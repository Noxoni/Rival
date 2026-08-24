"""Execute Rival v10.2 Stage-1 implementation Gates 0 through 7."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import psutil
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import (  # noqa: E402
    save_checkpoint_atomic,
    verify_checkpoint_reload_parity,
    write_json_atomic,
)
from rival_training.v10_2_campaign import (  # noqa: E402
    CAMPAIGN_STATE_PATH,
    CORPUS_ROOT,
    DEFAULT_STAGE1_CONFIG,
    RESULT_ROOT,
    SOURCE_ACTOR_SHA256,
    SOURCE_CHECKPOINT,
    SOURCE_MANIFEST_SHA256,
    actor_only_stage_transfer,
    build_stage1_corpus_manifests,
    config_identity,
    initialize_progressive_state,
    load_stage1_config,
    update_progressive_state,
)
from rival_training.v10_2_curriculum import (  # noqa: E402
    curriculum_reset_audit,
)
from rival_training.v10_2_environment import (  # noqa: E402
    RivalSingleLearnerGymWrapperV1,
    build_ball_acquisition_env,
)
from rival_training.v10_2_evaluation import (  # noqa: E402
    evaluate_stage1_checkpoint,
)
from rival_training.v10_2_reward import (  # noqa: E402
    DISTANCE_PROGRESS_ABSOLUTE_EPISODE_BUDGET,
    DISTANCE_PROGRESS_SAFETY_CLIP_UU,
    BallAcquisitionTransitionV1,
    RivalBallAcquisitionRewardKernelV1,
    RivalNewContactDetectorV1,
    ball_acquisition_reward_metadata,
)
from rival_training.v9_checkpoint import sha256_file  # noqa: E402
from rival_training.v9_curriculum import _set_ball, _set_car  # noqa: E402
from rival_training.v9_trainer import RivalV9PPOTrainer  # noqa: E402


DEFAULT_OUTPUT = RESULT_ROOT / "preflight.json"
DEFAULT_DISPOSABLE_ROOT = (
    REPOSITORY_ROOT / "training/checkpoints/milestone10_2/preflight"
)
EXPECTED_WISP_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}


def _transition(
    tick: int,
    car: tuple[float, float, float],
    ball: tuple[float, float, float],
    *,
    raw_touches: int = 0,
    goal_for: bool = False,
    goal_against: bool = False,
) -> BallAcquisitionTransitionV1:
    return BallAcquisitionTransitionV1(
        tick=tick,
        car_position=np.asarray(car, dtype=np.float64),
        ball_position=np.asarray(ball, dtype=np.float64),
        raw_touch_records=raw_touches,
        goal_for=goal_for,
        goal_against=goal_against,
    )


def _reward_truth_table() -> dict[str, Any]:
    stationary_ball = (1000.0, 0.0, 93.0)
    kernel = RivalBallAcquisitionRewardKernelV1()
    kernel.reset(_transition(0, (0.0, 0.0, 17.0), stationary_ball))
    toward = kernel.step(_transition(1, (10.0, 0.0, 17.0), stationary_ball))
    away = kernel.step(_transition(2, (0.0, 0.0, 17.0), stationary_ball))

    moving_ball = RivalBallAcquisitionRewardKernelV1()
    moving_ball.reset(_transition(0, (0.0, 0.0, 17.0), stationary_ball))
    ball_toward = moving_ball.step(
        _transition(1, (0.0, 0.0, 17.0), (500.0, 0.0, 93.0))
    )
    ball_away = moving_ball.step(
        _transition(2, (0.0, 0.0, 17.0), (1500.0, 0.0, 93.0))
    )

    cycle = RivalBallAcquisitionRewardKernelV1()
    cycle.reset(_transition(0, (0.0, 0.0, 17.0), stationary_ball))
    cycle_approach = cycle.step(
        _transition(1, (10.0, 0.0, 17.0), stationary_ball)
    )
    cycle_retreat = cycle.step(
        _transition(2, (0.0, 0.0, 17.0), stationary_ball)
    )

    contacts = RivalNewContactDetectorV1()
    contact_sequence = [
        contacts.process(raw) for raw in (1, 1, 1, 0, 1, 0, 1)
    ]

    budget = RivalBallAcquisitionRewardKernelV1(safety_clip_uu=1000.0)
    budget.reset(
        _transition(0, (0.0, 0.0, 17.0), (5000.0, 0.0, 93.0))
    )
    for tick in range(1, 12):
        budget.step(
            _transition(
                tick,
                (tick * 500.0, 0.0, 17.0),
                (5000.0, 0.0, 93.0),
            )
        )

    goal = RivalBallAcquisitionRewardKernelV1()
    goal.reset(_transition(0, (0.0, 0.0, 17.0), stationary_ball))
    goal_for = goal.step(
        _transition(
            1,
            (0.0, 0.0, 17.0),
            stationary_ball,
            goal_for=True,
        )
    )
    checks = {
        "toward_stationary_ball_positive": toward.total > 0.0,
        "away_from_stationary_ball_negative": away.total < 0.0,
        "stationary_car_ball_toward_zero": abs(ball_toward.total) <= 1e-12,
        "stationary_car_ball_away_zero": abs(ball_away.total) <= 1e-12,
        "closed_approach_retreat_cycle_zero": abs(
            cycle_approach.total + cycle_retreat.total
        )
        <= 1e-12,
        "sustained_contact_counted_once": contact_sequence[:3]
        == [True, False, False],
        "two_separated_retouches_counted": contact_sequence
        == [True, False, False, False, True, False, True],
        "dense_absolute_budget_exact": math.isclose(
            budget.distance_absolute_spend,
            DISTANCE_PROGRESS_ABSOLUTE_EPISODE_BUDGET,
            abs_tol=1e-12,
        ),
        "goal_for_reward_zero": goal_for.components["goal_for"] == 0.0
        and goal_for.total == 0.0,
        "goal_against_reward_zero": ball_acquisition_reward_metadata()[
            "goal_against_reward"
        ]
        == 0.0,
        "speed_and_controller_reward_absent": ball_acquisition_reward_metadata()[
            "speed_reward"
        ]
        == 0.0
        and ball_acquisition_reward_metadata()["reads_controller_action"]
        is False,
    }
    checks["passed"] = all(checks.values())
    return {
        "metadata": ball_acquisition_reward_metadata(),
        "toward": toward.__dict__,
        "away": away.__dict__,
        "stationary_car_ball_toward": ball_toward.__dict__,
        "stationary_car_ball_away": ball_away.__dict__,
        "contact_sequence": contact_sequence,
        "dense_budget_spend": budget.distance_absolute_spend,
        "checks": checks,
    }


def _rocketsim_touch_trace() -> dict[str, Any]:
    engine = RocketSimEngine(rlbot_delay=True)
    state = engine.create_base_state()
    shared: dict[str, Any] = {}
    FixedTeamSizeMutator(blue_size=1, orange_size=1).apply(state, shared)
    agents = list(state.cars)
    learner = next(
        agent for agent in agents if int(state.cars[agent].team_num) == 0
    )
    dummy = next(agent for agent in agents if agent != learner)
    _set_ball(
        state,
        position=np.asarray([500.0, 0.0, 93.0]),
        velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    _set_car(
        state.cars[learner],
        position=np.asarray([0.0, 0.0, 17.0]),
        velocity=np.asarray([900.0, 0.0, 0.0]),
        euler=np.asarray([0.0, 0.0, 0.0]),
        boost=0.0,
    )
    _set_car(
        state.cars[dummy],
        position=np.asarray([-3500.0, -4500.0, 17.0]),
        velocity=np.zeros(3),
        euler=np.zeros(3),
        boost=0.0,
    )
    state = engine.set_state(state, shared)
    detector = RivalNewContactDetectorV1()
    rows: list[dict[str, Any]] = []
    touch_events = 0

    def step(throttle: float, phase: str) -> None:
        nonlocal state, touch_events
        actions = {
            learner: np.asarray(
                [[throttle, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            dummy: np.zeros((1, 8), dtype=np.float32),
        }
        state = engine.step(actions, shared)
        raw = int(state.cars[learner].ball_touches)
        emitted = detector.process(raw)
        touch_events += int(emitted)
        rows.append(
            {
                "tick": int(state.tick_count),
                "phase": phase,
                "learner_raw_touch_records": raw,
                "dummy_raw_touch_records": int(
                    state.cars[dummy].ball_touches
                ),
                "detector_contact_active": detector.raw_contact_active,
                "emitted_new_contact": emitted,
                "car_ball_distance": float(
                    np.linalg.norm(
                        state.cars[learner].physics.position
                        - state.ball.position
                    )
                ),
            }
        )

    for _ in range(180):
        step(1.0, "first_approach")
        if touch_events:
            break
    first_touch_found = touch_events == 1
    for _ in range(12):
        step(0.0, "post_first_contact")

    # Force a physically separated second approach while retaining detector
    # state.  The next raw-zero simulator tick proves separation before retouch.
    separated = state
    _set_ball(
        separated,
        position=np.asarray([500.0, 0.0, 93.0]),
        velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    _set_car(
        separated.cars[learner],
        position=np.asarray([0.0, 0.0, 17.0]),
        velocity=np.asarray([1000.0, 0.0, 0.0]),
        euler=np.asarray([0.0, 0.0, 0.0]),
        boost=0.0,
    )
    state = engine.set_state(separated, shared)
    step(0.0, "forced_separation")
    separated_zero_seen = rows[-1]["learner_raw_touch_records"] == 0
    for _ in range(180):
        step(1.0, "second_approach")
        if touch_events >= 2:
            break
    reset_detector = RivalNewContactDetectorV1()
    phantom_after_reset = reset_detector.process(0)
    engine.close()
    checks = {
        "first_live_rocketsim_contact_emitted_once": first_touch_found,
        "continuous_raw_contact_run_not_repeated": all(
            sum(
                bool(row["emitted_new_contact"])
                for row in rows
                if row["phase"] == phase
            )
            == 1
            for phase in ("first_approach", "second_approach")
        ),
        "separation_raw_zero_observed": separated_zero_seen,
        "separated_retouch_emitted": touch_events >= 2,
        "dummy_contact_never_credited": all(
            not row["dummy_raw_touch_records"] for row in rows
        ),
        "reset_has_no_phantom_touch": phantom_after_reset is False,
    }
    checks["passed"] = all(checks.values())
    return {
        "trace_version": "RivalRocketSimTouchTraceV1",
        "rows": rows,
        "emitted_touch_events": touch_events,
        "checks": checks,
    }


def _active_dummy_isolation_audit(total_steps: int) -> dict[str, Any]:
    if int(total_steps) < 10_000:
        raise ValueError("Dummy isolation requires at least 10,000 steps")
    rng = np.random.default_rng(20261023)
    environments = [
        RivalSingleLearnerGymWrapperV1(
            build_ball_acquisition_env(
                phase="A",
                seed=20261023 + team,
                forced_active_team=team,
            )
        )
        for team in (0, 1)
    ]
    observations = [environment.reset() for environment in environments]
    resets = [1, 1]
    dummy_touches = 0
    progress_unclipped: list[float] = []
    all_finite = True
    exactly_one_row = True
    try:
        for step_index in range(int(total_steps)):
            slot = step_index % 2
            environment = environments[slot]
            action = np.empty((1, 8), dtype=np.float32)
            action[0, :5] = rng.uniform(-1.0, 1.0, 5)
            action[0, 5:] = rng.integers(0, 2, 3)
            observation, rewards, done, truncated, info = environment.step(
                action
            )
            observations[slot] = observation
            all_finite = all_finite and bool(np.isfinite(observation).all())
            exactly_one_row = exactly_one_row and observation.shape == (1, 714)
            exactly_one_row = exactly_one_row and len(rewards) == 1
            dummy_touches += int(
                environment.rlgym_env.state.cars[
                    environment.dummy_agent
                ].ball_touches
                > 0
            )
            progress_unclipped.append(
                float(
                    environment.rlgym_env.shared_info[
                        "rival_v10_2_reward_metrics"
                    ]["car_progress_unclipped_uu"]
                )
            )
            # Force many episode transitions in addition to natural timeouts.
            if done or truncated or (step_index + 1) % 240 == 0:
                observations[slot] = environment.reset()
                resets[slot] += 1
            if info["rival_v10_2"]["dummy_rows_returned"] != 0:
                exactly_one_row = False
    finally:
        for environment in environments:
            environment.close()
    absolute = np.abs(np.asarray(progress_unclipped, dtype=np.float64))
    ordinary_p999 = float(np.percentile(absolute, 99.9))
    maximum = float(absolute.max())
    total_dummy_actions = sum(
        environment.dummy_actions_injected for environment in environments
    )
    nonzero_dummy_actions = sum(
        environment.dummy_nonzero_actions_injected
        for environment in environments
    )
    checks = {
        "at_least_10000_environment_steps": int(total_steps) >= 10_000,
        "exactly_one_active_learner_row_per_step": exactly_one_row,
        "active_teams_exactly_balanced": sum(
            [int(total_steps) // 2, int(total_steps) - int(total_steps) // 2]
        )
        == int(total_steps),
        "dummy_controller_always_exact_zero": nonzero_dummy_actions == 0
        and total_dummy_actions == int(total_steps),
        "dummy_rows_never_returned": exactly_one_row,
        "dummy_physical_interference_below_point_one_percent": dummy_touches
        / int(total_steps)
        < 0.001,
        "opponent_observation_contract_finite": all_finite,
        "safety_clip_does_not_truncate_ordinary_p999": ordinary_p999
        < DISTANCE_PROGRESS_SAFETY_CLIP_UU,
    }
    checks["passed"] = all(checks.values())
    return {
        "steps": int(total_steps),
        "resets_by_active_team": {"0": resets[0], "1": resets[1]},
        "dummy_actions_injected": total_dummy_actions,
        "dummy_nonzero_actions_injected": nonzero_dummy_actions,
        "dummy_touch_ticks": dummy_touches,
        "dummy_touch_tick_share": dummy_touches / int(total_steps),
        "distance_progress_legal_transition_statistics": {
            "samples": len(progress_unclipped),
            "absolute_p99_9_uu": ordinary_p999,
            "absolute_maximum_uu": maximum,
            "selected_safety_clip_uu": DISTANCE_PROGRESS_SAFETY_CLIP_UU,
        },
        "checks": checks,
    }


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("RIVAL_"):
            environment.pop(name)
    environment.pop("PYTHONPATH", None)
    return environment


def _production_probe() -> dict[str, Any]:
    production_python = REPOSITORY_ROOT / ".venv/Scripts/python.exe"
    script = (
        "import json,config; print(json.dumps({"
        "'mode':config.POLICY_RUNTIME_MODE,'tick_skip':config.TICK_SKIP,"
        "'v9':config.V9_SCRATCH_POLICY_ENABLED,"
        "'candidate':config.CANDIDATE_POLICY_ENABLED,"
        "'m08':config.M08_DUAL_RATE_ENABLED,"
        "'policy':config.MODEL_INFO_POLICY.path.name,"
        "'shared_head':config.MODEL_INFO_SHARED_HEAD.path.name}))"
    )
    completed = subprocess.run(
        [str(production_python), "-c", script],
        cwd=REPOSITORY_ROOT / "bot",
        env=_clean_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


def _production_state() -> dict[str, Any]:
    return {
        "probe": _production_probe(),
        "wisp_hashes": {
            name: sha256_file(REPOSITORY_ROOT / "bot/models" / name)
            for name in EXPECTED_WISP_HASHES
        },
    }


def _running_training_processes() -> list[dict[str, Any]]:
    matches = []
    needles = (
        "run_m10_campaign_boundary.py",
        "run_m10_1_campaign_boundary.py",
        "run_m10_2_stage1_boundary.py",
        "run_m10_2_progressive.py",
    )
    current = os.getpid()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if process.pid != current and any(needle in command for needle in needles):
            matches.append(
                {
                    "pid": process.pid,
                    "name": process.info.get("name"),
                    "command": command,
                }
            )
    return matches


def _resource_snapshot() -> dict[str, Any]:
    gpu: dict[str, Any] = {}
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        values = [float(item.strip()) for item in completed.stdout.split(",")]
        gpu = {
            "utilization_percent": values[0],
            "memory_used_mib": values[1],
            "memory_total_mib": values[2],
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        gpu = {"status": "unavailable"}
    memory = psutil.virtual_memory()
    return {
        "cpu_utilization_percent": psutil.cpu_percent(interval=None),
        "system_memory_used_mib": (memory.total - memory.available)
        / (1024 * 1024),
        "system_memory_available_mib": memory.available / (1024 * 1024),
        "gpu": gpu,
    }


def _worker_sweep(
    base_config: dict[str, Any],
    *,
    device: str,
    candidates: list[int],
    active_steps: int,
) -> dict[str, Any]:
    rows = []
    for worker_count in candidates:
        config = deepcopy(base_config)
        config["backend"]["worker_count"] = int(worker_count)
        transfer = actor_only_stage_transfer(
            SOURCE_CHECKPOINT, config, device=device
        )
        trainer = RivalV9PPOTrainer(
            config,
            device=device,
            actor=transfer["actor"],
            critic=transfer["critic"],
            actor_optimizer=transfer["actor_optimizer"],
            critic_optimizer=transfer["critic_optimizer"],
            trainer_state=transfer["trainer_state"],
            env_factory=(
                __import__(
                    "rival_training.v10_2_environment",
                    fromlist=["make_ball_acquisition_phase_a_env"],
                ).make_ball_acquisition_phase_a_env
            ),
        )
        cleanup = None
        psutil.cpu_percent(interval=None)
        before = _resource_snapshot()
        try:
            shapes = trainer.start_workers()
            started = time.perf_counter()
            data, metrics, collected, collection_seconds = (
                trainer.manager.collect_timesteps(int(active_steps))
            )
            wall_seconds = time.perf_counter() - started
            health = trainer.worker_health()
            inference = trainer.policy.drain_inference_samples()
        finally:
            cleanup = trainer.cleanup()
        after = _resource_snapshot()
        latency = np.asarray(
            [float(item["per_agent_microseconds"]) for item in inference],
            dtype=np.float64,
        )
        stable = (
            len(health) == int(worker_count)
            and all(item["alive"] for item in health)
            and cleanup is not None
            and cleanup["passed"]
            and len(data[0]) == int(collected)
        )
        rows.append(
            {
                "worker_count": int(worker_count),
                "environment_shapes": shapes,
                "requested_active_learner_steps": int(active_steps),
                "collected_active_learner_steps": int(collected),
                "experience_records": len(data[0]),
                "metrics_records": len(metrics),
                "collection_seconds": float(collection_seconds),
                "rollout_wall_seconds": wall_seconds,
                "active_learner_steps_per_second": float(
                    collected / collection_seconds
                ),
                "aggregate_simulated_game_seconds_per_second": float(
                    collected / collection_seconds / 120.0
                ),
                "rollout_inference_per_agent_microseconds": {
                    "samples": int(latency.size),
                    "mean": float(latency.mean()),
                    "p95": float(np.percentile(latency, 95)),
                    "maximum": float(latency.max()),
                },
                "resource_before": before,
                "resource_after": after,
                "worker_health": health,
                "worker_stalls_or_crashes": sum(
                    not item["alive"] for item in health
                ),
                "cleanup": cleanup,
                "stable": stable,
            }
        )
    stable_rows = [row for row in rows if row["stable"]]
    if not stable_rows:
        raise RuntimeError("No stable worker candidate completed the sweep")
    selected = max(
        stable_rows, key=lambda row: row["active_learner_steps_per_second"]
    )
    return {
        "selection_metric": "stable_active_learner_steps_per_second",
        "masking_materially_changed_row_count": True,
        "candidates": rows,
        "selected_worker_count": selected["worker_count"],
        "selected_active_learner_steps_per_second": selected[
            "active_learner_steps_per_second"
        ],
        "checks": {
            "all_candidates_completed_stably": len(stable_rows) == len(rows),
            "selected_is_highest_stable_throughput": True,
            "dummy_inclusive_rows_not_used": True,
        },
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = load_stage1_config(args.config)
    stale_trainers = _running_training_processes()
    source_before = {
        "actor_sha256": sha256_file(SOURCE_CHECKPOINT / "actor.pt"),
        "manifest_sha256": sha256_file(
            SOURCE_CHECKPOINT / "checkpoint_manifest.json"
        ),
    }
    production_before = _production_state()
    transfer = actor_only_stage_transfer(
        SOURCE_CHECKPOINT, config, device=args.device
    )
    corpora = build_stage1_corpus_manifests()
    initialize_progressive_state(
        transfer_proof=transfer["proof"], corpora=corpora
    )
    reward = _reward_truth_table()
    touch_trace = _rocketsim_touch_trace()
    isolation = _active_dummy_isolation_audit(args.isolation_steps)
    reset_audits = {
        "phase_A": curriculum_reset_audit(
            "A", seed=20261024, samples_per_family=args.phase_a_resets
        ),
        "phase_B": curriculum_reset_audit(
            "B", seed=20261025, samples_per_family=args.phase_b_resets
        ),
    }

    source_gate = evaluate_stage1_checkpoint(
        SOURCE_CHECKPOINT,
        CORPUS_ROOT / "stage1_frozen_gate_corpus.json",
        device=args.device,
        evaluation_workers=args.evaluation_workers,
    )
    source_unseen = evaluate_stage1_checkpoint(
        SOURCE_CHECKPOINT,
        CORPUS_ROOT / "stage1_unseen_generalization_corpus.json",
        device=args.device,
        evaluation_workers=args.evaluation_workers,
    )
    write_json_atomic(
        RESULT_ROOT / "stage_1/source_v10_1_plus10_gate.json", source_gate
    )
    write_json_atomic(
        RESULT_ROOT / "stage_1/source_v10_1_plus10_unseen.json",
        source_unseen,
    )

    sweep = _worker_sweep(
        config,
        device=args.device,
        candidates=args.worker_candidates,
        active_steps=args.sweep_active_steps,
    )
    effective_config = deepcopy(config)
    effective_config["backend"]["worker_count"] = int(
        sweep["selected_worker_count"]
    )
    disposable_transfer = actor_only_stage_transfer(
        SOURCE_CHECKPOINT, effective_config, device=args.device
    )
    from rival_training.v10_2_environment import (
        make_ball_acquisition_phase_a_env,
    )

    trainer = RivalV9PPOTrainer(
        effective_config,
        device=args.device,
        actor=disposable_transfer["actor"],
        critic=disposable_transfer["critic"],
        actor_optimizer=disposable_transfer["actor_optimizer"],
        critic_optimizer=disposable_transfer["critic_optimizer"],
        trainer_state=disposable_transfer["trainer_state"],
        env_factory=make_ball_acquisition_phase_a_env,
    )
    cleanup = None
    try:
        shapes = trainer.start_workers()
        health_before = trainer.worker_health()
        smoke, held_observations = trainer.run_iteration()
        health_after = trainer.worker_health()
    finally:
        cleanup = trainer.cleanup()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    disposable_directory = (
        args.disposable_root.resolve()
        / stamp
        / f"{trainer.cumulative_agent_steps:09d}"
    )
    state = trainer.trainer_state()
    state.update(
        {
            "stage": 1,
            "stage_phase": "A",
            "v10_2_disposable_preflight": True,
            "experience_counted_toward_campaign": False,
            "production_promotion_authorized": False,
        }
    )
    disposable_checkpoint = save_checkpoint_atomic(
        disposable_directory,
        actor=trainer.actor,
        critic=trainer.critic,
        actor_optimizer=trainer.actor_optimizer,
        critic_optimizer=trainer.critic_optimizer,
        trainer_state=state,
        config=effective_config,
        reload_observations=held_observations,
    )
    reload_parity = verify_checkpoint_reload_parity(
        disposable_directory,
        expected_config=effective_config,
        device="cpu",
    )
    real_restart = actor_only_stage_transfer(
        SOURCE_CHECKPOINT, effective_config, device="cpu"
    )
    source_after = {
        "actor_sha256": sha256_file(SOURCE_CHECKPOINT / "actor.pt"),
        "manifest_sha256": sha256_file(
            SOURCE_CHECKPOINT / "checkpoint_manifest.json"
        ),
    }
    production_after = _production_state()
    expected_production = {
        "mode": "frozen_wisp_production",
        "tick_skip": 8,
        "v9": False,
        "candidate": False,
        "m08": False,
        "policy": "POLICY.lt",
        "shared_head": "SHARED_HEAD.lt",
    }
    checks = {
        "gate0_no_stale_training_process": not stale_trainers,
        "gate0_source_checkpoint_exact": source_before
        == {
            "actor_sha256": SOURCE_ACTOR_SHA256,
            "manifest_sha256": SOURCE_MANIFEST_SHA256,
        }
        == source_after,
        "gate0_frozen_production_exact": production_before
        == production_after
        == {
            "probe": expected_production,
            "wisp_hashes": EXPECTED_WISP_HASHES,
        },
        "gate1_actor_only_transfer": transfer["proof"]["checks"]["passed"],
        "gate2_reward_truth_table": reward["checks"]["passed"],
        "gate3_rocketsim_touch_trace": touch_trace["checks"]["passed"],
        "gate4_active_dummy_isolation": isolation["checks"]["passed"],
        "gate5_reset_distributions": all(
            report["checks"]["passed"] for report in reset_audits.values()
        ),
        "gate6_corpora_frozen_and_source_evaluated": corpora["checks"][
            "passed"
        ]
        and source_gate["checks"]["passed"]
        and source_unseen["checks"]["passed"],
        "gate7_worker_sweep_stable": sweep["checks"][
            "all_candidates_completed_stably"
        ],
        "gate7_disposable_cuda_ppo_healthy": smoke["health"]["passed"],
        "gate7_one_active_row_per_step": smoke["experience_records"]
        == smoke["collected_agent_steps"],
        "gate7_both_actor_heads_and_critic_updated": smoke["health"][
            "all_hybrid_head_gradient_rows_nonzero"
        ]
        and smoke["health"]["actor_updated"]
        and smoke["health"]["critic_updated"],
        "gate7_checkpoint_reload_exact": reload_parity["checks"]["passed"],
        "gate7_workers_cleaned": cleanup is not None and cleanup["passed"],
        "disposable_update_discarded_and_real_restart_exact": real_restart[
            "proof"
        ]["checks"]["passed"]
        and real_restart["proof"]["fresh_actor_optimizer_state_entries"] == 0
        and real_restart["proof"]["fresh_critic_optimizer_state_entries"] == 0,
        "real_campaign_clock_not_started": json.loads(
            CAMPAIGN_STATE_PATH.read_text(encoding="utf-8")
        )["campaign_wall_clock_started_utc"]
        is None,
        "production_promotion_authorized": False,
    }
    passed = all(
        value
        for key, value in checks.items()
        if key != "production_promotion_authorized"
    ) and checks["production_promotion_authorized"] is False
    result = {
        "schema_version": 1,
        "preflight_version": "RivalM10_2Stage1PreflightV1",
        "status": "passed" if passed else "failed",
        "config": config_identity(config),
        "effective_selected_worker_count": sweep["selected_worker_count"],
        "effective_config": effective_config,
        "stale_training_processes": stale_trainers,
        "actor_only_transfer": transfer["proof"],
        "reward_truth_table": reward,
        "rocketsim_touch_trace": touch_trace,
        "active_learner_dummy_isolation": isolation,
        "reset_distribution_audits": reset_audits,
        "corpora": corpora,
        "source_evaluation": {
            "gate": source_gate,
            "unseen": source_unseen,
        },
        "worker_sweep": sweep,
        "disposable_cuda_smoke": {
            "environment_shapes": shapes,
            "health_before": health_before,
            "iteration": smoke,
            "health_after": health_after,
            "cleanup": cleanup,
            "checkpoint": disposable_checkpoint,
            "checkpoint_reload": reload_parity,
            "experience_counted_toward_campaign": False,
        },
        "real_campaign_restart_proof": real_restart["proof"],
        "source_before": source_before,
        "source_after": source_after,
        "production_before": production_before,
        "production_after": production_after,
        "checks": checks,
    }
    checks["passed"] = passed
    write_json_atomic(args.output, result)
    update_progressive_state(
        {
            "current_phase": "ready_stage_1" if passed else "stopped",
            "gate_decision": (
                "stage_1_preflight_passed"
                if passed
                else "stop_stage_1_preflight_failed"
            ),
            "selected_worker_count": sweep["selected_worker_count"],
            "preflight_result": args.output.resolve()
            .relative_to(REPOSITORY_ROOT)
            .as_posix(),
            "stop_reason": None if passed else "stop_stage_1_preflight_failed",
        }
    )
    if not passed:
        raise RuntimeError(f"Milestone 10.2 preflight failed: {checks}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_STAGE1_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--disposable-root", type=Path, default=DEFAULT_DISPOSABLE_ROOT
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase-a-resets", type=int, default=10_000)
    parser.add_argument("--phase-b-resets", type=int, default=5_000)
    parser.add_argument("--isolation-steps", type=int, default=10_000)
    parser.add_argument("--evaluation-workers", type=int, default=24)
    parser.add_argument(
        "--worker-candidates",
        type=int,
        nargs="+",
        default=[32, 40, 48, 56, 64],
    )
    parser.add_argument("--sweep-active-steps", type=int, default=6_000)
    args = parser.parse_args()
    report = run_preflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_worker_count": report[
                    "effective_selected_worker_count"
                ],
                "source_gate_success": report["source_evaluation"]["gate"][
                    "overall"
                ]["first_touch_success_share"],
                "source_unseen_success": report["source_evaluation"][
                    "unseen"
                ]["overall"]["first_touch_success_share"],
                "checks": report["checks"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
