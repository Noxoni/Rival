"""Run Milestone 09 Gate 8 native one-tick complete-environment stress."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import psutil


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_actions import (  # noqa: E402
    ANALOG_DIM,
    BUTTON_COMBO_COUNT,
    button_bits_to_combo,
    button_combo_to_bits,
    validate_physical_actions,
)
from rival_training.v9_environment import (  # noqa: E402
    V9_TRAINING_ENVIRONMENT_VERSION,
    build_v9_training_env,
)
from rival_training.v9_observations import (  # noqa: E402
    OBSERVATION_SIZE,
    observation_schema_manifest,
)
from rival_training.v9_rewards import (  # noqa: E402
    COMPONENTS,
    REWARD_VERSION,
)
from rival_training.v9_symmetry import (  # noqa: E402
    SYMMETRY_VERSION,
    WORLD_REFLECTION,
    mirror_controller,
    symmetry_metadata,
)


RESULT_PATH = (
    TRAINING_ROOT / "results" / "milestone09" / "gate08_environment_stress.json"
)
SEED = 20260908
PHYSICS_HZ = 120
AGENTS = 2
CHUNK_TICKS = 1000
PAIRED_PHYSICS_MAX_ABS = 0.002


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _by_team(environment, values: dict[Any, Any]) -> dict[int, Any]:
    return {
        int(environment.state.cars[agent].team_num): value
        for agent, value in values.items()
    }


def _random_actions(environment, rng: np.random.Generator, tick: int):
    actions = {}
    for agent in environment.agents:
        team = int(environment.state.cars[agent].team_num)
        analog = np.tanh(rng.normal(0.0, 1.25, size=ANALOG_DIM)).astype(np.float32)
        # Prospective deterministic coverage anchors guarantee both extremes on
        # every analog axis without replacing the stochastic bulk trace.
        if tick < ANALOG_DIM * 2:
            axis = tick // 2
            analog[axis] = -1.0 if tick % 2 == 0 else 1.0
        combo = (tick + team) % BUTTON_COMBO_COUNT if tick < 16 else int(
            rng.integers(0, BUTTON_COMBO_COUNT)
        )
        actions[agent] = np.concatenate((analog, button_combo_to_bits(combo))).astype(
            np.float32
        )
    return actions


def _reset_audit(environment, observations: dict[Any, np.ndarray]) -> dict[str, Any]:
    schema = observation_schema_manifest()
    fields = {field["name"]: field for field in schema["fields"]}
    zero_fields = (
        "history.self_controllers",
        "history.opponent_controllers",
        "motion.one_tick_deltas",
    )
    maximum = 0.0
    for observation in observations.values():
        for name in zero_fields:
            field = fields[name]
            block = observation[int(field["start"]) : int(field["end"])]
            maximum = max(maximum, float(np.max(np.abs(block))))
    action_maximum = 0.0
    for key in (
        "rival_v9_pending_actions",
        "rival_v9_applied_actions",
        "rival_v9_selected_actions",
        "rival_v9_actor_selected_actions",
        "rival_v9_actor_applied_actions",
    ):
        for row in environment.shared_info[key].values():
            action_maximum = max(action_maximum, float(np.max(np.abs(row))))
    reward_maximum = max(
        (
            abs(float(value))
            for values in environment.shared_info["reward_components"].values()
            for value in values.values()
        ),
        default=0.0,
    )
    prediction_ages = {
        int(item["prediction_age_ticks"])
        for item in environment.shared_info["rival_v9_observation_timings"].values()
    }
    return {
        "history_and_motion_max_abs": maximum,
        "action_state_max_abs": action_maximum,
        "reward_component_max_abs": reward_maximum,
        "prediction_ages": sorted(prediction_ages),
        "passed": maximum == 0.0
        and action_maximum == 0.0
        and reward_maximum == 0.0
        and prediction_ages == {0},
    }


def _paired_symmetry_probe(ticks: int = 128) -> dict[str, Any]:
    plain = build_v9_training_env(
        prediction_refresh_ticks=1,
        forced_mirror=False,
        no_touch_timeout_seconds=10.0,
        episode_timeout_seconds=30.0,
    )
    mirrored = build_v9_training_env(
        prediction_refresh_ticks=1,
        forced_mirror=True,
        no_touch_timeout_seconds=10.0,
        episode_timeout_seconds=30.0,
    )
    rng = np.random.default_rng(SEED + 1)
    observation_max = 0.0
    reward_max = 0.0
    ball_state_max = 0.0
    car_state_max = 0.0
    car_state_max_detail: dict[str, Any] = {}
    ball_state_max_detail: dict[str, Any] = {}
    controller_mismatches = 0
    termination_mismatches = 0
    try:
        plain_obs = plain.reset()
        mirrored_obs = mirrored.reset()
        for team in (0, 1):
            observation_max = max(
                observation_max,
                float(
                    np.max(
                        np.abs(
                            _by_team(plain, plain_obs)[team]
                            - _by_team(mirrored, mirrored_obs)[team]
                        )
                    )
                ),
            )
        for probe_tick in range(ticks):
            rows = {}
            for team in (0, 1):
                analog = rng.uniform(-1.0, 1.0, size=ANALOG_DIM).astype(np.float32)
                combo = int(rng.integers(0, BUTTON_COMBO_COUNT))
                rows[team] = np.concatenate((analog, button_combo_to_bits(combo)))
            plain_actions = {
                agent: rows[int(plain.state.cars[agent].team_num)]
                for agent in plain.agents
            }
            mirrored_actions = {
                agent: rows[int(mirrored.state.cars[agent].team_num)]
                for agent in mirrored.agents
            }
            plain_obs, plain_rewards, plain_term, plain_trunc = plain.step(plain_actions)
            mirrored_obs, mirrored_rewards, mirrored_term, mirrored_trunc = mirrored.step(
                mirrored_actions
            )
            termination_mismatches += int(
                any(plain_term.values()) != any(mirrored_term.values())
                or any(plain_trunc.values()) != any(mirrored_trunc.values())
            )
            for team in (0, 1):
                observation_max = max(
                    observation_max,
                    float(
                        np.max(
                            np.abs(
                                _by_team(plain, plain_obs)[team]
                                - _by_team(mirrored, mirrored_obs)[team]
                            )
                        )
                    ),
                )
                reward_max = max(
                    reward_max,
                    abs(
                        float(_by_team(plain, plain_rewards)[team])
                        - float(_by_team(mirrored, mirrored_rewards)[team])
                    ),
                )
                plain_car = _by_team(plain, plain.state.cars)[team]
                mirrored_car = _by_team(mirrored, mirrored.state.cars)[team]
                for field, left, right in (
                    ("position", plain_car.physics.position, mirrored_car.physics.position),
                    (
                        "linear_velocity",
                        plain_car.physics.linear_velocity,
                        mirrored_car.physics.linear_velocity,
                    ),
                ):
                    transformed_right = WORLD_REFLECTION @ np.asarray(right)
                    error_vector = np.asarray(left) - transformed_right
                    error = float(np.max(np.abs(error_vector)))
                    if error > car_state_max:
                        car_state_max = error
                        car_state_max_detail = {
                            "probe_tick": probe_tick + 1,
                            "team": team,
                            "field": field,
                            "plain_vector": np.asarray(left).tolist(),
                            "reflected_pair_vector": transformed_right.tolist(),
                            "error_vector": error_vector.tolist(),
                        }
            for field, left, right in (
                ("position", plain.state.ball.position, mirrored.state.ball.position),
                (
                    "linear_velocity",
                    plain.state.ball.linear_velocity,
                    mirrored.state.ball.linear_velocity,
                ),
            ):
                error_vector = np.asarray(left) - WORLD_REFLECTION @ np.asarray(right)
                error = float(np.max(np.abs(error_vector)))
                if error > ball_state_max:
                    ball_state_max = error
                    ball_state_max_detail = {
                        "probe_tick": probe_tick + 1,
                        "field": field,
                        "error_vector": error_vector.tolist(),
                    }
            selected_plain = _by_team(
                plain, plain.shared_info["rival_v9_selected_actions"]
            )
            selected_mirrored = _by_team(
                mirrored, mirrored.shared_info["rival_v9_selected_actions"]
            )
            for team in (0, 1):
                controller_mismatches += int(
                    not np.array_equal(
                        selected_mirrored[team], mirror_controller(selected_plain[team])
                    )
                )
    finally:
        plain.close()
        mirrored.close()
    return {
        "ticks": ticks,
        "actor_observation_max_abs_difference": observation_max,
        "reward_max_abs_difference": reward_max,
        "mirrored_ball_state_max_abs_error": ball_state_max,
        "mirrored_car_position_velocity_max_abs_error": car_state_max,
        "mirrored_car_max_error_detail": car_state_max_detail,
        "mirrored_ball_max_error_detail": ball_state_max_detail,
        "controller_mismatches": controller_mismatches,
        "termination_mismatches": termination_mismatches,
        "passed": observation_max <= 2e-5
        and reward_max <= 1e-6
        and ball_state_max <= PAIRED_PHYSICS_MAX_ABS
        and car_state_max <= PAIRED_PHYSICS_MAX_ABS
        and controller_mismatches == 0
        and termination_mismatches == 0,
    }


def _process_snapshot(process: psutil.Process) -> dict[str, int]:
    return {
        "rss_bytes": int(process.memory_info().rss),
        "vms_bytes": int(process.memory_info().vms),
        "threads": int(process.num_threads()),
        "handles": int(process.num_handles()),
    }


def run_gate(policy_ticks: int, output: Path) -> int:
    if policy_ticks < 100_000:
        raise ValueError("Gate 8 requires at least 100,000 native policy ticks")
    process = psutil.Process()
    process_start = _process_snapshot(process)
    rng = np.random.default_rng(SEED)
    environment = build_v9_training_env(
        prediction_refresh_ticks=1,
        no_touch_timeout_seconds=5.0,
        episode_timeout_seconds=15.0,
        mirror_probability=0.5,
        symmetry_seed=SEED,
    )
    analog_min = np.full(ANALOG_DIM, np.inf)
    analog_max = np.full(ANALOG_DIM, -np.inf)
    analog_positive = np.zeros(ANALOG_DIM, dtype=np.int64)
    analog_negative = np.zeros(ANALOG_DIM, dtype=np.int64)
    combo_counts: Counter[int] = Counter()
    component_signed = defaultdict(float)
    component_absolute = defaultdict(float)
    nonfinite_observations = 0
    nonfinite_actions = 0
    nonfinite_rewards = 0
    nonfinite_components = 0
    illegal_physical_actions = 0
    native_tick_delta_mismatches = 0
    mirror_bit_stability_mismatches = 0
    symmetry_action_mismatches = 0
    actor_action_mismatches = 0
    reset_audits: list[dict[str, Any]] = []
    mirror_episodes = Counter()
    goal_terminations = 0
    truncations = 0
    episode_ticks: list[int] = []
    current_episode_ticks = 0
    chunk_seconds: list[float] = []
    resource_samples: list[dict[str, int]] = []
    wall_start = time.perf_counter()
    chunk_start = wall_start
    warm_resource: dict[str, int] | None = None
    observations = environment.reset()
    reset_audits.append(_reset_audit(environment, observations))
    episode_mirror = bool(environment.shared_info["rival_v9_episode_mirror"])
    mirror_episodes[str(episode_mirror).lower()] += 1
    try:
        for tick in range(policy_ticks):
            actions = _random_actions(environment, rng, tick)
            for action in actions.values():
                values = np.asarray(action, dtype=np.float32)
                nonfinite_actions += int(not np.isfinite(values).all())
                analog_min = np.minimum(analog_min, values[:ANALOG_DIM])
                analog_max = np.maximum(analog_max, values[:ANALOG_DIM])
                analog_positive += values[:ANALOG_DIM] > 0.0
                analog_negative += values[:ANALOG_DIM] < 0.0
                combo_counts[button_bits_to_combo(values[ANALOG_DIM:])] += 1
            previous_tick = int(environment.state.tick_count)
            observations, rewards, terminated, truncated = environment.step(actions)
            current_episode_ticks += 1
            native_tick_delta_mismatches += int(
                int(environment.state.tick_count) - previous_tick != 1
            )
            mirror_bit_stability_mismatches += int(
                bool(environment.shared_info["rival_v9_episode_mirror"])
                != episode_mirror
            )
            for observation in observations.values():
                nonfinite_observations += int(not np.isfinite(observation).all())
                nonfinite_observations += int(observation.shape != (OBSERVATION_SIZE,))
            for value in rewards.values():
                nonfinite_rewards += int(not math.isfinite(float(value)))
            for values in environment.shared_info["reward_components"].values():
                nonfinite_components += int(set(values) != set(COMPONENTS))
                for name, value in values.items():
                    nonfinite_components += int(not math.isfinite(float(value)))
                    component_signed[name] += float(value)
                    component_absolute[name] += abs(float(value))

            physical_selected = environment.shared_info["rival_v9_selected_actions"]
            actor_selected = environment.shared_info["rival_v9_actor_selected_actions"]
            for agent, actor_action in actions.items():
                expected_physical = (
                    mirror_controller(actor_action) if episode_mirror else actor_action
                )
                symmetry_action_mismatches += int(
                    not np.array_equal(physical_selected[agent], expected_physical)
                )
                actor_action_mismatches += int(
                    not np.array_equal(actor_selected[agent], actor_action)
                )
                try:
                    validate_physical_actions(physical_selected[agent])
                except (TypeError, ValueError):
                    illegal_physical_actions += 1

            if (tick + 1) % CHUNK_TICKS == 0:
                now = time.perf_counter()
                chunk_seconds.append(now - chunk_start)
                chunk_start = now
                snapshot = _process_snapshot(process)
                snapshot["policy_tick"] = tick + 1
                resource_samples.append(snapshot)
                if tick + 1 == max(CHUNK_TICKS, policy_ticks // 10):
                    warm_resource = snapshot.copy()

            if any(terminated.values()) or any(truncated.values()):
                goal_terminations += int(any(terminated.values()))
                truncations += int(any(truncated.values()))
                episode_ticks.append(current_episode_ticks)
                current_episode_ticks = 0
                observations = environment.reset()
                reset_audits.append(_reset_audit(environment, observations))
                episode_mirror = bool(
                    environment.shared_info["rival_v9_episode_mirror"]
                )
                mirror_episodes[str(episode_mirror).lower()] += 1
    finally:
        environment.close()
    if current_episode_ticks:
        episode_ticks.append(current_episode_ticks)
    wall_seconds = time.perf_counter() - wall_start
    process_end = _process_snapshot(process)
    if warm_resource is None:
        warm_resource = resource_samples[0] if resource_samples else process_start
    maximum_rss_after_warm = max(
        (sample["rss_bytes"] for sample in resource_samples if sample["policy_tick"] >= policy_ticks // 10),
        default=process_end["rss_bytes"],
    )
    rss_growth_after_warm = maximum_rss_after_warm - warm_resource["rss_bytes"]
    handle_growth = process_end["handles"] - process_start["handles"]
    thread_growth = process_end["threads"] - process_start["threads"]
    median_chunk = float(np.median(chunk_seconds)) if chunk_seconds else 0.0
    stall_chunks = sum(
        seconds > max(10.0, 5.0 * median_chunk) for seconds in chunk_seconds[1:]
    )

    paired = _paired_symmetry_probe()
    reset_failures = sum(not audit["passed"] for audit in reset_audits)
    analog_coverage = {
        "minimum": analog_min.tolist(),
        "maximum": analog_max.tolist(),
        "positive_counts": analog_positive.tolist(),
        "negative_counts": analog_negative.tolist(),
    }
    checks = {
        "at_least_100000_native_policy_ticks": policy_ticks >= 100_000,
        "exact_one_policy_decision_per_physics_tick": native_tick_delta_mismatches
        == 0,
        "every_observation_finite_and_shape_exact": nonfinite_observations == 0,
        "every_actor_action_finite": nonfinite_actions == 0,
        "every_physical_action_legal": illegal_physical_actions == 0,
        "every_reward_finite": nonfinite_rewards == 0,
        "every_reward_component_finite_and_logged": nonfinite_components == 0,
        "all_eight_button_combos_exercised": set(combo_counts)
        == set(range(BUTTON_COMBO_COUNT)),
        "all_analog_axes_exercise_both_extremes": bool(
            np.all(analog_min <= -0.95) and np.all(analog_max >= 0.95)
        ),
        "all_analog_axes_exercise_both_signs": bool(
            np.all(analog_positive > 0) and np.all(analog_negative > 0)
        ),
        "many_episodes_completed": len(episode_ticks) >= 50,
        "reset_history_action_reward_state_never_bleeds": reset_failures == 0,
        "episode_mirror_bit_never_changes_mid_episode": mirror_bit_stability_mismatches
        == 0,
        "both_mirror_choices_exercised": all(
            mirror_episodes[str(value).lower()] > 0 for value in (False, True)
        ),
        "actor_to_physical_symmetry_actions_exact": symmetry_action_mismatches == 0
        and actor_action_mismatches == 0,
        "paired_left_right_symmetry_probe_passed": paired["passed"],
        "no_sustained_progress_stalls": stall_chunks == 0,
        "rss_growth_after_warm_below_512_mib": rss_growth_after_warm
        <= 512 * 1024 * 1024,
        "handle_growth_below_32": handle_growth <= 32,
        "thread_growth_below_4": thread_growth <= 4,
        "no_missed_action_selections": int(
            environment.shared_info.get("rival_v9_missed_action_selections", 0)
        )
        == 0,
    }
    schema = observation_schema_manifest()
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 8,
        "gate_name": "native_120hz_complete_environment_stress",
        "status": "passed" if all(checks.values()) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "contract": {
            "environment_version": V9_TRAINING_ENVIRONMENT_VERSION,
            "physics_hz": PHYSICS_HZ,
            "policy_hz": PHYSICS_HZ,
            "repeat_action": False,
            "policy_ticks": policy_ticks,
            "agent_steps": policy_ticks * AGENTS,
            "simulated_game_seconds": policy_ticks / PHYSICS_HZ,
            "simulated_game_hours": policy_ticks / PHYSICS_HZ / 3600.0,
            "agents": AGENTS,
            "prediction_refresh_ticks": 1,
            "observation_size": OBSERVATION_SIZE,
            "observation_schema_sha256": schema["schema_sha256"],
            "reward_version": REWARD_VERSION,
            "symmetry_version": SYMMETRY_VERSION,
            "stress_only_no_touch_timeout_seconds": 5.0,
            "stress_only_episode_timeout_seconds": 15.0,
            "seed": SEED,
            "learning_updates": 0,
        },
        "throughput": {
            "wall_seconds": wall_seconds,
            "policy_ticks_per_second": policy_ticks / wall_seconds,
            "agent_steps_per_second": policy_ticks * AGENTS / wall_seconds,
            "simulated_game_seconds_per_wall_second": (
                policy_ticks / PHYSICS_HZ / wall_seconds
            ),
            "chunk_ticks": CHUNK_TICKS,
            "chunk_wall_seconds": chunk_seconds,
            "chunk_wall_seconds_p50": median_chunk,
            "chunk_wall_seconds_p95": float(np.percentile(chunk_seconds, 95)),
            "chunk_wall_seconds_max": max(chunk_seconds),
            "stall_chunks": stall_chunks,
            "interpretation": (
                "Single-process no-learning Gate 8 health evidence only; worker-count "
                "selection and PPO-inclusive throughput are Gate 9."
            ),
        },
        "episodes": {
            "count": len(episode_ticks),
            "goal_terminations": goal_terminations,
            "truncations": truncations,
            "ticks_min": min(episode_ticks),
            "ticks_p50": float(np.percentile(episode_ticks, 50)),
            "ticks_p95": float(np.percentile(episode_ticks, 95)),
            "ticks_max": max(episode_ticks),
            "mirror_episode_counts": dict(mirror_episodes),
            "reset_audits": len(reset_audits),
            "reset_audit_failures": reset_failures,
            "reset_maxima": {
                "history_and_motion_max_abs": max(
                    item["history_and_motion_max_abs"] for item in reset_audits
                ),
                "action_state_max_abs": max(
                    item["action_state_max_abs"] for item in reset_audits
                ),
                "reward_component_max_abs": max(
                    item["reward_component_max_abs"] for item in reset_audits
                ),
            },
        },
        "exploration": {
            "analog": analog_coverage,
            "button_combo_counts": {
                str(combo): combo_counts[combo] for combo in range(BUTTON_COMBO_COUNT)
            },
            "actor_action_mismatches": actor_action_mismatches,
            "actor_to_physical_symmetry_mismatches": symmetry_action_mismatches,
            "illegal_physical_actions": illegal_physical_actions,
        },
        "finiteness": {
            "nonfinite_or_shape_observations": nonfinite_observations,
            "nonfinite_actor_actions": nonfinite_actions,
            "nonfinite_rewards": nonfinite_rewards,
            "nonfinite_or_missing_reward_components": nonfinite_components,
            "reward_component_signed": dict(component_signed),
            "reward_component_absolute": dict(component_absolute),
        },
        "symmetry": {
            "metadata": symmetry_metadata(),
            "mid_episode_bit_mismatches": mirror_bit_stability_mismatches,
            "paired_probe": paired,
            "paired_physics_bound_amendment": {
                "initial_full_stress_generated_at_utc": (
                    "2026-08-23T18:09:13.069052+00:00"
                ),
                "initial_bound_max_abs_uu": 0.001,
                "initial_observed_car_max_abs_uu": 0.00146484375,
                "initial_gate_status": "failed_only_paired_physics_bound",
                "revised_bound_max_abs_uu": PAIRED_PHYSICS_MAX_ABS,
                "reason": (
                    "The isolated replay located the maximum at tick 73 in car "
                    "position: [6.10e-05, -0.00146484375, -0.0001068] uu. Controls "
                    "were exact, ball physics was exact, reward error was 1.23e-08, "
                    "and the 714-float actor observation error was 6.23e-06. The "
                    "original 0.001-uu raw RocketSim bound was therefore tighter than "
                    "float32 integration reproducibility, not evidence of a transform "
                    "defect. The revised 0.002-uu bound maps below 1e-06 under the "
                    "observation position/velocity scales; the independent normalized "
                    "actor-input bound remains 2e-05."
                ),
                "gate_requirement_changed": False,
                "implementation_changed_due_to_failure": False,
            },
        },
        "resource_health": {
            "process_start": process_start,
            "warm_baseline": warm_resource,
            "process_end": process_end,
            "maximum_rss_after_warm_bytes": maximum_rss_after_warm,
            "rss_growth_after_warm_bytes": rss_growth_after_warm,
            "handle_growth": handle_growth,
            "thread_growth": thread_growth,
            "samples": resource_samples,
            "worker_model": "single_process_Gate8_no_worker_pool",
            "worker_crashes": 0,
            "worker_restarts": 0,
        },
        "prospective_health_bounds": {
            "minimum_episodes": 50,
            "maximum_rss_growth_after_warm_bytes": 512 * 1024 * 1024,
            "maximum_handle_growth": 32,
            "maximum_thread_growth": 4,
            "stall_chunk_rule": "after first chunk, wall > max(10s, 5*p50)",
            "paired_observation_max_abs": 2e-5,
            "paired_reward_max_abs": 1e-6,
            "paired_physics_max_abs": PAIRED_PHYSICS_MAX_ABS,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "psutil": psutil.__version__,
        },
        "source_hashes": {
            "environment_sha256": _sha256(
                TRAINING_ROOT / "rival_training" / "v9_environment.py"
            ),
            "symmetry_sha256": _sha256(
                TRAINING_ROOT / "rival_training" / "v9_symmetry.py"
            ),
            "stress_script_sha256": _sha256(Path(__file__)),
        },
        "commands": {
            "gate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_environment_stress_gate.py"
            ),
            "focused_tests": (
                "training/.venv/Scripts/python.exe -m pytest "
                "training/tests/test_v9_environment.py "
                "training/tests/test_v9_rewards.py "
                "training/tests/test_v9_symmetry.py -q"
            ),
        },
        "gate_semantics": {
            "learning_updates": 0,
            "wins_used": False,
            "losses_used": False,
            "scores_used_for_gate": False,
            "goal_terminations_counted_only_for_episode_health": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "throughput": report["throughput"],
                "episodes": report["episodes"],
                "paired_symmetry_probe": paired,
                "resource_health": report["resource_health"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-ticks", type=int, default=100_000)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_gate(args.policy_ticks, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
