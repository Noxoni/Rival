"""Focused pre-training verification for Rival Milestone 10.1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import (  # noqa: E402
    checkpoint_record,
    save_checkpoint_atomic,
    write_json_atomic,
)
from rival_training.v10_1_campaign import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    M10_PLUS25_CHECKPOINT,
    compact_training_iteration,
    config_migration_report,
    contract_summary,
    load_m10_1_config,
    verify_exact_plus25_start,
)
from rival_training.v10_bootstrap_curriculum import (  # noqa: E402
    PHASE_WEIGHTS,
    RivalAgencyBootstrapCurriculumV1,
    curriculum_distribution_report,
)
from rival_training.v10_bootstrap_environment import (  # noqa: E402
    ENV_FACTORY_BY_PHASE,
    build_v10_bootstrap_env,
)
from rival_training.v10_bootstrap_metrics import (  # noqa: E402
    aggregate_v10_bootstrap_metrics,
    collect_v10_bootstrap_metric_vector,
)
from rival_training.v10_bootstrap_reward import (  # noqa: E402
    AERIAL_TOUCH_BONUS,
    BALL_TOUCH_BASE_REWARD,
    COMBINED_SHAPING_ABSOLUTE_EPISODE_BUDGET,
    CONCEDE_REWARD,
    GOAL_REWARD,
    BootstrapRewardEventsV1,
    RivalAgencyBootstrapRewardKernelV1,
    RivalLogicalTouchAuditorV1,
    RewardStateV1,
    reward_metadata,
    touch_chain_bonus,
)
from rival_training.v9_checkpoint import (  # noqa: E402
    config_sha256,
    load_v9_checkpoint,
    sha256_file,
)
from rival_training.v9_trainer import RivalV9PPOTrainer  # noqa: E402


DEFAULT_OUTPUT = REPOSITORY_ROOT / "training/results/milestone10_1/preflight.json"
DEFAULT_DISPOSABLE_ROOT = (
    REPOSITORY_ROOT / "training/checkpoints/milestone10_1/preflight"
)
EXPECTED_WISP_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}


def _state(
    tick: int,
    *,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ball_z: float = 92.75,
    surface: bool = True,
) -> RewardStateV1:
    return RewardStateV1(
        tick_index=tick,
        self_position=np.asarray([0.0, 0.0, 17.0]),
        self_linear_velocity=np.asarray(velocity),
        self_forward=np.asarray([0.0, 1.0, 0.0]),
        self_up=np.asarray([0.0, 0.0, 1.0]),
        self_boost=50.0,
        self_surface_contact=surface,
        self_boosting=False,
        self_supersonic=False,
        self_can_dodge=False,
        ball_position=np.asarray([0.0, 1000.0, ball_z]),
        ball_linear_velocity=np.zeros(3),
    )


def _reward_formula_report() -> dict[str, Any]:
    stationary = RivalAgencyBootstrapRewardKernelV1()
    stationary.reset(_state(0))
    stationary_step = stationary.step(_state(1), BootstrapRewardEventsV1())

    def speed_value(ticks: int, velocity: tuple[float, float, float]) -> float:
        kernel = RivalAgencyBootstrapRewardKernelV1()
        kernel.reset(_state(0, velocity=velocity))
        return kernel.step(
            _state(ticks, velocity=velocity), BootstrapRewardEventsV1()
        ).components["useful_speed_rate"]

    touch = RivalAgencyBootstrapRewardKernelV1()
    touch.reset(_state(0, ball_z=300.0, surface=False))
    touch_step = touch.step(
        _state(1, ball_z=300.0, surface=False),
        BootstrapRewardEventsV1(
            raw_touch_records=2,
            logical_touch=True,
            aerial_touch=True,
            touch_chain_length=5,
        ),
    )
    outcomes: dict[str, float] = {}
    for name, event in (
        ("goal", BootstrapRewardEventsV1(goal_for=True)),
        ("concede", BootstrapRewardEventsV1(goal_against=True)),
    ):
        kernel = RivalAgencyBootstrapRewardKernelV1()
        kernel.reset(_state(0, velocity=(0.0, 1800.0, 0.0)))
        outcomes[name] = kernel.step(
            _state(1, velocity=(0.0, 1800.0, 0.0)), event
        ).total
    budget = RivalAgencyBootstrapRewardKernelV1()
    budget.reset(_state(0, velocity=(0.0, 2300.0, 0.0), ball_z=400.0, surface=False))
    for tick in range(1, 5000):
        budget.step(
            _state(
                tick,
                velocity=(0.0, 2300.0, 0.0),
                ball_z=400.0,
                surface=False,
            ),
            BootstrapRewardEventsV1(
                raw_touch_records=1,
                logical_touch=True,
                aerial_touch=True,
                touch_chain_length=5,
            ),
        )
    auditor = RivalLogicalTouchAuditorV1()
    agents = ["blue", "orange"]
    auditor.reset(agents)

    def audit(tick: int, blue: int = 0, orange: int = 0):
        return auditor.process(
            agents,
            tick=tick,
            raw_touch_records={"blue": blue, "orange": orange},
            surface_contact={"blue": False, "orange": True},
            ball_z=300.0,
        )

    first = audit(10, blue=3)
    suppressed = audit(11, blue=1)
    separated = audit(18, blue=1)
    audit(19, orange=1)
    after_opponent = audit(20, blue=1)
    checks = {
        "stationary_unchanged_zero": abs(stationary_step.total) <= 1e-15,
        "toward_speed_exceeds_away": speed_value(4, (0.0, 1200.0, 0.0))
        > speed_value(4, (0.0, -1200.0, 0.0))
        > speed_value(4, (0.0, 0.0, 0.0)),
        "rate_integrates_1_2_4_ticks": abs(
            speed_value(4, (0.0, 1200.0, 0.0))
            - 4.0 * speed_value(1, (0.0, 1200.0, 0.0))
        )
        <= 1e-12,
        "ground_touch_base_exact": touch_step.proposals["ball_touch_event"]
        == BALL_TOUCH_BASE_REWARD,
        "aerial_touch_bonus_exact": touch_step.proposals["aerial_touch_event"]
        == AERIAL_TOUCH_BONUS,
        "chain_schedule_exact": [touch_chain_bonus(index) for index in range(1, 6)]
        == [0.0, 0.10, 0.20, 0.35, 0.50],
        "outcomes_exact": outcomes
        == {"goal": GOAL_REWARD, "concede": CONCEDE_REWARD},
        "combined_absolute_spend_bounded": sum(budget.absolute_spend.values())
        <= COMBINED_SHAPING_ABSOLUTE_EPISODE_BUDGET + 1e-12,
        "continuous_contact_debounced": first["blue"].logical_touch
        and not suppressed["blue"].logical_touch,
        "eight_tick_separation_rewards_second_touch": separated[
            "blue"
        ].touch_chain_length
        == 2,
        "opponent_touch_resets_chain_and_debounce": after_opponent[
            "blue"
        ].logical_touch
        and after_opponent["blue"].touch_chain_length == 1,
        "aerial_requires_physical_touch": first["blue"].aerial_touch,
        "kernel_has_no_action_argument": "action"
        not in inspect.signature(RivalAgencyBootstrapRewardKernelV1.step).parameters,
        "metadata_forbids_action_reward": reward_metadata()[
            "direct_action_press_rewards"
        ]
        is False,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "stationary_components": stationary_step.components,
        "speed": {
            "toward_four_ticks": speed_value(4, (0.0, 1200.0, 0.0)),
            "away_four_ticks": speed_value(4, (0.0, -1200.0, 0.0)),
            "toward_one_tick": speed_value(1, (0.0, 1200.0, 0.0)),
        },
        "touch_proposals": touch_step.proposals,
        "outcomes": outcomes,
        "final_absolute_spend": budget.absolute_spend,
    }


def _timeout_report() -> dict[str, Any]:
    environment = build_v10_bootstrap_env(
        phase="A", seed=20261044, forced_family="natural", forced_mirror=False
    )
    truncated_at: int | None = None
    reason: str | None = None
    try:
        observations = environment.reset()
        for tick in range(1, 1250):
            actions = {agent: np.zeros(8, dtype=np.float32) for agent in observations}
            observations, _, terminated, truncated = environment.step(actions)
            if any(terminated.values()):
                reason = "unexpected_goal"
                break
            if any(truncated.values()):
                truncated_at = tick
                reason = "no_touch_timeout"
                break
    finally:
        environment.close()
    return {
        "status": "passed" if truncated_at is not None and 1199 <= truncated_at <= 1202 else "failed",
        "truncated_at_native_tick": truncated_at,
        "reason": reason,
        "expected_seconds": 10.0,
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


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = load_m10_1_config(args.config)
    exact_start = verify_exact_plus25_start(args.checkpoint, device="cpu")
    source_loaded = load_v9_checkpoint(args.checkpoint, device=args.device)
    migration = config_migration_report(source_loaded["config"], config)
    source_before = checkpoint_record(args.checkpoint, manifest=source_loaded["manifest"])
    production_before = _production_state()
    formulas = _reward_formula_report()
    timeout = _timeout_report()
    engine = RocketSimEngine(rlbot_delay=True)
    team_size = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    distributions = {
        phase: curriculum_distribution_report(
            RivalAgencyBootstrapCurriculumV1(
                phase, seed=int(config["bootstrap"]["seed_base"]) + index * 10_000
            ),
            engine.create_base_state,
            team_size,
            samples=int(args.resets_per_phase),
        )
        for index, phase in enumerate(("A", "B", "C"))
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    disposable = args.disposable_root.resolve() / stamp / f"{source_before['cumulative_agent_steps']:09d}"
    disposable_state = dict(source_loaded["trainer_state"])
    disposable_state.update(
        {
            "v10_1_disposable_preflight": True,
            "v10_1_active_phase": "A",
            "production_promotion_authorized": False,
        }
    )
    disposable_record = save_checkpoint_atomic(
        disposable,
        actor=source_loaded["actor"],
        critic=source_loaded["critic"],
        actor_optimizer=source_loaded["actor_optimizer"],
        critic_optimizer=source_loaded["critic_optimizer"],
        trainer_state=disposable_state,
        config=config,
        reload_observations=source_loaded["reload_observations"],
    )
    disposable_loaded = load_v9_checkpoint(
        disposable, device=args.device, expected_config=config
    )
    trainer = RivalV9PPOTrainer(
        config,
        device=args.device,
        actor=disposable_loaded["actor"],
        critic=disposable_loaded["critic"],
        actor_optimizer=disposable_loaded["actor_optimizer"],
        critic_optimizer=disposable_loaded["critic_optimizer"],
        trainer_state=disposable_loaded["trainer_state"],
        env_factory=ENV_FACTORY_BY_PHASE["A"],
        collect_metrics_fn=collect_v10_bootstrap_metric_vector,
        aggregate_metrics_fn=aggregate_v10_bootstrap_metrics,
    )
    cleanup: dict[str, Any] | None = None
    try:
        shapes = trainer.start_workers()
        health_after_start = trainer.worker_health()
        smoke, _ = trainer.run_iteration(
            rollout_target_agent_steps=int(args.rollout_agent_steps),
            ppo_batch_agent_steps=int(args.rollout_agent_steps),
        )
        health_after_smoke = trainer.worker_health()
    finally:
        cleanup = trainer.cleanup()
    source_after_loaded = load_v9_checkpoint(args.checkpoint, device="cpu")
    source_after = checkpoint_record(
        args.checkpoint, manifest=source_after_loaded["manifest"]
    )
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
        "reward_formula_and_debounce_contract": formulas["status"] == "passed",
        "ten_second_dead_play_timeout": timeout["status"] == "passed",
        "all_three_10000_reset_distributions_pass": all(
            report["checks"]["passed"] for report in distributions.values()
        ),
        "phase_weights_exact": config["bootstrap"]["phase_weights"] == PHASE_WEIGHTS,
        "mirror_and_team_symmetry_statistically_balanced": all(
            report["checks"]["active_team_balance_within_two_percent"]
            and report["checks"]["left_right_ball_distribution_balanced"]
            for report in distributions.values()
        ),
        "exact_m10_plus25_fresh_reload": exact_start["checks"]["fresh_reload_exact"],
        "disposable_copy_is_new_config": disposable_record["config_version"]
        == "RivalM10_1TrainingConfigV1",
        "real_cuda_full_ppo_iteration_healthy": smoke["health"]["passed"]
        and int(smoke["collected_agent_steps"]) >= int(args.rollout_agent_steps),
        "both_action_branches_updated": smoke["health"][
            "all_hybrid_head_gradient_rows_nonzero"
        ],
        "all_56_workers_alive_before_and_after": len(health_after_start) == 56
        and len(health_after_smoke) == 56
        and all(item["alive"] for item in health_after_start + health_after_smoke),
        "worker_cleanup_passed": cleanup is not None and cleanup["passed"],
        "canonical_resume_checkpoint_unchanged": source_before == source_after,
        "frozen_wisp_files_unchanged": production_before["wisp_hashes"]
        == EXPECTED_WISP_HASHES
        == production_after["wisp_hashes"],
        "production_default_unchanged": production_before["probe"]
        == expected_production
        == production_after["probe"],
        "production_promotion_authorized": False,
    }
    result = {
        "schema_version": 1,
        "status": "passed" if all(
            value
            for key, value in checks.items()
            if key != "production_promotion_authorized"
        )
        and checks["production_promotion_authorized"] is False
        else "failed",
        "preflight_version": "RivalAgencyBootstrapPreflightV1",
        "config": {
            "path": "training/configs/milestone10_1.json",
            "file_sha256": sha256_file(args.config),
            "canonical_sha256": config_sha256(config),
            "migration": migration,
            "contract": contract_summary(config),
        },
        "exact_start_checkpoint": exact_start,
        "reward_formulas": formulas,
        "curriculum_distribution": distributions,
        "dead_play_timeout": timeout,
        "visual_inspection": {
            "status": "pending_separate_rlviser_family_sample",
            "required_families": list(PHASE_WEIGHTS["A"]),
        },
        "disposable_cuda_smoke": {
            "checkpoint": disposable_record,
            "environment_shapes": shapes,
            "health_after_start": health_after_start,
            "iteration": compact_training_iteration(smoke),
            "health_after_smoke": health_after_smoke,
            "cleanup": cleanup,
            "experience_counted_toward_campaign": False,
        },
        "source_checkpoint_before": source_before,
        "source_checkpoint_after": source_after,
        "production_before": production_before,
        "production_after": production_after,
        "checks": checks,
    }
    checks["passed"] = result["status"] == "passed"
    write_json_atomic(args.output, result)
    if result["status"] != "passed":
        raise RuntimeError(f"v10.1 preflight failed: {checks}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=M10_PLUS25_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--disposable-root", type=Path, default=DEFAULT_DISPOSABLE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resets-per-phase", type=int, default=10_000)
    parser.add_argument("--rollout-agent-steps", type=int, default=192_000)
    args = parser.parse_args()
    if args.resets_per_phase < 10_000:
        raise ValueError("Preflight requires at least 10,000 resets per phase")
    if args.rollout_agent_steps != 192_000:
        raise ValueError("Preflight requires one full frozen 192k PPO iteration")
    report = run_preflight(args)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
