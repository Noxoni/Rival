"""Train one isolated M10.8 GAE arm to a clean Stage-1 boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import (  # noqa: E402
    prune_rolling_checkpoints,
    write_json_atomic,
)
from rival_training.v10_6_environment import (  # noqa: E402
    make_ball_acquisition_phase_a_env,
)
from rival_training.v10_7_checkpoint import (  # noqa: E402
    checkpoint_record,
    load_checkpoint,
    save_checkpoint_atomic,
    verify_reload_parity,
)
from rival_training.v10_8_campaign import (  # noqa: E402
    ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR,
    ARM_LAMBDAS,
    BOUNDARY_HOURS,
    CHECKPOINT_ROOT,
    RESULT_ROOT,
    arm_slug,
    load_arm_config,
)
from rival_training.v10_8_credit import (  # noqa: E402
    aggregate_credit_diagnostics,
)
from rival_training.v10_8_trainer import RivalV10_8PPOTrainer  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _boundary_slug(hours: float) -> str:
    if float(hours) not in BOUNDARY_HOURS:
        raise ValueError(f"Unsupported M10.8 boundary: {hours}")
    return {0.5: "plus-000p5h", 1.0: "plus-001h"}[float(hours)]


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "iteration": report["iteration"],
        "collected_active_learner_steps": report["collected_agent_steps"],
        "experience_records": report["experience_records"],
        "cumulative_active_learner_steps": report["cumulative_agent_steps"],
        "simulated_game_hours": report["simulated_game_hours"],
        "collection_seconds": report["collection_seconds"],
        "update_wall_seconds": report["update_wall_seconds"],
        "iteration_wall_seconds": report["iteration_wall_seconds"],
        "active_learner_steps_per_second": report["agent_steps_per_second"],
        "cuda_peak_allocated_mib": report["cuda_peak_allocated_mib"],
        "reward": report["reward"],
        "gae": report["gae"],
        "rollout_inference": report["rollout_inference"],
        "ppo": report["ppo"],
        "actions": report["actions"],
        "credit_assignment": report["credit_assignment"],
        "health": report["health"],
    }


def run_boundary(args: argparse.Namespace) -> dict[str, Any]:
    arm = str(args.arm).upper()
    expected_lambda = ARM_LAMBDAS[arm]
    arm_directory = arm_slug(arm)
    boundary_hours = float(args.boundary_hours)
    slug = _boundary_slug(boundary_hours)
    target_steps = int(round(boundary_hours * ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR))
    config = load_arm_config(arm, args.config)
    preflight = _read(RESULT_ROOT / "preflight.json")
    if preflight.get("status") != "passed" or preflight.get("ppo_authorized") is not True:
        raise RuntimeError("M10.8 PPO is forbidden until preflight passes")
    source = args.source_checkpoint.resolve()
    loaded = load_checkpoint(source, device=args.device, expected_config=config)
    state = loaded["trainer_state"]
    if state.get("m10_8_arm") != arm:
        raise RuntimeError(f"M10.8 arm {arm} cannot load another arm's checkpoint")
    if float(loaded["config"]["ppo"]["gae_lambda"]) != expected_lambda:
        raise RuntimeError("Checkpoint lambda does not match the selected arm")
    expected_source_steps = 0 if boundary_hours == 0.5 else 216_000
    if int(state["cumulative_agent_steps"]) < expected_source_steps:
        raise RuntimeError("Source checkpoint has not reached the preceding boundary")
    if int(state["cumulative_agent_steps"]) >= target_steps:
        raise RuntimeError("Source checkpoint already reached this boundary")

    trainer = RivalV10_8PPOTrainer(
        config,
        device=args.device,
        actor=loaded["actor"],
        critic=loaded["critic"],
        actor_optimizer=loaded["actor_optimizer"],
        critic_optimizer=loaded["critic_optimizer"],
        trainer_state=state,
        env_factory=make_ball_acquisition_phase_a_env,
    )
    source_record = checkpoint_record(source, manifest=loaded["manifest"])
    checkpoint_root = args.checkpoint_root.resolve() / "arms" / arm_directory
    rolling_root = checkpoint_root / "rolling"
    boundary_parent = checkpoint_root / "boundaries" / slug
    result_root = RESULT_ROOT / "arms" / arm_directory
    campaign_state = result_root / "campaign_state.json"
    iterations: list[dict[str, Any]] = []
    rolling_records = []
    removed = []
    latest_held = None
    cleanup = None
    shapes = None
    try:
        shapes = trainer.start_workers()
        while trainer.cumulative_agent_steps < target_steps:
            remaining = target_steps - trainer.cumulative_agent_steps
            rollout = min(
                int(config["ppo"]["rollout_agent_steps_per_iteration"]), remaining
            )
            minibatch = int(config["ppo"]["minibatch_agent_steps"])
            batch = (rollout // minibatch) * minibatch or rollout
            iteration, held = trainer.run_iteration(
                rollout_target_agent_steps=rollout,
                ppo_batch_agent_steps=batch,
                maximum_cumulative_agent_steps=target_steps
                + 2 * int(config["backend"]["worker_count"]),
            )
            latest_held = held
            compact = _compact(iteration)
            iterations.append(compact)
            current_state = trainer.trainer_state()
            current_state.update(
                {
                    "campaign_id": config["campaign_id"],
                    "stage": 1,
                    "m10_8_arm": arm,
                    "gae_lambda": expected_lambda,
                    "source_boundary_checkpoint": source_record,
                    "active_boundary_hours": boundary_hours,
                    "production_promotion_authorized": False,
                }
            )
            destination = rolling_root / f"{trainer.cumulative_agent_steps:09d}"
            record = save_checkpoint_atomic(
                destination,
                actor=trainer.actor,
                critic=trainer.critic,
                actor_optimizer=trainer.actor_optimizer,
                critic_optimizer=trainer.critic_optimizer,
                trainer_state=current_state,
                config=config,
                reload_observations=held,
            )
            rolling_records.append(record)
            removed.extend(
                prune_rolling_checkpoints(
                    rolling_root,
                    keep=int(
                        config["stage_contract"][
                            "rolling_recovery_checkpoints_to_keep"
                        ]
                    ),
                )
            )
            write_json_atomic(
                campaign_state,
                {
                    "format": "rival-m10-8-arm-state-v1",
                    "arm": arm,
                    "lambda": expected_lambda,
                    "phase": "training",
                    "boundary_hours": boundary_hours,
                    "cumulative_active_learner_steps": trainer.cumulative_agent_steps,
                    "latest_clean_recovery_checkpoint": record,
                    "stage_2_authorized": False,
                    "production_promotion_authorized": False,
                },
            )
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "boundary_hours": boundary_hours,
                        "iteration": compact,
                        "rolling_checkpoint": record,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        cleanup = trainer.cleanup()
    if cleanup is None or not cleanup["passed"]:
        raise RuntimeError(f"M10.8 worker cleanup failed for arm {arm}: {cleanup}")
    if latest_held is None:
        raise RuntimeError("M10.8 boundary made no recoverable PPO update")

    final_state = trainer.trainer_state()
    final_state.update(
        {
            "campaign_id": config["campaign_id"],
            "stage": 1,
            "m10_8_arm": arm,
            "gae_lambda": expected_lambda,
            "source_boundary_checkpoint": source_record,
            "completed_boundary_hours": boundary_hours,
            "production_promotion_authorized": False,
        }
    )
    destination = boundary_parent / f"{trainer.cumulative_agent_steps:09d}"
    immutable = save_checkpoint_atomic(
        destination,
        actor=trainer.actor,
        critic=trainer.critic,
        actor_optimizer=trainer.actor_optimizer,
        critic_optimizer=trainer.critic_optimizer,
        trainer_state=final_state,
        config=config,
        reload_observations=latest_held,
    )
    parity = verify_reload_parity(destination, expected_config=config, device="cpu")
    reached = trainer.cumulative_agent_steps >= target_steps
    result = {
        "schema_version": 1,
        "training_boundary_version": "RivalM10_8GAEArmTrainingBoundaryV1",
        "status": "passed" if reached else "failed",
        "arm": arm,
        "gae_lambda": expected_lambda,
        "gamma": float(config["ppo"]["gamma"]),
        "boundary_hours": boundary_hours,
        "target_active_learner_steps": target_steps,
        "reached_active_learner_steps": trainer.cumulative_agent_steps,
        "reached_simulated_hours": trainer.cumulative_agent_steps
        / ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR,
        "source_checkpoint": source_record,
        "environment_shapes": shapes,
        "iterations": iterations,
        "aggregate_credit_assignment": aggregate_credit_diagnostics(
            row["credit_assignment"] for row in iterations
        ),
        "rolling_checkpoints_written": rolling_records,
        "rolling_checkpoints_removed": removed,
        "immutable_checkpoint": immutable,
        "immutable_checkpoint_reload": parity,
        "worker_cleanup": cleanup,
        "checks": {
            "boundary_reached": reached,
            "one_hour_ceiling_respected": trainer.cumulative_agent_steps <= 432_112,
            "all_iterations_healthy": all(row["health"]["passed"] for row in iterations),
            "all_credit_diagnostics_passed": all(
                row["credit_assignment"]["checks"]["passed"] for row in iterations
            ),
            "lambda_exact": all(
                row["gae"]["lambda"] == expected_lambda for row in iterations
            ),
            "checkpoint_reload_exact": parity["checks"]["passed"],
            "workers_cleaned": cleanup["passed"],
            "stage_2_authorized": False,
            "production_promotion_authorized": False,
        },
    }
    result["checks"]["passed"] = all(
        value
        for key, value in result["checks"].items()
        if key not in {"stage_2_authorized", "production_promotion_authorized"}
    )
    output = args.output or result_root / f"training_{slug}.json"
    write_json_atomic(output, result)
    write_json_atomic(
        campaign_state,
        {
            "format": "rival-m10-8-arm-state-v1",
            "arm": arm,
            "lambda": expected_lambda,
            "phase": "pending_evaluation",
            "boundary_hours": boundary_hours,
            "latest_clean_recovery_checkpoint": immutable,
            "stage_2_authorized": False,
            "production_promotion_authorized": False,
        },
    )
    if not result["checks"]["passed"]:
        raise RuntimeError(f"M10.8 arm {arm} boundary failed: {result['checks']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARM_LAMBDAS), required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-hours", type=float, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "training/configs/milestone10_8_stage1.json",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_boundary(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "arm": report["arm"],
                "boundary_hours": report["boundary_hours"],
                "checkpoint": report["immutable_checkpoint"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
