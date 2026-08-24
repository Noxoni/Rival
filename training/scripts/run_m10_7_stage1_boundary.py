"""Train M10.7 Stage 1 to one clean recoverable boundary."""

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
from rival_training.v10_7_campaign import (  # noqa: E402
    ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR,
    BOUNDARY_HOURS,
    CHECKPOINT_ROOT,
    DEFAULT_STAGE1_CONFIG,
    RESULT_ROOT,
    load_stage1_config,
)
from rival_training.v10_7_checkpoint import (  # noqa: E402
    checkpoint_record,
    load_checkpoint,
    save_checkpoint_atomic,
    verify_reload_parity,
)
from rival_training.v10_7_trainer import RivalV10_7PPOTrainer  # noqa: E402


CAMPAIGN_STATE = RESULT_ROOT / "stage1_campaign_state.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(hours: float) -> str:
    if float(hours) not in BOUNDARY_HOURS:
        raise ValueError(f"Unsupported M10.7 boundary: {hours}")
    return {0.5: "plus-000p5h", 1.0: "plus-001h", 2.5: "plus-002p5h"}[
        float(hours)
    ]


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
        "rollout_inference": report["rollout_inference"],
        "ppo": {
            "actor_loss": report["ppo"]["actor_loss"],
            "critic_loss": report["ppo"]["critic_loss"],
            "analog_entropy": report["ppo"]["analog_entropy"],
            "button_entropy": report["ppo"]["button_entropy"],
            "button_entropy_by_field": report["ppo"]["button_entropy_by_field"],
            "button_entropy_coefficient": report["ppo"][
                "button_entropy_coefficient"
            ],
            "button_entropy_schedule_active_learner_step": report["ppo"][
                "button_entropy_schedule_active_learner_step"
            ],
            "actor_update_magnitude": report["ppo"]["actor_update_magnitude"],
            "critic_update_magnitude": report["ppo"]["critic_update_magnitude"],
            "controller_branch_gradient_absolute_sums": report["ppo"][
                "controller_branch_gradient_absolute_sums"
            ],
        },
        "actions": report["actions"],
        "health": report["health"],
    }


def run_boundary(args: argparse.Namespace) -> dict[str, Any]:
    boundary_hours = float(args.boundary_hours)
    slug = _slug(boundary_hours)
    target_steps = int(round(boundary_hours * ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR))
    config = load_stage1_config(args.config)
    preflight = _read(RESULT_ROOT / "preflight.json")
    if preflight.get("status") != "passed" or preflight.get("ppo_authorized") is not True:
        raise RuntimeError("M10.7 PPO is forbidden until the complete preflight passes")
    source = args.source_checkpoint.resolve()
    loaded = load_checkpoint(source, device=args.device, expected_config=config)
    trainer = RivalV10_7PPOTrainer(
        config,
        device=args.device,
        actor=loaded["actor"],
        critic=loaded["critic"],
        actor_optimizer=loaded["actor_optimizer"],
        critic_optimizer=loaded["critic_optimizer"],
        trainer_state=loaded["trainer_state"],
        env_factory=make_ball_acquisition_phase_a_env,
    )
    if trainer.cumulative_agent_steps >= target_steps:
        raise RuntimeError("Source checkpoint already reached this boundary")
    if trainer.cumulative_agent_steps > 1_080_000:
        raise RuntimeError("Source checkpoint exceeds M10.7 Stage-1 authority")
    source_record = checkpoint_record(source, manifest=loaded["manifest"])
    rolling_root = args.checkpoint_root.resolve() / "rolling"
    boundary_parent = args.checkpoint_root.resolve() / "boundaries" / slug
    iterations = []
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
            if rollout <= 0:
                break
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
            state = trainer.trainer_state()
            state.update(
                {
                    "campaign_id": config["campaign_id"],
                    "stage": 1,
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
                trainer_state=state,
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
                CAMPAIGN_STATE,
                {
                    "format": "rival-m10-7-stage1-state-v1",
                    "campaign_id": config["campaign_id"],
                    "current_phase": "training",
                    "current_boundary_hours": boundary_hours,
                    "cumulative_active_learner_steps": trainer.cumulative_agent_steps,
                    "simulated_game_hours": trainer.cumulative_agent_steps
                    / ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR,
                    "latest_clean_recovery_checkpoint": record,
                    "production_promotion_authorized": False,
                },
            )
            print(
                json.dumps(
                    {
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
        raise RuntimeError(f"M10.7 worker cleanup failed: {cleanup}")
    if latest_held is None:
        raise RuntimeError("M10.7 boundary made no recoverable PPO update")
    final_state = trainer.trainer_state()
    final_state.update(
        {
            "campaign_id": config["campaign_id"],
            "stage": 1,
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
    shortfall = max(0, target_steps - trainer.cumulative_agent_steps)
    reached = trainer.cumulative_agent_steps >= target_steps
    result = {
        "schema_version": 1,
        "training_boundary_version": "RivalM10_7Stage1TrainingBoundaryV1",
        "status": "passed" if reached else "failed",
        "stage": 1,
        "phase": "A",
        "boundary_hours": boundary_hours,
        "target_active_learner_steps": target_steps,
        "reached_active_learner_steps": trainer.cumulative_agent_steps,
        "reached_simulated_hours": trainer.cumulative_agent_steps
        / ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR,
        "shortfall_active_learner_steps": shortfall,
        "source_checkpoint": source_record,
        "environment_shapes": shapes,
        "iterations": iterations,
        "rolling_checkpoints_written": rolling_records,
        "rolling_checkpoints_removed": removed,
        "immutable_checkpoint": immutable,
        "immutable_checkpoint_reload": parity,
        "worker_cleanup": cleanup,
        "checks": {
            "boundary_reached": reached,
            "stage_experience_ceiling_respected": trainer.cumulative_agent_steps
            <= 1_080_112,
            "all_iterations_healthy": all(row["health"]["passed"] for row in iterations),
            "checkpoint_reload_exact": parity["checks"]["passed"],
            "workers_cleaned": cleanup["passed"],
            "production_promotion_authorized": False,
        },
    }
    result["checks"]["passed"] = (
        all(
            value
            for key, value in result["checks"].items()
            if key != "production_promotion_authorized"
        )
        and result["checks"]["production_promotion_authorized"] is False
    )
    output = args.output or RESULT_ROOT / "stage_1" / f"training_{slug}.json"
    write_json_atomic(output, result)
    write_json_atomic(
        CAMPAIGN_STATE,
        {
            "format": "rival-m10-7-stage1-state-v1",
            "campaign_id": config["campaign_id"],
            "current_phase": "pending_evaluation",
            "current_boundary_hours": boundary_hours,
            "cumulative_active_learner_steps": trainer.cumulative_agent_steps,
            "simulated_game_hours": trainer.cumulative_agent_steps
            / ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR,
            "latest_clean_recovery_checkpoint": immutable,
            "production_promotion_authorized": False,
        },
    )
    if not result["checks"]["passed"]:
        raise RuntimeError(f"M10.7 boundary failed: {result['checks']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-hours", type=float, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_STAGE1_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_boundary(args)
    print(
        json.dumps(
            {
                "status": report["status"],
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
