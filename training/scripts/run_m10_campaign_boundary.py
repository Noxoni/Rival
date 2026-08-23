"""Train the frozen scratch foundation to one immutable M10 hour boundary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import (  # noqa: E402
    DEFAULT_M10_CONFIG_PATH,
    M09_FINAL_STEPS,
    boundary_slug,
    checkpoint_record,
    compact_training_iteration,
    load_m10_config,
    m10_config_migration_report,
    nominal_boundary_steps,
    portable_path,
    prune_rolling_checkpoints,
    save_checkpoint_atomic,
    verify_checkpoint_reload_parity,
    verify_exact_m09_start_checkpoint,
    write_json_atomic,
)
from rival_training.v9_checkpoint import load_v9_checkpoint  # noqa: E402
from rival_training.v9_trainer import RivalV9PPOTrainer  # noqa: E402


DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "training/checkpoints/milestone10"
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "training/results/raw/milestone10"


def _console_iteration(report: dict[str, Any], boundary: int) -> dict[str, Any]:
    return {
        "boundary_added_hours": boundary,
        "iteration": report["iteration"],
        "collected_agent_steps": report["collected_agent_steps"],
        "cumulative_agent_steps": report["cumulative_agent_steps"],
        "added_simulated_game_hours": (
            int(report["cumulative_agent_steps"]) - M09_FINAL_STEPS
        )
        / 864000.0,
        "agent_steps_per_second": report["agent_steps_per_second"],
        "iteration_wall_seconds": report["iteration_wall_seconds"],
        "reward_mean": report["reward"]["mean"],
        "actor_loss_mean": report["ppo"]["actor_loss"]["mean"],
        "critic_loss_mean": report["ppo"]["critic_loss"]["mean"],
        "health": report["health"]["passed"],
    }


def _lineage(source_state: dict[str, Any], source_record: dict[str, Any]) -> dict[str, Any]:
    existing = source_state.get("m10_initial_m09_checkpoint")
    if isinstance(existing, dict):
        return dict(existing)
    return dict(source_record)


def run_boundary(args: argparse.Namespace) -> dict[str, Any]:
    config = load_m10_config(args.config)
    boundary = int(args.boundary_added_hours)
    slug = boundary_slug(boundary)
    source_checkpoint = args.source_checkpoint.resolve()
    loaded = load_v9_checkpoint(source_checkpoint, device=args.device)
    migration = m10_config_migration_report(loaded["config"], config)
    source_state = loaded["trainer_state"]
    source_record = checkpoint_record(source_checkpoint, manifest=loaded["manifest"])
    if loaded["config"]["config_version"] == "RivalM09TrainingConfigV2PilotCurriculum":
        exact_start = verify_exact_m09_start_checkpoint(source_checkpoint, device="cpu")
    else:
        exact_start = None
        verify_checkpoint_reload_parity(
            source_checkpoint, expected_config=config, device="cpu"
        )

    source_steps = int(source_state["cumulative_agent_steps"])
    nominal_target = nominal_boundary_steps(boundary)
    final_nominal_target = int(config["campaign"]["nominal_cumulative_agent_step_target"])
    rollout_target = int(config["ppo"]["rollout_agent_steps_per_iteration"])
    worker_count = int(config["backend"]["worker_count"])
    final_alignment_ceiling = final_nominal_target + rollout_target + 2 * worker_count
    if source_steps < M09_FINAL_STEPS:
        raise RuntimeError("M10 source checkpoint predates the exact final M09 Gate 13 state")
    if source_steps >= nominal_target:
        raise RuntimeError(
            f"Source already reached {source_steps} steps for nominal {nominal_target}; "
            "refusing to create a second immutable boundary"
        )
    if source_steps > final_alignment_ceiling:
        raise RuntimeError("M10 source checkpoint exceeds normal final-iteration alignment")

    checkpoint_root = args.checkpoint_root.resolve()
    rolling_root = checkpoint_root / "rolling"
    boundary_parent = checkpoint_root / "boundaries" / slug
    raw_root = args.raw_root.resolve() / slug
    iteration_root = raw_root / "iterations"
    progress_path = raw_root / "training_progress.json"
    rolling_root.mkdir(parents=True, exist_ok=True)
    iteration_root.mkdir(parents=True, exist_ok=True)

    trainer = RivalV9PPOTrainer(
        config,
        device=args.device,
        actor=loaded["actor"],
        critic=loaded["critic"],
        actor_optimizer=loaded["actor_optimizer"],
        critic_optimizer=loaded["critic_optimizer"],
        trainer_state=source_state,
    )
    initial_identity = _lineage(source_state, source_record)
    prior_boundaries = list(source_state.get("m10_boundary_history", []))
    compact_iterations: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    removed_rolling: list[str] = []
    shapes = trainer.start_workers()
    cleanup: dict[str, Any] | None = None
    try:
        maximum_iterations = math.ceil((nominal_target - source_steps) / rollout_target) + 2
        while trainer.cumulative_agent_steps < nominal_target:
            if len(compact_iterations) >= maximum_iterations:
                raise RuntimeError("M10 boundary failed to advance within the alignment guard")
            iteration, held_observations = trainer.run_iteration(
                maximum_cumulative_agent_steps=final_alignment_ceiling
            )
            compact = compact_training_iteration(iteration)
            compact_iterations.append(compact)
            write_json_atomic(
                iteration_root / f"{trainer.cumulative_agent_steps:09d}.json", iteration
            )
            trainer_state = trainer.trainer_state()
            trainer_state.update(
                {
                    "m10_initial_m09_checkpoint": initial_identity,
                    "m10_latest_resume_source": source_record,
                    "m10_config_migration": migration,
                    "m10_boundary_history": prior_boundaries,
                    "m10_active_boundary_added_simulated_hours": boundary,
                    "m10_added_agent_steps": trainer.cumulative_agent_steps
                    - M09_FINAL_STEPS,
                    "m10_added_simulated_game_hours": (
                        trainer.cumulative_agent_steps - M09_FINAL_STEPS
                    )
                    / 864000.0,
                    "production_promotion_authorized": False,
                }
            )
            rolling_checkpoint = rolling_root / f"{trainer.cumulative_agent_steps:09d}"
            rolling_record = save_checkpoint_atomic(
                rolling_checkpoint,
                actor=trainer.actor,
                critic=trainer.critic,
                actor_optimizer=trainer.actor_optimizer,
                critic_optimizer=trainer.critic_optimizer,
                trainer_state=trainer_state,
                config=config,
                reload_observations=held_observations,
            )
            checkpoint_records.append(rolling_record)
            removed_rolling.extend(
                prune_rolling_checkpoints(
                    rolling_root,
                    keep=int(config["campaign"]["rolling_recovery_checkpoints_to_keep"]),
                )
            )
            progress = {
                "schema_version": 1,
                "status": "in_progress",
                "boundary_added_simulated_hours": boundary,
                "source_checkpoint": source_record,
                "nominal_target_cumulative_agent_steps": nominal_target,
                "final_alignment_ceiling_agent_steps": final_alignment_ceiling,
                "environment_shapes": shapes,
                "iterations": compact_iterations,
                "rolling_checkpoints_written": checkpoint_records,
                "rolling_checkpoints_removed": removed_rolling,
                "latest_rolling_checkpoint": rolling_record,
            }
            write_json_atomic(progress_path, progress)
            print(json.dumps(_console_iteration(iteration, boundary)), flush=True)

        reached_steps = trainer.cumulative_agent_steps
        boundary_history = prior_boundaries + [
            {
                "boundary_added_simulated_hours": boundary,
                "nominal_cumulative_agent_steps": nominal_target,
                "achieved_cumulative_agent_steps": reached_steps,
                "achieved_added_simulated_game_hours": (reached_steps - M09_FINAL_STEPS)
                / 864000.0,
            }
        ]
        boundary_state = trainer.trainer_state()
        boundary_state.update(
            {
                "m10_initial_m09_checkpoint": initial_identity,
                "m10_latest_resume_source": source_record,
                "m10_config_migration": migration,
                "m10_boundary_history": boundary_history,
                "m10_latest_completed_boundary_added_simulated_hours": boundary,
                "m10_added_agent_steps": reached_steps - M09_FINAL_STEPS,
                "m10_added_simulated_game_hours": (reached_steps - M09_FINAL_STEPS)
                / 864000.0,
                "production_promotion_authorized": False,
            }
        )
        immutable_checkpoint = boundary_parent / f"{reached_steps:09d}"
        immutable_record = save_checkpoint_atomic(
            immutable_checkpoint,
            actor=trainer.actor,
            critic=trainer.critic,
            actor_optimizer=trainer.actor_optimizer,
            critic_optimizer=trainer.critic_optimizer,
            trainer_state=boundary_state,
            config=config,
            reload_observations=held_observations,
        )
    finally:
        cleanup = trainer.cleanup()
    if cleanup is None or not cleanup["passed"]:
        raise RuntimeError(f"M10 worker cleanup failed: {cleanup}")

    reload_parity = verify_checkpoint_reload_parity(
        immutable_checkpoint, expected_config=config, device="cpu"
    )
    result = {
        "schema_version": 1,
        "status": "passed",
        "campaign_version": "RivalM10CampaignV1",
        "boundary_added_simulated_hours": boundary,
        "source_checkpoint": source_record,
        "exact_m09_start_verification": exact_start,
        "config_migration": migration,
        "nominal_target_cumulative_agent_steps": nominal_target,
        "final_alignment_ceiling_agent_steps": final_alignment_ceiling,
        "achieved": {
            "cumulative_agent_steps": reached_steps,
            "additional_agent_steps_from_m09": reached_steps - M09_FINAL_STEPS,
            "cumulative_simulated_game_hours": reached_steps / 864000.0,
            "additional_simulated_game_hours_from_m09": (
                reached_steps - M09_FINAL_STEPS
            )
            / 864000.0,
            "normal_iteration_alignment_agent_steps": reached_steps - nominal_target,
        },
        "environment_shapes": shapes,
        "iterations": compact_iterations,
        "rolling_checkpoints_written": checkpoint_records,
        "rolling_checkpoints_removed": removed_rolling,
        "immutable_checkpoint": immutable_record,
        "immutable_checkpoint_fresh_reload": reload_parity,
        "cleanup": cleanup,
        "checks": {
            "all_iterations_healthy": all(item["health"]["passed"] for item in compact_iterations),
            "boundary_reached": reached_steps >= nominal_target,
            "within_normal_final_iteration_alignment": reached_steps <= final_alignment_ceiling,
            "immutable_checkpoint_reload_exact": reload_parity["checks"]["passed"],
            "rolling_retention_at_least_two": int(
                config["campaign"]["rolling_recovery_checkpoints_to_keep"]
            )
            >= 2,
            "production_promotion_authorized": False,
        },
    }
    result["checks"]["passed"] = all(
        value for key, value in result["checks"].items() if key != "production_promotion_authorized"
    ) and result["checks"]["production_promotion_authorized"] is False
    if not result["checks"]["passed"]:
        raise RuntimeError(f"M10 boundary result failed: {result['checks']}")
    write_json_atomic(progress_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-added-hours", type=int, choices=(5, 10, 25, 50, 100), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_M10_CONFIG_PATH)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_boundary(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "boundary_added_simulated_hours": report[
                    "boundary_added_simulated_hours"
                ],
                "immutable_checkpoint": report["immutable_checkpoint"],
                "raw_report": portable_path(
                    args.raw_root.resolve()
                    / boundary_slug(args.boundary_added_hours)
                    / "training_progress.json"
                ),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
