"""Run one fresh-parent, checkpoint-every-iteration Gate 13 phase."""

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

from rival_training.v9_checkpoint import (  # noqa: E402
    DEFAULT_PILOT_CONFIG_PATH,
    load_m09_config,
    load_v9_checkpoint,
    pilot_config_migration_report,
    portable_path,
    save_v9_checkpoint,
    sha256_file,
)
from rival_training.v9_trainer import RivalV9PPOTrainer  # noqa: E402


def _write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _parse_targets(value: str) -> list[int]:
    targets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not targets or any(target <= 0 for target in targets):
        raise ValueError("--rollout-targets must contain positive integers")
    return targets


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "iteration": report["iteration"],
        "collected_agent_steps": report["collected_agent_steps"],
        "cumulative_agent_steps": report["cumulative_agent_steps"],
        "simulated_game_hours": report["simulated_game_hours"],
        "agent_steps_per_second": report["agent_steps_per_second"],
        "actor_update_magnitude": report["ppo"]["actor_update_magnitude"],
        "critic_update_magnitude": report["ppo"]["critic_update_magnitude"],
        "reward_mean": report["reward"]["mean"],
        "health": report["health"]["passed"],
    }


def run_phase(args: argparse.Namespace) -> dict[str, Any]:
    target_config = load_m09_config(args.config)
    loaded = load_v9_checkpoint(args.source_checkpoint, device=args.device)
    migration = pilot_config_migration_report(loaded["config"], target_config)
    source_state = loaded["trainer_state"]
    maximum_steps = int(target_config["pilot"]["maximum_cumulative_agent_steps"])
    if int(source_state["cumulative_agent_steps"]) >= maximum_steps:
        raise RuntimeError("Source checkpoint has no authorized Gate 13 budget left")
    targets = _parse_targets(args.rollout_targets)
    minibatch = int(target_config["ppo"]["minibatch_agent_steps"])
    if any(target % minibatch for target in targets):
        raise ValueError("Every pilot rollout target must preserve complete 48k minibatches")

    trainer = RivalV9PPOTrainer(
        target_config,
        device=args.device,
        actor=loaded["actor"],
        critic=loaded["critic"],
        actor_optimizer=loaded["actor_optimizer"],
        critic_optimizer=loaded["critic_optimizer"],
        trainer_state=source_state,
    )
    prior_gate11 = list(source_state.get("gate11_iteration_reports", []))
    prior_gate13 = list(source_state.get("gate13_iteration_reports", []))
    reports: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    shapes = trainer.start_workers()
    cleanup: dict[str, Any] | None = None
    try:
        for phase_index, target in enumerate(targets, start=1):
            iteration, held_observations = trainer.run_iteration(
                rollout_target_agent_steps=target,
                ppo_batch_agent_steps=target,
                maximum_cumulative_agent_steps=maximum_steps,
            )
            iteration["gate13_phase"] = str(args.phase)
            iteration["gate13_phase_iteration"] = phase_index
            reports.append(iteration)
            trainer_state = trainer.trainer_state()
            trainer_state.update(
                {
                    "gate11_iteration_reports": prior_gate11,
                    "gate13_iteration_reports": prior_gate13 + reports,
                    "gate13_config_migration": migration,
                    "gate13_initial_source_checkpoint": source_state.get(
                        "gate13_initial_source_checkpoint",
                        {
                            "directory": portable_path(args.source_checkpoint),
                            "manifest_sha256": sha256_file(
                                Path(args.source_checkpoint) / "checkpoint_manifest.json"
                            ),
                            "cumulative_agent_steps": int(source_state["cumulative_agent_steps"]),
                        },
                    ),
                    "gate13_latest_phase": str(args.phase),
                    "gate13_latest_phase_iteration": phase_index,
                    "production_promotion_authorized": False,
                }
            )
            checkpoint = args.checkpoint_root / (f"{trainer.cumulative_agent_steps:07d}")
            manifest = save_v9_checkpoint(
                checkpoint,
                actor=trainer.actor,
                critic=trainer.critic,
                actor_optimizer=trainer.actor_optimizer,
                critic_optimizer=trainer.critic_optimizer,
                trainer_state=trainer_state,
                config=target_config,
                reload_observations=held_observations,
            )
            checkpoint_record = {
                "directory": portable_path(checkpoint),
                "manifest_sha256": sha256_file(checkpoint / "checkpoint_manifest.json"),
                "actor_sha256": manifest["files"]["actor.pt"]["sha256"],
                "actor_size_bytes": manifest["files"]["actor.pt"]["size_bytes"],
                "cumulative_agent_steps": trainer.cumulative_agent_steps,
                "simulated_game_hours": trainer.cumulative_agent_steps / 864000.0,
            }
            checkpoints.append(checkpoint_record)
            progress = {
                "schema_version": 1,
                "status": "in_progress",
                "phase": str(args.phase),
                "source_checkpoint": portable_path(args.source_checkpoint),
                "rollout_targets": targets,
                "config_migration": migration,
                "environment_shapes": shapes,
                "iterations": reports,
                "checkpoints": checkpoints,
                "latest_checkpoint": checkpoint_record,
            }
            _write_progress(args.output, progress)
            print(json.dumps(_compact(iteration)), flush=True)
    finally:
        cleanup = trainer.cleanup()
    if not cleanup["passed"]:
        raise RuntimeError(f"Gate 13 phase worker cleanup failed: {cleanup}")
    result = {
        "schema_version": 1,
        "status": "passed",
        "phase": str(args.phase),
        "source_checkpoint": portable_path(args.source_checkpoint),
        "rollout_targets": targets,
        "config_migration": migration,
        "environment_shapes": shapes,
        "iterations": reports,
        "checkpoints": checkpoints,
        "latest_checkpoint": checkpoints[-1],
        "cleanup": cleanup,
    }
    _write_progress(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-targets", required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_PILOT_CONFIG_PATH)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_phase(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "phase": report["phase"],
                "latest_checkpoint": report["latest_checkpoint"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
