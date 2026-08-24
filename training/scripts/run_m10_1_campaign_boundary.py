"""Train Rival v10.1 to one clean agency-bootstrap evaluation boundary."""

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
    checkpoint_record,
    portable_path,
    prune_rolling_checkpoints,
    save_checkpoint_atomic,
    verify_checkpoint_reload_parity,
    write_json_atomic,
)
from rival_training.v10_1_campaign import (  # noqa: E402
    AGENT_STEPS_PER_SIMULATED_HOUR,
    DEFAULT_CONFIG_PATH,
    M10_PLUS25_STEPS,
    boundary_slug,
    compact_training_iteration,
    config_migration_report,
    load_m10_1_config,
    nominal_boundary_steps,
    verify_exact_plus25_start,
)
from rival_training.v10_bootstrap_environment import (  # noqa: E402
    ENV_FACTORY_BY_PHASE,
)
from rival_training.v10_bootstrap_metrics import (  # noqa: E402
    aggregate_v10_bootstrap_metrics,
    collect_v10_bootstrap_metric_vector,
)
from rival_training.v9_checkpoint import load_v9_checkpoint  # noqa: E402
from rival_training.v9_trainer import RivalV9PPOTrainer  # noqa: E402


DEFAULT_CHECKPOINT_ROOT = REPOSITORY_ROOT / "training/checkpoints/milestone10_1"
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "training/results/raw/milestone10_1"


def _console(report: dict[str, Any], boundary: float, phase: str) -> dict[str, Any]:
    metrics = report["pilot_metrics"]
    return {
        "boundary_added_bootstrap_hours": boundary,
        "phase": phase,
        "iteration": report["iteration"],
        "collected_agent_steps": report["collected_agent_steps"],
        "cumulative_agent_steps": report["cumulative_agent_steps"],
        "added_bootstrap_simulated_game_hours": (
            int(report["cumulative_agent_steps"]) - M10_PLUS25_STEPS
        )
        / AGENT_STEPS_PER_SIMULATED_HOUR,
        "agent_steps_per_second": report["agent_steps_per_second"],
        "iteration_wall_seconds": report["iteration_wall_seconds"],
        "reward_mean": report["reward"]["mean"],
        "logical_touches_per_100k": metrics[
            "interaction_rates_per_100k_agent_steps"
        ]["logical_touches"],
        "goals": metrics["goals"],
        "health": report["health"]["passed"],
    }


def _phase_authorized(
    source_state: dict[str, Any],
    source_record: dict[str, Any],
    requested: str,
    authorization_path: Path | None,
) -> bool:
    current = str(source_state.get("v10_1_active_phase", "A"))
    if requested == current:
        return True
    if authorization_path is None or not authorization_path.is_file():
        return False
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    return bool(
        authorization.get("status") == "passed"
        and authorization.get("source_checkpoint_manifest_sha256")
        == source_record["manifest_sha256"]
        and authorization.get("current_phase") == current
        and authorization.get("next_phase") == requested
        and authorization.get("phase_transition_authorized") is True
    )


def run_boundary(args: argparse.Namespace) -> dict[str, Any]:
    config = load_m10_1_config(args.config)
    boundary = float(args.boundary_added_hours)
    slug = boundary_slug(boundary)
    phase = str(args.phase).upper()
    source_checkpoint = args.source_checkpoint.resolve()
    loaded = load_v9_checkpoint(source_checkpoint, device=args.device)
    source_state = loaded["trainer_state"]
    source_record = checkpoint_record(source_checkpoint, manifest=loaded["manifest"])
    migration = config_migration_report(loaded["config"], config)
    exact_start = (
        verify_exact_plus25_start(source_checkpoint, device="cpu")
        if loaded["config"]["config_version"] == "RivalM10TrainingConfigV1"
        else None
    )
    if exact_start is not None and phase != "A":
        raise RuntimeError("The bootstrap must activate in Phase A")
    if not _phase_authorized(
        source_state, source_record, phase, args.phase_authorization
    ):
        raise RuntimeError(
            f"Phase {phase} is not authorized by the source checkpoint/evaluation"
        )
    source_steps = int(source_state["cumulative_agent_steps"])
    nominal_target = nominal_boundary_steps(boundary)
    rollout_target = int(config["ppo"]["rollout_agent_steps_per_iteration"])
    worker_count = int(config["backend"]["worker_count"])
    final_nominal = M10_PLUS25_STEPS + int(
        config["campaign"]["maximum_additional_agent_steps"]
    )
    final_ceiling = final_nominal + rollout_target + 2 * worker_count
    if source_steps < M10_PLUS25_STEPS:
        raise RuntimeError("v10.1 source predates the exact M10 +25 checkpoint")
    if source_steps >= nominal_target:
        raise RuntimeError("Source already reached the requested bootstrap boundary")

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
        env_factory=ENV_FACTORY_BY_PHASE[phase],
        collect_metrics_fn=collect_v10_bootstrap_metric_vector,
        aggregate_metrics_fn=aggregate_v10_bootstrap_metrics,
    )
    initial_identity = dict(
        source_state.get("v10_1_initial_m10_plus25_checkpoint", source_record)
    )
    prior_boundaries = list(source_state.get("v10_1_boundary_history", []))
    prior_phase_history = list(source_state.get("v10_1_phase_history", []))
    if not prior_phase_history or prior_phase_history[-1].get("phase") != phase:
        prior_phase_history.append(
            {
                "phase": phase,
                "activated_from_checkpoint_manifest_sha256": source_record[
                    "manifest_sha256"
                ],
                "activated_cumulative_agent_steps": source_steps,
            }
        )
    compact_iterations: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    removed_rolling: list[str] = []
    shapes = trainer.start_workers()
    cleanup: dict[str, Any] | None = None
    held_observations = loaded["reload_observations"]
    try:
        maximum_iterations = math.ceil((nominal_target - source_steps) / rollout_target) + 2
        while trainer.cumulative_agent_steps < nominal_target:
            if len(compact_iterations) >= maximum_iterations:
                raise RuntimeError("v10.1 boundary alignment guard failed")
            iteration, held_observations = trainer.run_iteration(
                maximum_cumulative_agent_steps=final_ceiling
            )
            compact = compact_training_iteration(iteration)
            compact_iterations.append(compact)
            write_json_atomic(
                iteration_root / f"{trainer.cumulative_agent_steps:09d}.json",
                iteration,
            )
            trainer_state = trainer.trainer_state()
            trainer_state.update(
                {
                    "v10_1_initial_m10_plus25_checkpoint": initial_identity,
                    "v10_1_latest_resume_source": source_record,
                    "v10_1_config_migration": migration,
                    "v10_1_boundary_history": prior_boundaries,
                    "v10_1_phase_history": prior_phase_history,
                    "v10_1_active_phase": phase,
                    "v10_1_active_boundary_added_simulated_hours": boundary,
                    "v10_1_added_agent_steps": trainer.cumulative_agent_steps
                    - M10_PLUS25_STEPS,
                    "v10_1_added_simulated_game_hours": (
                        trainer.cumulative_agent_steps - M10_PLUS25_STEPS
                    )
                    / AGENT_STEPS_PER_SIMULATED_HOUR,
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
            write_json_atomic(
                progress_path,
                {
                    "schema_version": 1,
                    "status": "in_progress",
                    "boundary_added_bootstrap_hours": boundary,
                    "phase": phase,
                    "source_checkpoint": source_record,
                    "nominal_target_cumulative_agent_steps": nominal_target,
                    "iterations": compact_iterations,
                    "rolling_checkpoints_written": checkpoint_records,
                    "rolling_checkpoints_removed": removed_rolling,
                    "latest_rolling_checkpoint": rolling_record,
                },
            )
            print(json.dumps(_console(iteration, boundary, phase)), flush=True)

        reached_steps = trainer.cumulative_agent_steps
        boundary_history = prior_boundaries + [
            {
                "boundary_added_bootstrap_hours": boundary,
                "phase": phase,
                "nominal_cumulative_agent_steps": nominal_target,
                "achieved_cumulative_agent_steps": reached_steps,
                "achieved_added_bootstrap_hours": (
                    reached_steps - M10_PLUS25_STEPS
                )
                / AGENT_STEPS_PER_SIMULATED_HOUR,
            }
        ]
        boundary_state = trainer.trainer_state()
        boundary_state.update(
            {
                "v10_1_initial_m10_plus25_checkpoint": initial_identity,
                "v10_1_latest_resume_source": source_record,
                "v10_1_config_migration": migration,
                "v10_1_boundary_history": boundary_history,
                "v10_1_phase_history": prior_phase_history,
                "v10_1_active_phase": phase,
                "v10_1_latest_completed_boundary_added_simulated_hours": boundary,
                "v10_1_added_agent_steps": reached_steps - M10_PLUS25_STEPS,
                "v10_1_added_simulated_game_hours": (
                    reached_steps - M10_PLUS25_STEPS
                )
                / AGENT_STEPS_PER_SIMULATED_HOUR,
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
        raise RuntimeError(f"v10.1 worker cleanup failed: {cleanup}")
    reload_parity = verify_checkpoint_reload_parity(
        immutable_checkpoint, expected_config=config, device="cpu"
    )
    result = {
        "schema_version": 1,
        "status": "passed",
        "campaign_version": "RivalAgencyBootstrapCampaignV1",
        "boundary_added_bootstrap_hours": boundary,
        "phase": phase,
        "source_checkpoint": source_record,
        "exact_m10_plus25_start_verification": exact_start,
        "config_migration": migration,
        "nominal_target_cumulative_agent_steps": nominal_target,
        "final_alignment_ceiling_agent_steps": final_ceiling,
        "achieved": {
            "cumulative_agent_steps": reached_steps,
            "additional_agent_steps_from_m10_plus25": reached_steps
            - M10_PLUS25_STEPS,
            "additional_bootstrap_simulated_game_hours": (
                reached_steps - M10_PLUS25_STEPS
            )
            / AGENT_STEPS_PER_SIMULATED_HOUR,
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
            "all_iterations_healthy": all(
                item["health"]["passed"] for item in compact_iterations
            ),
            "boundary_reached": reached_steps >= nominal_target,
            "within_absolute_campaign_ceiling": reached_steps <= final_ceiling,
            "immutable_checkpoint_reload_exact": reload_parity["checks"]["passed"],
            "production_promotion_authorized": False,
        },
    }
    result["checks"]["passed"] = all(
        value
        for key, value in result["checks"].items()
        if key != "production_promotion_authorized"
    ) and result["checks"]["production_promotion_authorized"] is False
    if not result["checks"]["passed"]:
        raise RuntimeError(f"v10.1 boundary failed: {result['checks']}")
    write_json_atomic(progress_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--boundary-added-hours",
        type=float,
        choices=(2.5, 5.0, 10.0, 15.0, 20.0, 25.0),
        required=True,
    )
    parser.add_argument("--phase", choices=("A", "B", "C"), required=True)
    parser.add_argument("--phase-authorization", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_boundary(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "boundary_added_bootstrap_hours": report[
                    "boundary_added_bootstrap_hours"
                ],
                "phase": report["phase"],
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
