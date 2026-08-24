"""Train Rival v10.2 Stage 1 to one clean evaluation boundary."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import (  # noqa: E402
    checkpoint_record,
    prune_rolling_checkpoints,
    save_checkpoint_atomic,
    verify_checkpoint_reload_parity,
    write_json_atomic,
)
from rival_training.v10_2_campaign import (  # noqa: E402
    CAMPAIGN_STATE_PATH,
    DEFAULT_STAGE1_CONFIG,
    RESULT_ROOT,
    SOURCE_CHECKPOINT,
    actor_only_stage_transfer,
    boundary_slug,
    load_stage1_config,
    nominal_stage1_steps,
    start_real_campaign_clock,
    update_progressive_state,
    wall_clock_status,
)
from rival_training.v10_2_environment import (  # noqa: E402
    BALL_ACQUISITION_ENV_FACTORY_BY_PHASE,
)
from rival_training.v9_checkpoint import load_v9_checkpoint  # noqa: E402
from rival_training.v9_trainer import RivalV9PPOTrainer  # noqa: E402


DEFAULT_CHECKPOINT_ROOT = (
    REPOSITORY_ROOT / "training/checkpoints/milestone10_2/stage_1"
)
STAGE_MAXIMUM_ACTIVE_STEPS = 6_480_000
EVALUATION_OVERHEAD_PROJECTION_SECONDS = 540.0


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _effective_config(path: Path) -> dict[str, Any]:
    config = load_stage1_config(path)
    preflight = _read(RESULT_ROOT / "preflight.json")
    if preflight.get("status") != "passed":
        raise RuntimeError("Stage 1 cannot start before preflight passes")
    selected = int(preflight["effective_selected_worker_count"])
    if selected not in (32, 40, 48, 56, 64):
        raise RuntimeError(f"Invalid preflight worker selection: {selected}")
    config = deepcopy(config)
    config["backend"]["worker_count"] = selected
    return config


def _compact_iteration(report: dict[str, Any]) -> dict[str, Any]:
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
            "batch_active_learner_steps": report["ppo"]["batch_agent_steps"],
            "actor_loss": report["ppo"]["actor_loss"],
            "critic_loss": report["ppo"]["critic_loss"],
            "analog_entropy": report["ppo"]["analog_entropy"],
            "button_entropy": report["ppo"]["button_entropy"],
            "actor_update_magnitude": report["ppo"][
                "actor_update_magnitude"
            ],
            "critic_update_magnitude": report["ppo"][
                "critic_update_magnitude"
            ],
            "head_gradient_absolute_sums": report["ppo"][
                "head_gradient_absolute_sums"
            ],
        },
        "actions": report["actions"],
        "health": report["health"],
    }


def _new_trainer(
    source: Path,
    config: dict[str, Any],
    *,
    phase: str,
    device: str,
) -> tuple[RivalV9PPOTrainer, dict[str, Any], Any]:
    if source.resolve() == SOURCE_CHECKPOINT.resolve():
        transfer = actor_only_stage_transfer(source, config, device=device)
        trainer = RivalV9PPOTrainer(
            config,
            device=device,
            actor=transfer["actor"],
            critic=transfer["critic"],
            actor_optimizer=transfer["actor_optimizer"],
            critic_optimizer=transfer["critic_optimizer"],
            trainer_state=transfer["trainer_state"],
            env_factory=BALL_ACQUISITION_ENV_FACTORY_BY_PHASE[phase],
        )
        return trainer, transfer["proof"]["source_checkpoint"], transfer
    loaded = load_v9_checkpoint(
        source,
        device=device,
        expected_config=config,
    )
    trainer = RivalV9PPOTrainer(
        config,
        device=device,
        actor=loaded["actor"],
        critic=loaded["critic"],
        actor_optimizer=loaded["actor_optimizer"],
        critic_optimizer=loaded["critic_optimizer"],
        trainer_state=loaded["trainer_state"],
        env_factory=BALL_ACQUISITION_ENV_FACTORY_BY_PHASE[phase],
    )
    return trainer, checkpoint_record(source, manifest=loaded["manifest"]), None


def run_boundary(args: argparse.Namespace) -> dict[str, Any]:
    boundary_hours = float(args.boundary_hours)
    slug = boundary_slug(boundary_hours)
    target_steps = nominal_stage1_steps(boundary_hours)
    phase = str(args.phase).upper()
    if phase not in BALL_ACQUISITION_ENV_FACTORY_BY_PHASE:
        raise ValueError(f"Unsupported Stage-1 phase: {phase}")
    config = _effective_config(args.config)
    state = _read(CAMPAIGN_STATE_PATH)
    if state["current_stage"] != 1 or state["current_phase"] not in {
        phase,
        "ready_stage_1",
    }:
        raise RuntimeError(
            "Progressive state does not authorize this Stage-1 phase: "
            f"{state['current_phase']} versus {phase}"
        )
    if state.get("stop_reason"):
        raise RuntimeError(f"Campaign is already stopped: {state['stop_reason']}")

    source = args.source_checkpoint.resolve()
    trainer, source_record, transfer = _new_trainer(
        source, config, phase=phase, device=args.device
    )
    if trainer.cumulative_agent_steps >= target_steps:
        raise RuntimeError(
            "Source has already reached the requested Stage-1 boundary"
        )
    if trainer.cumulative_agent_steps > STAGE_MAXIMUM_ACTIVE_STEPS:
        raise RuntimeError("Source exceeds the Stage-1 experience authority")

    start_real_campaign_clock()
    update_progressive_state(
        {
            "current_stage": 1,
            "current_skill": "ball_acquisition",
            "current_phase": phase,
            "gate_decision": "stage_1_training_in_progress",
        }
    )
    checkpoint_root = args.checkpoint_root.resolve()
    rolling_root = checkpoint_root / "rolling"
    boundary_parent = checkpoint_root / "boundaries" / slug
    iteration_rows: list[dict[str, Any]] = []
    rolling_records: list[dict[str, Any]] = []
    removed_rolling: list[str] = []
    projected_iteration_seconds = float(
        state.get("projected_next_iteration_seconds", 45.0)
    )
    cleanup = None
    wall_stop = False
    latest_held = None
    shapes = None
    try:
        shapes = trainer.start_workers()
        while trainer.cumulative_agent_steps < target_steps:
            clock = wall_clock_status(
                projected_iteration_and_boundary_seconds=(
                    projected_iteration_seconds
                    + EVALUATION_OVERHEAD_PROJECTION_SECONDS
                ),
                config=config,
            )
            if not clock["ordinary_iteration_allowed"]:
                wall_stop = True
                break
            remaining = target_steps - trainer.cumulative_agent_steps
            worker_reserve = 2 * int(config["backend"]["worker_count"])
            if target_steps == STAGE_MAXIMUM_ACTIVE_STEPS:
                rollout = min(
                    int(config["ppo"]["rollout_agent_steps_per_iteration"]),
                    max(0, remaining - worker_reserve),
                )
                maximum = STAGE_MAXIMUM_ACTIVE_STEPS
            else:
                rollout = min(
                    int(config["ppo"]["rollout_agent_steps_per_iteration"]),
                    remaining,
                )
                maximum = target_steps + worker_reserve
            minibatch = int(config["ppo"]["minibatch_agent_steps"])
            batch = (rollout // minibatch) * minibatch
            if batch < minibatch or rollout <= 0:
                break
            iteration, held = trainer.run_iteration(
                rollout_target_agent_steps=rollout,
                ppo_batch_agent_steps=batch,
                maximum_cumulative_agent_steps=maximum,
            )
            latest_held = held
            compact = _compact_iteration(iteration)
            iteration_rows.append(compact)
            recent = [
                float(row["iteration_wall_seconds"])
                for row in iteration_rows[-3:]
            ]
            projected_iteration_seconds = max(recent) * 1.25
            clock = wall_clock_status(
                projected_iteration_and_boundary_seconds=(
                    projected_iteration_seconds
                    + EVALUATION_OVERHEAD_PROJECTION_SECONDS
                ),
                config=config,
            )
            trainer_state = trainer.trainer_state()
            trainer_state.update(
                {
                    "stage": 1,
                    "stage_phase": phase,
                    "v10_2_source_checkpoint": source_record,
                    "v10_2_active_boundary_hours": boundary_hours,
                    "campaign_wall_clock": clock,
                    "production_promotion_authorized": False,
                }
            )
            rolling_directory = rolling_root / (
                f"{trainer.cumulative_agent_steps:09d}"
            )
            rolling_record = save_checkpoint_atomic(
                rolling_directory,
                actor=trainer.actor,
                critic=trainer.critic,
                actor_optimizer=trainer.actor_optimizer,
                critic_optimizer=trainer.critic_optimizer,
                trainer_state=trainer_state,
                config=config,
                reload_observations=held,
            )
            rolling_records.append(rolling_record)
            removed_rolling.extend(
                prune_rolling_checkpoints(
                    rolling_root,
                    keep=int(
                        config["stage_contract"][
                            "rolling_recovery_checkpoints_to_keep"
                        ]
                    ),
                )
            )
            update_progressive_state(
                {
                    "stage_active_learner_steps": trainer.cumulative_agent_steps,
                    "stage_simulated_hours": trainer.cumulative_agent_steps
                    / 432_000.0,
                    "total_progressive_active_learner_steps": (
                        trainer.cumulative_agent_steps
                    ),
                    "total_progressive_simulated_hours": (
                        trainer.cumulative_agent_steps / 432_000.0
                    ),
                    "campaign_wall_clock_elapsed_seconds": clock[
                        "elapsed_seconds"
                    ],
                    "campaign_wall_clock_remaining_seconds": clock[
                        "remaining_seconds"
                    ],
                    "projected_next_iteration_seconds": (
                        projected_iteration_seconds
                    ),
                    "latest_clean_recovery_checkpoint": rolling_record,
                }
            )
            print(
                json.dumps(
                    {
                        "stage": 1,
                        "phase": phase,
                        "boundary_hours": boundary_hours,
                        "iteration": compact,
                        "wall_clock": clock,
                        "rolling_checkpoint": rolling_record,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        cleanup = trainer.cleanup()
    if cleanup is None or not cleanup["passed"]:
        raise RuntimeError(f"Stage-1 worker cleanup failed: {cleanup}")
    if latest_held is None:
        raise RuntimeError("Stage-1 boundary made no recoverable PPO update")

    clock = wall_clock_status(config=config)
    trainer_state = trainer.trainer_state()
    trainer_state.update(
        {
            "stage": 1,
            "stage_phase": phase,
            "v10_2_source_checkpoint": source_record,
            "v10_2_completed_boundary_hours": boundary_hours,
            "campaign_wall_clock": clock,
            "production_promotion_authorized": False,
        }
    )
    if wall_stop:
        stop_parent = checkpoint_root / "wall-clock-stop"
        destination = stop_parent / f"{trainer.cumulative_agent_steps:09d}"
    else:
        destination = boundary_parent / f"{trainer.cumulative_agent_steps:09d}"
    immutable = save_checkpoint_atomic(
        destination,
        actor=trainer.actor,
        critic=trainer.critic,
        actor_optimizer=trainer.actor_optimizer,
        critic_optimizer=trainer.critic_optimizer,
        trainer_state=trainer_state,
        config=config,
        reload_observations=latest_held,
    )
    reload_parity = verify_checkpoint_reload_parity(
        destination,
        expected_config=config,
        device="cpu",
    )
    reached = trainer.cumulative_agent_steps >= target_steps
    status = "wall_clock_stop" if wall_stop else "passed" if reached else "failed"
    result = {
        "schema_version": 1,
        "training_boundary_version": "RivalM10_2Stage1TrainingBoundaryV1",
        "status": status,
        "stage": 1,
        "skill": "ball_acquisition",
        "phase": phase,
        "boundary_hours": boundary_hours,
        "target_active_learner_steps": target_steps,
        "reached_active_learner_steps": trainer.cumulative_agent_steps,
        "reached_simulated_hours": trainer.cumulative_agent_steps
        / 432_000.0,
        "source_checkpoint": source_record,
        "source_actor_only_transfer": transfer["proof"] if transfer else None,
        "environment_shapes": shapes,
        "iterations": iteration_rows,
        "rolling_checkpoints_written": rolling_records,
        "rolling_checkpoints_removed": removed_rolling,
        "immutable_checkpoint": immutable,
        "immutable_checkpoint_reload": reload_parity,
        "worker_cleanup": cleanup,
        "campaign_wall_clock": clock,
        "stop_reason": (
            "stop_progressive_overnight_wall_clock_budget_exhausted"
            if wall_stop
            else None
        ),
        "checks": {
            "one_trainable_row_per_collected_step": all(
                row["experience_records"]
                == row["collected_active_learner_steps"]
                for row in iteration_rows
            ),
            "all_iterations_healthy": all(
                row["health"]["passed"] for row in iteration_rows
            ),
            "stage_experience_ceiling_respected": (
                trainer.cumulative_agent_steps <= STAGE_MAXIMUM_ACTIVE_STEPS
            ),
            "immutable_checkpoint_reload_exact": reload_parity["checks"][
                "passed"
            ],
            "workers_cleaned": cleanup["passed"],
            "boundary_reached_or_wall_stop": reached or wall_stop,
            "production_promotion_authorized": False,
        },
    }
    result["checks"]["passed"] = all(
        value
        for key, value in result["checks"].items()
        if key != "production_promotion_authorized"
    ) and result["checks"]["production_promotion_authorized"] is False
    output = args.output or (
        RESULT_ROOT / "stage_1" / f"training_{slug}.json"
    )
    write_json_atomic(output, result)
    updates = {
        "stage_active_learner_steps": trainer.cumulative_agent_steps,
        "stage_simulated_hours": trainer.cumulative_agent_steps / 432_000.0,
        "total_progressive_active_learner_steps": (
            trainer.cumulative_agent_steps
        ),
        "total_progressive_simulated_hours": (
            trainer.cumulative_agent_steps / 432_000.0
        ),
        "campaign_wall_clock_elapsed_seconds": clock["elapsed_seconds"],
        "campaign_wall_clock_remaining_seconds": clock["remaining_seconds"],
        "current_evaluation_boundary": boundary_hours if reached else None,
        "latest_clean_recovery_checkpoint": immutable,
        "gate_decision": (
            "pending_stage_1_evaluation"
            if reached
            else "stop_progressive_overnight_wall_clock_budget_exhausted"
        ),
        "stop_reason": result["stop_reason"],
    }
    update_progressive_state(updates)
    if not result["checks"]["passed"]:
        raise RuntimeError(f"Stage-1 boundary failed: {result['checks']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-hours", type=float, required=True)
    parser.add_argument("--phase", choices=("A", "B"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_STAGE1_CONFIG)
    parser.add_argument(
        "--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_boundary(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "phase": report["phase"],
                "boundary_hours": report["boundary_hours"],
                "checkpoint": report["immutable_checkpoint"],
                "wall_clock": report["campaign_wall_clock"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
