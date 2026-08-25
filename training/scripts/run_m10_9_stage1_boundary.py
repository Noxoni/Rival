"""Train and evaluate M10.9 at one immutable Stage-1 boundary."""

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
from rival_training.v10_7_checkpoint import (  # noqa: E402
    checkpoint_record,
    load_checkpoint,
    save_checkpoint_atomic,
    verify_reload_parity,
)
from rival_training.v10_9_campaign import (  # noqa: E402
    ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR,
    BOUNDARY_HOURS,
    CHECKPOINT_ROOT,
    CORPUS_ROOT,
    GATE_CORPUS_FILENAME,
    RESULT_ROOT,
    load_stage1_config,
)
from rival_training.v10_9_credit import (  # noqa: E402
    aggregate_credit_diagnostics,
)
from rival_training.v10_9_evaluation import (  # noqa: E402
    capability_gap,
    evaluate_stage1_checkpoint,
)
from rival_training.v10_9_trainer import RivalV10_9PPOTrainer  # noqa: E402


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _boundary_slug(hours: float) -> str:
    return {0.25: "plus-000p25h", 0.5: "plus-000p5h", 1.0: "plus-001h"}[
        float(hours)
    ]


def _compact(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: report[key]
        for key in (
            "iteration",
            "collected_agent_steps",
            "experience_records",
            "cumulative_agent_steps",
            "cumulative_model_updates",
            "simulated_game_hours",
            "collection_seconds",
            "update_wall_seconds",
            "iteration_wall_seconds",
            "agent_steps_per_second",
            "cuda_peak_allocated_mib",
            "rollout_log_probability_reproduction",
            "reward",
            "gae",
            "ppo",
            "actions",
            "exploration",
            "credit_assignment",
            "health",
        )
    }


def run_boundary(args: argparse.Namespace) -> dict[str, Any]:
    boundary_hours = float(args.boundary_hours)
    if boundary_hours not in BOUNDARY_HOURS:
        raise ValueError(f"Unsupported M10.9 boundary: {boundary_hours}")
    slug = _boundary_slug(boundary_hours)
    target_steps = int(
        round(boundary_hours * ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR)
    )
    config = load_stage1_config(args.config)
    preflight = _read(RESULT_ROOT / "preflight.json")
    if preflight.get("status") != "passed" or preflight.get("ppo_authorized") is not True:
        raise RuntimeError("M10.9 PPO is forbidden until hard preflight passes")
    source = args.source_checkpoint.resolve()
    loaded = load_checkpoint(source, device=args.device, expected_config=config)
    state = loaded["trainer_state"]
    if state.get("m10_9_ppo_v2") is not True:
        raise RuntimeError("Source checkpoint is not in the M10.9 lineage")
    expected_source = {0.25: 0, 0.5: 108_000, 1.0: 216_000}[boundary_hours]
    if int(state["cumulative_agent_steps"]) < expected_source:
        raise RuntimeError("Source has not reached the preceding M10.9 boundary")
    if int(state["cumulative_agent_steps"]) >= target_steps:
        raise RuntimeError("Source already reached this boundary")
    trainer = RivalV10_9PPOTrainer(
        config,
        device=args.device,
        actor=loaded["actor"],
        critic=loaded["critic"],
        actor_optimizer=loaded["actor_optimizer"],
        critic_optimizer=loaded["critic_optimizer"],
        trainer_state=state,
    )
    source_record = checkpoint_record(source, manifest=loaded["manifest"])
    rolling_root = args.checkpoint_root.resolve() / "rolling"
    boundary_parent = args.checkpoint_root.resolve() / "boundaries" / slug
    campaign_state_path = RESULT_ROOT / "campaign_state.json"
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
            iteration, held = trainer.run_iteration(
                rollout_target_agent_steps=rollout,
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
                campaign_state_path,
                {
                    "format": "rival-m10-9-stage1-state-v1",
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
        raise RuntimeError(f"M10.9 worker cleanup failed: {cleanup}")
    if latest_held is None:
        raise RuntimeError("M10.9 boundary made no recoverable PPO update")
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
    deterministic = evaluate_stage1_checkpoint(
        destination,
        CORPUS_ROOT / GATE_CORPUS_FILENAME,
        deterministic=True,
        evaluation_workers=int(args.evaluation_workers),
    )
    stochastic = evaluate_stage1_checkpoint(
        destination,
        CORPUS_ROOT / GATE_CORPUS_FILENAME,
        deterministic=False,
        evaluation_workers=int(args.evaluation_workers),
    )
    deterministic_path = RESULT_ROOT / f"evaluation_{slug}_deterministic.json"
    stochastic_path = RESULT_ROOT / f"evaluation_{slug}_stochastic_ar1.json"
    write_json_atomic(deterministic_path, deterministic)
    write_json_atomic(stochastic_path, stochastic)
    reached = trainer.cumulative_agent_steps >= target_steps
    result = {
        "schema_version": 1,
        "training_boundary_version": "RivalM10_9PPOV2BoundaryV1",
        "status": "passed" if reached else "failed",
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
        "evaluation": {
            "deterministic_report": deterministic_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "stochastic_report": stochastic_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "deterministic_overall": deterministic["overall"],
            "stochastic_overall": stochastic["overall"],
            "stochastic_vs_deterministic_gap": capability_gap(
                deterministic, stochastic
            ),
        },
        "checks": {
            "boundary_reached": reached,
            "one_hour_ceiling_respected": trainer.cumulative_agent_steps <= 432_112,
            "all_iterations_healthy": all(
                row["health"]["passed"] for row in iterations
            ),
            "all_credit_diagnostics_passed": all(
                row["credit_assignment"]["checks"]["passed"]
                for row in iterations
            ),
            "all_advantage_signs_preserved": all(
                row["gae"]["raw_scaled_sign_agreement"] for row in iterations
            ),
            "checkpoint_reload_exact": parity["checks"]["passed"],
            "deterministic_evaluation_passed": deterministic["checks"]["passed"],
            "stochastic_evaluation_passed": stochastic["checks"]["passed"],
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
    output = args.output or RESULT_ROOT / f"training_{slug}.json"
    write_json_atomic(output, result)
    write_json_atomic(
        campaign_state_path,
        {
            "format": "rival-m10-9-stage1-state-v1",
            "phase": "complete" if boundary_hours == 1.0 else "pending_next_boundary",
            "boundary_hours": boundary_hours,
            "latest_clean_recovery_checkpoint": immutable,
            "stage_2_authorized": False,
            "production_promotion_authorized": False,
        },
    )
    if not result["checks"]["passed"]:
        raise RuntimeError(f"M10.9 boundary failed: {result['checks']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--boundary-hours", type=float, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "training/configs/milestone10_9_stage1.json",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-workers", type=int, default=24)
    args = parser.parse_args()
    report = run_boundary(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "boundary_hours": report["boundary_hours"],
                "checkpoint": report["immutable_checkpoint"],
                "deterministic_overall": report["evaluation"][
                    "deterministic_overall"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
