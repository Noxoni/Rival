"""Create one compact, committed M10 boundary evidence record."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import (  # noqa: E402
    boundary_slug,
    portable_path,
    sha256_file,
    write_json_atomic,
)


DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "training/results/raw/milestone10"
DEFAULT_RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10"


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise FloatingPointError("M10 boundary summary requires finite values")
    return {
        "samples": int(len(array)),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def _raw_record(path: Path) -> dict[str, Any]:
    return {
        "path": portable_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "git_ignored": portable_path(path).startswith("training/results/raw/"),
    }


def _reward_totals(evaluation: dict[str, Any]) -> dict[str, dict[str, float]]:
    totals: dict[str, dict[str, float]] = {}
    for key, summary in evaluation["metrics"]["reward_components"].items():
        component = key.split(".", 1)[1]
        row = totals.setdefault(component, {"cumulative": 0.0, "cumulative_absolute": 0.0})
        row["cumulative"] += float(summary["cumulative"])
        row["cumulative_absolute"] += float(summary["cumulative_absolute"])
    return totals


def _reward_exploit_audit(
    baseline: dict[str, Any], current: dict[str, Any]
) -> dict[str, Any]:
    baseline_reward = _reward_totals(baseline)
    current_reward = _reward_totals(current)
    baseline_shaping = sum(
        row["cumulative"] for name, row in baseline_reward.items() if name != "outcome"
    )
    current_shaping = sum(
        row["cumulative"] for name, row in current_reward.items() if name != "outcome"
    )
    baseline_signature = baseline["behavior_signature"]
    current_signature = current["behavior_signature"]
    touches_degenerated = current_signature["touches_per_100k_agent_steps"] < 0.5 * max(
        baseline_signature["touches_per_100k_agent_steps"], 1e-12
    )
    distance_degenerated = current_signature["mean_distance_to_ball"] > 1.25 * max(
        baseline_signature["mean_distance_to_ball"], 1e-12
    )
    shaping_improved = current_shaping > baseline_shaping + 1e-6
    clear_exploitation = bool(shaping_improved and touches_degenerated and distance_degenerated)
    return {
        "version": "RivalM10RewardExploitAuditV1",
        "baseline_non_outcome_shaping_cumulative": baseline_shaping,
        "current_non_outcome_shaping_cumulative": current_shaping,
        "shaping_cumulative_increased": shaping_improved,
        "touch_frequency_below_half_baseline": touches_degenerated,
        "mean_distance_over_125_percent_baseline": distance_degenerated,
        "clear_reward_exploitation_detected": clear_exploitation,
        "rule": (
            "clear only when non-outcome shaping rises while touches fall below half "
            "the fixed M09 baseline and mean distance exceeds 125 percent of baseline"
        ),
        "fixed_evaluation_reward_components": current_reward,
        "passed": not clear_exploitation,
    }


def _training_summary(training: dict[str, Any]) -> dict[str, Any]:
    iterations = training["iterations"]
    total_steps = sum(int(row["collected_agent_steps"]) for row in iterations)
    collection_seconds = sum(float(row["collection_seconds"]) for row in iterations)
    iteration_wall_seconds = sum(float(row["iteration_wall_seconds"]) for row in iterations)
    combo_counts = np.zeros(8, dtype=np.int64)
    reset_counts: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    reward_components: dict[str, dict[str, float]] = {}
    for row in iterations:
        combo_counts += np.asarray(row["actions"]["button_combo_counts"], dtype=np.int64)
        reset_counts.update(row["reset_counts"])
        event_counts.update(row["event_counts"])
        for key, summary in row["reward_components"].items():
            target = reward_components.setdefault(
                key, {"cumulative": 0.0, "cumulative_absolute": 0.0}
            )
            target["cumulative"] += float(summary["cumulative"])
            target["cumulative_absolute"] += float(summary["cumulative_absolute"])
    reset_total = sum(reset_counts.values())
    action_total = max(int(combo_counts.sum()), 1)
    trend_rows = [
        {
            "iteration": int(row["iteration"]),
            "cumulative_agent_steps": int(row["cumulative_agent_steps"]),
            "simulated_game_hours": float(row["simulated_game_hours"]),
            "agent_steps_per_second": float(row["agent_steps_per_second"]),
            "actor_loss_mean": float(row["ppo"]["actor_loss"]["mean"]),
            "critic_loss_mean": float(row["ppo"]["critic_loss"]["mean"]),
            "approximate_kl_mean": float(row["ppo"]["approximate_kl"]["mean"]),
            "clip_fraction_mean": float(row["ppo"]["clip_fraction"]["mean"]),
            "explained_variance_before_update": float(
                row["ppo"]["explained_variance_before_update"]
            ),
            "analog_entropy_mean": float(row["ppo"]["analog_entropy"]["mean"]),
            "button_entropy_mean": float(row["ppo"]["button_entropy"]["mean"]),
            "actor_gradient_norm_mean": float(row["ppo"]["actor_gradient_norm"]["mean"]),
            "critic_gradient_norm_mean": float(row["ppo"]["critic_gradient_norm"]["mean"]),
            "reward_mean": float(row["reward"]["mean"]),
        }
        for row in iterations
    ]
    return {
        "iterations_in_boundary_segment": len(iterations),
        "agent_steps_in_boundary_segment": total_steps,
        "collection_seconds": collection_seconds,
        "iteration_wall_seconds": iteration_wall_seconds,
        "sustained_agent_steps_per_collection_second": total_steps
        / max(collection_seconds, 1e-12),
        "simulated_game_hours_per_wall_hour": (total_steps / 864000.0)
        / max(iteration_wall_seconds / 3600.0, 1e-12),
        "throughput_per_iteration": _stats(
            [float(row["agent_steps_per_second"]) for row in iterations]
        ),
        "ppo_health": {
            "all_iterations_passed": all(row["health"]["passed"] for row in iterations),
            "all_update_metrics_finite": all(
                row["health"]["all_update_metrics_finite"] for row in iterations
            ),
            "all_workers_alive": all(
                row["health"]["rollout_workers_alive"] for row in iterations
            ),
            "all_actor_and_critic_updates_nonzero": all(
                row["health"]["actor_updated"] and row["health"]["critic_updated"]
                for row in iterations
            ),
            "all_hybrid_gradient_rows_nonzero": all(
                row["health"]["all_hybrid_head_gradient_rows_nonzero"]
                for row in iterations
            ),
            "approximate_kl_mean_by_iteration": _stats(
                [float(row["ppo"]["approximate_kl"]["mean"]) for row in iterations]
            ),
            "clip_fraction_mean_by_iteration": _stats(
                [float(row["ppo"]["clip_fraction"]["mean"]) for row in iterations]
            ),
            "explained_variance_before_update_by_iteration": _stats(
                [
                    float(row["ppo"]["explained_variance_before_update"])
                    for row in iterations
                ]
            ),
        },
        "action_and_exploration": {
            "aggregate_button_combo_counts": combo_counts.tolist(),
            "aggregate_button_combo_shares": (combo_counts / action_total).tolist(),
            "all_eight_combos_observed_every_iteration": all(
                row["actions"]["all_eight_button_combos_sampled"] for row in iterations
            ),
            "all_analog_axes_nontrivial_every_iteration": all(
                all(axis["nontrivial_range"] for axis in row["actions"]["analog"])
                for row in iterations
            ),
            "first_iteration": iterations[0]["actions"],
            "last_iteration": iterations[-1]["actions"],
            "analog_log_std_first": iterations[0]["ppo"]["analog_log_std"],
            "analog_log_std_last": iterations[-1]["ppo"]["analog_log_std"],
            "button_entropy_first": iterations[0]["ppo"]["button_entropy"],
            "button_entropy_last": iterations[-1]["ppo"]["button_entropy"],
        },
        "curriculum": {
            "reset_counts": dict(reset_counts),
            "reset_shares": {
                key: count / max(reset_total, 1) for key, count in reset_counts.items()
            },
        },
        "event_counts": dict(event_counts),
        "reward_component_totals": reward_components,
        "iteration_trend": trend_rows,
    }


def _compact_fixed(evaluation: dict[str, Any]) -> dict[str, Any]:
    metrics = evaluation["metrics"]
    return {
        "evaluation_version": evaluation["evaluation_version"],
        "checkpoint": evaluation["checkpoint"],
        "fixed_protocol": evaluation["fixed_protocol"],
        "agent_steps_evaluated": int(evaluation["agent_steps_evaluated"]),
        "actor_fingerprint": evaluation["actor_fingerprint"],
        "behavior_signature": evaluation["behavior_signature"],
        "scores_recorded_for_diagnostics_only": metrics[
            "scores_recorded_for_diagnostics_only"
        ],
        "event_counts": metrics["event_counts"],
        "movement_and_recovery": metrics["movement_and_recovery"],
        "reward_components": metrics["reward_components"],
        "fixed_action_diagnostics": evaluation["fixed_action_diagnostics"],
        "checks": evaluation["checks"],
    }


def _compact_frozen(frozen: dict[str, Any] | None) -> dict[str, Any] | None:
    if frozen is None:
        return None
    return {
        "comparison_version": frozen["comparison_version"],
        "comparisons": [
            {
                "label": row["label"],
                "candidate": row["candidate"],
                "reference": row["reference"],
                "protocol": row["protocol"],
                "candidate_metrics": {
                    "record": row["metrics"]["candidate"]["record"],
                    "behavior_signature": row["metrics"]["candidate"][
                        "behavior_signature"
                    ],
                    "event_rates_per_100k_agent_steps": row["metrics"]["candidate"][
                        "event_rates_per_100k_agent_steps"
                    ],
                },
                "reference_metrics": {
                    "record": row["metrics"]["reference"]["record"],
                    "behavior_signature": row["metrics"]["reference"][
                        "behavior_signature"
                    ],
                    "event_rates_per_100k_agent_steps": row["metrics"]["reference"][
                        "event_rates_per_100k_agent_steps"
                    ],
                },
                "checks": row["checks"],
            }
            for row in frozen["comparisons"]
        ],
        "checks": frozen["checks"],
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    boundary = int(args.boundary_added_hours)
    training_path = args.training or (
        DEFAULT_RAW_ROOT / boundary_slug(boundary) / "training_progress.json"
    )
    training = _json(training_path)
    evaluation = _json(args.evaluation)
    baseline = _json(args.baseline)
    frozen = _json(args.frozen) if args.frozen else None
    native = _json(args.native) if args.native else None
    immutable = training["immutable_checkpoint"]
    reward_audit = _reward_exploit_audit(baseline, evaluation)
    raw_paths = [Path(training_path), Path(args.evaluation), Path(args.baseline)]
    if args.frozen:
        raw_paths.append(Path(args.frozen))
    if args.native:
        raw_paths.append(Path(args.native))
    frozen_required = boundary >= 10
    native_required = boundary in (25, 100)
    checks = {
        "training_boundary_passed": training["status"] == "passed"
        and training["checks"]["passed"],
        "immutable_checkpoint_reload_exact": training[
            "immutable_checkpoint_fresh_reload"
        ]["checks"]["passed"],
        "fixed_evaluation_passed": evaluation["status"] == "passed"
        and evaluation["checks"]["passed"],
        "evaluation_used_immutable_checkpoint": evaluation["checkpoint"]["manifest_sha256"]
        == immutable["manifest_sha256"],
        "frozen_comparison_present_when_required": not frozen_required or frozen is not None,
        "frozen_comparison_passed_when_present": frozen is None
        or frozen["checks"]["passed"],
        "native_transfer_present_when_required": not native_required or native is not None,
        "native_transfer_passed_when_present": native is None
        or native.get("status") == "passed"
        and native.get("checks", {}).get("passed") is True,
        "reward_exploit_audit_passed": reward_audit["passed"],
        "production_promotion_authorized": False,
        "wins_and_losses_excluded_from_technical_pass_fail": True,
    }
    passed = all(
        value for key, value in checks.items() if key != "production_promotion_authorized"
    ) and checks["production_promotion_authorized"] is False
    checks["passed"] = passed
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "milestone": "10",
        "boundary_added_simulated_hours": boundary,
        "source_checkpoint": training["source_checkpoint"],
        "immutable_checkpoint": immutable,
        "target_and_achievement": {
            "nominal_target_cumulative_agent_steps": training[
                "nominal_target_cumulative_agent_steps"
            ],
            **training["achieved"],
        },
        "frozen_foundation": {
            "config_migration": training["config_migration"],
            "policy_observation_action_reward_ppo_worker_curriculum_unchanged": training[
                "config_migration"
            ]["checks"]["all_learning_semantics_exact"],
            "production_promotion_authorized": False,
        },
        "training": _training_summary(training),
        "fixed_deterministic_evaluation": _compact_fixed(evaluation),
        "m09_fixed_baseline": {
            "checkpoint": baseline["checkpoint"],
            "behavior_signature": baseline["behavior_signature"],
            "actor_fingerprint": baseline["actor_fingerprint"],
        },
        "reward_component_behavior_and_exploit_audit": reward_audit,
        "frozen_snapshot_comparison": _compact_frozen(frozen),
        "native_rlbot_transfer": native,
        "raw_local_evidence": [_raw_record(path) for path in raw_paths],
        "checks": checks,
    }
    write_json_atomic(args.output, result)
    if not passed:
        raise RuntimeError(f"M10 boundary finalization failed: {checks}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-added-hours", type=int, choices=(5, 10, 25, 50, 100), required=True)
    parser.add_argument("--training", type=Path)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--frozen", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_RESULT_ROOT / f"boundary_{boundary_slug(args.boundary_added_hours)}.json"
    report = finalize(args)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
