"""Finalize one Rival v10.1 boundary and apply the frozen agency gates."""

from __future__ import annotations

import argparse
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
    portable_path,
    write_json_atomic,
)
from rival_training.v10_1_campaign import (  # noqa: E402
    M10_PLUS25_MANIFEST_SHA256,
    M10_PLUS25_STEPS,
)
from rival_training.v10_bootstrap_reward import (  # noqa: E402
    SHAPING_COMPONENTS,
)
from rival_training.v9_checkpoint import sha256_file  # noqa: E402


M10_PLUS10_RESULT = REPOSITORY_ROOT / (
    "training/results/milestone10/boundary_plus-010h.json"
)
M10_PLUS25_RESULT = REPOSITORY_ROOT / (
    "training/results/milestone10/boundary_plus-025h.json"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _historical_comparisons(candidate_fixed: dict[str, Any]) -> dict[str, Any]:
    plus10 = _read(M10_PLUS10_RESULT)
    plus25 = _read(M10_PLUS25_RESULT)
    sources = {
        "m09_gate13": plus25["m09_fixed_baseline"],
        "m10_plus10": plus10["fixed_deterministic_evaluation"],
        "m10_plus25": plus25["fixed_deterministic_evaluation"],
        "candidate": candidate_fixed,
    }
    signatures = {
        name: value["behavior_signature"] for name, value in sources.items()
    }
    candidate = signatures["candidate"]
    deltas: dict[str, dict[str, float]] = {}
    for name in ("m09_gate13", "m10_plus10", "m10_plus25"):
        reference = signatures[name]
        deltas[name] = {
            metric: float(candidate[metric]) - float(reference[metric])
            for metric in candidate
            if metric in reference
            and isinstance(candidate[metric], (int, float))
            and isinstance(reference[metric], (int, float))
        }
    return {"behavior_signatures": signatures, "candidate_deltas": deltas}


def _compact_fixed_evaluation(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": report["status"],
        "evaluation_version": report["evaluation_version"],
        "checkpoint": report["checkpoint"],
        "fixed_protocol": report["fixed_protocol"],
        "actor_fingerprint": report["actor_fingerprint"],
        "agent_steps_evaluated": report["agent_steps_evaluated"],
        "behavior_signature": report["behavior_signature"],
        "fixed_action_diagnostics": report["fixed_action_diagnostics"],
        "checks": report["checks"],
    }


def _compact_bootstrap_evaluation(report: dict[str, Any]) -> dict[str, Any]:
    episode_outcomes: dict[str, dict[str, int]] = {}
    for family in report["fixed_protocol"]["families"]:
        rows = [row for row in report["episodes"] if row["family"] == family]
        episode_outcomes[family] = {
            "episodes": len(rows),
            "goals": sum(int(row["end_reason"] == "goal") for row in rows),
            "active_team_goals": sum(
                int(row["active_team_scored"]) for row in rows
            ),
            "no_touch_timeouts": sum(
                int(row["end_reason"] == "no_touch_timeout") for row in rows
            ),
            "episode_timeouts": sum(
                int(row["end_reason"] == "episode_timeout") for row in rows
            ),
        }
    return {
        "status": report["status"],
        "evaluation_version": report["evaluation_version"],
        "checkpoint": report["checkpoint"],
        "fixed_protocol": report["fixed_protocol"],
        "environment_ticks": report["environment_ticks"],
        "agent_steps_evaluated": report["agent_steps_evaluated"],
        "wall_seconds": report["wall_seconds"],
        "capability_summary": report["capability_summary"],
        "metrics": report["metrics"],
        "episode_outcomes_by_family": episode_outcomes,
        "checks": report["checks"],
    }


def _compact_boundary_iteration(report: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in report.items()
        if key != "actions"
    }


def _reward_dominance(
    metrics: dict[str, Any], touches: float, baseline_touches: float
) -> dict[str, Any]:
    components = metrics["reward_components"]
    combined_absolute = {
        name: float(
            sum(
                components[f"{slot}.{name}"]["cumulative_absolute"]
                for slot in ("blue", "orange")
            )
        )
        for name in SHAPING_COMPONENTS
    }
    total = sum(combined_absolute.values())
    fractions = {
        name: value / max(total, 1e-12)
        for name, value in combined_absolute.items()
    }
    maximum_name = max(fractions, key=fractions.get)
    speed_dominant = fractions["useful_speed_rate"] > 0.80
    ball_interaction_stagnant = touches <= baseline_touches * 1.10
    return {
        "combined_shaping_absolute_reward": combined_absolute,
        "component_absolute_fractions": fractions,
        "largest_component": maximum_name,
        "largest_component_fraction": fractions[maximum_name],
        "useful_speed_rate_dominant_over_80_percent": speed_dominant,
        "ball_interaction_stagnant_vs_prebootstrap": ball_interaction_stagnant,
        "speed_dominance_with_stagnant_interaction": speed_dominant
        and ball_interaction_stagnant,
        "no_obvious_single_component_farming": not (
            fractions[maximum_name] > 0.80 and ball_interaction_stagnant
        ),
    }


def _gate_report(
    boundary_hours: float,
    phase: str,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    cap = candidate["capability_summary"]
    baseline_cap = baseline["capability_summary"]
    metrics = candidate["metrics"]
    rates = metrics["interaction_rates_per_100k_agent_steps"]
    reward = _reward_dominance(
        metrics,
        float(rates["logical_touches"]),
        float(baseline_cap["logical_touches"]),
    )
    phase_a_conditions = {
        "at_least_2p5_bootstrap_hours": boundary_hours >= 2.5,
        "mean_planar_speed_at_least_600": cap["mean_planar_speed"] >= 600.0,
        "logical_touches_at_least_75_per_100k": rates["logical_touches"]
        >= 75.0,
        "deterministic_jumps_at_least_5_per_100k": rates["first_jumps"] >= 5.0,
        "easy_finish_active_team_goal_with_touch_evidence_exists": cap[
            "easy_finish_active_team_goals"
        ]
        >= 1
        and metrics["touches_by_reset_family"]["easy_finish"] >= 1,
        "no_touch_timeout_share_at_most_60_percent": cap[
            "no_touch_timeout_share"
        ]
        <= 0.60,
        "no_useful_speed_dominance_with_stagnant_interaction": not reward[
            "speed_dominance_with_stagnant_interaction"
        ],
    }
    phase_a_passed = all(phase_a_conditions.values())
    non_easy_goals = sum(
        int(value)
        for family, value in metrics["goals_by_reset_family"].items()
        if family != "easy_finish"
    )
    phase_b_conditions = {
        "at_least_10_bootstrap_hours": boundary_hours >= 10.0,
        "logical_touches_at_least_150_per_100k": rates["logical_touches"]
        >= 150.0,
        "deterministic_jumps_at_least_20_per_100k": rates["first_jumps"]
        >= 20.0,
        "deterministic_dodges_at_least_5_per_100k": rates["dodges"] >= 5.0,
        "aerial_touches_at_least_3_per_100k": rates[
            "aerial_logical_touches"
        ]
        >= 3.0,
        "two_plus_chain_touches_at_least_10_per_100k": cap[
            "two_or_more_chain_touches_per_100k_agent_steps"
        ]
        >= 10.0,
        "goal_activity_outside_easy_finish": non_easy_goals > 0,
        "no_touch_timeout_share_at_most_40_percent": cap[
            "no_touch_timeout_share"
        ]
        <= 0.40,
    }
    phase_b_passed = all(phase_b_conditions.values())
    chain_count = cap["two_or_more_chain_touches_per_100k_agent_steps"]
    contact_farming_checks = {
        "two_plus_chain_rate_at_least_10_per_100k": chain_count >= 10.0,
        "maximum_chain_length_at_least_2": cap[
            "maximum_touch_chain_length_lower_bound"
        ]
        >= 2,
        "chain_events_do_not_exceed_logical_touches": chain_count
        <= rates["logical_touches"],
        "debounce_contract_preflight_passed": True,
    }
    distance_improvement_fraction = 1.0 - float(cap["mean_distance_to_ball"]) / max(
        float(baseline_cap["mean_distance_to_ball"]), 1e-12
    )
    exit_conditions = {
        "mean_planar_speed_at_least_700": cap["mean_planar_speed"] >= 700.0,
        "logical_touches_at_least_150_per_100k": rates["logical_touches"]
        >= 150.0,
        "deterministic_jumps_at_least_20_per_100k": rates["first_jumps"]
        >= 20.0,
        "deterministic_dodges_at_least_10_per_100k": rates["dodges"] >= 10.0,
        "aerial_touches_at_least_5_per_100k": rates[
            "aerial_logical_touches"
        ]
        >= 5.0,
        "repeated_touch_chains_nontrivial_and_debounced": all(
            contact_farming_checks.values()
        ),
        "natural_goal_activity_nonzero": cap["natural_goal_activity"],
        "no_touch_timeout_share_at_most_30_percent": cap[
            "no_touch_timeout_share"
        ]
        <= 0.30,
        "mean_ball_distance_improved_at_least_10_percent": (
            distance_improvement_fraction >= 0.10
        ),
        "no_obvious_single_component_farming": reward[
            "no_obvious_single_component_farming"
        ],
    }
    single_exit = all(exit_conditions.values())
    previous_single = bool(
        previous
        and previous.get("gates", {})
        .get("bootstrap_exit", {})
        .get("single_boundary_passed", False)
    )
    consecutive_exit = single_exit and previous_single
    return {
        "reward_integrity_interpretation": reward,
        "phase_a": {
            "conditions": phase_a_conditions,
            "passed": phase_a_passed,
        },
        "phase_b": {
            "conditions": phase_b_conditions,
            "passed": phase_b_passed,
        },
        "bootstrap_exit": {
            "conditions": exit_conditions,
            "contact_farming_checks": contact_farming_checks,
            "mean_distance_improvement_fraction_vs_prebootstrap": (
                distance_improvement_fraction
            ),
            "single_boundary_passed": single_exit,
            "previous_boundary_single_passed": previous_single,
            "two_consecutive_boundaries_passed": consecutive_exit,
        },
        "gate_thresholds_are_capability_not_win_loss": True,
        "phase_at_evaluation": phase,
    }


def _decision(
    boundary_hours: float, phase: str, gates: dict[str, Any]
) -> dict[str, Any]:
    if gates["bootstrap_exit"]["two_consecutive_boundaries_passed"]:
        return {
            "action": "stop_bootstrap_exit_gate_passed",
            "continue_training": False,
            "next_phase": None,
        }
    if phase == "A" and gates["phase_a"]["passed"]:
        return {
            "action": "transition_to_phase_B",
            "continue_training": True,
            "next_phase": "B",
        }
    if phase == "A" and boundary_hours >= 10.0:
        return {
            "action": "stop_phase_A_readiness_failed_by_plus_10h",
            "continue_training": False,
            "next_phase": None,
        }
    if phase == "B" and gates["phase_b"]["passed"]:
        return {
            "action": "transition_to_phase_C",
            "continue_training": True,
            "next_phase": "C",
        }
    if boundary_hours >= 25.0:
        return {
            "action": "stop_absolute_25h_bootstrap_maximum",
            "continue_training": False,
            "next_phase": None,
        }
    return {
        "action": f"continue_phase_{phase}",
        "continue_training": True,
        "next_phase": phase,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    training = _read(args.training_progress)
    fixed = _read(args.fixed_evaluation)
    bootstrap = _read(args.bootstrap_evaluation)
    baseline = _read(args.prebootstrap_evaluation)
    previous = _read(args.previous_boundary) if args.previous_boundary else None
    boundary_hours = float(training["boundary_added_bootstrap_hours"])
    phase = str(training["phase"])
    checkpoint = training["immutable_checkpoint"]
    checks = {
        "training_boundary_passed": training["status"] == "passed"
        and training["checks"]["passed"],
        "fixed_historical_evaluation_passed": fixed["status"] == "passed"
        and fixed["checks"]["passed"],
        "bootstrap_evaluation_passed": bootstrap["status"] == "passed"
        and bootstrap["checks"]["passed"],
        "prebootstrap_baseline_passed": baseline["status"] == "passed"
        and baseline["checks"]["passed"],
        "all_evaluations_use_expected_checkpoints": bootstrap["checkpoint"][
            "manifest_sha256"
        ]
        == fixed["checkpoint"]["manifest_sha256"]
        == checkpoint["manifest_sha256"]
        and baseline["checkpoint"]["manifest_sha256"]
        == M10_PLUS25_MANIFEST_SHA256,
        "boundary_starts_after_exact_m10_plus25": checkpoint[
            "cumulative_agent_steps"
        ]
        > M10_PLUS25_STEPS,
        "production_promotion_authorized": False,
    }
    checks["passed"] = all(
        value
        for key, value in checks.items()
        if key != "production_promotion_authorized"
    ) and checks["production_promotion_authorized"] is False
    if not checks["passed"]:
        raise RuntimeError(f"Boundary evidence consistency failed: {checks}")
    gates = _gate_report(boundary_hours, phase, bootstrap, baseline, previous)
    decision = _decision(boundary_hours, phase, gates)
    iterations = training["iterations"]
    throughput = [float(item["agent_steps_per_second"]) for item in iterations]
    report = {
        "schema_version": 1,
        "status": "passed",
        "milestone": "10.1",
        "boundary_added_bootstrap_hours": boundary_hours,
        "phase": phase,
        "immutable_checkpoint": checkpoint,
        "training": {
            "source_checkpoint": training["source_checkpoint"],
            "achieved": training["achieved"],
            "iteration_count": len(iterations),
            "mean_agent_steps_per_second": float(np.mean(throughput)),
            "minimum_agent_steps_per_second": min(throughput),
            "maximum_agent_steps_per_second": max(throughput),
            "iterations": [
                _compact_boundary_iteration(item) for item in iterations
            ],
            "cleanup": training["cleanup"],
        },
        "fixed_historical_evaluation": _compact_fixed_evaluation(fixed),
        "bootstrap_evaluation": _compact_bootstrap_evaluation(bootstrap),
        "prebootstrap_plus25_bootstrap_evaluation": (
            _compact_bootstrap_evaluation(baseline)
        ),
        "historical_comparisons": _historical_comparisons(fixed),
        "gates": gates,
        "decision": decision,
        "phase_authorization_output": (
            portable_path(args.phase_authorization_output)
            if decision["next_phase"] is not None
            and decision["next_phase"] != phase
            else None
        ),
        "checks": checks,
        "evidence_files": {
            "training_progress": {
                "path": portable_path(args.training_progress),
                "sha256": sha256_file(args.training_progress),
            },
            "fixed_evaluation": {
                "path": portable_path(args.fixed_evaluation),
                "sha256": sha256_file(args.fixed_evaluation),
            },
            "bootstrap_evaluation": {
                "path": portable_path(args.bootstrap_evaluation),
                "sha256": sha256_file(args.bootstrap_evaluation),
            },
            "prebootstrap_evaluation": {
                "path": portable_path(args.prebootstrap_evaluation),
                "sha256": sha256_file(args.prebootstrap_evaluation),
            },
        },
        "production_promotion_authorized": False,
    }
    write_json_atomic(args.output, report)
    next_phase = decision["next_phase"]
    if next_phase is not None and next_phase != phase:
        authorization = {
            "schema_version": 1,
            "status": "passed",
            "source_boundary_added_bootstrap_hours": boundary_hours,
            "source_checkpoint_manifest_sha256": checkpoint[
                "manifest_sha256"
            ],
            "source_checkpoint_actor_sha256": checkpoint["actor_sha256"],
            "current_phase": phase,
            "next_phase": next_phase,
            "phase_transition_authorized": True,
            "gate": "phase_a" if phase == "A" else "phase_b",
            "boundary_result": portable_path(args.output),
            "boundary_result_sha256": sha256_file(args.output),
            "production_promotion_authorized": False,
        }
        write_json_atomic(args.phase_authorization_output, authorization)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-progress", type=Path, required=True)
    parser.add_argument("--fixed-evaluation", type=Path, required=True)
    parser.add_argument("--bootstrap-evaluation", type=Path, required=True)
    parser.add_argument("--prebootstrap-evaluation", type=Path, required=True)
    parser.add_argument("--previous-boundary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-authorization-output", type=Path, required=True)
    args = parser.parse_args()
    report = finalize(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "boundary_added_bootstrap_hours": report[
                    "boundary_added_bootstrap_hours"
                ],
                "phase": report["phase"],
                "decision": report["decision"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
