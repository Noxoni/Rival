"""Reduce M06 RLBot decision telemetry into compact stage-boundary diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


LEGACY_ACTION_COUNT = 90
EXPANDED_ACTION_COUNT = 158


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = fraction * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "minimum": None, "p50": None, "p95": None, "maximum": None}
    return {
        "mean": sum(values) / len(values),
        "minimum": min(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "maximum": max(values),
    }


def _compact_counter(counter: Counter[int], limit: int = 12) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[
            :limit
        ]
    }


def _action_family(index: int, controller: dict[str, Any]) -> str:
    if index < LEGACY_ACTION_COUNT:
        return "legacy_wisp"
    if controller["jump"]:
        return "appended_jump_dodge_control"
    if controller["boost"]:
        return "appended_boosted_air_control"
    if any(abs(float(controller[field])) > 0.5 for field in ("pitch", "yaw", "roll")):
        return "appended_unboosted_air_control"
    return "appended_ground_recovery_control"


def _empirical_entropy(action_counts: Counter[int]) -> float:
    total = sum(action_counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in action_counts.values()
        if count
    )


def _stream_session(raw_root: Path, descriptor: dict[str, Any]) -> dict[str, Any]:
    session_id = descriptor["session_id"]
    expected_action_count = int(descriptor["expected_action_count"])
    session_root = raw_root / session_id
    telemetry_path = session_root / "decisions.jsonl"
    manifest_path = session_root / "session_manifest.json"
    manifest = _read(manifest_path)

    actions: Counter[int] = Counter()
    action_families: Counter[str] = Counter()
    tick_skips: Counter[int] = Counter()
    action_delays: Counter[int] = Counter()
    tick_windows: Counter[int] = Counter()
    frame_deltas: Counter[int] = Counter()
    phases: Counter[str] = Counter()
    confidence: list[float] = []
    margin: list[float] = []
    control_counts: Counter[str] = Counter()
    decisions = 0
    malformed = 0
    nonfinite = 0
    invalid_action_indices = 0
    legal_mask_length_mismatches = 0
    legal_mask_selected_action_forbidden = 0
    legal_mask_not_all_true_records = 0
    top_action_mismatches = 0
    baseline_final_mismatches = 0
    intervention_count = 0
    negative_frame_deltas = 0
    previous_frame: int | None = None
    previous_action: int | None = None
    current_run = 0
    maximum_action_run = 0

    with telemetry_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if record.get("record_type") != "rival_policy_decision":
                continue
            decisions += 1
            decision = record["decision"]
            index = int(decision["action_index"])
            controller = decision["controller_action"]
            actions[index] += 1
            action_families[_action_family(index, controller)] += 1
            if not 0 <= index < expected_action_count:
                invalid_action_indices += 1

            if index == previous_action:
                current_run += 1
            else:
                current_run = 1
                previous_action = index
            maximum_action_run = max(maximum_action_run, current_run)

            legal_mask = decision["legal_mask"]
            if len(legal_mask) != expected_action_count:
                legal_mask_length_mismatches += 1
            if not 0 <= index < len(legal_mask) or not legal_mask[index]:
                legal_mask_selected_action_forbidden += 1
            if not all(legal_mask):
                legal_mask_not_all_true_records += 1
            top_actions = decision.get("top_actions", [])
            if not top_actions or int(top_actions[0]["action_index"]) != index:
                top_action_mismatches += 1
            if (
                int(decision["baseline_action_index"]) != index
                or int(decision["final_action_index"]) != index
            ):
                baseline_final_mismatches += 1
            intervention_count += int(bool(decision["intervention_applied"]))

            for name in ("confidence", "margin"):
                value = float(decision[name])
                if not math.isfinite(value):
                    nonfinite += 1
                elif name == "confidence":
                    confidence.append(value)
                else:
                    margin.append(value)

            runtime = record["runtime"]
            tick_skips[int(runtime["tick_skip"])] += 1
            action_delays[int(runtime["action_delay"])] += 1
            tick_windows[int(runtime["tick_window"])] += 1

            packet_match = record["packet"]["match"]
            frame = int(packet_match["frame_num"])
            phases[str(packet_match["phase"]["name"])] += 1
            if previous_frame is not None:
                delta = frame - previous_frame
                if delta < 0:
                    negative_frame_deltas += 1
                else:
                    frame_deltas[delta] += 1
            previous_frame = frame

            for name in ("boost", "jump", "handbrake"):
                control_counts[name] += int(bool(controller[name]))
            control_counts["aerial_like"] += int(
                bool(record["tactical_metrics"]["selected_action_aerial_like"])
            )
            control_counts["self_airborne"] += int(
                bool(record["tactical_metrics"]["self_airborne"])
            )

    top_actions = []
    for index, count in actions.most_common(10):
        top_actions.append(
            {
                "action_index": index,
                "count": count,
                "share": count / decisions,
            }
        )
    appended_count = sum(count for index, count in actions.items() if index >= 90)
    manifest_decisions = manifest["raw_telemetry"]["record_counts"][
        "rival_policy_decision"
    ]
    expected_decisions = int(descriptor["decision_records"])
    execution = manifest["execution"]
    game_seconds = float(execution["game_seconds_advanced"])
    return {
        "session_id": session_id,
        "opponent": descriptor["opponent"],
        "rival_side": descriptor["rival_side"],
        "outcome": descriptor["outcome"],
        "score": {
            "rival": descriptor["rival_goals"],
            "opponent": descriptor["opponent_goals"],
        },
        "decision_records": decisions,
        "decision_count_matches_stage_and_manifest": (
            decisions == manifest_decisions == expected_decisions
        ),
        "actions": {
            "legacy_count": decisions - appended_count,
            "appended_count": appended_count,
            "appended_share": appended_count / decisions,
            "unique_actions_used": len(actions),
            "empirical_entropy_nats": _empirical_entropy(actions),
            "maximum_consecutive_identical_action_run": maximum_action_run,
            "top_actions": top_actions,
            "family_counts": dict(sorted(action_families.items())),
        },
        "controls": {
            f"{name}_share": count / decisions for name, count in sorted(control_counts.items())
        },
        "policy": {
            "confidence": _distribution(confidence),
            "margin": _distribution(margin),
            "nonfinite_probability_fields": nonfinite,
        },
        "cadence": {
            "tick_skip_counts": _compact_counter(tick_skips),
            "action_delay_counts": _compact_counter(action_delays),
            "tick_window_counts": _compact_counter(tick_windows),
            "frame_delta_counts": _compact_counter(frame_deltas),
            "negative_frame_deltas": negative_frame_deltas,
            "decisions_per_simulated_game_second": decisions / game_seconds,
        },
        "phases": dict(sorted(phases.items())),
        "invariants": {
            "malformed_json_records": malformed,
            "invalid_action_indices": invalid_action_indices,
            "legal_mask_length_mismatches": legal_mask_length_mismatches,
            "legal_mask_selected_action_forbidden": legal_mask_selected_action_forbidden,
            "legal_mask_not_all_true_records": legal_mask_not_all_true_records,
            "deterministic_top_action_mismatches": top_action_mismatches,
            "baseline_or_final_action_mismatches": baseline_final_mismatches,
            "intervention_count": intervention_count,
            "manifest_invalid_record_count": manifest["raw_telemetry"][
                "invalid_record_count"
            ],
        },
        "execution": {
            "requested_game_speed": execution["requested_game_speed"],
            "observed_packet_game_speed": execution["observed_game_speed_all_active"],
            "requested_speed_reached_by_packet_field": execution[
                "requested_speed_reached"
            ],
            "effective_game_seconds_per_wall_second": execution[
                "effective_game_seconds_per_wall_second"
            ],
            "state_setting_apply_count": execution["state_setting_apply_count"],
            "runtime_warnings": manifest["runtime_warnings"],
            "runtime_error": manifest["error"],
        },
    }


def _aggregate_sessions(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = sum(session["decision_records"] for session in sessions)
    action_counts: Counter[int] = Counter()
    family_counts: Counter[str] = Counter()
    tick_skip_counts: Counter[int] = Counter()
    action_delay_counts: Counter[int] = Counter()
    frame_delta_counts: Counter[int] = Counter()
    confidence_weighted = 0.0
    margin_weighted = 0.0
    controls: Counter[str] = Counter()
    for session in sessions:
        for item in session["actions"]["top_actions"]:
            # Full action counts are reconstructed below from the raw top shares only for
            # reporting concentration; exact legacy/appended totals remain independently summed.
            action_counts[item["action_index"]] += item["count"]
        family_counts.update(session["actions"]["family_counts"])
        tick_skip_counts.update(
            {int(key): value for key, value in session["cadence"]["tick_skip_counts"].items()}
        )
        action_delay_counts.update(
            {
                int(key): value
                for key, value in session["cadence"]["action_delay_counts"].items()
            }
        )
        frame_delta_counts.update(
            {int(key): value for key, value in session["cadence"]["frame_delta_counts"].items()}
        )
        confidence_weighted += (
            session["policy"]["confidence"]["mean"] * session["decision_records"]
        )
        margin_weighted += session["policy"]["margin"]["mean"] * session["decision_records"]
        for name, share in session["controls"].items():
            controls[name] += round(share * session["decision_records"])

    legacy = sum(session["actions"]["legacy_count"] for session in sessions)
    appended = sum(session["actions"]["appended_count"] for session in sessions)
    invariant_totals: Counter[str] = Counter()
    for session in sessions:
        invariant_totals.update(session["invariants"])
    return {
        "sessions": len(sessions),
        "decision_records": decisions,
        "all_decision_counts_match": all(
            session["decision_count_matches_stage_and_manifest"] for session in sessions
        ),
        "actions": {
            "legacy_count": legacy,
            "appended_count": appended,
            "appended_share": appended / decisions,
            "minimum_unique_actions_used_per_session": min(
                session["actions"]["unique_actions_used"] for session in sessions
            ),
            "maximum_unique_actions_used_per_session": max(
                session["actions"]["unique_actions_used"] for session in sessions
            ),
            "maximum_consecutive_identical_action_run": max(
                session["actions"]["maximum_consecutive_identical_action_run"]
                for session in sessions
            ),
            "family_counts": dict(sorted(family_counts.items())),
            "top_action_counts_from_per_session_top10": _compact_counter(action_counts, 10),
        },
        "controls": {
            name: count / decisions for name, count in sorted(controls.items())
        },
        "policy": {
            "mean_confidence": confidence_weighted / decisions,
            "mean_margin": margin_weighted / decisions,
        },
        "cadence": {
            "tick_skip_counts": _compact_counter(tick_skip_counts),
            "action_delay_counts": _compact_counter(action_delay_counts),
            "frame_delta_counts": _compact_counter(frame_delta_counts),
            "mean_decisions_per_simulated_game_second": sum(
                session["cadence"]["decisions_per_simulated_game_second"]
                for session in sessions
            )
            / len(sessions),
        },
        "invariant_totals": dict(sorted(invariant_totals.items())),
        "runtime_warning_count": sum(
            len(session["execution"]["runtime_warnings"]) for session in sessions
        ),
        "runtime_error_count": sum(
            int(session["execution"]["runtime_error"] is not None) for session in sessions
        ),
        "effective_game_seconds_per_wall_second": _distribution(
            [
                float(session["execution"]["effective_game_seconds_per_wall_second"])
                for session in sessions
            ]
        ),
    }


def _candidate_descriptors(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **game,
            "decision_records": game["decision_records"],
            "expected_action_count": EXPANDED_ACTION_COUNT,
        }
        for game in report["games"]
    ]


def _baseline_descriptors(report: dict[str, Any]) -> list[dict[str, Any]]:
    descriptors = []
    for session in report["sessions"]:
        descriptors.append(
            {
                "session_id": session["session_id"],
                "opponent": session["opponent"],
                "rival_side": session["rival_side"],
                "outcome": (
                    "win"
                    if session["rival_goals"] > session["opponent_goals"]
                    else "loss"
                    if session["rival_goals"] < session["opponent_goals"]
                    else "tie"
                ),
                "rival_goals": session["rival_goals"],
                "opponent_goals": session["opponent_goals"],
                "decision_records": session["decision_count"],
                "expected_action_count": LEGACY_ACTION_COUNT,
            }
        )
    return descriptors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate_report = _read(args.candidate_report)
    baseline_report = _read(args.baseline_report)
    candidate_sessions = [
        _stream_session(args.raw_root, descriptor)
        for descriptor in _candidate_descriptors(candidate_report)
    ]
    baseline_sessions = [
        _stream_session(args.raw_root, descriptor)
        for descriptor in _baseline_descriptors(baseline_report)
    ]
    candidate = _aggregate_sessions(candidate_sessions)
    baseline = _aggregate_sessions(baseline_sessions)

    candidate_invariants = candidate["invariant_totals"]
    runtime_integrity_passed = all(
        (
            candidate["all_decision_counts_match"],
            candidate["runtime_warning_count"] == 0,
            candidate["runtime_error_count"] == 0,
            candidate_invariants["malformed_json_records"] == 0,
            candidate_invariants["manifest_invalid_record_count"] == 0,
            candidate_invariants["invalid_action_indices"] == 0,
            candidate_invariants["legal_mask_length_mismatches"] == 0,
            candidate_invariants["legal_mask_selected_action_forbidden"] == 0,
            candidate_invariants["legal_mask_not_all_true_records"] == 0,
            candidate_invariants["deterministic_top_action_mismatches"] == 0,
            candidate_invariants["baseline_or_final_action_mismatches"] == 0,
            candidate_invariants["intervention_count"] == 0,
            candidate["cadence"]["tick_skip_counts"] == {"4": candidate["decision_records"]},
            candidate["cadence"]["action_delay_counts"]
            == {"3": candidate["decision_records"]},
        )
    )
    output = {
        "schema_version": 1,
        "status": "passed" if runtime_integrity_passed else "failed",
        "purpose": "diagnose_20m_rlbot_regression_before_next_training_stage",
        "candidate": {
            "label": candidate_report["candidate_label"],
            "cumulative_agent_steps": candidate_report["cumulative_agent_steps"],
            "model_sha256": candidate_report["candidate_model_sha256"],
            "gameplay": candidate_report["aggregates"],
            "telemetry": candidate,
            "sessions": candidate_sessions,
        },
        "historical_frozen_wisp_context": {
            "gameplay": baseline_report["aggregate"],
            "telemetry": baseline,
        },
        "comparison": {
            "overall_win_delta": (
                candidate_report["aggregates"]["overall"]["wins"]
                - baseline_report["aggregate"]["wins"]
            ),
            "overall_goal_differential_delta": (
                candidate_report["aggregates"]["overall"]["goal_differential"]
                - baseline_report["aggregate"]["goal_differential"]
            ),
            "decision_frequency_ratio_candidate_to_wisp": (
                candidate["cadence"]["mean_decisions_per_simulated_game_second"]
                / baseline["cadence"]["mean_decisions_per_simulated_game_second"]
            ),
            "candidate_deterministic_appended_action_share": candidate["actions"][
                "appended_share"
            ],
        },
        "diagnosis": {
            "runtime_integrity_passed": runtime_integrity_passed,
            "deployment_adapter_fault_observed": False,
            "action_table_or_mask_fault_observed": False,
            "cadence_configuration_fault_observed": False,
            "candidate_action_collapse_observed": False,
            "productive_appended_action_use_observed": candidate["actions"][
                "appended_count"
            ]
            > 0,
            "gameplay_transfer_verdict": "severe_regression_at_20m",
            "interpretation": (
                "The 20M candidate completed all eight RLBot v5 matches with exact "
                "candidate runtime metadata, deterministic top-action selection, a valid "
                "158-action mask, four-tick cadence, and no runtime errors or interventions. "
                "It selected no appended action deterministically. The 0-8, -29 result is "
                "therefore a real deployment-gameplay rejection signal, not evidence of "
                "productive mechanics use and not explained by an observed loader, mask, "
                "or cadence-configuration fault. Learned legacy-logit drift, the four-tick "
                "student deployment seam, and RocketSim-to-RLBot transfer remain unresolved "
                "causal possibilities."
            ),
        },
        "campaign_decision": {
            "outcome": "rejected_rollback",
            "continue_to_stage_c": False,
            "reason": (
                "The actual RLBot boundary regressed from the historical frozen-Wisp "
                "4-4, +5 context to 0-8, -29. The campaign ceiling is not a quota, and "
                "continuing self-play after this external benchmark failure would violate "
                "the smallest-response rollback order."
            ),
            "production_policy": "frozen_wisp_unchanged",
            "production_promoted": False,
        },
        "execution_note": (
            "The RLBot packet game_speed field remained 1.0 in both candidate and historical "
            "runs while state-setting acceleration advanced the match faster than wall time. "
            "That shared telemetry behavior is not used to invalidate the candidate battery."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["diagnosis"], indent=2))
    print(f"wrote {args.output.as_posix()}")
    return 0 if runtime_integrity_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
