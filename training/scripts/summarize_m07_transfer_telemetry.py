"""Reduce the Milestone 07 transfer matrix telemetry into compact diagnostics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPOSITORY_ROOT / "evidence/raw"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "minimum": None,
            "p50": None,
            "p95": None,
            "maximum": None,
        }

    def percentile(fraction: float) -> float:
        location = fraction * (len(finite) - 1)
        lower = math.floor(location)
        upper = math.ceil(location)
        if lower == upper:
            return finite[lower]
        weight = location - lower
        return finite[lower] * (1 - weight) + finite[upper] * weight

    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "minimum": finite[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "maximum": finite[-1],
    }


def _counter(counter: Counter[Any], limit: int = 20) -> dict[str, int]:
    return {
        str(key): int(count)
        for key, count in counter.most_common(limit)
    }


def _touch_time(player: dict[str, Any]) -> float | None:
    touch = player.get("latest_touch") or {}
    value = touch.get("game_seconds")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _summarize_session(mode: str, definition: dict[str, Any], game: dict[str, Any]) -> dict[str, Any]:
    path = RAW_ROOT / game["session_id"] / "decisions.jsonl"
    expected_actions = int(definition["expected_action_count"])
    expected_tick = int(definition["tick_skip"])
    actions: Counter[int] = Counter()
    transitions: Counter[tuple[int, int]] = Counter()
    frame_deltas: Counter[int] = Counter()
    tick_windows: Counter[int] = Counter()
    confidence: list[float] = []
    margin: list[float] = []
    boost: list[float] = []
    eta_advantage: list[float] = []
    distance_advantage: list[float] = []
    control_counts: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    touches: Counter[str] = Counter()
    latest_touch = {"self": None, "opponent": None}
    possession_loss_transitions = 0
    previous_eta_favorable: bool | None = None
    previous_action: int | None = None
    previous_frame: int | None = None
    current_run = maximum_run = action_changes = comparisons = 0
    decisions = malformed = nonfinite = 0
    invariant_failures: Counter[str] = Counter()
    session_start = None
    session_end = None
    for line in path.open("r", encoding="utf-8"):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        if record.get("record_type") == "rival_session_start":
            session_start = record
            continue
        if record.get("record_type") == "rival_session_end":
            session_end = record
            continue
        if record.get("record_type") != "rival_policy_decision":
            continue
        decisions += 1
        decision = record["decision"]
        action = int(decision["action_index"])
        actions[action] += 1
        if previous_action is not None:
            comparisons += 1
            action_changes += int(action != previous_action)
            transitions[(previous_action, action)] += 1
        if action == previous_action:
            current_run += 1
        else:
            current_run = 1
        maximum_run = max(maximum_run, current_run)
        previous_action = action

        mask = decision["legal_mask"]
        if len(mask) != expected_actions:
            invariant_failures["legal_mask_length"] += 1
        if not 0 <= action < len(mask) or not mask[action]:
            invariant_failures["selected_action_illegal"] += 1
        if action >= 90:
            invariant_failures["selected_appended_action"] += 1
        if expected_actions == 158 and any(mask[90:]):
            invariant_failures["legacy_only_suffix_not_hard_masked"] += 1
        if not decision.get("top_actions") or int(
            decision["top_actions"][0]["action_index"]
        ) != action:
            invariant_failures["deterministic_top1_mismatch"] += 1
        if decision.get("intervention_applied"):
            invariant_failures["intervention_applied"] += 1

        for key, target in (("confidence", confidence), ("margin", margin)):
            value = float(decision[key])
            if math.isfinite(value):
                target.append(value)
            else:
                nonfinite += 1
        runtime = record["runtime"]
        if int(runtime["tick_skip"]) != expected_tick:
            invariant_failures["tick_skip_mismatch"] += 1
        if int(runtime["action_delay"]) != expected_tick - 1:
            invariant_failures["action_delay_mismatch"] += 1
        if not runtime.get("transfer_diagnostic_mode"):
            invariant_failures["diagnostic_mode_off"] += 1
        if expected_actions == 158 and not runtime.get("candidate_legacy_only"):
            invariant_failures["candidate_legacy_only_off"] += 1
        tick_windows[int(runtime["tick_window"])] += 1

        packet = record["packet"]
        packet_match = packet["match"]
        phases[str(packet_match["phase"]["name"])] += 1
        frame = int(packet_match["frame_num"])
        if previous_frame is not None:
            delta = frame - previous_frame
            if delta < 0:
                invariant_failures["negative_frame_delta"] += 1
            else:
                frame_deltas[delta] += 1
        previous_frame = frame

        tactical = record["tactical_metrics"]
        boost.append(float(tactical["self_boost"]))
        eta = float(tactical["possession_eta_advantage"])
        distance = float(tactical["distance_opponent_ball"]) - float(
            tactical["distance_self_ball"]
        )
        eta_advantage.append(eta)
        distance_advantage.append(distance)
        favorable = eta > 0
        if previous_eta_favorable is True and not favorable:
            possession_loss_transitions += 1
        previous_eta_favorable = favorable
        controller = decision["controller_action"]
        for key in ("boost", "jump", "handbrake"):
            control_counts[key] += int(bool(controller[key]))
        control_counts["airborne"] += int(bool(tactical["self_airborne"]))
        control_counts["low_boost"] += int(float(tactical["self_boost"]) < 20.0)
        control_counts["close_ball"] += int(float(tactical["ball_distance"]) < 1000.0)

        self_index = int(packet["self_index"])
        opponent_indices = list(packet["opponent_indices"])
        roles = {"self": self_index}
        if opponent_indices:
            roles["opponent"] = int(opponent_indices[0])
        for role, index in roles.items():
            value = _touch_time(packet["players"][index])
            if value is not None and (
                latest_touch[role] is None or value > float(latest_touch[role]) + 1e-4
            ):
                touches[role] += 1
                latest_touch[role] = value

    if session_start is None or session_end is None:
        invariant_failures["session_boundary_missing"] += 1
    if decisions != int(game["decision_records"]):
        invariant_failures["decision_count_mismatch"] += 1
    runtime_clean = bool(game["runtime_health"]["passed"])
    return {
        "session_id": game["session_id"],
        "opponent": game["opponent"],
        "rival_side": game["rival_side"],
        "outcome": game["outcome"],
        "score": {"rival": game["rival_goals"], "opponent": game["opponent_goals"]},
        "runtime_clean": runtime_clean,
        "known_runtime_anomaly": game.get("known_runtime_anomaly"),
        "raw": {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        },
        "decisions": decisions,
        "actions": {
            "unique": len(actions),
            "top_counts": _counter(actions),
            "appended_selected": sum(count for index, count in actions.items() if index >= 90),
            "change_share": action_changes / max(comparisons, 1),
            "maximum_identical_run": maximum_run,
            "top_transitions": _counter(transitions),
        },
        "policy": {
            "confidence": _distribution(confidence),
            "margin": _distribution(margin),
            "nonfinite_values": nonfinite,
        },
        "cadence": {
            "tick_skip": expected_tick,
            "action_delay": expected_tick - 1,
            "tick_window_counts": _counter(tick_windows),
            "frame_delta_counts": _counter(frame_deltas),
        },
        "gameplay_proxies": {
            "touches": {"self": touches["self"], "opponent": touches["opponent"]},
            "possession_eta_advantage": _distribution(eta_advantage),
            "eta_advantage_positive_share": sum(value > 0 for value in eta_advantage)
            / max(len(eta_advantage), 1),
            "distance_advantage": _distribution(distance_advantage),
            "distance_advantage_positive_share": sum(
                value > 0 for value in distance_advantage
            )
            / max(len(distance_advantage), 1),
            "possession_loss_transition_count": possession_loss_transitions,
            "self_boost": _distribution(boost),
            "controller_and_state_shares": {
                key: count / max(decisions, 1)
                for key, count in sorted(control_counts.items())
            },
        },
        "phases": dict(sorted(phases.items())),
        "invariants": {
            "malformed_json_records": malformed,
            "failure_counts": dict(sorted(invariant_failures.items())),
            "passed": not malformed and not invariant_failures and nonfinite == 0,
        },
    }


def _aggregate(mode: str, definition: dict[str, Any], sessions: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = sum(session["decisions"] for session in sessions)
    touches = Counter()
    action_counts: Counter[int] = Counter()
    confidence = []
    margin = []
    eta_positive_weighted = distance_positive_weighted = 0.0
    control_weighted: Counter[str] = Counter()
    for session in sessions:
        touches.update(session["gameplay_proxies"]["touches"])
        action_counts.update(
            {int(key): count for key, count in session["actions"]["top_counts"].items()}
        )
        if session["policy"]["confidence"]["mean"] is not None:
            confidence.extend(
                [float(session["policy"]["confidence"]["mean"])] * session["decisions"]
            )
            margin.extend(
                [float(session["policy"]["margin"]["mean"])] * session["decisions"]
            )
        eta_positive_weighted += (
            session["gameplay_proxies"]["eta_advantage_positive_share"]
            * session["decisions"]
        )
        distance_positive_weighted += (
            session["gameplay_proxies"]["distance_advantage_positive_share"]
            * session["decisions"]
        )
        for key, share in session["gameplay_proxies"][
            "controller_and_state_shares"
        ].items():
            control_weighted[key] += round(float(share) * session["decisions"])
    return {
        "mode": mode,
        "policy": definition["policy"],
        "tick_skip": definition["tick_skip"],
        "action_delay": definition["action_delay"],
        "gameplay": definition["aggregates"],
        "sessions": len(sessions),
        "runtime_clean_sessions": sum(session["runtime_clean"] for session in sessions),
        "decisions": decisions,
        "all_invariants_passed": all(session["invariants"]["passed"] for session in sessions),
        "appended_selected": sum(session["actions"]["appended_selected"] for session in sessions),
        "action_counts_from_per_session_top20": _counter(action_counts),
        "mean_confidence": sum(confidence) / max(len(confidence), 1),
        "mean_margin": sum(margin) / max(len(margin), 1),
        "touches": dict(touches),
        "eta_advantage_positive_share": eta_positive_weighted / max(decisions, 1),
        "distance_advantage_positive_share": distance_positive_weighted / max(decisions, 1),
        "controller_and_state_shares": {
            key: count / max(decisions, 1) for key, count in sorted(control_weighted.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPOSITORY_ROOT / "training/results/milestone07/transfer_matrix.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "training/results/milestone07/transfer_telemetry.json",
    )
    args = parser.parse_args()
    matrix = _read(args.matrix)
    modes = {}
    all_sessions = []
    for mode, definition in matrix["modes"].items():
        sessions = [
            _summarize_session(mode, definition, game) for game in definition["games"]
        ]
        modes[mode] = {
            "aggregate": _aggregate(mode, definition, sessions),
            "sessions": sessions,
        }
        all_sessions.extend(sessions)
    clean = sum(session["runtime_clean"] for session in all_sessions)
    output = {
        "schema_version": 1,
        "status": (
            "completed_with_runtime_anomaly"
            if clean != len(all_sessions)
            else "passed"
        ),
        "purpose": "milestone07_transfer_matrix_telemetry",
        "matrix_report": args.matrix.relative_to(REPOSITORY_ROOT).as_posix(),
        "modes": modes,
        "hard_invariants": {
            "sessions": len(all_sessions),
            "runtime_clean_sessions": clean,
            "telemetry_invariant_passed_sessions": sum(
                session["invariants"]["passed"] for session in all_sessions
            ),
            "deterministic_appended_action_selections": sum(
                session["actions"]["appended_selected"] for session in all_sessions
            ),
            "all_legacy_only_masks_and_selections_valid": all(
                session["invariants"]["passed"] for session in all_sessions
            ),
        },
        "known_excluded_sessions": matrix.get("excluded_sessions", []),
        "production_promoted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": output["status"], **output["hard_invariants"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
