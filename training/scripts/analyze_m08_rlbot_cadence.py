"""Reduce M08 forced-PASS RLBot telemetry into a strict cadence gate.

Match outcomes are deliberately excluded from the pass/fail calculation.  The
gate is based on controller clocks, decision sequences, runtime identity, and
forced-PASS invariants.  Scores are retained only as bounded behavioral context.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = (
    REPOSITORY_ROOT
    / "training/results/milestone08/zero_step_dual_rate_rlbot.json"
)
DEFAULT_SOFTWARE_GATE = (
    REPOSITORY_ROOT / "training/results/milestone08/pretraining_gates.json"
)
DEFAULT_M07_MATRIX = (
    REPOSITORY_ROOT / "training/results/milestone07/transfer_matrix.json"
)
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "training/results/milestone08/zero_step_transfer_gate.json"
)
EXPECTED_POLICY_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tick_histogram(times: list[float]) -> tuple[dict[str, int], int | None, float]:
    """Summarize uninterrupted intervals in 120-Hz physics ticks.

    Intervals over half a second are lifecycle gaps (startup, replay, or goal
    transition), not scheduler samples.  Nearby +/- one-tick packet jitter is
    retained and reported rather than rounded away.
    """
    ticks = [
        int(round((current - previous) * 120.0))
        for previous, current in zip(times, times[1:])
        if 0.0 <= current - previous <= 0.5
    ]
    counts = Counter(ticks)
    mode = None if not counts else counts.most_common(1)[0][0]
    near_mode = 0.0
    if mode is not None and ticks:
        near_mode = sum(abs(value - mode) <= 1 for value in ticks) / len(ticks)
    return (
        {str(key): counts[key] for key in sorted(counts)},
        mode,
        near_mode,
    )


def _session_gate(
    game: dict[str, Any], *, raw_root: Path
) -> dict[str, Any]:
    session_id = str(game["session_id"])
    path = raw_root / session_id / "decisions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing telemetry for {session_id}: {path}")

    record_counts: Counter[str] = Counter()
    invalid_json_records = 0
    wrong_session_records = 0
    start_record: dict[str, Any] | None = None
    end_record: dict[str, Any] | None = None
    strategic_count = 0
    mechanics_count = 0
    strategic_sequence_exact = True
    mechanics_sequence_exact = True
    strategic_actions_exact = True
    strategic_runtime_exact = True
    forced_pass_exact = True
    strategic_times: list[float] = []
    mechanics_times: list[float] = []
    strategic_phases: dict[int, str | None] = {}
    mechanics_per_strategic: Counter[int] = Counter()
    controller_sources: Counter[str] = Counter()

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_json_records += 1
                continue
            record_type = str(record.get("record_type"))
            record_counts[record_type] += 1
            if record.get("session_id") != session_id:
                wrong_session_records += 1

            if record_type == "rival_session_start":
                start_record = record
                continue
            if record_type == "rival_session_end":
                end_record = record
                continue
            if record_type == "rival_policy_decision":
                decision = record.get("decision") or {}
                runtime = record.get("runtime") or {}
                packet = record.get("packet") or {}
                match = packet.get("match") or {}
                strategic_sequence_exact &= decision.get("tick") == strategic_count
                action = decision.get("action_index")
                baseline = decision.get("baseline_action_index")
                final = decision.get("final_action_index")
                legal_mask = decision.get("legal_mask")
                strategic_actions_exact &= bool(
                    isinstance(action, int)
                    and 0 <= action < 90
                    and action == baseline == final
                    and isinstance(legal_mask, list)
                    and len(legal_mask) == 90
                )
                source = str(runtime.get("last_controller_source"))
                controller_sources[source] += 1
                strategic_runtime_exact &= bool(
                    runtime.get("deterministic") is True
                    and runtime.get("strategic_overrides_enabled") is False
                    and runtime.get("transfer_diagnostic_mode") is True
                    and runtime.get("candidate_legacy_only") is False
                    and runtime.get("tick_skip") == 8
                    and runtime.get("action_delay") == 7
                    and runtime.get("dual_rate_enabled") is True
                    and source in {"strategic", "strategic_pass_through"}
                )
                game_time = decision.get("game_time")
                if isinstance(game_time, (int, float)):
                    strategic_times.append(float(game_time))
                strategic_phases[strategic_count + 1] = (
                    (match.get("phase") or {}).get("name")
                )
                strategic_count += 1
                continue
            if record_type == "rival_mechanics_decision":
                mechanics_sequence_exact &= (
                    record.get("mechanics_decision") == mechanics_count
                )
                policy_tick = record.get("strategic_policy_tick")
                if isinstance(policy_tick, int):
                    mechanics_per_strategic[policy_tick] += 1
                else:
                    forced_pass_exact = False
                forced_pass_exact &= bool(
                    record.get("choice") == 0
                    and record.get("pass") is True
                    and record.get("override_selected") is False
                    and record.get("global_action_index") is None
                    and record.get("force_pass") is True
                    and record.get("mean_pass_probability") == 1.0
                    and record.get("scheduler") == "[previous,new,new,new]"
                )
                game_time = record.get("game_time")
                if isinstance(game_time, (int, float)):
                    mechanics_times.append(float(game_time))
                mechanics_count += 1

    strategic_hist, strategic_mode, strategic_near_mode = _tick_histogram(
        strategic_times
    )
    mechanics_hist, mechanics_mode, mechanics_near_mode = _tick_histogram(
        mechanics_times
    )
    partial_windows = []
    invalid_partial_windows = []
    for policy_tick in range(1, strategic_count + 1):
        decisions = mechanics_per_strategic[policy_tick]
        if decisions == 2:
            continue
        phase = strategic_phases.get(policy_tick)
        classification = (
            "startup"
            if policy_tick <= 2
            else "terminal"
            if policy_tick == strategic_count
            else "countdown_boundary"
            if phase == "Countdown"
            else "in_play"
        )
        entry = {
            "strategic_policy_tick": policy_tick,
            "mechanics_decisions": decisions,
            "phase": phase,
            "classification": classification,
        }
        partial_windows.append(entry)
        if decisions not in {1, 2} or classification == "in_play":
            invalid_partial_windows.append(entry)

    metadata = (start_record or {}).get("metadata") or {}
    policy_runtime = metadata.get("policy_runtime") or {}
    start_contract_exact = bool(
        policy_runtime.get("mode") == "m08_dual_rate_forced_pass"
        and policy_runtime.get("candidate_enabled") is False
        and policy_runtime.get("transfer_diagnostic_mode") is True
        and policy_runtime.get("legacy_only") is False
        and policy_runtime.get("tick_skip") == 8
        and policy_runtime.get("action_count") == 90
        and policy_runtime.get("dual_rate_enabled") is True
        and policy_runtime.get("dual_rate_version") == "RivalLiveDualRateV1"
        and policy_runtime.get("strategic_tick_skip") == 8
        and policy_runtime.get("mechanics_tick_skip") == 4
        and policy_runtime.get("mechanics_action_count") == 69
        and policy_runtime.get("mechanics_force_pass") is True
        and policy_runtime.get("mechanics_model_path") is None
        and metadata.get("rival_model_sha256") == EXPECTED_POLICY_HASHES
    )
    record_count_exact = bool(
        strategic_count == game.get("decision_records")
        and mechanics_count == game.get("mechanics_decision_records")
        and (end_record or {}).get("decision_record_count") == strategic_count
    )
    source_sequence_exact = bool(
        controller_sources.get("strategic", 0) == 1
        and controller_sources.get("strategic_pass_through", 0)
        == strategic_count - 1
        and len(controller_sources) == 2
    )
    ratio = (
        0.0 if strategic_count == 0 else mechanics_count / (2.0 * strategic_count)
    )
    checks = {
        "one_start_and_end_record": bool(
            record_counts["rival_session_start"] == 1
            and record_counts["rival_session_end"] == 1
        ),
        "no_invalid_or_cross_session_records": bool(
            invalid_json_records == 0 and wrong_session_records == 0
        ),
        "reported_record_counts_exact": record_count_exact,
        "runtime_identity_and_hashes_exact": start_contract_exact,
        "strategic_decision_sequence_contiguous": strategic_sequence_exact,
        "mechanics_decision_sequence_contiguous": mechanics_sequence_exact,
        "strategic_action_mask_and_passthrough_exact": strategic_actions_exact,
        "strategic_runtime_flags_exact": strategic_runtime_exact,
        "forced_pass_records_exact": forced_pass_exact,
        "controller_source_sequence_exact": source_sequence_exact,
        "strategic_clock_mode_is_8_ticks": bool(
            strategic_mode == 8 and strategic_near_mode >= 0.95
        ),
        "mechanics_clock_mode_is_4_ticks": bool(
            mechanics_mode == 4 and mechanics_near_mode >= 0.95
        ),
        "two_to_one_clock_ratio_sustained": ratio >= 0.999,
        "partial_windows_are_lifecycle_only": not invalid_partial_windows,
        "runner_completed_cleanly": bool(
            game.get("runner_status") == "complete"
            and (game.get("runtime_health") or {}).get("passed") is True
        ),
    }
    passed = all(checks.values())
    return {
        "session_id": session_id,
        "opponent": game["opponent"],
        "rival_side": game["rival_side"],
        "passed": passed,
        "checks": checks,
        "telemetry": {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "record_counts": dict(sorted(record_counts.items())),
            "invalid_json_records": invalid_json_records,
            "wrong_session_records": wrong_session_records,
        },
        "cadence": {
            "strategic_decisions": strategic_count,
            "mechanics_decisions": mechanics_count,
            "mechanics_to_expected_two_per_strategic_ratio": ratio,
            "strategic_interval_ticks_histogram": strategic_hist,
            "strategic_modal_interval_ticks": strategic_mode,
            "strategic_within_one_tick_of_mode_rate": strategic_near_mode,
            "mechanics_interval_ticks_histogram": mechanics_hist,
            "mechanics_modal_interval_ticks": mechanics_mode,
            "mechanics_within_one_tick_of_mode_rate": mechanics_near_mode,
            "controller_sources": dict(sorted(controller_sources.items())),
            "partial_mechanics_windows": partial_windows,
            "invalid_partial_mechanics_windows": invalid_partial_windows,
        },
        "outcome_context_not_used_by_gate": {
            "outcome": game["outcome"],
            "rival_goals": game["rival_goals"],
            "opponent_goals": game["opponent_goals"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--software-gate", type=Path, default=DEFAULT_SOFTWARE_GATE)
    parser.add_argument("--m07-matrix", type=Path, default=DEFAULT_M07_MATRIX)
    parser.add_argument("--raw-root", type=Path, default=REPOSITORY_ROOT / "evidence/raw")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    matrix = _read(args.matrix)
    software = _read(args.software_gate)
    m07 = _read(args.m07_matrix)
    games = matrix["modes"]["M8P"]["games"]
    sessions = [_session_gate(game, raw_root=args.raw_root) for game in games]
    technical_checks = {
        "offline_software_gate_passed": software.get("passed") is True,
        "four_balanced_full_games_present": bool(
            len(sessions) == 4
            and {(item["opponent"], item["rival_side"]) for item in sessions}
            == {
                ("nexto", "blue"),
                ("nexto", "orange"),
                ("wisp", "blue"),
                ("wisp", "orange"),
            }
        ),
        "all_live_cadence_sessions_passed": all(
            session["passed"] for session in sessions
        ),
    }
    passed = all(technical_checks.values())
    current_outcome = matrix["modes"]["M8P"]["aggregates"]["overall"]
    historical = {
        mode: m07["modes"][mode]["aggregates"]["overall"]
        for mode in ("P0", "Z8", "Z4")
    }
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "purpose": "milestone08_zero_step_dual_rate_rlbot_cadence_gate",
        "decision_rule": {
            "technical_cadence_is_the_gate": True,
            "wins_losses_or_score_used_by_gate": False,
            "strategic_contract": "8-tick clock; [P,P,P,P,P,S,S,S] proven offline",
            "mechanics_contract": "4-tick clock; [previous,M,M,M] proven offline",
            "forced_pass_contract": (
                "every mechanics record is PASS and the live controller source remains "
                "the strategic pass-through path"
            ),
            "lifecycle_rule": (
                "a one-decision mechanics window is allowed only at startup, terminal, "
                "or Countdown phase boundaries; any partial in-play window fails"
            ),
            "collapse_indicators": [
                "strategic modal interval changes from 8 physics ticks",
                "mechanics modal interval changes from 4 physics ticks",
                "non-contiguous, duplicated, stalled, or cross-session decisions",
                "mechanics/strategic decision ratio below 0.999 after lifecycle boundaries",
                "any non-PASS mechanics output in forced-PASS mode",
                "any controller source other than initial strategic or strategic_pass_through",
                "runtime identity, action-mask, action-index, or frozen-hash mismatch",
            ],
        },
        "technical_checks": technical_checks,
        "sessions": sessions,
        "offline_contract_evidence": {
            "path": args.software_gate.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _sha256(args.software_gate),
            "controller_trace_exact": software["fallback_invariant"]["checks"][
                "controller_trace_exact"
            ],
            "controller_trace_sha256": software["fallback_invariant"][
                "controller_trace_sha256"
            ],
            "strategic_temporal_schedule_exact": software["temporal_scheduler"][
                "exact"
            ],
        },
        "behavioral_context_not_used_by_gate": {
            "interpretation": (
                "Scores are a bounded smoke comparison only. A win cannot rescue broken "
                "cadence, and a loss cannot fail correct cadence."
            ),
            "m08_forced_pass": current_outcome,
            "historical_m07": historical,
            "z4_like_score_pattern_observed": bool(
                current_outcome["wins"] <= historical["Z4"]["wins"]
                and current_outcome["goal_differential"]
                <= historical["Z4"]["goal_differential"]
            ),
        },
        "known_external_note": {
            "session_id": "rival-v8-natural-wisp-orange-20260823T045312Z-8dab6a01",
            "component": "installed Wisp opponent",
            "classification": "shared_rough_eta_nonfinite_one_packet_recovered",
            "evidence_scope": "runner-console observation; not present in Rival telemetry",
            "effect": (
                "The match completed and Rival's telemetry stayed contiguous. The game score "
                "remains behavioral context only; the opponent event is not used to pass the "
                "Rival cadence gate."
            ),
        },
        "production_promoted": False,
        "ppo_authorized_by_zero_step_gate": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
