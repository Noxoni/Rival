"""Validate live M08 candidate cadence and the bounded severe-regression gate."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE = (
    REPOSITORY_ROOT
    / "training/results/milestone08/zero_step_dual_rate_rlbot.json"
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
    return {str(key): counts[key] for key in sorted(counts)}, mode, near_mode


def _mechanics_record_exact(record: dict[str, Any]) -> bool:
    choice = record.get("choice")
    if not isinstance(choice, int) or not 0 <= choice < 69:
        return False
    probability = record.get("mean_pass_probability")
    top = record.get("top_mechanics_choices")
    if not (
        isinstance(probability, (int, float))
        and math.isfinite(float(probability))
        and 0.0 <= float(probability) <= 1.0
        and isinstance(top, list)
        and len(top) == 5
    ):
        return False
    top_exact = all(
        isinstance(item, dict)
        and isinstance(item.get("choice"), int)
        and 0 <= item["choice"] < 69
        and isinstance(item.get("probability"), (int, float))
        and math.isfinite(float(item["probability"]))
        and 0.0 <= float(item["probability"]) <= 1.0
        for item in top
    )
    expected_global = None if choice == 0 else 89 + choice
    return bool(
        record.get("pass") is (choice == 0)
        and record.get("override_selected") is (choice != 0)
        and record.get("global_action_index") == expected_global
        and record.get("force_pass") is False
        and record.get("scheduler") == "[previous,new,new,new]"
        and top_exact
    )


def _session_gate(
    game: dict[str, Any],
    *,
    raw_root: Path,
    expected_model: Path,
    expected_runtime_label: str,
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
    mechanics_records_exact = True
    strategic_times: list[float] = []
    mechanics_times: list[float] = []
    strategic_phases: dict[int, str | None] = {}
    mechanics_per_strategic: Counter[int] = Counter()
    mechanics_choices: Counter[int] = Counter()
    controller_sources: Counter[str] = Counter()
    pass_probabilities: list[float] = []
    allowed_sources = {
        "strategic",
        "strategic_pass_through",
        "mechanics_delay_previous",
        "mechanics_override",
    }

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
            elif record_type == "rival_session_end":
                end_record = record
            elif record_type == "rival_policy_decision":
                decision = record.get("decision") or {}
                runtime = record.get("runtime") or {}
                packet = record.get("packet") or {}
                match = packet.get("match") or {}
                strategic_sequence_exact &= decision.get("tick") == strategic_count
                action = decision.get("action_index")
                legal_mask = decision.get("legal_mask")
                strategic_actions_exact &= bool(
                    isinstance(action, int)
                    and 0 <= action < 90
                    and action == decision.get("baseline_action_index")
                    == decision.get("final_action_index")
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
                    and source in allowed_sources
                )
                game_time = decision.get("game_time")
                if isinstance(game_time, (int, float)):
                    strategic_times.append(float(game_time))
                strategic_phases[strategic_count + 1] = (
                    (match.get("phase") or {}).get("name")
                )
                strategic_count += 1
            elif record_type == "rival_mechanics_decision":
                mechanics_sequence_exact &= (
                    record.get("mechanics_decision") == mechanics_count
                )
                policy_tick = record.get("strategic_policy_tick")
                if isinstance(policy_tick, int):
                    mechanics_per_strategic[policy_tick] += 1
                else:
                    mechanics_records_exact = False
                mechanics_records_exact &= _mechanics_record_exact(record)
                choice = record.get("choice")
                if isinstance(choice, int):
                    mechanics_choices[choice] += 1
                probability = record.get("mean_pass_probability")
                if isinstance(probability, (int, float)):
                    pass_probabilities.append(float(probability))
                game_time = record.get("game_time")
                if isinstance(game_time, (int, float)):
                    mechanics_times.append(float(game_time))
                mechanics_count += 1

    strategic_hist, strategic_mode, strategic_near = _tick_histogram(
        strategic_times
    )
    mechanics_hist, mechanics_mode, mechanics_near = _tick_histogram(
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
    model_path = policy_runtime.get("mechanics_model_path")
    model_path_exact = False
    if isinstance(model_path, str):
        model_path_exact = Path(model_path).resolve() == expected_model.resolve()
    start_contract_exact = bool(
        policy_runtime.get("mode") == expected_runtime_label
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
        and policy_runtime.get("mechanics_force_pass") is False
        and model_path_exact
        and metadata.get("rival_model_sha256") == EXPECTED_POLICY_HASHES
    )
    record_count_exact = bool(
        strategic_count == game.get("decision_records")
        and mechanics_count == game.get("mechanics_decision_records")
        and (end_record or {}).get("decision_record_count") == strategic_count
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
        "runtime_identity_model_path_and_hashes_exact": start_contract_exact,
        "strategic_decision_sequence_contiguous": strategic_sequence_exact,
        "mechanics_decision_sequence_contiguous": mechanics_sequence_exact,
        "strategic_action_mask_and_frozen_choice_exact": strategic_actions_exact,
        "strategic_runtime_flags_exact": strategic_runtime_exact,
        "mechanics_choice_mapping_and_probabilities_exact": mechanics_records_exact,
        "strategic_clock_mode_is_8_ticks": bool(
            strategic_mode == 8 and strategic_near >= 0.95
        ),
        "mechanics_clock_mode_is_4_ticks": bool(
            mechanics_mode == 4 and mechanics_near >= 0.95
        ),
        "two_to_one_clock_ratio_sustained": ratio >= 0.999,
        "partial_windows_are_lifecycle_only": not invalid_partial_windows,
        "controller_sources_are_dual_rate_contract_values": bool(
            controller_sources
            and set(controller_sources).issubset(allowed_sources)
            and controller_sources.get("strategic", 0) == 1
        ),
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
            "strategic_within_one_tick_of_mode_rate": strategic_near,
            "mechanics_interval_ticks_histogram": mechanics_hist,
            "mechanics_modal_interval_ticks": mechanics_mode,
            "mechanics_within_one_tick_of_mode_rate": mechanics_near,
            "controller_sources": dict(sorted(controller_sources.items())),
            "partial_mechanics_windows": partial_windows,
            "invalid_partial_mechanics_windows": invalid_partial_windows,
        },
        "mechanics": {
            "choice_counts": {
                str(key): mechanics_choices[key] for key in sorted(mechanics_choices)
            },
            "pass_count": mechanics_choices[0],
            "override_count": mechanics_count - mechanics_choices[0],
            "deterministic_override_rate": (
                0.0
                if mechanics_count == 0
                else (mechanics_count - mechanics_choices[0]) / mechanics_count
            ),
            "mean_pass_probability": (
                None
                if not pass_probabilities
                else sum(pass_probabilities) / len(pass_probabilities)
            ),
        },
        "outcome_context": {
            "outcome": game["outcome"],
            "rival_goals": game["rival_goals"],
            "opponent_goals": game["opponent_goals"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--candidate-export", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument("--maximum-goal-differential-drop", type=int, default=10)
    parser.add_argument("--raw-root", type=Path, default=REPOSITORY_ROOT / "evidence/raw")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    matrix = _read(args.matrix)
    candidate = _read(args.candidate_export)
    baseline = _read(args.baseline)
    games = matrix["modes"]["M8C"]["games"]
    model_path = REPOSITORY_ROOT / candidate["torchscript_export"]["path"]
    sessions = [
        _session_gate(
            game,
            raw_root=args.raw_root.resolve(),
            expected_model=model_path,
            expected_runtime_label=args.runtime_label,
        )
        for game in games
    ]

    candidate_outcome = matrix["modes"]["M8C"]["aggregates"]["overall"]
    baseline_outcome = baseline["modes"]["M8P"]["aggregates"]["overall"]
    differential_change = int(candidate_outcome["goal_differential"]) - int(
        baseline_outcome["goal_differential"]
    )
    technical_checks = {
        "candidate_export_passed": candidate.get("status") == "passed",
        "candidate_model_file_hash_exact": bool(
            model_path.is_file()
            and _sha256(model_path) == candidate["torchscript_export"]["sha256"]
            and matrix["modes"]["M8C"]["model_sha256"]
            == candidate["torchscript_export"]["sha256"]
        ),
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
        "all_live_candidate_cadence_sessions_passed": all(
            session["passed"] for session in sessions
        ),
    }
    transfer_safety = {
        "baseline_goal_differential": int(baseline_outcome["goal_differential"]),
        "candidate_goal_differential": int(candidate_outcome["goal_differential"]),
        "goal_differential_change": differential_change,
        "maximum_allowed_drop": int(args.maximum_goal_differential_drop),
        "no_severe_goal_differential_regression": differential_change
        >= -int(args.maximum_goal_differential_drop),
    }
    passed = all(technical_checks.values()) and transfer_safety[
        "no_severe_goal_differential_regression"
    ]
    total_mechanics = sum(
        session["cadence"]["mechanics_decisions"] for session in sessions
    )
    total_overrides = sum(
        session["mechanics"]["override_count"] for session in sessions
    )
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "purpose": "milestone08_candidate_rlbot_transfer_gate",
        "decision_rule": {
            "cadence_collapse_is_technical_not_score_based": True,
            "technical_cadence_checks": [
                "strategic modal interval is 8 physics ticks",
                "mechanics modal interval is 4 physics ticks",
                "decision sequences and two-to-one clock ratio remain contiguous",
                "runtime identity, mechanics model, frozen hashes and mappings are exact",
            ],
            "score_use_is_limited_to_explicit_severe_transfer_regression_gate": True,
            "wins_do_not_prove_mechanics_learning": True,
        },
        "technical_checks": technical_checks,
        "transfer_safety": transfer_safety,
        "sessions": sessions,
        "aggregate_mechanics": {
            "decisions": total_mechanics,
            "deterministic_overrides": total_overrides,
            "deterministic_override_rate": (
                0.0 if total_mechanics == 0 else total_overrides / total_mechanics
            ),
        },
        "candidate_outcome_context": candidate_outcome,
        "forced_pass_baseline_outcome_context": baseline_outcome,
        "candidate_export": {
            "path": args.candidate_export.resolve()
            .relative_to(REPOSITORY_ROOT.resolve())
            .as_posix(),
            "sha256": _sha256(args.candidate_export),
            "mechanics_model_path": candidate["torchscript_export"]["path"],
            "mechanics_model_sha256": candidate["torchscript_export"]["sha256"],
            "source_checkpoint": candidate["source_checkpoint"],
        },
        "production_promoted": False,
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
