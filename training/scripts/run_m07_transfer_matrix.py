"""Run the bounded Milestone 07 RLBot transfer diagnostic matrix."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import rlbot.managers
import psutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.references import sha256_file  # noqa: E402
from tools.evidence.runner import run_natural_match  # noqa: E402


ZERO_STEP_REPORT = REPOSITORY_ROOT / "training/results/milestone07/zero_step_export.json"
TRAINED_REPORT = REPOSITORY_ROOT / "training/results/milestone06/candidate_export_020m.json"
MODES = ("P0", "Z8", "Z4", "T8", "T4", "M8P", "M8C")
KNOWN_COMPLETED_ANOMALIES = {
    "rival-v7-natural-wisp-blue-20260823T025156Z-022710be": {
        "classification": "shared_eta_nan_recovered",
        "summary": (
            "During overtime, both Rival and the installed Wisp opponent raised the "
            "same one-packet ValueError while converting a non-finite rough_eta to an "
            "integer. Both bots recovered and the match completed. Retain the score "
            "for behavioral context, but do not classify runtime health as clean."
        ),
        "affected_components": ["rival", "installed_wisp_v2_75b"],
    }
}
KNOWN_EXCLUDED_SESSIONS = [
    {
        "session_id": "rival-v7-natural-nexto-blue-20260823T023355Z-c28f4765",
        "reason": "RLBot client disconnect/invalid offset; incomplete and excluded",
    },
    {
        "session_id": "rival-v7-natural-nexto-blue-20260823T024557Z-0497d782",
        "reason": "diagnostic runtime label was incorrect; aborted and excluded",
    },
    {
        "session_id": "rival-v7-natural-wisp-orange-20260823T025410Z-6fb2b633",
        "reason": (
            "aborted after the prior overtime/map state leaked into the next launch; "
            "excluded before switching to a fresh Rocket League process per game"
        ),
    },
]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path_record: str, expected_hash: str) -> Path:
    path = (REPOSITORY_ROOT / path_record).resolve()
    actual = sha256_file(path)
    if actual != expected_hash:
        raise RuntimeError(f"Artifact hash mismatch for {path}: expected {expected_hash}, got {actual}")
    return path


def _mode_definitions(
    mechanics_model: Path | None = None,
) -> dict[str, dict[str, Any]]:
    zero = _read(ZERO_STEP_REPORT)
    trained = _read(TRAINED_REPORT)
    if zero["status"] != "passed" or trained["status"] != "passed":
        raise RuntimeError("M07 matrix requires passing zero-step and M06 candidate exports")
    zero_actor = _artifact(
        zero["torchscript_export"]["path"], zero["torchscript_export"]["sha256"]
    )
    trained_actor = _artifact(
        trained["torchscript_export"]["path"],
        trained["torchscript_export"]["sha256"],
    )
    action_table = _artifact(
        trained["action_table"]["path"], trained["action_table"]["file_sha256"]
    )
    base = {
        "RIVAL_TRANSFER_DIAGNOSTIC_MODE": "1",
        "RIVAL_NATURAL_ADJUSTMENT_MODE": "off",
    }

    def candidate(actor: Path, tick_skip: int, runtime_label: str) -> dict[str, str]:
        return {
            **base,
            "RIVAL_CANDIDATE_MODEL_PATH": str(actor),
            "RIVAL_CANDIDATE_ACTION_TABLE_PATH": str(action_table),
            "RIVAL_CANDIDATE_LEGACY_ONLY": "1",
            "RIVAL_CANDIDATE_RUNTIME_LABEL": runtime_label,
            "RIVAL_TICK_SKIP": str(tick_skip),
        }

    definitions = {
        "P0": {
            "description": "frozen production Wisp, tick 8",
            "policy": "frozen_wisp_production",
            "model_sha256": {
                "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
                "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
            },
            "tick_skip": 8,
            "action_delay": 7,
            "expected_action_count": 90,
            "legacy_only": True,
            "environment": {
                **base,
                "RIVAL_TICK_SKIP": "8",
                "RIVAL_DIAGNOSTIC_CAPTURE_OBSERVATIONS": "1",
                "RIVAL_DIAGNOSTIC_OBSERVATION_STRIDE": "16",
            },
        },
        "Z8": {
            "description": "zero-step reconstructed student, legacy-only, tick 8",
            "policy": "zero_step_reconstructed_student",
            "model_sha256": zero["torchscript_export"]["sha256"],
            "tick_skip": 8,
            "action_delay": 7,
            "expected_action_count": 158,
            "legacy_only": True,
            "environment": candidate(zero_actor, 8, "m07_zero_step_legacy_only"),
        },
        "Z4": {
            "description": "zero-step reconstructed student, legacy-only, tick 4",
            "policy": "zero_step_reconstructed_student",
            "model_sha256": zero["torchscript_export"]["sha256"],
            "tick_skip": 4,
            "action_delay": 3,
            "expected_action_count": 158,
            "legacy_only": True,
            "environment": candidate(zero_actor, 4, "m07_zero_step_legacy_only"),
        },
        "T8": {
            "description": "rejected 20M actor, legacy-only, tick 8",
            "policy": "milestone06_20m_rejected_actor",
            "model_sha256": trained["torchscript_export"]["sha256"],
            "tick_skip": 8,
            "action_delay": 7,
            "expected_action_count": 158,
            "legacy_only": True,
            "environment": candidate(trained_actor, 8, "m07_20m_legacy_only"),
        },
        "T4": {
            "description": "rejected 20M actor, legacy-only, tick 4",
            "policy": "milestone06_20m_rejected_actor",
            "model_sha256": trained["torchscript_export"]["sha256"],
            "tick_skip": 4,
            "action_delay": 3,
            "expected_action_count": 158,
            "legacy_only": True,
            "environment": candidate(trained_actor, 4, "m07_20m_legacy_only"),
        },
        "M8P": {
            "description": (
                "M08 opt-in dual-rate runtime with the frozen strategic Wisp branch "
                "and mechanics forced PASS"
            ),
            "policy": "m08_dual_rate_forced_pass",
            "model_sha256": {
                "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
                "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
            },
            "tick_skip": 8,
            "action_delay": 7,
            "mechanics_tick_skip": 4,
            "expected_action_count": 69,
            "legacy_only": False,
            "environment": {
                **base,
                "RIVAL_M08_DUAL_RATE_ENABLED": "1",
                "RIVAL_M08_MECHANICS_FORCE_PASS": "1",
                "RIVAL_M08_ACTION_TABLE_PATH": str(action_table),
                "RIVAL_M08_RUNTIME_LABEL": "m08_dual_rate_forced_pass",
                "RIVAL_TICK_SKIP": "8",
            },
        },
    }
    if mechanics_model is not None:
        selected = mechanics_model.expanduser().resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"M08 mechanics model does not exist: {selected}")
        definitions["M8C"] = {
            "description": (
                "M08 opt-in dual-rate runtime with frozen strategic Wisp and a "
                "selected mechanics candidate"
            ),
            "policy": "m08_dual_rate_candidate",
            "model_sha256": sha256_file(selected),
            "tick_skip": 8,
            "action_delay": 7,
            "mechanics_tick_skip": 4,
            "expected_action_count": 69,
            "legacy_only": False,
            "environment": {
                **base,
                "RIVAL_M08_DUAL_RATE_ENABLED": "1",
                "RIVAL_M08_MECHANICS_FORCE_PASS": "0",
                "RIVAL_M08_MECHANICS_MODEL_PATH": str(selected),
                "RIVAL_M08_ACTION_TABLE_PATH": str(action_table),
                "RIVAL_M08_RUNTIME_LABEL": "m08_dual_rate_candidate",
                "RIVAL_M08_MECHANICS_DETERMINISTIC": "1",
                "RIVAL_TICK_SKIP": "8",
            },
        }
    return definitions


def _schedule(games_per_mode: int) -> list[dict[str, Any]]:
    result = []
    for repetition in range(games_per_mode // 4):
        for opponent in ("nexto", "wisp"):
            for rival_team in (0, 1):
                result.append(
                    {
                        "opponent": opponent,
                        "rival_team": rival_team,
                        "repetition": repetition + 1,
                    }
                )
    return result


def _compact_game(
    manifest: dict[str, Any], schedule: dict[str, Any], game_number: int
) -> dict[str, Any]:
    rival_team = int(schedule["rival_team"])
    score = manifest["final_score"]
    rival_goals = score["blue"] if rival_team == 0 else score["orange"]
    opponent_goals = score["orange"] if rival_team == 0 else score["blue"]
    if rival_goals is None or opponent_goals is None:
        outcome = "incomplete"
    elif rival_goals > opponent_goals:
        outcome = "win"
    elif rival_goals < opponent_goals:
        outcome = "loss"
    else:
        outcome = "tie"
    telemetry_path = (
        REPOSITORY_ROOT / "evidence/raw" / manifest["session_id"] / "decisions.jsonl"
    )
    session_end = None
    last_decision = None
    invalid_json_records = 0
    if telemetry_path.is_file():
        with telemetry_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid_json_records += 1
                    continue
                if record.get("record_type") == "rival_policy_decision":
                    last_decision = record
                elif record.get("record_type") == "rival_session_end":
                    session_end = record
    end_score_matches = bool(
        session_end is not None
        and session_end.get("final_score") == manifest["final_score"]
    )
    last_remaining = (
        None
        if last_decision is None
        else (last_decision.get("packet") or {}).get("match", {}).get(
            "game_time_remaining"
        )
    )
    decision_stream_reached_match_end = bool(
        isinstance(last_remaining, (int, float)) and float(last_remaining) <= 2.0
    )
    runtime_health = {
        "telemetry_session_end_present": session_end is not None,
        "telemetry_end_score_matches_manifest": end_score_matches,
        "decision_stream_reached_match_end": decision_stream_reached_match_end,
        "last_decision_game_time_remaining": last_remaining,
        "invalid_json_records": invalid_json_records,
        "runtime_warnings": manifest.get("runtime_warnings", []),
    }
    runtime_health["passed"] = all(
        (
            manifest["status"] == "complete",
            runtime_health["telemetry_session_end_present"],
            runtime_health["telemetry_end_score_matches_manifest"],
            runtime_health["decision_stream_reached_match_end"],
            runtime_health["invalid_json_records"] == 0,
            not runtime_health["runtime_warnings"],
        )
    )
    status = "complete" if runtime_health["passed"] else "invalid"
    return {
        "game": game_number,
        "session_id": manifest["session_id"],
        **schedule,
        "rival_side": "blue" if rival_team == 0 else "orange",
        "status": status,
        "runner_status": manifest["status"],
        "outcome": outcome,
        "rival_goals": rival_goals,
        "opponent_goals": opponent_goals,
        "decision_records": manifest.get("raw_telemetry", {})
        .get("record_counts", {})
        .get("rival_policy_decision", 0),
        "mechanics_decision_records": manifest.get("raw_telemetry", {})
        .get("record_counts", {})
        .get("rival_mechanics_decision", 0),
        "wall_duration_seconds": manifest["wall_duration_seconds"],
        "termination_reason": manifest["termination_reason"],
        "error": manifest.get("error"),
        "runtime_health": runtime_health,
    }


def _aggregate(games: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = games
    for game in games:
        groups[str(game["opponent"])].append(game)
    result = {}
    for key, records in groups.items():
        goals_for = sum(int(record["rival_goals"] or 0) for record in records)
        goals_against = sum(int(record["opponent_goals"] or 0) for record in records)
        result[key] = {
            "games": len(records),
            "complete": sum(record["status"] == "complete" for record in records),
            "completed_match_results": sum(
                record["runner_status"] == "complete" for record in records
            ),
            "runtime_clean": sum(
                record["runtime_health"]["passed"] for record in records
            ),
            "wins": sum(record["outcome"] == "win" for record in records),
            "losses": sum(record["outcome"] == "loss" for record in records),
            "ties": sum(record["outcome"] == "tie" for record in records),
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_differential": goals_for - goals_against,
        }
    return result


def _apply_known_annotations(report: dict[str, Any]) -> None:
    report["excluded_sessions"] = KNOWN_EXCLUDED_SESSIONS
    for entry in report.get("modes", {}).values():
        for game in entry.get("games", []):
            annotation = KNOWN_COMPLETED_ANOMALIES.get(game["session_id"])
            if annotation is None:
                continue
            game["known_runtime_anomaly"] = annotation
            game["runtime_health"]["passed"] = False
            game["status"] = "complete_with_runtime_anomaly"
            game["evidence_use"] = "behavioral_outcome_only"
        entry["aggregates"] = _aggregate(entry.get("games", []))


def _write(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _stop_rocket_league_for_clean_launch() -> list[dict[str, Any]]:
    """End only verified RocketLeague.exe processes before a fresh RLBot launch."""
    selected = []
    records = []
    for process in psutil.process_iter(("pid", "name", "exe")):
        try:
            name = str(process.info.get("name") or "").lower()
            executable = process.info.get("exe")
            if name != "rocketleague.exe" or not executable:
                continue
            path = Path(executable).resolve()
            if path.name.lower() != "rocketleague.exe":
                continue
            selected.append(process)
            records.append({"pid": process.pid, "path": str(path)})
            process.terminate()
        except (OSError, psutil.Error):
            continue
    if selected:
        _, alive = psutil.wait_procs(selected, timeout=20.0)
        if alive:
            raise RuntimeError(
                "Rocket League did not exit for the clean per-game relaunch: "
                + ", ".join(str(process.pid) for process in alive)
            )
        time.sleep(2.0)
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=["P0", "Z8", "Z4", "T8", "T4"],
    )
    parser.add_argument("--games-per-mode", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "training/results/milestone07/transfer_matrix.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--evaluation-label", default="milestone07_bounded_rlbot_transfer_matrix"
    )
    parser.add_argument("--session-version", default="v7")
    parser.add_argument("--session-source", default="milestone07_transfer_matrix")
    parser.add_argument("--experiment-milestone", default="m07-transfer-diagnosis")
    parser.add_argument("--lane-id", default="m07-transfer-diagnostic")
    parser.add_argument("--runtime-label")
    parser.add_argument("--mechanics-model", type=Path)
    args = parser.parse_args()
    definitions = _mode_definitions(args.mechanics_model)
    if "M8C" in args.modes and "M8C" not in definitions:
        parser.error("M8C requires --mechanics-model")
    if args.runtime_label:
        for mode in args.modes:
            runtime_key = (
                "RIVAL_M08_RUNTIME_LABEL"
                if mode in {"M8P", "M8C"}
                else "RIVAL_CANDIDATE_RUNTIME_LABEL"
            )
            definitions[mode]["environment"][runtime_key] = args.runtime_label
    report = (
        _read(args.output)
        if args.resume and args.output.is_file()
        else {
            "schema_version": 1,
            "status": "in_progress",
            "evaluation": args.evaluation_label,
            "rlbot_version_family": "RLBot v5",
            "game_speed": 5.0,
            "full_five_minute_soccar": True,
            "balanced_sides": True,
            "rocket_league_process_regime": "fresh_process_per_game",
            "maximum_games_per_ambiguous_mode": 8,
            "modes": {},
            "production_promoted": False,
        }
    )
    if args.evaluation_label == "milestone07_bounded_rlbot_transfer_matrix":
        report["rocket_league_process_regime"] = "mixed"
        report["process_hygiene"] = {
            "initial_regime": "one Rocket League process reused across sequential matches",
            "fresh_process_per_game_enabled_after_session": (
                "rival-v7-natural-wisp-blue-20260823T025156Z-022710be"
            ),
            "current_regime": "fresh verified RocketLeague.exe process per match",
            "reason": (
                "A completed overtime match exposed a shared one-packet ETA anomaly and the "
                "following launch inherited transient overtime/map state. Subsequent matches "
                "use clean application state without changing the bot or match configuration."
            ),
        }
        _apply_known_annotations(report)
    else:
        report["rocket_league_process_regime"] = "fresh_process_per_game"
        report["process_hygiene"] = {
            "initial_regime": "fresh verified RocketLeague.exe process per match",
            "current_regime": "fresh verified RocketLeague.exe process per match",
            "reason": "prospective M08 zero-step/pass-only transfer control",
        }
    manager: rlbot.managers.MatchManager | None = None
    started = time.perf_counter()
    failed = False
    try:
        for mode in args.modes:
            definition = definitions[mode]
            entry = report["modes"].setdefault(
                mode,
                {
                    **{key: value for key, value in definition.items() if key != "environment"},
                    "games": [],
                    "aggregates": {},
                },
            )
            desired_schedule = _schedule(args.games_per_mode)
            for index in range(len(entry["games"]), len(desired_schedule)):
                item = desired_schedule[index]
                stopped_processes = _stop_rocket_league_for_clean_launch()
                manager = rlbot.managers.MatchManager()
                try:
                    manifest = run_natural_match(
                        item["opponent"],
                        rival_team=item["rival_team"],
                        launcher="steam",
                        timeout=900.0,
                        game_speed=5.0,
                        challenge_mode="off",
                        lane_id=args.lane_id,
                        execution_regime="sequential_fresh_process_per_game",
                        session_version=args.session_version,
                        session_source=args.session_source,
                        experiment_milestone=args.experiment_milestone,
                        experiment_metadata={
                            "mode": mode,
                            "description": definition["description"],
                            "tick_skip": definition["tick_skip"],
                            "action_delay": definition["action_delay"],
                            "mechanics_tick_skip": definition.get("mechanics_tick_skip"),
                            "legacy_only": definition["legacy_only"],
                            "fresh_rocket_league_process": True,
                            "stopped_prior_processes": stopped_processes,
                            "production_promoted": False,
                        },
                        rival_environment_overrides=definition["environment"],
                        manager=manager,
                    )
                finally:
                    manager.shut_down()
                    manager = None
                game = _compact_game(manifest, item, index + 1)
                game["rocket_league_process_regime"] = "fresh_process_for_match"
                game["stopped_prior_rocket_league_processes"] = stopped_processes
                entry["games"].append(game)
                entry["aggregates"] = _aggregate(entry["games"])
                report["status"] = "in_progress"
                _write(args.output, report)
                if game["status"] != "complete":
                    failed = True
                    break
            if failed:
                break
            entry["aggregates"] = _aggregate(entry["games"])
    finally:
        if manager is not None:
            manager.shut_down()

    all_requested_complete = all(
        len(report["modes"].get(mode, {}).get("games", [])) >= args.games_per_mode
        and report["modes"][mode]["aggregates"]["overall"]["completed_match_results"]
        >= args.games_per_mode
        for mode in args.modes
    )
    has_runtime_anomaly = any(
        not game["runtime_health"]["passed"]
        for entry in report["modes"].values()
        for game in entry.get("games", [])
    )
    report["status"] = (
        "completed_with_runtime_anomaly"
        if all_requested_complete and not failed and has_runtime_anomaly
        else "passed"
        if all_requested_complete and not failed
        else "incomplete"
    )
    report["last_run_wall_seconds"] = time.perf_counter() - started
    report["completed_modes"] = sorted(
        mode
        for mode, entry in report["modes"].items()
        if entry.get("aggregates", {}).get("overall", {}).get(
            "completed_match_results", 0
        )
        >= len(entry.get("games", []))
        and len(entry.get("games", [])) >= 4
    )
    _write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if all_requested_complete and not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
