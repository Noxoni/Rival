"""Run balanced full-game RLBot v5 evaluation for an exported M06 candidate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any

import rlbot.managers


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.references import sha256_file  # noqa: E402
from tools.evidence.runner import run_natural_match  # noqa: E402


def _compact_game(
    manifest: dict[str, Any],
    *,
    opponent: str,
    rival_team: int,
    game_number: int,
) -> dict[str, Any]:
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
    return {
        "game": game_number,
        "session_id": manifest["session_id"],
        "opponent": opponent,
        "rival_team": rival_team,
        "rival_side": "blue" if rival_team == 0 else "orange",
        "status": manifest["status"],
        "outcome": outcome,
        "rival_goals": rival_goals,
        "opponent_goals": opponent_goals,
        "wall_duration_seconds": manifest["wall_duration_seconds"],
        "termination_reason": manifest["termination_reason"],
        "effective_game_speed": manifest.get("execution", {}).get(
            "effective_game_speed"
        ),
        "decision_records": manifest.get("raw_telemetry", {})
        .get("record_counts", {})
        .get("rival_policy_decision", 0),
        "error": manifest.get("error"),
    }


def _aggregate(games: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups["overall"] = games
    for game in games:
        groups[str(game["opponent"])].append(game)
    result = {}
    for name, records in groups.items():
        result[name] = {
            "games": len(records),
            "complete": sum(record["status"] == "complete" for record in records),
            "wins": sum(record["outcome"] == "win" for record in records),
            "losses": sum(record["outcome"] == "loss" for record in records),
            "ties": sum(record["outcome"] == "tie" for record in records),
            "goals_for": sum(int(record["rival_goals"] or 0) for record in records),
            "goals_against": sum(
                int(record["opponent_goals"] or 0) for record in records
            ),
        }
        result[name]["goal_differential"] = (
            result[name]["goals_for"] - result[name]["goals_against"]
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-report", type=Path, required=True)
    parser.add_argument("--games", type=int, choices=(8, 16), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export = json.loads(args.export_report.read_text(encoding="utf-8"))
    if export["status"] != "passed":
        raise RuntimeError("Refusing RLBot evaluation of a failed candidate export")
    model_path = (REPOSITORY_ROOT / export["torchscript_export"]["path"]).resolve()
    action_table_path = (REPOSITORY_ROOT / export["action_table"]["path"]).resolve()
    if sha256_file(model_path) != export["torchscript_export"]["sha256"]:
        raise RuntimeError("Candidate model changed after export")
    if sha256_file(action_table_path) != export["action_table"]["file_sha256"]:
        raise RuntimeError("Candidate action table changed after export")

    schedule = []
    repeats_per_side = args.games // 4
    for opponent in ("nexto", "wisp"):
        for repetition in range(repeats_per_side):
            for rival_team in (0, 1):
                schedule.append(
                    {
                        "opponent": opponent,
                        "rival_team": rival_team,
                        "repetition": repetition + 1,
                    }
                )
    manager = rlbot.managers.MatchManager()
    games = []
    started = time.perf_counter()
    try:
        for index, item in enumerate(schedule):
            manifest = run_natural_match(
                item["opponent"],
                rival_team=item["rival_team"],
                launcher="steam",
                timeout=900.0,
                game_speed=5.0,
                challenge_mode="off",
                lane_id="m06-stage-eval",
                execution_regime="sequential",
                session_version="v6",
                session_source="milestone06_rlbot_stage_evaluation",
                experiment_milestone="m06-serious-training",
                experiment_metadata={
                    "candidate_label": export["label"],
                    "cumulative_agent_steps": export["source_campaign_state"][
                        "cumulative_agent_steps"
                    ],
                    "candidate_model_sha256": export["torchscript_export"][
                        "sha256"
                    ],
                    "production_promoted": False,
                },
                rival_environment_overrides={
                    "RIVAL_CANDIDATE_MODEL_PATH": str(model_path),
                    "RIVAL_CANDIDATE_ACTION_TABLE_PATH": str(action_table_path),
                    "RIVAL_TICK_SKIP": "4",
                    "RIVAL_NATURAL_ADJUSTMENT_MODE": "off",
                },
                manager=manager,
            )
            compact = _compact_game(
                manifest,
                opponent=item["opponent"],
                rival_team=item["rival_team"],
                game_number=index + 1,
            )
            games.append(compact)
            if compact["status"] != "complete":
                break
    finally:
        manager.shut_down()

    aggregates = _aggregate(games)
    complete = len(games) == args.games and aggregates["overall"]["complete"] == args.games
    report = {
        "schema_version": 1,
        "status": "passed" if complete else "incomplete",
        "evaluation": (
            "final_16_game_promotion_context"
            if args.games == 16
            else "major_stage_8_game_context"
        ),
        "rlbot_version_family": "RLBot v5",
        "candidate_export_report": args.export_report.as_posix(),
        "candidate_label": export["label"],
        "cumulative_agent_steps": export["source_campaign_state"][
            "cumulative_agent_steps"
        ],
        "candidate_model_sha256": export["torchscript_export"]["sha256"],
        "candidate_action_table_sha256": export["action_table"][
            "logical_float32_sha256"
        ],
        "game_speed": 5.0,
        "full_five_minute_soccar": True,
        "balanced_sides": True,
        "games": games,
        "aggregates": aggregates,
        "historical_frozen_wisp_context": {
            "games": 8,
            "record": "4-4",
            "goals_for": 35,
            "goals_against": 30,
            "goal_differential": 5,
        },
        "promotion_decision": "not_evaluated_by_stage_runner",
        "production_promoted": False,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
