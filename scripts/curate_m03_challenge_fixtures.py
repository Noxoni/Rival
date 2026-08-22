from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.io import load_session  # noqa: E402
from tools.evidence.session import write_json  # noqa: E402


DEFAULT_EVENT_IDS = (
    "appa-2853d7379f43",
    "appa-1d8d681ee3a9",
    "appa-b91d6aa6af1e",
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _grounded_jump(record: Mapping[str, Any]) -> bool:
    packet = _mapping(record.get("packet"))
    players = packet.get("players")
    self_index = packet.get("self_index")
    if not isinstance(players, list) or not isinstance(self_index, int):
        return False
    if not 0 <= self_index < len(players):
        return False
    air_state = str(_mapping(_mapping(players[self_index]).get("air_state")).get("name"))
    action = _mapping(_mapping(record.get("decision")).get("controller_action"))
    return air_state == "OnGround" and bool(action.get("jump", False))


def _compact_record(record: Mapping[str, Any], decision_index: int) -> dict[str, Any]:
    packet = _mapping(record.get("packet"))
    state = _mapping(record.get("state"))
    return {
        "schema_version": record.get("schema_version"),
        "record_type": record.get("record_type"),
        "session_id": record.get("session_id"),
        "source_decision_index": decision_index,
        "source_raw_line": record.get("_raw_line"),
        "decision": dict(_mapping(record.get("decision"))),
        "tactical_metrics": dict(_mapping(record.get("tactical_metrics"))),
        "packet": {
            "self_index": packet.get("self_index"),
            "opponent_indices": packet.get("opponent_indices"),
            "players": packet.get("players"),
            "ball": packet.get("ball"),
            "match": packet.get("match"),
        },
        "state": {
            "self": state.get("self"),
            "opponent": state.get("opponent"),
            "ball": state.get("ball"),
            "score_diff": state.get("score_diff"),
            "game_time": state.get("game_time"),
            "seconds_remaining": state.get("seconds_remaining"),
        },
        "runtime": dict(_mapping(record.get("runtime"))),
    }


def curate(
    *,
    event_report: Path,
    raw_root: Path,
    output_dir: Path,
    event_ids: tuple[str, ...],
) -> list[Path]:
    report = json.loads(event_report.read_text(encoding="utf-8"))
    events = {
        str(event["event_id"]): event
        for event in report.get("events", [])
        if event.get("event_id") in event_ids
    }
    missing = sorted(set(event_ids) - set(events))
    if missing:
        raise ValueError(f"Event report does not contain requested ids: {missing}")

    sessions: dict[str, Any] = {}
    paths: list[Path] = []
    for event_id in event_ids:
        event = events[event_id]
        session_id = str(event["session_id"])
        session = sessions.get(session_id)
        if session is None:
            session = load_session(raw_root / session_id / "decisions.jsonl")
            sessions[session_id] = session
        if session.raw_sha256 != event["raw_telemetry_sha256"]:
            raise ValueError(f"Raw SHA mismatch for {session_id}")
        first, last = (int(value) for value in event["post_window_record_range"])
        last = min(last, len(session.decisions) - 1)
        jump_indices: list[int] = []
        previous_jump = False
        for index in range(first, last + 1):
            grounded_jump = _grounded_jump(session.decisions[index])
            if grounded_jump and not previous_jump:
                jump_indices.append(index)
            previous_jump = grounded_jump
        anchor = int(event["post_window_record_range"][0])
        selected = set(range(max(0, anchor - 2), min(len(session.decisions), anchor + 3)))
        for index in jump_indices:
            selected.update(
                range(max(0, index - 6), min(len(session.decisions), index + 5))
            )
        selected_indices = sorted(selected)
        fixture = {
            "fixture_schema_version": 2,
            "candidate_only": True,
            "detector_version": "rival-m02-events-v1",
            "evaluation_detector_version": "rival-m03-challenge-v1",
            "source_session_id": session.session_id,
            "source_event_id": event_id,
            "source_raw_sha256": session.raw_sha256,
            "event_class": "apparent_vs_actual_challenge",
            "opponent": session.opponent,
            "game_time_window": [event["start_game_time"], event["end_game_time"]],
            "expected_baseline_observation": event["ranking_explanation"],
            "record_selection": {
                "anchor_decision_index": anchor,
                "grounded_jump_initiation_indices": jump_indices,
                "selected_decision_indices": selected_indices,
                "note": (
                    "Compact anchor and grounded-jump neighborhoods; discontinuous gaps "
                    "are intentional and trigger estimator history reset."
                ),
            },
            "event": event,
            "records": [
                _compact_record(session.decisions[index], index)
                for index in selected_indices
            ],
        }
        path = output_dir / f"apparent_vs_actual_challenge__{event_id}.json"
        write_json(path, fixture)
        paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate the required M03 natural fixtures")
    parser.add_argument(
        "--event-report",
        type=Path,
        default=REPOSITORY_ROOT / "evidence" / "results" / "v2" / "candidate_events.json",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPOSITORY_ROOT / "evidence" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "fixtures" / "evidence",
    )
    parser.add_argument("--event-id", action="append")
    args = parser.parse_args()
    paths = curate(
        event_report=args.event_report,
        raw_root=args.raw_root,
        output_dir=args.output_dir,
        event_ids=tuple(args.event_id or DEFAULT_EVENT_IDS),
    )
    print(json.dumps([str(path) for path in paths], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
