from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import EvidenceSession
from .session import write_json


def curate_top_fixtures(
    events: Iterable[dict[str, Any]],
    sessions: Iterable[EvidenceSession],
    output_dir: Path,
) -> list[Path]:
    session_map = {session.session_id: session for session in sessions}
    selected: dict[str, dict[str, Any]] = {}
    for event in events:
        selected.setdefault(str(event["class"]), event)

    paths: list[Path] = []
    for event_class, event in sorted(selected.items()):
        session = session_map[event["session_id"]]
        anchor = int(event["post_window_record_range"][0])
        first = max(0, anchor - 2)
        last = min(len(session.decisions), anchor + 4)
        records = []
        for record in session.decisions[first:last]:
            compact = {key: value for key, value in record.items() if not key.startswith("_")}
            records.append(compact)
        fixture = {
            "fixture_schema_version": 1,
            "candidate_only": True,
            "detector_version": "rival-m02-events-v1",
            "source_session_id": session.session_id,
            "source_event_id": event["event_id"],
            "source_raw_sha256": session.raw_sha256,
            "event_class": event_class,
            "opponent": session.opponent,
            "game_time_window": [event["start_game_time"], event["end_game_time"]],
            "expected_baseline_observation": event["ranking_explanation"],
            "event": event,
            "records": records,
        }
        path = output_dir / f"{event_class}__{event['event_id']}.json"
        write_json(path, fixture)
        event["fixture_path"] = str(path)
        paths.append(path)
    return paths
