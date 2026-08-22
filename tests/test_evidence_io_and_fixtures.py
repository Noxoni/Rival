from __future__ import annotations

import json

from tools.evidence.events import DetectorParameters
from tools.evidence.fixtures import curate_top_fixtures
from tools.evidence.io import load_session


def _v1_decision(game_time: float = 1.0) -> dict:
    return {
        "schema_version": 1,
        "record_type": "rival_policy_decision",
        "decision": {
            "tick": 1,
            "game_time": game_time,
            "confidence": 0.5,
            "controller_action": {"jump": False, "boost": False, "pitch": 0.0},
            "top_actions": [],
        },
        "tactical_metrics": {
            "self_boost": 50.0,
            "ball_height": 100.0,
            "distance_self_ball": 400.0,
            "selected_action_aerial_like": False,
        },
        "state": {"game_time": game_time, "score_diff": 0},
        "runtime": {"strategic_overrides_enabled": False},
    }


def test_schema_v1_is_read_with_stable_legacy_session_id(tmp_path) -> None:
    raw = tmp_path / "legacy.jsonl"
    raw.write_text(json.dumps(_v1_decision()) + "\n", encoding="utf-8")

    session = load_session(raw)

    assert session.session_id == "legacy-v1-legacy"
    assert len(session.decisions) == 1
    assert session.decisions[0]["session_id"] == "legacy-v1-legacy"


def test_fixture_curation_preserves_source_hash_and_bounded_records(tmp_path) -> None:
    raw = tmp_path / "decisions.jsonl"
    start = {
        "schema_version": 2,
        "record_type": "rival_session_start",
        "session_id": "fixture-source",
        "metadata": {
            "source": "synthetic_test",
            "opponent": {"identity": "Synthetic"},
        },
    }
    decisions = []
    for tick in range(8):
        record = _v1_decision(float(tick))
        record.update(schema_version=2, session_id="fixture-source")
        record["decision"]["tick"] = tick
        decisions.append(record)
    raw.write_text(
        "\n".join(json.dumps(record) for record in [start, *decisions]) + "\n",
        encoding="utf-8",
    )
    session = load_session(raw)
    event = {
        "event_id": "synthetic-event",
        "class": "resource_stressed_aerial",
        "session_id": "fixture-source",
        "start_game_time": 2.0,
        "end_game_time": 6.0,
        "post_window_record_range": [4, 6],
        "ranking_explanation": ["synthetic fixture test"],
    }

    paths = curate_top_fixtures([event], [session], tmp_path / "fixtures")

    assert len(paths) == 1
    fixture = json.loads(paths[0].read_text(encoding="utf-8"))
    assert fixture["source_raw_sha256"] == session.raw_sha256
    assert fixture["source_event_id"] == "synthetic-event"
    assert 1 <= len(fixture["records"]) <= 6
    assert event["fixture_path"] == str(paths[0])
    assert DetectorParameters().to_record()["pre_window_seconds"] == 1.5
