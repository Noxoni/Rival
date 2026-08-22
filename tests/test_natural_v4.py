from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.run_m04_natural_batch as natural_batch


def test_v4_natural_schedule_balances_opponents_and_sides() -> None:
    schedule = natural_batch.build_schedule(4)

    assert len(schedule) == 8
    assert sum(item["opponent"] == "nexto" for item in schedule) == 4
    assert sum(item["opponent"] == "wisp" for item in schedule) == 4
    for opponent in natural_batch.OPPONENTS:
        opponent_cases = [item for item in schedule if item["opponent"] == opponent]
        assert sum(item["rival_team"] == 0 for item in opponent_cases) == 2
        assert sum(item["rival_team"] == 1 for item in opponent_cases) == 2


def test_v4_natural_health_uses_effective_speed_not_packet_echo(
    tmp_path,
    monkeypatch,
) -> None:
    raw_root = tmp_path / "raw"
    session_id = "natural-health"
    session_root = raw_root / session_id
    session_root.mkdir(parents=True)
    records = []
    for index in range(120):
        records.append(
            {
                "schema_version": 3,
                "record_type": "rival_policy_decision",
                "decision": {
                    "final_action_index": index % 10,
                    "game_time": index / 15.0,
                    "timestamp_unix_ns": 1_000_000_000 + index * 13_333_333,
                },
                "packet": {
                    "match": {
                        "phase": {"name": "Active"},
                        "scores": [
                            {"team": 0, "score": 0},
                            {"team": 1, "score": 0},
                        ],
                    },
                    "opponent_indices": [1],
                    "players": [
                        {},
                        {
                            "last_input": {
                                "throttle": 1.0,
                                "steer": 0.0,
                                "pitch": 0.0,
                                "yaw": 0.0,
                                "roll": 0.0,
                                "jump": False,
                                "boost": index % 2 == 0,
                                "handbrake": False,
                            }
                        },
                    ],
                },
            }
        )
    (session_root / "decisions.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(natural_batch, "RAW_EVIDENCE_ROOT", raw_root)
    manifest = {
        "status": "complete",
        "termination_reason": "match_phase_ended",
        "raw_telemetry": {"invalid_record_count": 0},
        "execution": {
            "game_seconds_advanced": 8.0,
            "effective_game_seconds_per_wall_second": 3.9,
            "observed_game_speed_all_active": {"median": 1.0},
        },
    }

    health = natural_batch.telemetry_health(session_id, manifest)

    assert health["accepted"] is True
    assert health["checks"]["effective_5x_progression"] is True
    assert health["decisions_per_game_second"] == 15.0
    assert health["distinct_action_indices"] == 10
    assert health["end_to_end_game_seconds_per_wall_second"] == 3.9
    assert health["sustained_in_play_speed"]["weighted_rate"] == pytest.approx(5.0)


def test_compact_manifest_omits_workstation_reference_paths(monkeypatch) -> None:
    monkeypatch.setattr(
        natural_batch,
        "telemetry_health",
        lambda _session_id, _manifest: {"accepted": True},
    )
    manifest = {
        "session_id": "path-sanitization",
        "opponent": {
            "key": "nexto",
            "identity": "Nexto",
            "root": "/fixture/installed/Nexto",
            "config_path": "/fixture/installed/Nexto/bot.toml",
            "executable_path": "/fixture/installed/Nexto/nexto.exe",
            "config_sha256": "a" * 64,
            "executable_sha256": "b" * 64,
        },
    }

    compact = natural_batch.compact_manifest(manifest)

    assert compact["opponent"] == {
        "key": "nexto",
        "identity": "Nexto",
        "config_sha256": "a" * 64,
        "executable_sha256": "b" * 64,
        "config_filename": Path("bot.toml").name,
        "executable_filename": Path("nexto.exe").name,
    }
    assert "person" not in json.dumps(compact)
