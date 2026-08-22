from __future__ import annotations

import json
from pathlib import Path

from tools.evidence.challenge_v3 import observe_decisions


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _record(game_time: float, *, jump: bool) -> dict:
    action = {
        "throttle": 1.0,
        "steer": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "jump": jump,
        "boost": False,
        "handbrake": False,
    }
    return {
        "schema_version": 3,
        "record_type": "rival_policy_decision",
        "decision": {
            "game_time": game_time,
            "controller_action": action,
            "baseline_controller_action": action,
            "final_controller_action": action,
            "action_index": 1 if jump else 0,
            "final_action_index": 1 if jump else 0,
        },
        "tactical_metrics": {
            "eta_self_ball": 0.40,
            "eta_opponent_ball": 1.20,
            "opponent_ball_closing_velocity": 180.0,
            "challenge_closing_velocity": 200.0,
        },
        "packet": {
            "self_index": 0,
            "opponent_indices": [1],
            "players": [
                {
                    "team": 0,
                    "player_id": 1,
                    "air_state": {"name": "OnGround"},
                    "demolished_timeout": -1.0,
                    "last_input": action,
                    "latest_touch": None,
                    "physics": {
                        "position": {"x": 0.0, "y": -400.0, "z": 17.0},
                        "velocity": {"x": 0.0, "y": 300.0, "z": 0.0},
                        "rotation": {
                            "forward": {"x": 0.0, "y": 1.0, "z": 0.0}
                        },
                    },
                },
                {
                    "team": 1,
                    "player_id": 2,
                    "air_state": {"name": "OnGround"},
                    "demolished_timeout": -1.0,
                    "last_input": {
                        **action,
                        "jump": True,
                        "boost": True,
                    },
                    "latest_touch": None,
                    "physics": {
                        "position": {"x": 700.0, "y": 700.0, "z": 17.0},
                        "velocity": {"x": 500.0, "y": 0.0, "z": 0.0},
                        "rotation": {
                            "forward": {"x": 1.0, "y": 0.0, "z": 0.0}
                        },
                    },
                },
            ],
            "ball": {
                "physics": {
                    "position": {"x": 0.0, "y": 0.0, "z": 100.0},
                    "velocity": {"x": 0.0, "y": 100.0, "z": 0.0},
                }
            },
            "match": {
                "seconds_elapsed": game_time,
                "phase": {"name": "Active"},
                "scores": [{"team": 0, "score": 0}, {"team": 1, "score": 0}],
            },
        },
    }


def test_narrow_metric_detects_grounded_jump_but_not_jump_or_boost_input_alone() -> None:
    observations = observe_decisions(
        [_record(1.0, jump=False), _record(1.08, jump=True)]
    )

    assert observations[0].estimate.state != "high"
    assert observations[0].premature_release_jump is False
    assert observations[1].final_jump_initiation is True
    assert observations[1].premature_release_jump is True


def test_existing_jump_fake_fixture_is_deterministic_and_broad_label_is_not_reused() -> None:
    path = (
        REPOSITORY_ROOT
        / "fixtures"
        / "evidence"
        / "apparent_vs_actual_challenge__appa-1662387304b0.json"
    )
    fixture = json.loads(path.read_text(encoding="utf-8"))

    first = observe_decisions(fixture["records"])
    second = observe_decisions(fixture["records"])

    assert fixture["event"]["derived_features"]["rival_release_like_response"] is True
    assert [item.estimate.to_record() for item in first] == [
        item.estimate.to_record() for item in second
    ]
    assert all(item.estimate.valid for item in first)
    assert sum(item.premature_release_jump for item in first) == 0
