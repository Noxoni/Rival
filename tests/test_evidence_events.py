from __future__ import annotations

from pathlib import Path

from tools.evidence.events import DetectorParameters, detect_events, segment_records
from tools.evidence.io import EvidenceSession


def _record(
    tick: int,
    game_time: float,
    *,
    boost: float = 20.0,
    distance: float = 800.0,
    ball_height: float = 100.0,
    aerial: bool = False,
    action_jump: bool = False,
    eta_advantage: float = 0.4,
    score: tuple[int, int] = (0, 0),
    self_touch: float | None = None,
    opponent_touch: float | None = None,
) -> dict:
    players = [
        {
            "latest_touch": None if self_touch is None else {"game_seconds": self_touch},
            "last_input": {"boost": False, "jump": False, "throttle": 1.0, "steer": 0.0},
        },
        {
            "latest_touch": (
                None if opponent_touch is None else {"game_seconds": opponent_touch}
            ),
            "last_input": {"boost": True, "jump": False, "throttle": 1.0, "steer": 0.0},
        },
    ]
    return {
        "schema_version": 2,
        "record_type": "rival_policy_decision",
        "session_id": "synthetic-session",
        "decision": {
            "tick": tick,
            "game_time": game_time,
            "confidence": 0.8,
            "controller_action": {
                "jump": action_jump,
                "boost": aerial,
                "pitch": 0.8 if aerial else 0.0,
            },
            "top_actions": [],
        },
        "tactical_metrics": {
            "self_boost": boost,
            "opponent_boost": 50.0,
            "ball_height": ball_height,
            "distance_self_ball": distance,
            "distance_opponent_ball": 900.0,
            "eta_self_ball": 0.8,
            "eta_opponent_ball": 1.2,
            "possession_eta_advantage": eta_advantage,
            "challenge_closing_velocity": 500.0,
            "opponent_ball_closing_velocity": 600.0,
            "self_airborne": aerial,
            "selected_action_aerial_like": aerial,
        },
        "state": {"game_time": game_time, "score_diff": score[0] - score[1]},
        "packet": {
            "self_index": 0,
            "opponent_indices": [1],
            "players": players,
            "match": {
                "phase": {"name": "Active"},
                "scores": [
                    {"team": 0, "score": score[0]},
                    {"team": 1, "score": score[1]},
                ],
            },
        },
    }


def _session() -> EvidenceSession:
    records = [
        _record(0, 10.0, boost=12.0, distance=700.0),
        _record(1, 10.2, boost=11.0, distance=1200.0, ball_height=800.0, aerial=True, action_jump=True),
        _record(2, 10.4, boost=7.0, distance=1300.0, ball_height=850.0, aerial=True),
        _record(3, 10.6, boost=7.0, distance=1350.0, opponent_touch=10.55),
        _record(4, 10.8, boost=40.0, distance=1500.0, eta_advantage=-0.3, opponent_touch=10.55),
        _record(5, 11.0, boost=39.0, distance=1600.0, eta_advantage=-0.5, opponent_touch=10.55),
    ]
    for index, record in enumerate(records):
        record["_decision_index"] = index
    return EvidenceSession(
        session_id="synthetic-session",
        raw_path=Path("synthetic.jsonl"),
        raw_sha256="0" * 64,
        metadata={
            "source": "controlled_probe",
            "opponent": {"identity": "Controlled probe (boost_then_brake)"},
        },
        manifest={
            "schedule": [
                {
                    "start_game_time": 10.0,
                    "end_game_time": 11.0,
                    "parameters": {"behavior": "boost_then_brake", "repetition": 1},
                }
            ]
        },
        decisions=records,
        warnings=[],
    )


def test_event_segmentation_stops_at_score_boundary() -> None:
    records = [_record(0, 1.0), _record(1, 1.2), _record(2, 1.4, score=(1, 0))]
    assert [len(segment) for segment in segment_records(records)] == [2, 1]


def test_event_detection_covers_required_classes_and_ranks_deterministically() -> None:
    params = DetectorParameters()
    first = detect_events([_session()], params)
    second = detect_events([_session()], params)

    assert {event["class"] for event in first} == {
        "resource_stressed_aerial",
        "boost_detour_possession_loss",
        "apparent_vs_actual_challenge",
    }
    assert [event["event_id"] for event in first] == [
        event["event_id"] for event in second
    ]
    assert [event["ranking_score"] for event in first] == sorted(
        (event["ranking_score"] for event in first), reverse=True
    )
    assert all(event["candidate_only"] is True for event in first)
    controlled = next(
        event for event in first if event["class"] == "apparent_vs_actual_challenge"
    )
    assert controlled["raw_features"]["ground_truth_behavior"] == "boost_then_brake"
    assert controlled["derived_features"]["ground_truth_committed"] is False
