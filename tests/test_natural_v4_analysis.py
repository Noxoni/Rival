from __future__ import annotations

from tools.evidence.natural_v4 import (
    NaturalAnalysisParameters,
    PATTERN_DEFINITIONS,
    _collapse_overlapping_events,
    _pattern_summary,
)


def _challenge_event(session_id: str, game_time: float, next_touch: str) -> dict:
    return {
        "class": "apparent_vs_actual_challenge",
        "session_id": session_id,
        "anchor_game_time": game_time,
        "ranking_score": 80.0,
        "derived_features": {
            "closing_aborted_in_window": True,
            "rival_release_like_response": True,
        },
        "outcome": {"next_touch": next_touch},
    }


def test_overlapping_natural_candidates_count_as_one_episode() -> None:
    events = [
        _challenge_event("one", 10.0, "opponent"),
        _challenge_event("one", 10.4, "opponent"),
        _challenge_event("one", 12.1, "self"),
        _challenge_event("two", 10.1, "self"),
    ]

    collapsed = _collapse_overlapping_events(events, 2.0)

    assert [(event["session_id"], event["anchor_game_time"]) for event in collapsed] == [
        ("one", 10.0),
        ("one", 12.1),
        ("two", 10.1),
    ]


def test_pattern_summary_charges_next_touch_and_near_term_goal_consequence() -> None:
    definition = PATTERN_DEFINITIONS[0]
    events = [
        _challenge_event("one", 10.0, "opponent"),
        _challenge_event("one", 10.5, "self"),
        _challenge_event("two", 20.0, "self"),
    ]
    goals = {
        "one": [{"game_time": 16.0, "rival_outcome": "conceded"}],
        "two": [{"game_time": 35.0, "rival_outcome": "scored"}],
    }

    summary = _pattern_summary(
        definition,
        events,
        goals,
        session_count=2,
        parameters=NaturalAnalysisParameters(),
    )

    assert summary["raw_candidate_count"] == 3
    assert summary["independent_episode_count"] == 2
    assert summary["matches_with_episode"] == 2
    assert summary["next_touch"]["opponent"] == 1
    assert summary["next_goal_within_window"]["conceded"] == 1
    assert summary["next_goal_within_window"]["none"] == 1
    assert summary["high_consequence_episode_count"] == 1
    assert summary["priority_score"] > 0.0
