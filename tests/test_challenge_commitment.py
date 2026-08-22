from __future__ import annotations

from dataclasses import replace
import math

import pytest

from strategy.challenge_commitment import (
    ChallengeCommitmentTracker,
    ChallengeSample,
    projected_closest_approach,
)


def _sample(
    game_time: float,
    *,
    opponent_position: tuple[float, float, float] = (0.0, 900.0, 17.0),
    opponent_velocity: tuple[float, float, float] = (0.0, -700.0, 0.0),
    opponent_forward: tuple[float, float, float] = (0.0, -1.0, 0.0),
    opponent_eta: float | None = 0.75,
    ball_closing: float | None = 700.0,
    rival_closing: float | None = 900.0,
    throttle: float = 1.0,
    steer: float = 0.0,
    jump: bool = False,
    boost: bool = False,
    phase: str = "Active",
    scores: tuple[int, ...] = (0, 0),
) -> ChallengeSample:
    return ChallengeSample(
        game_time=game_time,
        self_position=(0.0, -420.0, 17.0),
        self_velocity=(0.0, 350.0, 0.0),
        opponent_position=opponent_position,
        opponent_velocity=opponent_velocity,
        opponent_forward=opponent_forward,
        ball_position=(0.0, 0.0, 100.0),
        ball_velocity=(0.0, 120.0, 0.0),
        self_team=0,
        self_grounded=True,
        opponent_airborne=False,
        self_demoed=False,
        opponent_demoed=False,
        opponent_id=2,
        opponent_throttle=throttle,
        opponent_steer=steer,
        opponent_jump=jump,
        opponent_boost=boost,
        opponent_handbrake=False,
        opponent_input_available=True,
        self_eta_to_ball=0.45,
        opponent_eta_to_ball=opponent_eta,
        opponent_ball_closing_speed=ball_closing,
        opponent_rival_closing_speed=rival_closing,
        phase=phase,
        scores=scores,
    )


def test_projected_closest_approach_handles_intercept_and_zero_relative_speed() -> None:
    closest_time, miss = projected_closest_approach(
        (0.0, 1000.0, 0.0),
        (0.0, -1000.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        2.0,
    )
    assert closest_time == pytest.approx(1.0)
    assert miss == pytest.approx(0.0)

    stationary_time, stationary_miss = projected_closest_approach(
        (0.0, 1000.0, 0.0),
        (10.0, 20.0, 0.0),
        (0.0, 0.0, 0.0),
        (10.0, 20.0, 0.0),
        1.0,
    )
    assert stationary_time == 0.0
    assert stationary_miss == pytest.approx(1000.0)


def test_true_intersecting_trajectory_reaches_high_without_controller_inputs() -> None:
    tracker = ChallengeCommitmentTracker()
    sample = replace(
        _sample(
            1.0,
            opponent_position=(0.0, 700.0, 17.0),
            opponent_velocity=(0.0, -900.0, 0.0),
            opponent_eta=0.52,
            ball_closing=1020.0,
        ),
        opponent_input_available=False,
    )

    first = tracker.update(sample)
    second = tracker.update(replace(sample, game_time=1.08, opponent_eta_to_ball=0.62))

    assert first.valid and second.valid
    assert second.pressure_present is True
    assert second.state == "high"
    assert second.score >= tracker.parameters.high_threshold
    assert second.components["projected_miss_distance"] < 150.0
    assert math.isfinite(second.score)


def test_jump_and_boost_alone_do_not_force_high_commitment() -> None:
    tracker = ChallengeCommitmentTracker()
    sideways = _sample(
        2.0,
        opponent_position=(700.0, 700.0, 17.0),
        opponent_velocity=(500.0, 0.0, 0.0),
        opponent_forward=(1.0, 0.0, 0.0),
        opponent_eta=1.2,
        ball_closing=180.0,
        rival_closing=200.0,
        jump=True,
        boost=True,
    )

    estimate = tracker.update(sideways)

    assert estimate.pressure_present is True
    assert estimate.state != "high"
    assert estimate.components["opponent_jump"] is True
    assert estimate.components["opponent_boost"] is True


def test_clear_abort_reduces_commitment_within_one_policy_tick() -> None:
    tracker = ChallengeCommitmentTracker()
    committed = _sample(
        3.0,
        opponent_position=(0.0, 700.0, 17.0),
        opponent_velocity=(0.0, -900.0, 0.0),
        opponent_eta=0.52,
        ball_closing=1020.0,
    )
    tracker.update(committed)
    high = tracker.update(replace(committed, game_time=3.08, opponent_eta_to_ball=0.62))
    aborted = tracker.update(
        _sample(
            3.16,
            opponent_velocity=(0.0, 300.0, 0.0),
            opponent_forward=(1.0, 0.0, 0.0),
            opponent_eta=1.05,
            ball_closing=-300.0,
            rival_closing=-100.0,
            throttle=-1.0,
            steer=1.0,
        )
    )

    assert high.state == "high"
    assert aborted.abort_detected is True
    assert aborted.state != "high"
    assert "reverse_or_brake_input" in aborted.history["abort_signals"]
    assert "closing_speed_collapse" in aborted.history["abort_signals"]
    assert aborted.score < high.score


@pytest.mark.parametrize(
    ("next_sample", "expected_reason"),
    [
        (replace(_sample(5.0), game_time=0.25), "time_rewind"),
        (replace(_sample(1.08), scores=(1, 0)), "score_change"),
        (replace(_sample(1.08), phase="Countdown"), "phase_countdown"),
        (
            replace(_sample(1.08), opponent_position=(3000.0, 3000.0, 17.0)),
            "state_discontinuity",
        ),
    ],
)
def test_history_resets_deterministically(
    next_sample: ChallengeSample,
    expected_reason: str,
) -> None:
    tracker = ChallengeCommitmentTracker()
    tracker.update(_sample(1.0))

    result = tracker.update(next_sample)

    assert result.reset_reason == expected_reason
    assert result.history["sample_count"] == 1


def test_missing_player_and_demolition_do_not_reuse_stale_history() -> None:
    tracker = ChallengeCommitmentTracker()
    tracker.update(_sample(1.0))
    missing = tracker.update(None)
    repeated_missing = tracker.update(None)
    demo = tracker.update(replace(_sample(2.0), opponent_demoed=True))

    assert missing.valid is False
    assert repeated_missing.episode_id == missing.episode_id
    assert demo.valid is False
    assert demo.reset_reason == "demolition"
