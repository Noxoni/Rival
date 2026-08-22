from __future__ import annotations

from types import SimpleNamespace

import pytest

from analysis.tactical_metrics import build_state_snapshot, compute_tactical_metrics


def _vec(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def _action(**overrides: float | bool) -> SimpleNamespace:
    values: dict[str, float | bool] = {
        "throttle": 1.0,
        "steer": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "jump": False,
        "boost": False,
        "handbrake": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _player(
    *,
    index: int,
    x: float,
    velocity_x: float,
    boost: float,
    on_ground: bool,
    team: str,
) -> SimpleNamespace:
    player = SimpleNamespace(
        index=index,
        pos=_vec(x, 0.0, 17.0),
        vel=_vec(velocity_x, 0.0, 0.0),
        boost=boost,
        is_on_ground=on_ground,
        is_jumping=False,
        is_supersonic=False,
        team=SimpleNamespace(name=team),
        prev_action=_action(),
    )
    player.has_flip_or_jump = lambda: True
    return player


def test_tactical_metrics_handle_missing_opponent_safely() -> None:
    state = SimpleNamespace(
        ball=SimpleNamespace(pos=_vec(100.0, 0.0, 120.0), vel=_vec(0.0, 0.0, 0.0))
    )
    player = _player(
        index=0,
        x=0.0,
        velocity_x=10.0,
        boost=37.0,
        on_ground=True,
        team="BLUE",
    )

    metrics = compute_tactical_metrics(
        state,
        player,
        None,
        _action(boost=True),
        score_diff=-1,
        seconds_remaining=72.0,
    )

    assert metrics.self_boost == pytest.approx(37.0)
    assert metrics.opponent_boost is None
    assert metrics.distance_opponent_ball is None
    assert metrics.eta_opponent_ball is None
    assert metrics.opponent_airborne is None
    assert metrics.challenge_closing_velocity is None
    assert metrics.eta_self_ball is not None
    assert metrics.selected_action_uses_boost is True
    assert metrics.selected_action_aerial_like is False


def test_tactical_metrics_capture_eta_advantage_and_aerial_action() -> None:
    state = SimpleNamespace(
        ball=SimpleNamespace(pos=_vec(500.0, 0.0, 300.0), vel=_vec(0.0, 0.0, 0.0))
    )
    player = _player(
        index=0,
        x=0.0,
        velocity_x=500.0,
        boost=20.0,
        on_ground=False,
        team="BLUE",
    )
    opponent = _player(
        index=1,
        x=1000.0,
        velocity_x=-200.0,
        boost=50.0,
        on_ground=True,
        team="ORANGE",
    )

    metrics = compute_tactical_metrics(
        state,
        player,
        opponent,
        _action(pitch=1.0, boost=True),
        score_diff=1,
        seconds_remaining=30.0,
        eta_self_ball=0.8,
        eta_opponent_ball=1.1,
        eta_method="test_eta",
    )

    assert metrics.possession_eta_advantage == pytest.approx(0.3)
    assert metrics.self_airborne is True
    assert metrics.opponent_airborne is False
    assert metrics.selected_action_aerial_like is True
    assert metrics.eta_method == "test_eta"


def test_state_snapshot_records_boost_map_and_previous_action() -> None:
    player = _player(
        index=0,
        x=0.0,
        velocity_x=0.0,
        boost=12.0,
        on_ground=True,
        team="BLUE",
    )
    state = SimpleNamespace(
        ball=SimpleNamespace(pos=_vec(0.0, 100.0, 92.0), vel=_vec(0.0, 0.0, 0.0)),
        boost_pads=[True, False, True],
    )
    locations = [_vec(100.0, 0.0, 73.0), _vec(200.0, 0.0, 70.0), _vec(50.0, 0.0, 70.0)]

    snapshot = build_state_snapshot(
        state,
        player,
        None,
        score_diff=0,
        game_time=10.0,
        seconds_remaining=290.0,
        boost_locations=locations,
    )

    assert snapshot["opponent"] is None
    assert snapshot["self"]["previous_action"]["throttle"] == pytest.approx(1.0)
    assert snapshot["boost_map"]["active_large_pad_indices"] == [0]
    assert snapshot["boost_map"]["active_small_pad_indices"] == [2]
    assert snapshot["boost_map"]["nearby_active_opportunities"][0]["index"] == 2
