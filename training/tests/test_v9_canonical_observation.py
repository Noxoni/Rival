from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np

from rival_training.v9_canonical import (
    CANONICAL_ADAPTER_VERSION,
    CANONICAL_STATE_VERSION,
    STANDARD_GRAVITY_Z,
    STANDARD_PAD_IS_BIG,
    STANDARD_PAD_POSITIONS,
    RLBotCanonicalAdapterV1,
    RivalCanonicalStateV1,
    RocketSimCanonicalAdapterV1,
)
from rival_training.v9_observations import (
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    PREDICTION_COUNT,
    RivalObsV1Builder,
    observation_schema_manifest,
)


def _physics(
    position=(0.0, 0.0, 17.0),
    velocity=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
):
    return SimpleNamespace(
        position=np.asarray(position, dtype=np.float32),
        rotation_mtx=np.eye(3, dtype=np.float32),
        linear_velocity=np.asarray(velocity, dtype=np.float32),
        angular_velocity=np.asarray(angular, dtype=np.float32),
    )


def _rlgym_car(team: int, *, position, controller, touches=0):
    return SimpleNamespace(
        team_num=team,
        physics=_physics(position, (400.0, 50.0, 0.0), (0.1, -0.2, 0.3)),
        boost_amount=55.0,
        demo_respawn_timer=0.0,
        on_ground=True,
        is_boosting=bool(controller[6]),
        is_supersonic=False,
        handbrake=float(controller[7]),
        is_jumping=False,
        is_flipping=False,
        is_holding_jump=bool(controller[5]),
        has_jumped=False,
        has_double_jumped=False,
        has_flipped=False,
        can_flip=False,
        air_time_since_jump=0.0,
        jump_time=0.0,
        flip_time=0.0,
        flip_torque=np.zeros(3, dtype=np.float32),
        ball_touches=touches,
    )


def _flat_vec(values):
    return SimpleNamespace(x=float(values[0]), y=float(values[1]), z=float(values[2]))


def _flat_physics(source):
    return SimpleNamespace(
        location=_flat_vec(source.position),
        rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
        velocity=_flat_vec(source.linear_velocity),
        angular_velocity=_flat_vec(source.angular_velocity),
    )


def _flat_controller(values):
    return SimpleNamespace(
        throttle=float(values[0]),
        steer=float(values[1]),
        pitch=float(values[2]),
        yaw=float(values[3]),
        roll=float(values[4]),
        jump=bool(values[5]),
        boost=bool(values[6]),
        handbrake=bool(values[7]),
    )


def _equivalent_sources():
    self_action = np.asarray([0.25, -0.5, 0.75, -0.125, 0.625, 0, 1, 0], np.float32)
    opponent_action = np.asarray([-0.4, 0.3, -0.2, 0.1, -0.6, 0, 0, 1], np.float32)
    self_car = _rlgym_car(0, position=(-1000.0, -2000.0, 17.0), controller=self_action, touches=1)
    opponent_car = _rlgym_car(1, position=(900.0, 1800.0, 17.0), controller=opponent_action)
    ball = _physics((100.0, 200.0, 93.0), (500.0, -100.0, 20.0), (0.0, 0.5, 0.0))
    timers = np.zeros(34, dtype=np.float32)
    timers[0] = 2.0
    timers[3] = 7.0
    rlgym_state = SimpleNamespace(
        tick_count=1200,
        cars={"self": self_car, "opponent": opponent_car},
        ball=ball,
        boost_pad_timers=timers,
        goal_scored=False,
        config=SimpleNamespace(gravity=1.0),
    )
    shared = {
        "rival_v9_applied_actions": {
            "self": self_action,
            "opponent": opponent_action,
        },
        "score_by_team": {0: 2, 1: 1},
        "game_time_remaining": 290.0,
        "overtime": False,
        "kickoff": False,
        "active_play": True,
    }

    players = [
        SimpleNamespace(
            team=0,
            physics=_flat_physics(self_car.physics),
            boost=55.0,
            demolished_timeout=-1.0,
            is_supersonic=False,
            air_state="OnGround",
            has_jumped=False,
            has_double_jumped=False,
            has_dodged=False,
            dodge_timeout=-1.0,
            dodge_elapsed=0.0,
            dodge_dir=SimpleNamespace(x=0.0, y=0.0),
            last_input=_flat_controller(self_action),
            latest_touch=SimpleNamespace(game_seconds=10.0),
        ),
        SimpleNamespace(
            team=1,
            physics=_flat_physics(opponent_car.physics),
            boost=55.0,
            demolished_timeout=-1.0,
            is_supersonic=False,
            air_state="OnGround",
            has_jumped=False,
            has_double_jumped=False,
            has_dodged=False,
            dodge_timeout=-1.0,
            dodge_elapsed=0.0,
            dodge_dir=SimpleNamespace(x=0.0, y=0.0),
            last_input=_flat_controller(opponent_action),
            latest_touch=None,
        ),
    ]
    dynamic_pads = []
    for index, remaining in enumerate(timers):
        big = bool(STANDARD_PAD_IS_BIG[index])
        if remaining <= 0:
            dynamic_pads.append(SimpleNamespace(is_active=True, timer=0.0))
        else:
            dynamic_pads.append(
                SimpleNamespace(
                    is_active=False,
                    timer=(10.0 if big else 4.0) - float(remaining),
                )
            )
    packet = SimpleNamespace(
        players=players,
        balls=[SimpleNamespace(physics=_flat_physics(ball))],
        boost_pads=dynamic_pads,
        teams=[
            SimpleNamespace(team_index=0, score=2),
            SimpleNamespace(team_index=1, score=1),
        ],
        match_info=SimpleNamespace(
            seconds_elapsed=10.0,
            game_time_remaining=290.0,
            frame_num=1200,
            is_overtime=False,
            match_phase="Active",
            world_gravity_z=-650.0,
        ),
    )
    field_info = SimpleNamespace(
        boost_pads=[
            SimpleNamespace(
                location=_flat_vec(position), is_full_boost=bool(STANDARD_PAD_IS_BIG[index])
            )
            for index, position in enumerate(STANDARD_PAD_POSITIONS)
        ]
    )
    return rlgym_state, shared, packet, field_info


def _prediction_provider(ball):
    horizons = np.arange(1, PREDICTION_COUNT + 1, dtype=np.float32)[:, None]
    positions = ball.position[None, :] + horizons * np.asarray([10.0, 20.0, 5.0])
    velocities = ball.linear_velocity[None, :] + horizons * np.asarray([1.0, -2.0, 0.5])
    return positions.astype(np.float32), velocities.astype(np.float32)


def _assert_canonical_equal(left: RivalCanonicalStateV1, right: RivalCanonicalStateV1):
    left_payload = left.to_payload()
    right_payload = right.to_payload()

    def compare(a, b):
        if isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, list):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-6, rtol=0)
        elif isinstance(a, float):
            assert abs(a - b) <= 1e-6
        else:
            assert a == b

    compare(left_payload, right_payload)


def test_rocketsim_and_rlbot_thin_adapters_reach_same_canonical_state() -> None:
    rlgym_state, shared, packet, field_info = _equivalent_sources()
    training = RocketSimCanonicalAdapterV1().adapt(rlgym_state, "self", shared)
    deployment = RLBotCanonicalAdapterV1().adapt(packet, 0, field_info)
    _assert_canonical_equal(training, deployment)
    assert training.version == CANONICAL_STATE_VERSION
    assert training.adapter_version == CANONICAL_ADAPTER_VERSION
    assert training.pad_positions.shape == (34, 3)
    assert training.pad_time_until_active[0] == 2.0
    assert training.pad_time_until_active[3] == 7.0


def test_canonical_json_round_trip_is_observation_bit_identical() -> None:
    rlgym_state, shared, _, _ = _equivalent_sources()
    canonical = RocketSimCanonicalAdapterV1().adapt(rlgym_state, "self", shared)
    first = RivalObsV1Builder(prediction_provider=_prediction_provider)
    runtime_before = first.export_runtime_state()
    expected = first.build(canonical)

    serialized = json.dumps(canonical.to_payload(), sort_keys=True)
    restored = RivalCanonicalStateV1.from_payload(json.loads(serialized))
    second = RivalObsV1Builder(prediction_provider=_prediction_provider)
    second.load_runtime_state(json.loads(json.dumps(runtime_before)))
    actual = second.build(restored)
    assert np.array_equal(actual, expected)


def test_generated_schema_is_contiguous_complete_and_hashed() -> None:
    manifest = observation_schema_manifest()
    assert manifest["observation_version"] == OBSERVATION_VERSION
    assert manifest["canonical_state_version"] == CANONICAL_STATE_VERSION
    assert manifest["float_count"] == OBSERVATION_SIZE == 714
    assert manifest["running_standardization"] is False
    assert len(manifest["schema_sha256"]) == 64
    assert len(manifest["builder_source_sha256"]) == 64
    assert len(manifest["canonical_source_sha256"]) == 64
    offset = 0
    names = set()
    for field in manifest["fields"]:
        assert field["start"] == offset
        assert field["end"] > field["start"]
        assert field["name"] not in names
        assert field["normalization"]
        assert field["coordinate_frame"]
        assert field["canonical_source"]
        assert field["update_cadence"]
        assert field["reset_semantics"]
        names.add(field["name"])
        offset = field["end"]
    assert offset == OBSERVATION_SIZE
    assert manifest["entity_shapes"]["boost_pads"] == [34, 9]
    assert manifest["entity_shapes"]["prediction"] == [6, 12]


def test_runtime_history_motion_delta_and_prediction_cadence_are_explicit() -> None:
    rlgym_state, shared, _, _ = _equivalent_sources()
    canonical = RocketSimCanonicalAdapterV1().adapt(rlgym_state, "self", shared)
    calls = []

    def provider(ball):
        calls.append(ball.position.copy())
        return _prediction_provider(ball)

    builder = RivalObsV1Builder(prediction_refresh_ticks=4, prediction_provider=provider)
    ages = []
    for offset in range(5):
        self_physics = replace(
            canonical.self_car.physics,
            linear_velocity=canonical.self_car.physics.linear_velocity
            + np.asarray([float(offset), 0.0, 0.0], dtype=np.float32),
        )
        self_car = replace(
            canonical.self_car,
            physics=self_physics,
            latest_controller=np.asarray([offset / 4, 0, 0, 0, 0, 0, 0, 0], np.float32),
        )
        state = replace(canonical, tick_index=canonical.tick_index + offset, self_car=self_car)
        observation = builder.build(state)
        assert observation.shape == (OBSERVATION_SIZE,)
        assert np.isfinite(observation).all()
        ages.append(int(builder.last_timings["prediction_age_ticks"]))
    assert ages == [0, 1, 2, 3, 0]
    assert len(calls) == 2
    assert np.array_equal(np.stack(builder.self_history)[-1], state.self_car.latest_controller)
    np.testing.assert_allclose(builder.motion_delta[0, 0], 1.0, atol=0, rtol=0)


def test_team_inversion_is_fixed_and_never_depends_on_current_x_position() -> None:
    rlgym_state, shared, _, _ = _equivalent_sources()
    blue = RocketSimCanonicalAdapterV1().adapt(rlgym_state, "self", shared)
    orange = RocketSimCanonicalAdapterV1().adapt(rlgym_state, "opponent", shared)
    np.testing.assert_array_equal(blue.self_car.physics.position, [-1000.0, -2000.0, 17.0])
    np.testing.assert_array_equal(orange.self_car.physics.position, [-900.0, -1800.0, 17.0])
    np.testing.assert_array_equal(blue.pad_positions, orange.pad_positions)

    moved = replace(
        blue,
        self_car=replace(
            blue.self_car,
            physics=replace(
                blue.self_car.physics,
                position=np.asarray([1000.0, -2000.0, 17.0], dtype=np.float32),
            ),
        ),
    )
    first = RivalObsV1Builder(prediction_provider=_prediction_provider).build(blue)
    second = RivalObsV1Builder(prediction_provider=_prediction_provider).build(moved)
    position_field = next(
        field for field in observation_schema_manifest()["fields"] if field["name"] == "self.position"
    )
    assert first[position_field["start"]] == -second[position_field["start"]]
    assert np.array_equal(
        first[position_field["start"] + 1 : position_field["end"]],
        second[position_field["start"] + 1 : position_field["end"]],
    )


def test_default_shared_rocketsim_predictor_builds_finite_observation() -> None:
    # A real RocketSimEngine is constructed first by the environment smoke in the
    # adjacent test suite; this direct canonical case verifies the production
    # predictor implementation rather than only the deterministic test provider.
    rlgym_state, shared, _, _ = _equivalent_sources()
    canonical = RocketSimCanonicalAdapterV1().adapt(rlgym_state, "self", shared)
    builder = RivalObsV1Builder(prediction_refresh_ticks=4)
    observation = builder.build(canonical)
    assert observation.shape == (OBSERVATION_SIZE,)
    assert np.isfinite(observation).all()
    assert builder.last_timings["prediction_refreshed"] is True
    assert float(builder.last_timings["predictor_seconds"]) > 0
    assert canonical.gravity_z == STANDARD_GRAVITY_Z
