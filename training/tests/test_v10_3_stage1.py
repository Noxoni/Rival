"""Contract tests for Rival v10.3 Stage-1 V2 repair."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator

from rival_training.v10_2_campaign import boundary_ppo_batch_agent_steps
from rival_training.v10_2_curriculum import RivalBallAcquisitionCurriculumV1
from rival_training.v10_2_reward import (
    BallAcquisitionTransitionV1,
    RivalBallAcquisitionRewardKernelV1,
)
from rival_training.v10_3_campaign import (
    SOURCE_ACTOR_SHA256,
    SOURCE_MANIFEST_SHA256,
    load_stage1_config,
)
from rival_training.v10_3_curriculum import (
    ORDINARY_HEADING_ERROR_DEGREES,
    RivalBallAcquisitionCurriculumV2,
)
from rival_training.v10_3_environment import (
    RivalSingleLearnerGymWrapperV2,
    build_ball_acquisition_env,
)
from rival_training.v10_3_reward import (
    IDLE_PENALTY_RATE_PER_SIMULATED_SECOND,
    PHYSICAL_NEW_TOUCH_REWARD,
    BallAcquisitionTransitionV2,
    RivalBallAcquisitionRewardKernelV2,
    reward_truth_table_v2,
)


def _v2_transition(
    tick: int,
    car_x: float,
    *,
    ball_x: float = 1000.0,
    speed: float = 0.0,
    touches: int = 0,
) -> BallAcquisitionTransitionV2:
    return BallAcquisitionTransitionV2(
        tick=tick,
        car_position=np.asarray([car_x, 0.0, 17.0]),
        ball_position=np.asarray([ball_x, 0.0, 93.0]),
        car_linear_velocity=np.asarray([speed, 0.0, 0.0]),
        raw_touch_records=touches,
    )


def _wrapped_degrees(value: float) -> float:
    return abs(math.degrees(math.atan2(math.sin(value), math.cos(value))))


def _state_vectors(state) -> list[np.ndarray | float]:
    rows: list[np.ndarray | float] = [
        np.asarray(state.ball.position),
        np.asarray(state.ball.linear_velocity),
        np.asarray(state.ball.angular_velocity),
    ]
    for agent in sorted(state.cars, key=str):
        car = state.cars[agent]
        rows.extend(
            [
                np.asarray(car.physics.position),
                np.asarray(car.physics.linear_velocity),
                np.asarray(car.physics.angular_velocity),
                np.asarray(car.physics.euler_angles),
                float(car.boost_amount),
            ]
        )
    return rows


def test_v2_reward_truth_table_is_exact() -> None:
    report = reward_truth_table_v2()
    assert report["checks"]["passed"]
    assert report["metadata"]["generic_speed_reward"] == 0.0
    assert report["metadata"]["action_magnitude_reward"] == 0.0
    assert report["metadata"]["goal_for_reward"] == 0.0
    assert report["metadata"]["goal_against_reward"] == 0.0


def test_idle_penalty_is_tick_scaled_thresholded_and_stops_after_touch() -> None:
    kernel = RivalBallAcquisitionRewardKernelV2()
    kernel.reset(_v2_transition(0, 0.0))
    grace = kernel.step(_v2_transition(60, 0.0))
    multi_tick_idle = kernel.step(_v2_transition(66, 0.0))
    moving = kernel.step(_v2_transition(67, 0.0, speed=80.0))
    touch = kernel.step(_v2_transition(68, 0.0, touches=1))
    separated = kernel.step(_v2_transition(69, 0.0))
    after_touch = kernel.step(_v2_transition(120, 0.0))
    assert grace.idle_penalty == 0.0
    assert multi_tick_idle.idle_ticks == 6
    assert multi_tick_idle.idle_penalty == pytest.approx(
        IDLE_PENALTY_RATE_PER_SIMULATED_SECOND * 6 / 120
    )
    assert moving.idle_penalty == 0.0
    assert touch.components["physical_new_touch"] == PHYSICAL_NEW_TOUCH_REWARD
    assert touch.idle_penalty == 0.0
    assert separated.idle_penalty == 0.0
    assert after_touch.idle_penalty == 0.0


def test_v1_distance_and_touch_terms_are_byte_semantically_retained() -> None:
    v1 = RivalBallAcquisitionRewardKernelV1()
    v2 = RivalBallAcquisitionRewardKernelV2()
    v1.reset(
        BallAcquisitionTransitionV1(
            0, np.asarray([0.0, 0.0, 17.0]), np.asarray([1000.0, 0.0, 93.0])
        )
    )
    v2.reset(_v2_transition(0, 0.0))
    for tick, car_x, touches in ((1, 10.0, 0), (2, 20.0, 1), (3, 20.0, 1)):
        old = v1.step(
            BallAcquisitionTransitionV1(
                tick,
                np.asarray([car_x, 0.0, 17.0]),
                np.asarray([1000.0, 0.0, 93.0]),
                raw_touch_records=touches,
            )
        )
        new = v2.step(
            _v2_transition(tick, car_x, speed=100.0, touches=touches)
        )
        assert new.components["distance_progress"] == old.components[
            "distance_progress"
        ]
        assert new.components["physical_new_touch"] == old.components[
            "physical_new_touch"
        ]


@pytest.mark.parametrize(
    ("family", "limit"), ORDINARY_HEADING_ERROR_DEGREES.items()
)
def test_v2_ordinary_family_heading_limits(family: str, limit: float) -> None:
    engine = RocketSimEngine(rlbot_delay=True)
    teams = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    mutator = RivalBallAcquisitionCurriculumV2(
        "A", seed=20261033, forced_family=family
    )
    errors = []
    for _ in range(1000):
        state = engine.create_base_state()
        shared = {}
        teams.apply(state, shared)
        mutator.apply(state, shared)
        active = next(
            car
            for car in state.cars.values()
            if int(car.team_num) == int(shared["rival_v10_3_active_team"])
        )
        delta = np.asarray(state.ball.position) - np.asarray(
            active.physics.position
        )
        direction = math.atan2(float(delta[1]), float(delta[0]))
        yaw = float(active.physics.euler_angles[1])
        errors.append(_wrapped_degrees(yaw - direction))
    assert max(errors) <= limit + 1e-6


@pytest.mark.parametrize("family", ["awkward_heading", "natural_kickoff_holdout"])
def test_v2_locked_families_match_v1_for_same_seed(family: str) -> None:
    engine = RocketSimEngine(rlbot_delay=True)
    teams = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    old = RivalBallAcquisitionCurriculumV1(
        "A", seed=20261034, forced_family=family, forced_active_team=0
    )
    new = RivalBallAcquisitionCurriculumV2(
        "A", seed=20261034, forced_family=family, forced_active_team=0
    )
    for _ in range(50):
        old_state = engine.create_base_state()
        new_state = engine.create_base_state()
        old_shared = {}
        new_shared = {}
        teams.apply(old_state, old_shared)
        teams.apply(new_state, new_shared)
        old.apply(old_state, old_shared)
        new.apply(new_state, new_shared)
        for left, right in zip(
            _state_vectors(old_state), _state_vectors(new_state), strict=True
        ):
            np.testing.assert_array_equal(left, right)


def test_v2_single_learner_environment_keeps_frozen_contract() -> None:
    env = RivalSingleLearnerGymWrapperV2(
        build_ball_acquisition_env(
            phase="A",
            seed=20261035,
            forced_family="stationary_close",
            forced_active_team=0,
        )
    )
    try:
        observation = env.reset()
        assert observation.shape == (1, 714)
        action = np.zeros((1, 8), dtype=np.float32)
        next_observation, rewards, _, _, info = env.step(action)
        assert next_observation.shape == (1, 714)
        assert len(rewards) == 1
        assert info["rival_v10_2"]["dummy_rows_returned"] == 0
        assert env.rlgym_env.shared_info["rival_v10_3_reward_version"] == (
            "RivalBallAcquisitionRewardV2"
        )
    finally:
        env.close()


def test_v10_3_config_and_sub_minibatch_boundary_contract() -> None:
    config = load_stage1_config()
    assert config["policy_version"] == "RivalPolicyV1"
    assert config["observation_version"] == "RivalObsV1"
    assert config["action_version"] == "RivalActionV1"
    assert config["time_base"]["physics_hz"] == 120
    assert config["time_base"]["repeat_action"] is False
    assert config["time_base"]["one_tick_action_delay"] is True
    assert config["stage_contract"]["source_actor_sha256"] == SOURCE_ACTOR_SHA256
    assert SOURCE_MANIFEST_SHA256 == (
        "d1a785ef439b0127b5ab1a9ff1693ade1aa11d850151cd17b9733bbeb98dacb3"
    )
    assert boundary_ppo_batch_agent_steps(23_893, 24_000) == 23_893
