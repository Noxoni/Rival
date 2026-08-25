"""Contracts for the deliberately minimal M10.10 Stage-1 experiment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rival_training.v10_10_campaign import (
    PAIRED_ACTOR_STATE_SHA256,
    PAIRED_CRITIC_STATE_SHA256,
    clean_initialization,
    load_stage1_config,
)
from rival_training.v10_10_environment import (
    RivalActiveLearnerFirstTouchConditionV1,
    RivalSingleLearnerFirstTouchWrapperV1,
    build_first_touch_velocity_env,
)
from rival_training.v10_10_reward import (
    COMPONENTS,
    FIRST_PHYSICAL_TOUCH_REWARD,
    MAXIMUM_CAR_SPEED_UU_PER_SECOND,
    PHYSICS_HZ,
    FirstTouchVelocityTransitionV1,
    RivalFirstTouchVelocityRewardKernelV1,
    reward_truth_table_v10_10,
)


def _transition(
    tick: int,
    *,
    velocity=(0.0, 0.0, 0.0),
    ball=(5000.0, 0.0, 17.0),
    touches: int = 0,
) -> FirstTouchVelocityTransitionV1:
    return FirstTouchVelocityTransitionV1(
        tick=tick,
        car_position=np.asarray([0.0, 0.0, 17.0]),
        ball_position=np.asarray(ball, dtype=np.float64),
        car_linear_velocity=np.asarray(velocity, dtype=np.float64),
        raw_touch_records=touches,
    )


def test_reward_truth_table_proves_minimal_contract() -> None:
    report = reward_truth_table_v10_10()
    checks = {key: value for key, value in report["checks"].items() if key != "passed"}
    assert len(checks) == 14
    assert report["checks"]["passed"]
    assert all(checks.values())


def test_native_120hz_velocity_reward_integrates_in_physical_time() -> None:
    kernel = RivalFirstTouchVelocityRewardKernelV1()
    kernel.reset(_transition(0))
    steps = [
        kernel.step(
            _transition(
                tick,
                velocity=(MAXIMUM_CAR_SPEED_UU_PER_SECOND, 0.0, 0.0),
            )
        )
        for tick in range(1, PHYSICS_HZ + 1)
    ]
    assert all(
        step.normalized_directed_velocity == pytest.approx(1.0)
        for step in steps
    )
    assert sum(step.velocity_to_ball_reward for step in steps) == pytest.approx(
        1.0
    )
    assert kernel.velocity_to_ball_total == pytest.approx(1.0)


def test_reward_depends_on_car_velocity_not_ball_motion() -> None:
    stationary = RivalFirstTouchVelocityRewardKernelV1()
    stationary.reset(_transition(0, ball=(5000.0, 0.0, 17.0)))
    moved_ball = stationary.step(
        _transition(1, ball=(1000.0, 2500.0, 17.0))
    )
    assert moved_ball.directed_velocity_uu_per_second == 0.0
    assert moved_ball.velocity_to_ball_reward == 0.0


def test_first_touch_pays_once_and_requests_immediate_termination() -> None:
    kernel = RivalFirstTouchVelocityRewardKernelV1()
    kernel.reset(_transition(0))
    first = kernel.step(_transition(1, touches=1))
    sustained = kernel.step(_transition(2, touches=1))
    kernel.step(_transition(3, touches=0))
    later = kernel.step(_transition(4, touches=1))
    assert first.physical_touch_reward == FIRST_PHYSICAL_TOUCH_REWARD
    assert first.episode_should_terminate
    assert sustained.physical_touch_reward == 0.0
    assert not sustained.episode_should_terminate
    assert later.new_physical_touch
    assert later.physical_touch_reward == 0.0
    assert kernel.touch_total == FIRST_PHYSICAL_TOUCH_REWARD


def test_every_nonminimal_reward_component_is_zero() -> None:
    kernel = RivalFirstTouchVelocityRewardKernelV1()
    kernel.reset(_transition(0))
    step = kernel.step(_transition(1, velocity=(1000.0, 0.0, 0.0)))
    allowed = {"velocity_to_ball", "physical_new_touch"}
    assert all(
        step.components[name] == 0.0 for name in COMPONENTS if name not in allowed
    )


def test_first_touch_done_condition_uses_active_learner_raw_contact() -> None:
    raw = build_first_touch_velocity_env(
        phase="A",
        seed=202610103,
        forced_family="stationary_close",
        forced_active_team=0,
        forced_mirror=False,
    )
    wrapper = RivalSingleLearnerFirstTouchWrapperV1(raw)
    try:
        wrapper.reset()
        assert wrapper.active_agent is not None
        condition = RivalActiveLearnerFirstTouchConditionV1()
        agents = list(raw.state.cars)
        condition.reset(agents, raw.state, raw.shared_info)
        before = condition.is_done(agents, raw.state, raw.shared_info)
        assert not any(before.values())
        raw.state.cars[wrapper.active_agent].ball_touches = 1
        contact = condition.is_done(agents, raw.state, raw.shared_info)
        assert all(contact.values())
        sustained = condition.is_done(agents, raw.state, raw.shared_info)
        assert not any(sustained.values())
    finally:
        wrapper.close()


def test_environment_keeps_native_action_and_one_learner_contract() -> None:
    env = RivalSingleLearnerFirstTouchWrapperV1(
        build_first_touch_velocity_env(
            phase="A",
            seed=202610104,
            forced_family="stationary_close",
            forced_active_team=0,
        )
    )
    try:
        observation = env.reset()
        assert observation.shape == (1, 714)
        next_observation, rewards, _, _, info = env.step(
            np.zeros((1, 8), dtype=np.float32)
        )
        assert next_observation.shape == (1, 714)
        assert len(rewards) == 1
        assert info["rival_v10_2"]["dummy_rows_returned"] == 0
        assert env.dummy_nonzero_actions_injected == 0
        assert env.rlgym_env.transition_engine._rlbot_delay  # noqa: SLF001
        components = env.rlgym_env.shared_info["reward_components"][env.active_agent]
        assert set(components) == set(COMPONENTS)
        assert math.isfinite(float(rewards[0]))
    finally:
        env.close()


def test_config_freezes_m10_9_ppo_action_and_clean_source() -> None:
    config = load_stage1_config()
    assert config["ppo"]["gamma"] == 0.9987444968227265
    assert config["ppo"]["gae_lambda"] == 0.9983695094257663
    assert config["time_base"]["physics_hz"] == 120
    assert config["time_base"]["repeat_action"] is False
    assert config["time_base"]["one_tick_action_delay"] is True
    assert config["action_version"] == "RivalActionV1"
    assert config["stage_contract"]["stage_2_authorized"] is False
    assert config["stage_contract"]["production_promotion_authorized"] is False
    initialized = clean_initialization(config, device="cpu")
    proof = initialized["proof"]
    assert proof["checks"]["passed"]
    assert proof["actor_state_sha256"] == PAIRED_ACTOR_STATE_SHA256
    assert proof["critic_state_sha256"] == PAIRED_CRITIC_STATE_SHA256
    assert proof["m10_7_through_m10_9_trained_actor_used"] is False
