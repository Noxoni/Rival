"""Contract tests for the Stage-1-only V3 reward restart."""

from __future__ import annotations

import numpy as np
import pytest

from rival_training.v10_4_campaign import (
    SOURCE_ACTOR_SHA256,
    SUCCESS_DECISION,
    load_stage1_config,
)
from rival_training.v10_4_environment import (
    RivalSingleLearnerGymWrapperV3,
    build_ball_acquisition_env,
)
from rival_training.v10_4_reward import (
    AWAY_BALL_EPISODE_BUDGET,
    IDLE_GRACE_TICKS,
    IDLE_VIOLATION_PENALTY,
    PHYSICAL_NEW_TOUCH_REWARD,
    TOWARD_BALL_EPISODE_BUDGET,
    BallAcquisitionTransitionV3,
    RivalBallAcquisitionRewardKernelV3,
    reward_truth_table_v3,
)


def _transition(
    tick: int,
    car_x: float,
    *,
    ball_x: float = 5000.0,
    speed: float = 100.0,
    touches: int = 0,
) -> BallAcquisitionTransitionV3:
    return BallAcquisitionTransitionV3(
        tick=tick,
        car_position=np.asarray([car_x, 0.0, 17.0]),
        ball_position=np.asarray([ball_x, 0.0, 93.0]),
        car_linear_velocity=np.asarray([speed, 0.0, 0.0]),
        raw_touch_records=touches,
    )


def test_v3_reward_truth_table_is_exact() -> None:
    report = reward_truth_table_v3()
    assert report["checks"]["passed"]
    assert report["metadata"]["physical_new_touch_reward"] == 1.0
    assert report["metadata"]["toward_ball_episode_budget"] == 0.75
    assert report["metadata"]["away_ball_episode_budget"] == -0.75
    assert report["metadata"]["idle_violation_penalty"] == -0.80


def test_directional_progress_budgets_are_independent() -> None:
    kernel = RivalBallAcquisitionRewardKernelV3(safety_clip_uu=2300.0)
    kernel.reset(_transition(0, 0.0))
    kernel.step(_transition(1, 1000.0))
    toward = kernel.step(_transition(2, 2000.0))
    kernel.step(_transition(3, 1000.0))
    away = kernel.step(_transition(4, 0.0))
    assert kernel.toward_ball_spend == pytest.approx(TOWARD_BALL_EPISODE_BUDGET)
    assert kernel.away_ball_spend == pytest.approx(abs(AWAY_BALL_EPISODE_BUDGET))
    assert toward.toward_ball_budget_saturated
    assert away.away_ball_budget_saturated
    assert kernel.distance_total == pytest.approx(0.0)


def test_first_idle_tick_after_grace_immediately_pays_once() -> None:
    kernel = RivalBallAcquisitionRewardKernelV3()
    kernel.reset(_transition(0, 0.0, speed=0.0))
    grace = kernel.step(_transition(IDLE_GRACE_TICKS, 0.0, speed=0.0))
    violation = kernel.step(
        _transition(IDLE_GRACE_TICKS + 1, 0.0, speed=0.0)
    )
    repeated = kernel.step(
        _transition(IDLE_GRACE_TICKS + 2, 0.0, speed=0.0)
    )
    assert grace.idle_penalty == 0.0
    assert violation.idle_penalty == IDLE_VIOLATION_PENALTY == -0.80
    assert violation.idle_violation_triggered
    assert repeated.idle_penalty == 0.0
    assert kernel.idle_penalty_total == -0.80


def test_touch_on_first_post_grace_tick_avoids_idle_violation() -> None:
    kernel = RivalBallAcquisitionRewardKernelV3()
    kernel.reset(_transition(0, 0.0, speed=0.0))
    touch = kernel.step(
        _transition(IDLE_GRACE_TICKS + 1, 0.0, speed=0.0, touches=1)
    )
    assert touch.components["physical_new_touch"] == PHYSICAL_NEW_TOUCH_REWARD == 1.0
    assert touch.components["pre_touch_idle"] == 0.0
    assert not touch.idle_violation_triggered


def test_v3_environment_keeps_frozen_native_contract() -> None:
    env = RivalSingleLearnerGymWrapperV3(
        build_ball_acquisition_env(
            phase="A",
            seed=20261043,
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
        assert env.rlgym_env.shared_info["rival_v10_4_reward_version"] == (
            "RivalBallAcquisitionRewardV3"
        )
    finally:
        env.close()


def test_v10_4_config_is_stage1_only_and_restarts_exact_source() -> None:
    config = load_stage1_config()
    contract = config["stage_contract"]
    assert config["scope"] == "stage_1_only"
    assert config["time_base"]["physics_hz"] == 120
    assert config["time_base"]["repeat_action"] is False
    assert config["time_base"]["one_tick_action_delay"] is True
    assert contract["source_actor_sha256"] == SOURCE_ACTOR_SHA256
    assert contract["fresh_critic"] is True
    assert contract["fresh_actor_optimizer"] is True
    assert contract["fresh_critic_optimizer"] is True
    assert contract["success_decision"] == SUCCESS_DECISION
    assert contract["stage_2_authorized"] is False
    assert contract["production_promotion_authorized"] is False
