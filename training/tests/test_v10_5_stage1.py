"""Contract tests for the Stage-1 turn/approach/first-touch experiment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rival_training.v10_5_campaign import (
    SOURCE_ACTOR_SHA256,
    TERMINAL_DECISION,
    load_stage1_config,
)
from rival_training.v10_5_environment import (
    RivalSingleLearnerGymWrapperV4,
    build_ball_acquisition_env,
)
from rival_training.v10_5_reward import (
    APPROACH_NEGATIVE_EPISODE_BUDGET,
    APPROACH_POSITIVE_EPISODE_BUDGET,
    FIRST_PHYSICAL_TOUCH_REWARD,
    HEADING_NEGATIVE_EPISODE_BUDGET,
    HEADING_POSITIVE_EPISODE_BUDGET,
    IDLE_EPISODE_FLOOR,
    IDLE_GRACE_TICKS,
    IDLE_PENALTY_RATE_PER_SIMULATED_SECOND,
    MAXIMUM_POSITIVE_EPISODE_REWARD,
    NO_TOUCH_TIMEOUT_SECONDS,
    PHYSICS_HZ,
    BallAcquisitionTransitionV4,
    RivalBallAcquisitionRewardKernelV4,
    reward_truth_table_v4,
)


def _transition(
    tick: int,
    car_x: float,
    *,
    ball_x: float = 5000.0,
    forward_x: float = 1.0,
    speed: float = 100.0,
    touches: int = 0,
) -> BallAcquisitionTransitionV4:
    return BallAcquisitionTransitionV4(
        tick=tick,
        car_position=np.asarray([car_x, 0.0, 17.0]),
        ball_position=np.asarray([ball_x, 0.0, 17.0]),
        car_forward=np.asarray([forward_x, 0.0, 0.0]),
        car_linear_velocity=np.asarray([speed, 0.0, 0.0]),
        raw_touch_records=touches,
    )


def test_v4_reward_truth_table_proves_all_user_requirements() -> None:
    report = reward_truth_table_v4()
    assert report["checks"]["passed"]
    assert all(report["checks"].values())
    assert report["positive_total"] == MAXIMUM_POSITIVE_EPISODE_REWARD == 16.0
    assert report["metadata"]["full_idle_penalty"] == pytest.approx(-16.1)
    assert abs(report["metadata"]["full_idle_penalty"]) > report["positive_total"]


def test_idle_rate_starts_after_grace_and_totals_negative_16p1() -> None:
    kernel = RivalBallAcquisitionRewardKernelV4()
    kernel.reset(_transition(0, 0.0, speed=0.0))
    grace = kernel.step(_transition(IDLE_GRACE_TICKS, 0.0, speed=0.0))
    first = kernel.step(_transition(IDLE_GRACE_TICKS + 1, 0.0, speed=0.0))
    final = kernel.step(_transition(int(NO_TOUCH_TIMEOUT_SECONDS * PHYSICS_HZ), 0.0, speed=0.0))
    assert grace.idle_penalty == 0.0
    assert first.idle_penalty == pytest.approx(IDLE_PENALTY_RATE_PER_SIMULATED_SECOND / PHYSICS_HZ)
    assert final.cumulative_idle_penalty == pytest.approx(IDLE_EPISODE_FLOOR)
    assert final.cumulative_idle_seconds == pytest.approx(11.5)
    assert final.cumulative_idle_ticks == 1380


def test_heading_rewards_change_only_with_independent_caps() -> None:
    kernel = RivalBallAcquisitionRewardKernelV4()
    kernel.reset(_transition(0, 0.0, forward_x=-1.0))
    toward = kernel.step(_transition(1, 0.0, forward_x=1.0))
    held = kernel.step(_transition(2, 0.0, forward_x=1.0))
    away = kernel.step(_transition(3, 0.0, forward_x=-1.0))
    assert toward.components["heading_alignment"] == 3.0
    assert held.components["heading_alignment"] == 0.0
    assert away.components["heading_alignment"] == -3.0
    assert kernel.heading_positive_spend == HEADING_POSITIVE_EPISODE_BUDGET
    assert kernel.heading_negative_spend == abs(HEADING_NEGATIVE_EPISODE_BUDGET)


def test_distance_rewards_car_progress_not_ball_motion_with_independent_caps() -> None:
    kernel = RivalBallAcquisitionRewardKernelV4(safety_clip_uu=2300.0)
    kernel.reset(_transition(0, 0.0, ball_x=20_000.0))
    for tick in range(1, 5):
        kernel.step(_transition(tick, tick * 2300.0, ball_x=20_000.0))
    for tick in range(5, 9):
        kernel.step(_transition(tick, (8 - tick) * 2300.0, ball_x=20_000.0))
    assert kernel.approach_positive_spend == APPROACH_POSITIVE_EPISODE_BUDGET
    assert kernel.approach_negative_spend == abs(APPROACH_NEGATIVE_EPISODE_BUDGET)
    assert kernel.distance_total == pytest.approx(0.0)

    ball_only = RivalBallAcquisitionRewardKernelV4()
    ball_only.reset(_transition(0, 0.0, ball_x=1000.0))
    moved_ball = ball_only.step(_transition(1, 0.0, ball_x=500.0))
    assert moved_ball.total == 0.0
    assert moved_ball.components["distance_progress"] == 0.0


def test_only_first_separated_physical_touch_pays_exactly_10() -> None:
    kernel = RivalBallAcquisitionRewardKernelV4()
    kernel.reset(_transition(0, 0.0))
    first = kernel.step(_transition(1, 0.0, touches=1))
    held = kernel.step(_transition(2, 0.0, touches=1))
    kernel.step(_transition(3, 0.0, touches=0))
    later = kernel.step(_transition(4, 0.0, touches=1))
    assert first.components["physical_new_touch"] == FIRST_PHYSICAL_TOUCH_REWARD
    assert held.components["physical_new_touch"] == 0.0
    assert later.new_physical_touch
    assert later.components["physical_new_touch"] == 0.0
    assert kernel.touch_total == 10.0
    assert kernel.touch_count == 2


def test_v4_environment_keeps_frozen_native_contract() -> None:
    env = RivalSingleLearnerGymWrapperV4(
        build_ball_acquisition_env(
            phase="A",
            seed=20261053,
            forced_family="stationary_close",
            forced_active_team=0,
        )
    )
    try:
        observation = env.reset()
        assert observation.shape == (1, 714)
        next_observation, rewards, _, _, info = env.step(np.zeros((1, 8), dtype=np.float32))
        assert next_observation.shape == (1, 714)
        assert len(rewards) == 1
        assert info["rival_v10_2"]["dummy_rows_returned"] == 0
        shared = env.rlgym_env.shared_info
        assert shared["rival_v10_5_reward_version"] == ("RivalBallAcquisitionRewardV4")
        assert math.isfinite(shared["rival_v10_5_reward_metrics"]["alignment_now"])
    finally:
        env.close()


def test_v10_5_config_is_exact_stage1_only_fresh_restart() -> None:
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
    assert contract["maximum_active_learner_steps"] == 2_160_000
    assert contract["evaluation_boundaries_added_simulated_hours"] == [1.0, 2.5, 5.0]
    assert contract["terminal_decision"] == TERMINAL_DECISION
    assert contract["stage_2_authorized"] is False
    assert contract["production_promotion_authorized"] is False
