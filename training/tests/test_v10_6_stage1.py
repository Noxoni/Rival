"""Contract tests for the Stage-1 uncapped three-contact experiment."""

from __future__ import annotations

import math

import numpy as np
import pytest

from rival_training.v10_6_campaign import (
    SOURCE_ACTOR_SHA256,
    TERMINAL_DECISION,
    build_stage1_corpus_manifests,
    load_stage1_config,
)
from rival_training.v10_6_environment import (
    RivalSingleLearnerGymWrapperV5,
    build_ball_acquisition_env,
)
from rival_training.v10_6_reward import (
    ACQUISITION_GRACE_TICKS,
    ACQUISITION_TIME_PENALTY_RATE_PER_SECOND,
    FAILED_ACQUISITION_WINDOW_PENALTY,
    NO_TOUCH_TIMEOUT_SECONDS,
    PHYSICAL_CONTACT_REWARD,
    PHYSICS_HZ,
    BallAcquisitionTransitionV5,
    RivalBallAcquisitionRewardKernelV5,
    reward_truth_table_v5,
)


def _transition(
    tick: int,
    car_x: float = 0.0,
    *,
    ball_x: float = 5000.0,
    forward_x: float = 1.0,
    speed: float = 0.0,
    touches: int = 0,
) -> BallAcquisitionTransitionV5:
    return BallAcquisitionTransitionV5(
        tick=tick,
        car_position=np.asarray([car_x, 0.0, 17.0]),
        ball_position=np.asarray([ball_x, 0.0, 17.0]),
        car_forward=np.asarray([forward_x, 0.0, 0.0]),
        car_linear_velocity=np.asarray([speed, 0.0, 0.0]),
        raw_touch_records=touches,
    )


def _separate(kernel: RivalBallAcquisitionRewardKernelV5, tick: int) -> None:
    kernel.step(_transition(tick, touches=0))


def test_v5_reward_truth_table_proves_all_23_user_requirements() -> None:
    report = reward_truth_table_v5()
    checks = {key: value for key, value in report["checks"].items() if key != "passed"}
    assert len(checks) == 23
    assert report["checks"]["passed"]
    assert all(checks.values())
    assert report["metadata"]["failed_acquisition_window_penalty"] == pytest.approx(-16.1)
    assert report["metadata"]["maximum_contact_reward_per_episode"] == 30.0


def test_failed_window_is_speed_independent_negative_16p1_after_grace() -> None:
    results = []
    for speed in (0.0, 81.0, 2300.0):
        kernel = RivalBallAcquisitionRewardKernelV5()
        kernel.reset(_transition(0, speed=speed))
        grace = kernel.step(_transition(ACQUISITION_GRACE_TICKS, speed=speed))
        final = kernel.step(_transition(int(NO_TOUCH_TIMEOUT_SECONDS * PHYSICS_HZ), speed=speed))
        assert grace.acquisition_time_penalty == 0.0
        assert final.acquisition_time_penalty_ticks == 1380
        assert final.acquisition_time_penalty_seconds == 11.5
        results.append(final.cumulative_acquisition_time_penalty)
    assert results == pytest.approx([FAILED_ACQUISITION_WINDOW_PENALTY] * 3)


def test_each_rewarded_contact_stops_penalty_and_restarts_grace_then_third_stops_all() -> None:
    kernel = RivalBallAcquisitionRewardKernelV5()
    kernel.reset(_transition(0))
    first = kernel.step(_transition(60, touches=1))
    assert first.components["physical_new_touch"] == PHYSICAL_CONTACT_REWARD
    assert first.acquisition_time_penalty == 0.0
    _separate(kernel, 61)
    first_grace = kernel.step(_transition(120))
    first_eligible = kernel.step(_transition(121))
    assert first_grace.acquisition_time_penalty == 0.0
    assert first_eligible.acquisition_time_penalty == pytest.approx(
        ACQUISITION_TIME_PENALTY_RATE_PER_SECOND / PHYSICS_HZ
    )
    second = kernel.step(_transition(180, touches=1))
    assert second.components["physical_new_touch"] == PHYSICAL_CONTACT_REWARD
    _separate(kernel, 181)
    second_grace = kernel.step(_transition(240))
    second_eligible = kernel.step(_transition(241))
    assert second_grace.acquisition_time_penalty == 0.0
    assert second_eligible.acquisition_time_penalty < 0.0
    third = kernel.step(_transition(300, touches=1))
    assert third.components["physical_new_touch"] == PHYSICAL_CONTACT_REWARD
    _separate(kernel, 301)
    after_third = kernel.step(_transition(2000))
    fourth = kernel.step(_transition(2001, touches=1))
    assert after_third.acquisition_time_penalty == 0.0
    assert fourth.components["physical_new_touch"] == 0.0
    assert fourth.acquisition_time_penalty == 0.0
    assert kernel.touch_total == 30.0


def test_sustained_contact_counts_once_and_three_separated_contacts_pay() -> None:
    kernel = RivalBallAcquisitionRewardKernelV5()
    kernel.reset(_transition(0))
    rewards = []
    for tick, touches in ((1, 1), (2, 1), (3, 1), (4, 0), (5, 1), (6, 0), (7, 1)):
        rewards.append(kernel.step(_transition(tick, touches=touches)))
    assert [step.components["physical_new_touch"] for step in rewards] == [
        10.0,
        0.0,
        0.0,
        0.0,
        10.0,
        0.0,
        10.0,
    ]
    assert kernel.physical_contact_count == 3
    assert kernel.rewarded_contact_count == 3


def test_heading_delta_is_uncapped_and_round_trips_cancel() -> None:
    kernel = RivalBallAcquisitionRewardKernelV5()
    kernel.reset(_transition(0, forward_x=-1.0))
    toward = kernel.step(_transition(1, forward_x=1.0))
    held = kernel.step(_transition(2, forward_x=1.0))
    away = kernel.step(_transition(3, forward_x=-1.0))
    second_toward = kernel.step(_transition(4, forward_x=1.0))
    second_away = kernel.step(_transition(5, forward_x=-1.0))
    assert toward.heading_reward == 3.0
    assert held.heading_reward == 0.0
    assert away.heading_reward == -3.0
    assert toward.heading_reward + away.heading_reward == pytest.approx(0.0)
    assert kernel.heading_positive_total == 6.0
    assert kernel.heading_negative_total == -6.0
    assert second_toward.heading_reward == 3.0
    assert second_away.heading_reward == -3.0
    assert not hasattr(kernel, "heading_positive_spend")


def test_distance_progress_is_uncapped_cancels_and_rejects_ball_only_motion() -> None:
    kernel = RivalBallAcquisitionRewardKernelV5()
    kernel.reset(_transition(0, car_x=0.0))
    approach = kernel.step(_transition(1, car_x=50.0))
    reverse = kernel.step(_transition(2, car_x=0.0))
    assert approach.distance_reward > 0.0
    assert reverse.distance_reward < 0.0
    assert approach.distance_reward + reverse.distance_reward == pytest.approx(0.0)

    retreat = RivalBallAcquisitionRewardKernelV5()
    retreat.reset(_transition(0, car_x=0.0))
    trajectory = [retreat.step(_transition(tick, car_x=-64.0 * tick)) for tick in range(1, 121)]
    assert all(step.distance_reward < 0.0 for step in trajectory)
    assert retreat.distance_negative_total < -3.0
    assert not hasattr(retreat, "approach_negative_spend")

    ball_only = RivalBallAcquisitionRewardKernelV5()
    ball_only.reset(_transition(0, car_x=0.0, ball_x=1000.0))
    moved_ball = ball_only.step(_transition(1, car_x=0.0, ball_x=500.0))
    assert moved_ball.components["distance_progress"] == 0.0


def test_all_other_reward_components_are_zero() -> None:
    kernel = RivalBallAcquisitionRewardKernelV5()
    kernel.reset(_transition(0))
    step = kernel.step(_transition(1))
    assert step.components["goal_for"] == 0.0
    assert step.components["goal_against"] == 0.0
    config = load_stage1_config()
    for name in (
        "goal_reward",
        "concede_reward",
        "generic_speed_reward",
        "boost_reward",
        "throttle_reward",
        "steer_reward",
        "action_magnitude_reward",
        "jump_reward",
        "handbrake_reward",
        "named_mechanic_reward",
        "possession_reward",
        "aerial_reward",
        "recovery_reward",
    ):
        assert config["reward_contract"][name] == 0.0


def test_v5_environment_keeps_frozen_native_contract() -> None:
    env = RivalSingleLearnerGymWrapperV5(
        build_ball_acquisition_env(
            phase="A",
            seed=20261063,
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
        assert shared["rival_v10_6_reward_version"] == "RivalBallAcquisitionRewardV5"
        assert math.isfinite(shared["rival_v10_6_reward_metrics"]["alignment_now"])
    finally:
        env.close()


def test_v10_6_config_and_frozen_corpus_are_exact() -> None:
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
    corpora = build_stage1_corpus_manifests()
    assert corpora["checks"]["passed"]
    assert corpora["checks"]["m10_5_corpora_reused_without_rewrite"]
