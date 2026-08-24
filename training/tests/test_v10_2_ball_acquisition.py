"""Contract tests for Rival v10.2 Stage-1 ball acquisition."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator

from rival_training.v10_2_campaign import (
    boundary_ppo_batch_agent_steps,
    load_stage1_config,
)
from rival_training.v10_2_curriculum import (
    FAMILIES,
    RivalBallAcquisitionCurriculumV1,
    _legal_state,
)
from rival_training.v10_2_environment import (
    RivalSingleLearnerGymWrapperV1,
    build_ball_acquisition_env,
)
from rival_training.v10_2_reward import (
    DISTANCE_PROGRESS_ABSOLUTE_EPISODE_BUDGET,
    PHYSICAL_NEW_TOUCH_REWARD,
    BallAcquisitionTransitionV1,
    RivalBallAcquisitionRewardKernelV1,
    RivalNewContactDetectorV1,
)


def _transition(
    tick: int,
    car_x: float,
    ball_x: float,
    *,
    touches: int = 0,
    goal_for: bool = False,
    goal_against: bool = False,
) -> BallAcquisitionTransitionV1:
    return BallAcquisitionTransitionV1(
        tick=tick,
        car_position=np.asarray([car_x, 0.0, 17.0]),
        ball_position=np.asarray([ball_x, 0.0, 93.0]),
        raw_touch_records=touches,
        goal_for=goal_for,
        goal_against=goal_against,
    )


def test_boundary_batch_uses_smaller_final_update_without_zeroing_it() -> None:
    assert boundary_ppo_batch_agent_steps(96_000, 48_000) == 96_000
    assert boundary_ppo_batch_agent_steps(71_891, 48_000) == 48_000
    assert boundary_ppo_batch_agent_steps(23_893, 48_000) == 23_893
    with pytest.raises(ValueError):
        boundary_ppo_batch_agent_steps(0, 48_000)


def test_car_caused_progress_sign_and_moving_ball_invariance() -> None:
    kernel = RivalBallAcquisitionRewardKernelV1()
    kernel.reset(_transition(0, 0.0, 1000.0))
    toward = kernel.step(_transition(1, 10.0, 1000.0))
    away = kernel.step(_transition(2, 0.0, 1000.0))
    assert toward.components["distance_progress"] > 0.0
    assert away.components["distance_progress"] < 0.0
    assert toward.components["distance_progress"] == pytest.approx(
        -away.components["distance_progress"]
    )

    stationary = RivalBallAcquisitionRewardKernelV1()
    stationary.reset(_transition(0, 0.0, 1000.0))
    moving_toward = stationary.step(_transition(1, 0.0, 500.0))
    moving_away = stationary.step(_transition(2, 0.0, 1500.0))
    assert moving_toward.components["distance_progress"] == pytest.approx(0.0)
    assert moving_away.components["distance_progress"] == pytest.approx(0.0)


def test_closed_approach_retreat_cycle_has_no_free_dense_return() -> None:
    kernel = RivalBallAcquisitionRewardKernelV1()
    kernel.reset(_transition(0, 0.0, 1000.0))
    approach = kernel.step(_transition(1, 10.0, 1000.0))
    retreat = kernel.step(_transition(2, 0.0, 1000.0))
    assert (
        approach.components["distance_progress"]
        + retreat.components["distance_progress"]
    ) == pytest.approx(0.0, abs=1e-12)


def test_dense_budget_and_unbudgeted_touch_events_are_exact() -> None:
    kernel = RivalBallAcquisitionRewardKernelV1(safety_clip_uu=1000.0)
    kernel.reset(_transition(0, 0.0, 5000.0))
    steps = []
    for tick in range(1, 10):
        steps.append(kernel.step(_transition(tick, tick * 500.0, 5000.0)))
    assert kernel.distance_absolute_spend == pytest.approx(
        DISTANCE_PROGRESS_ABSOLUTE_EPISODE_BUDGET
    )
    assert steps[-1].distance_budget_saturated

    detector = RivalNewContactDetectorV1()
    events = [detector.process(value) for value in (1, 1, 1, 0, 1, 0, 1)]
    assert events == [True, False, False, False, True, False, True]
    assert sum(events) * PHYSICAL_NEW_TOUCH_REWARD == 3.0


def test_touch_values_ignore_ground_aerial_and_goals_are_neutral() -> None:
    kernel = RivalBallAcquisitionRewardKernelV1()
    kernel.reset(_transition(0, 0.0, 1000.0))
    goal_touch = kernel.step(
        _transition(1, 0.0, 1000.0, touches=1, goal_for=True)
    )
    assert goal_touch.components["physical_new_touch"] == 1.0
    assert goal_touch.components["goal_for"] == 0.0
    assert goal_touch.components["goal_against"] == 0.0
    assert goal_touch.total == 1.0

    opponent_goal = RivalBallAcquisitionRewardKernelV1()
    opponent_goal.reset(_transition(0, 0.0, 1000.0))
    concede = opponent_goal.step(
        _transition(1, 0.0, 1000.0, goal_against=True)
    )
    assert concede.total == 0.0


def test_reward_kernel_has_no_controller_action_input() -> None:
    signature = inspect.signature(RivalBallAcquisitionRewardKernelV1.step)
    assert tuple(signature.parameters) == ("self", "transition")


@pytest.mark.parametrize("family", FAMILIES)
def test_every_stage1_family_generates_legal_state(family: str) -> None:
    engine = RocketSimEngine(rlbot_delay=True)
    team_size = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    mutator = RivalBallAcquisitionCurriculumV1(
        "A", seed=20261021, forced_family=family
    )
    shared: dict = {}
    for _ in range(128):
        state = engine.create_base_state()
        team_size.apply(state, shared)
        mutator.apply(state, shared)
        assert _legal_state(state)
        assert shared["rival_v10_2_reset_family"] == family
        assert shared["rival_v10_2_active_team"] in (0, 1)


def test_single_learner_wrapper_returns_no_dummy_rows_and_zero_dummy_action() -> None:
    wrapper = RivalSingleLearnerGymWrapperV1(
        build_ball_acquisition_env(
            forced_family="stationary_close",
            forced_active_team=0,
            forced_mirror=False,
        )
    )
    try:
        observation = wrapper.reset()
        assert observation.shape == (1, 714)
        next_observation, rewards, done, truncated, info = wrapper.step(
            np.zeros((1, 8), dtype=np.float32)
        )
        assert next_observation.shape == (1, 714)
        assert len(rewards) == 1
        assert isinstance(done, bool)
        assert isinstance(truncated, bool)
        assert info["rival_v10_2"]["dummy_action"] == [0.0] * 8
        assert info["rival_v10_2"]["dummy_rows_returned"] == 0
        assert wrapper.dummy_actions_injected == 1
        assert wrapper.dummy_nonzero_actions_injected == 0
    finally:
        wrapper.close()


def test_stage1_config_freezes_authority() -> None:
    config = load_stage1_config()
    assert config["backend"]["worker_count"] == 56
    assert config["ppo"]["rollout_agent_steps_per_iteration"] == 96_000
    assert config["ppo"]["ppo_batch_agent_steps"] == 96_000
    assert config["ppo"]["minibatch_agent_steps"] == 24_000
    assert config["time_base"]["trainable_agents_per_environment"] == 1
    assert config["stage_contract"]["maximum_active_learner_steps"] == 6_480_000
    assert config["stage_contract"]["production_promotion_authorized"] is False
