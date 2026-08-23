from __future__ import annotations

import math

import numpy as np

from rival_training.v9_actions import ACTION_DIM
from rival_training.v9_environment import build_v9_training_env
from rival_training.v9_rewards import (
    COMPONENTS,
    GAMMA_120HZ,
    GOAL_REWARD,
    SHAPING_ABSOLUTE_EPISODE_BUDGETS,
    RewardEventsV1,
    RewardStateV1,
    RivalRewardKernelV1,
    reward_metadata,
    select_reward_phase,
)


def _state(
    tick: int,
    *,
    car_y: float = -1000.0,
    car_speed_y: float = 600.0,
    ball_y: float = 0.0,
    ball_speed_y: float = 300.0,
    boosting: bool = False,
    supersonic: bool = False,
    can_dodge: bool = False,
    surface: bool = True,
) -> RewardStateV1:
    return RewardStateV1(
        tick_index=tick,
        self_position=np.asarray([100.0, car_y, 17.0]),
        self_linear_velocity=np.asarray([0.0, car_speed_y, 0.0]),
        self_forward=np.asarray([0.0, 1.0, 0.0]),
        self_up=np.asarray([0.0, 0.0, 1.0]),
        self_boost=40.0,
        self_surface_contact=surface,
        self_boosting=boosting,
        self_supersonic=supersonic,
        self_can_dodge=can_dodge,
        ball_position=np.asarray([0.0, ball_y, 92.75]),
        ball_linear_velocity=np.asarray([0.0, ball_speed_y, 0.0]),
    )


def _integrate(period: int, *, goal_for: bool = False, goal_against: bool = False):
    phase = select_reward_phase(0.0)
    kernel = RivalRewardKernelV1()
    kernel.reset(_state(0))
    totals = {name: 0.0 for name in COMPONENTS}
    for tick in range(period, 241, period):
        fraction = tick / 240.0
        step = kernel.step(
            _state(
                tick,
                car_y=-1000.0 + 500.0 * fraction,
                car_speed_y=600.0 + 200.0 * fraction,
                ball_y=900.0 * fraction,
                ball_speed_y=300.0 + 500.0 * fraction,
            ),
            RewardEventsV1(
                goal_for=goal_for and tick == 240,
                goal_against=goal_against and tick == 240,
                self_touch=tick == 120,
            ),
            phase,
        )
        for name, value in step.components.items():
            totals[name] += value
    return totals


def test_reward_schedule_needs_physical_time_and_readiness() -> None:
    assert select_reward_phase(0.0).name == "foundation"
    assert select_reward_phase(100.0).name == "foundation"
    assert select_reward_phase(25.0, competence_ready=True).name == "competence"
    assert (
        select_reward_phase(249.0, competence_ready=True, mature_ready=True).name
        == "competence"
    )
    assert select_reward_phase(250.0, mature_ready=True).name == "mature"


def test_outcome_events_are_exactly_cadence_invariant() -> None:
    for goal_for, expected in ((True, GOAL_REWARD), (False, -GOAL_REWARD)):
        outcomes = []
        for period in (1, 2, 4):
            totals = _integrate(
                period,
                goal_for=goal_for,
                goal_against=not goal_for,
            )
            outcomes.append(totals["outcome"])
        assert outcomes == [expected, expected, expected]


def test_potential_integration_is_comparable_at_one_two_and_four_ticks() -> None:
    by_period = {period: _integrate(period) for period in (1, 2, 4)}
    for component in (
        "ball_progress_potential",
        "approach_control_potential",
        "recovery_potential",
    ):
        values = [by_period[period][component] for period in (1, 2, 4)]
        scale = max(max(abs(value) for value in values), 1e-9)
        assert (max(values) - min(values)) / scale < 0.08


def test_rate_term_integrates_by_physical_time_not_step_count() -> None:
    phase = select_reward_phase(0.0)
    totals = []
    for period in (1, 2, 4):
        kernel = RivalRewardKernelV1()
        kernel.reset(
            _state(0, car_speed_y=2300.0, boosting=True, supersonic=True)
        )
        total = 0.0
        for tick in range(period, 121, period):
            step = kernel.step(
                _state(
                    tick,
                    car_speed_y=2300.0,
                    boosting=True,
                    supersonic=True,
                ),
                RewardEventsV1(),
                phase,
            )
            total += step.components["boost_waste_rate"]
        totals.append(total)
    np.testing.assert_allclose(totals, [-0.015, -0.015, -0.015], atol=1e-12)


def test_team_inversion_produces_symmetric_reward() -> None:
    inversion = np.asarray([-1.0, -1.0, 1.0])
    blue_previous = _state(0, car_y=-1200.0, ball_y=100.0)
    blue_current = _state(1, car_y=-1190.0, ball_y=110.0, ball_speed_y=450.0)

    def invert_twice(state: RewardStateV1) -> RewardStateV1:
        # Construct an orange world-frame mirror, then apply the canonical
        # orange transform.  The resulting role-relative state must match blue.
        world_car = state.self_position * inversion
        world_velocity = state.self_linear_velocity * inversion
        world_forward = state.self_forward * inversion
        world_up = state.self_up * inversion
        world_ball = state.ball_position * inversion
        world_ball_velocity = state.ball_linear_velocity * inversion
        return RewardStateV1(
            tick_index=state.tick_index,
            self_position=world_car * inversion,
            self_linear_velocity=world_velocity * inversion,
            self_forward=world_forward * inversion,
            self_up=world_up * inversion,
            self_boost=state.self_boost,
            self_surface_contact=state.self_surface_contact,
            self_boosting=state.self_boosting,
            self_supersonic=state.self_supersonic,
            self_can_dodge=state.self_can_dodge,
            ball_position=world_ball * inversion,
            ball_linear_velocity=world_ball_velocity * inversion,
        )

    phase = select_reward_phase(0.0)
    blue = RivalRewardKernelV1()
    orange = RivalRewardKernelV1()
    blue.reset(blue_previous)
    orange.reset(invert_twice(blue_previous))
    blue_step = blue.step(blue_current, RewardEventsV1(self_touch=True), phase)
    orange_step = orange.step(
        invert_twice(blue_current), RewardEventsV1(self_touch=True), phase
    )
    assert blue_step.components == orange_step.components


def test_shaping_budgets_are_explicit_and_subordinate_to_goal() -> None:
    metadata = reward_metadata()
    assert metadata["combined_shaping_absolute_episode_budget"] < GOAL_REWARD
    assert sum(SHAPING_ABSOLUTE_EPISODE_BUDGETS.values()) == 8.75
    assert metadata["named_mechanic_identity_rewards"] is False
    assert math.isclose(metadata["gamma_120hz"], GAMMA_120HZ)


def test_complete_training_environment_logs_finite_components() -> None:
    environment = build_v9_training_env(prediction_refresh_ticks=1)
    try:
        observations = environment.reset()
        actions = {
            agent: np.zeros(ACTION_DIM, dtype=np.float32)
            for agent in environment.agents
        }
        observations, rewards, terminated, truncated = environment.step(actions)
        assert not any(terminated.values())
        assert not any(truncated.values())
        assert all(np.isfinite(observation).all() for observation in observations.values())
        assert all(math.isfinite(value) for value in rewards.values())
        assert environment.shared_info["rival_v9_reward_phase"] == "foundation"
        for components in environment.shared_info["reward_components"].values():
            assert tuple(components) == COMPONENTS
            assert all(math.isfinite(value) for value in components.values())
    finally:
        environment.close()
