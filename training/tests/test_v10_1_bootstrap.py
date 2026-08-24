from __future__ import annotations

import math

import numpy as np
import pytest
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator

from rival_training.v10_1_campaign import load_m10_1_config
from rival_training.v10_bootstrap_curriculum import (
    FAMILIES,
    PHASE_WEIGHTS,
    RivalAgencyBootstrapCurriculumV1,
    _legal_state,
)
from rival_training.v10_bootstrap_environment import build_v10_bootstrap_env
from rival_training.v10_bootstrap_reward import (
    AERIAL_TOUCH_BONUS,
    BALL_TOUCH_BASE_REWARD,
    COMBINED_SHAPING_ABSOLUTE_EPISODE_BUDGET,
    CONCEDE_REWARD,
    GOAL_REWARD,
    BootstrapRewardEventsV1,
    RivalAgencyBootstrapRewardKernelV1,
    RivalLogicalTouchAuditorV1,
    RewardStateV1,
    reward_metadata,
    touch_chain_bonus,
)
from rival_training.v9_rewards import REWARD_VERSION as NORMAL_REWARD_VERSION


def _state(
    tick: int,
    *,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ball: tuple[float, float, float] = (0.0, 1000.0, 92.75),
    surface_contact: bool = True,
) -> RewardStateV1:
    return RewardStateV1(
        tick_index=tick,
        self_position=np.asarray([0.0, 0.0, 17.0]),
        self_linear_velocity=np.asarray(velocity),
        self_forward=np.asarray([0.0, 1.0, 0.0]),
        self_up=np.asarray([0.0, 0.0, 1.0]),
        self_boost=50.0,
        self_surface_contact=surface_contact,
        self_boosting=False,
        self_supersonic=False,
        self_can_dodge=False,
        ball_position=np.asarray(ball),
        ball_linear_velocity=np.zeros(3),
    )


def test_stationary_unchanged_transition_has_zero_reward() -> None:
    kernel = RivalAgencyBootstrapRewardKernelV1()
    kernel.reset(_state(0))
    step = kernel.step(_state(1), BootstrapRewardEventsV1())
    assert step.total == pytest.approx(0.0, abs=1e-15)
    assert all(value == pytest.approx(0.0, abs=1e-15) for value in step.components.values())


def test_useful_speed_is_cadence_safe_and_directional() -> None:
    def transition(ticks: int, velocity: tuple[float, float, float]) -> float:
        kernel = RivalAgencyBootstrapRewardKernelV1()
        kernel.reset(_state(0, velocity=velocity))
        return kernel.step(
            _state(ticks, velocity=velocity), BootstrapRewardEventsV1()
        ).components["useful_speed_rate"]

    toward = transition(4, (0.0, 1200.0, 0.0))
    away = transition(4, (0.0, -1200.0, 0.0))
    stationary = transition(4, (0.0, 0.0, 0.0))
    assert toward > away > stationary
    assert transition(4, (0.0, 1200.0, 0.0)) == pytest.approx(
        4.0 * transition(1, (0.0, 1200.0, 0.0)), rel=1e-12
    )


def test_touch_aerial_and_chain_proposals_are_exact() -> None:
    kernel = RivalAgencyBootstrapRewardKernelV1()
    kernel.reset(_state(0, ball=(0.0, 1000.0, 300.0), surface_contact=False))
    first = kernel.step(
        _state(1, ball=(0.0, 1000.0, 300.0), surface_contact=False),
        BootstrapRewardEventsV1(
            raw_touch_records=3,
            logical_touch=True,
            aerial_touch=True,
            touch_chain_length=1,
        ),
    )
    assert first.proposals["ball_touch_event"] == BALL_TOUCH_BASE_REWARD
    assert first.proposals["aerial_touch_event"] == AERIAL_TOUCH_BONUS
    assert first.proposals["touch_chain_event"] == 0.0
    assert [touch_chain_bonus(index) for index in range(1, 6)] == [
        0.0,
        0.10,
        0.20,
        0.35,
        0.50,
    ]


def test_outcome_precedence_is_exact_and_unclipped() -> None:
    for event, expected in (
        (BootstrapRewardEventsV1(goal_for=True), GOAL_REWARD),
        (BootstrapRewardEventsV1(goal_against=True), CONCEDE_REWARD),
    ):
        kernel = RivalAgencyBootstrapRewardKernelV1()
        kernel.reset(_state(0, velocity=(0.0, 1800.0, 0.0)))
        step = kernel.step(_state(1, velocity=(0.0, 1800.0, 0.0)), event)
        assert step.total == expected
        assert all(step.components[name] == 0.0 for name in step.components if name != "outcome")


def test_combined_non_outcome_spend_never_exceeds_7_5() -> None:
    kernel = RivalAgencyBootstrapRewardKernelV1()
    kernel.reset(_state(0, velocity=(0.0, 2300.0, 0.0), surface_contact=False))
    for tick in range(1, 5000):
        kernel.step(
            _state(
                tick,
                velocity=(0.0, 2300.0, 0.0),
                ball=(0.0, 1000.0, 400.0),
                surface_contact=False,
            ),
            BootstrapRewardEventsV1(
                raw_touch_records=1,
                logical_touch=True,
                aerial_touch=True,
                touch_chain_length=5,
            ),
        )
    assert sum(kernel.absolute_spend.values()) <= (
        COMBINED_SHAPING_ABSOLUTE_EPISODE_BUDGET + 1e-12
    )
    assert kernel.absolute_spend["ball_touch_event"] == 2.0
    assert kernel.absolute_spend["aerial_touch_event"] == 1.5
    assert kernel.absolute_spend["touch_chain_event"] == 1.5


def test_touch_debounce_opponent_reset_and_chain_timeout() -> None:
    auditor = RivalLogicalTouchAuditorV1()
    agents = ["blue", "orange"]
    auditor.reset(agents)

    def step(tick: int, blue: int = 0, orange: int = 0):
        return auditor.process(
            agents,
            tick=tick,
            raw_touch_records={"blue": blue, "orange": orange},
            surface_contact={"blue": False, "orange": True},
            ball_z=300.0,
        )

    first = step(10, blue=4)
    assert first["blue"].logical_touch
    assert first["blue"].touch_chain_length == 1
    assert first["blue"].aerial_touch
    assert not step(11, blue=1)["blue"].logical_touch
    assert not step(17, blue=1)["blue"].logical_touch
    second = step(18, blue=1)
    assert second["blue"].touch_chain_length == 2
    opponent = step(19, orange=1)
    assert opponent["orange"].logical_touch
    after_opponent = step(20, blue=1)
    assert after_opponent["blue"].logical_touch
    assert after_opponent["blue"].touch_chain_length == 1
    after_gap = step(321, blue=1)
    assert after_gap["blue"].touch_chain_length == 1


def test_no_direct_action_or_removed_normal_reward_terms() -> None:
    metadata = reward_metadata()
    assert metadata["direct_action_press_rewards"] is False
    assert metadata["recovery_reward"] == 0.0
    assert metadata["boost_waste_penalty"] == 0.0
    assert metadata["dodge_resource_reward"] == 0.0
    assert NORMAL_REWARD_VERSION == "RivalScratchRewardV1"


@pytest.mark.parametrize("family", FAMILIES)
def test_every_curriculum_family_produces_finite_legal_state(family: str) -> None:
    engine = RocketSimEngine(rlbot_delay=True)
    teams = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    mutator = RivalAgencyBootstrapCurriculumV1(
        "A", seed=1000 + FAMILIES.index(family), forced_family=family
    )
    shared: dict[str, object] = {}
    for _ in range(64):
        state = engine.create_base_state()
        teams.apply(state, shared)
        mutator.apply(state, shared)
        assert shared["rival_v10_reset_family"] == family
        assert _legal_state(state)


def test_phase_weights_and_team_assignment_are_balanced() -> None:
    assert all(math.isclose(sum(weights.values()), 1.0) for weights in PHASE_WEIGHTS.values())
    engine = RocketSimEngine(rlbot_delay=True)
    teams = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    mutator = RivalAgencyBootstrapCurriculumV1("A", seed=20261001)
    counts = {0: 0, 1: 0}
    shared: dict[str, object] = {}
    for _ in range(2000):
        state = engine.create_base_state()
        teams.apply(state, shared)
        mutator.apply(state, shared)
        counts[int(shared["rival_v10_active_team"])] += 1
    assert abs(counts[0] / 2000.0 - 0.5) <= 0.04


def test_dead_play_truncates_at_ten_seconds() -> None:
    environment = build_v10_bootstrap_env(
        phase="A", seed=44, forced_family="natural", forced_mirror=False
    )
    try:
        observations = environment.reset()
        truncated_at = None
        for tick in range(1, 1250):
            actions = {agent: np.zeros(8, dtype=np.float32) for agent in observations}
            observations, _, terminated, truncated = environment.step(actions)
            assert not any(terminated.values())
            if any(truncated.values()):
                truncated_at = tick
                break
        assert truncated_at is not None
        assert 1199 <= truncated_at <= 1202
    finally:
        environment.close()


def test_m10_1_config_freezes_ppo_and_architecture() -> None:
    config = load_m10_1_config()
    assert config["backend"]["worker_count"] == 56
    assert config["ppo"]["minibatch_agent_steps"] == 48_000
    assert config["time_base"]["physics_hz"] == 120
    assert config["time_base"]["policy_hz"] == 120
    assert config["time_base"]["repeat_action"] is False
