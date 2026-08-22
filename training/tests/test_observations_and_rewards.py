from __future__ import annotations

import math

import numpy as np

from rival_training.environment import build_rlgym_env
from rival_training.observations import (
    OBSERVATION_SIZE,
    PLAYER_OBSERVATION_SIZE,
    observation_metadata,
)
from rival_training.rewards import COMPONENTS, reward_metadata


def test_observation_contract_is_explicit() -> None:
    metadata = observation_metadata()
    assert OBSERVATION_SIZE == 432
    assert PLAYER_OBSERVATION_SIZE == 51
    assert metadata["shape"] == [432]
    assert metadata["prediction_ticks"] == [22, 66, 198, 594]
    assert metadata["live_training_differences"]


def test_reward_is_outcome_dominant_and_componentized() -> None:
    metadata = reward_metadata()
    assert metadata["outcome_goal"] == 10.0
    assert metadata["outcome_concede"] == -10.0
    assert tuple(metadata["components"]) == COMPONENTS
    assert "named-mechanic" in metadata["shaping_policy"]


def test_actual_goal_and_concede_components_are_signed_ten() -> None:
    env = build_rlgym_env("mechanics4")
    try:
        env.reset()
        desired = env.state
        desired.ball.position[1] = 5300.0
        env.set_state(desired)
        state = env.state
        agents = env.agents
        env.reward_fn.reset(agents, state, env.shared_info)
        rewards = env.reward_fn.get_rewards(
            agents,
            state,
            {agent: True for agent in agents},
            {agent: False for agent in agents},
            env.shared_info,
        )
        assert state.goal_scored
        assert state.scoring_team == 0
        assert env.shared_info["reward_components"]["blue-0"]["outcome"] == 10.0
        assert env.shared_info["reward_components"]["orange-0"]["outcome"] == -10.0
        assert rewards["blue-0"] >= 10.0
        assert rewards["orange-0"] <= -10.0
    finally:
        env.close()


def test_random_rollout_rewards_and_components_stay_finite() -> None:
    env = build_rlgym_env("mechanics4")
    rng = np.random.default_rng(5)
    try:
        observations = env.reset()
        for _ in range(256):
            actions = {
                agent: np.array([rng.integers(0, 158)], dtype=np.int64)
                for agent in observations
            }
            observations, rewards, terminated, truncated = env.step(actions)
            assert all(np.isfinite(obs).all() for obs in observations.values())
            assert all(math.isfinite(value) for value in rewards.values())
            component_values = env.shared_info["reward_components"]
            assert all(
                math.isfinite(value)
                for values in component_values.values()
                for value in values.values()
            )
            if any(terminated.values()) or any(truncated.values()):
                observations = env.reset()
    finally:
        env.close()
