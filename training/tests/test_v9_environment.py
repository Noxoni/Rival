from __future__ import annotations

import numpy as np

from rival_training.v9_actions import ACTION_DIM
from rival_training.v9_environment import build_v9_diagnostic_env
from rival_training.v9_observations import OBSERVATION_SIZE


def test_v9_diagnostic_environment_is_native_one_tick_and_finite() -> None:
    environment = build_v9_diagnostic_env(prediction_refresh_ticks=4)
    try:
        observations = environment.reset()
        assert len(observations) == 2
        assert all(value.shape == (OBSERVATION_SIZE,) for value in observations.values())
        assert all(np.isfinite(value).all() for value in observations.values())
        initial_tick = int(environment.state.tick_count)
        actions = {
            agent: np.zeros(ACTION_DIM, dtype=np.float32)
            for agent in environment.agents
        }
        observations, rewards, terminated, truncated = environment.step(actions)
        assert int(environment.state.tick_count) == initial_tick + 1
        assert all(value == 0.0 for value in rewards.values())
        assert not any(terminated.values())
        assert not any(truncated.values())
        assert all(np.isfinite(value).all() for value in observations.values())
        assert {
            int(timing["prediction_age_ticks"])
            for timing in environment.shared_info[
                "rival_v9_observation_timings"
            ].values()
        } == {1}
    finally:
        environment.close()


def test_prediction_refresh_periods_expose_expected_age_cycle() -> None:
    for period in (1, 2, 4):
        environment = build_v9_diagnostic_env(prediction_refresh_ticks=period)
        try:
            environment.reset()
            observed_ages: list[int] = []
            for _ in range(period * 2):
                actions = {
                    agent: np.zeros(ACTION_DIM, dtype=np.float32)
                    for agent in environment.agents
                }
                environment.step(actions)
                timings = environment.shared_info["rival_v9_observation_timings"]
                observed_ages.append(
                    int(next(iter(timings.values()))["prediction_age_ticks"])
                )
            assert observed_ages == [index % period for index in range(1, period * 2 + 1)]
        finally:
            environment.close()
