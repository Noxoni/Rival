from __future__ import annotations

import numpy as np

from rival_training.v9_actions import ACTION_DIM
from rival_training.v9_environment import build_v9_diagnostic_env
from rival_training.v9_observations import (
    OBSERVATION_SIZE,
    observation_schema_manifest,
)


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


def test_observation_history_records_applied_not_newly_selected_controller() -> None:
    environment = build_v9_diagnostic_env(prediction_refresh_ticks=1)
    try:
        environment.reset()
        agents_by_team = {
            int(environment.state.cars[agent].team_num): agent
            for agent in environment.agents
        }
        first = {
            agents_by_team[0]: np.asarray(
                [1.0, -0.2, 0.3, -0.4, 0.5, 1, 1, 0], dtype=np.float32
            ),
            agents_by_team[1]: np.asarray(
                [-1.0, 0.6, -0.7, 0.8, -0.9, 0, 0, 1], dtype=np.float32
            ),
        }
        second = {
            agent: np.asarray(
                [0.25, 0.35, -0.45, 0.55, -0.65, 0, 1, 1], dtype=np.float32
            )
            for agent in environment.agents
        }
        first_observations, *_ = environment.step(first)
        history = next(
            field
            for field in observation_schema_manifest()["fields"]
            if field["name"] == "history.self_controllers"
        )
        start, end = int(history["start"]), int(history["end"])
        for agent, observation in first_observations.items():
            rows = observation[start:end].reshape(8, ACTION_DIM)
            np.testing.assert_array_equal(rows[-1], np.zeros(ACTION_DIM))
            np.testing.assert_array_equal(
                environment.shared_info["rival_v9_pending_actions"][agent],
                first[agent],
            )

        second_observations, *_ = environment.step(second)
        for agent, observation in second_observations.items():
            rows = observation[start:end].reshape(8, ACTION_DIM)
            np.testing.assert_array_equal(rows[-1], first[agent])
            np.testing.assert_array_equal(
                environment.shared_info["rival_v9_applied_actions"][agent],
                first[agent],
            )
            np.testing.assert_array_equal(
                environment.shared_info["rival_v9_pending_actions"][agent],
                second[agent],
            )
    finally:
        environment.close()
