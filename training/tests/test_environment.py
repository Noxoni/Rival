from __future__ import annotations

import numpy as np
import pytest

from rival_training.environment import build_rlgym_env


@pytest.mark.parametrize("cadence", ["legacy8", "mechanics4"])
def test_natural_1v1_reset_and_step_are_finite(cadence: str) -> None:
    env = build_rlgym_env(cadence)
    try:
        observations = env.reset()
        assert set(observations) == {"blue-0", "orange-0"}
        assert all(obs.shape == (432,) for obs in observations.values())
        assert all(np.isfinite(obs).all() for obs in observations.values())
        assert all(space == ("discrete", 158) for space in env.action_spaces.values())

        actions = {agent: np.array([0], dtype=np.int64) for agent in observations}
        next_obs, rewards, terminated, truncated = env.step(actions)
        assert all(obs.shape == (432,) for obs in next_obs.values())
        assert all(np.isfinite(obs).all() for obs in next_obs.values())
        assert all(np.isfinite(value) for value in rewards.values())
        assert not any(terminated.values())
        assert not any(truncated.values())
    finally:
        env.close()


def test_live_x_mirror_semantics_are_applied_and_recovered_in_previous_action() -> None:
    env = build_rlgym_env("mechanics4")
    try:
        observations = env.reset()
        parser = env.action_parser
        # Row 1 has non-zero steer/yaw and is therefore mirror-observable.
        index = next(
            i for i, row in enumerate(parser.lookup_table) if row[1] != 0 and row[3] != 0
        )
        actions = {agent: np.array([index], dtype=np.int64) for agent in observations}
        expected_world = {}
        for agent in observations:
            car = env.state.cars[agent]
            mirror = (car.team_num == 1) != (float(car.physics.position[0]) < 0)
            row = parser.lookup_table[index].copy()
            if mirror:
                row[[1, 3, 4]] *= -1
            expected_world[agent] = row
        env.step(actions)
        for agent, row in expected_world.items():
            assert np.array_equal(env.shared_info["previous_actions"][agent], row)
    finally:
        env.close()
