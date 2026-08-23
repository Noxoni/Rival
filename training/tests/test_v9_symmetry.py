from __future__ import annotations

import numpy as np

from rival_training.v9_actions import ACTION_DIM
from rival_training.v9_canonical import RocketSimCanonicalAdapterV1
from rival_training.v9_environment import build_v9_training_env
from rival_training.v9_symmetry import (
    CONTROLLER_SIGN,
    WORLD_REFLECTION,
    mirror_canonical_state,
    mirror_controller,
)


def _by_team(environment, values):
    return {
        int(environment.state.cars[agent].team_num): value
        for agent, value in values.items()
    }


def test_controller_mirror_has_documented_signs_and_is_involutive() -> None:
    action = np.asarray([0.8, -0.7, 0.6, -0.5, 0.4, 1, 0, 1], dtype=np.float32)
    mirrored = mirror_controller(action)
    np.testing.assert_array_equal(mirrored, action * CONTROLLER_SIGN)
    np.testing.assert_array_equal(mirror_controller(mirrored), action)


def test_canonical_mirror_is_exactly_involutive() -> None:
    environment = build_v9_training_env(forced_mirror=False)
    try:
        environment.reset()
        adapter = RocketSimCanonicalAdapterV1()
        agent = environment.agents[0]
        original = adapter.adapt(environment.state, agent, environment.shared_info)
        restored = mirror_canonical_state(mirror_canonical_state(original))
        assert restored.to_payload() == original.to_payload()
    finally:
        environment.close()


def test_paired_mirrored_environments_produce_symmetric_states_and_actor_obs() -> None:
    plain = build_v9_training_env(forced_mirror=False, prediction_refresh_ticks=1)
    mirrored = build_v9_training_env(forced_mirror=True, prediction_refresh_ticks=1)
    rng = np.random.default_rng(20260908)
    try:
        plain_obs = plain.reset()
        mirrored_obs = mirrored.reset()
        for team in (0, 1):
            np.testing.assert_allclose(
                _by_team(plain, plain_obs)[team],
                _by_team(mirrored, mirrored_obs)[team],
                atol=1e-6,
                rtol=0.0,
            )

        for _ in range(64):
            rows = {}
            for team in (0, 1):
                row = rng.uniform(-1.0, 1.0, size=ACTION_DIM).astype(np.float32)
                row[5:] = rng.integers(0, 2, size=3)
                rows[team] = row
            plain_actions = {
                agent: rows[int(plain.state.cars[agent].team_num)]
                for agent in plain.agents
            }
            mirrored_actions = {
                agent: rows[int(mirrored.state.cars[agent].team_num)]
                for agent in mirrored.agents
            }
            plain_obs, plain_reward, *_ = plain.step(plain_actions)
            mirrored_obs, mirrored_reward, *_ = mirrored.step(mirrored_actions)

            for team in (0, 1):
                np.testing.assert_allclose(
                    _by_team(plain, plain_obs)[team],
                    _by_team(mirrored, mirrored_obs)[team],
                    atol=2e-5,
                    rtol=0.0,
                )
                np.testing.assert_allclose(
                    _by_team(plain, plain_reward)[team],
                    _by_team(mirrored, mirrored_reward)[team],
                    atol=1e-7,
                    rtol=0.0,
                )

            plain_ball = np.asarray(plain.state.ball.position)
            mirrored_ball = np.asarray(mirrored.state.ball.position)
            np.testing.assert_allclose(
                WORLD_REFLECTION @ mirrored_ball,
                plain_ball,
                atol=1e-3,
                rtol=0.0,
            )
            assert plain.shared_info["rival_v9_episode_mirror"] is False
            assert mirrored.shared_info["rival_v9_episode_mirror"] is True

            plain_selected = _by_team(
                plain, plain.shared_info["rival_v9_selected_actions"]
            )
            mirrored_selected = _by_team(
                mirrored, mirrored.shared_info["rival_v9_selected_actions"]
            )
            for team in (0, 1):
                np.testing.assert_array_equal(
                    mirrored_selected[team], mirror_controller(plain_selected[team])
                )
    finally:
        plain.close()
        mirrored.close()


def test_random_mirror_bit_changes_only_at_reset() -> None:
    environment = build_v9_training_env(
        mirror_probability=0.5,
        symmetry_seed=7,
    )
    observed = set()
    try:
        for _ in range(16):
            environment.reset()
            bit = bool(environment.shared_info["rival_v9_episode_mirror"])
            observed.add(bit)
            actions = {
                agent: np.zeros(ACTION_DIM, dtype=np.float32)
                for agent in environment.agents
            }
            for _ in range(3):
                environment.step(actions)
                assert bool(environment.shared_info["rival_v9_episode_mirror"]) == bit
        assert observed == {False, True}
    finally:
        environment.close()
