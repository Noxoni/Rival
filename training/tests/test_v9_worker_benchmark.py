from __future__ import annotations

import numpy as np

from rival_training.v9_benchmark import GAMMA_120HZ, _gae
from rival_training.v9_environment import make_v9_training_gym_env
from rival_training.v9_policy import make_instrumented_rival_policy
from scripts.run_m09_worker_sweep_gate import _selected, _should_extend


def _result(workers: int, rate: float, *, stable: bool = True):
    return {
        "workers": workers,
        "stable": stable,
        "sustained_agent_steps_per_second_mean": rate,
    }


def test_worker_selection_uses_highest_stable_rate_not_largest_count() -> None:
    results = [
        _result(40, 8000.0),
        _result(48, 9000.0),
        _result(56, 8500.0),
        _result(64, 9500.0, stable=False),
    ]
    assert _selected(results)["workers"] == 48


def test_upper_sweep_extends_only_for_three_percent_stable_gain() -> None:
    assert _should_extend([_result(48, 9000), _result(56, 9300)])
    assert not _should_extend([_result(48, 9000), _result(56, 9200)])
    assert not _should_extend(
        [_result(48, 9000), _result(56, 9500, stable=False)]
    )


def test_gae_resets_at_terminal_boundaries() -> None:
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    values = np.zeros(3, dtype=np.float32)
    next_values = np.asarray([0.0, 0.0, 5.0], dtype=np.float32)
    dones = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    truncated = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    advantages, returns = _gae(rewards, values, next_values, dones, truncated)
    assert advantages[1] == 2.0
    assert advantages[2] == 3.0
    assert advantages[0] > 1.0
    assert advantages[0] < 1.0 + GAMMA_120HZ * 2.0
    np.testing.assert_array_equal(advantages, returns)


def test_instrumented_policy_records_real_batch_and_latency() -> None:
    policy = make_instrumented_rival_policy("cpu")
    observations = np.zeros((7, 714), dtype=np.float32)
    actions, log_probabilities = policy.get_action(observations)
    samples = policy.drain_inference_samples()
    assert actions.shape == (7, 8)
    assert log_probabilities.shape == (7,)
    assert len(samples) == 1
    assert samples[0]["batch_size"] == 7
    assert samples[0]["wall_seconds"] > 0.0
    assert policy.drain_inference_samples() == []


def test_pickle_safe_gym_factory_exposes_v9_shapes() -> None:
    environment = make_v9_training_gym_env()
    try:
        observations = environment.reset()
        assert observations.shape == (2, 714)
        assert environment.action_space.shape == (8,)
        actions = np.zeros((2, 8), dtype=np.float32)
        next_observations, rewards, done, truncated, _ = environment.step(actions)
        assert next_observations.shape == (2, 714)
        assert len(rewards) == 2
        assert not done
        assert not truncated
    finally:
        environment.close()
