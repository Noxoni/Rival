from __future__ import annotations

import numpy as np
import pytest
import torch

from rival_training.v9_checkpoint import (
    load_m09_config,
    load_v9_checkpoint,
    save_v9_checkpoint,
    verify_v9_checkpoint,
)
from rival_training.v9_observations import OBSERVATION_SIZE
from rival_training.v9_policy import RivalCriticV1, RivalPolicyV1
from rival_training.v9_trainer import compute_physical_time_gae, resolve_ppo_batch_size


def test_physical_time_gae_terminates_without_bootstrap() -> None:
    advantages, returns = compute_physical_time_gae(
        rewards=np.asarray([1.0], dtype=np.float32),
        values=np.asarray([0.25], dtype=np.float32),
        next_values=np.asarray([99.0], dtype=np.float32),
        terminated=np.asarray([1], dtype=np.uint8),
        truncated=np.asarray([0], dtype=np.uint8),
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages, [0.75])
    np.testing.assert_allclose(returns, [1.0])


def test_physical_time_gae_truncation_bootstraps_but_stops_recurrence() -> None:
    advantages, returns = compute_physical_time_gae(
        rewards=np.asarray([1.0, 2.0], dtype=np.float32),
        values=np.asarray([0.5, 0.25], dtype=np.float32),
        next_values=np.asarray([3.0, 4.0], dtype=np.float32),
        terminated=np.asarray([0, 0], dtype=np.uint8),
        truncated=np.asarray([1, 1], dtype=np.uint8),
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages, [3.2, 5.35], rtol=1e-6)
    np.testing.assert_allclose(returns, [3.7, 5.6], rtol=1e-6)


def test_m09_config_matches_runtime_contracts() -> None:
    config = load_m09_config()
    assert config["backend"]["worker_count"] == 56
    assert config["time_base"]["policy_hz"] == 120
    assert config["pilot"]["maximum_cumulative_agent_steps"] == 1_728_000
    assert config["pilot"]["gate11_steps_count_toward_pilot_ceiling"] is True


def test_worker_segment_shortfall_is_bounded() -> None:
    assert resolve_ppo_batch_size(191_938, 192_000, 56) == (191_938, 224)
    assert resolve_ppo_batch_size(192_010, 192_000, 56) == (192_000, 224)
    with pytest.raises(RuntimeError, match="maximum worker-segment shortfall"):
        resolve_ppo_batch_size(191_775, 192_000, 56)


def _populate_optimizer_state(
    actor: RivalPolicyV1,
    critic: RivalCriticV1,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
) -> None:
    observations = torch.zeros(2, OBSERVATION_SIZE)
    mean, log_std, button_logits = actor(observations)
    actor_loss = mean.square().mean() + log_std.square().mean() + button_logits.square().mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    actor_optimizer.step()
    critic_loss = critic(observations).square().mean()
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    critic_optimizer.step()


def test_v9_checkpoint_round_trip_includes_models_optimizers_and_contract(tmp_path) -> None:
    config = load_m09_config()
    actor = RivalPolicyV1()
    critic = RivalCriticV1()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    _populate_optimizer_state(actor, critic, actor_optimizer, critic_optimizer)
    corpus = np.zeros((3, OBSERVATION_SIZE), dtype=np.float32)
    manifest = save_v9_checkpoint(
        tmp_path,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        trainer_state={
            "completed_iterations": 2,
            "cumulative_agent_steps": 384_000,
            "cumulative_model_updates": 8,
        },
        config=config,
        reload_observations=corpus,
    )
    assert set(manifest["files"]) == {
        "actor.pt",
        "actor_optimizer.pt",
        "critic.pt",
        "critic_optimizer.pt",
        "reload_observations.npy",
        "trainer_state.json",
        "training_config.json",
    }
    verified = verify_v9_checkpoint(tmp_path, expected_config=config)
    assert verified["contract"]["action_version"] == "RivalActionV1"
    loaded = load_v9_checkpoint(tmp_path, device="cpu", expected_config=config)
    assert loaded["trainer_state"]["completed_iterations"] == 2
    assert loaded["actor_optimizer"].state
    assert loaded["critic_optimizer"].state
    np.testing.assert_array_equal(loaded["reload_observations"], corpus)
    for expected, actual in zip(actor.parameters(), loaded["actor"].parameters()):
        torch.testing.assert_close(expected, actual, atol=0.0, rtol=0.0)


def test_v9_checkpoint_detects_corruption(tmp_path) -> None:
    config = load_m09_config()
    actor = RivalPolicyV1()
    critic = RivalCriticV1()
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    _populate_optimizer_state(actor, critic, actor_optimizer, critic_optimizer)
    save_v9_checkpoint(
        tmp_path,
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        trainer_state={
            "completed_iterations": 1,
            "cumulative_agent_steps": 1,
            "cumulative_model_updates": 1,
        },
        config=config,
        reload_observations=np.zeros((1, OBSERVATION_SIZE), dtype=np.float32),
    )
    with (tmp_path / "actor.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(RuntimeError, match="actor.pt"):
        verify_v9_checkpoint(tmp_path, expected_config=config)
