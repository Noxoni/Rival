from __future__ import annotations

import numpy as np
import torch

from rival_training.v9_actions import ACTION_DIM, ANALOG_DIM
from rival_training.v9_policy import (
    SCHEMA_LAYOUT,
    RivalCriticV1,
    RivalPolicyV1,
    make_rival_policy,
    policy_metadata,
    trainable_parameter_count,
)


def _observations(batch: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260909)
    return torch.randn(
        batch,
        SCHEMA_LAYOUT["observation_size"],
        generator=generator,
        dtype=torch.float32,
    )


def test_actor_and_critic_have_schema_driven_shapes_and_no_shared_parameters() -> None:
    actor = RivalPolicyV1()
    critic = RivalCriticV1()
    observations = _observations()
    mean, log_std, button_logits = actor(observations)
    values = critic(observations)
    assert mean.shape == (8, 5)
    assert log_std.shape == (5,)
    assert button_logits.shape == (8, 8)
    assert values.shape == (8, 1)
    assert torch.isfinite(mean).all()
    assert torch.isfinite(log_std).all()
    assert torch.isfinite(button_logits).all()
    assert torch.isfinite(values).all()
    actor_storage = {parameter.data_ptr() for parameter in actor.parameters()}
    critic_storage = {parameter.data_ptr() for parameter in critic.parameters()}
    assert actor_storage.isdisjoint(critic_storage)


def test_actor_size_is_within_authorized_initial_target() -> None:
    actor_parameters = trainable_parameter_count(RivalPolicyV1())
    critic_parameters = trainable_parameter_count(RivalCriticV1())
    assert 1_000_000 <= actor_parameters <= 4_000_000
    assert 1_000_000 <= critic_parameters <= 4_000_000


def test_pad_encoder_is_entity_permutation_invariant() -> None:
    actor = RivalPolicyV1().eval()
    observations = _observations(3)
    permuted = observations.clone()
    start, end = SCHEMA_LAYOUT["pad_slice"]
    count, width = SCHEMA_LAYOUT["pad_shape"]
    rows = permuted[:, start:end].reshape(3, count, width)
    permutation = torch.randperm(count, generator=torch.Generator().manual_seed(91))
    permuted[:, start:end] = rows[:, permutation].reshape(3, -1)
    with torch.inference_mode():
        original_outputs = actor(observations)
        permuted_outputs = actor(permuted)
    for original, transformed in zip(original_outputs, permuted_outputs):
        torch.testing.assert_close(original, transformed, atol=2e-6, rtol=0.0)


def test_prediction_and_history_encoders_preserve_order() -> None:
    actor = RivalPolicyV1().eval()
    observations = _observations(2)
    prediction_reversed = observations.clone()
    start, end = SCHEMA_LAYOUT["prediction_slice"]
    count, width = SCHEMA_LAYOUT["prediction_shape"]
    rows = prediction_reversed[:, start:end].reshape(2, count, width)
    prediction_reversed[:, start:end] = rows.flip(1).reshape(2, -1)

    history_reversed = observations.clone()
    start, end = SCHEMA_LAYOUT["history_slice"]
    ticks, controller_width = SCHEMA_LAYOUT["self_history_shape"]
    half = ticks * controller_width
    for offset in (0, half):
        rows = history_reversed[:, start + offset : start + offset + half].reshape(
            2, ticks, controller_width
        )
        history_reversed[:, start + offset : start + offset + half] = rows.flip(
            1
        ).reshape(2, -1)

    with torch.inference_mode():
        baseline = actor(observations)[0]
        prediction_output = actor(prediction_reversed)[0]
        history_output = actor(history_reversed)[0]
    assert not torch.allclose(baseline, prediction_output, atol=1e-7, rtol=0.0)
    assert not torch.allclose(baseline, history_output, atol=1e-7, rtol=0.0)


def test_hybrid_policy_samples_legal_rows_and_backpropagates_all_heads() -> None:
    policy = make_rival_policy("cpu")
    observations = _observations(16)
    actions, rollout_log_probs = policy.get_action(observations)
    assert actions.shape == (16, ACTION_DIM)
    assert rollout_log_probs.shape == (16,)
    assert torch.all(actions[:, :ANALOG_DIM] >= -1.0)
    assert torch.all(actions[:, :ANALOG_DIM] <= 1.0)
    assert torch.equal(actions[:, ANALOG_DIM:], actions[:, ANALOG_DIM:].round())
    log_probs, entropy = policy.get_backprop_data(observations, actions)
    loss = -log_probs.mean() - 0.001 * entropy
    loss.backward()
    actor = policy.actor
    assert actor.action_head.analog_mean.weight.grad is not None
    assert float(actor.action_head.analog_mean.weight.grad.abs().sum()) > 0.0
    assert actor.action_head.analog_log_std.grad is not None
    assert float(actor.action_head.analog_log_std.grad.abs().sum()) > 0.0
    assert actor.action_head.button_logits.weight.grad is not None
    assert float(actor.action_head.button_logits.weight.grad.abs().sum()) > 0.0
    assert any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in actor.encoder.parameters()
    )


def test_seeded_initialization_is_reproducible_without_global_rng_side_effect() -> None:
    torch.manual_seed(123)
    before = torch.rand(3)
    first = RivalPolicyV1(seed=42)
    between = torch.rand(3)
    torch.manual_seed(123)
    expected_before = torch.rand(3)
    expected_between = torch.rand(3)
    second = RivalPolicyV1(seed=42)
    torch.testing.assert_close(before, expected_before)
    torch.testing.assert_close(between, expected_between)
    for left, right in zip(first.state_dict().values(), second.state_dict().values()):
        torch.testing.assert_close(left, right, atol=0.0, rtol=0.0)


def test_policy_metadata_freezes_schema_and_structured_architecture() -> None:
    metadata = policy_metadata()
    assert metadata["observation_schema_sha256"] == SCHEMA_LAYOUT["schema_sha256"]
    assert metadata["observation_size"] == 714
    assert metadata["logical_widths"]["fusion_input"] == 768
    assert metadata["logical_widths"]["fusion_output"] == 512
    assert metadata["actor_critic_parameters_shared"] is False
    assert metadata["recurrent"] is False
    assert metadata["batch_norm"] is False
    assert metadata["entity_shapes"]["pads"] == [34, 9]
    assert len(metadata["source_sha256"]) == 64


def test_numpy_inference_path_uses_exact_observation_contract() -> None:
    policy = make_rival_policy("cpu")
    observations = _observations(5).numpy()
    actions, log_probabilities = policy.get_action(observations, deterministic=True)
    assert actions.shape == (5, ACTION_DIM)
    assert log_probabilities.shape == (5,)
    assert np.isfinite(actions.numpy()).all()
    assert np.isfinite(log_probabilities.numpy()).all()
