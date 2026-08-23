from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from rival_training.v9_actions import (
    ACTION_DIM,
    ACTION_VERSION,
    ANALOG_DIM,
    BUTTON_COMBO_COUNT,
    LOG_STD_MAX,
    LOG_STD_MIN,
    TANH_EPSILON,
    RivalActionHeadV1,
    RivalActionV1Parser,
    RivalHybridDistribution,
    RivalHybridPolicy,
    action_metadata,
    button_bits_to_combo,
    button_combo_to_bits,
    validate_physical_actions,
)


class _TinyActor(nn.Module):
    def __init__(self, observation_size: int = 11) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(observation_size, 32), nn.SiLU())
        self.head = RivalActionHeadV1(32)

    def forward(self, observations: torch.Tensor):
        return self.head(self.encoder(observations))


@pytest.mark.parametrize("combo", range(BUTTON_COMBO_COUNT))
def test_all_button_combinations_round_trip_exactly(combo: int) -> None:
    bits = button_combo_to_bits(combo)
    assert bits.dtype == np.float32
    assert button_bits_to_combo(bits) == combo


def test_parser_emits_one_exact_physics_tick_and_updates_applied_history() -> None:
    parser = RivalActionV1Parser()
    shared: dict = {}
    parser.reset(["blue"], None, shared)
    action = np.asarray([0.125, -0.25, 0.375, -0.5, 0.625, 1, 0, 1], np.float32)
    parsed = parser.parse_actions({"blue": action}, None, shared)
    assert parser.get_action_space("blue") == ("continuous", ACTION_DIM)
    assert parser.repeats == 1
    assert parsed["blue"].shape == (1, ACTION_DIM)
    assert np.array_equal(parsed["blue"][0], action)
    assert np.array_equal(shared["previous_actions"]["blue"], action)
    assert np.array_equal(shared["rival_action_last_applied"]["blue"], action)
    assert shared["cadence_ticks"] == 1


def test_full_continuous_axes_and_steer_yaw_independence_survive_parser() -> None:
    rows = np.zeros((257, ACTION_DIM), dtype=np.float32)
    rows[:, :ANALOG_DIM] = np.linspace(-1.0, 1.0, len(rows), dtype=np.float32)[:, None]
    rows[:, 1] = np.linspace(1.0, -1.0, len(rows), dtype=np.float32)
    rows[:, 3] = np.sin(np.linspace(-math.pi, math.pi, len(rows))).astype(np.float32)
    validated = validate_physical_actions(rows)
    assert np.array_equal(validated, rows)
    assert len(np.unique(validated[:, 1])) == len(rows)
    assert not np.array_equal(validated[:, 1], validated[:, 3])


def test_representative_mechanics_traces_are_not_quantized_or_synthesized() -> None:
    # Capability traces only: these are never exposed as macros to the policy.
    traces = {
        "speedflip_flip_cancel": [
            [1, -0.73, -0.81, 0.42, -0.37, 1, 1, 0],
            [1, -0.19, 0.94, 0.08, -0.66, 0, 1, 0],
        ],
        "wavedash": [[0.67, 0.31, -0.92, -0.23, 0.14, 1, 0, 1]],
        "stall": [[0.0, 0.0, 0.0, 1.0, -1.0, 1, 0, 0]],
        "reset_followup": [[0.83, -0.12, 0.56, -0.71, 0.48, 1, 1, 0]],
        "wall_dash": [[1.0, 0.91, -0.44, 0.37, -0.28, 1, 1, 1]],
        "aerial_air_roll": [[0.41, -0.07, 0.63, -0.52, 0.79, 0, 1, 0]],
    }
    parser = RivalActionV1Parser()
    shared: dict = {}
    parser.reset(["agent"], None, shared)
    for rows in traces.values():
        for row in rows:
            expected = np.asarray(row, dtype=np.float32)
            actual = parser.parse_actions({"agent": expected}, None, shared)["agent"]
            assert np.array_equal(actual[0], expected)


@pytest.mark.parametrize(
    "invalid",
    [
        np.zeros(7, dtype=np.float32),
        np.asarray([1.01, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.asarray([0, 0, 0, 0, 0, 0.5, 0, 0], dtype=np.float32),
        np.asarray([0, 0, 0, 0, np.nan, 0, 0, 0], dtype=np.float32),
    ],
)
def test_parser_rejects_nonphysical_transport(invalid: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_physical_actions(invalid)


def test_tanh_gaussian_log_probability_matches_independent_reference() -> None:
    torch.manual_seed(911)
    mean = torch.randn(19, ANALOG_DIM, dtype=torch.float64) * 0.3
    log_std = torch.linspace(-1.1, 0.2, ANALOG_DIM, dtype=torch.float64)
    logits = torch.randn(19, BUTTON_COMBO_COUNT, dtype=torch.float64)
    distribution = RivalHybridDistribution(mean, log_std, logits)
    analog = torch.tanh(torch.randn(19, ANALOG_DIM, dtype=torch.float64) * 1.7)
    combos = torch.arange(19) % BUTTON_COMBO_COUNT
    bits = torch.stack(
        ((combos & 1), ((combos >> 1) & 1), ((combos >> 2) & 1)), dim=-1
    ).to(torch.float64)
    actions = torch.cat((analog, bits), dim=-1)

    bounded = analog.clamp(-1 + TANH_EPSILON, 1 - TANH_EPSILON)
    pre_tanh = torch.atanh(bounded)
    reference_analog = (
        Normal(mean, log_std.exp()).log_prob(pre_tanh)
        - torch.log(torch.clamp(1 - bounded.square(), min=TANH_EPSILON))
    ).sum(-1)
    reference_buttons = Categorical(logits=logits).log_prob(combos)
    assert torch.allclose(
        distribution.log_prob(actions), reference_analog + reference_buttons, atol=1e-11
    )


def test_mixed_log_probability_reproduces_stored_rollout_actions() -> None:
    torch.manual_seed(912)
    actor = _TinyActor()
    policy = RivalHybridPolicy(actor, "cpu")
    observations = torch.randn(64, 11)
    actions, rollout_log_probs = policy.get_action(observations)
    backprop_log_probs, entropy = policy.get_backprop_data(observations, actions)
    assert actions.shape == (64, ACTION_DIM)
    assert torch.equal(actions[:, ANALOG_DIM:], actions[:, ANALOG_DIM:].round())
    assert torch.allclose(rollout_log_probs, backprop_log_probs.detach(), atol=1e-6)
    assert torch.isfinite(entropy)
    assert policy.last_entropy["button_exact"] > 0


def test_seeded_hybrid_ppo_objective_has_finite_nonzero_branch_gradients() -> None:
    torch.manual_seed(913)
    actor = _TinyActor()
    policy = RivalHybridPolicy(actor, "cpu")
    observations = torch.randn(128, 11)
    with torch.no_grad():
        actions, old_log_probs = policy.get_action(observations)
    # Include exact near-saturation values to exercise inverse tanh stability.
    actions[:2, :ANALOG_DIM] = torch.tensor(
        [[0.999999, -0.999999, 0.9999, -0.9999, 0.0]] * 2
    )
    log_probs, entropy = policy.get_backprop_data(observations, actions)
    advantages = torch.linspace(-1.0, 1.0, len(observations))
    ratio = torch.exp(log_probs - old_log_probs)
    objective = -(ratio * advantages).mean() - 0.001 * entropy
    objective.backward()

    head = actor.head
    assert torch.isfinite(objective)
    assert head.analog_mean.weight.grad is not None
    assert torch.all(torch.isfinite(head.analog_mean.weight.grad))
    assert torch.all(head.analog_mean.weight.grad.abs().sum(dim=1) > 0)
    assert head.analog_log_std.grad is not None
    assert torch.all(torch.isfinite(head.analog_log_std.grad))
    assert torch.all(head.analog_log_std.grad.abs() > 0)
    assert head.button_logits.weight.grad is not None
    assert torch.all(torch.isfinite(head.button_logits.weight.grad))
    assert float(head.button_logits.weight.grad.abs().sum()) > 0


def test_deterministic_action_is_tanh_mean_plus_joint_argmax() -> None:
    torch.manual_seed(914)
    actor = _TinyActor().eval()
    policy = RivalHybridPolicy(actor, "cpu")
    observations = torch.randn(7, 11)
    actions, log_probabilities = policy.get_action(observations, deterministic=True)
    mean, _, logits = actor(observations)
    expected_analog = torch.tanh(mean)
    expected_combos = logits.argmax(dim=-1).numpy()
    assert torch.allclose(actions[:, :ANALOG_DIM], expected_analog)
    for index, combo in enumerate(expected_combos):
        assert np.array_equal(actions[index, ANALOG_DIM:].numpy(), button_combo_to_bits(combo))
    assert torch.isfinite(log_probabilities).all()


def test_metadata_freezes_native_unmasked_no_lookup_contract() -> None:
    metadata = action_metadata()
    assert metadata["version"] == ACTION_VERSION
    assert metadata["physics_hz"] == metadata["policy_hz"] == 120
    assert metadata["policy_decisions_per_physics_tick"] == 1
    assert metadata["repeat_action"] is False
    assert metadata["state_dependent_action_mask"] is False
    assert metadata["lookup_table"] is False
    assert metadata["transport_shape"] == [8]
    assert metadata["analog"]["log_std_bounds"] == [LOG_STD_MIN, LOG_STD_MAX]
    assert len(metadata["parser_source_sha256"]) == 64


def test_action_head_log_std_is_bounded() -> None:
    head = RivalActionHeadV1(4)
    with torch.no_grad():
        head.analog_log_std.copy_(torch.tensor([-99.0, -5.0, 0.0, 1.0, 99.0]))
    _, log_std, _ = head(torch.zeros(2, 4))
    assert float(log_std.detach().min()) == LOG_STD_MIN
    assert float(log_std.detach().max()) == LOG_STD_MAX
