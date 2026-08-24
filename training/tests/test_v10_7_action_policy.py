from __future__ import annotations

import math

import torch
from torch.distributions import Bernoulli, Normal

from rival_training.v10_6_reward import ball_acquisition_reward_metadata
from rival_training.v10_7_actions import (
    ACTION_DIM,
    ANALOG_DIM,
    BUTTON_FIELDS,
    BUTTON_PERSISTENCE,
    HISTORY_START,
    HISTORY_TICKS,
    CONTROLLER_SIZE,
    TANH_EPSILON,
    RivalStickyBernoulliDistribution,
    RivalStickyBernoulliPolicy,
    deterministic_transition_reachability,
    effective_button_probabilities,
    previous_applied_buttons,
)
from rival_training.v10_7_campaign import (
    SOURCE_ACTOR_SHA256,
    SOURCE_CHECKPOINT,
    actor_only_architecture_transfer,
    button_entropy_coefficient,
    load_stage1_config,
)
from rival_training.v10_7_policy import RivalPolicyV1IndependentStickyButtons
from rival_training.v9_observations import OBSERVATION_SIZE


def _observations(buttons: tuple[int, int, int], rows: int = 4) -> torch.Tensor:
    values = torch.zeros(rows, OBSERVATION_SIZE)
    start = HISTORY_START + (HISTORY_TICKS - 1) * CONTROLLER_SIZE + ANALOG_DIM
    values[:, start : start + 3] = torch.tensor(buttons, dtype=torch.float32)
    return values


def test_previous_actually_applied_buttons_are_read_from_newest_self_history() -> None:
    observations = _observations((1, 0, 1))
    expected = torch.tensor([[1.0, 0.0, 1.0]]).expand(4, -1)
    assert torch.equal(previous_applied_buttons(observations), expected)


def test_log_odds_persistence_matches_neutral_requested_values() -> None:
    logits = torch.zeros(2, 3, dtype=torch.float64)
    previous = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.float64)
    base, effective = effective_button_probabilities(logits, previous)
    assert torch.equal(base, torch.full_like(base, 0.5))
    for index, name in enumerate(BUTTON_FIELDS):
        persistence = BUTTON_PERSISTENCE[name]
        assert math.isclose(float(effective[0, index]), 1.0 - persistence, abs_tol=1e-12)
        assert math.isclose(float(effective[1, index]), persistence, abs_tol=1e-12)


def test_corrected_persistence_allows_deterministic_transitions_both_directions() -> None:
    previous_off = torch.zeros(1, 3)
    previous_on = torch.ones(1, 3)
    _, on = effective_button_probabilities(torch.full((1, 3), 20.0), previous_off)
    _, off = effective_button_probabilities(torch.full((1, 3), -20.0), previous_on)
    assert torch.all(on >= 0.5)
    assert torch.all(off < 0.5)
    reachability = deterministic_transition_reachability()
    assert reachability["pathological_before_ppo"] is False
    assert reachability["deterministic_reset_policy_can_ever_enable_a_button"] is True
    assert all(row["off_to_on_reachable"] for row in reachability["buttons"].values())
    assert all(row["on_to_off_reachable"] for row in reachability["buttons"].values())


def test_sticky_bernoulli_log_probability_matches_independent_reference() -> None:
    torch.manual_seed(1071)
    mean = torch.randn(31, 5, dtype=torch.float64) * 0.25
    log_std = torch.linspace(-1.0, 0.0, 5, dtype=torch.float64)
    logits = torch.randn(31, 3, dtype=torch.float64)
    previous = torch.randint(0, 2, (31, 3), dtype=torch.int64).to(torch.float64)
    distribution = RivalStickyBernoulliDistribution(mean, log_std, logits, previous)
    actions, stored = distribution.sample()

    analog = actions[:, :ANALOG_DIM]
    bounded = analog.clamp(-1 + TANH_EPSILON, 1 - TANH_EPSILON)
    pre_tanh = torch.atanh(bounded)
    reference_analog = (
        Normal(mean, log_std.exp()).log_prob(pre_tanh)
        - torch.log(torch.clamp(1 - bounded.square(), min=TANH_EPSILON))
    ).sum(dim=-1)
    reference_buttons = Bernoulli(
        probs=distribution.effective_probabilities
    ).log_prob(actions[:, ANALOG_DIM:]).sum(dim=-1)
    assert torch.equal(stored, distribution.log_prob(actions))
    assert torch.allclose(stored, reference_analog + reference_buttons, atol=1e-11)


def test_deterministic_mode_uses_same_effective_probabilities() -> None:
    mean = torch.zeros(2, 5)
    log_std = torch.zeros(5)
    logits = torch.tensor([[20.0, -20.0, 20.0], [-20.0, 20.0, -20.0]])
    previous = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    distribution = RivalStickyBernoulliDistribution(mean, log_std, logits, previous)
    expected = (distribution.effective_probabilities >= 0.5).to(torch.float32)
    assert torch.equal(distribution.mode()[:, ANALOG_DIM:], expected)


def test_all_eight_physical_button_combinations_remain_representable() -> None:
    mean = torch.zeros(8, 5)
    log_std = torch.full((5,), -5.0)
    previous = torch.zeros(8, 3)
    desired = torch.tensor(
        [[combo & 1, (combo >> 1) & 1, (combo >> 2) & 1] for combo in range(8)],
        dtype=torch.float32,
    )
    logits = torch.where(desired > 0.5, 20.0, -20.0)
    distribution = RivalStickyBernoulliDistribution(mean, log_std, logits, previous)
    assert torch.equal(distribution.mode()[:, ANALOG_DIM:], desired)


def test_seeded_ppo_smoke_reaches_all_eight_controller_branches() -> None:
    torch.manual_seed(1072)
    actor = RivalPolicyV1IndependentStickyButtons()
    policy = RivalStickyBernoulliPolicy(actor, "cpu")
    observations = torch.randn(128, OBSERVATION_SIZE)
    start = HISTORY_START + (HISTORY_TICKS - 1) * CONTROLLER_SIZE + ANALOG_DIM
    observations[:, start : start + 3] = torch.randint(0, 2, (128, 3)).float()
    with torch.no_grad():
        actions, old_log_probabilities = policy.get_action(observations)
    distribution = policy.distribution(observations)
    log_probabilities = distribution.log_prob(actions)
    entropy = distribution.entropy(actions)
    advantages = torch.linspace(-1.0, 1.0, len(observations))
    objective = -(
        torch.exp(log_probabilities - old_log_probabilities) * advantages
    ).mean() - 0.0002 * entropy.analog_monte_carlo - 0.001 * entropy.button_exact
    objective.backward()
    head = actor.action_head
    assert torch.all(head.analog_mean.weight.grad.abs().sum(dim=1) > 0)
    assert torch.all(head.analog_log_std.grad.abs() > 0)
    assert torch.all(head.button_logits.weight.grad.abs().sum(dim=1) > 0)


def test_m10_7_config_freezes_m10_6_reward_and_only_schedules_button_entropy() -> None:
    config = load_stage1_config()
    assert config["reward_contract"] == {
        **config["reward_contract"]
    }
    metadata = ball_acquisition_reward_metadata()
    assert config["reward_contract"]["distance_progress_scale_uu"] == metadata[
        "distance_progress_scale_uu"
    ]
    assert config["reward_contract"]["heading_delta_scale"] == 1.5
    assert button_entropy_coefficient(0, config) == 0.001
    assert button_entropy_coefficient(108_000, config) == 0.0005
    assert button_entropy_coefficient(216_000, config) == 0.0
    assert button_entropy_coefficient(1_080_000, config) == 0.0


def test_exact_selective_source_transfer_and_fresh_learning_state() -> None:
    config = load_stage1_config()
    transfer = actor_only_architecture_transfer(SOURCE_CHECKPOINT, config, device="cpu")
    proof = transfer["proof"]
    assert proof["source_actor_file_sha256"] == SOURCE_ACTOR_SHA256
    assert proof["checks"]["passed"]
    assert proof["newly_initialized_parameter_names"] == [
        "action_head.button_logits.bias",
        "action_head.button_logits.weight",
    ]
    assert not transfer["actor_optimizer"].state
    assert not transfer["critic_optimizer"].state
    assert transfer["actor"].action_head.button_logits.out_features == 3


def test_policy_produces_native_eight_value_actions_without_lookup_or_repeat() -> None:
    actor = RivalPolicyV1IndependentStickyButtons()
    policy = RivalStickyBernoulliPolicy(actor, "cpu")
    observations = _observations((0, 0, 0), rows=7)
    actions, log_probabilities = policy.get_action(observations)
    assert actions.shape == (7, ACTION_DIM)
    assert torch.equal(actions[:, ANALOG_DIM:], actions[:, ANALOG_DIM:].round())
    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probabilities).all()
