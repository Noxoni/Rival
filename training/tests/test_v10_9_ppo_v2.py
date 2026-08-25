from __future__ import annotations

import math

import numpy as np
import torch

from rival_training.v10_7_policy import RivalPolicyV1IndependentStickyButtons
from rival_training.v10_7_actions import (
    ACTION_DIM,
    CONTROLLER_SIZE,
    HISTORY_END,
    HISTORY_START,
    HISTORY_TICKS,
)
from rival_training.v10_8_credit import gae_physical_time_report
from rival_training.v10_9_actions import (
    ANALOG_DIM,
    AR_DT_SECONDS,
    AR_INNOVATION_STD,
    AR_RHO,
    AR_TAU_SECONDS,
    RivalARStickyBernoulliPolicy,
    independent_ar_log_probability,
    pack_rollout_actions,
    unpack_rollout_actions,
)
from rival_training.v10_9_campaign import load_stage1_config
from rival_training.v10_9_trainer import scale_advantages
from rival_training.v9_policy import SCHEMA_LAYOUT


def _observations(rows: int = 64) -> torch.Tensor:
    torch.manual_seed(7901)
    observations = torch.randn(rows, SCHEMA_LAYOUT["observation_size"])
    history = observations[:, HISTORY_START:HISTORY_END]
    self_rows = history[:, : HISTORY_TICKS * CONTROLLER_SIZE].reshape(
        rows, HISTORY_TICKS, CONTROLLER_SIZE
    )
    self_rows[:, -1, ANALOG_DIM:ACTION_DIM] = 0.0
    return observations


def test_ar_constants_are_exact_physical_time_conversion() -> None:
    assert AR_DT_SECONDS == 1.0 / 120.0
    assert AR_TAU_SECONDS == 0.075
    assert AR_RHO == math.exp(-(1.0 / 120.0) / 0.075)
    assert AR_INNOVATION_STD == math.sqrt(1.0 - AR_RHO**2)


def test_rollout_record_round_trip_keeps_physical_action_separate() -> None:
    physical = torch.zeros(7, 8)
    previous = torch.randn(7, ANALOG_DIM)
    initial = torch.tensor([[1.0], [0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    packed = pack_rollout_actions(physical, previous, initial)
    replay_physical, replay_previous, replay_initial = unpack_rollout_actions(packed)
    assert packed.shape == (7, 14)
    assert torch.equal(replay_physical, physical)
    assert torch.equal(replay_previous, previous)
    assert torch.equal(replay_initial, initial)


def test_conditional_ar_log_probability_replays_and_cross_checks() -> None:
    actor = RivalPolicyV1IndependentStickyButtons()
    policy = RivalARStickyBernoulliPolicy(actor, "cpu")
    observations = _observations()
    previous = torch.randn(len(observations), ANALOG_DIM)
    initial = torch.zeros(len(observations), 1)
    initial[::11] = 1.0
    distribution = policy.distribution(observations, previous, initial)
    torch.manual_seed(7902)
    sample = distribution.sample()
    replay = distribution.log_prob(sample.physical_action)
    independent = independent_ar_log_probability(
        analog_mean=distribution.analog_mean,
        analog_log_std=distribution.analog_log_std,
        button_probabilities=distribution.effective_probabilities,
        physical_actions=sample.physical_action,
        previous_epsilon=previous,
        initial=initial,
    )
    assert torch.max(torch.abs(replay - sample.log_probability)).item() == 0.0
    assert torch.max(torch.abs(replay - independent)).item() <= 2e-5


def test_initial_distribution_is_stationary_and_continuation_is_conditional() -> None:
    policy = RivalARStickyBernoulliPolicy(
        RivalPolicyV1IndependentStickyButtons(), "cpu"
    )
    observations = _observations(5)
    previous = torch.full((5, ANALOG_DIM), 2.0)
    initial = torch.tensor([[1.0], [0.0], [1.0], [0.0], [0.0]])
    distribution = policy.distribution(observations, previous, initial)
    assert torch.all(distribution.epsilon_mean[initial[:, 0] == 1] == 0.0)
    assert torch.all(distribution.epsilon_std[initial[:, 0] == 1] == 1.0)
    assert torch.allclose(
        distribution.epsilon_mean[initial[:, 0] == 0],
        torch.full((3, ANALOG_DIM), 2.0 * AR_RHO),
    )
    assert torch.allclose(
        distribution.epsilon_std[initial[:, 0] == 0],
        torch.full((3, ANALOG_DIM), AR_INNOVATION_STD),
    )


def test_deterministic_output_contains_no_ar_term() -> None:
    policy = RivalARStickyBernoulliPolicy(
        RivalPolicyV1IndependentStickyButtons(), "cpu"
    )
    observations = _observations(8)
    left = policy.distribution(
        observations, torch.full((8, ANALOG_DIM), -10.0), torch.zeros(8, 1)
    )
    right = policy.distribution(
        observations, torch.full((8, ANALOG_DIM), 10.0), torch.zeros(8, 1)
    )
    assert torch.equal(left.mode()[:, :ANALOG_DIM], right.mode()[:, :ANALOG_DIM])
    assert torch.equal(left.mode()[:, :ANALOG_DIM], torch.tanh(left.analog_mean))


def test_scale_only_advantages_preserve_sign_zero_and_one_rollout_scale() -> None:
    raw = np.asarray([-4.0, -1.899, -1.212, 0.0, 0.5, 7.0], dtype=np.float32)
    scaled, scale = scale_advantages(raw, epsilon=1e-6)
    assert scale == float(raw.std())
    assert np.array_equal(np.signbit(raw[raw != 0]), np.signbit(scaled[raw != 0]))
    assert scaled[3] == 0.0
    assert np.isfinite(scaled).all()
    assert scaled[1] < 0.0
    assert scaled[2] < 0.0
    chunks = [raw[:2] / scale, raw[2:4] / scale, raw[4:] / scale]
    assert np.array_equal(np.concatenate(chunks), scaled)


def test_arm_c_horizon_and_ppo_v2_schedule_are_frozen() -> None:
    config = load_stage1_config()
    report = gae_physical_time_report(
        gamma=config["ppo"]["gamma"],
        gae_lambda=config["ppo"]["gae_lambda"],
        arm="M10.9",
    )
    assert report["checks"]["passed"]
    assert abs(report["half_life_seconds"] - 2.0) < 1e-12
    assert config["ppo"]["advantage_scaling"] == "scale_only_no_mean_centering"
    assert config["ppo"]["actor_epochs"] == 2
    assert config["ppo"]["critic_epochs"] == 8
    assert config["ppo"]["actor_minibatch_agent_steps"] == 8192
    assert config["ppo"]["critic_minibatch_agent_steps"] == 8192
    assert config["ppo"]["actor_kl_stop_threshold"] == 0.015
