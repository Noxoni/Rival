"""Independent sticky-Bernoulli button policy for RivalActionV1.

The physical controller remains the eight-value native 120-Hz RivalActionV1
contract.  This module changes only the actor-side button distribution: jump,
boost, and handbrake are independent Bernoulli variables whose probabilities
are conditioned on the previous *actually applied* controller row already
stored in RivalObsV1.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Bernoulli, Normal

from .v9_actions import (
    ACTION_DIM,
    ACTION_VERSION,
    ANALOG_DIM,
    ANALOG_FIELDS,
    BUTTON_FIELDS,
    CONTROLLER_FIELDS,
    LOG_STD_MAX,
    LOG_STD_MIN,
    PHYSICS_HZ,
    POLICY_HZ,
    TANH_EPSILON,
    TIMING_VERSION,
)
from .v9_observations import observation_schema_manifest


BUTTON_POLICY_VERSION = "RivalIndependentStickyBernoulliButtonsV1"
BUTTON_DIM = len(BUTTON_FIELDS)
BUTTON_PERSISTENCE = {
    "jump": 0.95,
    "boost": 0.90,
    "handbrake": 0.90,
}
BUTTON_PERSISTENCE_TENSOR = tuple(BUTTON_PERSISTENCE[name] for name in BUTTON_FIELDS)


def _history_layout() -> tuple[int, int, int, int]:
    manifest = observation_schema_manifest()
    history = manifest["block_slices"]["controller_history"]
    history_ticks, controller_size = manifest["entity_shapes"][
        "self_controller_history"
    ]
    return (
        int(history["start"]),
        int(history["end"]),
        int(history_ticks),
        int(controller_size),
    )


HISTORY_START, HISTORY_END, HISTORY_TICKS, CONTROLLER_SIZE = _history_layout()


def previous_applied_buttons(observations: torch.Tensor) -> torch.Tensor:
    """Read the newest self-controller buttons from RivalObsV1 history."""

    expected_size = int(observation_schema_manifest()["float_count"])
    if observations.ndim != 2 or observations.shape[-1] != expected_size:
        raise ValueError(
            f"Expected batched RivalObsV1 shape (N, {expected_size}), "
            f"got {tuple(observations.shape)}"
        )
    history = observations[..., HISTORY_START:HISTORY_END]
    self_width = HISTORY_TICKS * CONTROLLER_SIZE
    self_rows = history[..., :self_width].reshape(
        -1, HISTORY_TICKS, CONTROLLER_SIZE
    )
    buttons = self_rows[..., -1, ANALOG_DIM:ACTION_DIM]
    rounded = buttons.round()
    if not bool(torch.allclose(buttons, rounded, atol=1e-6, rtol=0.0)):
        raise ValueError("RivalObsV1 newest applied button history is not binary")
    if not bool(torch.all((rounded >= 0) & (rounded <= 1))):
        raise ValueError("RivalObsV1 newest applied buttons must be in {0, 1}")
    return rounded


def effective_button_probabilities(
    button_logits: torch.Tensor,
    previous_buttons: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return base and exact persistence-conditioned Bernoulli probabilities."""

    if button_logits.shape[-1] != BUTTON_DIM:
        raise ValueError(f"Expected three button logits, got {button_logits.shape}")
    if previous_buttons.shape != button_logits.shape:
        raise ValueError(
            "Previous button shape must match logits: "
            f"{previous_buttons.shape} != {button_logits.shape}"
        )
    persistence = torch.as_tensor(
        BUTTON_PERSISTENCE_TENSOR,
        dtype=button_logits.dtype,
        device=button_logits.device,
    )
    base = torch.sigmoid(button_logits)
    previous = previous_buttons.to(dtype=button_logits.dtype)
    # A convex probability mixture with persistence > 0.5 makes deterministic
    # thresholding an absorbing latch. Apply the same requested persistence as
    # a neutral-logit prior instead: a neutral output retains the previous bit
    # with probability ``persistence``, while sufficiently strong learned
    # logits can still cross 0.5 in either direction.
    persistence_log_odds = torch.logit(persistence)
    effective = torch.sigmoid(
        button_logits + (2.0 * previous - 1.0) * persistence_log_odds
    )
    return base, effective


def deterministic_transition_reachability() -> dict[str, Any]:
    """Prove the threshold reachability implied by the requested persistence."""

    rows: dict[str, Any] = {}
    for name, persistence in BUTTON_PERSISTENCE.items():
        policy_logit_threshold = math.log(persistence / (1.0 - persistence))
        rows[name] = {
            "persistence": persistence,
            "neutral_logit_effective_probability_previous_off": 1.0 - persistence,
            "neutral_logit_effective_probability_previous_on": persistence,
            "off_to_on_policy_logit_threshold": policy_logit_threshold,
            "on_to_off_policy_logit_threshold": -policy_logit_threshold,
            "off_to_on_base_probability_threshold": persistence,
            "on_to_off_base_probability_threshold": 1.0 - persistence,
            "deterministic_threshold": 0.5,
            "off_to_on_reachable": True,
            "on_to_off_reachable": True,
            "state_is_absorbing": False,
        }
    return {
        "formula": (
            "sigmoid(policy_logit + (2*previous_bit-1)*logit(persistence))"
        ),
        "correction_reason": (
            "The originally requested convex probability mixture is an absorbing "
            "deterministic latch whenever persistence exceeds 0.5. The log-odds prior "
            "preserves neutral-output stickiness and makes both transitions learnable."
        ),
        "deterministic_rule": "effective_probability >= 0.5",
        "initial_reset_buttons": [0, 0, 0],
        "buttons": rows,
        "all_button_states_absorbing": all(
            row["state_is_absorbing"] for row in rows.values()
        ),
        "deterministic_reset_policy_can_ever_enable_a_button": any(
            row["off_to_on_reachable"] for row in rows.values()
        ),
        "pathological_before_ppo": False,
    }


@dataclass(frozen=True)
class StickyHybridEntropy:
    analog_monte_carlo: torch.Tensor
    button_exact: torch.Tensor
    button_by_field: torch.Tensor

    @property
    def mixed(self) -> torch.Tensor:
        return self.analog_monte_carlo + self.button_exact


class RivalStickyBernoulliDistribution:
    """Five tanh-Gaussian axes plus three sticky Bernoulli buttons."""

    def __init__(
        self,
        analog_mean: torch.Tensor,
        analog_log_std: torch.Tensor,
        button_logits: torch.Tensor,
        previous_buttons: torch.Tensor,
    ) -> None:
        if analog_mean.shape[-1] != ANALOG_DIM:
            raise ValueError(f"Expected five analog means, got {analog_mean.shape}")
        bounded_log_std = torch.clamp(analog_log_std, LOG_STD_MIN, LOG_STD_MAX)
        if bounded_log_std.ndim == 1:
            bounded_log_std = bounded_log_std.expand_as(analog_mean)
        if bounded_log_std.shape != analog_mean.shape:
            raise ValueError(
                f"Analog log-std shape {bounded_log_std.shape} does not match means "
                f"{analog_mean.shape}"
            )
        self.analog_mean = analog_mean
        self.analog_log_std = bounded_log_std
        self.button_logits = button_logits
        self.previous_buttons = previous_buttons.to(dtype=analog_mean.dtype)
        self.base_probabilities, self.effective_probabilities = (
            effective_button_probabilities(button_logits, self.previous_buttons)
        )
        self.normal = Normal(analog_mean, bounded_log_std.exp())
        self.bernoulli = Bernoulli(probs=self.effective_probabilities)

    @staticmethod
    def _atanh(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = actions.clamp(-1.0 + TANH_EPSILON, 1.0 - TANH_EPSILON)
        pre_tanh = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
        return bounded, pre_tanh

    def analog_log_prob(self, analog_actions: torch.Tensor) -> torch.Tensor:
        bounded, pre_tanh = self._atanh(analog_actions)
        base = self.normal.log_prob(pre_tanh)
        log_jacobian = torch.log(
            torch.clamp(1.0 - bounded.square(), min=TANH_EPSILON)
        )
        return (base - log_jacobian).sum(dim=-1)

    @staticmethod
    def _validate_buttons(actions: torch.Tensor) -> torch.Tensor:
        buttons = actions[..., ANALOG_DIM:ACTION_DIM]
        rounded = buttons.round()
        if not bool(torch.allclose(buttons, rounded, atol=0.0, rtol=0.0)):
            raise ValueError("Stored RivalActionV1 button values must be exact 0/1 bits")
        if not bool(torch.all((rounded >= 0) & (rounded <= 1))):
            raise ValueError("Stored RivalActionV1 button values must be in {0, 1}")
        return rounded

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected RivalActionV1 shape (..., 8), got {actions.shape}")
        buttons = self._validate_buttons(actions)
        return self.analog_log_prob(actions[..., :ANALOG_DIM]) + self.bernoulli.log_prob(
            buttons
        ).sum(dim=-1)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        analog = torch.tanh(self.normal.rsample())
        buttons = self.bernoulli.sample().to(dtype=analog.dtype)
        actions = torch.cat((analog, buttons), dim=-1)
        return actions, self.log_prob(actions)

    def mode(self) -> torch.Tensor:
        analog = torch.tanh(self.analog_mean)
        buttons = (self.effective_probabilities >= 0.5).to(dtype=analog.dtype)
        return torch.cat((analog, buttons), dim=-1)

    def entropy(self, stored_actions: torch.Tensor) -> StickyHybridEntropy:
        analog = -self.analog_log_prob(stored_actions[..., :ANALOG_DIM]).mean()
        by_field = self.bernoulli.entropy().mean(dim=0)
        return StickyHybridEntropy(
            analog_monte_carlo=analog,
            button_exact=by_field.sum(),
            button_by_field=by_field,
        )

    def diagnostics(self) -> dict[str, torch.Tensor]:
        deterministic = (self.effective_probabilities >= 0.5).to(
            dtype=self.effective_probabilities.dtype
        )
        return {
            "base_probability": self.base_probabilities,
            "effective_probability": self.effective_probabilities,
            "distance_from_half": (self.effective_probabilities - 0.5).abs(),
            "deterministic_bit": deterministic,
            "previous_bit": self.previous_buttons,
        }


class RivalActionHeadV2(nn.Module):
    """The transferred analog head plus a new independent three-logit head."""

    def __init__(
        self,
        embedding_size: int,
        *,
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.analog_mean = nn.Linear(int(embedding_size), ANALOG_DIM)
        self.button_logits = nn.Linear(int(embedding_size), BUTTON_DIM)
        self.analog_log_std = nn.Parameter(
            torch.full((ANALOG_DIM,), float(initial_log_std), dtype=torch.float32)
        )
        nn.init.normal_(self.analog_mean.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.analog_mean.bias)
        nn.init.normal_(self.button_logits.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.button_logits.bias)

    def forward(
        self, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.analog_mean(embedding)
        log_std = torch.clamp(self.analog_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, self.button_logits(embedding)


class RivalStickyBernoulliPolicy(nn.Module):
    """rlgym-ppo policy wrapper using observation-accountable persistence."""

    def __init__(self, actor: nn.Module, device: str | torch.device) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.action_dim = ACTION_DIM
        self.last_entropy: dict[str, Any] = {}

    def distribution(
        self, observations: torch.Tensor | Any
    ) -> RivalStickyBernoulliDistribution:
        values = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        mean, log_std, logits = self.actor(values)
        return RivalStickyBernoulliDistribution(
            mean,
            log_std,
            logits,
            previous_applied_buttons(values),
        )

    @torch.no_grad()
    def get_action(
        self,
        observations: torch.Tensor | Any,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        if deterministic:
            actions = distribution.mode()
            log_probabilities = distribution.log_prob(actions)
        else:
            actions, log_probabilities = distribution.sample()
        return actions.detach().cpu(), log_probabilities.detach().cpu()

    def get_backprop_data(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        physical = actions.to(self.device, dtype=torch.float32)
        log_probabilities = distribution.log_prob(physical)
        entropy = distribution.entropy(physical)
        self.last_entropy = {
            "analog_monte_carlo": float(entropy.analog_monte_carlo.detach().cpu()),
            "button_exact": float(entropy.button_exact.detach().cpu()),
            "button_by_field": {
                name: float(entropy.button_by_field[index].detach().cpu())
                for index, name in enumerate(BUTTON_FIELDS)
            },
            "mixed": float(entropy.mixed.detach().cpu()),
        }
        return log_probabilities, entropy.mixed


def button_policy_metadata() -> dict[str, Any]:
    canonical = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return {
        "version": BUTTON_POLICY_VERSION,
        "physical_action_version": ACTION_VERSION,
        "controller_fields": list(CONTROLLER_FIELDS),
        "physics_hz": PHYSICS_HZ,
        "policy_hz": POLICY_HZ,
        "policy_decisions_per_physics_tick": 1,
        "repeat_action": False,
        "analog_fields": list(ANALOG_FIELDS),
        "button_fields": list(BUTTON_FIELDS),
        "button_distribution": "three_independent_sticky_bernoulli_log_odds_prior",
        "button_persistence": dict(BUTTON_PERSISTENCE),
        "persistence_formula": (
            "sigmoid(policy_logit + (2*previous_bit-1)*logit(persistence))"
        ),
        "persistence_correction": (
            "Replaces the absorbing convex-mixture form; neutral logits retain the "
            "requested stickiness while strong logits can transition both ways."
        ),
        "previous_button_source": "RivalObsV1 newest self actually-applied controller history",
        "ppo_log_probability": (
            "sum(five tanh-Gaussian log probabilities) plus sum(three effective-probability "
            "Bernoulli log probabilities)"
        ),
        "deterministic_rule": "effective_probability >= 0.5",
        "lookup_table": False,
        "macro_actions": False,
        "state_dependent_action_mask": False,
        "timing_delay_version": TIMING_VERSION,
        "source_sha256": hashlib.sha256(canonical).hexdigest(),
    }
