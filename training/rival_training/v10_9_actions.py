"""PPO-accountable AR(1) analog exploration for M10.9."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

import torch
from torch import nn
from torch.distributions import Bernoulli, Normal

from .v10_7_actions import (
    StickyHybridEntropy,
    effective_button_probabilities,
    previous_applied_buttons,
)
from .v9_actions import (
    ACTION_DIM,
    ANALOG_DIM,
    ANALOG_FIELDS,
    LOG_STD_MAX,
    LOG_STD_MIN,
    PHYSICS_HZ,
    TANH_EPSILON,
)


AR_EXPLORATION_VERSION = "RivalAR1AnalogExplorationV1"
AR_TAU_SECONDS = 0.075
AR_DT_SECONDS = 1.0 / PHYSICS_HZ
AR_RHO = math.exp(-AR_DT_SECONDS / AR_TAU_SECONDS)
AR_INNOVATION_STD = math.sqrt(1.0 - AR_RHO * AR_RHO)
PREVIOUS_EPSILON_START = ACTION_DIM
PREVIOUS_EPSILON_END = PREVIOUS_EPSILON_START + ANALOG_DIM
INITIAL_FLAG_INDEX = PREVIOUS_EPSILON_END
ROLLOUT_ACTION_DIM = ACTION_DIM + ANALOG_DIM + 1


def unpack_rollout_actions(
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if actions.shape[-1] != ROLLOUT_ACTION_DIM:
        raise ValueError(
            f"Expected M10.9 rollout action width {ROLLOUT_ACTION_DIM}, "
            f"got {actions.shape}"
        )
    physical = actions[..., :ACTION_DIM]
    previous_epsilon = actions[..., PREVIOUS_EPSILON_START:PREVIOUS_EPSILON_END]
    initial = actions[..., INITIAL_FLAG_INDEX : INITIAL_FLAG_INDEX + 1]
    rounded = initial.round()
    if not bool(torch.equal(initial, rounded)) or not bool(
        torch.all((rounded >= 0) & (rounded <= 1))
    ):
        raise ValueError("M10.9 AR initial flags must be exact binary values")
    return physical, previous_epsilon, rounded


def pack_rollout_actions(
    physical: torch.Tensor,
    previous_epsilon: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    if physical.shape[-1] != ACTION_DIM:
        raise ValueError("Physical RivalActionV1 must have exactly eight fields")
    if previous_epsilon.shape != physical.shape[:-1] + (ANALOG_DIM,):
        raise ValueError("Previous AR epsilon shape does not match physical actions")
    if initial.shape != physical.shape[:-1] + (1,):
        raise ValueError("AR initial flag shape does not match physical actions")
    return torch.cat((physical, previous_epsilon, initial), dim=-1)


@dataclass(frozen=True)
class ARSample:
    physical_action: torch.Tensor
    log_probability: torch.Tensor
    epsilon: torch.Tensor


class RivalARStickyBernoulliDistribution:
    """Conditional AR epsilon density plus unchanged sticky buttons."""

    def __init__(
        self,
        analog_mean: torch.Tensor,
        analog_log_std: torch.Tensor,
        button_logits: torch.Tensor,
        previous_buttons: torch.Tensor,
        previous_epsilon: torch.Tensor,
        initial: torch.Tensor,
    ) -> None:
        if analog_mean.shape[-1] != ANALOG_DIM:
            raise ValueError("M10.9 requires five independent analog means")
        bounded_log_std = torch.clamp(analog_log_std, LOG_STD_MIN, LOG_STD_MAX)
        if bounded_log_std.ndim == 1:
            bounded_log_std = bounded_log_std.expand_as(analog_mean)
        if bounded_log_std.shape != analog_mean.shape:
            raise ValueError("Analog log-std shape does not match analog means")
        if previous_epsilon.shape != analog_mean.shape:
            raise ValueError("Previous epsilon shape does not match analog means")
        if initial.shape != analog_mean.shape[:-1] + (1,):
            raise ValueError("Initial flag must have shape (batch, 1)")
        self.analog_mean = analog_mean
        self.analog_log_std = bounded_log_std
        self.analog_std = bounded_log_std.exp()
        self.button_logits = button_logits
        self.previous_buttons = previous_buttons.to(dtype=analog_mean.dtype)
        self.previous_epsilon = previous_epsilon.to(dtype=analog_mean.dtype)
        self.initial = initial.to(dtype=analog_mean.dtype)
        self.base_probabilities, self.effective_probabilities = (
            effective_button_probabilities(button_logits, self.previous_buttons)
        )
        self.bernoulli = Bernoulli(probs=self.effective_probabilities)
        continuation_mean = AR_RHO * self.previous_epsilon
        continuation_std = torch.full_like(analog_mean, AR_INNOVATION_STD)
        self.epsilon_mean = torch.where(
            self.initial > 0.5, torch.zeros_like(analog_mean), continuation_mean
        )
        self.epsilon_std = torch.where(
            self.initial > 0.5, torch.ones_like(analog_mean), continuation_std
        )
        self.epsilon_distribution = Normal(self.epsilon_mean, self.epsilon_std)

    @staticmethod
    def _atanh(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = actions.clamp(-1.0 + TANH_EPSILON, 1.0 - TANH_EPSILON)
        return bounded, 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))

    def epsilon_from_action(self, analog_actions: torch.Tensor) -> torch.Tensor:
        _, pre_tanh = self._atanh(analog_actions)
        return (pre_tanh - self.analog_mean) / self.analog_std

    @staticmethod
    def _validate_buttons(actions: torch.Tensor) -> torch.Tensor:
        buttons = actions[..., ANALOG_DIM:ACTION_DIM]
        rounded = buttons.round()
        if not bool(torch.equal(buttons, rounded)) or not bool(
            torch.all((rounded >= 0) & (rounded <= 1))
        ):
            raise ValueError("Stored RivalActionV1 buttons must be exact binary bits")
        return rounded

    def analog_log_prob(self, analog_actions: torch.Tensor) -> torch.Tensor:
        bounded, _ = self._atanh(analog_actions)
        epsilon = self.epsilon_from_action(bounded)
        epsilon_log_prob = self.epsilon_distribution.log_prob(epsilon)
        scale_jacobian = self.analog_log_std
        tanh_jacobian = torch.log(
            torch.clamp(1.0 - bounded.square(), min=TANH_EPSILON)
        )
        return (epsilon_log_prob - scale_jacobian - tanh_jacobian).sum(dim=-1)

    def log_prob(self, physical_actions: torch.Tensor) -> torch.Tensor:
        if physical_actions.shape[-1] != ACTION_DIM:
            raise ValueError("Executed RivalActionV1 must contain exactly eight values")
        buttons = self._validate_buttons(physical_actions)
        return self.analog_log_prob(
            physical_actions[..., :ANALOG_DIM]
        ) + self.bernoulli.log_prob(buttons).sum(dim=-1)

    def sample(self) -> ARSample:
        epsilon = self.epsilon_distribution.rsample()
        analog = torch.tanh(self.analog_mean + self.analog_std * epsilon)
        buttons = self.bernoulli.sample().to(dtype=analog.dtype)
        physical = torch.cat((analog, buttons), dim=-1)
        return ARSample(physical, self.log_prob(physical), epsilon)

    def mode(self) -> torch.Tensor:
        analog = torch.tanh(self.analog_mean)
        buttons = (self.effective_probabilities >= 0.5).to(dtype=analog.dtype)
        return torch.cat((analog, buttons), dim=-1)

    def entropy(self, physical_actions: torch.Tensor) -> StickyHybridEntropy:
        analog = -self.analog_log_prob(
            physical_actions[..., :ANALOG_DIM]
        ).mean()
        by_field = self.bernoulli.entropy().mean(dim=0)
        return StickyHybridEntropy(
            analog_monte_carlo=analog,
            button_exact=by_field.sum(),
            button_by_field=by_field,
        )

    def diagnostics(self) -> dict[str, torch.Tensor]:
        return {
            "base_probability": self.base_probabilities,
            "effective_probability": self.effective_probabilities,
            "deterministic_bit": (self.effective_probabilities >= 0.5).to(
                self.analog_mean.dtype
            ),
            "previous_bit": self.previous_buttons,
            "epsilon_conditional_mean": self.epsilon_mean,
            "epsilon_conditional_std": self.epsilon_std,
            "initial": self.initial,
        }


class RivalARStickyBernoulliPolicy(nn.Module):
    """Actor wrapper for replay-exact stateful AR exploration."""

    def __init__(self, actor: nn.Module, device: str | torch.device) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.action_dim = ACTION_DIM
        self.inference_samples: list[dict[str, Any]] = []
        self._evaluation_epsilon: torch.Tensor | None = None
        self._evaluation_initial = True
        self._evaluation_generator: torch.Generator | None = None

    def distribution(
        self,
        observations: torch.Tensor | Any,
        previous_epsilon: torch.Tensor | Any | None = None,
        initial: torch.Tensor | Any | None = None,
    ) -> RivalARStickyBernoulliDistribution:
        values = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        mean, log_std, logits = self.actor(values)
        if previous_epsilon is None:
            previous = torch.zeros_like(mean)
        else:
            previous = torch.as_tensor(
                previous_epsilon, dtype=torch.float32, device=self.device
            )
        if initial is None:
            initial_values = torch.ones(
                mean.shape[:-1] + (1,), dtype=torch.float32, device=self.device
            )
        else:
            initial_values = torch.as_tensor(
                initial, dtype=torch.float32, device=self.device
            )
        return RivalARStickyBernoulliDistribution(
            mean,
            log_std,
            logits,
            previous_applied_buttons(values),
            previous,
            initial_values,
        )

    def distribution_for_replay(
        self, observations: torch.Tensor, rollout_actions: torch.Tensor
    ) -> tuple[RivalARStickyBernoulliDistribution, torch.Tensor]:
        physical, previous, initial = unpack_rollout_actions(rollout_actions)
        return self.distribution(observations, previous, initial), physical

    @torch.no_grad()
    def sample_with_context(
        self,
        observations: torch.Tensor | Any,
        previous_epsilon: torch.Tensor | Any,
        initial: torch.Tensor | Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(observations.shape[0])
        started = time.perf_counter()
        distribution = self.distribution(observations, previous_epsilon, initial)
        sample = distribution.sample()
        elapsed = time.perf_counter() - started
        self.inference_samples.append(
            {
                "batch_size": batch_size,
                "wall_seconds": elapsed,
                "per_agent_microseconds": elapsed * 1e6 / max(batch_size, 1),
            }
        )
        return (
            sample.physical_action.detach().cpu(),
            sample.log_probability.detach().cpu(),
            sample.epsilon.detach().cpu(),
        )

    def reset_exploration(self, *, seed: int | None = None) -> None:
        self._evaluation_epsilon = None
        self._evaluation_initial = True
        if seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed))
            self._evaluation_generator = generator
        else:
            self._evaluation_generator = None

    def evaluation_distribution(
        self, observations: torch.Tensor | Any
    ) -> RivalARStickyBernoulliDistribution:
        """Expose the exact next conditional distribution for evaluation evidence."""

        values = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        if self._evaluation_epsilon is None or self._evaluation_epsilon.shape != (
            len(values),
            ANALOG_DIM,
        ):
            previous = torch.zeros(
                len(values), ANALOG_DIM, dtype=torch.float32, device=self.device
            )
            initial = torch.ones(
                len(values), 1, dtype=torch.float32, device=self.device
            )
        else:
            previous = self._evaluation_epsilon
            initial = torch.full(
                (len(values), 1),
                float(self._evaluation_initial),
                dtype=torch.float32,
                device=self.device,
            )
        return self.distribution(values, previous, initial)

    @torch.no_grad()
    def get_action(
        self,
        observations: torch.Tensor | Any,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        if deterministic:
            distribution = self.distribution(values)
            physical = distribution.mode()
            return physical.detach().cpu(), distribution.log_prob(physical).detach().cpu()
        if self._evaluation_epsilon is None or self._evaluation_epsilon.shape != (
            len(values),
            ANALOG_DIM,
        ):
            self._evaluation_epsilon = torch.zeros(
                len(values), ANALOG_DIM, dtype=torch.float32, device=self.device
            )
            self._evaluation_initial = True
        initial = torch.full(
            (len(values), 1),
            float(self._evaluation_initial),
            device=self.device,
        )
        distribution = self.distribution(values, self._evaluation_epsilon, initial)
        if self._evaluation_generator is None:
            sample = distribution.sample()
        else:
            innovation = torch.randn(
                distribution.analog_mean.shape,
                device=self.device,
                generator=self._evaluation_generator,
            )
            epsilon = (
                distribution.epsilon_mean
                + distribution.epsilon_std * innovation
            )
            analog = torch.tanh(
                distribution.analog_mean + distribution.analog_std * epsilon
            )
            button_uniform = torch.rand(
                distribution.effective_probabilities.shape,
                device=self.device,
                generator=self._evaluation_generator,
            )
            buttons = (
                button_uniform < distribution.effective_probabilities
            ).to(analog.dtype)
            physical = torch.cat((analog, buttons), dim=-1)
            sample = ARSample(physical, distribution.log_prob(physical), epsilon)
        self._evaluation_epsilon = sample.epsilon.detach()
        self._evaluation_initial = False
        return sample.physical_action.detach().cpu(), sample.log_probability.detach().cpu()

    def get_backprop_data(
        self, observations: torch.Tensor, rollout_actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution, physical = self.distribution_for_replay(
            observations, rollout_actions
        )
        log_probability = distribution.log_prob(physical)
        return log_probability, distribution.entropy(physical).mixed

    def drain_inference_samples(self) -> list[dict[str, Any]]:
        rows = self.inference_samples
        self.inference_samples = []
        return rows


def independent_ar_log_probability(
    *,
    analog_mean: torch.Tensor,
    analog_log_std: torch.Tensor,
    button_probabilities: torch.Tensor,
    physical_actions: torch.Tensor,
    previous_epsilon: torch.Tensor,
    initial: torch.Tensor,
) -> torch.Tensor:
    """Independent formula used only to cross-check distribution replay."""

    bounded = physical_actions[..., :ANALOG_DIM].clamp(
        -1.0 + TANH_EPSILON, 1.0 - TANH_EPSILON
    )
    pre_tanh = torch.atanh(bounded)
    log_std = torch.clamp(analog_log_std, LOG_STD_MIN, LOG_STD_MAX)
    if log_std.ndim == 1:
        log_std = log_std.expand_as(analog_mean)
    std = log_std.exp()
    epsilon = (pre_tanh - analog_mean) / std
    conditional_mean = torch.where(
        initial > 0.5,
        torch.zeros_like(epsilon),
        AR_RHO * previous_epsilon,
    )
    conditional_std = torch.where(
        initial > 0.5,
        torch.ones_like(epsilon),
        torch.full_like(epsilon, AR_INNOVATION_STD),
    )
    analog = (
        Normal(conditional_mean, conditional_std).log_prob(epsilon)
        - log_std
        - torch.log(torch.clamp(1.0 - bounded.square(), min=TANH_EPSILON))
    ).sum(dim=-1)
    buttons = physical_actions[..., ANALOG_DIM:ACTION_DIM]
    button = Bernoulli(probs=button_probabilities).log_prob(buttons).sum(dim=-1)
    return analog + button


def ar_metadata() -> dict[str, Any]:
    return {
        "version": AR_EXPLORATION_VERSION,
        "physics_hz": PHYSICS_HZ,
        "tau_seconds": AR_TAU_SECONDS,
        "dt_seconds": AR_DT_SECONDS,
        "rho": AR_RHO,
        "innovation_standard_deviation": AR_INNOVATION_STD,
        "analog_fields": list(ANALOG_FIELDS),
        "executed_action_width": ACTION_DIM,
        "rollout_record_width": ROLLOUT_ACTION_DIM,
        "rollout_auxiliary_fields": [
            *[f"previous_epsilon_{name}" for name in ANALOG_FIELDS],
            "ar_initial_transition",
        ],
        "initial_distribution": "stationary epsilon ~ Normal(0,1)",
        "continuation_distribution": (
            "epsilon_t | epsilon_t-1 ~ Normal(rho*epsilon_t-1, sqrt(1-rho^2))"
        ),
        "deterministic_action": "tanh(policy_mean); no AR term",
        "action_repeat": False,
    }
