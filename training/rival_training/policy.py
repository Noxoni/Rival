"""rlgym-ppo policy interface backed by the reconstructed Wisp student."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch import nn

from .teacher import (
    EXPANDED_ACTION_COUNT,
    TEACHER_ACTION_COUNT,
    WispStudentActor,
    build_wisp_student,
)


APPENDED_PRIOR_VERSION = "RivalAppendedActionPriorV1"
MECHANICS_POLICY_VERSION = "RivalMechanicsPolicyV1"
MECHANICS_OBSERVATION_SIZE = 432
MECHANICS_ACTION_COUNT = 69


class StudentDiscretePolicy(nn.Module):
    """Drop-in replacement for rlgym-ppo's ``DiscreteFF`` policy."""

    def __init__(
        self,
        actor: WispStudentActor,
        device: str | torch.device,
        *,
        appended_logit_offset: float = 0.0,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.n_actions = actor.action_count
        self.register_buffer(
            "appended_logit_offset",
            torch.tensor(float(appended_logit_offset), dtype=torch.float32),
        )

    def set_appended_logit_offset(self, value: float) -> None:
        """Change the checkpointed exploration prior at a safe stage boundary."""
        converted = float(value)
        if not np.isfinite(converted):
            raise ValueError("Appended-action logit offset must be finite")
        self.appended_logit_offset.fill_(converted)

    def prior_state(self) -> dict[str, Any]:
        return {
            "version": APPENDED_PRIOR_VERSION,
            "legacy_action_count": TEACHER_ACTION_COUNT,
            "expanded_action_count": self.n_actions,
            "appended_logit_offset": float(self.appended_logit_offset.item()),
        }

    def logits(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(observations):
            observations = torch.as_tensor(observations, dtype=torch.float32)
        observations = observations.to(self.device, dtype=torch.float32)
        logits = self.actor(observations)
        if self.n_actions <= TEACHER_ACTION_COUNT:
            return logits
        # Do not mutate actor output in place: autograd must see exactly the same
        # additive prior in rollout sampling and PPO backpropability calculations.
        prior = torch.zeros_like(logits)
        prior[..., TEACHER_ACTION_COUNT:] = self.appended_logit_offset
        return logits + prior

    def get_output(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(observations), dim=-1)

    @torch.no_grad()
    def get_action(
        self,
        observations: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor | int, torch.Tensor | int]:
        probabilities = torch.clamp(self.get_output(observations), min=1e-11, max=1.0)
        probabilities = probabilities.view(-1, self.n_actions)
        if deterministic:
            actions = probabilities.argmax(dim=-1).cpu()
            if len(actions) == 1:
                return int(actions.item()), 0
            return actions, torch.zeros_like(actions, dtype=torch.float32)
        actions = torch.multinomial(probabilities, 1, replacement=True)
        log_probabilities = torch.log(probabilities).gather(-1, actions)
        return actions.flatten().cpu(), log_probabilities.flatten().cpu()

    def get_backprop_data(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions.long()
        probabilities = torch.clamp(self.get_output(observations), min=1e-11, max=1.0)
        probabilities = probabilities.view(-1, self.n_actions)
        log_probabilities = torch.log(probabilities)
        action_log_probabilities = log_probabilities.gather(-1, actions)
        entropy = -(log_probabilities * probabilities).sum(dim=-1).mean()
        return action_log_probabilities.to(self.device), entropy.to(self.device)


def make_student_policy(
    device: str | torch.device = "cuda",
    action_count: int = EXPANDED_ACTION_COUNT,
    *,
    appended_logit_offset: float = 0.0,
) -> StudentDiscretePolicy:
    return StudentDiscretePolicy(
        build_wisp_student(action_count),
        device,
        appended_logit_offset=appended_logit_offset,
    )


def attach_student_policy(ppo_learner, policy: StudentDiscretePolicy, learning_rate: float) -> None:
    """Install the Wisp-derived policy into an initialized rlgym-ppo PPOLearner."""
    ppo_learner.policy = policy
    ppo_learner.policy_optimizer = torch.optim.Adam(
        policy.parameters(), lr=learning_rate
    )


def normalize_bootstrap_actor_for_prior(
    actor: WispStudentActor,
    *,
    expected_bootstrap_bias: float = -12.0,
) -> WispStudentActor:
    """Move the bootstrap suppression from actor bias into an explicit prior.

    Adding ``expected_bootstrap_bias`` as the policy prior after this operation
    preserves every effective logit exactly. The separation lets later stages
    reach a genuine zero external prior without confusing a permanent -12 actor
    initialization with a learned policy preference.
    """
    output = actor.policy[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("Student actor output head must be Linear")
    appended = output.bias[TEACHER_ACTION_COUNT:]
    expected = torch.full_like(appended, float(expected_bootstrap_bias))
    if not torch.allclose(appended.detach(), expected, atol=1e-5, rtol=0.0):
        raise ValueError("Actor appended biases do not match the frozen bootstrap bias")
    with torch.no_grad():
        appended.sub_(float(expected_bootstrap_bias))
    return actor


def materialize_effective_actor(policy: StudentDiscretePolicy) -> WispStudentActor:
    """Return an actor whose raw logits exactly include the checkpointed prior."""
    actor = copy.deepcopy(policy.actor).to("cpu").eval()
    output = actor.policy[-1]
    if not isinstance(output, nn.Linear):
        raise TypeError("Student actor output head must be Linear")
    with torch.no_grad():
        output.bias[TEACHER_ACTION_COUNT:].add_(
            float(policy.appended_logit_offset.detach().cpu().item())
        )
    return actor


class MechanicsActor(nn.Module):
    """Independent M08 mechanics/recovery actor.

    This module intentionally contains no strategic-Wisp parameters.  The frozen
    strategic branch lives inside the dual-rate environment/runtime and therefore
    cannot accidentally enter the PPO optimizer through this policy.
    """

    def __init__(self, *, seed: int = 20260823) -> None:
        super().__init__()
        # Isolate initialization from the caller's global Torch RNG so checkpoint
        # reconstruction is deterministic without perturbing training seeds.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.policy = nn.Sequential(
                nn.Linear(MECHANICS_OBSERVATION_SIZE, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, MECHANICS_ACTION_COUNT),
            )
            for module in self.policy:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    nn.init.zeros_(module.bias)
            output = self.output_layer
            nn.init.normal_(output.weight, mean=0.0, std=0.005)
            nn.init.zeros_(output.bias)

    @property
    def output_layer(self) -> nn.Linear:
        output = self.policy[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("Mechanics actor output must be Linear")
        return output

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy(observations)


class MechanicsDiscretePolicy(nn.Module):
    """rlgym-ppo discrete-policy interface for the 69-output mechanics actor."""

    def __init__(
        self,
        actor: MechanicsActor,
        device: str | torch.device,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.n_actions = MECHANICS_ACTION_COUNT

    def logits(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(observations):
            observations = torch.as_tensor(observations, dtype=torch.float32)
        observations = observations.to(self.device, dtype=torch.float32)
        return self.actor(observations)

    def get_output(self, observations: np.ndarray | torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.logits(observations), dim=-1)

    @torch.no_grad()
    def get_action(
        self,
        observations: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor | int, torch.Tensor | int]:
        probabilities = torch.clamp(self.get_output(observations), min=1e-11, max=1.0)
        probabilities = probabilities.view(-1, self.n_actions)
        if deterministic:
            actions = probabilities.argmax(dim=-1).cpu()
            if len(actions) == 1:
                return int(actions.item()), 0
            return actions, torch.zeros_like(actions, dtype=torch.float32)
        actions = torch.multinomial(probabilities, 1, replacement=True)
        log_probabilities = torch.log(probabilities).gather(-1, actions)
        return actions.flatten().cpu(), log_probabilities.flatten().cpu()

    def get_backprop_data(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions.long()
        probabilities = torch.clamp(self.get_output(observations), min=1e-11, max=1.0)
        probabilities = probabilities.view(-1, self.n_actions)
        log_probabilities = torch.log(probabilities)
        selected = log_probabilities.gather(-1, actions)
        entropy = -(log_probabilities * probabilities).sum(dim=-1).mean()
        return selected.to(self.device), entropy.to(self.device)

    def prior_state(self) -> dict[str, Any]:
        return {
            "version": MECHANICS_POLICY_VERSION,
            "action_count": self.n_actions,
            "pass_index": 0,
            "pass_output_bias": float(
                self.actor.output_layer.bias[0].detach().cpu().item()
            ),
        }


def make_mechanics_policy(
    device: str | torch.device = "cuda:0",
    *,
    seed: int = 20260823,
) -> MechanicsDiscretePolicy:
    return MechanicsDiscretePolicy(MechanicsActor(seed=seed), device)


def attach_mechanics_policy(
    ppo_learner,
    policy: MechanicsDiscretePolicy,
    learning_rate: float,
) -> None:
    """Install only mechanics parameters into the PPO policy optimizer."""
    ppo_learner.policy = policy
    ppo_learner.policy_optimizer = torch.optim.Adam(
        policy.parameters(), lr=float(learning_rate)
    )


@torch.inference_mode()
def mechanics_prior_statistics(
    actor: MechanicsActor,
    observations: np.ndarray | torch.Tensor,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    selected_device = torch.device(device)
    actor = actor.to(selected_device).eval()
    values = torch.as_tensor(observations, dtype=torch.float32, device=selected_device)
    logits = actor(values)
    probabilities = torch.softmax(logits, dim=-1)
    pass_probability = probabilities[:, 0]
    deterministic = logits.argmax(dim=-1)
    conditional = probabilities[:, 1:] / torch.clamp(
        1.0 - pass_probability[:, None], min=1e-12
    )
    return {
        "observation_count": int(values.shape[0]),
        "mean_pass_probability": float(pass_probability.mean().item()),
        "minimum_pass_probability": float(pass_probability.min().item()),
        "maximum_pass_probability": float(pass_probability.max().item()),
        "mean_override_probability": float((1.0 - pass_probability).mean().item()),
        "deterministic_pass_rate": float((deterministic == 0).float().mean().item()),
        "deterministic_override_rate": float((deterministic != 0).float().mean().item()),
        "conditional_override_entropy": float(
            (-(conditional * torch.log(torch.clamp(conditional, min=1e-12))).sum(-1))
            .mean()
            .item()
        ),
        "finite": bool(torch.isfinite(logits).all().item()),
    }


def calibrate_mechanics_pass_prior(
    actor: MechanicsActor,
    observations: np.ndarray | torch.Tensor,
    *,
    target_override_probability: float,
    device: str | torch.device = "cpu",
    iterations: int = 60,
) -> dict[str, Any]:
    """Choose the PASS bias from measured natural-state probability mass."""
    target = float(target_override_probability)
    if not 0.0 < target < 1.0:
        raise ValueError("Target override probability must be between zero and one")
    selected_device = torch.device(device)
    actor = actor.to(selected_device).eval()
    values = torch.as_tensor(observations, dtype=torch.float32, device=selected_device)
    output = actor.output_layer
    with torch.no_grad():
        output.bias[0].zero_()
        base_logits = actor(values)
        other_logsumexp = torch.logsumexp(base_logits[:, 1:], dim=-1)
        pass_base = base_logits[:, 0]
        lower, upper = -20.0, 30.0
        for _ in range(int(iterations)):
            midpoint = (lower + upper) / 2.0
            pass_probability = torch.sigmoid(pass_base + midpoint - other_logsumexp)
            override = float((1.0 - pass_probability).mean().item())
            if override > target:
                lower = midpoint
            else:
                upper = midpoint
        calibrated = (lower + upper) / 2.0
        output.bias[0].fill_(calibrated)
    statistics = mechanics_prior_statistics(actor, values, device=selected_device)
    statistics.update(
        {
            "target_override_probability": target,
            "calibrated_pass_output_bias": float(calibrated),
            "absolute_target_error": abs(
                float(statistics["mean_override_probability"]) - target
            ),
        }
    )
    return statistics
