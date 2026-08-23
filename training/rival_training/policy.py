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
