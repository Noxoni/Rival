"""rlgym-ppo policy interface backed by the reconstructed Wisp student."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .teacher import EXPANDED_ACTION_COUNT, WispStudentActor, build_wisp_student


class StudentDiscretePolicy(nn.Module):
    """Drop-in replacement for rlgym-ppo's ``DiscreteFF`` policy."""

    def __init__(self, actor: WispStudentActor, device: str | torch.device) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.n_actions = actor.action_count

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
        action_log_probabilities = log_probabilities.gather(-1, actions)
        entropy = -(log_probabilities * probabilities).sum(dim=-1).mean()
        return action_log_probabilities.to(self.device), entropy.to(self.device)


def make_student_policy(
    device: str | torch.device = "cuda",
    action_count: int = EXPANDED_ACTION_COUNT,
) -> StudentDiscretePolicy:
    return StudentDiscretePolicy(build_wisp_student(action_count), device)


def attach_student_policy(ppo_learner, policy: StudentDiscretePolicy, learning_rate: float) -> None:
    """Install the Wisp-derived policy into an initialized rlgym-ppo PPOLearner."""
    ppo_learner.policy = policy
    ppo_learner.policy_optimizer = torch.optim.Adam(
        policy.parameters(), lr=learning_rate
    )
