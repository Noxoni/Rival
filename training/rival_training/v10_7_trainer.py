"""M10.7 PPO trainer binding for independent sticky Bernoulli buttons."""

from __future__ import annotations

from typing import Any

import torch

from .v10_7_campaign import button_entropy_coefficient
from .v10_7_checkpoint import load_checkpoint
from .v10_7_policy import (
    InstrumentedRivalStickyBernoulliPolicy,
    RivalPolicyV1IndependentStickyButtons,
)
from .v9_policy import RivalCriticV1
from .v9_trainer import RivalV9PPOTrainer


class RivalV10_7PPOTrainer(RivalV9PPOTrainer):
    """Use the proven PPO loop with the corrected button distribution."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        device: str | torch.device = "cuda:0",
        actor: RivalPolicyV1IndependentStickyButtons | None = None,
        critic: RivalCriticV1 | None = None,
        actor_optimizer: torch.optim.Optimizer | None = None,
        critic_optimizer: torch.optim.Optimizer | None = None,
        trainer_state: dict[str, Any] | None = None,
        env_factory=None,
    ) -> None:
        super().__init__(
            config,
            device=device,
            actor=actor or RivalPolicyV1IndependentStickyButtons(),
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            trainer_state=trainer_state,
            env_factory=env_factory,
            policy_factory=InstrumentedRivalStickyBernoulliPolicy,
            button_entropy_coefficient_fn=lambda step: button_entropy_coefficient(
                step, config
            ),
        )

    @classmethod
    def from_checkpoint(
        cls,
        directory: str,
        config: dict[str, Any],
        *,
        device: str | torch.device = "cuda:0",
        env_factory=None,
    ) -> "RivalV10_7PPOTrainer":
        loaded = load_checkpoint(
            directory, device=device, expected_config=config
        )
        return cls(
            config,
            device=device,
            actor=loaded["actor"],
            critic=loaded["critic"],
            actor_optimizer=loaded["actor_optimizer"],
            critic_optimizer=loaded["critic_optimizer"],
            trainer_state=loaded["trainer_state"],
            env_factory=env_factory,
        )
