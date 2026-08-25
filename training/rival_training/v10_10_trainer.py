"""M10.10 binding of the proven PPO V2 trainer to the minimal environment."""

from __future__ import annotations

from typing import Callable

import torch

from .v10_10_environment import make_first_touch_velocity_phase_a_env
from .v10_9_trainer import RivalV10_9PPOTrainer


class RivalV10_10PPOTrainer(RivalV10_9PPOTrainer):
    """Keep PPO V2 exact while changing only environment reward/termination."""

    def __init__(
        self,
        config,
        *,
        device: str | torch.device = "cuda:0",
        actor,
        critic,
        actor_optimizer,
        critic_optimizer,
        trainer_state=None,
        env_factory: Callable | None = None,
    ) -> None:
        super().__init__(
            config,
            device=device,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            trainer_state=trainer_state,
            env_factory=env_factory or make_first_touch_velocity_phase_a_env,
        )
