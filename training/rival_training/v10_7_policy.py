"""RivalPolicyV1 encoder/analog transfer with a new sticky button head."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import nn

from .v10_7_actions import RivalActionHeadV2, RivalStickyBernoulliPolicy
from .v9_policy import RivalStructuredEncoderV1


POLICY_VERSION = "RivalPolicyV1IndependentStickyButtonsV1"


class RivalPolicyV1IndependentStickyButtons(nn.Module):
    """Unchanged RivalPolicyV1 encoder plus transferred analog and new buttons."""

    def __init__(self, *, seed: int = 20261071) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.encoder = RivalStructuredEncoderV1()
            self.action_head = RivalActionHeadV2(self.encoder.output_width)

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.action_head(self.encoder(observations))


class InstrumentedRivalStickyBernoulliPolicy(RivalStickyBernoulliPolicy):
    """Central rollout policy with inference and button probability telemetry."""

    def __init__(
        self,
        actor: RivalPolicyV1IndependentStickyButtons,
        device: str | torch.device,
    ) -> None:
        super().__init__(actor, device)
        self.inference_samples: list[dict[str, Any]] = []

    @torch.no_grad()
    def get_action(
        self,
        observations,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(observations.shape[0])
        started = time.perf_counter()
        distribution = self.distribution(observations)
        if deterministic:
            actions = distribution.mode()
            log_probabilities = distribution.log_prob(actions)
        else:
            actions, log_probabilities = distribution.sample()
        elapsed = time.perf_counter() - started
        diagnostics = distribution.diagnostics()
        self.inference_samples.append(
            {
                "batch_size": batch_size,
                "wall_seconds": elapsed,
                "per_agent_microseconds": elapsed * 1e6 / max(batch_size, 1),
                "mean_base_probability": diagnostics["base_probability"]
                .mean(dim=0)
                .detach()
                .cpu()
                .tolist(),
                "mean_effective_probability": diagnostics["effective_probability"]
                .mean(dim=0)
                .detach()
                .cpu()
                .tolist(),
            }
        )
        return actions.detach().cpu(), log_probabilities.detach().cpu()

    def drain_inference_samples(self) -> list[dict[str, Any]]:
        samples = self.inference_samples
        self.inference_samples = []
        return samples
