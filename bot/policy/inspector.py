from __future__ import annotations

import time
from typing import Any, Callable

import torch

from .decision import ActionCandidate, ControllerAction, PolicyDecision, PolicyInference


class PolicyInspector:
    """Turns device-resident model output into a compact inspectable decision."""

    def __init__(self, top_n: int = 5) -> None:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        self.top_n = top_n

    @torch.inference_mode()
    def inspect(
        self,
        inference: PolicyInference,
        action_index: int,
        controller_action: Any,
        action_resolver: Callable[[int], Any],
        *,
        tick: int,
        game_time: float | None,
        timestamp_unix_ns: int | None = None,
    ) -> PolicyDecision:
        if inference.masked_logits.numel() == 0:
            raise ValueError("Cannot inspect an empty policy output")
        if action_index < 0 or action_index >= inference.masked_logits.numel():
            raise IndexError(f"Selected action index is out of range: {action_index}")

        legal_indices = torch.nonzero(inference.legal_mask, as_tuple=False).view(-1)
        if legal_indices.numel() == 0:
            raise ValueError("PolicyInference must contain at least one effective legal action")

        legal_logits = inference.masked_logits[legal_indices]
        legal_probabilities = torch.softmax(legal_logits, dim=0)
        k = min(self.top_n, int(legal_indices.numel()))
        top_logits, top_positions = torch.topk(legal_logits, k=k, largest=True, sorted=True)
        top_indices = legal_indices[top_positions]
        top_probabilities = legal_probabilities[top_positions]

        indices = [int(value) for value in top_indices.detach().cpu().tolist()]
        logits = [float(value) for value in top_logits.detach().cpu().tolist()]
        probabilities = [
            float(value) for value in top_probabilities.detach().cpu().tolist()
        ]
        candidates = tuple(
            ActionCandidate(
                action_index=index,
                controller_action=ControllerAction.from_action(action_resolver(index)),
                logit=logit,
                probability=probability,
            )
            for index, logit, probability in zip(indices, logits, probabilities)
        )

        selected_positions = torch.nonzero(
            legal_indices == action_index, as_tuple=False
        ).view(-1)
        if selected_positions.numel() != 1:
            raise ValueError(f"Selected action {action_index} is not legal")
        confidence = float(legal_probabilities[selected_positions[0]].item())
        margin = probabilities[0] - probabilities[1] if len(probabilities) > 1 else probabilities[0]

        return PolicyDecision(
            action_index=action_index,
            controller_action=ControllerAction.from_action(controller_action),
            raw_logits=inference.raw_logits,
            masked_logits=inference.masked_logits,
            legal_mask=inference.legal_mask,
            top_actions=candidates,
            confidence=confidence,
            margin=margin,
            tick=tick,
            timestamp_unix_ns=(
                time.time_ns() if timestamp_unix_ns is None else timestamp_unix_ns
            ),
            game_time=None if game_time is None else float(game_time),
            empty_mask_fallback=inference.empty_mask_fallback,
        )
