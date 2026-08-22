from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch


def _json_float(value: Any) -> float | None:
    result = float(value)
    return result if math.isfinite(result) else None


def _tensor_float_list(tensor: torch.Tensor) -> list[float | None]:
    return [_json_float(value) for value in tensor.detach().cpu().view(-1).tolist()]


@dataclass(frozen=True)
class ControllerAction:
    """Serializable controller values corresponding to one discrete policy action."""

    throttle: float
    steer: float
    pitch: float
    yaw: float
    roll: float
    jump: bool
    boost: bool
    handbrake: bool

    @classmethod
    def from_action(cls, action: Any) -> "ControllerAction":
        return cls(
            throttle=float(action.throttle),
            steer=float(action.steer),
            pitch=float(action.pitch),
            yaw=float(action.yaw),
            roll=float(action.roll),
            jump=bool(action.jump),
            boost=bool(action.boost),
            handbrake=bool(action.handbrake),
        )

    def to_record(self) -> dict[str, float | bool]:
        return {
            "throttle": self.throttle,
            "steer": self.steer,
            "pitch": self.pitch,
            "yaw": self.yaw,
            "roll": self.roll,
            "jump": self.jump,
            "boost": self.boost,
            "handbrake": self.handbrake,
        }


@dataclass(frozen=True)
class ActionCandidate:
    action_index: int
    controller_action: ControllerAction
    logit: float
    probability: float

    def to_record(self) -> dict[str, Any]:
        return {
            "action_index": self.action_index,
            "controller_action": self.controller_action.to_record(),
            "logit": self.logit,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class PolicyInference:
    """Device-resident policy tensors before controller-action parsing.

    The tensors stay on the inference device. Full CPU materialization happens only
    when verbose telemetry explicitly requests it.
    """

    raw_logits: torch.Tensor
    masked_logits: torch.Tensor
    legal_mask: torch.Tensor
    empty_mask_fallback: bool = False

    def select_action(self, deterministic: bool = True) -> int:
        if deterministic:
            return int(torch.argmax(self.masked_logits).item())

        from torch.distributions.categorical import Categorical

        return int(Categorical(logits=self.masked_logits).sample().item())


@dataclass(frozen=True)
class PolicyDecision:
    """One complete, inspectable Rival policy decision."""

    action_index: int
    controller_action: ControllerAction
    raw_logits: torch.Tensor
    masked_logits: torch.Tensor
    legal_mask: torch.Tensor
    top_actions: tuple[ActionCandidate, ...]
    confidence: float
    margin: float
    tick: int
    timestamp_unix_ns: int
    game_time: float | None
    empty_mask_fallback: bool = False

    def to_record(self, *, include_logits: bool = False) -> dict[str, Any]:
        record: dict[str, Any] = {
            "action_index": self.action_index,
            "controller_action": self.controller_action.to_record(),
            "legal_mask": [
                bool(value)
                for value in self.legal_mask.detach().cpu().view(-1).tolist()
            ],
            "top_actions": [candidate.to_record() for candidate in self.top_actions],
            "confidence": self.confidence,
            "margin": self.margin,
            "tick": self.tick,
            "timestamp_unix_ns": self.timestamp_unix_ns,
            "game_time": self.game_time,
            "empty_mask_fallback": self.empty_mask_fallback,
        }
        if include_logits:
            record["raw_logits"] = _tensor_float_list(self.raw_logits)
            record["masked_logits"] = _tensor_float_list(self.masked_logits)
        return record
