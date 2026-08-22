"""Single-observation inference and TorchScript export seam for later RLBot use."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actions import build_expanded_action_table
from .checkpoint import load_actor_checkpoint, portable_path
from .teacher import WispStudentActor, sha256_file


class RivalInferenceSession:
    def __init__(
        self,
        actor: WispStudentActor,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        self.action_table = build_expanded_action_table()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> "RivalInferenceSession":
        actor, _ = load_actor_checkpoint(path, device)
        return cls(actor, device)

    @torch.inference_mode()
    def infer(
        self,
        observation: np.ndarray,
        *,
        mirror_x: bool = False,
    ) -> dict[str, Any]:
        observation_array = np.asarray(observation, dtype=np.float32)
        if observation_array.shape != (432,):
            raise ValueError(f"Expected observation shape (432,), got {observation_array.shape}")
        tensor = torch.from_numpy(observation_array).to(self.device).unsqueeze(0)
        logits = self.actor(tensor).squeeze(0)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Deployment actor emitted non-finite logits")
        action_index = int(torch.argmax(logits).item())
        controller = self.action_table[action_index].copy()
        if mirror_x:
            controller[[1, 3, 4]] *= -1
        return {
            "action_index": action_index,
            "controller_action": controller,
            "logits": logits.detach().cpu().numpy(),
        }


def export_torchscript(
    actor: WispStudentActor,
    path: str | Path,
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cpu_actor = actor.to("cpu").eval()
    scripted = torch.jit.script(cpu_actor)
    torch.jit.save(scripted, str(destination))
    return {
        "path": portable_path(destination),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "input_shape": [1, 432],
        "output_shape": [1, actor.action_count],
    }
