"""Recurrent RLBot runtime for the single-network unified Rival V5 policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "rival2_live"))
from runtime import (  # noqa: E402
    ACTIVE_PHASES,
    RESET_PHASES,
    Rival2LiveAdapter,
    Rival2LiveRuntime,
    _phase_name,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


class Rival2UnifiedLiveRuntime:
    """Maintain one GRU state and one deterministic decision per physics tick."""

    def __init__(self, model_path: Path, manifest_path: Path, field_info: Any):
        self.model_path = model_path.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "RIVAL2_RLBOT_RECURRENT_DEPLOY_V1":
            raise RuntimeError("unsupported Rival unified RLBot artifact format")
        if self.manifest["source"].get("runtime_router") is not False:
            raise RuntimeError("Rival unified deployment must not use a router")
        contracts = self.manifest["contracts"]
        if (
            int(contracts["physics_hz"]) != 120
            or int(contracts["policy_hz"]) != 120
            or int(contracts["hold_ticks"]) != 1
        ):
            raise RuntimeError("Rival unified deployment requires native 120 Hz control")
        artifact = self.manifest["artifact"]
        actual = {
            "size_bytes": self.model_path.stat().st_size,
            "sha256": _sha256(self.model_path),
        }
        expected = {
            "size_bytes": int(artifact["size_bytes"]),
            "sha256": artifact["sha256"],
        }
        if actual != expected:
            raise RuntimeError(f"Rival unified model identity mismatch: {actual} != {expected}")
        self.model = torch.jit.load(str(self.model_path), map_location="cpu").eval()
        self.adapter = Rival2LiveAdapter(self.manifest, field_info)
        hidden_shape = self.manifest["recurrent"]["input_hidden_shape"]
        self.hidden = torch.zeros(
            (int(hidden_shape[0]), 1, int(hidden_shape[2])), dtype=torch.float32
        )
        self.zero = np.zeros(8, dtype=np.float32)
        self.current_action = self.zero.copy()
        self.last_frame: int | None = None
        self.last_decision_frame: int | None = None
        self.pending_reset = True
        self.last_phase = "Inactive"
        self.last_score: tuple[int, int] | None = None
        self.decisions = 0
        self.recurrent_resets = 0
        self.duplicate_packets = 0
        self.missed_physics_ticks = 0
        with torch.inference_mode():
            warmup = torch.zeros(
                (1, int(self.manifest["observation"]["dimension"])),
                dtype=torch.float32,
            )
            output, next_hidden = self.model(warmup, self.hidden)
        Rival2LiveRuntime._validate_action(output[0].numpy())
        if tuple(next_hidden.shape) != tuple(self.hidden.shape):
            raise RuntimeError("Rival unified recurrent output shape mismatch")

    @staticmethod
    def _score(packet: Any) -> tuple[int, int]:
        return tuple(int(packet.teams[index].score) for index in (0, 1))

    def _reset(self, packet: Any) -> None:
        self.adapter.reset(packet)
        # Torch 2.13 marks tensors created under inference_mode as inference
        # tensors, which cannot later be mutated outside that context.
        self.hidden = torch.zeros(tuple(self.hidden.shape), dtype=torch.float32)
        self.current_action.fill(0.0)
        self.last_decision_frame = None
        self.pending_reset = False
        self.recurrent_resets += 1

    def step(self, packet: Any, team: int) -> np.ndarray:
        phase = _phase_name(packet)
        score = self._score(packet) if len(packet.teams) >= 2 else (0, 0)
        if self.last_score is not None and score != self.last_score:
            self.pending_reset = True
        self.last_score = score
        if phase in RESET_PHASES and self.last_phase not in RESET_PHASES:
            self.pending_reset = True
        if phase not in ACTIVE_PHASES:
            self.last_phase = phase
            return self.zero.copy()
        if not packet.balls:
            return self.zero.copy()

        frame = int(packet.match_info.frame_num)
        if self.last_frame is not None and frame == self.last_frame:
            self.duplicate_packets += 1
            return self.current_action.copy()
        if self.last_frame is None or frame < self.last_frame:
            self.pending_reset = True
        if self.pending_reset:
            self._reset(packet)
        elif self.last_frame is not None:
            delta = max(1, frame - self.last_frame)
            self.missed_physics_ticks += max(0, delta - 1)
            self.adapter.advance(packet, delta)

        self.last_frame = frame
        self.last_phase = phase
        observation = self.adapter.observation(packet)[0, int(team)]
        with torch.inference_mode():
            action, next_hidden = self.model(
                torch.from_numpy(observation[None, :]), self.hidden
            )
        row = np.ascontiguousarray(action[0].numpy(), dtype=np.float32)
        Rival2LiveRuntime._validate_action(row)
        if not bool(torch.isfinite(next_hidden).all().item()):
            raise RuntimeError("Rival unified recurrent state became nonfinite")
        self.current_action = row
        self.hidden = next_hidden.contiguous()
        self.adapter.memory.previous_action[0, int(team)] = row
        self.adapter.memory.clear_interval_events()
        self.last_decision_frame = frame
        self.decisions += 1
        return self.current_action.copy()

    def summary(self) -> dict[str, Any]:
        return {
            "format": self.manifest["format"],
            "source": self.manifest["source"],
            "artifact": self.manifest["artifact"],
            "decisions": self.decisions,
            "recurrent_resets": self.recurrent_resets,
            "duplicate_packets": self.duplicate_packets,
            "missed_physics_ticks": self.missed_physics_ticks,
            "hold_ticks": 1,
            "runtime_router": False,
            "wheel_contact_semantics": (
                "aggregate AirState.OnGround broadcast to four fields"
            ),
        }


__all__ = ["Rival2UnifiedLiveRuntime"]
