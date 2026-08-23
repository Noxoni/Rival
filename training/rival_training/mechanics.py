"""Portable Milestone 08 mechanics-actor artifacts and fingerprints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .checkpoint import portable_path
from .policy import MechanicsActor
from .teacher import sha256_file


MECHANICS_CHECKPOINT_FORMAT = "rival-milestone08-mechanics-actor-v1"


def mechanics_state_sha256(actor: MechanicsActor) -> str:
    """Hash names, dtypes, shapes and exact tensor bytes in stable key order."""
    digest = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def save_mechanics_actor(
    path: str | Path,
    actor: MechanicsActor,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cpu_actor = actor.to("cpu").eval()
    payload = {
        "format": MECHANICS_CHECKPOINT_FORMAT,
        "state_dict": cpu_actor.state_dict(),
        "metadata": dict(metadata),
        "state_sha256": mechanics_state_sha256(cpu_actor),
    }
    torch.save(payload, destination)
    reloaded, loaded_metadata = load_mechanics_actor(destination, device="cpu")
    exact_reload = mechanics_state_sha256(reloaded) == payload["state_sha256"]
    if not exact_reload:
        raise RuntimeError("Mechanics actor changed during checkpoint reload")
    return {
        "path": portable_path(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "state_sha256": payload["state_sha256"],
        "fresh_reload_exact": exact_reload,
        "metadata": loaded_metadata,
    }


def load_mechanics_actor(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[MechanicsActor, dict[str, Any]]:
    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("format") != MECHANICS_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported mechanics actor format: {payload.get('format')}")
    actor = MechanicsActor()
    actor.load_state_dict(payload["state_dict"], strict=True)
    actual = mechanics_state_sha256(actor)
    if actual != payload["state_sha256"]:
        raise RuntimeError(
            f"Mechanics actor state fingerprint mismatch: {actual} != {payload['state_sha256']}"
        )
    return actor.to(torch.device(device)).eval(), dict(payload.get("metadata") or {})


def export_mechanics_torchscript(
    actor: MechanicsActor,
    path: str | Path,
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cpu_actor = actor.to("cpu").eval()
    sample = torch.zeros(4, 432, dtype=torch.float32)
    with torch.inference_mode():
        expected = cpu_actor(sample)
    traced = torch.jit.trace(cpu_actor, sample)
    traced.save(str(destination))
    loaded = torch.jit.load(str(destination), map_location="cpu").eval()
    with torch.inference_mode():
        actual = loaded(sample)
    if not torch.equal(expected, actual):
        raise RuntimeError("Mechanics TorchScript export changed deterministic logits")
    return {
        "path": portable_path(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
        "state_sha256": mechanics_state_sha256(cpu_actor),
        "fresh_reload_exact_logits": True,
        "output_shape": list(actual.shape),
    }
