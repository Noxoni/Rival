"""Portable actor exports plus resumable rlgym-ppo checkpoint metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .config import REPOSITORY_ROOT
from .teacher import WispStudentActor, sha256_file


ACTOR_CHECKPOINT_FORMAT = "rival-student-actor-v1"


def portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def save_actor_checkpoint(
    path: str | Path,
    actor: WispStudentActor,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": ACTOR_CHECKPOINT_FORMAT,
        "action_count": actor.action_count,
        "actor_state_dict": {
            key: value.detach().cpu() for key, value in actor.state_dict().items()
        },
        "metadata": metadata or {},
    }
    torch.save(payload, destination)
    return {
        "path": portable_path(destination),
        "sha256": sha256_file(destination),
        "size_bytes": destination.stat().st_size,
        "format": ACTOR_CHECKPOINT_FORMAT,
        "action_count": actor.action_count,
    }


def load_actor_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[WispStudentActor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != ACTOR_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported actor checkpoint format: {payload.get('format')}")
    actor = WispStudentActor(int(payload["action_count"]))
    actor.load_state_dict(payload["actor_state_dict"])
    actor.to(device).eval()
    return actor, dict(payload.get("metadata", {}))


def save_ppo_state(
    directory: str | Path,
    ppo_learner,
    trainer_state: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    ppo_learner.save_to(str(destination))
    state_path = destination / "RIVAL_TRAINER_STATE.json"
    state_path.write_text(json.dumps(trainer_state, indent=2) + "\n", encoding="utf-8")
    files = {}
    for path in sorted(destination.iterdir()):
        if path.is_file():
            files[path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return {
        "directory": portable_path(destination),
        "files": files,
        "trainer_state": trainer_state,
    }


def load_ppo_state(directory: str | Path, ppo_learner) -> dict[str, Any]:
    source = Path(directory)
    ppo_learner.load_from(str(source))
    state = json.loads((source / "RIVAL_TRAINER_STATE.json").read_text(encoding="utf-8"))
    ppo_learner.cumulative_model_updates = int(state["cumulative_model_updates"])
    return state
