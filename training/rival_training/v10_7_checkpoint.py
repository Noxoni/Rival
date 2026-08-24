"""Atomic checkpoints for the M10.7 sticky-button actor architecture."""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import torch

from .v10_7_actions import BUTTON_POLICY_VERSION, button_policy_metadata
from .v10_7_policy import POLICY_VERSION, RivalPolicyV1IndependentStickyButtons
from .v9_actions import ACTION_VERSION
from .v9_canonical import CANONICAL_ADAPTER_VERSION, CANONICAL_STATE_VERSION
from .v9_checkpoint import (
    action_schema_sha256,
    canonical_json_sha256,
    config_sha256,
    portable_path,
    sha256_file,
)
from .v9_observations import OBSERVATION_VERSION, observation_schema_manifest
from .v9_policy import CRITIC_VERSION, RivalCriticV1


CHECKPOINT_FORMAT = "rival-m10-7-sticky-bernoulli-ppo-checkpoint-v1"
MANIFEST_NAME = "checkpoint_manifest.json"


def checkpoint_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "config_version": config["config_version"],
        "config_sha256": config_sha256(config),
        "policy_version": POLICY_VERSION,
        "critic_version": CRITIC_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "observation_schema_sha256": observation_schema_manifest()["schema_sha256"],
        "physical_action_version": ACTION_VERSION,
        "physical_action_schema_sha256": action_schema_sha256(),
        "button_policy_version": BUTTON_POLICY_VERSION,
        "button_policy_metadata_sha256": canonical_json_sha256(button_policy_metadata()),
        "canonical_state_version": CANONICAL_STATE_VERSION,
        "canonical_adapter_version": CANONICAL_ADAPTER_VERSION,
        "reward_version": config["reward_version"],
        "reward_schedule_version": config["reward_schedule_version"],
        "environment_version": config["environment_version"],
        "backend": dict(config["backend"]),
    }


def _file_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != MANIFEST_NAME
    }


def save_checkpoint(
    directory: str | Path,
    *,
    actor: RivalPolicyV1IndependentStickyButtons,
    critic: RivalCriticV1,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any],
    config: dict[str, Any],
    reload_observations: np.ndarray,
) -> dict[str, Any]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    observations = np.asarray(reload_observations, dtype=np.float32)
    expected = int(observation_schema_manifest()["float_count"])
    if observations.ndim != 2 or observations.shape[1] != expected:
        raise ValueError(
            f"Reload observations must have shape (N, {expected}), got {observations.shape}"
        )
    contract = checkpoint_contract(config)
    torch.save(
        {"format": CHECKPOINT_FORMAT, "state_dict": actor.state_dict()},
        destination / "actor.pt",
    )
    torch.save(
        {"format": CHECKPOINT_FORMAT, "state_dict": critic.state_dict()},
        destination / "critic.pt",
    )
    torch.save(
        {"format": CHECKPOINT_FORMAT, "state_dict": actor_optimizer.state_dict()},
        destination / "actor_optimizer.pt",
    )
    torch.save(
        {"format": CHECKPOINT_FORMAT, "state_dict": critic_optimizer.state_dict()},
        destination / "critic_optimizer.pt",
    )
    np.save(destination / "reload_observations.npy", observations, allow_pickle=False)
    (destination / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    complete_state = {"format": CHECKPOINT_FORMAT, "contract": contract, **trainer_state}
    (destination / "trainer_state.json").write_text(
        json.dumps(complete_state, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "format": CHECKPOINT_FORMAT,
        "directory": portable_path(destination),
        "contract": contract,
        "trainer_state": complete_state,
        "files": _file_manifest(destination),
    }
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def verify_checkpoint(
    directory: str | Path,
    *,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(directory)
    manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported M10.7 checkpoint: {manifest.get('format')}")
    for name, expected in manifest["files"].items():
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file is missing: {name}")
        actual = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            raise RuntimeError(
                f"Checkpoint file verification failed for {name}: {actual} != {expected}"
            )
    stored_config = json.loads((source / "training_config.json").read_text(encoding="utf-8"))
    if manifest["contract"] != checkpoint_contract(stored_config):
        raise RuntimeError("M10.7 checkpoint contract differs from its config snapshot")
    if expected_config is not None and config_sha256(stored_config) != config_sha256(
        expected_config
    ):
        raise RuntimeError("M10.7 checkpoint config differs from requested config")
    return manifest


def load_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(directory)
    manifest = verify_checkpoint(source, expected_config=expected_config)
    selected = torch.device(device)
    actor = RivalPolicyV1IndependentStickyButtons().to(selected)
    critic = RivalCriticV1().to(selected)
    actor.load_state_dict(
        torch.load(source / "actor.pt", map_location=selected, weights_only=True)[
            "state_dict"
        ],
        strict=True,
    )
    critic.load_state_dict(
        torch.load(source / "critic.pt", map_location=selected, weights_only=True)[
            "state_dict"
        ],
        strict=True,
    )
    config = json.loads((source / "training_config.json").read_text(encoding="utf-8"))
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=float(config["ppo"]["actor_learning_rate"])
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["ppo"]["critic_learning_rate"])
    )
    actor_optimizer.load_state_dict(
        torch.load(
            source / "actor_optimizer.pt", map_location=selected, weights_only=True
        )["state_dict"]
    )
    critic_optimizer.load_state_dict(
        torch.load(
            source / "critic_optimizer.pt", map_location=selected, weights_only=True
        )["state_dict"]
    )
    return {
        "actor": actor,
        "critic": critic,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "trainer_state": manifest["trainer_state"],
        "config": config,
        "reload_observations": np.load(
            source / "reload_observations.npy", allow_pickle=False
        ),
        "manifest": manifest,
    }


def checkpoint_record(
    directory: str | Path, *, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    source = Path(directory)
    verified = manifest or verify_checkpoint(source)
    state = verified["trainer_state"]
    return {
        "directory": portable_path(source),
        "format": verified["format"],
        "manifest_sha256": sha256_file(source / MANIFEST_NAME),
        "actor_sha256": verified["files"]["actor.pt"]["sha256"],
        "actor_size_bytes": int(verified["files"]["actor.pt"]["size_bytes"]),
        "config_sha256": verified["contract"]["config_sha256"],
        "config_version": verified["contract"]["config_version"],
        "cumulative_agent_steps": int(state["cumulative_agent_steps"]),
        "simulated_game_hours": float(state["simulated_game_hours"]),
        "completed_iterations": int(state["completed_iterations"]),
        "cumulative_model_updates": int(state["cumulative_model_updates"]),
        "clean_boundary": bool(state["clean_boundary"]),
    }


def save_checkpoint_atomic(directory: str | Path, **kwargs: Any) -> dict[str, Any]:
    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    try:
        save_checkpoint(temporary, **kwargs)
        verify_checkpoint(temporary, expected_config=kwargs["config"])
        temporary.replace(destination)
        manifest_path = destination / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["directory"] = portable_path(destination)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return checkpoint_record(
            destination,
            manifest=verify_checkpoint(destination, expected_config=kwargs["config"]),
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _state_error(left: Any, right: Any) -> tuple[bool, float]:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, math.inf
        left_cpu = left.detach().cpu()
        right_cpu = right.detach().cpu()
        exact = bool(torch.equal(left_cpu, right_cpu))
        error = (
            0.0
            if left_cpu.numel() == 0 or exact
            else float(
                (left_cpu.to(torch.float64) - right_cpu.to(torch.float64)).abs().max()
            )
        )
        return exact, error
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        exact = left.shape == right.shape and left.dtype == right.dtype and np.array_equal(
            left, right
        )
        error = 0.0 if exact else math.inf
        return exact, error
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, math.inf
        rows = [_state_error(left[key], right[key]) for key in left]
        return all(row[0] for row in rows), max((row[1] for row in rows), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return False, math.inf
        rows = [_state_error(a, b) for a, b in zip(left, right, strict=True)]
        return all(row[0] for row in rows), max((row[1] for row in rows), default=0.0)
    exact = left == right
    return exact, 0.0 if exact else math.inf


def verify_reload_parity(
    directory: str | Path,
    *,
    expected_config: dict[str, Any],
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    source = Path(directory)
    first = load_checkpoint(source, device=device, expected_config=expected_config)
    second = load_checkpoint(source, device=device, expected_config=expected_config)
    comparisons = {
        "actor_state": _state_error(first["actor"].state_dict(), second["actor"].state_dict()),
        "critic_state": _state_error(first["critic"].state_dict(), second["critic"].state_dict()),
        "actor_optimizer_state": _state_error(
            first["actor_optimizer"].state_dict(), second["actor_optimizer"].state_dict()
        ),
        "critic_optimizer_state": _state_error(
            first["critic_optimizer"].state_dict(), second["critic_optimizer"].state_dict()
        ),
        "reload_observations": _state_error(
            first["reload_observations"], second["reload_observations"]
        ),
    }
    held = torch.as_tensor(first["reload_observations"], dtype=torch.float32, device=device)
    first["actor"].eval()
    second["actor"].eval()
    first["critic"].eval()
    second["critic"].eval()
    with torch.inference_mode():
        first_actor = first["actor"](held)
        second_actor = second["actor"](held)
        actor_rows = [
            _state_error(left, right)
            for left, right in zip(first_actor, second_actor, strict=True)
        ]
        comparisons["actor_outputs"] = (
            all(row[0] for row in actor_rows),
            max((row[1] for row in actor_rows), default=0.0),
        )
        comparisons["critic_outputs"] = _state_error(
            first["critic"](held), second["critic"](held)
        )
    state = first["trainer_state"]
    checks = {
        **{f"{name}_exact": row[0] for name, row in comparisons.items()},
        "trainer_clean_boundary": state.get("clean_boundary") is True,
        "partial_experience_buffer_empty": int(
            state.get("partial_experience_buffer_records", -1)
        )
        == 0,
    }
    checks["passed"] = all(checks.values())
    result = {
        "schema_version": 1,
        "checkpoint": checkpoint_record(source, manifest=first["manifest"]),
        "maximum_absolute_error": {
            name: row[1] for name, row in comparisons.items()
        },
        "checks": checks,
    }
    if not checks["passed"]:
        raise RuntimeError(f"M10.7 checkpoint reload parity failed: {result}")
    return result
