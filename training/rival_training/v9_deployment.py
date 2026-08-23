"""Versioned RivalPolicyV1 export and deployment-contract helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .v9_actions import ACTION_VERSION
from .v9_canonical import CANONICAL_ADAPTER_VERSION, CANONICAL_STATE_VERSION
from .v9_checkpoint import (
    action_schema_sha256,
    load_v9_checkpoint,
    portable_path,
    sha256_file,
)
from .v9_observations import (
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    observation_schema_manifest,
)
from .v9_policy import POLICY_VERSION, RivalPolicyV1
from .v9_rewards import REWARD_SCHEDULE_VERSION, REWARD_VERSION


EXPORT_FORMAT = "rival-v9-torchscript-deterministic-controller-v1"
EXPORT_SCHEMA_VERSION = 1
DEPLOYMENT_WRAPPER_VERSION = "RivalV9DeterministicControllerV1"
TRAINING_CONFIG_VERSION = "RivalM09TrainingConfigV1"
CONTROLLER_FIELDS = (
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "jump",
    "boost",
    "handbrake",
)


def canonical_source_sha256(path: str | Path) -> str:
    canonical = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


class RivalV9DeterministicController(nn.Module):
    """Export-only wrapper exposing parameters and the physical mode action."""

    def __init__(self, actor: RivalPolicyV1) -> None:
        super().__init__()
        self.actor = actor

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        analog_mean, analog_log_std, button_logits = self.actor(observations)
        combo = torch.argmax(button_logits, dim=-1)
        button_bits = torch.stack(
            (
                torch.bitwise_and(combo, 1),
                torch.bitwise_and(torch.bitwise_right_shift(combo, 1), 1),
                torch.bitwise_and(torch.bitwise_right_shift(combo, 2), 1),
            ),
            dim=-1,
        ).to(dtype=analog_mean.dtype)
        controller = torch.cat((torch.tanh(analog_mean), button_bits), dim=-1)
        return (
            analog_mean,
            analog_log_std.expand_as(analog_mean),
            button_logits,
            controller,
        )


def _reference_outputs(
    wrapper: RivalV9DeterministicController,
    observations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        outputs = wrapper(torch.from_numpy(observations))
    return tuple(
        np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        for value in outputs
    )


def _maximum_errors(
    expected: tuple[np.ndarray, ...],
    actual: tuple[np.ndarray, ...],
) -> dict[str, float]:
    names = ("analog_mean", "analog_log_std", "button_logits", "controller")
    return {
        name: float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        for name, left, right in zip(names, expected, actual)
    }


def export_v9_deployment(
    checkpoint_directory: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Export both practical formats and select the verified TorchScript seam."""

    checkpoint_path = Path(checkpoint_directory).resolve()
    destination = Path(output_directory).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    loaded = load_v9_checkpoint(checkpoint_path, device="cpu")
    actor = loaded["actor"].eval()
    observations = np.ascontiguousarray(
        loaded["reload_observations"], dtype=np.float32
    )
    if observations.shape != (32, OBSERVATION_SIZE):
        raise RuntimeError(
            "Gate 12 requires the checkpoint's 32-row held observation corpus; "
            f"got {observations.shape}"
        )
    wrapper = RivalV9DeterministicController(actor).eval()
    expected = _reference_outputs(wrapper, observations)

    example = torch.from_numpy(observations[:1])
    traced = torch.jit.trace(wrapper, example, strict=True)
    selected = torch.jit.optimize_for_inference(torch.jit.freeze(traced))
    selected_path = destination / "rival_v9_scratch.ts"
    torch.jit.save(selected, str(selected_path))

    modern_path = destination / "rival_v9_scratch.pt2"
    exported = torch.export.export(wrapper, (example,), strict=True)
    torch.export.save(exported, modern_path)

    reference_path = destination / "held_export_reference.npz"
    np.savez(
        reference_path,
        observations=observations,
        analog_mean=expected[0],
        analog_log_std=expected[1],
        button_logits=expected[2],
        controller=expected[3],
    )

    with torch.inference_mode():
        selected_outputs = selected(torch.from_numpy(observations))
    selected_arrays = tuple(
        np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        for value in selected_outputs
    )
    selected_errors = _maximum_errors(expected, selected_arrays)
    if max(selected_errors.values()) > 1e-5:
        raise RuntimeError(
            f"TorchScript export parity exceeded 1e-5: {selected_errors}"
        )

    config = loaded["config"]
    if config.get("config_version") != TRAINING_CONFIG_VERSION:
        raise RuntimeError("Checkpoint training-config version is not deployable v9")
    manifest_path = checkpoint_path / "checkpoint_manifest.json"
    model_record = {
        "path": portable_path(selected_path),
        "size_bytes": selected_path.stat().st_size,
        "sha256": sha256_file(selected_path),
    }
    metadata = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "format": EXPORT_FORMAT,
        "artifact": model_record,
        "contract": {
            "export_format": EXPORT_FORMAT,
            "deployment_wrapper_version": DEPLOYMENT_WRAPPER_VERSION,
            "policy_version": POLICY_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "observation_schema_sha256": observation_schema_manifest()[
                "schema_sha256"
            ],
            "observation_size": OBSERVATION_SIZE,
            "action_version": ACTION_VERSION,
            "action_schema_sha256": action_schema_sha256(),
            "canonical_state_version": CANONICAL_STATE_VERSION,
            "canonical_adapter_version": CANONICAL_ADAPTER_VERSION,
            "training_config_version": config["config_version"],
            "reward_version": REWARD_VERSION,
            "reward_schedule_version": REWARD_SCHEDULE_VERSION,
            "physics_hz": 120,
            "policy_hz": 120,
            "prediction_refresh_ticks": 1,
            "repeat_action": False,
            "controller_fields": list(CONTROLLER_FIELDS),
            "output_tensors": [
                "analog_mean",
                "analog_log_std",
                "button_logits",
                "controller",
            ],
            "deterministic_action": "tanh(analog_mean) plus categorical argmax",
            "state_dependent_action_mask": False,
        },
        "source_checkpoint": {
            "directory": portable_path(checkpoint_path),
            "checkpoint_manifest_sha256": sha256_file(manifest_path),
            "checkpoint_manifest_size_bytes": manifest_path.stat().st_size,
            "actor_sha256": loaded["manifest"]["files"]["actor.pt"]["sha256"],
            "actor_size_bytes": loaded["manifest"]["files"]["actor.pt"][
                "size_bytes"
            ],
            "cumulative_agent_steps": loaded["trainer_state"][
                "cumulative_agent_steps"
            ],
            "simulated_game_hours": loaded["trainer_state"][
                "simulated_game_hours"
            ],
        },
        "held_corpus": {
            "path": portable_path(reference_path),
            "sha256": sha256_file(reference_path),
            "size_bytes": reference_path.stat().st_size,
            "observations": int(observations.shape[0]),
            "observation_size": int(observations.shape[1]),
            "source": "Gate 11 clean-boundary reload corpus",
        },
        "selection": {
            "selected_format": "TorchScript frozen optimize_for_inference",
            "selected_for_live_runtime": True,
            "reason": (
                "Known Windows CPU loading seam; Gate 12 records a fresh-process "
                "latency/parity comparison with torch.export before final acceptance."
            ),
            "torch_export_candidate": {
                "path": portable_path(modern_path),
                "sha256": sha256_file(modern_path),
                "size_bytes": modern_path.stat().st_size,
            },
        },
        "export_parity": {
            "tolerance": 1e-5,
            "maximum_absolute_errors": selected_errors,
            "passed": max(selected_errors.values()) <= 1e-5,
        },
        "runtime_requirements": {
            "device": "cpu",
            "training_virtual_environment_required": False,
            "opt_in_only": True,
            "production_default_unchanged": True,
        },
        "source_hashes": {
            "deployment_source_sha256": canonical_source_sha256(Path(__file__)),
            "policy_source_sha256": canonical_source_sha256(
                Path(__file__).with_name("v9_policy.py")
            ),
            "observation_source_sha256": canonical_source_sha256(
                Path(__file__).with_name("v9_observations.py")
            ),
            "canonical_source_sha256": canonical_source_sha256(
                Path(__file__).with_name("v9_canonical.py")
            ),
        },
        "torch_version": torch.__version__,
    }
    metadata_path = destination / "rival_v9_scratch.metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "metadata": metadata,
        "metadata_path": portable_path(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "selected_path": portable_path(selected_path),
        "modern_path": portable_path(modern_path),
        "reference_path": portable_path(reference_path),
    }
