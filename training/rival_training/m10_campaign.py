"""Recoverable Milestone 10 campaign plumbing around the frozen v9 learner.

This module deliberately owns only campaign metadata, checkpoint lifecycle, and
reload verification.  Policy, environment, reward, action, observation, and PPO
semantics continue to live in the already-proven v9 modules.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any
import uuid

import numpy as np
import torch

from .v9_checkpoint import (
    DEFAULT_PILOT_CONFIG_PATH,
    config_sha256,
    load_m09_config,
    load_v9_checkpoint,
    portable_path,
    save_v9_checkpoint,
    sha256_file,
    verify_v9_checkpoint,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_M10_CONFIG_PATH = REPOSITORY_ROOT / "training/configs/milestone10.json"
M10_AUTHORITY_PATH = REPOSITORY_ROOT / "handoff/v10.0/M10_CAMPAIGN.json"
M09_FINAL_CHECKPOINT = REPOSITORY_ROOT / (
    "training/checkpoints/milestone09/"
    "gate13-20260823T200008Z/phase2/1680214"
)
M09_FINAL_STEPS = 1_680_214
M09_FINAL_SIMULATED_HOURS = 1.9446921296
M09_FINAL_ACTOR_SHA256 = (
    "12770f082c6cbe1fbab8809580dc775d1d78071825eb9481df4a16d9ee85fbe5"
)
M09_FINAL_MANIFEST_SHA256 = (
    "9b08b454f587248ed184a194aac9b57d904fe1c5ba5b8efe8e5b9e84c9ae469e"
)
AGENT_STEPS_PER_SIMULATED_HOUR = 864_000
M10_BOUNDARY_HOURS = (5, 10, 25, 50, 100)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(destination)


def load_m10_config(path: str | Path = DEFAULT_M10_CONFIG_PATH) -> dict[str, Any]:
    """Load M10 config and prove it is a campaign-only migration from M09."""

    config = load_m09_config(path)
    authority = _read_json(M10_AUTHORITY_PATH)
    source = load_m09_config(DEFAULT_PILOT_CONFIG_PATH)
    migration = m10_config_migration_report(source, config)
    expected_campaign = {
        "starting_cumulative_agent_steps": int(
            authority["resume"]["expected_cumulative_agent_steps"]
        ),
        "starting_simulated_game_hours": float(
            authority["resume"]["expected_simulated_game_hours"]
        ),
        "additional_agent_step_budget": int(authority["additional_budget"]["agent_steps"]),
        "additional_simulated_game_hour_budget": float(
            authority["additional_budget"]["simulated_game_hours"]
        ),
        "nominal_cumulative_agent_step_target": int(
            authority["additional_budget"]["nominal_cumulative_agent_steps"]
        ),
        "nominal_cumulative_simulated_game_hour_target": float(
            authority["additional_budget"]["nominal_cumulative_simulated_game_hours"]
        ),
        "evaluation_boundaries_added_simulated_hours": list(
            authority["evaluation_boundaries_added_simulated_hours"]
        ),
        "native_rlbot_boundaries_added_simulated_hours": list(
            authority["native_rlbot_boundaries_added_simulated_hours"]
        ),
        "rolling_recovery_checkpoints_to_keep": 2,
        "production_promotion_authorized": False,
    }
    checks = {
        "config_version_exact": config.get("config_version") == "RivalM10TrainingConfigV1",
        "campaign_id_exact": config.get("campaign_id") == authority["campaign_id"],
        "campaign_exact": config.get("campaign") == expected_campaign,
        "campaign_migration_only": migration["passed"],
        "resume_actor_identity_exact": authority["resume"]["expected_actor_sha256"]
        == M09_FINAL_ACTOR_SHA256,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Milestone 10 config/authority mismatch: {checks}")
    return config


def m10_config_migration_report(
    source: dict[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    """Prove M09 -> M10 changes campaign metadata and no learning semantics."""

    mutable_metadata = {"config_version", "campaign_id", "pilot", "campaign"}
    all_keys = set(source) | set(target)
    changed_keys = sorted(key for key in all_keys if source.get(key) != target.get(key))
    frozen_keys = sorted(all_keys - mutable_metadata)
    initial_migration = source.get("config_version") == "RivalM09TrainingConfigV2PilotCurriculum"
    same_m10_config = config_sha256(source) == config_sha256(target)
    allowed_initial_changes = sorted(mutable_metadata)
    checks = {
        "source_is_final_m09_or_same_m10": initial_migration or same_m10_config,
        "target_is_m10": target.get("config_version") == "RivalM10TrainingConfigV1",
        "changed_keys_are_campaign_metadata_only": (
            changed_keys == allowed_initial_changes if initial_migration else changed_keys == []
        ),
        "all_learning_semantics_exact": all(
            source.get(key) == target.get(key) for key in frozen_keys
        ),
        "m09_pilot_ceiling_removed": "pilot" not in target,
        "m10_campaign_present": isinstance(target.get("campaign"), dict),
    }
    report = {
        "source_config_version": source.get("config_version"),
        "source_config_sha256": config_sha256(source),
        "target_config_version": target.get("config_version"),
        "target_config_sha256": config_sha256(target),
        "changed_top_level_keys": changed_keys,
        "allowed_changed_top_level_keys": allowed_initial_changes if initial_migration else [],
        "frozen_top_level_keys": frozen_keys,
        "checks": checks,
    }
    report["passed"] = all(checks.values())
    if not report["passed"]:
        raise RuntimeError(f"Unauthorized M10 config migration: {report}")
    return report


def _state_error(left: Any, right: Any) -> tuple[bool, float]:
    """Return exact equality and maximum numeric error for nested torch state."""

    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, math.inf
        left_cpu = left.detach().cpu()
        right_cpu = right.detach().cpu()
        exact = bool(torch.equal(left_cpu, right_cpu))
        if left_cpu.numel() == 0:
            return exact, 0.0
        if left_cpu.dtype == torch.bool:
            return exact, 0.0 if exact else 1.0
        error = float((left_cpu.to(torch.float64) - right_cpu.to(torch.float64)).abs().max())
        return exact, error
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, math.inf
        exact = bool(np.array_equal(left, right))
        error = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        return exact, error
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, math.inf
        rows = [_state_error(left[key], right[key]) for key in left]
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            return False, math.inf
        rows = [_state_error(a, b) for a, b in zip(left, right, strict=True)]
    else:
        return left == right, 0.0 if left == right else math.inf
    return all(item[0] for item in rows), max((item[1] for item in rows), default=0.0)


def _all_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all()) if value.is_floating_point() else True
    if isinstance(value, np.ndarray):
        return bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else True
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def verify_checkpoint_reload_parity(
    directory: str | Path,
    *,
    expected_config: dict[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a full checkpoint twice and compare every learned/restored state."""

    source = Path(directory)
    first = load_v9_checkpoint(source, device=device, expected_config=expected_config)
    second = load_v9_checkpoint(source, device=device, expected_config=expected_config)
    actor_exact, actor_error = _state_error(first["actor"].state_dict(), second["actor"].state_dict())
    critic_exact, critic_error = _state_error(
        first["critic"].state_dict(), second["critic"].state_dict()
    )
    actor_optimizer_exact, actor_optimizer_error = _state_error(
        first["actor_optimizer"].state_dict(), second["actor_optimizer"].state_dict()
    )
    critic_optimizer_exact, critic_optimizer_error = _state_error(
        first["critic_optimizer"].state_dict(), second["critic_optimizer"].state_dict()
    )
    observations_exact, observation_error = _state_error(
        first["reload_observations"], second["reload_observations"]
    )
    held = torch.as_tensor(first["reload_observations"], dtype=torch.float32, device=device)
    first["actor"].eval()
    second["actor"].eval()
    first["critic"].eval()
    second["critic"].eval()
    with torch.inference_mode():
        first_actor_output = first["actor"](held)
        second_actor_output = second["actor"](held)
        actor_output_rows = [
            _state_error(left, right)
            for left, right in zip(first_actor_output, second_actor_output, strict=True)
        ]
        critic_output_exact, critic_output_error = _state_error(
            first["critic"](held), second["critic"](held)
        )
    actor_output_exact = all(row[0] for row in actor_output_rows)
    actor_output_error = max((row[1] for row in actor_output_rows), default=0.0)
    state = first["trainer_state"]
    checks = {
        "actor_state_exact": actor_exact,
        "critic_state_exact": critic_exact,
        "actor_optimizer_state_exact": actor_optimizer_exact,
        "critic_optimizer_state_exact": critic_optimizer_exact,
        "reload_observations_exact": observations_exact,
        "held_actor_outputs_exact": actor_output_exact,
        "held_critic_outputs_exact": critic_output_exact,
        "trainer_clean_boundary": state.get("clean_boundary") is True,
        "partial_experience_buffer_empty": int(state.get("partial_experience_buffer_records", -1))
        == 0,
        "all_model_optimizer_and_held_state_finite": all(
            _all_finite(item)
            for item in (
                first["actor"].state_dict(),
                first["critic"].state_dict(),
                first["actor_optimizer"].state_dict(),
                first["critic_optimizer"].state_dict(),
                first["reload_observations"],
            )
        ),
    }
    result = {
        "schema_version": 1,
        "checkpoint": checkpoint_record(source, manifest=first["manifest"]),
        "maximum_absolute_error": {
            "actor_state": actor_error,
            "critic_state": critic_error,
            "actor_optimizer_state": actor_optimizer_error,
            "critic_optimizer_state": critic_optimizer_error,
            "reload_observations": observation_error,
            "held_actor_outputs": actor_output_error,
            "held_critic_outputs": critic_output_error,
        },
        "checks": checks,
    }
    result["checks"]["passed"] = all(checks.values())
    if not result["checks"]["passed"]:
        raise RuntimeError(f"Checkpoint fresh-reload parity failed: {result}")
    return result


def verify_exact_m09_start_checkpoint(
    directory: str | Path = M09_FINAL_CHECKPOINT,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Verify the exact final M09 Gate 13 state required by M10 authority."""

    source = Path(directory)
    required = (
        "actor.pt",
        "critic.pt",
        "actor_optimizer.pt",
        "critic_optimizer.pt",
        "trainer_state.json",
        "training_config.json",
        "reload_observations.npy",
        "checkpoint_manifest.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Exact M09 Gate 13 checkpoint is incomplete: {missing}")
    manifest_sha = sha256_file(source / "checkpoint_manifest.json")
    if manifest_sha != M09_FINAL_MANIFEST_SHA256:
        raise RuntimeError(
            "M09 Gate 13 checkpoint manifest identity mismatch: "
            f"expected {M09_FINAL_MANIFEST_SHA256}, got {manifest_sha}"
        )
    parity = verify_checkpoint_reload_parity(source, device=device)
    identity = parity["checkpoint"]
    checks = {
        "manifest_sha256_exact": manifest_sha == M09_FINAL_MANIFEST_SHA256,
        "actor_sha256_exact": identity["actor_sha256"] == M09_FINAL_ACTOR_SHA256,
        "cumulative_agent_steps_exact": identity["cumulative_agent_steps"] == M09_FINAL_STEPS,
        "simulated_game_hours_exact_within_1e_9": abs(
            identity["simulated_game_hours"] - M09_FINAL_SIMULATED_HOURS
        )
        <= 1e-9,
        "fresh_reload_exact": parity["checks"]["passed"]
        and max(parity["maximum_absolute_error"].values()) == 0.0,
        "full_state_files_present": not missing,
    }
    result = {
        "schema_version": 1,
        "authority": "final_m09_gate13_checkpoint",
        "checkpoint": identity,
        "fresh_reload": parity,
        "checks": checks,
    }
    result["checks"]["passed"] = all(checks.values())
    if not result["checks"]["passed"]:
        raise RuntimeError(f"Exact M09 Gate 13 resume verification failed: {result}")
    return result


def checkpoint_record(
    directory: str | Path, *, manifest: dict[str, Any] | None = None
) -> dict[str, Any]:
    source = Path(directory)
    manifest = manifest or verify_v9_checkpoint(source)
    state = manifest["trainer_state"]
    return {
        "directory": portable_path(source),
        "format": manifest["format"],
        "manifest_sha256": sha256_file(source / "checkpoint_manifest.json"),
        "actor_sha256": manifest["files"]["actor.pt"]["sha256"],
        "actor_size_bytes": int(manifest["files"]["actor.pt"]["size_bytes"]),
        "config_sha256": manifest["contract"]["config_sha256"],
        "config_version": manifest["contract"]["config_version"],
        "cumulative_agent_steps": int(state["cumulative_agent_steps"]),
        "simulated_game_hours": float(state["simulated_game_hours"]),
        "completed_iterations": int(state["completed_iterations"]),
        "cumulative_model_updates": int(state["cumulative_model_updates"]),
        "clean_boundary": bool(state["clean_boundary"]),
    }


def save_checkpoint_atomic(
    directory: str | Path,
    *,
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    trainer_state: dict[str, Any],
    config: dict[str, Any],
    reload_observations: np.ndarray,
) -> dict[str, Any]:
    """Write and self-verify a checkpoint before atomically exposing it."""

    destination = Path(directory)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    try:
        save_v9_checkpoint(
            temporary,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            trainer_state=trainer_state,
            config=config,
            reload_observations=reload_observations,
        )
        verify_v9_checkpoint(temporary, expected_config=config)
        temporary.replace(destination)
        manifest_path = destination / "checkpoint_manifest.json"
        manifest = _read_json(manifest_path)
        manifest["directory"] = portable_path(destination)
        write_json_atomic(manifest_path, manifest)
        verified = verify_v9_checkpoint(destination, expected_config=config)
        return checkpoint_record(destination, manifest=verified)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def prune_rolling_checkpoints(root: str | Path, *, keep: int) -> list[str]:
    """Retain newest numeric rolling checkpoint directories with bounded deletion."""

    rolling_root = Path(root).resolve()
    if keep < 2:
        raise ValueError("M10 must retain at least two rolling recovery checkpoints")
    if not rolling_root.is_dir():
        return []
    candidates = []
    for child in rolling_root.iterdir():
        if child.is_symlink():
            raise RuntimeError(f"Refusing rolling retention across symlink: {child}")
        if child.is_dir() and child.name.isdecimal() and child.resolve().parent == rolling_root:
            candidates.append(child)
    candidates.sort(key=lambda item: int(item.name), reverse=True)
    removed: list[str] = []
    for child in candidates[keep:]:
        removed.append(portable_path(child))
        shutil.rmtree(child)
    return removed


def boundary_slug(added_hours: int | float) -> str:
    value = float(added_hours)
    if value not in M10_BOUNDARY_HOURS:
        raise ValueError(f"Unsupported M10 boundary: {added_hours}")
    return f"plus-{int(value):03d}h"


def nominal_boundary_steps(added_hours: int | float) -> int:
    return M09_FINAL_STEPS + int(round(float(added_hours) * AGENT_STEPS_PER_SIMULATED_HOUR))


def compact_training_iteration(report: dict[str, Any]) -> dict[str, Any]:
    ppo = report["ppo"]
    pilot = report["pilot_metrics"]
    return {
        "iteration": int(report["iteration"]),
        "collected_agent_steps": int(report["collected_agent_steps"]),
        "cumulative_agent_steps": int(report["cumulative_agent_steps"]),
        "simulated_game_hours": float(report["simulated_game_hours"]),
        "agent_steps_per_second": float(report["agent_steps_per_second"]),
        "collection_seconds": float(report["collection_seconds"]),
        "iteration_wall_seconds": float(report["iteration_wall_seconds"]),
        "rollout_wall_seconds": float(report["rollout_wall_seconds"]),
        "update_wall_seconds": float(report["update_wall_seconds"]),
        "cuda_peak_allocated_mib": float(report["cuda_peak_allocated_mib"]),
        "reward": report["reward"],
        "ppo": {
            "actor_loss": ppo["actor_loss"],
            "critic_loss": ppo["critic_loss"],
            "analog_entropy": ppo["analog_entropy"],
            "button_entropy": ppo["button_entropy"],
            "analog_log_std": ppo["analog_log_std"],
            "analog_std": ppo["analog_std"],
            "actor_gradient_norm": ppo["actor_gradient_norm"],
            "critic_gradient_norm": ppo["critic_gradient_norm"],
            "actor_update_magnitude": float(ppo["actor_update_magnitude"]),
            "critic_update_magnitude": float(ppo["critic_update_magnitude"]),
            "approximate_kl": ppo.get("approximate_kl"),
            "clip_fraction": ppo.get("clip_fraction"),
            "explained_variance_before_update": ppo.get("explained_variance_before_update"),
        },
        "actions": report["actions"],
        "rollout_inference": report["rollout_inference"],
        "reset_counts": pilot["reset_counts"],
        "scores_recorded_for_diagnostics_only": pilot[
            "scores_recorded_for_diagnostics_only"
        ],
        "reward_components": pilot["reward_components"],
        "event_counts": pilot["event_counts"],
        "health": report["health"],
    }
