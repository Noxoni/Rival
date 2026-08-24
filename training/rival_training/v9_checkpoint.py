"""Versioned, resumable checkpoints for the scratch Rival v9 trainer."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .v9_actions import ACTION_VERSION, action_metadata
from .v9_canonical import CANONICAL_ADAPTER_VERSION, CANONICAL_STATE_VERSION
from .v9_curriculum import (
    V9_PILOT_CURRICULUM_VERSION,
    V9_PILOT_CURRICULUM_WEIGHTS,
)
from .v9_environment import (
    V9_PILOT_ENVIRONMENT_VERSION,
    V9_TRAINING_ENVIRONMENT_VERSION,
)
from .v9_metrics import V9_PILOT_METRICS_VERSION
from .v9_observations import OBSERVATION_VERSION, observation_schema_manifest
from .v9_policy import (
    CRITIC_VERSION,
    POLICY_VERSION,
    RivalCriticV1,
    RivalPolicyV1,
)
from .v9_rewards import REWARD_SCHEDULE_VERSION, REWARD_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "training/configs/milestone09.json"
DEFAULT_PILOT_CONFIG_PATH = REPOSITORY_ROOT / "training/configs/milestone09_pilot.json"
CHECKPOINT_FORMAT = "rival-v9-hybrid-ppo-checkpoint-v1"
MANIFEST_NAME = "checkpoint_manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def portable_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"<external>/{resolved.name}"


def action_schema_sha256() -> str:
    return canonical_json_sha256(action_metadata())


def load_m09_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    observation = observation_schema_manifest()
    expected = {
        "policy_version": POLICY_VERSION,
        "critic_version": CRITIC_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "observation_schema_sha256": observation["schema_sha256"],
        "action_version": ACTION_VERSION,
        "action_schema_sha256": action_schema_sha256(),
        "canonical_state_version": CANONICAL_STATE_VERSION,
        "canonical_adapter_version": CANONICAL_ADAPTER_VERSION,
    }
    mismatches = {
        key: {"config": config.get(key), "runtime": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Milestone 09 config/runtime contract mismatch: {mismatches}")
    environment_version = str(config.get("environment_version"))
    reward_version = str(config.get("reward_version"))
    reward_schedule_version = str(config.get("reward_schedule_version"))
    if reward_version == REWARD_VERSION:
        if reward_schedule_version != REWARD_SCHEDULE_VERSION:
            raise RuntimeError("Historical scratch reward schedule version mismatch")
        allowed_environments = {
            V9_TRAINING_ENVIRONMENT_VERSION,
            V9_PILOT_ENVIRONMENT_VERSION,
        }
        if environment_version not in allowed_environments:
            raise RuntimeError(
                f"Milestone 09 environment version is not implemented: {environment_version}"
            )
        if environment_version == V9_PILOT_ENVIRONMENT_VERSION:
            curriculum = config.get("curriculum", {})
            expected_curriculum = {
                "version": V9_PILOT_CURRICULUM_VERSION,
                "seed_base": 20260913,
                "weights": V9_PILOT_CURRICULUM_WEIGHTS,
                "natural_is_majority": True,
                "metrics_version": V9_PILOT_METRICS_VERSION,
            }
            if curriculum != expected_curriculum:
                raise RuntimeError(
                    "Milestone 09 pilot curriculum contract mismatch: "
                    f"expected {expected_curriculum}, got {curriculum}"
                )
        elif "curriculum" in config:
            raise RuntimeError(
                "The Gate 11 environment config cannot silently carry pilot curriculum"
            )
    else:
        from .v10_bootstrap_curriculum import CURRICULUM_VERSION, PHASE_WEIGHTS
        from .v10_bootstrap_environment import (
            ENVIRONMENT_VERSION as BOOTSTRAP_ENVIRONMENT_VERSION,
            EPISODE_TIMEOUT_SECONDS,
            NO_TOUCH_TIMEOUT_SECONDS,
        )
        from .v10_bootstrap_metrics import METRICS_VERSION
        from .v10_bootstrap_reward import (
            REWARD_SCHEDULE_VERSION as BOOTSTRAP_REWARD_SCHEDULE_VERSION,
            REWARD_VERSION as BOOTSTRAP_REWARD_VERSION,
        )

        bootstrap = config.get("bootstrap", {})
        checks = {
            "reward_version": reward_version == BOOTSTRAP_REWARD_VERSION,
            "reward_schedule_version": reward_schedule_version
            == BOOTSTRAP_REWARD_SCHEDULE_VERSION,
            "environment_version": environment_version
            == BOOTSTRAP_ENVIRONMENT_VERSION,
            "config_version": config.get("config_version")
            == "RivalM10_1TrainingConfigV1",
            "curriculum_version": bootstrap.get("curriculum_version")
            == CURRICULUM_VERSION,
            "metrics_version": bootstrap.get("metrics_version") == METRICS_VERSION,
            "initial_phase": bootstrap.get("initial_phase") == "A",
            "phase_weights": bootstrap.get("phase_weights") == PHASE_WEIGHTS,
            "no_touch_timeout": float(
                bootstrap.get("no_touch_timeout_seconds", -1.0)
            )
            == NO_TOUCH_TIMEOUT_SECONDS,
            "episode_timeout": float(bootstrap.get("episode_timeout_seconds", -1.0))
            == EPISODE_TIMEOUT_SECONDS,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Milestone 10.1 bootstrap config mismatch: {checks}")
    time_base = config["time_base"]
    if (
        time_base["physics_hz"] != 120
        or time_base["policy_hz"] != 120
        or time_base["agents_per_environment"] != 2
        or time_base["repeat_action"] is not False
    ):
        raise RuntimeError("Milestone 09 time-base contract is invalid")
    ppo = config["ppo"]
    if ppo["ppo_batch_agent_steps"] > ppo["rollout_agent_steps_per_iteration"]:
        raise RuntimeError("PPO batch cannot exceed a clean-boundary rollout")
    if ppo["ppo_batch_agent_steps"] % ppo["minibatch_agent_steps"]:
        raise RuntimeError("PPO batch must contain complete minibatches")
    return config


def config_sha256(config: dict[str, Any]) -> str:
    return canonical_json_sha256(config)


def checkpoint_contract(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "config_version": config["config_version"],
        "config_sha256": config_sha256(config),
        "policy_version": POLICY_VERSION,
        "critic_version": CRITIC_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "observation_schema_sha256": observation_schema_manifest()["schema_sha256"],
        "action_version": ACTION_VERSION,
        "action_schema_sha256": action_schema_sha256(),
        "canonical_state_version": CANONICAL_STATE_VERSION,
        "canonical_adapter_version": CANONICAL_ADAPTER_VERSION,
        "reward_version": config["reward_version"],
        "reward_schedule_version": config["reward_schedule_version"],
        # Use the stored, loader-validated environment version so a later
        # prospective environment does not make historical checkpoints
        # unverifiable merely because the runtime gained a new implementation.
        "environment_version": config["environment_version"],
        "backend": dict(config["backend"]),
    }


def pilot_config_migration_report(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    """Prove the Gate 11 -> Gate 13 change is reset/metrics-only."""

    changed_keys = sorted(
        key for key in set(source) | set(target) if source.get(key) != target.get(key)
    )
    allowed_changed_keys = ["config_version", "curriculum", "environment_version"]
    frozen_keys = sorted((set(source) | set(target)) - set(allowed_changed_keys))
    checks = {
        "only_authorized_top_level_keys_changed": changed_keys == allowed_changed_keys,
        "all_policy_action_observation_reward_ppo_backend_time_contracts_exact": all(
            source.get(key) == target.get(key) for key in frozen_keys
        ),
        "source_environment_is_gate11_or_pilot": source.get("environment_version")
        in {V9_TRAINING_ENVIRONMENT_VERSION, V9_PILOT_ENVIRONMENT_VERSION},
        "target_environment_is_pilot": target.get("environment_version")
        == V9_PILOT_ENVIRONMENT_VERSION,
        "target_curriculum_matches_runtime": target.get("curriculum")
        == {
            "version": V9_PILOT_CURRICULUM_VERSION,
            "seed_base": 20260913,
            "weights": V9_PILOT_CURRICULUM_WEIGHTS,
            "natural_is_majority": True,
            "metrics_version": V9_PILOT_METRICS_VERSION,
        },
    }
    if source.get("environment_version") == V9_PILOT_ENVIRONMENT_VERSION:
        checks["only_authorized_top_level_keys_changed"] = changed_keys == []
        allowed_changed_keys = []
    report = {
        "source_config_version": source.get("config_version"),
        "target_config_version": target.get("config_version"),
        "changed_top_level_keys": changed_keys,
        "allowed_changed_top_level_keys": allowed_changed_keys,
        "frozen_top_level_keys": frozen_keys,
        "checks": checks,
    }
    report["passed"] = all(checks.values())
    if not report["passed"]:
        raise RuntimeError(f"Unauthorized Gate 13 config migration: {report}")
    return report


def _file_manifest(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name != MANIFEST_NAME
    }


def save_v9_checkpoint(
    directory: str | Path,
    *,
    actor: RivalPolicyV1,
    critic: RivalCriticV1,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any],
    config: dict[str, Any],
    reload_observations: np.ndarray,
) -> dict[str, Any]:
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    contract = checkpoint_contract(config)
    observations = np.asarray(reload_observations, dtype=np.float32)
    expected_size = int(observation_schema_manifest()["float_count"])
    if observations.ndim != 2 or observations.shape[1] != expected_size:
        raise ValueError(
            f"Reload corpus must have shape (N, {expected_size}), got {observations.shape}"
        )

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
    complete_state = {
        "format": CHECKPOINT_FORMAT,
        "contract": contract,
        **trainer_state,
    }
    (destination / "trainer_state.json").write_text(
        json.dumps(complete_state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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


def verify_v9_checkpoint(
    directory: str | Path,
    *,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(directory)
    manifest = json.loads((source / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported v9 checkpoint: {manifest.get('format')}")
    for name, expected in manifest["files"].items():
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file is missing: {name}")
        actual = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if actual != expected:
            raise RuntimeError(
                f"Checkpoint file verification failed for {name}: expected {expected}, got {actual}"
            )
    stored_config = json.loads((source / "training_config.json").read_text())
    if manifest["contract"] != checkpoint_contract(stored_config):
        raise RuntimeError("Checkpoint contract does not match its config snapshot")
    if expected_config is not None and config_sha256(stored_config) != config_sha256(
        expected_config
    ):
        raise RuntimeError("Checkpoint config does not match the requested run config")
    return manifest


def load_v9_checkpoint(
    directory: str | Path,
    *,
    device: str | torch.device,
    expected_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source = Path(directory)
    manifest = verify_v9_checkpoint(source, expected_config=expected_config)
    selected_device = torch.device(device)
    actor = RivalPolicyV1().to(selected_device)
    critic = RivalCriticV1().to(selected_device)
    actor_payload = torch.load(source / "actor.pt", map_location=selected_device, weights_only=True)
    critic_payload = torch.load(
        source / "critic.pt", map_location=selected_device, weights_only=True
    )
    actor.load_state_dict(actor_payload["state_dict"])
    critic.load_state_dict(critic_payload["state_dict"])

    config = json.loads((source / "training_config.json").read_text())
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=float(config["ppo"]["actor_learning_rate"])
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["ppo"]["critic_learning_rate"])
    )
    actor_optimizer.load_state_dict(
        torch.load(
            source / "actor_optimizer.pt",
            map_location=selected_device,
            weights_only=True,
        )["state_dict"]
    )
    critic_optimizer.load_state_dict(
        torch.load(
            source / "critic_optimizer.pt",
            map_location=selected_device,
            weights_only=True,
        )["state_dict"]
    )
    if not actor_optimizer.state or not critic_optimizer.state:
        raise RuntimeError("Checkpoint did not restore both optimizer states")
    return {
        "actor": actor,
        "critic": critic,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "trainer_state": manifest["trainer_state"],
        "config": config,
        "reload_observations": np.load(source / "reload_observations.npy", allow_pickle=False),
        "manifest": manifest,
    }
