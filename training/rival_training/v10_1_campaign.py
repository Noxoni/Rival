"""Milestone 10.1 agency-bootstrap authority and checkpoint helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .m10_campaign import (
    checkpoint_record,
    verify_checkpoint_reload_parity,
)
from .v10_bootstrap_curriculum import CURRICULUM_VERSION, PHASE_WEIGHTS
from .v10_bootstrap_environment import (
    ENVIRONMENT_VERSION,
    EPISODE_TIMEOUT_SECONDS,
    NO_TOUCH_TIMEOUT_SECONDS,
)
from .v10_bootstrap_metrics import METRICS_VERSION
from .v10_bootstrap_reward import (
    REWARD_SCHEDULE_VERSION,
    REWARD_VERSION,
    reward_metadata,
)
from .v9_checkpoint import config_sha256, load_m09_config, load_v9_checkpoint, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "training/configs/milestone10_1.json"
M10_CONFIG_PATH = REPOSITORY_ROOT / "training/configs/milestone10.json"
AUTHORITY_PATH = REPOSITORY_ROOT / "handoff/v10.1/M10_1_CAMPAIGN.json"
M10_PLUS25_CHECKPOINT = REPOSITORY_ROOT / (
    "training/checkpoints/milestone10/boundaries/plus-025h/023378810"
)
M10_PLUS25_STEPS = 23_378_810
M10_PLUS25_SIMULATED_HOURS = 27.05880787037037
M10_PLUS25_ACTOR_SHA256 = (
    "5d246a7eee8af22290f6f644a3e408f786551dc893bf10f19d487945329100c1"
)
M10_PLUS25_MANIFEST_SHA256 = (
    "903a38cdb85d8a171e207c18718d9651cae1ec905003f0fbf3729c5203202784"
)
AGENT_STEPS_PER_SIMULATED_HOUR = 864_000
BOUNDARY_HOURS = (2.5, 5.0, 10.0, 15.0, 20.0, 25.0)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_m10_1_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = load_m09_config(path)
    authority = _read_json(AUTHORITY_PATH)
    source = load_m09_config(M10_CONFIG_PATH)
    migration = config_migration_report(source, config)
    campaign = config["campaign"]
    checks = {
        "campaign_id": config["campaign_id"] == authority["campaign_id"],
        "reward_version": config["reward_version"]
        == authority["bootstrap_versions"]["reward"],
        "curriculum_version": config["bootstrap"]["curriculum_version"]
        == authority["bootstrap_versions"]["curriculum"],
        "environment_version": config["environment_version"]
        == authority["bootstrap_versions"]["environment"],
        "worker_count": config["backend"]["worker_count"]
        == authority["runtime"]["worker_count_start"],
        "timeouts": config["bootstrap"]["no_touch_timeout_seconds"]
        == authority["runtime"]["no_touch_timeout_seconds"]
        and config["bootstrap"]["episode_timeout_seconds"]
        == authority["runtime"]["episode_timeout_seconds"],
        "boundaries": campaign["evaluation_boundaries_added_simulated_hours"]
        == authority["evaluation_boundaries_added_bootstrap_hours"],
        "maximum_budget": campaign["maximum_additional_agent_steps"]
        == authority["maximum_additional_bootstrap_agent_steps"]
        and campaign["maximum_additional_simulated_game_hours"]
        == authority["maximum_additional_bootstrap_simulated_game_hours"],
        "exact_start_identity": campaign["starting_cumulative_agent_steps"]
        == M10_PLUS25_STEPS
        and campaign["starting_checkpoint_actor_sha256"]
        == M10_PLUS25_ACTOR_SHA256
        and campaign["starting_checkpoint_manifest_sha256"]
        == M10_PLUS25_MANIFEST_SHA256,
        "runtime_versions": config["reward_version"] == REWARD_VERSION
        and config["reward_schedule_version"] == REWARD_SCHEDULE_VERSION
        and config["environment_version"] == ENVIRONMENT_VERSION
        and config["bootstrap"]["curriculum_version"] == CURRICULUM_VERSION
        and config["bootstrap"]["metrics_version"] == METRICS_VERSION,
        "phase_weights": config["bootstrap"]["phase_weights"] == PHASE_WEIGHTS,
        "production_not_authorized": campaign["production_promotion_authorized"] is False,
        "migration_passed": migration["passed"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Milestone 10.1 config/authority mismatch: {checks}")
    return config


def config_migration_report(source: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    immutable = {
        "policy_version",
        "critic_version",
        "observation_version",
        "observation_schema_sha256",
        "action_version",
        "action_schema_sha256",
        "canonical_state_version",
        "canonical_adapter_version",
        "backend",
        "time_base",
        "ppo",
        "gate11",
    }
    changes = sorted(
        key for key in set(source) | set(target) if source.get(key) != target.get(key)
    )
    allowed = sorted(
        {
            "config_version",
            "campaign_id",
            "reward_version",
            "reward_schedule_version",
            "environment_version",
            "curriculum",
            "bootstrap",
            "campaign",
        }
    )
    checks = {
        "source_is_m10_or_same_bootstrap": source.get("config_version")
        in {"RivalM10TrainingConfigV1", "RivalM10_1TrainingConfigV1"},
        "target_is_bootstrap": target.get("config_version")
        == "RivalM10_1TrainingConfigV1",
        "immutable_architecture_and_ppo_exact": all(
            source.get(key) == target.get(key) for key in immutable
        ),
        "only_learning_distribution_and_campaign_changed": (
            set(changes) <= set(allowed)
            if source.get("config_version") == "RivalM10TrainingConfigV1"
            else changes == []
        ),
        "normal_reward_not_mutated": source.get("reward_version")
        == "RivalScratchRewardV1",
        "bootstrap_reward_is_new_version": target.get("reward_version")
        == REWARD_VERSION,
    }
    if source.get("config_version") == "RivalM10_1TrainingConfigV1":
        checks["normal_reward_not_mutated"] = True
    report = {
        "source_config_version": source.get("config_version"),
        "source_config_sha256": config_sha256(source),
        "target_config_version": target.get("config_version"),
        "target_config_sha256": config_sha256(target),
        "changed_top_level_keys": changes,
        "allowed_changed_top_level_keys": allowed,
        "immutable_top_level_keys": sorted(immutable),
        "checks": checks,
    }
    report["passed"] = all(checks.values())
    if not report["passed"]:
        raise RuntimeError(f"Unauthorized v10.1 config migration: {report}")
    return report


def verify_exact_plus25_start(
    checkpoint: str | Path = M10_PLUS25_CHECKPOINT,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    path = Path(checkpoint)
    loaded = load_v9_checkpoint(path, device=device)
    record = checkpoint_record(path, manifest=loaded["manifest"])
    checks = {
        "directory_exact": path.resolve() == M10_PLUS25_CHECKPOINT.resolve(),
        "actor_sha256_exact": record["actor_sha256"] == M10_PLUS25_ACTOR_SHA256,
        "manifest_sha256_exact": record["manifest_sha256"]
        == M10_PLUS25_MANIFEST_SHA256,
        "cumulative_steps_exact": record["cumulative_agent_steps"]
        == M10_PLUS25_STEPS,
        "trainer_clean_boundary": record["clean_boundary"],
        "actor_critic_optimizers_present": all(
            (path / name).is_file()
            for name in (
                "actor.pt",
                "critic.pt",
                "actor_optimizer.pt",
                "critic_optimizer.pt",
                "trainer_state.json",
            )
        ),
        "fresh_reload_exact": verify_checkpoint_reload_parity(
            path, expected_config=loaded["config"], device=device
        )["checks"]["passed"],
    }
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checkpoint": record,
        "checks": checks,
    }
    if report["status"] != "passed":
        raise RuntimeError(f"Exact M10 +25 bootstrap start failed: {checks}")
    return report


def boundary_slug(hours: int | float) -> str:
    value = float(hours)
    if value not in BOUNDARY_HOURS:
        raise ValueError(f"Unsupported v10.1 boundary: {hours}")
    return "plus-002p5h" if value == 2.5 else f"plus-{int(value):03d}h"


def nominal_boundary_steps(hours: int | float) -> int:
    return M10_PLUS25_STEPS + int(round(float(hours) * AGENT_STEPS_PER_SIMULATED_HOUR))


def checkpoint_identity(directory: str | Path) -> dict[str, Any]:
    loaded = load_v9_checkpoint(directory, device="cpu")
    return checkpoint_record(directory, manifest=loaded["manifest"])


def compact_training_iteration(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["pilot_metrics"]
    return {
        "iteration": int(report["iteration"]),
        "collected_agent_steps": int(report["collected_agent_steps"]),
        "cumulative_agent_steps": int(report["cumulative_agent_steps"]),
        "added_bootstrap_simulated_game_hours": (
            int(report["cumulative_agent_steps"]) - M10_PLUS25_STEPS
        )
        / AGENT_STEPS_PER_SIMULATED_HOUR,
        "agent_steps_per_second": float(report["agent_steps_per_second"]),
        "iteration_wall_seconds": float(report["iteration_wall_seconds"]),
        "reward": report["reward"],
        "ppo": {
            "actor_loss": report["ppo"]["actor_loss"],
            "critic_loss": report["ppo"]["critic_loss"],
            "analog_entropy": report["ppo"]["analog_entropy"],
            "button_entropy": report["ppo"]["button_entropy"],
            "actor_update_magnitude": report["ppo"]["actor_update_magnitude"],
            "critic_update_magnitude": report["ppo"]["critic_update_magnitude"],
            "approximate_kl": report["ppo"]["approximate_kl"],
            "clip_fraction": report["ppo"]["clip_fraction"],
        },
        "actions": report["actions"],
        "bootstrap_metrics": {
            "reset_counts": metrics["reset_counts"],
            "termination_counts": metrics["termination_counts"],
            "goals": metrics["goals"],
            "interaction_rates_per_100k_agent_steps": metrics[
                "interaction_rates_per_100k_agent_steps"
            ],
            "touch_chain": metrics["touch_chain"],
            "reward_integrity": metrics["reward_integrity"],
        },
        "health": report["health"],
    }


def contract_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_sha256": config_sha256(config),
        "reward": reward_metadata(),
        "curriculum_version": CURRICULUM_VERSION,
        "phase_weights": PHASE_WEIGHTS,
        "environment_version": ENVIRONMENT_VERSION,
        "metrics_version": METRICS_VERSION,
        "no_touch_timeout_seconds": NO_TOUCH_TIMEOUT_SECONDS,
        "episode_timeout_seconds": EPISODE_TIMEOUT_SECONDS,
        "source_checkpoint_manifest_sha256": sha256_file(
            M10_PLUS25_CHECKPOINT / "checkpoint_manifest.json"
        ),
    }
