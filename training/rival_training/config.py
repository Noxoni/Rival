"""Configuration helpers for Rival's committed training campaigns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRAINING_ROOT.parent
DEFAULT_CONFIG_PATH = TRAINING_ROOT / "configs" / "milestone05.json"
MILESTONE06_CONFIG_PATH = TRAINING_ROOT / "configs" / "milestone06.json"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_milestone06_config() -> dict[str, Any]:
    config = load_config(MILESTONE06_CONFIG_PATH)
    validate_milestone06_config(config)
    return config


def stage_config(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    for stage in config["stages"]:
        if stage["name"] == stage_name:
            return stage
    raise ValueError(f"Unknown Milestone 06 stage {stage_name!r}")


def validate_milestone06_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 2:
        raise ValueError("Milestone 06 config must use schema version 2")
    if config.get("campaign_ceiling_agent_steps") != 100_000_000:
        raise ValueError("Milestone 06 campaign ceiling must remain 100M agent-steps")
    environment = config["environment"]
    if environment["workers"] != 56 or environment["cadence_ticks"] != 4:
        raise ValueError(
            "Milestone 06 requires the measured 56-worker optimum and mechanics4 cadence"
        )
    if environment["worker_selection"]["evidence"] != (
        "training/results/milestone06/throughput_sweep.json"
    ):
        raise ValueError("Milestone 06 worker selection must reference its sweep evidence")
    ppo = config["ppo"]
    if ppo["batch_size"] % ppo["minibatch_size"]:
        raise ValueError("PPO batch size must be divisible by minibatch size")
    expected_gamma = 0.99**0.5
    if abs(float(ppo["gamma"]) - expected_gamma) > 1e-12:
        raise ValueError("Milestone 06 gamma must preserve the 8-tick 0.99 horizon")
    previous_end = 0
    for stage in config["stages"]:
        if int(stage["start_agent_steps"]) != previous_end:
            raise ValueError("Milestone 06 stages must be contiguous")
        previous_end = int(stage["end_agent_steps"])
        weights = stage["curriculum_weights"]
        if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
            raise ValueError(f"Curriculum weights do not sum to one: {stage['name']}")
        if float(weights["natural"]) < 0.75:
            raise ValueError(f"Natural resets lost majority status: {stage['name']}")
    if previous_end != config["campaign_ceiling_agent_steps"]:
        raise ValueError("Final stage must end at the campaign ceiling")


def resolve_repo_path(relative_path: str) -> Path:
    path = (REPOSITORY_ROOT / relative_path).resolve()
    path.relative_to(REPOSITORY_ROOT.resolve())
    return path
