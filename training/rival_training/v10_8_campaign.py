"""Frozen paired-arm contracts for the Milestone 10.8 GAE experiment."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .v10_6_campaign import CORPUS_ROOT, GATE_CORPUS_FILENAME
from .v10_7_actions import BUTTON_PERSISTENCE
from .v10_7_campaign import button_entropy_coefficient
from .v10_7_checkpoint import load_checkpoint
from .v9_checkpoint import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_CONFIG = REPOSITORY_ROOT / "training/configs/milestone10_8_stage1.json"
M10_7_CONFIG = REPOSITORY_ROOT / "training/configs/milestone10_7_stage1.json"
RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10_8"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "training/checkpoints/milestone10_8"
SOURCE_CHECKPOINT = (
    REPOSITORY_ROOT
    / "training/checkpoints/milestone10_7/stage_1/initialization/000000000"
)
SOURCE_ACTOR_SHA256 = (
    "1e58fa4f6ad107344fa7d163b53e1af2aad871ec01d16ed47776d61b20397548"
)
SOURCE_MANIFEST_SHA256 = (
    "301b6f3a65cf998bdffe39fcd23c353c5187d2c760cfd74ebbec356bf5fd2824"
)
CAMPAIGN_ID = "rival-v10-8-stage1-gae-credit-assignment"
ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR = 432_000
BOUNDARY_HOURS = (0.5, 1.0)
ARM_LAMBDAS = {
    "A": 0.9872585449014338,
    "B": 0.993608849045455,
    "C": 0.9983695094257663,
}


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def arm_slug(arm: str) -> str:
    selected = str(arm).upper()
    if selected not in ARM_LAMBDAS:
        raise ValueError(f"Unknown M10.8 arm: {arm}")
    return f"arm_{selected.lower()}"


def load_arm_config(
    arm: str, path: str | Path = DEFAULT_STAGE1_CONFIG
) -> dict[str, Any]:
    selected = str(arm).upper()
    arm_slug(selected)
    template = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = json.loads(M10_7_CONFIG.read_text(encoding="utf-8"))
    config = deepcopy(template)
    config["ppo"]["gae_lambda"] = ARM_LAMBDAS[selected]

    expected_experiment = {
        "variable": "ppo.gae_lambda",
        "selected_arm": None,
        "paired_initialization": "exact_m10_7_pre_ppo_initialization_checkpoint",
        "arms": {
            "A": {
                "label": "control_m10_7_horizon",
                "gae_lambda": ARM_LAMBDAS["A"],
            },
            "B": {
                "label": "physical_time_conversion_0p95_pow_1_over_8",
                "gae_lambda": ARM_LAMBDAS["B"],
            },
            "C": {
                "label": "two_second_half_life",
                "gae_lambda": ARM_LAMBDAS["C"],
            },
        },
    }
    frozen_ppo = deepcopy(frozen["ppo"])
    frozen_ppo["gae_lambda"] = ARM_LAMBDAS[selected]
    checks = {
        "config_identity_exact": config["config_version"]
        == "RivalM10_8Stage1GAEExperimentConfigV1"
        and config["campaign_id"] == CAMPAIGN_ID,
        "experiment_exact": config["experiment"] == expected_experiment,
        "ppo_only_lambda_differs_from_m10_7": config["ppo"] == frozen_ppo,
        "reward_frozen": config["reward_contract"] == frozen["reward_contract"],
        "backend_frozen": config["backend"] == frozen["backend"],
        "time_base_frozen": config["time_base"] == frozen["time_base"],
        "action_policy_frozen": config["button_policy"] == frozen["button_policy"],
        "architecture_frozen": all(
            config[key] == frozen[key]
            for key in (
                "policy_version",
                "critic_version",
                "observation_version",
                "observation_schema_sha256",
                "action_version",
                "action_schema_sha256",
                "button_policy_version",
                "canonical_state_version",
                "canonical_adapter_version",
                "reward_version",
                "reward_schedule_version",
                "curriculum_version",
                "environment_version",
            )
        ),
        "persistence_frozen": config["button_policy"]["persistence"]
        == BUTTON_PERSISTENCE,
        "source_exact": config["stage_contract"]["source_checkpoint"]
        == SOURCE_CHECKPOINT.relative_to(REPOSITORY_ROOT).as_posix()
        and config["stage_contract"]["source_actor_sha256"] == SOURCE_ACTOR_SHA256
        and config["stage_contract"]["source_manifest_sha256"]
        == SOURCE_MANIFEST_SHA256,
        "scope_exact": config["stage_contract"][
            "maximum_active_learner_steps_per_arm"
        ]
        == ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR
        and config["stage_contract"]["evaluation_boundaries_simulated_hours"]
        == list(BOUNDARY_HOURS)
        and config["stage_contract"]["automatically_continue_beyond_1h"] is False,
        "stage2_and_promotion_forbidden": config["stage_contract"][
            "stage_2_authorized"
        ]
        is False
        and config["stage_contract"]["production_promotion_authorized"] is False,
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"M10.8 frozen-contract mismatch for arm {selected}: {checks}")
    return config


def paired_initial_state(device: str | torch.device = "cpu") -> dict[str, Any]:
    if sha256_file(SOURCE_CHECKPOINT / "actor.pt") != SOURCE_ACTOR_SHA256:
        raise RuntimeError("M10.7 initialization actor file hash changed")
    if sha256_file(SOURCE_CHECKPOINT / "checkpoint_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("M10.7 initialization manifest hash changed")
    source_config = json.loads(M10_7_CONFIG.read_text(encoding="utf-8"))
    loaded = load_checkpoint(
        SOURCE_CHECKPOINT, device=device, expected_config=source_config
    )
    state = loaded["trainer_state"]
    if any(int(state[key]) != 0 for key in (
        "completed_iterations",
        "cumulative_agent_steps",
        "cumulative_model_updates",
    )):
        raise RuntimeError("M10.7 paired source is not the zero-update checkpoint")
    if loaded["actor_optimizer"].state or loaded["critic_optimizer"].state:
        raise RuntimeError("M10.7 paired source optimizers are not fresh")
    return loaded


def paired_contract_report() -> dict[str, Any]:
    configs = {arm: load_arm_config(arm) for arm in ARM_LAMBDAS}
    normalized = {}
    for arm, config in configs.items():
        row = deepcopy(config)
        row["ppo"]["gae_lambda"] = "EXPERIMENTAL_VARIABLE"
        normalized[arm] = row
    source = paired_initial_state("cpu")
    actor_hash = state_dict_sha256(source["actor"].state_dict())
    critic_hash = state_dict_sha256(source["critic"].state_dict())
    checks = {
        "all_arm_configs_identical_after_lambda_mask": len(
            {json.dumps(value, sort_keys=True) for value in normalized.values()}
        )
        == 1,
        "all_lambdas_exact": all(
            configs[arm]["ppo"]["gae_lambda"] == value
            for arm, value in ARM_LAMBDAS.items()
        ),
        "source_actor_file_exact": sha256_file(SOURCE_CHECKPOINT / "actor.pt")
        == SOURCE_ACTOR_SHA256,
        "source_manifest_exact": sha256_file(
            SOURCE_CHECKPOINT / "checkpoint_manifest.json"
        )
        == SOURCE_MANIFEST_SHA256,
        "source_zero_updates": int(source["trainer_state"]["cumulative_agent_steps"])
        == 0,
        "fresh_optimizers_empty": not source["actor_optimizer"].state
        and not source["critic_optimizer"].state,
    }
    checks["passed"] = all(checks.values())
    return {
        "version": "RivalM10_8PairedInitializationContractV1",
        "source_checkpoint": SOURCE_CHECKPOINT.relative_to(REPOSITORY_ROOT).as_posix(),
        "source_actor_file_sha256": sha256_file(SOURCE_CHECKPOINT / "actor.pt"),
        "source_manifest_sha256": sha256_file(
            SOURCE_CHECKPOINT / "checkpoint_manifest.json"
        ),
        "actor_state_sha256": actor_hash,
        "critic_state_sha256": critic_hash,
        "actor_optimizer_state_entries": len(source["actor_optimizer"].state),
        "critic_optimizer_state_entries": len(source["critic_optimizer"].state),
        "arm_lambdas": ARM_LAMBDAS,
        "checks": checks,
    }


def configuration_evidence() -> dict[str, Any]:
    config = load_arm_config("A")
    return {
        "template_path": DEFAULT_STAGE1_CONFIG.relative_to(REPOSITORY_ROOT).as_posix(),
        "template_sha256": sha256_file(DEFAULT_STAGE1_CONFIG),
        "m10_7_config_sha256": sha256_file(M10_7_CONFIG),
        "frozen_corpus": {
            "path": (CORPUS_ROOT / GATE_CORPUS_FILENAME)
            .relative_to(REPOSITORY_ROOT)
            .as_posix(),
            "sha256": sha256_file(CORPUS_ROOT / GATE_CORPUS_FILENAME),
            "episodes": 500,
        },
        "button_entropy_coefficients": {
            "at_0h": button_entropy_coefficient(0, config),
            "at_0p5h": button_entropy_coefficient(216_000, config),
            "at_1h": button_entropy_coefficient(432_000, config),
        },
    }
