"""Frozen contracts and paired initialization for M10.9 PPO V2."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import torch

from .v10_6_campaign import CORPUS_ROOT, GATE_CORPUS_FILENAME
from .v10_7_campaign import button_entropy_coefficient
from .v10_7_checkpoint import load_checkpoint
from .v10_8_campaign import state_dict_sha256
from .v10_9_actions import AR_RHO, AR_TAU_SECONDS, ar_metadata
from .v9_checkpoint import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_CONFIG = REPOSITORY_ROOT / "training/configs/milestone10_9_stage1.json"
M10_8_CONFIG = REPOSITORY_ROOT / "training/configs/milestone10_8_stage1.json"
M10_8_PREFLIGHT = REPOSITORY_ROOT / "training/results/milestone10_8/preflight.json"
SOURCE_CHECKPOINT = (
    REPOSITORY_ROOT
    / "training/checkpoints/milestone10_7/stage_1/initialization/000000000"
)
SOURCE_ACTOR_FILE_SHA256 = (
    "1e58fa4f6ad107344fa7d163b53e1af2aad871ec01d16ed47776d61b20397548"
)
SOURCE_MANIFEST_SHA256 = (
    "301b6f3a65cf998bdffe39fcd23c353c5187d2c760cfd74ebbec356bf5fd2824"
)
PAIRED_ACTOR_STATE_SHA256 = (
    "1bce479b61613b6284f94861ade03214f0d940be39dcce4499fea541178b0daf"
)
PAIRED_CRITIC_STATE_SHA256 = (
    "d204ecae323d911465bcd3a0f5541a9823928f470ebca2beb300f1edd0c1ab97"
)
CAMPAIGN_ID = "rival-v10-9-stage1-ppo-v2"
RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10_9"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "training/checkpoints/milestone10_9/stage_1"
INITIAL_CHECKPOINT = CHECKPOINT_ROOT / "initialization/000000000"
ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR = 432_000
BOUNDARY_HOURS = (0.25, 0.5, 1.0)


def load_stage1_config(path: str | Path = DEFAULT_STAGE1_CONFIG) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    previous = json.loads(M10_8_CONFIG.read_text(encoding="utf-8"))
    arm_c = deepcopy(previous)
    arm_c["ppo"]["gae_lambda"] = previous["experiment"]["arms"]["C"][
        "gae_lambda"
    ]
    checks = {
        "identity_exact": config["config_version"]
        == "RivalM10_9Stage1PPOV2ConfigV1"
        and config["campaign_id"] == CAMPAIGN_ID,
        "reward_frozen": config["reward_contract"] == previous["reward_contract"],
        "backend_frozen": config["backend"] == previous["backend"],
        "time_base_frozen": config["time_base"] == previous["time_base"],
        "button_policy_frozen": config["button_policy"] == previous["button_policy"],
        "architecture_frozen": all(
            config[key] == previous[key]
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
        "arm_c_gae_exact": config["ppo"]["gamma"]
        == arm_c["ppo"]["gamma"]
        and config["ppo"]["gae_lambda"] == arm_c["ppo"]["gae_lambda"],
        "advantage_scale_only": config["ppo"]["advantage_scaling"]
        == "scale_only_no_mean_centering",
        "actor_schedule_exact": config["ppo"]["actor_learning_rate"] == 5e-5
        and config["ppo"]["actor_epochs"] == 2
        and config["ppo"]["actor_minibatch_agent_steps"] == 8192
        and config["ppo"]["actor_kl_stop_threshold"] == 0.015,
        "critic_schedule_exact": config["ppo"]["critic_learning_rate"] == 2e-4
        and config["ppo"]["critic_epochs"] == 8
        and config["ppo"]["critic_minibatch_agent_steps"] == 8192
        and config["ppo"]["critic_loss"]
        == "mean_squared_error_fixed_rollout_returns",
        "ar_exact": config["analog_exploration"]["tau_seconds"]
        == AR_TAU_SECONDS
        and config["analog_exploration"]["version"]
        == "RivalAR1AnalogExplorationV1"
        and not config["analog_exploration"]["action_repeat"],
        "scope_exact": config["stage_contract"]["evaluation_boundaries_simulated_hours"]
        == list(BOUNDARY_HOURS)
        and config["stage_contract"]["maximum_active_learner_steps"] == 432_000
        and config["stage_contract"]["automatically_continue_beyond_1h"] is False,
        "stage2_and_promotion_forbidden": config["stage_contract"][
            "stage_2_authorized"
        ]
        is False
        and config["stage_contract"]["production_promotion_authorized"] is False,
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"M10.9 frozen contract failed: {checks}")
    return config


def paired_initialization(
    config: dict[str, Any], *, device: str | torch.device
) -> dict[str, Any]:
    if sha256_file(SOURCE_CHECKPOINT / "actor.pt") != SOURCE_ACTOR_FILE_SHA256:
        raise RuntimeError("M10.9 source actor file hash changed")
    if sha256_file(SOURCE_CHECKPOINT / "checkpoint_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("M10.9 source manifest hash changed")
    source_config = json.loads(
        (REPOSITORY_ROOT / "training/configs/milestone10_7_stage1.json").read_text(
            encoding="utf-8"
        )
    )
    loaded = load_checkpoint(
        SOURCE_CHECKPOINT, device=device, expected_config=source_config
    )
    actor_hash = state_dict_sha256(loaded["actor"].state_dict())
    critic_hash = state_dict_sha256(loaded["critic"].state_dict())
    if actor_hash != PAIRED_ACTOR_STATE_SHA256:
        raise RuntimeError("M10.9 paired actor state hash changed")
    if critic_hash != PAIRED_CRITIC_STATE_SHA256:
        raise RuntimeError("M10.9 paired critic state hash changed")
    actor_optimizer = torch.optim.Adam(
        loaded["actor"].parameters(), lr=float(config["ppo"]["actor_learning_rate"])
    )
    critic_optimizer = torch.optim.Adam(
        loaded["critic"].parameters(), lr=float(config["ppo"]["critic_learning_rate"])
    )
    proof = {
        "version": "RivalM10_9PairedInitializationV1",
        "source_checkpoint": SOURCE_CHECKPOINT.relative_to(REPOSITORY_ROOT).as_posix(),
        "source_actor_file_sha256": sha256_file(SOURCE_CHECKPOINT / "actor.pt"),
        "source_manifest_sha256": sha256_file(
            SOURCE_CHECKPOINT / "checkpoint_manifest.json"
        ),
        "actor_state_sha256": actor_hash,
        "critic_state_sha256": critic_hash,
        "fresh_actor_optimizer_state_entries": len(actor_optimizer.state),
        "fresh_critic_optimizer_state_entries": len(critic_optimizer.state),
        "m10_8_preflight_sha256": sha256_file(M10_8_PREFLIGHT),
        "m10_8_trained_actor_used": False,
        "m10_7_trained_actor_used": False,
        "checks": {
            "actor_exact": actor_hash == PAIRED_ACTOR_STATE_SHA256,
            "critic_exact": critic_hash == PAIRED_CRITIC_STATE_SHA256,
            "actor_optimizer_fresh": not actor_optimizer.state,
            "critic_optimizer_fresh": not critic_optimizer.state,
        },
    }
    proof["checks"]["passed"] = all(proof["checks"].values())
    return {
        "actor": loaded["actor"],
        "critic": loaded["critic"],
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "reload_observations": loaded["reload_observations"].copy(),
        "trainer_state": {
            "completed_iterations": 0,
            "cumulative_agent_steps": 0,
            "cumulative_model_updates": 0,
            "campaign_id": CAMPAIGN_ID,
            "stage": 1,
            "m10_9_ppo_v2": True,
            "source_checkpoint": proof["source_checkpoint"],
            "clean_boundary": True,
            "partial_experience_buffer_records": 0,
            "production_promotion_authorized": False,
        },
        "proof": proof,
    }


def configuration_evidence(config: dict[str, Any]) -> dict[str, Any]:
    preflight = json.loads(M10_8_PREFLIGHT.read_text(encoding="utf-8"))
    return {
        "config_path": DEFAULT_STAGE1_CONFIG.relative_to(REPOSITORY_ROOT).as_posix(),
        "config_sha256": sha256_file(DEFAULT_STAGE1_CONFIG),
        "m10_8_preflight_sha256": sha256_file(M10_8_PREFLIGHT),
        "m10_8_paired_initialization": preflight["paired_initialization"],
        "ar_exploration": ar_metadata(),
        "ar_rho": AR_RHO,
        "frozen_corpus": {
            "path": (CORPUS_ROOT / GATE_CORPUS_FILENAME)
            .relative_to(REPOSITORY_ROOT)
            .as_posix(),
            "sha256": sha256_file(CORPUS_ROOT / GATE_CORPUS_FILENAME),
            "episodes": 500,
        },
        "button_entropy_coefficients": {
            "at_0h": button_entropy_coefficient(0, config),
            "at_0p25h": button_entropy_coefficient(108_000, config),
            "at_0p5h": button_entropy_coefficient(216_000, config),
            "at_1h": button_entropy_coefficient(432_000, config),
        },
    }
