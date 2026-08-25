"""Frozen contracts and clean initialization for M10.10 first-touch training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .v10_6_campaign import CORPUS_ROOT, GATE_CORPUS_FILENAME
from .v10_7_checkpoint import load_checkpoint
from .v10_8_campaign import state_dict_sha256
from .v10_9_campaign import load_stage1_config as load_m10_9_config
from .v9_checkpoint import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_CONFIG = (
    REPOSITORY_ROOT / "training/configs/milestone10_10_stage1.json"
)
M10_9_CONFIG = REPOSITORY_ROOT / "training/configs/milestone10_9_stage1.json"
M10_9_FINAL = (
    REPOSITORY_ROOT / "training/results/milestone10_9/final_comparison.json"
)
SOURCE_CHECKPOINT = (
    REPOSITORY_ROOT
    / "training/checkpoints/milestone10_9/stage_1/initialization/000000000"
)
SOURCE_ACTOR_FILE_SHA256 = (
    "a4c2c51d1dadb51ceb3b417c8851a8c3104b2b33cfbc4d4ac6314ae699d732d4"
)
SOURCE_CRITIC_FILE_SHA256 = (
    "6fbd6487dfb50f3c1d1eaadc5b960209ff4a73be6976a2d139200530c207ebe5"
)
SOURCE_MANIFEST_SHA256 = (
    "9ec987ea8ea4a6ee83f664a12fc6bbe569a7e3bac522b4f69ae203281f672920"
)
PAIRED_ACTOR_STATE_SHA256 = (
    "1bce479b61613b6284f94861ade03214f0d940be39dcce4499fea541178b0daf"
)
PAIRED_CRITIC_STATE_SHA256 = (
    "d204ecae323d911465bcd3a0f5541a9823928f470ebca2beb300f1edd0c1ab97"
)
CAMPAIGN_ID = "rival-v10-10-minimal-first-touch"
RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10_10"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "training/checkpoints/milestone10_10/stage_1"
INITIAL_CHECKPOINT = CHECKPOINT_ROOT / "initialization/000000000"
ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR = 432_000
BOUNDARY_HOURS = (0.25, 0.5, 1.0)


def load_stage1_config(
    path: str | Path = DEFAULT_STAGE1_CONFIG,
) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    previous = json.loads(M10_9_CONFIG.read_text(encoding="utf-8"))
    reward = config["reward_contract"]
    zero_fields = (
        "distance_progress_reward",
        "heading_alignment_reward",
        "acquisition_time_penalty",
        "idle_penalty",
        "generic_speed_reward",
        "boost_reward",
        "throttle_reward",
        "steer_reward",
        "action_magnitude_reward",
        "jump_reward",
        "handbrake_reward",
        "goal_reward",
        "concede_reward",
        "possession_reward",
        "named_mechanic_reward",
        "aerial_reward",
        "recovery_reward",
    )
    checks = {
        "identity_exact": config["config_version"]
        == "RivalM10_10MinimalFirstTouchConfigV1"
        and config["campaign_id"] == CAMPAIGN_ID,
        "backend_frozen": config["backend"] == previous["backend"],
        "time_base_frozen": config["time_base"] == previous["time_base"],
        "ppo_v2_frozen": config["ppo"] == previous["ppo"],
        "button_policy_frozen": config["button_policy"]
        == previous["button_policy"],
        "analog_exploration_frozen": config["analog_exploration"]
        == previous["analog_exploration"],
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
                "curriculum_version",
            )
        ),
        "minimal_dense_reward_exact": reward["dense_reward"]
        == "velocity_to_ball_only"
        and reward["maximum_car_speed_uu_per_second"] == 2300.0
        and reward["native_tick_integration"]
        == "normalized_directed_velocity*tick_delta/120"
        and reward["reads_ball_velocity"] is False,
        "first_touch_exact": reward["rewarded_contact_limit"] == 1
        and reward["first_physical_touch_reward"] == 10.0
        and reward["second_and_later_touch_reward"] == 0.0
        and reward["terminate_on_first_physical_touch"] is True,
        "all_other_rewards_zero": all(reward[name] == 0.0 for name in zero_fields),
        "long_horizon_gae_exact": config["ppo"]["gamma"]
        == 0.9987444968227265
        and config["ppo"]["gae_lambda"] == 0.9983695094257663,
        "source_exact": config["stage_contract"]["source_checkpoint"]
        == SOURCE_CHECKPOINT.relative_to(REPOSITORY_ROOT).as_posix()
        and config["stage_contract"]["source_actor_file_sha256"]
        == SOURCE_ACTOR_FILE_SHA256,
        "scope_exact": config["stage_contract"][
            "evaluation_boundaries_simulated_hours"
        ]
        == list(BOUNDARY_HOURS)
        and config["stage_contract"]["maximum_active_learner_steps"]
        == ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR
        and config["stage_contract"]["automatically_continue_beyond_1h"]
        is False,
        "stage2_and_promotion_forbidden": config["stage_contract"][
            "stage_2_authorized"
        ]
        is False
        and config["stage_contract"]["production_promotion_authorized"]
        is False,
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"M10.10 frozen contract failed: {checks}")
    return config


def clean_initialization(
    config: dict[str, Any], *, device: str | torch.device
) -> dict[str, Any]:
    file_hashes = {
        "actor": sha256_file(SOURCE_CHECKPOINT / "actor.pt"),
        "critic": sha256_file(SOURCE_CHECKPOINT / "critic.pt"),
        "manifest": sha256_file(SOURCE_CHECKPOINT / "checkpoint_manifest.json"),
    }
    expected = {
        "actor": SOURCE_ACTOR_FILE_SHA256,
        "critic": SOURCE_CRITIC_FILE_SHA256,
        "manifest": SOURCE_MANIFEST_SHA256,
    }
    if file_hashes != expected:
        raise RuntimeError(
            f"M10.10 source initialization hashes changed: {file_hashes}"
        )
    source_config = load_m10_9_config()
    loaded = load_checkpoint(
        SOURCE_CHECKPOINT, device=device, expected_config=source_config
    )
    actor_hash = state_dict_sha256(loaded["actor"].state_dict())
    critic_hash = state_dict_sha256(loaded["critic"].state_dict())
    if actor_hash != PAIRED_ACTOR_STATE_SHA256:
        raise RuntimeError("M10.10 paired actor state hash changed")
    if critic_hash != PAIRED_CRITIC_STATE_SHA256:
        raise RuntimeError("M10.10 paired critic state hash changed")
    actor_optimizer = torch.optim.Adam(
        loaded["actor"].parameters(),
        lr=float(config["ppo"]["actor_learning_rate"]),
    )
    critic_optimizer = torch.optim.Adam(
        loaded["critic"].parameters(),
        lr=float(config["ppo"]["critic_learning_rate"]),
    )
    proof = {
        "version": "RivalM10_10CleanInitializationV1",
        "source_checkpoint": SOURCE_CHECKPOINT.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "source_file_sha256": file_hashes,
        "actor_state_sha256": actor_hash,
        "critic_state_sha256": critic_hash,
        "fresh_actor_optimizer_state_entries": len(actor_optimizer.state),
        "fresh_critic_optimizer_state_entries": len(critic_optimizer.state),
        "m10_7_through_m10_9_trained_actor_used": False,
        "checks": {
            "source_files_exact": file_hashes == expected,
            "actor_state_exact": actor_hash == PAIRED_ACTOR_STATE_SHA256,
            "critic_state_exact": critic_hash == PAIRED_CRITIC_STATE_SHA256,
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
            "m10_10_minimal_first_touch": True,
            "source_checkpoint": proof["source_checkpoint"],
            "clean_boundary": True,
            "partial_experience_buffer_records": 0,
            "production_promotion_authorized": False,
        },
        "proof": proof,
    }


def configuration_evidence(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": DEFAULT_STAGE1_CONFIG.relative_to(
            REPOSITORY_ROOT
        ).as_posix(),
        "config_sha256": sha256_file(DEFAULT_STAGE1_CONFIG),
        "m10_9_final_sha256": sha256_file(M10_9_FINAL),
        "frozen_m10_9_ppo": config["ppo"],
        "frozen_action": {
            "version": config["action_version"],
            "schema_sha256": config["action_schema_sha256"],
            "continuous_axes": ["throttle", "steer", "pitch", "yaw", "roll"],
            "binary_controls": ["jump", "boost", "handbrake"],
        },
        "frozen_corpus": {
            "path": (CORPUS_ROOT / GATE_CORPUS_FILENAME)
            .relative_to(REPOSITORY_ROOT)
            .as_posix(),
            "sha256": sha256_file(CORPUS_ROOT / GATE_CORPUS_FILENAME),
            "episodes": 500,
        },
    }
