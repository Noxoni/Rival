"""Frozen contracts and exact source transfer for Milestone 10.7."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .v10_2_campaign import (
    SOURCE_ACTOR_SHA256,
    SOURCE_CHECKPOINT,
    SOURCE_MANIFEST_SHA256,
)
from .v10_6_campaign import CORPUS_ROOT, GATE_CORPUS_FILENAME
from .v10_6_environment import BALL_ACQUISITION_ENVIRONMENT_VERSION
from .v10_6_reward import BALL_ACQUISITION_REWARD_VERSION
from .v10_7_actions import (
    BUTTON_POLICY_VERSION,
    BUTTON_PERSISTENCE,
    button_policy_metadata,
)
from .v10_7_policy import (
    POLICY_VERSION,
    RivalPolicyV1IndependentStickyButtons,
)
from .v9_actions import ACTION_VERSION
from .v9_checkpoint import action_schema_sha256, load_v9_checkpoint, sha256_file
from .v9_observations import OBSERVATION_VERSION, observation_schema_manifest
from .v9_policy import CRITIC_VERSION, RivalCriticV1


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_CONFIG = REPOSITORY_ROOT / "training/configs/milestone10_7_stage1.json"
RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10_7"
CHECKPOINT_ROOT = REPOSITORY_ROOT / "training/checkpoints/milestone10_7/stage_1"
BOUNDARY_HOURS = (0.5, 1.0, 2.5)
ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR = 432_000
CAMPAIGN_ID = "rival-v10-7-stage1-action-policy-correction"


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def load_stage1_config(path: str | Path = DEFAULT_STAGE1_CONFIG) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    frozen = json.loads(
        (REPOSITORY_ROOT / "training/configs/milestone10_6_stage1.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        "config_version": "RivalM10_7Stage1TrainingConfigV1",
        "campaign_id": CAMPAIGN_ID,
        "stage": 1,
        "scope": "stage_1_only",
        "policy_version": POLICY_VERSION,
        "critic_version": CRITIC_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "observation_schema_sha256": observation_schema_manifest()["schema_sha256"],
        "action_version": ACTION_VERSION,
        "action_schema_sha256": action_schema_sha256(),
        "button_policy_version": BUTTON_POLICY_VERSION,
        "reward_version": BALL_ACQUISITION_REWARD_VERSION,
        "reward_schedule_version": frozen["reward_schedule_version"],
        "curriculum_version": frozen["curriculum_version"],
        "environment_version": BALL_ACQUISITION_ENVIRONMENT_VERSION,
    }
    mismatches = {
        key: {"actual": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    frozen_ppo = dict(frozen["ppo"])
    expected_ppo = dict(frozen_ppo)
    expected_ppo["button_entropy_schedule"] = {
        "initial_coefficient": 0.001,
        "final_coefficient": 0.0,
        "anneal_active_learner_simulated_hours": 0.5,
        "remain_at_final_after_anneal": True,
    }
    contract = config["stage_contract"]
    checks = {
        "identity_exact": not mismatches,
        "reward_contract_byte_semantics_frozen": config["reward_contract"]
        == frozen["reward_contract"],
        "backend_frozen": config["backend"] == frozen["backend"],
        "time_base_frozen": config["time_base"] == frozen["time_base"],
        "ppo_only_button_entropy_schedule_added": config["ppo"] == expected_ppo,
        "canonical_state_frozen": all(
            config[key] == frozen[key]
            for key in ("canonical_state_version", "canonical_adapter_version")
        ),
        "source_exact": (
            contract["source_checkpoint"]
            == SOURCE_CHECKPOINT.relative_to(REPOSITORY_ROOT).as_posix()
            and contract["source_actor_sha256"] == SOURCE_ACTOR_SHA256
        ),
        "all_m10_2_through_m10_6_actors_forbidden": all(
            contract[f"forbid_m10_{version}_actor_as_source"] is True
            for version in range(2, 7)
        ),
        "transfer_and_fresh_state_exact": all(
            contract[key] is True
            for key in (
                "transfer_encoder_trunk_exactly",
                "transfer_five_analog_mean_head_weights_exactly",
                "transfer_five_analog_log_std_exactly",
                "replace_only_discrete_button_head",
                "fresh_critic",
                "fresh_actor_optimizer",
                "fresh_critic_optimizer",
                "dummy_excluded_from_ppo",
            )
        ),
        "persistence_exact": config["button_policy"]["persistence"]
        == BUTTON_PERSISTENCE,
        "boundaries_exact": (
            contract["evaluation_boundaries_added_simulated_hours"]
            == list(BOUNDARY_HOURS)
            and contract["maximum_active_learner_steps"] == 1_080_000
            and contract["automatically_continue_to_plus_5h"] is False
        ),
        "stage2_and_production_forbidden": (
            contract["stage_2_authorized"] is False
            and contract["production_promotion_authorized"] is False
        ),
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"Milestone 10.7 config mismatch: {mismatches}, {checks}")
    return config


def button_entropy_coefficient(
    cumulative_active_learner_steps: int,
    config: dict[str, Any],
) -> float:
    schedule = config["ppo"]["button_entropy_schedule"]
    anneal_steps = int(
        round(
            float(schedule["anneal_active_learner_simulated_hours"])
            * ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR
        )
    )
    fraction = min(max(int(cumulative_active_learner_steps), 0), anneal_steps) / float(
        anneal_steps
    )
    initial = float(schedule["initial_coefficient"])
    final = float(schedule["final_coefficient"])
    return initial + fraction * (final - initial)


def actor_only_architecture_transfer(
    source_checkpoint: str | Path,
    config: dict[str, Any],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Transfer only encoder/trunk and analog action parameters."""

    source = Path(source_checkpoint).resolve()
    if source != SOURCE_CHECKPOINT.resolve():
        raise RuntimeError("M10.7 requires the exact v10.1 +10h source checkpoint")
    if sha256_file(source / "actor.pt") != SOURCE_ACTOR_SHA256:
        raise RuntimeError("The v10.1 +10h source actor SHA-256 is not exact")
    if sha256_file(source / "checkpoint_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("The v10.1 +10h source manifest SHA-256 is not exact")

    loaded = load_v9_checkpoint(source, device=device)
    selected_device = torch.device(device)
    actor = RivalPolicyV1IndependentStickyButtons().to(selected_device)
    initial_button_state = {
        name: value.detach().clone()
        for name, value in actor.state_dict().items()
        if name.startswith("action_head.button_logits.")
    }
    source_state = loaded["actor"].state_dict()
    target_state = actor.state_dict()
    transfer_names = sorted(
        name
        for name in target_state
        if name.startswith("encoder.")
        or name in {
            "action_head.analog_mean.weight",
            "action_head.analog_mean.bias",
            "action_head.analog_log_std",
        }
    )
    new_names = sorted(set(target_state) - set(transfer_names))
    if new_names != [
        "action_head.button_logits.bias",
        "action_head.button_logits.weight",
    ]:
        raise RuntimeError(f"Unexpected new M10.7 actor parameters: {new_names}")
    with torch.no_grad():
        for name in transfer_names:
            target_state[name].copy_(source_state[name])
    actor.load_state_dict(target_state, strict=True)
    critic = RivalCriticV1(seed=20261072).to(selected_device)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=float(config["ppo"]["actor_learning_rate"])
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["ppo"]["critic_learning_rate"])
    )

    transferred_exact = {
        name: bool(torch.equal(actor.state_dict()[name], source_state[name]))
        for name in transfer_names
    }
    new_unchanged = {
        name: bool(torch.equal(actor.state_dict()[name], initial_button_state[name]))
        for name in new_names
    }
    held = torch.as_tensor(
        loaded["reload_observations"], dtype=torch.float32, device=selected_device
    )
    with torch.inference_mode():
        source_mean, source_log_std, _ = loaded["actor"](held)
        target_mean, target_log_std, _ = actor(held)
    proof = {
        "transfer_version": "RivalM10_7SelectiveActionHeadTransferV1",
        "source_checkpoint": SOURCE_CHECKPOINT.relative_to(REPOSITORY_ROOT).as_posix(),
        "source_actor_file_sha256": sha256_file(source / "actor.pt"),
        "source_manifest_sha256": sha256_file(source / "checkpoint_manifest.json"),
        "transferred_parameter_names": transfer_names,
        "transferred_parameter_count": sum(target_state[name].numel() for name in transfer_names),
        "newly_initialized_parameter_names": new_names,
        "newly_initialized_parameter_count": sum(target_state[name].numel() for name in new_names),
        "per_parameter_transfer_exact": transferred_exact,
        "per_parameter_new_initialization_preserved": new_unchanged,
        "source_analog_mean_output_sha256": _tensor_sha256(source_mean),
        "transferred_analog_mean_output_sha256": _tensor_sha256(target_mean),
        "source_analog_log_std_output_sha256": _tensor_sha256(source_log_std),
        "transferred_analog_log_std_output_sha256": _tensor_sha256(target_log_std),
        "new_button_head_state_sha256": _state_dict_sha256(initial_button_state),
        "complete_actor_state_sha256": _state_dict_sha256(actor.state_dict()),
        "fresh_critic_state_sha256": _state_dict_sha256(critic.state_dict()),
        "source_critic_state_not_loaded": True,
        "source_actor_optimizer_state_not_loaded": True,
        "source_critic_optimizer_state_not_loaded": True,
        "fresh_actor_optimizer_state_entries": len(actor_optimizer.state),
        "fresh_critic_optimizer_state_entries": len(critic_optimizer.state),
        "checks": {
            "source_actor_exact": sha256_file(source / "actor.pt") == SOURCE_ACTOR_SHA256,
            "all_encoder_trunk_and_analog_parameters_exact": all(
                transferred_exact.values()
            ),
            "only_button_head_newly_initialized": new_names
            == [
                "action_head.button_logits.bias",
                "action_head.button_logits.weight",
            ],
            "new_button_initialization_not_overwritten": all(new_unchanged.values()),
            "held_analog_mean_outputs_exact": torch.equal(source_mean, target_mean),
            "held_analog_log_std_outputs_exact": torch.equal(
                source_log_std, target_log_std
            ),
            "fresh_critic": True,
            "fresh_actor_optimizer_empty": not actor_optimizer.state,
            "fresh_critic_optimizer_empty": not critic_optimizer.state,
        },
    }
    proof["checks"]["passed"] = all(proof["checks"].values())
    if not proof["checks"]["passed"]:
        raise RuntimeError(f"M10.7 selective transfer failed: {proof}")
    return {
        "actor": actor,
        "critic": critic,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "reload_observations": loaded["reload_observations"].copy(),
        "trainer_state": {
            "completed_iterations": 0,
            "cumulative_agent_steps": 0,
            "cumulative_model_updates": 0,
            "campaign_id": CAMPAIGN_ID,
            "stage": 1,
            "stage_phase": "A",
            "source_checkpoint": proof["source_checkpoint"],
            "source_actor_sha256": SOURCE_ACTOR_SHA256,
            "selective_actor_transfer": True,
            "fresh_critic_and_optimizers": True,
            "clean_boundary": True,
            "partial_experience_buffer_records": 0,
            "production_promotion_authorized": False,
        },
        "proof": proof,
    }


def configuration_evidence(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "config_path": DEFAULT_STAGE1_CONFIG.relative_to(REPOSITORY_ROOT).as_posix(),
        "config_sha256": sha256_file(DEFAULT_STAGE1_CONFIG),
        "button_policy": button_policy_metadata(),
        "reward_contract_identical_to_m10_6": True,
        "frozen_gate_corpus": {
            "path": (CORPUS_ROOT / GATE_CORPUS_FILENAME)
            .relative_to(REPOSITORY_ROOT)
            .as_posix(),
            "sha256": sha256_file(CORPUS_ROOT / GATE_CORPUS_FILENAME),
            "episode_count": 500,
        },
        "button_entropy_coefficients": {
            "at_0h": button_entropy_coefficient(0, config),
            "at_0p25h": button_entropy_coefficient(108_000, config),
            "at_0p5h": button_entropy_coefficient(216_000, config),
            "at_1h": button_entropy_coefficient(432_000, config),
        },
    }
