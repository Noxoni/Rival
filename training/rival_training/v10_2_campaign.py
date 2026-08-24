"""Authority, transfer, corpus, and resumable state for Rival v10.2."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .m10_campaign import checkpoint_record, write_json_atomic
from .v10_2_curriculum import (
    BALL_ACQUISITION_CURRICULUM_VERSION,
    FAMILIES,
    PHASE_WEIGHTS,
)
from .v10_2_environment import BALL_ACQUISITION_ENVIRONMENT_VERSION
from .v10_2_reward import BALL_ACQUISITION_REWARD_VERSION
from .v9_actions import ACTION_VERSION
from .v9_checkpoint import (
    action_schema_sha256,
    config_sha256,
    load_v9_checkpoint,
    sha256_file,
)
from .v9_observations import OBSERVATION_VERSION, observation_schema_manifest
from .v9_policy import (
    CRITIC_VERSION,
    POLICY_VERSION,
    RivalCriticV1,
    RivalPolicyV1,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_CONFIG = (
    REPOSITORY_ROOT / "training/configs/milestone10_2_stage1.json"
)
PROGRESSIVE_AUTHORITY = (
    REPOSITORY_ROOT / "handoff/v10.2/M10_2_PROGRESSIVE_CAMPAIGN.json"
)
STAGE1_AUTHORITY = REPOSITORY_ROOT / "handoff/v10.2/M10_2_CAMPAIGN.json"
SOURCE_CHECKPOINT = REPOSITORY_ROOT / (
    "training/checkpoints/milestone10_1/boundaries/plus-010h/032019870"
)
SOURCE_ACTOR_SHA256 = (
    "e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6"
)
SOURCE_MANIFEST_SHA256 = (
    "d1a785ef439b0127b5ab1a9ff1693ade1aa11d850151cd17b9733bbeb98dacb3"
)
RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10_2"
CORPUS_ROOT = RESULT_ROOT / "stage_1/corpora"
CAMPAIGN_STATE_PATH = RESULT_ROOT / "progressive_campaign_state.json"
GATE_CORPUS_EPISODES_PER_FAMILY = 100
GENERALIZATION_CORPUS_EPISODES_PER_FAMILY = 50
ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR = 432_000
STAGE1_BOUNDARY_HOURS = (1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_stage1_config(
    path: str | Path = DEFAULT_STAGE1_CONFIG,
) -> dict[str, Any]:
    config = _read_json(path)
    progressive = _read_json(PROGRESSIVE_AUTHORITY)
    stage1 = _read_json(STAGE1_AUTHORITY)
    observation = observation_schema_manifest()
    expected = {
        "config_version": "RivalM10_2Stage1TrainingConfigV1",
        "campaign_id": stage1["campaign_id"],
        "stage": 1,
        "skill": "ball_acquisition",
        "policy_version": POLICY_VERSION,
        "critic_version": CRITIC_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "observation_schema_sha256": observation["schema_sha256"],
        "action_version": ACTION_VERSION,
        "action_schema_sha256": action_schema_sha256(),
        "reward_version": BALL_ACQUISITION_REWARD_VERSION,
        "curriculum_version": BALL_ACQUISITION_CURRICULUM_VERSION,
        "environment_version": BALL_ACQUISITION_ENVIRONMENT_VERSION,
        "evaluation_version": "RivalBallAcquisitionEvaluationV1",
    }
    mismatches = {
        key: {"config": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    ppo_authority = stage1["ppo"]
    ppo_expected = {
        "gamma": ppo_authority["gamma"],
        "gae_lambda": ppo_authority["gae_lambda"],
        "rollout_agent_steps_per_iteration": ppo_authority[
            "rollout_trainable_agent_steps_per_iteration"
        ],
        "ppo_batch_agent_steps": ppo_authority[
            "ppo_batch_trainable_agent_steps"
        ],
        "minibatch_agent_steps": ppo_authority[
            "minibatch_trainable_agent_steps"
        ],
        "epochs": ppo_authority["epochs"],
        "clip_range": ppo_authority["clip_range"],
        "actor_learning_rate": ppo_authority["actor_learning_rate"],
        "critic_learning_rate": ppo_authority["critic_learning_rate"],
        "analog_entropy_coefficient": ppo_authority[
            "analog_entropy_coefficient"
        ],
        "button_entropy_coefficient": ppo_authority[
            "button_entropy_coefficient"
        ],
        "max_gradient_norm": ppo_authority["max_gradient_norm"],
    }
    ppo_mismatches = {
        key: {"config": config["ppo"].get(key), "expected": value}
        for key, value in ppo_expected.items()
        if config["ppo"].get(key) != value
    }
    contract = config["stage_contract"]
    checks = {
        "top_level_contract_exact": not mismatches,
        "ppo_contract_exact": not ppo_mismatches,
        "worker_count_starts_at_56": config["backend"]["worker_count"]
        == 56,
        "one_trainable_plus_one_dummy": config["time_base"][
            "trainable_agents_per_environment"
        ]
        == 1
        and config["time_base"]["dummy_agents_per_environment"] == 1,
        "native_120hz_one_tick_delay": config["time_base"]["physics_hz"]
        == 120
        and config["time_base"]["policy_hz"] == 120
        and config["time_base"]["repeat_action"] is False
        and config["time_base"]["one_tick_action_delay"] is True,
        "source_actor_exact": contract["source_actor_sha256"]
        == SOURCE_ACTOR_SHA256,
        "stage_budget_exact": contract["maximum_active_learner_steps"]
        == int(stage1["budget"]["maximum_added_trainable_active_learner_steps"])
        and contract["maximum_learner_simulated_hours"]
        == float(stage1["budget"]["maximum_added_simulated_game_hours"]),
        "phase_weights_exact": stage1["phase_a_curriculum"]
        == PHASE_WEIGHTS["A"]
        and stage1["phase_b_curriculum"] == PHASE_WEIGHTS["B"],
        "wall_clock_exact": config["wall_clock"][
            "total_progressive_authority_seconds"
        ]
        == int(progressive["overnight_wall_clock"]["total_authority_hours"] * 3600)
        and config["wall_clock"]["finalization_reserve_seconds"]
        == int(
            progressive["overnight_wall_clock"][
                "finalization_reserve_minutes"
            ]
            * 60
        ),
        "production_promotion_not_authorized": contract[
            "production_promotion_authorized"
        ]
        is False,
    }
    if mismatches or ppo_mismatches or not all(checks.values()):
        raise RuntimeError(
            "Milestone 10.2 Stage-1 config mismatch: "
            f"mismatches={mismatches}, ppo={ppo_mismatches}, checks={checks}"
        )
    return config


def _tensor_tree_sha256(values: tuple[torch.Tensor, ...]) -> str:
    digest = hashlib.sha256()
    for tensor in values:
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def actor_only_stage_transfer(
    source_checkpoint: str | Path,
    config: dict[str, Any],
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    """Load actor weights exactly and create every learning state fresh."""

    source = Path(source_checkpoint)
    if sha256_file(source / "actor.pt") != SOURCE_ACTOR_SHA256:
        raise RuntimeError("The v10.1 +10h source actor hash is not exact")
    if sha256_file(source / "checkpoint_manifest.json") != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("The v10.1 +10h source manifest hash is not exact")
    loaded = load_v9_checkpoint(source, device=device)
    selected_device = torch.device(device)
    actor = RivalPolicyV1().to(selected_device)
    actor.load_state_dict(deepcopy(loaded["actor"].state_dict()), strict=True)
    critic = RivalCriticV1().to(selected_device)
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=float(config["ppo"]["actor_learning_rate"])
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["ppo"]["critic_learning_rate"])
    )
    held = torch.as_tensor(
        loaded["reload_observations"],
        dtype=torch.float32,
        device=selected_device,
    )
    with torch.inference_mode():
        source_outputs = loaded["actor"](held)
        transferred_outputs = actor(held)
    outputs_exact = all(
        torch.equal(left, right)
        for left, right in zip(source_outputs, transferred_outputs, strict=True)
    )
    maximum_error = max(
        float((left - right).abs().max().detach().cpu())
        for left, right in zip(source_outputs, transferred_outputs, strict=True)
    )
    source_record = checkpoint_record(source, manifest=loaded["manifest"])
    proof = {
        "transfer_version": "RivalActorOnlySkillTransferV1",
        "source_checkpoint": source_record,
        "source_actor_file_sha256": sha256_file(source / "actor.pt"),
        "transferred_actor_state_sha256": _state_dict_sha256(
            actor.state_dict()
        ),
        "fresh_critic_state_sha256": _state_dict_sha256(
            critic.state_dict()
        ),
        "held_observation_count": int(held.shape[0]),
        "source_output_sha256": _tensor_tree_sha256(source_outputs),
        "transferred_output_sha256": _tensor_tree_sha256(
            transferred_outputs
        ),
        "maximum_held_output_error": maximum_error,
        "source_critic_state_not_loaded": True,
        "source_actor_optimizer_state_not_loaded": True,
        "source_critic_optimizer_state_not_loaded": True,
        "fresh_actor_optimizer_state_entries": len(actor_optimizer.state),
        "fresh_critic_optimizer_state_entries": len(critic_optimizer.state),
        "checks": {
            "source_actor_file_exact": sha256_file(source / "actor.pt")
            == SOURCE_ACTOR_SHA256,
            "held_actor_outputs_exact": outputs_exact and maximum_error == 0.0,
            "fresh_critic": True,
            "fresh_actor_optimizer_empty": not actor_optimizer.state,
            "fresh_critic_optimizer_empty": not critic_optimizer.state,
        },
    }
    proof["checks"]["passed"] = all(proof["checks"].values())
    if not proof["checks"]["passed"]:
        raise RuntimeError(f"Actor-only transfer failed: {proof}")
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
            "stage": 1,
            "stage_phase": "A",
            "source_checkpoint": source_record,
            "source_actor_sha256": SOURCE_ACTOR_SHA256,
            "actor_weights_only_transfer": True,
            "fresh_critic_and_optimizers": True,
            "production_promotion_authorized": False,
        },
        "proof": proof,
    }


def build_stage1_corpus_manifests(
    *,
    output_root: str | Path = CORPUS_ROOT,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    def build(name: str, seed_base: int, per_family: int) -> dict[str, Any]:
        episodes = []
        index = 0
        for family_index, family in enumerate(FAMILIES):
            for family_episode in range(per_family):
                episodes.append(
                    {
                        "index": index,
                        "family": family,
                        "family_episode": family_episode,
                        "environment_seed": seed_base
                        + family_index * 100_000
                        + family_episode,
                        "active_team": family_episode % 2,
                        "mirror": bool((family_episode // 2) % 2),
                    }
                )
                index += 1
        manifest = {
            "schema_version": 1,
            "evaluation_version": "RivalBallAcquisitionEvaluationV1",
            "name": name,
            "seed_base": seed_base,
            "families": list(FAMILIES),
            "episodes_per_family": per_family,
            "episode_count": len(episodes),
            "episodes": episodes,
        }
        path = root / f"{name}.json"
        write_json_atomic(path, manifest)
        return {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(path),
            "episode_count": len(episodes),
        }

    gate = build(
        "stage1_frozen_gate_corpus",
        2026102100,
        GATE_CORPUS_EPISODES_PER_FAMILY,
    )
    unseen = build(
        "stage1_unseen_generalization_corpus",
        2026102200,
        GENERALIZATION_CORPUS_EPISODES_PER_FAMILY,
    )
    checks = {
        "gate_has_exactly_500_episodes": gate["episode_count"] == 500,
        "unseen_has_at_least_250_episodes": unseen["episode_count"] >= 250,
        "corpus_hashes_disjoint": gate["sha256"] != unseen["sha256"],
    }
    checks["passed"] = all(checks.values())
    return {"gate": gate, "unseen": unseen, "checks": checks}


def boundary_slug(hours: float) -> str:
    value = float(hours)
    if value not in STAGE1_BOUNDARY_HOURS:
        raise ValueError(f"Unsupported Stage-1 boundary: {hours}")
    return f"plus-{value:05.1f}h".replace(".0", "").replace(".", "p")


def nominal_stage1_steps(hours: float) -> int:
    value = float(hours)
    if value not in STAGE1_BOUNDARY_HOURS:
        raise ValueError(f"Unsupported Stage-1 boundary: {hours}")
    return int(round(value * ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR))


def initialize_progressive_state(
    *,
    path: str | Path = CAMPAIGN_STATE_PATH,
    transfer_proof: dict[str, Any],
    corpora: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    if destination.is_file():
        state = _read_json(destination)
        if state.get("format") != "rival-v10-2-progressive-state-v1":
            raise RuntimeError("Refusing an unknown progressive campaign state")
        return state
    state = {
        "format": "rival-v10-2-progressive-state-v1",
        "schema_version": 1,
        "campaign_id": "rival-v10-2-progressive-prerequisites-stage1-through-stage4",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "campaign_wall_clock_started_utc": None,
        "campaign_wall_clock_started_monotonic": None,
        "campaign_wall_clock_elapsed_seconds": 0.0,
        "campaign_wall_clock_remaining_seconds": 36000.0,
        "current_stage": 1,
        "current_skill": "ball_acquisition",
        "current_phase": "preflight",
        "source_checkpoint": transfer_proof["source_checkpoint"],
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "stage_active_learner_steps": 0,
        "stage_simulated_hours": 0.0,
        "total_progressive_active_learner_steps": 0,
        "total_progressive_simulated_hours": 0.0,
        "current_evaluation_boundary": None,
        "gate_decision": "pending_preflight",
        "passed_prerequisite_checkpoints": {},
        "latest_clean_recovery_checkpoint": None,
        "next_authorized_stage": None,
        "stop_reason": None,
        "corpora": corpora,
        "production_promotion_authorized": False,
    }
    write_json_atomic(destination, state)
    return state


def update_progressive_state(
    updates: dict[str, Any],
    *,
    path: str | Path = CAMPAIGN_STATE_PATH,
) -> dict[str, Any]:
    state = _read_json(path)
    state.update(deepcopy(updates))
    state["updated_utc"] = utc_now()
    write_json_atomic(path, state)
    return state


def start_real_campaign_clock(
    *, path: str | Path = CAMPAIGN_STATE_PATH
) -> dict[str, Any]:
    state = _read_json(path)
    if state["campaign_wall_clock_started_utc"] is None:
        state["campaign_wall_clock_started_utc"] = utc_now()
        state["campaign_wall_clock_started_monotonic"] = time.monotonic()
        state["current_phase"] = "A"
        state["gate_decision"] = "stage_1_training_in_progress"
        state["updated_utc"] = utc_now()
        write_json_atomic(path, state)
    return state


def wall_clock_status(
    *,
    state: dict[str, Any] | None = None,
    projected_iteration_and_boundary_seconds: float = 0.0,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    current = state or _read_json(CAMPAIGN_STATE_PATH)
    selected = config or load_stage1_config()
    started = current.get("campaign_wall_clock_started_monotonic")
    elapsed = (
        0.0 if started is None else max(0.0, time.monotonic() - float(started))
    )
    authority = float(
        selected["wall_clock"]["total_progressive_authority_seconds"]
    )
    reserve = float(selected["wall_clock"]["finalization_reserve_seconds"])
    remaining = max(0.0, authority - elapsed)
    projected = max(0.0, float(projected_iteration_and_boundary_seconds))
    return {
        "elapsed_seconds": elapsed,
        "remaining_seconds": remaining,
        "finalization_reserve_seconds": reserve,
        "projected_iteration_and_boundary_seconds": projected,
        "ordinary_iteration_allowed": remaining > reserve + projected,
        "wall_clock_exhausted": remaining <= reserve,
    }


def config_identity(config: dict[str, Any]) -> dict[str, str]:
    return {
        "path": DEFAULT_STAGE1_CONFIG.relative_to(REPOSITORY_ROOT).as_posix(),
        "file_sha256": sha256_file(DEFAULT_STAGE1_CONFIG),
        "canonical_sha256": config_sha256(config),
    }


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True
