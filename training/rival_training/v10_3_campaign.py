"""Authority, transfer, corpora, and state for Rival Milestone 10.3."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any

from .m10_campaign import write_json_atomic
from .v10_2_campaign import (
    SOURCE_ACTOR_SHA256,
    SOURCE_CHECKPOINT,
    SOURCE_MANIFEST_SHA256,
    actor_only_stage_transfer as _v10_2_actor_only_stage_transfer,
    boundary_ppo_batch_agent_steps as _boundary_ppo_batch_agent_steps,
    utc_now,
)
from .v10_2_curriculum import FAMILIES
from .v10_3_curriculum import (
    BALL_ACQUISITION_CURRICULUM_VERSION,
    ORDINARY_HEADING_ERROR_DEGREES,
)
from .v10_3_environment import BALL_ACQUISITION_ENVIRONMENT_VERSION
from .v10_3_reward import (
    BALL_ACQUISITION_REWARD_VERSION,
    IDLE_GRACE_SECONDS,
    IDLE_PENALTY_RATE_PER_SIMULATED_SECOND,
    IDLE_SPEED_THRESHOLD_UU_PER_SECOND,
)
from .v9_actions import ACTION_VERSION
from .v9_checkpoint import action_schema_sha256, config_sha256, sha256_file
from .v9_observations import OBSERVATION_VERSION, observation_schema_manifest
from .v9_policy import CRITIC_VERSION, POLICY_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_CONFIG = (
    REPOSITORY_ROOT / "training/configs/milestone10_3_stage1.json"
)
CAMPAIGN_AUTHORITY = REPOSITORY_ROOT / "handoff/v10.3/M10_3_CAMPAIGN.json"
RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10_3"
CORPUS_ROOT = RESULT_ROOT / "stage_1/corpora"
CAMPAIGN_STATE_PATH = RESULT_ROOT / "progressive_campaign_state.json"
GATE_CORPUS_FILENAME = "stage1_v2_frozen_gate_corpus.json"
UNSEEN_CORPUS_FILENAME = "stage1_v2_unseen_generalization_corpus.json"
GATE_CORPUS_EPISODES_PER_FAMILY = 100
GENERALIZATION_CORPUS_EPISODES_PER_FAMILY = 50
ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR = 432_000
STAGE1_BOUNDARY_HOURS = (1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)
EVALUATION_VERSION = "RivalBallAcquisitionEvaluationV2"
boundary_ppo_batch_agent_steps = _boundary_ppo_batch_agent_steps


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stage1_config(
    path: str | Path = DEFAULT_STAGE1_CONFIG,
) -> dict[str, Any]:
    config = _read_json(path)
    authority = _read_json(CAMPAIGN_AUTHORITY)
    stage = authority["stage_1_v2"]
    architecture = authority["frozen_architecture"]
    source = authority["source"]
    observation = observation_schema_manifest()
    expected = {
        "config_version": "RivalM10_3Stage1TrainingConfigV1",
        "campaign_id": authority["campaign_id"],
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
        "evaluation_version": EVALUATION_VERSION,
    }
    mismatches = {
        key: {"config": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    ppo = authority["ppo"]
    expected_ppo = {
        "gamma": ppo["gamma"],
        "gae_lambda": ppo["gae_lambda"],
        "rollout_agent_steps_per_iteration": ppo[
            "rollout_trainable_agent_steps_per_iteration"
        ],
        "ppo_batch_agent_steps": ppo["ppo_batch_trainable_agent_steps"],
        "minibatch_agent_steps": ppo["minibatch_trainable_agent_steps"],
        "epochs": ppo["epochs"],
        "clip_range": ppo["clip_range"],
        "actor_learning_rate": ppo["actor_learning_rate"],
        "critic_learning_rate": ppo["critic_learning_rate"],
        "analog_entropy_coefficient": ppo["analog_entropy_coefficient"],
        "button_entropy_coefficient": ppo["button_entropy_coefficient"],
        "max_gradient_norm": ppo["max_gradient_norm"],
    }
    ppo_mismatches = {
        key: {"config": config["ppo"].get(key), "expected": value}
        for key, value in expected_ppo.items()
        if config["ppo"].get(key) != value
    }
    reward_delta = stage["reward_delta"]
    curriculum_delta = stage["curriculum_delta"]
    contract = config["stage_contract"]
    checks = {
        "top_level_contract_exact": not mismatches,
        "ppo_contract_exact": not ppo_mismatches,
        "source_checkpoint_exact": (
            contract["source_checkpoint"] == source["checkpoint"]
            and contract["source_actor_sha256"]
            == source["expected_actor_sha256"]
            == SOURCE_ACTOR_SHA256
            and source["expected_manifest_sha256"] == SOURCE_MANIFEST_SHA256
        ),
        "m10_2_recovery_source_forbidden": (
            contract["forbid_m10_2_recovery_actor_as_source"] is True
            and source["explicitly_forbid_m10_2_recovery_actor_as_source"] is True
        ),
        "native_120hz_one_tick_delay": (
            config["time_base"]["physics_hz"]
            == architecture["physics_hz"]
            == 120
            and config["time_base"]["policy_hz"]
            == architecture["policy_hz"]
            == 120
            and config["time_base"]["repeat_action"]
            is architecture["repeat_action"]
            is False
            and config["time_base"]["one_tick_action_delay"]
            is architecture["one_tick_action_delay"]
            is True
        ),
        "one_trainable_plus_one_dummy": (
            config["time_base"]["trainable_agents_per_environment"] == 1
            and config["time_base"]["dummy_agents_per_environment"] == 1
        ),
        "v2_reward_delta_exact": (
            reward_delta["idle_grace_seconds"] == IDLE_GRACE_SECONDS
            and reward_delta["idle_speed_threshold_uu_per_s"]
            == IDLE_SPEED_THRESHOLD_UU_PER_SECOND
            and reward_delta["idle_penalty_rate_per_simulated_second"]
            == IDLE_PENALTY_RATE_PER_SIMULATED_SECOND
            and reward_delta["idle_only_before_first_touch"] is True
            and reward_delta["generic_speed_reward"] == 0.0
            and reward_delta["action_magnitude_reward"] == 0.0
            and reward_delta["goal_for_reward"] == 0.0
            and reward_delta["goal_against_reward"] == 0.0
        ),
        "v2_heading_delta_exact": (
            curriculum_delta["stationary_close_heading_error_degrees"]
            == ORDINARY_HEADING_ERROR_DEGREES["stationary_close"]
            and curriculum_delta["stationary_medium_heading_error_degrees"]
            == ORDINARY_HEADING_ERROR_DEGREES["stationary_medium"]
            and curriculum_delta["moving_chase_heading_error_degrees"]
            == ORDINARY_HEADING_ERROR_DEGREES["moving_chase"]
            and curriculum_delta["awkward_heading_unchanged"] is True
            and curriculum_delta["natural_kickoff_holdout_unchanged"] is True
        ),
        "stage_budget_exact": (
            contract["maximum_active_learner_steps"]
            == stage["maximum_active_learner_steps"]
            == 6_480_000
            and contract["maximum_learner_simulated_hours"]
            == stage["maximum_learner_simulated_hours"]
            == 15.0
        ),
        "boundary_ladder_exact": (
            contract["evaluation_boundaries_added_simulated_hours"]
            == stage["evaluation_boundaries_added_simulated_hours"]
            == list(STAGE1_BOUNDARY_HOURS)
        ),
        "wall_clock_exact": (
            config["wall_clock"]["total_progressive_authority_seconds"]
            == int(authority["progressive_ladder"]["global_wall_clock_hours"] * 3600)
            and config["wall_clock"]["finalization_reserve_seconds"]
            == int(
                authority["progressive_ladder"]["finalization_reserve_minutes"]
                * 60
            )
        ),
        "production_promotion_not_authorized": (
            contract["production_promotion_authorized"] is False
            and authority["production"]["promotion_authorized"] is False
        ),
    }
    if mismatches or ppo_mismatches or not all(checks.values()):
        raise RuntimeError(
            "Milestone 10.3 Stage-1 config mismatch: "
            f"mismatches={mismatches}, ppo={ppo_mismatches}, checks={checks}"
        )
    return config


def actor_only_stage_transfer(
    source_checkpoint: str | Path,
    config: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    source = Path(source_checkpoint).resolve()
    if source != SOURCE_CHECKPOINT.resolve():
        raise RuntimeError(
            "M10.3 Stage 1 must restart from v10.1 +10h; M10.2 recovery is forbidden"
        )
    result = _v10_2_actor_only_stage_transfer(source, config, device=device)
    result["trainer_state"].update(
        {
            "campaign_id": "rival-v10-3-stage1-anti-idle-progressive-retry",
            "m10_2_recovery_actor_used_as_source": False,
        }
    )
    result["proof"]["m10_2_recovery_actor_used_as_source"] = False
    return result


def build_stage1_corpus_manifests(
    *, output_root: str | Path = CORPUS_ROOT
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    def build(filename: str, name: str, seed_base: int, per_family: int) -> dict[str, Any]:
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
            "evaluation_version": EVALUATION_VERSION,
            "name": name,
            "seed_base": seed_base,
            "families": list(FAMILIES),
            "episodes_per_family": per_family,
            "episode_count": len(episodes),
            "episodes": episodes,
        }
        path = root / filename
        write_json_atomic(path, manifest)
        return {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(path),
            "episode_count": len(episodes),
        }

    gate = build(
        GATE_CORPUS_FILENAME,
        "stage1_v2_frozen_gate_corpus",
        2026103100,
        GATE_CORPUS_EPISODES_PER_FAMILY,
    )
    unseen = build(
        UNSEEN_CORPUS_FILENAME,
        "stage1_v2_unseen_generalization_corpus",
        2026103200,
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
        if state.get("format") != "rival-v10-3-progressive-state-v1":
            raise RuntimeError("Refusing an unknown progressive campaign state")
        return state
    state = {
        "format": "rival-v10-3-progressive-state-v1",
        "schema_version": 1,
        "campaign_id": "rival-v10-3-stage1-anti-idle-progressive-retry",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "campaign_wall_clock_started_utc": None,
        "campaign_wall_clock_started_monotonic": None,
        "campaign_wall_clock_elapsed_seconds": 0.0,
        "campaign_wall_clock_remaining_seconds": 36_000.0,
        "current_stage": 1,
        "current_skill": "ball_acquisition",
        "current_phase": "preflight",
        "source_checkpoint": transfer_proof["source_checkpoint"],
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "m10_2_recovery_actor_used_as_source": False,
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
    *, path: str | Path = CAMPAIGN_STATE_PATH,
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
