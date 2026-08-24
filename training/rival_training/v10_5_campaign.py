"""State and frozen contracts for the Stage-1 V4 reward experiment."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

from . import v10_4_campaign as previous
from .m10_campaign import write_json_atomic
from .v10_2_campaign import (
    SOURCE_ACTOR_SHA256,
    SOURCE_CHECKPOINT,
    SOURCE_MANIFEST_SHA256 as _SOURCE_MANIFEST_SHA256,
    actor_only_stage_transfer as _actor_only_stage_transfer,
    boundary_ppo_batch_agent_steps as _boundary_ppo_batch_agent_steps,
    utc_now,
)
from .v10_2_curriculum import FAMILIES
from .v10_3_curriculum import BALL_ACQUISITION_CURRICULUM_VERSION
from .v10_5_environment import BALL_ACQUISITION_ENVIRONMENT_VERSION
from .v10_5_reward import (
    APPROACH_NEGATIVE_EPISODE_BUDGET,
    APPROACH_POSITIVE_EPISODE_BUDGET,
    BALL_ACQUISITION_REWARD_VERSION,
    FIRST_PHYSICAL_TOUCH_REWARD,
    HEADING_ALIGNMENT_DELTA_SCALE,
    HEADING_NEGATIVE_EPISODE_BUDGET,
    HEADING_POSITIVE_EPISODE_BUDGET,
    IDLE_GRACE_SECONDS,
    IDLE_PENALTY_RATE_PER_SIMULATED_SECOND,
    IDLE_SPEED_THRESHOLD_UU_PER_SECOND,
    MAXIMUM_POSITIVE_EPISODE_REWARD,
    ball_acquisition_reward_metadata,
)
from .v9_actions import ACTION_VERSION
from .v9_checkpoint import action_schema_sha256, config_sha256, sha256_file
from .v9_observations import OBSERVATION_VERSION, observation_schema_manifest
from .v9_policy import CRITIC_VERSION, POLICY_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_CONFIG = REPOSITORY_ROOT / "training/configs/milestone10_5_stage1.json"
RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10_5"
CORPUS_ROOT = RESULT_ROOT / "stage_1/corpora"
CAMPAIGN_STATE_PATH = RESULT_ROOT / "stage1_campaign_state.json"
GATE_CORPUS_FILENAME = "stage1_v4_frozen_gate_corpus.json"
UNSEEN_CORPUS_FILENAME = "stage1_v4_unseen_generalization_corpus.json"
GATE_CORPUS_EPISODES_PER_FAMILY = 100
GENERALIZATION_CORPUS_EPISODES_PER_FAMILY = 50
ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR = 432_000
STAGE1_BOUNDARY_HOURS = (1.0, 2.5, 5.0)
EVALUATION_VERSION = "RivalBallAcquisitionEvaluationV4"
CAMPAIGN_ID = "rival-v10-5-stage1-turn-approach-first-touch"
STATE_FORMAT = "rival-v10-5-stage1-only-state-v1"
TERMINAL_DECISION = "stage1_turn_approach_touch_experiment_complete_at_plus_5h"
SOURCE_MANIFEST_SHA256 = _SOURCE_MANIFEST_SHA256
boundary_ppo_batch_agent_steps = _boundary_ppo_batch_agent_steps


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stage1_config(path: str | Path = DEFAULT_STAGE1_CONFIG) -> dict[str, Any]:
    config = _read_json(path)
    frozen = previous.load_stage1_config()
    observation = observation_schema_manifest()
    expected = {
        "config_version": "RivalM10_5Stage1TrainingConfigV1",
        "campaign_id": CAMPAIGN_ID,
        "stage": 1,
        "skill": "ball_acquisition",
        "scope": "stage_1_only",
        "policy_version": POLICY_VERSION,
        "critic_version": CRITIC_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "observation_schema_sha256": observation["schema_sha256"],
        "action_version": ACTION_VERSION,
        "action_schema_sha256": action_schema_sha256(),
        "reward_version": BALL_ACQUISITION_REWARD_VERSION,
        "reward_schedule_version": "RivalBallAcquisitionRewardScheduleV4",
        "curriculum_version": BALL_ACQUISITION_CURRICULUM_VERSION,
        "environment_version": BALL_ACQUISITION_ENVIRONMENT_VERSION,
        "evaluation_version": EVALUATION_VERSION,
    }
    mismatches = {
        key: {"config": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    metadata = ball_acquisition_reward_metadata()
    reward = config["reward_contract"]
    contract = config["stage_contract"]
    exact_reward_fields = {
        "first_physical_touch_reward": FIRST_PHYSICAL_TOUCH_REWARD,
        "subsequent_physical_touch_reward": 0.0,
        "maximum_touch_reward_per_episode": FIRST_PHYSICAL_TOUCH_REWARD,
        "heading_alignment_formula": metadata["heading_alignment_formula"],
        "heading_delta_scale": HEADING_ALIGNMENT_DELTA_SCALE,
        "heading_positive_episode_budget": HEADING_POSITIVE_EPISODE_BUDGET,
        "heading_negative_episode_budget": HEADING_NEGATIVE_EPISODE_BUDGET,
        "holding_heading_reward": 0.0,
        "distance_progress_formula": metadata["distance_progress_formula"],
        "approach_positive_episode_budget": APPROACH_POSITIVE_EPISODE_BUDGET,
        "approach_negative_episode_budget": APPROACH_NEGATIVE_EPISODE_BUDGET,
        "idle_grace_seconds": IDLE_GRACE_SECONDS,
        "idle_speed_threshold_uu_per_second": IDLE_SPEED_THRESHOLD_UU_PER_SECOND,
        "idle_penalty_rate_per_simulated_second": (IDLE_PENALTY_RATE_PER_SIMULATED_SECOND),
        "idle_penalty_per_eligible_tick": metadata["idle_penalty_per_eligible_tick"],
        "idle_eligible_seconds_before_no_touch_timeout": 11.5,
        "full_idle_penalty": -16.1,
        "maximum_positive_episode_reward": MAXIMUM_POSITIVE_EPISODE_REWARD,
        "generic_speed_reward": 0.0,
        "boost_reward": 0.0,
        "action_magnitude_reward": 0.0,
        "jump_reward": 0.0,
        "named_mechanic_rewards": 0.0,
        "goal_for_reward": 0.0,
        "goal_against_reward": 0.0,
    }
    reward_exact = set(reward) == set(exact_reward_fields) and all(
        reward[key] == value
        or (
            isinstance(value, float)
            and math.isclose(float(reward[key]), value, rel_tol=0.0, abs_tol=1e-12)
        )
        for key, value in exact_reward_fields.items()
    )
    checks = {
        "top_level_contract_exact": not mismatches,
        "backend_frozen": config["backend"] == frozen["backend"],
        "time_base_frozen": config["time_base"] == frozen["time_base"],
        "ppo_frozen": config["ppo"] == frozen["ppo"],
        "canonical_state_frozen": (
            config["canonical_state_version"] == frozen["canonical_state_version"]
            and config["canonical_adapter_version"] == frozen["canonical_adapter_version"]
        ),
        "source_exact_and_all_later_actors_forbidden": (
            contract["source_checkpoint"]
            == SOURCE_CHECKPOINT.relative_to(REPOSITORY_ROOT).as_posix()
            and contract["source_actor_sha256"] == SOURCE_ACTOR_SHA256
            and contract["forbid_m10_2_recovery_actor_as_source"] is True
            and contract["forbid_m10_3_actor_as_source"] is True
            and contract["forbid_m10_4_actor_as_source"] is True
        ),
        "fresh_stage1_state": (
            contract["actor_weights_only"] is True
            and contract["fresh_critic"] is True
            and contract["fresh_actor_optimizer"] is True
            and contract["fresh_critic_optimizer"] is True
        ),
        "reward_contract_exact": reward_exact,
        "stage1_only_scope": (
            contract["terminal_decision"] == TERMINAL_DECISION
            and contract["stage_2_authorized"] is False
            and contract["production_promotion_authorized"] is False
        ),
        "stage_budget_and_boundaries_exact": (
            contract["maximum_active_learner_steps"] == 2_160_000
            and contract["maximum_learner_simulated_hours"] == 5.0
            and contract["evaluation_boundaries_added_simulated_hours"]
            == list(STAGE1_BOUNDARY_HOURS)
        ),
    }
    if mismatches or not all(checks.values()):
        raise RuntimeError(f"Milestone 10.5 Stage-1 config mismatch: {mismatches}, checks={checks}")
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
            "M10.5 Stage 1 must restart from v10.1 +10h; all later actors are forbidden"
        )
    result = _actor_only_stage_transfer(source, config, device=device)
    result["trainer_state"].update(
        {
            "campaign_id": CAMPAIGN_ID,
            "scope": "stage_1_only",
            "m10_2_recovery_actor_used_as_source": False,
            "m10_3_actor_used_as_source": False,
            "m10_4_actor_used_as_source": False,
        }
    )
    result["proof"].update(
        {
            "m10_2_recovery_actor_used_as_source": False,
            "m10_3_actor_used_as_source": False,
            "m10_4_actor_used_as_source": False,
        }
    )
    return result


def build_stage1_corpus_manifests(*, output_root: str | Path = CORPUS_ROOT) -> dict[str, Any]:
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
                        "environment_seed": (seed_base + family_index * 100_000 + family_episode),
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
        "stage1_v4_frozen_gate_corpus",
        2026105100,
        GATE_CORPUS_EPISODES_PER_FAMILY,
    )
    unseen = build(
        UNSEEN_CORPUS_FILENAME,
        "stage1_v4_unseen_generalization_corpus",
        2026105200,
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
    return previous.boundary_slug(hours)


def nominal_stage1_steps(hours: float) -> int:
    if float(hours) not in STAGE1_BOUNDARY_HOURS:
        raise ValueError(f"Unsupported M10.5 boundary: {hours}")
    return int(round(float(hours) * ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR))


def initialize_progressive_state(
    *,
    path: str | Path = CAMPAIGN_STATE_PATH,
    transfer_proof: dict[str, Any],
    corpora: dict[str, Any],
) -> dict[str, Any]:
    destination = Path(path)
    if destination.is_file():
        state = _read_json(destination)
        if state.get("format") != STATE_FORMAT:
            raise RuntimeError("Refusing an unknown M10.5 campaign state")
        return state
    state = {
        "format": STATE_FORMAT,
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "scope": "stage_1_only",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "campaign_wall_clock_started_utc": None,
        "campaign_wall_clock_started_monotonic": None,
        "campaign_wall_clock_elapsed_seconds": 0.0,
        "campaign_wall_clock_remaining_seconds": 36_000.0,
        "current_stage": 1,
        "current_skill": "turn_approach_first_touch",
        "current_phase": "preflight",
        "source_checkpoint": transfer_proof["source_checkpoint"],
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "m10_2_recovery_actor_used_as_source": False,
        "m10_3_actor_used_as_source": False,
        "m10_4_actor_used_as_source": False,
        "stage_active_learner_steps": 0,
        "stage_simulated_hours": 0.0,
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
    updates: dict[str, Any], *, path: str | Path = CAMPAIGN_STATE_PATH
) -> dict[str, Any]:
    state = _read_json(path)
    state.update(deepcopy(updates))
    state["updated_utc"] = utc_now()
    write_json_atomic(path, state)
    return state


def start_real_campaign_clock(*, path: str | Path = CAMPAIGN_STATE_PATH) -> dict[str, Any]:
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
    elapsed = 0.0 if started is None else max(0.0, time.monotonic() - float(started))
    authority = float(selected["wall_clock"]["total_progressive_authority_seconds"])
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
