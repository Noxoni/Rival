"""State and frozen contracts for the Stage-1 uncapped reacquisition experiment."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import time
from typing import Any

from . import v10_5_campaign as previous
from .m10_campaign import write_json_atomic
from .v10_2_campaign import (
    SOURCE_ACTOR_SHA256,
    SOURCE_CHECKPOINT,
    SOURCE_MANIFEST_SHA256 as _SOURCE_MANIFEST_SHA256,
    utc_now,
)
from .v10_3_curriculum import BALL_ACQUISITION_CURRICULUM_VERSION
from .v10_6_environment import BALL_ACQUISITION_ENVIRONMENT_VERSION
from .v10_6_reward import (
    ACQUISITION_GRACE_SECONDS,
    ACQUISITION_TIME_PENALTY_RATE_PER_SECOND,
    BALL_ACQUISITION_REWARD_VERSION,
    FAILED_ACQUISITION_WINDOW_PENALTY,
    HEADING_ALIGNMENT_DELTA_SCALE,
    MAXIMUM_CONTACT_REWARD_PER_EPISODE,
    PHYSICAL_CONTACT_REWARD,
    REWARDED_CONTACT_LIMIT,
    ball_acquisition_reward_metadata,
)
from .v9_actions import ACTION_VERSION
from .v9_checkpoint import action_schema_sha256, config_sha256, sha256_file
from .v9_observations import OBSERVATION_VERSION, observation_schema_manifest
from .v9_policy import CRITIC_VERSION, POLICY_VERSION


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE1_CONFIG = REPOSITORY_ROOT / "training/configs/milestone10_6_stage1.json"
RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10_6"
CORPUS_ROOT = REPOSITORY_ROOT / "training/results/milestone10_5/stage_1/corpora"
CAMPAIGN_STATE_PATH = RESULT_ROOT / "stage1_campaign_state.json"
GATE_CORPUS_FILENAME = previous.GATE_CORPUS_FILENAME
UNSEEN_CORPUS_FILENAME = previous.UNSEEN_CORPUS_FILENAME
ACTIVE_LEARNER_STEPS_PER_SIMULATED_HOUR = 432_000
STAGE1_BOUNDARY_HOURS = (1.0, 2.5, 5.0)
EVALUATION_VERSION = "RivalBallAcquisitionEvaluationV5"
FROZEN_CORPUS_EVALUATION_VERSION = previous.EVALUATION_VERSION
CAMPAIGN_ID = "rival-v10-6-stage1-uncapped-reacquisition"
STATE_FORMAT = "rival-v10-6-stage1-only-state-v1"
TERMINAL_DECISION = "stage1_uncapped_reacquisition_experiment_complete_at_plus_5h"
SOURCE_MANIFEST_SHA256 = _SOURCE_MANIFEST_SHA256
boundary_ppo_batch_agent_steps = previous.boundary_ppo_batch_agent_steps


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_stage1_config(path: str | Path = DEFAULT_STAGE1_CONFIG) -> dict[str, Any]:
    config = _read_json(path)
    frozen = previous.load_stage1_config()
    metadata = ball_acquisition_reward_metadata()
    expected = {
        "config_version": "RivalM10_6Stage1TrainingConfigV1",
        "campaign_id": CAMPAIGN_ID,
        "stage": 1,
        "skill": "ball_acquisition",
        "scope": "stage_1_only",
        "policy_version": POLICY_VERSION,
        "critic_version": CRITIC_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "observation_schema_sha256": observation_schema_manifest()["schema_sha256"],
        "action_version": ACTION_VERSION,
        "action_schema_sha256": action_schema_sha256(),
        "reward_version": BALL_ACQUISITION_REWARD_VERSION,
        "reward_schedule_version": "RivalBallAcquisitionRewardScheduleV5",
        "curriculum_version": BALL_ACQUISITION_CURRICULUM_VERSION,
        "environment_version": BALL_ACQUISITION_ENVIRONMENT_VERSION,
        "evaluation_version": EVALUATION_VERSION,
    }
    mismatches = {
        key: {"config": config.get(key), "expected": value}
        for key, value in expected.items()
        if config.get(key) != value
    }
    reward = config["reward_contract"]
    exact_reward_fields = {
        "rewarded_contact_limit": REWARDED_CONTACT_LIMIT,
        "first_physical_touch_reward": PHYSICAL_CONTACT_REWARD,
        "second_physical_touch_reward": PHYSICAL_CONTACT_REWARD,
        "third_physical_touch_reward": PHYSICAL_CONTACT_REWARD,
        "fourth_and_later_touch_reward": 0.0,
        "maximum_touch_reward_per_episode": MAXIMUM_CONTACT_REWARD_PER_EPISODE,
        "requires_separated_new_contact": True,
        "heading_alignment_formula": metadata["heading_alignment_formula"],
        "heading_delta_scale": HEADING_ALIGNMENT_DELTA_SCALE,
        "heading_positive_episode_budget": None,
        "heading_negative_episode_budget": None,
        "holding_heading_reward": 0.0,
        "distance_progress_formula": metadata["distance_progress_formula"],
        "distance_progress_scale_uu": metadata["distance_progress_scale_uu"],
        "distance_progress_safety_clip_uu": metadata["distance_progress_safety_clip_uu"],
        "distance_positive_episode_budget": None,
        "distance_negative_episode_budget": None,
        "acquisition_grace_seconds": ACQUISITION_GRACE_SECONDS,
        "acquisition_time_penalty_rate_per_second": ACQUISITION_TIME_PENALTY_RATE_PER_SECOND,
        "acquisition_time_penalty_per_eligible_tick": (
            ACQUISITION_TIME_PENALTY_RATE_PER_SECOND / 120.0
        ),
        "failed_acquisition_window_penalty": FAILED_ACQUISITION_WINDOW_PENALTY,
        "time_penalty_depends_on_speed": False,
        "goal_reward": 0.0,
        "concede_reward": 0.0,
        "generic_speed_reward": 0.0,
        "boost_reward": 0.0,
        "throttle_reward": 0.0,
        "steer_reward": 0.0,
        "action_magnitude_reward": 0.0,
        "jump_reward": 0.0,
        "handbrake_reward": 0.0,
        "named_mechanic_reward": 0.0,
        "possession_reward": 0.0,
        "aerial_reward": 0.0,
        "recovery_reward": 0.0,
    }
    reward_exact = set(reward) == set(exact_reward_fields) and all(
        reward[key] == value
        or (
            isinstance(value, float)
            and math.isclose(float(reward[key]), value, rel_tol=0.0, abs_tol=1e-12)
        )
        for key, value in exact_reward_fields.items()
    )
    contract = config["stage_contract"]
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
            and all(
                contract[f"forbid_m10_{version}_actor_as_source"] is True
                for version in (2, 3, 4, 5)
            )
        ),
        "fresh_learning_state": all(
            contract[key] is True
            for key in (
                "actor_weights_only",
                "fresh_critic",
                "fresh_actor_optimizer",
                "fresh_critic_optimizer",
                "dummy_excluded_from_ppo",
            )
        ),
        "reward_contract_exact": reward_exact,
        "stage1_only_scope": (
            contract["terminal_decision"] == TERMINAL_DECISION
            and contract["stage_2_authorized"] is False
            and contract["production_promotion_authorized"] is False
        ),
        "budget_and_boundaries_exact": (
            contract["maximum_active_learner_steps"] == 2_160_000
            and contract["maximum_learner_simulated_hours"] == 5.0
            and contract["evaluation_boundaries_added_simulated_hours"]
            == list(STAGE1_BOUNDARY_HOURS)
        ),
    }
    if mismatches or not all(checks.values()):
        raise RuntimeError(f"Milestone 10.6 Stage-1 config mismatch: {mismatches}, {checks}")
    return config


def actor_only_stage_transfer(
    source_checkpoint: str | Path,
    config: dict[str, Any],
    *,
    device: str,
) -> dict[str, Any]:
    source = Path(source_checkpoint).resolve()
    if source != SOURCE_CHECKPOINT.resolve():
        raise RuntimeError("M10.6 must start from the exact v10.1 +10h checkpoint")
    result = previous.actor_only_stage_transfer(source, config, device=device)
    actor_flags = {
        "m10_2_actor_used_as_source": False,
        "m10_3_actor_used_as_source": False,
        "m10_4_actor_used_as_source": False,
        "m10_5_actor_used_as_source": False,
    }
    result["trainer_state"].update(
        {"campaign_id": CAMPAIGN_ID, "scope": "stage_1_only", **actor_flags}
    )
    result["proof"].update(actor_flags)
    return result


def build_stage1_corpus_manifests(*, output_root: str | Path = CORPUS_ROOT) -> dict[str, Any]:
    root = Path(output_root).resolve()
    if root != CORPUS_ROOT.resolve():
        raise RuntimeError("M10.6 must reuse the exact frozen M10.5 evaluation corpus")
    records: dict[str, Any] = {}
    for key, filename, expected_count in (
        ("gate", GATE_CORPUS_FILENAME, 500),
        ("unseen", UNSEEN_CORPUS_FILENAME, 250),
    ):
        path = root / filename
        manifest = _read_json(path)
        if (
            manifest.get("evaluation_version") != FROZEN_CORPUS_EVALUATION_VERSION
            or int(manifest.get("episode_count", -1)) != expected_count
        ):
            raise RuntimeError(f"Frozen M10.5 {key} corpus changed")
        records[key] = {
            "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(path),
            "episode_count": expected_count,
        }
    checks = {
        "gate_has_exactly_500_episodes": records["gate"]["episode_count"] == 500,
        "unseen_has_exactly_250_episodes": records["unseen"]["episode_count"] == 250,
        "corpus_hashes_disjoint": records["gate"]["sha256"] != records["unseen"]["sha256"],
        "m10_5_corpora_reused_without_rewrite": True,
    }
    checks["passed"] = all(checks.values())
    return {**records, "checks": checks}


def boundary_slug(hours: float) -> str:
    return previous.boundary_slug(hours)


def nominal_stage1_steps(hours: float) -> int:
    if float(hours) not in STAGE1_BOUNDARY_HOURS:
        raise ValueError(f"Unsupported M10.6 boundary: {hours}")
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
            raise RuntimeError("Refusing an unknown M10.6 campaign state")
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
        "current_skill": "turn_approach_reacquire_three_contacts",
        "current_phase": "preflight",
        "source_checkpoint": transfer_proof["source_checkpoint"],
        "source_actor_sha256": SOURCE_ACTOR_SHA256,
        "m10_2_actor_used_as_source": False,
        "m10_3_actor_used_as_source": False,
        "m10_4_actor_used_as_source": False,
        "m10_5_actor_used_as_source": False,
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
