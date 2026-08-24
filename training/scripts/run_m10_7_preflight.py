"""Execute the complete M10.7 action-policy preflight before real PPO."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import psutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import write_json_atomic  # noqa: E402
from rival_training.v10_6_environment import (  # noqa: E402
    make_ball_acquisition_phase_a_env,
)
from rival_training.v10_6_reward import reward_truth_table_v5  # noqa: E402
from rival_training.v10_7_actions import (  # noqa: E402
    deterministic_transition_reachability,
)
from rival_training.v10_7_campaign import (  # noqa: E402
    CHECKPOINT_ROOT,
    CORPUS_ROOT,
    DEFAULT_STAGE1_CONFIG,
    GATE_CORPUS_FILENAME,
    RESULT_ROOT,
    SOURCE_CHECKPOINT,
    actor_only_architecture_transfer,
    configuration_evidence,
    load_stage1_config,
)
from rival_training.v10_7_checkpoint import (  # noqa: E402
    load_checkpoint,
    save_checkpoint_atomic,
    verify_checkpoint,
    verify_reload_parity,
)
from rival_training.v10_7_diagnostics import (  # noqa: E402
    action_mapping_report,
    build_observation_corpus,
    button_policy_diagnostics,
    exact_log_probability_replay_report,
    gradient_smoke_report,
    observation_corpus_report,
    supervised_learnability_report,
)
from rival_training.v10_7_evaluation import (  # noqa: E402
    capability_gap,
    evaluate_stage1_checkpoint,
)
from rival_training.v10_7_trainer import RivalV10_7PPOTrainer  # noqa: E402
from rival_training.v9_checkpoint import sha256_file  # noqa: E402


DEFAULT_OUTPUT = RESULT_ROOT / "preflight.json"
INITIAL_CHECKPOINT = CHECKPOINT_ROOT / "initialization/000000000"
EXPECTED_WISP_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}


def _training_processes() -> list[dict[str, Any]]:
    needles = ("run_m10_7_preflight.py", "run_m10_7_stage1_boundary.py")
    current_process = psutil.Process(os.getpid())
    excluded = {int(current_process.pid)} | {
        int(process.pid) for process in current_process.parents()
    }
    matches = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if process.pid not in excluded and any(needle in command for needle in needles):
            matches.append(
                {
                    "pid": int(process.pid),
                    "name": process.info.get("name"),
                    "command": command,
                }
            )
    return matches


def _production_hashes() -> dict[str, Any]:
    actual = {
        name: sha256_file(REPOSITORY_ROOT / "bot/models" / name)
        for name in EXPECTED_WISP_HASHES
    }
    return {
        "expected": EXPECTED_WISP_HASHES,
        "actual": actual,
        "frozen_wisp_unchanged": actual == EXPECTED_WISP_HASHES,
    }


def _initial_checkpoint(transfer: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if INITIAL_CHECKPOINT.exists():
        verified = verify_checkpoint(INITIAL_CHECKPOINT, expected_config=config)
        loaded = load_checkpoint(INITIAL_CHECKPOINT, device="cpu", expected_config=config)
        if loaded["trainer_state"].get("source_actor_sha256") != transfer["trainer_state"][
            "source_actor_sha256"
        ]:
            raise RuntimeError("Existing M10.7 initialization checkpoint has wrong lineage")
        from rival_training.v10_7_checkpoint import checkpoint_record

        return checkpoint_record(INITIAL_CHECKPOINT, manifest=verified)
    state = dict(transfer["trainer_state"])
    state.update(
        {
            "simulated_game_seconds": 0.0,
            "simulated_game_hours": 0.0,
            "worker_count": int(config["backend"]["worker_count"]),
            "m10_7_initialization_checkpoint": True,
            "persistence_correction": "sticky_log_odds_prior",
        }
    )
    return save_checkpoint_atomic(
        INITIAL_CHECKPOINT,
        actor=transfer["actor"],
        critic=transfer["critic"],
        actor_optimizer=transfer["actor_optimizer"],
        critic_optimizer=transfer["critic_optimizer"],
        trainer_state=state,
        config=config,
        reload_observations=transfer["reload_observations"],
    )


def _ppo_smoke(config: dict[str, Any], *, device: str, rollout_steps: int) -> dict[str, Any]:
    transfer = actor_only_architecture_transfer(SOURCE_CHECKPOINT, config, device=device)
    smoke_config = deepcopy(config)
    trainer = RivalV10_7PPOTrainer(
        smoke_config,
        device=device,
        actor=transfer["actor"],
        critic=transfer["critic"],
        actor_optimizer=transfer["actor_optimizer"],
        critic_optimizer=transfer["critic_optimizer"],
        trainer_state=transfer["trainer_state"],
        env_factory=make_ball_acquisition_phase_a_env,
    )
    cleanup = None
    try:
        shapes = trainer.start_workers()
        health_after_start = trainer.worker_health()
        report, _ = trainer.run_iteration(
            rollout_target_agent_steps=int(rollout_steps),
            ppo_batch_agent_steps=int(rollout_steps),
            maximum_cumulative_agent_steps=int(rollout_steps)
            + 2 * int(config["backend"]["worker_count"]),
        )
    finally:
        cleanup = trainer.cleanup()
    compact = {
        "disposable": True,
        "source_recreated_from_untouched_exact_transfer": True,
        "rollout_active_learner_steps": int(report["collected_agent_steps"]),
        "environment_shapes": shapes,
        "health_after_start": health_after_start,
        "rollout_log_probability_reproduction": report[
            "rollout_log_probability_reproduction"
        ],
        "ppo": report["ppo"],
        "actions": report["actions"],
        "health": report["health"],
        "worker_cleanup": cleanup,
        "actor_discarded_after_smoke": True,
        "checks": {
            "ppo_health_passed": report["health"]["passed"],
            "all_eight_controller_branches_finite_nonzero": report["ppo"][
                "all_hybrid_head_gradient_rows_nonzero"
            ],
            "stored_rollout_log_probabilities_reproduced": report["health"][
                "rollout_log_probabilities_reproduced"
            ],
            "workers_cleaned": cleanup is not None and cleanup["passed"],
        },
    }
    compact["checks"]["passed"] = all(compact["checks"].values())
    return compact


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    processes_before = _training_processes()
    production_before = _production_hashes()
    config = load_stage1_config(args.config)
    transfer = actor_only_architecture_transfer(
        SOURCE_CHECKPOINT, config, device=args.device
    )
    initialization = _initial_checkpoint(transfer, config)
    reload_parity = verify_reload_parity(
        INITIAL_CHECKPOINT, expected_config=config, device="cpu"
    )
    reward = reward_truth_table_v5()
    action_mapping = action_mapping_report()
    observations, categories, geometry = build_observation_corpus(
        samples_per_category=int(args.observations_per_category),
        seed_base=2026107100,
    )
    corpus = observation_corpus_report(
        observations, categories, geometry, seed_base=2026107100
    )
    corpus["checks"]["passed"] = all(corpus["checks"].values())
    write_json_atomic(RESULT_ROOT / "button_observation_corpus.json", corpus)
    log_probability = exact_log_probability_replay_report(
        transfer["actor"], observations, device=args.device
    )
    buttons = button_policy_diagnostics(
        transfer["actor"],
        observations,
        device=args.device,
        stochastic_draws_per_state=int(args.button_draws_per_state),
    )
    gradient = gradient_smoke_report(
        transfer["actor"], observations, device=args.device
    )
    gradient["checks"]["passed"] = all(gradient["checks"].values())
    supervised = supervised_learnability_report(
        transfer["actor"],
        device=args.device,
        train_samples_per_category=int(args.supervised_train_per_category),
        validation_samples_per_category=int(args.supervised_validation_per_category),
        updates=int(args.supervised_updates),
    )
    smoke = _ppo_smoke(
        config, device=args.device, rollout_steps=int(args.smoke_rollout_steps)
    )
    deterministic = evaluate_stage1_checkpoint(
        INITIAL_CHECKPOINT,
        CORPUS_ROOT / GATE_CORPUS_FILENAME,
        deterministic=True,
        evaluation_workers=int(args.evaluation_workers),
    )
    stochastic = evaluate_stage1_checkpoint(
        INITIAL_CHECKPOINT,
        CORPUS_ROOT / GATE_CORPUS_FILENAME,
        deterministic=False,
        evaluation_workers=int(args.evaluation_workers),
    )
    write_json_atomic(RESULT_ROOT / "source_transfer_deterministic.json", deterministic)
    write_json_atomic(RESULT_ROOT / "source_transfer_stochastic.json", stochastic)
    gap = capability_gap(deterministic, stochastic)
    production_after = _production_hashes()
    processes_after = _training_processes()
    reachability = deterministic_transition_reachability()
    checks = {
        "no_preexisting_m10_7_training_processes": not processes_before,
        "exact_source_selective_transfer": transfer["proof"]["checks"]["passed"],
        "initial_checkpoint_reload_exact": reload_parity["checks"]["passed"],
        "m10_6_reward_truth_table_frozen_and_passed": reward["checks"]["passed"],
        "targeted_action_mapping_passed": action_mapping["checks"]["passed"],
        "frozen_eight_category_observation_corpus_passed": corpus["checks"]["passed"],
        "exact_physical_action_log_probability_replay_passed": log_probability[
            "checks"
        ]["passed"],
        "corrected_persistence_not_absorbing": (
            reachability["pathological_before_ppo"] is False
            and reachability["deterministic_reset_policy_can_ever_enable_a_button"]
        ),
        "offline_all_eight_branch_gradient_smoke_passed": gradient["checks"][
            "passed"
        ],
        "supervised_directional_learnability_passed": supervised["checks"][
            "passed"
        ],
        "disposable_real_ppo_smoke_passed": smoke["checks"]["passed"],
        "source_deterministic_500_episode_evaluation_passed": deterministic[
            "checks"
        ]["passed"],
        "source_stochastic_500_episode_evaluation_passed": stochastic["checks"][
            "passed"
        ],
        "frozen_wisp_unchanged": (
            production_before["frozen_wisp_unchanged"]
            and production_after == production_before
        ),
        "no_workers_remain": not processes_after,
    }
    checks["passed"] = all(checks.values())
    report = {
        "schema_version": 1,
        "preflight_version": "RivalM10_7ActionPolicyPreflightV1",
        "status": "passed" if checks["passed"] else "failed",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Stage-1-only action-policy correction; M10.6 reward frozen",
        "persistence_correction": {
            "original_formula_rejected": (
                "convex probability mixture with persistence above 0.5 creates an "
                "absorbing deterministic threshold latch"
            ),
            "implemented_formula": reachability["formula"],
            "requested_persistence_values_preserved": {
                "jump": 0.95,
                "boost": 0.90,
                "handbrake": 0.90,
            },
            "user_authorized_correction": True,
            "reachability": reachability,
        },
        "configuration": configuration_evidence(config),
        "source_transfer": transfer["proof"],
        "initialization_checkpoint": initialization,
        "initialization_checkpoint_reload": reload_parity,
        "reward_truth_table": reward,
        "action_mapping": action_mapping,
        "button_observation_corpus": corpus,
        "exact_log_probability_replay": log_probability,
        "source_transfer_button_policy_diagnostics": buttons,
        "offline_gradient_smoke": gradient,
        "supervised_learnability": supervised,
        "disposable_real_ppo_smoke": smoke,
        "source_transfer_capability": {
            "deterministic_report": (
                RESULT_ROOT / "source_transfer_deterministic.json"
            ).relative_to(REPOSITORY_ROOT).as_posix(),
            "stochastic_report": (
                RESULT_ROOT / "source_transfer_stochastic.json"
            ).relative_to(REPOSITORY_ROOT).as_posix(),
            "deterministic_overall": deterministic["overall"],
            "stochastic_overall": stochastic["overall"],
            "stochastic_vs_deterministic_capability_gap": gap,
        },
        "production_before": production_before,
        "production_after": production_after,
        "training_processes_before": processes_before,
        "training_processes_after": processes_after,
        "ppo_authorized": checks["passed"],
        "checks": checks,
    }
    write_json_atomic(args.output, report)
    if not checks["passed"]:
        raise RuntimeError(f"M10.7 preflight failed: {checks}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_STAGE1_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--observations-per-category", type=int, default=32)
    parser.add_argument("--button-draws-per-state", type=int, default=64)
    parser.add_argument("--supervised-train-per-category", type=int, default=128)
    parser.add_argument("--supervised-validation-per-category", type=int, default=32)
    parser.add_argument("--supervised-updates", type=int, default=400)
    parser.add_argument("--smoke-rollout-steps", type=int, default=24000)
    parser.add_argument("--evaluation-workers", type=int, default=24)
    args = parser.parse_args()
    report = run_preflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "ppo_authorized": report["ppo_authorized"],
                "initialization_checkpoint": report["initialization_checkpoint"],
                "checks": report["checks"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
