"""Execute the M10.9 PPO V2 hard preflight and disposable critic fit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import psutil
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import write_json_atomic  # noqa: E402
from rival_training.v10_6_reward import reward_truth_table_v5  # noqa: E402
from rival_training.v10_7_checkpoint import (  # noqa: E402
    checkpoint_record,
    load_checkpoint,
    save_checkpoint_atomic,
    verify_checkpoint,
    verify_reload_parity,
)
from rival_training.v10_7_diagnostics import (  # noqa: E402
    action_mapping_report,
    build_observation_corpus,
)
from rival_training.v10_8_credit import gae_physical_time_report  # noqa: E402
from rival_training.v10_9_actions import (  # noqa: E402
    ANALOG_DIM,
    AR_INNOVATION_STD,
    AR_RHO,
    RivalARStickyBernoulliPolicy,
    independent_ar_log_probability,
    pack_rollout_actions,
)
from rival_training.v10_9_campaign import (  # noqa: E402
    CORPUS_ROOT,
    GATE_CORPUS_FILENAME,
    INITIAL_CHECKPOINT,
    PAIRED_ACTOR_STATE_SHA256,
    PAIRED_CRITIC_STATE_SHA256,
    RESULT_ROOT,
    configuration_evidence,
    load_stage1_config,
    paired_initialization,
    state_dict_sha256,
)
from rival_training.v10_9_evaluation import (  # noqa: E402
    capability_gap,
    evaluate_stage1_checkpoint,
)
from rival_training.v10_9_trainer import (  # noqa: E402
    RivalV10_9PPOTrainer,
    scale_advantages,
)
from rival_training.v9_checkpoint import sha256_file  # noqa: E402


DEFAULT_OUTPUT = RESULT_ROOT / "preflight.json"
EXPECTED_WISP_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}


def _training_processes() -> list[dict[str, Any]]:
    needles = ("run_m10_9_preflight.py", "run_m10_9_stage1_boundary.py")
    current = psutil.Process(os.getpid())
    excluded = {current.pid, *(row.pid for row in current.parents())}
    rows = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if process.pid not in excluded and any(needle in command for needle in needles):
            rows.append(
                {
                    "pid": int(process.pid),
                    "name": process.info.get("name"),
                    "command": command,
                }
            )
    return rows


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


def _materialize_initialization(config: dict[str, Any]) -> dict[str, Any]:
    if INITIAL_CHECKPOINT.exists():
        manifest = verify_checkpoint(INITIAL_CHECKPOINT, expected_config=config)
        return checkpoint_record(INITIAL_CHECKPOINT, manifest=manifest)
    paired = paired_initialization(config, device="cpu")
    state = dict(paired["trainer_state"])
    state.update(
        {
            "simulated_game_seconds": 0.0,
            "simulated_game_hours": 0.0,
            "worker_count": int(config["backend"]["worker_count"]),
            "m10_9_initialization_checkpoint": True,
            "production_promotion_authorized": False,
        }
    )
    return save_checkpoint_atomic(
        INITIAL_CHECKPOINT,
        actor=paired["actor"],
        critic=paired["critic"],
        actor_optimizer=paired["actor_optimizer"],
        critic_optimizer=paired["critic_optimizer"],
        trainer_state=state,
        config=config,
        reload_observations=paired["reload_observations"],
    )


def _advantage_evidence(config: dict[str, Any]) -> dict[str, Any]:
    raw = np.asarray(
        [-7.1, -4.8, -2.4, -1.899, -1.212, -0.2, 0.0, 0.3, 3.4, 9.2],
        dtype=np.float32,
    )
    scaled, scale = scale_advantages(
        raw, epsilon=float(config["ppo"]["advantage_epsilon"])
    )
    centered = (raw - raw.mean()) / max(
        float(raw.std()), float(config["ppo"]["advantage_epsilon"])
    )
    return {
        "version": "RivalM10_9AdvantageSignPreflightV1",
        "m10_8_like_raw_values": raw.tolist(),
        "rollout_wide_scale": scale,
        "scale_only_values": scaled.tolist(),
        "historical_mean_centered_values_for_demonstration_only": centered.tolist(),
        "negative_1p212_scale_only": float(
            scaled[np.flatnonzero(np.isclose(raw, -1.212))[0]]
        ),
        "negative_1p899_scale_only": float(
            scaled[np.flatnonzero(np.isclose(raw, -1.899))[0]]
        ),
        "mean_centered_negative_flip_count": int(
            np.sum((raw < 0.0) & (centered > 0.0))
        ),
        "checks": {
            "negative_remains_negative": bool(np.all(scaled[raw < 0] < 0)),
            "positive_remains_positive": bool(np.all(scaled[raw > 0] > 0)),
            "zero_remains_zero": bool(np.all(scaled[raw == 0] == 0)),
            "finite": bool(np.isfinite(scaled).all()),
            "one_scale_reused_across_minibatches": bool(
                np.array_equal(
                    np.concatenate([raw[:3] / scale, raw[3:7] / scale, raw[7:] / scale]),
                    scaled,
                )
            ),
            "preserved_failure_mode_reproduced": bool(
                np.any((raw < 0.0) & (centered > 0.0))
            ),
        },
    }


def _ar_synthetic_evidence() -> dict[str, Any]:
    rng = np.random.default_rng(2026107901)
    episodes = 512
    ticks = 512
    epsilon = np.empty((episodes, ticks, ANALOG_DIM), dtype=np.float64)
    epsilon[:, 0] = rng.standard_normal((episodes, ANALOG_DIM))
    for tick in range(1, ticks):
        epsilon[:, tick] = (
            AR_RHO * epsilon[:, tick - 1]
            + AR_INNOVATION_STD * rng.standard_normal((episodes, ANALOG_DIM))
        )
    autocorrelation = {}
    deviations = []
    for lag in (1, 3, 6, 12, 24):
        empirical = [
            float(
                np.corrcoef(
                    epsilon[:, :-lag, axis].reshape(-1),
                    epsilon[:, lag:, axis].reshape(-1),
                )[0, 1]
            )
            for axis in range(ANALOG_DIM)
        ]
        expected = AR_RHO**lag
        difference = [abs(value - expected) for value in empirical]
        deviations.extend(difference)
        autocorrelation[str(lag)] = {
            "physical_seconds": lag / 120.0,
            "expected": expected,
            "empirical": empirical,
            "absolute_deviation": difference,
        }
    return {
        "version": "RivalM10_9ARSyntheticPreflightV1",
        "rho": AR_RHO,
        "episodes": episodes,
        "ticks_per_episode": ticks,
        "epsilon_mean": epsilon.reshape(-1, ANALOG_DIM).mean(axis=0).tolist(),
        "epsilon_std": epsilon.reshape(-1, ANALOG_DIM).std(axis=0).tolist(),
        "autocorrelation": autocorrelation,
        "maximum_analytical_vs_measured_deviation": max(deviations),
        "resets_are_independent_stationary_draws": True,
        "checks": {
            "maximum_autocorrelation_error_below_0p015": max(deviations) < 0.015,
            "stationary_mean_near_zero": bool(
                np.max(np.abs(epsilon.reshape(-1, ANALOG_DIM).mean(axis=0))) < 0.02
            ),
            "stationary_std_near_one": bool(
                np.max(
                    np.abs(epsilon.reshape(-1, ANALOG_DIM).std(axis=0) - 1.0)
                )
                < 0.03
            ),
        },
    }


def _offline_log_probability_evidence(actor, *, device: str) -> dict[str, Any]:
    observations, _, _ = build_observation_corpus(
        samples_per_category=4, seed_base=2026107902
    )
    policy = RivalARStickyBernoulliPolicy(actor, device)
    generator = torch.Generator(device=device).manual_seed(2026107903)
    previous = torch.randn(
        len(observations), ANALOG_DIM, generator=generator, device=device
    )
    initial = torch.zeros(len(observations), 1, device=device)
    initial[::7] = 1.0
    obs = torch.as_tensor(observations, dtype=torch.float32, device=device)
    distribution = policy.distribution(obs, previous, initial)
    torch.manual_seed(2026107904)
    sample = distribution.sample()
    packed = pack_rollout_actions(sample.physical_action, previous, initial)
    replay_distribution, physical = policy.distribution_for_replay(obs, packed)
    replay = replay_distribution.log_prob(physical)
    independent = independent_ar_log_probability(
        analog_mean=distribution.analog_mean,
        analog_log_std=distribution.analog_log_std,
        button_probabilities=distribution.effective_probabilities,
        physical_actions=sample.physical_action,
        previous_epsilon=previous,
        initial=initial,
    )
    same_error = float((sample.log_probability - replay).abs().max().detach().cpu())
    independent_error = float((replay - independent).abs().max().detach().cpu())
    return {
        "version": "RivalM10_9OfflineARLogProbabilityReplayV1",
        "samples": len(observations),
        "same_policy_maximum_abs_error": same_error,
        "independent_formula_maximum_abs_error": independent_error,
        "checks": {
            "same_policy_effectively_zero": same_error <= 2e-5,
            "independent_formula_agrees": independent_error <= 2e-5,
            "stored_record_contains_previous_epsilon": packed.shape[1] == 14,
            "executed_physical_action_remains_eight_fields": physical.shape[1] == 8,
        },
    }


def _disposable_real_rollout(
    config: dict[str, Any], *, device: str, rollout_steps: int
) -> dict[str, Any]:
    loaded = load_checkpoint(
        INITIAL_CHECKPOINT, device=device, expected_config=config
    )
    actor_before = state_dict_sha256(loaded["actor"].state_dict())
    critic_before = state_dict_sha256(loaded["critic"].state_dict())
    trainer = RivalV10_9PPOTrainer(
        config,
        device=device,
        actor=loaded["actor"],
        critic=loaded["critic"],
        actor_optimizer=loaded["actor_optimizer"],
        critic_optimizer=loaded["critic_optimizer"],
        trainer_state=loaded["trainer_state"],
    )
    cleanup = None
    try:
        shapes = trainer.start_workers()
        rollout = trainer.collect_prepared_rollout(int(rollout_steps))
        critic = trainer.optimize_critic(rollout, disposable=True)
        actor = trainer.optimize_actor(rollout)
        health = trainer.worker_health()
    finally:
        cleanup = trainer.cleanup()
    return {
        "version": "RivalM10_9DisposableRealRolloutPreflightV1",
        "disposable": True,
        "environment_shapes": shapes,
        "rollout_active_learner_steps": rollout.collected,
        "experience_records": len(rollout.observations),
        "collection_seconds": rollout.collection_seconds,
        "source_actor_state_sha256": actor_before,
        "source_critic_state_sha256": critic_before,
        "log_probability_replay": rollout.replay,
        "advantage": {
            "raw": {
                "mean": float(rollout.advantages.mean()),
                "standard_deviation": float(rollout.advantages.std()),
                "minimum": float(rollout.advantages.min()),
                "maximum": float(rollout.advantages.max()),
            },
            "scaled": {
                "mean": float(rollout.scaled_advantages.mean()),
                "standard_deviation": float(rollout.scaled_advantages.std()),
                "minimum": float(rollout.scaled_advantages.min()),
                "maximum": float(rollout.scaled_advantages.max()),
            },
            "scale": rollout.advantage_scale,
            "sign_agreement": bool(
                np.array_equal(
                    np.signbit(rollout.advantages[rollout.advantages != 0]),
                    np.signbit(
                        rollout.scaled_advantages[rollout.advantages != 0]
                    ),
                )
            ),
        },
        "ar_exploration": rollout.exploration,
        "credit_assignment": rollout.credit,
        "critic_learnability": critic,
        "actor_smoke": actor,
        "worker_health_before_cleanup": health,
        "worker_cleanup": cleanup,
        "disposable_actor_and_critic_discarded": True,
        "checks": {
            "source_actor_exact": actor_before == PAIRED_ACTOR_STATE_SHA256,
            "source_critic_exact": critic_before == PAIRED_CRITIC_STATE_SHA256,
            "same_policy_rollout_log_probability_replay": rollout.replay[
                "same_policy_replay"
            ]["passed"],
            "independent_rollout_log_probability_replay": rollout.replay[
                "independent_formula"
            ]["passed"],
            "advantage_signs_preserved": bool(
                np.array_equal(
                    np.signbit(rollout.advantages[rollout.advantages != 0]),
                    np.signbit(
                        rollout.scaled_advantages[rollout.advantages != 0]
                    ),
                )
            ),
            "ar_autocorrelation_matches": rollout.exploration[
                "maximum_analytical_vs_measured_deviation"
            ]
            < 0.03,
            "ar_reset_records_present": rollout.exploration[
                "initial_transition_records"
            ]
            > 0,
            "critic_held_out_ev_improved": critic["held_out_ev_improvement"] > 0.0,
            "critic_held_out_loss_improved": critic["held_out_loss_improvement"] > 0.0,
            "critic_eight_epochs": critic["epochs_executed"] == 8,
            "critic_substantially_more_than_four_steps": critic["optimizer_steps"]
            > 4,
            "actor_one_or_two_epochs_per_kl_rule": 1
            <= actor["epochs_executed"]
            <= 2,
            "actor_all_eight_branches_receive_gradient": actor[
                "all_controller_branches_finite_nonzero"
            ],
            "workers_cleaned": cleanup is not None and cleanup["passed"],
        },
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    processes_before = _training_processes()
    production_before = _production_hashes()
    config = load_stage1_config(args.config)
    initialization = _materialize_initialization(config)
    initialization_reload = verify_reload_parity(
        INITIAL_CHECKPOINT, expected_config=config, device="cpu"
    )
    paired = paired_initialization(config, device="cpu")["proof"]
    action_mapping = action_mapping_report()
    reward = reward_truth_table_v5()
    gae = gae_physical_time_report(
        gamma=float(config["ppo"]["gamma"]),
        gae_lambda=float(config["ppo"]["gae_lambda"]),
        arm="M10.9-Arm-C",
    )
    advantages = _advantage_evidence(config)
    ar_synthetic = _ar_synthetic_evidence()
    loaded_cpu = load_checkpoint(
        INITIAL_CHECKPOINT, device="cpu", expected_config=config
    )
    offline_replay = _offline_log_probability_evidence(
        loaded_cpu["actor"], device="cpu"
    )
    real_rollout = _disposable_real_rollout(
        config, device=args.device, rollout_steps=int(args.rollout_steps)
    )
    corpus = CORPUS_ROOT / GATE_CORPUS_FILENAME
    deterministic = evaluate_stage1_checkpoint(
        INITIAL_CHECKPOINT,
        corpus,
        deterministic=True,
        evaluation_workers=int(args.evaluation_workers),
    )
    stochastic = evaluate_stage1_checkpoint(
        INITIAL_CHECKPOINT,
        corpus,
        deterministic=False,
        evaluation_workers=int(args.evaluation_workers),
    )
    det_path = RESULT_ROOT / "initialization_deterministic.json"
    sto_path = RESULT_ROOT / "initialization_stochastic_ar1.json"
    write_json_atomic(det_path, deterministic)
    write_json_atomic(sto_path, stochastic)
    production_after = _production_hashes()
    processes_after = _training_processes()
    nominal_actor_minibatches = math.ceil(
        96_000 / int(config["ppo"]["actor_minibatch_agent_steps"])
    )
    nominal_critic_train = 96_000 - int(
        config["ppo"]["critic_validation_agent_steps"]
    )
    nominal_critic_minibatches = math.ceil(
        nominal_critic_train / int(config["ppo"]["critic_minibatch_agent_steps"])
    )
    optimizer_schedule = {
        "actor": {
            "epochs": 2,
            "minibatches_per_epoch": nominal_actor_minibatches,
            "nominal_optimizer_steps": 2 * nominal_actor_minibatches,
            "subject_to_kl_early_stop": True,
        },
        "critic": {
            "epochs": 8,
            "training_samples_after_heldout": nominal_critic_train,
            "minibatches_per_epoch": nominal_critic_minibatches,
            "nominal_optimizer_steps": 8 * nominal_critic_minibatches,
            "final_partial_minibatch_retained": True,
        },
        "loops_decoupled": True,
    }
    checks = {
        "no_preexisting_m10_9_processes": not processes_before,
        "paired_initialization_exact": paired["checks"]["passed"],
        "initial_checkpoint_reload_exact": initialization_reload["checks"]["passed"],
        "all_eight_soccar_channels_map": action_mapping["checks"]["passed"],
        "m10_6_reward_truth_table_frozen": reward["checks"]["passed"],
        "arm_c_gae_horizon_exact": gae["checks"]["passed"],
        "advantage_scaling_truth_table": all(advantages["checks"].values()),
        "ar_synthetic_autocorrelation": all(ar_synthetic["checks"].values()),
        "offline_log_probability_replay": all(offline_replay["checks"].values()),
        "real_rollout_and_critic_diagnostic": all(real_rollout["checks"].values()),
        "actor_critic_schedules_decoupled": optimizer_schedule["loops_decoupled"],
        "critic_nominal_steps_exceed_old_four": optimizer_schedule["critic"][
            "nominal_optimizer_steps"
        ]
        > 4,
        "initial_deterministic_evaluation": deterministic["checks"]["passed"],
        "initial_stochastic_evaluation": stochastic["checks"]["passed"],
        "frozen_wisp_unchanged": production_before == production_after
        and production_after["frozen_wisp_unchanged"],
        "no_workers_remain": not processes_after,
    }
    checks["passed"] = all(checks.values())
    report = {
        "schema_version": 1,
        "preflight_version": "RivalM10_9PPOV2PreflightV1",
        "status": "passed" if checks["passed"] else "failed",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration_evidence(config),
        "paired_initialization": paired,
        "initialization_checkpoint": initialization,
        "initialization_checkpoint_reload": initialization_reload,
        "action_mapping": action_mapping,
        "reward_truth_table": reward,
        "gae_physical_time_proof": gae,
        "advantage_sign_evidence": advantages,
        "ar_synthetic_evidence": ar_synthetic,
        "offline_log_probability_replay": offline_replay,
        "optimizer_schedule": optimizer_schedule,
        "disposable_real_rollout_and_critic": real_rollout,
        "initialization_capability": {
            "deterministic_report": det_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "stochastic_report": sto_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "deterministic_overall": deterministic["overall"],
            "stochastic_overall": stochastic["overall"],
            "stochastic_vs_deterministic_gap": capability_gap(
                deterministic, stochastic
            ),
        },
        "production_before": production_before,
        "production_after": production_after,
        "processes_before": processes_before,
        "processes_after": processes_after,
        "checks": checks,
        "ppo_authorized": checks["passed"],
        "stage_2_authorized": False,
        "production_promotion_authorized": False,
    }
    write_json_atomic(args.output, report)
    if not checks["passed"]:
        raise RuntimeError(f"M10.9 preflight failed: {checks}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "training/configs/milestone10_9_stage1.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-steps", type=int, default=96000)
    parser.add_argument("--evaluation-workers", type=int, default=24)
    args = parser.parse_args()
    report = run_preflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "ppo_authorized": report["ppo_authorized"],
                "critic_held_out_ev_improvement": report[
                    "disposable_real_rollout_and_critic"
                ]["critic_learnability"]["held_out_ev_improvement"],
                "checks": report["checks"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
