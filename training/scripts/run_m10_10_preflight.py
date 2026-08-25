"""Run the M10.10 minimal-reward hard preflight and initial evaluations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import psutil
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import write_json_atomic  # noqa: E402
from rival_training.v10_7_checkpoint import (  # noqa: E402
    checkpoint_record,
    save_checkpoint_atomic,
    verify_checkpoint,
    verify_reload_parity,
)
from rival_training.v10_7_diagnostics import action_mapping_report  # noqa: E402
from rival_training.v10_8_credit import gae_physical_time_report  # noqa: E402
from rival_training.v10_10_campaign import (  # noqa: E402
    CORPUS_ROOT,
    GATE_CORPUS_FILENAME,
    INITIAL_CHECKPOINT,
    RESULT_ROOT,
    clean_initialization,
    configuration_evidence,
    load_stage1_config,
)
from rival_training.v10_10_evaluation import (  # noqa: E402
    capability_gap,
    evaluate_stage1_checkpoint,
)
from rival_training.v10_10_reward import (  # noqa: E402
    reward_truth_table_v10_10,
)
from rival_training.v10_10_trainer import RivalV10_10PPOTrainer  # noqa: E402
from rival_training.v9_checkpoint import sha256_file  # noqa: E402


EXPECTED_WISP_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}
M10_9_PREFLIGHT = REPOSITORY_ROOT / "training/results/milestone10_9/preflight.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_processes() -> list[dict[str, Any]]:
    needles = ("run_m10_10_preflight.py", "run_m10_10_stage1_boundary.py")
    current = psutil.Process(os.getpid())
    excluded = {current.pid, *(parent.pid for parent in current.parents())}
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
    initialized = clean_initialization(config, device="cpu")
    state = dict(initialized["trainer_state"])
    state.update(
        {
            "simulated_game_seconds": 0.0,
            "simulated_game_hours": 0.0,
            "worker_count": int(config["backend"]["worker_count"]),
            "m10_10_initialization_checkpoint": True,
            "production_promotion_authorized": False,
        }
    )
    return save_checkpoint_atomic(
        INITIAL_CHECKPOINT,
        actor=initialized["actor"],
        critic=initialized["critic"],
        actor_optimizer=initialized["actor_optimizer"],
        critic_optimizer=initialized["critic_optimizer"],
        trainer_state=state,
        config=config,
        reload_observations=initialized["reload_observations"],
    )


def _disposable_real_rollout(
    config: dict[str, Any], *, device: str
) -> dict[str, Any]:
    initialized = clean_initialization(config, device=device)
    trainer = RivalV10_10PPOTrainer(
        config,
        device=device,
        actor=initialized["actor"],
        critic=initialized["critic"],
        actor_optimizer=initialized["actor_optimizer"],
        critic_optimizer=initialized["critic_optimizer"],
        trainer_state=initialized["trainer_state"],
    )
    cleanup = None
    shapes = None
    try:
        shapes = trainer.start_workers()
        rollout = trainer.collect_prepared_rollout(24_000)
        actor = trainer.optimize_actor(rollout)
        critic = trainer.optimize_critic(rollout, disposable=True)
        worker_health = trainer.worker_health()
    finally:
        cleanup = trainer.cleanup()
    if cleanup is None:
        raise RuntimeError("M10.10 disposable rollout cleanup was not attempted")
    checks = {
        "real_reward_rows_collected": rollout.collected >= 24_000,
        "all_rewards_finite": bool(torch.isfinite(torch.as_tensor(rollout.rewards)).all()),
        "same_policy_log_probability_replay": rollout.replay[
            "same_policy_replay"
        ]["passed"],
        "independent_log_probability_replay": rollout.replay[
            "independent_formula"
        ]["passed"],
        "advantage_signs_preserved": bool(
            torch.equal(
                torch.signbit(torch.as_tensor(rollout.advantages[rollout.advantages != 0])),
                torch.signbit(
                    torch.as_tensor(
                        rollout.scaled_advantages[rollout.advantages != 0]
                    )
                ),
            )
        ),
        "actor_updated_all_branches": actor[
            "all_controller_branches_finite_nonzero"
        ],
        "critic_eight_epochs": critic["epochs_executed"] == 8,
        "critic_held_out_finite": all(
            torch.isfinite(torch.tensor(value))
            for value in (
                critic["before"]["held_out"]["loss"],
                critic["after"]["held_out"]["loss"],
                critic["after"]["held_out"]["explained_variance"],
            )
        ),
        "workers_healthy": len(worker_health)
        == int(config["backend"]["worker_count"])
        and all(row["alive"] for row in worker_health),
        "workers_cleaned": cleanup["passed"],
    }
    checks["passed"] = all(checks.values())
    return {
        "version": "RivalM10_10DisposableRealRolloutV1",
        "disposable": True,
        "environment_shapes": shapes,
        "collected_agent_steps": rollout.collected,
        "reward": {
            "minimum": float(rollout.rewards.min()),
            "maximum": float(rollout.rewards.max()),
            "mean": float(rollout.rewards.mean()),
            "nonzero_share": float((rollout.rewards != 0.0).mean()),
        },
        "log_probability_replay": rollout.replay,
        "advantage_scale": rollout.advantage_scale,
        "actor": actor,
        "critic": critic,
        "ar_exploration": rollout.exploration,
        "worker_health_before_cleanup": worker_health,
        "worker_cleanup": cleanup,
        "disposable_actor_and_critic_discarded": True,
        "checks": checks,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    processes_before = _training_processes()
    if processes_before:
        raise RuntimeError(f"M10.10 process collision: {processes_before}")
    production_before = _production_hashes()
    config = load_stage1_config(args.config)
    initialized = clean_initialization(config, device="cpu")
    initialization = _materialize_initialization(config)
    reload = verify_reload_parity(
        INITIAL_CHECKPOINT, expected_config=config, device="cpu"
    )
    action_mapping = action_mapping_report()
    reward = reward_truth_table_v10_10()
    m10_9_preflight = _read(M10_9_PREFLIGHT)
    gae = gae_physical_time_report(
        gamma=float(config["ppo"]["gamma"]),
        gae_lambda=float(config["ppo"]["gae_lambda"]),
        arm="M10.10",
    )
    disposable = _disposable_real_rollout(config, device=args.device)
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
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    deterministic_path = RESULT_ROOT / "initialization_deterministic.json"
    stochastic_path = RESULT_ROOT / "initialization_stochastic_ar1.json"
    write_json_atomic(deterministic_path, deterministic)
    write_json_atomic(stochastic_path, stochastic)
    production_after = _production_hashes()
    processes_after = _training_processes()
    checks = {
        "no_preexisting_m10_10_processes": not processes_before,
        "configuration_frozen": load_stage1_config(args.config) == config,
        "clean_source_lineage": initialized["proof"]["checks"]["passed"],
        "initial_checkpoint_reload_exact": reload["checks"]["passed"],
        "all_eight_soccar_channels_map": action_mapping["checks"]["passed"],
        "minimal_reward_truth_table": reward["checks"]["passed"],
        "arm_c_gae_horizon_exact": gae["checks"]["passed"]
        and abs(float(gae["half_life_seconds"]) - 2.0) < 1e-12,
        "m10_9_ppo_v2_infrastructure_proven": m10_9_preflight["checks"][
            "passed"
        ],
        "new_reward_real_rollout_passed": disposable["checks"]["passed"],
        "initial_deterministic_evaluation": deterministic["checks"]["passed"],
        "initial_stochastic_evaluation": stochastic["checks"]["passed"],
        "frozen_wisp_unchanged": production_before["frozen_wisp_unchanged"]
        and production_after["frozen_wisp_unchanged"]
        and production_before["actual"] == production_after["actual"],
        "no_workers_remain": not processes_after,
    }
    checks["passed"] = all(checks.values())
    report = {
        "schema_version": 1,
        "preflight_version": "RivalM10_10MinimalFirstTouchPreflightV1",
        "status": "passed" if checks["passed"] else "failed",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration_evidence(config),
        "clean_initialization": initialized["proof"],
        "initialization_checkpoint": initialization,
        "initialization_checkpoint_reload": reload,
        "action_mapping": action_mapping,
        "reward_truth_table": reward,
        "gae_physical_time_proof": gae,
        "m10_9_proven_infrastructure": {
            "path": M10_9_PREFLIGHT.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(M10_9_PREFLIGHT),
            "checks": m10_9_preflight["checks"],
        },
        "disposable_real_rollout": disposable,
        "initialization_capability": {
            "deterministic_report": deterministic_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "stochastic_report": stochastic_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
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
        "ppo_authorized": checks["passed"],
        "stage_2_authorized": False,
        "production_promotion_authorized": False,
        "checks": checks,
    }
    write_json_atomic(args.output, report)
    if not checks["passed"]:
        raise RuntimeError(f"M10.10 preflight failed: {checks}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "training/configs/milestone10_10_stage1.json",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "preflight.json"
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-workers", type=int, default=24)
    args = parser.parse_args()
    report = run_preflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "ppo_authorized": report["ppo_authorized"],
                "initial_deterministic_first_touch": report[
                    "initialization_capability"
                ]["deterministic_overall"]["first_touch_success_share"],
                "initial_stochastic_first_touch": report[
                    "initialization_capability"
                ]["stochastic_overall"]["first_touch_success_share"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
