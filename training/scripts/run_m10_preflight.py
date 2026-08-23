"""Focused preflight for the M10 sustained campaign."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import (  # noqa: E402
    DEFAULT_M10_CONFIG_PATH,
    M09_FINAL_CHECKPOINT,
    compact_training_iteration,
    load_m10_config,
    m10_config_migration_report,
    verify_exact_m09_start_checkpoint,
    write_json_atomic,
)
from rival_training.v9_checkpoint import (  # noqa: E402
    config_sha256,
    load_v9_checkpoint,
    sha256_file,
)
from rival_training.v9_trainer import RivalV9PPOTrainer  # noqa: E402


DEFAULT_OUTPUT = REPOSITORY_ROOT / "training/results/milestone10/preflight.json"
EXPECTED_WISP_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("RIVAL_"):
            environment.pop(name)
    environment.pop("PYTHONPATH", None)
    return environment


def _production_probe() -> dict[str, Any]:
    production_python = REPOSITORY_ROOT / ".venv/Scripts/python.exe"
    script = (
        "import json,config; print(json.dumps({"
        "'mode':config.POLICY_RUNTIME_MODE,"
        "'tick_skip':config.TICK_SKIP,"
        "'v9':config.V9_SCRATCH_POLICY_ENABLED,"
        "'candidate':config.CANDIDATE_POLICY_ENABLED,"
        "'m08':config.M08_DUAL_RATE_ENABLED,"
        "'policy':config.MODEL_INFO_POLICY.path.name,"
        "'shared_head':config.MODEL_INFO_SHARED_HEAD.path.name}))"
    )
    completed = subprocess.run(
        [str(production_python), "-c", script],
        cwd=REPOSITORY_ROOT / "bot",
        env=_clean_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    config = load_m10_config(args.config)
    exact_start = verify_exact_m09_start_checkpoint(args.checkpoint, device="cpu")
    loaded = load_v9_checkpoint(args.checkpoint, device=args.device)
    migration = m10_config_migration_report(loaded["config"], config)
    actor_hash_before = sha256_file(Path(args.checkpoint) / "actor.pt")
    trainer = RivalV9PPOTrainer(
        config,
        device=args.device,
        actor=loaded["actor"],
        critic=loaded["critic"],
        actor_optimizer=loaded["actor_optimizer"],
        critic_optimizer=loaded["critic_optimizer"],
        trainer_state=loaded["trainer_state"],
    )
    cleanup: dict[str, Any] | None = None
    try:
        shapes = trainer.start_workers()
        health_after_start = trainer.worker_health()
        smoke, _ = trainer.run_iteration(
            rollout_target_agent_steps=int(args.rollout_agent_steps),
            ppo_batch_agent_steps=int(args.rollout_agent_steps),
        )
        health_after_rollout = trainer.worker_health()
    finally:
        cleanup = trainer.cleanup()
    actor_hash_after = sha256_file(Path(args.checkpoint) / "actor.pt")
    wisp_hashes = {
        name: sha256_file(REPOSITORY_ROOT / "bot/models" / name)
        for name in EXPECTED_WISP_HASHES
    }
    production = _production_probe()
    expected_production = {
        "mode": "frozen_wisp_production",
        "tick_skip": 8,
        "v9": False,
        "candidate": False,
        "m08": False,
        "policy": "POLICY.lt",
        "shared_head": "SHARED_HEAD.lt",
    }
    checks = {
        "exact_m09_checkpoint_verified_before_smoke": exact_start["checks"]["passed"],
        "m10_config_campaign_only_migration": migration["passed"],
        "worker_count_exactly_56": len(health_after_start) == 56,
        "all_56_workers_alive_after_start": len(health_after_start) == 56
        and all(item["alive"] for item in health_after_start),
        "short_rollout_and_update_finite": smoke["health"]["passed"],
        "all_56_workers_alive_after_rollout": len(health_after_rollout) == 56
        and all(item["alive"] for item in health_after_rollout),
        "smoke_did_not_mutate_source_checkpoint": actor_hash_before == actor_hash_after,
        "worker_cleanup_passed": cleanup is not None and cleanup["passed"],
        "frozen_wisp_hashes_unchanged": wisp_hashes == EXPECTED_WISP_HASHES,
        "production_default_remains_frozen_wisp": production == expected_production,
        "production_promotion_authorized": False,
    }
    result = {
        "schema_version": 1,
        "status": "passed",
        "preflight_version": "RivalM10FocusedPreflightV1",
        "config": {
            "path": "training/configs/milestone10.json",
            "file_sha256": sha256_file(args.config),
            "canonical_sha256": config_sha256(config),
            "version": config["config_version"],
            "migration": migration,
        },
        "exact_start_checkpoint": exact_start,
        "smoke": {
            "purpose": "discarded_in_memory_finite_56_worker_rollout_preflight",
            "rollout_agent_steps_requested": int(args.rollout_agent_steps),
            "environment_shapes": shapes,
            "health_after_start": health_after_start,
            "iteration": compact_training_iteration(smoke),
            "health_after_rollout": health_after_rollout,
            "source_actor_sha256_before": actor_hash_before,
            "source_actor_sha256_after": actor_hash_after,
            "checkpoint_written": False,
            "experience_counted_toward_m10_campaign": False,
            "cleanup": cleanup,
        },
        "production": {
            "probe": production,
            "expected": expected_production,
            "wisp_hashes": wisp_hashes,
            "expected_wisp_hashes": EXPECTED_WISP_HASHES,
        },
        "checks": checks,
    }
    result["checks"]["passed"] = all(
        value for key, value in checks.items() if key != "production_promotion_authorized"
    ) and checks["production_promotion_authorized"] is False
    if not result["checks"]["passed"]:
        result["status"] = "failed"
        write_json_atomic(args.output, result)
        raise RuntimeError(f"M10 focused preflight failed: {result['checks']}")
    write_json_atomic(args.output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=M09_FINAL_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_M10_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-agent-steps", type=int, default=48000)
    args = parser.parse_args()
    if args.rollout_agent_steps <= 0 or args.rollout_agent_steps % 48000:
        raise ValueError("Preflight rollout must preserve complete 48k minibatches")
    report = run_preflight(args)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
