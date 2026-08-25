"""Validate and materialize the paired M10.8 GAE-arm initialization."""

from __future__ import annotations

import argparse
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
from rival_training.v10_6_reward import reward_truth_table_v5  # noqa: E402
from rival_training.v10_7_checkpoint import (  # noqa: E402
    checkpoint_record,
    load_checkpoint,
    save_checkpoint_atomic,
    verify_checkpoint,
    verify_reload_parity,
)
from rival_training.v10_7_evaluation import (  # noqa: E402
    capability_gap,
    evaluate_stage1_checkpoint,
)
from rival_training.v10_8_campaign import (  # noqa: E402
    ARM_LAMBDAS,
    CHECKPOINT_ROOT,
    CORPUS_ROOT,
    GATE_CORPUS_FILENAME,
    RESULT_ROOT,
    SOURCE_CHECKPOINT,
    configuration_evidence,
    load_arm_config,
    paired_contract_report,
    paired_initial_state,
    state_dict_sha256,
)
from rival_training.v10_8_credit import gae_physical_time_report  # noqa: E402
from rival_training.v9_checkpoint import sha256_file  # noqa: E402


DEFAULT_OUTPUT = RESULT_ROOT / "preflight.json"
EXPECTED_WISP_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}


def _training_processes() -> list[dict[str, Any]]:
    needles = ("run_m10_8_preflight.py", "run_m10_8_stage1_boundary.py")
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


def _clean_trainer_state(source: dict[str, Any], arm: str) -> dict[str, Any]:
    state = {
        key: value
        for key, value in source["trainer_state"].items()
        if key not in {"format", "contract"}
    }
    state.update(
        {
            "campaign_id": "rival-v10-8-stage1-gae-credit-assignment",
            "m10_8_arm": arm,
            "m10_8_paired_initialization": True,
            "m10_8_source_checkpoint": SOURCE_CHECKPOINT.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "clean_boundary": True,
            "production_promotion_authorized": False,
        }
    )
    return state


def _materialize_arm_initialization(
    arm: str, source: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    destination = CHECKPOINT_ROOT / f"arms/arm_{arm.lower()}/initialization/000000000"
    if destination.exists():
        manifest = verify_checkpoint(destination, expected_config=config)
        record = checkpoint_record(destination, manifest=manifest)
    else:
        record = save_checkpoint_atomic(
            destination,
            actor=source["actor"],
            critic=source["critic"],
            actor_optimizer=source["actor_optimizer"],
            critic_optimizer=source["critic_optimizer"],
            trainer_state=_clean_trainer_state(source, arm),
            config=config,
            reload_observations=source["reload_observations"],
        )
    loaded = load_checkpoint(destination, device="cpu", expected_config=config)
    parity = verify_reload_parity(destination, expected_config=config, device="cpu")
    return {
        "arm": arm,
        "lambda": config["ppo"]["gae_lambda"],
        "checkpoint": record,
        "actor_state_sha256": state_dict_sha256(loaded["actor"].state_dict()),
        "critic_state_sha256": state_dict_sha256(loaded["critic"].state_dict()),
        "actor_optimizer_state_entries": len(loaded["actor_optimizer"].state),
        "critic_optimizer_state_entries": len(loaded["critic_optimizer"].state),
        "reload_parity": parity,
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = datetime.now(timezone.utc).isoformat()
    processes_before = _training_processes()
    production_before = _production_hashes()
    m10_7_preflight_path = REPOSITORY_ROOT / "training/results/milestone10_7/preflight.json"
    m10_7_preflight = json.loads(m10_7_preflight_path.read_text(encoding="utf-8"))
    paired = paired_contract_report()
    source = paired_initial_state("cpu")
    arms = {
        arm: _materialize_arm_initialization(arm, source, load_arm_config(arm))
        for arm in ARM_LAMBDAS
    }
    gae = {
        arm: gae_physical_time_report(
            gamma=float(load_arm_config(arm)["ppo"]["gamma"]),
            gae_lambda=value,
            arm=arm,
        )
        for arm, value in ARM_LAMBDAS.items()
    }
    reward = reward_truth_table_v5()

    corpus = CORPUS_ROOT / GATE_CORPUS_FILENAME
    deterministic = evaluate_stage1_checkpoint(
        SOURCE_CHECKPOINT,
        corpus,
        deterministic=True,
        evaluation_workers=int(args.evaluation_workers),
    )
    stochastic = evaluate_stage1_checkpoint(
        SOURCE_CHECKPOINT,
        corpus,
        deterministic=False,
        evaluation_workers=int(args.evaluation_workers),
    )
    det_path = RESULT_ROOT / "initialization_deterministic.json"
    sto_path = RESULT_ROOT / "initialization_stochastic.json"
    write_json_atomic(det_path, deterministic)
    write_json_atomic(sto_path, stochastic)
    initialization = {
        "shared_by_all_arms": True,
        "deterministic_report": det_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "stochastic_report": sto_path.relative_to(REPOSITORY_ROOT).as_posix(),
        "deterministic_overall": deterministic["overall"],
        "stochastic_overall": stochastic["overall"],
        "stochastic_vs_deterministic_gap": capability_gap(
            deterministic, stochastic
        ),
    }
    production_after = _production_hashes()
    processes_after = _training_processes()
    actor_hashes = {row["actor_state_sha256"] for row in arms.values()}
    critic_hashes = {row["critic_state_sha256"] for row in arms.values()}
    checks = {
        "no_preexisting_m10_8_training_processes": not processes_before,
        "m10_7_preflight_was_passed": m10_7_preflight.get("status") == "passed"
        and m10_7_preflight.get("ppo_authorized") is True,
        "paired_contract_passed": paired["checks"]["passed"],
        "all_arm_actor_states_identical": len(actor_hashes) == 1
        and next(iter(actor_hashes)) == paired["actor_state_sha256"],
        "all_arm_critic_states_identical": len(critic_hashes) == 1
        and next(iter(critic_hashes)) == paired["critic_state_sha256"],
        "all_arm_optimizers_fresh": all(
            row["actor_optimizer_state_entries"] == 0
            and row["critic_optimizer_state_entries"] == 0
            for row in arms.values()
        ),
        "all_arm_reload_parity_exact": all(
            row["reload_parity"]["checks"]["passed"] for row in arms.values()
        ),
        "all_gae_analytical_and_synthetic_proofs_passed": all(
            row["checks"]["passed"] for row in gae.values()
        ),
        "reward_contract_truth_table_passed": reward["checks"]["passed"],
        "initial_deterministic_evaluation_passed": deterministic["checks"]["passed"],
        "initial_stochastic_evaluation_passed": stochastic["checks"]["passed"],
        "frozen_wisp_unchanged": production_before["frozen_wisp_unchanged"]
        and production_after["frozen_wisp_unchanged"]
        and production_before["actual"] == production_after["actual"],
        "no_m10_8_processes_after_preflight": not processes_after,
    }
    checks["passed"] = all(checks.values())
    report = {
        "schema_version": 1,
        "preflight_version": "RivalM10_8GAECreditAssignmentPreflightV1",
        "status": "passed" if checks["passed"] else "failed",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": configuration_evidence(),
        "paired_initialization": paired,
        "arm_initializations": arms,
        "gae_physical_time_proofs": gae,
        "reward_truth_table": reward,
        "inherited_m10_7_preflight": {
            "path": m10_7_preflight_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(m10_7_preflight_path),
            "status": m10_7_preflight.get("status"),
            "ppo_authorized": m10_7_preflight.get("ppo_authorized"),
        },
        "initialization_capability": initialization,
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
        raise RuntimeError(f"M10.8 preflight failed: {checks}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation-workers", type=int, default=24)
    args = parser.parse_args()
    report = run_preflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "ppo_authorized": report["ppo_authorized"],
                "actor_state_sha256": report["paired_initialization"][
                    "actor_state_sha256"
                ],
                "gae_horizons": {
                    arm: {
                        "gamma_times_lambda": row["gamma_times_lambda"],
                        "half_life_seconds": row["half_life_seconds"],
                    }
                    for arm, row in report["gae_physical_time_proofs"].items()
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
