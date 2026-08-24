"""Repair only the self-ancestor false positive in completed M10.7 preflight."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import psutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import write_json_atomic  # noqa: E402
from rival_training.v10_7_campaign import (  # noqa: E402
    DEFAULT_STAGE1_CONFIG,
    RESULT_ROOT,
    load_stage1_config,
)
from rival_training.v10_7_checkpoint import verify_reload_parity  # noqa: E402
from rival_training.v9_checkpoint import sha256_file  # noqa: E402
import run_m10_7_preflight as preflight  # noqa: E402


def main() -> int:
    path = RESULT_ROOT / "preflight.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("preflight_version") != "RivalM10_7ActionPolicyPreflightV1":
        raise RuntimeError("Refusing to repair an unknown preflight report")
    false_checks = {
        key
        for key, value in report["checks"].items()
        if key != "passed" and value is not True
    }
    if false_checks != {
        "no_preexisting_m10_7_training_processes",
        "no_workers_remain",
    }:
        raise RuntimeError(
            f"Process-audit repair cannot alter substantive failures: {false_checks}"
        )
    recorded = report["training_processes_before"] + report[
        "training_processes_after"
    ]
    recorded_ids = {int(row["pid"]) for row in recorded}
    launchers_exact = bool(recorded_ids) and all(
        row["name"] in {"python.exe", "pwsh.exe"}
        and "run_m10_7_preflight.py" in row["command"]
        and "spawn_main" not in row["command"]
        and "run_m10_7_stage1_boundary.py" not in row["command"]
        for row in recorded
    )
    recorded_pids_exited = all(not psutil.pid_exists(pid) for pid in recorded_ids)
    if not launchers_exact or not recorded_pids_exited:
        raise RuntimeError(
            "Recorded false positives are not exact exited preflight launchers: "
            f"rows={recorded}, exited={recorded_pids_exited}"
        )
    remaining = preflight._training_processes()  # noqa: SLF001
    if remaining:
        raise RuntimeError(f"Real M10.7 training processes remain: {remaining}")
    config = load_stage1_config(DEFAULT_STAGE1_CONFIG)
    parity = verify_reload_parity(
        preflight.INITIAL_CHECKPOINT, expected_config=config, device="cpu"
    )
    deterministic = json.loads(
        (RESULT_ROOT / "source_transfer_deterministic.json").read_text(encoding="utf-8")
    )
    stochastic = json.loads(
        (RESULT_ROOT / "source_transfer_stochastic.json").read_text(encoding="utf-8")
    )
    if not (
        parity["checks"]["passed"]
        and deterministic["checks"]["passed"]
        and stochastic["checks"]["passed"]
    ):
        raise RuntimeError("Completed checkpoint/evaluation evidence no longer verifies")
    production = preflight._production_hashes()  # noqa: SLF001
    if not production["frozen_wisp_unchanged"]:
        raise RuntimeError("Frozen Wisp hashes changed during process-audit repair")
    report["process_audit_repair"] = {
        "version": "RivalM10_7PreflightProcessAuditRepairV1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            "The original detector excluded only the inner Python PID and counted "
            "its PowerShell and venv-launcher ancestors as external M10.7 workers."
        ),
        "recorded_false_positive_rows": recorded,
        "all_recorded_rows_are_exact_preflight_launchers_not_spawn_workers": launchers_exact,
        "all_recorded_launcher_pids_exited": recorded_pids_exited,
        "corrected_detector_excludes_current_process_and_all_ancestors": True,
        "real_remaining_m10_7_processes": remaining,
        "initial_checkpoint_manifest_sha256": sha256_file(
            preflight.INITIAL_CHECKPOINT / "checkpoint_manifest.json"
        ),
        "initial_checkpoint_reload_reverified": parity["checks"]["passed"],
        "deterministic_evaluation_reverified": deterministic["checks"]["passed"],
        "stochastic_evaluation_reverified": stochastic["checks"]["passed"],
        "production_reverified": production,
    }
    report["training_processes_before"] = []
    report["training_processes_after"] = []
    report["checks"]["no_preexisting_m10_7_training_processes"] = True
    report["checks"]["no_workers_remain"] = True
    report["checks"]["passed"] = all(
        value for key, value in report["checks"].items() if key != "passed"
    )
    report["status"] = "passed" if report["checks"]["passed"] else "failed"
    report["ppo_authorized"] = report["checks"]["passed"]
    write_json_atomic(path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "ppo_authorized": report["ppo_authorized"],
                "process_audit_repair": report["process_audit_repair"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
