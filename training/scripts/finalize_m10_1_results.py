"""Run and record the final Milestone 10.1 stop verification."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import numpy as np
import psutil
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v10_1_campaign import (  # noqa: E402
    M10_PLUS25_ACTOR_SHA256,
    M10_PLUS25_MANIFEST_SHA256,
    M10_PLUS25_STEPS,
    load_m10_1_config,
    verify_exact_plus25_start,
)
from rival_training.v9_checkpoint import (  # noqa: E402
    load_v9_checkpoint,
    portable_path,
    sha256_file,
)


PRODUCTION_PYTHON = REPOSITORY_ROOT / ".venv/Scripts/python.exe"
TRAINING_PYTHON = REPOSITORY_ROOT / "training/.venv/Scripts/python.exe"
RESULT_ROOT = REPOSITORY_ROOT / "training/results/milestone10_1"
FINAL_CHECKPOINT = REPOSITORY_ROOT / (
    "training/checkpoints/milestone10_1/boundaries/plus-010h/032019870"
)
FINAL_OUTPUT = RESULT_ROOT / "final_verification.json"
EXPECTED_FINAL_ACTOR_SHA256 = (
    "e6b9fd1a38dbb5c6670711a8b47769a628d2f20caf7c88738f372d0839b7a3b6"
)
EXPECTED_FINAL_MANIFEST_SHA256 = (
    "d1a785ef439b0127b5ab1a9ff1693ade1aa11d850151cd17b9733bbeb98dacb3"
)
EXPECTED_FINAL_STEPS = 32_019_870
EXPECTED_WISP_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("RIVAL_"):
            environment.pop(name)
    environment.pop("PYTHONPATH", None)
    return environment


def _command(
    name: str,
    argv: list[str],
    *,
    environment: dict[str, str] | None = None,
    timeout: float = 1_800.0,
) -> tuple[dict[str, Any], subprocess.CompletedProcess[str]]:
    started = time.perf_counter()
    completed = subprocess.run(
        argv,
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    combined = completed.stdout + "\n" + completed.stderr
    passed_match = re.search(r"(\d+) passed", combined)
    warnings_match = re.search(r"(\d+) warnings?", combined)
    display_command = " ".join(argv)
    display_command = display_command.replace(
        str(REPOSITORY_ROOT) + os.sep, ""
    ).replace(REPOSITORY_ROOT.as_posix() + "/", "")
    summary = {
        "name": name,
        "command": display_command,
        "return_code": completed.returncode,
        "wall_seconds": time.perf_counter() - started,
        "passed": completed.returncode == 0,
        "pytest_passed_count": int(passed_match.group(1)) if passed_match else None,
        "pytest_warning_count": (
            int(warnings_match.group(1)) if warnings_match else None
        ),
    }
    return summary, completed


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _production_probe() -> dict[str, Any]:
    script = (
        "import json,config; print(json.dumps({"
        "'mode':config.POLICY_RUNTIME_MODE,'tick_skip':config.TICK_SKIP,"
        "'v9':config.V9_SCRATCH_POLICY_ENABLED,"
        "'candidate':config.CANDIDATE_POLICY_ENABLED,"
        "'m08':config.M08_DUAL_RATE_ENABLED,"
        "'policy':config.MODEL_INFO_POLICY.path.name,"
        "'shared_head':config.MODEL_INFO_SHARED_HEAD.path.name}))"
    )
    completed = subprocess.run(
        [str(PRODUCTION_PYTHON), "-c", script],
        cwd=REPOSITORY_ROOT / "bot",
        env=_clean_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


def _checkpoint_reload() -> dict[str, Any]:
    config = load_m10_1_config()
    first = load_v9_checkpoint(
        FINAL_CHECKPOINT, device="cpu", expected_config=config
    )
    second = load_v9_checkpoint(
        FINAL_CHECKPOINT, device="cpu", expected_config=config
    )
    maximum_error = max(
        float((left - right).detach().abs().max())
        for left, right in zip(
            first["actor"].parameters(),
            second["actor"].parameters(),
            strict=True,
        )
    )
    observations = torch.from_numpy(
        np.asarray(first["reload_observations"], dtype=np.float32)
    )
    with torch.inference_mode():
        mean, log_std, button_logits = first["actor"](observations)
    return {
        "directory": portable_path(FINAL_CHECKPOINT),
        "actor_sha256": sha256_file(FINAL_CHECKPOINT / "actor.pt"),
        "manifest_sha256": sha256_file(
            FINAL_CHECKPOINT / "checkpoint_manifest.json"
        ),
        "cumulative_agent_steps": int(
            first["trainer_state"]["cumulative_agent_steps"]
        ),
        "simulated_game_hours": float(
            first["trainer_state"]["simulated_game_hours"]
        ),
        "completed_iterations": int(
            first["trainer_state"]["completed_iterations"]
        ),
        "actor_optimizer_state_entries": len(first["actor_optimizer"].state),
        "critic_optimizer_state_entries": len(first["critic_optimizer"].state),
        "second_reload_maximum_parameter_error": maximum_error,
        "held_observation_count": int(len(observations)),
        "held_output_shapes": {
            "analog_mean": list(mean.shape),
            "analog_log_std": list(log_std.expand_as(mean).shape),
            "button_logits": list(button_logits.shape),
        },
        "held_outputs_finite": bool(
            torch.isfinite(mean).all()
            and torch.isfinite(log_std).all()
            and torch.isfinite(button_logits).all()
        ),
    }


def _result_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = []
    parsed: dict[str, Any] = {}
    for path in sorted(RESULT_ROOT.glob("*.json")):
        if path.name == FINAL_OUTPUT.name:
            continue
        payload = _read(path)
        parsed[path.name] = payload
        records.append(
            {
                "path": portable_path(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "status": payload.get("status"),
            }
        )
    return records, parsed


def _ignored_checkpoint_files() -> list[dict[str, Any]]:
    targets = [
        FINAL_CHECKPOINT / "actor.pt",
        FINAL_CHECKPOINT / "critic.pt",
        FINAL_CHECKPOINT / "actor_optimizer.pt",
        FINAL_CHECKPOINT / "critic_optimizer.pt",
        REPOSITORY_ROOT / "training/tools/rlviser/rlviser.exe",
    ]
    records = []
    for path in targets:
        ignored = _git(
            "check-ignore", "-q", "--", str(path), check=False
        ).returncode == 0
        tracked = _git(
            "ls-files", "--error-unmatch", str(path), check=False
        ).returncode == 0
        records.append(
            {
                "path": portable_path(path),
                "exists": path.exists(),
                "ignored": ignored,
                "tracked": tracked,
                "passed": path.exists() and ignored and not tracked,
            }
        )
    return records


def _running_campaign_processes() -> list[dict[str, Any]]:
    found = []
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if "run_m10_1_campaign_boundary.py" in command:
                found.append(
                    {
                        "pid": int(process.info["pid"]),
                        "name": process.info.get("name"),
                    }
                )
        except (OSError, psutil.Error):
            continue
    return found


def finalize() -> dict[str, Any]:
    production_tests, _ = _command(
        "production_pytest",
        [
            str(PRODUCTION_PYTHON),
            "-m",
            "pytest",
            "tests",
            "-q",
            "--basetemp",
            str(REPOSITORY_ROOT / ".tmp/pytest-m10-1-final-production"),
        ],
        environment=_clean_environment(),
    )
    training_tests, _ = _command(
        "complete_training_pytest",
        [
            str(TRAINING_PYTHON),
            "-m",
            "pytest",
            "training/tests",
            "-q",
            "--basetemp",
            str(REPOSITORY_ROOT / ".tmp/pytest-m10-1-final-training"),
        ],
    )
    ruff, _ = _command(
        "ruff_training",
        [str(TRAINING_PYTHON), "-m", "ruff", "check", "training"],
    )
    compile_check, _ = _command(
        "compileall_training",
        [
            str(TRAINING_PYTHON),
            "-m",
            "compileall",
            "-q",
            "training/rival_training",
            "training/scripts",
            "training/tests",
        ],
    )
    selftest_command, selftest_completed = _command(
        "frozen_wisp_selftest",
        [str(PRODUCTION_PYTHON), "bot/bot.py", "--self-test"],
        environment=_clean_environment(),
    )
    selftest_payload = json.loads(selftest_completed.stdout)
    selftest = {
        **selftest_command,
        "status": selftest_payload["status"],
        "collision_mesh_count": selftest_payload["collision_mesh_count"],
        "finite_logits": selftest_payload["finite_logits"],
        "observation_shape": selftest_payload["observation_shape"],
        "policy_output_shape": selftest_payload["policy_output_shape"],
    }
    diff_check, _ = _command("git_diff_check", ["git", "diff", "--check"])

    inventory, parsed = _result_inventory()
    preflight = parsed["preflight.json"]
    visual = parsed["preflight_rlviser_visual_inspection.json"]
    plus2p5 = parsed["boundary_plus-002p5h.json"]
    plus5 = parsed["boundary_plus-005h.json"]
    plus10 = parsed["boundary_plus-010h.json"]
    checkpoint = _checkpoint_reload()
    exact_start = verify_exact_plus25_start(device="cpu")
    ignored = _ignored_checkpoint_files()
    wisp_hashes = {
        name: sha256_file(REPOSITORY_ROOT / "bot/models" / name)
        for name in EXPECTED_WISP_HASHES
    }
    production_probe = _production_probe()
    stash = _git("stash", "list").stdout.splitlines()
    local_head = _git("rev-parse", "HEAD").stdout.strip()
    remote_head = _git("ls-remote", "origin", "refs/heads/main").stdout.split()[0]
    branch = _git("branch", "--show-current").stdout.strip()
    later_paths = [
        REPOSITORY_ROOT / "training/checkpoints/milestone10_1/boundaries/plus-015h",
        REPOSITORY_ROOT / "training/checkpoints/milestone10_1/boundaries/plus-020h",
        REPOSITORY_ROOT / "training/checkpoints/milestone10_1/boundaries/plus-025h",
        RESULT_ROOT / "boundary_plus-015h.json",
        RESULT_ROOT / "boundary_plus-020h.json",
        RESULT_ROOT / "boundary_plus-025h.json",
        REPOSITORY_ROOT / "training/checkpoints/milestone10/boundaries/plus-050h",
        REPOSITORY_ROOT / "training/checkpoints/milestone10/boundaries/plus-100h",
    ]
    running = _running_campaign_processes()
    checks = {
        "production_tests_passed": production_tests["passed"],
        "complete_training_tests_passed": training_tests["passed"],
        "ruff_training_passed": ruff["passed"],
        "compileall_training_passed": compile_check["passed"],
        "git_diff_check_passed": diff_check["passed"],
        "frozen_wisp_selftest_passed": selftest["passed"]
        and selftest["status"] == "pass"
        and selftest["finite_logits"],
        "frozen_wisp_hashes_unchanged": wisp_hashes == EXPECTED_WISP_HASHES,
        "production_default_remains_frozen_wisp": production_probe
        == {
            "mode": "frozen_wisp_production",
            "tick_skip": 8,
            "v9": False,
            "candidate": False,
            "m08": False,
            "policy": "POLICY.lt",
            "shared_head": "SHARED_HEAD.lt",
        },
        "preflight_and_visual_inspection_passed": preflight["status"] == "passed"
        and preflight["checks"]["passed"]
        and visual["status"] == "passed"
        and visual["checks"]["passed"],
        "exact_m10_plus25_source_preserved": exact_start["status"] == "passed"
        and exact_start["checkpoint"]["cumulative_agent_steps"] == M10_PLUS25_STEPS
        and exact_start["checkpoint"]["actor_sha256"] == M10_PLUS25_ACTOR_SHA256
        and exact_start["checkpoint"]["manifest_sha256"]
        == M10_PLUS25_MANIFEST_SHA256,
        "all_three_required_boundaries_passed_evidence_checks": all(
            result["status"] == "passed" and result["checks"]["passed"]
            for result in (plus2p5, plus5, plus10)
        ),
        "phase_a_never_authorized_phase_b": not plus2p5["gates"]["phase_a"][
            "passed"
        ]
        and not plus5["gates"]["phase_a"]["passed"]
        and not plus10["gates"]["phase_a"]["passed"],
        "hard_review_stop_decision_exact": plus10["decision"]
        == {
            "action": "stop_phase_A_readiness_failed_by_plus_10h",
            "continue_training": False,
            "next_phase": None,
        },
        "no_later_m10_or_m10_1_boundaries_exist": not any(
            path.exists() for path in later_paths
        ),
        "final_checkpoint_actor_exact": checkpoint["actor_sha256"]
        == EXPECTED_FINAL_ACTOR_SHA256,
        "final_checkpoint_manifest_exact": checkpoint["manifest_sha256"]
        == EXPECTED_FINAL_MANIFEST_SHA256,
        "final_checkpoint_steps_exact": checkpoint["cumulative_agent_steps"]
        == EXPECTED_FINAL_STEPS,
        "final_checkpoint_second_reload_exact": checkpoint[
            "second_reload_maximum_parameter_error"
        ]
        == 0.0,
        "final_checkpoint_outputs_and_optimizer_state_present": checkpoint[
            "held_outputs_finite"
        ]
        and checkpoint["actor_optimizer_state_entries"] > 0
        and checkpoint["critic_optimizer_state_entries"] > 0,
        "checkpoint_and_rlviser_artifacts_ignored_untracked": all(
            item["passed"] for item in ignored
        ),
        "historical_v4_stash_preserved": any(
            "rival-v4-paused-superseded-before-v4.1" in line for line in stash
        ),
        "no_campaign_process_running": not running,
        "main_aligned_with_origin_before_final_commit": branch == "main"
        and local_head == remote_head,
        "production_promotion_authorized": False,
    }
    checks["passed"] = all(
        value
        for key, value in checks.items()
        if key != "production_promotion_authorized"
    ) and checks["production_promotion_authorized"] is False
    report = {
        "schema_version": 1,
        "status": "passed" if checks["passed"] else "failed",
        "milestone": "10.1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "conclusion": {
            "campaign_stopped": True,
            "stop_reason": "phase_A_readiness_failed_by_plus_10h",
            "phase_b_activated": False,
            "additional_m10_1_training_authorized": False,
            "production_promotion_decision": "not_authorized_not_promoted",
        },
        "commands": {
            "production_tests": production_tests,
            "training_tests": training_tests,
            "ruff": ruff,
            "compileall": compile_check,
            "wisp_selftest": selftest,
            "git_diff_check": diff_check,
        },
        "source_checkpoint": exact_start,
        "final_checkpoint": checkpoint,
        "boundary_results": inventory,
        "production": {
            "expected_wisp_hashes": EXPECTED_WISP_HASHES,
            "actual_wisp_hashes": wisp_hashes,
            "default_probe": production_probe,
            "promoted": False,
        },
        "ignored_artifacts": ignored,
        "repository_before_final_commit": {
            "branch": branch,
            "local_head": local_head,
            "origin_main": remote_head,
            "historical_stash": stash,
        },
        "running_campaign_processes": running,
        "checks": checks,
    }
    FINAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if report["status"] != "passed":
        raise RuntimeError(f"M10.1 final verification failed: {checks}")
    return report


def main() -> int:
    report = finalize()
    print(json.dumps({"status": report["status"], "output": str(FINAL_OUTPUT)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
