"""Run and record Milestone 09 repository/final verification."""

from __future__ import annotations

import argparse
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

from rival_training.v9_checkpoint import (  # noqa: E402
    DEFAULT_PILOT_CONFIG_PATH,
    load_m09_config,
    load_v9_checkpoint,
    portable_path,
    sha256_file,
)


PRODUCTION_PYTHON = REPOSITORY_ROOT / ".venv/Scripts/python.exe"
TRAINING_PYTHON = REPOSITORY_ROOT / "training/.venv/Scripts/python.exe"
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
    timeout: float = 900.0,
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
    elapsed = time.perf_counter() - started
    combined = completed.stdout + "\n" + completed.stderr
    passed_match = re.search(r"(\d+) passed", combined)
    warnings_match = re.search(r"(\d+) warnings?", combined)
    summary = {
        "name": name,
        "command": " ".join(argv).replace(str(REPOSITORY_ROOT) + os.sep, ""),
        "return_code": completed.returncode,
        "wall_seconds": elapsed,
        "passed": completed.returncode == 0,
        "pytest_passed_count": int(passed_match.group(1)) if passed_match else None,
        "pytest_warning_count": (int(warnings_match.group(1)) if warnings_match else None),
    }
    return summary, completed


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _result_inventory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = REPOSITORY_ROOT / "training/results/milestone09"
    records = []
    parsed: dict[str, Any] = {}
    for path in sorted(root.glob("*.json")):
        if path.name == "gate14_final_verification.json":
            continue
        payload = _json(path)
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


def _checkpoint_reload(gate13: dict[str, Any]) -> dict[str, Any]:
    directory = REPOSITORY_ROOT / gate13["checkpoints"]["final"]["directory"]
    config = load_m09_config(DEFAULT_PILOT_CONFIG_PATH)
    first = load_v9_checkpoint(directory, device="cpu", expected_config=config)
    second = load_v9_checkpoint(directory, device="cpu", expected_config=config)
    maximum_error = max(
        float((left - right).detach().abs().max())
        for left, right in zip(
            first["actor"].parameters(), second["actor"].parameters(), strict=True
        )
    )
    with torch.inference_mode():
        mean, log_std, logits = first["actor"](
            torch.from_numpy(np.asarray(first["reload_observations"], dtype=np.float32))
        )
    return {
        "directory": portable_path(directory),
        "manifest_sha256": sha256_file(directory / "checkpoint_manifest.json"),
        "actor_sha256": sha256_file(directory / "actor.pt"),
        "cumulative_agent_steps": int(first["trainer_state"]["cumulative_agent_steps"]),
        "simulated_game_hours": float(first["trainer_state"]["simulated_game_hours"]),
        "actor_optimizer_state_entries": len(first["actor_optimizer"].state),
        "critic_optimizer_state_entries": len(first["critic_optimizer"].state),
        "second_reload_maximum_parameter_error": maximum_error,
        "held_output_shape": {
            "analog_mean": list(mean.shape),
            "analog_log_std": list(log_std.expand_as(mean).shape),
            "button_logits": list(logits.shape),
        },
        "held_outputs_finite": bool(
            torch.isfinite(mean).all()
            and torch.isfinite(log_std).all()
            and torch.isfinite(logits).all()
        ),
    }


def _export_reload(gate12: dict[str, Any]) -> dict[str, Any]:
    export = gate12["export"]
    artifact = REPOSITORY_ROOT / export["artifact"]["path"]
    metadata = REPOSITORY_ROOT / export["metadata_path"]
    reference = REPOSITORY_ROOT / export["held_corpus"]["path"]
    model = torch.jit.load(str(artifact), map_location="cpu").eval()
    corpus = np.load(reference, allow_pickle=False)
    observations = np.asarray(corpus["observations"], dtype=np.float32)
    with torch.inference_mode():
        outputs = model(torch.from_numpy(observations))
    names = ("analog_mean", "analog_log_std", "button_logits", "controller")
    maximum_errors = {
        name: float(
            np.max(
                np.abs(
                    value.detach().cpu().numpy().astype(np.float64)
                    - np.asarray(corpus[name], dtype=np.float64)
                )
            )
        )
        for name, value in zip(names, outputs, strict=True)
    }
    return {
        "artifact": {
            "path": portable_path(artifact),
            "size_bytes": artifact.stat().st_size,
            "sha256": sha256_file(artifact),
            "matches_committed_evidence": (
                artifact.stat().st_size == export["artifact"]["size_bytes"]
                and sha256_file(artifact) == export["artifact"]["sha256"]
            ),
        },
        "metadata": {
            "path": portable_path(metadata),
            "size_bytes": metadata.stat().st_size,
            "sha256": sha256_file(metadata),
            "matches_committed_evidence": sha256_file(metadata) == export["metadata_sha256"],
        },
        "held_reference": {
            "path": portable_path(reference),
            "size_bytes": reference.stat().st_size,
            "sha256": sha256_file(reference),
            "matches_committed_evidence": (
                reference.stat().st_size == export["held_corpus"]["size_bytes"]
                and sha256_file(reference) == export["held_corpus"]["sha256"]
            ),
        },
        "held_observations": int(len(observations)),
        "maximum_absolute_errors": maximum_errors,
        "tolerance": float(export["export_parity"]["tolerance"]),
        "parity_passed": max(maximum_errors.values())
        <= float(export["export_parity"]["tolerance"]),
        "all_outputs_finite": all(bool(torch.isfinite(value).all()) for value in outputs),
    }


def _hygiene_scan(paths: list[Path]) -> dict[str, Any]:
    absolute_pattern = re.compile(r"\b[A-Za-z]:[\\/]")
    secret_patterns = {
        "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{16,}"),
        "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
        "aws_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
        "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    }
    absolute_hits = []
    secret_hits = []
    for path in paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if absolute_pattern.search(line):
                absolute_hits.append({"path": portable_path(path), "line": line_number})
            for name, pattern in secret_patterns.items():
                if pattern.search(line):
                    secret_hits.append(
                        {"path": portable_path(path), "line": line_number, "kind": name}
                    )
    return {
        "files_scanned": len(paths),
        "absolute_path_hits": absolute_hits,
        "apparent_secret_hits": secret_hits,
        "passed": not absolute_hits and not secret_hits,
    }


def _production_probe() -> dict[str, Any]:
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
        [str(PRODUCTION_PYTHON), "-c", script],
        cwd=REPOSITORY_ROOT / "bot",
        env=_clean_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


def _ignored_artifacts(gate13: dict[str, Any], gate12: dict[str, Any]) -> list[dict[str, Any]]:
    targets = [
        REPOSITORY_ROOT / gate13["checkpoints"]["final"]["directory"] / "actor.pt",
        REPOSITORY_ROOT / gate12["export"]["artifact"]["path"],
        REPOSITORY_ROOT / gate12["export"]["held_corpus"]["path"],
        REPOSITORY_ROOT / "training/tools/rlviser/rlviser.exe",
        REPOSITORY_ROOT / "evidence/raw/rival-v9-gate12-live-20260823T193500Z/native_packets.jsonl",
    ]
    records = []
    for target in targets:
        ignored = _git("check-ignore", "-q", "--", str(target), check=False).returncode == 0
        tracked = _git("ls-files", "--error-unmatch", str(target), check=False).returncode == 0
        records.append(
            {
                "path": portable_path(target),
                "exists": target.exists(),
                "ignored": ignored,
                "tracked": tracked,
                "passed": target.exists() and ignored and not tracked,
            }
        )
    return records


def finalize() -> dict[str, Any]:
    result_inventory, parsed = _result_inventory()
    gate13 = parsed["gate13_scratch_pilot.json"]
    gate12 = parsed["gate12_export_live_inference.json"]

    production_tests, _ = _command(
        "production_pytest",
        [
            str(PRODUCTION_PYTHON),
            "-m",
            "pytest",
            "--basetemp",
            ".pytest_tmp/m09_gate14_evidence_production",
            "tests",
            "-q",
        ],
        environment=_clean_environment(),
    )
    v9_test_files = [str(path) for path in sorted((TRAINING_ROOT / "tests").glob("test_v9_*.py"))]
    scratch_tests, _ = _command(
        "scratch_v9_pytest",
        [
            str(TRAINING_PYTHON),
            "-m",
            "pytest",
            "--basetemp",
            ".pytest_tmp/m09_gate14_evidence_scratch",
            *v9_test_files,
            "-q",
        ],
    )
    ruff_files = [
        *sorted((TRAINING_ROOT / "rival_training").glob("v9_*.py")),
        *sorted((TRAINING_ROOT / "scripts").glob("run_m09_*.py")),
        *sorted((TRAINING_ROOT / "scripts").glob("finalize_m09_*.py")),
        *sorted((TRAINING_ROOT / "tests").glob("test_v9_*.py")),
    ]
    ruff, _ = _command(
        "ruff",
        [
            str(TRAINING_PYTHON),
            "-m",
            "ruff",
            "check",
            *[str(path) for path in ruff_files],
        ],
    )
    compile_check, _ = _command(
        "compileall",
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
        "selected_action_index": selftest_payload["selected_action_index"],
        "compatibility_action_index": selftest_payload["compatibility_action_index"],
        "models": selftest_payload["models"],
    }
    diff_check, _ = _command("git_diff_check", ["git", "diff", "--check"])

    wisp_hashes = {
        name: sha256_file(REPOSITORY_ROOT / "bot/models" / name) for name in EXPECTED_WISP_HASHES
    }
    production_probe = _production_probe()
    checkpoint_reload = _checkpoint_reload(gate13)
    export_reload = _export_reload(gate12)
    ignored = _ignored_artifacts(gate13, gate12)

    hygiene_paths = [
        *sorted((REPOSITORY_ROOT / "training/results/milestone09").glob("*.json")),
        REPOSITORY_ROOT / "training/configs/milestone09.json",
        REPOSITORY_ROOT / "training/configs/milestone09_pilot.json",
        REPOSITORY_ROOT / "docs/MILESTONE_09_RESULTS.md",
    ]
    hygiene = _hygiene_scan(hygiene_paths)
    required_status_files = [
        "gate01_action_contract.json",
        "gate02_canonical_schema.json",
        "gate03_native_capture.json",
        "gate03_observation_parity.json",
        "gate04_prediction_cadence.json",
        "gate05_timing_parity.json",
        "gate06_transition_audit.json",
        "gate07_reward_cadence.json",
        "gate08_environment_stress.json",
        "gate09_worker_sweep.json",
        "gate10_backend_decision.json",
        "gate11_hybrid_ppo.json",
        "gate12_cpu_runtime_probe.json",
        "gate12_export_live_inference.json",
        "gate12_scratch_live_smoke.json",
        "gate13_scratch_pilot.json",
    ]
    geometry_checks = parsed["rlbot_v5_geometry_authority.json"]["checks"]
    local_head = _git("rev-parse", "HEAD").stdout.strip()
    remote_head = _git("ls-remote", "origin", "refs/heads/main").stdout.split()[0]
    branch = _git("branch", "--show-current").stdout.strip()
    stash = _git("stash", "list").stdout.splitlines()
    status_lines = [
        line
        for line in _git("status", "--porcelain", "--untracked-files=all").stdout.splitlines()
        if line
    ]
    allowed_worktree_paths = {
        "docs/MILESTONE_09_RESULTS.md",
        "training/scripts/finalize_m09_gate14.py",
        "training/results/milestone09/gate14_final_verification.json",
    }
    scoped_status = []
    unexpected_status = []
    for line in status_lines:
        path = line[3:].replace("\\", "/")
        record = {"status": line[:2], "path": path}
        scoped_status.append(record)
        if path not in allowed_worktree_paths:
            unexpected_status.append(record)
    running_training_processes = []
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        try:
            command = " ".join(process.info.get("cmdline") or [])
            if any(
                token in command
                for token in (
                    "run_m09_scratch_pilot_phase.py",
                    "run_m09_hybrid_ppo_phase.py",
                )
            ):
                running_training_processes.append(int(process.info["pid"]))
        except (OSError, psutil.Error):
            continue

    checks = {
        "production_suite_90_passed": production_tests["passed"]
        and production_tests["pytest_passed_count"] == 90,
        "scratch_v9_suite_78_passed": scratch_tests["passed"]
        and scratch_tests["pytest_passed_count"] == 78,
        "ruff_passed": ruff["passed"],
        "compile_passed": compile_check["passed"],
        "git_diff_check_passed": diff_check["passed"],
        "frozen_wisp_selftest_passed": selftest["passed"] and selftest["status"] == "pass",
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
        "all_required_gate_results_passed": all(
            parsed[name].get("status") == "passed" for name in required_status_files
        ),
        "geometry_authority_checks_passed": all(geometry_checks.values()),
        "all_committed_result_json_parsed": len(result_inventory) == len(parsed),
        "final_checkpoint_manifest_matches_gate13": checkpoint_reload["manifest_sha256"]
        == gate13["checkpoints"]["final"]["manifest_sha256"],
        "final_checkpoint_actor_matches_gate13": checkpoint_reload["actor_sha256"]
        == gate13["checkpoints"]["final"]["files"]["actor.pt"]["sha256"],
        "final_checkpoint_second_reload_exact": checkpoint_reload[
            "second_reload_maximum_parameter_error"
        ]
        == 0.0,
        "final_checkpoint_outputs_finite": checkpoint_reload["held_outputs_finite"],
        "gate12_export_artifacts_match_evidence": all(
            export_reload[name]["matches_committed_evidence"]
            for name in ("artifact", "metadata", "held_reference")
        ),
        "gate12_export_independent_parity_passed": export_reload["parity_passed"]
        and export_reload["all_outputs_finite"],
        "large_artifacts_present_ignored_and_untracked": all(item["passed"] for item in ignored),
        "hygiene_scan_passed": hygiene["passed"],
        "historical_stash_preserved": any(
            "rival-v4-paused-superseded-before-v4.1" in line for line in stash
        ),
        "m08_final_boundary_preserved": (REPOSITORY_ROOT / "docs/MILESTONE_08_RESULTS.md").is_file()
        and _git("cat-file", "-e", "0b8f313^{commit}", check=False).returncode == 0,
        "main_aligned_with_origin_before_final_commit": branch == "main"
        and local_head == remote_head,
        "only_gate14_files_uncommitted": not unexpected_status,
        "no_training_process_running": not running_training_processes,
        "gate13_no_additional_training_or_promotion": not gate13["authority"][
            "additional_training_authorized"
        ]
        and gate13["conclusion"]["promotion_decision"] == "not_authorized_not_promoted",
    }
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 14,
        "gate_name": "repository and final verification",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "commands": {
            "production_tests": production_tests,
            "scratch_tests": scratch_tests,
            "ruff": ruff,
            "compile": compile_check,
            "wisp_selftest": selftest,
            "git_diff_check": diff_check,
        },
        "gate_results": result_inventory,
        "production": {
            "expected_model_hashes": EXPECTED_WISP_HASHES,
            "actual_model_hashes": wisp_hashes,
            "default_probe": production_probe,
            "promoted": False,
        },
        "independent_artifact_reload": {
            "final_gate13_checkpoint": checkpoint_reload,
            "selected_gate12_export": export_reload,
        },
        "repository_hygiene": {
            "ignored_artifacts": ignored,
            "content_scan": hygiene,
            "historical_stash_entry": next(
                (line for line in stash if "rival-v4-paused-superseded-before-v4.1" in line),
                None,
            ),
            "pre_final_commit_head": local_head,
            "pre_final_commit_origin_main": remote_head,
            "branch": branch,
            "allowed_precommit_worktree_scope": scoped_status,
            "unexpected_worktree_status": unexpected_status,
            "running_training_processes": running_training_processes,
        },
        "documentation": {
            "path": "docs/MILESTONE_09_RESULTS.md",
            "sha256": sha256_file(REPOSITORY_ROOT / "docs/MILESTONE_09_RESULTS.md"),
            "final_remote_sha_recorded_in_completion_handoff": True,
        },
        "checks": checks,
        "conclusion": {
            "all_gates_0_through_14_passed": status == "passed",
            "production_default": "frozen_wisp_production",
            "scratch_candidate_promoted": False,
            "final_checkpoint_agent_steps": gate13["authority"]["final_cumulative_agent_steps"],
            "final_checkpoint_simulated_game_hours": gate13["authority"][
                "final_simulated_game_hours"
            ],
            "promotion_decision": "not_authorized_not_promoted",
            "remaining_required_action": (
                "Commit/push this compact Gate 14 result and documentation, then "
                "read back origin/main and report the final SHA."
            ),
        },
        "reproduction_commands": {
            "production_tests": (
                ".venv/Scripts/python.exe -m pytest --basetemp "
                ".pytest_tmp/m09_gate14_evidence_production tests -q"
            ),
            "scratch_tests": (
                "training/.venv/Scripts/python.exe -m pytest --basetemp "
                ".pytest_tmp/m09_gate14_evidence_scratch training/tests/test_v9_*.py -q"
            ),
            "ruff": (
                "training/.venv/Scripts/python.exe -m ruff check "
                "training/rival_training/v9_*.py training/scripts/run_m09_*.py "
                "training/scripts/finalize_m09_*.py training/tests/test_v9_*.py"
            ),
            "selftest": ".venv/Scripts/python.exe bot/bot.py --self-test",
            "finalize": (
                "training/.venv/Scripts/python.exe training/scripts/finalize_m09_gate14.py"
            ),
        },
    }
    if status != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Gate 14 failed checks: {failed}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "training/results/milestone09/gate14_final_verification.json",
    )
    args = parser.parse_args()
    report = finalize()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": portable_path(args.output),
                "checks": len(report["checks"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
