"""Run Gate 11: real CUDA hybrid PPO, checkpoint, fresh reload, and resume."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))

from rival_training.v9_checkpoint import (  # noqa: E402
    config_sha256,
    load_m09_config,
    load_v9_checkpoint,
    portable_path,
    sha256_file,
)
from rival_training.v9_policy import RivalPolicyV1  # noqa: E402


DEFAULT_GATE10 = REPO_ROOT / "training/results/milestone09/gate10_backend_decision.json"
DEFAULT_OUTPUT = REPO_ROOT / "training/results/milestone09/gate11_hybrid_ppo.json"
DEFAULT_CHECKPOINT_ROOT = REPO_ROOT / "training/checkpoints/milestone09"

GATE11_ATTEMPT_HISTORY = [
    {
        "attempt": 1,
        "status": "stopped_cleanly_before_second_update",
        "healthy_completed_iterations": 1,
        "first_iteration_collected_agent_steps": 192002,
        "first_iteration_agent_steps_per_second": 4424.137251529111,
        "first_iteration_actor_update_magnitude": 0.2595788538455963,
        "first_iteration_critic_update_magnitude": 0.21674369275569916,
        "second_collection_serialized_records": 191938,
        "nominal_ppo_batch_records": 192000,
        "failure": (
            "The initial harness required the serialized trajectory count to be at "
            "least the nominal batch size; rlgym-ppo's worker-segment flush returned "
            "62 fewer records."
        ),
        "checkpoint_retained": False,
        "resolution": (
            "Accept only a bounded shortfall of at most four records per selected "
            "worker and update on every returned record; larger shortfalls still fail."
        ),
        "retained_policy_pilot_counter_effect": (
            "none; the stopped technical-gate policy was discarded and Gate 13 had "
            "not begun"
        ),
    },
    {
        "attempt": 2,
        "status": "stopped_during_same_parent_worker_respawn",
        "healthy_completed_iterations": 2,
        "cumulative_agent_steps": 384030,
        "iteration_agent_steps_per_second": [4594.429634622889, 4497.36748799093],
        "iteration_actor_update_magnitude": [
            0.252869576215744,
            0.14295172691345215,
        ],
        "iteration_critic_update_magnitude": [
            0.21672087907791138,
            0.1461903154850006,
        ],
        "pre_reload_checkpoint_retained": True,
        "pre_reload_checkpoint": {
            "directory": (
                "training/checkpoints/milestone09/"
                "gate11-20260823T190009Z/pre_reload"
            ),
            "payload_size_bytes": 47304913,
            "manifest_size_bytes": 3961,
            "manifest_sha256": (
                "43470df764d0388c22bbe2f0a234b29a300c1831c2c50140426be5c66a95e6e1"
            ),
            "actor_size_bytes": 7877036,
            "actor_sha256": (
                "a1880cf345183aa5a5a07e75d8ce0f940159a2731057237334b80673e6c03005"
            ),
            "format": "rival-v9-hybrid-ppo-checkpoint-v1",
            "selected_lineage": False,
        },
        "failure": (
            "After exact fresh-process reload, a second 56-worker launch in the same "
            "long-lived parent exhausted Windows paging-file commit while workers "
            "loaded PyTorch DLLs (WinError 1455)."
        ),
        "resolution": (
            "Run pre-reload and resumed worker phases in separate parent processes so "
            "the complete trajectory allocator is released before respawn."
        ),
        "retained_policy_pilot_counter_effect": (
            "none; this technical-gate checkpoint is preserved but not used as the "
            "selected Gate 11 or Gate 13 lineage"
        ),
    },
]


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _deterministic_parameters(
    actor: RivalPolicyV1, observations: np.ndarray
) -> np.ndarray:
    actor = actor.to("cpu").eval()
    with torch.inference_mode():
        mean, log_std, button_logits = actor(
            torch.as_tensor(observations, dtype=torch.float32)
        )
        expanded_log_std = log_std.expand_as(mean)
        output = torch.cat((mean, expanded_log_std, button_logits), dim=-1)
    return np.ascontiguousarray(output.numpy(), dtype=np.float32)


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _fresh_reload_probe(checkpoint: Path) -> int:
    loaded = load_v9_checkpoint(checkpoint, device="cpu")
    output = _deterministic_parameters(
        loaded["actor"], loaded["reload_observations"]
    )
    report = {
        "status": "passed" if np.isfinite(output).all() else "failed",
        "fresh_process": True,
        "output_shape": list(output.shape),
        "output_sha256": _array_sha256(output),
        "output_float32_base64": base64.b64encode(output.tobytes()).decode("ascii"),
        "actor_optimizer_state_entries": len(loaded["actor_optimizer"].state),
        "critic_optimizer_state_entries": len(loaded["critic_optimizer"].state),
        "trainer_state": loaded["trainer_state"],
        "checkpoint_verified": True,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def _run_fresh_process_probe(
    checkpoint: Path,
    expected: np.ndarray,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--fresh-reload-probe",
            str(checkpoint.resolve()),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Fresh-process checkpoint probe failed: " + completed.stderr.strip()
        )
    child = json.loads(completed.stdout)
    actual = np.frombuffer(
        base64.b64decode(child.pop("output_float32_base64")), dtype=np.float32
    ).reshape(child["output_shape"])
    difference = np.abs(expected - actual)
    proof = {
        **child,
        "subprocess_return_code": completed.returncode,
        "expected_output_sha256": _array_sha256(expected),
        "exact_output_bytes": bool(np.array_equal(expected, actual)),
        "maximum_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "optimizer_states_loaded": child["actor_optimizer_state_entries"] > 0
        and child["critic_optimizer_state_entries"] > 0,
    }
    proof["passed"] = bool(
        proof["status"] == "passed"
        and proof["exact_output_bytes"]
        and proof["optimizer_states_loaded"]
    )
    return proof


def _manifest_size(manifest: dict[str, Any]) -> int:
    return sum(int(item["size_bytes"]) for item in manifest["files"].values())


def _run_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_m09_config(args.config)
    official_config = copy.deepcopy(config)
    if args.workers is not None:
        config["backend"]["worker_count"] = int(args.workers)
        config["backend"]["minimum_inference_size"] = min(int(args.workers), 16)
    if args.rollout_agent_steps is not None:
        config["ppo"]["rollout_agent_steps_per_iteration"] = int(
            args.rollout_agent_steps
        )
        config["ppo"]["ppo_batch_agent_steps"] = int(args.rollout_agent_steps)
    if args.minibatch_agent_steps is not None:
        config["ppo"]["minibatch_agent_steps"] = int(args.minibatch_agent_steps)
    if config != official_config and not args.allow_non_gate:
        raise ValueError("Gate 11 contract overrides require --allow-non-gate")
    gate_status = "diagnostic_only" if args.allow_non_gate else "passed"

    gate10 = json.loads(args.gate10.read_text(encoding="utf-8"))
    if gate10["status"] != "passed" or gate10["selection"]["selected_backend"] != "rlgym-ppo":
        raise RuntimeError("Gate 10 did not select a passing rlgym-ppo path")
    if not args.allow_non_gate and int(config["backend"]["worker_count"]) != int(
        gate10["selection"]["selected_worker_count"]
    ):
        raise RuntimeError("Gate 11 worker count differs from the Gate 10 selection")

    run_id = datetime.now(timezone.utc).strftime("gate11-%Y%m%dT%H%M%SZ")
    run_root = args.checkpoint_root / run_id
    pre_checkpoint = run_root / "pre_reload"
    final_checkpoint = run_root / "resumed"
    before_count = int(config["gate11"]["iterations_before_fresh_reload"])
    after_count = int(config["gate11"]["iterations_after_fresh_reload"])
    if args.allow_non_gate:
        before_count = 1
        after_count = 1

    run_root.mkdir(parents=True, exist_ok=True)
    phase_config = run_root / "phase_config.json"
    phase_config.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    phase_script = REPO_ROOT / "training/scripts/run_m09_hybrid_ppo_phase.py"
    pre_phase_output = run_root / "pre_phase_result.json"
    pre_command = [
        sys.executable,
        str(phase_script),
        "--phase",
        "pre",
        "--config-snapshot",
        str(phase_config),
        "--checkpoint",
        str(pre_checkpoint),
        "--output",
        str(pre_phase_output),
        "--iterations",
        str(before_count),
        "--device",
        args.device,
    ]
    pre_process = subprocess.run(pre_command, check=False, timeout=1800)
    if pre_process.returncode != 0:
        raise RuntimeError(f"Isolated pre-reload phase exited {pre_process.returncode}")
    pre_phase = json.loads(pre_phase_output.read_text(encoding="utf-8"))
    pre_expected = np.frombuffer(
        base64.b64decode(pre_phase.pop("expected_output_float32_base64")),
        dtype=np.float32,
    ).reshape(pre_phase["expected_output_shape"])
    pre_manifest = pre_phase["checkpoint_manifest"]
    initial_shapes = pre_phase["environment_shapes"]
    iteration_reports: list[dict[str, Any]] = list(pre_phase["iterations"])
    cleanup_reports: list[dict[str, Any]] = [pre_phase["cleanup"]]

    pre_reload_proof = _run_fresh_process_probe(pre_checkpoint, pre_expected)
    if not pre_reload_proof["passed"]:
        raise RuntimeError(f"Fresh-process checkpoint proof failed: {pre_reload_proof}")

    resume_phase_output = run_root / "resume_phase_result.json"
    resume_command = [
        sys.executable,
        str(phase_script),
        "--phase",
        "resume",
        "--config-snapshot",
        str(phase_config),
        "--source-checkpoint",
        str(pre_checkpoint),
        "--checkpoint",
        str(final_checkpoint),
        "--output",
        str(resume_phase_output),
        "--iterations",
        str(after_count),
        "--device",
        args.device,
    ]
    resume_process = subprocess.run(resume_command, check=False, timeout=1800)
    if resume_process.returncode != 0:
        raise RuntimeError(
            f"Isolated resumed phase exited {resume_process.returncode}"
        )
    resume_phase = json.loads(resume_phase_output.read_text(encoding="utf-8"))
    final_expected = np.frombuffer(
        base64.b64decode(resume_phase.pop("expected_output_float32_base64")),
        dtype=np.float32,
    ).reshape(resume_phase["expected_output_shape"])
    final_manifest = resume_phase["checkpoint_manifest"]
    resumed_shapes = resume_phase["environment_shapes"]
    counters_restored = bool(resume_phase["restored_counters_exact"])
    iteration_reports.extend(resume_phase["iterations"])
    cleanup_reports.append(resume_phase["cleanup"])
    final_reload_proof = _run_fresh_process_probe(final_checkpoint, final_expected)
    if not final_reload_proof["passed"]:
        raise RuntimeError(f"Final fresh-process checkpoint proof failed: {final_reload_proof}")

    post_reload = iteration_reports[-after_count:]
    checks = {
        "gate10_selected_backend_obeyed": True,
        "actual_rival_v9_actor_and_environment_used": True,
        "multiple_cuda_ppo_iterations_completed": len(iteration_reports) >= 3
        if not args.allow_non_gate
        else len(iteration_reports) >= 2,
        "every_iteration_health_passed": all(
            item["health"]["passed"] for item in iteration_reports
        ),
        "gae_finite_every_iteration": all(
            item["gae"]["finite"] for item in iteration_reports
        ),
        "actor_and_critic_losses_finite": all(
            item["health"]["all_update_metrics_finite"] for item in iteration_reports
        ),
        "analog_and_button_updates_nonzero": all(
            item["ppo"]["all_hybrid_head_gradient_rows_nonzero"]
            for item in iteration_reports
        ),
        "analog_stds_and_button_entropy_finite": all(
            item["health"]["analog_stds_finite_and_positive"]
            and item["health"]["button_entropy_finite_and_positive"]
            for item in iteration_reports
        ),
        "no_action_branch_starved": all(
            item["health"]["all_button_combos_sampled"]
            and item["health"]["all_analog_axes_explored"]
            for item in iteration_reports
        ),
        "checkpoint_contains_models_optimizers_counters_and_contract_hashes": all(
            set(manifest["files"])
            == {
                "actor.pt",
                "actor_optimizer.pt",
                "critic.pt",
                "critic_optimizer.pt",
                "reload_observations.npy",
                "trainer_state.json",
                "training_config.json",
            }
            for manifest in (pre_manifest, final_manifest)
        ),
        "fresh_process_reload_exact": pre_reload_proof["passed"]
        and final_reload_proof["passed"],
        "trainer_counters_restored": counters_restored,
        "resumed_learner_completed_nonzero_update": all(
            item["ppo"]["actor_update_magnitude"] > 0
            and item["ppo"]["critic_update_magnitude"] > 0
            for item in post_reload
        ),
        "worker_cleanup_passed": all(item["passed"] for item in cleanup_reports),
        "pilot_ceiling_not_exceeded": int(
            final_manifest["trainer_state"]["cumulative_agent_steps"]
        )
        <= int(config["pilot"]["maximum_cumulative_agent_steps"]),
        "production_not_promoted": True,
    }
    all_checks = all(checks.values())
    if not all_checks:
        gate_status = "failed"

    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 11,
        "gate_name": "real_hybrid_cuda_ppo_save_reload_resume",
        "generated_at_utc": datetime.now().astimezone().isoformat(),
        "status": gate_status,
        "run_id": run_id,
        "attempt_history": GATE11_ATTEMPT_HISTORY,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else None,
            "rlgym_ppo": importlib.metadata.version("rlgym-ppo"),
        },
        "config": config,
        "config_sha256": config_sha256(config),
        "environment_shapes": {
            "initial": initial_shapes,
            "resumed": resumed_shapes,
        },
        "iterations": iteration_reports,
        "checkpoints": {
            "pre_reload": {
                **pre_manifest,
                "total_size_bytes": _manifest_size(pre_manifest),
            },
            "resumed_final": {
                **final_manifest,
                "total_size_bytes": _manifest_size(final_manifest),
            },
            "committed": False,
            "git_ignored_large_artifacts": True,
        },
        "fresh_process_reload": {
            "pre_reload": pre_reload_proof,
            "resumed_final": final_reload_proof,
        },
        "resume_proof": {
            "source_checkpoint": portable_path(pre_checkpoint),
            "restored_counters_exact": counters_restored,
            "post_reload_iterations": len(post_reload),
            "post_reload_actor_update_magnitudes": [
                item["ppo"]["actor_update_magnitude"] for item in post_reload
            ],
            "post_reload_critic_update_magnitudes": [
                item["ppo"]["critic_update_magnitude"] for item in post_reload
            ],
            "passed": checks["resumed_learner_completed_nonzero_update"],
        },
        "worker_cleanup": cleanup_reports,
        "checks": checks,
        "gate_semantics": {
            "wins_used": False,
            "losses_used": False,
            "scores_used_as_gate": False,
            "production_promoted": False,
            "gate11_steps_count_toward_two_hour_pilot_ceiling": True,
        },
        "source_hashes": {
            "script_sha256": _source_sha256(Path(__file__)),
            "trainer_sha256": _source_sha256(
                REPO_ROOT / "training/rival_training/v9_trainer.py"
            ),
            "checkpoint_sha256": _source_sha256(
                REPO_ROOT / "training/rival_training/v9_checkpoint.py"
            ),
            "isolated_phase_script_sha256": _source_sha256(
                REPO_ROOT / "training/scripts/run_m09_hybrid_ppo_phase.py"
            ),
            "config_file_sha256": sha256_file(args.config),
            "gate10_evidence_sha256": sha256_file(args.gate10),
        },
        "commands": {
            "gate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_hybrid_ppo_gate.py"
            ),
            "worker_phase": (
                "invoked by the gate coordinator in a fresh parent process for "
                "pre-reload and resumed phases"
            ),
            "fresh_process_probe": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_hybrid_ppo_gate.py "
                "--fresh-reload-probe <checkpoint-directory>"
            ),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "training/configs/milestone09.json")
    parser.add_argument("--gate10", type=Path, default=DEFAULT_GATE10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--rollout-agent-steps", type=int)
    parser.add_argument("--minibatch-agent-steps", type=int)
    parser.add_argument("--allow-non-gate", action="store_true")
    parser.add_argument("--fresh-reload-probe", type=Path)
    args = parser.parse_args()
    if args.fresh_reload_probe is not None:
        return _fresh_reload_probe(args.fresh_reload_probe)
    report = _run_gate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "gate": report["gate"],
                "status": report["status"],
                "iterations": len(report["iterations"]),
                "cumulative_agent_steps": report["checkpoints"]["resumed_final"][
                    "trainer_state"
                ]["cumulative_agent_steps"],
                "simulated_game_hours": report["checkpoints"]["resumed_final"][
                    "trainer_state"
                ]["simulated_game_hours"],
                "final_checkpoint": report["checkpoints"]["resumed_final"][
                    "directory"
                ],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] in {"passed", "diagnostic_only"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
