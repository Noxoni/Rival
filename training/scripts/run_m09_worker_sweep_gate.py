"""Run Milestone 09 Gate 9 actual-workload worker and throughput sweep."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import psutil
import torch


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_benchmark import (  # noqa: E402
    benchmark_v9_ppo_iteration,
    benchmark_v9_worker_count,
)
from rival_training.v9_environment import (  # noqa: E402
    V9_TRAINING_ENVIRONMENT_VERSION,
    make_v9_training_gym_env,
)
from rival_training.v9_observations import observation_schema_manifest  # noqa: E402
from rival_training.v9_policy import policy_metadata  # noqa: E402


RESULT_PATH = (
    TRAINING_ROOT / "results" / "milestone09" / "gate09_worker_sweep.json"
)
BASE_CANDIDATES = (16, 24, 32, 40, 48, 56)
UPPER_INCREMENT = 8
MAXIMUM_EXTENSION_WORKERS = 96
MATERIAL_IMPROVEMENT = 0.03


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [result for result in results if result.get("stable")]


def _selected(results: list[dict[str, Any]]) -> dict[str, Any]:
    stable = _stable(results)
    if not stable:
        raise RuntimeError("No reliably restartable stable v9 worker count")
    return max(
        stable,
        key=lambda result: result["sustained_agent_steps_per_second_mean"],
    )


def _should_extend(results: list[dict[str, Any]]) -> bool:
    if len(results) < 2:
        return False
    current = results[-1]
    previous = results[-2]
    if not current.get("stable") or not previous.get("stable"):
        return False
    current_rate = current["sustained_agent_steps_per_second_mean"]
    previous_rate = previous["sustained_agent_steps_per_second_mean"]
    return bool(current_rate >= (1.0 + MATERIAL_IMPROVEMENT) * previous_rate)


def _print_result(result: dict[str, Any]) -> None:
    print(
        json.dumps(
            {
                "workers": result["workers"],
                "status": result["status"],
                "stable": result.get("stable", False),
                "agent_steps_per_second": result.get(
                    "sustained_agent_steps_per_second_mean"
                ),
                "simulated_game_seconds_per_second": result.get(
                    "aggregate_simulated_game_seconds_per_second_mean"
                ),
                "cpu_percent": result.get("resources", {})
                .get("cpu_utilization_percent", {})
                .get("mean"),
                "gpu_percent": result.get("resources", {})
                .get("gpu_utilization_percent", {})
                .get("mean"),
                "restart_passed": result.get("restart_reliability", {}).get(
                    "passed", False
                ),
                "errors": result.get("errors_or_stalls", []),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def run_gate(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("Gate 9 actual actor sweep requires CUDA")
    if args.candidates:
        candidates = tuple(int(value) for value in args.candidates.split(","))
    else:
        candidates = BASE_CANDIDATES
    if not args.allow_non_gate and candidates != BASE_CANDIDATES:
        raise ValueError("Official Gate 9 must begin with 16,24,32,40,48,56")

    results: list[dict[str, Any]] = []
    for count in candidates:
        result = benchmark_v9_worker_count(
            count,
            make_v9_training_gym_env,
            warmup_agent_steps=args.warmup_agent_steps,
            measured_agent_steps_per_window=args.window_agent_steps,
            measured_windows=args.windows,
            device=args.device,
            restart_agent_steps=args.restart_agent_steps,
        )
        results.append(result)
        _print_result(result)

    if not args.candidates:
        next_count = candidates[-1] + UPPER_INCREMENT
        while (
            next_count <= MAXIMUM_EXTENSION_WORKERS and _should_extend(results)
        ):
            result = benchmark_v9_worker_count(
                next_count,
                make_v9_training_gym_env,
                warmup_agent_steps=args.warmup_agent_steps,
                measured_agent_steps_per_window=args.window_agent_steps,
                measured_windows=args.windows,
                device=args.device,
                restart_agent_steps=args.restart_agent_steps,
            )
            results.append(result)
            _print_result(result)
            next_count += UPPER_INCREMENT

    selected = _selected(results)
    stable_ranked = sorted(
        _stable(results),
        key=lambda result: result["sustained_agent_steps_per_second_mean"],
        reverse=True,
    )
    leading_counts = [int(result["workers"]) for result in stable_ranked[:2]]
    ppo_iterations = []
    if not args.skip_ppo_iteration:
        for count in leading_counts:
            print(f"Gate 9 PPO-inclusive leading-candidate iteration: {count} workers", flush=True)
            result = benchmark_v9_ppo_iteration(
                count,
                make_v9_training_gym_env,
                rollout_agent_steps=args.ppo_rollout_agent_steps,
                minibatch_size=args.ppo_minibatch_size,
                device=args.device,
            )
            ppo_iterations.append(result)
            print(
                json.dumps(
                    {
                        "workers": count,
                        "status": result["status"],
                        "iteration_wall_seconds": result.get("iteration_wall_seconds"),
                        "rollout_wall_seconds": result.get("rollout_wall_seconds"),
                        "update_wall_seconds": result.get("ppo_update_wall_seconds"),
                        "errors": result.get("errors", []),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    base_complete = set(result["workers"] for result in results).issuperset(
        BASE_CANDIDATES
    )
    extension_stopped_for_reason = bool(
        results[-1]["workers"] >= MAXIMUM_EXTENSION_WORKERS
        or not _should_extend(results)
    )
    checks = {
        "base_16_to_56_sweep_complete": base_complete,
        "every_candidate_completed_without_error": all(
            result["status"] in {"stable", "unstable"} for result in results
        ),
        "at_least_one_stable_candidate": bool(stable_ranked),
        "selected_is_highest_stable_sustained_agent_step_rate": selected
        == _selected(results),
        "selected_candidate_restart_passed": selected["restart_reliability"][
            "passed"
        ],
        "selected_candidate_has_no_worker_crash_or_stall": selected[
            "worker_crashes_or_stalls"
        ]
        == 0,
        "selected_candidate_cleanup_passed": selected["cleanup"]["passed"],
        "upper_extension_stopped_at_saturation_instability_or_cap": (
            extension_stopped_for_reason
        ),
        "actual_v9_actor_and_environment_used": True,
        "cuda_used": str(args.device).startswith("cuda")
        and torch.cuda.is_available(),
        "ppo_iteration_timed_on_two_leading_candidates": args.skip_ppo_iteration
        or (
            len(ppo_iterations) == min(2, len(stable_ranked))
            and all(result["status"] == "passed" for result in ppo_iterations)
        ),
    }
    official = not args.allow_non_gate and not args.skip_ppo_iteration
    status = "passed" if all(checks.values()) and official else "diagnostic_only"
    schema = observation_schema_manifest()
    policy = policy_metadata()
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 9,
        "gate_name": "actual_v9_worker_throughput_sweep",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "contract": {
            "environment_version": V9_TRAINING_ENVIRONMENT_VERSION,
            "policy_version": policy["policy_version"],
            "actor_parameter_count": policy["actor_parameter_count"],
            "observation_schema_sha256": schema["schema_sha256"],
            "observation_size": schema["float_count"],
            "physics_hz": 120,
            "policy_hz": 120,
            "agents_per_environment": 2,
            "prediction_refresh_ticks": 1,
            "repeat_action": False,
            "device": args.device,
            "min_inference_size": "min(workers,16)",
            "standardize_observations": False,
            "base_candidates": list(BASE_CANDIDATES),
            "upper_increment": UPPER_INCREMENT,
            "maximum_extension_workers": MAXIMUM_EXTENSION_WORKERS,
            "material_upper_improvement_fraction": MATERIAL_IMPROVEMENT,
            "warmup_agent_steps": args.warmup_agent_steps,
            "measured_agent_steps_per_window": args.window_agent_steps,
            "measured_windows": args.windows,
            "restart_agent_steps": args.restart_agent_steps,
        },
        "results": results,
        "selection": {
            "selected_worker_count": int(selected["workers"]),
            "selected_sustained_agent_steps_per_second": selected[
                "sustained_agent_steps_per_second_mean"
            ],
            "selected_aggregate_simulated_game_seconds_per_second": selected[
                "aggregate_simulated_game_seconds_per_second_mean"
            ],
            "selected_simulated_game_hours_per_wall_hour": selected[
                "simulated_game_hours_per_wall_hour"
            ],
            "stable_ranking": [
                {
                    "workers": int(result["workers"]),
                    "agent_steps_per_second": result[
                        "sustained_agent_steps_per_second_mean"
                    ],
                }
                for result in stable_ranked
            ],
            "criterion": (
                "maximum stable sustained total agent-steps/sec among candidates "
                "whose workers remained alive, cleaned up, and passed same-count restart"
            ),
            "m06_56_worker_result_transferred_without_measurement": False,
        },
        "ppo_inclusive_leading_candidate_iterations": ppo_iterations,
        "ppo_iteration_scope": {
            "rollout_agent_steps": args.ppo_rollout_agent_steps,
            "minibatch_size": args.ppo_minibatch_size,
            "epochs": 1,
            "purpose": "Gate 9 candidate wall-time comparison",
            "not_gate11": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_logical_count": psutil.cpu_count(logical=True),
            "cpu_physical_count": psutil.cpu_count(logical=False),
            "system_memory_total_mib": psutil.virtual_memory().total / (1024 * 1024),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(torch.device(args.device))
                if torch.cuda.is_available()
                else None
            ),
        },
        "source_hashes": {
            "policy_sha256": _sha256(
                TRAINING_ROOT / "rival_training" / "v9_policy.py"
            ),
            "benchmark_sha256": _sha256(
                TRAINING_ROOT / "rival_training" / "v9_benchmark.py"
            ),
            "environment_sha256": _sha256(
                TRAINING_ROOT / "rival_training" / "v9_environment.py"
            ),
            "script_sha256": _sha256(Path(__file__)),
        },
        "commands": {
            "gate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_worker_sweep_gate.py"
            ),
            "tests": (
                "training/.venv/Scripts/python.exe -m pytest "
                "training/tests/test_v9_policy.py "
                "training/tests/test_v9_worker_benchmark.py -q"
            ),
        },
        "gate_semantics": {
            "wins_used": False,
            "losses_used": False,
            "scores_used": False,
            "trained_checkpoint_used": False,
            "randomly_initialized_scratch_actor_used": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "checks": checks,
                "selection": report["selection"],
                "ppo_iterations": ppo_iterations,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if status in {"passed", "diagnostic_only"} else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str)
    parser.add_argument("--allow-non-gate", action="store_true")
    parser.add_argument("--skip-ppo-iteration", action="store_true")
    parser.add_argument("--warmup-agent-steps", type=int, default=4096)
    parser.add_argument("--window-agent-steps", type=int, default=20_000)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--restart-agent-steps", type=int, default=4096)
    parser.add_argument("--ppo-rollout-agent-steps", type=int, default=48_000)
    parser.add_argument("--ppo-minibatch-size", type=int, default=12_000)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args()


def main() -> int:
    return run_gate(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
