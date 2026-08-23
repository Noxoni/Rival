"""Find the highest-throughput stable M06 environment count on this machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import psutil
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.benchmark import (  # noqa: E402
    benchmark_worker_count_sustained,
    select_sustained_worker_count,
)
from rival_training.environment import campaign_environment_factory  # noqa: E402


INITIAL_CANDIDATES = (24, 32, 40, 48, 56, 64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-agent-steps", type=int, default=25000)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--maximum-workers", type=int, default=128)
    parser.add_argument("--material-improvement", type=float, default=0.03)
    args = parser.parse_args()
    if args.window_agent_steps < 4096 or args.windows < 2:
        parser.error("sustained sweep requires >=4096 steps/window and >=2 windows")
    if args.maximum_workers < 64:
        parser.error("maximum workers must include the required sweep through 64")

    results = []
    candidates = list(INITIAL_CANDIDATES)
    started = time.perf_counter()
    factory = campaign_environment_factory("stage_a")
    index = 0
    stop_reason = None
    while index < len(candidates):
        count = candidates[index]
        report = benchmark_worker_count_sustained(
            count,
            factory,
            warmup_agent_steps=4096,
            measured_agent_steps_per_window=args.window_agent_steps,
            measured_windows=args.windows,
        )
        results.append(report)
        print(json.dumps(report, indent=2), flush=True)
        index += 1
        if count < 64:
            continue
        if not report.get("stable"):
            stop_reason = f"worker count {count} failed the sustained stability gate"
            break
        previous_stable = [item for item in results[:-1] if item.get("stable")]
        previous_best = max(
            (
                float(item["sustained_agent_steps_per_second_mean"])
                for item in previous_stable
            ),
            default=0.0,
        )
        current_rate = float(report["sustained_agent_steps_per_second_mean"])
        improvement = (
            (current_rate - previous_best) / previous_best if previous_best > 0 else 1.0
        )
        report["improvement_over_previous_best"] = improvement
        if improvement < args.material_improvement:
            stop_reason = (
                f"worker count {count} improved only {improvement:.2%}, below the "
                f"{args.material_improvement:.2%} material-gain threshold"
            )
            break
        next_count = count + 8
        if next_count > args.maximum_workers:
            stop_reason = f"reached explicit safety maximum {args.maximum_workers}"
            break
        candidates.append(next_count)

    selected = select_sustained_worker_count(results)
    selected_result = next(item for item in results if item["workers"] == selected)
    report = {
        "schema_version": 1,
        "status": "passed",
        "selection_rule": "highest mean sustained stable agent-steps/sec",
        "required_initial_candidates": list(INITIAL_CANDIDATES),
        "executed_candidates": [item["workers"] for item in results],
        "extension_increment": 8,
        "material_improvement_threshold": args.material_improvement,
        "stop_reason": stop_reason or "candidate list exhausted",
        "measurement": {
            "warmup_agent_steps": 4096,
            "agent_steps_per_window": args.window_agent_steps,
            "windows_per_candidate": args.windows,
            "environment": "Milestone 06 Stage A Reward V2 curriculum",
            "policy": "frozen Wisp-derived 158-action actor",
            "includes_ppo_update": False,
        },
        "machine": {
            "platform": platform.platform(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "memory_gib": psutil.virtual_memory().total / (1024**3),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "results": results,
        "selected_worker_count": selected,
        "selected_sustained_agent_steps_per_second": selected_result[
            "sustained_agent_steps_per_second_mean"
        ],
        "selected_aggregate_simulated_game_seconds_per_second": selected_result[
            "aggregate_simulated_game_seconds_per_second_mean"
        ],
        "ppo_iteration_timing": None,
        "wall_seconds": time.perf_counter() - started,
    }
    output = REPOSITORY_ROOT / "training/results/milestone06/throughput_sweep.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
