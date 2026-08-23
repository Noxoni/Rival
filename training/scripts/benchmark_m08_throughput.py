"""Short 48/56/64 throughput sanity check for the M08 dual-rate pipeline."""

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
from rival_training.config import load_milestone08_config  # noqa: E402
from rival_training.environment import make_dual_rate_gym_env  # noqa: E402
from rival_training.mechanics import load_mechanics_actor  # noqa: E402
from rival_training.policy import MechanicsDiscretePolicy  # noqa: E402


CANDIDATES = (48, 56, 64)


def calibrated_policy(*, device: str) -> MechanicsDiscretePolicy:
    actor, _ = load_mechanics_actor(
        REPOSITORY_ROOT
        / "training/artifacts/milestone08/mechanics_initial_v1.pt",
        device="cpu",
    )
    return MechanicsDiscretePolicy(actor, device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-agent-steps", type=int, default=10000)
    parser.add_argument("--windows", type=int, default=2)
    args = parser.parse_args()
    if args.window_agent_steps < 4096 or args.windows < 2:
        parser.error("M08 sanity check requires >=4096 steps and >=2 windows")
    config = load_milestone08_config()
    results = []
    started = time.perf_counter()
    for count in CANDIDATES:
        result = benchmark_worker_count_sustained(
            count,
            make_dual_rate_gym_env,
            warmup_agent_steps=2048,
            measured_agent_steps_per_window=args.window_agent_steps,
            measured_windows=args.windows,
            policy_factory=calibrated_policy,
        )
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    selected = select_sustained_worker_count(results)
    selected_result = next(item for item in results if item["workers"] == selected)
    m06_rate = 12039.091910850184
    m08_rate = float(selected_result["sustained_agent_steps_per_second_mean"])
    report = {
        "schema_version": 1,
        "status": "passed",
        "purpose": "milestone08_short_dual_rate_worker_sanity",
        "selection_rule": "highest mean sustained stable agent-steps/sec",
        "authorized_candidates": list(CANDIDATES),
        "measurement": {
            "warmup_agent_steps": 2048,
            "agent_steps_per_window": args.window_agent_steps,
            "windows_per_candidate": args.windows,
            "environment": "M08 dual-rate Reward V2 majority-natural curriculum",
            "policy": "calibrated independent 69-output mechanics actor",
            "includes_frozen_strategic_inference_in_workers": True,
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
        "configured_start_worker_count": int(config["environment"]["workers"]),
        "selected_sustained_agent_steps_per_second": m08_rate,
        "selected_aggregate_simulated_game_seconds_per_second": selected_result[
            "aggregate_simulated_game_seconds_per_second_mean"
        ],
        "material_change_from_m06": abs(m08_rate - m06_rate) / m06_rate >= 0.10,
        "m06_selected_agent_steps_per_second": m06_rate,
        "wall_seconds": time.perf_counter() - started,
    }
    output = (
        REPOSITORY_ROOT / "training/results/milestone08/throughput_sanity.json"
    )
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
