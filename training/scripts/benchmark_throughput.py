"""Benchmark the required 8/12/16/24 RLGym worker candidates."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import sys

import psutil
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.benchmark import (  # noqa: E402
    benchmark_worker_count,
    select_stable_worker_count,
)
from rival_training.environment import make_gym_env_mechanics4  # noqa: E402


def main() -> None:
    candidates = [8, 12, 16, 24]
    results = []
    for count in candidates:
        result = benchmark_worker_count(count, make_gym_env_mechanics4)
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)
    selected = select_stable_worker_count(results)
    report = {
        "schema_version": 1,
        "status": "passed",
        "machine": {
            "platform": platform.platform(),
            "physical_cpu_cores": psutil.cpu_count(logical=False),
            "logical_cpu_cores": psutil.cpu_count(logical=True),
            "memory_gib": psutil.virtual_memory().total / (1024**3),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "candidate_order": candidates,
        "results": results,
        "selection_rule": "highest measured finite stable agent_steps_per_second",
        "selected_worker_count": selected,
    }
    output = REPOSITORY_ROOT / "training/results/throughput_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
