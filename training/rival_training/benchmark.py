"""Bounded rlgym-ppo worker throughput measurement."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import psutil
import torch
from rlgym_ppo.batched_agents import BatchedAgentManager

from .policy import make_student_policy


def benchmark_worker_count(
    worker_count: int,
    env_factory: Callable,
    *,
    cadence_ticks: int = 4,
    warmup_agent_steps: int = 512,
    measured_agent_steps: int = 4096,
    device: str = "cuda",
) -> dict[str, Any]:
    policy = make_student_policy(device=device)
    manager = BatchedAgentManager(
        policy,
        min_inference_size=min(worker_count, 16),
        seed=20260822,
        standardize_obs=False,
    )
    result: dict[str, Any] = {
        "workers": worker_count,
        "status": "error",
        "cadence_ticks": cadence_ticks,
        "warmup_agent_steps_target": warmup_agent_steps,
        "measured_agent_steps_target": measured_agent_steps,
        "min_inference_processes": min(worker_count, 16),
        "expected_steady_inference_agent_batch": 2 * min(worker_count, 16),
    }
    try:
        manager.init_processes(
            n_processes=worker_count,
            build_env_fn=env_factory,
            spawn_delay=None,
            render=False,
            shm_buffer_size=8192,
        )
        manager.collect_timesteps(warmup_agent_steps)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        psutil.cpu_percent(interval=None)
        wall_start = time.perf_counter()
        _, _, collected_steps, package_collection_seconds = manager.collect_timesteps(
            measured_agent_steps
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - wall_start
        cpu_percent = psutil.cpu_percent(interval=None)
        steps_per_second = collected_steps / package_collection_seconds
        # Each RocketSim transition advances one shared 1v1 game by cadence_ticks;
        # rlgym-ppo counts both agents as separate steps.
        simulated_game_seconds = collected_steps * cadence_ticks / (2 * 120)
        result.update(
            {
                "status": "stable",
                "collected_agent_steps": collected_steps,
                "package_collection_seconds": package_collection_seconds,
                "wall_seconds": wall_seconds,
                "agent_steps_per_second": steps_per_second,
                "simulated_game_seconds": simulated_game_seconds,
                "simulated_game_seconds_per_wall_second": simulated_game_seconds
                / package_collection_seconds,
                "cpu_percent_sample": cpu_percent,
                "gpu_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / (1024 * 1024)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "finite": all(
                    math.isfinite(value)
                    for value in (
                        package_collection_seconds,
                        wall_seconds,
                        steps_per_second,
                        cpu_percent,
                    )
                ),
                "errors_or_stalls": [],
            }
        )
    except Exception as error:
        result["errors_or_stalls"] = [f"{type(error).__name__}: {error}"]
    finally:
        manager.cleanup()
    return result


def select_stable_worker_count(results: list[dict[str, Any]]) -> int:
    stable = [
        result
        for result in results
        if result.get("status") == "stable" and result.get("finite")
    ]
    if not stable:
        raise RuntimeError(f"No stable worker benchmark result: {results}")
    return int(max(stable, key=lambda item: item["agent_steps_per_second"])["workers"])
