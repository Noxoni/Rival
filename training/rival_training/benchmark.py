"""Bounded rlgym-ppo worker throughput measurement."""

from __future__ import annotations

import math
import subprocess
import threading
import time
from typing import Any, Callable

import psutil
import numpy as np
import torch
from rlgym_ppo.batched_agents import BatchedAgentManager

from .policy import make_student_policy


class _ResourceSampler:
    def __init__(self, manager, interval_seconds: float = 0.25) -> None:
        self.manager = manager
        self.interval_seconds = interval_seconds
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _gpu_sample() -> tuple[float | None, float | None]:
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2.0,
                creationflags=creation_flags,
            )
            first = result.stdout.strip().splitlines()[0]
            utilization, memory = (float(value.strip()) for value in first.split(","))
            return utilization, memory
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None, None

    def _worker_rss_mib(self) -> float:
        total = 0
        for item in getattr(self.manager, "processes", []):
            process = item[0]
            try:
                total += psutil.Process(process.pid).memory_info().rss
            except (psutil.Error, AttributeError, TypeError):
                continue
        return total / (1024 * 1024)

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            gpu_utilization, gpu_memory = self._gpu_sample()
            memory = psutil.virtual_memory()
            self.samples.append(
                {
                    "cpu_percent": float(psutil.cpu_percent(interval=None)),
                    "system_memory_used_mib": float(memory.used / (1024 * 1024)),
                    "system_memory_available_mib": float(
                        memory.available / (1024 * 1024)
                    ),
                    "worker_rss_mib": self._worker_rss_mib(),
                    "gpu_utilization_percent": (
                        float(gpu_utilization) if gpu_utilization is not None else math.nan
                    ),
                    "gpu_memory_used_mib": (
                        float(gpu_memory) if gpu_memory is not None else math.nan
                    ),
                }
            )
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> list[dict[str, float]]:
        self._stop.set()
        self._thread.join(timeout=3.0)
        return self.samples


def _sample_stats(samples: list[dict[str, float]], name: str) -> dict[str, Any]:
    values = np.asarray([sample[name] for sample in samples], dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return {"samples": 0, "mean": None, "minimum": None, "maximum": None}
    return {
        "samples": int(values.size),
        "mean": float(values.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


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


def benchmark_worker_count_sustained(
    worker_count: int,
    env_factory: Callable,
    *,
    cadence_ticks: int = 4,
    warmup_agent_steps: int = 4096,
    measured_agent_steps_per_window: int = 25000,
    measured_windows: int = 3,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Measure sustained rollout throughput and resource health at one count."""
    policy = make_student_policy(device=device)
    manager = BatchedAgentManager(
        policy,
        min_inference_size=min(worker_count, 16),
        seed=20260827,
        standardize_obs=False,
    )
    result: dict[str, Any] = {
        "workers": worker_count,
        "status": "error",
        "cadence_ticks": cadence_ticks,
        "warmup_agent_steps_target": warmup_agent_steps,
        "measured_agent_steps_per_window_target": measured_agent_steps_per_window,
        "measured_windows": measured_windows,
        "min_inference_processes": min(worker_count, 16),
        "errors_or_stalls": [],
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
        windows = []
        all_resource_samples: list[dict[str, float]] = []
        for index in range(measured_windows):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            sampler = _ResourceSampler(manager)
            sampler.start()
            wall_start = time.perf_counter()
            _, _, collected_steps, package_seconds = manager.collect_timesteps(
                measured_agent_steps_per_window
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - wall_start
            resource_samples = sampler.stop()
            all_resource_samples.extend(resource_samples)
            steps_per_second = collected_steps / package_seconds
            simulated_seconds = collected_steps * cadence_ticks / (2 * 120)
            environment_decisions = collected_steps / 2
            windows.append(
                {
                    "window": index + 1,
                    "collected_agent_steps": int(collected_steps),
                    "package_collection_seconds": float(package_seconds),
                    "wall_seconds": float(wall_seconds),
                    "agent_steps_per_second": float(steps_per_second),
                    "aggregate_simulated_game_seconds_per_second": float(
                        simulated_seconds / package_seconds
                    ),
                    "mean_environment_decision_latency_ms": float(
                        package_seconds * worker_count * 1000
                        / max(environment_decisions, 1)
                    ),
                    "gpu_peak_torch_allocated_mib": (
                        float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                }
            )
        rates = np.asarray(
            [window["agent_steps_per_second"] for window in windows],
            dtype=np.float64,
        )
        process_health = []
        for item in manager.processes:
            process = item[0]
            process_health.append(
                {
                    "pid": int(process.pid),
                    "alive_after_measurement": bool(process.is_alive()),
                    "exit_code": process.exitcode,
                }
            )
        worker_health = {
            "worker_count": len(process_health),
            "alive_after_measurement": sum(
                item["alive_after_measurement"] for item in process_health
            ),
            "crashed_or_stalled": sum(
                not item["alive_after_measurement"] for item in process_health
            ),
            "exit_code_counts": {
                str(code): sum(item["exit_code"] == code for item in process_health)
                for code in sorted(
                    {item["exit_code"] for item in process_health},
                    key=lambda value: (value is not None, value or 0),
                )
            },
        }
        stable = bool(
            np.isfinite(rates).all()
            and rates.size == measured_windows
            and rates.mean() > 0
            and rates.std() / rates.mean() <= 0.20
            and all(item["alive_after_measurement"] for item in process_health)
        )
        result.update(
            {
                "status": "stable" if stable else "unstable",
                "windows": windows,
                "sustained_agent_steps_per_second_mean": float(rates.mean()),
                "sustained_agent_steps_per_second_median": float(np.median(rates)),
                "sustained_agent_steps_per_second_minimum": float(rates.min()),
                "sustained_agent_steps_per_second_maximum": float(rates.max()),
                "window_rate_coefficient_of_variation": float(
                    rates.std() / rates.mean()
                ),
                "aggregate_simulated_game_seconds_per_second_mean": float(
                    np.mean(
                        [
                            window[
                                "aggregate_simulated_game_seconds_per_second"
                            ]
                            for window in windows
                        ]
                    )
                ),
                "rollout_inference_latency": {
                    "interpretation": (
                        "mean wall latency for one environment decision, including "
                        "central policy inference and RocketSim stepping"
                    ),
                    "mean_environment_decision_latency_ms": float(
                        np.mean(
                            [
                                window["mean_environment_decision_latency_ms"]
                                for window in windows
                            ]
                        )
                    ),
                },
                "resources": {
                    name: _sample_stats(all_resource_samples, name)
                    for name in (
                        "cpu_percent",
                        "gpu_utilization_percent",
                        "gpu_memory_used_mib",
                        "system_memory_used_mib",
                        "system_memory_available_mib",
                        "worker_rss_mib",
                    )
                },
                "worker_process_health": worker_health,
                "finite": bool(np.isfinite(rates).all()),
                "stable": stable,
            }
        )
        if not stable:
            result["errors_or_stalls"].append(
                "sustained stability gate failed (rate variance, finite metrics, or worker liveness)"
            )
    except Exception as error:
        result["errors_or_stalls"].append(f"{type(error).__name__}: {error}")
    finally:
        manager.cleanup()
    return result


def select_sustained_worker_count(results: list[dict[str, Any]]) -> int:
    stable = [result for result in results if result.get("stable")]
    if not stable:
        raise RuntimeError(f"No stable sustained worker result: {results}")
    return int(
        max(
            stable,
            key=lambda item: item["sustained_agent_steps_per_second_mean"],
        )["workers"]
    )
