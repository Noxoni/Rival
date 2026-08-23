"""Measured worker and PPO-iteration throughput for the actual Rival v9 path."""

from __future__ import annotations

import gc
import math
import subprocess
import threading
import time
from typing import Any, Callable

import numpy as np
import psutil
import torch
from rlgym_ppo.batched_agents import BatchedAgentManager

from .v9_policy import (
    InstrumentedRivalHybridPolicy,
    RivalCriticV1,
    make_instrumented_rival_policy,
)


PHYSICS_HZ = 120
AGENTS = 2
GAMMA_120HZ = 0.99 ** (1.0 / 8.0)
GAE_LAMBDA_120HZ = 0.95 ** (1.0 / 4.0)


def _stats(values: list[float]) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return {
            "samples": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "samples": int(finite.size),
        "mean": float(finite.mean()),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


class V9ResourceSampler:
    def __init__(self, manager: BatchedAgentManager, interval_seconds: float = 0.5) -> None:
        self.manager = manager
        self.interval_seconds = float(interval_seconds)
        self.samples: list[dict[str, float]] = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    @staticmethod
    def _gpu() -> tuple[float, float]:
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
            utilization, memory = result.stdout.strip().splitlines()[0].split(",")
            return float(utilization.strip()), float(memory.strip())
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return math.nan, math.nan

    def _worker_rss_mib(self) -> float:
        total = 0
        for item in self.manager.processes:
            if item is None:
                continue
            process = item[0]
            try:
                total += psutil.Process(process.pid).memory_info().rss
            except (psutil.Error, AttributeError, TypeError):
                continue
        return total / (1024 * 1024)

    def _run(self) -> None:
        psutil.cpu_percent(interval=None)
        while not self.stop_event.is_set():
            gpu_utilization, gpu_memory = self._gpu()
            virtual = psutil.virtual_memory()
            commit = psutil.swap_memory()
            self.samples.append(
                {
                    "cpu_utilization_percent": float(psutil.cpu_percent(interval=None)),
                    "gpu_utilization_percent": gpu_utilization,
                    "gpu_memory_used_mib": gpu_memory,
                    "system_memory_used_mib": virtual.used / (1024 * 1024),
                    "system_memory_available_mib": virtual.available / (1024 * 1024),
                    "system_commit_used_mib": commit.used / (1024 * 1024),
                    "worker_rss_mib": self._worker_rss_mib(),
                }
            )
            self.stop_event.wait(self.interval_seconds)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> list[dict[str, float]]:
        self.stop_event.set()
        self.thread.join(timeout=3.0)
        return self.samples


def _resource_summary(samples: list[dict[str, float]]) -> dict[str, Any]:
    names = (
        "cpu_utilization_percent",
        "gpu_utilization_percent",
        "gpu_memory_used_mib",
        "system_memory_used_mib",
        "system_memory_available_mib",
        "system_commit_used_mib",
        "worker_rss_mib",
    )
    return {name: _stats([sample[name] for sample in samples]) for name in names}


def _inference_summary(samples: list[dict[str, float | int]]) -> dict[str, Any]:
    batch_counts: dict[str, int] = {}
    for sample in samples:
        key = str(int(sample["batch_size"]))
        batch_counts[key] = batch_counts.get(key, 0) + 1
    return {
        "calls": len(samples),
        "batch_size_counts": batch_counts,
        "batch_size": _stats([float(sample["batch_size"]) for sample in samples]),
        "batch_wall_milliseconds": _stats(
            [float(sample["wall_seconds"]) * 1000.0 for sample in samples]
        ),
        "per_agent_microseconds": _stats(
            [float(sample["per_agent_microseconds"]) for sample in samples]
        ),
        "includes": "actor forward, hybrid sampling/log-probability, and GPU-to-CPU action transfer",
    }


def _process_health(manager: BatchedAgentManager) -> list[dict[str, Any]]:
    health = []
    for item in manager.processes:
        if item is None:
            continue
        process = item[0]
        health.append(
            {
                "pid": int(process.pid),
                "alive": bool(process.is_alive()),
                "exit_code": process.exitcode,
            }
        )
    return health


def benchmark_v9_worker_count(
    worker_count: int,
    env_factory: Callable,
    *,
    warmup_agent_steps: int = 4096,
    measured_agent_steps_per_window: int = 20_000,
    measured_windows: int = 3,
    device: str = "cuda:0",
    restart_agent_steps: int = 4096,
) -> dict[str, Any]:
    """Measure sustained rollout throughput, resources, inference, and restart."""

    policy = make_instrumented_rival_policy(device=device)
    manager = BatchedAgentManager(
        policy,
        min_inference_size=min(int(worker_count), 16),
        seed=20260909,
        standardize_obs=False,
    )
    result: dict[str, Any] = {
        "workers": int(worker_count),
        "status": "error",
        "warmup_agent_steps_target": int(warmup_agent_steps),
        "measured_agent_steps_per_window_target": int(
            measured_agent_steps_per_window
        ),
        "measured_windows": int(measured_windows),
        "min_inference_size": min(int(worker_count), 16),
        "errors_or_stalls": [],
    }
    worker_pids: list[int] = []
    try:
        shapes = manager.init_processes(
            n_processes=int(worker_count),
            build_env_fn=env_factory,
            spawn_delay=None,
            render=False,
            shm_buffer_size=8192,
        )
        worker_pids = [int(item[0].pid) for item in manager.processes]
        manager.collect_timesteps(int(warmup_agent_steps))
        policy.drain_inference_samples()
        windows = []
        all_resources: list[dict[str, float]] = []
        all_inference: list[dict[str, float | int]] = []
        for window_index in range(int(measured_windows)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            sampler = V9ResourceSampler(manager)
            sampler.start()
            started = time.perf_counter()
            _, _, collected, package_seconds = manager.collect_timesteps(
                int(measured_agent_steps_per_window)
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_seconds = time.perf_counter() - started
            resources = sampler.stop()
            inference = policy.drain_inference_samples()
            all_resources.extend(resources)
            all_inference.extend(inference)
            rate = float(collected / package_seconds)
            simulated_rate = rate / (AGENTS * PHYSICS_HZ)
            windows.append(
                {
                    "window": window_index + 1,
                    "collected_agent_steps": int(collected),
                    "package_collection_seconds": float(package_seconds),
                    "wall_seconds": float(wall_seconds),
                    "agent_steps_per_second": rate,
                    "aggregate_simulated_game_seconds_per_second": simulated_rate,
                    "simulated_game_hours_per_wall_hour": simulated_rate,
                    "torch_cuda_peak_allocated_mib": (
                        float(torch.cuda.max_memory_allocated() / (1024 * 1024))
                        if torch.cuda.is_available()
                        else 0.0
                    ),
                    "inference": _inference_summary(inference),
                }
            )
        health = _process_health(manager)
        rates = np.asarray(
            [window["agent_steps_per_second"] for window in windows], dtype=np.float64
        )
        coefficient = float(rates.std() / rates.mean())
        stable = bool(
            np.isfinite(rates).all()
            and coefficient <= 0.15
            and float(rates.min() / rates.mean()) >= 0.80
            and len(health) == int(worker_count)
            and all(item["alive"] for item in health)
        )
        result.update(
            {
                "status": "stable" if stable else "unstable",
                "stable": stable,
                "environment_shapes": [
                    item.tolist() if hasattr(item, "tolist") else item for item in shapes
                ],
                "windows": windows,
                "sustained_agent_steps_per_second_mean": float(rates.mean()),
                "sustained_agent_steps_per_second_median": float(np.median(rates)),
                "sustained_agent_steps_per_second_minimum": float(rates.min()),
                "sustained_agent_steps_per_second_maximum": float(rates.max()),
                "window_rate_coefficient_of_variation": coefficient,
                "aggregate_simulated_game_seconds_per_second_mean": float(
                    rates.mean() / (AGENTS * PHYSICS_HZ)
                ),
                "simulated_game_hours_per_wall_hour": float(
                    rates.mean() / (AGENTS * PHYSICS_HZ)
                ),
                "inference": _inference_summary(all_inference),
                "resources": _resource_summary(all_resources),
                "worker_process_health": health,
                "worker_crashes_or_stalls": sum(not item["alive"] for item in health),
                "finite": bool(np.isfinite(rates).all()),
            }
        )
    except Exception as error:
        result["errors_or_stalls"].append(f"{type(error).__name__}: {error}")
    finally:
        manager.cleanup()

    restart = {
        "attempted": True,
        "agent_steps_target": int(restart_agent_steps),
        "passed": False,
        "errors": [],
    }
    restart_manager = BatchedAgentManager(
        policy,
        min_inference_size=min(int(worker_count), 16),
        seed=20261909,
        standardize_obs=False,
    )
    try:
        restart_manager.init_processes(
            n_processes=int(worker_count),
            build_env_fn=env_factory,
            spawn_delay=None,
            render=False,
            shm_buffer_size=8192,
        )
        policy.drain_inference_samples()
        started = time.perf_counter()
        _, _, collected, elapsed = restart_manager.collect_timesteps(
            int(restart_agent_steps)
        )
        health = _process_health(restart_manager)
        restart.update(
            {
                "collected_agent_steps": int(collected),
                "wall_seconds": float(time.perf_counter() - started),
                "package_collection_seconds": float(elapsed),
                "agent_steps_per_second": float(collected / elapsed),
                "workers_alive": sum(item["alive"] for item in health),
                "passed": len(health) == int(worker_count)
                and all(item["alive"] for item in health)
                and int(collected) >= int(restart_agent_steps),
            }
        )
    except Exception as error:
        restart["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        restart_manager.cleanup()
    result["restart_reliability"] = restart

    lingering = []
    for pid in worker_pids:
        try:
            if psutil.Process(pid).is_running():
                lingering.append(pid)
        except psutil.NoSuchProcess:
            continue
    result["cleanup"] = {
        "measured_worker_pids": worker_pids,
        "lingering_measured_worker_pids": lingering,
        "passed": not lingering,
    }
    if not restart["passed"]:
        result["errors_or_stalls"].append("same-count clean restart failed")
    if lingering:
        result["errors_or_stalls"].append("worker processes lingered after cleanup")
    result["stable"] = bool(
        result.get("stable", False)
        and restart["passed"]
        and not lingering
        and not result["errors_or_stalls"]
    )
    if not result["stable"] and result["status"] == "stable":
        result["status"] = "unstable"
    gc.collect()
    return result


def _batched_values(
    critic: RivalCriticV1,
    observations: np.ndarray,
    device: torch.device,
    batch_size: int = 8192,
) -> np.ndarray:
    output = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            values = critic(
                torch.as_tensor(
                    observations[start : start + batch_size],
                    dtype=torch.float32,
                    device=device,
                )
            )
            output.append(values.squeeze(-1).cpu().numpy())
    return np.concatenate(output)


def _gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    dones: np.ndarray,
    truncated: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    accumulator = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        terminal = bool(dones[index]) or bool(truncated[index])
        mask = 0.0 if terminal else 1.0
        delta = rewards[index] + GAMMA_120HZ * next_values[index] * mask - values[index]
        accumulator = delta + GAMMA_120HZ * GAE_LAMBDA_120HZ * mask * accumulator
        advantages[index] = accumulator
    returns = advantages + values
    return advantages, returns


def benchmark_v9_ppo_iteration(
    worker_count: int,
    env_factory: Callable,
    *,
    rollout_agent_steps: int = 48_000,
    minibatch_size: int = 12_000,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """One measured real hybrid rollout+GAE+update; Gate 11 remains separate."""

    selected_device = torch.device(device)
    policy: InstrumentedRivalHybridPolicy = make_instrumented_rival_policy(
        device=selected_device
    )
    critic = RivalCriticV1().to(selected_device)
    actor_optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    manager = BatchedAgentManager(
        policy,
        min_inference_size=min(int(worker_count), 16),
        seed=20260911,
        standardize_obs=False,
    )
    result: dict[str, Any] = {
        "workers": int(worker_count),
        "rollout_agent_steps_target": int(rollout_agent_steps),
        "minibatch_size": int(minibatch_size),
        "epochs": 1,
        "status": "error",
        "errors": [],
    }
    try:
        manager.init_processes(
            n_processes=int(worker_count),
            build_env_fn=env_factory,
            spawn_delay=None,
            render=False,
            shm_buffer_size=8192,
        )
        manager.collect_timesteps(4096)
        policy.drain_inference_samples()
        iteration_started = time.perf_counter()
        rollout_started = time.perf_counter()
        data, _, collected, package_seconds = manager.collect_timesteps(
            int(rollout_agent_steps)
        )
        rollout_wall = time.perf_counter() - rollout_started
        (
            observations,
            actions,
            old_log_probs,
            rewards,
            next_observations,
            dones,
            truncated,
        ) = data
        inference = policy.drain_inference_samples()
        values = _batched_values(critic, observations, selected_device)
        next_values = _batched_values(critic, next_observations, selected_device)
        advantages, returns = _gae(
            rewards.astype(np.float32),
            values,
            next_values,
            dones,
            truncated,
        )
        advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-6)
        indices = np.arange(len(observations))
        rng = np.random.default_rng(20260911)
        rng.shuffle(indices)
        actor_grad_norms: list[float] = []
        critic_grad_norms: list[float] = []
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        entropies: list[float] = []
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        update_started = time.perf_counter()
        for start in range(0, len(indices), int(minibatch_size)):
            batch = indices[start : start + int(minibatch_size)]
            obs = torch.as_tensor(
                observations[batch], dtype=torch.float32, device=selected_device
            )
            action = torch.as_tensor(
                actions[batch], dtype=torch.float32, device=selected_device
            )
            old_logp = torch.as_tensor(
                old_log_probs[batch], dtype=torch.float32, device=selected_device
            ).reshape(-1)
            advantage = torch.as_tensor(
                advantages[batch], dtype=torch.float32, device=selected_device
            )
            target = torch.as_tensor(
                returns[batch], dtype=torch.float32, device=selected_device
            )
            logp, entropy = policy.get_backprop_data(obs, action)
            logp = logp.reshape(-1)
            ratio = torch.exp(logp - old_logp)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
            actor_loss = -torch.minimum(unclipped, clipped).mean() - 0.001 * entropy
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            actor_optimizer.step()

            predicted = critic(obs).squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(predicted, target)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()
            actor_grad_norms.append(float(actor_norm.detach().cpu()))
            critic_grad_norms.append(float(critic_norm.detach().cpu()))
            actor_losses.append(float(actor_loss.detach().cpu()))
            critic_losses.append(float(critic_loss.detach().cpu()))
            entropies.append(float(entropy.detach().cpu()))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        update_wall = time.perf_counter() - update_started
        total_wall = time.perf_counter() - iteration_started
        finite = all(
            np.isfinite(values).all()
            for values in (
                advantages,
                returns,
                np.asarray(actor_losses),
                np.asarray(critic_losses),
                np.asarray(entropies),
                np.asarray(actor_grad_norms),
                np.asarray(critic_grad_norms),
            )
        )
        nonzero = min(actor_grad_norms) > 0.0 and min(critic_grad_norms) > 0.0
        result.update(
            {
                "status": "passed" if finite and nonzero else "failed",
                "collected_agent_steps": int(collected),
                "rollout_package_seconds": float(package_seconds),
                "rollout_wall_seconds": rollout_wall,
                "ppo_update_wall_seconds": update_wall,
                "iteration_wall_seconds": total_wall,
                "agent_steps_per_second": float(collected / package_seconds),
                "inference": _inference_summary(inference),
                "minibatches": len(actor_losses),
                "actor_loss": _stats(actor_losses),
                "critic_loss": _stats(critic_losses),
                "entropy": _stats(entropies),
                "actor_gradient_norm": _stats(actor_grad_norms),
                "critic_gradient_norm": _stats(critic_grad_norms),
                "analog_head_gradient_nonzero": bool(
                    policy.actor.action_head.analog_mean.weight.grad is not None
                    and policy.actor.action_head.analog_mean.weight.grad.abs().sum() > 0
                ),
                "button_head_gradient_nonzero": bool(
                    policy.actor.action_head.button_logits.weight.grad is not None
                    and policy.actor.action_head.button_logits.weight.grad.abs().sum() > 0
                ),
                "finite": finite,
                "nonzero_actor_and_critic_updates": nonzero,
                "scope": (
                    "Gate 9 one-iteration leading-candidate benchmark at 48k steps; "
                    "not the multiple-iteration save/reload/resume acceptance in Gate 11"
                ),
            }
        )
    except Exception as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    finally:
        manager.cleanup()
    return result
