"""Rival-owned exact-hybrid PPO trainer on rlgym-ppo's worker path."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import psutil
import torch
from rlgym_ppo.batched_agents import BatchedAgentManager

from .v9_actions import ANALOG_DIM, BUTTON_COMBO_COUNT, RivalHybridDistribution
from .v9_checkpoint import load_v9_checkpoint
from .v9_environment import make_v9_training_gym_env
from .v9_policy import (
    InstrumentedRivalHybridPolicy,
    RivalCriticV1,
    RivalPolicyV1,
)


ROLLOUT_LOG_PROBABILITY_TOLERANCE = 1e-3


def resolve_ppo_batch_size(
    experience_records: int, nominal_batch_size: int, worker_count: int
) -> tuple[int, int]:
    maximum_segment_shortfall = 4 * int(worker_count)
    if int(experience_records) < int(nominal_batch_size) - maximum_segment_shortfall:
        raise RuntimeError(
            f"Collected {experience_records} records for a nominal "
            f"{nominal_batch_size}-record PPO batch; maximum worker-segment "
            f"shortfall is {maximum_segment_shortfall}"
        )
    return min(int(nominal_batch_size), int(experience_records)), maximum_segment_shortfall


def _stats(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size or not np.isfinite(array).all():
        raise FloatingPointError("Cannot summarize empty or non-finite values")
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def compute_physical_time_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE without joining independent/partial worker trajectories.

    A true termination suppresses next-state bootstrapping. A collection or
    time-limit truncation ends the reverse recurrence but retains the provided
    next-state bootstrap value.
    """

    arrays = [
        np.asarray(item).reshape(-1)
        for item in (rewards, values, next_values, terminated, truncated)
    ]
    if len({len(item) for item in arrays}) != 1:
        raise ValueError("GAE inputs must have equal lengths")
    reward_values, value_values, following_values, done_values, cut_values = arrays
    advantages = np.zeros(len(reward_values), dtype=np.float32)
    accumulator = 0.0
    for index in range(len(reward_values) - 1, -1, -1):
        done = bool(done_values[index])
        cut = bool(cut_values[index])
        bootstrap = 0.0 if done else 1.0
        continuation = 0.0 if done or cut else 1.0
        delta = (
            float(reward_values[index])
            + gamma * float(following_values[index]) * bootstrap
            - float(value_values[index])
        )
        accumulator = delta + gamma * gae_lambda * continuation * accumulator
        advantages[index] = accumulator
    returns = advantages + np.asarray(value_values, dtype=np.float32)
    return advantages, returns


def _batched_values(
    critic: RivalCriticV1,
    observations: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    output: list[np.ndarray] = []
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


def _rollout_log_probability_error(
    policy: InstrumentedRivalHybridPolicy,
    observations: np.ndarray,
    actions: np.ndarray,
    stored_log_probabilities: np.ndarray,
    *,
    batch_size: int = 8192,
    tolerance: float = ROLLOUT_LOG_PROBABILITY_TOLERANCE,
) -> dict[str, Any]:
    differences: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            obs = torch.as_tensor(
                observations[start : start + batch_size],
                dtype=torch.float32,
                device=policy.device,
            )
            physical = torch.as_tensor(
                actions[start : start + batch_size],
                dtype=torch.float32,
                device=policy.device,
            )
            distribution = policy.distribution(obs)
            replayed = distribution.log_prob(physical).cpu().numpy().reshape(-1)
            stored = np.asarray(
                stored_log_probabilities[start : start + batch_size], dtype=np.float32
            ).reshape(-1)
            differences.append(np.abs(replayed - stored))
    error = np.concatenate(differences)
    return {
        "finite": bool(np.isfinite(error).all()),
        "mean_abs_error": float(error.mean()),
        "p99_abs_error": float(np.percentile(error, 99)),
        "maximum_abs_error": float(error.max()),
        "tolerance": float(tolerance),
        "within_tolerance": bool(float(error.max()) <= float(tolerance)),
        "tolerance_basis": (
            "same weights/actions across asynchronous rollout versus 8192-row "
            "CUDA recomputation; strict same-batch math remains unit-tested"
        ),
    }


def _action_diagnostics(actions: np.ndarray) -> dict[str, Any]:
    physical = np.asarray(actions, dtype=np.float32)
    analog = physical[:, :ANALOG_DIM]
    buttons = np.rint(physical[:, ANALOG_DIM:]).astype(np.int64)
    combos = buttons[:, 0] + 2 * buttons[:, 1] + 4 * buttons[:, 2]
    combo_counts = np.bincount(combos, minlength=BUTTON_COMBO_COUNT)
    analog_rows = []
    for index, name in enumerate(("throttle", "steer", "pitch", "yaw", "roll")):
        row = _stats(analog[:, index])
        row.update(
            {
                "field": name,
                "absolute_over_0_95_share": float(
                    np.mean(np.abs(analog[:, index]) > 0.95)
                ),
                "nontrivial_range": bool(np.ptp(analog[:, index]) > 0.1),
            }
        )
        analog_rows.append(row)
    return {
        "sample_count": int(len(physical)),
        "all_actions_finite": bool(np.isfinite(physical).all()),
        "analog": analog_rows,
        "button_combo_counts": combo_counts.tolist(),
        "button_combo_shares": (combo_counts / max(combo_counts.sum(), 1)).tolist(),
        "all_eight_button_combos_sampled": bool(np.all(combo_counts > 0)),
        "marginal_button_shares": {
            "jump": float(buttons[:, 0].mean()),
            "boost": float(buttons[:, 1].mean()),
            "handbrake": float(buttons[:, 2].mean()),
        },
    }


def _gradient_rows(actor: RivalPolicyV1) -> dict[str, list[float]]:
    head = actor.action_head
    if (
        head.analog_mean.weight.grad is None
        or head.analog_log_std.grad is None
        or head.button_logits.weight.grad is None
    ):
        raise RuntimeError("One or more hybrid action heads received no gradient")
    return {
        "analog_mean_abs_sum_by_axis":
            head.analog_mean.weight.grad.detach().abs().sum(dim=1).cpu().tolist(),
        "analog_log_std_abs_by_axis":
            head.analog_log_std.grad.detach().abs().cpu().tolist(),
        "button_logit_abs_sum_by_combo":
            head.button_logits.weight.grad.detach().abs().sum(dim=1).cpu().tolist(),
    }


def _add_gradient_rows(
    aggregate: dict[str, np.ndarray], current: dict[str, list[float]]
) -> None:
    for key, values in current.items():
        aggregate[key] += np.asarray(values, dtype=np.float64)


class RivalV9PPOTrainer:
    """Clean-boundary PPO trainer using the proven rlgym-ppo worker manager."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        device: str | torch.device = "cuda:0",
        actor: RivalPolicyV1 | None = None,
        critic: RivalCriticV1 | None = None,
        actor_optimizer: torch.optim.Optimizer | None = None,
        critic_optimizer: torch.optim.Optimizer | None = None,
        trainer_state: dict[str, Any] | None = None,
        env_factory: Callable = make_v9_training_gym_env,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Gate 11 requires CUDA")
        self.actor = (actor or RivalPolicyV1()).to(self.device)
        self.critic = (critic or RivalCriticV1()).to(self.device)
        self.policy = InstrumentedRivalHybridPolicy(self.actor, self.device)
        ppo = config["ppo"]
        self.actor_optimizer = actor_optimizer or torch.optim.Adam(
            self.actor.parameters(), lr=float(ppo["actor_learning_rate"])
        )
        self.critic_optimizer = critic_optimizer or torch.optim.Adam(
            self.critic.parameters(), lr=float(ppo["critic_learning_rate"])
        )
        state = trainer_state or {}
        self.completed_iterations = int(state.get("completed_iterations", 0))
        self.cumulative_agent_steps = int(state.get("cumulative_agent_steps", 0))
        self.cumulative_model_updates = int(state.get("cumulative_model_updates", 0))
        self.env_factory = env_factory
        self.manager: BatchedAgentManager | None = None
        self.worker_pids: list[int] = []

    @classmethod
    def from_checkpoint(
        cls,
        directory: str,
        config: dict[str, Any],
        *,
        device: str | torch.device = "cuda:0",
    ) -> "RivalV9PPOTrainer":
        loaded = load_v9_checkpoint(
            directory, device=device, expected_config=config
        )
        return cls(
            config,
            device=device,
            actor=loaded["actor"],
            critic=loaded["critic"],
            actor_optimizer=loaded["actor_optimizer"],
            critic_optimizer=loaded["critic_optimizer"],
            trainer_state=loaded["trainer_state"],
        )

    def start_workers(self) -> list[Any]:
        if self.manager is not None:
            raise RuntimeError("Workers already started")
        backend = self.config["backend"]
        seed = int(self.config["gate11"]["seed"]) + self.completed_iterations
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.manager = BatchedAgentManager(
            self.policy,
            min_inference_size=int(backend["minimum_inference_size"]),
            seed=seed,
            standardize_obs=bool(backend["standardize_observations"]),
        )
        shapes = self.manager.init_processes(
            n_processes=int(backend["worker_count"]),
            build_env_fn=self.env_factory,
            spawn_delay=None,
            render=False,
            shm_buffer_size=8192,
        )
        self.worker_pids = [int(item[0].pid) for item in self.manager.processes]
        return [item.tolist() if hasattr(item, "tolist") else item for item in shapes]

    def worker_health(self) -> list[dict[str, Any]]:
        if self.manager is None:
            return []
        return [
            {
                "pid": int(item[0].pid),
                "alive": bool(item[0].is_alive()),
                "exit_code": item[0].exitcode,
            }
            for item in self.manager.processes
            if item is not None
        ]

    def trainer_state(self) -> dict[str, Any]:
        return {
            "completed_iterations": self.completed_iterations,
            "cumulative_agent_steps": self.cumulative_agent_steps,
            "cumulative_model_updates": self.cumulative_model_updates,
            "simulated_game_seconds": self.cumulative_agent_steps / 240.0,
            "simulated_game_hours": self.cumulative_agent_steps / 864000.0,
            "worker_count": int(self.config["backend"]["worker_count"]),
            "clean_boundary": True,
            "partial_experience_buffer_records": 0,
        }

    def run_iteration(self) -> tuple[dict[str, Any], np.ndarray]:
        if self.manager is None:
            raise RuntimeError("start_workers must be called before PPO")
        ppo = self.config["ppo"]
        rollout_target = int(ppo["rollout_agent_steps_per_iteration"])
        nominal_batch_size = int(ppo["ppo_batch_agent_steps"])
        minibatch_size = int(ppo["minibatch_agent_steps"])
        started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        rollout_started = time.perf_counter()
        data, collected_metrics, collected, collection_seconds = (
            self.manager.collect_timesteps(rollout_target)
        )
        rollout_wall = time.perf_counter() - rollout_started
        (
            observations,
            actions,
            old_log_probabilities,
            rewards,
            next_observations,
            terminated,
            truncated,
        ) = data
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        batch_size, maximum_segment_shortfall = resolve_ppo_batch_size(
            len(observations),
            nominal_batch_size,
            int(self.config["backend"]["worker_count"]),
        )
        rollout_log_probabilities = _rollout_log_probability_error(
            self.policy, observations, actions, old_log_probabilities
        )
        if not rollout_log_probabilities["within_tolerance"]:
            raise RuntimeError(
                "Rollout/backprop log probabilities diverged before PPO: "
                f"{rollout_log_probabilities}"
            )

        values = _batched_values(self.critic, observations, self.device)
        next_values = _batched_values(self.critic, next_observations, self.device)
        advantages, returns = compute_physical_time_gae(
            rewards,
            values,
            next_values,
            terminated,
            truncated,
            gamma=float(ppo["gamma"]),
            gae_lambda=float(ppo["gae_lambda"]),
        )
        gae_finite = bool(np.isfinite(advantages).all() and np.isfinite(returns).all())
        if not gae_finite:
            raise FloatingPointError("GAE produced a non-finite value")
        normalized_advantages = (advantages - advantages.mean()) / max(
            float(advantages.std()), float(ppo["advantage_epsilon"])
        )

        rng = np.random.default_rng(
            int(self.config["gate11"]["seed"]) + self.completed_iterations
        )
        indices = rng.choice(len(observations), size=batch_size, replace=False)
        rng.shuffle(indices)
        actor_before = torch.nn.utils.parameters_to_vector(
            self.actor.parameters()
        ).detach().cpu()
        critic_before = torch.nn.utils.parameters_to_vector(
            self.critic.parameters()
        ).detach().cpu()
        gradient_aggregate = {
            "analog_mean_abs_sum_by_axis": np.zeros(ANALOG_DIM, dtype=np.float64),
            "analog_log_std_abs_by_axis": np.zeros(ANALOG_DIM, dtype=np.float64),
            "button_logit_abs_sum_by_combo": np.zeros(
                BUTTON_COMBO_COUNT, dtype=np.float64
            ),
        }
        actor_losses: list[float] = []
        critic_losses: list[float] = []
        analog_entropies: list[float] = []
        button_entropies: list[float] = []
        mixed_log_ratio_maxima: list[float] = []
        actor_gradient_norms: list[float] = []
        critic_gradient_norms: list[float] = []
        update_started = time.perf_counter()
        for _epoch in range(int(ppo["epochs"])):
            for start in range(0, batch_size, minibatch_size):
                batch = indices[start : start + minibatch_size]
                obs = torch.as_tensor(
                    observations[batch], dtype=torch.float32, device=self.device
                )
                physical = torch.as_tensor(
                    actions[batch], dtype=torch.float32, device=self.device
                )
                old_logp = torch.as_tensor(
                    old_log_probabilities[batch],
                    dtype=torch.float32,
                    device=self.device,
                ).reshape(-1)
                advantage = torch.as_tensor(
                    normalized_advantages[batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                target = torch.as_tensor(
                    returns[batch], dtype=torch.float32, device=self.device
                )

                distribution: RivalHybridDistribution = self.policy.distribution(obs)
                logp = distribution.log_prob(physical)
                entropy = distribution.entropy(physical)
                ratio = torch.exp(logp - old_logp)
                surrogate = ratio * advantage
                clipped = torch.clamp(
                    ratio,
                    1.0 - float(ppo["clip_range"]),
                    1.0 + float(ppo["clip_range"]),
                ) * advantage
                actor_loss = (
                    -torch.minimum(surrogate, clipped).mean()
                    - float(ppo["analog_entropy_coefficient"])
                    * entropy.analog_monte_carlo
                    - float(ppo["button_entropy_coefficient"]) * entropy.button_exact
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                _add_gradient_rows(gradient_aggregate, _gradient_rows(self.actor))
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), float(ppo["max_gradient_norm"])
                )
                self.actor_optimizer.step()

                predicted = self.critic(obs).squeeze(-1)
                critic_loss = torch.nn.functional.mse_loss(predicted, target)
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(ppo["max_gradient_norm"])
                )
                self.critic_optimizer.step()

                actor_losses.append(float(actor_loss.detach().cpu()))
                critic_losses.append(float(critic_loss.detach().cpu()))
                analog_entropies.append(float(entropy.analog_monte_carlo.detach().cpu()))
                button_entropies.append(float(entropy.button_exact.detach().cpu()))
                mixed_log_ratio_maxima.append(
                    float((logp - old_logp).detach().abs().max().cpu())
                )
                actor_gradient_norms.append(float(actor_norm.detach().cpu()))
                critic_gradient_norms.append(float(critic_norm.detach().cpu()))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_started

        actor_after = torch.nn.utils.parameters_to_vector(
            self.actor.parameters()
        ).detach().cpu()
        critic_after = torch.nn.utils.parameters_to_vector(
            self.critic.parameters()
        ).detach().cpu()
        actor_update = float(torch.linalg.vector_norm(actor_after - actor_before))
        critic_update = float(torch.linalg.vector_norm(critic_after - critic_before))
        gradients = {key: value.tolist() for key, value in gradient_aggregate.items()}
        all_gradient_branches_nonzero = all(
            bool(np.all(np.asarray(value) > 0.0)) for value in gradients.values()
        )
        action_diagnostics = _action_diagnostics(actions)
        final_log_std = self.actor.action_head.analog_log_std.detach().cpu().numpy()
        numeric_groups = (
            np.asarray(actor_losses),
            np.asarray(critic_losses),
            np.asarray(analog_entropies),
            np.asarray(button_entropies),
            np.asarray(actor_gradient_norms),
            np.asarray(critic_gradient_norms),
            final_log_std,
        )
        all_update_metrics_finite = all(np.isfinite(item).all() for item in numeric_groups)
        self.completed_iterations += 1
        self.cumulative_agent_steps += int(collected)
        self.cumulative_model_updates += int(
            int(ppo["epochs"]) * math.ceil(batch_size / minibatch_size)
        )
        health = self.worker_health()
        report = {
            "iteration": self.completed_iterations,
            "collected_agent_steps": int(collected),
            "experience_records": int(len(observations)),
            "cumulative_agent_steps": self.cumulative_agent_steps,
            "cumulative_model_updates": self.cumulative_model_updates,
            "simulated_game_seconds": self.cumulative_agent_steps / 240.0,
            "simulated_game_hours": self.cumulative_agent_steps / 864000.0,
            "collection_seconds": float(collection_seconds),
            "rollout_wall_seconds": float(rollout_wall),
            "update_wall_seconds": float(update_seconds),
            "iteration_wall_seconds": float(time.perf_counter() - started),
            "agent_steps_per_second": float(collected / collection_seconds),
            "cuda_peak_allocated_mib": float(
                torch.cuda.max_memory_allocated() / (1024 * 1024)
                if torch.cuda.is_available()
                else 0.0
            ),
            "rlgym_ppo_metrics_records": int(len(collected_metrics)),
            "rollout_log_probability_reproduction": rollout_log_probabilities,
            "reward": _stats(rewards),
            "gae": {
                "finite": gae_finite,
                "advantage": _stats(advantages),
                "return": _stats(returns),
                "normalized_advantage": _stats(normalized_advantages),
                "gamma": float(ppo["gamma"]),
                "lambda": float(ppo["gae_lambda"]),
                "true_termination_suppresses_bootstrap": True,
                "truncation_bootstraps_but_stops_recurrence": True,
            },
            "ppo": {
                "batch_agent_steps": batch_size,
                "nominal_batch_agent_steps": nominal_batch_size,
                "worker_segment_record_shortfall": max(
                    0, nominal_batch_size - len(observations)
                ),
                "maximum_allowed_worker_segment_shortfall": maximum_segment_shortfall,
                "minibatch_agent_steps": minibatch_size,
                "epochs": int(ppo["epochs"]),
                "actor_loss": _stats(np.asarray(actor_losses)),
                "critic_loss": _stats(np.asarray(critic_losses)),
                "analog_entropy": _stats(np.asarray(analog_entropies)),
                "button_entropy": _stats(np.asarray(button_entropies)),
                "maximum_absolute_log_ratio_per_minibatch": mixed_log_ratio_maxima,
                "actor_gradient_norm": _stats(np.asarray(actor_gradient_norms)),
                "critic_gradient_norm": _stats(np.asarray(critic_gradient_norms)),
                "actor_update_magnitude": actor_update,
                "critic_update_magnitude": critic_update,
                "head_gradient_absolute_sums": gradients,
                "all_hybrid_head_gradient_rows_nonzero": all_gradient_branches_nonzero,
                "analog_log_std": final_log_std.tolist(),
                "analog_std": np.exp(final_log_std).tolist(),
            },
            "actions": action_diagnostics,
            "worker_health": health,
            "health": {
                "all_update_metrics_finite": all_update_metrics_finite,
                "gae_finite": gae_finite,
                "actor_updated": actor_update > 0.0,
                "critic_updated": critic_update > 0.0,
                "all_hybrid_head_gradient_rows_nonzero": all_gradient_branches_nonzero,
                "analog_stds_finite_and_positive": bool(
                    np.isfinite(final_log_std).all() and np.all(np.exp(final_log_std) > 0)
                ),
                "button_entropy_finite_and_positive": bool(
                    np.isfinite(button_entropies).all()
                    and min(button_entropies) > 0.0
                ),
                "all_button_combos_sampled": action_diagnostics[
                    "all_eight_button_combos_sampled"
                ],
                "all_analog_axes_explored": all(
                    row["nontrivial_range"] for row in action_diagnostics["analog"]
                ),
                "rollout_workers_alive": len(health)
                == int(self.config["backend"]["worker_count"])
                and all(item["alive"] for item in health),
                "rollout_log_probabilities_reproduced": rollout_log_probabilities[
                    "within_tolerance"
                ],
            },
        }
        report["health"]["passed"] = all(report["health"].values())
        if not report["health"]["passed"]:
            raise RuntimeError(f"Gate 11 PPO iteration failed health checks: {report}")
        held_count = int(self.config["gate11"]["held_reload_observation_count"])
        return report, observations[:held_count].copy()

    def cleanup(self) -> dict[str, Any]:
        if self.manager is None:
            return {"attempted": False, "passed": True, "worker_pids": []}
        pids = list(self.worker_pids)
        self.manager.cleanup()
        self.manager = None
        deadline = time.monotonic() + 5.0
        lingering = [pid for pid in pids if psutil.pid_exists(pid)]
        while lingering and time.monotonic() < deadline:
            time.sleep(0.05)
            lingering = [pid for pid in pids if psutil.pid_exists(pid)]
        return {
            "attempted": True,
            "worker_pids": pids,
            "lingering_worker_pids": lingering,
            "passed": not lingering,
        }
