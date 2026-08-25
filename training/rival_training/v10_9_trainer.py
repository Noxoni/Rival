"""Milestone 10.9 PPO V2 trainer."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np
import psutil
import torch

from .v10_6_environment import make_ball_acquisition_phase_a_env
from .v10_7_campaign import button_entropy_coefficient
from .v10_7_checkpoint import load_checkpoint
from .v10_9_actions import (
    ANALOG_DIM,
    ANALOG_FIELDS,
    AR_RHO,
    RivalARStickyBernoulliPolicy,
    independent_ar_log_probability,
    unpack_rollout_actions,
)
from .v10_9_credit import credit_assignment_window_diagnostics
from .v10_9_manager import RivalAR1BatchedAgentManager
from .v9_policy import RivalCriticV1
from .v9_trainer import (
    _add_gradient_rows,
    _batched_values,
    _empty_gradient_rows,
    _gradient_rows,
    _stats,
    compute_physical_time_gae,
    resolve_ppo_batch_size,
)


LOG_PROBABILITY_TOLERANCE = 1e-3
AR_LAGS = (1, 3, 6, 12, 24)


def scale_advantages(
    raw_advantages: np.ndarray, *, epsilon: float
) -> tuple[np.ndarray, float]:
    """Scale a complete rollout without shifting its zero or changing signs."""

    raw = np.asarray(raw_advantages, dtype=np.float32)
    if not np.isfinite(raw).all():
        raise FloatingPointError("Raw advantages must be finite")
    scale = max(float(raw.std()), float(epsilon))
    scaled = raw / scale
    if not np.isfinite(scaled).all():
        raise FloatingPointError("Scaled advantages must be finite")
    nonzero = raw != 0.0
    if not bool(np.array_equal(np.signbit(raw[nonzero]), np.signbit(scaled[nonzero]))):
        raise RuntimeError("Scale-only advantage normalization changed a sign")
    if not bool(np.all(scaled[~nonzero] == 0.0)):
        raise RuntimeError("Scale-only advantage normalization changed zero")
    return scaled, scale


def _explained_variance(targets: np.ndarray, predictions: np.ndarray) -> float:
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    variance = float(target.var())
    return 0.0 if variance <= 1e-12 else 1.0 - float((target - prediction).var()) / variance


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if len(a) < 2 or float(a.std()) <= 1e-12 or float(b.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _critic_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    target = np.asarray(targets, dtype=np.float32).reshape(-1)
    prediction = np.asarray(predictions, dtype=np.float32).reshape(-1)
    return {
        "samples": int(len(target)),
        "loss": float(np.mean(np.square(prediction - target))),
        "explained_variance": _explained_variance(target, prediction),
        "value": _stats(prediction),
        "return": _stats(target),
        "value_return_correlation": _correlation(prediction, target),
    }


def _axis_rows(values: np.ndarray) -> dict[str, Any]:
    return {
        name: _stats(np.asarray(values)[:, index])
        for index, name in enumerate(ANALOG_FIELDS)
    }


def _ar_exploration_diagnostics(
    policy: RivalARStickyBernoulliPolicy,
    observations: np.ndarray,
    rollout_actions: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
) -> dict[str, Any]:
    means = []
    log_stds = []
    epsilons = []
    physical, previous, initial = unpack_rollout_actions(
        torch.as_tensor(rollout_actions, dtype=torch.float32)
    )
    physical_np = physical.numpy()
    with torch.inference_mode():
        for start in range(0, len(observations), 8192):
            obs = torch.as_tensor(
                observations[start : start + 8192],
                dtype=torch.float32,
                device=policy.device,
            )
            previous_batch = previous[start : start + 8192].to(policy.device)
            initial_batch = initial[start : start + 8192].to(policy.device)
            distribution = policy.distribution(obs, previous_batch, initial_batch)
            means.append(distribution.analog_mean.cpu().numpy())
            log_stds.append(distribution.analog_log_std.cpu().numpy())
            epsilons.append(
                distribution.epsilon_from_action(
                    physical[start : start + 8192, :ANALOG_DIM].to(policy.device)
                )
                .cpu()
                .numpy()
            )
    mean = np.concatenate(means)
    log_std = np.concatenate(log_stds)
    epsilon = np.concatenate(epsilons)
    deterministic = np.tanh(mean)
    boundary = np.asarray(terminated, dtype=bool).reshape(-1) | np.asarray(
        truncated, dtype=bool
    ).reshape(-1)
    segment = np.zeros(len(epsilon), dtype=np.int64)
    if len(epsilon) > 1:
        segment[1:] = np.cumsum(boundary[:-1])
    autocorrelation: dict[str, Any] = {}
    deviations = []
    for lag in AR_LAGS:
        mask = segment[lag:] == segment[:-lag]
        empirical = []
        for axis in range(ANALOG_DIM):
            empirical.append(
                _correlation(epsilon[:-lag, axis][mask], epsilon[lag:, axis][mask])
            )
        expected = AR_RHO**lag
        deviation = [abs(value - expected) for value in empirical]
        deviations.extend(deviation)
        autocorrelation[str(lag)] = {
            "physical_seconds": lag / 120.0,
            "expected": expected,
            "empirical_by_axis": dict(zip(ANALOG_FIELDS, empirical)),
            "absolute_deviation_by_axis": dict(zip(ANALOG_FIELDS, deviation)),
            "eligible_pairs": int(mask.sum()),
        }
    return {
        "rho": AR_RHO,
        "policy_mean": _axis_rows(mean),
        "log_std": _axis_rows(log_std),
        "std": _axis_rows(np.exp(log_std)),
        "stochastic_executed_action": _axis_rows(physical_np[:, :ANALOG_DIM]),
        "deterministic_executed_action": _axis_rows(deterministic),
        "epsilon": _axis_rows(epsilon),
        "epsilon_autocorrelation": autocorrelation,
        "maximum_analytical_vs_measured_deviation": float(max(deviations)),
        "saturation_share_abs_action_over_0p95": {
            name: float(np.mean(np.abs(physical_np[:, index]) > 0.95))
            for index, name in enumerate(ANALOG_FIELDS)
        },
        "deterministic_minus_tanh_mean_max_abs": float(
            np.max(np.abs(deterministic - np.tanh(mean)))
        ),
        "initial_transition_records": int(initial.numpy().sum()),
    }


def _replay_log_probabilities(
    policy: RivalARStickyBernoulliPolicy,
    observations: np.ndarray,
    rollout_actions: np.ndarray,
    stored: np.ndarray,
) -> dict[str, Any]:
    replay_differences = []
    independent_differences = []
    with torch.inference_mode():
        for start in range(0, len(observations), 8192):
            obs = torch.as_tensor(
                observations[start : start + 8192],
                dtype=torch.float32,
                device=policy.device,
            )
            actions = torch.as_tensor(
                rollout_actions[start : start + 8192],
                dtype=torch.float32,
                device=policy.device,
            )
            distribution, physical = policy.distribution_for_replay(obs, actions)
            replay = distribution.log_prob(physical)
            independent = independent_ar_log_probability(
                analog_mean=distribution.analog_mean,
                analog_log_std=distribution.analog_log_std,
                button_probabilities=distribution.effective_probabilities,
                physical_actions=physical,
                previous_epsilon=distribution.previous_epsilon,
                initial=distribution.initial,
            )
            expected = torch.as_tensor(
                np.asarray(stored[start : start + 8192]).reshape(-1),
                dtype=torch.float32,
                device=policy.device,
            )
            replay_differences.append((replay - expected).abs().cpu().numpy())
            independent_differences.append((replay - independent).abs().cpu().numpy())
    replay_error = np.concatenate(replay_differences)
    independent_error = np.concatenate(independent_differences)
    return {
        "same_policy_replay": {
            "maximum_abs_error": float(replay_error.max()),
            "mean_abs_error": float(replay_error.mean()),
            "tolerance": LOG_PROBABILITY_TOLERANCE,
            "passed": bool(replay_error.max() <= LOG_PROBABILITY_TOLERANCE),
        },
        "independent_formula": {
            "maximum_abs_error": float(independent_error.max()),
            "mean_abs_error": float(independent_error.mean()),
            "tolerance": 2e-5,
            "passed": bool(independent_error.max() <= 2e-5),
        },
    }


@dataclass
class PreparedRollout:
    observations: np.ndarray
    rollout_actions: np.ndarray
    old_log_probabilities: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    scaled_advantages: np.ndarray
    advantage_scale: float
    returns: np.ndarray
    replay: dict[str, Any]
    exploration: dict[str, Any]
    credit: dict[str, Any]
    collected: int
    collection_seconds: float
    collected_metrics_count: int


class RivalV10_9PPOTrainer:
    """Separate actor/critic PPO with sign-safe GAE weights and AR exploration."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        device: str | torch.device = "cuda:0",
        actor,
        critic: RivalCriticV1,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        trainer_state: dict[str, Any] | None = None,
        env_factory: Callable | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("M10.9 requires CUDA")
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.policy = RivalARStickyBernoulliPolicy(self.actor, self.device)
        state = trainer_state or {}
        self.completed_iterations = int(state.get("completed_iterations", 0))
        self.cumulative_agent_steps = int(state.get("cumulative_agent_steps", 0))
        self.cumulative_model_updates = int(state.get("cumulative_model_updates", 0))
        self.env_factory = env_factory or make_ball_acquisition_phase_a_env
        self.manager: RivalAR1BatchedAgentManager | None = None
        self.worker_pids: list[int] = []

    @classmethod
    def from_checkpoint(
        cls,
        directory: str,
        config: dict[str, Any],
        *,
        device: str | torch.device = "cuda:0",
        env_factory: Callable | None = None,
    ) -> "RivalV10_9PPOTrainer":
        loaded = load_checkpoint(directory, device=device, expected_config=config)
        return cls(
            config,
            device=device,
            actor=loaded["actor"],
            critic=loaded["critic"],
            actor_optimizer=loaded["actor_optimizer"],
            critic_optimizer=loaded["critic_optimizer"],
            trainer_state=loaded["trainer_state"],
            env_factory=env_factory,
        )

    def start_workers(self) -> list[Any]:
        if self.manager is not None:
            raise RuntimeError("Workers already started")
        backend = self.config["backend"]
        seed = int(self.config["gate11"]["seed"]) + self.completed_iterations
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.manager = RivalAR1BatchedAgentManager(
            self.policy,
            min_inference_size=int(backend["minimum_inference_size"]),
            seed=seed,
            standardize_obs=bool(backend["standardize_observations"]),
        )
        shapes = self.manager.init_processes(
            n_processes=int(backend["worker_count"]),
            build_env_fn=self.env_factory,
            collect_metrics_fn=None,
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
        per_hour = int(
            self.config["time_base"][
                "active_learner_steps_per_simulated_game_hour"
            ]
        )
        return {
            "completed_iterations": self.completed_iterations,
            "cumulative_agent_steps": self.cumulative_agent_steps,
            "cumulative_model_updates": self.cumulative_model_updates,
            "simulated_game_hours": self.cumulative_agent_steps / per_hour,
            "worker_count": int(self.config["backend"]["worker_count"]),
            "clean_boundary": True,
            "partial_experience_buffer_records": 0,
            "m10_9_ppo_v2": True,
        }

    def collect_prepared_rollout(self, target: int) -> PreparedRollout:
        if self.manager is None:
            raise RuntimeError("start_workers must be called first")
        data, metrics, collected, collection_seconds = self.manager.collect_timesteps(
            int(target)
        )
        observations, actions, old_logp, rewards, next_obs, done, truncated = data
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        old_logp = np.asarray(old_logp, dtype=np.float32).reshape(-1)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        next_obs = np.asarray(next_obs, dtype=np.float32)
        done = np.asarray(done).reshape(-1)
        truncated = np.asarray(truncated).reshape(-1)
        replay = _replay_log_probabilities(
            self.policy, observations, actions, old_logp
        )
        if not all(row["passed"] for row in replay.values()):
            raise RuntimeError(f"M10.9 exact log-probability replay failed: {replay}")
        values = _batched_values(self.critic, observations, self.device)
        next_values = _batched_values(self.critic, next_obs, self.device)
        ppo = self.config["ppo"]
        advantages, returns = compute_physical_time_gae(
            rewards,
            values,
            next_values,
            done,
            truncated,
            gamma=float(ppo["gamma"]),
            gae_lambda=float(ppo["gae_lambda"]),
        )
        scaled, scale = scale_advantages(
            advantages, epsilon=float(ppo["advantage_epsilon"])
        )
        physical, _, _ = unpack_rollout_actions(torch.as_tensor(actions))
        credit = credit_assignment_window_diagnostics(
            observations=observations,
            actions=physical.numpy(),
            rewards=rewards,
            next_observations=next_obs,
            terminated=done,
            truncated=truncated,
            advantages=advantages,
            scaled_advantages=scaled,
        )
        exploration = _ar_exploration_diagnostics(
            self.policy, observations, actions, done, truncated
        )
        return PreparedRollout(
            observations,
            actions,
            old_logp,
            rewards,
            next_obs,
            done,
            truncated,
            values,
            advantages,
            scaled,
            scale,
            returns,
            replay,
            exploration,
            credit,
            int(collected),
            float(collection_seconds),
            len(metrics),
        )

    def _critic_validation_indices(
        self, count: int, *, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(count)
        validation_count = min(
            int(self.config["ppo"]["critic_validation_agent_steps"]), count // 5
        )
        return order[validation_count:], order[:validation_count]

    def optimize_critic(
        self,
        rollout: PreparedRollout,
        *,
        disposable: bool = False,
    ) -> dict[str, Any]:
        ppo = self.config["ppo"]
        seed = int(self.config["gate11"]["seed"]) + self.completed_iterations
        train_indices, validation_indices = self._critic_validation_indices(
            len(rollout.observations), seed=seed + 90_000
        )

        def metrics(indices: np.ndarray) -> dict[str, Any]:
            predictions = _batched_values(
                self.critic, rollout.observations[indices], self.device
            )
            return _critic_metrics(rollout.returns[indices], predictions)

        before = {
            "training": metrics(train_indices),
            "held_out": metrics(validation_indices),
        }
        epochs = []
        gradient_norms = []
        losses = []
        step_count = 0
        for epoch in range(int(ppo["critic_epochs"])):
            rng = np.random.default_rng(seed + 91_000 + epoch)
            shuffled = train_indices.copy()
            rng.shuffle(shuffled)
            for start in range(
                0, len(shuffled), int(ppo["critic_minibatch_agent_steps"])
            ):
                batch = shuffled[
                    start : start + int(ppo["critic_minibatch_agent_steps"])
                ]
                obs = torch.as_tensor(
                    rollout.observations[batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                target = torch.as_tensor(
                    rollout.returns[batch], dtype=torch.float32, device=self.device
                )
                prediction = self.critic(obs).squeeze(-1)
                loss = torch.nn.functional.mse_loss(prediction, target)
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(ppo["max_gradient_norm"])
                )
                self.critic_optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradient_norms.append(float(norm.detach().cpu()))
                step_count += 1
            epochs.append(
                {
                    "epoch": epoch + 1,
                    "training": metrics(train_indices),
                    "held_out": metrics(validation_indices),
                }
            )
        result = {
            "disposable": disposable,
            "validation_indices_sha256": __import__("hashlib")
            .sha256(validation_indices.tobytes())
            .hexdigest(),
            "training_samples": int(len(train_indices)),
            "held_out_samples": int(len(validation_indices)),
            "before": before,
            "after_each_epoch": epochs,
            "after": epochs[-1],
            "optimizer_steps": step_count,
            "epochs_executed": len(epochs),
            "minibatch_loss": _stats(np.asarray(losses)),
            "gradient_norm": _stats(np.asarray(gradient_norms)),
            "held_out_ev_improvement": (
                epochs[-1]["held_out"]["explained_variance"]
                - before["held_out"]["explained_variance"]
            ),
            "held_out_loss_improvement": (
                before["held_out"]["loss"] - epochs[-1]["held_out"]["loss"]
            ),
        }
        return result

    def optimize_actor(self, rollout: PreparedRollout) -> dict[str, Any]:
        ppo = self.config["ppo"]
        seed = int(self.config["gate11"]["seed"]) + self.completed_iterations
        count, shortfall = resolve_ppo_batch_size(
            len(rollout.observations),
            min(
                int(ppo["ppo_batch_agent_steps"]), len(rollout.observations)
            ),
            int(self.config["backend"]["worker_count"]),
        )
        selection = np.random.default_rng(seed + 80_000).choice(
            len(rollout.observations), size=count, replace=False
        )
        gradient_aggregate = _empty_gradient_rows(self.actor)
        actor_losses = []
        kls = []
        clips = []
        gradient_norms = []
        epoch_rows = []
        actor_before = torch.nn.utils.parameters_to_vector(
            self.actor.parameters()
        ).detach().cpu()
        entropy_step = self.cumulative_agent_steps + rollout.collected // 2
        button_coefficient = button_entropy_coefficient(entropy_step, self.config)
        step_count = 0
        stopped_early = False
        for epoch in range(int(ppo["actor_epochs"])):
            order = selection.copy()
            np.random.default_rng(seed + 81_000 + epoch).shuffle(order)
            epoch_kls = []
            epoch_clips = []
            epoch_losses = []
            for start in range(
                0, len(order), int(ppo["actor_minibatch_agent_steps"])
            ):
                batch = order[start : start + int(ppo["actor_minibatch_agent_steps"])]
                obs = torch.as_tensor(
                    rollout.observations[batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                actions = torch.as_tensor(
                    rollout.rollout_actions[batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                old_logp = torch.as_tensor(
                    rollout.old_log_probabilities[batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                advantage = torch.as_tensor(
                    rollout.scaled_advantages[batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                distribution, physical = self.policy.distribution_for_replay(obs, actions)
                logp = distribution.log_prob(physical)
                entropy = distribution.entropy(physical)
                log_ratio = logp - old_logp
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - float(ppo["clip_range"]),
                    1.0 + float(ppo["clip_range"]),
                )
                loss = (
                    -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
                    - float(ppo["analog_entropy_coefficient"])
                    * entropy.analog_monte_carlo
                    - float(button_coefficient) * entropy.button_exact
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                _add_gradient_rows(gradient_aggregate, _gradient_rows(self.actor))
                norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), float(ppo["max_gradient_norm"])
                )
                self.actor_optimizer.step()
                kl = float(((ratio - 1.0) - log_ratio).detach().mean().cpu())
                clip = float(
                    (torch.abs(ratio - 1.0) > float(ppo["clip_range"]))
                    .float()
                    .mean()
                    .detach()
                    .cpu()
                )
                value = float(loss.detach().cpu())
                actor_losses.append(value)
                kls.append(kl)
                clips.append(clip)
                gradient_norms.append(float(norm.detach().cpu()))
                epoch_kls.append(kl)
                epoch_clips.append(clip)
                epoch_losses.append(value)
                step_count += 1
            mean_kl = float(np.mean(epoch_kls))
            epoch_rows.append(
                {
                    "epoch": epoch + 1,
                    "optimizer_steps": len(epoch_kls),
                    "actor_loss": _stats(np.asarray(epoch_losses)),
                    "approximate_kl": _stats(np.asarray(epoch_kls)),
                    "clip_fraction": _stats(np.asarray(epoch_clips)),
                    "mean_kl_exceeds_stop_threshold": mean_kl
                    > float(ppo["actor_kl_stop_threshold"]),
                }
            )
            if mean_kl > float(ppo["actor_kl_stop_threshold"]):
                stopped_early = True
                break
        actor_after = torch.nn.utils.parameters_to_vector(
            self.actor.parameters()
        ).detach().cpu()
        gradients = {key: value.tolist() for key, value in gradient_aggregate.items()}
        branch_key = next(key for key in gradients if key.startswith("button_logit"))
        branch = {
            name: {
                "mean_weight_absolute_sum": gradients[
                    "analog_mean_abs_sum_by_axis"
                ][index],
                "log_std_absolute_sum": gradients[
                    "analog_log_std_abs_by_axis"
                ][index],
            }
            for index, name in enumerate(ANALOG_FIELDS)
        }
        branch.update(
            {
                name: {"logit_weight_absolute_sum": gradients[branch_key][index]}
                for index, name in enumerate(("jump", "boost", "handbrake"))
            }
        )
        return {
            "batch_agent_steps": count,
            "worker_segment_shortfall_allowance": shortfall,
            "epochs_authorized": int(ppo["actor_epochs"]),
            "epochs_executed": len(epoch_rows),
            "optimizer_steps": step_count,
            "kl_stopped_early": stopped_early,
            "kl_stop_threshold": float(ppo["actor_kl_stop_threshold"]),
            "epochs": epoch_rows,
            "actor_loss": _stats(np.asarray(actor_losses)),
            "approximate_kl": _stats(np.asarray(kls)),
            "clip_fraction": _stats(np.asarray(clips)),
            "gradient_norm": _stats(np.asarray(gradient_norms)),
            "actor_update_magnitude": float(
                torch.linalg.vector_norm(actor_after - actor_before)
            ),
            "gradient_absolute_sums": gradients,
            "controller_branch_gradient_absolute_sums": branch,
            "all_controller_branches_finite_nonzero": all(
                math.isfinite(value) and value > 0.0
                for row in branch.values()
                for value in row.values()
            ),
            "button_entropy_coefficient": float(button_coefficient),
            "button_entropy_schedule_step": int(entropy_step),
        }

    def run_iteration(
        self,
        *,
        rollout_target_agent_steps: int | None = None,
        maximum_cumulative_agent_steps: int | None = None,
    ) -> tuple[dict[str, Any], np.ndarray]:
        ppo = self.config["ppo"]
        target = int(
            ppo["rollout_agent_steps_per_iteration"]
            if rollout_target_agent_steps is None
            else rollout_target_agent_steps
        )
        reserve = 2 * int(self.config["backend"]["worker_count"])
        if maximum_cumulative_agent_steps is not None:
            remaining = int(maximum_cumulative_agent_steps) - self.cumulative_agent_steps
            if target > remaining - reserve:
                raise RuntimeError("M10.9 requested rollout can breach boundary ceiling")
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        rollout = self.collect_prepared_rollout(target)
        if (
            maximum_cumulative_agent_steps is not None
            and self.cumulative_agent_steps + rollout.collected
            > int(maximum_cumulative_agent_steps)
        ):
            raise RuntimeError("M10.9 rollout exceeded reserved boundary ceiling")
        actor_before = torch.nn.utils.parameters_to_vector(
            self.actor.parameters()
        ).detach().cpu()
        critic_before = torch.nn.utils.parameters_to_vector(
            self.critic.parameters()
        ).detach().cpu()
        update_started = time.perf_counter()
        actor_report = self.optimize_actor(rollout)
        critic_report = self.optimize_critic(rollout)
        torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_started
        actor_update = float(
            torch.linalg.vector_norm(
                torch.nn.utils.parameters_to_vector(self.actor.parameters())
                .detach()
                .cpu()
                - actor_before
            )
        )
        critic_update = float(
            torch.linalg.vector_norm(
                torch.nn.utils.parameters_to_vector(self.critic.parameters())
                .detach()
                .cpu()
                - critic_before
            )
        )
        self.completed_iterations += 1
        self.cumulative_agent_steps += rollout.collected
        self.cumulative_model_updates += (
            int(actor_report["optimizer_steps"])
            + int(critic_report["optimizer_steps"])
        )
        physical, _, _ = unpack_rollout_actions(
            torch.as_tensor(rollout.rollout_actions)
        )
        physical_np = physical.numpy()
        final_log_std = self.actor.action_head.analog_log_std.detach().cpu().numpy()
        health_rows = self.worker_health()
        sign_agreement = bool(
            np.array_equal(
                np.signbit(rollout.advantages[rollout.advantages != 0]),
                np.signbit(rollout.scaled_advantages[rollout.advantages != 0]),
            )
        )
        report = {
            "iteration": self.completed_iterations,
            "collected_agent_steps": rollout.collected,
            "experience_records": int(len(rollout.observations)),
            "cumulative_agent_steps": self.cumulative_agent_steps,
            "cumulative_model_updates": self.cumulative_model_updates,
            "simulated_game_hours": self.cumulative_agent_steps / 432_000.0,
            "collection_seconds": rollout.collection_seconds,
            "update_wall_seconds": update_seconds,
            "iteration_wall_seconds": time.perf_counter() - started,
            "agent_steps_per_second": rollout.collected / rollout.collection_seconds,
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated()
            / (1024 * 1024),
            "rollout_log_probability_reproduction": rollout.replay,
            "reward": _stats(rollout.rewards),
            "gae": {
                "gamma": float(ppo["gamma"]),
                "lambda": float(ppo["gae_lambda"]),
                "raw_advantage": _stats(rollout.advantages),
                "scaled_advantage": _stats(rollout.scaled_advantages),
                "advantage_scale": rollout.advantage_scale,
                "raw_scaled_sign_agreement": sign_agreement,
                "zero_preserved": bool(
                    np.all(
                        rollout.scaled_advantages[rollout.advantages == 0.0] == 0.0
                    )
                ),
                "return": _stats(rollout.returns),
            },
            "ppo": {
                "actor": actor_report,
                "critic": critic_report,
                "actor_update_magnitude": actor_update,
                "critic_update_magnitude": critic_update,
                "analog_log_std": final_log_std.tolist(),
                "analog_std": np.exp(final_log_std).tolist(),
            },
            "actions": {
                "continuous": _axis_rows(physical_np[:, :ANALOG_DIM]),
                "button_shares": {
                    name: float(physical_np[:, ANALOG_DIM + index].mean())
                    for index, name in enumerate(("jump", "boost", "handbrake"))
                },
            },
            "exploration": rollout.exploration,
            "credit_assignment": rollout.credit,
            "worker_health": health_rows,
        }
        report["health"] = {
            "raw_scaled_sign_agreement": sign_agreement,
            "log_probability_replay": all(
                row["passed"] for row in rollout.replay.values()
            ),
            "actor_updated": actor_update > 0.0,
            "critic_updated": critic_update > 0.0,
            "all_controller_branches_finite_nonzero": actor_report[
                "all_controller_branches_finite_nonzero"
            ],
            "ppo_metrics_finite": all(
                math.isfinite(value)
                for value in (
                    actor_update,
                    critic_update,
                    actor_report["approximate_kl"]["mean"],
                    critic_report["after"]["held_out"]["loss"],
                )
            ),
            "workers_alive": len(health_rows)
            == int(self.config["backend"]["worker_count"])
            and all(row["alive"] for row in health_rows),
            "boundary_ceiling_respected": maximum_cumulative_agent_steps is None
            or self.cumulative_agent_steps <= int(maximum_cumulative_agent_steps),
        }
        report["health"]["passed"] = all(report["health"].values())
        if not report["health"]["passed"]:
            raise RuntimeError(f"M10.9 PPO V2 health check failed: {report['health']}")
        held_count = int(self.config["gate11"]["held_reload_observation_count"])
        return report, rollout.observations[:held_count].copy()

    def cleanup(self) -> dict[str, Any]:
        if self.manager is None:
            return {"attempted": False, "passed": True, "worker_pids": []}
        pids = list(self.worker_pids)
        manager_diagnostics = self.manager.ar_state_diagnostics()
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
            "ar_manager": manager_diagnostics,
            "passed": not lingering,
        }
