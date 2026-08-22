"""Bounded multi-update PPO proof using rlgym-ppo's rollout and learner stack."""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch
from rlgym_ppo import Learner
from rlgym_ppo.ppo import PPOLearner

from .checkpoint import load_ppo_state, save_actor_checkpoint, save_ppo_state
from .policy import attach_student_policy, make_student_policy
from .teacher import EXPANDED_ACTION_COUNT


def _fresh_package_ppo(
    *,
    device: str,
    batch_size: int,
    minibatch_size: int,
    epochs: int,
    policy_learning_rate: float,
    critic_learning_rate: float,
) -> PPOLearner:
    learner = PPOLearner(
        obs_space_size=432,
        act_space_size=EXPANDED_ACTION_COUNT,
        policy_type=0,
        policy_layer_sizes=(64,),
        critic_layer_sizes=(256, 256),
        continuous_var_range=(0.1, 1.0),
        batch_size=batch_size,
        n_epochs=epochs,
        policy_lr=policy_learning_rate,
        critic_lr=critic_learning_rate,
        clip_range=0.2,
        ent_coef=0.005,
        mini_batch_size=minibatch_size,
        device=device,
    )
    attach_student_policy(
        learner,
        make_student_policy(device=device),
        policy_learning_rate,
    )
    return learner


def _finite_report(report: dict[str, Any]) -> bool:
    numeric = [
        float(value)
        for value in report.values()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    return all(math.isfinite(value) for value in numeric)


def run_bounded_ppo_smoke(
    env_factory: Callable,
    *,
    worker_count: int,
    checkpoint_directory: str | Path,
    actor_checkpoint_path: str | Path,
    device: str = "cuda:0",
    iterations_before_reload: int = 2,
    iterations_after_reload: int = 1,
    agent_steps_per_iteration: int = 2048,
    batch_size: int = 1024,
    minibatch_size: int = 256,
    epochs: int = 2,
    policy_learning_rate: float = 1e-5,
    critic_learning_rate: float = 1e-4,
) -> dict[str, Any]:
    if iterations_before_reload < 1 or iterations_after_reload < 1:
        raise ValueError("Smoke must exercise updates both before and after reload")
    checkpoint_directory = Path(checkpoint_directory)
    orchestrator = Learner(
        env_create_function=env_factory,
        n_proc=worker_count,
        min_inference_size=min(worker_count, 16),
        render=False,
        timestep_limit=(iterations_before_reload + iterations_after_reload)
        * agent_steps_per_iteration,
        exp_buffer_size=max(batch_size * 2, agent_steps_per_iteration * 2),
        ts_per_iteration=agent_steps_per_iteration,
        standardize_returns=False,
        standardize_obs=False,
        policy_layer_sizes=(64,),
        critic_layer_sizes=(256, 256),
        ppo_epochs=epochs,
        ppo_batch_size=batch_size,
        ppo_minibatch_size=minibatch_size,
        ppo_ent_coef=0.005,
        ppo_clip_range=0.2,
        policy_lr=policy_learning_rate,
        critic_lr=critic_learning_rate,
        log_to_wandb=False,
        checkpoint_load_folder=None,
        checkpoints_save_folder=str(checkpoint_directory),
        add_unix_timestamp=False,
        random_seed=20260822,
        device=device,
    )
    attach_student_policy(
        orchestrator.ppo_learner,
        make_student_policy(device=device),
        policy_learning_rate,
    )
    orchestrator.agent.policy = orchestrator.ppo_learner.policy

    iteration_reports: list[dict[str, Any]] = []
    overall_action_counts = np.zeros(EXPANDED_ACTION_COUNT, dtype=np.int64)
    checkpoint_manifest: dict[str, Any] | None = None
    reload_proof: dict[str, Any] = {}
    try:
        total_iterations = iterations_before_reload + iterations_after_reload
        for iteration in range(total_iterations):
            phase = "before_reload" if iteration < iterations_before_reload else "after_reload"
            start = time.perf_counter()
            experience, _, collected_steps, collection_seconds = (
                orchestrator.agent.collect_timesteps(agent_steps_per_iteration)
            )
            experience_records = int(experience[0].shape[0])
            actions_taken = np.asarray(experience[1]).astype(np.int64).reshape(-1)
            action_counts = np.bincount(
                actions_taken, minlength=EXPANDED_ACTION_COUNT
            )[:EXPANDED_ACTION_COUNT]
            overall_action_counts += action_counts
            rewards_collected = np.asarray(experience[3], dtype=np.float64).reshape(-1)
            orchestrator.add_new_experience(experience)
            package_report = orchestrator.ppo_learner.learn(
                orchestrator.experience_buffer
            )
            iteration_seconds = time.perf_counter() - start
            report = {
                "iteration": iteration + 1,
                "phase": phase,
                "collected_agent_steps": collected_steps,
                "experience_records": experience_records,
                "mean_reward": float(rewards_collected.mean()),
                "minimum_reward": float(rewards_collected.min()),
                "maximum_reward": float(rewards_collected.max()),
                "appended_action_count": int(action_counts[90:].sum()),
                "appended_action_rate": float(
                    action_counts[90:].sum() / max(action_counts.sum(), 1)
                ),
                "top_action_counts": [
                    {"action_index": int(index), "count": int(action_counts[index])}
                    for index in np.argsort(action_counts)[-10:][::-1]
                    if action_counts[index] > 0
                ],
                "collection_seconds": collection_seconds,
                "agent_steps_per_second": collected_steps / collection_seconds,
                "iteration_seconds": iteration_seconds,
                **package_report,
            }
            report["all_metrics_finite"] = _finite_report(report)
            if not report["all_metrics_finite"]:
                raise FloatingPointError(f"Non-finite PPO report: {report}")
            if report["Policy Update Magnitude"] <= 0:
                raise RuntimeError(f"PPO iteration performed no policy update: {report}")
            iteration_reports.append(report)
            orchestrator.experience_buffer.clear()

            if iteration + 1 == iterations_before_reload:
                sample = torch.randn(32, 432, device=device)
                with torch.no_grad():
                    logits_before = orchestrator.ppo_learner.policy.logits(sample).cpu()
                trainer_state = {
                    "completed_iterations": iteration + 1,
                    "cumulative_agent_steps": orchestrator.agent.cumulative_timesteps,
                    "cumulative_model_updates": orchestrator.ppo_learner.cumulative_model_updates,
                    "worker_count": worker_count,
                }
                checkpoint_manifest = save_ppo_state(
                    checkpoint_directory,
                    orchestrator.ppo_learner,
                    trainer_state,
                )
                reloaded = _fresh_package_ppo(
                    device=device,
                    batch_size=batch_size,
                    minibatch_size=minibatch_size,
                    epochs=epochs,
                    policy_learning_rate=policy_learning_rate,
                    critic_learning_rate=critic_learning_rate,
                )
                restored_state = load_ppo_state(checkpoint_directory, reloaded)
                with torch.no_grad():
                    logits_after = reloaded.policy.logits(sample).cpu()
                reload_proof = {
                    "fresh_instance": True,
                    "restored_trainer_state": restored_state,
                    "max_abs_logit_error": float(
                        (logits_before - logits_after).abs().max().item()
                    ),
                    "exact_logits": bool(torch.equal(logits_before, logits_after)),
                    "optimizer_state_loaded": bool(
                        reloaded.policy_optimizer.state_dict()["state"]
                    ),
                }
                if not reload_proof["exact_logits"]:
                    raise RuntimeError(f"Checkpoint reload changed inference: {reload_proof}")
                orchestrator.ppo_learner = reloaded
                orchestrator.agent.policy = reloaded.policy

        actor_manifest = save_actor_checkpoint(
            actor_checkpoint_path,
            orchestrator.ppo_learner.policy.actor,
            {
                "source": "milestone05_bounded_ppo_smoke",
                "cumulative_model_updates": orchestrator.ppo_learner.cumulative_model_updates,
                "cumulative_agent_steps": orchestrator.agent.cumulative_timesteps,
            },
        )
        action_head = orchestrator.ppo_learner.policy.actor.policy[-1]
        appended_weight_norm = float(
            action_head.weight[90:].detach().norm().cpu().item()
        )
        appended_bias_mean = float(
            action_head.bias[90:].detach().mean().cpu().item()
        )
        resume_passed = iteration_reports[-1]["Policy Update Magnitude"] > 0
        if not resume_passed:
            raise RuntimeError("Reloaded PPO learner did not perform a policy update")
        return {
            "schema_version": 1,
            "status": "passed",
            "rlgym_ppo_integration": {
                "rollout": "rlgym_ppo.batched_agents.BatchedAgentManager via Learner",
                "update": "rlgym_ppo.ppo.PPOLearner",
                "experience": "rlgym_ppo.ppo.ExperienceBuffer and Learner.add_new_experience",
            },
            "worker_count": worker_count,
            "device": device,
            "iterations_before_reload": iterations_before_reload,
            "iterations_after_reload": iterations_after_reload,
            "agent_steps_per_iteration_target": agent_steps_per_iteration,
            "batch_size": batch_size,
            "minibatch_size": minibatch_size,
            "epochs": epochs,
            "iteration_reports": iteration_reports,
            "checkpoint_manifest": checkpoint_manifest,
            "reload_proof": reload_proof,
            "resume_proof": {
                "post_reload_iterations": iterations_after_reload,
                "post_reload_update_magnitude": iteration_reports[-1][
                    "Policy Update Magnitude"
                ],
                "passed": resume_passed,
            },
            "expanded_head_after_smoke": {
                "appended_weight_norm": appended_weight_norm,
                "appended_bias_mean": appended_bias_mean,
                "sampled_action_count": int(overall_action_counts.sum()),
                "appended_action_count": int(overall_action_counts[90:].sum()),
                "appended_action_rate": float(
                    overall_action_counts[90:].sum()
                    / max(overall_action_counts.sum(), 1)
                ),
                "full_action_counts": overall_action_counts.tolist(),
            },
            "actor_checkpoint": actor_manifest,
        }
    finally:
        orchestrator.cleanup()
