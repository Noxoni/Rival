"""Resumable staged Milestone 06 PPO campaign orchestration."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from rlgym_ppo import Learner
from rlgym_ppo.ppo import PPOLearner

from .checkpoint import load_actor_checkpoint, portable_path
from .config import (
    REPOSITORY_ROOT,
    canonical_config_sha256,
    load_milestone06_config,
    stage_config,
)
from .environment import campaign_environment_factory
from .metrics import (
    CampaignMetricsCollector,
    aggregate_campaign_metrics,
    merge_campaign_metric_reports,
)
from .policy import (
    StudentDiscretePolicy,
    attach_student_policy,
    normalize_bootstrap_actor_for_prior,
)
from .teacher import EXPANDED_ACTION_COUNT, sha256_file


CAMPAIGN_CHECKPOINT_STATE = "RIVAL_CAMPAIGN_STATE.json"
CAMPAIGN_CHECKPOINT_FORMAT = "rival-milestone06-full-ppo-v1"


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return True


def _bootstrap_policy(
    config: dict[str, Any],
    *,
    device: str,
    appended_logit_offset: float,
) -> StudentDiscretePolicy:
    path = REPOSITORY_ROOT / "training/artifacts/bootstrap/wisp_student_expanded_v1.pt"
    expected = config["frozen_fingerprints"]["bootstrap_actor_sha256"]
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"Bootstrap actor hash mismatch: expected {expected}, got {actual}")
    actor, _ = load_actor_checkpoint(path, "cpu")
    normalize_bootstrap_actor_for_prior(
        actor,
        expected_bootstrap_bias=float(
            config["action_exploration"]["bootstrap_appended_bias"]
        ),
    )
    return StudentDiscretePolicy(
        actor,
        device,
        appended_logit_offset=appended_logit_offset,
    )


def make_campaign_ppo(
    config: dict[str, Any],
    *,
    device: str,
    appended_logit_offset: float,
) -> PPOLearner:
    ppo = config["ppo"]
    learner = PPOLearner(
        obs_space_size=432,
        act_space_size=EXPANDED_ACTION_COUNT,
        policy_type=0,
        policy_layer_sizes=(64,),
        critic_layer_sizes=(256, 256),
        continuous_var_range=(0.1, 1.0),
        batch_size=int(ppo["batch_size"]),
        n_epochs=int(ppo["epochs"]),
        policy_lr=float(ppo["policy_learning_rate"]),
        critic_lr=float(ppo["critic_learning_rate"]),
        clip_range=float(ppo["clip_range"]),
        ent_coef=float(ppo["entropy_coefficient"]),
        mini_batch_size=int(ppo["minibatch_size"]),
        device=device,
    )
    attach_student_policy(
        learner,
        _bootstrap_policy(
            config,
            device=device,
            appended_logit_offset=appended_logit_offset,
        ),
        float(ppo["policy_learning_rate"]),
    )
    return learner


def _checkpoint_files(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def load_campaign_state(directory: str | Path) -> dict[str, Any]:
    source = Path(directory)
    state = json.loads((source / CAMPAIGN_CHECKPOINT_STATE).read_text(encoding="utf-8"))
    if state.get("format") != CAMPAIGN_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported campaign checkpoint: {state.get('format')}")
    return state


def verify_campaign_checkpoint(
    directory: str | Path,
    source_ppo: PPOLearner,
    config: dict[str, Any],
    *,
    require_optimizer_state: bool,
) -> dict[str, Any]:
    source = Path(directory)
    state = load_campaign_state(source)
    if state["config_sha256"] != canonical_config_sha256(config):
        raise RuntimeError("Campaign checkpoint config hash mismatch")
    reloaded = make_campaign_ppo(
        config,
        device=source_ppo.device,
        appended_logit_offset=float(
            state["action_exploration_prior"]["appended_logit_offset"]
        ),
    )
    sample_generator = torch.Generator(device="cpu").manual_seed(20260828)
    sample = torch.randn(64, 432, generator=sample_generator).to(source_ppo.device)
    with torch.inference_mode():
        before = source_ppo.policy.logits(sample).detach().cpu()
    reloaded.load_from(str(source))
    with torch.inference_mode():
        after = reloaded.policy.logits(sample).detach().cpu()
    optimizer_loaded = bool(reloaded.policy_optimizer.state_dict()["state"])
    critic_optimizer_loaded = bool(reloaded.value_optimizer.state_dict()["state"])
    proof = {
        "fresh_instance": True,
        "exact_logits": bool(torch.equal(before, after)),
        "max_abs_logit_error": float((before - after).abs().max().item()),
        "policy_optimizer_state_loaded": optimizer_loaded,
        "critic_optimizer_state_loaded": critic_optimizer_loaded,
        "prior_state_reloaded": reloaded.policy.prior_state(),
        "state_file_parse_passed": True,
    }
    if not proof["exact_logits"]:
        raise RuntimeError(f"Campaign checkpoint reload changed logits: {proof}")
    if require_optimizer_state and not (optimizer_loaded and critic_optimizer_loaded):
        raise RuntimeError(f"Campaign checkpoint lost optimizer state: {proof}")
    return proof


def save_campaign_checkpoint(
    ppo_learner: PPOLearner,
    trainer_state: dict[str, Any],
    config: dict[str, Any],
    *,
    checkpoint_root: str | Path | None = None,
) -> dict[str, Any]:
    steps = int(trainer_state["cumulative_agent_steps"])
    stage = str(trainer_state["stage"])
    offset = float(
        trainer_state["action_exploration_prior"]["appended_logit_offset"]
    )
    offset_tag = f"{offset:+.1f}".replace("+", "p").replace("-", "m").replace(".", "p")
    root = (
        REPOSITORY_ROOT / "training/checkpoints/milestone06"
        if checkpoint_root is None
        else Path(checkpoint_root)
    )
    destination = root / f"{steps:09d}_{stage}_{offset_tag}"
    destination.mkdir(parents=True, exist_ok=True)
    ppo_learner.save_to(str(destination))
    state = {
        "format": CAMPAIGN_CHECKPOINT_FORMAT,
        "schema_version": 1,
        "config_sha256": canonical_config_sha256(config),
        **trainer_state,
    }
    state_path = destination / CAMPAIGN_CHECKPOINT_STATE
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    reload_proof = verify_campaign_checkpoint(
        destination,
        ppo_learner,
        config,
        require_optimizer_state=int(trainer_state["cumulative_model_updates"]) > 0,
    )
    return {
        "directory": portable_path(destination),
        "files": _checkpoint_files(destination),
        "trainer_state": state,
        "fresh_reload_proof": reload_proof,
    }


def _load_full_checkpoint(
    directory: Path,
    orchestrator: Learner,
    config: dict[str, Any],
) -> dict[str, Any]:
    state = load_campaign_state(directory)
    if state["config_sha256"] != canonical_config_sha256(config):
        raise RuntimeError("Refusing resume from a different serialized M06 config")
    orchestrator.ppo_learner.load_from(str(directory))
    orchestrator.ppo_learner.cumulative_model_updates = int(
        state["cumulative_model_updates"]
    )
    orchestrator.agent.cumulative_timesteps = int(state["cumulative_agent_steps"])
    orchestrator.agent.average_reward = state.get("policy_average_reward")
    orchestrator.epoch = int(state["completed_iterations"])
    orchestrator.agent.policy = orchestrator.ppo_learner.policy
    return state


def _build_orchestrator(
    config: dict[str, Any],
    stage_name: str,
    *,
    worker_count: int,
    device: str,
    appended_logit_offset: float,
) -> Learner:
    ppo = config["ppo"]
    learner = Learner(
        env_create_function=campaign_environment_factory(stage_name),
        metrics_logger=CampaignMetricsCollector(),
        n_proc=worker_count,
        min_inference_size=min(worker_count, 16),
        render=False,
        timestep_limit=int(config["campaign_ceiling_agent_steps"]),
        exp_buffer_size=int(ppo["experience_buffer_size"]),
        ts_per_iteration=int(ppo["agent_steps_per_iteration"]),
        standardize_returns=bool(ppo["standardize_returns"]),
        standardize_obs=bool(ppo["standardize_observations"]),
        policy_layer_sizes=(64,),
        critic_layer_sizes=(256, 256),
        ppo_epochs=int(ppo["epochs"]),
        ppo_batch_size=int(ppo["batch_size"]),
        ppo_minibatch_size=int(ppo["minibatch_size"]),
        ppo_ent_coef=float(ppo["entropy_coefficient"]),
        ppo_clip_range=float(ppo["clip_range"]),
        gae_lambda=float(ppo["gae_lambda"]),
        gae_gamma=float(ppo["gamma"]),
        policy_lr=float(ppo["policy_learning_rate"]),
        critic_lr=float(ppo["critic_learning_rate"]),
        log_to_wandb=False,
        checkpoint_load_folder=None,
        checkpoints_save_folder=str(
            REPOSITORY_ROOT / "training/checkpoints/milestone06/package_unused"
        ),
        add_unix_timestamp=False,
        random_seed=int(config["seeds"]["training"]),
        n_checkpoints_to_keep=1000,
        device=device,
    )
    policy = _bootstrap_policy(
        config,
        device=device,
        appended_logit_offset=appended_logit_offset,
    )
    attach_student_policy(
        learner.ppo_learner,
        policy,
        float(ppo["policy_learning_rate"]),
    )
    learner.agent.policy = policy
    return learner


def _action_report(
    actions: np.ndarray,
    policy: StudentDiscretePolicy,
    observations: np.ndarray,
) -> dict[str, Any]:
    action_indices = np.asarray(actions, dtype=np.int64).reshape(-1)
    counts = np.bincount(action_indices, minlength=EXPANDED_ACTION_COUNT)[
        :EXPANDED_ACTION_COUNT
    ]
    sample = observations[: min(len(observations), 4096)]
    with torch.inference_mode():
        probabilities = policy.get_output(sample)
        appended_mass = probabilities[:, 90:].sum(dim=-1)
    return {
        "sampled_action_count": int(counts.sum()),
        "legacy_action_count": int(counts[:90].sum()),
        "appended_action_count": int(counts[90:].sum()),
        "appended_action_share": float(counts[90:].sum() / max(counts.sum(), 1)),
        "mean_appended_probability_mass": float(appended_mass.mean().item()),
        "full_action_counts": counts.tolist(),
        "top_action_counts": [
            {"action_index": int(index), "count": int(counts[index])}
            for index in np.argsort(counts)[-12:][::-1]
            if counts[index] > 0
        ],
    }


def run_campaign_stage(
    stage_name: str,
    *,
    appended_logit_offset: float,
    resume_directory: str | Path | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    config = load_milestone06_config()
    stage = stage_config(config, stage_name)
    throughput_path = REPOSITORY_ROOT / "training/results/milestone06/throughput_sweep.json"
    throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
    worker_count = int(throughput["selected_worker_count"])
    configured_workers = int(config["environment"]["workers"])
    if worker_count != configured_workers:
        raise RuntimeError(
            f"Measured worker optimum {worker_count} != serialized config {configured_workers}"
        )
    orchestrator = _build_orchestrator(
        config,
        stage_name,
        worker_count=worker_count,
        device=device,
        appended_logit_offset=appended_logit_offset,
    )
    raw_directory = REPOSITORY_ROOT / "training/results/raw/milestone06"
    raw_directory.mkdir(parents=True, exist_ok=True)
    raw_log = raw_directory / f"{stage_name}_iterations.jsonl"
    prior_history: list[dict[str, Any]] = []
    checkpoint_manifests: list[dict[str, Any]] = []
    iteration_reports: list[dict[str, Any]] = []
    evaluation_reports: list[dict[str, Any]] = []
    source_checkpoint = None
    campaign_health_passed = True
    started = time.perf_counter()
    try:
        if resume_directory is None:
            if int(stage["start_agent_steps"]) != 0:
                raise ValueError(f"{stage_name} requires a resume checkpoint")
            prior_history.append(
                {
                    "cumulative_agent_steps": 0,
                    "stage": stage_name,
                    "from_bootstrap_effective_bias": float(
                        config["action_exploration"]["bootstrap_appended_bias"]
                    ),
                    "appended_logit_offset": float(appended_logit_offset),
                    "reason": "measured natural-observation calibration",
                }
            )
            initial_state = {
                "campaign_id": config["campaign_id"],
                "stage": stage_name,
                "completed_iterations": 0,
                "cumulative_agent_steps": 0,
                "cumulative_model_updates": 0,
                "worker_count": worker_count,
                "policy_average_reward": None,
                "action_exploration_prior": orchestrator.ppo_learner.policy.prior_state(),
                "prior_history": prior_history,
                "source_checkpoint": None,
            }
            checkpoint_manifests.append(
                save_campaign_checkpoint(orchestrator.ppo_learner, initial_state, config)
            )
        else:
            source = Path(resume_directory).resolve()
            source_checkpoint = portable_path(source)
            restored = _load_full_checkpoint(source, orchestrator, config)
            prior_history = list(restored.get("prior_history") or [])
            previous_stage = str(restored["stage"])
            previous_offset = float(
                restored["action_exploration_prior"]["appended_logit_offset"]
            )
            if previous_stage == stage_name and not math.isclose(
                previous_offset, appended_logit_offset, abs_tol=1e-12
            ):
                raise ValueError("Appended prior may not change within a stage")
            if previous_stage != stage_name:
                if int(restored["cumulative_agent_steps"]) < int(
                    stage["start_agent_steps"]
                ):
                    raise ValueError("Resume checkpoint has not reached the stage boundary")
                orchestrator.experience_buffer.clear()
                orchestrator.ppo_learner.policy.set_appended_logit_offset(
                    appended_logit_offset
                )
                prior_history.append(
                    {
                        "cumulative_agent_steps": int(
                            restored["cumulative_agent_steps"]
                        ),
                        "stage": stage_name,
                        "previous_stage": previous_stage,
                        "previous_appended_logit_offset": previous_offset,
                        "appended_logit_offset": float(appended_logit_offset),
                        "reason": "explicit healthy stage-boundary relaxation decision",
                    }
                )

        cumulative = int(orchestrator.agent.cumulative_timesteps)
        end_steps = int(stage["end_agent_steps"])
        if cumulative >= end_steps:
            raise ValueError(f"Checkpoint already reached {stage_name} end boundary")
        interval = int(config["checkpointing"]["interval_agent_steps"])
        evaluation_interval = int(
            config["checkpointing"]["evaluation_interval_agent_steps"]
        )
        next_checkpoint = (cumulative // interval + 1) * interval
        next_evaluation = (cumulative // evaluation_interval + 1) * evaluation_interval
        completed_iterations = int(orchestrator.epoch)
        while cumulative < end_steps:
            remaining = end_steps - cumulative
            target = min(int(config["ppo"]["agent_steps_per_iteration"]), remaining)
            if (
                end_steps == int(config["campaign_ceiling_agent_steps"])
                and remaining <= int(config["ppo"]["agent_steps_per_iteration"])
            ):
                target = max(1, target - 4 * worker_count)
            iteration_started = time.perf_counter()
            experience, collected_metrics, collected_steps, collection_seconds = (
                orchestrator.agent.collect_timesteps(target)
            )
            observations = np.asarray(experience[0], dtype=np.float32)
            actions = np.asarray(experience[1])
            rewards = np.asarray(experience[3], dtype=np.float64).reshape(-1)
            action_report = _action_report(
                actions,
                orchestrator.ppo_learner.policy,
                observations,
            )
            metric_report = aggregate_campaign_metrics(collected_metrics)
            orchestrator.add_new_experience(experience)
            buffer_records = int(orchestrator.experience_buffer.rewards.shape[0])
            if buffer_records < int(config["ppo"]["batch_size"]):
                raise RuntimeError(
                    f"PPO buffer has {buffer_records} records, below material batch size"
                )
            ppo_report = orchestrator.ppo_learner.learn(orchestrator.experience_buffer)
            completed_iterations += 1
            cumulative = int(orchestrator.agent.cumulative_timesteps)
            iteration_report = {
                "stage": stage_name,
                "iteration": completed_iterations,
                "cumulative_agent_steps": cumulative,
                "collected_agent_steps": int(collected_steps),
                "experience_records": int(len(observations)),
                "experience_buffer_records": buffer_records,
                "mean_reward": float(rewards.mean()),
                "minimum_reward": float(rewards.min()),
                "maximum_reward": float(rewards.max()),
                "collection_seconds": float(collection_seconds),
                "agent_steps_per_second": float(collected_steps / collection_seconds),
                "iteration_wall_seconds": float(time.perf_counter() - iteration_started),
                "action_exploration_prior": orchestrator.ppo_learner.policy.prior_state(),
                "actions": action_report,
                "rollout_metrics": metric_report,
                "ppo": ppo_report,
            }
            iteration_report["health"] = {
                "all_metrics_finite": _finite_tree(iteration_report),
                "policy_updated": float(ppo_report["Policy Update Magnitude"]) > 0.0,
                "appended_share_below_rejection_gate": (
                    action_report["appended_action_share"]
                    <= float(
                        config["evaluation"][
                            "maximum_appended_action_share_for_health"
                        ]
                    )
                ),
            }
            iteration_report["health"]["passed"] = all(
                iteration_report["health"].values()
            )
            if not iteration_report["health"]["passed"]:
                raise RuntimeError(f"Campaign health gate failed: {iteration_report}")
            iteration_reports.append(iteration_report)
            with raw_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(iteration_report, separators=(",", ":")) + "\n")
            print(
                json.dumps(
                    {
                        "stage": stage_name,
                        "iteration": completed_iterations,
                        "steps": cumulative,
                        "updates": ppo_report["Cumulative Model Updates"],
                        "steps_per_second": iteration_report["agent_steps_per_second"],
                        "appended_share": action_report["appended_action_share"],
                        "policy_update": ppo_report["Policy Update Magnitude"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if cumulative >= next_checkpoint or cumulative >= end_steps:
                trainer_state = {
                    "campaign_id": config["campaign_id"],
                    "stage": stage_name,
                    "completed_iterations": completed_iterations,
                    "cumulative_agent_steps": cumulative,
                    "cumulative_model_updates": int(
                        orchestrator.ppo_learner.cumulative_model_updates
                    ),
                    "worker_count": worker_count,
                    "policy_average_reward": float(rewards.mean()),
                    "action_exploration_prior": (
                        orchestrator.ppo_learner.policy.prior_state()
                    ),
                    "prior_history": prior_history,
                    "source_checkpoint": source_checkpoint,
                }
                checkpoint_manifests.append(
                    save_campaign_checkpoint(
                        orchestrator.ppo_learner,
                        trainer_state,
                        config,
                    )
                )
                while next_checkpoint <= cumulative:
                    next_checkpoint += interval
            if cumulative >= next_evaluation:
                from .evaluation import evaluate_frozen_wisp  # noqa: PLC0415

                baseline_path = (
                    REPOSITORY_ROOT
                    / "training/results/milestone06/headless_wisp_preflight.json"
                )
                baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
                evaluation = evaluate_frozen_wisp(
                    orchestrator.ppo_learner.policy,
                    {
                        "kind": "in_memory_policy_at_verified_checkpoint",
                        "checkpoint": checkpoint_manifests[-1],
                    },
                    games=int(config["evaluation"]["headless_games"]),
                    seed=int(config["seeds"]["evaluation"]),
                    device=device,
                    baseline=baseline,
                )
                evaluation_output = (
                    REPOSITORY_ROOT
                    / "training/results/milestone06"
                    / f"headless_wisp_{cumulative:09d}.json"
                )
                evaluation_output.write_text(
                    json.dumps(evaluation, indent=2) + "\n",
                    encoding="utf-8",
                )
                evaluation_reports.append(
                    {
                        "cumulative_agent_steps": cumulative,
                        "path": portable_path(evaluation_output),
                        "status": evaluation["status"],
                        "outcomes": evaluation["outcomes"],
                        "appended_action_share": evaluation["actions"][
                            "appended_action_share"
                        ],
                        "mechanics_recovery": evaluation["mechanics_recovery"],
                        "health": evaluation["health"],
                    }
                )
                print(
                    json.dumps(
                        {
                            "evaluation_boundary": cumulative,
                            "wins": evaluation["outcomes"]["wins"],
                            "losses": evaluation["outcomes"]["losses"],
                            "ties": evaluation["outcomes"]["ties"],
                            "goal_differential": evaluation["outcomes"][
                                "goal_differential"
                            ],
                            "appended_share": evaluation["actions"][
                                "appended_action_share"
                            ],
                            "health_passed": evaluation["health"]["passed"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                while next_evaluation <= cumulative:
                    next_evaluation += evaluation_interval
                if not evaluation["health"]["passed"]:
                    campaign_health_passed = False
                    break
            if cumulative >= end_steps:
                break

        merged_metrics = merge_campaign_metric_reports(
            [item["rollout_metrics"] for item in iteration_reports]
        )
        aggregate_action_counts = np.sum(
            [np.asarray(item["actions"]["full_action_counts"]) for item in iteration_reports],
            axis=0,
            dtype=np.int64,
        )
        total_actions = int(aggregate_action_counts.sum())
        action_share_history = [
            {
                "cumulative_agent_steps": item["cumulative_agent_steps"],
                "sampled_appended_action_share": item["actions"][
                    "appended_action_share"
                ],
                "mean_appended_probability_mass": item["actions"][
                    "mean_appended_probability_mass"
                ],
            }
            for item in iteration_reports
        ]
        ppo_history = [
            {
                "cumulative_agent_steps": item["cumulative_agent_steps"],
                "policy_entropy": item["ppo"]["Policy Entropy"],
                "mean_kl_divergence": item["ppo"]["Mean KL Divergence"],
                "value_function_loss": item["ppo"]["Value Function Loss"],
                "policy_update_magnitude": item["ppo"]["Policy Update Magnitude"],
                "iteration_wall_seconds": item["iteration_wall_seconds"],
            }
            for item in iteration_reports
        ]
        reached_stage_boundary = cumulative >= end_steps and campaign_health_passed
        summary = {
            "schema_version": 1,
            "status": (
                "completed_stage_boundary"
                if reached_stage_boundary
                else "rejected_at_evaluation_boundary"
            ),
            "stage": stage_name,
            "stage_target_end_agent_steps": end_steps,
            "cumulative_agent_steps": cumulative,
            "cumulative_model_updates": int(
                orchestrator.ppo_learner.cumulative_model_updates
            ),
            "iterations_this_invocation": len(iteration_reports),
            "worker_count": worker_count,
            "config_sha256": canonical_config_sha256(config),
            "action_exploration_prior": orchestrator.ppo_learner.policy.prior_state(),
            "prior_history": prior_history,
            "aggregate_rollout_metrics": merged_metrics,
            "aggregate_actions": {
                "sampled_action_count": total_actions,
                "legacy_action_count": int(aggregate_action_counts[:90].sum()),
                "appended_action_count": int(aggregate_action_counts[90:].sum()),
                "appended_action_share": float(
                    aggregate_action_counts[90:].sum() / max(total_actions, 1)
                ),
                "full_action_counts": aggregate_action_counts.tolist(),
            },
            "action_share_history": action_share_history,
            "ppo_history": ppo_history,
            "headless_evaluations": evaluation_reports,
            "health": {
                "all_iteration_gates_passed": True,
                "all_headless_gates_passed": campaign_health_passed,
                "passed": campaign_health_passed,
            },
            "latest_iteration": iteration_reports[-1],
            "checkpoint_manifests": checkpoint_manifests,
            "latest_checkpoint": checkpoint_manifests[-1],
            "raw_iteration_log": portable_path(raw_log),
            "wall_seconds": time.perf_counter() - started,
            "exact_same_stage_resume_command": (
                None
                if reached_stage_boundary or not campaign_health_passed
                else (
                    "training/.venv/Scripts/python.exe "
                    "training/scripts/run_m06_campaign.py "
                    f"--stage {stage_name} --appended-offset {appended_logit_offset} "
                    f"--resume {checkpoint_manifests[-1]['directory']}"
                )
            ),
            "next_stage_requires_explicit_health_gated_prior_decision": (
                reached_stage_boundary and stage_name != "stage_d"
            ),
        }
        output = raw_directory / f"{stage_name}_run_summary.json"
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        orchestrator.cleanup()
