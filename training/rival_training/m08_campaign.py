"""Bounded, resumable PPO orchestration for the M08 mechanics branch only."""

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

from .config import (
    REPOSITORY_ROOT,
    canonical_config_sha256,
    load_milestone08_config,
)
from .environment import make_dual_rate_gym_env
from .m08_metrics import (
    M08MetricsCollector,
    aggregate_m08_metrics,
    mechanics_action_report,
)
from .mechanics import load_mechanics_actor, mechanics_state_sha256
from .policy import MechanicsDiscretePolicy, attach_mechanics_policy
from .teacher import sha256_file


M08_STATE_FILE = "RIVAL_M08_STATE.json"
M08_CHECKPOINT_FORMAT = "rival-milestone08-full-ppo-v1"


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return True


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def frozen_strategic_proof(config: dict[str, Any]) -> dict[str, Any]:
    paths = {
        "wisp_policy": REPOSITORY_ROOT / "bot/models/POLICY.lt",
        "wisp_shared_head": REPOSITORY_ROOT / "bot/models/SHARED_HEAD.lt",
        "zero_step_torchscript": (
            REPOSITORY_ROOT
            / "training/artifacts/milestone07/zero_step_actor.ts"
        ),
    }
    expected = {
        "wisp_policy": config["frozen_fingerprints"]["wisp_policy_sha256"],
        "wisp_shared_head": config["frozen_fingerprints"]["wisp_shared_head_sha256"],
        "zero_step_torchscript": config["frozen_fingerprints"][
            "zero_step_torchscript_sha256"
        ],
    }
    artifacts = {}
    for name, path in paths.items():
        actual = sha256_file(path)
        artifacts[name] = {
            "path": _portable(path),
            "sha256": actual,
            "expected_sha256": expected[name],
            "unchanged": actual == expected[name],
        }
    return {
        "artifacts": artifacts,
        "all_unchanged": all(item["unchanged"] for item in artifacts.values()),
        "optimizer_contains_strategic_parameters": False,
        "architecture_reason": (
            "The strategic actor is instantiated only inside environment workers; "
            "the central PPO policy contains MechanicsActor parameters exclusively."
        ),
    }


def _initial_actor_path() -> Path:
    return REPOSITORY_ROOT / "training/artifacts/milestone08/mechanics_initial_v1.pt"


def _initial_actor(config: dict[str, Any], device: str) -> MechanicsDiscretePolicy:
    report_path = (
        REPOSITORY_ROOT
        / "training/results/milestone08/mechanics_prior_calibration.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("M08 mechanics-prior calibration has not passed")
    actor_path = _initial_actor_path()
    if sha256_file(actor_path) != report["checkpoint"]["sha256"]:
        raise RuntimeError("M08 initial mechanics checkpoint hash mismatch")
    actor, _ = load_mechanics_actor(actor_path, device="cpu")
    return MechanicsDiscretePolicy(actor, device)


def make_m08_ppo(config: dict[str, Any], *, device: str) -> PPOLearner:
    ppo = config["ppo"]
    learner = PPOLearner(
        obs_space_size=432,
        act_space_size=69,
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
    attach_mechanics_policy(
        learner,
        _initial_actor(config, device),
        float(ppo["policy_learning_rate"]),
    )
    return learner


def _build_orchestrator(
    config: dict[str, Any],
    *,
    worker_count: int,
    device: str,
) -> Learner:
    ppo = config["ppo"]
    learner = Learner(
        env_create_function=make_dual_rate_gym_env,
        metrics_logger=M08MetricsCollector(),
        n_proc=int(worker_count),
        min_inference_size=min(int(worker_count), 16),
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
            REPOSITORY_ROOT / "training/checkpoints/milestone08/package_unused"
        ),
        add_unix_timestamp=False,
        random_seed=int(config["seeds"]["training"]),
        n_checkpoints_to_keep=1000,
        device=device,
    )
    policy = _initial_actor(config, device)
    attach_mechanics_policy(
        learner.ppo_learner,
        policy,
        float(ppo["policy_learning_rate"]),
    )
    learner.agent.policy = policy
    return learner


def _checkpoint_files(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def load_m08_state(directory: str | Path) -> dict[str, Any]:
    source = Path(directory)
    state = json.loads((source / M08_STATE_FILE).read_text(encoding="utf-8"))
    if state.get("format") != M08_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported M08 checkpoint: {state.get('format')}")
    return state


def verify_m08_checkpoint(
    directory: str | Path,
    source_ppo: PPOLearner,
    config: dict[str, Any],
    *,
    require_optimizer_state: bool,
) -> dict[str, Any]:
    source = Path(directory)
    state = load_m08_state(source)
    if state["config_sha256"] != canonical_config_sha256(config):
        raise RuntimeError("M08 checkpoint config hash mismatch")
    reloaded = make_m08_ppo(config, device=str(source_ppo.device))
    sample = torch.randn(
        64,
        432,
        generator=torch.Generator(device="cpu").manual_seed(20260903),
    ).to(source_ppo.device)
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
        "state_file_parse_passed": True,
        "mechanics_state_sha256": mechanics_state_sha256(
            source_ppo.policy.actor.to("cpu")
        ),
    }
    source_ppo.policy.actor.to(source_ppo.device)
    if not proof["exact_logits"]:
        raise RuntimeError(f"M08 checkpoint reload changed logits: {proof}")
    if require_optimizer_state and not (optimizer_loaded and critic_optimizer_loaded):
        raise RuntimeError(f"M08 checkpoint lost optimizer state: {proof}")
    return proof


def save_m08_checkpoint(
    ppo: PPOLearner,
    trainer_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    steps = int(trainer_state["cumulative_agent_steps"])
    destination = (
        REPOSITORY_ROOT / "training/checkpoints/milestone08" / f"{steps:09d}"
    )
    destination.mkdir(parents=True, exist_ok=True)
    ppo.save_to(str(destination))
    state = {
        "format": M08_CHECKPOINT_FORMAT,
        "schema_version": 1,
        "config_sha256": canonical_config_sha256(config),
        **trainer_state,
    }
    (destination / M08_STATE_FILE).write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )
    reload = verify_m08_checkpoint(
        destination,
        ppo,
        config,
        require_optimizer_state=int(trainer_state["cumulative_model_updates"]) > 0,
    )
    return {
        "directory": _portable(destination),
        "files": _checkpoint_files(destination),
        "trainer_state": state,
        "fresh_reload_proof": reload,
    }


def _load_full_checkpoint(
    directory: Path,
    orchestrator: Learner,
    config: dict[str, Any],
) -> dict[str, Any]:
    state = load_m08_state(directory)
    if state["config_sha256"] != canonical_config_sha256(config):
        raise RuntimeError("Refusing M08 resume with a different serialized config")
    orchestrator.ppo_learner.load_from(str(directory))
    orchestrator.ppo_learner.cumulative_model_updates = int(
        state["cumulative_model_updates"]
    )
    orchestrator.agent.cumulative_timesteps = int(state["cumulative_agent_steps"])
    orchestrator.agent.average_reward = state.get("policy_average_reward")
    orchestrator.epoch = int(state["completed_iterations"])
    orchestrator.agent.policy = orchestrator.ppo_learner.policy
    return state


def _optimizer_ownership(ppo: PPOLearner) -> dict[str, Any]:
    policy_ids = {id(parameter) for parameter in ppo.policy.parameters()}
    optimizer_ids = {
        id(parameter)
        for group in ppo.policy_optimizer.param_groups
        for parameter in group["params"]
    }
    names = [name for name, _ in ppo.policy.named_parameters()]
    return {
        "policy_parameter_names": names,
        "policy_parameter_count": sum(
            parameter.numel() for parameter in ppo.policy.parameters()
        ),
        "optimizer_exactly_matches_policy_parameters": policy_ids == optimizer_ids,
        "strategic_named_parameter_present": any(
            "strategic" in name.lower() or "wisp" in name.lower() for name in names
        ),
    }


def run_m08_training_boundary(
    target_agent_steps: int,
    *,
    resume_directory: str | Path | None = None,
    worker_count: int | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    config = load_milestone08_config()
    target = int(target_agent_steps)
    if target not in config["boundaries_agent_steps"]:
        raise ValueError("M08 target must be one of 500k, 1M, 2M or 5M")
    workers = int(worker_count or config["environment"]["workers"])
    if workers not in config["environment"]["sanity_worker_candidates"]:
        raise ValueError("M08 worker count must come from the 48/56/64 sanity set")
    strategic_before = frozen_strategic_proof(config)
    if not strategic_before["all_unchanged"]:
        raise RuntimeError("Frozen strategic fingerprint failed before M08 PPO")
    orchestrator = _build_orchestrator(config, worker_count=workers, device=device)
    raw_directory = REPOSITORY_ROOT / "training/results/raw/milestone08"
    raw_directory.mkdir(parents=True, exist_ok=True)
    raw_log = raw_directory / "iterations.jsonl"
    checkpoint_manifests: list[dict[str, Any]] = []
    iteration_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    source_checkpoint = None
    try:
        if resume_directory is None:
            initial_state = {
                "campaign_id": config["campaign_id"],
                "completed_iterations": 0,
                "cumulative_agent_steps": 0,
                "cumulative_model_updates": 0,
                "worker_count": workers,
                "policy_average_reward": None,
                "source_checkpoint": None,
                "mechanics_prior": orchestrator.ppo_learner.policy.prior_state(),
            }
            checkpoint_manifests.append(
                save_m08_checkpoint(orchestrator.ppo_learner, initial_state, config)
            )
        else:
            source = Path(resume_directory).resolve()
            source_checkpoint = _portable(source)
            restored = _load_full_checkpoint(source, orchestrator, config)
            if int(restored["cumulative_agent_steps"]) >= target:
                raise ValueError("Resume checkpoint already reached requested boundary")
            if int(restored["worker_count"]) != workers:
                raise ValueError("Worker count may change only through prospective evidence")

        ownership = _optimizer_ownership(orchestrator.ppo_learner)
        if not ownership["optimizer_exactly_matches_policy_parameters"] or ownership[
            "strategic_named_parameter_present"
        ]:
            raise RuntimeError(f"M08 optimizer ownership failed: {ownership}")
        cumulative = int(orchestrator.agent.cumulative_timesteps)
        completed_iterations = int(orchestrator.epoch)
        tolerance = 4 * workers
        while cumulative < target - tolerance:
            remaining = target - cumulative
            collect_target = min(
                int(config["ppo"]["agent_steps_per_iteration"]), remaining
            )
            if remaining <= int(config["ppo"]["agent_steps_per_iteration"]):
                collect_target = max(1, collect_target - tolerance)
            iteration_started = time.perf_counter()
            experience, metrics, collected_steps, collection_seconds = (
                orchestrator.agent.collect_timesteps(collect_target)
            )
            observations = np.asarray(experience[0], dtype=np.float32)
            actions = np.asarray(experience[1])
            rewards = np.asarray(experience[3], dtype=np.float64).reshape(-1)
            sample = observations[: min(4096, len(observations))]
            with torch.inference_mode():
                probabilities = (
                    orchestrator.ppo_learner.policy.get_output(sample)
                    .detach()
                    .cpu()
                    .numpy()
                )
            action_report = mechanics_action_report(actions, probabilities)
            metric_report = aggregate_m08_metrics(metrics)
            orchestrator.add_new_experience(experience)
            buffer_records = int(orchestrator.experience_buffer.rewards.shape[0])
            if buffer_records < int(config["ppo"]["batch_size"]):
                raise RuntimeError("M08 PPO experience buffer is below its batch size")
            ppo_report = orchestrator.ppo_learner.learn(orchestrator.experience_buffer)
            completed_iterations += 1
            cumulative = int(orchestrator.agent.cumulative_timesteps)
            iteration = {
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
                "iteration_wall_seconds": float(
                    time.perf_counter() - iteration_started
                ),
                "actions": action_report,
                "rollout_metrics": metric_report,
                "ppo": ppo_report,
            }
            iteration["health"] = {
                "all_metrics_finite": _finite_tree(iteration),
                "policy_updated": float(ppo_report["Policy Update Magnitude"]) > 0.0,
                "sampled_override_bounded": action_report["sampled_override_rate"]
                <= float(config["evaluation"]["maximum_sampled_override_share"]),
            }
            iteration["health"]["passed"] = all(iteration["health"].values())
            if not iteration["health"]["passed"]:
                raise RuntimeError(f"M08 iteration health gate failed: {iteration}")
            iteration_reports.append(iteration)
            with raw_log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(iteration, separators=(",", ":")) + "\n")
            print(
                json.dumps(
                    {
                        "iteration": completed_iterations,
                        "steps": cumulative,
                        "steps_per_second": iteration["agent_steps_per_second"],
                        "override_share": action_report["sampled_override_rate"],
                        "entropy": ppo_report["Policy Entropy"],
                        "policy_update": ppo_report["Policy Update Magnitude"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if cumulative > int(config["campaign_ceiling_agent_steps"]):
            raise RuntimeError("M08 collection exceeded the authorized 5M ceiling")
        trainer_state = {
            "campaign_id": config["campaign_id"],
            "completed_iterations": completed_iterations,
            "cumulative_agent_steps": cumulative,
            "requested_boundary_agent_steps": target,
            "cumulative_model_updates": int(
                orchestrator.ppo_learner.cumulative_model_updates
            ),
            "worker_count": workers,
            "policy_average_reward": float(iteration_reports[-1]["mean_reward"]),
            "source_checkpoint": source_checkpoint,
            "mechanics_prior": orchestrator.ppo_learner.policy.prior_state(),
        }
        checkpoint_manifests.append(
            save_m08_checkpoint(orchestrator.ppo_learner, trainer_state, config)
        )
        strategic_after = frozen_strategic_proof(config)
        aggregate_counts = np.sum(
            [
                np.asarray(item["actions"]["full_mechanics_action_counts"])
                for item in iteration_reports
            ],
            axis=0,
            dtype=np.int64,
        )
        total = int(aggregate_counts.sum())
        summary = {
            "schema_version": 1,
            "status": "completed_boundary",
            "requested_boundary_agent_steps": target,
            "cumulative_agent_steps": cumulative,
            "within_authorized_ceiling": cumulative
            <= int(config["campaign_ceiling_agent_steps"]),
            "cumulative_model_updates": int(
                orchestrator.ppo_learner.cumulative_model_updates
            ),
            "iterations_this_invocation": len(iteration_reports),
            "worker_count": workers,
            "config_sha256": canonical_config_sha256(config),
            "optimizer_ownership": ownership,
            "strategic_branch_before": strategic_before,
            "strategic_branch_after": strategic_after,
            "strategic_unchanged": strategic_before == strategic_after,
            "aggregate_actions": {
                "sampled_action_count": total,
                "pass_count": int(aggregate_counts[0]),
                "override_count": int(aggregate_counts[1:].sum()),
                "sampled_override_rate": float(
                    aggregate_counts[1:].sum() / max(total, 1)
                ),
                "full_mechanics_action_counts": aggregate_counts.tolist(),
            },
            "action_probability_history": [
                {
                    "cumulative_agent_steps": item["cumulative_agent_steps"],
                    "mean_pass_probability": item["actions"][
                        "mean_pass_probability"
                    ],
                    "sampled_override_rate": item["actions"][
                        "sampled_override_rate"
                    ],
                    "deterministic_override_rate": item["actions"][
                        "deterministic_override_rate"
                    ],
                }
                for item in iteration_reports
            ],
            "ppo_history": [
                {
                    "cumulative_agent_steps": item["cumulative_agent_steps"],
                    "policy_entropy": item["ppo"]["Policy Entropy"],
                    "mean_kl_divergence": item["ppo"]["Mean KL Divergence"],
                    "value_function_loss": item["ppo"]["Value Function Loss"],
                    "policy_update_magnitude": item["ppo"][
                        "Policy Update Magnitude"
                    ],
                    "collection_seconds": item["collection_seconds"],
                    "iteration_wall_seconds": item["iteration_wall_seconds"],
                    "agent_steps_per_second": item["agent_steps_per_second"],
                }
                for item in iteration_reports
            ],
            "latest_rollout_metrics": iteration_reports[-1]["rollout_metrics"],
            "latest_iteration": iteration_reports[-1],
            "checkpoint_manifests": checkpoint_manifests,
            "latest_checkpoint": checkpoint_manifests[-1],
            "raw_iteration_log": _portable(raw_log),
            "wall_seconds": time.perf_counter() - started,
            "production_promoted": False,
            "health": {
                "all_iterations_passed": True,
                "strategic_unchanged": strategic_before == strategic_after,
                "checkpoint_reload_exact": checkpoint_manifests[-1][
                    "fresh_reload_proof"
                ]["exact_logits"],
                "passed": True,
            },
        }
        output = (
            REPOSITORY_ROOT
            / "training/results/milestone08"
            / f"training_{target:09d}.json"
        )
        output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary
    finally:
        orchestrator.cleanup()
