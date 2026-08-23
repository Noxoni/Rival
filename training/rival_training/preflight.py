"""Measured serious-PPO and deployment preflight for Milestone 06."""

from __future__ import annotations

import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .campaign import (
    _action_report,
    _build_orchestrator,
    make_campaign_ppo,
    save_campaign_checkpoint,
)
from .checkpoint import load_actor_checkpoint, portable_path, save_actor_checkpoint
from .config import REPOSITORY_ROOT, canonical_config_sha256, load_milestone06_config
from .deploy import export_torchscript, make_exact_policy_export
from .metrics import aggregate_campaign_metrics
from .policy import materialize_effective_actor
from .teacher import sha256_file


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return math.isfinite(float(value))
    return True


def run_serious_ppo_preflight(*, device: str = "cuda:0") -> dict[str, Any]:
    """Run one real configured rollout/update and prove reload/export parity."""
    config = load_milestone06_config()
    config_hash = canonical_config_sha256(config)
    results_root = REPOSITORY_ROOT / "training/results/milestone06"
    throughput_path = results_root / "throughput_sweep.json"
    calibration_path = results_root / "action_prior_calibration.json"
    throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    workers = int(throughput["selected_worker_count"])
    offset = float(calibration["selected_appended_logit_offset"])
    if workers != int(config["environment"]["workers"]):
        raise RuntimeError("Serious preflight worker count does not match measured config")
    if calibration["config_sha256"] != config_hash:
        raise RuntimeError("Action-prior calibration was not run against this config")

    orchestrator = _build_orchestrator(
        config,
        "stage_a",
        worker_count=workers,
        device=device,
        appended_logit_offset=offset,
    )
    started = time.perf_counter()
    checkpoint_manifest: dict[str, Any] | None = None
    try:
        target = int(config["ppo"]["agent_steps_per_iteration"])
        collection_started = time.perf_counter()
        experience, collected_metrics, collected_steps, package_collection_seconds = (
            orchestrator.agent.collect_timesteps(target)
        )
        collection_wall_seconds = time.perf_counter() - collection_started
        observations = np.asarray(experience[0], dtype=np.float32)
        actions = np.asarray(experience[1])
        rewards = np.asarray(experience[3], dtype=np.float64).reshape(-1)
        action_report = _action_report(
            actions,
            orchestrator.ppo_learner.policy,
            observations,
        )
        rollout_metrics = aggregate_campaign_metrics(collected_metrics)
        orchestrator.add_new_experience(experience)
        buffer_records = int(orchestrator.experience_buffer.rewards.shape[0])
        if buffer_records < int(config["ppo"]["batch_size"]):
            raise RuntimeError("Real preflight did not fill the configured PPO batch")

        update_started = time.perf_counter()
        ppo_report = orchestrator.ppo_learner.learn(orchestrator.experience_buffer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ppo_update_wall_seconds = time.perf_counter() - update_started
        iteration_wall_seconds = time.perf_counter() - collection_started
        if float(ppo_report["Policy Update Magnitude"]) <= 0:
            raise RuntimeError("Serious preflight produced no actor update")

        trainer_state = {
            "campaign_id": config["campaign_id"],
            "stage": "preflight",
            "completed_iterations": 1,
            "cumulative_agent_steps": int(orchestrator.agent.cumulative_timesteps),
            "cumulative_model_updates": int(
                orchestrator.ppo_learner.cumulative_model_updates
            ),
            "worker_count": workers,
            "policy_average_reward": float(rewards.mean()),
            "action_exploration_prior": (
                orchestrator.ppo_learner.policy.prior_state()
            ),
            "prior_history": [
                {
                    "cumulative_agent_steps": 0,
                    "stage": "preflight",
                    "appended_logit_offset": offset,
                    "reason": "measured natural-observation calibration",
                }
            ],
            "source_checkpoint": None,
            "preflight_only_not_campaign_steps": True,
        }
        checkpoint_manifest = save_campaign_checkpoint(
            orchestrator.ppo_learner,
            trainer_state,
            config,
            checkpoint_root=(
                REPOSITORY_ROOT / "training/checkpoints/milestone06_preflight"
            ),
        )

        effective_actor = materialize_effective_actor(orchestrator.ppo_learner.policy)
        actor_path = (
            REPOSITORY_ROOT / "training/artifacts/milestone06/preflight_actor.pt"
        )
        actor_manifest = save_actor_checkpoint(
            actor_path,
            effective_actor,
            {
                "source": "milestone06_serious_ppo_preflight",
                "config_sha256": config_hash,
                "appended_prior_baked_into_actor": offset,
                "campaign_steps_counted": False,
            },
        )
        torchscript_path = (
            REPOSITORY_ROOT / "training/artifacts/milestone06/preflight_actor.ts"
        )
        cpu_ppo = make_campaign_ppo(
            config,
            device="cpu",
            appended_logit_offset=offset,
        )
        cpu_ppo.load_from(
            str(REPOSITORY_ROOT / checkpoint_manifest["directory"])
        )
        exact_export = make_exact_policy_export(cpu_ppo.policy)
        torchscript_manifest = export_torchscript(exact_export, torchscript_path)
        reloaded_actor, reloaded_metadata = load_actor_checkpoint(actor_path, "cpu")
        scripted = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
        sample = torch.randn(
            128,
            432,
            generator=torch.Generator(device="cpu").manual_seed(20260829),
        )
        with torch.inference_mode():
            effective_logits = effective_actor(sample)
            actor_logits = reloaded_actor(sample)
            policy_logits = cpu_ppo.policy.logits(sample)
            exact_export_logits = exact_export(sample)
            scripted_logits = scripted(sample)
        deployment_parity = {
            "sample_shape": list(sample.shape),
            "actor_checkpoint_exact": bool(torch.equal(effective_logits, actor_logits)),
            "checkpoint_policy_to_exact_export_exact": bool(
                torch.equal(policy_logits, exact_export_logits)
            ),
            "exact_export_to_torchscript_exact": bool(
                torch.equal(exact_export_logits, scripted_logits)
            ),
            "actor_checkpoint_max_abs_error": float(
                (effective_logits - actor_logits).abs().max().item()
            ),
            "checkpoint_policy_to_exact_export_max_abs_error": float(
                (policy_logits - exact_export_logits).abs().max().item()
            ),
            "exact_export_to_torchscript_max_abs_error": float(
                (exact_export_logits - scripted_logits).abs().max().item()
            ),
            "all_logits_finite": bool(torch.isfinite(scripted_logits).all().item()),
            "actor_metadata": reloaded_metadata,
        }
        deployment_parity["passed"] = all(
            deployment_parity[key]
            for key in (
                "actor_checkpoint_exact",
                "checkpoint_policy_to_exact_export_exact",
                "exact_export_to_torchscript_exact",
                "all_logits_finite",
            )
        )
        timing = {
            "worker_count": workers,
            "target_agent_steps": target,
            "collected_agent_steps": int(collected_steps),
            "experience_records": int(len(observations)),
            "experience_buffer_records": buffer_records,
            "package_collection_seconds": float(package_collection_seconds),
            "collection_wall_seconds": float(collection_wall_seconds),
            "ppo_update_wall_seconds": float(ppo_update_wall_seconds),
            "iteration_wall_seconds": float(iteration_wall_seconds),
            "agent_steps_per_collection_second": float(
                collected_steps / package_collection_seconds
            ),
            "agent_steps_per_iteration_wall_second": float(
                collected_steps / iteration_wall_seconds
            ),
        }
        report: dict[str, Any] = {
            "schema_version": 1,
            "status": "passed",
            "scope": "one configured serious PPO iteration; excluded from campaign budget",
            "config_sha256": config_hash,
            "worker_count": workers,
            "selected_appended_logit_offset": offset,
            "ppo_configuration": config["ppo"],
            "timing": timing,
            "reward_summary": {
                "count": int(rewards.size),
                "mean": float(rewards.mean()),
                "minimum": float(rewards.min()),
                "maximum": float(rewards.max()),
            },
            "actions": action_report,
            "rollout_metrics": rollout_metrics,
            "ppo": ppo_report,
            "checkpoint": checkpoint_manifest,
            "actor_checkpoint": actor_manifest,
            "torchscript_export": torchscript_manifest,
            "deployment_parity": deployment_parity,
            "all_metrics_finite": True,
            "wall_seconds": float(time.perf_counter() - started),
        }
        report["all_metrics_finite"] = _all_finite(report)
        if not report["all_metrics_finite"] or not deployment_parity["passed"]:
            report["status"] = "failed"
            raise RuntimeError(f"Serious PPO preflight health failure: {report}")

        output = results_root / "serious_ppo_preflight.json"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        throughput["measurement"]["includes_ppo_update"] = True
        throughput["ppo_iteration_timing"] = {
            **timing,
            "evidence": portable_path(output),
            "checkpoint_reload_passed": checkpoint_manifest[
                "fresh_reload_proof"
            ]["exact_logits"],
            "deployment_export_passed": deployment_parity["passed"],
        }
        throughput_path.write_text(
            json.dumps(throughput, indent=2) + "\n",
            encoding="utf-8",
        )
        return report
    finally:
        orchestrator.cleanup()


def verify_export_with_runtime(
    torchscript_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Load a candidate export in any Python runtime and run finite inference."""
    source = Path(torchscript_path)
    actual_hash = sha256_file(source)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise RuntimeError("Candidate TorchScript hash does not match expected value")
    module = torch.jit.load(str(source), map_location="cpu").eval()
    sample = torch.randn(
        64,
        432,
        generator=torch.Generator(device="cpu").manual_seed(20260830),
    )
    with torch.inference_mode():
        output = module(sample)
    report = {
        "path": portable_path(source),
        "sha256": actual_hash,
        "torch_version": torch.__version__,
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
    }
    if report["output_shape"] != [64, 158] or not report["finite"]:
        raise RuntimeError(f"Candidate export runtime verification failed: {report}")
    return report
