"""Measured calibration for the checkpointed appended-action logit prior."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rlgym_ppo.batched_agents import BatchedAgentManager

from .checkpoint import load_actor_checkpoint
from .environment import make_natural_campaign_gym_env
from .policy import StudentDiscretePolicy, normalize_bootstrap_actor_for_prior
from .teacher import TEACHER_ACTION_COUNT


def collect_natural_observations(
    bootstrap_actor_path: str | Path,
    *,
    worker_count: int,
    target_agent_observations: int,
    seed: int,
    device: str = "cuda:0",
) -> tuple[np.ndarray, dict[str, Any]]:
    actor, metadata = load_actor_checkpoint(bootstrap_actor_path, "cpu")
    normalize_bootstrap_actor_for_prior(actor)
    policy = StudentDiscretePolicy(
        actor,
        device,
        appended_logit_offset=-12.0,
    )
    manager = BatchedAgentManager(
        policy,
        min_inference_size=min(worker_count, 16),
        seed=seed,
        standardize_obs=False,
    )
    batches: list[np.ndarray] = []
    try:
        manager.init_processes(
            n_processes=worker_count,
            build_env_fn=make_natural_campaign_gym_env,
            spawn_delay=None,
            render=False,
            shm_buffer_size=8192,
        )
        collected = 0
        while collected < target_agent_observations:
            experience, _, _, _ = manager.collect_timesteps(
                target_agent_observations - collected
            )
            observations = np.asarray(experience[0], dtype=np.float32)
            if observations.ndim != 2 or observations.shape[1] != 432:
                raise ValueError(f"Unexpected natural observation shape {observations.shape}")
            batches.append(observations)
            collected += len(observations)
    finally:
        manager.cleanup()
    observations = np.ascontiguousarray(
        np.concatenate(batches, axis=0)[:target_agent_observations],
        dtype=np.float32,
    )
    if not np.isfinite(observations).all():
        raise FloatingPointError("Non-finite natural calibration observations")
    metadata_record = {
        "bootstrap_metadata": metadata,
        "shape": list(observations.shape),
        "canonical_float32_sha256": hashlib.sha256(observations.tobytes()).hexdigest(),
        "mean": float(observations.mean()),
        "standard_deviation": float(observations.std()),
        "minimum": float(observations.min()),
        "maximum": float(observations.max()),
        "raw_observations_committed": False,
    }
    return observations, metadata_record


@torch.inference_mode()
def calibrate_appended_offsets(
    actor,
    observations: np.ndarray,
    *,
    candidate_offsets: list[float],
    minimum_probability_mass: float,
    maximum_probability_mass: float,
    minimum_sampled_share: float,
    maximum_deterministic_share: float,
    minimum_legacy_top1_retention: float,
    seed: int,
    device: str = "cuda:0",
    inference_batch_size: int = 4096,
) -> dict[str, Any]:
    actor = actor.to(device).eval()
    raw_batches = []
    for start in range(0, len(observations), inference_batch_size):
        batch = torch.from_numpy(observations[start : start + inference_batch_size]).to(
            device
        )
        raw = actor(batch)
        if not torch.isfinite(raw).all():
            raise FloatingPointError("Actor emitted non-finite calibration logits")
        raw_batches.append(raw.cpu())
    raw_logits = torch.cat(raw_batches, dim=0)
    legacy_top1 = raw_logits[:, :TEACHER_ACTION_COUNT].argmax(dim=-1)
    results = []
    for candidate_index, offset in enumerate(candidate_offsets):
        logits = raw_logits.clone()
        logits[:, TEACHER_ACTION_COUNT:] += float(offset)
        probabilities = torch.softmax(logits, dim=-1)
        appended_mass = probabilities[:, TEACHER_ACTION_COUNT:].sum(dim=-1)
        generator = torch.Generator(device="cpu").manual_seed(seed + candidate_index)
        sampled = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        top1 = logits.argmax(dim=-1)
        sampled_share = float((sampled >= TEACHER_ACTION_COUNT).float().mean().item())
        deterministic_share = float((top1 >= TEACHER_ACTION_COUNT).float().mean().item())
        legacy_retention = float((top1 == legacy_top1).float().mean().item())
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(
            dim=-1
        )
        mean_mass = float(appended_mass.mean().item())
        safe = bool(
            minimum_probability_mass <= mean_mass <= maximum_probability_mass
            and sampled_share >= minimum_sampled_share
            and deterministic_share <= maximum_deterministic_share
            and legacy_retention >= minimum_legacy_top1_retention
        )
        result = {
            "appended_logit_offset": float(offset),
            "mean_appended_probability_mass": mean_mass,
            "p95_appended_probability_mass": float(
                torch.quantile(appended_mass, 0.95).item()
            ),
            "sampled_appended_action_share": sampled_share,
            "deterministic_appended_action_share": deterministic_share,
            "legacy_top1_retention": legacy_retention,
            "mean_action_entropy": float(entropy.mean().item()),
            "safe_minority_exploration": safe,
        }
        if not all(
            math.isfinite(float(value))
            for value in result.values()
            if isinstance(value, (int, float))
        ):
            raise FloatingPointError(f"Non-finite exploration calibration: {result}")
        results.append(result)
    selected = next(
        (item for item in results if item["safe_minority_exploration"]), None
    )
    if selected is None:
        raise RuntimeError(f"No candidate produced safe minority exploration: {results}")
    return {
        "candidate_results": results,
        "selection_rule": (
            "most-suppressed candidate satisfying probability, sampled-share, "
            "deterministic-share, and legacy-retention gates"
        ),
        "selected_appended_logit_offset": selected["appended_logit_offset"],
        "selected": selected,
    }
