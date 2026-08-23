"""Measure and checkpoint the Milestone 08 mechanics PASS prior."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.actions import action_metadata  # noqa: E402
from rival_training.config import (  # noqa: E402
    canonical_config_sha256,
    load_milestone08_config,
)
from rival_training.environment import (  # noqa: E402
    build_dual_rate_env,
    dual_rate_environment_metadata,
)
from rival_training.mechanics import (  # noqa: E402
    export_mechanics_torchscript,
    save_mechanics_actor,
)
from rival_training.policy import (  # noqa: E402
    MechanicsActor,
    calibrate_mechanics_pass_prior,
    mechanics_prior_statistics,
)
from rival_training.teacher import sha256_file  # noqa: E402


def collect_natural_observations(count: int, seed: int) -> tuple[np.ndarray, dict]:
    env = build_dual_rate_env(seed=seed, natural_only=True, force_pass=True)
    rows: list[np.ndarray] = []
    episodes = 0
    decisions = 0
    started = time.perf_counter()
    try:
        observations = env.reset()
        while len(rows) < count:
            for value in observations.values():
                if len(rows) < count:
                    rows.append(np.asarray(value, dtype=np.float32).copy())
            actions = {
                agent: np.asarray([0], dtype=np.int64) for agent in observations
            }
            observations, _, terminated, truncated = env.step(actions)
            decisions += 1
            if any(terminated.values()) or any(truncated.values()):
                episodes += 1
                observations = env.reset()
    finally:
        env.close()
    values = np.stack(rows)
    return values, {
        "observation_count": int(len(values)),
        "environment_decisions": decisions,
        "completed_episodes": episodes,
        "wall_seconds": time.perf_counter() - started,
        "all_finite": bool(np.isfinite(values).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the M08 mechanics prior. Existing evidence is protected "
            "unless --overwrite is supplied explicitly."
        )
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing corpus, actor, export, and compact report",
    )
    args = parser.parse_args()

    dataset_path = (
        REPOSITORY_ROOT
        / "training/datasets/milestone08/natural_prior_observations.npy"
    )
    checkpoint_path = (
        REPOSITORY_ROOT
        / "training/artifacts/milestone08/mechanics_initial_v1.pt"
    )
    export_path = (
        REPOSITORY_ROOT
        / "training/artifacts/milestone08/mechanics_initial_v1.ts"
    )
    output = (
        REPOSITORY_ROOT
        / "training/results/milestone08/mechanics_prior_calibration.json"
    )
    protected_paths = (dataset_path, checkpoint_path, export_path, output)
    existing = [path for path in protected_paths if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "refusing to replace existing M08 calibration evidence; pass "
            f"--overwrite only for an explicitly authorized rerun: {existing}"
        )

    config = load_milestone08_config()
    prior = config["mechanics_prior"]
    observations, corpus = collect_natural_observations(
        int(prior["natural_observations"]),
        int(config["seeds"]["prior_corpus"]),
    )
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(dataset_path, observations, allow_pickle=False)

    actor = MechanicsActor(seed=int(config["seeds"]["actor_initialization"]))
    before = mechanics_prior_statistics(actor, observations)
    calibration = calibrate_mechanics_pass_prior(
        actor,
        observations,
        target_override_probability=float(
            prior["target_sampled_override_probability"]
        ),
    )
    with torch.inference_mode():
        probabilities = torch.softmax(
            actor(torch.from_numpy(observations)), dim=-1
        )
        generator = torch.Generator(device="cpu").manual_seed(
            int(config["seeds"]["audit"])
        )
        sampled = torch.multinomial(
            probabilities,
            num_samples=1,
            replacement=True,
            generator=generator,
        ).reshape(-1)
    sampled_override_rate = float((sampled != 0).float().mean().item())
    sampled_counts = torch.bincount(sampled, minlength=69).tolist()
    artifact_metadata = {
        "campaign_id": config["campaign_id"],
        "config_sha256": canonical_config_sha256(config),
        "prior_calibration": calibration,
        "architecture": "432-256-LayerNorm-ReLU-256-LayerNorm-ReLU-69",
        "pass_index": 0,
        "global_mapping": {"mechanics_1": 90, "mechanics_68": 157},
    }
    checkpoint = save_mechanics_actor(
        checkpoint_path,
        actor,
        artifact_metadata,
    )
    export = export_mechanics_torchscript(
        actor,
        export_path,
    )
    minimum = float(prior["minimum_sampled_override_probability"])
    maximum = float(prior["maximum_sampled_override_probability"])
    gates = {
        "corpus_finite": corpus["all_finite"],
        "probability_override_within_bounds": minimum
        <= calibration["mean_override_probability"]
        <= maximum,
        "sampled_override_within_bounds": minimum
        <= sampled_override_rate
        <= maximum,
        "deterministic_pass_rate": calibration["deterministic_pass_rate"]
        >= float(prior["minimum_deterministic_pass_rate"]),
        "override_outputs_numerically_non_starved": (
            calibration["conditional_override_entropy"] >= 0.95 * math.log(68)
            and sum(count > 0 for count in sampled_counts[1:]) >= 60
        ),
        "checkpoint_reload_exact": checkpoint["fresh_reload_exact"],
        "torchscript_reload_exact": export["fresh_reload_exact_logits"],
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(gates.values()) else "failed",
        "purpose": "milestone08_natural_state_mechanics_pass_prior",
        "config_sha256": canonical_config_sha256(config),
        "environment": dual_rate_environment_metadata(),
        "action_contract": action_metadata(),
        "corpus": {
            **corpus,
            "path": "training/datasets/milestone08/natural_prior_observations.npy",
            "size_bytes": dataset_path.stat().st_size,
            "sha256": sha256_file(dataset_path),
        },
        "uncalibrated": before,
        "calibrated": calibration,
        "sampled_audit": {
            "seed": int(config["seeds"]["audit"]),
            "sample_count": int(len(sampled)),
            "sampled_pass_rate": 1.0 - sampled_override_rate,
            "sampled_override_rate": sampled_override_rate,
            "mechanics_action_counts": sampled_counts,
        },
        "checkpoint": checkpoint,
        "torchscript_export": export,
        "gates": gates,
        "production_promoted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
