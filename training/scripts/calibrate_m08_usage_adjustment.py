"""Calibrate a prospective M08 checkpoint PASS-bias reduction."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.config import (  # noqa: E402
    canonical_config_sha256,
    load_milestone08_config,
)
from rival_training.m08_campaign import (  # noqa: E402
    frozen_strategic_proof,
    load_m08_state,
    make_m08_ppo,
)
from rival_training.teacher import sha256_file  # noqa: E402


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _file_manifest(directory: Path) -> dict[str, dict[str, int | str]]:
    return {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def _statistics(logits: torch.Tensor, pass_bias_delta: float) -> dict[str, object]:
    adjusted = logits.detach().clone()
    adjusted[:, 0].add_(float(pass_bias_delta))
    probabilities = torch.softmax(adjusted, dim=-1)
    choices = adjusted.argmax(dim=-1)
    pass_probability = probabilities[:, 0]
    conditional = probabilities[:, 1:] / torch.clamp(
        1.0 - pass_probability[:, None], min=1e-12
    )
    return {
        "mean_pass_probability": float(pass_probability.mean().item()),
        "minimum_pass_probability": float(pass_probability.min().item()),
        "maximum_pass_probability": float(pass_probability.max().item()),
        "mean_override_probability": float(
            (1.0 - pass_probability).mean().item()
        ),
        "deterministic_pass_rate": float((choices == 0).float().mean().item()),
        "deterministic_override_rate": float(
            (choices != 0).float().mean().item()
        ),
        "conditional_override_entropy": float(
            (-(conditional * torch.log(torch.clamp(conditional, min=1e-12)))
             .sum(dim=-1).mean().item())
        ),
        "finite": bool(torch.isfinite(adjusted).all().item()),
    }


def _calibrate_delta(
    logits: torch.Tensor,
    *,
    target_override_probability: float,
    iterations: int = 64,
) -> float:
    target = float(target_override_probability)
    if not 0.0 < target < 1.0:
        raise ValueError("Target override probability must be between zero and one")
    lower, upper = -20.0, 20.0
    for _ in range(int(iterations)):
        midpoint = (lower + upper) / 2.0
        measured = float(
            _statistics(logits, midpoint)["mean_override_probability"]
        )
        if measured > target:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--target-mean-override-probability", type=float, default=0.10
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_milestone08_config()
    source = args.checkpoint.resolve()
    source_files_before = _file_manifest(source)
    state = load_m08_state(source)
    if state["config_sha256"] != canonical_config_sha256(config):
        raise RuntimeError("Adjustment source checkpoint config hash mismatch")
    if state.get("mechanics_usage_adjustment_history", []):
        raise RuntimeError("Refusing to recalibrate an already-adjusted checkpoint")

    dataset = (
        REPOSITORY_ROOT
        / "training/datasets/milestone08/natural_prior_observations.npy"
    )
    observations = np.load(dataset, allow_pickle=False)
    ppo = make_m08_ppo(config, device="cuda:0")
    ppo.load_from(str(source))
    ppo.policy.eval()
    with torch.inference_mode():
        logits = ppo.policy.logits(
            torch.from_numpy(observations).to("cuda:0")
        )
    before = _statistics(logits, 0.0)
    target = float(args.target_mean_override_probability)
    delta = _calibrate_delta(
        logits,
        target_override_probability=target,
    )
    after = _statistics(logits, delta)
    source_bias = float(
        ppo.policy.actor.output_layer.bias[0].detach().cpu().item()
    )

    adjusted_logits = logits.detach().clone()
    adjusted_logits[:, 0].add_(delta)
    probabilities = torch.softmax(adjusted_logits, dim=-1).cpu()
    generator = torch.Generator(device="cpu").manual_seed(
        int(config["seeds"]["audit"]) + 8
    )
    sampled = torch.multinomial(
        probabilities,
        num_samples=1,
        replacement=True,
        generator=generator,
    ).reshape(-1)
    counts = torch.bincount(sampled, minlength=69)
    sampled_rate = float((sampled != 0).float().mean().item())
    maximum = float(config["evaluation"]["maximum_sampled_override_share"])
    checks = {
        "source_is_clean_2m_boundary": int(state["cumulative_agent_steps"])
        == 1_999_776,
        "source_has_no_prior_usage_adjustment": not state.get(
            "mechanics_usage_adjustment_history", []
        ),
        "target_is_increased_but_bounded": float(
            before["mean_override_probability"]
        )
        < target
        <= maximum,
        "negative_finite_pass_bias_delta": math.isfinite(delta) and delta < 0.0,
        "mean_probability_hits_target": abs(
            float(after["mean_override_probability"]) - target
        )
        <= 1e-6,
        "sampled_rate_near_target": abs(sampled_rate - target) <= 0.02,
        "sampled_rate_bounded": sampled_rate <= maximum,
        "all_override_outputs_sampled": int((counts[1:] > 0).sum().item()) == 68,
        "statistics_finite": bool(before["finite"] and after["finite"]),
        "strategic_branch_unchanged": frozen_strategic_proof(config)[
            "all_unchanged"
        ],
        "source_checkpoint_unchanged": source_files_before == _file_manifest(source),
    }
    report = {
        "schema_version": 1,
        "status": "authorized" if all(checks.values()) else "rejected",
        "purpose": (
            "Prospective user-directed M08 mechanics exposure increase after the "
            "untouched 2M boundary showed zero deterministic overrides."
        ),
        "config_sha256": canonical_config_sha256(config),
        "source_checkpoint": {
            "directory": _portable(source),
            "agent_steps": int(state["cumulative_agent_steps"]),
            "files": source_files_before,
        },
        "natural_observation_corpus": {
            "path": _portable(dataset),
            "observation_count": int(len(observations)),
            "size_bytes": dataset.stat().st_size,
            "sha256": sha256_file(dataset),
        },
        "target_mean_override_probability": target,
        "pass_bias_adjustment": {
            "source_bias": source_bias,
            "source_mean_override_probability": float(
                before["mean_override_probability"]
            ),
            "delta": float(delta),
            "adjusted_bias": float(source_bias + delta),
            "application_point": (
                "after exact full-PPO checkpoint restore and before the first "
                "post-2M rollout"
            ),
            "optimizer_state_policy": "preserve exact loaded Adam state",
        },
        "before": before,
        "after": after,
        "sampled_audit": {
            "seed": int(config["seeds"]["audit"]) + 8,
            "sample_count": int(len(sampled)),
            "override_rate": sampled_rate,
            "action_counts": counts.tolist(),
        },
        "checks": checks,
        "production_promoted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "authorized" else 2


if __name__ == "__main__":
    raise SystemExit(main())
