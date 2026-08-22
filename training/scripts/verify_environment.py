"""Capture dependency, CUDA, action, observation, reward, and env smoke evidence."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.actions import action_metadata  # noqa: E402
from rival_training.environment import (  # noqa: E402
    build_rlgym_env,
    environment_metadata,
)
from rival_training.observations import observation_metadata  # noqa: E402
from rival_training.rewards import reward_metadata  # noqa: E402


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> None:
    cuda_probe = {
        "torch": torch.__version__,
        "available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
    }
    if torch.cuda.is_available():
        matrix = torch.randn(512, 512, device="cuda")
        result = matrix @ matrix
        torch.cuda.synchronize()
        cuda_probe["matmul_finite"] = bool(torch.isfinite(result).all().item())

    cadence_smokes = {}
    rng = np.random.default_rng(20260822)
    for cadence in ("legacy8", "mechanics4"):
        env = build_rlgym_env(cadence)
        try:
            observations = env.reset()
            reset_agents = sorted(str(agent) for agent in observations)
            reset_finite = all(np.isfinite(obs).all() for obs in observations.values())
            actions = {
                agent: np.array([rng.integers(0, 158)], dtype=np.int64)
                for agent in observations
            }
            next_obs, rewards, terminated, truncated = env.step(actions)
            cadence_smokes[cadence] = {
                "agents": reset_agents,
                "reset_shapes": {
                    str(agent): list(obs.shape) for agent, obs in observations.items()
                },
                "reset_finite": reset_finite,
                "step_shapes": {
                    str(agent): list(obs.shape) for agent, obs in next_obs.items()
                },
                "step_observations_finite": all(
                    np.isfinite(obs).all() for obs in next_obs.values()
                ),
                "step_rewards_finite": all(np.isfinite(value) for value in rewards.values()),
                "terminated": {str(key): bool(value) for key, value in terminated.items()},
                "truncated": {str(key): bool(value) for key, value in truncated.items()},
                "reward_components": env.shared_info.get("reward_components"),
            }
        finally:
            env.close()

    report = {
        "schema_version": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "rlgym": package_version("rlgym"),
            "rlgym-api": package_version("rlgym-api"),
            "rlgym-rocket-league": package_version("rlgym-rocket-league"),
            "rlgym-tools": package_version("rlgym-tools"),
            "RocketSim": package_version("RocketSim"),
            "rlgym-ppo": package_version("rlgym-ppo"),
            "numpy": package_version("numpy"),
            "torch": torch.__version__,
        },
        "rlgym_ppo_source_commit": "4ffd2e924198bf4b2d59f4bf280b29919d7c07ea",
        "cuda": cuda_probe,
        "environment": environment_metadata(),
        "actions": action_metadata(),
        "observation": observation_metadata(),
        "reward": reward_metadata(),
        "cadence_smokes": cadence_smokes,
        "passed": all(
            item["reset_finite"]
            and item["step_observations_finite"]
            and item["step_rewards_finite"]
            for item in cadence_smokes.values()
        ),
    }
    output = REPOSITORY_ROOT / "training/results/environment_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
