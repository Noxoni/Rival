"""Run several thousand natural random decisions and assert finite training state."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.environment import build_rlgym_env  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decisions", type=int, default=5000)
    args = parser.parse_args()
    env = build_rlgym_env("mechanics4")
    rng = np.random.default_rng(20260822)
    component_totals: dict[str, float] = defaultdict(float)
    episode_count = 1
    goal_terminations = 0
    truncations = 0
    start = time.perf_counter()
    observations = env.reset()
    try:
        for _ in range(args.decisions):
            actions = {
                agent: np.array([rng.integers(0, 158)], dtype=np.int64)
                for agent in observations
            }
            observations, rewards, terminated, truncated = env.step(actions)
            if not all(np.isfinite(obs).all() for obs in observations.values()):
                raise FloatingPointError("Non-finite observation during stress run")
            if not all(np.isfinite(value) for value in rewards.values()):
                raise FloatingPointError("Non-finite reward during stress run")
            for values in env.shared_info.get("reward_components", {}).values():
                for name, value in values.items():
                    if not np.isfinite(value):
                        raise FloatingPointError(f"Non-finite {name} reward")
                    component_totals[name] += float(value)
            if any(terminated.values()) or any(truncated.values()):
                goal_terminations += int(any(terminated.values()))
                truncations += int(any(truncated.values()))
                episode_count += 1
                observations = env.reset()
    finally:
        env.close()
    elapsed = time.perf_counter() - start
    report = {
        "schema_version": 1,
        "status": "passed",
        "cadence": "mechanics4",
        "decisions": args.decisions,
        "agent_steps": args.decisions * 2,
        "physics_ticks_per_environment": args.decisions * 4,
        "simulated_seconds_per_environment": args.decisions * 4 / 120,
        "wall_seconds": elapsed,
        "environment_decisions_per_second": args.decisions / elapsed,
        "episodes": episode_count,
        "goal_terminations": goal_terminations,
        "truncations": truncations,
        "reward_component_totals": dict(component_totals),
        "all_observations_and_rewards_finite": True,
    }
    output = REPOSITORY_ROOT / "training/results/environment_stress.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
