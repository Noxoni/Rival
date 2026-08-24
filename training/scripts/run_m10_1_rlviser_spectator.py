"""Watch one isolated Rival v10.1 bootstrap environment in RLViser."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.spectator import (  # noqa: E402
    RLVISER_BINARY_PATH,
    _rlviser_process_running,
    _set_low_impact_process_defaults,
    verify_rlviser_binary,
)
from rival_training.v10_bootstrap_curriculum import (  # noqa: E402
    FAMILIES,
    RivalAgencyBootstrapCurriculumV1,
)
from rival_training.v10_bootstrap_environment import (  # noqa: E402
    build_v10_bootstrap_env,
)
from rival_training.v9_checkpoint import portable_path  # noqa: E402
from rival_training.v9_spectator import LoadedV9SpectatorPolicy  # noqa: E402


SPECTATOR_VERSION = "RivalAgencyBootstrapRLViserSpectatorV1"
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "training/checkpoints/milestone10/boundaries/plus-025h/023378810"
)


def _curriculum(environment) -> RivalAgencyBootstrapCurriculumV1:
    for mutator in getattr(environment.state_mutator, "mutators", ()):
        if isinstance(mutator, RivalAgencyBootstrapCurriculumV1):
            return mutator
    raise RuntimeError("Bootstrap curriculum mutator was not found")


def _initial_state_report(environment, family: str) -> dict[str, Any]:
    state = environment.state
    cars = sorted(state.cars.values(), key=lambda car: int(car.team_num))
    return {
        "family": family,
        "ball_position": np.asarray(state.ball.position, dtype=float).tolist(),
        "ball_velocity": np.asarray(
            state.ball.linear_velocity, dtype=float
        ).tolist(),
        "cars": [
            {
                "team": int(car.team_num),
                "position": np.asarray(car.physics.position, dtype=float).tolist(),
                "velocity": np.asarray(
                    car.physics.linear_velocity, dtype=float
                ).tolist(),
                "boost": float(car.boost_amount),
            }
            for car in cars
        ],
    }


def run_spectator(args: argparse.Namespace) -> dict[str, Any]:
    from rlgym.rocket_league.rlviser import RLViserRenderer

    if args.seconds_per_family <= 0.0:
        raise ValueError("seconds-per-family must be positive")
    families = list(FAMILIES) if args.family == "all" else [args.family]
    _set_low_impact_process_defaults()
    binary = verify_rlviser_binary()
    policy = LoadedV9SpectatorPolicy(args.checkpoint, device=args.device)
    original_working_directory = Path.cwd()
    os.chdir(RLVISER_BINARY_PATH.parent)
    renderer = RLViserRenderer(tick_rate=120.0)
    environment = build_v10_bootstrap_env(
        phase=args.phase,
        seed=args.seed,
        forced_family=families[0],
        forced_mirror=False,
        renderer=renderer,
    )
    curriculum = _curriculum(environment)
    family_reports: list[dict[str, Any]] = []
    renderer_process_verified = False
    wall_started = time.perf_counter()
    try:
        for index, family in enumerate(families):
            curriculum.forced_family = family
            curriculum.forced_active_team = index % 2
            observations = environment.reset()
            family_started = time.perf_counter()
            initial = _initial_state_report(environment, family)
            decisions = 0
            episodes = 0
            missed_pacing_deadlines = 0
            next_deadline = family_started
            print(
                f"RLVISER_FAMILY {index + 1}/{len(families)} {family}",
                flush=True,
            )
            while time.perf_counter() - family_started < args.seconds_per_family:
                actions = policy.actions(observations)
                environment.render()
                if not renderer_process_verified:
                    launch_deadline = time.perf_counter() + 5.0
                    while time.perf_counter() < launch_deadline:
                        if _rlviser_process_running():
                            renderer_process_verified = True
                            break
                        time.sleep(0.05)
                    if not renderer_process_verified:
                        raise RuntimeError("RLViser did not launch within five seconds")
                observations, _, terminated, truncated = environment.step(actions)
                decisions += 1
                next_deadline += 1.0 / 120.0 / args.playback_speed
                remaining = next_deadline - time.perf_counter()
                if remaining > 0.0:
                    time.sleep(remaining)
                else:
                    missed_pacing_deadlines += 1
                    next_deadline = time.perf_counter()
                if any(terminated.values()) or any(truncated.values()):
                    episodes += 1
                    observations = environment.reset()
            family_reports.append(
                {
                    **initial,
                    "wall_seconds": time.perf_counter() - family_started,
                    "decisions": decisions,
                    "episodes_completed": episodes,
                    "missed_pacing_deadlines": missed_pacing_deadlines,
                    "rendered": True,
                }
            )
    finally:
        environment.close()
        os.chdir(original_working_directory)
    return {
        "schema_version": 1,
        "status": "completed",
        "spectator_version": SPECTATOR_VERSION,
        "policy": policy.source,
        "phase": args.phase,
        "families": family_reports,
        "family_order": families,
        "renderer_process_verified": renderer_process_verified,
        "rlviser_binary": binary,
        "tick_skip": 1,
        "physics_and_policy_hz": 120,
        "playback_speed": args.playback_speed,
        "wall_seconds": time.perf_counter() - wall_started,
        "single_environment": True,
        "separate_process": True,
        "training_workers_rendered": False,
        "disabled_by_default": True,
        "visual_inspection": {
            "status": "requires_human_observation_while_viewer_is_open",
            "command": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m10_1_rlviser_spectator.py "
                f"--checkpoint {portable_path(policy.directory)}"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render one optional, isolated v10.1 bootstrap environment. "
            "Training workers are never rendered."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--phase", choices=("A", "B", "C"), default="A")
    parser.add_argument("--family", choices=("all", *FAMILIES), default="all")
    parser.add_argument("--seconds-per-family", type=float, default=6.0)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261051)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_spectator(args)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
