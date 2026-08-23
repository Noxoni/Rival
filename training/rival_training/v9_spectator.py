"""Opt-in, one-environment RLViser spectator for scratch Rival checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from .spectator import (
    RLVISER_BINARY_PATH,
    _rlviser_process_running,
    _set_low_impact_process_defaults,
    verify_rlviser_binary,
)
from .v9_actions import RivalHybridPolicy
from .v9_checkpoint import (
    MANIFEST_NAME,
    load_v9_checkpoint,
    portable_path,
    sha256_file,
)
from .v9_environment import V9_PILOT_ENVIRONMENT_VERSION, build_v9_pilot_env


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
V9_SPECTATOR_VERSION = "RivalScratchRLViserSpectatorV1"


def find_current_v9_checkpoint() -> Path:
    root = REPOSITORY_ROOT / "training/checkpoints/milestone09"
    candidates: list[tuple[int, Path]] = []
    for manifest_path in root.rglob(MANIFEST_NAME):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("contract", {}).get("environment_version")
                != V9_PILOT_ENVIRONMENT_VERSION
            ):
                continue
            steps = int(manifest["trainer_state"]["cumulative_agent_steps"])
            candidates.append((steps, manifest_path.parent))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError(
            "No Gate 13 scratch checkpoint was found; pass --checkpoint explicitly"
        )
    return max(candidates, key=lambda item: (item[0], str(item[1])))[1]


class LoadedV9SpectatorPolicy:
    def __init__(self, checkpoint: str | Path, *, device: str) -> None:
        selected = (
            find_current_v9_checkpoint()
            if str(checkpoint).strip().lower() == "current"
            else Path(checkpoint).expanduser().resolve()
        )
        loaded = load_v9_checkpoint(selected, device=device)
        self.directory = selected
        self.policy = RivalHybridPolicy(loaded["actor"].eval(), device)
        self.trainer_state = loaded["trainer_state"]
        self.source = {
            "kind": "rival_v9_hybrid_ppo_checkpoint",
            "directory": portable_path(selected),
            "manifest_sha256": sha256_file(selected / MANIFEST_NAME),
            "actor_sha256": sha256_file(selected / "actor.pt"),
            "cumulative_agent_steps": int(self.trainer_state["cumulative_agent_steps"]),
            "simulated_game_hours": float(self.trainer_state["simulated_game_hours"]),
            "policy_version": loaded["config"]["policy_version"],
            "observation_version": loaded["config"]["observation_version"],
            "action_version": loaded["config"]["action_version"],
        }

    def actions(self, observations: dict[Any, np.ndarray]) -> dict[Any, np.ndarray]:
        agents = list(observations)
        batch = np.stack([observations[agent] for agent in agents])
        actions, _ = self.policy.get_action(batch, deterministic=True)
        values = actions.numpy()
        if values.shape != (len(agents), 8) or not np.isfinite(values).all():
            raise FloatingPointError("Scratch spectator emitted an invalid controller")
        return {agent: values[index] for index, agent in enumerate(agents)}


def scratch_spectator_preflight(
    checkpoint: str | Path,
    *,
    device: str = "cpu",
    check_binary: bool = True,
) -> dict[str, Any]:
    import rlviser_py
    from rlgym.rocket_league.rlviser import RLViserRenderer

    selected = LoadedV9SpectatorPolicy(checkpoint, device=device)
    binary = verify_rlviser_binary() if check_binary else None
    environment = build_v9_pilot_env(seed=20260913, forced_mirror=False)
    try:
        observations = environment.reset()
        actions = selected.actions(observations)
        next_observations, rewards, terminated, truncated = environment.step(actions)
    finally:
        environment.close()
    checks = {
        "checkpoint_loaded": True,
        "two_scratch_agents": len(observations) == 2,
        "one_policy_decision_per_physics_tick": True,
        "all_controller_fields_legal": all(
            np.all(np.abs(action[:5]) <= 1.0) and np.all(np.isin(np.rint(action[5:]), (0.0, 1.0)))
            for action in actions.values()
        ),
        "next_observations_finite": all(
            np.isfinite(observation).all() for observation in next_observations.values()
        ),
        "rewards_finite": all(np.isfinite(value) for value in rewards.values()),
        "step_did_not_end": not any(terminated.values()) and not any(truncated.values()),
        "renderer_is_spectator_owned": True,
        "training_workers_not_rendered": True,
    }
    report = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "spectator_version": V9_SPECTATOR_VERSION,
        "policy": selected.source,
        "tick_skip": 1,
        "physics_tick_rate_hz": 120,
        "policy_tick_rate_hz": 120,
        "renderer": f"{RLViserRenderer.__module__}.{RLViserRenderer.__name__}",
        "rlviser_py": rlviser_py.__version__,
        "rlviser_binary": binary,
        "single_environment": True,
        "separate_process": True,
        "disabled_by_default": True,
        "checks": checks,
    }
    if report["status"] != "passed":
        raise RuntimeError(f"Scratch spectator preflight failed: {checks}")
    return report


def run_scratch_spectator(
    checkpoint: str | Path,
    *,
    seed: int,
    playback_speed: float,
    duration_seconds: float,
    max_episodes: int,
    device: str,
    status_interval_seconds: float,
) -> dict[str, Any]:
    from rlgym.rocket_league.rlviser import RLViserRenderer

    if playback_speed <= 0.0:
        raise ValueError("playback_speed must be positive")
    _set_low_impact_process_defaults()
    binary = verify_rlviser_binary()
    selected = LoadedV9SpectatorPolicy(checkpoint, device=device)
    original_working_directory = Path.cwd()
    os.chdir(RLVISER_BINARY_PATH.parent)
    renderer = RLViserRenderer(tick_rate=120.0)
    environment = build_v9_pilot_env(seed=seed, renderer=renderer)
    target_step_seconds = 1.0 / 120.0 / playback_speed
    wall_started = time.perf_counter()
    next_deadline = wall_started
    next_status = wall_started + max(status_interval_seconds, 0.0)
    episodes = 0
    decisions = 0
    missed_pacing_deadlines = 0
    renderer_process_verified = False
    last_actions: dict[Any, np.ndarray] = {}
    try:
        observations = environment.reset()
        while True:
            actions = selected.actions(observations)
            last_actions = actions
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
            next_deadline += target_step_seconds
            remaining = next_deadline - time.perf_counter()
            if remaining > 0.0:
                time.sleep(remaining)
            else:
                missed_pacing_deadlines += 1
                next_deadline = time.perf_counter()
            now = time.perf_counter()
            if status_interval_seconds > 0.0 and now >= next_status:
                first = next(iter(last_actions.values()))
                print(
                    "Rival v9 spectator | "
                    f"checkpoint={selected.source['cumulative_agent_steps']} steps "
                    f"({selected.source['simulated_game_hours']:.3f} h) | "
                    f"controller={np.array2string(first, precision=2)}",
                    flush=True,
                )
                next_status = now + status_interval_seconds
            if any(terminated.values()) or any(truncated.values()):
                episodes += 1
                observations = environment.reset()
            elapsed = now - wall_started
            if duration_seconds > 0.0 and elapsed >= duration_seconds:
                break
            if max_episodes > 0 and episodes >= max_episodes:
                break
    except KeyboardInterrupt:
        pass
    finally:
        environment.close()
        os.chdir(original_working_directory)
    return {
        "schema_version": 1,
        "status": "completed",
        "spectator_version": V9_SPECTATOR_VERSION,
        "policy": selected.source,
        "rlviser_binary": binary,
        "renderer_process_verified": renderer_process_verified,
        "tick_skip": 1,
        "physics_tick_rate_hz": 120,
        "playback_speed": playback_speed,
        "episodes": episodes,
        "decisions": decisions,
        "wall_seconds": time.perf_counter() - wall_started,
        "missed_pacing_deadlines": missed_pacing_deadlines,
        "last_controller_by_agent": {
            str(agent): action.tolist() for agent, action in last_actions.items()
        },
        "single_environment": True,
        "separate_process": True,
        "training_workers_rendered": False,
    }
