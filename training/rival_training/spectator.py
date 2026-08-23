"""Isolated, real-time RocketSim spectator for a selected Rival policy.

Nothing imports this module from the headless training or RLBot diagnostic paths.
The single rendered environment exists only in the explicit spectator process.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import psutil
from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
import torch

from .actions import RivalActionParser
from .campaign import CAMPAIGN_CHECKPOINT_STATE
from .checkpoint import load_actor_checkpoint, portable_path
from .observations import WispCompatibleObs
from .policy import StudentDiscretePolicy
from .rewards import RivalRewardV1
from .teacher import EXPANDED_ACTION_COUNT, FrozenWispReference, WispStudentActor, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SPECTATOR_VERSION = "RivalRLViserSpectatorV1"
RLVISER_BINARY_PATH = REPOSITORY_ROOT / "training/tools/rlviser/rlviser.exe"
RLVISER_BINARY_SHA256 = (
    "518a04f711c68de81008a51cb90a61808847d11a0ce8a102a87017cf6f94f8ad"
)


@dataclass(frozen=True)
class LoadedSpectatorPolicy:
    model: torch.nn.Module
    action_count: int
    source: dict[str, Any]

    @torch.inference_mode()
    def action(self, observation: np.ndarray, *, legacy_only: bool = False) -> int:
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=next(self.model.parameters()).device,
        ).unsqueeze(0)
        if hasattr(self.model, "logits"):
            logits = self.model.logits(tensor)
        else:
            logits = self.model(tensor)
        logits = logits.reshape(-1)
        if logits.numel() != self.action_count:
            raise RuntimeError(
                f"Spectator policy declared {self.action_count} actions but emitted "
                f"{logits.numel()} logits"
            )
        if not bool(torch.isfinite(logits).all().item()):
            raise FloatingPointError("Spectator policy emitted non-finite logits")
        if legacy_only:
            logits = logits[:90]
        return int(logits.argmax().item())


def _device_for_model(model: torch.nn.Module, device: str) -> torch.nn.Module:
    return model.to(torch.device(device)).eval()


def _load_campaign_policy(directory: Path, device: str) -> LoadedSpectatorPolicy:
    state_path = directory / CAMPAIGN_CHECKPOINT_STATE
    policy_path = directory / "PPO_POLICY.pt"
    if not state_path.is_file() or not policy_path.is_file():
        raise FileNotFoundError(
            f"Campaign checkpoint needs {state_path.name} and {policy_path.name}: {directory}"
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    actor = WispStudentActor(EXPANDED_ACTION_COUNT)
    policy = StudentDiscretePolicy(actor, "cpu")
    policy.load_state_dict(
        torch.load(policy_path, map_location="cpu", weights_only=True), strict=True
    )
    policy = _device_for_model(policy, device)
    return LoadedSpectatorPolicy(
        model=policy,
        action_count=EXPANDED_ACTION_COUNT,
        source={
            "kind": "campaign_checkpoint",
            "directory": portable_path(directory),
            "campaign_state_sha256": sha256_file(state_path),
            "policy_sha256": sha256_file(policy_path),
            "cumulative_agent_steps": int(state["cumulative_agent_steps"]),
            "stage": state["stage"],
            "appended_logit_offset": float(
                state["action_exploration_prior"]["appended_logit_offset"]
            ),
        },
    )


def find_current_campaign_checkpoint() -> Path:
    root = REPOSITORY_ROOT / "training/checkpoints/milestone06"
    candidates: list[tuple[int, Path]] = []
    for state_path in root.glob(f"*/{CAMPAIGN_CHECKPOINT_STATE}"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            candidates.append((int(state["cumulative_agent_steps"]), state_path.parent))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError(
            "No local Milestone 06 campaign checkpoint was found under "
            f"{root}. Supply --checkpoint with an actor checkpoint or TorchScript export."
        )
    return max(candidates, key=lambda item: (item[0], str(item[1])))[1]


def load_spectator_policy(
    checkpoint: str | Path | None,
    *,
    device: str = "cpu",
) -> LoadedSpectatorPolicy:
    """Load frozen Wisp, an actor checkpoint/export, or a full campaign checkpoint."""
    if checkpoint is None or str(checkpoint).strip().lower() == "frozen-wisp":
        model = _device_for_model(FrozenWispReference(), device)
        return LoadedSpectatorPolicy(
            model=model,
            action_count=90,
            source={
                "kind": "frozen_wisp",
                "policy_sha256": sha256_file(REPOSITORY_ROOT / "bot/models/POLICY.lt"),
                "shared_head_sha256": sha256_file(
                    REPOSITORY_ROOT / "bot/models/SHARED_HEAD.lt"
                ),
            },
        )

    selected = str(checkpoint).strip()
    path = (
        find_current_campaign_checkpoint()
        if selected.lower() == "current"
        else Path(selected).expanduser().resolve()
    )
    if path.is_dir():
        return _load_campaign_policy(path, device)
    if not path.is_file():
        raise FileNotFoundError(f"Spectator checkpoint does not exist: {path}")
    if path.name == "PPO_POLICY.pt":
        return _load_campaign_policy(path.parent, device)
    if path.suffix.lower() in {".ts", ".lt"}:
        model = _device_for_model(torch.jit.load(str(path), map_location="cpu"), device)
        sample = torch.zeros(1, 432, dtype=torch.float32, device=torch.device(device))
        with torch.inference_mode():
            action_count = int(model(sample).reshape(-1).numel())
        if action_count not in {90, EXPANDED_ACTION_COUNT}:
            raise ValueError(
                f"Expected a 90- or 158-action TorchScript policy, got {action_count}"
            )
        return LoadedSpectatorPolicy(
            model=model,
            action_count=action_count,
            source={
                "kind": "torchscript_actor",
                "path": portable_path(path),
                "sha256": sha256_file(path),
            },
        )

    actor, metadata = load_actor_checkpoint(path, "cpu")
    actor = _device_for_model(actor, device)
    return LoadedSpectatorPolicy(
        model=actor,
        action_count=actor.action_count,
        source={
            "kind": "portable_actor_checkpoint",
            "path": portable_path(path),
            "sha256": sha256_file(path),
            "metadata": metadata,
        },
    )


def resolve_tick_skip(checkpoint: str | Path | None, requested: str | int) -> int:
    if str(requested).lower() != "auto":
        tick_skip = int(requested)
        if tick_skip not in {4, 8}:
            raise ValueError("Spectator tick skip must be 4 or 8")
        return tick_skip
    return 8 if checkpoint is None or str(checkpoint).lower() == "frozen-wisp" else 4


def build_spectator_environment(
    tick_skip: int,
    *,
    seed: int,
    renderer: Any | None,
) -> RLGym:
    """Build one spectator-owned environment; never used by rollout workers."""
    cadence = {4: "mechanics4", 8: "legacy8"}.get(tick_skip)
    if cadence is None:
        raise ValueError("Spectator tick skip must be 4 or 8")
    transition_engine = RocketSimEngine(rlbot_delay=True)
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            KickoffMutator(),
        ),
        obs_builder=WispCompatibleObs(seed=seed),
        action_parser=RivalActionParser(cadence=cadence),
        reward_fn=RivalRewardV1(),
        transition_engine=transition_engine,
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(30.0),
            TimeoutCondition(300.0),
        ),
        shared_info_provider=None,
        renderer=renderer,
    )


def _set_low_impact_process_defaults() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (AttributeError, psutil.Error):
        pass


def verify_rlviser_binary() -> dict[str, Any]:
    if not RLVISER_BINARY_PATH.is_file():
        raise FileNotFoundError(
            f"RLViser executable is missing: {RLVISER_BINARY_PATH}. Run "
            "./training/install_rlviser_spectator.ps1 first."
        )
    actual_hash = sha256_file(RLVISER_BINARY_PATH)
    if actual_hash != RLVISER_BINARY_SHA256:
        raise RuntimeError(
            "RLViser executable hash mismatch; rerun the optional spectator installer: "
            f"expected {RLVISER_BINARY_SHA256}, got {actual_hash}"
        )
    return {
        "version": "0.8.2",
        "path": portable_path(RLVISER_BINARY_PATH),
        "sha256": actual_hash,
        "size_bytes": RLVISER_BINARY_PATH.stat().st_size,
    }


def _rlviser_process_running() -> bool:
    expected = RLVISER_BINARY_PATH.resolve()
    for process in psutil.process_iter(("exe", "name")):
        try:
            executable = process.info.get("exe")
            if executable and Path(executable).resolve() == expected:
                return True
        except (OSError, psutil.Error):
            continue
    return False


def spectator_preflight(
    checkpoint: str | Path | None,
    *,
    tick_skip: int,
    device: str,
    check_rlviser: bool = True,
    check_binary: bool = True,
) -> dict[str, Any]:
    """Load everything and step once without opening a renderer window."""
    if check_rlviser:
        import rlviser_py
        from rlgym.rocket_league.rlviser import RLViserRenderer

        rlviser_version = rlviser_py.__version__
        renderer_class = f"{RLViserRenderer.__module__}.{RLViserRenderer.__name__}"
    else:
        rlviser_version = None
        renderer_class = None
    binary = verify_rlviser_binary() if check_binary else None
    policy = load_spectator_policy(checkpoint, device=device)
    opponent = load_spectator_policy(None, device=device)
    env = build_spectator_environment(tick_skip, seed=20260901, renderer=None)
    try:
        observations = env.reset()
        agents = list(observations)
        actions = {
            agents[0]: np.asarray([policy.action(observations[agents[0]])]),
            agents[1]: np.asarray([opponent.action(observations[agents[1]])]),
        }
        next_observations, _, _, _ = env.step(actions)
    finally:
        env.close()
    return {
        "schema_version": 1,
        "status": "passed",
        "spectator_version": SPECTATOR_VERSION,
        "policy": policy.source,
        "action_count": policy.action_count,
        "tick_skip": tick_skip,
        "physics_tick_rate_hz": 120,
        "renderer": renderer_class,
        "rlviser_py": rlviser_version,
        "rlviser_binary": binary,
        "single_environment": True,
        "separate_process": True,
        "headless_environment_modified": False,
        "reset_agent_count": len(agents),
        "step_observation_shapes": {
            str(agent): list(np.asarray(value).shape)
            for agent, value in next_observations.items()
        },
    }


def run_spectator(
    checkpoint: str | Path | None,
    *,
    opponent_mode: str,
    selected_team: int,
    tick_skip: int,
    legacy_only: bool,
    seed: int,
    playback_speed: float,
    duration_seconds: float,
    max_episodes: int,
    device: str,
) -> dict[str, Any]:
    """Render a paced single environment until its explicit viewer bound is reached."""
    if playback_speed <= 0:
        raise ValueError("playback_speed must be positive")
    if selected_team not in {0, 1}:
        raise ValueError("selected_team must be blue (0) or orange (1)")
    if opponent_mode not in {"frozen-wisp", "selected"}:
        raise ValueError("opponent_mode must be frozen-wisp or selected")

    from rlgym.rocket_league.rlviser import RLViserRenderer

    _set_low_impact_process_defaults()
    binary = verify_rlviser_binary()
    selected = load_spectator_policy(checkpoint, device=device)
    opponent = (
        selected
        if opponent_mode == "selected"
        else load_spectator_policy(None, device=device)
    )
    if legacy_only and selected.action_count < 90:
        raise ValueError("Legacy-only mode requires at least 90 policy actions")
    original_working_directory = Path.cwd()
    os.chdir(RLVISER_BINARY_PATH.parent)
    renderer = RLViserRenderer(tick_rate=120.0 / tick_skip)
    env = build_spectator_environment(tick_skip, seed=seed, renderer=renderer)
    target_step_seconds = tick_skip / 120.0 / playback_speed
    wall_started = time.perf_counter()
    next_deadline = wall_started
    episodes = 0
    decisions = 0
    renderer_process_verified = False
    selected_action_counts = np.zeros(selected.action_count, dtype=np.int64)
    try:
        observations = env.reset()
        while True:
            selected_agent = next(
                agent
                for agent in observations
                if env.state.cars[agent].team_num == selected_team
            )
            opponent_agent = next(agent for agent in observations if agent != selected_agent)
            selected_action = selected.action(
                observations[selected_agent], legacy_only=legacy_only
            )
            opponent_action = opponent.action(
                observations[opponent_agent],
                legacy_only=(opponent_mode == "selected" and legacy_only),
            )
            env.render()
            if not renderer_process_verified:
                launch_deadline = time.perf_counter() + 5.0
                while time.perf_counter() < launch_deadline:
                    if _rlviser_process_running():
                        renderer_process_verified = True
                        break
                    time.sleep(0.05)
                if not renderer_process_verified:
                    raise RuntimeError(
                        "RLViser did not launch within five seconds; executable and "
                        "Python bridge are present but no viewer process was observed"
                    )
            observations, _, terminated, truncated = env.step(
                {
                    selected_agent: np.asarray([selected_action], dtype=np.int64),
                    opponent_agent: np.asarray([opponent_action], dtype=np.int64),
                }
            )
            selected_action_counts[selected_action] += 1
            decisions += 1
            next_deadline += target_step_seconds
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_deadline = time.perf_counter()

            if any(terminated.values()) or any(truncated.values()):
                episodes += 1
                observations = env.reset()
            elapsed = time.perf_counter() - wall_started
            if duration_seconds > 0 and elapsed >= duration_seconds:
                break
            if max_episodes > 0 and episodes >= max_episodes:
                break
    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        os.chdir(original_working_directory)

    return {
        "spectator_version": SPECTATOR_VERSION,
        "policy": selected.source,
        "opponent": opponent.source,
        "rlviser_binary": binary,
        "renderer_process_verified": renderer_process_verified,
        "selected_team": "blue" if selected_team == 0 else "orange",
        "tick_skip": tick_skip,
        "legacy_only": legacy_only,
        "playback_speed": playback_speed,
        "episodes": episodes,
        "decisions": decisions,
        "wall_seconds": time.perf_counter() - wall_started,
        "selected_action_counts": selected_action_counts.tolist(),
    }
