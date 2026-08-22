"""Natural headless 1v1 RocketSim environment construction."""

from __future__ import annotations

from typing import Any

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
from rlgym_ppo.util import RLGymV2GymWrapper

from .actions import CADENCE_TICKS, RivalActionParser
from .observations import WispCompatibleObs
from .rewards import RivalRewardV1


ENVIRONMENT_VERSION = "RivalNatural1v1RocketSimV1"


def build_rlgym_env(
    cadence: str = "mechanics4",
    *,
    seed: int = 20260822,
    no_touch_timeout_seconds: float = 30.0,
    episode_timeout_seconds: float = 300.0,
) -> RLGym:
    """Build the renderer-free default distribution: ordinary 1v1 kickoffs/play."""
    if cadence not in CADENCE_TICKS:
        raise ValueError(f"Unknown cadence: {cadence}")
    # RocketSimEngine initializes the packaged soccar collision meshes. Construct it
    # before the observation builder creates its BallPredictor.
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
            NoTouchTimeoutCondition(no_touch_timeout_seconds),
            TimeoutCondition(episode_timeout_seconds),
        ),
        shared_info_provider=None,
        renderer=None,
    )


def make_gym_env_mechanics4() -> RLGymV2GymWrapper:
    """Pickle-safe factory for rlgym-ppo worker processes."""
    return RLGymV2GymWrapper(build_rlgym_env("mechanics4"))


def make_gym_env_legacy8() -> RLGymV2GymWrapper:
    """Pickle-safe 8-tick Wisp-cadence factory."""
    return RLGymV2GymWrapper(build_rlgym_env("legacy8"))


def environment_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "environment_version": ENVIRONMENT_VERSION,
        "distribution": "natural headless 1v1 self-play",
        "transition_engine": "rlgym.rocket_league.sim.RocketSimEngine",
        "rlbot_action_delay": True,
        "renderer": None,
        "physics_tick_rate_hz": 120,
        "cadence_modes": CADENCE_TICKS,
        "termination": "goal",
        "truncation": {
            "no_touch_seconds": 30.0,
            "episode_seconds": 300.0,
        },
    }
