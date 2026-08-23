"""Natural headless 1v1 RocketSim environment construction."""

from __future__ import annotations

from functools import partial
import random
from typing import Any

import gym
import numpy as np
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
from .config import load_milestone06_config, stage_config
from .curriculum import CURRICULUM_FAMILIES, RivalCurriculumMutator
from .metrics import build_campaign_metric_vector
from .observations import WispCompatibleObs
from .rewards import RivalRewardV1, RivalRewardV2, reward_v2_metadata


ENVIRONMENT_VERSION = "RivalNatural1v1RocketSimV1"
CAMPAIGN_ENVIRONMENT_VERSION = "RivalCurriculum1v1RocketSimV2"


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


def build_campaign_env(
    stage_name: str,
    *,
    seed: int | None = None,
    natural_only: bool = False,
) -> RLGym:
    """Build Reward V2 plus a stage-weighted broad majority-natural reset mix."""
    config = load_milestone06_config()
    stage = stage_config(config, stage_name)
    selected_seed = int(config["seeds"]["training"] if seed is None else seed)
    weights = dict(stage["curriculum_weights"])
    if natural_only:
        weights = {name: float(name == "natural") for name in CURRICULUM_FAMILIES}
    transition_engine = RocketSimEngine(rlbot_delay=True)
    curriculum = RivalCurriculumMutator(weights, seed=selected_seed)
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            curriculum,
        ),
        obs_builder=WispCompatibleObs(seed=selected_seed),
        action_parser=RivalActionParser(cadence="mechanics4"),
        reward_fn=RivalRewardV2(cadence_ticks=4),
        transition_engine=transition_engine,
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(
                float(config["environment"]["no_touch_timeout_seconds"])
            ),
            TimeoutCondition(float(config["environment"]["episode_timeout_seconds"])),
        ),
        shared_info_provider=None,
        renderer=None,
    )


class CampaignSeededDiscrete(gym.spaces.Discrete):
    """Seed the environment components when rlgym-ppo seeds its action space."""

    def __init__(self, n: int, seed_callback) -> None:
        super().__init__(n=n)
        self._seed_callback = seed_callback

    def seed(self, seed: int | None = None):
        if seed is not None:
            self._seed_callback(int(seed))
        return super().seed(seed)


class CampaignGymWrapper(RLGymV2GymWrapper):
    def __init__(self, rlgym_env: RLGym) -> None:
        super().__init__(rlgym_env)
        self.action_space = CampaignSeededDiscrete(
            int(self.action_space.n), self._seed_components
        )

    def _seed_components(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        if hasattr(self.rlgym_env.state_mutator, "mutators"):
            for mutator in self.rlgym_env.state_mutator.mutators:
                if hasattr(mutator, "seed"):
                    mutator.seed(seed)
        if hasattr(self.rlgym_env.obs_builder, "seed"):
            self.rlgym_env.obs_builder.seed(seed)

    def step(self, actions):
        observations, rewards, done, truncated, info = super().step(actions)
        info["state"] = build_campaign_metric_vector(
            self.rlgym_env.state, self.rlgym_env.shared_info
        )
        return observations, rewards, done, truncated, info


def make_campaign_gym_env(stage_name: str) -> CampaignGymWrapper:
    return CampaignGymWrapper(build_campaign_env(stage_name))


def make_natural_campaign_gym_env() -> CampaignGymWrapper:
    return CampaignGymWrapper(build_campaign_env("stage_a", natural_only=True))


def campaign_environment_factory(stage_name: str):
    """Return a pickle-safe stage-specific factory for worker processes."""
    return partial(make_campaign_gym_env, stage_name)


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


def campaign_environment_metadata(stage_name: str) -> dict[str, Any]:
    config = load_milestone06_config()
    stage = stage_config(config, stage_name)
    return {
        "schema_version": 2,
        "environment_version": CAMPAIGN_ENVIRONMENT_VERSION,
        "distribution": "majority-natural weighted broad 1v1 curriculum",
        "stage": stage_name,
        "curriculum_weights": stage["curriculum_weights"],
        "transition_engine": "rlgym.rocket_league.sim.RocketSimEngine",
        "rlbot_action_delay": True,
        "renderer": None,
        "physics_tick_rate_hz": 120,
        "cadence": "mechanics4",
        "cadence_ticks": 4,
        "termination": "goal",
        "truncation": {
            "no_touch_seconds": config["environment"]["no_touch_timeout_seconds"],
            "episode_seconds": config["environment"]["episode_timeout_seconds"],
        },
        "reward": reward_v2_metadata(),
    }
