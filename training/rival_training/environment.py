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

from .actions import CADENCE_TICKS, DualRateActionParser, RivalActionParser
from .config import load_milestone06_config, load_milestone08_config, stage_config
from .curriculum import CURRICULUM_FAMILIES, RivalCurriculumMutator
from .metrics import build_campaign_metric_vector
from .m08_metrics import build_m08_metric_vector
from .observations import WispCompatibleObs
from .rewards import RivalRewardV1, RivalRewardV2, reward_v2_metadata


ENVIRONMENT_VERSION = "RivalNatural1v1RocketSimV1"
CAMPAIGN_ENVIRONMENT_VERSION = "RivalCurriculum1v1RocketSimV2"
DUAL_RATE_ENVIRONMENT_VERSION = "RivalDualRate1v1RocketSimV1"


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


class M08GymWrapper(CampaignGymWrapper):
    """Transport override-context and bounded follow-up diagnostics to PPO."""

    def __init__(self, rlgym_env: RLGym) -> None:
        self._override_windows: dict[Any, dict[str, float]] = {}
        super().__init__(rlgym_env)

    def reset(self):
        self._override_windows = {}
        return super().reset()

    def step(self, actions):
        # Call the base rlgym-ppo wrapper directly so the M06 vector is not built
        # and its one-shot reset marker is not consumed twice.
        observations, rewards, done, truncated, info = RLGymV2GymWrapper.step(
            self, actions
        )
        decisions = self.rlgym_env.shared_info.get("dual_rate_last_decisions", {})
        mechanics = self.rlgym_env.shared_info.get("mechanics_metrics", {})
        for agent, decision in decisions.items():
            if bool(decision.get("override_selected", False)):
                self._override_windows[agent] = {
                    "remaining": 30.0,
                    "active": 1.0,
                    "useful_touch": 0.0,
                    "goal_for": 0.0,
                    "goal_against": 0.0,
                }
        for agent, window in list(self._override_windows.items()):
            window["active"] = 1.0
            metrics = mechanics.get(agent, {})
            if float(metrics.get("aerial_useful_touches", 0.0)) > 0.0:
                window["useful_touch"] = 1.0
            car = self.rlgym_env.state.cars.get(agent)
            if self.rlgym_env.state.goal_scored and car is not None:
                if self.rlgym_env.state.scoring_team == car.team_num:
                    window["goal_for"] = 1.0
                else:
                    window["goal_against"] = 1.0
            window["remaining"] -= 1.0
        info["state"] = build_m08_metric_vector(
            self.rlgym_env.state,
            self.rlgym_env.shared_info,
            self._override_windows,
        )
        for agent, window in list(self._override_windows.items()):
            window["useful_touch"] = 0.0
            window["goal_for"] = 0.0
            window["goal_against"] = 0.0
            if window["remaining"] <= 0.0:
                del self._override_windows[agent]
        return observations, rewards, done, truncated, info


def make_campaign_gym_env(stage_name: str) -> CampaignGymWrapper:
    return CampaignGymWrapper(build_campaign_env(stage_name))


def make_natural_campaign_gym_env() -> CampaignGymWrapper:
    return CampaignGymWrapper(build_campaign_env("stage_a", natural_only=True))


def build_dual_rate_env(
    *,
    seed: int = 20260823,
    natural_only: bool = False,
    mechanics_disabled: bool = False,
    force_pass: bool = False,
    anchor_team: int | None = None,
) -> RLGym:
    """Build the explicit 8-tick strategic/4-tick mechanics environment."""
    config = load_milestone08_config()
    weights = dict(config["environment"]["curriculum_weights"])
    if natural_only:
        weights = {name: float(name == "natural") for name in CURRICULUM_FAMILIES}
    # Controller rows already encode both temporal delays; the transition engine
    # must apply each row before its physics tick without adding another delay.
    transition_engine = RocketSimEngine(rlbot_delay=False)
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            RivalCurriculumMutator(weights, seed=seed),
        ),
        obs_builder=WispCompatibleObs(seed=seed + 1),
        action_parser=DualRateActionParser(
            mechanics_disabled=mechanics_disabled,
            force_pass=force_pass,
            anchor_team=anchor_team,
            seed=seed + 2,
        ),
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


def make_dual_rate_gym_env() -> CampaignGymWrapper:
    return M08GymWrapper(build_dual_rate_env())


def make_dual_rate_pass_gym_env() -> CampaignGymWrapper:
    return M08GymWrapper(build_dual_rate_env(force_pass=True))


def make_dual_rate_anchor_blue_gym_env() -> CampaignGymWrapper:
    return M08GymWrapper(build_dual_rate_env(anchor_team=0))


def make_dual_rate_anchor_orange_gym_env() -> CampaignGymWrapper:
    return M08GymWrapper(build_dual_rate_env(anchor_team=1))


def dual_rate_environment_metadata() -> dict[str, Any]:
    config = load_milestone08_config()
    return {
        "schema_version": 1,
        "environment_version": DUAL_RATE_ENVIRONMENT_VERSION,
        "distribution": "majority-natural dual-rate 1v1 self-play",
        "curriculum_weights": config["environment"]["curriculum_weights"],
        "transition_engine": "rlgym.rocket_league.sim.RocketSimEngine",
        "rlbot_action_delay": False,
        "renderer": None,
        "physics_tick_rate_hz": 120,
        "strategic_cadence_ticks": 8,
        "mechanics_cadence_ticks": 4,
        "mechanics_action_count": 69,
        "pass_index": 0,
        "anchor_modes": ["self_play", "frozen_wisp_blue", "frozen_wisp_orange"],
        "reward": reward_v2_metadata(),
    }


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
