"""Native-120-Hz Rival v9 environment construction.

This module contains only the scratch environment boundary. Legacy Wisp
observation/action/reward classes are intentionally not imported.
"""

from __future__ import annotations

import math
import multiprocessing
from typing import Any, Mapping

import gym
import numpy as np
from rlgym.api import AgentID, ObsBuilder, RLGym, RewardFunction, StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_RESTING_HEIGHT
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    MutatorSequence,
)
from rlgym_ppo.util import RLGymV2GymWrapper

from .v9_actions import RivalActionV1Parser
from .v9_canonical import RocketSimCanonicalAdapterV1
from .v9_curriculum import (
    V9_PILOT_CURRICULUM_WEIGHTS,
    RivalV9PilotCurriculumMutator,
)
from .v9_metrics import RivalV9PilotMetricTracker
from .v9_observations import (
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    RivalObsV1Builder,
    observation_schema_manifest,
)
from .v9_rewards import RivalScratchRewardV1
from .v9_symmetry import (
    RivalEpisodeSymmetryActionParser,
    mirror_canonical_state,
)


V9_ENVIRONMENT_VERSION = "RivalScratch1v1RocketSimV2OneTickDelay"
V9_TRAINING_ENVIRONMENT_VERSION = "RivalScratch1v1RocketSimV3OneTickDelayEpisodeSymmetry"
V9_PILOT_ENVIRONMENT_VERSION = (
    "RivalScratch1v1RocketSimV4OneTickDelayEpisodeSymmetryPilotCurriculum"
)


class RivalV9ContinuousGymWrapper(RLGymV2GymWrapper):
    """Repair rlgym-ppo's discrete-only v2 wrapper action-space discovery."""

    def __init__(self, rlgym_env: RLGym) -> None:
        super().__init__(rlgym_env)
        self.is_discrete = False
        # rlgym-ppo 1.3.13 uses an exact type(Box) comparison when reporting
        # action-space kind. Keep this as the concrete Gym Box; worker-specific
        # curriculum/symmetry seeds come from the process identity in the
        # pickle-safe factories below.
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0] * 5 + [0.0] * 3, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )


class RivalV9PilotGymWrapper(RivalV9ContinuousGymWrapper):
    """Transport fixed diagnostic vectors without altering rollout behavior."""

    def __init__(self, rlgym_env: RLGym) -> None:
        self.metric_tracker = RivalV9PilotMetricTracker()
        super().__init__(rlgym_env)
        self.metric_tracker.reset(self.rlgym_env.state)

    def reset(self):
        observations = super().reset()
        self.metric_tracker.reset(self.rlgym_env.state)
        return observations

    def step(self, actions):
        observations, rewards, done, truncated, info = super().step(actions)
        info["state"] = self.metric_tracker.build(self.rlgym_env.state, self.rlgym_env.shared_info)
        return observations, rewards, done, truncated, info


class RivalV9DeterministicKickoffMutator(StateMutator[GameState]):
    """Fixed standard 1v1 kickoff used to isolate diagnostic variables."""

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        state.ball.position = np.asarray([0.0, 0.0, BALL_RESTING_HEIGHT], dtype=np.float32)
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        for car in state.cars.values():
            if int(car.team_num) == 0:
                position = (0.0, -4608.0, 17.0)
                yaw = math.pi / 2.0
            else:
                position = (0.0, 4608.0, 17.0)
                yaw = -math.pi / 2.0
            car.physics.position = np.asarray(position, dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.physics.euler_angles = np.asarray([0.0, yaw, 0.0], dtype=np.float32)
            car.boost_amount = 33.3
        shared_info["kickoff"] = True


class RivalObsV1RLGymBuilder(ObsBuilder[AgentID, np.ndarray, GameState, tuple[str, int]]):
    """Training-only thin adapter around the shared canonical observation."""

    def __init__(
        self,
        *,
        prediction_refresh_ticks: int = 4,
        apply_episode_mirror: bool = False,
    ) -> None:
        self.prediction_refresh_ticks = int(prediction_refresh_ticks)
        self.apply_episode_mirror = bool(apply_episode_mirror)
        self.adapter = RocketSimCanonicalAdapterV1()
        self.builders: dict[AgentID, RivalObsV1Builder] = {}

    def get_obs_space(self, agent: AgentID) -> tuple[str, int]:
        del agent
        return "real", OBSERVATION_SIZE

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        del initial_state
        self.adapter.reset()
        self.builders = {
            agent: RivalObsV1Builder(prediction_refresh_ticks=self.prediction_refresh_ticks)
            for agent in agents
        }
        shared_info["rival_observation_version"] = OBSERVATION_VERSION
        shared_info["rival_observation_schema_sha256"] = observation_schema_manifest()[
            "schema_sha256"
        ]
        shared_info["rival_prediction_refresh_ticks"] = self.prediction_refresh_ticks
        shared_info["rival_v9_observation_episode_mirror_enabled"] = self.apply_episode_mirror

    def build_obs(
        self,
        agents: list[AgentID],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, np.ndarray]:
        observations: dict[AgentID, np.ndarray] = {}
        per_agent: dict[AgentID, dict[str, float | bool]] = {}
        for agent in agents:
            canonical = self.adapter.adapt(state, agent, shared_info)
            if self.apply_episode_mirror and bool(
                shared_info.get("rival_v9_episode_mirror", False)
            ):
                canonical = mirror_canonical_state(canonical)
            observation = self.builders[agent].build(canonical)
            observations[agent] = observation
            per_agent[agent] = dict(self.builders[agent].last_timings)
        shared_info["rival_v9_observation_timings"] = per_agent
        return observations


class RivalV9ZeroReward(RewardFunction[AgentID, GameState, float]):
    """Explicit zero reward for pre-reward diagnostic gates."""

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        del initial_state
        shared_info["rival_v9_reward_mode"] = "diagnostic_zero"
        shared_info["reward_components"] = {agent: {"diagnostic_zero": 0.0} for agent in agents}

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        del state, is_terminated, is_truncated, shared_info
        return {agent: 0.0 for agent in agents}


def build_v9_diagnostic_env(
    *,
    prediction_refresh_ticks: int = 4,
    no_touch_timeout_seconds: float = 30.0,
    episode_timeout_seconds: float = 300.0,
    rlbot_delay: bool = True,
) -> RLGym:
    """Build the complete v9 action/observation path without learning/reward."""

    transition_engine = RocketSimEngine(rlbot_delay=rlbot_delay)
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            RivalV9DeterministicKickoffMutator(),
        ),
        obs_builder=RivalObsV1RLGymBuilder(prediction_refresh_ticks=prediction_refresh_ticks),
        action_parser=RivalActionV1Parser(),
        reward_fn=RivalV9ZeroReward(),
        transition_engine=transition_engine,
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(no_touch_timeout_seconds),
            TimeoutCondition(episode_timeout_seconds),
        ),
        shared_info_provider=None,
        renderer=None,
    )


def build_v9_training_env(
    *,
    prediction_refresh_ticks: int = 1,
    no_touch_timeout_seconds: float = 30.0,
    episode_timeout_seconds: float = 300.0,
    rlbot_delay: bool = True,
    mirror_probability: float = 0.5,
    symmetry_seed: int = 20260908,
    forced_mirror: bool | None = None,
    curriculum_weights: Mapping[str, float] | None = None,
    curriculum_seed: int = 20260913,
    renderer: Any | None = None,
) -> RLGym:
    """Build the complete one-tick scratch path including Reward V1.

    This is separate from :func:`build_v9_diagnostic_env` so Gates 1--6 keep
    their explicit zero-reward isolation.  Gate 8 and later training use this
    complete environment.
    """

    transition_engine = RocketSimEngine(rlbot_delay=rlbot_delay)
    reset_mutator: StateMutator[GameState] = (
        RivalV9DeterministicKickoffMutator()
        if curriculum_weights is None
        else RivalV9PilotCurriculumMutator(curriculum_weights, seed=curriculum_seed)
    )
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            reset_mutator,
        ),
        obs_builder=RivalObsV1RLGymBuilder(
            prediction_refresh_ticks=prediction_refresh_ticks,
            apply_episode_mirror=True,
        ),
        action_parser=RivalEpisodeSymmetryActionParser(
            mirror_probability=mirror_probability,
            seed=symmetry_seed,
            forced_mirror=forced_mirror,
        ),
        reward_fn=RivalScratchRewardV1(),
        transition_engine=transition_engine,
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(no_touch_timeout_seconds),
            TimeoutCondition(episode_timeout_seconds),
        ),
        shared_info_provider=None,
        renderer=renderer,
    )


def build_v9_pilot_env(
    *,
    seed: int = 20260913,
    forced_mirror: bool | None = None,
    renderer: Any | None = None,
) -> RLGym:
    """Build the versioned majority-natural Gate 13 pilot distribution."""

    return build_v9_training_env(
        symmetry_seed=seed + 100_000,
        forced_mirror=forced_mirror,
        curriculum_weights=V9_PILOT_CURRICULUM_WEIGHTS,
        curriculum_seed=seed,
        renderer=renderer,
    )


def make_v9_training_gym_env() -> RivalV9ContinuousGymWrapper:
    """Pickle-safe actual-v9 worker factory for rlgym-ppo.

    ``BatchedAgentManager`` does not pass its per-process seed into the factory,
    so derive only the augmentation RNG offset from multiprocessing's stable
    worker identity. RocketSim state remains governed by the environment's
    explicit reset/curriculum implementation.
    """

    identity = multiprocessing.current_process()._identity  # noqa: SLF001
    worker_offset = int(identity[-1]) if identity else 0
    return RivalV9ContinuousGymWrapper(
        build_v9_training_env(symmetry_seed=20260908 + worker_offset)
    )


def make_v9_pilot_gym_env() -> RivalV9PilotGymWrapper:
    """Pickle-safe Gate 13 worker with curriculum and metric transport."""

    identity = multiprocessing.current_process()._identity  # noqa: SLF001
    worker_offset = int(identity[-1]) if identity else 0
    return RivalV9PilotGymWrapper(build_v9_pilot_env(seed=20260913 + worker_offset))
