"""Native-120-Hz Rival v10.1 agency-bootstrap environment seam."""

from __future__ import annotations

import multiprocessing
from typing import Any

from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

from .v10_bootstrap_curriculum import RivalAgencyBootstrapCurriculumV1
from .v10_bootstrap_metrics import RivalAgencyBootstrapMetricTrackerV1
from .v10_bootstrap_reward import RivalAgencyBootstrapRewardV1
from .v9_environment import RivalObsV1RLGymBuilder, RivalV9ContinuousGymWrapper
from .v9_symmetry import RivalEpisodeSymmetryActionParser


ENVIRONMENT_VERSION = "RivalAgencyBootstrapEnvV1"
NO_TOUCH_TIMEOUT_SECONDS = 10.0
EPISODE_TIMEOUT_SECONDS = 120.0


class RivalAgencyBootstrapGymWrapperV1(RivalV9ContinuousGymWrapper):
    """Transport bootstrap metrics without changing rollout observations/actions."""

    def __init__(self, rlgym_env: RLGym) -> None:
        self.metric_tracker = RivalAgencyBootstrapMetricTrackerV1()
        self.last_raw_touch_tick = 0
        self.episode_start_tick = 0
        super().__init__(rlgym_env)
        self.metric_tracker.reset(self.rlgym_env.state, self.rlgym_env.shared_info)

    def reset(self):
        observations = super().reset()
        tick = int(self.rlgym_env.state.tick_count)
        self.last_raw_touch_tick = tick
        self.episode_start_tick = tick
        self.metric_tracker.reset(self.rlgym_env.state, self.rlgym_env.shared_info)
        return observations

    def step(self, actions):
        observations, rewards, done, truncated, info = super().step(actions)
        state = self.rlgym_env.state
        tick = int(state.tick_count)
        if any(int(car.ball_touches) > 0 for car in state.cars.values()):
            self.last_raw_touch_tick = tick
        reason: str | None = None
        if bool(done):
            reason = "goal"
        elif bool(truncated):
            no_touch_ticks = tick - self.last_raw_touch_tick
            reason = (
                "no_touch_timeout"
                if no_touch_ticks >= int(NO_TOUCH_TIMEOUT_SECONDS * 120) - 1
                else "episode_timeout"
            )
        info["state"] = self.metric_tracker.build(
            state,
            self.rlgym_env.shared_info,
            termination_reason=reason,
        )
        return observations, rewards, done, truncated, info


def build_v10_bootstrap_env(
    *,
    phase: str = "A",
    seed: int = 20261001,
    forced_family: str | None = None,
    forced_active_team: int | None = None,
    forced_mirror: bool | None = None,
    renderer: Any | None = None,
) -> RLGym:
    """Build one versioned interaction-dense 1v1 RocketSim environment."""

    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            RivalAgencyBootstrapCurriculumV1(
                phase,
                seed=seed,
                forced_family=forced_family,
                forced_active_team=forced_active_team,
            ),
        ),
        obs_builder=RivalObsV1RLGymBuilder(
            prediction_refresh_ticks=1,
            apply_episode_mirror=True,
        ),
        action_parser=RivalEpisodeSymmetryActionParser(
            mirror_probability=0.5,
            seed=seed + 100_000,
            forced_mirror=forced_mirror,
        ),
        reward_fn=RivalAgencyBootstrapRewardV1(),
        transition_engine=RocketSimEngine(rlbot_delay=True),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            NoTouchTimeoutCondition(NO_TOUCH_TIMEOUT_SECONDS),
            TimeoutCondition(EPISODE_TIMEOUT_SECONDS),
        ),
        shared_info_provider=None,
        renderer=renderer,
    )


def _worker_offset() -> int:
    identity = multiprocessing.current_process()._identity  # noqa: SLF001
    return int(identity[-1]) if identity else 0


def make_v10_phase_a_gym_env() -> RivalAgencyBootstrapGymWrapperV1:
    return RivalAgencyBootstrapGymWrapperV1(
        build_v10_bootstrap_env(phase="A", seed=20261001 + _worker_offset())
    )


def make_v10_phase_b_gym_env() -> RivalAgencyBootstrapGymWrapperV1:
    return RivalAgencyBootstrapGymWrapperV1(
        build_v10_bootstrap_env(phase="B", seed=20261001 + _worker_offset())
    )


def make_v10_phase_c_gym_env() -> RivalAgencyBootstrapGymWrapperV1:
    return RivalAgencyBootstrapGymWrapperV1(
        build_v10_bootstrap_env(phase="C", seed=20261001 + _worker_offset())
    )


ENV_FACTORY_BY_PHASE = {
    "A": make_v10_phase_a_gym_env,
    "B": make_v10_phase_b_gym_env,
    "C": make_v10_phase_c_gym_env,
}
