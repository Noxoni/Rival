"""Stage-1-only RocketSim environment for the V4 reward experiment."""

from __future__ import annotations

import multiprocessing
from typing import Any

from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, TimeoutCondition
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

from .v10_2_environment import (
    EPISODE_TIMEOUT_SECONDS,
    NO_ACTIVE_LEARNER_TOUCH_TIMEOUT_SECONDS,
    RivalActiveLearnerNoTouchTimeoutV1,
    RivalSingleLearnerGymWrapperV1,
)
from .v10_3_curriculum import RivalBallAcquisitionCurriculumV2
from .v10_5_reward import RivalBallAcquisitionRewardV4
from .v9_environment import RivalObsV1RLGymBuilder
from .v9_symmetry import RivalEpisodeSymmetryActionParser


BALL_ACQUISITION_ENVIRONMENT_VERSION = "RivalBallAcquisitionEnvV4"


class RivalSingleLearnerGymWrapperV4(RivalSingleLearnerGymWrapperV1):
    """V4 marker around the unchanged learner/dummy isolation wrapper."""


def build_ball_acquisition_env(
    *,
    phase: str = "A",
    seed: int = 20261051,
    forced_family: str | None = None,
    forced_active_team: int | None = None,
    forced_mirror: bool | None = None,
    renderer: Any | None = None,
) -> RLGym:
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            RivalBallAcquisitionCurriculumV2(
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
        reward_fn=RivalBallAcquisitionRewardV4(),
        transition_engine=RocketSimEngine(rlbot_delay=True),
        termination_cond=GoalCondition(),
        truncation_cond=AnyCondition(
            RivalActiveLearnerNoTouchTimeoutV1(NO_ACTIVE_LEARNER_TOUCH_TIMEOUT_SECONDS),
            TimeoutCondition(EPISODE_TIMEOUT_SECONDS),
        ),
        shared_info_provider=None,
        renderer=renderer,
    )


def _worker_offset() -> int:
    identity = multiprocessing.current_process()._identity  # noqa: SLF001
    return int(identity[-1]) if identity else 0


def make_ball_acquisition_phase_a_env() -> RivalSingleLearnerGymWrapperV4:
    return RivalSingleLearnerGymWrapperV4(
        build_ball_acquisition_env(phase="A", seed=20261051 + _worker_offset())
    )


def make_ball_acquisition_phase_b_env() -> RivalSingleLearnerGymWrapperV4:
    return RivalSingleLearnerGymWrapperV4(
        build_ball_acquisition_env(phase="B", seed=20261051 + _worker_offset())
    )


BALL_ACQUISITION_ENV_FACTORY_BY_PHASE = {
    "A": make_ball_acquisition_phase_a_env,
    "B": make_ball_acquisition_phase_b_env,
}
