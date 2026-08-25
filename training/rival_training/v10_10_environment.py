"""Stage-1 environment that ends on the learner's first physical ball touch."""

from __future__ import annotations

import multiprocessing
from typing import Any

from rlgym.api import AgentID, DoneCondition, RLGym
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.done_conditions import AnyCondition, GoalCondition, TimeoutCondition
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, MutatorSequence

from .v10_2_environment import (
    EPISODE_TIMEOUT_SECONDS,
    NO_ACTIVE_LEARNER_TOUCH_TIMEOUT_SECONDS,
    RivalActiveLearnerNoTouchTimeoutV1,
    RivalSingleLearnerGymWrapperV1,
    _agent_for_team,
)
from .v10_2_reward import RivalNewContactDetectorV1
from .v10_3_curriculum import RivalBallAcquisitionCurriculumV2
from .v10_10_reward import RivalFirstTouchVelocityRewardV1
from .v9_environment import RivalObsV1RLGymBuilder
from .v9_symmetry import RivalEpisodeSymmetryActionParser


FIRST_TOUCH_ENVIRONMENT_VERSION = "RivalFirstTouchVelocityEnvV1"


class RivalActiveLearnerFirstTouchConditionV1(
    DoneCondition[AgentID, GameState]
):
    """Terminate all agent rows on the active learner's first new touch run."""

    def __init__(self) -> None:
        self.active_agent: AgentID | None = None
        self.detector = RivalNewContactDetectorV1()

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        active_team = int(shared_info["rival_v10_2_active_team"])
        self.active_agent = _agent_for_team(initial_state, active_team)
        if self.active_agent not in agents:
            raise RuntimeError("Active learner missing from first-touch agent set")
        self.detector.reset()

    def is_done(
        self,
        agents: list[AgentID],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, bool]:
        del shared_info
        if self.active_agent is None:
            raise RuntimeError("First-touch condition must be reset before stepping")
        new_contact = self.detector.process(
            int(state.cars[self.active_agent].ball_touches)
        )
        return {agent: new_contact for agent in agents}


class RivalSingleLearnerFirstTouchWrapperV1(RivalSingleLearnerGymWrapperV1):
    """Preserve one learner row while exposing exact first-touch termination."""

    def step(self, actions):
        observation, rewards, done, truncated, info = super().step(actions)
        metrics = self.rlgym_env.shared_info["rival_v10_10_reward_metrics"]
        if done and bool(metrics["new_physical_touch"]):
            info["rival_v10_2"]["termination_reason"] = "first_touch"
        return observation, rewards, done, truncated, info


def build_first_touch_velocity_env(
    *,
    phase: str = "A",
    seed: int = 202610101,
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
        reward_fn=RivalFirstTouchVelocityRewardV1(),
        transition_engine=RocketSimEngine(rlbot_delay=True),
        termination_cond=AnyCondition(
            RivalActiveLearnerFirstTouchConditionV1(),
            GoalCondition(),
        ),
        truncation_cond=AnyCondition(
            RivalActiveLearnerNoTouchTimeoutV1(
                NO_ACTIVE_LEARNER_TOUCH_TIMEOUT_SECONDS
            ),
            TimeoutCondition(EPISODE_TIMEOUT_SECONDS),
        ),
        shared_info_provider=None,
        renderer=renderer,
    )


def _worker_offset() -> int:
    identity = multiprocessing.current_process()._identity  # noqa: SLF001
    return int(identity[-1]) if identity else 0


def make_first_touch_velocity_phase_a_env() -> RivalSingleLearnerFirstTouchWrapperV1:
    return RivalSingleLearnerFirstTouchWrapperV1(
        build_first_touch_velocity_env(
            phase="A", seed=202610101 + _worker_offset()
        )
    )


FIRST_TOUCH_ENV_FACTORY_BY_PHASE = {
    "A": make_first_touch_velocity_phase_a_env,
}
