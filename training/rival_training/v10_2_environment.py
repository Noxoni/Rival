"""One-trainable-learner RocketSim environment for Rival v10.2 Stage 1."""

from __future__ import annotations

from contextlib import redirect_stdout
import io
import multiprocessing
from typing import Any

import gym
import numpy as np
from rlgym.api import AgentID, DoneCondition, RLGym
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    MutatorSequence,
)

from .v10_2_curriculum import RivalBallAcquisitionCurriculumV1
from .v10_2_reward import RivalBallAcquisitionRewardV1
from .v9_environment import RivalObsV1RLGymBuilder, RivalV9ContinuousGymWrapper
from .v9_symmetry import RivalEpisodeSymmetryActionParser


BALL_ACQUISITION_ENVIRONMENT_VERSION = "RivalBallAcquisitionEnvV1"
NO_ACTIVE_LEARNER_TOUCH_TIMEOUT_SECONDS = 12.0
EPISODE_TIMEOUT_SECONDS = 45.0
ZERO_CONTROLLER = np.zeros(8, dtype=np.float32)


def _agent_for_team(state: GameState, team: int) -> AgentID:
    matches = [
        agent
        for agent, car in state.cars.items()
        if int(car.team_num) == int(team)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one car for team {team}, got {matches}")
    return matches[0]


class RivalActiveLearnerNoTouchTimeoutV1(DoneCondition[AgentID, GameState]):
    """Timeout measured only from genuine active-learner raw touch records."""

    def __init__(self, timeout_seconds: float) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.active_agent: AgentID | None = None
        self.last_touch_tick: int | None = None

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        active_team = int(shared_info["rival_v10_2_active_team"])
        self.active_agent = _agent_for_team(initial_state, active_team)
        if self.active_agent not in agents:
            raise RuntimeError("Active learner missing from timeout agent set")
        self.last_touch_tick = int(initial_state.tick_count)

    def is_done(
        self,
        agents: list[AgentID],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, bool]:
        del shared_info
        if self.active_agent is None or self.last_touch_tick is None:
            raise RuntimeError("Timeout condition must be reset before stepping")
        if int(state.cars[self.active_agent].ball_touches) > 0:
            self.last_touch_tick = int(state.tick_count)
        elapsed = (int(state.tick_count) - self.last_touch_tick) / 120.0
        done = elapsed >= self.timeout_seconds
        return {agent: done for agent in agents}


class RivalSingleLearnerGymWrapperV1(RivalV9ContinuousGymWrapper):
    """Expose only the active learner to rlgym-ppo.

    The wrapped RLGym environment still contains and observes two cars.  This
    wrapper returns one observation/reward trajectory and injects an exact-zero
    action for the inert dummy before the simulator step.  Consequently no
    dummy observation, action, log-probability, reward, advantage, return, or
    loss row can enter the worker manager.
    """

    def __init__(self, rlgym_env: RLGym) -> None:
        self.active_agent: AgentID | None = None
        self.dummy_agent: AgentID | None = None
        self.dummy_actions_injected = 0
        self.dummy_nonzero_actions_injected = 0
        self.learner_steps_returned = 0
        self.episode_start_tick = 0
        self.last_active_touch_tick = 0
        # rlgym-ppo 1.3.13 prints an unconditional discovery warning for the
        # deliberate one-time reset in its wrapper constructor.  The behavior
        # is retained, but suppress the message so hundreds of deterministic
        # evaluation environments do not flood evidence logs.
        with redirect_stdout(io.StringIO()):
            super().__init__(rlgym_env)
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0] * 5 + [0.0] * 3, dtype=np.float32),
            high=np.ones(8, dtype=np.float32),
            dtype=np.float32,
        )

    def _select_agents(self, observations: dict[AgentID, np.ndarray]) -> None:
        active_team = int(self.rlgym_env.shared_info["rival_v10_2_active_team"])
        self.active_agent = _agent_for_team(self.rlgym_env.state, active_team)
        dummies = [agent for agent in observations if agent != self.active_agent]
        if len(dummies) != 1:
            raise RuntimeError(
                f"Expected one inert dummy, found active={self.active_agent}, "
                f"agents={list(observations)}"
            )
        self.dummy_agent = dummies[0]
        self.agent_map = {0: self.active_agent}

    def reset(self):
        observations = self.rlgym_env.reset()
        self._select_agents(observations)
        tick = int(self.rlgym_env.state.tick_count)
        self.episode_start_tick = tick
        self.last_active_touch_tick = tick
        self.obs_buffer = np.asarray(
            [observations[self.active_agent]], dtype=np.float32
        )
        return self.obs_buffer

    def step(self, actions):
        if self.active_agent is None or self.dummy_agent is None:
            raise RuntimeError("Single-learner wrapper must be reset before step")
        physical = np.asarray(actions, dtype=np.float32)
        if physical.shape != (1, 8):
            raise ValueError(
                f"Single learner expects action shape (1, 8), got {physical.shape}"
            )
        dummy_action = ZERO_CONTROLLER.copy()
        self.dummy_actions_injected += 1
        self.dummy_nonzero_actions_injected += int(bool(np.any(dummy_action)))
        action_dict = {
            self.active_agent: physical[0],
            self.dummy_agent: dummy_action,
        }
        observations, rewards, terminated, truncated = self.rlgym_env.step(
            action_dict
        )
        state = self.rlgym_env.state
        tick = int(state.tick_count)
        if int(state.cars[self.active_agent].ball_touches) > 0:
            self.last_active_touch_tick = tick
        active_done = bool(terminated[self.active_agent])
        active_truncated = bool(truncated[self.active_agent])
        if active_done:
            reason = "goal"
        elif active_truncated:
            no_touch_ticks = tick - self.last_active_touch_tick
            reason = (
                "no_touch_timeout"
                if no_touch_ticks
                >= int(NO_ACTIVE_LEARNER_TOUCH_TIMEOUT_SECONDS * 120) - 1
                else "episode_timeout"
            )
        else:
            reason = None
        self.obs_buffer = np.asarray(
            [observations[self.active_agent]], dtype=np.float32
        )
        self.learner_steps_returned += 1
        info = {
            "state": np.asarray(
                [
                    float(tick),
                    float(self.rlgym_env.shared_info["rival_v10_2_active_team"]),
                    float(
                        self.rlgym_env.shared_info[
                            "rival_v10_2_reward_metrics"
                        ].get("new_physical_touch", False)
                    ),
                    float(
                        self.rlgym_env.shared_info[
                            "rival_v10_2_reward_metrics"
                        ].get("car_progress_clipped_uu", 0.0)
                    ),
                ],
                dtype=np.float32,
            ),
            "rival_v10_2": {
                "active_agent": self.active_agent,
                "dummy_agent": self.dummy_agent,
                "dummy_action": dummy_action.tolist(),
                "dummy_rows_returned": 0,
                "termination_reason": reason,
                "reset_family": self.rlgym_env.shared_info[
                    "rival_v10_2_reset_family"
                ],
            },
        }
        return (
            self.obs_buffer,
            [float(rewards[self.active_agent])],
            active_done,
            active_truncated,
            info,
        )


def build_ball_acquisition_env(
    *,
    phase: str = "A",
    seed: int = 20261021,
    forced_family: str | None = None,
    forced_active_team: int | None = None,
    forced_mirror: bool | None = None,
    renderer: Any | None = None,
) -> RLGym:
    return RLGym(
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            RivalBallAcquisitionCurriculumV1(
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
        reward_fn=RivalBallAcquisitionRewardV1(),
        transition_engine=RocketSimEngine(rlbot_delay=True),
        termination_cond=GoalCondition(),
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


def make_ball_acquisition_phase_a_env() -> RivalSingleLearnerGymWrapperV1:
    return RivalSingleLearnerGymWrapperV1(
        build_ball_acquisition_env(
            phase="A",
            seed=20261021 + _worker_offset(),
        )
    )


def make_ball_acquisition_phase_b_env() -> RivalSingleLearnerGymWrapperV1:
    return RivalSingleLearnerGymWrapperV1(
        build_ball_acquisition_env(
            phase="B",
            seed=20261021 + _worker_offset(),
        )
    )


BALL_ACQUISITION_ENV_FACTORY_BY_PHASE = {
    "A": make_ball_acquisition_phase_a_env,
    "B": make_ball_acquisition_phase_b_env,
}
