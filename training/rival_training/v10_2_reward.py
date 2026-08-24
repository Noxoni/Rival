"""Milestone 10.2 prerequisite rewards.

Stage 1 deliberately contains only car-caused progress toward the current ball
position and genuine new learner-ball contacts.  The pure kernel is kept
separate from the RLGym adapter so the reward contract can be exhaustively
truth-table tested without a simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState


BALL_ACQUISITION_REWARD_VERSION = "RivalBallAcquisitionRewardV1"
DISTANCE_PROGRESS_SCALE_UU = 2300.0
DISTANCE_PROGRESS_ABSOLUTE_EPISODE_BUDGET = 0.75
# This provisional safety ceiling is over three times the maximum ordinary
# 2300 uu/s native-tick displacement.  Gate 2 measures legal transition tails
# before real training and refuses the campaign if this would truncate them.
DISTANCE_PROGRESS_SAFETY_CLIP_UU = 64.0
PHYSICAL_NEW_TOUCH_REWARD = 1.0

COMPONENTS = (
    "distance_progress",
    "physical_new_touch",
    "goal_for",
    "goal_against",
)


@dataclass(frozen=True)
class BallAcquisitionTransitionV1:
    """The only state needed by the pure Stage-1 reward kernel."""

    tick: int
    car_position: np.ndarray
    ball_position: np.ndarray
    raw_touch_records: int = 0
    goal_for: bool = False
    goal_against: bool = False

    def __post_init__(self) -> None:
        car = np.asarray(self.car_position, dtype=np.float64)
        ball = np.asarray(self.ball_position, dtype=np.float64)
        if car.shape != (3,) or ball.shape != (3,):
            raise ValueError("Car and ball positions must each have shape (3,)")
        if not np.isfinite(car).all() or not np.isfinite(ball).all():
            raise FloatingPointError("Reward positions must be finite")
        if int(self.raw_touch_records) < 0:
            raise ValueError("raw_touch_records cannot be negative")
        if self.goal_for and self.goal_against:
            raise ValueError("A transition cannot be goal-for and goal-against")


@dataclass(frozen=True)
class BallAcquisitionRewardStepV1:
    total: float
    components: Mapping[str, float]
    car_progress_unclipped_uu: float
    car_progress_clipped_uu: float
    distance_absolute_spend: float
    distance_budget_saturated: bool
    new_physical_touch: bool
    raw_touch_records: int


class RivalNewContactDetectorV1:
    """Convert authoritative per-tick RocketSim touches into new contacts.

    RLGym exposes ``Car.ball_touches`` for the current physics transition.  A
    run of non-zero records is one sustained contact.  A subsequent non-zero
    record is a new contact only after RocketSim emitted at least one zero-touch
    transition, so rapid genuinely separated retouches are not suppressed by
    an arbitrary time debounce.
    """

    def __init__(self) -> None:
        self._raw_contact_active = False

    @property
    def raw_contact_active(self) -> bool:
        return self._raw_contact_active

    def reset(self) -> None:
        self._raw_contact_active = False

    def process(self, raw_touch_records: int) -> bool:
        raw = max(0, int(raw_touch_records))
        new_contact = raw > 0 and not self._raw_contact_active
        self._raw_contact_active = raw > 0
        return new_contact


class RivalBallAcquisitionRewardKernelV1:
    """Pure native-120-Hz Stage-1 reward implementation."""

    def __init__(self, *, safety_clip_uu: float = DISTANCE_PROGRESS_SAFETY_CLIP_UU) -> None:
        if not math.isfinite(safety_clip_uu) or safety_clip_uu <= 0.0:
            raise ValueError("safety_clip_uu must be finite and positive")
        self.safety_clip_uu = float(safety_clip_uu)
        self.previous: BallAcquisitionTransitionV1 | None = None
        self.distance_absolute_spend = 0.0
        self.distance_total = 0.0
        self.touch_total = 0.0
        self.touch_count = 0
        self.distance_budget_saturated = False
        self.touch_detector = RivalNewContactDetectorV1()

    def reset(self, initial: BallAcquisitionTransitionV1) -> None:
        self.previous = initial
        self.distance_absolute_spend = 0.0
        self.distance_total = 0.0
        self.touch_total = 0.0
        self.touch_count = 0
        self.distance_budget_saturated = False
        self.touch_detector.reset()

    def _budget_distance(self, proposal: float) -> float:
        remaining = max(
            0.0,
            DISTANCE_PROGRESS_ABSOLUTE_EPISODE_BUDGET
            - self.distance_absolute_spend,
        )
        value = math.copysign(min(abs(proposal), remaining), proposal)
        self.distance_absolute_spend += abs(value)
        self.distance_budget_saturated = (
            self.distance_absolute_spend
            >= DISTANCE_PROGRESS_ABSOLUTE_EPISODE_BUDGET - 1e-12
        )
        return value

    def step(
        self, transition: BallAcquisitionTransitionV1
    ) -> BallAcquisitionRewardStepV1:
        if self.previous is None:
            raise RuntimeError("Reward kernel must be reset before stepping")
        if int(transition.tick) <= int(self.previous.tick):
            raise ValueError("Reward transitions require increasing native ticks")

        # The current ball position is intentionally used in both distances.
        # Ball motion alone therefore cannot pay the learner.
        current_ball = np.asarray(transition.ball_position, dtype=np.float64)
        previous_car = np.asarray(self.previous.car_position, dtype=np.float64)
        current_car = np.asarray(transition.car_position, dtype=np.float64)
        progress_unclipped = float(
            np.linalg.norm(previous_car - current_ball)
            - np.linalg.norm(current_car - current_ball)
        )
        progress_clipped = float(
            np.clip(
                progress_unclipped,
                -self.safety_clip_uu,
                self.safety_clip_uu,
            )
        )
        distance_reward = self._budget_distance(
            progress_clipped / DISTANCE_PROGRESS_SCALE_UU
        )
        new_touch = self.touch_detector.process(transition.raw_touch_records)
        touch_reward = PHYSICAL_NEW_TOUCH_REWARD if new_touch else 0.0

        # Goals are diagnostics/termination only in Stage 1.
        components = {
            "distance_progress": distance_reward,
            "physical_new_touch": touch_reward,
            "goal_for": 0.0,
            "goal_against": 0.0,
        }
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise FloatingPointError(f"Non-finite acquisition reward: {components}")
        if (
            self.distance_absolute_spend
            > DISTANCE_PROGRESS_ABSOLUTE_EPISODE_BUDGET + 1e-12
        ):
            raise RuntimeError("Distance shaping exceeded its episode budget")

        self.distance_total += distance_reward
        self.touch_total += touch_reward
        self.touch_count += int(new_touch)
        self.previous = transition
        return BallAcquisitionRewardStepV1(
            total=total,
            components=components,
            car_progress_unclipped_uu=progress_unclipped,
            car_progress_clipped_uu=progress_clipped,
            distance_absolute_spend=self.distance_absolute_spend,
            distance_budget_saturated=self.distance_budget_saturated,
            new_physical_touch=new_touch,
            raw_touch_records=max(0, int(transition.raw_touch_records)),
        )


def _agent_for_team(state: GameState, team: int) -> AgentID:
    matches = [
        agent
        for agent, car in state.cars.items()
        if int(car.team_num) == int(team)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one car for active team {team}, got {matches}")
    return matches[0]


class RivalBallAcquisitionRewardV1(RewardFunction[AgentID, GameState, float]):
    """RLGym adapter which pays only the episode's active learner."""

    def __init__(self) -> None:
        self.active_agent: AgentID | None = None
        self.active_team: int | None = None
        self.kernel = RivalBallAcquisitionRewardKernelV1()

    @staticmethod
    def _transition(
        state: GameState,
        agent: AgentID,
        *,
        goal_for: bool = False,
        goal_against: bool = False,
    ) -> BallAcquisitionTransitionV1:
        car = state.cars[agent]
        return BallAcquisitionTransitionV1(
            tick=int(state.tick_count),
            car_position=np.asarray(car.physics.position, dtype=np.float64),
            ball_position=np.asarray(state.ball.position, dtype=np.float64),
            raw_touch_records=max(0, int(car.ball_touches)),
            goal_for=goal_for,
            goal_against=goal_against,
        )

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        active_team = int(shared_info["rival_v10_2_active_team"])
        active_agent = _agent_for_team(initial_state, active_team)
        if active_agent not in agents:
            raise RuntimeError("Active learner is missing from the RLGym agent set")
        self.active_team = active_team
        self.active_agent = active_agent
        self.kernel.reset(self._transition(initial_state, active_agent))
        shared_info["rival_v10_2_active_agent"] = active_agent
        shared_info["rival_v10_2_reward_version"] = (
            BALL_ACQUISITION_REWARD_VERSION
        )
        shared_info["reward_components"] = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        shared_info["rival_v10_2_reward_metrics"] = {}

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        del is_terminated, is_truncated
        if self.active_agent is None or self.active_team is None:
            raise RuntimeError("Reward adapter must be reset before stepping")
        goal_now = bool(state.goal_scored)
        scoring_team = int(state.scoring_team) if goal_now else -1
        step = self.kernel.step(
            self._transition(
                state,
                self.active_agent,
                goal_for=goal_now and scoring_team == self.active_team,
                goal_against=goal_now and scoring_team != self.active_team,
            )
        )
        components = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        components[self.active_agent] = dict(step.components)
        shared_info["reward_components"] = components
        shared_info["rival_v10_2_reward_metrics"] = {
            "active_agent": self.active_agent,
            "active_team": self.active_team,
            "car_progress_unclipped_uu": step.car_progress_unclipped_uu,
            "car_progress_clipped_uu": step.car_progress_clipped_uu,
            "distance_absolute_spend": step.distance_absolute_spend,
            "distance_budget_saturated": step.distance_budget_saturated,
            "new_physical_touch": step.new_physical_touch,
            "raw_touch_records": step.raw_touch_records,
            "touch_count": self.kernel.touch_count,
            "distance_total": self.kernel.distance_total,
            "touch_total": self.kernel.touch_total,
        }
        return {
            agent: step.total if agent == self.active_agent else 0.0
            for agent in agents
        }


def ball_acquisition_reward_metadata() -> dict[str, Any]:
    return {
        "reward_version": BALL_ACQUISITION_REWARD_VERSION,
        "distance_progress_formula": (
            "distance(previous_car,current_ball)-"
            "distance(current_car,current_ball)"
        ),
        "distance_progress_scale_uu": DISTANCE_PROGRESS_SCALE_UU,
        "distance_progress_safety_clip_uu": (
            DISTANCE_PROGRESS_SAFETY_CLIP_UU
        ),
        "distance_progress_absolute_episode_budget": (
            DISTANCE_PROGRESS_ABSOLUTE_EPISODE_BUDGET
        ),
        "physical_new_touch_reward": PHYSICAL_NEW_TOUCH_REWARD,
        "touch_detection": (
            "authoritative_raw_touch_run_with_zero-record_separation"
        ),
        "speed_reward": 0.0,
        "goal_for_reward": 0.0,
        "goal_against_reward": 0.0,
        "reads_controller_action": False,
        "future_skill_rewards": 0.0,
    }
