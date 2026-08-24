"""User-authorized Milestone 10.4 Stage-1-only acquisition reward."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState

from .v10_2_reward import (
    DISTANCE_PROGRESS_SAFETY_CLIP_UU,
    DISTANCE_PROGRESS_SCALE_UU,
    PHYSICAL_NEW_TOUCH_REWARD,
    RivalNewContactDetectorV1,
    _agent_for_team,
)


BALL_ACQUISITION_REWARD_VERSION = "RivalBallAcquisitionRewardV3"
TOWARD_BALL_EPISODE_BUDGET = 0.75
AWAY_BALL_EPISODE_BUDGET = -0.75
IDLE_VIOLATION_PENALTY = -0.80
IDLE_GRACE_SECONDS = 0.5
IDLE_SPEED_THRESHOLD_UU_PER_SECOND = 80.0
PHYSICS_HZ = 120
IDLE_GRACE_TICKS = int(IDLE_GRACE_SECONDS * PHYSICS_HZ)

COMPONENTS = (
    "distance_progress",
    "physical_new_touch",
    "pre_touch_idle",
    "goal_for",
    "goal_against",
)


@dataclass(frozen=True)
class BallAcquisitionTransitionV3:
    tick: int
    car_position: np.ndarray
    ball_position: np.ndarray
    car_linear_velocity: np.ndarray | None = None
    raw_touch_records: int = 0
    goal_for: bool = False
    goal_against: bool = False

    def __post_init__(self) -> None:
        car = np.asarray(self.car_position, dtype=np.float64)
        ball = np.asarray(self.ball_position, dtype=np.float64)
        velocity = np.asarray(
            np.zeros(3)
            if self.car_linear_velocity is None
            else self.car_linear_velocity,
            dtype=np.float64,
        )
        if car.shape != (3,) or ball.shape != (3,) or velocity.shape != (3,):
            raise ValueError("Car position, ball position, and velocity require shape (3,)")
        if not all(np.isfinite(value).all() for value in (car, ball, velocity)):
            raise FloatingPointError("Reward transition vectors must be finite")
        if int(self.raw_touch_records) < 0:
            raise ValueError("raw_touch_records cannot be negative")
        if self.goal_for and self.goal_against:
            raise ValueError("A transition cannot be goal-for and goal-against")


@dataclass(frozen=True)
class BallAcquisitionRewardStepV3:
    total: float
    components: Mapping[str, float]
    car_progress_unclipped_uu: float
    car_progress_clipped_uu: float
    distance_absolute_spend: float
    distance_budget_saturated: bool
    toward_ball_spend: float
    away_ball_spend: float
    toward_ball_budget_saturated: bool
    away_ball_budget_saturated: bool
    new_physical_touch: bool
    raw_touch_records: int
    learner_speed_uu_per_second: float
    idle_ticks: int
    idle_seconds: float
    idle_penalty: float
    cumulative_idle_ticks: int
    cumulative_idle_seconds: float
    cumulative_idle_penalty: float
    idle_penalty_saturated: bool
    idle_violation_triggered: bool
    first_touch_occurred: bool


class RivalBallAcquisitionRewardKernelV3:
    """Separate directional budgets plus timeout-calibrated pre-touch idle."""

    def __init__(self, *, safety_clip_uu: float = DISTANCE_PROGRESS_SAFETY_CLIP_UU) -> None:
        if not math.isfinite(safety_clip_uu) or safety_clip_uu <= 0.0:
            raise ValueError("safety_clip_uu must be finite and positive")
        self.safety_clip_uu = float(safety_clip_uu)
        self.previous: BallAcquisitionTransitionV3 | None = None
        self.reset_tick = 0
        self.toward_ball_spend = 0.0
        self.away_ball_spend = 0.0
        self.distance_total = 0.0
        self.touch_total = 0.0
        self.touch_count = 0
        self.first_touch_occurred = False
        self.idle_ticks = 0
        self.idle_seconds = 0.0
        self.idle_penalty_total = 0.0
        self.idle_violation_triggered = False
        self.touch_detector = RivalNewContactDetectorV1()

    @property
    def distance_absolute_spend(self) -> float:
        return self.toward_ball_spend + self.away_ball_spend

    @property
    def distance_budget_saturated(self) -> bool:
        return self.toward_ball_budget_saturated or self.away_ball_budget_saturated

    @property
    def toward_ball_budget_saturated(self) -> bool:
        return self.toward_ball_spend >= TOWARD_BALL_EPISODE_BUDGET - 1e-12

    @property
    def away_ball_budget_saturated(self) -> bool:
        return self.away_ball_spend >= abs(AWAY_BALL_EPISODE_BUDGET) - 1e-12

    @property
    def idle_penalty_saturated(self) -> bool:
        return self.idle_penalty_total <= IDLE_VIOLATION_PENALTY + 1e-12

    def reset(self, initial: BallAcquisitionTransitionV3) -> None:
        self.previous = initial
        self.reset_tick = int(initial.tick)
        self.toward_ball_spend = 0.0
        self.away_ball_spend = 0.0
        self.distance_total = 0.0
        self.touch_total = 0.0
        self.touch_count = 0
        self.first_touch_occurred = False
        self.idle_ticks = 0
        self.idle_seconds = 0.0
        self.idle_penalty_total = 0.0
        self.idle_violation_triggered = False
        self.touch_detector.reset()

    def _budget_distance(self, proposal: float) -> float:
        if proposal >= 0.0:
            remaining = max(0.0, TOWARD_BALL_EPISODE_BUDGET - self.toward_ball_spend)
            value = min(proposal, remaining)
            self.toward_ball_spend += value
            return value
        remaining = max(0.0, abs(AWAY_BALL_EPISODE_BUDGET) - self.away_ball_spend)
        magnitude = min(abs(proposal), remaining)
        self.away_ball_spend += magnitude
        return -magnitude

    def step(
        self, transition: BallAcquisitionTransitionV3
    ) -> BallAcquisitionRewardStepV3:
        if self.previous is None:
            raise RuntimeError("Reward kernel must be reset before stepping")
        if int(transition.tick) <= int(self.previous.tick):
            raise ValueError("Reward transitions require increasing native ticks")

        current_ball = np.asarray(transition.ball_position, dtype=np.float64)
        previous_car = np.asarray(self.previous.car_position, dtype=np.float64)
        current_car = np.asarray(transition.car_position, dtype=np.float64)
        progress_unclipped = float(
            np.linalg.norm(previous_car - current_ball)
            - np.linalg.norm(current_car - current_ball)
        )
        progress_clipped = float(
            np.clip(progress_unclipped, -self.safety_clip_uu, self.safety_clip_uu)
        )
        distance_reward = self._budget_distance(
            progress_clipped / DISTANCE_PROGRESS_SCALE_UU
        )

        new_touch = self.touch_detector.process(transition.raw_touch_records)
        touch_reward = PHYSICAL_NEW_TOUCH_REWARD if new_touch else 0.0
        learner_speed = float(
            np.linalg.norm(
                np.asarray(
                    np.zeros(3)
                    if transition.car_linear_velocity is None
                    else transition.car_linear_velocity,
                    dtype=np.float64,
                )
            )
        )
        eligible_start_tick = max(
            int(self.previous.tick), self.reset_tick + IDLE_GRACE_TICKS
        )
        eligible_ticks = max(0, int(transition.tick) - eligible_start_tick)
        idle_now = (
            not self.first_touch_occurred
            and not new_touch
            and learner_speed < IDLE_SPEED_THRESHOLD_UU_PER_SECOND
        )
        step_idle_ticks = eligible_ticks if idle_now else 0
        step_idle_seconds = step_idle_ticks / float(PHYSICS_HZ)
        idle_violation = bool(
            idle_now
            and step_idle_ticks > 0
            and not self.idle_violation_triggered
        )
        idle_penalty = IDLE_VIOLATION_PENALTY if idle_violation else 0.0
        self.idle_ticks += step_idle_ticks
        self.idle_seconds += step_idle_seconds
        self.idle_penalty_total += idle_penalty
        self.idle_violation_triggered = (
            self.idle_violation_triggered or idle_violation
        )
        if new_touch:
            self.first_touch_occurred = True

        components = {
            "distance_progress": distance_reward,
            "physical_new_touch": touch_reward,
            "pre_touch_idle": idle_penalty,
            "goal_for": 0.0,
            "goal_against": 0.0,
        }
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise FloatingPointError(f"Non-finite acquisition reward: {components}")
        if self.toward_ball_spend > TOWARD_BALL_EPISODE_BUDGET + 1e-12:
            raise RuntimeError("Toward-ball shaping exceeded its episode budget")
        if self.away_ball_spend > abs(AWAY_BALL_EPISODE_BUDGET) + 1e-12:
            raise RuntimeError("Away-ball shaping exceeded its episode budget")
        if self.idle_penalty_total < IDLE_VIOLATION_PENALTY - 1e-12:
            raise RuntimeError("Idle shaping exceeded its episode penalty")

        self.distance_total += distance_reward
        self.touch_total += touch_reward
        self.touch_count += int(new_touch)
        self.previous = transition
        return BallAcquisitionRewardStepV3(
            total=total,
            components=components,
            car_progress_unclipped_uu=progress_unclipped,
            car_progress_clipped_uu=progress_clipped,
            distance_absolute_spend=self.distance_absolute_spend,
            distance_budget_saturated=self.distance_budget_saturated,
            toward_ball_spend=self.toward_ball_spend,
            away_ball_spend=self.away_ball_spend,
            toward_ball_budget_saturated=self.toward_ball_budget_saturated,
            away_ball_budget_saturated=self.away_ball_budget_saturated,
            new_physical_touch=new_touch,
            raw_touch_records=max(0, int(transition.raw_touch_records)),
            learner_speed_uu_per_second=learner_speed,
            idle_ticks=step_idle_ticks,
            idle_seconds=step_idle_seconds,
            idle_penalty=idle_penalty,
            cumulative_idle_ticks=self.idle_ticks,
            cumulative_idle_seconds=self.idle_seconds,
            cumulative_idle_penalty=self.idle_penalty_total,
            idle_penalty_saturated=self.idle_penalty_saturated,
            idle_violation_triggered=self.idle_violation_triggered,
            first_touch_occurred=self.first_touch_occurred,
        )


class RivalBallAcquisitionRewardV3(RewardFunction[AgentID, GameState, float]):
    def __init__(self) -> None:
        self.active_agent: AgentID | None = None
        self.active_team: int | None = None
        self.kernel = RivalBallAcquisitionRewardKernelV3()

    @staticmethod
    def _transition(
        state: GameState,
        agent: AgentID,
        *,
        goal_for: bool = False,
        goal_against: bool = False,
    ) -> BallAcquisitionTransitionV3:
        car = state.cars[agent]
        return BallAcquisitionTransitionV3(
            tick=int(state.tick_count),
            car_position=np.asarray(car.physics.position, dtype=np.float64),
            ball_position=np.asarray(state.ball.position, dtype=np.float64),
            car_linear_velocity=np.asarray(car.physics.linear_velocity, dtype=np.float64),
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
        active_team = int(
            shared_info.get(
                "rival_v10_4_active_team",
                shared_info.get(
                    "rival_v10_3_active_team", shared_info["rival_v10_2_active_team"]
                ),
            )
        )
        active_agent = _agent_for_team(initial_state, active_team)
        if active_agent not in agents:
            raise RuntimeError("Active learner is missing from the RLGym agent set")
        self.active_team = active_team
        self.active_agent = active_agent
        self.kernel.reset(self._transition(initial_state, active_agent))
        shared_info["rival_v10_4_active_agent"] = active_agent
        shared_info["rival_v10_4_reward_version"] = BALL_ACQUISITION_REWARD_VERSION
        shared_info["rival_v10_3_active_agent"] = active_agent
        shared_info["rival_v10_2_active_agent"] = active_agent
        shared_info["reward_components"] = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        shared_info["rival_v10_4_reward_metrics"] = {}
        shared_info["rival_v10_3_reward_metrics"] = {}
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
        metrics = {
            "active_agent": self.active_agent,
            "active_team": self.active_team,
            "car_progress_unclipped_uu": step.car_progress_unclipped_uu,
            "car_progress_clipped_uu": step.car_progress_clipped_uu,
            "distance_absolute_spend": step.distance_absolute_spend,
            "distance_budget_saturated": step.distance_budget_saturated,
            "toward_ball_spend": step.toward_ball_spend,
            "away_ball_spend": step.away_ball_spend,
            "toward_ball_budget_saturated": step.toward_ball_budget_saturated,
            "away_ball_budget_saturated": step.away_ball_budget_saturated,
            "new_physical_touch": step.new_physical_touch,
            "raw_touch_records": step.raw_touch_records,
            "touch_count": self.kernel.touch_count,
            "distance_total": self.kernel.distance_total,
            "touch_total": self.kernel.touch_total,
            "learner_speed_uu_per_second": step.learner_speed_uu_per_second,
            "idle_ticks": step.idle_ticks,
            "idle_seconds": step.idle_seconds,
            "idle_penalty": step.idle_penalty,
            "cumulative_idle_ticks": step.cumulative_idle_ticks,
            "cumulative_idle_seconds": step.cumulative_idle_seconds,
            "cumulative_idle_penalty": step.cumulative_idle_penalty,
            "idle_penalty_saturated": step.idle_penalty_saturated,
            "idle_violation_triggered": step.idle_violation_triggered,
            "first_touch_occurred": step.first_touch_occurred,
        }
        shared_info["rival_v10_4_reward_metrics"] = metrics
        shared_info["rival_v10_3_reward_metrics"] = metrics
        shared_info["rival_v10_2_reward_metrics"] = metrics
        return {
            agent: step.total if agent == self.active_agent else 0.0
            for agent in agents
        }


def ball_acquisition_reward_metadata() -> dict[str, Any]:
    return {
        "reward_version": BALL_ACQUISITION_REWARD_VERSION,
        "distance_progress_formula": (
            "distance(previous_car,current_ball)-distance(current_car,current_ball)"
        ),
        "distance_progress_scale_uu": DISTANCE_PROGRESS_SCALE_UU,
        "distance_progress_safety_clip_uu": DISTANCE_PROGRESS_SAFETY_CLIP_UU,
        "toward_ball_episode_budget": TOWARD_BALL_EPISODE_BUDGET,
        "away_ball_episode_budget": AWAY_BALL_EPISODE_BUDGET,
        "physical_new_touch_reward": PHYSICAL_NEW_TOUCH_REWARD,
        "idle_grace_seconds": IDLE_GRACE_SECONDS,
        "idle_speed_threshold_uu_per_second": IDLE_SPEED_THRESHOLD_UU_PER_SECOND,
        "idle_violation_penalty": IDLE_VIOLATION_PENALTY,
        "idle_penalty_mode": "one_shot_on_first_stationary_tick_after_grace",
        "idle_only_before_first_touch": True,
        "generic_speed_reward": 0.0,
        "action_magnitude_reward": 0.0,
        "goal_for_reward": 0.0,
        "goal_against_reward": 0.0,
        "reads_controller_action": False,
        "future_skill_rewards": 0.0,
    }


def reward_truth_table_v3() -> dict[str, Any]:
    def transition(
        tick: int,
        car_x: float,
        ball_x: float = 5000.0,
        *,
        speed: float = 100.0,
        touches: int = 0,
    ) -> BallAcquisitionTransitionV3:
        return BallAcquisitionTransitionV3(
            tick=tick,
            car_position=np.asarray([car_x, 0.0, 17.0]),
            ball_position=np.asarray([ball_x, 0.0, 93.0]),
            car_linear_velocity=np.asarray([speed, 0.0, 0.0]),
            raw_touch_records=touches,
        )

    directional = RivalBallAcquisitionRewardKernelV3(safety_clip_uu=2300.0)
    directional.reset(transition(0, 0.0))
    toward_1 = directional.step(transition(1, 1000.0))
    toward_2 = directional.step(transition(2, 2000.0))
    away_1 = directional.step(transition(3, 1000.0))
    away_2 = directional.step(transition(4, 0.0))

    idle = RivalBallAcquisitionRewardKernelV3()
    idle.reset(transition(0, 0.0, speed=0.0))
    grace = idle.step(transition(IDLE_GRACE_TICKS, 0.0, speed=0.0))
    violation = idle.step(transition(IDLE_GRACE_TICKS + 1, 0.0, speed=0.0))
    repeated_idle = idle.step(transition(IDLE_GRACE_TICKS + 2, 0.0, speed=0.0))

    touch = RivalBallAcquisitionRewardKernelV3()
    touch.reset(transition(0, 0.0, speed=0.0))
    first_touch = touch.step(transition(61, 0.0, speed=0.0, touches=1))
    touch.step(transition(62, 0.0, speed=0.0))
    after_touch = touch.step(transition(1440, 0.0, speed=0.0))

    ball_motion = RivalBallAcquisitionRewardKernelV3()
    ball_motion.reset(transition(0, 0.0, 1000.0))
    ball_toward = ball_motion.step(transition(1, 0.0, 500.0))

    checks = {
        "touch_pays_exactly_one": (
            first_touch.components["physical_new_touch"] == 1.0
        ),
        "toward_ball_is_capped_at_positive_0p75": math.isclose(
            directional.toward_ball_spend, 0.75, rel_tol=0.0, abs_tol=1e-12
        )
        and toward_1.components["distance_progress"] > 0.0
        and toward_2.components["distance_progress"] > 0.0,
        "away_ball_is_separately_capped_at_negative_0p75": math.isclose(
            directional.away_ball_spend, 0.75, rel_tol=0.0, abs_tol=1e-12
        )
        and away_1.components["distance_progress"] < 0.0
        and away_2.components["distance_progress"] < 0.0,
        "idle_grace_is_free": grace.components["pre_touch_idle"] == 0.0,
        "first_idle_tick_after_grace_immediately_pays_negative_0p80": math.isclose(
            idle.idle_penalty_total,
            -0.80,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and violation.idle_penalty == -0.80
        and violation.idle_penalty_saturated,
        "idle_penalty_is_one_shot_not_per_tick": repeated_idle.idle_penalty == 0.0,
        "idle_stops_on_and_after_touch": (
            first_touch.components["pre_touch_idle"] == 0.0
            and after_touch.components["pre_touch_idle"] == 0.0
        ),
        "ball_motion_toward_stationary_car_not_positive": ball_toward.total <= 0.0,
        "goals_speed_actions_future_skills_zero": (
            ball_acquisition_reward_metadata()["generic_speed_reward"] == 0.0
            and ball_acquisition_reward_metadata()["action_magnitude_reward"] == 0.0
            and ball_acquisition_reward_metadata()["goal_for_reward"] == 0.0
            and ball_acquisition_reward_metadata()["goal_against_reward"] == 0.0
            and ball_acquisition_reward_metadata()["future_skill_rewards"] == 0.0
        ),
    }
    checks["passed"] = all(checks.values())
    return {
        "truth_table_version": "RivalBallAcquisitionRewardV3TruthTableV1",
        "metadata": ball_acquisition_reward_metadata(),
        "observations": {
            "toward_1": dict(toward_1.components),
            "toward_2": dict(toward_2.components),
            "away_1": dict(away_1.components),
            "away_2": dict(away_2.components),
            "grace": dict(grace.components),
            "idle_violation": dict(violation.components),
            "repeated_idle": dict(repeated_idle.components),
            "first_touch": dict(first_touch.components),
            "after_touch": dict(after_touch.components),
            "ball_motion_toward_stationary_car": dict(ball_toward.components),
        },
        "checks": checks,
    }
