"""Stage-1-only turn, approach, and first-touch reward hierarchy."""

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
    RivalNewContactDetectorV1,
    _agent_for_team,
)


BALL_ACQUISITION_REWARD_VERSION = "RivalBallAcquisitionRewardV4"
FIRST_PHYSICAL_TOUCH_REWARD = 10.0
HEADING_ALIGNMENT_DELTA_SCALE = 1.5
HEADING_POSITIVE_EPISODE_BUDGET = 3.0
HEADING_NEGATIVE_EPISODE_BUDGET = -3.0
APPROACH_POSITIVE_EPISODE_BUDGET = 3.0
APPROACH_NEGATIVE_EPISODE_BUDGET = -3.0
IDLE_GRACE_SECONDS = 0.5
IDLE_SPEED_THRESHOLD_UU_PER_SECOND = 80.0
IDLE_PENALTY_RATE_PER_SIMULATED_SECOND = -1.4
NO_TOUCH_TIMEOUT_SECONDS = 12.0
PHYSICS_HZ = 120
IDLE_GRACE_TICKS = int(IDLE_GRACE_SECONDS * PHYSICS_HZ)
IDLE_ELIGIBLE_SECONDS = NO_TOUCH_TIMEOUT_SECONDS - IDLE_GRACE_SECONDS
IDLE_EPISODE_FLOOR = IDLE_PENALTY_RATE_PER_SIMULATED_SECOND * IDLE_ELIGIBLE_SECONDS
MAXIMUM_POSITIVE_EPISODE_REWARD = (
    FIRST_PHYSICAL_TOUCH_REWARD + HEADING_POSITIVE_EPISODE_BUDGET + APPROACH_POSITIVE_EPISODE_BUDGET
)

COMPONENTS = (
    "heading_alignment",
    "distance_progress",
    "physical_new_touch",
    "pre_touch_idle",
    "goal_for",
    "goal_against",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return np.zeros(3, dtype=np.float64) if norm <= 1e-12 else value / norm


@dataclass(frozen=True)
class BallAcquisitionTransitionV4:
    tick: int
    car_position: np.ndarray
    ball_position: np.ndarray
    car_forward: np.ndarray
    car_linear_velocity: np.ndarray | None = None
    raw_touch_records: int = 0
    goal_for: bool = False
    goal_against: bool = False

    def __post_init__(self) -> None:
        car = np.asarray(self.car_position, dtype=np.float64)
        ball = np.asarray(self.ball_position, dtype=np.float64)
        forward = np.asarray(self.car_forward, dtype=np.float64)
        velocity = np.asarray(
            np.zeros(3) if self.car_linear_velocity is None else self.car_linear_velocity,
            dtype=np.float64,
        )
        if any(value.shape != (3,) for value in (car, ball, forward, velocity)):
            raise ValueError("Reward transition vectors require shape (3,)")
        if not all(np.isfinite(value).all() for value in (car, ball, forward, velocity)):
            raise FloatingPointError("Reward transition vectors must be finite")
        if float(np.linalg.norm(forward)) <= 1e-12:
            raise ValueError("car_forward must be non-zero")
        if int(self.raw_touch_records) < 0:
            raise ValueError("raw_touch_records cannot be negative")
        if self.goal_for and self.goal_against:
            raise ValueError("A transition cannot be goal-for and goal-against")


@dataclass(frozen=True)
class BallAcquisitionRewardStepV4:
    total: float
    components: Mapping[str, float]
    alignment_previous: float
    alignment_now: float
    alignment_delta: float
    heading_positive_spend: float
    heading_negative_spend: float
    heading_positive_budget_saturated: bool
    heading_negative_budget_saturated: bool
    car_progress_unclipped_uu: float
    car_progress_clipped_uu: float
    distance_absolute_spend: float
    distance_budget_saturated: bool
    approach_positive_spend: float
    approach_negative_spend: float
    approach_positive_budget_saturated: bool
    approach_negative_budget_saturated: bool
    new_physical_touch: bool
    rewarded_first_physical_touch: bool
    raw_touch_records: int
    learner_speed_uu_per_second: float
    idle_ticks: int
    idle_seconds: float
    idle_penalty: float
    cumulative_idle_ticks: int
    cumulative_idle_seconds: float
    cumulative_idle_penalty: float
    idle_penalty_saturated: bool
    first_touch_occurred: bool


class RivalBallAcquisitionRewardKernelV4:
    """Pure native-120-Hz implementation of the requested reward hierarchy."""

    def __init__(self, *, safety_clip_uu: float = DISTANCE_PROGRESS_SAFETY_CLIP_UU) -> None:
        if not math.isfinite(safety_clip_uu) or safety_clip_uu <= 0.0:
            raise ValueError("safety_clip_uu must be finite and positive")
        self.safety_clip_uu = float(safety_clip_uu)
        self.previous: BallAcquisitionTransitionV4 | None = None
        self.reset_tick = 0
        self.heading_positive_spend = 0.0
        self.heading_negative_spend = 0.0
        self.approach_positive_spend = 0.0
        self.approach_negative_spend = 0.0
        self.heading_total = 0.0
        self.distance_total = 0.0
        self.touch_total = 0.0
        self.touch_count = 0
        self.first_touch_occurred = False
        self.idle_ticks = 0
        self.idle_seconds = 0.0
        self.idle_penalty_total = 0.0
        self.touch_detector = RivalNewContactDetectorV1()

    @staticmethod
    def _alignment(transition: BallAcquisitionTransitionV4) -> float:
        direction = _unit(
            np.asarray(transition.ball_position, dtype=np.float64)
            - np.asarray(transition.car_position, dtype=np.float64)
        )
        forward = _unit(np.asarray(transition.car_forward, dtype=np.float64))
        return float(np.clip(np.dot(forward, direction), -1.0, 1.0))

    @property
    def distance_absolute_spend(self) -> float:
        return self.approach_positive_spend + self.approach_negative_spend

    @property
    def distance_budget_saturated(self) -> bool:
        return self.approach_positive_budget_saturated or self.approach_negative_budget_saturated

    @property
    def heading_positive_budget_saturated(self) -> bool:
        return self.heading_positive_spend >= HEADING_POSITIVE_EPISODE_BUDGET - 1e-12

    @property
    def heading_negative_budget_saturated(self) -> bool:
        return self.heading_negative_spend >= abs(HEADING_NEGATIVE_EPISODE_BUDGET) - 1e-12

    @property
    def approach_positive_budget_saturated(self) -> bool:
        return self.approach_positive_spend >= APPROACH_POSITIVE_EPISODE_BUDGET - 1e-12

    @property
    def approach_negative_budget_saturated(self) -> bool:
        return self.approach_negative_spend >= abs(APPROACH_NEGATIVE_EPISODE_BUDGET) - 1e-12

    @property
    def idle_penalty_saturated(self) -> bool:
        return self.idle_penalty_total <= IDLE_EPISODE_FLOOR + 1e-12

    def reset(self, initial: BallAcquisitionTransitionV4) -> None:
        self.previous = initial
        self.reset_tick = int(initial.tick)
        self.heading_positive_spend = 0.0
        self.heading_negative_spend = 0.0
        self.approach_positive_spend = 0.0
        self.approach_negative_spend = 0.0
        self.heading_total = 0.0
        self.distance_total = 0.0
        self.touch_total = 0.0
        self.touch_count = 0
        self.first_touch_occurred = False
        self.idle_ticks = 0
        self.idle_seconds = 0.0
        self.idle_penalty_total = 0.0
        self.touch_detector.reset()

    @staticmethod
    def _separate_budget(
        proposal: float,
        positive_spend: float,
        negative_spend: float,
        positive_budget: float,
        negative_budget: float,
    ) -> tuple[float, float, float]:
        if proposal >= 0.0:
            remaining = max(0.0, positive_budget - positive_spend)
            value = min(proposal, remaining)
            return value, positive_spend + value, negative_spend
        remaining = max(0.0, abs(negative_budget) - negative_spend)
        magnitude = min(abs(proposal), remaining)
        return -magnitude, positive_spend, negative_spend + magnitude

    def step(self, transition: BallAcquisitionTransitionV4) -> BallAcquisitionRewardStepV4:
        if self.previous is None:
            raise RuntimeError("Reward kernel must be reset before stepping")
        if int(transition.tick) <= int(self.previous.tick):
            raise ValueError("Reward transitions require increasing native ticks")

        alignment_previous = self._alignment(self.previous)
        alignment_now = self._alignment(transition)
        alignment_delta = alignment_now - alignment_previous
        heading_reward, self.heading_positive_spend, self.heading_negative_spend = (
            self._separate_budget(
                HEADING_ALIGNMENT_DELTA_SCALE * alignment_delta,
                self.heading_positive_spend,
                self.heading_negative_spend,
                HEADING_POSITIVE_EPISODE_BUDGET,
                HEADING_NEGATIVE_EPISODE_BUDGET,
            )
        )

        current_ball = np.asarray(transition.ball_position, dtype=np.float64)
        previous_car = np.asarray(self.previous.car_position, dtype=np.float64)
        current_car = np.asarray(transition.car_position, dtype=np.float64)
        progress_unclipped = float(
            np.linalg.norm(previous_car - current_ball) - np.linalg.norm(current_car - current_ball)
        )
        progress_clipped = float(
            np.clip(progress_unclipped, -self.safety_clip_uu, self.safety_clip_uu)
        )
        distance_reward, self.approach_positive_spend, self.approach_negative_spend = (
            self._separate_budget(
                progress_clipped / DISTANCE_PROGRESS_SCALE_UU,
                self.approach_positive_spend,
                self.approach_negative_spend,
                APPROACH_POSITIVE_EPISODE_BUDGET,
                APPROACH_NEGATIVE_EPISODE_BUDGET,
            )
        )

        new_touch = self.touch_detector.process(transition.raw_touch_records)
        rewarded_first_touch = bool(new_touch and not self.first_touch_occurred)
        touch_reward = FIRST_PHYSICAL_TOUCH_REWARD if rewarded_first_touch else 0.0
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
        eligible_start_tick = max(int(self.previous.tick), self.reset_tick + IDLE_GRACE_TICKS)
        eligible_ticks = max(0, int(transition.tick) - eligible_start_tick)
        idle_now = (
            not self.first_touch_occurred
            and not new_touch
            and learner_speed < IDLE_SPEED_THRESHOLD_UU_PER_SECOND
        )
        step_idle_ticks = eligible_ticks if idle_now else 0
        step_idle_seconds = step_idle_ticks / float(PHYSICS_HZ)
        proposed_idle_penalty = IDLE_PENALTY_RATE_PER_SIMULATED_SECOND * step_idle_seconds
        remaining_idle_magnitude = max(0.0, abs(IDLE_EPISODE_FLOOR) - abs(self.idle_penalty_total))
        idle_penalty = -min(abs(proposed_idle_penalty), remaining_idle_magnitude)
        self.idle_ticks += step_idle_ticks
        self.idle_seconds += step_idle_seconds
        self.idle_penalty_total += idle_penalty
        if new_touch:
            self.first_touch_occurred = True

        components = {
            "heading_alignment": heading_reward,
            "distance_progress": distance_reward,
            "physical_new_touch": touch_reward,
            "pre_touch_idle": idle_penalty,
            "goal_for": 0.0,
            "goal_against": 0.0,
        }
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise FloatingPointError(f"Non-finite acquisition reward: {components}")
        if self.heading_positive_spend > HEADING_POSITIVE_EPISODE_BUDGET + 1e-12:
            raise RuntimeError("Positive heading reward exceeded its episode budget")
        if self.heading_negative_spend > abs(HEADING_NEGATIVE_EPISODE_BUDGET) + 1e-12:
            raise RuntimeError("Negative heading reward exceeded its episode budget")
        if self.approach_positive_spend > APPROACH_POSITIVE_EPISODE_BUDGET + 1e-12:
            raise RuntimeError("Positive approach reward exceeded its episode budget")
        if self.approach_negative_spend > abs(APPROACH_NEGATIVE_EPISODE_BUDGET) + 1e-12:
            raise RuntimeError("Negative approach reward exceeded its episode budget")
        if self.idle_penalty_total < IDLE_EPISODE_FLOOR - 1e-12:
            raise RuntimeError("Idle penalty exceeded its episode floor")
        if self.touch_total + touch_reward > FIRST_PHYSICAL_TOUCH_REWARD + 1e-12:
            raise RuntimeError("Touch reward exceeded the first-touch-only budget")

        self.heading_total += heading_reward
        self.distance_total += distance_reward
        self.touch_total += touch_reward
        self.touch_count += int(new_touch)
        self.previous = transition
        return BallAcquisitionRewardStepV4(
            total=total,
            components=components,
            alignment_previous=alignment_previous,
            alignment_now=alignment_now,
            alignment_delta=alignment_delta,
            heading_positive_spend=self.heading_positive_spend,
            heading_negative_spend=self.heading_negative_spend,
            heading_positive_budget_saturated=self.heading_positive_budget_saturated,
            heading_negative_budget_saturated=self.heading_negative_budget_saturated,
            car_progress_unclipped_uu=progress_unclipped,
            car_progress_clipped_uu=progress_clipped,
            distance_absolute_spend=self.distance_absolute_spend,
            distance_budget_saturated=self.distance_budget_saturated,
            approach_positive_spend=self.approach_positive_spend,
            approach_negative_spend=self.approach_negative_spend,
            approach_positive_budget_saturated=self.approach_positive_budget_saturated,
            approach_negative_budget_saturated=self.approach_negative_budget_saturated,
            new_physical_touch=new_touch,
            rewarded_first_physical_touch=rewarded_first_touch,
            raw_touch_records=max(0, int(transition.raw_touch_records)),
            learner_speed_uu_per_second=learner_speed,
            idle_ticks=step_idle_ticks,
            idle_seconds=step_idle_seconds,
            idle_penalty=idle_penalty,
            cumulative_idle_ticks=self.idle_ticks,
            cumulative_idle_seconds=self.idle_seconds,
            cumulative_idle_penalty=self.idle_penalty_total,
            idle_penalty_saturated=self.idle_penalty_saturated,
            first_touch_occurred=self.first_touch_occurred,
        )


class RivalBallAcquisitionRewardV4(RewardFunction[AgentID, GameState, float]):
    def __init__(self) -> None:
        self.active_agent: AgentID | None = None
        self.active_team: int | None = None
        self.kernel = RivalBallAcquisitionRewardKernelV4()

    @staticmethod
    def _transition(
        state: GameState,
        agent: AgentID,
        *,
        goal_for: bool = False,
        goal_against: bool = False,
    ) -> BallAcquisitionTransitionV4:
        car = state.cars[agent]
        return BallAcquisitionTransitionV4(
            tick=int(state.tick_count),
            car_position=np.asarray(car.physics.position, dtype=np.float64),
            ball_position=np.asarray(state.ball.position, dtype=np.float64),
            car_forward=np.asarray(car.physics.forward, dtype=np.float64),
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
                "rival_v10_5_active_team",
                shared_info.get("rival_v10_3_active_team", shared_info["rival_v10_2_active_team"]),
            )
        )
        active_agent = _agent_for_team(initial_state, active_team)
        if active_agent not in agents:
            raise RuntimeError("Active learner is missing from the RLGym agent set")
        self.active_team = active_team
        self.active_agent = active_agent
        self.kernel.reset(self._transition(initial_state, active_agent))
        shared_info["rival_v10_5_active_agent"] = active_agent
        shared_info["rival_v10_5_reward_version"] = BALL_ACQUISITION_REWARD_VERSION
        shared_info["rival_v10_3_active_agent"] = active_agent
        shared_info["rival_v10_2_active_agent"] = active_agent
        shared_info["reward_components"] = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        shared_info["rival_v10_5_reward_metrics"] = {}
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
        components = {agent: {name: 0.0 for name in COMPONENTS} for agent in agents}
        components[self.active_agent] = dict(step.components)
        shared_info["reward_components"] = components
        metrics = {
            "active_agent": self.active_agent,
            "active_team": self.active_team,
            "alignment_previous": step.alignment_previous,
            "alignment_now": step.alignment_now,
            "alignment_delta": step.alignment_delta,
            "heading_positive_spend": step.heading_positive_spend,
            "heading_negative_spend": step.heading_negative_spend,
            "heading_positive_budget_saturated": step.heading_positive_budget_saturated,
            "heading_negative_budget_saturated": step.heading_negative_budget_saturated,
            "heading_total": self.kernel.heading_total,
            "car_progress_unclipped_uu": step.car_progress_unclipped_uu,
            "car_progress_clipped_uu": step.car_progress_clipped_uu,
            "distance_absolute_spend": step.distance_absolute_spend,
            "distance_budget_saturated": step.distance_budget_saturated,
            "approach_positive_spend": step.approach_positive_spend,
            "approach_negative_spend": step.approach_negative_spend,
            "approach_positive_budget_saturated": step.approach_positive_budget_saturated,
            "approach_negative_budget_saturated": step.approach_negative_budget_saturated,
            "new_physical_touch": step.new_physical_touch,
            "rewarded_first_physical_touch": step.rewarded_first_physical_touch,
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
            "first_touch_occurred": step.first_touch_occurred,
        }
        shared_info["rival_v10_5_reward_metrics"] = metrics
        shared_info["rival_v10_3_reward_metrics"] = metrics
        shared_info["rival_v10_2_reward_metrics"] = metrics
        return {agent: step.total if agent == self.active_agent else 0.0 for agent in agents}


def ball_acquisition_reward_metadata() -> dict[str, Any]:
    return {
        "reward_version": BALL_ACQUISITION_REWARD_VERSION,
        "first_physical_touch_reward": FIRST_PHYSICAL_TOUCH_REWARD,
        "subsequent_physical_touch_reward": 0.0,
        "maximum_touch_reward_per_episode": FIRST_PHYSICAL_TOUCH_REWARD,
        "heading_alignment_formula": "dot(car_forward,normalize(ball_position-car_position))",
        "heading_delta_scale": HEADING_ALIGNMENT_DELTA_SCALE,
        "heading_positive_episode_budget": HEADING_POSITIVE_EPISODE_BUDGET,
        "heading_negative_episode_budget": HEADING_NEGATIVE_EPISODE_BUDGET,
        "holding_heading_reward": 0.0,
        "distance_progress_formula": (
            "distance(previous_car,current_ball)-distance(current_car,current_ball)"
        ),
        "distance_progress_scale_uu": DISTANCE_PROGRESS_SCALE_UU,
        "distance_progress_safety_clip_uu": DISTANCE_PROGRESS_SAFETY_CLIP_UU,
        "approach_positive_episode_budget": APPROACH_POSITIVE_EPISODE_BUDGET,
        "approach_negative_episode_budget": APPROACH_NEGATIVE_EPISODE_BUDGET,
        "idle_grace_seconds": IDLE_GRACE_SECONDS,
        "idle_speed_threshold_uu_per_second": IDLE_SPEED_THRESHOLD_UU_PER_SECOND,
        "idle_penalty_rate_per_simulated_second": IDLE_PENALTY_RATE_PER_SIMULATED_SECOND,
        "idle_penalty_per_eligible_tick": IDLE_PENALTY_RATE_PER_SIMULATED_SECOND / PHYSICS_HZ,
        "idle_eligible_seconds_before_no_touch_timeout": IDLE_ELIGIBLE_SECONDS,
        "full_idle_penalty": IDLE_EPISODE_FLOOR,
        "maximum_positive_episode_reward": MAXIMUM_POSITIVE_EPISODE_REWARD,
        "goal_for_reward": 0.0,
        "goal_against_reward": 0.0,
        "generic_speed_reward": 0.0,
        "boost_reward": 0.0,
        "action_magnitude_reward": 0.0,
        "jump_reward": 0.0,
        "named_mechanic_rewards": 0.0,
        "reads_controller_action": False,
        "future_skill_rewards": 0.0,
    }


def reward_truth_table_v4() -> dict[str, Any]:
    def transition(
        tick: int,
        car_x: float,
        ball_x: float,
        *,
        forward_x: float = 1.0,
        forward_y: float = 0.0,
        speed: float = 100.0,
        touches: int = 0,
    ) -> BallAcquisitionTransitionV4:
        return BallAcquisitionTransitionV4(
            tick=tick,
            car_position=np.asarray([car_x, 0.0, 17.0]),
            # Keep this synthetic truth-table geometry exactly collinear so
            # full -1 -> +1 alignment and pure ball translation are exact.
            ball_position=np.asarray([ball_x, 0.0, 17.0]),
            car_forward=np.asarray([forward_x, forward_y, 0.0]),
            car_linear_velocity=np.asarray([speed, 0.0, 0.0]),
            raw_touch_records=touches,
        )

    idle = RivalBallAcquisitionRewardKernelV4()
    idle.reset(transition(0, 0.0, 5000.0, speed=0.0))
    idle_grace = idle.step(transition(IDLE_GRACE_TICKS, 0.0, 5000.0, speed=0.0))
    idle_full = idle.step(
        transition(int(NO_TOUCH_TIMEOUT_SECONDS * PHYSICS_HZ), 0.0, 5000.0, speed=0.0)
    )

    positive = RivalBallAcquisitionRewardKernelV4(safety_clip_uu=2300.0)
    positive.reset(transition(0, 0.0, 20_000.0, forward_x=-1.0))
    turn_toward = positive.step(transition(1, 0.0, 20_000.0, forward_x=1.0))
    approach_1 = positive.step(transition(2, 2300.0, 20_000.0))
    approach_2 = positive.step(transition(3, 4600.0, 20_000.0))
    positive.step(transition(4, 6900.0, 20_000.0))
    first_touch = positive.step(transition(5, 9200.0, 20_000.0, touches=1))
    positive.step(transition(6, 9200.0, 20_000.0, touches=0))
    later_touch = positive.step(transition(7, 9200.0, 20_000.0, touches=1))

    hold = RivalBallAcquisitionRewardKernelV4()
    hold.reset(transition(0, 0.0, 5000.0))
    hold_same = hold.step(transition(1, 0.0, 5000.0))

    negative_heading = RivalBallAcquisitionRewardKernelV4()
    negative_heading.reset(transition(0, 0.0, 5000.0, forward_x=1.0))
    turn_away = negative_heading.step(transition(1, 0.0, 5000.0, forward_x=-1.0))

    progress = RivalBallAcquisitionRewardKernelV4()
    progress.reset(transition(0, 0.0, 5000.0))
    approach = progress.step(transition(1, 10.0, 5000.0))
    away = progress.step(transition(2, 0.0, 5000.0))

    ball_motion = RivalBallAcquisitionRewardKernelV4()
    ball_motion.reset(transition(0, 0.0, 1000.0))
    ball_toward_stationary = ball_motion.step(transition(1, 0.0, 500.0))

    positive_total = (
        positive.heading_positive_spend + positive.approach_positive_spend + positive.touch_total
    )
    metadata = ball_acquisition_reward_metadata()
    checks = {
        "sitting_idle_full_eligible_window_is_negative_16p1": (
            idle_grace.components["pre_touch_idle"] == 0.0
            and math.isclose(
                idle_full.components["pre_touch_idle"],
                -16.1,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(idle.idle_penalty_total, -16.1, rel_tol=0.0, abs_tol=1e-12)
        ),
        "full_positive_episode_reward_cannot_exceed_16": math.isclose(
            positive_total, 16.0, rel_tol=0.0, abs_tol=1e-12
        )
        and positive_total <= MAXIMUM_POSITIVE_EPISODE_REWARD,
        "full_idle_magnitude_exceeds_all_positive_rewards": abs(IDLE_EPISODE_FLOOR)
        > MAXIMUM_POSITIVE_EPISODE_REWARD,
        "turning_toward_ball_is_positive": turn_toward.components["heading_alignment"] == 3.0,
        "holding_same_heading_is_zero": hold_same.components["heading_alignment"] == 0.0,
        "turning_away_is_negative": turn_away.components["heading_alignment"] == -3.0,
        "approaching_is_positive": approach.components["distance_progress"] > 0.0,
        "moving_away_is_negative": away.components["distance_progress"] < 0.0,
        "stationary_car_not_rewarded_when_ball_moves_toward_it": (
            ball_toward_stationary.total == 0.0
        ),
        "first_touch_is_positive_10_exactly_once": (
            first_touch.components["physical_new_touch"] == 10.0
            and first_touch.rewarded_first_physical_touch
        ),
        "later_touches_add_zero": (
            later_touch.new_physical_touch
            and not later_touch.rewarded_first_physical_touch
            and later_touch.components["physical_new_touch"] == 0.0
            and positive.touch_total == 10.0
        ),
        "approach_budget_reaches_positive_3": (
            approach_1.components["distance_progress"] > 0.0
            and approach_2.components["distance_progress"] > 0.0
            and positive.approach_positive_spend == 3.0
        ),
        "all_other_rewards_zero": all(
            metadata[name] == 0.0
            for name in (
                "goal_for_reward",
                "goal_against_reward",
                "generic_speed_reward",
                "boost_reward",
                "action_magnitude_reward",
                "jump_reward",
                "named_mechanic_rewards",
                "future_skill_rewards",
            )
        ),
    }
    checks["passed"] = all(checks.values())
    return {
        "truth_table_version": "RivalBallAcquisitionRewardV4TruthTableV1",
        "metadata": metadata,
        "observations": {
            "idle_grace": dict(idle_grace.components),
            "idle_full_window": dict(idle_full.components),
            "turn_toward": dict(turn_toward.components),
            "hold_same_heading": dict(hold_same.components),
            "turn_away": dict(turn_away.components),
            "approach": dict(approach.components),
            "away": dict(away.components),
            "ball_motion_toward_stationary_car": dict(ball_toward_stationary.components),
            "first_touch": dict(first_touch.components),
            "later_touch": dict(later_touch.components),
        },
        "positive_total": positive_total,
        "checks": checks,
    }
