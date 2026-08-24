"""Uncapped Stage-1 trajectory reward with three rewarded contacts."""

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


BALL_ACQUISITION_REWARD_VERSION = "RivalBallAcquisitionRewardV5"
REWARDED_CONTACT_LIMIT = 3
PHYSICAL_CONTACT_REWARD = 10.0
MAXIMUM_CONTACT_REWARD_PER_EPISODE = REWARDED_CONTACT_LIMIT * PHYSICAL_CONTACT_REWARD
HEADING_ALIGNMENT_DELTA_SCALE = 1.5
ACQUISITION_GRACE_SECONDS = 0.5
ACQUISITION_TIME_PENALTY_RATE_PER_SECOND = -1.4
NO_TOUCH_TIMEOUT_SECONDS = 12.0
PHYSICS_HZ = 120
ACQUISITION_GRACE_TICKS = int(ACQUISITION_GRACE_SECONDS * PHYSICS_HZ)
ACQUISITION_ELIGIBLE_SECONDS = NO_TOUCH_TIMEOUT_SECONDS - ACQUISITION_GRACE_SECONDS
FAILED_ACQUISITION_WINDOW_PENALTY = (
    ACQUISITION_TIME_PENALTY_RATE_PER_SECOND * ACQUISITION_ELIGIBLE_SECONDS
)

COMPONENTS = (
    "heading_alignment",
    "distance_progress",
    "physical_new_touch",
    "acquisition_time",
    "goal_for",
    "goal_against",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return np.zeros(3, dtype=np.float64) if norm <= 1e-12 else value / norm


@dataclass(frozen=True)
class BallAcquisitionTransitionV5:
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
class BallAcquisitionRewardStepV5:
    total: float
    components: Mapping[str, float]
    alignment_previous: float
    alignment_now: float
    alignment_delta: float
    heading_reward: float
    car_progress_unclipped_uu: float
    car_progress_clipped_uu: float
    distance_reward: float
    new_physical_touch: bool
    rewarded_physical_touch: bool
    physical_contact_count: int
    rewarded_contact_count: int
    raw_touch_records: int
    learner_speed_uu_per_second: float
    acquisition_target: int | None
    acquisition_started_tick: int | None
    acquisition_elapsed_ticks: int
    acquisition_time_penalty_ticks: int
    acquisition_time_penalty_seconds: float
    acquisition_time_penalty: float
    cumulative_acquisition_time_penalty_ticks: int
    cumulative_acquisition_time_penalty_seconds: float
    cumulative_acquisition_time_penalty: float


class RivalBallAcquisitionRewardKernelV5:
    """Pure native-120-Hz implementation without episode spend caps."""

    def __init__(self, *, safety_clip_uu: float = DISTANCE_PROGRESS_SAFETY_CLIP_UU) -> None:
        if not math.isfinite(safety_clip_uu) or safety_clip_uu <= 0.0:
            raise ValueError("safety_clip_uu must be finite and positive")
        self.safety_clip_uu = float(safety_clip_uu)
        self.previous: BallAcquisitionTransitionV5 | None = None
        self.acquisition_started_tick: int | None = None
        self.physical_contact_count = 0
        self.rewarded_contact_count = 0
        self.heading_total = 0.0
        self.heading_positive_total = 0.0
        self.heading_negative_total = 0.0
        self.distance_total = 0.0
        self.distance_positive_total = 0.0
        self.distance_negative_total = 0.0
        self.touch_total = 0.0
        self.acquisition_time_penalty_ticks = 0
        self.acquisition_time_penalty_seconds = 0.0
        self.acquisition_time_penalty_total = 0.0
        self.touch_detector = RivalNewContactDetectorV1()

    @staticmethod
    def _alignment(transition: BallAcquisitionTransitionV5) -> float:
        direction = _unit(
            np.asarray(transition.ball_position, dtype=np.float64)
            - np.asarray(transition.car_position, dtype=np.float64)
        )
        forward = _unit(np.asarray(transition.car_forward, dtype=np.float64))
        return float(np.clip(np.dot(forward, direction), -1.0, 1.0))

    @property
    def acquisition_target(self) -> int | None:
        if self.rewarded_contact_count >= REWARDED_CONTACT_LIMIT:
            return None
        return self.rewarded_contact_count + 1

    def reset(self, initial: BallAcquisitionTransitionV5) -> None:
        self.previous = initial
        self.acquisition_started_tick = int(initial.tick)
        self.physical_contact_count = 0
        self.rewarded_contact_count = 0
        self.heading_total = 0.0
        self.heading_positive_total = 0.0
        self.heading_negative_total = 0.0
        self.distance_total = 0.0
        self.distance_positive_total = 0.0
        self.distance_negative_total = 0.0
        self.touch_total = 0.0
        self.acquisition_time_penalty_ticks = 0
        self.acquisition_time_penalty_seconds = 0.0
        self.acquisition_time_penalty_total = 0.0
        self.touch_detector.reset()

    def _eligible_penalty_ticks(
        self, transition: BallAcquisitionTransitionV5, *, rewarded_contact: bool
    ) -> int:
        if self.previous is None or self.acquisition_started_tick is None:
            return 0
        if self.rewarded_contact_count >= REWARDED_CONTACT_LIMIT:
            return 0
        eligible_start = self.acquisition_started_tick + ACQUISITION_GRACE_TICKS
        end_tick = int(transition.tick) - int(rewarded_contact)
        return max(0, end_tick - max(int(self.previous.tick), eligible_start))

    def step(self, transition: BallAcquisitionTransitionV5) -> BallAcquisitionRewardStepV5:
        if self.previous is None:
            raise RuntimeError("Reward kernel must be reset before stepping")
        if int(transition.tick) <= int(self.previous.tick):
            raise ValueError("Reward transitions require increasing native ticks")

        alignment_previous = self._alignment(self.previous)
        alignment_now = self._alignment(transition)
        alignment_delta = alignment_now - alignment_previous
        heading_reward = HEADING_ALIGNMENT_DELTA_SCALE * alignment_delta

        current_ball = np.asarray(transition.ball_position, dtype=np.float64)
        previous_car = np.asarray(self.previous.car_position, dtype=np.float64)
        current_car = np.asarray(transition.car_position, dtype=np.float64)
        progress_unclipped = float(
            np.linalg.norm(previous_car - current_ball) - np.linalg.norm(current_car - current_ball)
        )
        progress_clipped = float(
            np.clip(progress_unclipped, -self.safety_clip_uu, self.safety_clip_uu)
        )
        distance_reward = progress_clipped / DISTANCE_PROGRESS_SCALE_UU

        new_contact = self.touch_detector.process(transition.raw_touch_records)
        rewarded_contact = bool(
            new_contact and self.rewarded_contact_count < REWARDED_CONTACT_LIMIT
        )
        touch_reward = PHYSICAL_CONTACT_REWARD if rewarded_contact else 0.0
        target_before_contact = self.acquisition_target
        acquisition_started_before_contact = self.acquisition_started_tick
        penalty_ticks = self._eligible_penalty_ticks(transition, rewarded_contact=rewarded_contact)
        penalty_seconds = penalty_ticks / float(PHYSICS_HZ)
        time_penalty = ACQUISITION_TIME_PENALTY_RATE_PER_SECOND * penalty_seconds
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

        if new_contact:
            self.physical_contact_count += 1
        if rewarded_contact:
            self.rewarded_contact_count += 1
            self.acquisition_started_tick = (
                None
                if self.rewarded_contact_count >= REWARDED_CONTACT_LIMIT
                else int(transition.tick)
            )

        self.heading_total += heading_reward
        self.heading_positive_total += max(0.0, heading_reward)
        self.heading_negative_total += min(0.0, heading_reward)
        self.distance_total += distance_reward
        self.distance_positive_total += max(0.0, distance_reward)
        self.distance_negative_total += min(0.0, distance_reward)
        self.touch_total += touch_reward
        self.acquisition_time_penalty_ticks += penalty_ticks
        self.acquisition_time_penalty_seconds += penalty_seconds
        self.acquisition_time_penalty_total += time_penalty

        components = {
            "heading_alignment": heading_reward,
            "distance_progress": distance_reward,
            "physical_new_touch": touch_reward,
            "acquisition_time": time_penalty,
            "goal_for": 0.0,
            "goal_against": 0.0,
        }
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise FloatingPointError(f"Non-finite acquisition reward: {components}")
        if self.touch_total > MAXIMUM_CONTACT_REWARD_PER_EPISODE + 1e-12:
            raise RuntimeError("Touch reward exceeded the three-contact maximum")

        acquisition_elapsed = (
            0
            if acquisition_started_before_contact is None
            else int(transition.tick) - acquisition_started_before_contact
        )
        self.previous = transition
        return BallAcquisitionRewardStepV5(
            total=total,
            components=components,
            alignment_previous=alignment_previous,
            alignment_now=alignment_now,
            alignment_delta=alignment_delta,
            heading_reward=heading_reward,
            car_progress_unclipped_uu=progress_unclipped,
            car_progress_clipped_uu=progress_clipped,
            distance_reward=distance_reward,
            new_physical_touch=new_contact,
            rewarded_physical_touch=rewarded_contact,
            physical_contact_count=self.physical_contact_count,
            rewarded_contact_count=self.rewarded_contact_count,
            raw_touch_records=max(0, int(transition.raw_touch_records)),
            learner_speed_uu_per_second=learner_speed,
            acquisition_target=target_before_contact,
            acquisition_started_tick=acquisition_started_before_contact,
            acquisition_elapsed_ticks=acquisition_elapsed,
            acquisition_time_penalty_ticks=penalty_ticks,
            acquisition_time_penalty_seconds=penalty_seconds,
            acquisition_time_penalty=time_penalty,
            cumulative_acquisition_time_penalty_ticks=(self.acquisition_time_penalty_ticks),
            cumulative_acquisition_time_penalty_seconds=(self.acquisition_time_penalty_seconds),
            cumulative_acquisition_time_penalty=self.acquisition_time_penalty_total,
        )


class RivalBallAcquisitionRewardV5(RewardFunction[AgentID, GameState, float]):
    def __init__(self) -> None:
        self.active_agent: AgentID | None = None
        self.active_team: int | None = None
        self.kernel = RivalBallAcquisitionRewardKernelV5()

    @staticmethod
    def _transition(
        state: GameState,
        agent: AgentID,
        *,
        goal_for: bool = False,
        goal_against: bool = False,
    ) -> BallAcquisitionTransitionV5:
        car = state.cars[agent]
        return BallAcquisitionTransitionV5(
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
                "rival_v10_6_active_team",
                shared_info.get("rival_v10_3_active_team", shared_info["rival_v10_2_active_team"]),
            )
        )
        active_agent = _agent_for_team(initial_state, active_team)
        if active_agent not in agents:
            raise RuntimeError("Active learner is missing from the RLGym agent set")
        self.active_team = active_team
        self.active_agent = active_agent
        self.kernel.reset(self._transition(initial_state, active_agent))
        shared_info["rival_v10_6_active_agent"] = active_agent
        shared_info["rival_v10_6_reward_version"] = BALL_ACQUISITION_REWARD_VERSION
        shared_info["rival_v10_3_active_agent"] = active_agent
        shared_info["rival_v10_2_active_agent"] = active_agent
        shared_info["reward_components"] = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        shared_info["rival_v10_6_reward_metrics"] = {}
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
            "heading_total": self.kernel.heading_total,
            "heading_positive_total": self.kernel.heading_positive_total,
            "heading_negative_total": self.kernel.heading_negative_total,
            "heading_positive_budget_saturated": False,
            "heading_negative_budget_saturated": False,
            "car_progress_unclipped_uu": step.car_progress_unclipped_uu,
            "car_progress_clipped_uu": step.car_progress_clipped_uu,
            "distance_reward": step.distance_reward,
            "distance_total": self.kernel.distance_total,
            "distance_positive_total": self.kernel.distance_positive_total,
            "distance_negative_total": self.kernel.distance_negative_total,
            "distance_absolute_spend": (
                self.kernel.distance_positive_total - self.kernel.distance_negative_total
            ),
            "distance_budget_saturated": False,
            "approach_positive_budget_saturated": False,
            "approach_negative_budget_saturated": False,
            "new_physical_touch": step.new_physical_touch,
            "rewarded_physical_touch": step.rewarded_physical_touch,
            "rewarded_first_physical_touch": (
                step.rewarded_physical_touch and step.rewarded_contact_count == 1
            ),
            "raw_touch_records": step.raw_touch_records,
            "touch_count": step.physical_contact_count,
            "rewarded_touch_count": step.rewarded_contact_count,
            "touch_total": self.kernel.touch_total,
            "learner_speed_uu_per_second": step.learner_speed_uu_per_second,
            "acquisition_target": step.acquisition_target,
            "acquisition_started_tick": step.acquisition_started_tick,
            "acquisition_elapsed_ticks": step.acquisition_elapsed_ticks,
            "acquisition_time_penalty_ticks": step.acquisition_time_penalty_ticks,
            "acquisition_time_penalty_seconds": step.acquisition_time_penalty_seconds,
            "acquisition_time_penalty": step.acquisition_time_penalty,
            "cumulative_acquisition_time_penalty_ticks": (
                step.cumulative_acquisition_time_penalty_ticks
            ),
            "cumulative_acquisition_time_penalty_seconds": (
                step.cumulative_acquisition_time_penalty_seconds
            ),
            "cumulative_acquisition_time_penalty": (step.cumulative_acquisition_time_penalty),
            # Backward-compatible evaluation aliases.
            "idle_ticks": step.acquisition_time_penalty_ticks,
            "idle_seconds": step.acquisition_time_penalty_seconds,
            "idle_penalty": step.acquisition_time_penalty,
            "cumulative_idle_ticks": step.cumulative_acquisition_time_penalty_ticks,
            "cumulative_idle_seconds": step.cumulative_acquisition_time_penalty_seconds,
            "cumulative_idle_penalty": step.cumulative_acquisition_time_penalty,
            "idle_penalty_saturated": False,
            "first_touch_occurred": step.rewarded_contact_count >= 1,
        }
        shared_info["rival_v10_6_reward_metrics"] = metrics
        shared_info["rival_v10_3_reward_metrics"] = metrics
        shared_info["rival_v10_2_reward_metrics"] = metrics
        return {agent: step.total if agent == self.active_agent else 0.0 for agent in agents}


def ball_acquisition_reward_metadata() -> dict[str, Any]:
    return {
        "reward_version": BALL_ACQUISITION_REWARD_VERSION,
        "rewarded_contact_limit": REWARDED_CONTACT_LIMIT,
        "physical_contact_reward": PHYSICAL_CONTACT_REWARD,
        "maximum_contact_reward_per_episode": MAXIMUM_CONTACT_REWARD_PER_EPISODE,
        "requires_separated_new_contact": True,
        "heading_alignment_formula": "dot(car_forward,normalize(ball_position-car_position))",
        "heading_delta_scale": HEADING_ALIGNMENT_DELTA_SCALE,
        "heading_positive_episode_budget": None,
        "heading_negative_episode_budget": None,
        "holding_heading_reward": 0.0,
        "distance_progress_formula": (
            "distance(previous_car,current_ball)-distance(current_car,current_ball)"
        ),
        "distance_progress_scale_uu": DISTANCE_PROGRESS_SCALE_UU,
        "distance_progress_safety_clip_uu": DISTANCE_PROGRESS_SAFETY_CLIP_UU,
        "distance_positive_episode_budget": None,
        "distance_negative_episode_budget": None,
        "acquisition_grace_seconds": ACQUISITION_GRACE_SECONDS,
        "acquisition_time_penalty_rate_per_second": (ACQUISITION_TIME_PENALTY_RATE_PER_SECOND),
        "acquisition_time_penalty_per_eligible_tick": (
            ACQUISITION_TIME_PENALTY_RATE_PER_SECOND / PHYSICS_HZ
        ),
        "acquisition_eligible_seconds_before_no_touch_timeout": (ACQUISITION_ELIGIBLE_SECONDS),
        "failed_acquisition_window_penalty": FAILED_ACQUISITION_WINDOW_PENALTY,
        "time_penalty_depends_on_speed": False,
        "goal_for_reward": 0.0,
        "goal_against_reward": 0.0,
        "generic_speed_reward": 0.0,
        "boost_reward": 0.0,
        "throttle_reward": 0.0,
        "steer_reward": 0.0,
        "action_magnitude_reward": 0.0,
        "jump_reward": 0.0,
        "handbrake_reward": 0.0,
        "named_mechanic_reward": 0.0,
        "possession_reward": 0.0,
        "aerial_reward": 0.0,
        "recovery_reward": 0.0,
        "reads_controller_action": False,
    }


def reward_truth_table_v5() -> dict[str, Any]:
    def transition(
        tick: int,
        car_x: float = 0.0,
        ball_x: float = 5000.0,
        *,
        forward_x: float = 1.0,
        speed: float = 0.0,
        touches: int = 0,
    ) -> BallAcquisitionTransitionV5:
        return BallAcquisitionTransitionV5(
            tick=tick,
            car_position=np.asarray([car_x, 0.0, 17.0]),
            ball_position=np.asarray([ball_x, 0.0, 17.0]),
            car_forward=np.asarray([forward_x, 0.0, 0.0]),
            car_linear_velocity=np.asarray([speed, 0.0, 0.0]),
            raw_touch_records=touches,
        )

    idle = RivalBallAcquisitionRewardKernelV5()
    idle.reset(transition(0, speed=0.0))
    idle_grace = idle.step(transition(ACQUISITION_GRACE_TICKS, speed=0.0))
    idle_full = idle.step(transition(int(NO_TOUCH_TIMEOUT_SECONDS * PHYSICS_HZ), speed=0.0))
    fast = RivalBallAcquisitionRewardKernelV5()
    fast.reset(transition(0, speed=1500.0))
    fast.step(transition(ACQUISITION_GRACE_TICKS, speed=1500.0))
    fast_full = fast.step(transition(int(NO_TOUCH_TIMEOUT_SECONDS * PHYSICS_HZ), speed=1500.0))

    contacts = RivalBallAcquisitionRewardKernelV5()
    contacts.reset(transition(0, speed=100.0))
    contacts.step(transition(ACQUISITION_GRACE_TICKS, speed=100.0))
    first = contacts.step(transition(ACQUISITION_GRACE_TICKS + 1, speed=100.0, touches=1))
    sustained = contacts.step(transition(ACQUISITION_GRACE_TICKS + 2, speed=100.0, touches=1))
    contacts.step(transition(ACQUISITION_GRACE_TICKS + 3, speed=100.0, touches=0))
    first_restart_grace = contacts.step(
        transition(2 * ACQUISITION_GRACE_TICKS + 1, speed=100.0, touches=0)
    )
    first_restart_penalty = contacts.step(
        transition(2 * ACQUISITION_GRACE_TICKS + 2, speed=100.0, touches=0)
    )
    second = contacts.step(transition(2 * ACQUISITION_GRACE_TICKS + 3, speed=100.0, touches=1))
    contacts.step(transition(2 * ACQUISITION_GRACE_TICKS + 4, speed=100.0, touches=0))
    second_restart_grace = contacts.step(
        transition(3 * ACQUISITION_GRACE_TICKS + 3, speed=100.0, touches=0)
    )
    second_restart_penalty = contacts.step(
        transition(3 * ACQUISITION_GRACE_TICKS + 4, speed=100.0, touches=0)
    )
    third = contacts.step(transition(3 * ACQUISITION_GRACE_TICKS + 5, speed=100.0, touches=1))
    contacts.step(transition(3 * ACQUISITION_GRACE_TICKS + 6, speed=100.0, touches=0))
    after_third = contacts.step(transition(5 * ACQUISITION_GRACE_TICKS + 6, speed=100.0, touches=0))
    fourth = contacts.step(transition(5 * ACQUISITION_GRACE_TICKS + 7, speed=100.0, touches=1))

    heading = RivalBallAcquisitionRewardKernelV5()
    heading.reset(transition(0, forward_x=-1.0, speed=100.0))
    turn_toward = heading.step(transition(1, forward_x=1.0, speed=100.0))
    hold = heading.step(transition(2, forward_x=1.0, speed=100.0))
    turn_back = heading.step(transition(3, forward_x=-1.0, speed=100.0))
    second_toward = heading.step(transition(4, forward_x=1.0, speed=100.0))
    second_back = heading.step(transition(5, forward_x=-1.0, speed=100.0))

    distance = RivalBallAcquisitionRewardKernelV5()
    distance.reset(transition(0, car_x=0.0, ball_x=20_000.0, speed=100.0))
    approach = distance.step(transition(1, car_x=50.0, ball_x=20_000.0, speed=100.0))
    reverse = distance.step(transition(2, car_x=0.0, ball_x=20_000.0, speed=100.0))
    away_steps = []
    for tick in range(3, 123):
        car_x = -(tick - 2) * DISTANCE_PROGRESS_SAFETY_CLIP_UU
        away_steps.append(
            distance.step(transition(tick, car_x=car_x, ball_x=20_000.0, speed=100.0))
        )

    ball_motion = RivalBallAcquisitionRewardKernelV5()
    ball_motion.reset(transition(0, car_x=0.0, ball_x=1000.0, speed=0.0))
    ball_toward = ball_motion.step(transition(1, car_x=0.0, ball_x=500.0, speed=0.0))

    metadata = ball_acquisition_reward_metadata()
    zero_reward_names = (
        "goal_for_reward",
        "goal_against_reward",
        "generic_speed_reward",
        "boost_reward",
        "throttle_reward",
        "steer_reward",
        "action_magnitude_reward",
        "jump_reward",
        "handbrake_reward",
        "named_mechanic_reward",
        "possession_reward",
        "aerial_reward",
        "recovery_reward",
    )
    checks = {
        "failed_window_grace_0p5_eligible_11p5_penalty_negative_16p1": (
            idle_grace.acquisition_time_penalty == 0.0
            and idle_full.acquisition_time_penalty_ticks == 1380
            and math.isclose(idle.acquisition_time_penalty_total, -16.1, rel_tol=0.0, abs_tol=1e-12)
        ),
        "fast_and_stationary_time_penalties_identical": math.isclose(
            fast_full.acquisition_time_penalty,
            idle_full.acquisition_time_penalty,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "time_penalty_stops_on_rewarded_contact": (first.acquisition_time_penalty == 0.0),
        "timer_grace_restarts_after_first_contact": (
            first_restart_grace.acquisition_time_penalty == 0.0
            and first_restart_penalty.acquisition_time_penalty
            == ACQUISITION_TIME_PENALTY_RATE_PER_SECOND / PHYSICS_HZ
        ),
        "timer_grace_restarts_after_second_contact": (
            second_restart_grace.acquisition_time_penalty == 0.0
            and second_restart_penalty.acquisition_time_penalty
            == ACQUISITION_TIME_PENALTY_RATE_PER_SECOND / PHYSICS_HZ
        ),
        "after_third_no_touch_reward_or_time_penalty": (
            after_third.acquisition_time_penalty == 0.0
            and fourth.components["physical_new_touch"] == 0.0
            and fourth.acquisition_time_penalty == 0.0
        ),
        "first_separated_contact_positive_10": (first.components["physical_new_touch"] == 10.0),
        "second_separated_contact_positive_10": (second.components["physical_new_touch"] == 10.0),
        "third_separated_contact_positive_10": (third.components["physical_new_touch"] == 10.0),
        "fourth_and_later_contacts_zero": (
            fourth.new_physical_touch and fourth.components["physical_new_touch"] == 0.0
        ),
        "sustained_contact_counts_once": (
            not sustained.new_physical_touch and sustained.components["physical_new_touch"] == 0.0
        ),
        "turning_toward_positive": turn_toward.heading_reward == 3.0,
        "holding_alignment_zero": hold.heading_reward == 0.0,
        "turning_away_negative": turn_back.heading_reward == -3.0,
        "heading_round_trip_cancels": math.isclose(
            turn_toward.heading_reward + turn_back.heading_reward,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "no_heading_spend_caps": (
            heading.heading_positive_total == 6.0
            and heading.heading_negative_total == -6.0
            and second_toward.heading_reward == 3.0
            and second_back.heading_reward == -3.0
        ),
        "approaching_positive": approach.distance_reward > 0.0,
        "moving_away_negative": reverse.distance_reward < 0.0,
        "continued_away_accumulates_negative": (
            all(step.distance_reward < 0.0 for step in away_steps)
            and math.isclose(
                sum(step.distance_reward for step in away_steps),
                -120.0 * DISTANCE_PROGRESS_SAFETY_CLIP_UU / DISTANCE_PROGRESS_SCALE_UU,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ),
        "no_distance_spend_caps": distance.distance_negative_total < -3.0,
        "approach_reverse_cancels": math.isclose(
            approach.distance_reward + reverse.distance_reward,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "ball_motion_stationary_learner_zero_distance_reward": (ball_toward.distance_reward == 0.0),
        "all_other_reward_components_zero": all(
            metadata[name] == 0.0 for name in zero_reward_names
        ),
    }
    checks["passed"] = all(checks.values())
    return {
        "truth_table_version": "RivalBallAcquisitionRewardV5TruthTableV1",
        "metadata": metadata,
        "observations": {
            "idle_grace": dict(idle_grace.components),
            "idle_full": dict(idle_full.components),
            "fast_full": dict(fast_full.components),
            "first_contact": dict(first.components),
            "sustained_contact": dict(sustained.components),
            "first_restart_grace": dict(first_restart_grace.components),
            "first_restart_penalty": dict(first_restart_penalty.components),
            "second_contact": dict(second.components),
            "second_restart_grace": dict(second_restart_grace.components),
            "second_restart_penalty": dict(second_restart_penalty.components),
            "third_contact": dict(third.components),
            "after_third": dict(after_third.components),
            "fourth_contact": dict(fourth.components),
            "turn_toward": dict(turn_toward.components),
            "hold": dict(hold.components),
            "turn_back": dict(turn_back.components),
            "second_toward": dict(second_toward.components),
            "second_back": dict(second_back.components),
            "approach": dict(approach.components),
            "reverse": dict(reverse.components),
            "continued_away": [dict(step.components) for step in away_steps],
            "ball_motion": dict(ball_toward.components),
        },
        "checks": checks,
    }
