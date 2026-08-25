"""Minimal physical-time Stage-1 reward: velocity toward ball plus first touch."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState

from .v10_2_reward import RivalNewContactDetectorV1, _agent_for_team


FIRST_TOUCH_VELOCITY_REWARD_VERSION = "RivalFirstTouchVelocityRewardV1"
PHYSICS_HZ = 120
MAXIMUM_CAR_SPEED_UU_PER_SECOND = 2300.0
FIRST_PHYSICAL_TOUCH_REWARD = 10.0
REWARDED_CONTACT_LIMIT = 1

COMPONENTS = (
    "velocity_to_ball",
    "physical_new_touch",
    "distance_progress",
    "heading_alignment",
    "acquisition_time",
    "idle",
    "generic_speed",
    "boost",
    "throttle",
    "steer",
    "action_magnitude",
    "jump",
    "handbrake",
    "goal_for",
    "goal_against",
    "possession",
    "named_mechanic",
    "aerial",
    "recovery",
)


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return np.zeros(3, dtype=np.float64) if norm <= 1e-12 else value / norm


@dataclass(frozen=True)
class FirstTouchVelocityTransitionV1:
    tick: int
    car_position: np.ndarray
    ball_position: np.ndarray
    car_linear_velocity: np.ndarray
    raw_touch_records: int = 0

    def __post_init__(self) -> None:
        vectors = tuple(
            np.asarray(value, dtype=np.float64)
            for value in (
                self.car_position,
                self.ball_position,
                self.car_linear_velocity,
            )
        )
        if any(value.shape != (3,) for value in vectors):
            raise ValueError("Reward transition vectors require shape (3,)")
        if not all(np.isfinite(value).all() for value in vectors):
            raise FloatingPointError("Reward transition vectors must be finite")
        if int(self.raw_touch_records) < 0:
            raise ValueError("raw_touch_records cannot be negative")


@dataclass(frozen=True)
class FirstTouchVelocityRewardStepV1:
    total: float
    components: Mapping[str, float]
    tick_delta: int
    simulated_seconds: float
    direction_to_ball: np.ndarray
    directed_velocity_uu_per_second: float
    normalized_directed_velocity: float
    velocity_to_ball_reward: float
    new_physical_touch: bool
    rewarded_first_physical_touch: bool
    physical_touch_reward: float
    physical_contact_count: int
    rewarded_contact_count: int
    episode_should_terminate: bool
    raw_touch_records: int


class RivalFirstTouchVelocityRewardKernelV1:
    """Pure reward kernel integrated in simulated seconds at native 120 Hz."""

    def __init__(self) -> None:
        self.previous: FirstTouchVelocityTransitionV1 | None = None
        self.touch_detector = RivalNewContactDetectorV1()
        self.physical_contact_count = 0
        self.rewarded_contact_count = 0
        self.velocity_to_ball_total = 0.0
        self.touch_total = 0.0

    def reset(self, initial: FirstTouchVelocityTransitionV1) -> None:
        self.previous = initial
        self.touch_detector.reset()
        self.physical_contact_count = 0
        self.rewarded_contact_count = 0
        self.velocity_to_ball_total = 0.0
        self.touch_total = 0.0

    def step(
        self, transition: FirstTouchVelocityTransitionV1
    ) -> FirstTouchVelocityRewardStepV1:
        if self.previous is None:
            raise RuntimeError("Reward kernel must be reset before stepping")
        tick_delta = int(transition.tick) - int(self.previous.tick)
        if tick_delta <= 0:
            raise ValueError("Reward transitions require increasing native ticks")

        direction = _unit(
            np.asarray(transition.ball_position, dtype=np.float64)
            - np.asarray(transition.car_position, dtype=np.float64)
        )
        directed_velocity = float(
            np.dot(
                np.asarray(transition.car_linear_velocity, dtype=np.float64),
                direction,
            )
        )
        normalized = directed_velocity / MAXIMUM_CAR_SPEED_UU_PER_SECOND
        simulated_seconds = tick_delta / float(PHYSICS_HZ)
        velocity_reward = normalized * simulated_seconds

        new_contact = self.touch_detector.process(transition.raw_touch_records)
        rewarded_touch = bool(new_contact and self.rewarded_contact_count == 0)
        touch_reward = FIRST_PHYSICAL_TOUCH_REWARD if rewarded_touch else 0.0
        if new_contact:
            self.physical_contact_count += 1
        if rewarded_touch:
            self.rewarded_contact_count += 1

        components = {name: 0.0 for name in COMPONENTS}
        components["velocity_to_ball"] = velocity_reward
        components["physical_new_touch"] = touch_reward
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise FloatingPointError(f"Non-finite first-touch reward: {components}")

        self.velocity_to_ball_total += velocity_reward
        self.touch_total += touch_reward
        self.previous = transition
        return FirstTouchVelocityRewardStepV1(
            total=total,
            components=components,
            tick_delta=tick_delta,
            simulated_seconds=simulated_seconds,
            direction_to_ball=direction,
            directed_velocity_uu_per_second=directed_velocity,
            normalized_directed_velocity=normalized,
            velocity_to_ball_reward=velocity_reward,
            new_physical_touch=new_contact,
            rewarded_first_physical_touch=rewarded_touch,
            physical_touch_reward=touch_reward,
            physical_contact_count=self.physical_contact_count,
            rewarded_contact_count=self.rewarded_contact_count,
            episode_should_terminate=rewarded_touch,
            raw_touch_records=max(0, int(transition.raw_touch_records)),
        )


class RivalFirstTouchVelocityRewardV1(RewardFunction[AgentID, GameState, float]):
    def __init__(self) -> None:
        self.active_agent: AgentID | None = None
        self.active_team: int | None = None
        self.kernel = RivalFirstTouchVelocityRewardKernelV1()

    @staticmethod
    def _transition(
        state: GameState, agent: AgentID
    ) -> FirstTouchVelocityTransitionV1:
        car = state.cars[agent]
        return FirstTouchVelocityTransitionV1(
            tick=int(state.tick_count),
            car_position=np.asarray(car.physics.position, dtype=np.float64),
            ball_position=np.asarray(state.ball.position, dtype=np.float64),
            car_linear_velocity=np.asarray(
                car.physics.linear_velocity, dtype=np.float64
            ),
            raw_touch_records=max(0, int(car.ball_touches)),
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
                shared_info.get(
                    "rival_v10_3_active_team",
                    shared_info["rival_v10_2_active_team"],
                ),
            )
        )
        active_agent = _agent_for_team(initial_state, active_team)
        if active_agent not in agents:
            raise RuntimeError("Active learner is missing from the RLGym agent set")
        self.active_team = active_team
        self.active_agent = active_agent
        self.kernel.reset(self._transition(initial_state, active_agent))
        shared_info["rival_v10_10_active_agent"] = active_agent
        shared_info["rival_v10_10_reward_version"] = (
            FIRST_TOUCH_VELOCITY_REWARD_VERSION
        )
        shared_info["reward_components"] = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        shared_info["rival_v10_10_reward_metrics"] = {}
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
        step = self.kernel.step(self._transition(state, self.active_agent))
        components = {agent: {name: 0.0 for name in COMPONENTS} for agent in agents}
        components[self.active_agent] = dict(step.components)
        shared_info["reward_components"] = components
        metrics = {
            "active_agent": self.active_agent,
            "active_team": self.active_team,
            "tick_delta": step.tick_delta,
            "simulated_seconds": step.simulated_seconds,
            "direction_to_ball": step.direction_to_ball.tolist(),
            "directed_velocity_uu_per_second": (
                step.directed_velocity_uu_per_second
            ),
            "normalized_directed_velocity": step.normalized_directed_velocity,
            "velocity_to_ball_reward": step.velocity_to_ball_reward,
            "velocity_to_ball_total": self.kernel.velocity_to_ball_total,
            "new_physical_touch": step.new_physical_touch,
            "rewarded_physical_touch": step.rewarded_first_physical_touch,
            "rewarded_first_physical_touch": (
                step.rewarded_first_physical_touch
            ),
            "raw_touch_records": step.raw_touch_records,
            "touch_count": step.physical_contact_count,
            "rewarded_touch_count": step.rewarded_contact_count,
            "touch_total": self.kernel.touch_total,
            "first_touch_occurred": step.rewarded_contact_count >= 1,
            "episode_should_terminate": step.episode_should_terminate,
            # Compatibility fields are deliberately exact zero.
            "alignment_previous": 0.0,
            "alignment_now": 0.0,
            "alignment_delta": 0.0,
            "heading_total": 0.0,
            "car_progress_unclipped_uu": 0.0,
            "car_progress_clipped_uu": 0.0,
            "distance_reward": 0.0,
            "distance_total": 0.0,
            "distance_budget_saturated": False,
            "idle_ticks": 0,
            "idle_seconds": 0.0,
            "idle_penalty": 0.0,
            "cumulative_idle_penalty": 0.0,
            "cumulative_acquisition_time_penalty": 0.0,
        }
        for key in (
            "rival_v10_10_reward_metrics",
            "rival_v10_6_reward_metrics",
            "rival_v10_3_reward_metrics",
            "rival_v10_2_reward_metrics",
        ):
            shared_info[key] = metrics
        return {agent: step.total if agent == self.active_agent else 0.0 for agent in agents}


def first_touch_velocity_reward_metadata() -> dict[str, Any]:
    return {
        "reward_version": FIRST_TOUCH_VELOCITY_REWARD_VERSION,
        "physics_hz": PHYSICS_HZ,
        "maximum_car_speed_uu_per_second": MAXIMUM_CAR_SPEED_UU_PER_SECOND,
        "directed_velocity_formula": (
            "dot(car_linear_velocity,normalize(ball_position-car_position))"
        ),
        "instantaneous_normalization": "directed_velocity / 2300",
        "native_tick_integration": "normalized_directed_velocity * tick_delta / 120",
        "full_speed_toward_reward_per_simulated_second": 1.0,
        "full_speed_away_reward_per_simulated_second": -1.0,
        "first_physical_touch_reward": FIRST_PHYSICAL_TOUCH_REWARD,
        "rewarded_contact_limit": REWARDED_CONTACT_LIMIT,
        "terminate_on_first_touch": True,
        "reads_ball_velocity": False,
        "reads_controller_action": False,
        **{
            name: 0.0
            for name in (
                "distance_progress_reward",
                "heading_alignment_reward",
                "acquisition_time_penalty",
                "idle_penalty",
                "generic_speed_reward",
                "boost_reward",
                "throttle_reward",
                "steer_reward",
                "action_magnitude_reward",
                "jump_reward",
                "handbrake_reward",
                "goal_for_reward",
                "goal_against_reward",
                "possession_reward",
                "named_mechanic_reward",
                "aerial_reward",
                "recovery_reward",
            )
        },
    }


def reward_truth_table_v10_10() -> dict[str, Any]:
    def transition(
        tick: int,
        *,
        car=(0.0, 0.0, 17.0),
        ball=(5000.0, 0.0, 17.0),
        velocity=(0.0, 0.0, 0.0),
        touches: int = 0,
    ) -> FirstTouchVelocityTransitionV1:
        return FirstTouchVelocityTransitionV1(
            tick=tick,
            car_position=np.asarray(car, dtype=np.float64),
            ball_position=np.asarray(ball, dtype=np.float64),
            car_linear_velocity=np.asarray(velocity, dtype=np.float64),
            raw_touch_records=touches,
        )

    def one(velocity, *, ball=(5000.0, 0.0, 17.0), touches=0):
        kernel = RivalFirstTouchVelocityRewardKernelV1()
        kernel.reset(transition(0))
        return kernel, kernel.step(
            transition(1, ball=ball, velocity=velocity, touches=touches)
        )

    _, stationary = one((0.0, 0.0, 0.0))
    _, toward_slow = one((500.0, 0.0, 0.0))
    _, toward_fast = one((1500.0, 0.0, 0.0))
    _, toward_full = one((2300.0, 0.0, 0.0))
    _, perpendicular = one((0.0, 2000.0, 0.0))
    _, away_slow = one((-500.0, 0.0, 0.0))
    _, away_fast = one((-1500.0, 0.0, 0.0))
    _, ball_motion = one((0.0, 0.0, 0.0), ball=(1000.0, 500.0, 17.0))
    touch_kernel, touch = one((0.0, 0.0, 0.0), touches=1)
    sustained = touch_kernel.step(transition(2, touches=1))
    touch_kernel.step(transition(3, touches=0))
    later = touch_kernel.step(transition(4, touches=1))
    metadata = first_touch_velocity_reward_metadata()
    zero_components = tuple(name for name in COMPONENTS if name not in {
        "velocity_to_ball", "physical_new_touch"
    })
    checks = {
        "stationary_zero": stationary.velocity_to_ball_reward == 0.0,
        "moving_directly_toward_positive": toward_slow.velocity_to_ball_reward > 0.0,
        "faster_toward_more_positive": (
            toward_fast.velocity_to_ball_reward > toward_slow.velocity_to_ball_reward
        ),
        "full_speed_toward_normalized_maximum": math.isclose(
            toward_full.normalized_directed_velocity, 1.0, abs_tol=1e-12
        ) and math.isclose(
            toward_full.velocity_to_ball_reward, 1.0 / PHYSICS_HZ, abs_tol=1e-12
        ),
        "perpendicular_approximately_zero": math.isclose(
            perpendicular.velocity_to_ball_reward, 0.0, abs_tol=1e-12
        ),
        "moving_directly_away_negative": away_slow.velocity_to_ball_reward < 0.0,
        "faster_away_more_negative": (
            away_fast.velocity_to_ball_reward < away_slow.velocity_to_ball_reward
        ),
        "ball_motion_stationary_car_zero": ball_motion.velocity_to_ball_reward == 0.0,
        "first_genuine_touch_positive_10": (
            touch.new_physical_touch
            and touch.physical_touch_reward == FIRST_PHYSICAL_TOUCH_REWARD
        ),
        "episode_terminates_on_first_touch": touch.episode_should_terminate,
        "sustained_contact_not_recounted": (
            not sustained.new_physical_touch
            and sustained.physical_touch_reward == 0.0
        ),
        "later_separated_touch_zero": (
            later.new_physical_touch and later.physical_touch_reward == 0.0
        ),
        "every_other_component_zero": all(
            step.components[name] == 0.0
            for step in (
                stationary,
                toward_slow,
                toward_fast,
                toward_full,
                perpendicular,
                away_slow,
                away_fast,
                ball_motion,
                touch,
            )
            for name in zero_components
        ),
        "metadata_every_other_reward_zero": all(
            value == 0.0
            for key, value in metadata.items()
            if key.endswith("_reward") or key.endswith("_penalty")
            if key
            not in {
                "first_physical_touch_reward",
            }
        ),
    }
    checks["passed"] = all(checks.values())
    return {
        "truth_table_version": "RivalFirstTouchVelocityRewardTruthTableV1",
        "metadata": metadata,
        "observations": {
            name: {
                "directed_velocity_uu_per_second": step.directed_velocity_uu_per_second,
                "normalized_directed_velocity": step.normalized_directed_velocity,
                "velocity_to_ball_reward": step.velocity_to_ball_reward,
                "physical_touch_reward": step.physical_touch_reward,
                "total": step.total,
            }
            for name, step in (
                ("stationary", stationary),
                ("toward_slow", toward_slow),
                ("toward_fast", toward_fast),
                ("toward_full", toward_full),
                ("perpendicular", perpendicular),
                ("away_slow", away_slow),
                ("away_fast", away_fast),
                ("ball_motion_stationary_car", ball_motion),
                ("first_touch", touch),
                ("sustained_touch", sustained),
                ("later_separated_touch", later),
            )
        },
        "checks": checks,
    }
