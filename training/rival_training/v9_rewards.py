"""Cadence-safe outcome-dominant rewards for the Rival v9 scratch policy.

The reward kernel consumes the same team-normalized canonical state used by
``RivalObsV1``.  It is intentionally separated from the legacy M06/M08 reward
implementations: v9 runs at one policy decision per 120-Hz physics tick, so a
30-Hz per-step reward cannot be copied safely.

Dense terms are either potential differences using the PPO's physical-time
discount or bounded rates multiplied by physical elapsed time.  Every shaping
family also has a declared absolute episode budget.  The combined shaping
budget is below one goal event, preserving the explicit outcome-dominant
contract even if a policy discovers an unexpected farming strategy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState

from .v9_canonical import (
    PHYSICS_HZ,
    RivalCanonicalStateV1,
    RocketSimCanonicalAdapterV1,
)
from .v9_soccar_geometry import BACK_WALL_Y, BALL_MAX_SPEED, CAR_MAX_SPEED


REWARD_VERSION = "RivalScratchRewardV1"
REWARD_SCHEDULE_VERSION = "RivalScratchRewardScheduleV1"
GAMMA_120HZ = 0.99 ** (1.0 / 8.0)
GOAL_REWARD = 10.0
CONCEDE_REWARD = -10.0

COMPONENTS = (
    "outcome",
    "ball_progress_potential",
    "approach_control_potential",
    "touch_quality_event",
    "recovery_potential",
    "boost_waste_rate",
    "dodge_resource_event",
)
SHAPING_COMPONENTS = COMPONENTS[1:]

# Absolute episode spend, not signed return.  These caps are deliberately
# conservative and sum to 7.25, below one +/-10 goal event.
SHAPING_ABSOLUTE_EPISODE_BUDGETS: Mapping[str, float] = {
    "ball_progress_potential": 2.0,
    "approach_control_potential": 1.0,
    "touch_quality_event": 2.0,
    "recovery_potential": 2.5,
    "boost_waste_rate": 1.0,
    "dodge_resource_event": 0.25,
}


@dataclass(frozen=True)
class RewardWeightsV1:
    """Weights for one explicitly versioned scratch-learning phase."""

    ball_progress_potential: float
    approach_control_potential: float
    touch_quality_event: float
    recovery_potential: float
    boost_waste_rate_per_second: float
    dodge_resource_event: float


@dataclass(frozen=True)
class RewardPhaseV1:
    name: str
    minimum_simulated_game_hours: float
    requires_readiness_gate: bool
    weights: RewardWeightsV1


REWARD_PHASES: tuple[RewardPhaseV1, ...] = (
    RewardPhaseV1(
        name="foundation",
        minimum_simulated_game_hours=0.0,
        requires_readiness_gate=False,
        weights=RewardWeightsV1(
            ball_progress_potential=0.08,
            approach_control_potential=0.05,
            touch_quality_event=0.12,
            recovery_potential=0.04,
            boost_waste_rate_per_second=0.015,
            dodge_resource_event=0.005,
        ),
    ),
    RewardPhaseV1(
        name="competence",
        minimum_simulated_game_hours=25.0,
        requires_readiness_gate=True,
        weights=RewardWeightsV1(
            ball_progress_potential=0.06,
            approach_control_potential=0.02,
            touch_quality_event=0.10,
            recovery_potential=0.04,
            boost_waste_rate_per_second=0.0125,
            dodge_resource_event=0.002,
        ),
    ),
    RewardPhaseV1(
        name="mature",
        minimum_simulated_game_hours=250.0,
        requires_readiness_gate=True,
        weights=RewardWeightsV1(
            ball_progress_potential=0.05,
            approach_control_potential=0.005,
            touch_quality_event=0.08,
            recovery_potential=0.03,
            boost_waste_rate_per_second=0.01,
            dodge_resource_event=0.0,
        ),
    ),
)
_PHASE_BY_NAME = {phase.name: phase for phase in REWARD_PHASES}


def select_reward_phase(
    simulated_game_hours: float,
    *,
    competence_ready: bool = False,
    mature_ready: bool = False,
) -> RewardPhaseV1:
    """Select a phase only after both its time floor and readiness gate pass.

    The v9 pilot is capped at two simulated game-hours, so it remains in the
    foundation phase.  Future campaigns cannot transition merely because a raw
    step counter crossed a threshold: they must explicitly publish readiness
    metrics through the corresponding flag.
    """

    hours = max(0.0, float(simulated_game_hours))
    if mature_ready and hours >= _PHASE_BY_NAME["mature"].minimum_simulated_game_hours:
        return _PHASE_BY_NAME["mature"]
    if (
        competence_ready
        and hours >= _PHASE_BY_NAME["competence"].minimum_simulated_game_hours
    ):
        return _PHASE_BY_NAME["competence"]
    return _PHASE_BY_NAME["foundation"]


@dataclass(frozen=True)
class RewardStateV1:
    """Small immutable team-normalized view used by the pure reward kernel."""

    tick_index: int
    self_position: np.ndarray
    self_linear_velocity: np.ndarray
    self_forward: np.ndarray
    self_up: np.ndarray
    self_boost: float
    self_surface_contact: bool
    self_boosting: bool
    self_supersonic: bool
    self_can_dodge: bool
    ball_position: np.ndarray
    ball_linear_velocity: np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "self_position",
            "self_linear_velocity",
            "self_forward",
            "self_up",
            "ball_position",
            "ball_linear_velocity",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {value.shape}")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains a non-finite value")
            object.__setattr__(self, name, value)
        if not math.isfinite(float(self.self_boost)):
            raise ValueError("self_boost must be finite")


def reward_state_from_canonical(state: RivalCanonicalStateV1) -> RewardStateV1:
    """Project the shared canonical state without introducing new feature math."""

    car = state.self_car
    return RewardStateV1(
        tick_index=int(state.tick_index),
        self_position=car.physics.position,
        self_linear_velocity=car.physics.linear_velocity,
        self_forward=car.physics.forward,
        self_up=car.physics.up,
        self_boost=float(car.boost),
        self_surface_contact=bool(car.surface_contact),
        self_boosting=bool(car.boosting),
        self_supersonic=bool(car.supersonic),
        self_can_dodge=bool(car.can_dodge),
        ball_position=state.ball.position,
        ball_linear_velocity=state.ball.linear_velocity,
    )


@dataclass(frozen=True)
class RewardEventsV1:
    goal_for: bool = False
    goal_against: bool = False
    self_touch: bool = False
    touch_event_count: int = 0
    touch_quality_sum: float | None = None
    touch_velocity_gain_sum: float | None = None

    def __post_init__(self) -> None:
        if self.goal_for and self.goal_against:
            raise ValueError("A transition cannot be both a goal for and goal against")
        if self.touch_event_count < 0:
            raise ValueError("touch_event_count cannot be negative")
        count = self.touch_event_count or int(self.self_touch)
        if count == 0 and (
            self.touch_quality_sum is not None
            or self.touch_velocity_gain_sum is not None
        ):
            raise ValueError("Touch aggregates require at least one touch event")
        for name in ("touch_quality_sum", "touch_velocity_gain_sum"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class RewardStepV1:
    total: float
    components: Mapping[str, float]
    proposals: Mapping[str, float]
    delta_ticks: int
    phase: str
    budget_clipped: tuple[str, ...]
    detectors: Mapping[str, float]


def _safe_unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        return np.zeros(3, dtype=np.float64)
    return np.asarray(vector, dtype=np.float64) / norm


def _ball_progress_potential(state: RewardStateV1) -> float:
    # In the canonical perspective +Y always points toward the opponent goal.
    position = float(np.clip(state.ball_position[1] / BACK_WALL_Y, -1.0, 1.0))
    useful_velocity = float(
        np.clip(state.ball_linear_velocity[1] / BALL_MAX_SPEED, -1.0, 1.0)
    )
    return float(np.clip(0.72 * position + 0.28 * useful_velocity, -1.0, 1.0))


def _approach_control_potential(state: RewardStateV1) -> float:
    to_ball = state.ball_position - state.self_position
    direction = _safe_unit(to_ball)
    distance = float(np.linalg.norm(to_ball))
    distance_quality = 1.0 - 2.0 * float(np.clip(distance / 6000.0, 0.0, 1.0))
    closing_quality = float(
        np.clip(
            np.dot(state.self_linear_velocity, direction) / CAR_MAX_SPEED,
            -1.0,
            1.0,
        )
    )
    facing_quality = float(np.clip(np.dot(state.self_forward, direction), -1.0, 1.0))
    return float(
        np.clip(
            0.55 * distance_quality + 0.25 * closing_quality + 0.20 * facing_quality,
            -1.0,
            1.0,
        )
    )


def _recovery_potential(state: RewardStateV1) -> float:
    alignment = float(np.clip(state.self_up[2], -1.0, 1.0))
    speed_quality = 2.0 * float(
        np.clip(np.linalg.norm(state.self_linear_velocity) / CAR_MAX_SPEED, 0.0, 1.0)
    ) - 1.0
    goal_side = 1.0 if state.self_position[1] <= state.ball_position[1] else -1.0
    surface_bonus = 1.0 if state.self_surface_contact else -1.0
    return float(
        np.clip(
            0.40 * alignment
            + 0.25 * speed_quality
            + 0.20 * goal_side
            + 0.15 * surface_bonus,
            -1.0,
            1.0,
        )
    )


def reward_potentials(state: RewardStateV1) -> dict[str, float]:
    values = {
        "ball_progress_potential": _ball_progress_potential(state),
        "approach_control_potential": _approach_control_potential(state),
        "recovery_potential": _recovery_potential(state),
    }
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values.values()):
        raise FloatingPointError(f"Invalid Rival v9 reward potentials: {values}")
    return values


def touch_quality_from_transition(
    previous: RewardStateV1, current: RewardStateV1
) -> tuple[float, float]:
    """Return bounded useful touch quality and its attacking-velocity gain.

    Exposing this pure calculation lets cadence audits preserve every native
    event's immediate consequence when several 120-Hz events fall inside one
    wider diagnostic sampling interval.  The production path remains one tick.
    """

    velocity_gain = float(
        np.clip(
            (
                current.ball_linear_velocity[1]
                - previous.ball_linear_velocity[1]
            )
            / BALL_MAX_SPEED,
            -1.0,
            1.0,
        )
    )
    resulting_control = float(
        np.clip(current.ball_linear_velocity[1] / BALL_MAX_SPEED, -1.0, 1.0)
    )
    quality = 0.75 * velocity_gain + 0.25 * resulting_control
    return float(quality), velocity_gain


class RivalRewardKernelV1:
    """Pure cadence-aware per-perspective reward transition kernel."""

    def __init__(self) -> None:
        self.previous: RewardStateV1 | None = None
        self.previous_potentials: dict[str, float] = {}
        self.absolute_spend = {name: 0.0 for name in SHAPING_COMPONENTS}
        self.component_totals = {name: 0.0 for name in COMPONENTS}
        self.component_absolute_totals = {name: 0.0 for name in COMPONENTS}

    def reset(self, initial_state: RewardStateV1) -> None:
        self.previous = initial_state
        self.previous_potentials = reward_potentials(initial_state)
        self.absolute_spend = {name: 0.0 for name in SHAPING_COMPONENTS}
        self.component_totals = {name: 0.0 for name in COMPONENTS}
        self.component_absolute_totals = {name: 0.0 for name in COMPONENTS}

    def _budget(self, name: str, proposal: float) -> tuple[float, bool]:
        maximum = float(SHAPING_ABSOLUTE_EPISODE_BUDGETS[name])
        remaining = max(0.0, maximum - self.absolute_spend[name])
        value = math.copysign(min(abs(proposal), remaining), proposal)
        self.absolute_spend[name] += abs(value)
        return value, abs(value - proposal) > 1e-15

    def step(
        self,
        state: RewardStateV1,
        events: RewardEventsV1,
        phase: RewardPhaseV1,
    ) -> RewardStepV1:
        if self.previous is None:
            raise RuntimeError("RivalRewardKernelV1 must be reset before step")
        delta_ticks = int(state.tick_index) - int(self.previous.tick_index)
        if delta_ticks <= 0:
            raise ValueError(
                "Reward transitions require strictly increasing native tick indices; "
                f"got {self.previous.tick_index} -> {state.tick_index}"
            )
        elapsed_seconds = delta_ticks / PHYSICS_HZ
        gamma_delta = GAMMA_120HZ**delta_ticks
        weights = phase.weights
        current_potentials = reward_potentials(state)

        outcome = 0.0
        if events.goal_for:
            outcome = GOAL_REWARD
        elif events.goal_against:
            outcome = CONCEDE_REWARD

        proposals = {
            "outcome": outcome,
            "ball_progress_potential": weights.ball_progress_potential
            * (
                gamma_delta * current_potentials["ball_progress_potential"]
                - self.previous_potentials["ball_progress_potential"]
            ),
            "approach_control_potential": weights.approach_control_potential
            * (
                gamma_delta * current_potentials["approach_control_potential"]
                - self.previous_potentials["approach_control_potential"]
            ),
            "touch_quality_event": 0.0,
            "recovery_potential": weights.recovery_potential
            * (
                gamma_delta * current_potentials["recovery_potential"]
                - self.previous_potentials["recovery_potential"]
            ),
            "boost_waste_rate": 0.0,
            "dodge_resource_event": 0.0,
        }

        touch_count = events.touch_event_count or int(events.self_touch)
        touch_velocity_gain = 0.0
        if touch_count:
            if events.touch_quality_sum is None:
                touch_quality, touch_velocity_gain = touch_quality_from_transition(
                    self.previous, state
                )
            else:
                touch_quality = float(events.touch_quality_sum)
                touch_velocity_gain = float(events.touch_velocity_gain_sum or 0.0)
            proposals["touch_quality_event"] = weights.touch_quality_event * touch_quality

        speed = float(np.linalg.norm(state.self_linear_velocity))
        wasteful_boost = bool(
            state.self_boosting
            and state.self_supersonic
            and speed >= CAR_MAX_SPEED - 10.0
        )
        if wasteful_boost:
            proposals["boost_waste_rate"] = (
                -weights.boost_waste_rate_per_second * elapsed_seconds
            )

        acquired_airborne_dodge = bool(
            not self.previous.self_can_dodge
            and state.self_can_dodge
            and not state.self_surface_contact
        )
        if acquired_airborne_dodge:
            proposals["dodge_resource_event"] = weights.dodge_resource_event

        if not all(math.isfinite(value) for value in proposals.values()):
            raise FloatingPointError(f"Non-finite Rival v9 reward proposal: {proposals}")

        components = {"outcome": outcome}
        clipped: list[str] = []
        for name in SHAPING_COMPONENTS:
            components[name], was_clipped = self._budget(name, proposals[name])
            if was_clipped:
                clipped.append(name)
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise FloatingPointError(f"Non-finite Rival v9 total reward: {components}")

        for name, value in components.items():
            self.component_totals[name] += value
            self.component_absolute_totals[name] += abs(value)
        self.previous = state
        self.previous_potentials = current_potentials
        return RewardStepV1(
            total=total,
            components=components,
            proposals=proposals,
            delta_ticks=delta_ticks,
            phase=phase.name,
            budget_clipped=tuple(clipped),
            detectors={
                "self_touch": float(touch_count),
                "touch_velocity_gain": touch_velocity_gain,
                "wasteful_boost": float(wasteful_boost),
                "airborne_dodge_acquired": float(acquired_airborne_dodge),
            },
        )


def _audit_record(
    kernels: Mapping[AgentID, RivalRewardKernelV1],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for name in COMPONENTS:
        output[name] = {
            "cumulative_signed": float(
                sum(kernel.component_totals[name] for kernel in kernels.values())
            ),
            "cumulative_absolute": float(
                sum(kernel.component_absolute_totals[name] for kernel in kernels.values())
            ),
        }
        if name in SHAPING_ABSOLUTE_EPISODE_BUDGETS:
            output[name]["absolute_episode_budget_per_agent"] = float(
                SHAPING_ABSOLUTE_EPISODE_BUDGETS[name]
            )
    return output


class RivalScratchRewardV1(RewardFunction[AgentID, GameState, float]):
    """RLGym wrapper around the shared pure v9 reward transition kernel."""

    def __init__(self) -> None:
        self.adapter = RocketSimCanonicalAdapterV1()
        self.kernels: dict[AgentID, RivalRewardKernelV1] = {}
        self.team_by_agent: dict[AgentID, int] = {}
        self.last_goal_tick: int | None = None

    @staticmethod
    def _phase(shared_info: Mapping[str, Any]) -> RewardPhaseV1:
        return select_reward_phase(
            float(shared_info.get("rival_v9_simulated_game_hours", 0.0)),
            competence_ready=bool(
                shared_info.get("rival_v9_reward_competence_ready", False)
            ),
            mature_ready=bool(shared_info.get("rival_v9_reward_mature_ready", False)),
        )

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        self.adapter.reset()
        self.team_by_agent = {
            agent: int(initial_state.cars[agent].team_num) for agent in agents
        }
        self.kernels = {agent: RivalRewardKernelV1() for agent in agents}
        for agent in agents:
            canonical = self.adapter.adapt(initial_state, agent, shared_info)
            self.kernels[agent].reset(reward_state_from_canonical(canonical))
        self.last_goal_tick = None
        phase = self._phase(shared_info)
        shared_info["rival_v9_reward_mode"] = REWARD_VERSION
        shared_info["rival_v9_reward_schedule_version"] = REWARD_SCHEDULE_VERSION
        shared_info["rival_v9_reward_phase"] = phase.name
        shared_info["reward_components"] = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        shared_info["reward_component_proposals"] = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        shared_info["reward_component_audit"] = _audit_record(self.kernels)

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        del is_terminated, is_truncated
        phase = self._phase(shared_info)
        tick = int(state.tick_count)
        goal_now = bool(state.goal_scored) and self.last_goal_tick != tick
        if goal_now:
            self.last_goal_tick = tick
        scoring_team = int(state.scoring_team) if goal_now else -1

        steps: dict[AgentID, RewardStepV1] = {}
        for agent in agents:
            canonical = self.adapter.adapt(state, agent, shared_info)
            team = self.team_by_agent[agent]
            events = RewardEventsV1(
                goal_for=goal_now and scoring_team == team,
                goal_against=goal_now and scoring_team != team,
                self_touch=int(state.cars[agent].ball_touches) > 0,
            )
            steps[agent] = self.kernels[agent].step(
                reward_state_from_canonical(canonical), events, phase
            )

        shared_info["rival_v9_reward_phase"] = phase.name
        shared_info["reward_components"] = {
            agent: dict(step.components) for agent, step in steps.items()
        }
        shared_info["reward_component_proposals"] = {
            agent: dict(step.proposals) for agent, step in steps.items()
        }
        shared_info["reward_detectors"] = {
            agent: dict(step.detectors) for agent, step in steps.items()
        }
        shared_info["reward_budget_clipped"] = {
            agent: list(step.budget_clipped) for agent, step in steps.items()
        }
        shared_info["reward_component_audit"] = _audit_record(self.kernels)
        return {agent: step.total for agent, step in steps.items()}


def reward_metadata() -> dict[str, Any]:
    """Machine-readable contract embedded in gates/checkpoints/exports."""

    phases = []
    for phase in REWARD_PHASES:
        phases.append(
            {
                "name": phase.name,
                "minimum_simulated_game_hours": phase.minimum_simulated_game_hours,
                "requires_readiness_gate": phase.requires_readiness_gate,
                "weights": asdict(phase.weights),
            }
        )
    return {
        "reward_version": REWARD_VERSION,
        "schedule_version": REWARD_SCHEDULE_VERSION,
        "physics_hz": PHYSICS_HZ,
        "gamma_120hz": GAMMA_120HZ,
        "goal_reward": GOAL_REWARD,
        "concede_reward": CONCEDE_REWARD,
        "components": list(COMPONENTS),
        "shaping_absolute_episode_budgets": dict(
            SHAPING_ABSOLUTE_EPISODE_BUDGETS
        ),
        "combined_shaping_absolute_episode_budget": float(
            sum(SHAPING_ABSOLUTE_EPISODE_BUDGETS.values())
        ),
        "phases": phases,
        "dense_term_contract": (
            "potential terms use weight*(gamma_120hz**delta_ticks*Phi(next)-"
            "Phi(previous)); rate terms use bounded_rate*delta_ticks/120"
        ),
        "named_mechanic_identity_rewards": False,
        "mechanics_events_are_diagnostics_first": True,
    }
