"""Temporary agency-bootstrap reward for Rival Milestone 10.1.

This module is intentionally separate from :mod:`v9_rewards`.  It preserves the
canonical state and native-120-Hz timing contracts while changing only the
learning signal used during the bounded agency bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState

from .v9_canonical import PHYSICS_HZ, RocketSimCanonicalAdapterV1
from .v9_rewards import GAMMA_120HZ, RewardStateV1, reward_state_from_canonical
from .v9_soccar_geometry import BACK_WALL_Y, BALL_MAX_SPEED, CAR_MAX_SPEED


REWARD_VERSION = "RivalAgencyBootstrapRewardV1"
REWARD_SCHEDULE_VERSION = "RivalAgencyBootstrapRewardScheduleV1"
GOAL_REWARD = 10.0
CONCEDE_REWARD = -10.0
TOUCH_DEBOUNCE_NATIVE_TICKS = 8
TOUCH_CHAIN_TIMEOUT_NATIVE_TICKS = 300
AERIAL_TOUCH_MINIMUM_BALL_Z = 180.0

COMPONENTS = (
    "outcome",
    "useful_speed_rate",
    "ball_approach_potential",
    "ball_touch_event",
    "aerial_touch_event",
    "touch_chain_event",
    "ball_progress_potential",
)
SHAPING_COMPONENTS = COMPONENTS[1:]
SHAPING_ABSOLUTE_EPISODE_BUDGETS: Mapping[str, float] = {
    "useful_speed_rate": 1.0,
    "ball_approach_potential": 1.0,
    "ball_touch_event": 2.0,
    "aerial_touch_event": 1.5,
    "touch_chain_event": 1.5,
    "ball_progress_potential": 0.5,
}
COMBINED_SHAPING_ABSOLUTE_EPISODE_BUDGET = float(
    sum(SHAPING_ABSOLUTE_EPISODE_BUDGETS.values())
)

USEFUL_SPEED_MAX_RATE_PER_SECOND = 0.015
BALL_APPROACH_WEIGHT = 0.20
BALL_TOUCH_BASE_REWARD = 0.30
AERIAL_TOUCH_BONUS = 0.45
BALL_PROGRESS_WEIGHT = 0.05
TOUCH_CHAIN_BONUSES = (0.0, 0.10, 0.20, 0.35, 0.50)


def _safe_unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return np.zeros(3, dtype=np.float64) if norm <= 1e-9 else value / norm


def _approach_potential(state: RewardStateV1) -> float:
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


def _ball_progress_potential(state: RewardStateV1) -> float:
    position = float(np.clip(state.ball_position[1] / BACK_WALL_Y, -1.0, 1.0))
    velocity = float(
        np.clip(state.ball_linear_velocity[1] / BALL_MAX_SPEED, -1.0, 1.0)
    )
    return float(np.clip(0.72 * position + 0.28 * velocity, -1.0, 1.0))


def reward_potentials(state: RewardStateV1) -> dict[str, float]:
    values = {
        "ball_approach_potential": _approach_potential(state),
        "ball_progress_potential": _ball_progress_potential(state),
    }
    if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values.values()):
        raise FloatingPointError(f"Invalid bootstrap reward potentials: {values}")
    return values


def useful_speed_rate(state: RewardStateV1, *, delta_ticks: int) -> float:
    speed = float(np.linalg.norm(state.self_linear_velocity))
    speed_norm = float(np.clip(speed / CAR_MAX_SPEED, 0.0, 1.0))
    velocity_dir = _safe_unit(state.self_linear_velocity)
    to_ball = _safe_unit(state.ball_position - state.self_position)
    toward_ball = max(0.0, float(np.dot(velocity_dir, to_ball)))
    useful_factor = 0.35 + 0.65 * toward_ball
    rate = USEFUL_SPEED_MAX_RATE_PER_SECOND * speed_norm**2 * useful_factor
    return float(rate * int(delta_ticks) / PHYSICS_HZ)


def touch_chain_bonus(chain_length: int) -> float:
    if chain_length <= 0:
        return 0.0
    return float(TOUCH_CHAIN_BONUSES[min(chain_length, 5) - 1])


@dataclass(frozen=True)
class BootstrapRewardEventsV1:
    goal_for: bool = False
    goal_against: bool = False
    raw_touch_records: int = 0
    logical_touch: bool = False
    aerial_touch: bool = False
    touch_chain_length: int = 0

    def __post_init__(self) -> None:
        if self.goal_for and self.goal_against:
            raise ValueError("A transition cannot be both goal-for and goal-against")
        if self.raw_touch_records < 0:
            raise ValueError("raw_touch_records cannot be negative")
        if self.aerial_touch and not self.logical_touch:
            raise ValueError("An aerial reward requires a logical touch")
        if self.logical_touch != (self.touch_chain_length > 0):
            raise ValueError("Logical touch and chain length must agree")


@dataclass(frozen=True)
class BootstrapRewardStepV1:
    total: float
    components: Mapping[str, float]
    proposals: Mapping[str, float]
    delta_ticks: int
    budget_clipped: tuple[str, ...]
    detectors: Mapping[str, float]


class RivalAgencyBootstrapRewardKernelV1:
    """Pure per-agent cadence-aware bootstrap reward kernel."""

    def __init__(self) -> None:
        self.previous: RewardStateV1 | None = None
        self.previous_potentials: dict[str, float] = {}
        self.absolute_spend = {name: 0.0 for name in SHAPING_COMPONENTS}
        self.component_totals = {name: 0.0 for name in COMPONENTS}
        self.component_absolute_totals = {name: 0.0 for name in COMPONENTS}
        self.budget_clip_counts = {name: 0 for name in SHAPING_COMPONENTS}

    def reset(self, initial_state: RewardStateV1) -> None:
        self.previous = initial_state
        self.previous_potentials = reward_potentials(initial_state)
        self.absolute_spend = {name: 0.0 for name in SHAPING_COMPONENTS}
        self.component_totals = {name: 0.0 for name in COMPONENTS}
        self.component_absolute_totals = {name: 0.0 for name in COMPONENTS}
        self.budget_clip_counts = {name: 0 for name in SHAPING_COMPONENTS}

    def _budget(self, name: str, proposal: float) -> tuple[float, bool]:
        remaining = max(
            0.0,
            float(SHAPING_ABSOLUTE_EPISODE_BUDGETS[name])
            - self.absolute_spend[name],
        )
        value = math.copysign(min(abs(proposal), remaining), proposal)
        self.absolute_spend[name] += abs(value)
        clipped = abs(value - proposal) > 1e-15
        if clipped:
            self.budget_clip_counts[name] += 1
        return value, clipped

    def step(
        self,
        state: RewardStateV1,
        events: BootstrapRewardEventsV1,
    ) -> BootstrapRewardStepV1:
        if self.previous is None:
            raise RuntimeError("Bootstrap reward kernel must be reset before step")
        delta_ticks = int(state.tick_index) - int(self.previous.tick_index)
        if delta_ticks <= 0:
            raise ValueError("Bootstrap reward transitions require increasing native ticks")
        current_potentials = reward_potentials(state)
        outcome = (
            GOAL_REWARD
            if events.goal_for
            else CONCEDE_REWARD
            if events.goal_against
            else 0.0
        )
        proposals = {
            "outcome": outcome,
            "useful_speed_rate": useful_speed_rate(state, delta_ticks=delta_ticks),
            "ball_approach_potential": BALL_APPROACH_WEIGHT
            * (
                current_potentials["ball_approach_potential"]
                - self.previous_potentials["ball_approach_potential"]
            ),
            "ball_touch_event": BALL_TOUCH_BASE_REWARD if events.logical_touch else 0.0,
            "aerial_touch_event": AERIAL_TOUCH_BONUS if events.aerial_touch else 0.0,
            "touch_chain_event": touch_chain_bonus(events.touch_chain_length),
            "ball_progress_potential": BALL_PROGRESS_WEIGHT
            * (
                current_potentials["ball_progress_potential"]
                - self.previous_potentials["ball_progress_potential"]
            ),
        }
        # Outcome precedence keeps goal/concede transitions exactly +/-10.
        if outcome:
            for name in SHAPING_COMPONENTS:
                proposals[name] = 0.0
        if not all(math.isfinite(value) for value in proposals.values()):
            raise FloatingPointError(f"Non-finite bootstrap reward proposal: {proposals}")

        components = {"outcome": outcome}
        clipped: list[str] = []
        for name in SHAPING_COMPONENTS:
            components[name], was_clipped = self._budget(name, proposals[name])
            if was_clipped:
                clipped.append(name)
        total = float(sum(components.values()))
        if not math.isfinite(total):
            raise FloatingPointError(f"Non-finite bootstrap total: {components}")
        if sum(self.absolute_spend.values()) > COMBINED_SHAPING_ABSOLUTE_EPISODE_BUDGET + 1e-12:
            raise RuntimeError("Bootstrap shaping exceeded the 7.5 episode ceiling")
        for name, value in components.items():
            self.component_totals[name] += value
            self.component_absolute_totals[name] += abs(value)
        self.previous = state
        self.previous_potentials = current_potentials
        return BootstrapRewardStepV1(
            total=total,
            components=components,
            proposals=proposals,
            delta_ticks=delta_ticks,
            budget_clipped=tuple(clipped),
            detectors={
                "raw_touch_records": float(events.raw_touch_records),
                "logical_touch": float(events.logical_touch),
                "aerial_touch": float(events.aerial_touch),
                "touch_chain_length": float(events.touch_chain_length),
            },
        )


def _audit_record(
    kernels: Mapping[AgentID, RivalAgencyBootstrapRewardKernelV1],
) -> dict[str, Any]:
    per_agent: dict[str, Any] = {}
    for agent, kernel in kernels.items():
        per_agent[str(agent)] = {
            "components": {
                name: {
                    "cumulative_signed": float(kernel.component_totals[name]),
                    "cumulative_absolute": float(kernel.component_absolute_totals[name]),
                    **(
                        {
                            "absolute_episode_budget": float(
                                SHAPING_ABSOLUTE_EPISODE_BUDGETS[name]
                            ),
                            "budget_clip_count": int(kernel.budget_clip_counts[name]),
                        }
                        if name in SHAPING_COMPONENTS
                        else {}
                    ),
                }
                for name in COMPONENTS
            },
            "combined_non_outcome_absolute_spend": float(
                sum(kernel.absolute_spend.values())
            ),
            "combined_non_outcome_absolute_budget": (
                COMBINED_SHAPING_ABSOLUTE_EPISODE_BUDGET
            ),
        }
    return {"per_agent": per_agent}


@dataclass(frozen=True)
class LogicalTouchDecisionV1:
    raw_touch_records: int
    logical_touch: bool
    aerial_touch: bool
    touch_chain_length: int


class RivalLogicalTouchAuditorV1:
    """Pure multi-agent debounce and possession-chain state machine."""

    def __init__(self) -> None:
        self.last_rewarded_touch_tick: dict[AgentID, int | None] = {}
        self.last_opponent_logical_tick: dict[AgentID, int | None] = {}
        self.last_chain_touch_tick: dict[AgentID, int | None] = {}
        self.chain_length: dict[AgentID, int] = {}

    def reset(self, agents: list[AgentID]) -> None:
        self.last_rewarded_touch_tick = {agent: None for agent in agents}
        self.last_opponent_logical_tick = {agent: None for agent in agents}
        self.last_chain_touch_tick = {agent: None for agent in agents}
        self.chain_length = {agent: 0 for agent in agents}

    def process(
        self,
        agents: list[AgentID],
        *,
        tick: int,
        raw_touch_records: Mapping[AgentID, int],
        surface_contact: Mapping[AgentID, bool],
        ball_z: float,
        goal: bool = False,
    ) -> dict[AgentID, LogicalTouchDecisionV1]:
        if set(self.chain_length) != set(agents):
            raise RuntimeError("Logical-touch auditor must be reset for the active agents")
        raw = {agent: max(0, int(raw_touch_records.get(agent, 0))) for agent in agents}
        logical: dict[AgentID, bool] = {}
        for agent in agents:
            last_self = self.last_rewarded_touch_tick[agent]
            last_opponent = self.last_opponent_logical_tick[agent]
            opponent_since_self = bool(
                last_opponent is not None
                and (last_self is None or last_opponent > last_self)
            )
            logical[agent] = bool(
                raw[agent] > 0
                and (
                    last_self is None
                    or tick - last_self >= TOUCH_DEBOUNCE_NATIVE_TICKS
                    or opponent_since_self
                )
            )
        decisions: dict[AgentID, LogicalTouchDecisionV1] = {}
        for agent in agents:
            opponent_logical = any(logical[other] for other in agents if other != agent)
            last_chain = self.last_chain_touch_tick[agent]
            if (
                goal
                or opponent_logical
                or (
                    last_chain is not None
                    and tick - last_chain > TOUCH_CHAIN_TIMEOUT_NATIVE_TICKS
                )
            ):
                self.chain_length[agent] = 0
            if logical[agent]:
                self.chain_length[agent] += 1
                self.last_rewarded_touch_tick[agent] = tick
                self.last_chain_touch_tick[agent] = tick
            if opponent_logical:
                self.last_opponent_logical_tick[agent] = tick
            decisions[agent] = LogicalTouchDecisionV1(
                raw_touch_records=raw[agent],
                logical_touch=logical[agent],
                aerial_touch=bool(
                    logical[agent]
                    and not bool(surface_contact[agent])
                    and float(ball_z) >= AERIAL_TOUCH_MINIMUM_BALL_Z
                ),
                touch_chain_length=(self.chain_length[agent] if logical[agent] else 0),
            )
        return decisions


class RivalAgencyBootstrapRewardV1(RewardFunction[AgentID, GameState, float]):
    """RLGym wrapper with centralized logical-touch debounce and chain state."""

    def __init__(self) -> None:
        self.adapter = RocketSimCanonicalAdapterV1()
        self.kernels: dict[AgentID, RivalAgencyBootstrapRewardKernelV1] = {}
        self.team_by_agent: dict[AgentID, int] = {}
        self.last_goal_tick: int | None = None
        self.touch_auditor = RivalLogicalTouchAuditorV1()

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
        self.kernels = {
            agent: RivalAgencyBootstrapRewardKernelV1() for agent in agents
        }
        for agent in agents:
            canonical = self.adapter.adapt(initial_state, agent, shared_info)
            self.kernels[agent].reset(reward_state_from_canonical(canonical))
        self.last_goal_tick = None
        self.touch_auditor.reset(agents)
        shared_info["rival_v10_reward_mode"] = REWARD_VERSION
        shared_info["rival_v10_reward_schedule_version"] = REWARD_SCHEDULE_VERSION
        shared_info["reward_components"] = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        shared_info["reward_component_proposals"] = {
            agent: {name: 0.0 for name in COMPONENTS} for agent in agents
        }
        shared_info["rival_v10_touch_metrics"] = {
            agent: {
                "raw_touch_records": 0,
                "logical_touch": False,
                "aerial_touch": False,
                "touch_chain_length": 0,
            }
            for agent in agents
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
        tick = int(state.tick_count)
        goal_now = bool(state.goal_scored) and self.last_goal_tick != tick
        if goal_now:
            self.last_goal_tick = tick
        scoring_team = int(state.scoring_team) if goal_now else -1
        raw = {agent: max(0, int(state.cars[agent].ball_touches)) for agent in agents}

        states: dict[AgentID, RewardStateV1] = {}
        for agent in agents:
            canonical = self.adapter.adapt(state, agent, shared_info)
            states[agent] = reward_state_from_canonical(canonical)
        decisions = self.touch_auditor.process(
            agents,
            tick=tick,
            raw_touch_records=raw,
            surface_contact={
                agent: states[agent].self_surface_contact for agent in agents
            },
            ball_z=float(state.ball.position[2]),
            goal=goal_now,
        )
        metrics: dict[AgentID, dict[str, Any]] = {}
        events: dict[AgentID, BootstrapRewardEventsV1] = {}
        for agent in agents:
            decision = decisions[agent]
            team = self.team_by_agent[agent]
            events[agent] = BootstrapRewardEventsV1(
                goal_for=goal_now and scoring_team == team,
                goal_against=goal_now and scoring_team != team,
                raw_touch_records=decision.raw_touch_records,
                logical_touch=decision.logical_touch,
                aerial_touch=decision.aerial_touch,
                touch_chain_length=decision.touch_chain_length,
            )
            metrics[agent] = {
                "raw_touch_records": decision.raw_touch_records,
                "logical_touch": decision.logical_touch,
                "aerial_touch": decision.aerial_touch,
                "touch_chain_length": decision.touch_chain_length,
            }

        steps = {
            agent: self.kernels[agent].step(states[agent], events[agent])
            for agent in agents
        }
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
        shared_info["rival_v10_touch_metrics"] = metrics
        shared_info["reward_component_audit"] = _audit_record(self.kernels)
        return {agent: step.total for agent, step in steps.items()}


def reward_metadata() -> dict[str, Any]:
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
        "combined_shaping_absolute_episode_budget": (
            COMBINED_SHAPING_ABSOLUTE_EPISODE_BUDGET
        ),
        "touch_debounce_native_ticks": TOUCH_DEBOUNCE_NATIVE_TICKS,
        "touch_chain_timeout_native_ticks": TOUCH_CHAIN_TIMEOUT_NATIVE_TICKS,
        "aerial_touch_minimum_ball_z": AERIAL_TOUCH_MINIMUM_BALL_Z,
        "direct_action_press_rewards": False,
        "recovery_reward": 0.0,
        "boost_waste_penalty": 0.0,
        "dodge_resource_reward": 0.0,
        "outcome_precedence_exact": True,
    }
