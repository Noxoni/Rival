"""Outcome-dominant, independently instrumented reward v1."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y, SIDE_WALL_X


REWARD_VERSION = "RivalOutcomeRewardV1"
COMPONENTS = ("outcome", "possession", "progress", "boost_efficiency", "recovery")


class RivalRewardV1(RewardFunction[AgentID, GameState, float]):
    """Small modular shaping terms beneath a signed 10-point goal outcome."""

    def __init__(self) -> None:
        self.previous_ball_y = 0.0
        self.previous_boost: dict[AgentID, float] = {}
        self.previous_on_ground: dict[AgentID, bool] = {}
        self.recovery_active: dict[AgentID, bool] = {}
        self.component_totals: dict[str, float] = defaultdict(float)
        self.component_samples = 0

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        self.previous_ball_y = float(initial_state.ball.position[1])
        self.previous_boost = {
            agent: float(initial_state.cars[agent].boost_amount) for agent in agents
        }
        self.previous_on_ground = {
            agent: bool(initial_state.cars[agent].on_ground) for agent in agents
        }
        self.recovery_active = {agent: False for agent in agents}
        shared_info["reward_components"] = {
            agent: {component: 0.0 for component in COMPONENTS} for agent in agents
        }

    @staticmethod
    def _distance_to_ball(state: GameState, agent: AgentID) -> float:
        return float(
            np.linalg.norm(state.cars[agent].physics.position - state.ball.position)
        )

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        del is_terminated, is_truncated
        ball_delta_y = float(state.ball.position[1]) - self.previous_ball_y
        distances = {agent: self._distance_to_ball(state, agent) for agent in agents}
        components_by_agent: dict[AgentID, dict[str, float]] = {}
        rewards: dict[AgentID, float] = {}

        for agent in agents:
            car = state.cars[agent]
            attack_direction = 1.0 if car.team_num == 0 else -1.0
            opponent_distances = [
                distances[other]
                for other in agents
                if state.cars[other].team_num != car.team_num
            ]
            opponent_distance = min(opponent_distances, default=distances[agent])
            touch = car.ball_touches > 0

            outcome = 0.0
            if state.goal_scored:
                outcome = 10.0 if state.scoring_team == car.team_num else -10.0

            signed_progress = attack_direction * ball_delta_y
            progress = 0.02 * float(np.clip(signed_progress / 1000.0, -1.0, 1.0))

            control_advantage = float(
                np.clip((opponent_distance - distances[agent]) / 2000.0, -1.0, 1.0)
            )
            possession = 0.0
            if touch:
                possession = 0.05
                possession += 0.10 * max(0.0, control_advantage)
                possession += 0.10 * float(
                    np.clip(signed_progress / 1000.0, -1.0, 1.0)
                )

            previous_boost = self.previous_boost.get(agent, float(car.boost_amount))
            spent = max(0.0, previous_boost - float(car.boost_amount)) / 100.0
            useful_spend = touch or signed_progress > 40.0
            boost_efficiency = -0.02 * spent
            if useful_spend:
                boost_efficiency += 0.01 * spent

            opponent_controls = opponent_distance + 100.0 < distances[agent]
            if not car.on_ground and opponent_controls:
                self.recovery_active[agent] = True
            landed = car.on_ground and not self.previous_on_ground.get(agent, True)
            recovery = 0.0
            if landed and self.recovery_active.get(agent, False):
                own_goal_y = -BACK_WALL_Y if car.team_num == 0 else BACK_WALL_Y
                to_own_goal = np.array(
                    [0.0, own_goal_y, 0.0], dtype=np.float32
                ) - car.physics.position
                goal_side = float(
                    np.dot(car.physics.linear_velocity, to_own_goal)
                    / max(np.linalg.norm(to_own_goal) * 2300.0, 1.0)
                )
                speed_value = float(np.linalg.norm(car.physics.linear_velocity)) / 2300.0
                recovery = 0.02 + 0.01 * float(np.clip(goal_side, -1.0, 1.0))
                recovery += 0.01 * float(np.clip(speed_value, 0.0, 1.0))
                recovery += 0.005 * float(np.clip(car.boost_amount / 100.0, 0.0, 1.0))
                self.recovery_active[agent] = False

            component_values = {
                "outcome": outcome,
                "possession": possession,
                "progress": progress,
                "boost_efficiency": boost_efficiency,
                "recovery": recovery,
            }
            if not all(math.isfinite(value) for value in component_values.values()):
                raise FloatingPointError(f"Non-finite reward components: {component_values}")
            total = float(sum(component_values.values()))
            if not math.isfinite(total):
                raise FloatingPointError(f"Non-finite reward total for {agent}: {total}")
            components_by_agent[agent] = component_values
            rewards[agent] = total
            for name, value in component_values.items():
                self.component_totals[name] += value

            self.previous_boost[agent] = float(car.boost_amount)
            self.previous_on_ground[agent] = bool(car.on_ground)

        self.previous_ball_y = float(state.ball.position[1])
        self.component_samples += len(agents)
        shared_info["reward_components"] = components_by_agent
        shared_info["reward_component_totals"] = dict(self.component_totals)
        shared_info["reward_component_samples"] = self.component_samples
        return rewards


def reward_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reward_version": REWARD_VERSION,
        "components": list(COMPONENTS),
        "outcome_goal": 10.0,
        "outcome_concede": -10.0,
        "shaping_policy": "All shaping terms are small and independently logged; no named-mechanic reward is enabled.",
    }


REWARD_V2_VERSION = "RivalRewardV2"
V2_COMPONENTS = (
    "outcome",
    "possession",
    "progress",
    "boost_efficiency",
    "recovery",
    "mechanics_resource",
)
MECHANICS_METRICS = (
    "airborne_dodge_uses",
    "dodge_resource_acquisitions",
    "productive_resource_followups",
    "wavedash_like_events",
    "wall_speed_recovery_events",
    "aerial_boost_spent",
    "aerial_useful_distance",
    "aerial_useful_touches",
    "aerial_commitment_ends",
    "aerial_end_boost",
    "recovery_window_starts",
    "recovery_completions",
    "recovery_time_seconds",
    "recovery_concessions",
)


def _has_dodge_resource(car) -> bool:
    return bool(
        car.on_ground
        or (
            not car.has_flipped
            and not car.has_double_jumped
            and float(car.air_time_since_jump) < 1.25
        )
    )


class RivalRewardV2(RewardFunction[AgentID, GameState, float]):
    """Outcome-dominant reward plus bounded mechanics/recovery instrumentation.

    Named tricks are deliberately absent. The low-weight mechanics component
    rewards generic resource acquisition and productive use; the richer event
    stream is primarily diagnostic.
    """

    def __init__(self, *, cadence_ticks: int = 4) -> None:
        self.cadence_ticks = int(cadence_ticks)
        self.seconds_per_step = self.cadence_ticks / 120.0
        self.previous_ball_y = 0.0
        self.previous_boost: dict[AgentID, float] = {}
        self.previous_on_ground: dict[AgentID, bool] = {}
        self.previous_has_dodge: dict[AgentID, bool] = {}
        self.previous_has_flipped: dict[AgentID, bool] = {}
        self.previous_has_double_jumped: dict[AgentID, bool] = {}
        self.previous_position: dict[AgentID, np.ndarray] = {}
        self.previous_ball_distance: dict[AgentID, float] = {}
        self.previous_speed: dict[AgentID, float] = {}
        self.recovery_started_step: dict[AgentID, int | None] = {}
        self.aerial_active: dict[AgentID, bool] = {}
        self.resource_followup_steps: dict[AgentID, int] = {}
        self.wall_event_cooldown: dict[AgentID, int] = {}
        self.last_touch_agent: AgentID | None = None
        self.step_count = 0
        self.component_totals: dict[str, float] = defaultdict(float)
        self.component_absolute_totals: dict[str, float] = defaultdict(float)
        self.component_counts: dict[str, int] = defaultdict(int)
        self.component_minimums: dict[str, float] = {}
        self.component_maximums: dict[str, float] = {}

    @staticmethod
    def _distance_to_ball(state: GameState, agent: AgentID) -> float:
        return float(
            np.linalg.norm(state.cars[agent].physics.position - state.ball.position)
        )

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        self.previous_ball_y = float(initial_state.ball.position[1])
        self.previous_boost = {
            agent: float(initial_state.cars[agent].boost_amount) for agent in agents
        }
        self.previous_on_ground = {
            agent: bool(initial_state.cars[agent].on_ground) for agent in agents
        }
        self.previous_has_dodge = {
            agent: _has_dodge_resource(initial_state.cars[agent]) for agent in agents
        }
        self.previous_has_flipped = {
            agent: bool(initial_state.cars[agent].has_flipped) for agent in agents
        }
        self.previous_has_double_jumped = {
            agent: bool(initial_state.cars[agent].has_double_jumped) for agent in agents
        }
        self.previous_position = {
            agent: np.asarray(initial_state.cars[agent].physics.position).copy()
            for agent in agents
        }
        self.previous_ball_distance = {
            agent: self._distance_to_ball(initial_state, agent) for agent in agents
        }
        self.previous_speed = {
            agent: float(
                np.linalg.norm(initial_state.cars[agent].physics.linear_velocity)
            )
            for agent in agents
        }
        self.recovery_started_step = {agent: None for agent in agents}
        self.aerial_active = {agent: False for agent in agents}
        self.resource_followup_steps = {agent: 0 for agent in agents}
        self.wall_event_cooldown = {agent: 0 for agent in agents}
        self.last_touch_agent = None
        self.step_count = 0
        shared_info["reward_components"] = {
            agent: {component: 0.0 for component in V2_COMPONENTS} for agent in agents
        }
        shared_info["mechanics_metrics"] = {
            agent: {metric: 0.0 for metric in MECHANICS_METRICS} for agent in agents
        }

    @staticmethod
    def _goal_side(car, ball_position: np.ndarray) -> bool:
        if car.team_num == 0:
            return float(car.physics.position[1]) <= float(ball_position[1])
        return float(car.physics.position[1]) >= float(ball_position[1])

    def _record_components(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.component_totals[name] += value
            self.component_absolute_totals[name] += abs(value)
            self.component_counts[name] += 1
            self.component_minimums[name] = min(
                value, self.component_minimums.get(name, value)
            )
            self.component_maximums[name] = max(
                value, self.component_maximums.get(name, value)
            )

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        del is_terminated, is_truncated
        self.step_count += 1
        ball_delta_y = float(state.ball.position[1]) - self.previous_ball_y
        distances = {agent: self._distance_to_ball(state, agent) for agent in agents}
        touching_agents = [agent for agent in agents if state.cars[agent].ball_touches > 0]
        new_touch_agent = touching_agents[0] if touching_agents else None
        previous_controller = self.last_touch_agent
        components_by_agent: dict[AgentID, dict[str, float]] = {}
        metrics_by_agent: dict[AgentID, dict[str, float]] = {}
        rewards: dict[AgentID, float] = {}

        for agent in agents:
            car = state.cars[agent]
            attack_direction = 1.0 if car.team_num == 0 else -1.0
            opponent_distances = [
                distances[other]
                for other in agents
                if state.cars[other].team_num != car.team_num
            ]
            opponent_distance = min(opponent_distances, default=distances[agent])
            touch = car.ball_touches > 0
            signed_progress = attack_direction * ball_delta_y
            speed = float(np.linalg.norm(car.physics.linear_velocity))
            has_dodge = _has_dodge_resource(car)
            previous_ground = self.previous_on_ground.get(agent, bool(car.on_ground))
            previous_distance = self.previous_ball_distance.get(agent, distances[agent])
            useful_closing_distance = max(0.0, previous_distance - distances[agent])
            previous_boost = self.previous_boost.get(agent, float(car.boost_amount))
            boost_spent = max(0.0, previous_boost - float(car.boost_amount)) / 100.0
            metrics = {name: 0.0 for name in MECHANICS_METRICS}

            outcome = 0.0
            if state.goal_scored:
                outcome = 10.0 if state.scoring_team == car.team_num else -10.0

            progress = 0.02 * float(np.clip(signed_progress / 1000.0, -1.0, 1.0))
            control_advantage = float(
                np.clip((opponent_distance - distances[agent]) / 2000.0, -1.0, 1.0)
            )
            possession = 0.0
            if touch:
                possession = 0.05
                possession += 0.10 * max(0.0, control_advantage)
                possession += 0.10 * float(
                    np.clip(signed_progress / 1000.0, -1.0, 1.0)
                )

            boost_efficiency = 0.0
            if not car.on_ground and boost_spent > 0.0:
                metrics["aerial_boost_spent"] = boost_spent
                boost_efficiency = -0.01 * boost_spent
                if touch or signed_progress > 40.0 or useful_closing_distance > 20.0:
                    boost_efficiency += 0.01 * boost_spent

            if previous_ground and not car.on_ground:
                self.aerial_active[agent] = True
            if self.aerial_active.get(agent, False) and not car.on_ground:
                metrics["aerial_useful_distance"] = useful_closing_distance
                if touch:
                    metrics["aerial_useful_touches"] = 1.0
            if self.aerial_active.get(agent, False) and car.on_ground and not previous_ground:
                metrics["aerial_commitment_ends"] = 1.0
                metrics["aerial_end_boost"] = float(car.boost_amount)
                self.aerial_active[agent] = False

            acquired_resource = bool(
                not self.previous_has_dodge.get(agent, has_dodge)
                and has_dodge
                and not car.on_ground
            )
            if acquired_resource:
                metrics["dodge_resource_acquisitions"] = 1.0
                self.resource_followup_steps[agent] = 60
            used_airborne_dodge = bool(
                not car.on_ground
                and (
                    (
                        not self.previous_has_flipped.get(agent, car.has_flipped)
                        and car.has_flipped
                    )
                    or (
                        not self.previous_has_double_jumped.get(
                            agent, car.has_double_jumped
                        )
                        and car.has_double_jumped
                    )
                )
            )
            if used_airborne_dodge:
                metrics["airborne_dodge_uses"] = 1.0

            productive_followup = bool(
                self.resource_followup_steps.get(agent, 0) > 0
                and (touch or signed_progress > 120.0 or outcome > 0.0)
            )
            if productive_followup:
                metrics["productive_resource_followups"] = 1.0
                self.resource_followup_steps[agent] = 0
            elif self.resource_followup_steps.get(agent, 0) > 0:
                self.resource_followup_steps[agent] -= 1

            current_action = np.asarray(
                shared_info.get("previous_actions", {}).get(agent, np.zeros(8)),
                dtype=np.float32,
            )
            horizontal_speed = float(np.linalg.norm(car.physics.linear_velocity[:2]))
            wavedash_like = bool(
                not previous_ground
                and car.on_ground
                and current_action[5] > 0.5
                and horizontal_speed > 700.0
                and abs(float(car.physics.linear_velocity[2])) < 650.0
            )
            if wavedash_like:
                metrics["wavedash_like_events"] = 1.0

            position = np.asarray(car.physics.position)
            wall_contact = bool(
                any(car.wheels_with_contact)
                and (
                    abs(float(position[0])) > SIDE_WALL_X - 180.0
                    or abs(float(position[1])) > BACK_WALL_Y - 180.0
                    or float(position[2]) > 180.0
                )
            )
            if self.wall_event_cooldown.get(agent, 0) > 0:
                self.wall_event_cooldown[agent] -= 1
            wall_speed_recovery = bool(
                wall_contact
                and self.wall_event_cooldown.get(agent, 0) == 0
                and speed > 600.0
                and speed - self.previous_speed.get(agent, speed) > 150.0
            )
            if wall_speed_recovery:
                metrics["wall_speed_recovery_events"] = 1.0
                self.wall_event_cooldown[agent] = 15

            possession_lost = bool(
                previous_controller == agent
                and new_touch_agent is not None
                and state.cars[new_touch_agent].team_num != car.team_num
            )
            opponent_controls_airborne = bool(
                not car.on_ground and opponent_distance + 100.0 < distances[agent]
            )
            if (
                self.recovery_started_step.get(agent) is None
                and (possession_lost or opponent_controls_airborne)
            ):
                self.recovery_started_step[agent] = self.step_count
                metrics["recovery_window_starts"] = 1.0

            recovery = 0.0
            recovery_start = self.recovery_started_step.get(agent)
            if recovery_start is not None:
                if state.goal_scored and state.scoring_team != car.team_num:
                    metrics["recovery_concessions"] = 1.0
                    self.recovery_started_step[agent] = None
                else:
                    recovered = bool(
                        car.on_ground
                        and speed >= 300.0
                        and (
                            self._goal_side(car, state.ball.position)
                            or distances[agent] <= 1800.0
                        )
                    )
                    if recovered:
                        elapsed = (self.step_count - recovery_start) * self.seconds_per_step
                        metrics["recovery_completions"] = 1.0
                        metrics["recovery_time_seconds"] = elapsed
                        speed_quality = float(np.clip(speed / 2300.0, 0.0, 1.0))
                        boost_quality = float(
                            np.clip(float(car.boost_amount) / 100.0, 0.0, 1.0)
                        )
                        recovery = 0.02 + 0.005 * speed_quality + 0.005 * boost_quality
                        self.recovery_started_step[agent] = None

            mechanics_resource = 0.0
            mechanics_resource += 0.005 * metrics["dodge_resource_acquisitions"]
            mechanics_resource += 0.01 * metrics["productive_resource_followups"]
            mechanics_resource += 0.003 * metrics["wavedash_like_events"]
            mechanics_resource += 0.003 * metrics["aerial_useful_touches"]
            mechanics_resource = float(np.clip(mechanics_resource, -0.02, 0.02))

            component_values = {
                "outcome": outcome,
                "possession": possession,
                "progress": progress,
                "boost_efficiency": boost_efficiency,
                "recovery": recovery,
                "mechanics_resource": mechanics_resource,
            }
            if not all(math.isfinite(value) for value in component_values.values()):
                raise FloatingPointError(f"Non-finite Reward V2 components: {component_values}")
            if not all(math.isfinite(value) for value in metrics.values()):
                raise FloatingPointError(f"Non-finite mechanics metrics: {metrics}")
            total = float(sum(component_values.values()))
            if not math.isfinite(total):
                raise FloatingPointError(f"Non-finite Reward V2 total for {agent}: {total}")
            components_by_agent[agent] = component_values
            metrics_by_agent[agent] = metrics
            rewards[agent] = total
            self._record_components(component_values)

            self.previous_boost[agent] = float(car.boost_amount)
            self.previous_on_ground[agent] = bool(car.on_ground)
            self.previous_has_dodge[agent] = has_dodge
            self.previous_has_flipped[agent] = bool(car.has_flipped)
            self.previous_has_double_jumped[agent] = bool(car.has_double_jumped)
            self.previous_position[agent] = np.asarray(car.physics.position).copy()
            self.previous_ball_distance[agent] = distances[agent]
            self.previous_speed[agent] = speed

        if new_touch_agent is not None:
            self.last_touch_agent = new_touch_agent
        self.previous_ball_y = float(state.ball.position[1])
        shared_info["reward_components"] = components_by_agent
        shared_info["mechanics_metrics"] = metrics_by_agent
        shared_info["reward_component_audit"] = {
            name: {
                "count": self.component_counts[name],
                "cumulative_signed": self.component_totals[name],
                "cumulative_absolute": self.component_absolute_totals[name],
                "minimum": self.component_minimums.get(name, 0.0),
                "maximum": self.component_maximums.get(name, 0.0),
            }
            for name in V2_COMPONENTS
        }
        return rewards


def reward_v2_metadata() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "reward_version": REWARD_V2_VERSION,
        "components": list(V2_COMPONENTS),
        "mechanics_metrics": list(MECHANICS_METRICS),
        "outcome_goal": 10.0,
        "outcome_concede": -10.0,
        "mechanics_absolute_step_cap": 0.02,
        "named_mechanic_rewards_enabled": False,
        "wavedash_event_interpretation": (
            "bounded landing-plus-jump-and-speed proxy; diagnostic, not a named-trick objective"
        ),
    }
