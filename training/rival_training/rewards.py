"""Outcome-dominant, independently instrumented reward v1."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

import numpy as np
from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y


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
