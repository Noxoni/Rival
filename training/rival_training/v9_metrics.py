"""Fixed Gate 13 behavior, reward, recovery, and mechanic-like metrics.

The event signatures in this module are diagnostics only.  They never affect
rewards, reset selection, policy actions, or PPO updates, and deliberately use
``*_like`` names where a short state/controller signature cannot prove a named
Rocket League mechanic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np
from rlgym.rocket_league.common_values import BACK_WALL_Y, CEILING_Z, SIDE_WALL_X

from .v9_actions import ACTION_DIM, ANALOG_FIELDS, BUTTON_FIELDS
from .v9_rewards import COMPONENTS
from .v9_curriculum import V9_PILOT_CURRICULUM_FAMILIES


V9_PILOT_METRICS_VERSION = "RivalScratchPilotMetricsV1"
AGENT_SLOTS = ("blue", "orange")
CONTINUOUS_FIELDS = (
    "movement_speed",
    "movement_planar_speed",
    "movement_distance_to_ball",
    "movement_boost",
    "movement_airborne_distance_step",
    "movement_boost_spent",
    "recovery_landing_speed_weighted",
    "recovery_landing_uprightness_weighted",
)
EVENT_FIELDS = (
    "touch_event",
    "aerial_touch_event",
    "airborne_tick",
    "recovery_landing_like_event",
    "first_jump_event",
    "dodge_or_double_jump_event",
    "directional_dodge_event",
    "dodge_resource_acquired_like_event",
    "touch_after_resource_like_event",
    "air_roll_active_tick",
    "aerial_possession_like_tick",
    "flip_cancel_like_event",
    "stall_like_event",
    "wavedash_like_event",
    "wall_dash_like_event",
    "zap_dash_like_event",
    "wall_or_ceiling_recovery_like_event",
    "ceiling_contact_retained_dodge_like_event",
)
ACTION_FIELDS = tuple(f"action_{name}" for name in ANALOG_FIELDS + BUTTON_FIELDS)


def _metric_fields() -> tuple[str, ...]:
    fields = [f"reset.{name}" for name in V9_PILOT_CURRICULUM_FAMILIES]
    fields.extend(("score.blue_goal", "score.orange_goal"))
    for slot in AGENT_SLOTS:
        fields.extend(f"{slot}.reward.{name}" for name in COMPONENTS)
        fields.extend(f"{slot}.{name}" for name in CONTINUOUS_FIELDS)
        fields.extend(f"{slot}.{name}" for name in EVENT_FIELDS)
        fields.extend(f"{slot}.{name}" for name in ACTION_FIELDS)
    return tuple(fields)


V9_PILOT_METRIC_FIELDS = _metric_fields()
V9_PILOT_METRIC_INDEX = {name: index for index, name in enumerate(V9_PILOT_METRIC_FIELDS)}
V9_PILOT_METRIC_VECTOR_SIZE = len(V9_PILOT_METRIC_FIELDS)


@dataclass
class _PreviousCar:
    speed: float
    boost: float
    on_ground: bool
    has_jumped: bool
    has_dodge_or_double_jump: bool
    has_flip: bool
    air_ticks: int = 0
    dodge_recent_ticks: int = 10_000
    resource_window_ticks: int = 0


def _speed(car) -> float:
    return float(np.linalg.norm(car.physics.linear_velocity))


def _on_side_or_back_wall(car) -> bool:
    position = np.asarray(car.physics.position, dtype=np.float64)
    wheels = tuple(bool(value) for value in car.wheels_with_contact)
    near_wall = (
        abs(float(position[0])) >= SIDE_WALL_X - 190.0
        or abs(float(position[1])) >= BACK_WALL_Y - 190.0
    )
    return bool(any(wheels) and near_wall and float(position[2]) > 80.0)


def _on_ceiling(car) -> bool:
    return bool(
        any(bool(value) for value in car.wheels_with_contact)
        and float(car.physics.position[2]) >= CEILING_Z - 170.0
    )


def _uprightness(car) -> float:
    rotation = np.asarray(car.physics.rotation_mtx, dtype=np.float64)
    return float(np.clip(rotation[2, 2], -1.0, 1.0))


class RivalV9PilotMetricTracker:
    """Build one fixed-size transport vector after every environment step."""

    def __init__(self) -> None:
        self.previous: dict[Any, _PreviousCar] = {}

    def reset(self, state) -> None:
        self.previous = {
            agent: _PreviousCar(
                speed=_speed(car),
                boost=float(car.boost_amount),
                on_ground=bool(car.on_ground),
                has_jumped=bool(car.has_jumped),
                has_dodge_or_double_jump=bool(car.has_flipped or car.has_double_jumped),
                has_flip=bool(car.has_flip),
            )
            for agent, car in state.cars.items()
        }

    def build(self, state, shared_info: dict[str, Any]) -> np.ndarray:
        if set(self.previous) != set(state.cars):
            self.reset(state)
        vector = np.zeros(V9_PILOT_METRIC_VECTOR_SIZE, dtype=np.float32)
        if shared_info.pop("rival_v9_reset_family_pending_metric", False):
            family = str(shared_info.get("rival_v9_reset_family", "natural"))
            vector[V9_PILOT_METRIC_INDEX[f"reset.{family}"]] = 1.0
        if bool(state.goal_scored) and int(state.scoring_team) in (0, 1):
            side = "blue" if int(state.scoring_team) == 0 else "orange"
            vector[V9_PILOT_METRIC_INDEX[f"score.{side}_goal"]] = 1.0

        agents = sorted(state.cars, key=lambda agent: int(state.cars[agent].team_num))
        reward_components = shared_info.get("reward_components", {})
        applied_actions = shared_info.get("rival_v9_actor_applied_actions", {})
        for slot, agent in zip(AGENT_SLOTS, agents, strict=True):
            car = state.cars[agent]
            previous = self.previous[agent]
            speed = _speed(car)
            planar_speed = float(np.linalg.norm(car.physics.linear_velocity[:2]))
            distance_to_ball = float(np.linalg.norm(car.physics.position - state.ball.position))
            airborne = not bool(car.on_ground)
            touch_count = int(car.ball_touches)
            action = np.asarray(
                applied_actions.get(agent, np.zeros(ACTION_DIM)), dtype=np.float32
            ).reshape(-1)
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise FloatingPointError("Pilot metrics received an invalid controller")
            dodge_or_double_jump = bool(car.has_flipped or car.has_double_jumped)
            first_jump = bool(car.has_jumped) and not previous.has_jumped
            dodge_event = dodge_or_double_jump and not previous.has_dodge_or_double_jump
            ground_landing = (
                bool(car.on_ground)
                and previous.air_ticks >= 6
                and float(car.physics.position[2]) < 110.0
            )
            wall_contact = _on_side_or_back_wall(car)
            ceiling_contact = _on_ceiling(car)
            surface_recovery = (
                bool(car.on_ground) or wall_contact or ceiling_contact
            ) and previous.air_ticks >= 6
            landing_like = ground_landing or surface_recovery
            speed_delta = speed - previous.speed
            flip_torque = np.asarray(car.flip_torque, dtype=np.float64)
            directional_dodge = bool(dodge_event and np.linalg.norm(flip_torque[:2]) > 0.1)
            resource_acquired = bool(
                airborne and not previous.has_flip and bool(car.has_flip) and touch_count > 0
            )
            touch_after_resource = bool(
                touch_count > 0 and previous.resource_window_ticks > 0 and not resource_acquired
            )
            flip_cancel_like = bool(
                bool(car.is_flipping)
                and previous.dodge_recent_ticks <= 24
                and abs(float(action[2])) >= 0.6
                and abs(float(flip_torque[1])) > 0.1
                and float(action[2]) * float(flip_torque[1]) < 0.0
            )
            stall_like = bool(
                airborne
                and action[5] > 0.5
                and action[7] > 0.5
                and abs(float(action[4])) > 0.7
                and abs(float(action[3])) > 0.7
                and float(action[4]) * float(action[3]) < 0.0
            )
            wavedash_like = bool(
                ground_landing and previous.dodge_recent_ticks <= 12 and speed >= 500.0
            )
            wall_dash_like = bool(
                wall_contact and action[5] > 0.5 and speed >= 600.0 and speed_delta > 15.0
            )
            zap_dash_like = bool(
                ground_landing
                and action[5] > 0.5
                and action[6] > 0.5
                and speed >= 1000.0
                and speed_delta > 30.0
            )
            ceiling_retained_dodge = bool(ceiling_contact and bool(car.has_flip))
            values = {
                "movement_speed": speed,
                "movement_planar_speed": planar_speed,
                "movement_distance_to_ball": distance_to_ball,
                "movement_boost": float(car.boost_amount),
                "movement_airborne_distance_step": speed / 120.0 if airborne else 0.0,
                "movement_boost_spent": max(0.0, previous.boost - float(car.boost_amount)),
                "recovery_landing_speed_weighted": speed if landing_like else 0.0,
                "recovery_landing_uprightness_weighted": (
                    _uprightness(car) if landing_like else 0.0
                ),
                "touch_event": float(touch_count),
                "aerial_touch_event": float(touch_count if airborne else 0),
                "airborne_tick": float(airborne),
                "recovery_landing_like_event": float(landing_like),
                "first_jump_event": float(first_jump),
                "dodge_or_double_jump_event": float(dodge_event),
                "directional_dodge_event": float(directional_dodge),
                "dodge_resource_acquired_like_event": float(resource_acquired),
                "touch_after_resource_like_event": float(touch_after_resource),
                "air_roll_active_tick": float(airborne and abs(float(action[4])) > 0.2),
                "aerial_possession_like_tick": float(airborne and distance_to_ball < 600.0),
                "flip_cancel_like_event": float(flip_cancel_like),
                "stall_like_event": float(stall_like),
                "wavedash_like_event": float(wavedash_like),
                "wall_dash_like_event": float(wall_dash_like),
                "zap_dash_like_event": float(zap_dash_like),
                "wall_or_ceiling_recovery_like_event": float(
                    surface_recovery and (wall_contact or ceiling_contact)
                ),
                "ceiling_contact_retained_dodge_like_event": float(ceiling_retained_dodge),
            }
            for name, value in reward_components.get(agent, {}).items():
                if name in COMPONENTS:
                    vector[V9_PILOT_METRIC_INDEX[f"{slot}.reward.{name}"]] = float(value)
            for name, value in values.items():
                vector[V9_PILOT_METRIC_INDEX[f"{slot}.{name}"]] = float(value)
            for index, name in enumerate(ACTION_FIELDS):
                vector[V9_PILOT_METRIC_INDEX[f"{slot}.{name}"]] = action[index]

            air_ticks = previous.air_ticks + 1 if airborne else 0
            dodge_recent = 0 if dodge_event else min(previous.dodge_recent_ticks + 1, 10_000)
            resource_window = (
                120 if resource_acquired else max(0, previous.resource_window_ticks - 1)
            )
            self.previous[agent] = _PreviousCar(
                speed=speed,
                boost=float(car.boost_amount),
                on_ground=bool(car.on_ground),
                has_jumped=bool(car.has_jumped),
                has_dodge_or_double_jump=dodge_or_double_jump,
                has_flip=bool(car.has_flip),
                air_ticks=air_ticks,
                dodge_recent_ticks=dodge_recent,
                resource_window_ticks=resource_window,
            )

        if not np.isfinite(vector).all():
            raise FloatingPointError("Pilot metric vector contains non-finite values")
        return vector


def collect_v9_pilot_metric_vector(value: np.ndarray) -> np.ndarray:
    """Pickle-safe rlgym-ppo child-process metric encoder."""

    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (V9_PILOT_METRIC_VECTOR_SIZE,):
        raise ValueError(f"Unexpected Rival v9 pilot metric shape {vector.shape}")
    if not np.isfinite(vector).all():
        raise FloatingPointError("Non-finite Rival v9 pilot metric vector")
    return vector


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return {
            "samples": 0,
            "mean": 0.0,
            "standard_deviation": 0.0,
            "minimum": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "maximum": 0.0,
            "cumulative": 0.0,
            "cumulative_absolute": 0.0,
        }
    if not np.isfinite(array).all():
        raise FloatingPointError("Cannot summarize non-finite pilot metrics")
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
        "cumulative": float(array.sum()),
        "cumulative_absolute": float(np.abs(array).sum()),
    }


def aggregate_v9_pilot_metrics(vectors: Iterable[np.ndarray]) -> dict[str, Any]:
    rows = np.asarray(list(vectors), dtype=np.float64)
    if rows.size == 0:
        rows = np.zeros((0, V9_PILOT_METRIC_VECTOR_SIZE), dtype=np.float64)
    rows = rows.reshape(-1, V9_PILOT_METRIC_VECTOR_SIZE)
    if not np.isfinite(rows).all():
        raise FloatingPointError("Collected pilot metrics contain non-finite values")

    reset_counts = {
        name: int(round(rows[:, V9_PILOT_METRIC_INDEX[f"reset.{name}"]].sum()))
        for name in V9_PILOT_CURRICULUM_FAMILIES
    }
    reset_total = sum(reset_counts.values())
    goals = {
        side: int(round(rows[:, V9_PILOT_METRIC_INDEX[f"score.{side}_goal"]].sum()))
        for side in AGENT_SLOTS
    }
    rewards: dict[str, Any] = {}
    continuous: dict[str, Any] = {}
    events: dict[str, int] = {}
    action_rows: list[np.ndarray] = []
    for slot in AGENT_SLOTS:
        for name in COMPONENTS:
            rewards[f"{slot}.{name}"] = _summary(
                rows[:, V9_PILOT_METRIC_INDEX[f"{slot}.reward.{name}"]]
            )
        for name in CONTINUOUS_FIELDS:
            continuous[f"{slot}.{name}"] = _summary(
                rows[:, V9_PILOT_METRIC_INDEX[f"{slot}.{name}"]]
            )
        for name in EVENT_FIELDS:
            events[f"{slot}.{name}"] = int(
                round(rows[:, V9_PILOT_METRIC_INDEX[f"{slot}.{name}"]].sum())
            )
        action_rows.append(
            np.stack(
                [rows[:, V9_PILOT_METRIC_INDEX[f"{slot}.{name}"]] for name in ACTION_FIELDS],
                axis=1,
            )
        )
    actions = np.concatenate(action_rows, axis=0) if action_rows else np.zeros((0, 8))
    analog = actions[:, :5]
    buttons = np.rint(actions[:, 5:]).astype(np.int64)
    combo = buttons[:, 0] + 2 * buttons[:, 1] + 4 * buttons[:, 2]
    combo_counts = np.bincount(combo, minlength=8) if len(combo) else np.zeros(8, int)
    action_exploration = {
        "samples": int(len(actions)),
        "analog": {
            name: {
                **_summary(analog[:, index]),
                "nontrivial_range": bool(len(analog) and float(np.ptp(analog[:, index])) > 0.1),
                "absolute_over_0_95_share": float(np.mean(np.abs(analog[:, index]) > 0.95))
                if len(analog)
                else 0.0,
            }
            for index, name in enumerate(ANALOG_FIELDS)
        },
        "button_combo_counts": combo_counts.tolist(),
        "button_combo_shares": (combo_counts / max(int(combo_counts.sum()), 1)).tolist(),
        "marginal_button_shares": {
            name: float(buttons[:, index].mean()) if len(buttons) else 0.0
            for index, name in enumerate(BUTTON_FIELDS)
        },
    }
    report = {
        "schema_version": 1,
        "metrics_version": V9_PILOT_METRICS_VERSION,
        "metric_vector_size": V9_PILOT_METRIC_VECTOR_SIZE,
        "metric_vector_count": int(len(rows)),
        "agent_metric_samples": int(len(actions)),
        "reset_counts": reset_counts,
        "reset_shares": {
            name: float(count / reset_total) if reset_total else 0.0
            for name, count in reset_counts.items()
        },
        "scores_recorded_for_diagnostics_only": goals,
        "reward_components": rewards,
        "movement_and_recovery": continuous,
        "event_counts": events,
        "action_exploration": action_exploration,
        "named_mechanic_claims": False,
        "mechanic_like_detectors_are_diagnostics_only": True,
        "finite": True,
    }
    numeric_values = [
        value
        for section in (rewards, continuous)
        for summary in section.values()
        for value in summary.values()
        if isinstance(value, (int, float))
    ]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise FloatingPointError("Pilot aggregate contains a non-finite summary")
    return report


def metric_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metrics_version": V9_PILOT_METRICS_VERSION,
        "vector_size": V9_PILOT_METRIC_VECTOR_SIZE,
        "fields": list(V9_PILOT_METRIC_FIELDS),
        "reset_families": list(V9_PILOT_CURRICULUM_FAMILIES),
        "reward_components": list(COMPONENTS),
        "continuous_fields": list(CONTINUOUS_FIELDS),
        "event_fields": list(EVENT_FIELDS),
        "action_fields": list(ACTION_FIELDS),
        "cadence_hz": 120,
        "reward_influence": False,
    }
