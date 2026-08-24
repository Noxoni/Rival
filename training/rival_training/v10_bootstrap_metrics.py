"""Fixed transport metrics for the Rival v10.1 agency bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import numpy as np

from .v10_bootstrap_curriculum import FAMILIES
from .v10_bootstrap_reward import (
    COMPONENTS,
    SHAPING_ABSOLUTE_EPISODE_BUDGETS,
    SHAPING_COMPONENTS,
)
from .v9_actions import ACTION_DIM, ANALOG_FIELDS, BUTTON_FIELDS


METRICS_VERSION = "RivalAgencyBootstrapMetricsV1"
AGENT_SLOTS = ("blue", "orange")
TERMINATION_REASONS = ("goal", "no_touch_timeout", "episode_timeout")
MOTION_FIELDS = (
    "speed",
    "planar_speed",
    "distance_to_ball",
    "speed_over_500",
    "speed_over_1000",
    "speed_over_1500",
    "speed_over_2000",
)
INTERACTION_FIELDS = (
    "raw_touch_record",
    "logical_touch",
    "aerial_logical_touch",
    "touch_chain_start",
    "chain_length_1",
    "chain_length_2",
    "chain_length_3",
    "chain_length_4",
    "chain_length_5plus",
    "first_touch_event",
    "first_touch_time_seconds",
    "episode_self_touches_ge1",
    "episode_self_touches_ge2",
    "episode_self_touches_ge3",
    "first_jump_event",
    "dodge_event",
)
ACTION_FIELDS = tuple(f"action_{name}" for name in ANALOG_FIELDS + BUTTON_FIELDS)


def _fields() -> tuple[str, ...]:
    fields: list[str] = [f"reset.{name}" for name in FAMILIES]
    fields.extend(f"termination.{name}" for name in TERMINATION_REASONS)
    fields.extend(("score.blue_goal", "score.orange_goal"))
    fields.extend(f"goal_family.{name}" for name in FAMILIES)
    for slot in AGENT_SLOTS:
        fields.extend(f"{slot}.reward.{name}" for name in COMPONENTS)
        fields.extend(f"{slot}.budget_clip.{name}" for name in SHAPING_COMPONENTS)
        fields.extend(f"{slot}.{name}" for name in MOTION_FIELDS)
        fields.extend(f"{slot}.{name}" for name in INTERACTION_FIELDS)
        fields.extend(f"{slot}.{name}" for name in ACTION_FIELDS)
        fields.extend(f"{slot}.logical_touch_family.{name}" for name in FAMILIES)
    return tuple(fields)


METRIC_FIELDS = _fields()
METRIC_INDEX = {name: index for index, name in enumerate(METRIC_FIELDS)}
METRIC_VECTOR_SIZE = len(METRIC_FIELDS)


@dataclass
class _Previous:
    has_jumped: bool
    has_dodge_or_double_jump: bool


class RivalAgencyBootstrapMetricTrackerV1:
    """Build one fixed diagnostic vector after every environment transition."""

    def __init__(self) -> None:
        self.previous: dict[Any, _Previous] = {}
        self.episode_touch_counts: dict[Any, int] = {}
        self.episode_start_tick = 0
        self.current_family = "natural"

    def reset(self, state, shared_info: dict[str, Any] | None = None) -> None:
        self.previous = {
            agent: _Previous(
                has_jumped=bool(car.has_jumped),
                has_dodge_or_double_jump=bool(car.has_flipped or car.has_double_jumped),
            )
            for agent, car in state.cars.items()
        }
        self.episode_touch_counts = {agent: 0 for agent in state.cars}
        self.episode_start_tick = int(state.tick_count)
        if shared_info is not None:
            self.current_family = str(
                shared_info.get("rival_v10_reset_family", "natural")
            )

    def build(
        self,
        state,
        shared_info: dict[str, Any],
        *,
        termination_reason: str | None = None,
    ) -> np.ndarray:
        if set(self.previous) != set(state.cars):
            self.reset(state, shared_info)
        vector = np.zeros(METRIC_VECTOR_SIZE, dtype=np.float32)
        if shared_info.pop("rival_v10_reset_family_pending_metric", False):
            self.current_family = str(
                shared_info.get("rival_v10_reset_family", "natural")
            )
            vector[METRIC_INDEX[f"reset.{self.current_family}"]] = 1.0
        if termination_reason is not None:
            if termination_reason not in TERMINATION_REASONS:
                raise ValueError(f"Unknown bootstrap termination reason: {termination_reason}")
            vector[METRIC_INDEX[f"termination.{termination_reason}"]] = 1.0
        if bool(state.goal_scored) and int(state.scoring_team) in (0, 1):
            side = "blue" if int(state.scoring_team) == 0 else "orange"
            vector[METRIC_INDEX[f"score.{side}_goal"]] = 1.0
            vector[METRIC_INDEX[f"goal_family.{self.current_family}"]] = 1.0

        agents = sorted(state.cars, key=lambda agent: int(state.cars[agent].team_num))
        reward_components = shared_info.get("reward_components", {})
        touch_metrics = shared_info.get("rival_v10_touch_metrics", {})
        budget_clipped = shared_info.get("reward_budget_clipped", {})
        applied_actions = shared_info.get("rival_v9_actor_applied_actions", {})
        for slot, agent in zip(AGENT_SLOTS, agents, strict=True):
            car = state.cars[agent]
            previous = self.previous[agent]
            speed = float(np.linalg.norm(car.physics.linear_velocity))
            planar_speed = float(np.linalg.norm(car.physics.linear_velocity[:2]))
            distance = float(np.linalg.norm(car.physics.position - state.ball.position))
            details = touch_metrics.get(agent, {})
            raw = int(details.get("raw_touch_records", int(car.ball_touches)))
            logical = bool(details.get("logical_touch", False))
            aerial = bool(details.get("aerial_touch", False))
            chain_length = int(details.get("touch_chain_length", 0))
            previous_episode_touches = self.episode_touch_counts[agent]
            first_touch = logical and previous_episode_touches == 0
            if logical:
                self.episode_touch_counts[agent] += 1
            action = np.asarray(
                applied_actions.get(agent, np.zeros(ACTION_DIM)), dtype=np.float32
            ).reshape(-1)
            if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                raise FloatingPointError("Bootstrap metrics received an invalid controller")
            first_jump = bool(car.has_jumped) and not previous.has_jumped
            dodge_state = bool(car.has_flipped or car.has_double_jumped)
            dodge = dodge_state and not previous.has_dodge_or_double_jump
            values = {
                "speed": speed,
                "planar_speed": planar_speed,
                "distance_to_ball": distance,
                "speed_over_500": float(speed > 500.0),
                "speed_over_1000": float(speed > 1000.0),
                "speed_over_1500": float(speed > 1500.0),
                "speed_over_2000": float(speed > 2000.0),
                "raw_touch_record": float(raw),
                "logical_touch": float(logical),
                "aerial_logical_touch": float(aerial),
                "touch_chain_start": float(logical and chain_length == 1),
                "chain_length_1": float(logical and chain_length == 1),
                "chain_length_2": float(logical and chain_length == 2),
                "chain_length_3": float(logical and chain_length == 3),
                "chain_length_4": float(logical and chain_length == 4),
                "chain_length_5plus": float(logical and chain_length >= 5),
                "first_touch_event": float(first_touch),
                "first_touch_time_seconds": (
                    (int(state.tick_count) - self.episode_start_tick) / 120.0
                    if first_touch
                    else 0.0
                ),
                "episode_self_touches_ge1": float(
                    termination_reason is not None
                    and self.episode_touch_counts[agent] >= 1
                ),
                "episode_self_touches_ge2": float(
                    termination_reason is not None
                    and self.episode_touch_counts[agent] >= 2
                ),
                "episode_self_touches_ge3": float(
                    termination_reason is not None
                    and self.episode_touch_counts[agent] >= 3
                ),
                "first_jump_event": float(first_jump),
                "dodge_event": float(dodge),
            }
            for name, value in reward_components.get(agent, {}).items():
                if name in COMPONENTS:
                    vector[METRIC_INDEX[f"{slot}.reward.{name}"]] = float(value)
            for name in budget_clipped.get(agent, ()):
                if name in SHAPING_COMPONENTS:
                    vector[METRIC_INDEX[f"{slot}.budget_clip.{name}"]] = 1.0
            for name, value in values.items():
                vector[METRIC_INDEX[f"{slot}.{name}"]] = float(value)
            for index, name in enumerate(ACTION_FIELDS):
                vector[METRIC_INDEX[f"{slot}.{name}"]] = action[index]
            if logical:
                vector[
                    METRIC_INDEX[f"{slot}.logical_touch_family.{self.current_family}"]
                ] = 1.0
            self.previous[agent] = _Previous(
                has_jumped=bool(car.has_jumped),
                has_dodge_or_double_jump=dodge_state,
            )
        if not np.isfinite(vector).all():
            raise FloatingPointError("Bootstrap metric vector contains non-finite values")
        return vector


def collect_v10_bootstrap_metric_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (METRIC_VECTOR_SIZE,):
        raise ValueError(f"Unexpected bootstrap metric shape: {vector.shape}")
    if not np.isfinite(vector).all():
        raise FloatingPointError("Non-finite bootstrap metric vector")
    return vector


def _summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
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
        raise FloatingPointError("Cannot summarize non-finite bootstrap metrics")
    return {
        "samples": int(len(array)),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
        "cumulative": float(array.sum()),
        "cumulative_absolute": float(np.abs(array).sum()),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) < 2 or float(a.std()) <= 1e-12 or float(b.std()) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(a, b)[0, 1])
    return value if math.isfinite(value) else 0.0


def aggregate_v10_bootstrap_metrics(vectors: Iterable[np.ndarray]) -> dict[str, Any]:
    rows = np.asarray(list(vectors), dtype=np.float64)
    if rows.size == 0:
        rows = np.zeros((0, METRIC_VECTOR_SIZE), dtype=np.float64)
    rows = rows.reshape(-1, METRIC_VECTOR_SIZE)
    if not np.isfinite(rows).all():
        raise FloatingPointError("Collected bootstrap metrics contain non-finite values")
    reset_counts = {
        name: int(round(rows[:, METRIC_INDEX[f"reset.{name}"]].sum()))
        for name in FAMILIES
    }
    reset_total = sum(reset_counts.values())
    terminations = {
        name: int(round(rows[:, METRIC_INDEX[f"termination.{name}"]].sum()))
        for name in TERMINATION_REASONS
    }
    termination_total = sum(terminations.values())
    goals = {
        side: int(round(rows[:, METRIC_INDEX[f"score.{side}_goal"]].sum()))
        for side in AGENT_SLOTS
    }
    goals_by_family = {
        name: int(round(rows[:, METRIC_INDEX[f"goal_family.{name}"]].sum()))
        for name in FAMILIES
    }
    reward_components: dict[str, Any] = {}
    budget_clips: dict[str, int] = {}
    motion: dict[str, Any] = {}
    interaction_counts: dict[str, int] = {}
    touch_by_family: dict[str, int] = {name: 0 for name in FAMILIES}
    action_rows: list[np.ndarray] = []
    total_reward_rows: list[np.ndarray] = []
    logical_rows: list[np.ndarray] = []
    speed_rows: list[np.ndarray] = []
    for slot in AGENT_SLOTS:
        slot_reward = np.zeros(len(rows), dtype=np.float64)
        for name in COMPONENTS:
            values = rows[:, METRIC_INDEX[f"{slot}.reward.{name}"]]
            reward_components[f"{slot}.{name}"] = _summary(values)
            slot_reward += values
        for name in SHAPING_COMPONENTS:
            budget_clips[f"{slot}.{name}"] = int(
                round(rows[:, METRIC_INDEX[f"{slot}.budget_clip.{name}"]].sum())
            )
        for name in MOTION_FIELDS:
            motion[f"{slot}.{name}"] = _summary(
                rows[:, METRIC_INDEX[f"{slot}.{name}"]]
            )
        for name in INTERACTION_FIELDS:
            if name == "first_touch_time_seconds":
                mask = rows[:, METRIC_INDEX[f"{slot}.first_touch_event"]] > 0.5
                motion[f"{slot}.time_to_first_touch_seconds"] = _summary(
                    rows[mask, METRIC_INDEX[f"{slot}.{name}"]]
                )
            else:
                interaction_counts[f"{slot}.{name}"] = int(
                    round(rows[:, METRIC_INDEX[f"{slot}.{name}"]].sum())
                )
        for family in FAMILIES:
            touch_by_family[family] += int(
                round(
                    rows[:, METRIC_INDEX[f"{slot}.logical_touch_family.{family}"]].sum()
                )
            )
        action_rows.append(
            np.stack(
                [
                    rows[:, METRIC_INDEX[f"{slot}.{name}"]]
                    for name in ACTION_FIELDS
                ],
                axis=1,
            )
        )
        total_reward_rows.append(slot_reward)
        logical_rows.append(rows[:, METRIC_INDEX[f"{slot}.logical_touch"]])
        speed_rows.append(rows[:, METRIC_INDEX[f"{slot}.planar_speed"]])
    actions = np.concatenate(action_rows, axis=0) if action_rows else np.zeros((0, 8))
    analog = actions[:, :5]
    buttons = np.rint(actions[:, 5:]).astype(np.int64)
    combos = buttons[:, 0] + 2 * buttons[:, 1] + 4 * buttons[:, 2]
    combo_counts = np.bincount(combos, minlength=8) if len(combos) else np.zeros(8, int)
    agent_steps = max(int(len(rows) * len(AGENT_SLOTS)), 1)

    def combined_count(name: str) -> int:
        return sum(interaction_counts[f"{slot}.{name}"] for slot in AGENT_SLOTS)

    chain_histogram = {
        "1": combined_count("chain_length_1"),
        "2": combined_count("chain_length_2"),
        "3": combined_count("chain_length_3"),
        "4": combined_count("chain_length_4"),
        "5+": combined_count("chain_length_5plus"),
    }
    maximum_chain = next(
        (
            value
            for key, value in (("5+", 5), ("4", 4), ("3", 3), ("2", 2), ("1", 1))
            if chain_histogram[key] > 0
        ),
        0,
    )
    episode_agent_denominator = max(termination_total * 2, 1)
    first_touch_times = np.concatenate(
        [
            rows[
                rows[:, METRIC_INDEX[f"{slot}.first_touch_event"]] > 0.5,
                METRIC_INDEX[f"{slot}.first_touch_time_seconds"],
            ]
            for slot in AGENT_SLOTS
        ]
    )
    combined_total_reward = np.concatenate(total_reward_rows)
    combined_logical = np.concatenate(logical_rows)
    combined_planar_speed = np.concatenate(speed_rows)
    total_logical_touches = combined_count("logical_touch")
    simulated_agent_seconds = agent_steps / 120.0
    reward_integrity = {
        "component_budget_fraction_across_completed_episodes": {
            name: float(
                sum(
                    reward_components[f"{slot}.{name}"]["cumulative_absolute"]
                    for slot in AGENT_SLOTS
                )
                / max(
                    termination_total
                    * 2
                    * float(SHAPING_ABSOLUTE_EPISODE_BUDGETS[name]),
                    1e-12,
                )
            )
            for name in SHAPING_COMPONENTS
        },
        "budget_clip_counts": budget_clips,
        "total_reward_per_logical_touch": float(
            combined_total_reward.sum() / max(total_logical_touches, 1)
        ),
        "total_reward_per_simulated_agent_second": float(
            combined_total_reward.sum() / max(simulated_agent_seconds, 1e-12)
        ),
        "reward_correlation_with_logical_touch": _correlation(
            combined_total_reward, combined_logical
        ),
        "reward_correlation_with_planar_speed": _correlation(
            combined_total_reward, combined_planar_speed
        ),
        "direct_action_press_rewards": False,
    }
    report = {
        "schema_version": 1,
        "metrics_version": METRICS_VERSION,
        "metric_vector_size": METRIC_VECTOR_SIZE,
        "metric_vector_count": int(len(rows)),
        "agent_metric_samples": int(agent_steps),
        "reset_counts": reset_counts,
        "reset_shares": {
            name: count / max(reset_total, 1) for name, count in reset_counts.items()
        },
        "termination_counts": terminations,
        "termination_shares": {
            name: count / max(termination_total, 1)
            for name, count in terminations.items()
        },
        "goals": goals,
        "goals_by_reset_family": goals_by_family,
        "reward_components": reward_components,
        "reward_integrity": reward_integrity,
        "motion": motion,
        "interaction_counts": interaction_counts,
        "interaction_rates_per_100k_agent_steps": {
            "raw_touch_records": combined_count("raw_touch_record")
            * 100_000.0
            / agent_steps,
            "logical_touches": total_logical_touches * 100_000.0 / agent_steps,
            "aerial_logical_touches": combined_count("aerial_logical_touch")
            * 100_000.0
            / agent_steps,
            "touch_chain_starts": combined_count("touch_chain_start")
            * 100_000.0
            / agent_steps,
            "first_jumps": combined_count("first_jump_event")
            * 100_000.0
            / agent_steps,
            "dodges": combined_count("dodge_event") * 100_000.0 / agent_steps,
        },
        "touches_by_reset_family": touch_by_family,
        "touch_chain": {
            "maximum_observed_chain_length_lower_bound": maximum_chain,
            "histogram": chain_histogram,
            "two_or_more_chain_touches_per_100k_agent_steps": (
                sum(chain_histogram[key] for key in ("2", "3", "4", "5+"))
                * 100_000.0
                / agent_steps
            ),
        },
        "episode_touch_shares": {
            f"at_least_{threshold}": float(
                sum(
                    interaction_counts[f"{slot}.episode_self_touches_ge{threshold}"]
                    for slot in AGENT_SLOTS
                )
                / episode_agent_denominator
            )
            for threshold in (1, 2, 3)
        },
        "time_to_first_touch_seconds": _summary(first_touch_times),
        "action_diagnostics": {
            "samples": int(len(actions)),
            "analog": {
                name: {
                    **_summary(analog[:, index]),
                    "absolute_over_0_95_share": float(
                        np.mean(np.abs(analog[:, index]) > 0.95)
                    )
                    if len(analog)
                    else 0.0,
                }
                for index, name in enumerate(ANALOG_FIELDS)
            },
            "button_combo_counts": combo_counts.tolist(),
            "button_combo_shares": (
                combo_counts / max(int(combo_counts.sum()), 1)
            ).tolist(),
            "marginal_button_shares": {
                name: float(buttons[:, index].mean()) if len(buttons) else 0.0
                for index, name in enumerate(BUTTON_FIELDS)
            },
        },
        "scores_are_capability_evidence_but_not_the_only_gate": True,
        "finite": True,
    }
    return report


def metric_schema() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "metrics_version": METRICS_VERSION,
        "vector_size": METRIC_VECTOR_SIZE,
        "fields": list(METRIC_FIELDS),
        "reset_families": list(FAMILIES),
        "reward_components": list(COMPONENTS),
        "cadence_hz": 120,
        "reward_influence": False,
    }
