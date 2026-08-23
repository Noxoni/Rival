"""Compact dual-rate rollout metrics for Milestone 08."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .actions import action_family
from .metrics import (
    CAMPAIGN_METRIC_VECTOR_SIZE,
    aggregate_campaign_metrics,
    build_campaign_metric_vector,
)


M08_CONTEXTS = (
    "ground_near_ball",
    "airborne",
    "wall_or_ceiling",
    "awkward_recovery",
    "other",
)
_AGENT_SLOTS = 2
_PER_AGENT = 9 + len(M08_CONTEXTS)
M08_METRIC_VECTOR_SIZE = CAMPAIGN_METRIC_VECTOR_SIZE + _AGENT_SLOTS * _PER_AGENT


def mechanics_context(car, state) -> str:
    position = np.asarray(car.physics.position, dtype=np.float32)
    up = np.asarray(car.physics.up, dtype=np.float32)
    distance = float(np.linalg.norm(position - np.asarray(state.ball.position)))
    wall = abs(float(position[0])) > 3900.0 or abs(float(position[1])) > 4900.0
    ceiling = float(position[2]) > 1800.0
    if wall or ceiling:
        return "wall_or_ceiling"
    if not car.on_ground and float(position[2]) > 100.0:
        return "airborne"
    if float(up[2]) < 0.45 or (not car.on_ground and float(position[2]) < 180.0):
        return "awkward_recovery"
    if car.on_ground and distance <= 900.0:
        return "ground_near_ball"
    return "other"


def build_m08_metric_vector(
    state,
    shared_info: dict[str, Any],
    short_windows: dict[Any, dict[str, float]],
) -> np.ndarray:
    base = build_campaign_metric_vector(state, shared_info)
    vector = np.zeros(M08_METRIC_VECTOR_SIZE, dtype=np.float32)
    vector[:CAMPAIGN_METRIC_VECTOR_SIZE] = base
    agents = sorted(state.cars, key=lambda agent: state.cars[agent].team_num)
    decisions = shared_info.get("dual_rate_last_decisions", {})
    for slot, agent in enumerate(agents[:_AGENT_SLOTS]):
        offset = CAMPAIGN_METRIC_VECTOR_SIZE + slot * _PER_AGENT
        decision = decisions.get(agent, {})
        requested = int(decision.get("requested_mechanics_choice", 0))
        applied = int(decision.get("applied_mechanics_choice", 0))
        window = short_windows.get(agent, {})
        vector[offset + 0] = 1.0
        vector[offset + 1] = float(requested == 0)
        vector[offset + 2] = float(requested != 0)
        vector[offset + 3] = float(applied != 0)
        vector[offset + 4] = float(decision.get("strategic_decision", False))
        vector[offset + 5] = float(window.get("active", 0.0))
        vector[offset + 6] = float(window.get("useful_touch", 0.0))
        vector[offset + 7] = float(window.get("goal_for", 0.0))
        vector[offset + 8] = float(window.get("goal_against", 0.0))
        context = mechanics_context(state.cars[agent], state)
        vector[offset + 9 + M08_CONTEXTS.index(context)] = float(applied != 0)
    if not np.isfinite(vector).all():
        raise FloatingPointError("Non-finite Milestone 08 metric vector")
    return vector


class M08MetricsCollector:
    def collect_metrics(self, value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (M08_METRIC_VECTOR_SIZE,):
            raise ValueError(f"Unexpected M08 metric vector shape {result.shape}")
        return result

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps) -> None:
        del collected_metrics, wandb_run, cumulative_timesteps


def aggregate_m08_metrics(vectors: Iterable[np.ndarray]) -> dict[str, Any]:
    rows = np.asarray(list(vectors), dtype=np.float64)
    if rows.size == 0:
        rows = np.zeros((0, M08_METRIC_VECTOR_SIZE), dtype=np.float64)
    rows = rows.reshape(-1, M08_METRIC_VECTOR_SIZE)
    base = aggregate_campaign_metrics(rows[:, :CAMPAIGN_METRIC_VECTOR_SIZE])
    totals = {
        "decision_agent_records": 0,
        "requested_pass": 0,
        "requested_override": 0,
        "applied_override": 0,
        "strategic_boundaries": 0,
        "override_window_agent_records": 0,
        "useful_touches_after_override": 0,
        "goals_for_after_override": 0,
        "goals_against_after_override": 0,
    }
    contexts = {name: 0 for name in M08_CONTEXTS}
    keys = tuple(totals)
    for slot in range(_AGENT_SLOTS):
        offset = CAMPAIGN_METRIC_VECTOR_SIZE + slot * _PER_AGENT
        for index, key in enumerate(keys):
            totals[key] += int(round(rows[:, offset + index].sum()))
        for index, name in enumerate(M08_CONTEXTS):
            contexts[name] += int(round(rows[:, offset + 9 + index].sum()))
    decisions = max(totals["decision_agent_records"], 1)
    applied = max(totals["applied_override"], 1)
    base["dual_rate"] = {
        **totals,
        "requested_pass_share": totals["requested_pass"] / decisions,
        "requested_override_share": totals["requested_override"] / decisions,
        "applied_override_share": totals["applied_override"] / decisions,
        "override_context_counts": contexts,
        "override_context_shares": {
            name: count / applied for name, count in contexts.items()
        },
        "short_window_definition": (
            "most-recent applied override, through 30 subsequent mechanics decisions "
            "or until replaced/reset"
        ),
    }
    return base


def mechanics_action_report(
    actions: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    indices = np.asarray(actions, dtype=np.int64).reshape(-1)
    counts = np.bincount(indices, minlength=69)[:69]
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1, 69)
    deterministic = probs.argmax(axis=-1)
    appended_counts = [
        {
            "mechanics_choice": int(index),
            "global_action_index": int(89 + index),
            "family": action_family(89 + index),
            "count": int(counts[index]),
        }
        for index in range(1, 69)
        if counts[index]
    ]
    return {
        "sampled_action_count": int(counts.sum()),
        "pass_count": int(counts[0]),
        "override_count": int(counts[1:].sum()),
        "sampled_pass_rate": float(counts[0] / max(counts.sum(), 1)),
        "sampled_override_rate": float(counts[1:].sum() / max(counts.sum(), 1)),
        "mean_pass_probability": float(probs[:, 0].mean()),
        "mean_override_probability": float((1.0 - probs[:, 0]).mean()),
        "deterministic_pass_rate": float(np.mean(deterministic == 0)),
        "deterministic_override_rate": float(np.mean(deterministic != 0)),
        "full_mechanics_action_counts": counts.tolist(),
        "appended_action_distribution": appended_counts,
    }
