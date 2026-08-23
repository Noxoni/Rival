"""Compact local metrics transport for multiprocess Milestone 06 rollouts."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .curriculum import CURRICULUM_FAMILIES
from .rewards import MECHANICS_METRICS, V2_COMPONENTS


METRICS_SCHEMA_VERSION = 1
_AGENT_SLOTS = 2
_COMPONENT_VALUES = _AGENT_SLOTS * len(V2_COMPONENTS)
_MECHANICS_START = _COMPONENT_VALUES
_MECHANICS_VALUES = _AGENT_SLOTS * len(MECHANICS_METRICS)
_RESET_START = _MECHANICS_START + _MECHANICS_VALUES
_GOAL_START = _RESET_START + len(CURRICULUM_FAMILIES)
CAMPAIGN_METRIC_VECTOR_SIZE = _GOAL_START + 2


def build_campaign_metric_vector(state, shared_info: dict[str, Any]) -> np.ndarray:
    """Encode two-agent rewards/events plus one-shot reset and goal markers."""
    vector = np.zeros(CAMPAIGN_METRIC_VECTOR_SIZE, dtype=np.float32)
    agents = sorted(state.cars, key=lambda agent: state.cars[agent].team_num)
    components = shared_info.get("reward_components", {})
    mechanics = shared_info.get("mechanics_metrics", {})
    for slot, agent in enumerate(agents[:_AGENT_SLOTS]):
        component_offset = slot * len(V2_COMPONENTS)
        for index, name in enumerate(V2_COMPONENTS):
            vector[component_offset + index] = float(
                components.get(agent, {}).get(name, 0.0)
            )
        mechanics_offset = _MECHANICS_START + slot * len(MECHANICS_METRICS)
        for index, name in enumerate(MECHANICS_METRICS):
            vector[mechanics_offset + index] = float(
                mechanics.get(agent, {}).get(name, 0.0)
            )
    if shared_info.pop("reset_family_pending_metric", False):
        family = str(shared_info.get("reset_family", "natural"))
        vector[_RESET_START + CURRICULUM_FAMILIES.index(family)] = 1.0
    if state.goal_scored and state.scoring_team in (0, 1):
        vector[_GOAL_START + int(state.scoring_team)] = 1.0
    if not np.isfinite(vector).all():
        raise FloatingPointError("Non-finite campaign metric vector")
    return vector


class CampaignMetricsCollector:
    """rlgym-ppo callback transporting a vector already built by the wrapper."""

    def collect_metrics(self, value: np.ndarray) -> np.ndarray:
        result = np.asarray(value, dtype=np.float32)
        if result.shape != (CAMPAIGN_METRIC_VECTOR_SIZE,):
            raise ValueError(f"Unexpected campaign metric vector shape {result.shape}")
        return result

    def report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps) -> None:
        del collected_metrics, wandb_run, cumulative_timesteps


def _summary(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "absolute_mean": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "cumulative_signed": 0.0,
            "cumulative_absolute": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "absolute_mean": float(np.abs(values).mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "cumulative_signed": float(values.sum()),
        "cumulative_absolute": float(np.abs(values).sum()),
    }


def aggregate_campaign_metrics(vectors: Iterable[np.ndarray]) -> dict[str, Any]:
    rows = np.asarray(list(vectors), dtype=np.float64)
    if rows.size == 0:
        rows = np.zeros((0, CAMPAIGN_METRIC_VECTOR_SIZE), dtype=np.float64)
    rows = rows.reshape(-1, CAMPAIGN_METRIC_VECTOR_SIZE)
    if not np.isfinite(rows).all():
        raise FloatingPointError("Non-finite collected campaign metrics")
    reward_components: dict[str, Any] = {}
    for index, name in enumerate(V2_COMPONENTS):
        values = rows[:, [index, len(V2_COMPONENTS) + index]].reshape(-1)
        reward_components[name] = _summary(values)
    shaping_absolute = sum(
        float(record["cumulative_absolute"])
        for name, record in reward_components.items()
        if name != "outcome"
    )
    for name, record in reward_components.items():
        record["share_of_total_absolute_shaping"] = (
            0.0
            if name == "outcome" or shaping_absolute <= 0.0
            else float(record["cumulative_absolute"] / shaping_absolute)
        )

    mechanics_metrics: dict[str, Any] = {}
    for index, name in enumerate(MECHANICS_METRICS):
        first = _MECHANICS_START + index
        second = _MECHANICS_START + len(MECHANICS_METRICS) + index
        values = rows[:, [first, second]].reshape(-1)
        mechanics_metrics[name] = _summary(values)

    reset_counts = {
        name: int(round(rows[:, _RESET_START + index].sum()))
        for index, name in enumerate(CURRICULUM_FAMILIES)
    }
    reset_total = sum(reset_counts.values())
    reset_shares = {
        name: count / reset_total if reset_total else 0.0
        for name, count in reset_counts.items()
    }
    goals = {
        "blue": int(round(rows[:, _GOAL_START].sum())),
        "orange": int(round(rows[:, _GOAL_START + 1].sum())),
    }
    result = {
        "schema_version": METRICS_SCHEMA_VERSION,
        "metric_vector_count": int(len(rows)),
        "reward_components": reward_components,
        "mechanics_recovery": mechanics_metrics,
        "curriculum_reset_counts": reset_counts,
        "curriculum_reset_shares": reset_shares,
        "goals": goals,
    }
    numeric = [
        value
        for record in reward_components.values()
        for value in record.values()
        if isinstance(value, (int, float))
    ]
    if not all(math.isfinite(float(value)) for value in numeric):
        raise FloatingPointError("Non-finite aggregate campaign metrics")
    return result


def merge_campaign_metric_reports(reports: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge iteration reports by their sufficient count/sum/min/max statistics."""
    reports = list(reports)
    if not reports:
        return aggregate_campaign_metrics([])

    def merge_summaries(name: str, section: str) -> dict[str, float | int]:
        items = [report[section][name] for report in reports]
        count = sum(int(item["count"]) for item in items)
        signed = sum(float(item["cumulative_signed"]) for item in items)
        absolute = sum(float(item["cumulative_absolute"]) for item in items)
        return {
            "count": count,
            "mean": signed / count if count else 0.0,
            "absolute_mean": absolute / count if count else 0.0,
            "minimum": min(float(item["minimum"]) for item in items),
            "maximum": max(float(item["maximum"]) for item in items),
            "cumulative_signed": signed,
            "cumulative_absolute": absolute,
        }

    components = {
        name: merge_summaries(name, "reward_components") for name in V2_COMPONENTS
    }
    shaping_absolute = sum(
        float(item["cumulative_absolute"])
        for name, item in components.items()
        if name != "outcome"
    )
    for name, item in components.items():
        item["share_of_total_absolute_shaping"] = (
            0.0
            if name == "outcome" or shaping_absolute <= 0
            else float(item["cumulative_absolute"] / shaping_absolute)
        )
    mechanics = {
        name: merge_summaries(name, "mechanics_recovery")
        for name in MECHANICS_METRICS
    }
    reset_counts = {
        family: sum(report["curriculum_reset_counts"][family] for report in reports)
        for family in CURRICULUM_FAMILIES
    }
    reset_total = sum(reset_counts.values())
    return {
        "schema_version": METRICS_SCHEMA_VERSION,
        "metric_vector_count": sum(report["metric_vector_count"] for report in reports),
        "reward_components": components,
        "mechanics_recovery": mechanics,
        "curriculum_reset_counts": reset_counts,
        "curriculum_reset_shares": {
            name: value / reset_total if reset_total else 0.0
            for name, value in reset_counts.items()
        },
        "goals": {
            side: sum(report["goals"][side] for report in reports)
            for side in ("blue", "orange")
        },
    }
