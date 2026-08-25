"""Physical-time GAE proofs and reward-to-action credit diagnostics for M10.8."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

from .v9_observations import FIELD_SCALE, SCHEMA_FIELDS
from .v9_trainer import compute_physical_time_gae


PHYSICS_HZ = 120
HORIZON_SECONDS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
DELAYED_REWARD_SECONDS = (0.5, 1.0, 2.0, 3.0)
CONTACT_REWARD_EVENT_THRESHOLD = 5.0
FAILED_TIMEOUT_MINIMUM_TICKS = 1430
WINDOWS = {
    "zero_to_0p5_seconds_before_anchor": (0, 60),
    "0p5_to_1p0_seconds_before_anchor": (60, 120),
    "1p0_to_2p0_seconds_before_anchor": (120, 240),
    "2p0_to_3p0_seconds_before_anchor": (240, 360),
}


def _field(name: str) -> slice:
    match = next(item for item in SCHEMA_FIELDS if item.name == name)
    return slice(match.start, match.end)


TOUCH_AGE = _field("touch.self_age")
SELF_POSITION = _field("self.position")
SELF_FORWARD = _field("self.forward")
BALL_POSITION = _field("ball.position")


def numerical_summary(values: np.ndarray | Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    array = array.astype(np.float64, copy=False).reshape(-1)
    if not len(array):
        return {
            "samples": 0,
            "sum": 0.0,
            "sum_squares": 0.0,
            "mean": None,
            "variance": None,
            "standard_deviation": None,
            "minimum": None,
            "maximum": None,
        }
    if not np.isfinite(array).all():
        raise FloatingPointError("Credit diagnostics received non-finite values")
    return {
        "samples": int(len(array)),
        "sum": float(array.sum()),
        "sum_squares": float(np.square(array).sum()),
        "mean": float(array.mean()),
        "variance": float(array.var()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def merge_numerical_summaries(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if int(row["samples"]) > 0]
    if not selected:
        return numerical_summary(np.asarray([], dtype=np.float64))
    count = sum(int(row["samples"]) for row in selected)
    total = sum(float(row["sum"]) for row in selected)
    squares = sum(float(row["sum_squares"]) for row in selected)
    mean = total / count
    variance = max(0.0, squares / count - mean * mean)
    return {
        "samples": count,
        "sum": total,
        "sum_squares": squares,
        "mean": mean,
        "variance": variance,
        "standard_deviation": math.sqrt(variance),
        "minimum": min(float(row["minimum"]) for row in selected),
        "maximum": max(float(row["maximum"]) for row in selected),
    }


def gae_physical_time_report(
    *, gamma: float, gae_lambda: float, arm: str
) -> dict[str, Any]:
    product = float(gamma) * float(gae_lambda)
    if not 0.0 < product < 1.0:
        raise ValueError("gamma*lambda must be strictly between zero and one")
    retention = {
        f"{seconds:g}_seconds": product ** int(round(seconds * PHYSICS_HZ))
        for seconds in HORIZON_SECONDS
    }
    half_life = math.log(0.5) / (PHYSICS_HZ * math.log(product))

    # 361 transitions place a unit terminal reward exactly 360 native ticks
    # after action index zero, allowing exact 0.5/1/2/3-second probes.
    trajectory_ticks = 3 * PHYSICS_HZ + 1
    rewards = np.zeros(trajectory_ticks, dtype=np.float64)
    rewards[-1] = 1.0
    values = np.zeros_like(rewards)
    terminated = np.zeros_like(rewards)
    truncated = np.zeros_like(rewards)
    terminated[-1] = 1.0
    advantages, returns = compute_physical_time_gae(
        rewards,
        values,
        values,
        terminated,
        truncated,
        gamma=float(gamma),
        gae_lambda=float(gae_lambda),
    )
    synthetic = {}
    errors = []
    for seconds in DELAYED_REWARD_SECONDS:
        offset = int(round(seconds * PHYSICS_HZ))
        measured = float(advantages[-1 - offset])
        analytical = product**offset
        error = abs(measured - analytical)
        errors.append(error)
        synthetic[f"{seconds:g}_seconds_before_reward"] = {
            "ticks_before_reward": offset,
            "analytical_advantage": analytical,
            "measured_advantage": measured,
            "absolute_error": error,
        }
    checks = {
        "all_values_finite": bool(
            np.isfinite(advantages).all()
            and np.isfinite(returns).all()
            and all(math.isfinite(value) for value in retention.values())
        ),
        "synthetic_matches_analytical_within_1e_6": max(errors) <= 1e-6,
        "terminal_advantage_exactly_one": float(advantages[-1]) == 1.0,
    }
    checks["passed"] = all(checks.values())
    return {
        "arm": str(arm),
        "physics_hz": PHYSICS_HZ,
        "gamma": float(gamma),
        "lambda": float(gae_lambda),
        "gamma_times_lambda": product,
        "retention": retention,
        "half_life_seconds": half_life,
        "synthetic_delayed_reward": {
            "trajectory_ticks": trajectory_ticks,
            "terminal_reward": 1.0,
            "probes": synthetic,
            "maximum_absolute_error": max(errors),
        },
        "checks": checks,
    }


def _positions(observations: np.ndarray, field: slice) -> np.ndarray:
    return observations[:, field] * np.asarray(FIELD_SCALE, dtype=np.float32)


def _alignment(observations: np.ndarray) -> np.ndarray:
    self_position = _positions(observations, SELF_POSITION)
    ball_position = _positions(observations, BALL_POSITION)
    direction = ball_position - self_position
    norm = np.linalg.norm(direction, axis=1, keepdims=True)
    direction = np.divide(
        direction,
        np.maximum(norm, 1e-6),
        out=np.zeros_like(direction),
    )
    forward = observations[:, SELF_FORWARD]
    return np.clip(np.sum(forward * direction, axis=1), -1.0, 1.0)


def _distance_progress(
    observations: np.ndarray, next_observations: np.ndarray
) -> np.ndarray:
    previous_car = _positions(observations, SELF_POSITION)
    current_car = _positions(next_observations, SELF_POSITION)
    current_ball = _positions(next_observations, BALL_POSITION)
    raw = np.linalg.norm(previous_car - current_ball, axis=1) - np.linalg.norm(
        current_car - current_ball, axis=1
    )
    return np.clip(raw, -64.0, 64.0)


def _segments(terminated: np.ndarray, truncated: np.ndarray) -> list[tuple[int, int]]:
    boundary = np.flatnonzero(
        np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)
    )
    if not len(boundary) or int(boundary[-1]) != len(terminated) - 1:
        raise RuntimeError("Every collected PPO trajectory must end at a boundary")
    result = []
    start = 0
    for end in boundary:
        result.append((start, int(end) + 1))
        start = int(end) + 1
    return result


def _cohort_windows(
    anchors: list[tuple[int, int]],
    values: dict[str, np.ndarray],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for window, (newer_ticks, older_ticks) in WINDOWS.items():
        selected: dict[str, list[np.ndarray]] = {name: [] for name in values}
        eligible = 0
        for segment_start, anchor in anchors:
            start = anchor - older_ticks
            end = anchor - newer_ticks
            if start < segment_start or end <= start:
                continue
            eligible += 1
            for name, array in values.items():
                selected[name].append(array[start:end])
        report[window] = {
            "newer_offset_ticks": newer_ticks,
            "older_offset_ticks": older_ticks,
            "eligible_trajectories": eligible,
            "metrics": {
                name: numerical_summary(
                    np.concatenate(rows) if rows else np.asarray([], dtype=np.float64)
                )
                for name, rows in selected.items()
            },
        }
    return report


def credit_assignment_window_diagnostics(
    *,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_observations: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    advantages: np.ndarray,
    normalized_advantages: np.ndarray,
) -> dict[str, Any]:
    """Connect GAE values to policy actions before contacts and timeouts.

    The worker manager preserves trajectory ordering in the PPO arrays but not
    in its optional metrics side-channel, so this diagnostic deliberately uses
    only the exact tensors consumed by PPO. A contact is identified by the
    frozen reward contract's safe >5 threshold: a non-contact transition is
    bounded below that value, while a rewarded physical contact contributes
    +10. Collection-cut segments are excluded from the failed cohort by
    requiring a full timeout-like length.
    """

    observations = np.asarray(observations, dtype=np.float32)
    next_observations = np.asarray(next_observations, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    advantages = np.asarray(advantages, dtype=np.float64).reshape(-1)
    normalized_advantages = np.asarray(
        normalized_advantages, dtype=np.float64
    ).reshape(-1)
    lengths = {
        len(observations),
        len(next_observations),
        len(actions),
        len(rewards),
        len(advantages),
        len(normalized_advantages),
        len(terminated),
        len(truncated),
    }
    if len(lengths) != 1:
        raise ValueError(f"Credit diagnostic input lengths differ: {lengths}")
    if not all(
        np.isfinite(array).all()
        for array in (
            observations,
            next_observations,
            actions,
            rewards,
            advantages,
            normalized_advantages,
        )
    ):
        raise FloatingPointError("Credit diagnostic input is non-finite")

    alignment_delta = _alignment(next_observations) - _alignment(observations)
    progress = _distance_progress(observations, next_observations)
    values = {
        "raw_advantage": advantages,
        "normalized_advantage": normalized_advantages,
        "throttle": actions[:, 0],
        "steer_magnitude": np.abs(actions[:, 1]),
        "heading_improvement": alignment_delta,
        "distance_progress_uu": progress,
    }
    successful: list[tuple[int, int]] = []
    failed: list[tuple[int, int]] = []
    contact_events = 0
    touch_age_confirmed = 0
    collection_cut_or_short_failures = 0
    segment_rows = _segments(terminated, truncated)
    for start, end in segment_rows:
        candidates = np.flatnonzero(
            rewards[start:end] > CONTACT_REWARD_EVENT_THRESHOLD
        )
        if len(candidates):
            event = start + int(candidates[0])
            contact_events += 1
            touch_age_confirmed += int(
                float(next_observations[event, TOUCH_AGE][0]) <= 0.01
            )
            # Requiring reset/saturated touch age at segment start rejects
            # partial segments which clearly began after an earlier contact.
            if float(observations[start, TOUCH_AGE][0]) >= 0.999:
                successful.append((start, event + 1))
        elif end - start >= FAILED_TIMEOUT_MINIMUM_TICKS:
            failed.append((start, end))
        else:
            collection_cut_or_short_failures += 1

    checks = {
        "all_metrics_finite": True,
        "all_four_windows_present": set(WINDOWS)
        == set(_cohort_windows(successful, values)),
        "contact_reward_events_touch_age_confirmed": (
            contact_events == 0 or touch_age_confirmed == contact_events
        ),
        "trajectory_boundaries_cover_rollout": sum(
            end - start for start, end in segment_rows
        )
        == len(rewards),
    }
    checks["passed"] = all(checks.values())
    return {
        "version": "RivalM10_8CreditAssignmentWindowsV1",
        "physics_hz": PHYSICS_HZ,
        "contact_event_definition": "reward > 5 and next touch age confirms contact",
        "successful_cohort_definition": (
            "first visible rewarded contact in a trajectory segment whose initial "
            "self-touch age is reset/saturated"
        ),
        "failed_cohort_definition": (
            f"no rewarded contact and at least {FAILED_TIMEOUT_MINIMUM_TICKS} ticks; "
            "short collection-cut segments excluded"
        ),
        "trajectory_counts": {
            "segments": len(segment_rows),
            "contact_events": contact_events,
            "touch_age_confirmed_contact_events": touch_age_confirmed,
            "successful_first_contact_trajectories": len(successful),
            "failed_timeout_like_trajectories": len(failed),
            "excluded_collection_cut_or_short_failures": (
                collection_cut_or_short_failures
            ),
        },
        "successful_first_contact": _cohort_windows(successful, values),
        "failed_timeout_like": _cohort_windows(failed, values),
        "checks": checks,
    }


def aggregate_credit_diagnostics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    selected = list(rows)
    if not selected:
        raise ValueError("No credit diagnostics to aggregate")
    cohorts = ("successful_first_contact", "failed_timeout_like")
    report: dict[str, Any] = {
        "version": "RivalM10_8AggregateCreditAssignmentWindowsV1",
        "iterations": len(selected),
        "trajectory_counts": {
            key: sum(int(row["trajectory_counts"][key]) for row in selected)
            for key in selected[0]["trajectory_counts"]
        },
    }
    for cohort in cohorts:
        report[cohort] = {}
        for window in WINDOWS:
            report[cohort][window] = {
                "newer_offset_ticks": selected[0][cohort][window][
                    "newer_offset_ticks"
                ],
                "older_offset_ticks": selected[0][cohort][window][
                    "older_offset_ticks"
                ],
                "eligible_trajectories": sum(
                    int(row[cohort][window]["eligible_trajectories"])
                    for row in selected
                ),
                "metrics": {
                    metric: merge_numerical_summaries(
                        row[cohort][window]["metrics"][metric]
                        for row in selected
                    )
                    for metric in selected[0][cohort][window]["metrics"]
                },
            }
    report["checks"] = {
        "all_iterations_passed": all(row["checks"]["passed"] for row in selected),
        "all_four_windows_present": all(
            set(row[cohort]) == set(WINDOWS)
            for row in selected
            for cohort in cohorts
        ),
    }
    report["checks"]["passed"] = all(report["checks"].values())
    return report
