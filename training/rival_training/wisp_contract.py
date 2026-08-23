"""Pure-Python kernels for the versioned Wisp 432 observation contract.

The live bot keeps these calculations in :mod:`bot.eta`.  This module mirrors
that behavior without importing RLBot so RocketSim workers and parity tools use
the same numerical contract.  Intentional legacy behavior (including the case-C
``boost_dur`` accumulation order) is preserved because changing it would change
the frozen Wisp policy input.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
import math

import numpy as np


CONTRACT_VERSION = "Wisp432ContractV2"
BALL_RADIUS = 91.25
CAR_MAX_SPEED = 2300.0
CAR_MAX_THROTTLE_SPEED = 1410.0
BOOST_ACCELERATION = 991.66
MAX_PREDICTION_TICK = 599


def solve_exp_segment(v0: float, x0: float, v_inf: float) -> float:
    """Match ``bot.eta._solve_exp_segment`` including its two Newton steps."""
    amplitude = v_inf - v0
    time = x0 / max(v0, 1.0)
    for _ in range(2):
        exponential = math.exp(-time)
        remaining_distance = (
            v_inf * time + (v0 - v_inf) * (1 - exponential) - x0
        )
        remaining_velocity = v_inf - amplitude * exponential
        time -= remaining_distance / remaining_velocity
    return time


def solve_quad_segment(v0: float, x0: float, acceleration: float) -> float:
    """Match ``bot.eta._solve_quad_segment`` and its single Newton step."""
    time = x0 / max(v0, 1.0)
    remaining_distance = v0 * time + 0.5 * acceleration * time * time - x0
    remaining_velocity = v0 + acceleration * time
    return time - remaining_distance / remaining_velocity


def linear_eta(v0: float, x0: float, boost_duration: float) -> float:
    """Literal pure-Python port of the frozen production linear ETA routine."""
    throttle_asymptote = 1556.0
    boosted_asymptote = throttle_asymptote + BOOST_ACCELERATION
    elapsed = 0.0

    if v0 >= CAR_MAX_SPEED:
        return x0 / CAR_MAX_SPEED

    if v0 < CAR_MAX_THROTTLE_SPEED and boost_duration > 0:
        destination_time = solve_exp_segment(v0, x0, boosted_asymptote)
        if 0.0 <= destination_time <= boost_duration:
            return destination_time

        time_to_throttle_limit = math.log(
            (boosted_asymptote - v0)
            / (boosted_asymptote - CAR_MAX_THROTTLE_SPEED)
        )
        segment_time = min(time_to_throttle_limit, boost_duration)
        exponential = math.exp(-segment_time)
        end_velocity = boosted_asymptote - (boosted_asymptote - v0) * exponential
        segment_distance = (
            boosted_asymptote * segment_time
            + (v0 - boosted_asymptote) * (1 - exponential)
        )
        x0 -= segment_distance
        v0 = end_velocity
        boost_duration -= segment_time
        elapsed += segment_time

    if v0 < CAR_MAX_THROTTLE_SPEED and boost_duration <= 0:
        destination_time = solve_exp_segment(v0, x0, throttle_asymptote)
        time_to_throttle_limit = math.log(
            (throttle_asymptote - v0)
            / (throttle_asymptote - CAR_MAX_THROTTLE_SPEED)
        )
        if 0.0 <= destination_time < time_to_throttle_limit:
            return destination_time

        exponential = math.exp(-time_to_throttle_limit)
        end_velocity = throttle_asymptote - (
            throttle_asymptote - v0
        ) * exponential
        segment_distance = (
            throttle_asymptote * time_to_throttle_limit
            + (v0 - throttle_asymptote) * (1 - exponential)
        )
        x0 -= segment_distance
        v0 = end_velocity
        elapsed += time_to_throttle_limit

    if v0 >= CAR_MAX_THROTTLE_SPEED and boost_duration > 0:
        destination_time = solve_quad_segment(v0, x0, BOOST_ACCELERATION)
        if 0.0 <= destination_time <= boost_duration:
            return destination_time

        time_to_max_speed = (CAR_MAX_SPEED - v0) / BOOST_ACCELERATION
        if time_to_max_speed <= boost_duration:
            x0 -= (
                v0 * time_to_max_speed
                + 0.5 * BOOST_ACCELERATION * time_to_max_speed * time_to_max_speed
            )
            return elapsed + time_to_max_speed + x0 / CAR_MAX_SPEED

        x0 -= (
            v0 * boost_duration
            + 0.5 * BOOST_ACCELERATION * boost_duration * boost_duration
        )
        v0 += BOOST_ACCELERATION * boost_duration
        boost_duration = 0.0
        # Deliberately match the production order at bot/eta.py:131-132.
        elapsed += boost_duration

    return elapsed + x0 / v0


class WispEtaState:
    """Explicit per-observer/per-player state for production-compatible ETA."""

    def __init__(self) -> None:
        self._cache: dict[tuple[Hashable, Hashable], float] = {}

    def reset(self) -> None:
        self._cache.clear()

    def value(
        self,
        observer: Hashable,
        player: Hashable,
        position: np.ndarray,
        velocity: np.ndarray,
        boost_amount: float,
        prediction_position: Callable[[int], np.ndarray],
    ) -> float:
        """Run the live two-pass slice-selection/update contract."""
        key = (observer, player)
        time = self._cache.setdefault(key, 0.0)
        player_position = np.asarray(position, dtype=np.float64)
        player_velocity = np.asarray(velocity, dtype=np.float64)
        for _ in range(2):
            tick = min(int(time * 120), MAX_PREDICTION_TICK)
            delta = np.asarray(prediction_position(tick), dtype=np.float64) - player_position
            distance = float(np.linalg.norm(delta))
            if distance <= 0.0:
                # The live expression is undefined here. Keep the worker finite while
                # preserving the previous estimate; this only protects pathological
                # exact car/ball overlap resets.
                return float(time)
            initial_velocity = float(np.dot(player_velocity, delta / distance))
            time = linear_eta(
                initial_velocity,
                distance - 1.5 * BALL_RADIUS,
                float(boost_amount) / 33.3,
            )
        self._cache[key] = float(time)
        return float(time)


def ramp_handbrake(previous_value: float, applied: bool, ticks: int) -> float:
    """Match the live adapter's 5 units/second analog handbrake ramp."""
    direction = 1.0 if applied else -1.0
    return float(np.clip(previous_value + direction * ticks * (5.0 / 120.0), 0.0, 1.0))
