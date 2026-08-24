"""Milestone 10.3 Stage-1 V2 acquisition curriculum."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BALL_RESTING_HEIGHT
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator

from .v10_2_curriculum import (
    FAMILIES,
    RivalBallAcquisitionCurriculumV1,
    _legal_state,
    _planar_distance,
    _summary,
    _yaw_to,
    _yaw_vector,
)
from .v9_curriculum import _set_ball, _set_car


BALL_ACQUISITION_CURRICULUM_VERSION = "RivalBallAcquisitionCurriculumV2"
ORDINARY_HEADING_ERROR_DEGREES: Mapping[str, float] = {
    "stationary_close": 15.0,
    "stationary_medium": 30.0,
    "moving_chase": 45.0,
}


def _wrapped_degrees(value: float) -> float:
    return abs(math.degrees(math.atan2(math.sin(value), math.cos(value))))


class RivalBallAcquisitionCurriculumV2(RivalBallAcquisitionCurriculumV1):
    """V1 distribution with repaired ordinary-family initial orientation."""

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        super().apply(state, shared_info)
        shared_info["rival_v10_3_curriculum_version"] = (
            BALL_ACQUISITION_CURRICULUM_VERSION
        )
        shared_info["rival_v10_3_curriculum_phase"] = self.phase
        shared_info["rival_v10_3_curriculum_seed"] = self._seed
        shared_info["rival_v10_3_reset_family"] = shared_info[
            "rival_v10_2_reset_family"
        ]
        shared_info["rival_v10_3_active_team"] = shared_info[
            "rival_v10_2_active_team"
        ]

    def _stationary(
        self,
        state: GameState,
        *,
        distance_range: tuple[float, float],
        car_speed_max: float,
        boost_range: tuple[float, float],
        awkward: bool,
        heading_error_degrees: float | None = None,
    ) -> None:
        active, dummy = self._roles(state)
        ball = np.asarray(
            [
                self._rng.uniform(-1900.0, 1900.0),
                self._rng.uniform(-2600.0, 2600.0),
                BALL_RESTING_HEIGHT,
            ],
            dtype=np.float64,
        )
        car_position, bearing = self._safe_car_position(ball, distance_range)
        if awkward:
            error_magnitude = self._rng.uniform(math.pi / 2.0, math.pi)
            yaw = bearing + self._rng.choice((-1.0, 1.0)) * error_magnitude
            velocity_yaw = bearing + self._rng.uniform(1.6, 3.1)
        else:
            if heading_error_degrees is None:
                raise RuntimeError("V2 ordinary stationary reset needs heading limit")
            limit = math.radians(float(heading_error_degrees))
            yaw = bearing + self._rng.uniform(-limit, limit)
            velocity_yaw = yaw
        speed = self._rng.uniform(0.0, car_speed_max)
        _set_ball(
            state,
            position=ball,
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        _set_car(
            active,
            position=car_position,
            velocity=_yaw_vector(velocity_yaw) * speed,
            angular_velocity=self._rng.uniform(-0.8, 0.8, 3),
            euler=np.asarray([0.0, yaw, 0.0]),
            boost=self._rng.uniform(*boost_range),
        )
        self._place_dummy(dummy, ball, car_position)

    def _apply_stationary_close(self, state: GameState) -> None:
        self._stationary(
            state,
            distance_range=(400.0, 1400.0),
            car_speed_max=700.0,
            boost_range=(0.0, 60.0),
            awkward=False,
            heading_error_degrees=15.0,
        )

    def _apply_stationary_medium(self, state: GameState) -> None:
        self._stationary(
            state,
            distance_range=(1400.0, 3500.0),
            car_speed_max=1100.0,
            boost_range=(0.0, 80.0),
            awkward=False,
            heading_error_degrees=30.0,
        )

    def _apply_moving_chase(self, state: GameState) -> None:
        active, dummy = self._roles(state)
        ball = np.asarray(
            [
                self._rng.uniform(-1800.0, 1800.0),
                self._rng.uniform(-2400.0, 2400.0),
                self._rng.uniform(BALL_RESTING_HEIGHT, 180.0),
            ],
            dtype=np.float64,
        )
        ball_yaw = self._rng.uniform(-math.pi, math.pi)
        ball_speed = self._rng.uniform(200.0, 1400.0)
        ball_velocity = _yaw_vector(ball_yaw) * ball_speed
        if ball[2] > BALL_RESTING_HEIGHT + 5.0:
            ball_velocity[2] = self._rng.uniform(-120.0, 260.0)
        car_position, _ = self._safe_car_position(ball, (700.0, 3000.0))
        current_direction = _yaw_to(car_position, ball)
        heading = current_direction + self._rng.uniform(
            -math.pi / 4.0, math.pi / 4.0
        )
        _set_ball(
            state,
            position=ball,
            velocity=ball_velocity,
            angular_velocity=self._rng.uniform(-4.0, 4.0, 3),
        )
        _set_car(
            active,
            position=car_position,
            velocity=_yaw_vector(heading) * self._rng.uniform(0.0, 1100.0),
            euler=np.asarray([0.0, heading, 0.0]),
            boost=self._rng.uniform(10.0, 80.0),
        )
        self._place_dummy(dummy, ball, car_position)

    def _apply_awkward_heading(self, state: GameState) -> None:
        self._stationary(
            state,
            distance_range=(500.0, 2500.0),
            car_speed_max=1200.0,
            boost_range=(0.0, 70.0),
            awkward=True,
        )


def curriculum_reset_audit(
    phase: str,
    *,
    seed: int,
    samples_per_family: int,
) -> dict[str, Any]:
    minimum = 10_000 if str(phase).upper() == "A" else 5_000
    if int(samples_per_family) < minimum:
        raise ValueError(
            f"Phase {phase} requires at least {minimum} resets per family"
        )
    engine = RocketSimEngine(rlbot_delay=True)
    team_size = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    family_reports: dict[str, Any] = {}
    for family_index, family in enumerate(FAMILIES):
        mutator = RivalBallAcquisitionCurriculumV2(
            phase,
            seed=int(seed) + family_index * 100_000,
            forced_family=family,
        )
        team_counts: Counter[int] = Counter()
        values: dict[str, list[float]] = defaultdict(list)
        legal = True
        dummy_non_interfering = True
        for _ in range(int(samples_per_family)):
            state = engine.create_base_state()
            shared: dict[str, Any] = {}
            team_size.apply(state, shared)
            mutator.apply(state, shared)
            active_team = int(shared["rival_v10_3_active_team"])
            active = next(
                car
                for car in state.cars.values()
                if int(car.team_num) == active_team
            )
            dummy = next(
                car
                for car in state.cars.values()
                if int(car.team_num) != active_team
            )
            team_counts[active_team] += 1
            car = np.asarray(active.physics.position, dtype=np.float64)
            ball = np.asarray(state.ball.position, dtype=np.float64)
            yaw = float(active.physics.euler_angles[1])
            values["heading_error_degrees"].append(
                _wrapped_degrees(yaw - _yaw_to(car, ball))
            )
            values["planar_car_ball_distance"].append(
                _planar_distance(car, ball)
            )
            values["ball_x"].append(float(ball[0]))
            values["dummy_ball_distance"].append(
                _planar_distance(dummy.physics.position, ball)
            )
            legal = legal and _legal_state(state)
            if family != "natural_kickoff_holdout":
                dummy_non_interfering = dummy_non_interfering and (
                    _planar_distance(dummy.physics.position, ball) >= 2500.0
                )
        heading_errors = values["heading_error_degrees"]
        heading_contract = True
        if family in ORDINARY_HEADING_ERROR_DEGREES:
            heading_contract = max(heading_errors) <= (
                ORDINARY_HEADING_ERROR_DEGREES[family] + 1e-6
            )
        elif family == "awkward_heading":
            heading_contract = min(heading_errors) >= 90.0 - 1e-6
        team_share = team_counts[0] / int(samples_per_family)
        checks = {
            "all_physics_finite_and_legal": legal,
            "active_team_balance_within_two_percent": abs(team_share - 0.5)
            <= 0.02,
            "left_right_geometry_balanced": abs(float(np.mean(values["ball_x"])))
            <= 100.0,
            "dummy_initially_non_interfering": dummy_non_interfering,
            "v2_heading_contract_exact": heading_contract,
        }
        checks["passed"] = all(checks.values())
        family_reports[family] = {
            "samples": int(samples_per_family),
            "active_team_counts": {
                "0": int(team_counts[0]),
                "1": int(team_counts[1]),
            },
            "distributions": {
                name: _summary(rows) for name, rows in values.items()
            },
            "checks": checks,
        }

    mixture_samples = int(samples_per_family) * len(FAMILIES)
    mixed = RivalBallAcquisitionCurriculumV2(
        phase, seed=int(seed) + 900_000
    )
    counts: Counter[str] = Counter()
    for _ in range(mixture_samples):
        state = engine.create_base_state()
        shared = {}
        team_size.apply(state, shared)
        mixed.apply(state, shared)
        counts[str(shared["rival_v10_3_reset_family"])] += 1
    shares = {family: counts[family] / mixture_samples for family in FAMILIES}
    configured = {
        family: float(mixed.weights[index])
        for index, family in enumerate(FAMILIES)
    }
    checks = {
        "required_resets_per_family_completed": int(samples_per_family)
        >= minimum,
        "all_family_audits_passed": all(
            report["checks"]["passed"] for report in family_reports.values()
        ),
        "configured_mixture_within_one_percent": max(
            abs(shares[family] - configured[family]) for family in FAMILIES
        )
        <= 0.01,
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": 1,
        "curriculum_version": BALL_ACQUISITION_CURRICULUM_VERSION,
        "phase": str(phase).upper(),
        "samples_per_family": int(samples_per_family),
        "total_forced_family_samples": int(samples_per_family) * len(FAMILIES),
        "family_reports": family_reports,
        "mixed_distribution": {
            "samples": mixture_samples,
            "counts": {family: int(counts[family]) for family in FAMILIES},
            "shares": shares,
            "configured": configured,
        },
        "checks": checks,
    }
