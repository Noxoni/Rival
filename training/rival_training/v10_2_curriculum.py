"""Ball-acquisition reset curriculum for Rival Milestone 10.2 Stage 1."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Any, Mapping

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import (
    BACK_WALL_Y,
    BALL_RESTING_HEIGHT,
    CEILING_Z,
    SIDE_WALL_X,
)
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
)

from .v9_curriculum import _set_ball, _set_car


BALL_ACQUISITION_CURRICULUM_VERSION = "RivalBallAcquisitionCurriculumV1"
FAMILIES = (
    "stationary_close",
    "stationary_medium",
    "moving_chase",
    "awkward_heading",
    "natural_kickoff_holdout",
)
PHASE_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "A": {
        "stationary_close": 0.30,
        "stationary_medium": 0.25,
        "moving_chase": 0.20,
        "awkward_heading": 0.20,
        "natural_kickoff_holdout": 0.05,
    },
    "B": {
        "stationary_close": 0.10,
        "stationary_medium": 0.25,
        "moving_chase": 0.30,
        "awkward_heading": 0.25,
        "natural_kickoff_holdout": 0.10,
    },
}


def _yaw_vector(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)


def _yaw_to(source: np.ndarray, target: np.ndarray) -> float:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(
        source, dtype=np.float64
    )
    return float(math.atan2(float(delta[1]), float(delta[0])))


def _planar_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(
            np.asarray(left, dtype=np.float64)[:2]
            - np.asarray(right, dtype=np.float64)[:2]
        )
    )


class RivalBallAcquisitionCurriculumV1(StateMutator[GameState]):
    """Broad randomized reachable ground-ball acquisition starts."""

    def __init__(
        self,
        phase: str = "A",
        *,
        seed: int,
        weights: Mapping[str, float] | None = None,
        forced_family: str | None = None,
        forced_active_team: int | None = None,
    ) -> None:
        phase = str(phase).upper()
        if phase not in PHASE_WEIGHTS:
            raise ValueError(f"Unsupported acquisition phase: {phase}")
        selected = dict(PHASE_WEIGHTS[phase] if weights is None else weights)
        if set(selected) != set(FAMILIES):
            raise ValueError(f"Curriculum requires exactly {FAMILIES}")
        values = np.asarray(
            [float(selected[name]) for name in FAMILIES], dtype=np.float64
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Curriculum weights must be finite and non-negative")
        if not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("Curriculum weights must sum to one")
        if forced_family is not None and forced_family not in FAMILIES:
            raise ValueError(f"Unknown forced family: {forced_family}")
        if forced_active_team not in (None, 0, 1):
            raise ValueError("forced_active_team must be 0, 1, or None")
        self.phase = phase
        self.weights = values
        self.forced_family = forced_family
        self.forced_active_team = forced_active_team
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self._last_active_team = 0

    def seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    @staticmethod
    def _cars(state: GameState) -> tuple[Car, Car]:
        blue = next(
            car for car in state.cars.values() if int(car.team_num) == 0
        )
        orange = next(
            car for car in state.cars.values() if int(car.team_num) == 1
        )
        return blue, orange

    def _roles(self, state: GameState) -> tuple[Car, Car]:
        blue, orange = self._cars(state)
        active_team = (
            int(self.forced_active_team)
            if self.forced_active_team is not None
            else int(self._rng.integers(0, 2))
        )
        self._last_active_team = active_team
        return (blue, orange) if active_team == 0 else (orange, blue)

    def _safe_car_position(
        self,
        ball: np.ndarray,
        distance_range: tuple[float, float],
    ) -> tuple[np.ndarray, float]:
        for _ in range(256):
            bearing = self._rng.uniform(-math.pi, math.pi)
            distance = self._rng.uniform(*distance_range)
            position = np.asarray(ball, dtype=np.float64) - (
                _yaw_vector(bearing) * distance
            )
            position[2] = 17.0
            if (
                abs(float(position[0])) <= SIDE_WALL_X - 350.0
                and abs(float(position[1])) <= BACK_WALL_Y - 500.0
            ):
                return position, bearing
        raise RuntimeError("Could not sample a legal acquisition car position")

    def _place_dummy(
        self,
        dummy: Car,
        ball: np.ndarray,
        active_position: np.ndarray,
    ) -> None:
        corners = [
            np.asarray([x, y, 17.0], dtype=np.float64)
            for x in (-SIDE_WALL_X + 550.0, SIDE_WALL_X - 550.0)
            for y in (-BACK_WALL_Y + 650.0, BACK_WALL_Y - 650.0)
        ]
        position = max(
            corners,
            key=lambda value: min(
                _planar_distance(value, ball),
                _planar_distance(value, active_position),
            ),
        )
        yaw = _yaw_to(position, ball) + math.pi
        _set_car(
            dummy,
            position=position,
            velocity=np.zeros(3),
            euler=np.asarray([0.0, yaw, 0.0]),
            boost=0.0,
        )

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        family = self.forced_family
        if family is None:
            family = FAMILIES[
                int(self._rng.choice(len(FAMILIES), p=self.weights))
            ]
        getattr(self, f"_apply_{family}")(state)
        shared_info["rival_v10_2_curriculum_version"] = (
            BALL_ACQUISITION_CURRICULUM_VERSION
        )
        shared_info["rival_v10_2_curriculum_phase"] = self.phase
        shared_info["rival_v10_2_curriculum_seed"] = self._seed
        shared_info["rival_v10_2_reset_family"] = family
        shared_info["rival_v10_2_active_team"] = self._last_active_team

    def _stationary(
        self,
        state: GameState,
        *,
        distance_range: tuple[float, float],
        car_speed_max: float,
        boost_range: tuple[float, float],
        awkward: bool,
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
            yaw = self._rng.uniform(-math.pi, math.pi)
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
        )

    def _apply_stationary_medium(self, state: GameState) -> None:
        self._stationary(
            state,
            distance_range=(1400.0, 3500.0),
            car_speed_max=1100.0,
            boost_range=(0.0, 80.0),
            awkward=False,
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
        intercept_yaw = _yaw_to(car_position, ball) + self._rng.uniform(
            -1.4, 1.4
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
            velocity=_yaw_vector(intercept_yaw)
            * self._rng.uniform(0.0, 1100.0),
            euler=np.asarray([0.0, self._rng.uniform(-math.pi, math.pi), 0.0]),
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

    def _apply_natural_kickoff_holdout(self, state: GameState) -> None:
        index = int(self._rng.integers(0, len(KickoffMutator.SPAWN_BLUE_POS)))
        _set_ball(
            state,
            position=np.asarray([0.0, 0.0, BALL_RESTING_HEIGHT]),
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
        )
        blue, orange = self._cars(state)
        _set_car(
            blue,
            position=KickoffMutator.SPAWN_BLUE_POS[index],
            velocity=np.zeros(3),
            euler=np.asarray(
                [0.0, KickoffMutator.SPAWN_BLUE_YAW[index], 0.0]
            ),
            boost=33.3,
        )
        _set_car(
            orange,
            position=KickoffMutator.SPAWN_ORANGE_POS[index],
            velocity=np.zeros(3),
            euler=np.asarray(
                [0.0, KickoffMutator.SPAWN_ORANGE_YAW[index], 0.0]
            ),
            boost=33.3,
        )
        self._last_active_team = (
            int(self.forced_active_team)
            if self.forced_active_team is not None
            else int(self._rng.integers(0, 2))
        )


def _legal_state(state: GameState) -> bool:
    vectors = [
        state.ball.position,
        state.ball.linear_velocity,
        state.ball.angular_velocity,
    ]
    for car in state.cars.values():
        vectors.extend(
            (
                car.physics.position,
                car.physics.linear_velocity,
                car.physics.angular_velocity,
                car.physics.euler_angles,
            )
        )
        position = np.asarray(car.physics.position, dtype=np.float64)
        if not 0.0 <= float(car.boost_amount) <= 100.0:
            return False
        if (
            abs(float(position[0])) > SIDE_WALL_X
            or abs(float(position[1])) > BACK_WALL_Y
            or not 0.0 <= float(position[2]) <= CEILING_Z
        ):
            return False
    ball = np.asarray(state.ball.position, dtype=np.float64)
    return bool(
        all(np.isfinite(vector).all() for vector in vectors)
        and abs(float(ball[0])) <= SIDE_WALL_X
        and abs(float(ball[1])) <= BACK_WALL_Y
        and BALL_RESTING_HEIGHT <= float(ball[2]) <= CEILING_Z
        and float(np.linalg.norm(state.ball.linear_velocity)) <= 6000.0
        and all(
            float(np.linalg.norm(car.physics.linear_velocity)) <= 2300.0
            for car in state.cars.values()
        )
    )


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p50": float(np.percentile(array, 50)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
    }


def curriculum_reset_audit(
    phase: str,
    *,
    seed: int,
    samples_per_family: int,
) -> dict[str, Any]:
    """Audit forced families and the configured mixed distribution."""

    minimum = 10_000 if str(phase).upper() == "A" else 5_000
    if int(samples_per_family) < minimum:
        raise ValueError(
            f"Phase {phase} requires at least {minimum} resets per family"
        )
    engine = RocketSimEngine(rlbot_delay=True)
    team_size = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    family_reports: dict[str, Any] = {}
    all_legal = True
    all_non_interfering = True
    for family_index, family in enumerate(FAMILIES):
        mutator = RivalBallAcquisitionCurriculumV1(
            phase,
            seed=int(seed) + family_index * 100_000,
            forced_family=family,
        )
        active_counts: Counter[int] = Counter()
        stats: dict[str, list[float]] = defaultdict(list)
        family_legal = True
        non_interfering = True
        for _ in range(int(samples_per_family)):
            state = engine.create_base_state()
            shared: dict[str, Any] = {}
            team_size.apply(state, shared)
            mutator.apply(state, shared)
            active_team = int(shared["rival_v10_2_active_team"])
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
            active_counts[active_team] += 1
            stats["planar_car_ball_distance"].append(
                _planar_distance(active.physics.position, state.ball.position)
            )
            stats["ball_speed"].append(
                float(np.linalg.norm(state.ball.linear_velocity))
            )
            stats["ball_height"].append(float(state.ball.position[2]))
            stats["ball_x"].append(float(state.ball.position[0]))
            stats["dummy_ball_distance"].append(
                _planar_distance(dummy.physics.position, state.ball.position)
            )
            family_legal = family_legal and _legal_state(state)
            if family != "natural_kickoff_holdout":
                non_interfering = non_interfering and (
                    _planar_distance(dummy.physics.position, state.ball.position)
                    >= 2500.0
                )
        team_share = active_counts[0] / int(samples_per_family)
        family_reports[family] = {
            "samples": int(samples_per_family),
            "active_team_counts": {
                "0": int(active_counts[0]),
                "1": int(active_counts[1]),
            },
            "distributions": {
                name: _summary(values) for name, values in stats.items()
            },
            "checks": {
                "all_physics_finite_and_legal": family_legal,
                "active_team_balance_within_two_percent": abs(
                    team_share - 0.5
                )
                <= 0.02,
                "left_right_geometry_balanced": abs(
                    float(np.mean(stats["ball_x"]))
                )
                <= 100.0,
                "dummy_initially_non_interfering": non_interfering,
            },
        }
        family_reports[family]["checks"]["passed"] = all(
            family_reports[family]["checks"].values()
        )
        all_legal = all_legal and family_legal
        all_non_interfering = all_non_interfering and non_interfering

    mixture_samples = int(samples_per_family) * len(FAMILIES)
    mixed = RivalBallAcquisitionCurriculumV1(
        phase,
        seed=int(seed) + 900_000,
    )
    counts: Counter[str] = Counter()
    for _ in range(mixture_samples):
        state = engine.create_base_state()
        shared = {}
        team_size.apply(state, shared)
        mixed.apply(state, shared)
        counts[str(shared["rival_v10_2_reset_family"])] += 1
    shares = {name: counts[name] / mixture_samples for name in FAMILIES}
    configured = {
        name: float(mixed.weights[index])
        for index, name in enumerate(FAMILIES)
    }
    maximum_share_error = max(
        abs(shares[name] - configured[name]) for name in FAMILIES
    )
    checks = {
        "required_resets_per_family_completed": int(samples_per_family)
        >= minimum,
        "all_family_audits_passed": all(
            row["checks"]["passed"] for row in family_reports.values()
        ),
        "all_physics_finite_and_legal": all_legal,
        "dummy_initial_geometry_non_interfering": all_non_interfering,
        "configured_mixture_within_one_percent": maximum_share_error <= 0.01,
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": 1,
        "curriculum_version": BALL_ACQUISITION_CURRICULUM_VERSION,
        "phase": str(phase).upper(),
        "samples_per_family": int(samples_per_family),
        "total_forced_family_samples": int(samples_per_family)
        * len(FAMILIES),
        "family_reports": family_reports,
        "mixed_distribution": {
            "samples": mixture_samples,
            "counts": {name: int(counts[name]) for name in FAMILIES},
            "shares": shares,
            "configured": configured,
            "maximum_absolute_share_error": maximum_share_error,
        },
        "checks": checks,
    }
