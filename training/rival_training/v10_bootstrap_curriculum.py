"""Interaction-dense reset curriculum for Rival Milestone 10.1."""

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
from rlgym.rocket_league.state_mutators import KickoffMutator

from .v9_curriculum import _set_ball, _set_car


CURRICULUM_VERSION = "RivalAgencyBootstrapCurriculumV1"
FAMILIES = (
    "ground_acquisition",
    "moving_ball_chase",
    "touch_chain",
    "easy_aerial_contact",
    "easy_finish",
    "natural",
)
PHASE_WEIGHTS: Mapping[str, Mapping[str, float]] = {
    "A": {
        "ground_acquisition": 0.30,
        "moving_ball_chase": 0.20,
        "touch_chain": 0.20,
        "easy_aerial_contact": 0.15,
        "easy_finish": 0.10,
        "natural": 0.05,
    },
    "B": {
        "ground_acquisition": 0.20,
        "moving_ball_chase": 0.15,
        "touch_chain": 0.20,
        "easy_aerial_contact": 0.15,
        "easy_finish": 0.10,
        "natural": 0.20,
    },
    "C": {
        "ground_acquisition": 0.15,
        "moving_ball_chase": 0.10,
        "touch_chain": 0.15,
        "easy_aerial_contact": 0.10,
        "easy_finish": 0.10,
        "natural": 0.40,
    },
}


def _yaw_vector(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw), math.sin(yaw), 0.0], dtype=np.float64)


def _yaw_to(source: np.ndarray, target: np.ndarray) -> float:
    delta = np.asarray(target, dtype=np.float64) - np.asarray(source, dtype=np.float64)
    return float(math.atan2(float(delta[1]), float(delta[0])))


def _rotate2(vector: np.ndarray, angle: float) -> np.ndarray:
    x, y = float(vector[0]), float(vector[1])
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray([cosine * x - sine * y, sine * x + cosine * y, 0.0])


class RivalAgencyBootstrapCurriculumV1(StateMutator[GameState]):
    """Broad randomized Phase A/B/C interaction reset distribution."""

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
            raise ValueError(f"Unsupported bootstrap curriculum phase: {phase}")
        selected = dict(PHASE_WEIGHTS[phase] if weights is None else weights)
        if set(selected) != set(FAMILIES):
            raise ValueError(f"Bootstrap curriculum requires exactly {FAMILIES}")
        values = np.asarray([float(selected[name]) for name in FAMILIES], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Bootstrap curriculum weights must be finite and non-negative")
        if not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("Bootstrap curriculum weights must sum to one")
        if forced_family is not None and forced_family not in FAMILIES:
            raise ValueError(f"Unknown forced family: {forced_family}")
        if forced_active_team not in (None, 0, 1):
            raise ValueError("forced_active_team must be 0, 1, or None")
        self.phase = phase
        self.weights = values
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)
        self.forced_family = forced_family
        self.forced_active_team = forced_active_team
        self._last_active_team = 0

    def seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    @staticmethod
    def _cars(state: GameState) -> tuple[Car, Car]:
        blue = next(car for car in state.cars.values() if int(car.team_num) == 0)
        orange = next(car for car in state.cars.values() if int(car.team_num) == 1)
        return blue, orange

    def _roles(self, state: GameState) -> tuple[Car, Car, float]:
        blue, orange = self._cars(state)
        active_team = (
            int(self.forced_active_team)
            if self.forced_active_team is not None
            else int(self._rng.integers(0, 2))
        )
        self._last_active_team = active_team
        return (blue, orange, 1.0) if active_team == 0 else (orange, blue, -1.0)

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        family = self.forced_family
        if family is None:
            family = FAMILIES[int(self._rng.choice(len(FAMILIES), p=self.weights))]
        getattr(self, f"_apply_{family}")(state)
        shared_info["rival_v10_curriculum_version"] = CURRICULUM_VERSION
        shared_info["rival_v10_curriculum_phase"] = self.phase
        shared_info["rival_v10_curriculum_seed"] = self._seed
        shared_info["rival_v10_reset_family"] = family
        shared_info["rival_v10_active_team"] = self._last_active_team
        shared_info["rival_v10_reset_family_pending_metric"] = True

    def _apply_natural(self, state: GameState) -> None:
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
            euler=np.asarray([0.0, KickoffMutator.SPAWN_BLUE_YAW[index], 0.0]),
            boost=33.3,
        )
        _set_car(
            orange,
            position=KickoffMutator.SPAWN_ORANGE_POS[index],
            velocity=np.zeros(3),
            euler=np.asarray([0.0, KickoffMutator.SPAWN_ORANGE_YAW[index], 0.0]),
            boost=33.3,
        )
        self._last_active_team = int(self._rng.integers(0, 2))

    def _apply_ground_acquisition(self, state: GameState) -> None:
        active, opponent, attack = self._roles(state)
        ball = np.asarray(
            [
                self._rng.uniform(-2500.0, 2500.0),
                self._rng.uniform(-3000.0, 3000.0),
                BALL_RESTING_HEIGHT,
            ]
        )
        ball_speed = self._rng.uniform(0.0, 700.0)
        ball_yaw = self._rng.uniform(-math.pi, math.pi)
        _set_ball(
            state,
            position=ball,
            velocity=_yaw_vector(ball_yaw) * ball_speed,
            angular_velocity=self._rng.uniform(-3.0, 3.0, 3),
        )
        distance = self._rng.uniform(600.0, 2200.0)
        approach_yaw = self._rng.uniform(-math.pi, math.pi)
        car_position = ball - _yaw_vector(approach_yaw) * distance
        yaw = _yaw_to(car_position, ball) + self._rng.uniform(-math.pi / 4.0, math.pi / 4.0)
        speed = self._rng.uniform(0.0, 800.0)
        _set_car(
            active,
            position=car_position + np.asarray([0.0, 0.0, 17.0 - BALL_RESTING_HEIGHT]),
            velocity=_yaw_vector(yaw) * speed,
            euler=np.asarray([0.0, yaw, 0.0]),
            boost=self._rng.uniform(20.0, 80.0),
        )
        opponent_distance = self._rng.uniform(2200.0, 4500.0)
        lateral = self._rng.uniform(-1800.0, 1800.0)
        opponent_position = ball + np.asarray([lateral, attack * opponent_distance, 17.0 - BALL_RESTING_HEIGHT])
        opponent_yaw = _yaw_to(opponent_position, ball) + self._rng.uniform(-0.5, 0.5)
        _set_car(
            opponent,
            position=opponent_position,
            velocity=_yaw_vector(opponent_yaw) * self._rng.uniform(0.0, 700.0),
            euler=np.asarray([0.0, opponent_yaw, 0.0]),
            boost=self._rng.uniform(10.0, 80.0),
        )

    def _apply_moving_ball_chase(self, state: GameState) -> None:
        active, opponent, attack = self._roles(state)
        ball = np.asarray(
            [
                self._rng.uniform(-2400.0, 2400.0),
                self._rng.uniform(-2800.0, 2800.0),
                self._rng.uniform(BALL_RESTING_HEIGHT, 230.0),
            ]
        )
        ball_yaw = self._rng.uniform(-math.pi, math.pi)
        ball_speed = self._rng.uniform(300.0, 1400.0)
        ball_velocity = _yaw_vector(ball_yaw) * ball_speed
        ball_velocity[2] = self._rng.uniform(-80.0, 350.0) if ball[2] > 110.0 else 0.0
        _set_ball(
            state,
            position=ball,
            velocity=ball_velocity,
            angular_velocity=self._rng.uniform(-4.0, 4.0, 3),
        )
        intercept_direction = _rotate2(
            _safe_horizontal(ball_velocity), self._rng.uniform(-1.15, 1.15)
        )
        distance = self._rng.uniform(800.0, 3000.0)
        car_position = ball - intercept_direction * distance
        car_position[2] = 17.0
        yaw = _yaw_to(car_position, ball) + self._rng.uniform(-0.65, 0.65)
        _set_car(
            active,
            position=car_position,
            velocity=_yaw_vector(yaw) * self._rng.uniform(100.0, 1100.0),
            euler=np.asarray([0.0, yaw, 0.0]),
            boost=self._rng.uniform(15.0, 80.0),
        )
        opponent_position = ball + np.asarray(
            [self._rng.uniform(-1600.0, 1600.0), attack * self._rng.uniform(2400.0, 4400.0), 17.0 - ball[2]]
        )
        opponent_yaw = _yaw_to(opponent_position, ball)
        _set_car(
            opponent,
            position=opponent_position,
            velocity=_yaw_vector(opponent_yaw) * self._rng.uniform(0.0, 900.0),
            euler=np.asarray([0.0, opponent_yaw, 0.0]),
            boost=self._rng.uniform(10.0, 80.0),
        )

    def _apply_touch_chain(self, state: GameState) -> None:
        active, opponent, attack = self._roles(state)
        yaw = attack * math.pi / 2.0 + self._rng.uniform(-0.65, 0.65)
        car_position = np.asarray(
            [self._rng.uniform(-2500.0, 2500.0), self._rng.uniform(-2800.0, 2800.0), 17.0]
        )
        forward = _yaw_vector(yaw)
        lateral = np.asarray([-forward[1], forward[0], 0.0])
        ball = car_position + forward * self._rng.uniform(150.0, 750.0) + lateral * self._rng.uniform(-260.0, 260.0)
        ball[2] = BALL_RESTING_HEIGHT
        car_speed = self._rng.uniform(400.0, 1500.0)
        ball_speed = self._rng.uniform(250.0, 1300.0)
        _set_ball(
            state,
            position=ball,
            velocity=forward * ball_speed + lateral * self._rng.uniform(-250.0, 250.0),
            angular_velocity=self._rng.uniform(-3.0, 3.0, 3),
        )
        _set_car(
            active,
            position=car_position,
            velocity=forward * car_speed + lateral * self._rng.uniform(-180.0, 180.0),
            euler=np.asarray([0.0, yaw, 0.0]),
            boost=self._rng.uniform(10.0, 60.0),
        )
        opponent_position = ball + np.asarray(
            [self._rng.uniform(-1500.0, 1500.0), attack * self._rng.uniform(1800.0, 4200.0), 17.0 - BALL_RESTING_HEIGHT]
        )
        opponent_yaw = _yaw_to(opponent_position, ball)
        _set_car(
            opponent,
            position=opponent_position,
            velocity=_yaw_vector(opponent_yaw) * self._rng.uniform(0.0, 900.0),
            euler=np.asarray([0.0, opponent_yaw, 0.0]),
            boost=self._rng.uniform(10.0, 70.0),
        )

    def _apply_easy_aerial_contact(self, state: GameState) -> None:
        active, opponent, attack = self._roles(state)
        sidewall = bool(self._rng.random() < 0.20)
        if sidewall:
            side = float(self._rng.choice((-1.0, 1.0)))
            ball_x = side * self._rng.uniform(3000.0, 3700.0)
        else:
            ball_x = self._rng.uniform(-2500.0, 2500.0)
        ball = np.asarray(
            [ball_x, self._rng.uniform(-2800.0, 2800.0), self._rng.uniform(250.0, 900.0)]
        )
        ball_yaw = self._rng.uniform(-math.pi, math.pi)
        _set_ball(
            state,
            position=ball,
            velocity=_yaw_vector(ball_yaw) * self._rng.uniform(0.0, 900.0) + np.asarray([0.0, 0.0, self._rng.uniform(-120.0, 300.0)]),
            angular_velocity=self._rng.uniform(-4.0, 4.0, 3),
        )
        horizontal_yaw = attack * math.pi / 2.0 + self._rng.uniform(-0.55, 0.55)
        horizontal = _yaw_vector(horizontal_yaw)
        car_position = ball - horizontal * self._rng.uniform(500.0, 1700.0)
        car_position[2] = 17.0
        yaw = _yaw_to(car_position, ball) + self._rng.uniform(-math.pi / 6.0, math.pi / 6.0)
        _set_car(
            active,
            position=car_position,
            velocity=_yaw_vector(yaw) * self._rng.uniform(0.0, 850.0),
            euler=np.asarray([0.0, yaw, 0.0]),
            boost=self._rng.uniform(30.0, 80.0),
        )
        opponent_position = ball + np.asarray(
            [self._rng.uniform(-1500.0, 1500.0), attack * self._rng.uniform(2000.0, 4500.0), 17.0 - ball[2]]
        )
        opponent_yaw = _yaw_to(opponent_position, ball)
        _set_car(
            opponent,
            position=opponent_position,
            velocity=_yaw_vector(opponent_yaw) * self._rng.uniform(0.0, 700.0),
            euler=np.asarray([0.0, opponent_yaw, 0.0]),
            boost=self._rng.uniform(10.0, 70.0),
        )

    def _apply_easy_finish(self, state: GameState) -> None:
        active, opponent, attack = self._roles(state)
        ball_y = attack * (BACK_WALL_Y - self._rng.uniform(700.0, 2500.0))
        ball = np.asarray(
            [self._rng.uniform(-1150.0, 1150.0), ball_y, BALL_RESTING_HEIGHT]
        )
        ball_velocity = np.asarray(
            [self._rng.uniform(-350.0, 350.0), attack * self._rng.uniform(0.0, 850.0), 0.0]
        )
        _set_ball(
            state,
            position=ball,
            velocity=ball_velocity,
            angular_velocity=self._rng.uniform(-3.0, 3.0, 3),
        )
        active_position = ball + np.asarray(
            [self._rng.uniform(-420.0, 420.0), -attack * self._rng.uniform(300.0, 1200.0), 17.0 - BALL_RESTING_HEIGHT]
        )
        active_yaw = _yaw_to(active_position, ball) + self._rng.uniform(-0.35, 0.35)
        _set_car(
            active,
            position=active_position,
            velocity=_yaw_vector(active_yaw) * self._rng.uniform(0.0, 950.0),
            euler=np.asarray([0.0, active_yaw, 0.0]),
            boost=self._rng.uniform(20.0, 80.0),
        )
        defender_position = ball + np.asarray(
            [self._rng.choice((-1.0, 1.0)) * self._rng.uniform(900.0, 2200.0), attack * self._rng.uniform(700.0, 2200.0), 17.0 - BALL_RESTING_HEIGHT]
        )
        defender_yaw = _yaw_to(defender_position, ball) + self._rng.uniform(-0.8, 0.8)
        _set_car(
            opponent,
            position=defender_position,
            velocity=_yaw_vector(defender_yaw) * self._rng.uniform(0.0, 850.0),
            euler=np.asarray([0.0, defender_yaw, 0.0]),
            boost=self._rng.uniform(0.0, 55.0),
        )


def _safe_horizontal(vector: np.ndarray) -> np.ndarray:
    horizontal = np.asarray([vector[0], vector[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    if norm <= 1e-9:
        return np.asarray([1.0, 0.0, 0.0])
    return horizontal / norm


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
        if not 0.0 <= float(car.boost_amount) <= 100.0:
            return False
        position = np.asarray(car.physics.position)
        if (
            abs(float(position[0])) > SIDE_WALL_X
            or abs(float(position[1])) > BACK_WALL_Y
            or not 0.0 <= float(position[2]) <= CEILING_Z
        ):
            return False
    ball_position = np.asarray(state.ball.position)
    return bool(
        all(np.isfinite(vector).all() for vector in vectors)
        and abs(float(ball_position[0])) <= SIDE_WALL_X
        and abs(float(ball_position[1])) <= BACK_WALL_Y
        and BALL_RESTING_HEIGHT <= float(ball_position[2]) <= CEILING_Z
        and float(np.linalg.norm(state.ball.linear_velocity)) <= 6000.0
        and all(
            float(np.linalg.norm(car.physics.linear_velocity)) <= 2300.0
            for car in state.cars.values()
        )
    )


def curriculum_distribution_report(
    mutator: RivalAgencyBootstrapCurriculumV1,
    state_factory,
    team_size_mutator,
    *,
    samples: int,
) -> dict[str, Any]:
    if int(samples) < 10_000:
        raise ValueError("Bootstrap distribution validation requires at least 10,000 resets")
    counts: Counter[str] = Counter()
    active_teams: Counter[int] = Counter()
    stats: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    shared_info: dict[str, Any] = {}
    all_legal = True
    for _ in range(int(samples)):
        state = state_factory()
        team_size_mutator.apply(state, shared_info)
        mutator.apply(state, shared_info)
        family = str(shared_info["rival_v10_reset_family"])
        active_team = int(shared_info["rival_v10_active_team"])
        active = next(car for car in state.cars.values() if int(car.team_num) == active_team)
        counts[family] += 1
        active_teams[active_team] += 1
        stats[family]["active_car_ball_distance"].append(
            float(np.linalg.norm(active.physics.position - state.ball.position))
        )
        stats[family]["ball_height"].append(float(state.ball.position[2]))
        stats[family]["active_car_speed"].append(
            float(np.linalg.norm(active.physics.linear_velocity))
        )
        stats[family]["active_car_boost"].append(float(active.boost_amount))
        stats[family]["ball_x"].append(float(state.ball.position[0]))
        all_legal = all_legal and _legal_state(state)
    shares = {name: counts[name] / int(samples) for name in FAMILIES}
    configured = {
        name: float(mutator.weights[index]) for index, name in enumerate(FAMILIES)
    }
    distribution_error = {
        name: abs(shares[name] - configured[name]) for name in FAMILIES
    }

    def summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "minimum": float(array.min()),
            "p05": float(np.percentile(array, 5)),
            "p50": float(np.percentile(array, 50)),
            "p95": float(np.percentile(array, 95)),
            "maximum": float(array.max()),
            "mean": float(array.mean()),
        }

    checks = {
        "at_least_10000_resets": int(samples) >= 10_000,
        "configured_weights_sum_to_one": math.isclose(
            sum(configured.values()), 1.0, abs_tol=1e-12
        ),
        "empirical_shares_within_two_percent": max(distribution_error.values()) <= 0.02,
        "all_reset_physics_finite_and_legal": all_legal,
        "active_team_balance_within_two_percent": abs(
            active_teams[0] / int(samples) - 0.5
        )
        <= 0.02,
        "left_right_ball_distribution_balanced": abs(
            float(
                np.mean(
                    [
                        value
                        for family in FAMILIES
                        for value in stats[family]["ball_x"]
                    ]
                )
            )
        )
        <= 100.0,
    }
    checks["passed"] = all(checks.values())
    return {
        "schema_version": 1,
        "curriculum_version": CURRICULUM_VERSION,
        "phase": mutator.phase,
        "samples": int(samples),
        "counts": {name: int(counts[name]) for name in FAMILIES},
        "shares": shares,
        "configured_weights": configured,
        "absolute_share_error": distribution_error,
        "active_team_counts": {str(team): int(active_teams[team]) for team in (0, 1)},
        "family_distributions": {
            family: {
                name: summary(values) for name, values in stats[family].items()
            }
            for family in FAMILIES
        },
        "checks": checks,
    }
