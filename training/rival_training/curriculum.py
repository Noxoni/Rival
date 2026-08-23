"""Majority-natural broad state curriculum for Milestone 06."""

from __future__ import annotations

from collections import Counter
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


CURRICULUM_VERSION = "RivalBroadCurriculumV1"
CURRICULUM_FAMILIES = (
    "natural",
    "aerial_wall_possession",
    "recovery",
    "low_resource_aerial",
)


def _clip_field(position: np.ndarray, *, car: bool = False) -> np.ndarray:
    padding = 130.0 if car else 100.0
    result = np.asarray(position, dtype=np.float32).copy()
    result[0] = np.clip(result[0], -SIDE_WALL_X + padding, SIDE_WALL_X - padding)
    result[1] = np.clip(result[1], -BACK_WALL_Y + padding, BACK_WALL_Y - padding)
    result[2] = np.clip(result[2], 17.0 if car else BALL_RESTING_HEIGHT, CEILING_Z - padding)
    return result


def _set_car(
    car: Car,
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    euler: np.ndarray,
    boost: float,
    angular_velocity: np.ndarray | None = None,
) -> None:
    car.physics.position = _clip_field(position, car=True)
    car.physics.linear_velocity = np.asarray(velocity, dtype=np.float32)
    car.physics.angular_velocity = np.asarray(
        angular_velocity if angular_velocity is not None else np.zeros(3),
        dtype=np.float32,
    )
    car.physics.euler_angles = np.asarray(euler, dtype=np.float32)
    car.boost_amount = float(np.clip(boost, 0.0, 100.0))


class RivalCurriculumMutator(StateMutator[GameState]):
    """Choose one broad reset family while preserving a natural majority."""

    def __init__(self, weights: Mapping[str, float], *, seed: int) -> None:
        if set(weights) != set(CURRICULUM_FAMILIES):
            raise ValueError(f"Expected curriculum families {CURRICULUM_FAMILIES}")
        values = np.asarray([float(weights[name]) for name in CURRICULUM_FAMILIES])
        if not np.isfinite(values).all() or np.any(values < 0) or values.sum() <= 0:
            raise ValueError("Curriculum weights must be finite and non-negative")
        self.weights = values / values.sum()
        if float(self.weights[0]) < 0.75:
            raise ValueError("Natural 1v1 must remain at least 75% of resets")
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def seed(self, seed: int) -> None:
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        index = int(self._rng.choice(len(CURRICULUM_FAMILIES), p=self.weights))
        family = CURRICULUM_FAMILIES[index]
        if family == "natural":
            self._apply_natural(state)
        elif family == "aerial_wall_possession":
            self._apply_aerial_wall_possession(state)
        elif family == "recovery":
            self._apply_recovery(state)
        elif family == "low_resource_aerial":
            self._apply_low_resource_aerial(state)
        else:  # pragma: no cover - tuple and choice make this unreachable
            raise AssertionError(f"Unsupported curriculum family {family}")
        counts = shared_info.setdefault("curriculum_reset_counts", {})
        counts[family] = int(counts.get(family, 0)) + 1
        shared_info["reset_family"] = family
        shared_info["reset_family_pending_metric"] = True
        shared_info["curriculum_seed"] = self._seed

    def _cars_by_team(self, state: GameState) -> tuple[Car, Car]:
        blue = next(car for car in state.cars.values() if car.team_num == 0)
        orange = next(car for car in state.cars.values() if car.team_num == 1)
        return blue, orange

    def _apply_natural(self, state: GameState) -> None:
        # Reproduce the ordinary symmetric 1v1 kickoff family using the mutator's
        # checkpointable seed rather than process-global randomness.
        index = int(self._rng.integers(0, len(KickoffMutator.SPAWN_BLUE_POS)))
        state.ball.position = np.array(
            [0.0, 0.0, BALL_RESTING_HEIGHT], dtype=np.float32
        )
        state.ball.linear_velocity = np.zeros(3, dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)
        blue, orange = self._cars_by_team(state)
        _set_car(
            blue,
            position=KickoffMutator.SPAWN_BLUE_POS[index],
            velocity=np.zeros(3),
            euler=np.array([0.0, KickoffMutator.SPAWN_BLUE_YAW[index], 0.0]),
            boost=33.3,
        )
        _set_car(
            orange,
            position=KickoffMutator.SPAWN_ORANGE_POS[index],
            velocity=np.zeros(3),
            euler=np.array([0.0, KickoffMutator.SPAWN_ORANGE_YAW[index], 0.0]),
            boost=33.3,
        )

    def _random_team_assignment(self, state: GameState) -> tuple[Car, Car, float]:
        blue, orange = self._cars_by_team(state)
        if bool(self._rng.integers(0, 2)):
            return blue, orange, 1.0
        return orange, blue, -1.0

    def _apply_aerial_wall_possession(self, state: GameState) -> None:
        possessor, defender, attack = self._random_team_assignment(state)
        side = float(self._rng.choice((-1.0, 1.0)))
        backboard_case = bool(self._rng.random() < 0.25)
        if backboard_case:
            ball_position = np.array(
                [
                    self._rng.uniform(-2200.0, 2200.0),
                    attack * self._rng.uniform(3900.0, 4900.0),
                    self._rng.uniform(500.0, 1700.0),
                ]
            )
        else:
            ball_position = np.array(
                [
                    side * self._rng.uniform(3150.0, 3950.0),
                    self._rng.uniform(-3900.0, 3900.0),
                    self._rng.uniform(350.0, 1750.0),
                ]
            )
        state.ball.position = _clip_field(ball_position)
        state.ball.linear_velocity = np.array(
            [
                self._rng.uniform(-900.0, 900.0),
                attack * self._rng.uniform(100.0, 1300.0),
                self._rng.uniform(-250.0, 650.0),
            ],
            dtype=np.float32,
        )
        state.ball.angular_velocity = self._rng.uniform(-4.0, 4.0, 3).astype(np.float32)
        approach = np.array(
            [
                self._rng.uniform(-650.0, 650.0),
                -attack * self._rng.uniform(250.0, 850.0),
                self._rng.uniform(-450.0, 250.0),
            ]
        )
        _set_car(
            possessor,
            position=state.ball.position + approach,
            velocity=state.ball.linear_velocity
            + self._rng.uniform(-450.0, 450.0, 3),
            euler=np.array(
                [
                    self._rng.uniform(-1.0, 1.0),
                    self._rng.uniform(-math.pi, math.pi),
                    self._rng.uniform(-1.4, 1.4),
                ]
            ),
            boost=self._rng.uniform(15.0, 100.0),
            angular_velocity=self._rng.uniform(-3.0, 3.0, 3),
        )
        pressure = np.array(
            [
                self._rng.uniform(-1800.0, 1800.0),
                -attack * self._rng.uniform(900.0, 2600.0),
                -state.ball.position[2] + 17.0,
            ]
        )
        _set_car(
            defender,
            position=state.ball.position + pressure,
            velocity=np.array(
                [self._rng.uniform(-900.0, 900.0), attack * self._rng.uniform(300.0, 1500.0), 0.0]
            ),
            euler=np.array([0.0, attack * math.pi / 2, 0.0]),
            boost=self._rng.uniform(10.0, 100.0),
        )

    def _apply_recovery(self, state: GameState) -> None:
        recovering, opponent, attack = self._random_team_assignment(state)
        state.ball.position = _clip_field(
            np.array(
                [
                    self._rng.uniform(-3000.0, 3000.0),
                    self._rng.uniform(-3400.0, 3400.0),
                    self._rng.uniform(100.0, 900.0),
                ]
            )
        )
        state.ball.linear_velocity = self._rng.uniform(
            [-1000.0, -1500.0, -300.0], [1000.0, 1500.0, 600.0]
        ).astype(np.float32)
        state.ball.angular_velocity = self._rng.uniform(-4.0, 4.0, 3).astype(np.float32)
        near_wall = bool(self._rng.random() < 0.45)
        if near_wall:
            x = float(self._rng.choice((-1.0, 1.0))) * self._rng.uniform(3300.0, 3950.0)
        else:
            x = self._rng.uniform(-3200.0, 3200.0)
        _set_car(
            recovering,
            position=np.array(
                [x, self._rng.uniform(-3800.0, 3800.0), self._rng.uniform(180.0, 1500.0)]
            ),
            velocity=self._rng.uniform(
                [-1800.0, -1800.0, -900.0], [1800.0, 1800.0, 900.0]
            ),
            euler=self._rng.uniform(
                [-math.pi, -math.pi, -math.pi], [math.pi, math.pi, math.pi]
            ),
            boost=self._rng.uniform(0.0, 65.0),
            angular_velocity=self._rng.uniform(-5.0, 5.0, 3),
        )
        _set_car(
            opponent,
            position=state.ball.position
            + np.array(
                [self._rng.uniform(-900.0, 900.0), -attack * self._rng.uniform(350.0, 1400.0), -state.ball.position[2] + 17.0]
            ),
            velocity=np.array(
                [self._rng.uniform(-700.0, 700.0), attack * self._rng.uniform(500.0, 1700.0), 0.0]
            ),
            euler=np.array([0.0, attack * math.pi / 2, 0.0]),
            boost=self._rng.uniform(20.0, 100.0),
        )

    def _apply_low_resource_aerial(self, state: GameState) -> None:
        attacker, defender, attack = self._random_team_assignment(state)
        state.ball.position = _clip_field(
            np.array(
                [
                    self._rng.uniform(-2800.0, 2800.0),
                    self._rng.uniform(-3000.0, 3000.0),
                    self._rng.uniform(450.0, 1500.0),
                ]
            )
        )
        state.ball.linear_velocity = np.array(
            [
                self._rng.uniform(-700.0, 700.0),
                attack * self._rng.uniform(250.0, 1200.0),
                self._rng.uniform(-150.0, 500.0),
            ],
            dtype=np.float32,
        )
        state.ball.angular_velocity = self._rng.uniform(-3.0, 3.0, 3).astype(np.float32)
        _set_car(
            attacker,
            position=state.ball.position
            + np.array(
                [
                    self._rng.uniform(-600.0, 600.0),
                    -attack * self._rng.uniform(550.0, 1300.0),
                    -self._rng.uniform(250.0, 700.0),
                ]
            ),
            velocity=np.array(
                [self._rng.uniform(-600.0, 600.0), attack * self._rng.uniform(700.0, 1900.0), self._rng.uniform(150.0, 800.0)]
            ),
            euler=np.array(
                [self._rng.uniform(-0.7, 0.4), attack * math.pi / 2, self._rng.uniform(-0.5, 0.5)]
            ),
            boost=self._rng.uniform(0.0, 35.0),
        )
        _set_car(
            defender,
            position=state.ball.position
            + np.array(
                [self._rng.uniform(-1400.0, 1400.0), attack * self._rng.uniform(900.0, 2200.0), -state.ball.position[2] + 17.0]
            ),
            velocity=np.array(
                [self._rng.uniform(-700.0, 700.0), -attack * self._rng.uniform(200.0, 1200.0), 0.0]
            ),
            euler=np.array([0.0, -attack * math.pi / 2, 0.0]),
            boost=self._rng.uniform(15.0, 100.0),
        )


def curriculum_distribution_smoke(
    mutator: RivalCurriculumMutator,
    state_factory,
    team_size_mutator,
    *,
    samples: int,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    shared_info: dict[str, Any] = {}
    for _ in range(samples):
        state = state_factory()
        team_size_mutator.apply(state, shared_info)
        mutator.apply(state, shared_info)
        counts[str(shared_info["reset_family"])] += 1
        vectors = [state.ball.position, state.ball.linear_velocity]
        vectors.extend(car.physics.position for car in state.cars.values())
        vectors.extend(car.physics.linear_velocity for car in state.cars.values())
        if not all(np.isfinite(vector).all() for vector in vectors):
            raise FloatingPointError("Curriculum mutator produced non-finite physics")
    return {
        "samples": samples,
        "counts": {name: int(counts[name]) for name in CURRICULUM_FAMILIES},
        "shares": {
            name: float(counts[name] / max(samples, 1)) for name in CURRICULUM_FAMILIES
        },
        "configured_weights": {
            name: float(mutator.weights[index])
            for index, name in enumerate(CURRICULUM_FAMILIES)
        },
    }
