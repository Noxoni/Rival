"""Versioned majority-natural reset curriculum for the Rival v9 pilot.

The reset families intentionally describe broad state distributions rather than
named mechanics.  They only change initial RocketSim state; action, observation,
reward, policy, and PPO contracts remain untouched.
"""

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


V9_PILOT_CURRICULUM_VERSION = "RivalScratchResetCurriculumV1"
V9_PILOT_CURRICULUM_FAMILIES = (
    "natural",
    "ground_possession_challenge",
    "wall_aerial_ceiling_possession",
    "awkward_recovery_landing",
    "low_resource",
)
V9_PILOT_CURRICULUM_WEIGHTS = {
    "natural": 0.70,
    "ground_possession_challenge": 0.10,
    "wall_aerial_ceiling_possession": 0.08,
    "awkward_recovery_landing": 0.08,
    "low_resource": 0.04,
}


def _clip_position(position: np.ndarray, *, minimum_z: float, padding: float = 120.0) -> np.ndarray:
    result = np.asarray(position, dtype=np.float32).copy()
    result[0] = np.clip(result[0], -SIDE_WALL_X + padding, SIDE_WALL_X - padding)
    result[1] = np.clip(result[1], -BACK_WALL_Y + padding, BACK_WALL_Y - padding)
    result[2] = np.clip(result[2], minimum_z, CEILING_Z - padding)
    return result


def _set_ball(
    state: GameState,
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> None:
    state.ball.position = _clip_position(
        position, minimum_z=float(BALL_RESTING_HEIGHT), padding=100.0
    )
    state.ball.linear_velocity = np.asarray(velocity, dtype=np.float32)
    state.ball.angular_velocity = np.asarray(angular_velocity, dtype=np.float32)


def _set_car(
    car: Car,
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    euler: np.ndarray,
    boost: float,
    angular_velocity: np.ndarray | None = None,
) -> None:
    car.physics.position = _clip_position(position, minimum_z=17.0)
    car.physics.linear_velocity = np.asarray(velocity, dtype=np.float32)
    car.physics.angular_velocity = np.asarray(
        np.zeros(3) if angular_velocity is None else angular_velocity,
        dtype=np.float32,
    )
    car.physics.euler_angles = np.asarray(euler, dtype=np.float32)
    car.boost_amount = float(np.clip(boost, 0.0, 100.0))


class RivalV9PilotCurriculumMutator(StateMutator[GameState]):
    """Sample the authoritative 70/10/8/8/4 scratch reset mixture."""

    def __init__(
        self,
        weights: Mapping[str, float] = V9_PILOT_CURRICULUM_WEIGHTS,
        *,
        seed: int,
    ) -> None:
        if set(weights) != set(V9_PILOT_CURRICULUM_FAMILIES):
            raise ValueError(
                f"Pilot curriculum requires exactly these families: {V9_PILOT_CURRICULUM_FAMILIES}"
            )
        values = np.asarray(
            [float(weights[name]) for name in V9_PILOT_CURRICULUM_FAMILIES],
            dtype=np.float64,
        )
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError("Pilot curriculum weights must be finite and non-negative")
        if not np.isclose(values.sum(), 1.0, atol=1e-12, rtol=0.0):
            raise ValueError("Pilot curriculum weights must sum to one")
        if values[0] <= 0.5:
            raise ValueError("Natural 1v1 must remain the majority reset family")
        self.weights = values
        self._seed = int(seed)
        self._rng = np.random.default_rng(self._seed)

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
        if bool(self._rng.integers(0, 2)):
            return blue, orange, 1.0
        return orange, blue, -1.0

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        index = int(self._rng.choice(len(V9_PILOT_CURRICULUM_FAMILIES), p=self.weights))
        family = V9_PILOT_CURRICULUM_FAMILIES[index]
        getattr(self, f"_apply_{family}")(state)
        shared_info["rival_v9_curriculum_version"] = V9_PILOT_CURRICULUM_VERSION
        shared_info["rival_v9_curriculum_seed"] = self._seed
        shared_info["rival_v9_reset_family"] = family
        shared_info["rival_v9_reset_family_pending_metric"] = True

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

    def _apply_ground_possession_challenge(self, state: GameState) -> None:
        possessor, challenger, attack = self._roles(state)
        ball = np.asarray(
            [
                self._rng.uniform(-2500.0, 2500.0),
                self._rng.uniform(-3000.0, 3000.0),
                BALL_RESTING_HEIGHT,
            ]
        )
        lateral = self._rng.uniform(-350.0, 350.0)
        _set_ball(
            state,
            position=ball,
            velocity=np.asarray(
                [self._rng.uniform(-500.0, 500.0), attack * self._rng.uniform(0.0, 900.0), 0.0]
            ),
            angular_velocity=self._rng.uniform(-3.0, 3.0, 3),
        )
        _set_car(
            possessor,
            position=ball
            + np.asarray(
                [lateral, -attack * self._rng.uniform(250.0, 850.0), 17.0 - BALL_RESTING_HEIGHT]
            ),
            velocity=np.asarray(
                [self._rng.uniform(-500.0, 500.0), attack * self._rng.uniform(300.0, 1500.0), 0.0]
            ),
            euler=np.asarray([0.0, attack * math.pi / 2.0 + self._rng.uniform(-0.45, 0.45), 0.0]),
            boost=self._rng.uniform(10.0, 80.0),
        )
        _set_car(
            challenger,
            position=ball
            + np.asarray(
                [
                    self._rng.uniform(-1100.0, 1100.0),
                    attack * self._rng.uniform(700.0, 1900.0),
                    17.0 - BALL_RESTING_HEIGHT,
                ]
            ),
            velocity=np.asarray(
                [self._rng.uniform(-700.0, 700.0), -attack * self._rng.uniform(300.0, 1500.0), 0.0]
            ),
            euler=np.asarray([0.0, -attack * math.pi / 2.0 + self._rng.uniform(-0.6, 0.6), 0.0]),
            boost=self._rng.uniform(5.0, 100.0),
        )

    def _apply_wall_aerial_ceiling_possession(self, state: GameState) -> None:
        possessor, defender, attack = self._roles(state)
        case = str(self._rng.choice(("sidewall", "aerial", "ceiling"), p=(0.45, 0.40, 0.15)))
        if case == "sidewall":
            side = float(self._rng.choice((-1.0, 1.0)))
            ball = np.asarray(
                [
                    side * self._rng.uniform(3250.0, 3920.0),
                    self._rng.uniform(-3500.0, 3500.0),
                    self._rng.uniform(300.0, 1450.0),
                ]
            )
        elif case == "ceiling":
            ball = np.asarray(
                [
                    self._rng.uniform(-2400.0, 2400.0),
                    self._rng.uniform(-3000.0, 3000.0),
                    self._rng.uniform(1550.0, 1900.0),
                ]
            )
        else:
            ball = np.asarray(
                [
                    self._rng.uniform(-2800.0, 2800.0),
                    self._rng.uniform(-3200.0, 3200.0),
                    self._rng.uniform(450.0, 1700.0),
                ]
            )
        _set_ball(
            state,
            position=ball,
            velocity=np.asarray(
                [
                    self._rng.uniform(-850.0, 850.0),
                    attack * self._rng.uniform(100.0, 1250.0),
                    self._rng.uniform(-250.0, 650.0),
                ]
            ),
            angular_velocity=self._rng.uniform(-4.0, 4.0, 3),
        )
        _set_car(
            possessor,
            position=ball
            + np.asarray(
                [
                    self._rng.uniform(-600.0, 600.0),
                    -attack * self._rng.uniform(250.0, 900.0),
                    self._rng.uniform(-450.0, 200.0),
                ]
            ),
            velocity=state.ball.linear_velocity + self._rng.uniform(-400.0, 400.0, 3),
            euler=self._rng.uniform([-1.1, -math.pi, -1.5], [1.1, math.pi, 1.5]),
            boost=self._rng.uniform(15.0, 100.0),
            angular_velocity=self._rng.uniform(-3.0, 3.0, 3),
        )
        _set_car(
            defender,
            position=ball
            + np.asarray(
                [
                    self._rng.uniform(-1700.0, 1700.0),
                    attack * self._rng.uniform(900.0, 2400.0),
                    17.0 - ball[2],
                ]
            ),
            velocity=np.asarray(
                [self._rng.uniform(-750.0, 750.0), -attack * self._rng.uniform(250.0, 1400.0), 0.0]
            ),
            euler=np.asarray([0.0, -attack * math.pi / 2.0, 0.0]),
            boost=self._rng.uniform(10.0, 100.0),
        )

    def _apply_awkward_recovery_landing(self, state: GameState) -> None:
        recovering, opponent, attack = self._roles(state)
        _set_ball(
            state,
            position=np.asarray(
                [
                    self._rng.uniform(-2800.0, 2800.0),
                    self._rng.uniform(-3200.0, 3200.0),
                    self._rng.uniform(100.0, 850.0),
                ]
            ),
            velocity=self._rng.uniform([-1000.0, -1400.0, -350.0], [1000.0, 1400.0, 650.0]),
            angular_velocity=self._rng.uniform(-4.0, 4.0, 3),
        )
        near_sidewall = bool(self._rng.random() < 0.45)
        recovery_x = (
            float(self._rng.choice((-1.0, 1.0))) * self._rng.uniform(3300.0, 3920.0)
            if near_sidewall
            else self._rng.uniform(-3000.0, 3000.0)
        )
        _set_car(
            recovering,
            position=np.asarray(
                [recovery_x, self._rng.uniform(-3600.0, 3600.0), self._rng.uniform(180.0, 1500.0)]
            ),
            velocity=self._rng.uniform([-1750.0, -1750.0, -850.0], [1750.0, 1750.0, 850.0]),
            euler=self._rng.uniform([-math.pi, -math.pi, -math.pi], [math.pi, math.pi, math.pi]),
            boost=self._rng.uniform(0.0, 60.0),
            angular_velocity=self._rng.uniform(-5.0, 5.0, 3),
        )
        _set_car(
            opponent,
            position=state.ball.position
            + np.asarray(
                [
                    self._rng.uniform(-1000.0, 1000.0),
                    -attack * self._rng.uniform(350.0, 1500.0),
                    17.0 - state.ball.position[2],
                ]
            ),
            velocity=np.asarray(
                [self._rng.uniform(-700.0, 700.0), attack * self._rng.uniform(350.0, 1600.0), 0.0]
            ),
            euler=np.asarray([0.0, attack * math.pi / 2.0, 0.0]),
            boost=self._rng.uniform(15.0, 100.0),
        )

    def _apply_low_resource(self, state: GameState) -> None:
        attacker, defender, attack = self._roles(state)
        airborne = bool(self._rng.random() < 0.35)
        ball_z = self._rng.uniform(250.0, 1050.0) if airborne else BALL_RESTING_HEIGHT
        ball = np.asarray(
            [self._rng.uniform(-2600.0, 2600.0), self._rng.uniform(-3000.0, 3000.0), ball_z]
        )
        _set_ball(
            state,
            position=ball,
            velocity=np.asarray(
                [
                    self._rng.uniform(-650.0, 650.0),
                    attack * self._rng.uniform(100.0, 1000.0),
                    self._rng.uniform(-150.0, 450.0) if airborne else 0.0,
                ]
            ),
            angular_velocity=self._rng.uniform(-3.0, 3.0, 3),
        )
        _set_car(
            attacker,
            position=ball
            + np.asarray(
                [
                    self._rng.uniform(-650.0, 650.0),
                    -attack * self._rng.uniform(350.0, 1100.0),
                    self._rng.uniform(-300.0, 100.0) if airborne else 17.0 - ball_z,
                ]
            ),
            velocity=np.asarray(
                [
                    self._rng.uniform(-650.0, 650.0),
                    attack * self._rng.uniform(250.0, 1300.0),
                    self._rng.uniform(-250.0, 350.0) if airborne else 0.0,
                ]
            ),
            euler=np.asarray(
                [
                    self._rng.uniform(-0.7, 0.5),
                    attack * math.pi / 2.0 + self._rng.uniform(-0.5, 0.5),
                    self._rng.uniform(-0.6, 0.6),
                ]
            ),
            boost=self._rng.uniform(0.0, 12.0),
        )
        _set_car(
            defender,
            position=ball
            + np.asarray(
                [
                    self._rng.uniform(-1400.0, 1400.0),
                    attack * self._rng.uniform(900.0, 2200.0),
                    17.0 - ball_z,
                ]
            ),
            velocity=np.asarray(
                [self._rng.uniform(-650.0, 650.0), -attack * self._rng.uniform(200.0, 1200.0), 0.0]
            ),
            euler=np.asarray([0.0, -attack * math.pi / 2.0, 0.0]),
            boost=self._rng.uniform(0.0, 18.0),
        )


def curriculum_distribution_report(
    mutator: RivalV9PilotCurriculumMutator,
    state_factory,
    team_size_mutator,
    *,
    samples: int,
) -> dict[str, Any]:
    """Exercise reset sampling without stepping policy or reward code."""

    counts: Counter[str] = Counter()
    shared_info: dict[str, Any] = {}
    for _ in range(int(samples)):
        state = state_factory()
        team_size_mutator.apply(state, shared_info)
        mutator.apply(state, shared_info)
        family = str(shared_info["rival_v9_reset_family"])
        counts[family] += 1
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
                )
            )
        if not all(np.isfinite(vector).all() for vector in vectors):
            raise FloatingPointError("Pilot curriculum produced non-finite physics")
    return {
        "version": V9_PILOT_CURRICULUM_VERSION,
        "samples": int(samples),
        "counts": {name: int(counts[name]) for name in V9_PILOT_CURRICULUM_FAMILIES},
        "shares": {
            name: float(counts[name] / max(int(samples), 1))
            for name in V9_PILOT_CURRICULUM_FAMILIES
        },
        "configured_weights": {
            name: float(mutator.weights[index])
            for index, name in enumerate(V9_PILOT_CURRICULUM_FAMILIES)
        },
    }
