"""A tested 432-value RLGym observation adapter for the frozen Wisp teacher.

The feature count and ordering follow ``bot/obs_builder.py``. RocketSim supplies
the four irregular-horizon ball predictions directly. A few live-only signals
(the exact arena SDF landing normal, RLBot scoreboard, and Wisp's cached ETA)
use documented deterministic equivalents so training does not depend on RLBot.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import RocketSim as rs
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import (
    BACK_WALL_Y,
    BALL_RADIUS,
    BOOST_LOCATIONS,
    CEILING_Z,
    GOAL_CENTER_TO_POST,
    GOAL_HEIGHT,
    SIDE_WALL_X,
)


OBSERVATION_VERSION = "WispCompatible432RLGymV1"
OBSERVATION_SIZE = 432
PLAYER_OBSERVATION_SIZE = 51
POS_COEF = 1.0 / 5000.0
VEL_COEF = 1.0 / 2300.0
ANG_VEL_COEF = 1.0 / 3.0
PREDICTION_TICKS = (22, 66, 198, 594)
PREDICTION_STEP_TICKS = 22
MAX_PLAYERS_PER_TEAM = 3
DEMO_RESPAWN_TIME = 3.0
CAR_MAX_SPEED = 2300.0

BLUE_GOAL_CENTER = np.array([0.0, -BACK_WALL_Y, GOAL_HEIGHT / 2], dtype=np.float32)
ORANGE_GOAL_CENTER = np.array([0.0, BACK_WALL_Y, GOAL_HEIGHT / 2], dtype=np.float32)

# RLBot/Wisp and RLGym enumerate a few symmetric boost pads differently. Preserve
# the table consumed by Wisp, then map RocketSim timers by physical coordinates.
WISP_BOOST_LOCATIONS = np.asarray(
    [
        (0.0, -4240.0, 70.0),
        (-1792.0, -4184.0, 70.0),
        (1792.0, -4184.0, 70.0),
        (-3072.0, -4096.0, 73.0),
        (3072.0, -4096.0, 73.0),
        (-940.0, -3308.0, 70.0),
        (940.0, -3308.0, 70.0),
        (0.0, -2816.0, 70.0),
        (-3584.0, -2484.0, 70.0),
        (3584.0, -2484.0, 70.0),
        (-1788.0, -2300.0, 70.0),
        (1788.0, -2300.0, 70.0),
        (-2048.0, -1036.0, 70.0),
        (2048.0, -1036.0, 70.0),
        (0.0, -1024.0, 70.0),
        (-3584.0, 0.0, 73.0),
        (-1024.0, 0.0, 70.0),
        (1024.0, 0.0, 70.0),
        (3584.0, 0.0, 73.0),
        (0.0, 1024.0, 70.0),
        (-2048.0, 1036.0, 70.0),
        (2048.0, 1036.0, 70.0),
        (-1788.0, 2300.0, 70.0),
        (1788.0, 2300.0, 70.0),
        (-3584.0, 2484.0, 70.0),
        (3584.0, 2484.0, 70.0),
        (0.0, 2816.0, 70.0),
        (-940.0, 3310.0, 70.0),
        (940.0, 3308.0, 70.0),
        (-3072.0, 4096.0, 73.0),
        (3072.0, 4096.0, 73.0),
        (-1792.0, 4184.0, 70.0),
        (1792.0, 4184.0, 70.0),
        (0.0, 4240.0, 70.0),
    ],
    dtype=np.float32,
)
RLGYM_BOOST_LOCATIONS = np.asarray(BOOST_LOCATIONS, dtype=np.float32)


def _coordinate_mapping(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    mapping = []
    for position in target:
        distances = np.sum((source - position) ** 2, axis=1)
        index = int(np.argmin(distances))
        if float(distances[index]) > 1e-3:
            raise RuntimeError(f"Unable to map boost location {position.tolist()}")
        mapping.append(index)
    if len(set(mapping)) != len(mapping):
        raise RuntimeError("Boost pad coordinate mapping is not one-to-one")
    return np.asarray(mapping, dtype=np.int64)


RLGYM_TO_WISP_PAD_INDICES = _coordinate_mapping(
    RLGYM_BOOST_LOCATIONS, WISP_BOOST_LOCATIONS
)
WISP_MIRRORED_PAD_INDICES = np.asarray(
    [
        item[0]
        for item in sorted(
            enumerate(WISP_BOOST_LOCATIONS),
            key=lambda item: float(item[1][1] * 10000 - item[1][0]),
        )
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class CanonicalPhysics:
    position: np.ndarray
    rotation_mtx: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    @property
    def forward(self) -> np.ndarray:
        return self.rotation_mtx[:, 0]

    @property
    def right(self) -> np.ndarray:
        return self.rotation_mtx[:, 1]

    @property
    def up(self) -> np.ndarray:
        return self.rotation_mtx[:, 2]

    def local(self, vector: np.ndarray) -> np.ndarray:
        return self.rotation_mtx.T @ vector


def _canonical_physics(
    physics: PhysicsObject,
    invert: bool,
    mirror_x: bool,
) -> CanonicalPhysics:
    position = np.asarray(physics.position, dtype=np.float32).copy()
    rotation = np.asarray(physics.rotation_mtx, dtype=np.float32).copy()
    velocity = np.asarray(physics.linear_velocity, dtype=np.float32).copy()
    angular = np.asarray(physics.angular_velocity, dtype=np.float32).copy()
    if invert:
        scale = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
        position *= scale
        rotation *= scale[:, None]
        velocity *= scale
        angular *= scale
    if mirror_x:
        position[0] *= -1
        velocity[0] *= -1
        rotation[:, 0] *= np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        rotation[:, 1] *= np.array([1.0, -1.0, -1.0], dtype=np.float32)
        rotation[:, 2] *= np.array([-1.0, 1.0, 1.0], dtype=np.float32)
        angular *= np.array([1.0, -1.0, -1.0], dtype=np.float32)
    return CanonicalPhysics(position, rotation, velocity, angular)


def _canonical_vector(vector: np.ndarray, invert: bool, mirror_x: bool) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32).copy()
    if invert:
        result *= np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    if mirror_x:
        result[0] *= -1
    return result


def _normalized(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-7)


def _corner_wall_distance(x: float, y: float) -> float:
    p = np.array([abs(x), abs(y)], dtype=np.float32)
    start = np.array([SIDE_WALL_X - 1152, BACK_WALL_Y], dtype=np.float32)
    end = np.array([SIDE_WALL_X, BACK_WALL_Y - 1152], dtype=np.float32)
    segment = end - start
    parameter = float(np.dot(p - start, segment) / np.dot(segment, segment))
    closest = start + np.clip(parameter, 0.0, 1.0) * segment
    return float(np.linalg.norm(p - closest))


def _closest_wall_distance(position: np.ndarray) -> float:
    x, y = float(position[0]), float(position[1])
    return min(
        SIDE_WALL_X - abs(x),
        BACK_WALL_Y - abs(y),
        _corner_wall_distance(x, y),
    )


def _is_goal_post_between(first: np.ndarray, second: np.ndarray) -> bool:
    # Robust segment approximation of Wisp's post/crossbar intersection feature.
    for goal_y in (-BACK_WALL_Y, BACK_WALL_Y):
        dy = float(second[1] - first[1])
        if abs(dy) < 1e-7:
            continue
        t = (goal_y - float(first[1])) / dy
        if 0.0 <= t <= 1.0:
            crossing = first + t * (second - first)
            near_post = abs(abs(float(crossing[0])) - GOAL_CENTER_TO_POST) < BALL_RADIUS
            near_bar = abs(float(crossing[2]) - GOAL_HEIGHT) < BALL_RADIUS
            if near_post or near_bar:
                return True
    return False


def _turn_radius(speed: float) -> float:
    speed = abs(speed)
    if speed == 0:
        return 0.0
    if speed < 500:
        curvature = 0.006900 - 5.84e-6 * speed
    elif speed < 1000:
        curvature = 0.005610 - 3.26e-6 * speed
    elif speed < 1500:
        curvature = 0.004300 - 1.95e-6 * speed
    elif speed < 1750:
        curvature = 0.003025 - 1.1e-6 * speed
    else:
        curvature = 0.001800 - 4e-7 * min(speed, 2499.0)
    return 1.0 / max(curvature, 1e-7)


def _landing_normal(physics: CanonicalPhysics) -> np.ndarray:
    position = physics.position.copy()
    velocity = physics.linear_velocity.copy()
    dt = 0.25
    for _ in range(40):
        velocity[2] -= 650.0 * dt
        position += velocity * dt
        distances = (
            position[2] - 17.0,
            SIDE_WALL_X - abs(float(position[0])),
            BACK_WALL_Y - abs(float(position[1])),
            CEILING_Z - float(position[2]),
        )
        surface = int(np.argmin(distances))
        if distances[surface] <= 0:
            if surface == 0:
                return np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if surface == 1:
                return np.array([-math.copysign(1.0, position[0]), 0.0, 0.0], dtype=np.float32)
            if surface == 2:
                return np.array([0.0, -math.copysign(1.0, position[1]), 0.0], dtype=np.float32)
            return np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return np.array([0.0, 0.0, 1.0], dtype=np.float32)


def _rough_eta(car: Car, ball_position: np.ndarray) -> float:
    if car.is_demoed:
        return 10.0
    delta = ball_position - np.asarray(car.physics.position, dtype=np.float32)
    distance = max(0.0, float(np.linalg.norm(delta)) - 1.5 * BALL_RADIUS)
    direction = _normalized(delta)
    closing_speed = max(0.0, float(np.dot(car.physics.linear_velocity, direction)))
    boost_assist = 650.0 * min(max(float(car.boost_amount) / 100.0, 0.0), 1.0)
    effective_speed = min(CAR_MAX_SPEED, max(400.0, closing_speed + boost_assist))
    return min(10.0, distance / effective_speed)


class WispCompatibleObs(ObsBuilder[AgentID, np.ndarray, GameState, tuple[str, int]]):
    """Build Wisp-ordered observations from natural RocketSim states."""

    def __init__(self, seed: int = 20260822) -> None:
        self._rng = np.random.default_rng(seed)
        self._predictor = rs.BallPredictor(rs.GameMode.SOCCAR)
        self._prediction_tick: int | None = None
        self._prediction: list[tuple[np.ndarray, np.ndarray]] = []

    def get_obs_space(self, agent: AgentID) -> tuple[str, int]:
        return "real", OBSERVATION_SIZE

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        self._prediction_tick = None
        self._prediction = []

    def _ball_prediction(self, state: GameState) -> list[tuple[np.ndarray, np.ndarray]]:
        if self._prediction_tick == state.tick_count:
            return self._prediction
        ball = state.ball
        ball_state = rs.BallState(
            pos=rs.Vec(*ball.position),
            rot_mat=rs.RotMat(*ball.rotation_mtx.transpose().flatten()),
            vel=rs.Vec(*ball.linear_velocity),
            ang_vel=rs.Vec(*ball.angular_velocity),
        )
        steps = max(PREDICTION_TICKS) // PREDICTION_STEP_TICKS + 1
        prediction = self._predictor.get_ball_prediction(
            ball_state,
            0,
            steps,
            PREDICTION_STEP_TICKS,
        )
        selected = []
        for ticks in PREDICTION_TICKS:
            item = prediction[ticks // PREDICTION_STEP_TICKS]
            selected.append(
                (
                    np.array([item.pos.x, item.pos.y, item.pos.z], dtype=np.float32),
                    np.array([item.vel.x, item.vel.y, item.vel.z], dtype=np.float32),
                )
            )
        self._prediction_tick = state.tick_count
        self._prediction = selected
        return selected

    def build_obs(
        self,
        agents: list[AgentID],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, np.ndarray]:
        prediction = self._ball_prediction(state)
        return {
            agent: self._build_one(agent, state, shared_info, prediction)
            for agent in agents
        }

    def _player_features(
        self,
        car: Car,
        invert: bool,
        mirror_x: bool,
        ball: CanonicalPhysics,
    ) -> np.ndarray:
        physics = _canonical_physics(car.physics, invert, mirror_x)
        dodge_forward = _normalized(physics.forward * np.array([1.0, 1.0, 0.0]))
        dodge_right = np.array(
            [
                dodge_forward[1] if mirror_x else -dodge_forward[1],
                -dodge_forward[0] if mirror_x else dodge_forward[0],
                0.0,
            ],
            dtype=np.float32,
        )

        relative_position = ball.position - physics.position
        relative_velocity = ball.linear_velocity - physics.linear_velocity
        own_shot = _normalized(BLUE_GOAL_CENTER - ball.position)
        opponent_shot = _normalized(ORANGE_GOAL_CENTER - ball.position)
        has_flip_or_jump = car.on_ground or (
            not car.has_flipped
            and not car.has_double_jumped
            and float(car.air_time_since_jump) < 1.25
        )
        has_flip_reset = not car.on_ground and has_flip_or_jump and not car.has_jumped

        features: list[float] = []
        features.extend(physics.position * POS_COEF)
        features.extend(physics.forward)
        features.extend(physics.up)
        features.extend(physics.linear_velocity * VEL_COEF)
        features.extend(physics.angular_velocity * ANG_VEL_COEF)
        features.extend(physics.local(physics.angular_velocity) * ANG_VEL_COEF)
        features.append(float(np.dot(physics.forward, physics.linear_velocity) * VEL_COEF))
        features.extend(physics.local(relative_position) * POS_COEF)
        features.extend(physics.local(relative_velocity) * VEL_COEF)
        features.extend(
            [
                float(np.dot(dodge_forward, relative_position) * POS_COEF),
                float(np.dot(dodge_right, relative_position) * POS_COEF),
                float(np.dot(dodge_forward, relative_velocity) * VEL_COEF),
                float(np.dot(dodge_right, relative_velocity) * VEL_COEF),
            ]
        )
        features.extend(physics.local(own_shot) * POS_COEF)
        features.extend(physics.local(opponent_shot) * POS_COEF)
        features.extend(
            [
                float(np.dot(dodge_forward, own_shot) * POS_COEF),
                float(np.dot(dodge_right, own_shot) * POS_COEF),
                float(np.dot(dodge_forward, opponent_shot) * POS_COEF),
                float(np.dot(dodge_right, opponent_shot) * POS_COEF),
            ]
        )
        features.extend(
            [
                float(car.boost_amount) / 100.0,
                float(car.on_ground),
                float(has_flip_or_jump),
                float(car.is_demoed),
                float(car.is_jumping),
                float(has_flip_reset),
                float(abs(float(physics.position[1])) > BACK_WALL_Y - 10),
                float(_is_goal_post_between(physics.position, ball.position)),
                _closest_wall_distance(physics.position) * POS_COEF,
                _rough_eta(car, np.asarray(ball.position)),
            ]
        )

        if car.is_demoed:
            features = [0.0] * len(features)
        features.extend(
            [
                float(car.is_demoed),
                float(car.demo_respawn_timer) / DEMO_RESPAWN_TIME,
            ]
        )
        result = np.asarray(features, dtype=np.float32)
        if result.shape != (PLAYER_OBSERVATION_SIZE,):
            raise AssertionError(f"Unexpected per-player observation shape {result.shape}")
        return result

    def _build_one(
        self,
        agent: AgentID,
        state: GameState,
        shared_info: dict[str, Any],
        prediction: list[tuple[np.ndarray, np.ndarray]],
    ) -> np.ndarray:
        car = state.cars[agent]
        invert = car.team_num == 1
        mirror_x = invert != (float(car.physics.position[0]) < 0)
        physics = _canonical_physics(car.physics, invert, mirror_x)
        ball = _canonical_physics(state.ball, invert, mirror_x)

        features: list[float] = []
        features.extend(ball.position * POS_COEF)
        features.extend(ball.linear_velocity * VEL_COEF)
        features.extend(ball.angular_velocity * ANG_VEL_COEF)
        my_goal = ORANGE_GOAL_CENTER if invert else BLUE_GOAL_CENTER
        opponent_goal = BLUE_GOAL_CENTER if invert else ORANGE_GOAL_CENTER
        features.extend((my_goal - ball.position) * POS_COEF)
        features.extend((opponent_goal - ball.position) * POS_COEF)
        features.append(float(not np.any(ball.position[:2])))

        for predicted_position, predicted_velocity in prediction:
            position = _canonical_vector(predicted_position, invert, mirror_x)
            velocity = _canonical_vector(predicted_velocity, invert, mirror_x)
            features.extend(position * POS_COEF)
            features.extend(velocity * VEL_COEF)
            features.extend(physics.local(position - physics.position) * POS_COEF)
            features.extend(physics.local(velocity - physics.linear_velocity) * VEL_COEF)

        timers = np.asarray(state.boost_pad_timers, dtype=np.float32)[
            RLGYM_TO_WISP_PAD_INDICES
        ]
        if invert:
            timers = timers[::-1]
        if mirror_x:
            timers = timers[WISP_MIRRORED_PAD_INDICES]
        for timer in timers:
            features.append(1.0 if timer <= 0 else 1.0 / (1.0 + float(timer)))

        # Match Wisp's original-world close-pad computation and left/right mirror.
        world_position = np.asarray(car.physics.position, dtype=np.float32)
        world_forward = np.asarray(car.physics.forward, dtype=np.float32)
        soon_position = world_position + world_forward * 420.0
        distances = np.sum((WISP_BOOST_LOCATIONS - soon_position) ** 2, axis=1)
        for pad_index in np.argsort(distances)[:5]:
            relative = car.physics.rotation_mtx.T @ (
                WISP_BOOST_LOCATIONS[pad_index] - world_position
            )
            features.append(float(relative[0] * POS_COEF))
            lateral = -relative[1] if mirror_x else relative[1]
            features.append(float(lateral * POS_COEF))

        previous = np.asarray(
            shared_info.get("previous_actions", {}).get(agent, np.zeros(8)),
            dtype=np.float32,
        ).copy()
        if mirror_x:
            previous[[1, 3, 4]] *= -1
        features.extend(previous)

        position = np.asarray(car.physics.position, dtype=np.float32)
        features.extend(
            [
                (BACK_WALL_Y - abs(float(position[1]))) * POS_COEF,
                (SIDE_WALL_X - abs(float(position[0]))) * POS_COEF,
                _corner_wall_distance(float(position[0]), float(position[1])) * POS_COEF,
            ]
        )
        features.extend(physics.local(_landing_normal(physics)))
        features.append(float(np.clip(shared_info.get("score_diff", 0), -1, 1)))
        forward_speed = float(np.dot(car.physics.linear_velocity, car.physics.forward))
        features.extend(
            [
                _turn_radius(min(abs(forward_speed), CAR_MAX_SPEED)) / 1300.0,
                float(car.ball_touches > 0),
                float(car.handbrake),
            ]
        )

        self_features = self._player_features(car, invert, mirror_x, ball)
        features.extend(self_features)
        teammates: list[np.ndarray] = []
        opponents: list[np.ndarray] = []
        for other_agent, other_car in state.cars.items():
            if other_agent == agent:
                continue
            target = teammates if other_car.team_num == car.team_num else opponents
            target.append(self._player_features(other_car, invert, mirror_x, ball))
        while len(teammates) < MAX_PLAYERS_PER_TEAM - 1:
            teammates.append(np.zeros(PLAYER_OBSERVATION_SIZE, dtype=np.float32))
        while len(opponents) < MAX_PLAYERS_PER_TEAM:
            opponents.append(np.zeros(PLAYER_OBSERVATION_SIZE, dtype=np.float32))
        self._rng.shuffle(teammates)
        self._rng.shuffle(opponents)
        for item in teammates[: MAX_PLAYERS_PER_TEAM - 1]:
            features.extend(item)
        for item in opponents[:MAX_PLAYERS_PER_TEAM]:
            features.extend(item)

        observation = np.asarray(features, dtype=np.float32)
        if observation.shape != (OBSERVATION_SIZE,):
            raise AssertionError(f"Unexpected Wisp observation shape {observation.shape}")
        if not np.isfinite(observation).all():
            bad = np.flatnonzero(~np.isfinite(observation))
            raise FloatingPointError(f"Non-finite Wisp observation indices: {bad[:10]}")
        return observation


def observation_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "observation_version": OBSERVATION_VERSION,
        "shape": [OBSERVATION_SIZE],
        "player_feature_size": PLAYER_OBSERVATION_SIZE,
        "prediction_ticks": list(PREDICTION_TICKS),
        "normalization": {
            "position": POS_COEF,
            "linear_velocity": VEL_COEF,
            "angular_velocity": ANG_VEL_COEF,
        },
        "live_training_differences": [
            "RocketSim BallPredictor replaces RLBot ball-prediction flatbuffer slices at the same tick horizons.",
            "A deterministic box-surface landing normal approximates Wisp's live arena-SDF query.",
            "A bounded deterministic kinematic ETA replaces Wisp's process-global cached rough_eta.",
            "The episodic training scoreboard is zero unless a curriculum wrapper supplies score_diff.",
            "RLGym car state supplies previous actions selected by the training action parser.",
        ],
    }
