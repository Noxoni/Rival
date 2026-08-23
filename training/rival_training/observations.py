"""Versioned 432-value strategic observation adapter for frozen Wisp.

The feature count and ordering follow ``bot/obs_builder.py``. RocketSim supplies
the prediction slices while the stateful ETA and controller/event fields target
the production Wisp semantics explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
import RocketSim as rs
from rlgym.api import AgentID, ObsBuilder
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import (
    BACK_WALL_Y,
    BOOST_LOCATIONS,
    CEILING_Z,
    GOAL_CENTER_TO_POST,
    GOAL_HEIGHT,
    SIDE_WALL_X,
)

from .wisp_contract import CONTRACT_VERSION, WispEtaState


def _ensure_rocketsim_meshes() -> None:
    """Initialize RocketSim before constructing a standalone BallPredictor."""
    import rlgym.rocket_league.sim.rocketsim_engine as engine_module

    mesh_path = Path(engine_module.__file__).resolve().parent / "collision_meshes"
    try:
        rs.init(str(mesh_path))
    except Exception:
        # RocketSim initialization is process-global and safely already complete
        # when a transition engine was constructed first.
        pass


OBSERVATION_VERSION = CONTRACT_VERSION
OBSERVATION_SIZE = 432
PLAYER_OBSERVATION_SIZE = 51
POS_COEF = 1.0 / 5000.0
VEL_COEF = 1.0 / 2300.0
ANG_VEL_COEF = 1.0 / 3.0
PREDICTION_TICKS = (22, 66, 198, 594)
PREDICTION_STEP_TICKS = 1
MAX_PLAYERS_PER_TEAM = 3
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
    """Literal vector form of the frozen ``utils.is_goal_post_between``."""
    first_not_goal = abs(float(first[1])) < BACK_WALL_Y
    if first_not_goal and abs(float(second[1])) < BACK_WALL_Y:
        return False
    inside = np.asarray(second if first_not_goal else first, dtype=np.float64).copy()
    other = np.asarray(first if first_not_goal else second, dtype=np.float64).copy()
    if inside[1] < 0:
        inside[1] *= -1
        other[1] *= -1
    with np.errstate(divide="ignore", invalid="ignore"):
        goal_x = inside[0] + GOAL_CENTER_TO_POST
        goal_y = inside[1] - BACK_WALL_Y
        ball_x = other[0] + GOAL_CENTER_TO_POST
        ball_y = other[1] - BACK_WALL_Y
        if ball_x < 0 and goal_y / goal_x >= ball_y / ball_x:
            return True
        goal_x = inside[0] - GOAL_CENTER_TO_POST
        ball_x = other[0] - GOAL_CENTER_TO_POST
        if ball_x > 0 and goal_y / goal_x <= ball_y / ball_x:
            return True
        goal_z = inside[2] - GOAL_HEIGHT
        ball_z = other[2] - GOAL_HEIGHT
        if ball_z > 0 and goal_z / goal_y >= ball_z / ball_y:
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


def _sdf_wall_distance(point: np.ndarray) -> float:
    """Numpy port of the frozen production ``arena_sdf.sdf_wall_dist``."""
    roundness = 280.0
    inverse_sqrt_two = 1.0 / math.sqrt(2.0)
    center = np.array([0.0, 0.0, CEILING_Z / 2], dtype=np.float64)
    semi_size = np.array(
        [SIDE_WALL_X, BACK_WALL_Y, CEILING_Z / 2], dtype=np.float64
    )
    rotation_45 = np.array(
        [
            [inverse_sqrt_two, -inverse_sqrt_two, 0.0],
            [inverse_sqrt_two, inverse_sqrt_two, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    corner_size = np.array(
        [
            inverse_sqrt_two * 8064.0,
            inverse_sqrt_two * 8064.0,
            CEILING_Z / 2,
        ],
        dtype=np.float64,
    )
    base_q = np.abs(point - center) - semi_size + roundness
    base_distance = float(np.linalg.norm(np.maximum(base_q, 0.0))) + min(
        float(np.max(base_q)), 0.0
    )
    corner_q = np.abs(rotation_45.T @ point - center) - corner_size + roundness
    corner_distance = float(np.linalg.norm(np.maximum(corner_q, 0.0))) + min(
        float(np.max(corner_q)), 0.0
    )
    base_corner_distance = max(base_distance, corner_distance) - roundness
    goal_center = np.array([0.0, 0.0, GOAL_HEIGHT / 2], dtype=np.float64)
    goal_size = np.array(
        [GOAL_CENTER_TO_POST, 6000.0, GOAL_HEIGHT / 2], dtype=np.float64
    )
    goal_q = np.abs(point - goal_center) - goal_size + roundness
    goal_distance = float(np.linalg.norm(np.maximum(goal_q, 0.0))) + min(
        float(np.max(goal_q)), 0.0
    )
    return -min(base_corner_distance, goal_distance)


def _sdf_normal(point: np.ndarray) -> np.ndarray:
    delta = 0.0004
    offsets = np.eye(3, dtype=np.float64) * delta
    gradient = np.asarray(
        [
            _sdf_wall_distance(point + offset)
            - _sdf_wall_distance(point - offset)
            for offset in offsets
        ],
        dtype=np.float64,
    )
    length = float(np.linalg.norm(gradient))
    if length <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return np.asarray(gradient / length, dtype=np.float32)


def _landing_normal(position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
    """Port ``utils.normal_at_landing`` using the same arena SDF and stepping."""
    current_position = np.asarray(position, dtype=np.float64).copy()
    current_velocity = np.asarray(velocity, dtype=np.float64).copy()
    time_step = 0.25
    remaining = 10.0
    while _sdf_wall_distance(current_position) > 0.0 and remaining > 0.0:
        current_velocity[2] -= 650.0 * time_step
        current_position += current_velocity * time_step
        remaining -= time_step
    current_position -= 0.5 * current_velocity * time_step
    return _sdf_normal(current_position)


class WispCompatibleObs(ObsBuilder[AgentID, np.ndarray, GameState, tuple[str, int]]):
    """Build ``Wisp432ContractV2`` observations from RocketSim states."""

    def __init__(self, seed: int = 20260822) -> None:
        self._rng = np.random.default_rng(seed)
        _ensure_rocketsim_meshes()
        self._predictor = rs.BallPredictor(rs.GameMode.SOCCAR)
        self._prediction_tick: int | None = None
        self._prediction: list[tuple[np.ndarray, np.ndarray]] = []
        self._eta_state = WispEtaState()

    def seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

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

    def reset_process_session(self) -> None:
        """Clear process-lifetime temporal state for an explicit new bot session.

        Normal RLGym episode resets intentionally do not clear ETA: production
        retains ``rough_eta.cache`` across goals and timer rewinds in the same bot
        process.  A new worker/bot process starts with the constructor's empty
        cache; tests and offline tools can request that boundary explicitly here.
        """
        self._eta_state.reset()

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
        # RocketSim prediction[0] is the unadvanced source state; RLBot slice 0
        # corresponds to RocketSim prediction[1].  Keep one extra state so live
        # slice 599 maps to prediction[600].
        steps = 601
        prediction = self._predictor.get_ball_prediction(
            ball_state,
            0,
            steps,
            PREDICTION_STEP_TICKS,
        )
        selected = []
        for item in prediction:
            selected.append(
                (
                    np.array([item.pos.x, item.pos.y, item.pos.z], dtype=np.float32),
                    np.array([item.vel.x, item.vel.y, item.vel.z], dtype=np.float32),
                )
            )
        self._prediction_tick = state.tick_count
        self._prediction = selected
        return selected

    @staticmethod
    def _temporal_value(
        shared_info: dict[str, Any],
        field: str,
        agent: AgentID,
        default: float,
    ) -> float:
        return float(shared_info.get(field, {}).get(agent, default))

    def _eta(
        self,
        observer: AgentID,
        player: AgentID,
        car: Car,
        prediction: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        if car.is_demoed:
            return 10.0
        return self._eta_state.value(
            observer,
            player,
            car.physics.position,
            car.physics.linear_velocity,
            car.boost_amount,
            lambda tick: prediction[tick + 1][0],
        )

    def _advance_post_observation_eta(
        self,
        observer: AgentID,
        state: GameState,
        prediction: list[tuple[np.ndarray, np.ndarray]],
    ) -> None:
        """Mirror the production tactical-metrics ETA calls after inference.

        ``bot.bot.Rival.update_action`` calls ``rough_eta`` once while building
        each 1v1 player block and then once more for the controlled car and its
        closest opponent.  The second call only affects the next decision's
        cache, but omitting it was a material domain shift in M07.
        """
        controlled = state.cars[observer]
        self._eta(observer, observer, controlled, prediction)
        opponents = [
            (agent, car)
            for agent, car in state.cars.items()
            if car.team_num != controlled.team_num
        ]
        if opponents:
            opponent, opponent_car = min(
                opponents,
                key=lambda item: float(
                    np.sum(
                        (
                            np.asarray(item[1].physics.position)
                            - np.asarray(state.ball.position)
                        )
                        ** 2
                    )
                ),
            )
            self._eta(observer, opponent, opponent_car, prediction)

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
        observer: AgentID,
        player: AgentID,
        invert: bool,
        mirror_x: bool,
        ball: CanonicalPhysics,
        prediction: list[tuple[np.ndarray, np.ndarray]],
        shared_info: dict[str, Any],
    ) -> np.ndarray:
        physics = _canonical_physics(car.physics, invert, mirror_x)
        dodge_forward = _normalized(physics.forward * np.array([1.0, 1.0, 0.0]))
        # Preserve a frozen Wisp implementation quirk.  ``RotMat.right`` returns
        # a copy, so the component assignments in ``dodge_relative_rot_mat`` do
        # not mutate the identity row; its right axis is therefore always world
        # +Y.  Replacing this with the intended perpendicular axis was one of the
        # previously unrecognized V1 domain shifts.
        dodge_right = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        relative_position = ball.position - physics.position
        relative_velocity = ball.linear_velocity - physics.linear_velocity
        own_shot = _normalized(BLUE_GOAL_CENTER - ball.position)
        opponent_shot = _normalized(ORANGE_GOAL_CENTER - ball.position)
        flag_overrides = shared_info.get("wisp_player_flags", {}).get(player, {})
        on_ground = bool(flag_overrides.get("on_ground", car.on_ground))
        has_flip_or_jump = bool(flag_overrides.get("has_flip_or_jump", on_ground or (
            not car.has_flipped
            and not car.has_double_jumped
            and float(car.air_time_since_jump) < 1.25
        )))
        is_jumping = bool(flag_overrides.get("is_jumping", car.is_jumping))
        has_flip_reset = bool(flag_overrides.get(
            "has_flip_reset", not on_ground and has_flip_or_jump and not car.has_jumped
        ))

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
                float(on_ground),
                float(has_flip_or_jump),
                float(car.is_demoed),
                float(is_jumping),
                float(has_flip_reset),
                float(abs(float(physics.position[1])) > BACK_WALL_Y - 10),
                float(
                    _is_goal_post_between(
                        np.asarray(car.physics.position), ball.position
                    )
                ),
                _closest_wall_distance(physics.position) * POS_COEF,
                self._eta(observer, player, car, prediction),
            ]
        )

        if car.is_demoed:
            features = [0.0] * len(features)
        features.extend(
            [
                float(car.is_demoed),
                # The production RocketSimStateAdapter intentionally remains
                # frozen; it sets ``is_demoed`` but leaves Player's initialized
                # ``demo_respawn_timer`` at zero.
                0.0,
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

        for prediction_tick in PREDICTION_TICKS:
            predicted_position, predicted_velocity = prediction[prediction_tick + 1]
            position = _canonical_vector(predicted_position, invert, mirror_x)
            velocity = _canonical_vector(predicted_velocity, invert, mirror_x)
            features.extend(position * POS_COEF)
            features.extend(velocity * VEL_COEF)
            features.extend(physics.local(position - physics.position) * POS_COEF)
            features.extend(physics.local(velocity - physics.linear_velocity) * VEL_COEF)

        timers = np.asarray(state.boost_pad_timers, dtype=np.float32)[
            RLGYM_TO_WISP_PAD_INDICES
        ]
        active_override = shared_info.get("wisp_boost_pad_active")
        if active_override is None:
            active = timers <= 0
        else:
            active = np.asarray(active_override, dtype=bool).copy()
        if invert:
            timers = timers[::-1]
            active = active[::-1]
        if mirror_x:
            timers = timers[WISP_MIRRORED_PAD_INDICES]
            active = active[WISP_MIRRORED_PAD_INDICES]
        for is_active, timer in zip(active, timers, strict=True):
            features.append(1.0 if is_active else 1.0 / (1.0 + float(timer)))

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
        features.extend(
            physics.local(
                _landing_normal(car.physics.position, car.physics.linear_velocity)
            )
        )
        features.append(float(np.clip(shared_info.get("score_diff", 0), -1, 1)))
        # Preserve the frozen builder's mixed-frame expression exactly:
        # ``player.vel.dot(canonical_phys.rot_mat.forward)``.
        forward_speed = float(np.dot(car.physics.linear_velocity, physics.forward))
        features.extend(
            [
                _turn_radius(min(abs(forward_speed), CAR_MAX_SPEED)) / 1300.0,
                self._temporal_value(
                    shared_info,
                    "wisp_ball_touched_step",
                    agent,
                    float(car.ball_touches > 0),
                ),
                self._temporal_value(
                    shared_info,
                    "wisp_handbrake_values",
                    agent,
                    float(car.handbrake),
                ),
            ]
        )

        self_features = self._player_features(
            car, agent, agent, invert, mirror_x, ball, prediction, shared_info
        )
        features.extend(self_features)
        teammates: list[np.ndarray] = []
        opponents: list[np.ndarray] = []
        for other_agent, other_car in state.cars.items():
            if other_agent == agent:
                continue
            target = teammates if other_car.team_num == car.team_num else opponents
            target.append(
                self._player_features(
                    other_car,
                    agent,
                    other_agent,
                    invert,
                    mirror_x,
                    ball,
                    prediction,
                    shared_info,
                )
            )
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
        self._advance_post_observation_eta(agent, state, prediction)
        return observation


def observation_metadata() -> dict[str, Any]:
    return {
        "schema_version": 2,
        "observation_version": OBSERVATION_VERSION,
        "shape": [OBSERVATION_SIZE],
        "player_feature_size": PLAYER_OBSERVATION_SIZE,
        "prediction_ticks": list(PREDICTION_TICKS),
        "normalization": {
            "position": POS_COEF,
            "linear_velocity": VEL_COEF,
            "angular_velocity": ANG_VEL_COEF,
        },
        "contract_semantics": [
            "RocketSim BallPredictor replaces RLBot ball-prediction flatbuffer slices at the same tick horizons.",
            "A deterministic box-surface landing normal approximates Wisp's live arena-SDF query.",
            "The production two-pass 120-Hz cached rough_eta kernel persists across ordinary episode resets and is cleared only at an explicit process-session boundary.",
            "The episodic training scoreboard is zero unless a curriculum wrapper supplies score_diff.",
            "Previous controls, touch-step, and analog handbrake values refer to applied controller state at the decision boundary.",
        ],
        # Historical metadata key retained for downstream M05-M07 readers.
        "live_training_differences": [
            "RocketSim BallPredictor is the permitted prediction source.",
            "Landing normal remains the documented deterministic approximation.",
        ],
    }


# Keep the historical import name stable while making the contract version explicit.
Wisp432ContractV2 = WispCompatibleObs
