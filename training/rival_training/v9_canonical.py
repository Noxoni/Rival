"""Shared train/deploy canonical state for the Rival v9 scratch policy.

The two public adapters in this module deliberately use duck-typed source
objects.  The core module therefore imports neither RLGym nor RLBot and can be
loaded unchanged in both virtual environments.  All coordinate conversion and
timer normalization ends here; observation feature math lives in one separate
shared builder.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .v9_soccar_geometry import (
    ROCKETSIM_PAD_ORB_POSITIONS,
    STANDARD_GRAVITY_MAGNITUDE,
    STANDARD_GOAL_CENTERS,
    STANDARD_GOAL_HEIGHTS,
    STANDARD_GOAL_WIDTHS,
    STANDARD_PAD_IS_BIG,
    STANDARD_PAD_POSITIONS,
)


CANONICAL_STATE_VERSION = "RivalCanonicalStateV1"
CANONICAL_ADAPTER_VERSION = "RivalCanonicalAdapterV1"
PHYSICS_HZ = 120
AIR_STATES = ("OnGround", "Jumping", "DoubleJumping", "Dodging", "InAir")
AIR_STATE_INDEX = {name: index for index, name in enumerate(AIR_STATES)}
TEAM_INVERSION = np.asarray([-1.0, -1.0, 1.0], dtype=np.float32)
STANDARD_GRAVITY_Z = -STANDARD_GRAVITY_MAGNITUDE


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    return np.ascontiguousarray(result, dtype=np.float32)


def _vec3(value: Any | None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return _array(value, (3,), "vector")
    if isinstance(value, (list, tuple)):
        return _array(value, (3,), "vector")
    return np.asarray(
        [
            float(getattr(value, "x", 0.0)),
            float(getattr(value, "y", 0.0)),
            float(getattr(value, "z", 0.0)),
        ],
        dtype=np.float32,
    )


def _vec2(value: Any | None) -> np.ndarray:
    if value is None:
        return np.zeros(2, dtype=np.float32)
    if isinstance(value, np.ndarray):
        return _array(value, (2,), "vector2")
    if isinstance(value, (list, tuple)):
        return _array(value, (2,), "vector2")
    return np.asarray(
        [float(getattr(value, "x", 0.0)), float(getattr(value, "y", 0.0))],
        dtype=np.float32,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _rotation_from_flat(rotator: Any | None) -> np.ndarray:
    if rotator is None:
        return np.eye(3, dtype=np.float32)
    pitch = _safe_float(getattr(rotator, "pitch", 0.0))
    yaw = _safe_float(getattr(rotator, "yaw", 0.0))
    roll = _safe_float(getattr(rotator, "roll", 0.0))
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cr, sr = math.cos(roll), math.sin(roll)
    forward = np.asarray([cp * cy, cp * sy, sp], dtype=np.float32)
    up = np.asarray(
        [
            -cy * sp * cr - sy * sr,
            -sy * sp * cr + cy * sr,
            cp * cr,
        ],
        dtype=np.float32,
    )
    right = np.cross(up, forward).astype(np.float32)
    return np.ascontiguousarray(np.stack((forward, right, up), axis=1))


def _canonical_vector(value: np.ndarray, invert: bool) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    if invert:
        result *= TEAM_INVERSION
    return result


def _canonical_rotation(value: np.ndarray, invert: bool) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).copy()
    if invert:
        result *= TEAM_INVERSION[:, None]
    return np.ascontiguousarray(result, dtype=np.float32)


def _controller(value: Any | None) -> np.ndarray:
    if value is None:
        return np.zeros(8, dtype=np.float32)
    if isinstance(value, np.ndarray) or isinstance(value, (list, tuple)):
        result = np.asarray(value, dtype=np.float32).reshape(-1)
        if result.shape != (8,):
            raise ValueError(f"Physical controller must have eight fields, got {result.shape}")
        return result.copy()
    return np.asarray(
        [
            _safe_float(getattr(value, "throttle", 0.0)),
            _safe_float(getattr(value, "steer", 0.0)),
            _safe_float(getattr(value, "pitch", 0.0)),
            _safe_float(getattr(value, "yaw", 0.0)),
            _safe_float(getattr(value, "roll", 0.0)),
            float(bool(getattr(value, "jump", False))),
            float(bool(getattr(value, "boost", False))),
            float(bool(getattr(value, "handbrake", False))),
        ],
        dtype=np.float32,
    )


def _enum_name(value: Any) -> str:
    return str(value).split(".")[-1]


@dataclass(frozen=True)
class CanonicalPhysicsV1:
    position: np.ndarray
    rotation_mtx: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _array(self.position, (3,), "position"))
        object.__setattr__(
            self, "rotation_mtx", _array(self.rotation_mtx, (3, 3), "rotation_mtx")
        )
        object.__setattr__(
            self,
            "linear_velocity",
            _array(self.linear_velocity, (3,), "linear_velocity"),
        )
        object.__setattr__(
            self,
            "angular_velocity",
            _array(self.angular_velocity, (3,), "angular_velocity"),
        )

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
        return np.asarray(self.rotation_mtx.T @ vector, dtype=np.float32)

    def to_payload(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "rotation_mtx": self.rotation_mtx.tolist(),
            "linear_velocity": self.linear_velocity.tolist(),
            "angular_velocity": self.angular_velocity.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CanonicalPhysicsV1:
        return cls(
            payload["position"],
            payload["rotation_mtx"],
            payload["linear_velocity"],
            payload["angular_velocity"],
        )


@dataclass(frozen=True)
class CanonicalCarV1:
    physics: CanonicalPhysicsV1
    boost: float
    demo_time_remaining: float
    surface_contact: bool
    boosting: bool
    supersonic: bool
    handbrake: float
    air_state: int
    jump_held: bool
    has_jumped: bool
    has_double_jumped: bool
    has_dodged: bool
    can_dodge: bool
    air_time: float
    jump_hold_elapsed: float
    dodge_window_remaining: float
    dodge_elapsed: float
    dodge_direction: np.ndarray
    latest_controller: np.ndarray

    def __post_init__(self) -> None:
        if not 0 <= int(self.air_state) < len(AIR_STATES):
            raise ValueError(f"Invalid canonical air state {self.air_state}")
        object.__setattr__(
            self,
            "dodge_direction",
            _array(self.dodge_direction, (2,), "dodge_direction"),
        )
        object.__setattr__(
            self,
            "latest_controller",
            _array(self.latest_controller, (8,), "latest_controller"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "physics": self.physics.to_payload(),
            "boost": float(self.boost),
            "demo_time_remaining": float(self.demo_time_remaining),
            "surface_contact": bool(self.surface_contact),
            "boosting": bool(self.boosting),
            "supersonic": bool(self.supersonic),
            "handbrake": float(self.handbrake),
            "air_state": int(self.air_state),
            "jump_held": bool(self.jump_held),
            "has_jumped": bool(self.has_jumped),
            "has_double_jumped": bool(self.has_double_jumped),
            "has_dodged": bool(self.has_dodged),
            "can_dodge": bool(self.can_dodge),
            "air_time": float(self.air_time),
            "jump_hold_elapsed": float(self.jump_hold_elapsed),
            "dodge_window_remaining": float(self.dodge_window_remaining),
            "dodge_elapsed": float(self.dodge_elapsed),
            "dodge_direction": self.dodge_direction.tolist(),
            "latest_controller": self.latest_controller.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CanonicalCarV1:
        return cls(
            physics=CanonicalPhysicsV1.from_payload(payload["physics"]),
            boost=float(payload["boost"]),
            demo_time_remaining=float(payload["demo_time_remaining"]),
            surface_contact=bool(payload["surface_contact"]),
            boosting=bool(payload["boosting"]),
            supersonic=bool(payload["supersonic"]),
            handbrake=float(payload["handbrake"]),
            air_state=int(payload["air_state"]),
            jump_held=bool(payload["jump_held"]),
            has_jumped=bool(payload["has_jumped"]),
            has_double_jumped=bool(payload["has_double_jumped"]),
            has_dodged=bool(payload["has_dodged"]),
            can_dodge=bool(payload["can_dodge"]),
            air_time=float(payload["air_time"]),
            jump_hold_elapsed=float(payload["jump_hold_elapsed"]),
            dodge_window_remaining=float(payload["dodge_window_remaining"]),
            dodge_elapsed=float(payload["dodge_elapsed"]),
            dodge_direction=payload["dodge_direction"],
            latest_controller=payload["latest_controller"],
        )


@dataclass(frozen=True)
class RivalCanonicalStateV1:
    tick_index: int
    seconds_elapsed: float
    game_time_remaining: float
    score_diff: int
    overtime: bool
    kickoff: bool
    active_play: bool
    gravity_z: float
    self_touch_age: float
    opponent_touch_age: float
    last_toucher: int
    self_car: CanonicalCarV1
    opponent_car: CanonicalCarV1
    ball: CanonicalPhysicsV1
    goal_centers: np.ndarray
    goal_widths: np.ndarray
    goal_heights: np.ndarray
    pad_positions: np.ndarray
    pad_is_big: np.ndarray
    pad_active: np.ndarray
    pad_time_until_active: np.ndarray
    version: str = CANONICAL_STATE_VERSION
    adapter_version: str = CANONICAL_ADAPTER_VERSION

    def __post_init__(self) -> None:
        if self.version != CANONICAL_STATE_VERSION:
            raise ValueError(f"Unexpected canonical state version {self.version!r}")
        if not 0 <= int(self.last_toucher) <= 2:
            raise ValueError("last_toucher must be self=0, opponent=1 or none=2")
        object.__setattr__(
            self, "goal_centers", _array(self.goal_centers, (2, 3), "goal_centers")
        )
        object.__setattr__(
            self, "goal_widths", _array(self.goal_widths, (2,), "goal_widths")
        )
        object.__setattr__(
            self, "goal_heights", _array(self.goal_heights, (2,), "goal_heights")
        )
        object.__setattr__(
            self, "pad_positions", _array(self.pad_positions, (34, 3), "pad_positions")
        )
        object.__setattr__(
            self, "pad_is_big", _array(self.pad_is_big, (34,), "pad_is_big")
        )
        object.__setattr__(
            self, "pad_active", _array(self.pad_active, (34,), "pad_active")
        )
        object.__setattr__(
            self,
            "pad_time_until_active",
            _array(self.pad_time_until_active, (34,), "pad_time_until_active"),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "adapter_version": self.adapter_version,
            "tick_index": int(self.tick_index),
            "seconds_elapsed": float(self.seconds_elapsed),
            "game_time_remaining": float(self.game_time_remaining),
            "score_diff": int(self.score_diff),
            "overtime": bool(self.overtime),
            "kickoff": bool(self.kickoff),
            "active_play": bool(self.active_play),
            "gravity_z": float(self.gravity_z),
            "self_touch_age": float(self.self_touch_age),
            "opponent_touch_age": float(self.opponent_touch_age),
            "last_toucher": int(self.last_toucher),
            "self_car": self.self_car.to_payload(),
            "opponent_car": self.opponent_car.to_payload(),
            "ball": self.ball.to_payload(),
            "goal_centers": self.goal_centers.tolist(),
            "goal_widths": self.goal_widths.tolist(),
            "goal_heights": self.goal_heights.tolist(),
            "pad_positions": self.pad_positions.tolist(),
            "pad_is_big": self.pad_is_big.tolist(),
            "pad_active": self.pad_active.tolist(),
            "pad_time_until_active": self.pad_time_until_active.tolist(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RivalCanonicalStateV1:
        return cls(
            version=str(payload["version"]),
            adapter_version=str(payload["adapter_version"]),
            tick_index=int(payload["tick_index"]),
            seconds_elapsed=float(payload["seconds_elapsed"]),
            game_time_remaining=float(payload["game_time_remaining"]),
            score_diff=int(payload["score_diff"]),
            overtime=bool(payload["overtime"]),
            kickoff=bool(payload["kickoff"]),
            active_play=bool(payload["active_play"]),
            gravity_z=float(payload["gravity_z"]),
            self_touch_age=float(payload["self_touch_age"]),
            opponent_touch_age=float(payload["opponent_touch_age"]),
            last_toucher=int(payload["last_toucher"]),
            self_car=CanonicalCarV1.from_payload(payload["self_car"]),
            opponent_car=CanonicalCarV1.from_payload(payload["opponent_car"]),
            ball=CanonicalPhysicsV1.from_payload(payload["ball"]),
            goal_centers=payload["goal_centers"],
            goal_widths=payload["goal_widths"],
            goal_heights=payload["goal_heights"],
            pad_positions=payload["pad_positions"],
            pad_is_big=payload["pad_is_big"],
            pad_active=payload["pad_active"],
            pad_time_until_active=payload["pad_time_until_active"],
        )


class _AdapterTemporalState:
    def __init__(self) -> None:
        self.last_tick: int | None = None
        self.last_surface_tick: dict[Hashable, int] = {}
        self.jump_start_tick: dict[Hashable, int] = {}
        self.double_jump_start_tick: dict[Hashable, int] = {}
        self.previous_double_jump: dict[Hashable, bool] = {}
        self.touch_seconds: dict[Hashable, float] = {}

    def reset(self) -> None:
        self.__init__()

    def begin_tick(self, tick: int) -> None:
        if self.last_tick is not None and tick < self.last_tick:
            self.reset()
        self.last_tick = tick

    def air_time(self, key: Hashable, tick: int, surface: bool) -> float:
        if surface:
            self.last_surface_tick[key] = tick
            return 0.0
        if key not in self.last_surface_tick:
            self.last_surface_tick[key] = tick
        return max(0.0, (tick - self.last_surface_tick[key]) / PHYSICS_HZ)

    def jump_hold(self, key: Hashable, tick: int, air_state: str) -> float:
        if air_state == "Jumping":
            self.jump_start_tick.setdefault(key, tick)
            return min(0.2, max(0.0, (tick - self.jump_start_tick[key]) / PHYSICS_HZ))
        self.jump_start_tick.pop(key, None)
        return 0.0

    def rlgym_air_state(self, key: Hashable, tick: int, car: Any) -> str:
        surface = bool(getattr(car, "on_ground", False))
        has_double = bool(getattr(car, "has_double_jumped", False))
        previous_double = self.previous_double_jump.get(key, has_double)
        if has_double and not previous_double:
            self.double_jump_start_tick[key] = tick
        self.previous_double_jump[key] = has_double
        if surface:
            self.double_jump_start_tick.pop(key, None)
            return "OnGround"
        if bool(getattr(car, "is_flipping", False)):
            return "Dodging"
        double_start = self.double_jump_start_tick.get(key)
        if double_start is not None and tick - double_start < 13:
            return "DoubleJumping"
        if bool(getattr(car, "is_jumping", False)):
            return "Jumping"
        return "InAir"


def _canonical_physics_from_rlgym(physics: Any, invert: bool) -> CanonicalPhysicsV1:
    return CanonicalPhysicsV1(
        _canonical_vector(_vec3(physics.position), invert),
        _canonical_rotation(np.asarray(physics.rotation_mtx, dtype=np.float32), invert),
        _canonical_vector(_vec3(physics.linear_velocity), invert),
        _canonical_vector(_vec3(physics.angular_velocity), invert),
    )


def _canonical_physics_from_rlbot(physics: Any, invert: bool) -> CanonicalPhysicsV1:
    return CanonicalPhysicsV1(
        _canonical_vector(_vec3(getattr(physics, "location", None)), invert),
        _canonical_rotation(_rotation_from_flat(getattr(physics, "rotation", None)), invert),
        _canonical_vector(_vec3(getattr(physics, "velocity", None)), invert),
        _canonical_vector(_vec3(getattr(physics, "angular_velocity", None)), invert),
    )


def _pad_mapping(source_positions: np.ndarray, invert: bool) -> np.ndarray:
    transformed = np.asarray(source_positions, dtype=np.float32).copy()
    if invert:
        transformed *= TEAM_INVERSION
    # RLBot v5 FieldInfo exposes pickup-volume anchors while RLGym's standard
    # table uses floating-orb centers and a slightly different source order.
    # Pad identity is therefore defined by XY.  The canonical representation is
    # the RLBot v5 FieldInfo anchor table from the useful-values authority page.
    mapping: list[int] = []
    used: set[int] = set()
    for target in STANDARD_PAD_POSITIONS:
        distances = np.sum((transformed[:, :2] - target[:2]) ** 2, axis=1)
        for index in np.argsort(distances):
            candidate = int(index)
            if candidate not in used:
                # The two public tables also differ by up to two units on a
                # handful of asymmetric pad Y coordinates (for example
                # 3308/3310).  Five units is a bounded identity tolerance, far
                # below the distance between distinct pads.
                if float(distances[candidate]) > 25.0:
                    raise ValueError(
                        f"Unable to map canonical boost pad {target.tolist()}; "
                        f"nearest XY squared error is {float(distances[candidate])}"
                    )
                mapping.append(candidate)
                used.add(candidate)
                break
    if len(mapping) != 34:
        raise ValueError(f"Expected 34 unique boost-pad mappings, got {len(mapping)}")
    return np.asarray(mapping, dtype=np.int64)


def _score_by_team(value: Any) -> dict[int, int]:
    if isinstance(value, Mapping):
        return {int(key): int(score) for key, score in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return {index: int(score) for index, score in enumerate(value)}
    return {0: 0, 1: 0}


class RocketSimCanonicalAdapterV1:
    """Thin RLGym/RocketSim source adapter into ``RivalCanonicalStateV1``."""

    source_domain = "rocketsim_rlgym_v2"

    def __init__(self) -> None:
        self.temporal = _AdapterTemporalState()

    def reset(self) -> None:
        self.temporal.reset()

    def _car(
        self,
        agent: Hashable,
        car: Any,
        *,
        tick: int,
        invert: bool,
        controller: np.ndarray,
    ) -> CanonicalCarV1:
        surface = bool(getattr(car, "on_ground", False))
        air_state_name = self.temporal.rlgym_air_state(agent, tick, car)
        has_jumped = bool(getattr(car, "has_jumped", False))
        has_double = bool(getattr(car, "has_double_jumped", False))
        has_dodged = bool(getattr(car, "has_flipped", False))
        jump_held = bool(getattr(car, "is_holding_jump", controller[5] > 0.5))
        can_dodge = bool(getattr(car, "can_flip", False))
        air_time_since_jump = max(0.0, _safe_float(getattr(car, "air_time_since_jump", 0.0)))
        if can_dodge and not has_jumped:
            dodge_window = 1.45
        elif can_dodge:
            dodge_window = max(0.0, 1.25 - air_time_since_jump)
        else:
            dodge_window = 0.0
        torque = np.asarray(getattr(car, "flip_torque", np.zeros(3)), dtype=np.float32)
        dodge_direction = np.zeros(2, dtype=np.float32)
        if torque.shape == (3,):
            dodge_direction = np.asarray([torque[1], -torque[0]], dtype=np.float32)
            length = float(np.linalg.norm(dodge_direction))
            if length > 1e-6:
                dodge_direction /= length
        return CanonicalCarV1(
            physics=_canonical_physics_from_rlgym(car.physics, invert),
            boost=_safe_float(getattr(car, "boost_amount", 0.0)),
            demo_time_remaining=max(
                0.0, _safe_float(getattr(car, "demo_respawn_timer", 0.0))
            ),
            surface_contact=surface,
            boosting=bool(getattr(car, "is_boosting", False)),
            supersonic=bool(getattr(car, "is_supersonic", False)),
            handbrake=_safe_float(getattr(car, "handbrake", controller[7])),
            air_state=AIR_STATE_INDEX[air_state_name],
            jump_held=jump_held,
            has_jumped=has_jumped,
            has_double_jumped=has_double,
            has_dodged=has_dodged,
            can_dodge=can_dodge,
            air_time=self.temporal.air_time(agent, tick, surface),
            jump_hold_elapsed=max(0.0, _safe_float(getattr(car, "jump_time", 0.0))),
            dodge_window_remaining=dodge_window,
            dodge_elapsed=max(0.0, _safe_float(getattr(car, "flip_time", 0.0))),
            dodge_direction=dodge_direction,
            latest_controller=controller,
        )

    def adapt(
        self,
        state: Any,
        observer: Hashable,
        shared_info: Mapping[str, Any] | None = None,
    ) -> RivalCanonicalStateV1:
        shared = {} if shared_info is None else shared_info
        tick = int(getattr(state, "tick_count", 0))
        self.temporal.begin_tick(tick)
        cars = getattr(state, "cars", {})
        if observer not in cars:
            raise KeyError(f"Observer {observer!r} is absent from RocketSim state")
        self_source = cars[observer]
        opponents = [
            (agent, car)
            for agent, car in cars.items()
            if int(getattr(car, "team_num", 0)) != int(self_source.team_num)
        ]
        if len(opponents) != 1:
            raise ValueError(f"RivalObsV1 requires exactly one opponent, got {len(opponents)}")
        opponent, opponent_source = opponents[0]
        self_team = int(self_source.team_num)
        invert = self_team == 1

        applied = shared.get("rival_v9_applied_actions", {})
        if not applied:
            applied = shared.get("rival_action_last_applied", {})
        if not applied:
            applied = shared.get("previous_actions", {})
        self_controller = _controller(applied.get(observer) if isinstance(applied, Mapping) else None)
        opponent_controller = _controller(
            applied.get(opponent) if isinstance(applied, Mapping) else None
        )

        seconds_elapsed = tick / PHYSICS_HZ
        for agent, car in ((observer, self_source), (opponent, opponent_source)):
            if int(getattr(car, "ball_touches", 0)) > 0:
                self.temporal.touch_seconds[agent] = seconds_elapsed
        self_touch_seconds = self.temporal.touch_seconds.get(observer)
        opponent_touch_seconds = self.temporal.touch_seconds.get(opponent)
        self_touch_age = (
            10.0 if self_touch_seconds is None else max(0.0, seconds_elapsed - self_touch_seconds)
        )
        opponent_touch_age = (
            10.0
            if opponent_touch_seconds is None
            else max(0.0, seconds_elapsed - opponent_touch_seconds)
        )
        if self_touch_seconds is None and opponent_touch_seconds is None:
            last_toucher = 2
        elif opponent_touch_seconds is None or (
            self_touch_seconds is not None and self_touch_seconds >= opponent_touch_seconds
        ):
            last_toucher = 0
        else:
            last_toucher = 1

        timers = np.asarray(getattr(state, "boost_pad_timers", np.zeros(34)), dtype=np.float32)
        if timers.shape != (34,):
            raise ValueError(f"RocketSim boost-pad timers must have shape (34,), got {timers.shape}")
        pad_map = _pad_mapping(ROCKETSIM_PAD_ORB_POSITIONS, invert)
        canonical_timers = np.maximum(0.0, timers[pad_map]).astype(np.float32)
        pad_active = (canonical_timers <= 1e-6).astype(np.float32)

        scores = _score_by_team(shared.get("score_by_team", {0: 0, 1: 0}))
        config = getattr(state, "config", None)
        gravity_multiplier = _safe_float(getattr(config, "gravity", 1.0), 1.0)
        return RivalCanonicalStateV1(
            tick_index=tick,
            seconds_elapsed=seconds_elapsed,
            game_time_remaining=_safe_float(
                shared.get("game_time_remaining", max(0.0, 300.0 - seconds_elapsed))
            ),
            score_diff=scores.get(self_team, 0) - scores.get(1 - self_team, 0),
            overtime=bool(shared.get("overtime", False)),
            kickoff=bool(shared.get("kickoff", False)),
            active_play=bool(
                shared.get("active_play", not bool(getattr(state, "goal_scored", False)))
            ),
            gravity_z=STANDARD_GRAVITY_Z * gravity_multiplier,
            self_touch_age=self_touch_age,
            opponent_touch_age=opponent_touch_age,
            last_toucher=last_toucher,
            self_car=self._car(
                observer,
                self_source,
                tick=tick,
                invert=invert,
                controller=self_controller,
            ),
            opponent_car=self._car(
                opponent,
                opponent_source,
                tick=tick,
                invert=invert,
                controller=opponent_controller,
            ),
            ball=_canonical_physics_from_rlgym(state.ball, invert),
            goal_centers=STANDARD_GOAL_CENTERS,
            goal_widths=STANDARD_GOAL_WIDTHS,
            goal_heights=STANDARD_GOAL_HEIGHTS,
            pad_positions=STANDARD_PAD_POSITIONS,
            pad_is_big=STANDARD_PAD_IS_BIG.astype(np.float32),
            pad_active=pad_active,
            pad_time_until_active=canonical_timers,
        )


class RLBotCanonicalAdapterV1:
    """Thin RLBot v5 GamePacket/FieldInfo adapter into the shared canonical state."""

    source_domain = "rlbot_v5_gamepacket"

    def __init__(self) -> None:
        self.temporal = _AdapterTemporalState()

    def reset(self) -> None:
        self.temporal.reset()

    def _car(
        self,
        key: Hashable,
        player: Any,
        *,
        tick: int,
        invert: bool,
    ) -> CanonicalCarV1:
        controls = _controller(getattr(player, "last_input", None))
        air_state_name = _enum_name(getattr(player, "air_state", "InAir"))
        if air_state_name not in AIR_STATE_INDEX:
            air_state_name = "InAir"
        surface = air_state_name == "OnGround"
        has_jumped = bool(getattr(player, "has_jumped", False))
        has_double = bool(getattr(player, "has_double_jumped", False))
        has_dodged = bool(getattr(player, "has_dodged", False))
        dodge_timeout = _safe_float(getattr(player, "dodge_timeout", -1.0), -1.0)
        flip_reset = air_state_name == "InAir" and not has_jumped and not has_double and not has_dodged
        can_dodge = bool(
            not surface
            and not controls[5] > 0.5
            and not has_double
            and not has_dodged
            and (dodge_timeout >= 0.0 or flip_reset)
        )
        dodge_window = 1.45 if can_dodge and flip_reset else max(0.0, dodge_timeout)
        physics = getattr(player, "physics", None)
        if physics is None:
            raise ValueError("RLBot PlayerInfo is missing required physics")
        return CanonicalCarV1(
            physics=_canonical_physics_from_rlbot(physics, invert),
            boost=_safe_float(getattr(player, "boost", 0.0)),
            demo_time_remaining=max(
                0.0, _safe_float(getattr(player, "demolished_timeout", -1.0), -1.0)
            ),
            surface_contact=surface,
            boosting=bool(controls[6] > 0.5 and _safe_float(getattr(player, "boost", 0.0)) > 0),
            supersonic=bool(getattr(player, "is_supersonic", False)),
            handbrake=float(controls[7]),
            air_state=AIR_STATE_INDEX[air_state_name],
            jump_held=bool(controls[5] > 0.5),
            has_jumped=has_jumped,
            has_double_jumped=has_double,
            has_dodged=has_dodged,
            can_dodge=can_dodge,
            air_time=self.temporal.air_time(key, tick, surface),
            jump_hold_elapsed=self.temporal.jump_hold(key, tick, air_state_name),
            dodge_window_remaining=dodge_window,
            dodge_elapsed=max(0.0, _safe_float(getattr(player, "dodge_elapsed", 0.0))),
            dodge_direction=_vec2(getattr(player, "dodge_dir", None)),
            latest_controller=controls,
        )

    def adapt(
        self,
        packet: Any,
        self_index: int,
        field_info: Any | None,
    ) -> RivalCanonicalStateV1:
        players = list(getattr(packet, "players", None) or [])
        if not 0 <= int(self_index) < len(players):
            raise IndexError(f"RLBot self index {self_index} outside {len(players)} players")
        self_player = players[int(self_index)]
        self_team = int(getattr(self_player, "team", 0))
        opponents = [
            (index, player)
            for index, player in enumerate(players)
            if index != int(self_index) and int(getattr(player, "team", 0)) != self_team
        ]
        if len(opponents) != 1:
            raise ValueError(f"RivalObsV1 requires exactly one opponent, got {len(opponents)}")
        opponent_index, opponent_player = opponents[0]
        invert = self_team == 1
        match = getattr(packet, "match_info", None)
        seconds_elapsed = _safe_float(getattr(match, "seconds_elapsed", 0.0))
        frame = getattr(match, "frame_num", None)
        tick = int(frame) if frame is not None else int(round(seconds_elapsed * PHYSICS_HZ))
        self.temporal.begin_tick(tick)

        touches: dict[int, float | None] = {}
        for index, player in ((int(self_index), self_player), (opponent_index, opponent_player)):
            touch = getattr(player, "latest_touch", None)
            touches[index] = (
                None
                if touch is None
                else _safe_float(getattr(touch, "game_seconds", 0.0))
            )
        self_touch_seconds = touches[int(self_index)]
        opponent_touch_seconds = touches[opponent_index]
        self_touch_age = (
            10.0 if self_touch_seconds is None else max(0.0, seconds_elapsed - self_touch_seconds)
        )
        opponent_touch_age = (
            10.0
            if opponent_touch_seconds is None
            else max(0.0, seconds_elapsed - opponent_touch_seconds)
        )
        if self_touch_seconds is None and opponent_touch_seconds is None:
            last_toucher = 2
        elif opponent_touch_seconds is None or (
            self_touch_seconds is not None and self_touch_seconds >= opponent_touch_seconds
        ):
            last_toucher = 0
        else:
            last_toucher = 1

        balls = list(getattr(packet, "balls", None) or [])
        if not balls or getattr(balls[0], "physics", None) is None:
            raise ValueError("RLBot GamePacket is missing the Soccar ball physics")
        source_goals = list(getattr(field_info, "goals", None) or [])
        if source_goals:
            goals_by_team = {
                int(getattr(goal, "team_num", -1)): goal for goal in source_goals
            }
            if self_team not in goals_by_team or 1 - self_team not in goals_by_team:
                raise ValueError(
                    "RLBot standard-Soccar FieldInfo must expose one goal for each team"
                )
            blue_location = _vec3(getattr(goals_by_team[0], "location", None))
            orange_location = _vec3(getattr(goals_by_team[1], "location", None))
            expected_xy = STANDARD_GOAL_CENTERS[:, :2]
            actual_xy = np.stack((blue_location[:2], orange_location[:2]))
            if not np.allclose(actual_xy, expected_xy, atol=1.0, rtol=0.0):
                raise ValueError(
                    "RivalObsV1 supports standard Soccar goals at x=0, y=+/-5120; "
                    f"FieldInfo reported {actual_xy.tolist()}"
                )

        # RLBot v5 beta currently reports the map's goal/scoring volume
        # (observed on Stadium_P as roughly 1920 x 752 at z=312), while the
        # useful-values authority and RocketSim define the physical opening as
        # 1785.51 x 642.775. Rival's actor fields are explicitly physical
        # opening/post geometry, so both source adapters canonicalize to the
        # same documented standard-Soccar values. Raw FieldInfo goal metadata
        # remains captured and independently audited as a separate runtime
        # source rather than silently masquerading as physical posts.
        goal_centers = STANDARD_GOAL_CENTERS
        goal_widths = STANDARD_GOAL_WIDTHS
        goal_heights = STANDARD_GOAL_HEIGHTS
        static_pads = list(getattr(field_info, "boost_pads", None) or [])
        dynamic_pads = list(getattr(packet, "boost_pads", None) or [])
        if static_pads:
            source_positions = np.stack(
                [_vec3(getattr(pad, "location", None)) for pad in static_pads]
            ).astype(np.float32)
            source_big = np.asarray(
                [bool(getattr(pad, "is_full_boost", False)) for pad in static_pads]
            )
        else:
            source_positions = STANDARD_PAD_POSITIONS.copy()
            source_big = STANDARD_PAD_IS_BIG.copy()
        if len(source_positions) != 34 or len(dynamic_pads) != 34:
            raise ValueError(
                f"RLBot canonical adapter requires 34 pads, got static={len(source_positions)} "
                f"dynamic={len(dynamic_pads)}"
            )
        pad_map = _pad_mapping(source_positions, invert)
        canonical_big = source_big[pad_map]
        pad_active = np.asarray(
            [bool(getattr(dynamic_pads[index], "is_active", False)) for index in pad_map],
            dtype=np.float32,
        )
        elapsed = np.asarray(
            [_safe_float(getattr(dynamic_pads[index], "timer", 0.0)) for index in pad_map],
            dtype=np.float32,
        )
        respawn = np.where(canonical_big, 10.0, 4.0).astype(np.float32)
        time_until_active = np.where(
            pad_active > 0.5, 0.0, np.maximum(0.0, respawn - elapsed)
        ).astype(np.float32)

        scores = {
            int(getattr(team, "team_index", index)): int(getattr(team, "score", 0))
            for index, team in enumerate(list(getattr(packet, "teams", None) or []))
        }
        phase_name = _enum_name(getattr(match, "match_phase", "Inactive"))
        return RivalCanonicalStateV1(
            tick_index=tick,
            seconds_elapsed=seconds_elapsed,
            game_time_remaining=_safe_float(getattr(match, "game_time_remaining", 0.0)),
            score_diff=scores.get(self_team, 0) - scores.get(1 - self_team, 0),
            overtime=bool(getattr(match, "is_overtime", False)),
            kickoff=phase_name in {"Countdown", "Kickoff"},
            active_play=phase_name in {"Kickoff", "Active"},
            gravity_z=_safe_float(getattr(match, "world_gravity_z", STANDARD_GRAVITY_Z), STANDARD_GRAVITY_Z),
            self_touch_age=self_touch_age,
            opponent_touch_age=opponent_touch_age,
            last_toucher=last_toucher,
            self_car=self._car(int(self_index), self_player, tick=tick, invert=invert),
            opponent_car=self._car(
                opponent_index, opponent_player, tick=tick, invert=invert
            ),
            ball=_canonical_physics_from_rlbot(balls[0].physics, invert),
            goal_centers=goal_centers,
            goal_widths=goal_widths,
            goal_heights=goal_heights,
            pad_positions=STANDARD_PAD_POSITIONS,
            pad_is_big=canonical_big.astype(np.float32),
            pad_active=pad_active,
            pad_time_until_active=time_until_active,
        )
