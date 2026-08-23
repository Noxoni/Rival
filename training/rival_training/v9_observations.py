"""One shared RivalObsV1 feature implementation for training and deployment."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import RocketSim as rs

from .v9_canonical import (
    AIR_STATES,
    CANONICAL_ADAPTER_VERSION,
    CANONICAL_STATE_VERSION,
    RivalCanonicalStateV1,
)
from .v9_soccar_geometry import (
    BACK_NET_Y,
    BACK_WALL_Y,
    BALL_MAX_SPEED,
    CEILING_Z,
    CAR_MAX_ANGULAR_SPEED,
    CAR_MAX_SPEED,
    CORNER_ENDPOINT_OFFSET,
    GOAL_HALF_WIDTH,
    GOAL_HEIGHT,
    GEOMETRY_VERSION,
    SIDE_WALL_X,
    USEFUL_GAME_VALUES_URL,
)


OBSERVATION_VERSION = "RivalObsV1"
OBSERVATION_SCHEMA_VERSION = 1
PREDICTION_HORIZONS_SECONDS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)
PREDICTION_HORIZON_TICKS = tuple(round(value * 120) for value in PREDICTION_HORIZONS_SECONDS)
PREDICTION_COUNT = len(PREDICTION_HORIZONS_SECONDS)
PREDICTION_FEATURES = 12
PAD_COUNT = 34
PAD_FEATURES = 9
HISTORY_TICKS = 8
CONTROLLER_SIZE = 8

ANGULAR_SPEED_SCALE = CAR_MAX_ANGULAR_SPEED
FIELD_SCALE = np.asarray([SIDE_WALL_X, BACK_WALL_Y, CEILING_Z], dtype=np.float32)
VECTOR_SCALE = BALL_MAX_SPEED
@dataclass(frozen=True)
class ObservationFieldV1:
    name: str
    block: str
    start: int
    end: int
    shape: list[int]
    dtype: str
    normalization: str
    coordinate_frame: str
    canonical_source: str
    update_cadence: str
    reset_semantics: str


class _SchemaBuilder:
    def __init__(self) -> None:
        self.fields: list[ObservationFieldV1] = []
        self.offset = 0

    def add(
        self,
        name: str,
        block: str,
        shape: tuple[int, ...],
        normalization: str,
        coordinate_frame: str,
        canonical_source: str,
        update_cadence: str = "every physics tick",
        reset_semantics: str = "stateless",
    ) -> None:
        size = int(np.prod(shape))
        self.fields.append(
            ObservationFieldV1(
                name=name,
                block=block,
                start=self.offset,
                end=self.offset + size,
                shape=list(shape),
                dtype="float32",
                normalization=normalization,
                coordinate_frame=coordinate_frame,
                canonical_source=canonical_source,
                update_cadence=update_cadence,
                reset_semantics=reset_semantics,
            )
        )
        self.offset += size


def _build_schema_fields() -> list[ObservationFieldV1]:
    schema = _SchemaBuilder()
    block = "match_control"
    schema.add("match.score_diff", block, (1,), "clip[-5,5]/5", "Rival perspective", "score_diff")
    schema.add("match.time_remaining", block, (1,), "clip[0,300]/300", "match", "game_time_remaining")
    schema.add("match.overtime", block, (1,), "boolean", "match", "overtime")
    schema.add("match.kickoff", block, (1,), "boolean", "match", "kickoff")
    schema.add("match.active_play", block, (1,), "boolean", "match", "active_play")
    schema.add("match.gravity", block, (1,), "gravity_z/650", "world Z", "gravity_z")
    schema.add("touch.self_age", block, (1,), "clip[0,10]/10", "match time", "self_touch_age")
    schema.add("touch.opponent_age", block, (1,), "clip[0,10]/10", "match time", "opponent_touch_age")
    schema.add("touch.last_toucher", block, (3,), "one-hot", "Rival perspective", "last_toucher")
    schema.add(
        "prediction.age",
        block,
        (1,),
        "clip[0,4]/4 ticks",
        "physics ticks",
        "shared predictor cache",
        "every tick; predictor refresh configurable",
        "zero/empty on episode reset",
    )
    schema.add("intercept.self", block, (1,), "clip[0,10]/10", "seconds", "shared pure intercept proxy")
    schema.add("intercept.opponent", block, (1,), "clip[0,10]/10", "seconds", "shared pure intercept proxy")
    schema.add("intercept.advantage", block, (1,), "clip[-10,10]/10", "Rival perspective", "opponent minus self")

    block = "self_car"
    for name, shape, normalization, frame, source in (
        ("self.position", (3,), "divide field XYZ extents", "team", "self_car.physics.position"),
        ("self.forward", (3,), "unit vector", "team", "self_car.physics.forward"),
        ("self.up", (3,), "unit vector", "team", "self_car.physics.up"),
        ("self.linear_velocity", (3,), "divide 2300", "team", "self_car.physics.linear_velocity"),
        ("self.angular_velocity", (3,), "divide 5.5", "team", "self_car.physics.angular_velocity"),
        ("self.local_linear_velocity", (3,), "divide 2300", "car local", "shared rotation transform"),
        ("self.local_angular_velocity", (3,), "divide 5.5", "car local", "shared rotation transform"),
        ("self.speed", (1,), "clip[0,2300]/2300", "scalar", "linear_velocity norm"),
        ("self.signed_forward_speed", (1,), "clip[-2300,2300]/2300", "car forward", "velocity dot forward"),
        ("self.boost", (1,), "clip[0,100]/100", "scalar", "self_car.boost"),
        ("self.demo_time", (1,), "clip[0,3]/3", "seconds", "self_car.demo_time_remaining"),
        ("self.surface_contact", (1,), "boolean", "car", "self_car.surface_contact"),
        ("self.boosting", (1,), "boolean", "car", "self_car.boosting"),
        ("self.supersonic", (1,), "boolean", "car", "self_car.supersonic"),
        ("self.handbrake", (1,), "clip[0,1]", "controller", "self_car.handbrake"),
        ("self.air_state", (5,), "one-hot", "car", "self_car.air_state"),
        ("self.jump_held", (1,), "boolean", "controller", "self_car.jump_held"),
        ("self.has_jumped", (1,), "boolean", "car resource", "self_car.has_jumped"),
        ("self.has_double_jumped", (1,), "boolean", "car resource", "self_car.has_double_jumped"),
        ("self.has_dodged", (1,), "boolean", "car resource", "self_car.has_dodged"),
        ("self.can_dodge", (1,), "boolean", "car resource", "self_car.can_dodge"),
        ("self.air_time", (1,), "clip[0,5]/5", "seconds", "self_car.air_time"),
        ("self.jump_hold_elapsed", (1,), "clip[0,0.2]/0.2", "seconds", "self_car.jump_hold_elapsed"),
        ("self.dodge_window_remaining", (1,), "clip[0,1.45]/1.45", "seconds", "self_car.dodge_window_remaining"),
        ("self.dodge_elapsed", (1,), "clip[0,0.95]/0.95", "seconds", "self_car.dodge_elapsed"),
        ("self.dodge_direction", (2,), "unit vector", "canonical flip frame", "self_car.dodge_direction"),
        ("self.surface_distances", (5,), "nonnegative floor/ceiling/side/goal-aware-back/corner clearances divided by physical extents", "team", "shared RLBot-v5 planar standard-Soccar helper; curved ramps/posts approximated"),
        ("self.nearest_surface_normal", (3,), "unit vector", "car local", "shared RLBot-v5 planar standard-Soccar helper"),
        ("self.surface_up_alignment", (1,), "dot product", "car local", "shared RLBot-v5 planar standard-Soccar helper"),
        ("self.surface_signed_velocity", (1,), "divide 2300", "nearest surface normal", "shared RLBot-v5 planar standard-Soccar helper"),
        ("self.goal_centers_local", (2, 3), "divide 6000", "car local", "shared documented physical standard-Soccar goal centers"),
    ):
        schema.add(name, block, shape, normalization, frame, source)

    block = "opponent_car"
    for name, shape, normalization, frame, source in (
        ("opponent.position", (3,), "divide field XYZ extents", "team", "opponent_car.physics.position"),
        ("opponent.forward", (3,), "unit vector", "team", "opponent_car.physics.forward"),
        ("opponent.up", (3,), "unit vector", "team", "opponent_car.physics.up"),
        ("opponent.linear_velocity", (3,), "divide 2300", "team", "opponent_car.physics.linear_velocity"),
        ("opponent.angular_velocity", (3,), "divide 5.5", "team", "opponent_car.physics.angular_velocity"),
        ("opponent.relative_position", (3,), "divide 6000", "Rival car local", "shared relative transform"),
        ("opponent.relative_velocity", (3,), "divide 2300", "Rival car local", "shared relative transform"),
        ("opponent.speed", (1,), "clip[0,2300]/2300", "scalar", "linear_velocity norm"),
        ("opponent.signed_forward_speed", (1,), "clip[-2300,2300]/2300", "opponent forward", "velocity dot forward"),
        ("opponent.boost", (1,), "clip[0,100]/100", "scalar", "opponent_car.boost"),
        ("opponent.demo_time", (1,), "clip[0,3]/3", "seconds", "opponent_car.demo_time_remaining"),
        ("opponent.surface_contact", (1,), "boolean", "car", "opponent_car.surface_contact"),
        ("opponent.boosting", (1,), "boolean", "car", "opponent_car.boosting"),
        ("opponent.supersonic", (1,), "boolean", "car", "opponent_car.supersonic"),
        ("opponent.handbrake", (1,), "clip[0,1]", "controller", "opponent_car.handbrake"),
        ("opponent.air_state", (5,), "one-hot", "car", "opponent_car.air_state"),
        ("opponent.jump_held", (1,), "boolean", "controller", "opponent_car.jump_held"),
        ("opponent.has_jumped", (1,), "boolean", "car resource", "opponent_car.has_jumped"),
        ("opponent.has_double_jumped", (1,), "boolean", "car resource", "opponent_car.has_double_jumped"),
        ("opponent.has_dodged", (1,), "boolean", "car resource", "opponent_car.has_dodged"),
        ("opponent.can_dodge", (1,), "boolean", "car resource", "opponent_car.can_dodge"),
        ("opponent.air_time", (1,), "clip[0,5]/5", "seconds", "opponent_car.air_time"),
        ("opponent.jump_hold_elapsed", (1,), "clip[0,0.2]/0.2", "seconds", "opponent_car.jump_hold_elapsed"),
        ("opponent.dodge_window_remaining", (1,), "clip[0,1.45]/1.45", "seconds", "opponent_car.dodge_window_remaining"),
        ("opponent.dodge_elapsed", (1,), "clip[0,0.95]/0.95", "seconds", "opponent_car.dodge_elapsed"),
        ("opponent.dodge_direction", (2,), "unit vector", "canonical flip frame", "opponent_car.dodge_direction"),
        ("opponent.latest_controller", (8,), "native bounds", "physical controller", "opponent_car.latest_controller"),
        ("opponent.ball_local", (2, 3), "position/6000; velocity/2300", "opponent car local", "shared relative transform"),
        ("opponent.surface_distances", (5,), "nonnegative floor/ceiling/side/goal-aware-back/corner clearances divided by physical extents", "team", "shared RLBot-v5 planar standard-Soccar helper; curved ramps/posts approximated"),
        ("opponent.nearest_surface_normal", (3,), "unit vector", "opponent car local", "shared RLBot-v5 planar standard-Soccar helper"),
        ("opponent.surface_up_alignment", (1,), "dot product", "opponent car local", "shared RLBot-v5 planar standard-Soccar helper"),
        ("opponent.surface_signed_velocity", (1,), "divide 2300", "nearest surface normal", "shared RLBot-v5 planar standard-Soccar helper"),
        ("opponent.goal_side_of_ball", (1,), "boolean", "Rival perspective", "shared team-frame comparison"),
    ):
        schema.add(name, block, shape, normalization, frame, source)

    block = "ball_goal"
    for name, shape, normalization, frame, source in (
        ("ball.position", (3,), "divide field XYZ extents", "team", "ball.position"),
        ("ball.linear_velocity", (3,), "divide 2300", "team", "ball.linear_velocity"),
        ("ball.angular_velocity", (3,), "divide 5.5", "team", "ball.angular_velocity"),
        ("ball.self_local", (2, 3), "position/6000; velocity/2300", "Rival car local", "shared relative transform"),
        ("ball.opponent_local", (2, 3), "position/6000; velocity/2300", "opponent car local", "shared relative transform"),
        ("ball.speed_distances_closing", (5,), "speed/2300; distances/6000; closing/2300", "scalar", "shared relative math"),
        ("ball.goal_centers", (2, 3), "divide 6000", "team relative", "shared documented physical standard-Soccar goal centers"),
        ("ball.goal_posts", (4, 3), "divide 6000", "team relative", "shared documented physical standard-Soccar goal centers and widths"),
    ):
        schema.add(name, block, shape, normalization, frame, source)

    schema.add(
        "prediction.horizons",
        "prediction",
        (PREDICTION_COUNT, PREDICTION_FEATURES),
        "team position/field; team velocity/2300; local position/6000; local velocity/2300",
        "team and Rival car local",
        "shared RocketSim BallPredictor",
        "cached refresh every configured 1/2/4 ticks; age emitted every tick",
        "cache cleared on episode/reset discontinuity",
    )
    schema.add(
        "boost_pads.entities",
        "boost_pads",
        (PAD_COUNT, PAD_FEATURES),
        "XY/field; relative/6000; ball distance/6000; flags; time/10",
        "team and Rival car local",
        "canonical fixed-order pads",
    )
    schema.add(
        "history.self_controllers",
        "controller_history",
        (HISTORY_TICKS, CONTROLLER_SIZE),
        "native controller bounds",
        "physical controller",
        "applied self controllers",
        "every physics tick",
        "zero-filled ring buffer on episode reset",
    )
    schema.add(
        "history.opponent_controllers",
        "controller_history",
        (HISTORY_TICKS, CONTROLLER_SIZE),
        "native controller bounds",
        "physical controller",
        "applied opponent controllers",
        "every physics tick",
        "zero-filled ring buffer on episode reset",
    )
    schema.add(
        "motion.one_tick_deltas",
        "motion_delta",
        (3, 6),
        "linear delta/2300; angular delta/5.5",
        "team",
        "shared previous canonical physics",
        "every physics tick",
        "zero on reset/discontinuity",
    )
    return schema.fields


SCHEMA_FIELDS = _build_schema_fields()
OBSERVATION_SIZE = SCHEMA_FIELDS[-1].end
if OBSERVATION_SIZE != 714:
    raise AssertionError(f"RivalObsV1 schema drifted from 714 to {OBSERVATION_SIZE}")


def _canonical_source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def observation_schema_manifest() -> dict[str, Any]:
    block_slices: dict[str, dict[str, int]] = {}
    for field in SCHEMA_FIELDS:
        if field.block not in block_slices:
            block_slices[field.block] = {"start": field.start, "end": field.end}
        else:
            block_slices[field.block]["end"] = field.end
    source = Path(__file__)
    canonical_source = source.with_name("v9_canonical.py")
    geometry_source = source.with_name("v9_soccar_geometry.py")
    payload: dict[str, Any] = {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "canonical_state_version": CANONICAL_STATE_VERSION,
        "canonical_adapter_version": CANONICAL_ADAPTER_VERSION,
        "float_count": OBSERVATION_SIZE,
        "dtype": "float32",
        "running_standardization": False,
        "team_frame": "Rival always attacks positive Y; no state-dependent X mirror",
        "standard_soccar_geometry": {
            "version": GEOMETRY_VERSION,
            "authority": USEFUL_GAME_VALUES_URL,
            "surface_scope": (
                "Exact documented planes, 45-degree corner segment, and rectangular goal "
                "recess; curved ramps/posts are not represented as exact mesh clearances."
            ),
        },
        "fields": [asdict(field) for field in SCHEMA_FIELDS],
        "block_slices": block_slices,
        "entity_shapes": {
            "boost_pads": [PAD_COUNT, PAD_FEATURES],
            "prediction": [PREDICTION_COUNT, PREDICTION_FEATURES],
            "self_controller_history": [HISTORY_TICKS, CONTROLLER_SIZE],
            "opponent_controller_history": [HISTORY_TICKS, CONTROLLER_SIZE],
            "motion_deltas": [3, 6],
        },
        "model_groups": {
            "core_blocks": ["match_control", "self_car", "opponent_car", "ball_goal", "motion_delta"],
            "pad_block": "boost_pads",
            "prediction_block": "prediction",
            "history_block": "controller_history",
        },
        "builder_source_sha256": _canonical_source_sha256(source),
        "canonical_source_sha256": _canonical_source_sha256(canonical_source),
        "geometry_source_sha256": _canonical_source_sha256(geometry_source),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["schema_sha256"] = hashlib.sha256(serialized).hexdigest()
    return payload


def _one_hot(index: int, size: int) -> np.ndarray:
    result = np.zeros(size, dtype=np.float32)
    result[int(index)] = 1.0
    return result


def _clip_scale(value: Any, minimum: float, maximum: float, scale: float) -> np.ndarray:
    return np.asarray([np.clip(float(value), minimum, maximum) / scale], dtype=np.float32)


def _closing_speed(
    origin_position: np.ndarray,
    origin_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray,
) -> float:
    displacement = target_position - origin_position
    distance = float(np.linalg.norm(displacement))
    if distance <= 1e-6:
        return 0.0
    return float(np.dot(origin_velocity - target_velocity, displacement / distance))


def _corner_distance(position: np.ndarray) -> float:
    return _corner_distance_and_normal(position)[0]


def _corner_distance_and_normal(position: np.ndarray) -> tuple[float, np.ndarray]:
    point = np.abs(np.asarray(position[:2], dtype=np.float64))
    start = np.asarray(
        [SIDE_WALL_X - CORNER_ENDPOINT_OFFSET, BACK_WALL_Y], dtype=np.float64
    )
    end = np.asarray(
        [SIDE_WALL_X, BACK_WALL_Y - CORNER_ENDPOINT_OFFSET], dtype=np.float64
    )
    segment = end - start
    raw_parameter = float(np.dot(point - start, segment) / np.dot(segment, segment))
    parameter = float(np.clip(raw_parameter, 0.0, 1.0))
    closest = start + parameter * segment
    inward = point - closest
    length = float(np.linalg.norm(inward))
    penetrates_corner_plane = (
        0.0 <= raw_parameter <= 1.0
        and float(point[0] + point[1]) >= SIDE_WALL_X + BACK_WALL_Y - CORNER_ENDPOINT_OFFSET
    )
    if penetrates_corner_plane or length <= 1e-9:
        inward = np.asarray([-1.0, -1.0], dtype=np.float64) / math.sqrt(2.0)
        if penetrates_corner_plane:
            length = 0.0
    else:
        inward /= length
    inward *= np.asarray(
        [math.copysign(1.0, float(position[0]) or 1.0), math.copysign(1.0, float(position[1]) or 1.0)],
        dtype=np.float64,
    )
    return max(0.0, length), np.asarray([inward[0], inward[1], 0.0], dtype=np.float32)


def _surface_candidates(position: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """Analytic inward-facing standard-Soccar surfaces, including goal recesses."""

    x, y, z = (float(value) for value in position)
    sign_x = math.copysign(1.0, x or 1.0)
    sign_y = math.copysign(1.0, y or 1.0)
    corner_distance, corner_normal = _corner_distance_and_normal(position)
    candidates = [
        (max(0.0, z), np.asarray([0.0, 0.0, 1.0], dtype=np.float32)),
        (
            max(0.0, CEILING_Z - z),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        ),
        (
            max(0.0, SIDE_WALL_X - abs(x)),
            np.asarray([-sign_x, 0.0, 0.0], dtype=np.float32),
        ),
        (corner_distance, corner_normal),
    ]
    inside_goal_lane = abs(x) <= GOAL_HALF_WIDTH and z <= GOAL_HEIGHT
    back_extent = BACK_NET_Y if inside_goal_lane else BACK_WALL_Y
    candidates.append(
        (
            max(0.0, back_extent - abs(y)),
            np.asarray([0.0, -sign_y, 0.0], dtype=np.float32),
        )
    )
    if inside_goal_lane and abs(y) >= BACK_WALL_Y:
        candidates.extend(
            [
                (
                    max(0.0, GOAL_HALF_WIDTH - abs(x)),
                    np.asarray([-sign_x, 0.0, 0.0], dtype=np.float32),
                ),
                (
                    max(0.0, GOAL_HEIGHT - z),
                    np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                ),
            ]
        )
    return candidates


def _soccar_surface_distance(position: np.ndarray) -> float:
    """Nonnegative clearance to the nearest shared standard-Soccar surface."""

    return min(distance for distance, _normal in _surface_candidates(position))


def _soccar_surface_normal(position: np.ndarray) -> np.ndarray:
    """Inward normal of the nearest shared standard-Soccar surface."""

    return min(_surface_candidates(position), key=lambda item: item[0])[1]


def _surface_features(physics) -> tuple[np.ndarray, np.ndarray, float, float]:
    position = physics.position
    inside_goal_lane = (
        abs(float(position[0])) <= GOAL_HALF_WIDTH
        and float(position[2]) <= GOAL_HEIGHT
    )
    back_extent = BACK_NET_Y if inside_goal_lane else BACK_WALL_Y
    distances = np.asarray(
        [
            max(0.0, float(position[2])),
            max(0.0, CEILING_Z - float(position[2])),
            max(0.0, SIDE_WALL_X - abs(float(position[0]))),
            max(0.0, back_extent - abs(float(position[1]))),
            _corner_distance(position),
        ],
        dtype=np.float32,
    )
    normal_world = _soccar_surface_normal(position)
    normal_local = physics.local(normal_world)
    alignment = float(np.dot(physics.up, normal_world))
    signed_velocity = float(np.dot(physics.linear_velocity, normal_world))
    normalized_distances = np.asarray(
        [
            distances[0] / CEILING_Z,
            distances[1] / CEILING_Z,
            distances[2] / SIDE_WALL_X,
            distances[3] / back_extent,
            distances[4] / SIDE_WALL_X,
        ],
        dtype=np.float32,
    )
    return normalized_distances, normal_local, alignment, signed_velocity / CAR_MAX_SPEED


def deterministic_intercept_time(car, ball) -> float:
    """Pure deployable kinematic proxy; no Wisp/process-global ETA cache."""
    displacement = ball.position - car.physics.position
    distance = float(np.linalg.norm(displacement))
    if distance <= 1e-6:
        return 0.0
    direction = displacement / distance
    projected_speed = max(0.0, float(np.dot(car.physics.linear_velocity, direction)))
    boost_fraction = float(np.clip(car.boost / 100.0, 0.0, 1.0))
    acceleration = 900.0 + 500.0 * boost_fraction
    discriminant = projected_speed * projected_speed + 2.0 * acceleration * distance
    drive_time = (-projected_speed + math.sqrt(max(0.0, discriminant))) / acceleration
    alignment = float(np.clip(np.dot(car.physics.forward, direction), -1.0, 1.0))
    turn_time = math.acos(alignment) / 3.5
    vertical_penalty = max(0.0, float(displacement[2]) - 150.0) / 1100.0
    demo_penalty = max(0.0, float(car.demo_time_remaining))
    return float(np.clip(max(drive_time, turn_time) + vertical_penalty + demo_penalty, 0.0, 10.0))


def _ensure_rocketsim(mesh_directory: str | Path | None) -> None:
    candidates: list[Path | None] = []
    if mesh_directory is not None:
        candidates.append(Path(mesh_directory).resolve())
    repository_meshes = Path(__file__).resolve().parents[2] / "bot" / "collision_meshes"
    if repository_meshes.is_dir():
        candidates.append(repository_meshes)
    candidates.append(None)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            if candidate is None:
                rs.init()
            else:
                rs.init(str(candidate))
            return
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        # Initialization is process-global.  A prior RocketSimEngine or RLBot
        # adapter may already own the initialized mesh registry even when a
        # repeated init call reports an error; predictor construction is the
        # final authoritative check.
        return


def _default_prediction_provider(ball) -> tuple[np.ndarray, np.ndarray]:
    predictor = rs.BallPredictor(rs.GameMode.SOCCAR)
    ball_state = rs.BallState(
        pos=rs.Vec(*ball.position),
        rot_mat=rs.RotMat(*ball.rotation_mtx.transpose().flatten()),
        vel=rs.Vec(*ball.linear_velocity),
        ang_vel=rs.Vec(*ball.angular_velocity),
    )
    prediction = predictor.get_ball_prediction(
        ball_state,
        0,
        max(PREDICTION_HORIZON_TICKS) + 1,
        1,
    )
    positions = np.empty((PREDICTION_COUNT, 3), dtype=np.float32)
    velocities = np.empty((PREDICTION_COUNT, 3), dtype=np.float32)
    for index, tick in enumerate(PREDICTION_HORIZON_TICKS):
        item = prediction[tick]
        positions[index] = (item.pos.x, item.pos.y, item.pos.z)
        velocities[index] = (item.vel.x, item.vel.y, item.vel.z)
    return positions, velocities


class RivalObsV1Builder:
    """Stateful shared builder consuming only ``RivalCanonicalStateV1``."""

    def __init__(
        self,
        *,
        prediction_refresh_ticks: int = 4,
        collision_mesh_directory: str | Path | None = None,
        prediction_provider: Callable[[Any], tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        if int(prediction_refresh_ticks) not in {1, 2, 4}:
            raise ValueError("Prediction refresh must be one of 1, 2 or 4 physics ticks")
        self.prediction_refresh_ticks = int(prediction_refresh_ticks)
        _ensure_rocketsim(collision_mesh_directory)
        self._prediction_provider = prediction_provider or _default_prediction_provider
        self.last_timings: dict[str, float | bool] = {}
        self.reset()

    def reset(self) -> None:
        zero = np.zeros(CONTROLLER_SIZE, dtype=np.float32)
        self.self_history: deque[np.ndarray] = deque(
            [zero.copy() for _ in range(HISTORY_TICKS)], maxlen=HISTORY_TICKS
        )
        self.opponent_history: deque[np.ndarray] = deque(
            [zero.copy() for _ in range(HISTORY_TICKS)], maxlen=HISTORY_TICKS
        )
        self.previous_motion: np.ndarray | None = None
        self.motion_delta = np.zeros((3, 6), dtype=np.float32)
        self.prediction_tick: int | None = None
        self.prediction_positions = np.zeros((PREDICTION_COUNT, 3), dtype=np.float32)
        self.prediction_velocities = np.zeros((PREDICTION_COUNT, 3), dtype=np.float32)
        self.last_tick: int | None = None

    @staticmethod
    def _motion(state: RivalCanonicalStateV1) -> np.ndarray:
        return np.asarray(
            [
                np.concatenate(
                    (state.self_car.physics.linear_velocity, state.self_car.physics.angular_velocity)
                ),
                np.concatenate(
                    (
                        state.opponent_car.physics.linear_velocity,
                        state.opponent_car.physics.angular_velocity,
                    )
                ),
                np.concatenate((state.ball.linear_velocity, state.ball.angular_velocity)),
            ],
            dtype=np.float32,
        )

    def _advance_runtime(self, state: RivalCanonicalStateV1) -> None:
        tick = int(state.tick_index)
        if self.last_tick is not None and tick < self.last_tick:
            self.reset()
        if self.last_tick == tick:
            return
        gap = 1 if self.last_tick is None else max(1, tick - self.last_tick)
        repeats = min(HISTORY_TICKS, gap)
        for _ in range(repeats):
            self.self_history.append(state.self_car.latest_controller.copy())
            self.opponent_history.append(state.opponent_car.latest_controller.copy())
        current_motion = self._motion(state)
        if self.previous_motion is None:
            self.motion_delta.fill(0.0)
        else:
            self.motion_delta = (current_motion - self.previous_motion) / float(gap)
        self.previous_motion = current_motion
        self.last_tick = tick

    def _prediction(self, state: RivalCanonicalStateV1) -> tuple[np.ndarray, np.ndarray, int, bool, float]:
        tick = int(state.tick_index)
        refresh = (
            self.prediction_tick is None
            or tick < self.prediction_tick
            or tick - self.prediction_tick >= self.prediction_refresh_ticks
        )
        predictor_seconds = 0.0
        if refresh:
            started = time.perf_counter()
            positions, velocities = self._prediction_provider(state.ball)
            predictor_seconds = time.perf_counter() - started
            self.prediction_positions = np.asarray(positions, dtype=np.float32).reshape(
                PREDICTION_COUNT, 3
            )
            self.prediction_velocities = np.asarray(velocities, dtype=np.float32).reshape(
                PREDICTION_COUNT, 3
            )
            if not np.isfinite(self.prediction_positions).all() or not np.isfinite(
                self.prediction_velocities
            ).all():
                raise FloatingPointError("Shared ball predictor returned a non-finite value")
            self.prediction_tick = tick
        age = max(0, tick - int(self.prediction_tick))
        return (
            self.prediction_positions,
            self.prediction_velocities,
            age,
            refresh,
            predictor_seconds,
        )

    def export_runtime_state(self) -> dict[str, Any]:
        return {
            "version": OBSERVATION_VERSION,
            "prediction_refresh_ticks": self.prediction_refresh_ticks,
            "self_history": np.stack(self.self_history).tolist(),
            "opponent_history": np.stack(self.opponent_history).tolist(),
            "previous_motion": (
                None if self.previous_motion is None else self.previous_motion.tolist()
            ),
            "motion_delta": self.motion_delta.tolist(),
            "prediction_tick": self.prediction_tick,
            "prediction_positions": self.prediction_positions.tolist(),
            "prediction_velocities": self.prediction_velocities.tolist(),
            "last_tick": self.last_tick,
        }

    def load_runtime_state(self, payload: Mapping[str, Any]) -> None:
        if payload.get("version") != OBSERVATION_VERSION:
            raise ValueError(f"Unexpected observation runtime version {payload.get('version')!r}")
        if int(payload["prediction_refresh_ticks"]) != self.prediction_refresh_ticks:
            raise ValueError("Prediction refresh cadence differs from serialized runtime state")
        self.self_history = deque(
            _array_rows(payload["self_history"], (HISTORY_TICKS, CONTROLLER_SIZE)),
            maxlen=HISTORY_TICKS,
        )
        self.opponent_history = deque(
            _array_rows(payload["opponent_history"], (HISTORY_TICKS, CONTROLLER_SIZE)),
            maxlen=HISTORY_TICKS,
        )
        previous = payload["previous_motion"]
        self.previous_motion = (
            None if previous is None else np.asarray(previous, dtype=np.float32).reshape(3, 6)
        )
        self.motion_delta = np.asarray(payload["motion_delta"], dtype=np.float32).reshape(3, 6)
        self.prediction_tick = (
            None if payload["prediction_tick"] is None else int(payload["prediction_tick"])
        )
        self.prediction_positions = np.asarray(
            payload["prediction_positions"], dtype=np.float32
        ).reshape(PREDICTION_COUNT, 3)
        self.prediction_velocities = np.asarray(
            payload["prediction_velocities"], dtype=np.float32
        ).reshape(PREDICTION_COUNT, 3)
        self.last_tick = None if payload["last_tick"] is None else int(payload["last_tick"])

    def snapshot_payload(self, state: RivalCanonicalStateV1) -> dict[str, Any]:
        return {
            "canonical_state": state.to_payload(),
            "observation_runtime_before_build": self.export_runtime_state(),
            "observation_schema_sha256": observation_schema_manifest()["schema_sha256"],
        }

    def build(self, state: RivalCanonicalStateV1) -> np.ndarray:
        if state.version != CANONICAL_STATE_VERSION:
            raise ValueError(f"Expected {CANONICAL_STATE_VERSION}, got {state.version}")
        started = time.perf_counter()
        self._advance_runtime(state)
        prediction_positions, prediction_velocities, prediction_age, refreshed, predictor_seconds = self._prediction(state)
        emitted: list[tuple[str, np.ndarray]] = []

        def emit(name: str, value: Any) -> None:
            emitted.append((name, np.asarray(value, dtype=np.float32).reshape(-1)))

        self_intercept = deterministic_intercept_time(state.self_car, state.ball)
        opponent_intercept = deterministic_intercept_time(state.opponent_car, state.ball)
        emit("match.score_diff", _clip_scale(state.score_diff, -5, 5, 5))
        emit("match.time_remaining", _clip_scale(state.game_time_remaining, 0, 300, 300))
        emit("match.overtime", [state.overtime])
        emit("match.kickoff", [state.kickoff])
        emit("match.active_play", [state.active_play])
        emit("match.gravity", [np.clip(state.gravity_z / 650.0, -3.0, 3.0)])
        emit("touch.self_age", _clip_scale(state.self_touch_age, 0, 10, 10))
        emit("touch.opponent_age", _clip_scale(state.opponent_touch_age, 0, 10, 10))
        emit("touch.last_toucher", _one_hot(state.last_toucher, 3))
        emit("prediction.age", _clip_scale(prediction_age, 0, 4, 4))
        emit("intercept.self", _clip_scale(self_intercept, 0, 10, 10))
        emit("intercept.opponent", _clip_scale(opponent_intercept, 0, 10, 10))
        emit("intercept.advantage", _clip_scale(opponent_intercept - self_intercept, -10, 10, 10))

        self._emit_self_car(emit, state)
        self._emit_opponent_car(emit, state)
        self._emit_ball(emit, state)

        prediction_rows = []
        for position, velocity in zip(prediction_positions, prediction_velocities):
            relative_position = state.self_car.physics.local(position - state.self_car.physics.position)
            relative_velocity = state.self_car.physics.local(velocity - state.self_car.physics.linear_velocity)
            prediction_rows.append(
                np.concatenate(
                    (
                        position / FIELD_SCALE,
                        velocity / CAR_MAX_SPEED,
                        relative_position / VECTOR_SCALE,
                        relative_velocity / CAR_MAX_SPEED,
                    )
                )
            )
        emit("prediction.horizons", np.asarray(prediction_rows, dtype=np.float32))

        pad_rows = []
        for index in range(PAD_COUNT):
            position = state.pad_positions[index]
            relative = state.self_car.physics.local(position - state.self_car.physics.position)
            pad_rows.append(
                np.concatenate(
                    (
                        position[:2] / FIELD_SCALE[:2],
                        relative / VECTOR_SCALE,
                        [float(np.linalg.norm(position - state.ball.position)) / VECTOR_SCALE],
                        [state.pad_is_big[index]],
                        [state.pad_active[index]],
                        [np.clip(state.pad_time_until_active[index], 0.0, 10.0) / 10.0],
                    )
                )
            )
        emit("boost_pads.entities", np.asarray(pad_rows, dtype=np.float32))
        emit("history.self_controllers", np.stack(self.self_history))
        emit("history.opponent_controllers", np.stack(self.opponent_history))
        normalized_delta = self.motion_delta.copy()
        normalized_delta[:, :3] /= CAR_MAX_SPEED
        normalized_delta[:, 3:] /= ANGULAR_SPEED_SCALE
        emit("motion.one_tick_deltas", normalized_delta)

        if len(emitted) != len(SCHEMA_FIELDS):
            raise AssertionError(
                f"Builder emitted {len(emitted)} fields for {len(SCHEMA_FIELDS)} schema fields"
            )
        values: list[np.ndarray] = []
        for (name, value), field in zip(emitted, SCHEMA_FIELDS):
            if name != field.name:
                raise AssertionError(f"Observation emission order drift: {name} != {field.name}")
            if value.size != field.end - field.start:
                raise AssertionError(
                    f"Observation field {name} emitted {value.size}, expected {field.end - field.start}"
                )
            values.append(value)
        observation = np.ascontiguousarray(np.concatenate(values), dtype=np.float32)
        if observation.shape != (OBSERVATION_SIZE,):
            raise AssertionError(f"RivalObsV1 shape drift: {observation.shape}")
        if not np.isfinite(observation).all():
            indices = np.flatnonzero(~np.isfinite(observation))
            raise FloatingPointError(f"RivalObsV1 contains non-finite indices {indices[:20].tolist()}")
        total_seconds = time.perf_counter() - started
        self.last_timings = {
            "observation_seconds": total_seconds,
            "predictor_seconds": predictor_seconds,
            "prediction_refreshed": refreshed,
            "prediction_age_ticks": float(prediction_age),
        }
        return observation

    @staticmethod
    def _emit_self_car(emit, state: RivalCanonicalStateV1) -> None:
        car = state.self_car
        physics = car.physics
        emit("self.position", physics.position / FIELD_SCALE)
        emit("self.forward", physics.forward)
        emit("self.up", physics.up)
        emit("self.linear_velocity", physics.linear_velocity / CAR_MAX_SPEED)
        emit("self.angular_velocity", physics.angular_velocity / ANGULAR_SPEED_SCALE)
        emit("self.local_linear_velocity", physics.local(physics.linear_velocity) / CAR_MAX_SPEED)
        emit("self.local_angular_velocity", physics.local(physics.angular_velocity) / ANGULAR_SPEED_SCALE)
        speed = float(np.linalg.norm(physics.linear_velocity))
        emit("self.speed", [np.clip(speed, 0, CAR_MAX_SPEED) / CAR_MAX_SPEED])
        emit("self.signed_forward_speed", [np.clip(np.dot(physics.linear_velocity, physics.forward), -CAR_MAX_SPEED, CAR_MAX_SPEED) / CAR_MAX_SPEED])
        emit("self.boost", [np.clip(car.boost, 0, 100) / 100])
        emit("self.demo_time", [np.clip(car.demo_time_remaining, 0, 3) / 3])
        emit("self.surface_contact", [car.surface_contact])
        emit("self.boosting", [car.boosting])
        emit("self.supersonic", [car.supersonic])
        emit("self.handbrake", [np.clip(car.handbrake, 0, 1)])
        emit("self.air_state", _one_hot(car.air_state, len(AIR_STATES)))
        emit("self.jump_held", [car.jump_held])
        emit("self.has_jumped", [car.has_jumped])
        emit("self.has_double_jumped", [car.has_double_jumped])
        emit("self.has_dodged", [car.has_dodged])
        emit("self.can_dodge", [car.can_dodge])
        emit("self.air_time", [np.clip(car.air_time, 0, 5) / 5])
        emit("self.jump_hold_elapsed", [np.clip(car.jump_hold_elapsed, 0, 0.2) / 0.2])
        emit("self.dodge_window_remaining", [np.clip(car.dodge_window_remaining, 0, 1.45) / 1.45])
        emit("self.dodge_elapsed", [np.clip(car.dodge_elapsed, 0, 0.95) / 0.95])
        emit("self.dodge_direction", car.dodge_direction)
        distances, normal, alignment, signed_velocity = _surface_features(physics)
        emit("self.surface_distances", distances)
        emit("self.nearest_surface_normal", normal)
        emit("self.surface_up_alignment", [alignment])
        emit("self.surface_signed_velocity", [signed_velocity])
        emit(
            "self.goal_centers_local",
            np.stack(
                (
                    physics.local(state.goal_centers[0] - physics.position) / VECTOR_SCALE,
                    physics.local(state.goal_centers[1] - physics.position) / VECTOR_SCALE,
                )
            ),
        )

    @staticmethod
    def _emit_opponent_car(emit, state: RivalCanonicalStateV1) -> None:
        rival = state.self_car.physics
        car = state.opponent_car
        physics = car.physics
        emit("opponent.position", physics.position / FIELD_SCALE)
        emit("opponent.forward", physics.forward)
        emit("opponent.up", physics.up)
        emit("opponent.linear_velocity", physics.linear_velocity / CAR_MAX_SPEED)
        emit("opponent.angular_velocity", physics.angular_velocity / ANGULAR_SPEED_SCALE)
        emit("opponent.relative_position", rival.local(physics.position - rival.position) / VECTOR_SCALE)
        emit("opponent.relative_velocity", rival.local(physics.linear_velocity - rival.linear_velocity) / CAR_MAX_SPEED)
        speed = float(np.linalg.norm(physics.linear_velocity))
        emit("opponent.speed", [np.clip(speed, 0, CAR_MAX_SPEED) / CAR_MAX_SPEED])
        emit("opponent.signed_forward_speed", [np.clip(np.dot(physics.linear_velocity, physics.forward), -CAR_MAX_SPEED, CAR_MAX_SPEED) / CAR_MAX_SPEED])
        emit("opponent.boost", [np.clip(car.boost, 0, 100) / 100])
        emit("opponent.demo_time", [np.clip(car.demo_time_remaining, 0, 3) / 3])
        emit("opponent.surface_contact", [car.surface_contact])
        emit("opponent.boosting", [car.boosting])
        emit("opponent.supersonic", [car.supersonic])
        emit("opponent.handbrake", [np.clip(car.handbrake, 0, 1)])
        emit("opponent.air_state", _one_hot(car.air_state, len(AIR_STATES)))
        emit("opponent.jump_held", [car.jump_held])
        emit("opponent.has_jumped", [car.has_jumped])
        emit("opponent.has_double_jumped", [car.has_double_jumped])
        emit("opponent.has_dodged", [car.has_dodged])
        emit("opponent.can_dodge", [car.can_dodge])
        emit("opponent.air_time", [np.clip(car.air_time, 0, 5) / 5])
        emit("opponent.jump_hold_elapsed", [np.clip(car.jump_hold_elapsed, 0, 0.2) / 0.2])
        emit("opponent.dodge_window_remaining", [np.clip(car.dodge_window_remaining, 0, 1.45) / 1.45])
        emit("opponent.dodge_elapsed", [np.clip(car.dodge_elapsed, 0, 0.95) / 0.95])
        emit("opponent.dodge_direction", car.dodge_direction)
        emit("opponent.latest_controller", car.latest_controller)
        emit(
            "opponent.ball_local",
            np.stack(
                (
                    physics.local(state.ball.position - physics.position) / VECTOR_SCALE,
                    physics.local(state.ball.linear_velocity - physics.linear_velocity) / CAR_MAX_SPEED,
                )
            ),
        )
        distances, normal, alignment, signed_velocity = _surface_features(physics)
        emit("opponent.surface_distances", distances)
        emit("opponent.nearest_surface_normal", normal)
        emit("opponent.surface_up_alignment", [alignment])
        emit("opponent.surface_signed_velocity", [signed_velocity])
        emit("opponent.goal_side_of_ball", [float(physics.position[1] >= state.ball.position[1])])

    @staticmethod
    def _emit_ball(emit, state: RivalCanonicalStateV1) -> None:
        ball = state.ball
        rival = state.self_car.physics
        opponent = state.opponent_car.physics
        emit("ball.position", ball.position / FIELD_SCALE)
        emit("ball.linear_velocity", ball.linear_velocity / CAR_MAX_SPEED)
        emit("ball.angular_velocity", ball.angular_velocity / ANGULAR_SPEED_SCALE)
        emit(
            "ball.self_local",
            np.stack(
                (
                    rival.local(ball.position - rival.position) / VECTOR_SCALE,
                    rival.local(ball.linear_velocity - rival.linear_velocity) / CAR_MAX_SPEED,
                )
            ),
        )
        emit(
            "ball.opponent_local",
            np.stack(
                (
                    opponent.local(ball.position - opponent.position) / VECTOR_SCALE,
                    opponent.local(ball.linear_velocity - opponent.linear_velocity) / CAR_MAX_SPEED,
                )
            ),
        )
        emit(
            "ball.speed_distances_closing",
            [
                np.linalg.norm(ball.linear_velocity) / CAR_MAX_SPEED,
                np.linalg.norm(ball.position - rival.position) / VECTOR_SCALE,
                _closing_speed(rival.position, rival.linear_velocity, ball.position, ball.linear_velocity) / CAR_MAX_SPEED,
                np.linalg.norm(ball.position - opponent.position) / VECTOR_SCALE,
                _closing_speed(opponent.position, opponent.linear_velocity, ball.position, ball.linear_velocity) / CAR_MAX_SPEED,
            ],
        )
        emit(
            "ball.goal_centers",
            np.stack(
                (
                    (state.goal_centers[0] - ball.position) / VECTOR_SCALE,
                    (state.goal_centers[1] - ball.position) / VECTOR_SCALE,
                )
            ),
        )
        own_half_width = float(state.goal_widths[0]) / 2.0
        opponent_half_width = float(state.goal_widths[1]) / 2.0
        own_left_post = state.goal_centers[0] + np.asarray(
            [-own_half_width, 0.0, 0.0], dtype=np.float32
        )
        own_right_post = state.goal_centers[0] + np.asarray(
            [own_half_width, 0.0, 0.0], dtype=np.float32
        )
        opponent_left_post = state.goal_centers[1] + np.asarray(
            [-opponent_half_width, 0.0, 0.0], dtype=np.float32
        )
        opponent_right_post = state.goal_centers[1] + np.asarray(
            [opponent_half_width, 0.0, 0.0], dtype=np.float32
        )
        emit(
            "ball.goal_posts",
            np.stack(
                (
                    (own_left_post - ball.position) / VECTOR_SCALE,
                    (own_right_post - ball.position) / VECTOR_SCALE,
                    (opponent_left_post - ball.position) / VECTOR_SCALE,
                    (opponent_right_post - ball.position) / VECTOR_SCALE,
                )
            ),
        )


def _array_rows(value: Any, shape: tuple[int, int]) -> list[np.ndarray]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"Serialized runtime array expected {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Serialized runtime array contains non-finite values")
    return [row.copy() for row in array]
