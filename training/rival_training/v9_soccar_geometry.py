"""Authoritative standard-Soccar geometry shared by Rival v9.

The primary numeric source is RLBot v5's ``Useful game values`` page:
https://wiki.rlbot.org/v5/botmaking/useful-game-values/

RLBot's v5 ``FieldInfo`` remains the runtime authority for map-specific goal
identity/volume metadata and boost-pad data:
https://github.com/RLBot/flatbuffers-schema/blob/main/schema/gamedata.fbs

The page explicitly says that the approximately 256-uu wall-bottom ramps are
not circular.  Consequently, the constants below support exact planar Soccar
features and a bounded analytic corner/goal helper, not an exact collision-mesh
distance claim.  Exact curved collision geometry lives in the map collision
meshes described by:
https://wiki.rlbot.org/v5/miscellaneous/extracting-map-meshes/
"""

from __future__ import annotations

from typing import Any

import numpy as np


GEOMETRY_VERSION = "RLBotV5StandardSoccarGeometryV1"
USEFUL_GAME_VALUES_URL = "https://wiki.rlbot.org/v5/botmaking/useful-game-values/"
GAME_DATA_URL = "https://wiki.rlbot.org/v5/botmaking/game-data/"
FLATBUFFER_SCHEMA_URL = (
    "https://github.com/RLBot/flatbuffers-schema/blob/main/schema/gamedata.fbs"
)
COLLISION_MESH_URL = (
    "https://wiki.rlbot.org/v5/miscellaneous/extracting-map-meshes/"
)

FLOOR_Z = 0.0
SIDE_WALL_X = 4096.0
SIDE_WALL_LENGTH = 7936.0
BACK_WALL_Y = 5120.0
BACK_WALL_LENGTH = 5888.0
CEILING_Z = 2044.0
GOAL_HEIGHT = 642.775
GOAL_HALF_WIDTH = 892.755
GOAL_WIDTH = 2.0 * GOAL_HALF_WIDTH
GOAL_DEPTH = 880.0
BACK_NET_Y = BACK_WALL_Y + GOAL_DEPTH
GOAL_CENTER_Z = GOAL_HEIGHT / 2.0
CORNER_WALL_LENGTH = 1629.174
CORNER_PLANE_INTERCEPT = 8064.0
# The endpoint offset from either full wall extent is
# 5120 - (8064 - 4096) = 1152.
CORNER_ENDPOINT_OFFSET = BACK_WALL_Y - (CORNER_PLANE_INTERCEPT - SIDE_WALL_X)
WALL_BOTTOM_RAMP_APPROX_RADIUS = 256.0
STANDARD_GRAVITY_MAGNITUDE = 650.0
BALL_RADIUS = 91.25
BALL_MAX_SPEED = 6000.0
BALL_MAX_ANGULAR_SPEED = 6.0
CAR_MAX_SPEED = 2300.0
CAR_SUPERSONIC_THRESHOLD = 2200.0
CAR_MAX_NO_BOOST_SPEED = 1410.0
CAR_MAX_ANGULAR_SPEED = 5.5
BOOST_CONSUMPTION_PER_SECOND = 33.3

BLUE_GOAL_CENTER = np.asarray(
    [0.0, -BACK_WALL_Y, GOAL_CENTER_Z], dtype=np.float32
)
ORANGE_GOAL_CENTER = np.asarray(
    [0.0, BACK_WALL_Y, GOAL_CENTER_Z], dtype=np.float32
)
BLUE_GOAL_BACK = np.asarray([0.0, -BACK_NET_Y, GOAL_CENTER_Z], dtype=np.float32)
ORANGE_GOAL_BACK = np.asarray([0.0, BACK_NET_Y, GOAL_CENTER_Z], dtype=np.float32)
STANDARD_GOAL_CENTERS = np.stack((BLUE_GOAL_CENTER, ORANGE_GOAL_CENTER))
STANDARD_GOAL_WIDTHS = np.asarray([GOAL_WIDTH, GOAL_WIDTH], dtype=np.float32)
STANDARD_GOAL_HEIGHTS = np.asarray([GOAL_HEIGHT, GOAL_HEIGHT], dtype=np.float32)

# RLBot's documented FieldInfo anchor coordinates, in RLBot order (ascending Y,
# then X).  Small-pad Z is rounded to 0.082 on the wiki; live FieldInfo commonly
# reports approximately 0.0820159912.  Identity mapping therefore uses XY.
STANDARD_PAD_POSITIONS = np.asarray(
    [
        (0.0, -4240.0, 0.082),
        (-1792.0, -4184.0, 0.082),
        (1792.0, -4184.0, 0.082),
        (-3072.0, -4096.0, 8.0),
        (3072.0, -4096.0, 8.0),
        (-940.0, -3308.0, 0.082),
        (940.0, -3308.0, 0.082),
        (0.0, -2816.0, 0.082),
        (-3584.0, -2484.0, 0.082),
        (3584.0, -2484.0, 0.082),
        (-1788.0, -2302.0, 0.082),
        (1788.0, -2302.0, 0.082),
        (-2048.0, -1036.0, 0.082),
        (2048.0, -1036.0, 0.082),
        (0.0, -1024.0, 0.082),
        (-3584.0, 0.0, 8.0),
        (-1024.0, 0.0, 0.082),
        (1024.0, 0.0, 0.082),
        (3584.0, 0.0, 8.0),
        (0.0, 1024.0, 0.082),
        (-2048.0, 1036.0, 0.082),
        (2048.0, 1036.0, 0.082),
        (-1788.0, 2302.0, 0.082),
        (1788.0, 2302.0, 0.082),
        (-3584.0, 2484.0, 0.082),
        (3584.0, 2484.0, 0.082),
        (0.0, 2816.0, 0.082),
        (-940.0, 3308.0, 0.082),
        (940.0, 3308.0, 0.082),
        (-3072.0, 4096.0, 8.0),
        (3072.0, 4096.0, 8.0),
        (-1792.0, 4184.0, 0.082),
        (1792.0, 4184.0, 0.082),
        (0.0, 4240.0, 0.082),
    ],
    dtype=np.float32,
)
STANDARD_PAD_IS_BIG = np.asarray(
    [index in {3, 4, 15, 18, 29, 30} for index in range(34)], dtype=np.bool_
)

# RLGym 2.0's supported source order/orb-center representation.  This remains
# source-adapter data only: Rival's canonical order and positions are the RLBot
# FieldInfo anchors above.  A nearest-XY map reconciles the two public tables,
# including their few documented 2-uu differences.
ROCKETSIM_PAD_ORB_POSITIONS = np.asarray(
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
        (0.0, -1024.0, 70.0),
        (2048.0, -1036.0, 70.0),
        (-3584.0, 0.0, 73.0),
        (-1024.0, 0.0, 70.0),
        (1024.0, 0.0, 70.0),
        (3584.0, 0.0, 73.0),
        (-2048.0, 1036.0, 70.0),
        (0.0, 1024.0, 70.0),
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


def geometry_authority_manifest() -> dict[str, Any]:
    """Return compact, machine-readable provenance for the frozen constants."""

    return {
        "geometry_version": GEOMETRY_VERSION,
        "authority": {
            "standard_values": USEFUL_GAME_VALUES_URL,
            "runtime_data": GAME_DATA_URL,
            "runtime_schema": FLATBUFFER_SCHEMA_URL,
            "exact_collision_mesh_guidance": COLLISION_MESH_URL,
        },
        "coordinate_convention": {
            "units": "unreal units (uu)",
            "axes": "X is left; Y is longitudinal; Z is up",
            "blue_goal_direction": "negative Y",
            "yaw_zero": "positive X",
            "yaw_positive": "clockwise",
        },
        "standard_soccar": {
            "floor_z": FLOOR_Z,
            "side_wall_abs_x": SIDE_WALL_X,
            "side_wall_length": SIDE_WALL_LENGTH,
            "back_wall_abs_y": BACK_WALL_Y,
            "back_wall_length": BACK_WALL_LENGTH,
            "ceiling_z": CEILING_Z,
            "goal_height": GOAL_HEIGHT,
            "goal_half_width": GOAL_HALF_WIDTH,
            "goal_width_derived": GOAL_WIDTH,
            "goal_depth": GOAL_DEPTH,
            "goal_back_abs_y_derived": BACK_NET_Y,
            "corner_wall_length": CORNER_WALL_LENGTH,
            "corner_plane_axis_intercept": CORNER_PLANE_INTERCEPT,
            "corner_plane_per_quadrant": "abs(x) + abs(y) = 8064",
            "wall_bottom_ramp_radius_approximate": WALL_BOTTOM_RAMP_APPROX_RADIUS,
            "wall_bottom_ramp_exact": False,
        },
        "derived_standard_goal_centers": {
            "blue": BLUE_GOAL_CENTER.tolist(),
            "orange": ORANGE_GOAL_CENTER.tolist(),
            "basis": "documented physical back-wall opening and half goal height",
            "field_info_caveat": (
                "RLBot v5 beta may report a larger goal/scoring volume; captured runtime "
                "metadata is audited separately and is not treated as physical post geometry."
            ),
        },
        "standard_spawns": {
            "kickoff": {
                "blue": {
                    "right_corner": {"xy": [-2048.0, -2560.0], "yaw_pi": 0.25},
                    "left_corner": {"xy": [2048.0, -2560.0], "yaw_pi": 0.75},
                    "back_right": {"xy": [-256.0, -3840.0], "yaw_pi": 0.5},
                    "back_left": {"xy": [256.0, -3840.0], "yaw_pi": 0.5},
                    "far_back_center": {"xy": [0.0, -4608.0], "yaw_pi": 0.5},
                },
                "orange": {
                    "right_corner": {"xy": [2048.0, 2560.0], "yaw_pi": -0.75},
                    "left_corner": {"xy": [-2048.0, 2560.0], "yaw_pi": -0.25},
                    "back_right": {"xy": [256.0, 3840.0], "yaw_pi": -0.5},
                    "back_left": {"xy": [-256.0, 3840.0], "yaw_pi": -0.5},
                    "far_back_center": {"xy": [0.0, 4608.0], "yaw_pi": -0.5},
                },
            },
            "demolished": {
                "blue_xy": [
                    [-2304.0, -4608.0],
                    [-2688.0, -4608.0],
                    [2304.0, -4608.0],
                    [2688.0, -4608.0],
                ],
                "orange_xy": [
                    [2304.0, 4608.0],
                    [2688.0, 4608.0],
                    [-2304.0, 4608.0],
                    [-2688.0, 4608.0],
                ],
                "blue_yaw_pi": 0.5,
                "orange_yaw_pi": -0.5,
            },
        },
        "physics_values_used_by_rival": {
            "standard_gravity_magnitude": STANDARD_GRAVITY_MAGNITUDE,
            "ball_radius": BALL_RADIUS,
            "ball_max_speed": BALL_MAX_SPEED,
            "ball_max_angular_speed": BALL_MAX_ANGULAR_SPEED,
            "car_max_speed": CAR_MAX_SPEED,
            "car_supersonic_threshold": CAR_SUPERSONIC_THRESHOLD,
            "car_max_no_boost_speed": CAR_MAX_NO_BOOST_SPEED,
            "car_max_angular_speed": CAR_MAX_ANGULAR_SPEED,
            "boost_consumption_per_second": BOOST_CONSUMPTION_PER_SECOND,
        },
        "boost_pads": {
            "canonical_order": "RLBot FieldInfo: ascending y, then x",
            "count": int(STANDARD_PAD_POSITIONS.shape[0]),
            "small_count": int(np.count_nonzero(~STANDARD_PAD_IS_BIG)),
            "big_count": int(np.count_nonzero(STANDARD_PAD_IS_BIG)),
            "field_info_anchor_positions": STANDARD_PAD_POSITIONS.tolist(),
            "small_respawn_seconds": 4.0,
            "big_respawn_seconds": 10.0,
            "small_boost": 12.0,
            "big_boost": 100.0,
        },
        "scope_limit": (
            "Planar walls, 45-degree corner segment, and rectangular goal recess only; "
            "curved ramps/posts require collision-mesh queries for exact clearance."
        ),
    }
