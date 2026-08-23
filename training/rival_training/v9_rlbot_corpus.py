"""RLBot-v5 telemetry reconstruction and source-audit helpers for Rival v9.

The historical Rival telemetry stores a lossless JSON projection of every
RLBot field consumed by ``RLBotCanonicalAdapterV1``.  This module reconstructs
duck-typed packet objects from that projection so the exact production adapter
and shared observation builder can be exercised without launching Rocket
League.  It deliberately does not depend on the RLBot Python package.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .v9_canonical import (
    AIR_STATE_INDEX,
    PHYSICS_HZ,
    STANDARD_GRAVITY_Z,
    TEAM_INVERSION,
    RivalCanonicalStateV1,
)
from .v9_soccar_geometry import (
    STANDARD_GOAL_CENTERS,
    STANDARD_GOAL_HEIGHTS,
    STANDARD_GOAL_WIDTHS,
)


NATIVE_CORPUS_VERSION = "RivalV9NativePacketCorpusV2"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return float(default)
    return converted if math.isfinite(converted) else float(default)


def _vec3(value: Mapping[str, Any] | None) -> SimpleNamespace:
    source = value or {}
    return SimpleNamespace(
        x=_number(source.get("x")),
        y=_number(source.get("y")),
        z=_number(source.get("z")),
    )


def _vec2(value: Mapping[str, Any] | None) -> SimpleNamespace:
    source = value or {}
    return SimpleNamespace(x=_number(source.get("x")), y=_number(source.get("y")))


def _rotation(value: Mapping[str, Any] | None) -> SimpleNamespace:
    source = value or {}
    return SimpleNamespace(
        pitch=_number(source.get("pitch")),
        yaw=_number(source.get("yaw")),
        roll=_number(source.get("roll")),
    )


def _physics(value: Mapping[str, Any] | None) -> SimpleNamespace:
    source = value or {}
    return SimpleNamespace(
        location=_vec3(source.get("position")),
        rotation=_rotation(source.get("rotation")),
        velocity=_vec3(source.get("velocity")),
        angular_velocity=_vec3(source.get("angular_velocity")),
    )


def _controller(value: Mapping[str, Any] | None) -> SimpleNamespace:
    source = value or {}
    return SimpleNamespace(
        throttle=_number(source.get("throttle")),
        steer=_number(source.get("steer")),
        pitch=_number(source.get("pitch")),
        yaw=_number(source.get("yaw")),
        roll=_number(source.get("roll")),
        jump=bool(source.get("jump", False)),
        boost=bool(source.get("boost", False)),
        handbrake=bool(source.get("handbrake", False)),
    )


def _touch(value: Mapping[str, Any] | None) -> SimpleNamespace | None:
    if value is None:
        return None
    return SimpleNamespace(
        game_seconds=_number(value.get("game_seconds")),
        location=_vec3(value.get("location")),
        normal=_vec3(value.get("normal")),
        ball_index=int(_number(value.get("ball_index"))),
    )


def _enum_name(value: Any, default: str) -> str:
    if isinstance(value, Mapping):
        return str(value.get("name") or default)
    if value is None:
        return default
    return str(value).split(".")[-1]


def snapshot_to_rlbot_sources(
    snapshot: Mapping[str, Any],
) -> tuple[SimpleNamespace, SimpleNamespace, int]:
    """Reconstruct a packet/field-info pair consumed by the real v9 adapter."""

    players = []
    for source in snapshot.get("players", []):
        players.append(
            SimpleNamespace(
                team=int(_number(source.get("team"))),
                physics=_physics(source.get("physics")),
                boost=_number(source.get("boost")),
                demolished_timeout=_number(source.get("demolished_timeout"), -1.0),
                is_supersonic=bool(source.get("is_supersonic", False)),
                air_state=_enum_name(source.get("air_state"), "InAir"),
                has_jumped=bool(source.get("has_jumped", False)),
                has_double_jumped=bool(source.get("has_double_jumped", False)),
                has_dodged=bool(source.get("has_dodged", False)),
                dodge_timeout=_number(source.get("dodge_timeout"), -1.0),
                dodge_elapsed=_number(source.get("dodge_elapsed")),
                dodge_dir=_vec2(source.get("dodge_dir")),
                last_input=_controller(source.get("last_input")),
                latest_touch=_touch(source.get("latest_touch")),
            )
        )

    ball_source = snapshot.get("ball") or {}
    pads = list(snapshot.get("boost_pads", []))
    dynamic_pads = [
        SimpleNamespace(
            is_active=bool(pad.get("is_active", False)),
            timer=_number(pad.get("timer")),
        )
        for pad in pads
    ]
    static_pads = [
        SimpleNamespace(
            location=_vec3(pad.get("location")),
            is_full_boost=bool(pad.get("is_full_boost", False)),
        )
        for pad in pads
    ]
    static_goals = [
        SimpleNamespace(
            team_num=int(_number(goal.get("team_num"), -1)),
            location=_vec3(goal.get("location")),
            direction=_vec3(goal.get("direction")),
            width=_number(goal.get("width")),
            height=_number(goal.get("height")),
        )
        for goal in snapshot.get("goals", [])
    ]
    match = snapshot.get("match") or {}
    teams = [
        SimpleNamespace(
            team_index=int(_number(team.get("team"), index)),
            score=int(_number(team.get("score"))),
        )
        for index, team in enumerate(match.get("scores", []))
    ]
    packet = SimpleNamespace(
        players=players,
        balls=[SimpleNamespace(physics=_physics(ball_source.get("physics")))],
        boost_pads=dynamic_pads,
        teams=teams,
        match_info=SimpleNamespace(
            seconds_elapsed=_number(match.get("seconds_elapsed")),
            game_time_remaining=_number(match.get("game_time_remaining")),
            frame_num=int(
                _number(
                    match.get("frame_num"),
                    round(_number(match.get("seconds_elapsed")) * PHYSICS_HZ),
                )
            ),
            is_overtime=bool(match.get("is_overtime", False)),
            match_phase=_enum_name(match.get("phase"), "Inactive"),
            world_gravity_z=_number(
                match.get("world_gravity_z"), STANDARD_GRAVITY_Z
            ),
            game_speed=_number(match.get("game_speed"), 1.0),
        ),
    )
    field_info = SimpleNamespace(boost_pads=static_pads, goals=static_goals)
    return packet, field_info, int(snapshot.get("self_index", 0))


@dataclass
class SourceAuditAccumulator:
    """Independent packet-to-canonical comparisons accumulated over a corpus."""

    comparisons: int = 0
    mismatches: int = 0
    maximum_abs_error: float = 0.0

    def compare_scalar(self, actual: Any, expected: Any, *, tolerance: float = 1e-5) -> None:
        left = float(actual)
        right = float(expected)
        self.comparisons += 1
        error = abs(left - right)
        self.maximum_abs_error = max(self.maximum_abs_error, error)
        if not math.isfinite(error) or error > tolerance:
            self.mismatches += 1

    def compare_array(self, actual: Any, expected: Any, *, tolerance: float = 1e-5) -> None:
        left = np.asarray(actual, dtype=np.float64).reshape(-1)
        right = np.asarray(expected, dtype=np.float64).reshape(-1)
        if left.shape != right.shape:
            self.comparisons += max(left.size, right.size, 1)
            self.mismatches += max(left.size, right.size, 1)
            self.maximum_abs_error = math.inf
            return
        errors = np.abs(left - right)
        self.comparisons += int(errors.size)
        if errors.size:
            maximum = float(np.max(errors))
            self.maximum_abs_error = max(self.maximum_abs_error, maximum)
            self.mismatches += int(np.count_nonzero(~np.isfinite(errors) | (errors > tolerance)))

    def to_record(self) -> dict[str, Any]:
        return {
            "comparisons": self.comparisons,
            "mismatches": self.mismatches,
            "maximum_abs_error": self.maximum_abs_error,
            "passed": self.mismatches == 0 and math.isfinite(self.maximum_abs_error),
        }


def _snapshot_vector(value: Mapping[str, Any] | None) -> np.ndarray:
    source = value or {}
    return np.asarray(
        [_number(source.get("x")), _number(source.get("y")), _number(source.get("z"))],
        dtype=np.float32,
    )


def _snapshot_controller(value: Mapping[str, Any] | None) -> np.ndarray:
    source = value or {}
    return np.asarray(
        [
            _number(source.get("throttle")),
            _number(source.get("steer")),
            _number(source.get("pitch")),
            _number(source.get("yaw")),
            _number(source.get("roll")),
            float(bool(source.get("jump", False))),
            float(bool(source.get("boost", False))),
            float(bool(source.get("handbrake", False))),
        ],
        dtype=np.float32,
    )


def audit_canonical_against_snapshot(
    canonical: RivalCanonicalStateV1,
    snapshot: Mapping[str, Any],
    audit: dict[str, SourceAuditAccumulator],
) -> None:
    """Compare adapter output with independently decoded telemetry source fields."""

    self_index = int(snapshot.get("self_index", 0))
    players = list(snapshot.get("players", []))
    self_source = players[self_index]
    self_team = int(_number(self_source.get("team")))
    opponent_source = next(
        player
        for index, player in enumerate(players)
        if index != self_index and int(_number(player.get("team"))) != self_team
    )
    invert = self_team == 1
    transform = TEAM_INVERSION if invert else np.ones(3, dtype=np.float32)
    match = snapshot.get("match") or {}

    audit["match"].compare_scalar(canonical.tick_index, match.get("frame_num", 0), tolerance=0)
    audit["match"].compare_scalar(canonical.seconds_elapsed, match.get("seconds_elapsed", 0))
    audit["match"].compare_scalar(
        canonical.game_time_remaining, match.get("game_time_remaining", 0)
    )
    scores = {
        int(_number(value.get("team"), index)): int(_number(value.get("score")))
        for index, value in enumerate(match.get("scores", []))
    }
    audit["match"].compare_scalar(
        canonical.score_diff, scores.get(self_team, 0) - scores.get(1 - self_team, 0), tolerance=0
    )
    audit["match"].compare_scalar(canonical.overtime, bool(match.get("is_overtime", False)), tolerance=0)
    phase = _enum_name(match.get("phase"), "Inactive")
    audit["match"].compare_scalar(
        canonical.kickoff, phase in {"Countdown", "Kickoff"}, tolerance=0
    )
    audit["match"].compare_scalar(
        canonical.active_play, phase in {"Kickoff", "Active"}, tolerance=0
    )

    for label, source, car in (
        ("self", self_source, canonical.self_car),
        ("opponent", opponent_source, canonical.opponent_car),
    ):
        physics = source.get("physics") or {}
        audit[f"{label}_physics"].compare_array(
            car.physics.position,
            _snapshot_vector(physics.get("position")) * transform,
        )
        audit[f"{label}_physics"].compare_array(
            car.physics.linear_velocity,
            _snapshot_vector(physics.get("velocity")) * transform,
        )
        audit[f"{label}_physics"].compare_array(
            car.physics.angular_velocity,
            _snapshot_vector(physics.get("angular_velocity")) * transform,
        )
        rotation = physics.get("rotation") or {}
        audit[f"{label}_physics"].compare_array(
            car.physics.forward,
            _snapshot_vector(rotation.get("forward")) * transform,
            tolerance=2e-5,
        )
        audit[f"{label}_physics"].compare_array(
            car.physics.up,
            _snapshot_vector(rotation.get("up")) * transform,
            tolerance=2e-5,
        )
        audit[f"{label}_controller"].compare_array(
            car.latest_controller, _snapshot_controller(source.get("last_input"))
        )
        audit[f"{label}_resources"].compare_scalar(car.boost, source.get("boost", 0))
        audit[f"{label}_resources"].compare_scalar(
            car.demo_time_remaining,
            max(0.0, _number(source.get("demolished_timeout"), -1.0)),
        )
        air_name = _enum_name(source.get("air_state"), "InAir")
        expected_air = AIR_STATE_INDEX.get(air_name, AIR_STATE_INDEX["InAir"])
        audit[f"{label}_air_dodge"].compare_scalar(car.air_state, expected_air, tolerance=0)
        audit[f"{label}_air_dodge"].compare_scalar(
            car.has_jumped, bool(source.get("has_jumped", False)), tolerance=0
        )
        audit[f"{label}_air_dodge"].compare_scalar(
            car.has_double_jumped,
            bool(source.get("has_double_jumped", False)),
            tolerance=0,
        )
        audit[f"{label}_air_dodge"].compare_scalar(
            car.has_dodged, bool(source.get("has_dodged", False)), tolerance=0
        )
        audit[f"{label}_air_dodge"].compare_scalar(
            car.dodge_elapsed, max(0.0, _number(source.get("dodge_elapsed")))
        )
        audit[f"{label}_air_dodge"].compare_array(
            car.dodge_direction,
            _snapshot_vector(source.get("dodge_dir"))[:2],
        )
        surface = air_name == "OnGround"
        controls = _snapshot_controller(source.get("last_input"))
        has_double = bool(source.get("has_double_jumped", False))
        has_dodged = bool(source.get("has_dodged", False))
        dodge_timeout = _number(source.get("dodge_timeout"), -1.0)
        flip_reset = (
            air_name == "InAir"
            and not bool(source.get("has_jumped", False))
            and not has_double
            and not has_dodged
        )
        expected_can_dodge = (
            not surface
            and not controls[5] > 0.5
            and not has_double
            and not has_dodged
            and (dodge_timeout >= 0.0 or flip_reset)
        )
        expected_window = (
            1.45
            if expected_can_dodge and flip_reset
            else max(0.0, dodge_timeout)
        )
        audit[f"{label}_air_dodge"].compare_scalar(
            car.can_dodge, expected_can_dodge, tolerance=0
        )
        audit[f"{label}_air_dodge"].compare_scalar(
            car.dodge_window_remaining, expected_window
        )

    ball_source = (snapshot.get("ball") or {}).get("physics") or {}
    audit["ball_physics"].compare_array(
        canonical.ball.position,
        _snapshot_vector(ball_source.get("position")) * transform,
    )
    audit["ball_physics"].compare_array(
        canonical.ball.linear_velocity,
        _snapshot_vector(ball_source.get("velocity")) * transform,
    )
    audit["ball_physics"].compare_array(
        canonical.ball.angular_velocity,
        _snapshot_vector(ball_source.get("angular_velocity")) * transform,
    )

    goals = list(snapshot.get("goals", []))
    if goals:
        goals_by_team = {
            int(_number(goal.get("team_num"), -1)): goal for goal in goals
        }
        audit["goals"].compare_array(canonical.goal_centers, STANDARD_GOAL_CENTERS)
        audit["goals"].compare_array(canonical.goal_widths, STANDARD_GOAL_WIDTHS)
        audit["goals"].compare_array(canonical.goal_heights, STANDARD_GOAL_HEIGHTS)
        for team in (0, 1):
            source = goals_by_team[team]
            audit["goals"].compare_array(
                _snapshot_vector(source.get("location"))[:2],
                STANDARD_GOAL_CENTERS[team, :2],
            )
            expected_direction = np.asarray(
                [0.0, 1.0 if team == 0 else -1.0, 0.0], dtype=np.float32
            )
            audit["goals"].compare_array(
                _snapshot_vector(source.get("direction")),
                expected_direction,
                tolerance=1e-5,
            )

    pads = list(snapshot.get("boost_pads", []))
    for index, pad in enumerate(pads):
        canonical_position = _snapshot_vector(pad.get("location")) * transform
        distances = np.sum(
            (canonical.pad_positions[:, :2] - canonical_position[:2]) ** 2, axis=1
        )
        canonical_index = int(np.argmin(distances))
        audit["boost_pads"].compare_scalar(
            canonical.pad_active[canonical_index], bool(pad.get("is_active", False)), tolerance=0
        )
        respawn = 10.0 if bool(pad.get("is_full_boost", False)) else 4.0
        expected_remaining = (
            0.0
            if bool(pad.get("is_active", False))
            else max(0.0, respawn - _number(pad.get("timer")))
        )
        audit["boost_pads"].compare_scalar(
            canonical.pad_time_until_active[canonical_index], expected_remaining
        )
        audit["boost_pads"].compare_scalar(
            canonical.pad_is_big[canonical_index],
            bool(pad.get("is_full_boost", False)),
            tolerance=0,
        )

    seconds = _number(match.get("seconds_elapsed"))
    touch_seconds = []
    for source in (self_source, opponent_source):
        touch = source.get("latest_touch")
        touch_seconds.append(
            None if touch is None else _number(touch.get("game_seconds"))
        )
    expected_ages = [
        10.0 if value is None else max(0.0, seconds - value)
        for value in touch_seconds
    ]
    audit["touch"].compare_scalar(canonical.self_touch_age, expected_ages[0])
    audit["touch"].compare_scalar(canonical.opponent_touch_age, expected_ages[1])
    if touch_seconds[0] is None and touch_seconds[1] is None:
        expected_toucher = 2
    elif touch_seconds[1] is None or (
        touch_seconds[0] is not None and touch_seconds[0] >= touch_seconds[1]
    ):
        expected_toucher = 0
    else:
        expected_toucher = 1
    audit["touch"].compare_scalar(canonical.last_toucher, expected_toucher, tolerance=0)


def packet_coverage(snapshot: Mapping[str, Any]) -> set[str]:
    """Return naturally observed Gate-3 state categories for one packet."""

    categories: set[str] = set()
    match = snapshot.get("match") or {}
    phase = _enum_name(match.get("phase"), "Inactive")
    if phase in {"Countdown", "Kickoff"}:
        categories.add("kickoff")
    remaining = _number(match.get("game_time_remaining"), 300.0)
    if remaining <= 30.0:
        categories.add("late_clock")
    if bool(match.get("is_overtime", False)):
        categories.add("overtime")

    # Both cars are actor inputs.  A state category is covered when it is
    # naturally present in either the self or opponent portion of the packet;
    # the source-field audit and canonical serialization cover both roles.
    for player in snapshot.get("players", []):
        physics = player.get("physics") or {}
        position = _snapshot_vector(physics.get("position"))
        rotation = physics.get("rotation") or {}
        up = _snapshot_vector(rotation.get("up"))
        air_name = _enum_name(player.get("air_state"), "InAir")
        controls = player.get("last_input") or {}
        boost = _number(player.get("boost"))
        if air_name == "OnGround" and phase == "Active":
            categories.add("normal_ground_play")
        if boost <= 10.0:
            categories.add("low_boost")
        if boost >= 80.0:
            categories.add("high_boost")
        touch = player.get("latest_touch")
        if touch is not None:
            age = _number(match.get("seconds_elapsed")) - _number(
                touch.get("game_seconds")
            )
            if -1e-3 <= age <= 2.0 / PHYSICS_HZ + 1e-3:
                categories.add("ball_contact")
        if air_name == "Jumping" and bool(controls.get("jump", False)):
            categories.add("first_jump_hold")
        if (
            air_name == "InAir"
            and bool(player.get("has_jumped", False))
            and not bool(player.get("has_double_jumped", False))
            and not bool(player.get("has_dodged", False))
            and not bool(controls.get("jump", False))
        ):
            categories.add("first_jump_release")
        if air_name == "DoubleJumping" or bool(player.get("has_double_jumped", False)):
            categories.add("double_jump")
        dodge_direction = _snapshot_vector(player.get("dodge_dir"))[:2]
        if air_name == "Dodging" and float(np.linalg.norm(dodge_direction)) > 0.2:
            categories.add("directional_dodge")
            vertical_direction = float(dodge_direction[1])
            pitch = _number(controls.get("pitch"))
            if abs(vertical_direction) > 0.2 and pitch * vertical_direction < -0.2:
                categories.add("flip_cancel")
        if (
            air_name == "InAir"
            and position[2] > 150.0
            and not bool(player.get("has_jumped", False))
            and not bool(player.get("has_double_jumped", False))
            and not bool(player.get("has_dodged", False))
        ):
            categories.add("airborne_reset_resource")
        wall_distance = min(
            4096.0 - abs(float(position[0])),
            5120.0 - abs(float(position[1])),
        )
        velocity = _snapshot_vector(physics.get("velocity"))
        if air_name == "OnGround" and position[2] > 100.0 and wall_distance < 180.0:
            categories.add("wall_contact")
        # PlayerInfo has no general collision-contact boolean.  A car center
        # within 80 uu of the 2044-uu ceiling while its normal velocity is
        # stalled is a bounded physical collision-envelope test.  It catches
        # the observed ceiling impact plateau without labeling high aerials.
        ceiling_gap = 2044.0 - float(position[2])
        if 0.0 <= ceiling_gap <= 80.0 and abs(float(velocity[2])) <= 60.0:
            categories.add("ceiling_contact")
        if air_name != "OnGround" and (
            up[2] < 0.35 or abs(float(position[0])) > 3900.0
        ):
            categories.add("awkward_recovery")
        if _number(player.get("demolished_timeout"), -1.0) >= 0.0:
            categories.add("demolition_respawn")
    return categories
