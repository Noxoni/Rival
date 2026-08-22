from __future__ import annotations

import math
from typing import Any


def _number(value: Any) -> float | int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(converted):
        return None
    if isinstance(value, int):
        return int(value)
    return converted


def _vector(value: Any) -> dict[str, float | int | None] | None:
    if value is None:
        return None
    return {
        "x": _number(getattr(value, "x", None)),
        "y": _number(getattr(value, "y", None)),
        "z": _number(getattr(value, "z", None)),
    }


def _rotation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    pitch = _number(getattr(value, "pitch", None))
    yaw = _number(getattr(value, "yaw", None))
    roll = _number(getattr(value, "roll", None))
    record: dict[str, Any] = {"pitch": pitch, "yaw": yaw, "roll": roll}
    if all(isinstance(component, (float, int)) for component in (pitch, yaw, roll)):
        pitch_f, yaw_f, roll_f = float(pitch), float(yaw), float(roll)
        cp, sp = math.cos(pitch_f), math.sin(pitch_f)
        cy, sy = math.cos(yaw_f), math.sin(yaw_f)
        cr, sr = math.cos(roll_f), math.sin(roll_f)
        record["forward"] = {"x": cp * cy, "y": cp * sy, "z": sp}
        record["up"] = {
            "x": -cy * sp * cr - sy * sr,
            "y": -sy * sp * cr + cy * sr,
            "z": cp * cr,
        }
    return record


def _physics(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "position": _vector(getattr(value, "location", None)),
        "velocity": _vector(getattr(value, "velocity", None)),
        "rotation": _rotation(getattr(value, "rotation", None)),
        "angular_velocity": _vector(getattr(value, "angular_velocity", None)),
    }


def _controller(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "throttle": _number(getattr(value, "throttle", None)),
        "steer": _number(getattr(value, "steer", None)),
        "pitch": _number(getattr(value, "pitch", None)),
        "yaw": _number(getattr(value, "yaw", None)),
        "roll": _number(getattr(value, "roll", None)),
        "jump": bool(getattr(value, "jump", False)),
        "boost": bool(getattr(value, "boost", False)),
        "handbrake": bool(getattr(value, "handbrake", False)),
        "use_item": bool(getattr(value, "use_item", False)),
    }


def _touch(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "game_seconds": _number(getattr(value, "game_seconds", None)),
        "location": _vector(getattr(value, "location", None)),
        "normal": _vector(getattr(value, "normal", None)),
        "ball_index": _number(getattr(value, "ball_index", None)),
    }


def _score(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        field: _number(getattr(value, field, None))
        for field in (
            "score",
            "goals",
            "own_goals",
            "assists",
            "saves",
            "shots",
            "demolitions",
        )
    }


def _enum(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    try:
        raw = int(value)
    except (TypeError, ValueError):
        raw = None
    return {"name": str(value).split(".")[-1], "value": raw}


def _accolades(values: Any) -> list[Any]:
    if values is None:
        return []
    result: list[Any] = []
    try:
        iterator = iter(values)
    except TypeError:
        return [str(values)]
    for value in iterator:
        if isinstance(value, (str, bool, int, float)):
            result.append(value)
        else:
            result.append(str(value))
    return result


def extract_player(player: Any, packet_index: int) -> dict[str, Any]:
    """Serialize v5 PlayerInfo fields without assuming every runtime has them."""

    return {
        "packet_index": packet_index,
        "name": getattr(player, "name", None),
        "player_id": _number(getattr(player, "player_id", None)),
        "team": _number(getattr(player, "team", None)),
        "is_bot": bool(getattr(player, "is_bot", False)),
        "physics": _physics(getattr(player, "physics", None)),
        "boost": _number(getattr(player, "boost", None)),
        "is_supersonic": bool(getattr(player, "is_supersonic", False)),
        "last_input": _controller(getattr(player, "last_input", None)),
        "latest_touch": _touch(getattr(player, "latest_touch", None)),
        "air_state": _enum(getattr(player, "air_state", None)),
        "has_jumped": bool(getattr(player, "has_jumped", False)),
        "has_double_jumped": bool(getattr(player, "has_double_jumped", False)),
        "has_dodged": bool(getattr(player, "has_dodged", False)),
        "dodge_elapsed": _number(getattr(player, "dodge_elapsed", None)),
        "dodge_timeout": _number(getattr(player, "dodge_timeout", None)),
        "dodge_dir": _vector(getattr(player, "dodge_dir", None)),
        "demolished_timeout": _number(getattr(player, "demolished_timeout", None)),
        "score_info": _score(getattr(player, "score_info", None)),
        "accolades": _accolades(getattr(player, "accolades", None)),
    }


def _boost_pads(packet: Any, field_info: Any) -> list[dict[str, Any]]:
    dynamic = list(getattr(packet, "boost_pads", None) or [])
    static = list(getattr(field_info, "boost_pads", None) or []) if field_info else []
    result: list[dict[str, Any]] = []
    for index, state in enumerate(dynamic):
        pad = static[index] if index < len(static) else None
        result.append(
            {
                "index": index,
                "is_active": bool(getattr(state, "is_active", False)),
                "timer": _number(getattr(state, "timer", None)),
                "is_full_boost": (
                    bool(getattr(pad, "is_full_boost", False)) if pad is not None else None
                ),
                "location": _vector(getattr(pad, "location", None)),
            }
        )
    return result


def extract_packet_snapshot(
    packet: Any,
    self_index: int,
    field_info: Any = None,
) -> dict[str, Any]:
    """Extract analysis-only data directly from an RLBot v5 GamePacket."""

    players = list(getattr(packet, "players", None) or [])
    player_records = [extract_player(player, index) for index, player in enumerate(players)]
    self_team = (
        getattr(players[self_index], "team", None)
        if 0 <= self_index < len(players)
        else None
    )
    opponent_indices = [
        index
        for index, player in enumerate(players)
        if index != self_index and getattr(player, "team", None) != self_team
    ]
    balls = list(getattr(packet, "balls", None) or [])
    ball = balls[0] if balls else None
    match_info = getattr(packet, "match_info", None)
    teams = list(getattr(packet, "teams", None) or [])

    return {
        "self_index": self_index if 0 <= self_index < len(players) else None,
        "opponent_indices": opponent_indices,
        "players": player_records,
        "ball": None if ball is None else {"physics": _physics(getattr(ball, "physics", None))},
        "boost_pads": _boost_pads(packet, field_info),
        "match": {
            "seconds_elapsed": _number(getattr(match_info, "seconds_elapsed", None)),
            "game_time_remaining": _number(
                getattr(match_info, "game_time_remaining", None)
            ),
            "frame_num": _number(getattr(match_info, "frame_num", None)),
            "phase": _enum(getattr(match_info, "match_phase", None)),
            "is_overtime": bool(getattr(match_info, "is_overtime", False)),
            "game_speed": _number(getattr(match_info, "game_speed", None)),
            "scores": [
                {
                    "team": _number(getattr(team, "team_index", index)),
                    "score": _number(getattr(team, "score", None)),
                }
                for index, team in enumerate(teams)
            ],
        },
    }
