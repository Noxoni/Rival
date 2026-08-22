from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Sequence


def _vec3(value: Any) -> tuple[float, float, float]:
    return float(value.x), float(value.y), float(value.z)


def _vector_record(value: Any) -> dict[str, float]:
    x, y, z = _vec3(value)
    return {"x": x, "y": y, "z": z}


def _distance(first: Any, second: Any) -> float:
    ax, ay, az = _vec3(first)
    bx, by, bz = _vec3(second)
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2 + (bz - az) ** 2)


def _distance_2d(first: Any, second: Any) -> float:
    ax, ay, _ = _vec3(first)
    bx, by, _ = _vec3(second)
    return math.hypot(bx - ax, by - ay)


def _closing_velocity(
    source_position: Any,
    source_velocity: Any,
    target_position: Any,
    target_velocity: Any,
) -> float:
    sx, sy, sz = _vec3(source_position)
    svx, svy, svz = _vec3(source_velocity)
    tx, ty, tz = _vec3(target_position)
    tvx, tvy, tvz = _vec3(target_velocity)
    dx, dy, dz = tx - sx, ty - sy, tz - sz
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance <= 1e-6:
        return 0.0
    # Positive means the separation is shrinking.
    relative_x, relative_y, relative_z = svx - tvx, svy - tvy, svz - tvz
    return (relative_x * dx + relative_y * dy + relative_z * dz) / distance


def _constant_velocity_eta(distance: float, closing_velocity: float) -> float | None:
    if distance <= 1e-6:
        return 0.0
    if closing_velocity <= 1e-6:
        return None
    return distance / closing_velocity


def _action_record(action: Any | None) -> dict[str, float | bool] | None:
    if action is None:
        return None
    return {
        "throttle": float(action.throttle),
        "steer": float(action.steer),
        "pitch": float(action.pitch),
        "yaw": float(action.yaw),
        "roll": float(action.roll),
        "jump": bool(action.jump),
        "boost": bool(action.boost),
        "handbrake": bool(action.handbrake),
    }


@dataclass(frozen=True)
class TacticalMetrics:
    """Measurements only; none of these fields alter policy control."""

    self_boost: float
    opponent_boost: float | None
    ball_height: float
    ball_distance: float
    distance_self_ball: float
    distance_opponent_ball: float | None
    eta_self_ball: float | None
    eta_opponent_ball: float | None
    eta_method: str
    challenge_closing_velocity: float | None
    self_ball_closing_velocity: float
    opponent_ball_closing_velocity: float | None
    possession_eta_advantage: float | None
    self_airborne: bool
    opponent_airborne: bool | None
    selected_action_uses_boost: bool
    selected_action_uses_jump: bool
    selected_action_aerial_like: bool
    score_diff: int
    seconds_remaining: float | None

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def compute_tactical_metrics(
    state: Any,
    player: Any,
    opponent: Any | None,
    selected_action: Any,
    *,
    score_diff: int,
    seconds_remaining: float | None,
    eta_self_ball: float | None = None,
    eta_opponent_ball: float | None = None,
    eta_method: str = "constant_velocity_closing",
) -> TacticalMetrics:
    ball = state.ball
    self_distance = _distance(player.pos, ball.pos)
    self_ball_closing = _closing_velocity(
        player.pos, player.vel, ball.pos, ball.vel
    )

    opponent_distance: float | None = None
    opponent_ball_closing: float | None = None
    challenge_closing: float | None = None
    if opponent is not None:
        opponent_distance = _distance(opponent.pos, ball.pos)
        opponent_ball_closing = _closing_velocity(
            opponent.pos, opponent.vel, ball.pos, ball.vel
        )
        challenge_closing = _closing_velocity(
            player.pos, player.vel, opponent.pos, opponent.vel
        )

    if eta_self_ball is None:
        eta_self_ball = _constant_velocity_eta(self_distance, self_ball_closing)
    if opponent is not None and eta_opponent_ball is None:
        assert opponent_distance is not None
        assert opponent_ball_closing is not None
        eta_opponent_ball = _constant_velocity_eta(
            opponent_distance, opponent_ball_closing
        )

    possession_eta_advantage = None
    if eta_self_ball is not None and eta_opponent_ball is not None:
        possession_eta_advantage = eta_opponent_ball - eta_self_ball

    airborne = not bool(player.is_on_ground)
    action_uses_jump = bool(selected_action.jump)
    action_uses_boost = bool(selected_action.boost)
    rotational_air_input = any(
        abs(float(value)) > 1e-6
        for value in (
            selected_action.pitch,
            selected_action.yaw,
            selected_action.roll,
        )
    )
    aerial_like = action_uses_jump or (
        airborne and (action_uses_boost or rotational_air_input)
    )

    return TacticalMetrics(
        self_boost=float(player.boost),
        opponent_boost=None if opponent is None else float(opponent.boost),
        ball_height=float(ball.pos.z),
        ball_distance=self_distance,
        distance_self_ball=self_distance,
        distance_opponent_ball=opponent_distance,
        eta_self_ball=None if eta_self_ball is None else float(eta_self_ball),
        eta_opponent_ball=(
            None if eta_opponent_ball is None else float(eta_opponent_ball)
        ),
        eta_method=eta_method,
        challenge_closing_velocity=challenge_closing,
        self_ball_closing_velocity=self_ball_closing,
        opponent_ball_closing_velocity=opponent_ball_closing,
        possession_eta_advantage=possession_eta_advantage,
        self_airborne=airborne,
        opponent_airborne=(
            None if opponent is None else not bool(opponent.is_on_ground)
        ),
        selected_action_uses_boost=action_uses_boost,
        selected_action_uses_jump=action_uses_jump,
        selected_action_aerial_like=aerial_like,
        score_diff=int(score_diff),
        seconds_remaining=(
            None if seconds_remaining is None else float(seconds_remaining)
        ),
    )


def _player_snapshot(player: Any | None) -> dict[str, Any] | None:
    if player is None:
        return None
    supersonic = getattr(player, "is_supersonic", None)
    return {
        "index": int(getattr(player, "index", -1)),
        "team": getattr(getattr(player, "team", None), "name", None),
        "boost": float(player.boost),
        "position": _vector_record(player.pos),
        "velocity": _vector_record(player.vel),
        "supersonic": None if supersonic is None else bool(supersonic),
        "on_ground": bool(player.is_on_ground),
        "airborne": not bool(player.is_on_ground),
        "is_jumping": bool(getattr(player, "is_jumping", False)),
        "has_flip_or_jump": bool(player.has_flip_or_jump()),
        "previous_action": _action_record(getattr(player, "prev_action", None)),
    }


def _boost_snapshot(
    state: Any,
    player: Any,
    boost_locations: Sequence[Any] | None,
    nearby_limit: int,
) -> dict[str, Any]:
    active_values = [bool(value) for value in getattr(state, "boost_pads", [])]
    result: dict[str, Any] = {"active_pad_count": sum(active_values)}
    if boost_locations is None:
        result["active_pad_indices"] = [
            index for index, active in enumerate(active_values) if active
        ]
        return result

    active_large: list[int] = []
    active_small: list[int] = []
    opportunities: list[dict[str, Any]] = []
    for index, (active, location) in enumerate(
        zip(active_values, boost_locations)
    ):
        if not active:
            continue
        is_large = float(location.z) > 71.5
        (active_large if is_large else active_small).append(index)
        opportunities.append(
            {
                "index": index,
                "large": is_large,
                "distance_2d": _distance_2d(player.pos, location),
                "position": _vector_record(location),
            }
        )

    opportunities.sort(key=lambda item: item["distance_2d"])
    result.update(
        {
            "active_large_pad_indices": active_large,
            "active_small_pad_indices": active_small,
            "nearby_active_opportunities": opportunities[:nearby_limit],
        }
    )
    return result


def build_state_snapshot(
    state: Any,
    player: Any,
    opponent: Any | None,
    *,
    score_diff: int,
    game_time: float | None,
    seconds_remaining: float | None,
    boost_locations: Sequence[Any] | None = None,
    nearby_boost_limit: int = 5,
) -> dict[str, Any]:
    return {
        "self": _player_snapshot(player),
        "opponent": _player_snapshot(opponent),
        "ball": {
            "position": _vector_record(state.ball.pos),
            "velocity": _vector_record(state.ball.vel),
        },
        "boost_map": _boost_snapshot(
            state, player, boost_locations, nearby_boost_limit
        ),
        "score_diff": int(score_diff),
        "game_time": None if game_time is None else float(game_time),
        "seconds_remaining": (
            None if seconds_remaining is None else float(seconds_remaining)
        ),
    }
