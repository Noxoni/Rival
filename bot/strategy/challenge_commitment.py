from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping


Vector3 = tuple[float, float, float]


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _vector(value: Any) -> Vector3 | None:
    record = _mapping(value)
    components = tuple(_finite(record.get(axis)) for axis in ("x", "y", "z"))
    if any(component is None for component in components):
        return None
    return float(components[0]), float(components[1]), float(components[2])


def _attribute_vector(value: Any) -> Vector3 | None:
    components = tuple(_finite(_field(value, axis)) for axis in ("x", "y", "z"))
    if any(component is None for component in components):
        return None
    return float(components[0]), float(components[1]), float(components[2])


def _forward_from_rotation(value: Any) -> Vector3 | None:
    pitch = _finite(_field(value, "pitch"))
    yaw = _finite(_field(value, "yaw"))
    if pitch is None or yaw is None:
        return None
    cosine_pitch = math.cos(pitch)
    return cosine_pitch * math.cos(yaw), cosine_pitch * math.sin(yaw), math.sin(pitch)


def _enum_name(value: Any) -> str:
    if value is None:
        return ""
    return str(value).split(".")[-1]


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return first[0] - second[0], first[1] - second[1], first[2] - second[2]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(left * right for left, right in zip(first, second))


def _magnitude(value: Vector3) -> float:
    return math.sqrt(_dot(value, value))


def _distance(first: Vector3, second: Vector3) -> float:
    return _magnitude(_subtract(first, second))


def _alignment(direction: Vector3, reference: Vector3) -> float:
    denominator = _magnitude(direction) * _magnitude(reference)
    if denominator <= 1e-6:
        return 0.0
    return max(-1.0, min(1.0, _dot(direction, reference) / denominator))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def projected_closest_approach(
    opponent_position: Vector3,
    opponent_velocity: Vector3,
    ball_position: Vector3,
    ball_velocity: Vector3,
    horizon_seconds: float,
) -> tuple[float, float]:
    """Return constant-relative-velocity closest time and miss distance."""

    relative_position = _subtract(ball_position, opponent_position)
    relative_velocity = _subtract(opponent_velocity, ball_velocity)
    speed_squared = _dot(relative_velocity, relative_velocity)
    if speed_squared <= 1e-9:
        return 0.0, _magnitude(relative_position)
    closest_time = max(
        0.0,
        min(
            float(horizon_seconds),
            _dot(relative_position, relative_velocity) / speed_squared,
        ),
    )
    miss_vector = tuple(
        relative_position[index] - relative_velocity[index] * closest_time
        for index in range(3)
    )
    return closest_time, _magnitude(miss_vector)


@dataclass(frozen=True)
class ChallengeCommitmentParameters:
    version: str = "m03-commitment-v1"
    history_window_seconds: float = 0.40
    maximum_history_gap_seconds: float = 0.50
    discontinuity_distance: float = 1200.0
    projection_horizon_seconds: float = 0.80
    pressure_distance: float = 1900.0
    pressure_eta: float = 1.40
    pressure_min_closing_speed: float = 150.0
    projected_miss_reference: float = 450.0
    low_threshold: float = 0.34
    high_threshold: float = 0.70
    abort_closing_speed: float = 100.0
    abort_eta_growth: float = 0.14
    abort_miss_growth: float = 180.0
    abort_steer: float = 0.70
    abort_forward_alignment: float = 0.40
    reverse_throttle: float = -0.20

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_threshold < self.high_threshold <= 1.0:
            raise ValueError("commitment thresholds must satisfy 0 <= low < high <= 1")
        if self.history_window_seconds <= 0.0:
            raise ValueError("history_window_seconds must be positive")
        if self.projection_horizon_seconds <= 0.0:
            raise ValueError("projection_horizon_seconds must be positive")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChallengeSample:
    game_time: float
    self_position: Vector3
    self_velocity: Vector3
    opponent_position: Vector3
    opponent_velocity: Vector3
    opponent_forward: Vector3
    ball_position: Vector3
    ball_velocity: Vector3
    self_team: int
    self_grounded: bool
    opponent_airborne: bool
    self_demoed: bool
    opponent_demoed: bool
    opponent_id: int | str | None
    opponent_throttle: float
    opponent_steer: float
    opponent_jump: bool
    opponent_boost: bool
    opponent_handbrake: bool
    opponent_input_available: bool
    self_eta_to_ball: float | None
    opponent_eta_to_ball: float | None
    opponent_ball_closing_speed: float | None
    opponent_rival_closing_speed: float | None
    phase: str | None
    scores: tuple[int, ...]
    self_latest_touch_time: float | None = None
    opponent_latest_touch_time: float | None = None

    @classmethod
    def from_packet(
        cls,
        packet: Any,
        self_index: int,
        tactical_metrics: Any,
    ) -> "ChallengeSample | None":
        """Build a minimal sample directly from an RLBot v5 packet.

        This avoids constructing the much larger telemetry packet snapshot when normal
        telemetry is disabled. It deliberately accepts ``Any`` so the estimator stays
        testable with small packet-shaped fakes and across compatible RLBot v5 builds.
        """

        players = list(_field(packet, "players") or [])
        balls = list(_field(packet, "balls") or [])
        if not 0 <= self_index < len(players) or not balls:
            return None
        self_player = players[self_index]
        self_team_value = _finite(_field(self_player, "team"))
        if self_team_value is None:
            return None
        self_team = int(self_team_value)
        opponents = [
            (index, player)
            for index, player in enumerate(players)
            if index != self_index and _finite(_field(player, "team")) != self_team_value
        ]
        ball_physics = _field(balls[0], "physics")
        ball_position = _attribute_vector(_field(ball_physics, "location"))
        ball_velocity = _attribute_vector(_field(ball_physics, "velocity"))
        if ball_position is None or ball_velocity is None or not opponents:
            return None

        def opponent_distance(item: tuple[int, Any]) -> float:
            physics = _field(item[1], "physics")
            position = _attribute_vector(_field(physics, "location"))
            return math.inf if position is None else _distance(position, ball_position)

        opponent_index, opponent = min(opponents, key=opponent_distance)
        self_physics = _field(self_player, "physics")
        opponent_physics = _field(opponent, "physics")
        self_position = _attribute_vector(_field(self_physics, "location"))
        self_velocity = _attribute_vector(_field(self_physics, "velocity"))
        opponent_position = _attribute_vector(_field(opponent_physics, "location"))
        opponent_velocity = _attribute_vector(_field(opponent_physics, "velocity"))
        opponent_forward = _forward_from_rotation(_field(opponent_physics, "rotation"))
        if any(
            value is None
            for value in (
                self_position,
                self_velocity,
                opponent_position,
                opponent_velocity,
                opponent_forward,
            )
        ):
            return None

        match_info = _field(packet, "match_info")
        game_time = _finite(_field(match_info, "seconds_elapsed"))
        if game_time is None:
            return None
        last_input = _field(opponent, "last_input")
        self_touch = _field(self_player, "latest_touch")
        opponent_touch = _field(opponent, "latest_touch")
        scores = tuple(
            int(score)
            for team in list(_field(packet, "teams") or [])
            if (score := _finite(_field(team, "score"))) is not None
        )
        self_air_state = _enum_name(_field(self_player, "air_state"))
        opponent_air_state = _enum_name(_field(opponent, "air_state"))
        return cls(
            game_time=game_time,
            self_position=self_position,  # type: ignore[arg-type]
            self_velocity=self_velocity,  # type: ignore[arg-type]
            opponent_position=opponent_position,  # type: ignore[arg-type]
            opponent_velocity=opponent_velocity,  # type: ignore[arg-type]
            opponent_forward=opponent_forward,  # type: ignore[arg-type]
            ball_position=ball_position,
            ball_velocity=ball_velocity,
            self_team=self_team,
            self_grounded=self_air_state == "OnGround",
            opponent_airborne=opponent_air_state not in {"", "None", "OnGround"},
            self_demoed=(_finite(_field(self_player, "demolished_timeout")) or 0.0)
            > 0.0,
            opponent_demoed=(_finite(_field(opponent, "demolished_timeout")) or 0.0)
            > 0.0,
            opponent_id=_field(opponent, "player_id", opponent_index),
            opponent_throttle=_finite(_field(last_input, "throttle")) or 0.0,
            opponent_steer=_finite(_field(last_input, "steer")) or 0.0,
            opponent_jump=bool(_field(last_input, "jump", False)),
            opponent_boost=bool(_field(last_input, "boost", False)),
            opponent_handbrake=bool(_field(last_input, "handbrake", False)),
            opponent_input_available=last_input is not None,
            self_eta_to_ball=_finite(_field(tactical_metrics, "eta_self_ball")),
            opponent_eta_to_ball=_finite(
                _field(tactical_metrics, "eta_opponent_ball")
            ),
            opponent_ball_closing_speed=_finite(
                _field(tactical_metrics, "opponent_ball_closing_velocity")
            ),
            opponent_rival_closing_speed=_finite(
                _field(tactical_metrics, "challenge_closing_velocity")
            ),
            phase=_enum_name(_field(match_info, "match_phase")) or None,
            scores=scores,
            self_latest_touch_time=_finite(_field(self_touch, "game_seconds")),
            opponent_latest_touch_time=_finite(
                _field(opponent_touch, "game_seconds")
            ),
        )

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "ChallengeSample | None":
        packet = _mapping(record.get("packet"))
        metrics = _mapping(record.get("tactical_metrics"))
        decision = _mapping(record.get("decision"))
        players = packet.get("players")
        if not isinstance(players, list):
            return None
        self_index = packet.get("self_index")
        opponent_indices = packet.get("opponent_indices")
        if not isinstance(self_index, int) or not isinstance(opponent_indices, list):
            return None
        if not opponent_indices or not isinstance(opponent_indices[0], int):
            return None
        opponent_index = opponent_indices[0]
        if not 0 <= self_index < len(players) or not 0 <= opponent_index < len(players):
            return None

        self_player = _mapping(players[self_index])
        opponent = _mapping(players[opponent_index])
        self_physics = _mapping(self_player.get("physics"))
        opponent_physics = _mapping(opponent.get("physics"))
        opponent_rotation = _mapping(opponent_physics.get("rotation"))
        ball = _mapping(packet.get("ball"))
        ball_physics = _mapping(ball.get("physics"))
        positions = (
            _vector(self_physics.get("position")),
            _vector(opponent_physics.get("position")),
            _vector(ball_physics.get("position")),
        )
        velocities = (
            _vector(self_physics.get("velocity")),
            _vector(opponent_physics.get("velocity")),
            _vector(ball_physics.get("velocity")),
        )
        opponent_forward = _vector(opponent_rotation.get("forward"))
        game_time = _finite(decision.get("game_time"))
        if game_time is None:
            game_time = _finite(_mapping(packet.get("match")).get("seconds_elapsed"))
        if (
            game_time is None
            or any(value is None for value in positions)
            or any(value is None for value in velocities)
            or opponent_forward is None
        ):
            return None

        last_input = _mapping(opponent.get("last_input"))
        air_state = str(_mapping(opponent.get("air_state")).get("name") or "")
        scores: list[int] = []
        for score in _mapping(packet.get("match")).get("scores") or []:
            value = _finite(_mapping(score).get("score"))
            if value is not None:
                scores.append(int(value))
        self_touch = _mapping(self_player.get("latest_touch"))
        opponent_touch = _mapping(opponent.get("latest_touch"))
        self_team = _finite(self_player.get("team"))
        if self_team is None:
            return None

        return cls(
            game_time=game_time,
            self_position=positions[0],  # type: ignore[arg-type]
            self_velocity=velocities[0],  # type: ignore[arg-type]
            opponent_position=positions[1],  # type: ignore[arg-type]
            opponent_velocity=velocities[1],  # type: ignore[arg-type]
            opponent_forward=opponent_forward,
            ball_position=positions[2],  # type: ignore[arg-type]
            ball_velocity=velocities[2],  # type: ignore[arg-type]
            self_team=int(self_team),
            self_grounded=str(_mapping(self_player.get("air_state")).get("name"))
            == "OnGround",
            opponent_airborne=air_state not in {"", "OnGround"},
            self_demoed=(_finite(self_player.get("demolished_timeout")) or 0.0) > 0.0,
            opponent_demoed=(_finite(opponent.get("demolished_timeout")) or 0.0)
            > 0.0,
            opponent_id=opponent.get("player_id", opponent_index),
            opponent_throttle=_finite(last_input.get("throttle")) or 0.0,
            opponent_steer=_finite(last_input.get("steer")) or 0.0,
            opponent_jump=bool(last_input.get("jump", False)),
            opponent_boost=bool(last_input.get("boost", False)),
            opponent_handbrake=bool(last_input.get("handbrake", False)),
            opponent_input_available=bool(last_input),
            self_eta_to_ball=_finite(metrics.get("eta_self_ball")),
            opponent_eta_to_ball=_finite(metrics.get("eta_opponent_ball")),
            opponent_ball_closing_speed=_finite(
                metrics.get("opponent_ball_closing_velocity")
            ),
            opponent_rival_closing_speed=_finite(
                metrics.get("challenge_closing_velocity")
            ),
            phase=(
                str(
                    _mapping(_mapping(packet.get("match")).get("phase")).get("name")
                    or ""
                )
                or None
            ),
            scores=tuple(scores),
            self_latest_touch_time=_finite(self_touch.get("game_seconds")),
            opponent_latest_touch_time=_finite(opponent_touch.get("game_seconds")),
        )

    @property
    def defensive_emergency(self) -> bool:
        own_goal_y = -5120.0 if self.self_team == 0 else 5120.0
        goal_distance = abs(own_goal_y - self.ball_position[1])
        goalward_speed = (
            -self.ball_velocity[1] if self.self_team == 0 else self.ball_velocity[1]
        )
        return goal_distance <= 1500.0 and goalward_speed >= 300.0

    @property
    def reset_or_kickoff(self) -> bool:
        if self.phase in {"Countdown", "GoalScored", "Ended"}:
            return True
        ball_center_distance = math.hypot(self.ball_position[0], self.ball_position[1])
        return self.phase == "Kickoff" and ball_center_distance <= 300.0


@dataclass(frozen=True)
class ChallengeCommitmentEstimate:
    valid: bool
    score: float
    state: str
    pressure_present: bool
    abort_detected: bool
    components: Mapping[str, Any]
    history: Mapping[str, Any]
    reset_reason: str | None
    episode_id: int

    @classmethod
    def unavailable(
        cls, reason: str, *, episode_id: int = 0
    ) -> "ChallengeCommitmentEstimate":
        return cls(
            valid=False,
            score=0.0,
            state="low",
            pressure_present=False,
            abort_detected=False,
            components={},
            history={"sample_count": 0, "unavailable_reason": reason},
            reset_reason=reason,
            episode_id=episode_id,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "score": self.score,
            "state": self.state,
            "pressure_present": self.pressure_present,
            "abort_detected": self.abort_detected,
            "components": dict(self.components),
            "history": dict(self.history),
            "reset_reason": self.reset_reason,
            "episode_id": self.episode_id,
        }


@dataclass(frozen=True)
class _HistoryPoint:
    sample: ChallengeSample
    ball_closing: float
    opponent_eta: float | None
    miss_distance: float
    forward_alignment: float
    pressure_present: bool


class ChallengeCommitmentTracker:
    """Short-history physical commitment estimator with explicit reset handling."""

    def __init__(self, parameters: ChallengeCommitmentParameters | None = None) -> None:
        self.parameters = parameters or ChallengeCommitmentParameters()
        self._history: deque[_HistoryPoint] = deque()
        self._last_sample: ChallengeSample | None = None
        self._pressure_active = False
        self._episode_id = 0
        self._pending_reset_reason: str | None = "tracker_initialized"

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def reset(self, reason: str = "explicit_reset") -> None:
        self._history.clear()
        self._last_sample = None
        self._pressure_active = False
        self._episode_id += 1
        self._pending_reset_reason = reason

    def _reset_reason(self, sample: ChallengeSample) -> str | None:
        previous = self._last_sample
        if previous is None:
            return None
        delta = sample.game_time - previous.game_time
        if delta < -1e-3:
            return "time_rewind"
        if delta > self.parameters.maximum_history_gap_seconds:
            return "history_gap"
        if previous.scores and sample.scores and previous.scores != sample.scores:
            return "score_change"
        if previous.opponent_id != sample.opponent_id:
            return "opponent_changed"
        if sample.reset_or_kickoff and not previous.reset_or_kickoff:
            return f"phase_{(sample.phase or 'reset').lower()}"
        if delta <= 0.25:
            positions = (
                (previous.self_position, sample.self_position),
                (previous.opponent_position, sample.opponent_position),
                (previous.ball_position, sample.ball_position),
            )
            if any(
                _distance(before, after) > self.parameters.discontinuity_distance
                for before, after in positions
            ):
                return "state_discontinuity"
        return None

    def update(self, sample: ChallengeSample | None) -> ChallengeCommitmentEstimate:
        if sample is None:
            if self._last_sample is not None or self._history:
                self.reset("missing_or_invalid_sample")
            else:
                self._pending_reset_reason = "missing_or_invalid_sample"
            return ChallengeCommitmentEstimate.unavailable(
                "missing_or_invalid_sample", episode_id=self._episode_id
            )
        if sample.self_demoed or sample.opponent_demoed:
            if self._pending_reset_reason != "demolition":
                self.reset("demolition")
            return ChallengeCommitmentEstimate.unavailable(
                "demolition", episode_id=self._episode_id
            )

        detected_reset = self._reset_reason(sample)
        if detected_reset is not None:
            self.reset(detected_reset)
        reset_reason = self._pending_reset_reason
        self._pending_reset_reason = None

        parameters = self.parameters
        relative_ball = _subtract(sample.ball_position, sample.opponent_position)
        relative_velocity = _subtract(sample.opponent_velocity, sample.ball_velocity)
        opponent_distance = _magnitude(relative_ball)
        self_distance = _distance(sample.self_position, sample.ball_position)
        opponent_speed = _magnitude(sample.opponent_velocity)
        forward_alignment = _alignment(sample.opponent_forward, relative_ball)
        velocity_alignment = _alignment(relative_velocity, relative_ball)
        closest_time, miss_distance = projected_closest_approach(
            sample.opponent_position,
            sample.opponent_velocity,
            sample.ball_position,
            sample.ball_velocity,
            parameters.projection_horizon_seconds,
        )
        ball_closing = sample.opponent_ball_closing_speed
        if ball_closing is None:
            ball_closing = _dot(relative_velocity, relative_ball) / max(
                opponent_distance, 1e-6
            )
        rival_closing = sample.opponent_rival_closing_speed or 0.0
        pressure_present = bool(
            opponent_distance <= parameters.pressure_distance
            and (
                (
                    sample.opponent_eta_to_ball is not None
                    and sample.opponent_eta_to_ball <= parameters.pressure_eta
                )
                or ball_closing >= parameters.pressure_min_closing_speed
                or rival_closing >= parameters.pressure_min_closing_speed
            )
        )

        while self._history and (
            sample.game_time - self._history[0].sample.game_time
            > parameters.history_window_seconds
        ):
            self._history.popleft()
        oldest = self._history[0] if self._history else None
        maximum_prior_closing = max(
            (point.ball_closing for point in self._history),
            default=ball_closing,
        )
        abort_signals: list[str] = []
        if (
            sample.opponent_input_available
            and sample.opponent_throttle <= parameters.reverse_throttle
        ):
            abort_signals.append("reverse_or_brake_input")
        if (
            sample.opponent_input_available
            and abs(sample.opponent_steer) >= parameters.abort_steer
            and forward_alignment <= parameters.abort_forward_alignment
        ):
            abort_signals.append("hard_steer_away")
        if maximum_prior_closing >= 400.0 and ball_closing <= parameters.abort_closing_speed:
            abort_signals.append("closing_speed_collapse")
        if oldest is not None:
            if (
                oldest.opponent_eta is not None
                and sample.opponent_eta_to_ball is not None
                and sample.opponent_eta_to_ball - oldest.opponent_eta
                >= parameters.abort_eta_growth
            ):
                abort_signals.append("eta_opening")
            if miss_distance - oldest.miss_distance >= parameters.abort_miss_growth:
                abort_signals.append("projected_miss_opening")
        if velocity_alignment <= -0.15:
            abort_signals.append("velocity_away_from_ball")
        abort_detected = bool(abort_signals)

        pressure_points = [
            point for point in self._history if point.pressure_present
        ]
        pressure_start = (
            pressure_points[0].sample.game_time if pressure_points else sample.game_time
        )
        sustained_seconds = (
            max(0.0, sample.game_time - pressure_start) if pressure_present else 0.0
        )
        proximity_component = _clamp01(
            (parameters.pressure_distance - opponent_distance)
            / max(parameters.pressure_distance - 250.0, 1.0)
        )
        eta_component = (
            0.0
            if sample.opponent_eta_to_ball is None
            else _clamp01(
                (parameters.pressure_eta - sample.opponent_eta_to_ball)
                / parameters.pressure_eta
            )
        )
        closing_component = _clamp01(
            (ball_closing - parameters.pressure_min_closing_speed) / 1050.0
        )
        rival_closing_component = _clamp01(
            (rival_closing - parameters.pressure_min_closing_speed) / 1200.0
        )
        forward_component = _clamp01(forward_alignment)
        velocity_component = _clamp01(velocity_alignment)
        intercept_component = _clamp01(
            1.0 - miss_distance / parameters.projected_miss_reference
        )
        sustained_component = _clamp01(sustained_seconds / 0.20)
        throttle_component = _clamp01(sample.opponent_throttle)

        score = (
            0.12 * proximity_component
            + 0.14 * eta_component
            + 0.14 * closing_component
            + 0.05 * rival_closing_component
            + 0.14 * forward_component
            + 0.14 * velocity_component
            + 0.18 * intercept_component
            + 0.05 * sustained_component
            + 0.02 * throttle_component
            + 0.01 * float(sample.opponent_boost)
            + 0.01 * float(sample.opponent_jump)
        )

        trends: dict[str, float | None] = {
            "closing_speed_delta": None,
            "opponent_eta_delta": None,
            "projected_miss_delta": None,
            "forward_alignment_delta": None,
        }
        trend_adjustment = 0.0
        if oldest is not None:
            closing_delta = ball_closing - oldest.ball_closing
            eta_delta = (
                None
                if oldest.opponent_eta is None or sample.opponent_eta_to_ball is None
                else sample.opponent_eta_to_ball - oldest.opponent_eta
            )
            miss_delta = miss_distance - oldest.miss_distance
            alignment_delta = forward_alignment - oldest.forward_alignment
            trends = {
                "closing_speed_delta": closing_delta,
                "opponent_eta_delta": eta_delta,
                "projected_miss_delta": miss_delta,
                "forward_alignment_delta": alignment_delta,
            }
            trend_adjustment += 0.025 * max(-1.0, min(1.0, closing_delta / 500.0))
            if eta_delta is not None:
                trend_adjustment += 0.025 * max(-1.0, min(1.0, -eta_delta / 0.25))
            trend_adjustment += 0.025 * max(-1.0, min(1.0, -miss_delta / 250.0))
            trend_adjustment += 0.015 * max(-1.0, min(1.0, alignment_delta / 0.35))
        score += trend_adjustment
        if abort_detected:
            score -= min(0.42, 0.28 + 0.05 * (len(abort_signals) - 1))
        score = _clamp01(score)
        if not pressure_present:
            state = "low"
        elif score >= parameters.high_threshold and not abort_detected:
            state = "high"
        elif score < parameters.low_threshold:
            state = "low"
        else:
            state = "ambiguous"

        if pressure_present and not self._pressure_active:
            self._episode_id += 1
        self._pressure_active = pressure_present
        point = _HistoryPoint(
            sample=sample,
            ball_closing=ball_closing,
            opponent_eta=sample.opponent_eta_to_ball,
            miss_distance=miss_distance,
            forward_alignment=forward_alignment,
            pressure_present=pressure_present,
        )
        self._history.append(point)
        self._last_sample = sample

        components: dict[str, Any] = {
            "opponent_distance_to_ball": opponent_distance,
            "rival_distance_to_ball": self_distance,
            "opponent_eta_to_ball": sample.opponent_eta_to_ball,
            "rival_eta_to_ball": sample.self_eta_to_ball,
            "opponent_ball_closing_speed": ball_closing,
            "opponent_rival_closing_speed": rival_closing,
            "opponent_speed": opponent_speed,
            "forward_ball_alignment": forward_alignment,
            "velocity_ball_alignment": velocity_alignment,
            "projected_closest_time": closest_time,
            "projected_miss_distance": miss_distance,
            "opponent_throttle": sample.opponent_throttle,
            "opponent_steer": sample.opponent_steer,
            "opponent_jump": sample.opponent_jump,
            "opponent_boost": sample.opponent_boost,
            "opponent_handbrake": sample.opponent_handbrake,
            "opponent_airborne": sample.opponent_airborne,
            "input_available": sample.opponent_input_available,
            "proximity_component": proximity_component,
            "eta_component": eta_component,
            "closing_component": closing_component,
            "rival_closing_component": rival_closing_component,
            "forward_component": forward_component,
            "velocity_component": velocity_component,
            "intercept_component": intercept_component,
            "sustained_pressure_component": sustained_component,
            "input_throttle_component": throttle_component,
        }
        history_record: dict[str, Any] = {
            "sample_count": len(self._history),
            "window_seconds": (
                sample.game_time - self._history[0].sample.game_time
                if self._history
                else 0.0
            ),
            "sustained_pressure_seconds": sustained_seconds,
            "trend_adjustment": trend_adjustment,
            "trends": trends,
            "abort_signals": abort_signals,
        }
        return ChallengeCommitmentEstimate(
            valid=True,
            score=score,
            state=state,
            pressure_present=pressure_present,
            abort_detected=abort_detected,
            components=components,
            history=history_record,
            reset_reason=reset_reason,
            episode_id=self._episode_id,
        )
