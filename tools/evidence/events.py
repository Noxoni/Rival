from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Iterable

from .io import EvidenceSession


DETECTOR_VERSION = "rival-m02-events-v1"


@dataclass(frozen=True)
class DetectorParameters:
    pre_window_seconds: float = 1.5
    post_window_seconds: float = 4.0
    aerial_min_ball_height: float = 300.0
    aerial_low_boost_reference: float = 30.0
    aerial_min_distance: float = 650.0
    boost_pickup_min_gain: float = 5.0
    detour_min_distance_increase: float = 150.0
    challenge_max_opponent_ball_distance: float = 1900.0
    challenge_min_closing_speed: float = 250.0
    challenge_max_self_ball_distance: float = 1000.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def _value(mapping: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def _game_time(record: dict[str, Any]) -> float:
    value = _float(_value(record, "decision", "game_time"))
    if value is not None:
        return value
    value = _float(_value(record, "state", "game_time"))
    return value if value is not None else 0.0


def _score_tuple(record: dict[str, Any]) -> tuple[Any, ...]:
    scores = _value(record, "packet", "match", "scores") or []
    if scores:
        return tuple(item.get("score") for item in scores if isinstance(item, dict))
    return (_value(record, "state", "score_diff"),)


def _phase(record: dict[str, Any]) -> str | None:
    return _value(record, "packet", "match", "phase", "name")


def segment_records(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Split decisions at score/reset/time-rewind boundaries."""

    if not records:
        return []
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_time: float | None = None
    previous_score: tuple[Any, ...] | None = None
    for record in records:
        game_time = _game_time(record)
        score = _score_tuple(record)
        phase = _phase(record)
        boundary = bool(
            current
            and (
                (previous_time is not None and game_time + 0.1 < previous_time)
                or (previous_score is not None and score != previous_score)
                or phase in {"Countdown", "Kickoff"}
            )
        )
        if boundary:
            segments.append(current)
            current = []
        current.append(record)
        previous_time = game_time
        previous_score = score
    if current:
        segments.append(current)
    return segments


def _window(
    segment: list[dict[str, Any]],
    anchor_index: int,
    before: float,
    after: float,
) -> tuple[int, int, list[dict[str, Any]]]:
    anchor_time = _game_time(segment[anchor_index])
    first = anchor_index
    last = anchor_index
    while first > 0 and _game_time(segment[first - 1]) >= anchor_time - before:
        first -= 1
    while last + 1 < len(segment) and _game_time(segment[last + 1]) <= anchor_time + after:
        last += 1
    return first, last, segment[first : last + 1]


def _player(record: dict[str, Any], role: str) -> dict[str, Any] | None:
    packet = record.get("packet") or {}
    players = packet.get("players") or []
    if role == "self":
        index = packet.get("self_index")
    else:
        indices = packet.get("opponent_indices") or []
        index = indices[0] if indices else None
    if isinstance(index, int) and 0 <= index < len(players):
        value = players[index]
        return value if isinstance(value, dict) else None
    return None


def _latest_touch_time(record: dict[str, Any], role: str) -> float | None:
    return _float(_value(_player(record, role), "latest_touch", "game_seconds"))


def _touch_outcome(window: list[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, Any]:
    self_anchor = _latest_touch_time(anchor, "self")
    opponent_anchor = _latest_touch_time(anchor, "opponent")
    touches: list[tuple[float, str]] = []
    max_self = self_anchor
    max_opponent = opponent_anchor
    for record in window:
        self_touch = _latest_touch_time(record, "self")
        opponent_touch = _latest_touch_time(record, "opponent")
        if self_touch is not None and (max_self is None or self_touch > max_self + 1e-4):
            touches.append((self_touch, "self"))
            max_self = self_touch
        if opponent_touch is not None and (
            max_opponent is None or opponent_touch > max_opponent + 1e-4
        ):
            touches.append((opponent_touch, "opponent"))
            max_opponent = opponent_touch
    touches.sort()
    return {
        "next_touch": touches[0][1] if touches else "none",
        "touch_sequence": [role for _, role in touches],
        "self_touch_count": sum(role == "self" for _, role in touches),
        "opponent_touch_count": sum(role == "opponent" for _, role in touches),
    }


def _event_id(session_id: str, event_class: str, anchor_tick: Any) -> str:
    token = f"{session_id}|{event_class}|{anchor_tick}".encode("utf-8")
    return f"{event_class[:4]}-{hashlib.sha256(token).hexdigest()[:12]}"


def _schedule_case(session: EvidenceSession, game_time: float) -> dict[str, Any] | None:
    for entry in session.manifest.get("schedule", []):
        if not isinstance(entry, dict):
            continue
        start = _float(entry.get("start_game_time"))
        end = _float(entry.get("end_game_time"))
        if start is not None and end is not None and start <= game_time <= end:
            return entry
    return None


def _envelope(
    session: EvidenceSession,
    event_class: str,
    segment: list[dict[str, Any]],
    anchor_index: int,
    first: int,
    last: int,
    raw: dict[str, Any],
    derived: dict[str, Any],
    outcome: dict[str, Any],
    score: float,
    explanation: list[str],
) -> dict[str, Any]:
    anchor = segment[anchor_index]
    anchor_tick = _value(anchor, "decision", "tick")
    replay = (session.manifest.get("replays") or [None])[0]
    replay_path = replay.get("path") if isinstance(replay, dict) else None
    global_first = segment[first].get("_decision_index")
    global_last = segment[last].get("_decision_index")
    return {
        "event_id": _event_id(session.session_id, event_class, anchor_tick),
        "class": event_class,
        "candidate_only": True,
        "session_id": session.session_id,
        "opponent": session.opponent,
        "source": session.source,
        "start_game_time": _game_time(segment[first]),
        "end_game_time": _game_time(segment[last]),
        "anchor_game_time": _game_time(anchor),
        "anchor_decision_tick": anchor_tick,
        "pre_window_record_range": [global_first, anchor.get("_decision_index")],
        "post_window_record_range": [anchor.get("_decision_index"), global_last],
        "raw_features": raw,
        "derived_features": derived,
        "outcome": outcome,
        "ranking_score": round(max(0.0, min(100.0, score)), 3),
        "ranking_explanation": explanation,
        "raw_telemetry_path": str(session.raw_path),
        "raw_telemetry_sha256": session.raw_sha256,
        "replay_path": replay_path,
        "replay_timestamp": _game_time(anchor),
        "fixture_path": None,
    }


def detect_resource_stressed_aerials(
    session: EvidenceSession,
    params: DetectorParameters,
) -> list[dict[str, Any]]:
    probe_family = _value(session.manifest, "probe", "family")
    if session.source == "controlled_probe" and probe_family != "resource_aerial":
        return []
    events: list[dict[str, Any]] = []
    for segment in segment_records(session.decisions):
        previous_aerial = False
        for index, anchor in enumerate(segment):
            metrics = anchor.get("tactical_metrics") or {}
            aerial = bool(metrics.get("selected_action_aerial_like"))
            ball_height = _float(metrics.get("ball_height")) or 0.0
            distance = _float(metrics.get("distance_self_ball")) or 0.0
            boost = _float(metrics.get("self_boost"))
            transition = aerial and not previous_aerial
            previous_aerial = aerial
            if not transition or ball_height < params.aerial_min_ball_height:
                continue
            if distance < params.aerial_min_distance and (
                boost is None or boost > params.aerial_low_boost_reference
            ):
                continue
            first, last, window = _window(
                segment,
                index,
                params.pre_window_seconds,
                params.post_window_seconds,
            )
            boost_values = [
                value
                for record in window
                if (value := _float(_value(record, "tactical_metrics", "self_boost")))
                is not None
            ]
            outcome = _touch_outcome(window[index - first :], anchor)
            recovered = any(
                not bool(_value(record, "tactical_metrics", "self_airborne"))
                for record in window[index - first + 1 :]
            )
            min_boost = min(boost_values) if boost_values else boost
            boost_spent = max(0.0, (boost or 0.0) - (min_boost or 0.0))
            low_component = max(
                0.0,
                (params.aerial_low_boost_reference - (boost or 0.0))
                / params.aerial_low_boost_reference,
            )
            score = (
                20.0
                + 25.0 * low_component
                + min(15.0, max(0.0, ball_height - 300.0) / 80.0)
                + min(10.0, max(0.0, distance - 650.0) / 150.0)
                + (18.0 if outcome["next_touch"] != "self" else -12.0)
                + (12.0 if not recovered else 0.0)
            )
            explanation = ["aerial-like action transition with elevated ball"]
            if low_component > 0:
                explanation.append("low starting boost relative to detector reference")
            if outcome["next_touch"] == "opponent":
                explanation.append("opponent recorded the next touch")
            elif outcome["next_touch"] == "none":
                explanation.append("no later touch was observed in the bounded window")
            if not recovered:
                explanation.append("no grounded recovery was observed in the bounded window")
            events.append(
                _envelope(
                    session,
                    "resource_stressed_aerial",
                    segment,
                    index,
                    first,
                    last,
                    {
                        "start_boost": boost,
                        "ball_height": ball_height,
                        "distance_to_ball": distance,
                        "eta_self_ball": metrics.get("eta_self_ball"),
                        "eta_opponent_ball": metrics.get("eta_opponent_ball"),
                        "selected_action": _value(anchor, "decision", "controller_action"),
                        "top_actions": _value(anchor, "decision", "top_actions"),
                        "probe_case": _schedule_case(session, _game_time(anchor)),
                    },
                    {
                        "min_boost": min_boost,
                        "boost_spent": boost_spent,
                        "low_resource_component": low_component,
                        "recovered_to_ground": recovered,
                    },
                    outcome,
                    score,
                    explanation,
                )
            )
    return events


def detect_boost_detours(
    session: EvidenceSession,
    params: DetectorParameters,
) -> list[dict[str, Any]]:
    probe_family = _value(session.manifest, "probe", "family")
    if session.source == "controlled_probe" and probe_family != "boost_detour":
        # State-setting changes boost and geometry discontinuously. Treating those
        # transitions as live pad pickups would create cross-probe artifacts.
        return []
    events: list[dict[str, Any]] = []
    for segment in segment_records(session.decisions):
        for index in range(1, len(segment)):
            anchor = segment[index]
            previous = segment[index - 1]
            boost = _float(_value(anchor, "tactical_metrics", "self_boost"))
            previous_boost = _float(_value(previous, "tactical_metrics", "self_boost"))
            if boost is None or previous_boost is None:
                continue
            gain = boost - previous_boost
            if gain < params.boost_pickup_min_gain:
                continue
            first, last, window = _window(
                segment,
                index,
                params.pre_window_seconds,
                params.post_window_seconds,
            )
            before = segment[first:index] or [previous]
            after = segment[index : last + 1]
            before_distance = min(
                (
                    value
                    for record in before
                    if (
                        value := _float(
                            _value(record, "tactical_metrics", "distance_self_ball")
                        )
                    )
                    is not None
                ),
                default=None,
            )
            anchor_distance = _float(
                _value(anchor, "tactical_metrics", "distance_self_ball")
            )
            distance_added = (
                None
                if before_distance is None or anchor_distance is None
                else anchor_distance - before_distance
            )
            eta_before = next(
                (
                    value
                    for record in reversed(before)
                    if (
                        value := _float(
                            _value(record, "tactical_metrics", "possession_eta_advantage")
                        )
                    )
                    is not None
                ),
                None,
            )
            eta_after_values = [
                value
                for record in after
                if (
                    value := _float(
                        _value(record, "tactical_metrics", "possession_eta_advantage")
                    )
                )
                is not None
            ]
            eta_after = eta_after_values[-1] if eta_after_values else None
            possession_flip = bool(
                eta_before is not None
                and eta_after is not None
                and eta_before > 0.0
                and eta_after < 0.0
            )
            if (
                (distance_added is None or distance_added < params.detour_min_distance_increase)
                and not possession_flip
            ):
                continue
            outcome = _touch_outcome(after, anchor)
            score = (
                15.0
                + min(20.0, gain)
                + min(20.0, max(0.0, distance_added or 0.0) / 30.0)
                + (25.0 if possession_flip else 0.0)
                + (15.0 if outcome["next_touch"] == "opponent" else 0.0)
            )
            explanation = ["boost increased across a decision interval"]
            if distance_added is not None and distance_added > 0:
                explanation.append("distance to ball increased before the pickup")
            if possession_flip:
                explanation.append("ETA possession proxy changed from favorable to unfavorable")
            if outcome["next_touch"] == "opponent":
                explanation.append("opponent recorded the next touch")
            events.append(
                _envelope(
                    session,
                    "boost_detour_possession_loss",
                    segment,
                    index,
                    first,
                    last,
                    {
                        "boost_before": previous_boost,
                        "boost_after": boost,
                        "distance_before": before_distance,
                        "distance_at_pickup": anchor_distance,
                        "eta_advantage_before": eta_before,
                        "eta_advantage_after": eta_after,
                        "boost_pads": _value(anchor, "packet", "boost_pads"),
                    },
                    {
                        "boost_gained": gain,
                        "distance_added_to_ball": distance_added,
                        "possession_flip": possession_flip,
                    },
                    outcome,
                    score,
                    explanation,
                )
            )
    return events


def _challenge_features(record: dict[str, Any]) -> dict[str, Any]:
    metrics = record.get("tactical_metrics") or {}
    opponent = _player(record, "opponent") or {}
    last_input = opponent.get("last_input") or {}
    return {
        "challenger_closing_speed": metrics.get("challenge_closing_velocity"),
        "challenger_ball_closing_speed": metrics.get("opponent_ball_closing_velocity"),
        "challenger_eta_to_ball": metrics.get("eta_opponent_ball"),
        "challenger_distance_to_ball": metrics.get("distance_opponent_ball"),
        "rival_distance_to_ball": metrics.get("distance_self_ball"),
        "challenger_boost_input": last_input.get("boost"),
        "challenger_jump_input": last_input.get("jump"),
        "challenger_throttle_input": last_input.get("throttle"),
        "challenger_steer_input": last_input.get("steer"),
    }


def detect_apparent_challenges(
    session: EvidenceSession,
    params: DetectorParameters,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    schedule = session.manifest.get("schedule", [])
    probe_family = _value(session.manifest, "probe", "family")
    if session.source == "controlled_probe" and probe_family != "fake_challenge":
        return events
    controlled = (
        session.source == "controlled_probe"
        and probe_family == "fake_challenge"
        and schedule
    )
    if controlled:
        for entry in schedule:
            start = _float(entry.get("start_game_time"))
            end = _float(entry.get("end_game_time"))
            if start is None or end is None:
                continue
            candidates = [
                record
                for record in session.decisions
                if start <= _game_time(record) <= end
            ]
            if not candidates:
                continue
            segment = candidates
            anchor_index = min(
                range(len(segment)),
                key=lambda index: abs(_game_time(segment[index]) - start),
            )
            first, last, window = _window(segment, anchor_index, 0.0, end - start)
            anchor = segment[anchor_index]
            parameters = entry.get("parameters") or {}
            behavior = parameters.get("behavior") or _value(
                session.manifest, "probe", "behavior"
            )
            fake = behavior != "true_commit"
            actions = [
                _value(record, "decision", "controller_action") or {} for record in window
            ]
            rival_release = any(
                bool(action.get("jump"))
                or abs(float(action.get("pitch") or 0.0)) > 0.6
                or bool(action.get("boost"))
                for action in actions
            )
            outcome = _touch_outcome(window, anchor)
            distance_values = [
                value
                for record in window
                if (
                    value := _float(
                        _value(record, "tactical_metrics", "distance_self_ball")
                    )
                )
                is not None
            ]
            separation_increase = (
                max(distance_values) - distance_values[0] if distance_values else None
            )
            confidence = _float(_value(anchor, "decision", "confidence")) or 0.0
            score = (
                20.0
                + (20.0 if fake else 0.0)
                + (20.0 if rival_release else 0.0)
                + min(20.0, max(0.0, separation_increase or 0.0) / 40.0)
                + (15.0 if outcome["next_touch"] == "opponent" else 0.0)
                + 5.0 * confidence
            )
            explanation = [f"controlled probe ground truth: {behavior}"]
            if fake and rival_release:
                explanation.append("Rival used a release-like jump/boost/rotation response")
            if outcome["next_touch"] == "opponent":
                explanation.append("opponent recorded the next touch")
            events.append(
                _envelope(
                    session,
                    "apparent_vs_actual_challenge",
                    segment,
                    anchor_index,
                    first,
                    last,
                    {
                        "ground_truth_behavior": behavior,
                        "probe_parameters": parameters,
                        "commitment_components": _challenge_features(anchor),
                        "selected_action": _value(anchor, "decision", "controller_action"),
                        "top_actions": _value(anchor, "decision", "top_actions"),
                    },
                    {
                        "ground_truth_committed": not fake,
                        "rival_release_like_response": rival_release,
                        "rival_ball_separation_increase": separation_increase,
                        "policy_confidence": confidence,
                    },
                    outcome,
                    score,
                    explanation,
                )
            )
        return events

    for segment in segment_records(session.decisions):
        previous_candidate = False
        for index, anchor in enumerate(segment):
            metrics = anchor.get("tactical_metrics") or {}
            opponent_distance = _float(metrics.get("distance_opponent_ball"))
            self_distance = _float(metrics.get("distance_self_ball"))
            closing = _float(metrics.get("challenge_closing_velocity"))
            candidate = bool(
                opponent_distance is not None
                and opponent_distance <= params.challenge_max_opponent_ball_distance
                and self_distance is not None
                and self_distance <= params.challenge_max_self_ball_distance
                and closing is not None
                and closing >= params.challenge_min_closing_speed
            )
            transition = candidate and not previous_candidate
            previous_candidate = candidate
            if not transition:
                continue
            first, last, window = _window(
                segment,
                index,
                params.pre_window_seconds,
                params.post_window_seconds,
            )
            closing_after = [
                value
                for record in window[index - first + 1 :]
                if (
                    value := _float(
                        _value(record, "tactical_metrics", "challenge_closing_velocity")
                    )
                )
                is not None
            ]
            aborted = any(value < 50.0 for value in closing_after)
            actions = [
                _value(record, "decision", "controller_action") or {}
                for record in window[index - first :]
            ]
            rival_release = any(bool(action.get("jump")) for action in actions)
            outcome = _touch_outcome(window[index - first :], anchor)
            score = (
                15.0
                + (25.0 if aborted else 0.0)
                + (20.0 if rival_release else 0.0)
                + (20.0 if outcome["next_touch"] == "opponent" else 0.0)
                + min(15.0, max(0.0, closing - params.challenge_min_closing_speed) / 40.0)
            )
            explanation = ["natural-match opponent closing threshold crossed"]
            if aborted:
                explanation.append("closing speed later fell below the abort reference")
            if rival_release:
                explanation.append("Rival jumped during the bounded response window")
            events.append(
                _envelope(
                    session,
                    "apparent_vs_actual_challenge",
                    segment,
                    index,
                    first,
                    last,
                    {
                        "ground_truth_behavior": None,
                        "commitment_components": _challenge_features(anchor),
                        "selected_action": _value(anchor, "decision", "controller_action"),
                        "top_actions": _value(anchor, "decision", "top_actions"),
                    },
                    {
                        "natural_commitment_inference_only": True,
                        "closing_aborted_in_window": aborted,
                        "rival_release_like_response": rival_release,
                    },
                    outcome,
                    score,
                    explanation,
                )
            )
    return events


def detect_events(
    sessions: Iterable[EvidenceSession],
    params: DetectorParameters | None = None,
) -> list[dict[str, Any]]:
    params = params or DetectorParameters()
    events: list[dict[str, Any]] = []
    for session in sessions:
        events.extend(detect_resource_stressed_aerials(session, params))
        events.extend(detect_boost_detours(session, params))
        events.extend(detect_apparent_challenges(session, params))
    return sorted(
        events,
        key=lambda event: (
            -float(event["ranking_score"]),
            event["class"],
            event["session_id"],
            float(event["anchor_game_time"]),
            event["event_id"],
        ),
    )
