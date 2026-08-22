from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .events import DETECTOR_VERSION, DetectorParameters, detect_events
from .io import EvidenceSession, load_session
from .references import sha256_file
from .session import utc_now


ANALYZER_VERSION = "rival-m04p1-natural-v1"


@dataclass(frozen=True)
class NaturalAnalysisParameters:
    goal_consequence_window_seconds: float = 10.0
    minimum_episode_separation_seconds: float = 2.0
    favorable_eta_threshold_seconds: float = 0.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PatternDefinition:
    name: str
    event_class: str
    description: str
    predicate: Callable[[Mapping[str, Any]], bool]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _decision_time(record: Mapping[str, Any]) -> float | None:
    value = _finite(_mapping(record.get("decision")).get("game_time"))
    if value is not None:
        return value
    return _finite(_mapping(record.get("state")).get("game_time"))


def _score_tuple(record: Mapping[str, Any]) -> tuple[int, int] | None:
    scores = _mapping(_mapping(record.get("packet")).get("match")).get("scores")
    if not isinstance(scores, list):
        return None
    by_team: dict[int, int] = {}
    for item in scores:
        entry = _mapping(item)
        team = entry.get("team")
        score = entry.get("score")
        if isinstance(team, int) and isinstance(score, int):
            by_team[team] = score
    if 0 not in by_team or 1 not in by_team:
        return None
    return by_team[0], by_team[1]


def _phase(record: Mapping[str, Any]) -> str | None:
    phase = _mapping(_mapping(_mapping(record.get("packet")).get("match")).get("phase"))
    name = phase.get("name")
    return str(name) if name is not None else None


def _player(record: Mapping[str, Any], role: str) -> Mapping[str, Any]:
    packet = _mapping(record.get("packet"))
    players = packet.get("players")
    if not isinstance(players, list):
        return {}
    if role == "self":
        index = packet.get("self_index")
    else:
        indices = packet.get("opponent_indices")
        index = indices[0] if isinstance(indices, list) and indices else None
    if not isinstance(index, int) or not 0 <= index < len(players):
        return {}
    return _mapping(players[index])


def _touch_time(record: Mapping[str, Any], role: str) -> float | None:
    return _finite(_mapping(_player(record, role).get("latest_touch")).get("game_seconds"))


def _touch_sequence(decisions: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    observed: set[tuple[str, float]] = set()
    for record in decisions:
        for role in ("self", "opponent"):
            value = _touch_time(record, role)
            if value is not None:
                observed.add((role, value))
    return [
        {"role": role, "game_time": game_time}
        for role, game_time in sorted(observed, key=lambda item: (item[1], item[0]))
    ]


def _next_touch(
    touches: Iterable[Mapping[str, Any]],
    anchor: float,
) -> Mapping[str, Any] | None:
    return next(
        (
            touch
            for touch in touches
            if float(touch["game_time"]) > anchor + 1e-4
        ),
        None,
    )


def _goal_events(
    decisions: Iterable[Mapping[str, Any]],
    rival_team: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous: tuple[int, int] | None = None
    for record in decisions:
        score = _score_tuple(record)
        game_time = _decision_time(record)
        if score is None or game_time is None:
            continue
        if previous is None or score[0] < previous[0] or score[1] < previous[1]:
            previous = score
            continue
        for team in (0, 1):
            for _ in range(max(0, score[team] - previous[team])):
                events.append(
                    {
                        "game_time": game_time,
                        "scoring_team": team,
                        "rival_outcome": "scored" if team == rival_team else "conceded",
                    }
                )
        previous = score
    return events


def _next_goal(
    goals: Iterable[Mapping[str, Any]],
    anchor: float,
    window_seconds: float,
) -> Mapping[str, Any] | None:
    return next(
        (
            goal
            for goal in goals
            if anchor <= float(goal["game_time"]) <= anchor + window_seconds
        ),
        None,
    )


def _collapse_overlapping_events(
    events: Iterable[Mapping[str, Any]],
    minimum_separation_seconds: float,
) -> list[Mapping[str, Any]]:
    kept: list[Mapping[str, Any]] = []
    last_by_session: dict[str, float] = {}
    ordered = sorted(
        events,
        key=lambda event: (
            str(event.get("session_id")),
            float(event.get("anchor_game_time") or 0.0),
            -float(event.get("ranking_score") or 0.0),
        ),
    )
    for event in ordered:
        session_id = str(event.get("session_id"))
        anchor = float(event.get("anchor_game_time") or 0.0)
        previous = last_by_session.get(session_id)
        if previous is not None and anchor - previous < minimum_separation_seconds:
            continue
        kept.append(event)
        last_by_session[session_id] = anchor
    return kept


PATTERN_DEFINITIONS = (
    PatternDefinition(
        name="apparent_pressure_release_after_closing_abort",
        event_class="apparent_vs_actual_challenge",
        description=(
            "opponent closing threshold crossed, later collapsed, and Rival jumped "
            "inside the natural response window"
        ),
        predicate=lambda event: bool(
            _mapping(event.get("derived_features")).get("closing_aborted_in_window")
            and _mapping(event.get("derived_features")).get(
                "rival_release_like_response"
            )
        ),
    ),
    PatternDefinition(
        name="boost_pickup_with_eta_possession_flip",
        event_class="boost_detour_possession_loss",
        description=(
            "boost pickup window in which the ETA possession proxy changed from "
            "favorable to unfavorable"
        ),
        predicate=lambda event: bool(
            _mapping(event.get("derived_features")).get("possession_flip")
        ),
    ),
    PatternDefinition(
        name="low_resource_aerial_commitment",
        event_class="resource_stressed_aerial",
        description=(
            "aerial-like action transition with an elevated ball and less than the "
            "30-boost detector reference"
        ),
        predicate=lambda event: float(
            _mapping(event.get("derived_features")).get("low_resource_component")
            or 0.0
        )
        > 0.0,
    ),
)


def _pattern_summary(
    definition: PatternDefinition,
    events: list[Mapping[str, Any]],
    goal_events: Mapping[str, list[dict[str, Any]]],
    session_count: int,
    parameters: NaturalAnalysisParameters,
) -> dict[str, Any]:
    candidates = [
        event
        for event in events
        if event.get("class") == definition.event_class
        and definition.predicate(event)
    ]
    episodes = _collapse_overlapping_events(
        candidates,
        parameters.minimum_episode_separation_seconds,
    )
    touch_counts: Counter[str] = Counter()
    goal_counts: Counter[str] = Counter()
    high_consequence = 0
    for event in episodes:
        touch = str(_mapping(event.get("outcome")).get("next_touch") or "none")
        touch_counts[touch] += 1
        session_id = str(event.get("session_id"))
        anchor = float(event.get("anchor_game_time") or 0.0)
        goal = _next_goal(
            goal_events.get(session_id, []),
            anchor,
            parameters.goal_consequence_window_seconds,
        )
        goal_outcome = "none" if goal is None else str(goal["rival_outcome"])
        goal_counts[goal_outcome] += 1
        if touch == "opponent" or goal_outcome == "conceded":
            high_consequence += 1

    count = len(episodes)
    events_per_match = count / session_count if session_count else 0.0
    opponent_rate = touch_counts["opponent"] / count if count else 0.0
    conceded_rate = goal_counts["conceded"] / count if count else 0.0
    frequency_component = math.log1p(events_per_match)
    consequence_multiplier = 1.0 + opponent_rate + 4.0 * conceded_rate
    return {
        "pattern": definition.name,
        "description": definition.description,
        "source_event_class": definition.event_class,
        "raw_candidate_count": len(candidates),
        "independent_episode_count": count,
        "matches_with_episode": len(
            {str(event.get("session_id")) for event in episodes}
        ),
        "episodes_per_match": events_per_match,
        "next_touch": {
            "self": touch_counts["self"],
            "opponent": touch_counts["opponent"],
            "none": touch_counts["none"],
            "opponent_rate": opponent_rate,
        },
        "next_goal_within_window": {
            "window_seconds": parameters.goal_consequence_window_seconds,
            "scored": goal_counts["scored"],
            "conceded": goal_counts["conceded"],
            "none": goal_counts["none"],
            "conceded_rate": conceded_rate,
        },
        "high_consequence_episode_count": high_consequence,
        "priority_components": {
            "frequency_log1p_episodes_per_match": frequency_component,
            "opponent_next_touch_rate": opponent_rate,
            "goal_conceded_rate": conceded_rate,
            "consequence_multiplier": consequence_multiplier,
        },
        "priority_score": frequency_component * consequence_multiplier,
    }


def _session_metrics(
    session: EvidenceSession,
    batch_session: Mapping[str, Any],
    parameters: NaturalAnalysisParameters,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rival_team = int(batch_session.get("rival_team") or 0)
    final_score = _mapping(batch_session.get("final_score"))
    rival_side = "blue" if rival_team == 0 else "orange"
    opponent_side = "orange" if rival_team == 0 else "blue"
    rival_goals = int(final_score.get(rival_side) or 0)
    opponent_goals = int(final_score.get(opponent_side) or 0)
    active_decisions = [record for record in session.decisions if _phase(record) == "Active"]
    eta_values = [
        value
        for record in active_decisions
        if (
            value := _finite(
                _mapping(record.get("tactical_metrics")).get(
                    "possession_eta_advantage"
                )
            )
        )
        is not None
    ]
    touches = _touch_sequence(session.decisions)
    possession_losses = [
        touch
        for previous, touch in zip(touches, touches[1:])
        if previous["role"] == "self" and touch["role"] == "opponent"
    ]
    goals = _goal_events(session.decisions, rival_team)
    conceded_after_loss = sum(
        bool(
            (
                goal := _next_goal(
                    goals,
                    float(loss["game_time"]),
                    parameters.goal_consequence_window_seconds,
                )
            )
            and goal["rival_outcome"] == "conceded"
        )
        for loss in possession_losses
    )
    scored_after_loss = sum(
        bool(
            (
                goal := _next_goal(
                    goals,
                    float(loss["game_time"]),
                    parameters.goal_consequence_window_seconds,
                )
            )
            and goal["rival_outcome"] == "scored"
        )
        for loss in possession_losses
    )
    raw_expected = _mapping(batch_session.get("raw_telemetry")).get("sha256")
    intervention_decisions = [
        (record, _mapping(record.get("natural_adjustment")))
        for record in session.decisions
        if isinstance(record.get("natural_adjustment"), Mapping)
    ]
    applied_outcome_touches: Counter[str] = Counter()
    applied_outcome_goals: Counter[str] = Counter()
    applied_severities: list[float] = []
    reason_counts: Counter[str] = Counter()
    for record, adjustment in intervention_decisions:
        reason_counts[str(adjustment.get("reason") or "unknown")] += 1
        if not bool(adjustment.get("applied")):
            continue
        anchor = _decision_time(record)
        if anchor is None:
            applied_outcome_touches["none"] += 1
            applied_outcome_goals["none"] += 1
            continue
        next_touch = _next_touch(touches, anchor)
        applied_outcome_touches[
            "none" if next_touch is None else str(next_touch["role"])
        ] += 1
        goal = _next_goal(
            goals,
            anchor,
            parameters.goal_consequence_window_seconds,
        )
        applied_outcome_goals[
            "none" if goal is None else str(goal["rival_outcome"])
        ] += 1
        severity = _finite(adjustment.get("severity"))
        if severity is not None:
            applied_severities.append(severity)
    return (
        {
            "session_id": session.session_id,
            "schedule_index": batch_session.get("schedule_index"),
            "opponent": _mapping(batch_session.get("opponent")).get("key"),
            "rival_side": rival_side,
            "final_score": dict(final_score),
            "rival_goals": rival_goals,
            "opponent_goals": opponent_goals,
            "goal_differential": rival_goals - opponent_goals,
            "decision_count": len(session.decisions),
            "active_decision_count": len(active_decisions),
            "eta_observation_count": len(eta_values),
            "favorable_eta_observation_count": sum(
                value > parameters.favorable_eta_threshold_seconds
                for value in eta_values
            ),
            "touch_count": dict(Counter(touch["role"] for touch in touches)),
            "possession_loss_transition_count": len(possession_losses),
            "goals_conceded_within_window_after_possession_loss": conceded_after_loss,
            "goals_scored_within_window_after_possession_loss": scored_after_loss,
            "natural_adjustment_record_count": len(intervention_decisions),
            "natural_adjustment_eligible_decision_count": sum(
                bool(adjustment.get("eligible"))
                for _, adjustment in intervention_decisions
            ),
            "natural_adjustment_applied_decision_count": sum(
                bool(adjustment.get("applied"))
                for _, adjustment in intervention_decisions
            ),
            "natural_adjustment_hypothetical_change_count": sum(
                adjustment.get("hypothetical_action") is not None
                for _, adjustment in intervention_decisions
            ),
            "natural_adjustment_applied_next_touch": {
                "self": applied_outcome_touches["self"],
                "opponent": applied_outcome_touches["opponent"],
                "none": applied_outcome_touches["none"],
            },
            "natural_adjustment_applied_next_goal": {
                "window_seconds": parameters.goal_consequence_window_seconds,
                "scored": applied_outcome_goals["scored"],
                "conceded": applied_outcome_goals["conceded"],
                "none": applied_outcome_goals["none"],
            },
            "natural_adjustment_applied_severity": {
                "minimum": min(applied_severities) if applied_severities else None,
                "mean": (
                    sum(applied_severities) / len(applied_severities)
                    if applied_severities
                    else None
                ),
                "maximum": max(applied_severities) if applied_severities else None,
            },
            "natural_adjustment_reason_counts": dict(sorted(reason_counts.items())),
            "raw_telemetry_sha256": session.raw_sha256,
            "raw_hash_matches_batch": session.raw_sha256 == raw_expected,
            "loader_warnings": session.warnings,
        },
        goals,
    )


def _aggregate_sessions(sessions: list[Mapping[str, Any]]) -> dict[str, Any]:
    eta_count = sum(int(session["eta_observation_count"]) for session in sessions)
    favorable_count = sum(
        int(session["favorable_eta_observation_count"]) for session in sessions
    )
    goal_differentials = [int(session["goal_differential"]) for session in sessions]
    applied_next_touch = sum(
        (
            Counter(_mapping(session.get("natural_adjustment_applied_next_touch")))
            for session in sessions
        ),
        Counter(),
    )
    applied_next_goal = sum(
        (
            Counter(
                {
                    key: value
                    for key, value in _mapping(
                        session.get("natural_adjustment_applied_next_goal")
                    ).items()
                    if key != "window_seconds"
                }
            )
            for session in sessions
        ),
        Counter(),
    )
    applied_severities = [
        float(value)
        for session in sessions
        if (
            value := _finite(
                _mapping(session.get("natural_adjustment_applied_severity")).get(
                    "mean"
                )
            )
        )
        is not None
    ]
    return {
        "match_count": len(sessions),
        "wins": sum(value > 0 for value in goal_differentials),
        "losses": sum(value < 0 for value in goal_differentials),
        "ties": sum(value == 0 for value in goal_differentials),
        "goals_for": sum(int(session["rival_goals"]) for session in sessions),
        "goals_against": sum(int(session["opponent_goals"]) for session in sessions),
        "goal_differential": sum(goal_differentials),
        "decision_count": sum(int(session["decision_count"]) for session in sessions),
        "active_decision_count": sum(
            int(session["active_decision_count"]) for session in sessions
        ),
        "favorable_eta": {
            "threshold_seconds": 0.0,
            "observation_count": eta_count,
            "favorable_count": favorable_count,
            "share": favorable_count / eta_count if eta_count else None,
        },
        "touches": dict(
            sum(
                (
                    Counter(_mapping(session.get("touch_count")))
                    for session in sessions
                ),
                Counter(),
            )
        ),
        "possession_loss_transition_count": sum(
            int(session["possession_loss_transition_count"]) for session in sessions
        ),
        "goals_conceded_within_window_after_possession_loss": sum(
            int(session["goals_conceded_within_window_after_possession_loss"])
            for session in sessions
        ),
        "goals_scored_within_window_after_possession_loss": sum(
            int(session["goals_scored_within_window_after_possession_loss"])
            for session in sessions
        ),
        "natural_adjustment": {
            "record_count": sum(
                int(session["natural_adjustment_record_count"]) for session in sessions
            ),
            "eligible_decision_count": sum(
                int(session["natural_adjustment_eligible_decision_count"])
                for session in sessions
            ),
            "applied_decision_count": sum(
                int(session["natural_adjustment_applied_decision_count"])
                for session in sessions
            ),
            "hypothetical_change_count": sum(
                int(session["natural_adjustment_hypothetical_change_count"])
                for session in sessions
            ),
            "applied_next_touch": {
                "self": applied_next_touch["self"],
                "opponent": applied_next_touch["opponent"],
                "none": applied_next_touch["none"],
            },
            "applied_next_goal": {
                "scored": applied_next_goal["scored"],
                "conceded": applied_next_goal["conceded"],
                "none": applied_next_goal["none"],
            },
            "mean_session_applied_severity": (
                sum(applied_severities) / len(applied_severities)
                if applied_severities
                else None
            ),
        },
    }


def build_natural_analysis(
    batch_path: Path,
    raw_root: Path,
    parameters: NaturalAnalysisParameters | None = None,
) -> dict[str, Any]:
    parameters = parameters or NaturalAnalysisParameters()
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    if not isinstance(batch, dict):
        raise ValueError("natural batch report must be a JSON object")
    batch_sessions = batch.get("sessions")
    if not isinstance(batch_sessions, list) or not batch_sessions:
        raise ValueError("natural batch report has no sessions")

    loaded: list[EvidenceSession] = []
    compact_sessions: list[dict[str, Any]] = []
    goal_events: dict[str, list[dict[str, Any]]] = {}
    for item in batch_sessions:
        batch_session = _mapping(item)
        session_id = str(batch_session.get("session_id") or "")
        raw_path = raw_root / session_id / "decisions.jsonl"
        if not session_id or not raw_path.is_file():
            raise FileNotFoundError(f"missing raw telemetry for session {session_id!r}")
        session = load_session(raw_path)
        metrics, goals = _session_metrics(session, batch_session, parameters)
        loaded.append(session)
        compact_sessions.append(metrics)
        goal_events[session.session_id] = goals

    detector_parameters = DetectorParameters()
    events = detect_events(loaded, detector_parameters)
    patterns = [
        _pattern_summary(
            definition,
            events,
            goal_events,
            len(loaded),
            parameters,
        )
        for definition in PATTERN_DEFINITIONS
    ]
    patterns.sort(
        key=lambda pattern: (
            -float(pattern["priority_score"]),
            str(pattern["pattern"]),
        )
    )
    event_counts = Counter(str(event.get("class")) for event in events)
    return {
        "report_schema_version": 1,
        "generated_utc": utc_now(),
        "analyzer_version": ANALYZER_VERSION,
        "analysis_parameters": parameters.to_record(),
        "source_batch": {
            "filename": batch_path.name,
            "sha256": sha256_file(batch_path),
            "protocol": batch.get("protocol"),
            "batch_id": batch.get("batch_id"),
            "phase": batch.get("phase"),
            "adjustment_mode": batch.get("adjustment_mode"),
            "parameter_version": batch.get("parameter_version"),
        },
        "detector": {
            "version": DETECTOR_VERSION,
            "parameters": detector_parameters.to_record(),
            "candidate_only": True,
            "raw_event_counts": dict(sorted(event_counts.items())),
            "overlap_control": (
                "same-pattern anchors within the configured per-session separation "
                "are counted once"
            ),
        },
        "raw_hashes_verified": all(
            bool(session["raw_hash_matches_batch"]) for session in compact_sessions
        ),
        "sessions": compact_sessions,
        "aggregate": _aggregate_sessions(compact_sessions),
        "ranked_patterns": patterns,
        "highest_priority_pattern": patterns[0]["pattern"] if patterns else None,
        "interpretation_limits": [
            "natural trajectories are unpaired and pattern counts are observational",
            "event detectors are candidate screens rather than ground-truth labels",
            "scores and goal differential are context, not a standalone skill claim",
        ],
    }


def markdown_summary(report: Mapping[str, Any]) -> str:
    source = _mapping(report.get("source_batch"))
    aggregate = _mapping(report.get("aggregate"))
    favorable = _mapping(aggregate.get("favorable_eta"))
    adjustment = _mapping(aggregate.get("natural_adjustment"))
    lines = [
        f"# Natural-play analysis: {source.get('phase', 'unknown')}",
        "",
        (
            f"Analyzer `{report.get('analyzer_version')}` processed "
            f"**{aggregate.get('match_count', 0)}** full natural matches and "
            f"**{aggregate.get('decision_count', 0):,}** policy decisions."
        ),
        "",
        "## Aggregate context",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
        f"| Wins / losses / ties | {aggregate.get('wins', 0)} / {aggregate.get('losses', 0)} / {aggregate.get('ties', 0)} |",
        f"| Goals for / against | {aggregate.get('goals_for', 0)} / {aggregate.get('goals_against', 0)} |",
        f"| Goal differential | {aggregate.get('goal_differential', 0):+d} |",
        f"| Favorable ETA share | {float(favorable.get('share') or 0.0):.4f} |",
        f"| Possession-loss transitions | {aggregate.get('possession_loss_transition_count', 0)} |",
        (
            "| Goals conceded within the consequence window after loss | "
            f"{aggregate.get('goals_conceded_within_window_after_possession_loss', 0)} |"
        ),
        f"| Adjustment applied decisions | {adjustment.get('applied_decision_count', 0)} |",
        (
            "| Applied next touch self / opponent / none | "
            f"{_mapping(adjustment.get('applied_next_touch')).get('self', 0)} / "
            f"{_mapping(adjustment.get('applied_next_touch')).get('opponent', 0)} / "
            f"{_mapping(adjustment.get('applied_next_touch')).get('none', 0)} |"
        ),
        "",
        "Scores are context only; natural trajectories are not paired skill evidence.",
        "",
        "## Ranked recurring patterns",
        "",
        "| Pattern | Episodes | Matches | Opponent next touch | Conceded next goal | Priority |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pattern_value in report.get("ranked_patterns") or []:
        pattern = _mapping(pattern_value)
        touch = _mapping(pattern.get("next_touch"))
        goal = _mapping(pattern.get("next_goal_within_window"))
        lines.append(
            f"| `{pattern.get('pattern')}` | {pattern.get('independent_episode_count', 0)} "
            f"| {pattern.get('matches_with_episode', 0)} "
            f"| {touch.get('opponent', 0)} ({float(touch.get('opponent_rate') or 0.0):.3f}) "
            f"| {goal.get('conceded', 0)} ({float(goal.get('conceded_rate') or 0.0):.3f}) "
            f"| {float(pattern.get('priority_score') or 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The priority score combines log-scaled cross-match frequency with "
                "opponent-next-touch and near-term-concession rates. It ranks what to "
                "inspect; it is not a causal effect estimate."
            ),
            "",
            "## Session ledger",
            "",
            "| # | Opponent | Rival side | Score | Decisions | Raw SHA-256 |",
            "| ---: | --- | --- | ---: | ---: | --- |",
        ]
    )
    for session_value in report.get("sessions") or []:
        session = _mapping(session_value)
        score = _mapping(session.get("final_score"))
        lines.append(
            f"| {session.get('schedule_index')} | {session.get('opponent')} "
            f"| {session.get('rival_side')} | {score.get('blue', 0)}-{score.get('orange', 0)} "
            f"| {session.get('decision_count', 0):,} | `{session.get('raw_telemetry_sha256')}` |"
        )
    lines.extend(
        [
            "",
            "All raw hashes matched the compact batch manifest: "
            f"**{str(bool(report.get('raw_hashes_verified'))).lower()}**.",
            "",
        ]
    )
    return "\n".join(lines)
