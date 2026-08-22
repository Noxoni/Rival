from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

try:
    from strategy.challenge_commitment import (
        ChallengeCommitmentEstimate,
        ChallengeCommitmentTracker,
        ChallengeSample,
    )
except ModuleNotFoundError:  # Repository-tool import without pytest's bot path.
    from bot.strategy.challenge_commitment import (
        ChallengeCommitmentEstimate,
        ChallengeCommitmentTracker,
        ChallengeSample,
    )

from .io import EvidenceSession, load_sessions
from .session import utc_now, write_json


DETECTOR_VERSION = "rival-m03-challenge-v1"


@dataclass(frozen=True)
class ChallengeMetricParameters:
    control_distance: float = 650.0
    maximum_control_ball_height: float = 230.0
    maximum_eta_disadvantage: float = 0.30
    unavoidable_opponent_eta: float = 0.18
    unavoidable_opponent_distance: float = 260.0
    intervention_goal_window_seconds: float = 3.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionObservation:
    record: Mapping[str, Any]
    sample: ChallengeSample | None
    estimate: ChallengeCommitmentEstimate
    baseline_action: Mapping[str, Any]
    final_action: Mapping[str, Any]
    final_jump_initiation: bool
    challenge_candidate: bool
    premature_release_jump: bool
    intervention_eligible: bool
    intervention_applied: bool

    @property
    def game_time(self) -> float:
        value = (self.record.get("decision") or {}).get("game_time")
        return float(value) if isinstance(value, (int, float)) else 0.0


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _decision_action(record: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    decision = _mapping(record.get("decision"))
    explicit = decision.get(field)
    if isinstance(explicit, Mapping):
        return explicit
    return _mapping(decision.get("controller_action"))


def observe_decisions(
    records: Iterable[Mapping[str, Any]],
    parameters: ChallengeMetricParameters | None = None,
) -> list[DecisionObservation]:
    """Recompute the v3 physical metric for comparable schema-v2/v3 records."""

    parameters = parameters or ChallengeMetricParameters()
    tracker = ChallengeCommitmentTracker()
    observations: list[DecisionObservation] = []
    previous_final_jump = False
    for record in records:
        sample = ChallengeSample.from_record(record)
        estimate = tracker.update(sample)
        if estimate.reset_reason is not None:
            previous_final_jump = False
        baseline = _decision_action(record, "baseline_controller_action")
        final = _decision_action(record, "final_controller_action")
        final_jump = bool(final.get("jump", False))
        jump_initiation = bool(
            sample is not None
            and sample.self_grounded
            and final_jump
            and not previous_final_jump
        )
        previous_final_jump = final_jump
        rival_distance = _number(
            estimate.components.get("rival_distance_to_ball")
        )
        opponent_distance = _number(
            estimate.components.get("opponent_distance_to_ball")
        )
        ball_height = None if sample is None else sample.ball_position[2]
        eta_advantage = (
            None
            if sample is None
            or sample.self_eta_to_ball is None
            or sample.opponent_eta_to_ball is None
            else sample.opponent_eta_to_ball - sample.self_eta_to_ball
        )
        possession_plausible = bool(
            sample is not None
            and sample.self_grounded
            and rival_distance is not None
            and rival_distance <= parameters.control_distance
            and ball_height is not None
            and ball_height <= parameters.maximum_control_ball_height
            and (
                eta_advantage is None
                or eta_advantage >= -parameters.maximum_eta_disadvantage
            )
            and not sample.reset_or_kickoff
        )
        unavoidable = bool(
            sample is not None
            and (
                (
                    sample.opponent_eta_to_ball is not None
                    and sample.opponent_eta_to_ball
                    <= parameters.unavoidable_opponent_eta
                )
                or (
                    opponent_distance is not None
                    and opponent_distance <= parameters.unavoidable_opponent_distance
                )
            )
        )
        challenge_candidate = bool(
            estimate.valid
            and estimate.pressure_present
            and possession_plausible
            and not unavoidable
        )
        premature = bool(
            jump_initiation
            and challenge_candidate
            and estimate.score < tracker.parameters.high_threshold
        )
        calibration = _mapping(record.get("challenge_calibration"))
        observations.append(
            DecisionObservation(
                record=record,
                sample=sample,
                estimate=estimate,
                baseline_action=baseline,
                final_action=final,
                final_jump_initiation=jump_initiation,
                challenge_candidate=challenge_candidate,
                premature_release_jump=premature,
                intervention_eligible=bool(calibration.get("eligible", False)),
                intervention_applied=bool(calibration.get("applied", False)),
            )
        )
    return observations


def _touch_time(record: Mapping[str, Any], role: str) -> float | None:
    packet = _mapping(record.get("packet"))
    players = packet.get("players")
    if not isinstance(players, list):
        return None
    if role == "self":
        index = packet.get("self_index")
    else:
        indices = packet.get("opponent_indices")
        index = indices[0] if isinstance(indices, list) and indices else None
    if not isinstance(index, int) or not 0 <= index < len(players):
        return None
    return _number(_mapping(_mapping(players[index]).get("latest_touch")).get("game_seconds"))


def _touch_outcome(observations: list[DecisionObservation]) -> dict[str, Any]:
    if not observations:
        return {
            "next_touch": "none",
            "time_to_next_self_touch": None,
            "time_to_next_opponent_touch": None,
            "touch_sequence": [],
        }
    anchor = observations[0].game_time
    initial = {
        role: _touch_time(observations[0].record, role)
        for role in ("self", "opponent")
    }
    touches: list[tuple[float, str]] = []
    seen = set(initial.values())
    for observation in observations[1:]:
        for role in ("self", "opponent"):
            value = _touch_time(observation.record, role)
            if value is not None and value not in seen and value >= anchor - 0.1:
                touches.append((value, role))
                seen.add(value)
    touches.sort()
    first_by_role = {
        role: next((time for time, touch_role in touches if touch_role == role), None)
        for role in ("self", "opponent")
    }
    return {
        "next_touch": touches[0][1] if touches else "none",
        "time_to_next_self_touch": (
            None
            if first_by_role["self"] is None
            else first_by_role["self"] - anchor
        ),
        "time_to_next_opponent_touch": (
            None
            if first_by_role["opponent"] is None
            else first_by_role["opponent"] - anchor
        ),
        "touch_sequence": [role for _, role in touches],
    }


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.mean(finite) if finite else None


def _median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    return statistics.median(finite) if finite else None


def _case_metrics(
    observations: list[DecisionObservation],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    distances = [
        _number(observation.estimate.components.get("rival_distance_to_ball"))
        for observation in observations
    ]
    distance_values = [value for value in distances if value is not None]
    eta_advantages = [
        (
            None
            if observation.sample is None
            or observation.sample.self_eta_to_ball is None
            or observation.sample.opponent_eta_to_ball is None
            else observation.sample.opponent_eta_to_ball
            - observation.sample.self_eta_to_ball
        )
        for observation in observations
    ]
    likely_seconds = 0.0
    for first, second, advantage in zip(
        observations,
        observations[1:],
        eta_advantages,
    ):
        if advantage is not None and advantage >= 0.0:
            likely_seconds += max(0.0, second.game_time - first.game_time)
    intervention_indices = [
        index
        for index, observation in enumerate(observations)
        if observation.intervention_applied
    ]
    release_delays: list[float | None] = []
    for index in intervention_indices:
        release = next(
            (
                observation
                for observation in observations[index + 1 :]
                if observation.final_jump_initiation
            ),
            None,
        )
        release_delays.append(
            None
            if release is None
            else 1000.0 * (release.game_time - observations[index].game_time)
        )
    consecutive = 0
    maximum_consecutive = 0
    for observation in observations:
        consecutive = consecutive + 1 if observation.intervention_applied else 0
        maximum_consecutive = max(maximum_consecutive, consecutive)
    releases = [
        observation
        for observation in observations
        if observation.final_jump_initiation
    ]
    return {
        "behavior": parameters.get("behavior"),
        "repetition": parameters.get("repetition"),
        "parameters": dict(parameters),
        "decision_count": len(observations),
        "challenge_candidate_decisions": sum(
            observation.challenge_candidate for observation in observations
        ),
        "premature_release_jump_count": sum(
            observation.premature_release_jump for observation in observations
        ),
        "premature_release_jump_case": any(
            observation.premature_release_jump for observation in observations
        ),
        "grounded_release_jump_count": len(releases),
        "release_delay_ms": release_delays,
        "commitment_at_release": [
            {
                "game_time": observation.game_time,
                "score": observation.estimate.score,
                "state": observation.estimate.state,
                "premature": observation.premature_release_jump,
            }
            for observation in releases
        ],
        "touch_outcome": _touch_outcome(observations),
        "maximum_ball_separation_increase": (
            None
            if not distance_values
            else max(distance_values) - distance_values[0]
        ),
        "ending_ball_separation_change": (
            None
            if not distance_values
            else distance_values[-1] - distance_values[0]
        ),
        "eta_advantage_start": next(
            (value for value in eta_advantages if value is not None), None
        ),
        "eta_advantage_end": next(
            (value for value in reversed(eta_advantages) if value is not None), None
        ),
        "self_likely_next_toucher_seconds": likely_seconds,
        "eligible_decisions": sum(
            observation.intervention_eligible for observation in observations
        ),
        "interventions_applied": len(intervention_indices),
        "maximum_consecutive_deferrals": maximum_consecutive,
        "high_commitment_observed": any(
            observation.estimate.state == "high" for observation in observations
        ),
        "safety_exclusions": dict(
            Counter(
                str(calibration.get("safety_exclusion"))
                for observation in observations
                if (
                    calibration := _mapping(
                        observation.record.get("challenge_calibration")
                    )
                ).get("safety_exclusion")
            )
        ),
    }


def _session_mode(session: EvidenceSession) -> str:
    value = _mapping(session.metadata.get("challenge_calibration")).get("mode")
    if value:
        return str(value)
    for record in session.decisions:
        value = _mapping(record.get("challenge_calibration")).get("mode")
        if value:
            return str(value)
    # Schema-v1/v2 sessions predate the calibrator and are the frozen off baseline.
    return "off"


def analyze_session(
    session: EvidenceSession,
    parameters: ChallengeMetricParameters | None = None,
) -> dict[str, Any]:
    parameters = parameters or ChallengeMetricParameters()
    observations = observe_decisions(session.decisions, parameters)
    mode = _session_mode(session)
    base: dict[str, Any] = {
        "session_id": session.session_id,
        "source": session.source,
        "opponent": session.opponent,
        "mode": mode,
        "baseline_origin": (
            "legacy_schema_v1_v2"
            if session.decisions
            and session.decisions[0].get("schema_version") in (1, 2)
            else "m03_runtime"
        ),
        "raw_sha256": session.raw_sha256,
        "warnings": session.warnings,
        "decision_count": len(observations),
        "challenge_candidate_decisions": sum(
            observation.challenge_candidate for observation in observations
        ),
        "premature_release_jump_count": sum(
            observation.premature_release_jump for observation in observations
        ),
        "eligible_decisions": sum(
            observation.intervention_eligible for observation in observations
        ),
        "interventions_applied": sum(
            observation.intervention_applied for observation in observations
        ),
        "action_distribution_in_challenge_windows": dict(
            Counter(
                str(
                    _mapping(observation.record.get("decision")).get(
                        "final_action_index",
                        _mapping(observation.record.get("decision")).get(
                            "action_index"
                        ),
                    )
                )
                for observation in observations
                if observation.challenge_candidate
            )
        ),
    }
    base["challenge_candidates_per_1000_decisions"] = (
        1000.0 * base["challenge_candidate_decisions"] / len(observations)
        if observations
        else 0.0
    )
    base["premature_release_jumps_per_1000_decisions"] = (
        1000.0 * base["premature_release_jump_count"] / len(observations)
        if observations
        else 0.0
    )
    base["interventions_per_1000_decisions"] = (
        1000.0 * base["interventions_applied"] / len(observations)
        if observations
        else 0.0
    )

    schedule = session.manifest.get("schedule") or []
    if session.source == "controlled_probe" and schedule:
        cases = []
        for entry in schedule:
            start = _number(entry.get("start_game_time"))
            end = _number(entry.get("end_game_time"))
            if start is None or end is None:
                continue
            case_observations = [
                observation
                for observation in observations
                if start <= observation.game_time <= end
            ]
            cases.append(
                _case_metrics(
                    case_observations,
                    _mapping(entry.get("parameters")),
                )
            )
        base["cases"] = cases
    return base


def _aggregate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len(cases),
        "premature_release_jump_cases": sum(
            bool(case["premature_release_jump_case"]) for case in cases
        ),
        "premature_release_jump_count": sum(
            int(case["premature_release_jump_count"]) for case in cases
        ),
        "self_next_touch_cases": sum(
            case["touch_outcome"]["next_touch"] == "self" for case in cases
        ),
        "opponent_next_touch_cases": sum(
            case["touch_outcome"]["next_touch"] == "opponent" for case in cases
        ),
        "none_next_touch_cases": sum(
            case["touch_outcome"]["next_touch"] == "none" for case in cases
        ),
        "mean_maximum_ball_separation_increase": _mean(
            case["maximum_ball_separation_increase"] for case in cases
        ),
        "median_maximum_ball_separation_increase": _median(
            case["maximum_ball_separation_increase"] for case in cases
        ),
        "mean_ending_eta_advantage": _mean(
            case["eta_advantage_end"] for case in cases
        ),
        "eligible_decisions": sum(int(case["eligible_decisions"]) for case in cases),
        "interventions_applied": sum(
            int(case["interventions_applied"]) for case in cases
        ),
        "maximum_consecutive_deferrals": max(
            (int(case["maximum_consecutive_deferrals"]) for case in cases),
            default=0,
        ),
        "high_commitment_cases": sum(
            bool(case["high_commitment_observed"]) for case in cases
        ),
    }


def paired_controlled_summary(session_results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for session in session_results:
        for case in session.get("cases", []):
            behavior = str(case.get("behavior") or "unknown")
            grouped.setdefault(behavior, {}).setdefault(session["mode"], []).append(case)
    by_behavior = {
        behavior: {
            mode: _aggregate_cases(cases)
            for mode, cases in sorted(modes.items())
        }
        for behavior, modes in sorted(grouped.items())
    }
    fake_behaviors = {
        "boost_then_brake",
        "boost_then_veer",
        "jump_fake",
        "delayed_challenge",
    }
    fake_by_mode: dict[str, list[dict[str, Any]]] = {}
    true_by_mode: dict[str, list[dict[str, Any]]] = {}
    for session in session_results:
        for case in session.get("cases", []):
            target = fake_by_mode if case.get("behavior") in fake_behaviors else true_by_mode
            target.setdefault(session["mode"], []).append(case)
    fake_aggregate = {
        mode: _aggregate_cases(cases) for mode, cases in sorted(fake_by_mode.items())
    }
    true_aggregate = {
        mode: _aggregate_cases(cases) for mode, cases in sorted(true_by_mode.items())
    }
    baseline = fake_aggregate.get("off", {})
    treatment = fake_aggregate.get("intervene", {})
    baseline_cases = int(baseline.get("premature_release_jump_cases", 0))
    treatment_cases = (
        int(treatment.get("premature_release_jump_cases", 0))
        if treatment
        else None
    )
    relative_reduction = (
        None
        if baseline_cases == 0 or treatment_cases is None
        else (baseline_cases - treatment_cases) / baseline_cases
    )
    refined_metric_invalidates_broad_target = baseline_cases < 10
    return {
        "by_behavior": by_behavior,
        "fake_pressure_aggregate": fake_aggregate,
        "true_commit_aggregate": true_aggregate,
        "paired_gate_inputs": {
            "premature_release_case_relative_reduction": relative_reduction,
            "refined_metric_invalidates_original_50_percent_target": (
                refined_metric_invalidates_broad_target
            ),
            "baseline_refined_cases_out_of_20": baseline_cases,
            "treatment_refined_cases_out_of_20": treatment_cases,
            "treatment_self_next_touch_delta": (
                None
                if not treatment
                else int(treatment.get("self_next_touch_cases", 0))
                - int(baseline.get("self_next_touch_cases", 0))
            ),
        },
    }


def build_report(
    inputs: list[Path],
    parameters: ChallengeMetricParameters | None = None,
) -> dict[str, Any]:
    parameters = parameters or ChallengeMetricParameters()
    sessions = load_sessions(inputs)
    session_results = [analyze_session(session, parameters) for session in sessions]
    controlled = [
        result for result in session_results if result["source"] == "controlled_probe"
    ]
    natural = [
        result for result in session_results if result["source"] == "natural_match"
    ]
    return {
        "report_schema_version": 1,
        "generated_utc": utc_now(),
        "detector_version": DETECTOR_VERSION,
        "metric_parameters": parameters.to_record(),
        "premature_release_jump_definition": (
            "grounded final jump/dodge initiation while Rival is within the ground-control "
            "distance/height/ETA gate, apparent pressure is present, commitment is below "
            "the high threshold, and the opponent is outside the unavoidable-intercept boundary"
        ),
        "sessions": session_results,
        "paired_controlled": paired_controlled_summary(controlled),
        "natural_sessions": natural,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze Rival telemetry with the versioned M03 challenge metric"
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_report(args.inputs)
    write_json(args.output, report)
    print(
        json.dumps(
            {
                "detector_version": DETECTOR_VERSION,
                "sessions": len(report["sessions"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
