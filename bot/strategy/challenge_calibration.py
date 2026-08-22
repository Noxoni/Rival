from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Callable, Mapping

import torch

try:
    from policy.decision import ControllerAction, PolicyDecision
except ModuleNotFoundError:  # Package import used by offline repository tools.
    from bot.policy.decision import ControllerAction, PolicyDecision

from .challenge_commitment import (
    ChallengeCommitmentEstimate,
    ChallengeCommitmentParameters,
    ChallengeCommitmentTracker,
    ChallengeSample,
)


class ChallengeCalibrationMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    INTERVENE = "intervene"

    @classmethod
    def parse(cls, value: str | "ChallengeCalibrationMode") -> "ChallengeCalibrationMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"challenge calibration mode must be one of {choices}; got {value!r}"
            ) from exc


@dataclass(frozen=True)
class ChallengeCalibrationParameters:
    """Named conservative intervention parameters for the bounded M03 experiment."""

    version: str = "m03-conservative-v1"
    commitment: ChallengeCommitmentParameters = field(
        default_factory=ChallengeCommitmentParameters
    )
    control_distance: float = 650.0
    maximum_control_ball_height: float = 230.0
    maximum_eta_disadvantage: float = 0.30
    recent_touch_seconds: float = 1.25
    unavoidable_opponent_eta: float = 0.18
    unavoidable_opponent_distance: float = 260.0
    minimum_history_samples: int = 2
    maximum_logit_gap: float = 0.85
    maximum_probability_gap: float = 0.30
    maximum_baseline_confidence: float = 0.65
    minimum_continuation_throttle: float = 0.0
    maximum_deferral_policy_ticks: int = 1

    def __post_init__(self) -> None:
        if self.control_distance <= 0.0:
            raise ValueError("control_distance must be positive")
        if self.maximum_control_ball_height <= 0.0:
            raise ValueError("maximum_control_ball_height must be positive")
        if self.minimum_history_samples < 1:
            raise ValueError("minimum_history_samples must be at least one")
        if self.maximum_logit_gap < 0.0:
            raise ValueError("maximum_logit_gap cannot be negative")
        if not 0.0 <= self.maximum_probability_gap <= 1.0:
            raise ValueError("maximum_probability_gap must be within [0, 1]")
        if not 0.0 <= self.maximum_baseline_confidence <= 1.0:
            raise ValueError("maximum_baseline_confidence must be within [0, 1]")
        if self.maximum_deferral_policy_ticks not in (1, 2):
            raise ValueError("maximum_deferral_policy_ticks must be one or two")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["commitment"] = self.commitment.to_record()
        return record


@dataclass(frozen=True)
class ContinuationCandidate:
    action_index: int
    controller_action: ControllerAction
    logit: float
    probability: float

    def to_record(self) -> dict[str, Any]:
        return {
            "action_index": self.action_index,
            "controller_action": self.controller_action.to_record(),
            "logit": self.logit,
            "probability": self.probability,
        }


@dataclass(frozen=True)
class ChallengeCalibrationDecision:
    mode: ChallengeCalibrationMode
    parameter_version: str
    baseline_action_index: int
    baseline_controller_action: ControllerAction
    final_action_index: int
    final_controller_action: ControllerAction
    hypothetical_action_index: int | None
    hypothetical_controller_action: ControllerAction | None
    eligible: bool
    applied: bool
    reason: str
    gate_failures: tuple[str, ...]
    gate_components: Mapping[str, Any]
    safety_exclusion: str | None
    estimate: ChallengeCommitmentEstimate
    continuation: ContinuationCandidate | None
    baseline_logit: float | None
    baseline_probability: float
    logit_gap: float | None
    probability_gap: float | None
    deferrals_used: int
    remaining_deferral_budget: int

    @classmethod
    def exact_baseline(
        cls,
        baseline: PolicyDecision,
        parameters: ChallengeCalibrationParameters,
    ) -> "ChallengeCalibrationDecision":
        return cls(
            mode=ChallengeCalibrationMode.OFF,
            parameter_version=parameters.version,
            baseline_action_index=baseline.action_index,
            baseline_controller_action=baseline.controller_action,
            final_action_index=baseline.action_index,
            final_controller_action=baseline.controller_action,
            hypothetical_action_index=None,
            hypothetical_controller_action=None,
            eligible=False,
            applied=False,
            reason="mode_off_exact_baseline",
            gate_failures=("mode_off",),
            gate_components={},
            safety_exclusion=None,
            estimate=ChallengeCommitmentEstimate.unavailable("mode_off"),
            continuation=None,
            baseline_logit=None,
            baseline_probability=baseline.confidence,
            logit_gap=None,
            probability_gap=None,
            deferrals_used=0,
            remaining_deferral_budget=parameters.maximum_deferral_policy_ticks,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "parameter_version": self.parameter_version,
            "baseline_action": {
                "action_index": self.baseline_action_index,
                "controller_action": self.baseline_controller_action.to_record(),
                "logit": self.baseline_logit,
                "probability": self.baseline_probability,
            },
            "final_action": {
                "action_index": self.final_action_index,
                "controller_action": self.final_controller_action.to_record(),
            },
            "hypothetical_action": (
                None
                if self.hypothetical_action_index is None
                or self.hypothetical_controller_action is None
                else {
                    "action_index": self.hypothetical_action_index,
                    "controller_action": self.hypothetical_controller_action.to_record(),
                }
            ),
            "eligible": self.eligible,
            "applied": self.applied,
            "reason": self.reason,
            "gate_failures": list(self.gate_failures),
            "gate_components": dict(self.gate_components),
            "safety_exclusion": self.safety_exclusion,
            "commitment": self.estimate.to_record(),
            "continuation_candidate": (
                None if self.continuation is None else self.continuation.to_record()
            ),
            "logit_gap": self.logit_gap,
            "probability_gap": self.probability_gap,
            "deferrals_used": self.deferrals_used,
            "remaining_deferral_budget": self.remaining_deferral_budget,
        }


def _finite_tensor_value(tensor: torch.Tensor, index: int) -> float | None:
    value = float(tensor[index].detach().cpu().item())
    return value if math.isfinite(value) else None


class ChallengeCalibrationController:
    """Stateful, one-episode-bounded re-ranker over legal Wisp actions."""

    def __init__(
        self,
        mode: str | ChallengeCalibrationMode = ChallengeCalibrationMode.OFF,
        parameters: ChallengeCalibrationParameters | None = None,
    ) -> None:
        self.mode = ChallengeCalibrationMode.parse(mode)
        self.parameters = parameters or ChallengeCalibrationParameters()
        self.tracker = ChallengeCommitmentTracker(self.parameters.commitment)
        self._episode_id: int | None = None
        self._deferrals_used = 0
        self._previous_final_jump = False

    def reset(self, reason: str = "calibration_reset") -> None:
        self.tracker.reset(reason)
        self._episode_id = None
        self._deferrals_used = 0
        self._previous_final_jump = False

    def _continuation_candidate(
        self,
        baseline: PolicyDecision,
        action_resolver: Callable[[int], Any],
    ) -> ContinuationCandidate | None:
        legal_indices = torch.nonzero(baseline.legal_mask, as_tuple=False).view(-1)
        if legal_indices.numel() == 0:
            return None
        legal_logits = baseline.masked_logits[legal_indices]
        legal_probabilities = torch.softmax(legal_logits, dim=0)
        candidates: list[tuple[float, int, int, ControllerAction]] = []
        for position, raw_index in enumerate(legal_indices.detach().cpu().tolist()):
            index = int(raw_index)
            action = ControllerAction.from_action(action_resolver(index))
            if action.jump or action.throttle < self.parameters.minimum_continuation_throttle:
                continue
            logit = _finite_tensor_value(baseline.masked_logits, index)
            if logit is None:
                continue
            candidates.append((logit, -index, position, action))
        if not candidates:
            return None
        logit, negative_index, position, action = max(candidates)
        index = -negative_index
        probability = float(legal_probabilities[position].detach().cpu().item())
        return ContinuationCandidate(index, action, logit, probability)

    def evaluate(
        self,
        baseline: PolicyDecision,
        sample: ChallengeSample | None,
        action_resolver: Callable[[int], Any],
    ) -> ChallengeCalibrationDecision:
        parameters = self.parameters
        if self.mode is ChallengeCalibrationMode.OFF:
            return ChallengeCalibrationDecision.exact_baseline(baseline, parameters)

        estimate = self.tracker.update(sample)
        if estimate.episode_id != self._episode_id:
            self._episode_id = estimate.episode_id
            self._deferrals_used = 0
        if estimate.reset_reason is not None:
            self._previous_final_jump = False

        baseline_logit = _finite_tensor_value(
            baseline.masked_logits, baseline.action_index
        )
        continuation = self._continuation_candidate(baseline, action_resolver)
        logit_gap = (
            None
            if baseline_logit is None or continuation is None
            else baseline_logit - continuation.logit
        )
        probability_gap = (
            None
            if continuation is None
            else baseline.confidence - continuation.probability
        )

        sample_count = int(estimate.history.get("sample_count", 0) or 0)
        components = estimate.components
        opponent_distance = components.get("opponent_distance_to_ball")
        ball_height = None if sample is None else sample.ball_position[2]
        eta_advantage = (
            None
            if sample is None
            or sample.self_eta_to_ball is None
            or sample.opponent_eta_to_ball is None
            else sample.opponent_eta_to_ball - sample.self_eta_to_ball
        )
        recent_self_touch = bool(
            sample is not None
            and sample.self_latest_touch_time is not None
            and 0.0
            <= sample.game_time - sample.self_latest_touch_time
            <= parameters.recent_touch_seconds
        )
        gate_components: dict[str, Any] = {
            "self_grounded": None if sample is None else sample.self_grounded,
            "ball_distance": components.get("rival_distance_to_ball"),
            "ball_height": ball_height,
            "eta_advantage": eta_advantage,
            "recent_self_touch": recent_self_touch,
            "pressure_present": estimate.pressure_present,
            "commitment_state": estimate.state,
            "history_sample_count": sample_count,
            "baseline_jump": baseline.controller_action.jump,
            "grounded_jump_initiation": bool(
                baseline.controller_action.jump and not self._previous_final_jump
            ),
            "opponent_distance_to_ball": opponent_distance,
            "opponent_eta_to_ball": None if sample is None else sample.opponent_eta_to_ball,
            "deferrals_used": self._deferrals_used,
        }

        failures: list[str] = []
        safety_exclusion: str | None = None
        if not estimate.valid or sample is None:
            failures.append("invalid_estimate")
        if sample is not None and sample.reset_or_kickoff:
            failures.append("reset_or_kickoff")
        if sample is not None and not sample.self_grounded:
            failures.append("rival_airborne")
        if sample_count < parameters.minimum_history_samples:
            failures.append("insufficient_history")
        if not baseline.controller_action.jump:
            failures.append("baseline_not_grounded_jump")
        elif self._previous_final_jump:
            failures.append("baseline_jump_not_initiation")
        rival_distance = components.get("rival_distance_to_ball")
        if not isinstance(rival_distance, (int, float)) or (
            float(rival_distance) > parameters.control_distance
        ):
            failures.append("outside_control_distance")
        if ball_height is None or ball_height > parameters.maximum_control_ball_height:
            failures.append("ball_too_high_for_ground_control")
        if eta_advantage is not None and eta_advantage < -parameters.maximum_eta_disadvantage:
            failures.append("eta_not_competitive")
        if not estimate.pressure_present:
            failures.append("no_apparent_pressure")
        if estimate.state != "ambiguous":
            failures.append(f"commitment_{estimate.state}")

        if sample is not None and sample.defensive_emergency:
            safety_exclusion = "defensive_emergency"
        elif sample is not None and (
            (
                sample.opponent_eta_to_ball is not None
                and sample.opponent_eta_to_ball <= parameters.unavoidable_opponent_eta
            )
            or (
                isinstance(opponent_distance, (int, float))
                and float(opponent_distance) <= parameters.unavoidable_opponent_distance
            )
        ):
            safety_exclusion = "unavoidable_intercept"
        elif continuation is None:
            safety_exclusion = "no_legal_non_jump_continuation"
        if safety_exclusion is not None:
            failures.append(safety_exclusion)

        if logit_gap is None or logit_gap > parameters.maximum_logit_gap:
            failures.append("model_logit_preference_too_strong")
        if (
            probability_gap is None
            or probability_gap > parameters.maximum_probability_gap
        ):
            failures.append("model_probability_preference_too_strong")
        if baseline.confidence > parameters.maximum_baseline_confidence:
            failures.append("baseline_confidence_too_high")
        if self._deferrals_used >= parameters.maximum_deferral_policy_ticks:
            failures.append("deferral_budget_exhausted")

        eligible = not failures
        hypothetical_index = continuation.action_index if eligible and continuation else None
        hypothetical_action = (
            continuation.controller_action if eligible and continuation else None
        )
        applied = bool(
            eligible
            and continuation is not None
            and self.mode is ChallengeCalibrationMode.INTERVENE
        )
        final_index = continuation.action_index if applied and continuation else baseline.action_index
        final_action = (
            continuation.controller_action
            if applied and continuation
            else baseline.controller_action
        )
        if applied:
            self._deferrals_used += 1
            reason = "ambiguous_pressure_one_tick_deferral"
        elif eligible:
            reason = "observe_hypothetical_deferral"
        else:
            reason = failures[0]

        self._previous_final_jump = final_action.jump
        return ChallengeCalibrationDecision(
            mode=self.mode,
            parameter_version=parameters.version,
            baseline_action_index=baseline.action_index,
            baseline_controller_action=baseline.controller_action,
            final_action_index=final_index,
            final_controller_action=final_action,
            hypothetical_action_index=hypothetical_index,
            hypothetical_controller_action=hypothetical_action,
            eligible=eligible,
            applied=applied,
            reason=reason,
            gate_failures=tuple(failures),
            gate_components=gate_components,
            safety_exclusion=safety_exclusion,
            estimate=estimate,
            continuation=continuation,
            baseline_logit=baseline_logit,
            baseline_probability=baseline.confidence,
            logit_gap=logit_gap,
            probability_gap=probability_gap,
            deferrals_used=self._deferrals_used,
            remaining_deferral_budget=max(
                0, parameters.maximum_deferral_policy_ticks - self._deferrals_used
            ),
        )
