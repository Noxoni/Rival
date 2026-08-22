from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Callable, Mapping

import torch

try:
    from analysis.tactical_metrics import TacticalMetrics
    from policy.decision import ControllerAction, PolicyDecision
except ModuleNotFoundError:  # Package imports used by repository tests/tools.
    from bot.analysis.tactical_metrics import TacticalMetrics
    from bot.policy.decision import ControllerAction, PolicyDecision

from .challenge_commitment import ChallengeSample


SUPPORTED_PARAMETER_VERSION = "m04p1-low-resource-aerial-v1"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _finite_tensor_value(tensor: torch.Tensor, index: int) -> float | None:
    value = float(tensor[index].detach().cpu().item())
    return value if math.isfinite(value) else None


class NaturalAdjustmentMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    INTERVENE = "intervene"

    @classmethod
    def parse(cls, value: str | "NaturalAdjustmentMode") -> "NaturalAdjustmentMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"natural adjustment mode must be one of {choices}; got {value!r}"
            ) from exc


@dataclass(frozen=True)
class NaturalAdjustmentParameters:
    """Prospective parameters selected from the v4.1 natural baseline."""

    version: str = SUPPORTED_PARAMETER_VERSION
    maximum_boost: float = 30.0
    minimum_ball_height: float = 300.0
    minimum_ball_distance: float = 650.0
    maximum_eta_advantage: float = 0.0
    eta_disadvantage_reference: float = 1.0
    ball_height_reference: float = 700.0
    ball_distance_reference: float = 1850.0
    minimum_logit_penalty: float = 0.15
    maximum_logit_penalty: float = 0.55

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("natural adjustment parameter version cannot be empty")
        if self.maximum_boost <= 0.0:
            raise ValueError("maximum_boost must be positive")
        if self.minimum_ball_height <= 0.0 or self.minimum_ball_distance <= 0.0:
            raise ValueError("ball height and distance thresholds must be positive")
        if self.eta_disadvantage_reference <= 0.0:
            raise ValueError("eta_disadvantage_reference must be positive")
        if self.ball_height_reference <= 0.0 or self.ball_distance_reference <= 0.0:
            raise ValueError("ball severity references must be positive")
        if self.minimum_logit_penalty < 0.0:
            raise ValueError("minimum_logit_penalty cannot be negative")
        if self.maximum_logit_penalty < self.minimum_logit_penalty:
            raise ValueError("maximum_logit_penalty must be at least the minimum")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class NaturalAdjustmentSample:
    self_boost: float
    ball_height: float
    distance_self_ball: float
    possession_eta_advantage: float | None
    self_grounded: bool
    phase: str | None
    defensive_emergency: bool

    @classmethod
    def from_live(
        cls,
        live_sample: ChallengeSample | None,
        tactical_metrics: TacticalMetrics,
    ) -> "NaturalAdjustmentSample | None":
        if live_sample is None:
            return None
        return cls(
            self_boost=float(tactical_metrics.self_boost),
            ball_height=float(tactical_metrics.ball_height),
            distance_self_ball=float(tactical_metrics.distance_self_ball),
            possession_eta_advantage=tactical_metrics.possession_eta_advantage,
            self_grounded=bool(live_sample.self_grounded),
            phase=live_sample.phase,
            defensive_emergency=live_sample.defensive_emergency,
        )


@dataclass(frozen=True)
class NaturalAdjustmentCandidate:
    action_index: int
    controller_action: ControllerAction
    raw_logit: float
    adjusted_logit: float
    raw_probability: float
    penalty: float

    def to_record(self) -> dict[str, Any]:
        return {
            "action_index": self.action_index,
            "controller_action": self.controller_action.to_record(),
            "raw_logit": self.raw_logit,
            "adjusted_logit": self.adjusted_logit,
            "raw_probability": self.raw_probability,
            "penalty": self.penalty,
        }


@dataclass(frozen=True)
class NaturalAdjustmentDecision:
    mode: NaturalAdjustmentMode
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
    state_features: Mapping[str, Any]
    severity_components: Mapping[str, float]
    severity: float
    logit_penalty: float
    baseline_raw_logit: float | None
    baseline_adjusted_logit: float | None
    candidate: NaturalAdjustmentCandidate | None

    @classmethod
    def exact_baseline(
        cls,
        baseline: PolicyDecision,
        parameters: NaturalAdjustmentParameters,
    ) -> "NaturalAdjustmentDecision":
        return cls(
            mode=NaturalAdjustmentMode.OFF,
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
            state_features={},
            severity_components={},
            severity=0.0,
            logit_penalty=0.0,
            baseline_raw_logit=None,
            baseline_adjusted_logit=None,
            candidate=None,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "parameter_version": self.parameter_version,
            "baseline_action": {
                "action_index": self.baseline_action_index,
                "controller_action": self.baseline_controller_action.to_record(),
                "raw_logit": self.baseline_raw_logit,
                "adjusted_logit": self.baseline_adjusted_logit,
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
                    "controller_action": (
                        self.hypothetical_controller_action.to_record()
                    ),
                }
            ),
            "eligible": self.eligible,
            "applied": self.applied,
            "reason": self.reason,
            "gate_failures": list(self.gate_failures),
            "state_features": dict(self.state_features),
            "severity_components": dict(self.severity_components),
            "severity": self.severity,
            "logit_penalty": self.logit_penalty,
            "candidate": None if self.candidate is None else self.candidate.to_record(),
        }


class NaturalAdjustmentController:
    """One-transition graded re-ranker for low-resource losing-race aerials."""

    def __init__(
        self,
        mode: str | NaturalAdjustmentMode = NaturalAdjustmentMode.OFF,
        parameters: NaturalAdjustmentParameters | None = None,
    ) -> None:
        self.mode = NaturalAdjustmentMode.parse(mode)
        self.parameters = parameters or NaturalAdjustmentParameters()
        if (
            self.mode is not NaturalAdjustmentMode.OFF
            and self.parameters.version != SUPPORTED_PARAMETER_VERSION
        ):
            raise ValueError(
                "unsupported active natural adjustment parameter version: "
                f"{self.parameters.version!r}"
            )
        self._previous_baseline_aerial_like = False

    def reset(self, _reason: str = "natural_adjustment_reset") -> None:
        self._previous_baseline_aerial_like = False

    @staticmethod
    def _aerial_like(action: ControllerAction, grounded: bool) -> bool:
        rotational = any(
            abs(value) > 1e-6 for value in (action.pitch, action.yaw, action.roll)
        )
        return action.jump or (not grounded and (action.boost or rotational))

    @staticmethod
    def _resource_commitment(action: ControllerAction, grounded: bool) -> bool:
        return action.jump if grounded else action.boost

    def _severity(
        self,
        sample: NaturalAdjustmentSample,
    ) -> tuple[dict[str, float], float, float]:
        parameters = self.parameters
        eta = sample.possession_eta_advantage or 0.0
        components = {
            "boost_deficit": _clamp01(
                (parameters.maximum_boost - sample.self_boost)
                / parameters.maximum_boost
            ),
            "eta_disadvantage": _clamp01(
                (parameters.maximum_eta_advantage - eta)
                / parameters.eta_disadvantage_reference
            ),
            "ball_height": _clamp01(
                (sample.ball_height - parameters.minimum_ball_height)
                / parameters.ball_height_reference
            ),
            "ball_distance": _clamp01(
                (sample.distance_self_ball - parameters.minimum_ball_distance)
                / parameters.ball_distance_reference
            ),
        }
        severity = (
            0.45 * components["boost_deficit"]
            + 0.30 * components["eta_disadvantage"]
            + 0.15 * components["ball_height"]
            + 0.10 * components["ball_distance"]
        )
        penalty = parameters.minimum_logit_penalty + severity * (
            parameters.maximum_logit_penalty - parameters.minimum_logit_penalty
        )
        return components, severity, penalty

    def _adjusted_candidate(
        self,
        baseline: PolicyDecision,
        grounded: bool,
        penalty: float,
        action_resolver: Callable[[int], Any],
    ) -> NaturalAdjustmentCandidate | None:
        legal_indices = torch.nonzero(baseline.legal_mask, as_tuple=False).view(-1)
        if legal_indices.numel() == 0:
            return None
        legal_logits = baseline.masked_logits[legal_indices]
        probabilities = torch.softmax(legal_logits, dim=0)
        candidates: list[tuple[float, int, NaturalAdjustmentCandidate]] = []
        for position, raw_index in enumerate(legal_indices.detach().cpu().tolist()):
            index = int(raw_index)
            raw_logit = _finite_tensor_value(baseline.masked_logits, index)
            if raw_logit is None:
                continue
            action = ControllerAction.from_action(action_resolver(index))
            action_penalty = (
                penalty if self._resource_commitment(action, grounded) else 0.0
            )
            adjusted_logit = raw_logit - action_penalty
            candidate = NaturalAdjustmentCandidate(
                action_index=index,
                controller_action=action,
                raw_logit=raw_logit,
                adjusted_logit=adjusted_logit,
                raw_probability=float(probabilities[position].detach().cpu().item()),
                penalty=action_penalty,
            )
            # Match torch.argmax's stable first-index tie behavior.
            candidates.append((adjusted_logit, -index, candidate))
        return max(candidates, default=(0.0, 0, None))[2]

    def evaluate(
        self,
        baseline: PolicyDecision,
        sample: NaturalAdjustmentSample | None,
        action_resolver: Callable[[int], Any],
    ) -> NaturalAdjustmentDecision:
        parameters = self.parameters
        if self.mode is NaturalAdjustmentMode.OFF:
            return NaturalAdjustmentDecision.exact_baseline(baseline, parameters)

        if sample is None:
            self.reset("missing_live_sample")
            return self._blocked(
                baseline,
                reason="missing_live_sample",
                failures=("missing_live_sample",),
            )
        if sample.phase != "Active":
            self.reset("reset_or_kickoff")
            return self._blocked(
                baseline,
                sample=sample,
                reason="reset_or_kickoff",
                failures=("reset_or_kickoff",),
            )

        baseline_aerial = self._aerial_like(
            baseline.controller_action,
            sample.self_grounded,
        )
        aerial_transition = baseline_aerial and not self._previous_baseline_aerial_like
        self._previous_baseline_aerial_like = baseline_aerial
        state_features = {
            "self_boost": sample.self_boost,
            "ball_height": sample.ball_height,
            "distance_self_ball": sample.distance_self_ball,
            "possession_eta_advantage": sample.possession_eta_advantage,
            "self_grounded": sample.self_grounded,
            "phase": sample.phase,
            "defensive_emergency": sample.defensive_emergency,
            "baseline_aerial_like": baseline_aerial,
            "aerial_transition": aerial_transition,
            "baseline_resource_commitment": self._resource_commitment(
                baseline.controller_action,
                sample.self_grounded,
            ),
        }
        failures: list[str] = []
        if not aerial_transition:
            failures.append("not_aerial_transition")
        if not state_features["baseline_resource_commitment"]:
            failures.append("baseline_not_resource_commitment")
        if sample.self_boost >= parameters.maximum_boost:
            failures.append("boost_not_low")
        if sample.ball_height < parameters.minimum_ball_height:
            failures.append("ball_not_elevated")
        if sample.distance_self_ball < parameters.minimum_ball_distance:
            failures.append("ball_not_distant")
        eta = sample.possession_eta_advantage
        if eta is None:
            failures.append("eta_unavailable")
        elif eta > parameters.maximum_eta_advantage:
            failures.append("eta_favorable")
        if sample.defensive_emergency:
            failures.append("defensive_emergency")
        if failures:
            return self._blocked(
                baseline,
                sample=sample,
                state_features=state_features,
                reason="state_gate_blocked",
                failures=tuple(failures),
            )

        components, severity, penalty = self._severity(sample)
        candidate = self._adjusted_candidate(
            baseline,
            sample.self_grounded,
            penalty,
            action_resolver,
        )
        baseline_logit = _finite_tensor_value(
            baseline.masked_logits,
            baseline.action_index,
        )
        baseline_adjusted = (
            None if baseline_logit is None else baseline_logit - penalty
        )
        changes_argmax = bool(
            candidate is not None and candidate.action_index != baseline.action_index
        )
        if not changes_argmax:
            return NaturalAdjustmentDecision(
                mode=self.mode,
                parameter_version=parameters.version,
                baseline_action_index=baseline.action_index,
                baseline_controller_action=baseline.controller_action,
                final_action_index=baseline.action_index,
                final_controller_action=baseline.controller_action,
                hypothetical_action_index=None,
                hypothetical_controller_action=None,
                eligible=True,
                applied=False,
                reason="adjusted_argmax_unchanged",
                gate_failures=(),
                state_features=state_features,
                severity_components=components,
                severity=severity,
                logit_penalty=penalty,
                baseline_raw_logit=baseline_logit,
                baseline_adjusted_logit=baseline_adjusted,
                candidate=candidate,
            )

        assert candidate is not None
        applied = self.mode is NaturalAdjustmentMode.INTERVENE
        final_index = candidate.action_index if applied else baseline.action_index
        final_action = (
            candidate.controller_action if applied else baseline.controller_action
        )
        return NaturalAdjustmentDecision(
            mode=self.mode,
            parameter_version=parameters.version,
            baseline_action_index=baseline.action_index,
            baseline_controller_action=baseline.controller_action,
            final_action_index=final_index,
            final_controller_action=final_action,
            hypothetical_action_index=candidate.action_index,
            hypothetical_controller_action=candidate.controller_action,
            eligible=True,
            applied=applied,
            reason=(
                "graded_resource_possession_rerank"
                if applied
                else "observe_graded_resource_possession_rerank"
            ),
            gate_failures=(),
            state_features=state_features,
            severity_components=components,
            severity=severity,
            logit_penalty=penalty,
            baseline_raw_logit=baseline_logit,
            baseline_adjusted_logit=baseline_adjusted,
            candidate=candidate,
        )

    def _blocked(
        self,
        baseline: PolicyDecision,
        *,
        sample: NaturalAdjustmentSample | None = None,
        state_features: Mapping[str, Any] | None = None,
        reason: str,
        failures: tuple[str, ...],
    ) -> NaturalAdjustmentDecision:
        features = dict(state_features or {})
        if sample is not None and not features:
            features = {
                "self_boost": sample.self_boost,
                "ball_height": sample.ball_height,
                "distance_self_ball": sample.distance_self_ball,
                "possession_eta_advantage": sample.possession_eta_advantage,
                "self_grounded": sample.self_grounded,
                "phase": sample.phase,
                "defensive_emergency": sample.defensive_emergency,
            }
        return NaturalAdjustmentDecision(
            mode=self.mode,
            parameter_version=self.parameters.version,
            baseline_action_index=baseline.action_index,
            baseline_controller_action=baseline.controller_action,
            final_action_index=baseline.action_index,
            final_controller_action=baseline.controller_action,
            hypothetical_action_index=None,
            hypothetical_controller_action=None,
            eligible=False,
            applied=False,
            reason=reason,
            gate_failures=failures,
            state_features=features,
            severity_components={},
            severity=0.0,
            logit_penalty=0.0,
            baseline_raw_logit=None,
            baseline_adjusted_logit=None,
            candidate=None,
        )
