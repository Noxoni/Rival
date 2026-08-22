from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from policy.decision import ControllerAction, PolicyDecision
from strategy.challenge_calibration import (
    ChallengeCalibrationController,
    ChallengeCalibrationMode,
    ChallengeCalibrationParameters,
)
from strategy.challenge_commitment import (
    ChallengeCommitmentEstimate,
    ChallengeSample,
)


class _Action:
    def __init__(
        self,
        *,
        throttle: float = 1.0,
        jump: bool = False,
        boost: bool = False,
    ) -> None:
        self.throttle = throttle
        self.steer = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.jump = jump
        self.boost = boost
        self.handbrake = False


ACTIONS = {
    0: _Action(throttle=1.0),
    1: _Action(throttle=1.0, jump=True),
    2: _Action(throttle=-1.0),
    3: _Action(throttle=1.0, boost=True),
}


def _resolve(index: int) -> _Action:
    return ACTIONS[index]


def _decision(
    *,
    logits: tuple[float, ...] = (2.55, 3.0, 2.9, 2.4),
    legal: tuple[bool, ...] = (True, True, False, True),
) -> PolicyDecision:
    raw = torch.tensor(logits)
    legal_mask = torch.tensor(legal)
    masked = raw.clone()
    masked[~legal_mask] = -1e10
    legal_probabilities = torch.softmax(masked[legal_mask], dim=0)
    legal_indices = torch.nonzero(legal_mask, as_tuple=False).view(-1)
    selected_position = int(
        torch.nonzero(legal_indices == 1, as_tuple=False).view(-1)[0].item()
    )
    confidence = float(legal_probabilities[selected_position].item())
    return PolicyDecision(
        action_index=1,
        controller_action=ControllerAction.from_action(ACTIONS[1]),
        raw_logits=raw,
        masked_logits=masked,
        legal_mask=legal_mask,
        top_actions=(),
        confidence=confidence,
        margin=0.0,
        tick=10,
        timestamp_unix_ns=123,
        game_time=2.0,
    )


def _sample(**changes: Any) -> ChallengeSample:
    values: dict[str, Any] = {
        "game_time": 2.0,
        "self_position": (0.0, -400.0, 17.0),
        "self_velocity": (0.0, 300.0, 0.0),
        "opponent_position": (150.0, 850.0, 17.0),
        "opponent_velocity": (250.0, -420.0, 0.0),
        "opponent_forward": (0.3, -0.95, 0.0),
        "ball_position": (0.0, 0.0, 105.0),
        "ball_velocity": (0.0, 180.0, 0.0),
        "self_team": 0,
        "self_grounded": True,
        "opponent_airborne": False,
        "self_demoed": False,
        "opponent_demoed": False,
        "opponent_id": 2,
        "opponent_throttle": 0.7,
        "opponent_steer": 0.0,
        "opponent_jump": False,
        "opponent_boost": False,
        "opponent_handbrake": False,
        "opponent_input_available": True,
        "self_eta_to_ball": 0.42,
        "opponent_eta_to_ball": 0.58,
        "opponent_ball_closing_speed": 520.0,
        "opponent_rival_closing_speed": 600.0,
        "phase": "Active",
        "scores": (0, 0),
    }
    values.update(changes)
    return ChallengeSample(**values)


def _estimate(
    *,
    state: str = "ambiguous",
    episode_id: int = 4,
    pressure: bool = True,
) -> ChallengeCommitmentEstimate:
    return ChallengeCommitmentEstimate(
        valid=True,
        score={"low": 0.2, "ambiguous": 0.52, "high": 0.82}[state],
        state=state,
        pressure_present=pressure,
        abort_detected=False,
        components={
            "rival_distance_to_ball": 410.0,
            "opponent_distance_to_ball": 860.0,
            "projected_miss_distance": 220.0,
        },
        history={"sample_count": 3, "trends": {}},
        reset_reason=None,
        episode_id=episode_id,
    )


class _Tracker:
    def __init__(self, estimates: list[ChallengeCommitmentEstimate]) -> None:
        self.estimates = estimates
        self.index = 0

    def update(self, _sample: ChallengeSample | None) -> ChallengeCommitmentEstimate:
        value = self.estimates[min(self.index, len(self.estimates) - 1)]
        self.index += 1
        return value

    def reset(self, _reason: str = "test") -> None:
        self.index = 0


def _controller(
    mode: str,
    estimates: list[ChallengeCommitmentEstimate] | None = None,
    parameters: ChallengeCalibrationParameters | None = None,
) -> ChallengeCalibrationController:
    controller = ChallengeCalibrationController(mode, parameters)
    controller.tracker = _Tracker(estimates or [_estimate()])  # type: ignore[assignment]
    return controller


def test_off_mode_is_an_inert_exact_baseline_path() -> None:
    controller = ChallengeCalibrationController("off")
    baseline = _decision()

    result = controller.evaluate(
        baseline,
        _sample(),
        lambda _index: pytest.fail("off mode must not resolve another action"),
    )

    assert result.mode is ChallengeCalibrationMode.OFF
    assert result.final_action_index == baseline.action_index
    assert result.final_controller_action == baseline.controller_action
    assert result.hypothetical_action_index is None
    assert result.reason == "mode_off_exact_baseline"


def test_observe_logs_hypothetical_but_returns_baseline() -> None:
    controller = _controller("observe")
    baseline = _decision()

    result = controller.evaluate(baseline, _sample(), _resolve)

    assert result.eligible is True
    assert result.applied is False
    assert result.final_action_index == baseline.action_index
    assert result.hypothetical_action_index == 0
    assert result.continuation is not None
    assert result.continuation.action_index == 0
    assert result.reason == "observe_hypothetical_deferral"


def test_intervene_selects_highest_logit_existing_legal_non_jump_action() -> None:
    controller = _controller("intervene")
    baseline = _decision()

    result = controller.evaluate(baseline, _sample(), _resolve)

    assert result.applied is True
    assert result.final_action_index == 0
    assert baseline.legal_mask[result.final_action_index]
    assert result.final_controller_action == ControllerAction.from_action(ACTIONS[0])
    assert result.final_controller_action.jump is False
    assert result.logit_gap == pytest.approx(0.45)


def test_strong_model_preference_and_gate_exclusions_preserve_baseline() -> None:
    parameters = ChallengeCalibrationParameters(maximum_logit_gap=0.25)
    controller = _controller("intervene", parameters=parameters)

    strong = controller.evaluate(_decision(), _sample(), _resolve)

    assert strong.applied is False
    assert "model_logit_preference_too_strong" in strong.gate_failures
    assert strong.final_action_index == 1

    airborne = _controller("intervene").evaluate(
        _decision(),
        _sample(self_grounded=False),
        _resolve,
    )
    assert airborne.applied is False
    assert "rival_airborne" in airborne.gate_failures


@pytest.mark.parametrize("maximum_ticks", [1, 2])
def test_deferral_budget_is_strictly_bounded(maximum_ticks: int) -> None:
    parameters = ChallengeCalibrationParameters(
        maximum_deferral_policy_ticks=maximum_ticks
    )
    controller = _controller("intervene", parameters=parameters)

    results = [
        controller.evaluate(
            replace(_decision(), tick=10 + index),
            replace(_sample(), game_time=2.0 + 0.07 * index),
            _resolve,
        )
        for index in range(maximum_ticks + 1)
    ]

    assert sum(result.applied for result in results) == maximum_ticks
    assert results[-1].applied is False
    assert "deferral_budget_exhausted" in results[-1].gate_failures
    assert results[-1].remaining_deferral_budget == 0


def test_high_commitment_disables_deferral_immediately() -> None:
    controller = _controller(
        "intervene",
        estimates=[_estimate(), _estimate(state="high")],
        parameters=ChallengeCalibrationParameters(maximum_deferral_policy_ticks=2),
    )

    first = controller.evaluate(_decision(), _sample(), _resolve)
    second = controller.evaluate(
        replace(_decision(), tick=11),
        replace(_sample(), game_time=2.07),
        _resolve,
    )

    assert first.applied is True
    assert second.applied is False
    assert "commitment_high" in second.gate_failures
    assert second.final_action_index == 1
