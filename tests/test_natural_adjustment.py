from __future__ import annotations

from typing import Any

import pytest
import torch

from policy.decision import ControllerAction, PolicyDecision
from strategy.natural_adjustment import (
    NaturalAdjustmentController,
    NaturalAdjustmentMode,
    NaturalAdjustmentParameters,
    NaturalAdjustmentSample,
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
        self.pitch = -1.0 if jump else 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.jump = jump
        self.boost = boost
        self.handbrake = jump


ACTIONS = {
    0: _Action(),
    1: _Action(jump=True, boost=True),
    2: _Action(boost=True),
}


def _resolve(index: int) -> _Action:
    return ACTIONS[index]


def _decision(
    *,
    selected: int = 1,
    logits: tuple[float, ...] = (2.8, 3.0, 2.5),
) -> PolicyDecision:
    raw = torch.tensor(logits)
    legal = torch.tensor([True, True, True])
    probabilities = torch.softmax(raw, dim=0)
    return PolicyDecision(
        action_index=selected,
        controller_action=ControllerAction.from_action(ACTIONS[selected]),
        raw_logits=raw,
        masked_logits=raw,
        legal_mask=legal,
        top_actions=(),
        confidence=float(probabilities[selected].item()),
        margin=0.0,
        tick=7,
        timestamp_unix_ns=123,
        game_time=3.5,
    )


def _sample(**changes: Any) -> NaturalAdjustmentSample:
    values: dict[str, Any] = {
        "self_boost": 5.0,
        "ball_height": 850.0,
        "distance_self_ball": 1600.0,
        "possession_eta_advantage": -0.6,
        "self_grounded": True,
        "phase": "Active",
        "defensive_emergency": False,
    }
    values.update(changes)
    return NaturalAdjustmentSample(**values)


def test_off_mode_is_exact_and_does_not_resolve_actions() -> None:
    baseline = _decision()
    controller = NaturalAdjustmentController("off")

    result = controller.evaluate(
        baseline,
        _sample(),
        lambda _index: pytest.fail("off mode must not resolve another action"),
    )

    assert result.mode is NaturalAdjustmentMode.OFF
    assert result.final_action_index == baseline.action_index
    assert result.final_controller_action == baseline.controller_action
    assert result.reason == "mode_off_exact_baseline"


def test_observe_and_intervene_use_existing_legal_adjusted_argmax() -> None:
    baseline = _decision()
    observed = NaturalAdjustmentController("observe").evaluate(
        baseline,
        _sample(),
        _resolve,
    )
    applied = NaturalAdjustmentController("intervene").evaluate(
        baseline,
        _sample(),
        _resolve,
    )

    assert observed.eligible is True
    assert observed.applied is False
    assert observed.final_action_index == baseline.action_index
    assert observed.hypothetical_action_index == 0
    assert applied.applied is True
    assert applied.final_action_index == 0
    assert bool(baseline.legal_mask[applied.final_action_index])
    assert applied.final_controller_action == ControllerAction.from_action(ACTIONS[0])
    assert 0.15 <= applied.logit_penalty <= 0.55


def test_adjustment_is_one_transition_bounded_and_resets() -> None:
    controller = NaturalAdjustmentController("intervene")

    first = controller.evaluate(_decision(), _sample(), _resolve)
    repeated = controller.evaluate(_decision(), _sample(), _resolve)
    reset = controller.evaluate(
        _decision(selected=0, logits=(3.0, 2.0, 1.0)),
        _sample(),
        _resolve,
    )
    next_transition = controller.evaluate(_decision(), _sample(), _resolve)

    assert first.applied is True
    assert repeated.applied is False
    assert "not_aerial_transition" in repeated.gate_failures
    assert reset.applied is False
    assert next_transition.applied is True


@pytest.mark.parametrize(
    ("changes", "failure"),
    [
        ({"self_boost": 30.0}, "boost_not_low"),
        ({"ball_height": 299.0}, "ball_not_elevated"),
        ({"distance_self_ball": 649.0}, "ball_not_distant"),
        ({"possession_eta_advantage": 0.01}, "eta_favorable"),
        ({"defensive_emergency": True}, "defensive_emergency"),
        ({"phase": "Kickoff"}, "reset_or_kickoff"),
    ],
)
def test_state_and_safety_gates_preserve_baseline(
    changes: dict[str, Any],
    failure: str,
) -> None:
    baseline = _decision()
    result = NaturalAdjustmentController("intervene").evaluate(
        baseline,
        _sample(**changes),
        _resolve,
    )

    assert result.applied is False
    assert result.final_action_index == baseline.action_index
    assert failure in result.gate_failures


def test_logit_penalty_is_graded_by_live_resource_and_race_state() -> None:
    mild = NaturalAdjustmentController("observe").evaluate(
        _decision(),
        _sample(
            self_boost=29.0,
            ball_height=301.0,
            distance_self_ball=651.0,
            possession_eta_advantage=-0.01,
        ),
        _resolve,
    )
    severe = NaturalAdjustmentController("observe").evaluate(
        _decision(),
        _sample(
            self_boost=0.0,
            ball_height=1200.0,
            distance_self_ball=3000.0,
            possession_eta_advantage=-2.0,
        ),
        _resolve,
    )

    assert mild.eligible is True
    assert severe.eligible is True
    assert mild.logit_penalty < severe.logit_penalty
    assert severe.logit_penalty == pytest.approx(
        NaturalAdjustmentParameters().maximum_logit_penalty
    )
