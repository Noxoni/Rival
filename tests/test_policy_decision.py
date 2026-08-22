from __future__ import annotations

from pathlib import Path

import pytest
import torch

from backend.model import ActivationType, ModelInfo, ModelSet
from policy.decision import ControllerAction, PolicyInference
from policy.inspector import PolicyInspector


class DummyAction:
    def __init__(self, index: int = 0) -> None:
        self.throttle = float(index)
        self.steer = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.jump = False
        self.boost = bool(index % 2)
        self.handbrake = False


def _fake_model(logits: torch.Tensor) -> ModelSet:
    model = ModelSet.__new__(ModelSet)
    model.device = torch.device("cpu")
    model._action_disabled_logit = torch.tensor(-1e10, dtype=torch.float32)
    model._forward_logits = lambda _obs: logits.clone()
    return model


def test_compatibility_wrapper_preserves_masked_argmax() -> None:
    logits = torch.tensor([100.0, 3.0, 8.0, -2.0])
    legal_mask = torch.tensor([False, True, True, False])
    model = _fake_model(logits)

    inference = model.infer_policy(torch.zeros(1), legal_mask)
    expected = int(torch.argmax(inference.masked_logits).item())

    assert expected == 2
    assert model.get_action(torch.zeros(1), legal_mask, deterministic=True) == expected
    assert inference.raw_logits.tolist() == logits.tolist()
    assert inference.masked_logits[0].item() == pytest.approx(-1e10)
    assert inference.legal_mask.tolist() == legal_mask.tolist()


def test_empty_mask_preserves_wisp_allow_everything_fallback() -> None:
    logits = torch.tensor([-2.0, 7.0, 1.0])
    model = _fake_model(logits)

    inference = model.infer_policy(torch.zeros(1), torch.zeros(3, dtype=torch.bool))

    assert inference.empty_mask_fallback is True
    assert inference.legal_mask.tolist() == [True, True, True]
    assert inference.select_action(deterministic=True) == 1


def test_mask_shape_must_match_policy_output() -> None:
    model = _fake_model(torch.tensor([1.0, 2.0, 3.0]))

    with pytest.raises(ValueError, match="Action mask length"):
        model.infer_policy(torch.zeros(1), torch.tensor([True, False]))


def test_inspector_top_n_ranking_and_probabilities_are_consistent() -> None:
    raw = torch.tensor([0.0, 2.0, 1.0, 99.0])
    legal = torch.tensor([True, True, True, False])
    masked = raw.clone()
    masked[~legal] = -1e10
    inference = PolicyInference(raw, masked, legal)

    decision = PolicyInspector(top_n=3).inspect(
        inference,
        action_index=1,
        controller_action=DummyAction(1),
        action_resolver=DummyAction,
        tick=42,
        game_time=12.5,
        timestamp_unix_ns=123456,
    )

    assert decision.action_index == 1
    assert decision.controller_action == ControllerAction.from_action(DummyAction(1))
    assert [candidate.action_index for candidate in decision.top_actions] == [1, 2, 0]
    probabilities = [candidate.probability for candidate in decision.top_actions]
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) == pytest.approx(1.0)
    assert decision.confidence == pytest.approx(probabilities[0])
    assert decision.margin == pytest.approx(probabilities[0] - probabilities[1])
    assert decision.tick == 42
    assert decision.game_time == pytest.approx(12.5)
    assert decision.timestamp_unix_ns == 123456

    compact = decision.to_record()
    verbose = decision.to_record(include_logits=True)
    assert "raw_logits" not in compact
    assert verbose["raw_logits"] == raw.tolist()
    assert verbose["masked_logits"][3] == pytest.approx(-1e10)


@pytest.fixture(scope="module")
def real_wisp_model() -> ModelSet:
    model_root = Path(__file__).resolve().parents[1] / "bot" / "models"
    return ModelSet(
        ModelInfo(model_root / "POLICY.lt", ActivationType.RELU),
        ModelInfo(model_root / "SHARED_HEAD.lt", ActivationType.RELU),
        device="cpu",
    )


def test_real_wisp_model_reaches_inference_with_expected_shapes(
    real_wisp_model: ModelSet,
) -> None:
    torch.manual_seed(20260822)
    observation = torch.randn(432, dtype=torch.float32)
    legal_mask = torch.ones(90, dtype=torch.bool)
    legal_mask[::7] = False

    inference = real_wisp_model.infer_policy(observation, legal_mask)
    expected = int(torch.argmax(inference.masked_logits).item())
    compatibility_action = real_wisp_model.get_action(
        observation, legal_mask, deterministic=True
    )

    assert tuple(inference.raw_logits.shape) == (90,)
    assert tuple(inference.masked_logits.shape) == (90,)
    assert torch.isfinite(inference.raw_logits).all()
    assert compatibility_action == expected
    assert legal_mask[compatibility_action]
