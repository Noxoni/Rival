from __future__ import annotations

from dataclasses import replace
import json

import torch

from analysis.tactical_metrics import TacticalMetrics
from policy.decision import (
    ActionCandidate,
    ControllerAction,
    PolicyDecision,
)
from telemetry.decision_logger import DecisionTelemetryLogger
from strategy.challenge_calibration import (
    ChallengeCalibrationDecision,
    ChallengeCalibrationMode,
    ChallengeCalibrationParameters,
)
from strategy.natural_adjustment import (
    NaturalAdjustmentDecision,
    NaturalAdjustmentMode,
    NaturalAdjustmentParameters,
)


ACTION = ControllerAction(1.0, 0.0, 0.0, 0.0, 0.0, False, True, False)


def _decision() -> PolicyDecision:
    raw = torch.tensor([0.5, 1.5])
    legal = torch.tensor([True, True])
    return PolicyDecision(
        action_index=1,
        controller_action=ACTION,
        raw_logits=raw,
        masked_logits=raw,
        legal_mask=legal,
        top_actions=(ActionCandidate(1, ACTION, 1.5, 0.7310586),),
        confidence=0.7310586,
        margin=0.4621172,
        tick=7,
        timestamp_unix_ns=123456789,
        game_time=3.5,
    )


def _metrics() -> TacticalMetrics:
    return TacticalMetrics(
        self_boost=20.0,
        opponent_boost=None,
        ball_height=100.0,
        ball_distance=500.0,
        distance_self_ball=500.0,
        distance_opponent_ball=None,
        eta_self_ball=1.0,
        eta_opponent_ball=None,
        eta_method="test",
        challenge_closing_velocity=None,
        self_ball_closing_velocity=500.0,
        opponent_ball_closing_velocity=None,
        possession_eta_advantage=None,
        self_airborne=False,
        opponent_airborne=None,
        selected_action_uses_boost=True,
        selected_action_uses_jump=False,
        selected_action_aerial_like=False,
        score_diff=0,
        seconds_remaining=300.0,
    )


def test_disabled_telemetry_creates_no_output(tmp_path) -> None:
    output = tmp_path / "missing" / "decisions.jsonl"
    logger = DecisionTelemetryLogger(output, enabled=False)

    assert logger.log(_decision(), _metrics(), {}, {}) is False
    logger.close()
    assert not output.exists()
    assert not output.parent.exists()


def test_enabled_telemetry_writes_one_machine_readable_record(tmp_path) -> None:
    output = tmp_path / "decisions.jsonl"
    logger = DecisionTelemetryLogger(
        output,
        enabled=True,
        session_metadata={"session_id": "test-session", "source": "synthetic_test"},
    )

    assert logger.log(
        _decision(),
        _metrics(),
        {"score_diff": 0},
        {"strategic_overrides_enabled": False},
    )
    logger.close()

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["record_type"] for record in records] == [
        "rival_session_start",
        "rival_policy_decision",
        "rival_session_end",
    ]
    record = records[1]
    assert record["schema_version"] == 3
    assert record["session_id"] == "test-session"
    assert record["decision"]["action_index"] == 1
    assert record["decision"]["baseline_action_index"] == 1
    assert record["decision"]["final_action_index"] == 1
    assert record["decision"]["legal_mask"] == [True, True]
    assert "raw_logits" not in record["decision"]
    assert record["runtime"]["strategic_overrides_enabled"] is False
    assert records[0]["metadata"]["source"] == "synthetic_test"
    assert records[2]["decision_record_count"] == 1


def test_schema_v3_serializes_baseline_final_and_challenge_explanation(tmp_path) -> None:
    output = tmp_path / "treatment.jsonl"
    baseline = _decision()
    continuation = ControllerAction(1.0, 0.0, 0.0, 0.0, 0.0, False, False, False)
    calibration = replace(
        ChallengeCalibrationDecision.exact_baseline(
            baseline,
            ChallengeCalibrationParameters(),
        ),
        mode=ChallengeCalibrationMode.INTERVENE,
        final_action_index=0,
        final_controller_action=continuation,
        hypothetical_action_index=0,
        hypothetical_controller_action=continuation,
        eligible=True,
        applied=True,
        reason="ambiguous_pressure_one_tick_deferral",
    )
    logger = DecisionTelemetryLogger(output, enabled=True)

    logger.log(baseline, _metrics(), {}, {}, calibration=calibration)
    logger.close()

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    record = records[1]
    assert record["decision"]["action_index"] == 0
    assert record["decision"]["baseline_action_index"] == 1
    assert record["decision"]["final_action_index"] == 0
    assert record["decision"]["hypothetical_action_index"] == 0
    assert record["decision"]["intervention_applied"] is True
    assert record["challenge_calibration"]["mode"] == "intervene"
    assert record["challenge_calibration"]["baseline_action"]["action_index"] == 1
    assert record["challenge_calibration"]["final_action"]["action_index"] == 0


def test_schema_v3_serializes_natural_adjustment_as_final_action_layer(tmp_path) -> None:
    output = tmp_path / "natural-treatment.jsonl"
    baseline = _decision()
    continuation = ControllerAction(1.0, 0.0, 0.0, 0.0, 0.0, False, False, False)
    natural = replace(
        NaturalAdjustmentDecision.exact_baseline(
            baseline,
            NaturalAdjustmentParameters(),
        ),
        mode=NaturalAdjustmentMode.INTERVENE,
        final_action_index=0,
        final_controller_action=continuation,
        hypothetical_action_index=0,
        hypothetical_controller_action=continuation,
        eligible=True,
        applied=True,
        reason="graded_resource_possession_rerank",
    )
    logger = DecisionTelemetryLogger(output, enabled=True)

    logger.log(
        baseline,
        _metrics(),
        {},
        {},
        natural_adjustment=natural,
    )
    logger.close()

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    record = records[1]
    assert record["decision"]["action_index"] == 0
    assert record["decision"]["baseline_action_index"] == 1
    assert record["decision"]["final_action_index"] == 0
    assert record["decision"]["intervention_applied"] is True
    assert record["natural_adjustment"]["mode"] == "intervene"
    assert record["natural_adjustment"]["baseline_action"]["action_index"] == 1
    assert record["natural_adjustment"]["final_action"]["action_index"] == 0


def test_verbose_telemetry_includes_raw_and_masked_logits(tmp_path) -> None:
    output = tmp_path / "verbose.jsonl"
    logger = DecisionTelemetryLogger(output, enabled=True, include_logits=True)
    logger.log(_decision(), _metrics(), {}, {})
    logger.close()

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    record = records[1]
    assert record["decision"]["raw_logits"] == [0.5, 1.5]
    assert record["decision"]["masked_logits"] == [0.5, 1.5]
