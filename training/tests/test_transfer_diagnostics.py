from __future__ import annotations

from pathlib import Path

from rival_training.transfer_diagnostics import export_zero_step_diagnostic_actor


def test_zero_step_export_preserves_frozen_first_90_deployment_logits(tmp_path: Path) -> None:
    report = export_zero_step_diagnostic_actor(
        tmp_path / "zero_step_actor.ts",
        sample_count=16,
        seed=7,
    )

    parity = report["random_tensor_deployment_parity"]
    assert report["status"] == "passed"
    assert report["source_checkpoint"]["optimizer_steps"] == 0
    assert parity["first_90_top1_agreement"] == 1.0
    assert parity["max_abs_first_90_logit_error"] <= 1e-6
    assert parity["appended_weights_exact_zero"] is True
    assert parity["appended_bias_exact_negative_12"] is True
