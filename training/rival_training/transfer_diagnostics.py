"""Milestone 07 policy-transfer gates shared by scripts and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import load_actor_checkpoint, portable_path
from .deploy import export_torchscript
from .teacher import EXPECTED_TEACHER_HASHES, FrozenWispReference, sha256_file
from .wisp_actions import action_table_fingerprint


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ZERO_STEP_CHECKPOINT = (
    REPOSITORY_ROOT / "training/artifacts/bootstrap/wisp_student_expanded_v1.pt"
)
EXPECTED_ZERO_STEP_SHA256 = (
    "0ae817b0dd15330c3a4e8ee469803c01ad4bde4fbe07d42a606d1ac513015a37"
)
EXPANDED_ACTION_TABLE = REPOSITORY_ROOT / "bot/models/RIVAL_ACTIONS_V1.npy"
EXPECTED_EXPANDED_ACTION_TABLE_FILE_SHA256 = (
    "bed450ded25a7bc624b17695913626fd478d00e11936413c7f5af5022585de2f"
)
EXPECTED_EXPANDED_ACTION_TABLE_LOGICAL_SHA256 = (
    "38ed338273ae09736d81d3e7fb69c45d91397e45d50f1ae97101e3737c0ecd20"
)


def _production_wisp_logits(observations: torch.Tensor) -> torch.Tensor:
    # The frozen artifacts are container modules without ``forward``. Production
    # ModelSet reconstructs their Linear/LayerNorm/ReLU sequence before inference;
    # FrozenWispReference is the same parameter-copy path used by the bootstrap.
    reference = FrozenWispReference().eval()
    with torch.inference_mode():
        return reference(observations)


def export_zero_step_diagnostic_actor(
    destination: str | Path,
    *,
    sample_count: int = 1024,
    seed: int = 20260902,
) -> dict[str, Any]:
    """Export the untouched reconstructed actor and gate the live deployment graph."""
    source_hash = sha256_file(ZERO_STEP_CHECKPOINT)
    if source_hash != EXPECTED_ZERO_STEP_SHA256:
        raise RuntimeError(
            f"Zero-step checkpoint hash mismatch: expected {EXPECTED_ZERO_STEP_SHA256}, "
            f"got {source_hash}"
        )
    for relative, expected in EXPECTED_TEACHER_HASHES.items():
        actual = sha256_file(REPOSITORY_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen teacher hash mismatch for {relative}: {actual}")
    table_file_hash = sha256_file(EXPANDED_ACTION_TABLE)
    table = np.load(EXPANDED_ACTION_TABLE, allow_pickle=False)
    table_logical_hash = action_table_fingerprint(table)
    if (
        table_file_hash != EXPECTED_EXPANDED_ACTION_TABLE_FILE_SHA256
        or table_logical_hash != EXPECTED_EXPANDED_ACTION_TABLE_LOGICAL_SHA256
    ):
        raise RuntimeError("Expanded action table changed before M07 diagnostics")

    actor, actor_metadata = load_actor_checkpoint(ZERO_STEP_CHECKPOINT, "cpu")
    export = export_torchscript(actor, destination)
    scripted = torch.jit.load(str(Path(destination)), map_location="cpu").eval()
    observations = torch.randn(
        sample_count,
        432,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    expected = _production_wisp_logits(observations)
    with torch.inference_mode():
        actual_all = scripted(observations)
    actual = actual_all[:, :90]
    difference = (expected - actual).abs()
    expected_top1 = expected.argmax(dim=-1)
    actual_top1 = actual.argmax(dim=-1)
    appended_weights = actor.policy[-1].weight[90:].detach()
    appended_biases = actor.policy[-1].bias[90:].detach()
    parity = {
        "sample_count": sample_count,
        "seed": seed,
        "first_90_exact": bool(torch.equal(expected, actual)),
        "first_90_allclose_atol_1e-6_rtol_1e-6": bool(
            torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
        ),
        "max_abs_first_90_logit_error": float(difference.max().item()),
        "mean_abs_first_90_logit_error": float(difference.mean().item()),
        "first_90_top1_agreement": float(
            (expected_top1 == actual_top1).float().mean().item()
        ),
        "finite": bool(torch.isfinite(actual_all).all().item()),
        "appended_weights_exact_zero": bool(
            torch.equal(appended_weights, torch.zeros_like(appended_weights))
        ),
        "appended_bias_exact_negative_12": bool(
            torch.equal(appended_biases, torch.full_like(appended_biases, -12.0))
        ),
    }
    parity["passed"] = all(
        (
            parity["first_90_allclose_atol_1e-6_rtol_1e-6"],
            parity["first_90_top1_agreement"] == 1.0,
            parity["finite"],
            parity["appended_weights_exact_zero"],
            parity["appended_bias_exact_negative_12"],
        )
    )
    if not parity["passed"]:
        raise RuntimeError(f"Zero-step diagnostic export parity failed: {parity}")

    return {
        "schema_version": 1,
        "status": "passed",
        "purpose": "milestone07_zero_optimizer_step_deployment_control",
        "source_checkpoint": {
            "path": portable_path(ZERO_STEP_CHECKPOINT),
            "sha256": source_hash,
            "metadata": actor_metadata,
            "optimizer_steps": 0,
            "ppo_updates": 0,
        },
        "torchscript_export": export,
        "frozen_teacher_hashes": EXPECTED_TEACHER_HASHES,
        "action_table": {
            "path": portable_path(EXPANDED_ACTION_TABLE),
            "shape": list(table.shape),
            "file_sha256": table_file_hash,
            "logical_float32_sha256": table_logical_hash,
        },
        "random_tensor_deployment_parity": parity,
        "runtime_requirement": {
            "transfer_diagnostic_mode": True,
            "legacy_only": True,
            "allowed_tick_skip": [4, 8],
        },
        "production_promoted": False,
    }
