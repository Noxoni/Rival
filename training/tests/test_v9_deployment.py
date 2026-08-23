from __future__ import annotations

import numpy as np
import torch

from rival_training.v9_actions import RivalHybridDistribution
from rival_training.v9_checkpoint import action_schema_sha256
from rival_training.v9_deployment import (
    ACTION_VERSION,
    CONTROLLER_FIELDS,
    RivalV9DeterministicController,
)
from rival_training.v9_observations import OBSERVATION_SIZE
from rival_training.v9_policy import RivalPolicyV1


def test_deterministic_export_wrapper_matches_hybrid_distribution_mode() -> None:
    actor = RivalPolicyV1().eval()
    wrapper = RivalV9DeterministicController(actor).eval()
    observations = torch.linspace(
        -1.0,
        1.0,
        3 * OBSERVATION_SIZE,
        dtype=torch.float32,
    ).reshape(3, OBSERVATION_SIZE)
    with torch.inference_mode():
        mean, log_std, logits, controller = wrapper(observations)
        expected = RivalHybridDistribution(mean, log_std, logits).mode()
    torch.testing.assert_close(controller, expected, atol=0.0, rtol=0.0)
    assert controller.shape == (3, 8)


def test_deterministic_export_wrapper_emits_only_legal_controllers() -> None:
    generator = torch.Generator().manual_seed(20260912)
    observations = torch.randn(
        (64, OBSERVATION_SIZE), generator=generator, dtype=torch.float32
    )
    wrapper = RivalV9DeterministicController(RivalPolicyV1().eval()).eval()
    with torch.inference_mode():
        mean, log_std, logits, controllers = wrapper(observations)
    assert all(torch.isfinite(value).all() for value in (mean, log_std, logits, controllers))
    assert bool(torch.all(controllers[:, :5] >= -1.0))
    assert bool(torch.all(controllers[:, :5] <= 1.0))
    assert set(np.unique(controllers[:, 5:].numpy())).issubset({0.0, 1.0})


def test_deployment_contract_keeps_native_controller_order() -> None:
    assert ACTION_VERSION == "RivalActionV1"
    assert action_schema_sha256() == (
        "0121360ac73546911cc04dd6971ab5c53d1629c82589c00c45cb6b298a8f4163"
    )
    assert CONTROLLER_FIELDS == (
        "throttle",
        "steer",
        "pitch",
        "yaw",
        "roll",
        "jump",
        "boost",
        "handbrake",
    )
