from __future__ import annotations

import numpy as np
import torch

from rival_training.live_policy_parity import _divergence, _masked_policy_metrics


def test_masked_policy_metrics_and_divergence_are_stable() -> None:
    first_logits = torch.tensor([[1.0, 9.0, 2.0], [3.0, 2.0, 1.0]])
    second_logits = torch.tensor([[1.0, -4.0, 2.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True], [True, True, True]])
    first = _masked_policy_metrics(first_logits, mask)
    second = _masked_policy_metrics(second_logits, mask)
    kl_first_second, kl_second_first, js = _divergence(
        first["probabilities"], second["probabilities"]
    )

    assert first["top1"].tolist() == [2, 0]
    assert second["top1"].tolist() == [2, 2]
    assert torch.isfinite(kl_first_second).all()
    assert torch.isfinite(kl_second_first).all()
    assert torch.isfinite(js).all()
    assert np.all(js.numpy() >= 0)
    assert js[0].item() < 1e-8
