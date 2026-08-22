from __future__ import annotations

import numpy as np
import torch

from rival_training.checkpoint import load_actor_checkpoint, save_actor_checkpoint
from rival_training.deploy import RivalInferenceSession, export_torchscript
from rival_training.teacher import build_wisp_student


def test_actor_checkpoint_reload_and_inference_export(tmp_path) -> None:
    actor = build_wisp_student()
    checkpoint = tmp_path / "actor.pt"
    manifest = save_actor_checkpoint(checkpoint, actor, {"test": True})
    loaded, metadata = load_actor_checkpoint(checkpoint)
    assert manifest["sha256"]
    assert metadata == {"test": True}

    observation = np.random.default_rng(1).standard_normal(432).astype(np.float32)
    before = RivalInferenceSession(actor).infer(observation)
    after = RivalInferenceSession(loaded).infer(observation)
    assert before["action_index"] == after["action_index"]
    assert np.array_equal(before["logits"], after["logits"])

    export_path = tmp_path / "actor.ts"
    export_torchscript(loaded, export_path)
    scripted = torch.jit.load(str(export_path)).eval()
    with torch.no_grad():
        scripted_logits = scripted(torch.from_numpy(observation).unsqueeze(0))
    assert torch.allclose(
        torch.from_numpy(after["logits"]), scripted_logits.squeeze(0), atol=1e-6, rtol=1e-6
    )
