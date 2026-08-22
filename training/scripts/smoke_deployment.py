"""Verify checkpoint inference, mirror semantics, and TorchScript export/reload."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.checkpoint import load_actor_checkpoint  # noqa: E402
from rival_training.deploy import RivalInferenceSession, export_torchscript  # noqa: E402


def main() -> None:
    actor_path = REPOSITORY_ROOT / "training/artifacts/ppo/milestone05_smoke_actor.pt"
    actor, metadata = load_actor_checkpoint(actor_path, "cpu")
    session = RivalInferenceSession(actor, "cpu")
    observation = np.random.default_rng(20260822).standard_normal(432).astype(np.float32)
    normal = session.infer(observation, mirror_x=False)
    mirrored = session.infer(observation, mirror_x=True)
    expected_mirror = normal["controller_action"].copy()
    expected_mirror[[1, 3, 4]] *= -1
    export_path = REPOSITORY_ROOT / "training/artifacts/deploy/milestone05_actor.ts"
    export = export_torchscript(actor, export_path)
    scripted = torch.jit.load(str(export_path), map_location="cpu").eval()
    with torch.no_grad():
        eager_logits = actor(torch.from_numpy(observation).unsqueeze(0))
        scripted_logits = scripted(torch.from_numpy(observation).unsqueeze(0))
    report = {
        "schema_version": 1,
        "status": "passed",
        "checkpoint_metadata": metadata,
        "input_shape": list(observation.shape),
        "action_index": normal["action_index"],
        "controller_action": normal["controller_action"].tolist(),
        "logits_shape": list(normal["logits"].shape),
        "logits_finite": bool(np.isfinite(normal["logits"]).all()),
        "mirror_controller_exact": bool(
            np.array_equal(expected_mirror, mirrored["controller_action"])
        ),
        "torchscript_export": export,
        "torchscript_max_abs_error": float(
            (eager_logits - scripted_logits).abs().max().item()
        ),
        "torchscript_allclose": bool(
            torch.allclose(eager_logits, scripted_logits, atol=1e-6, rtol=1e-6)
        ),
        "production_runtime_replaced": False,
    }
    if not all(
        [report["logits_finite"], report["mirror_controller_exact"], report["torchscript_allclose"]]
    ):
        raise SystemExit(f"Deployment smoke failed: {report}")
    output = REPOSITORY_ROOT / "training/results/deployment_smoke.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
