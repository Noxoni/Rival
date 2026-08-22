from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = REPOSITORY_ROOT / "bot"
sys.path.insert(0, str(BOT_ROOT))

from backend.model import ModelSet  # noqa: E402
import config  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    torch.manual_seed(20260822)
    models = ModelSet(
        config.MODEL_INFO_POLICY,
        config.MODEL_INFO_SHARED_HEAD,
        device="cpu",
    )
    observation = torch.linspace(-1.0, 1.0, 432, dtype=torch.float32)
    legal_mask = torch.ones(90, dtype=torch.bool)
    inference = models.infer_policy(observation, legal_mask)
    selected = inference.select_action(deterministic=True)
    compatibility_selected = models.get_action(
        observation, legal_mask, deterministic=True
    )

    result = {
        "status": "pass" if selected == compatibility_selected else "fail",
        "observation_shape": list(observation.shape),
        "policy_output_shape": list(inference.raw_logits.shape),
        "selected_action_index": selected,
        "compatibility_action_index": compatibility_selected,
        "finite_logits": bool(torch.isfinite(inference.raw_logits).all().item()),
        "models": {
            "POLICY.lt": _sha256(config.MODEL_INFO_POLICY.path),
            "SHARED_HEAD.lt": _sha256(config.MODEL_INFO_SHARED_HEAD.path),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" and result["finite_logits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
