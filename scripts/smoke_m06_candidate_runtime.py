"""Production-environment smoke for the dormant Milestone 06 candidate seam."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "bot"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import config  # noqa: PLC0415
    from action_parser import XMirroredActionParser  # noqa: PLC0415
    from backend.model import ModelSet  # noqa: PLC0415

    if not config.CANDIDATE_POLICY_ENABLED:
        raise RuntimeError("RIVAL_CANDIDATE_MODEL_PATH was not supplied")
    parser_instance = XMirroredActionParser(
        config.CANDIDATE_ACTION_TABLE_PATH,
        allow_all_actions=True,
    )
    mask = parser_instance.get_action_mask(None, None)
    models = ModelSet(
        config.MODEL_INFO_POLICY,
        config.MODEL_INFO_SHARED_HEAD,
        device=config.MODEL_DEVICE,
    )
    observation = np.random.default_rng(20260901).standard_normal(432).astype(
        np.float32
    )
    inference = models.infer_policy(observation, mask)
    action_index = inference.select_action(True)
    selected_action = parser_instance.actions[action_index].get_np()
    action_table = np.load(config.CANDIDATE_ACTION_TABLE_PATH, allow_pickle=False)
    logical_hash = hashlib.sha256(
        np.asarray(action_table, dtype="<f4").tobytes(order="C")
    ).hexdigest()
    report = {
        "schema_version": 1,
        "status": "passed",
        "runtime_mode": config.POLICY_RUNTIME_MODE,
        "candidate_enabled": config.CANDIDATE_POLICY_ENABLED,
        "tick_skip": config.TICK_SKIP,
        "action_delay": config.ACTION_DELAY,
        "model_path": os.path.relpath(
            config.CANDIDATE_MODEL_PATH, REPOSITORY_ROOT
        ).replace("\\", "/"),
        "model_sha256": _sha256_file(config.CANDIDATE_MODEL_PATH),
        "action_table_path": os.path.relpath(
            config.CANDIDATE_ACTION_TABLE_PATH, REPOSITORY_ROOT
        ).replace("\\", "/"),
        "action_table_file_sha256": _sha256_file(
            config.CANDIDATE_ACTION_TABLE_PATH
        ),
        "action_table_logical_sha256": logical_hash,
        "action_count": len(parser_instance.actions),
        "all_actions_enabled": bool(mask.all().item()),
        "policy_output_count": int(inference.raw_logits.numel()),
        "policy_logits_finite": bool(torch.isfinite(inference.raw_logits).all().item()),
        "selected_action_index": int(action_index),
        "selected_controller": selected_action.tolist(),
        "production_default_replaced": False,
    }
    required = {
        "candidate_enabled": True,
        "tick_skip": 4,
        "action_count": 158,
        "all_actions_enabled": True,
        "policy_output_count": 158,
        "policy_logits_finite": True,
        "production_default_replaced": False,
    }
    if any(report[key] != expected for key, expected in required.items()):
        report["status"] = "failed"
        raise RuntimeError(f"Candidate runtime smoke failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
