"""Offline-only M10 export checks when live RLBot/Rocket League use is prohibited."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import (  # noqa: E402
    boundary_slug,
    checkpoint_record,
    portable_path,
    sha256_file,
    verify_checkpoint_reload_parity,
    write_json_atomic,
)
from rival_training.v9_checkpoint import load_v9_checkpoint  # noqa: E402
from rival_training.v9_deployment import export_v9_deployment  # noqa: E402


DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "training/artifacts/milestone10"
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "training/results/raw/milestone10"
EXPECTED_WISP_HASHES = {
    "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}


def _latency(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise FloatingPointError("Offline export latency samples are invalid")
    return {
        "samples": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def run_offline_transfer(args: argparse.Namespace) -> dict[str, Any]:
    boundary = int(args.boundary_added_hours)
    if boundary not in (25, 100):
        raise ValueError("M10 offline transfer checks are scoped to +25h and +100h")
    checkpoint = args.checkpoint.resolve()
    slug = boundary_slug(boundary)
    artifact_directory = (args.artifact_root / slug).resolve()
    loaded = load_v9_checkpoint(checkpoint, device="cpu")
    identity = checkpoint_record(checkpoint, manifest=loaded["manifest"])
    reload_parity = verify_checkpoint_reload_parity(
        checkpoint, expected_config=loaded["config"], device="cpu"
    )
    export = export_v9_deployment(checkpoint, artifact_directory)
    metadata = export["metadata"]
    model_path = REPOSITORY_ROOT / export["selected_path"]
    metadata_path = REPOSITORY_ROOT / export["metadata_path"]
    reference_path = REPOSITORY_ROOT / export["reference_path"]
    held = np.load(reference_path, allow_pickle=False)
    observations = np.asarray(held["observations"], dtype=np.float32)
    expected = tuple(
        np.asarray(held[name], dtype=np.float32)
        for name in ("analog_mean", "analog_log_std", "button_logits", "controller")
    )
    model = torch.jit.load(str(model_path), map_location="cpu").eval()
    with torch.inference_mode():
        actual_tensors = model(torch.from_numpy(observations))
    actual = tuple(tensor.detach().cpu().numpy() for tensor in actual_tensors)
    errors = {
        name: float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        for name, left, right in zip(
            ("analog_mean", "analog_log_std", "button_logits", "controller"),
            expected,
            actual,
            strict=True,
        )
    }
    controller = actual[-1]
    example = torch.from_numpy(observations[:1])
    with torch.inference_mode():
        for _ in range(int(args.warmup_samples)):
            model(example)
        latency_ms = []
        for _ in range(int(args.latency_samples)):
            started = time.perf_counter_ns()
            output = model(example)
            latency_ms.append((time.perf_counter_ns() - started) / 1e6)
            if not all(bool(torch.isfinite(value).all()) for value in output):
                raise FloatingPointError("Offline exported model emitted a non-finite tensor")
    contract = metadata["contract"]
    checkpoint_contract = loaded["manifest"]["contract"]
    wisp_hashes = {
        name: sha256_file(REPOSITORY_ROOT / "bot/models" / name)
        for name in EXPECTED_WISP_HASHES
    }
    latency = _latency(latency_ms)
    checks = {
        "checkpoint_fresh_reload_exact": reload_parity["checks"]["passed"],
        "export_source_checkpoint_exact": metadata["source_checkpoint"][
            "checkpoint_manifest_sha256"
        ]
        == identity["manifest_sha256"],
        "export_source_actor_exact": metadata["source_checkpoint"]["actor_sha256"]
        == identity["actor_sha256"],
        "export_parity_within_1e_5": metadata["export_parity"]["passed"]
        and max(errors.values()) <= 1e-5,
        "policy_schema_identity_exact": contract["policy_version"]
        == checkpoint_contract["policy_version"],
        "observation_schema_identity_exact": contract["observation_version"]
        == checkpoint_contract["observation_version"]
        and contract["observation_schema_sha256"]
        == checkpoint_contract["observation_schema_sha256"],
        "action_schema_identity_exact": contract["action_version"]
        == checkpoint_contract["action_version"]
        and contract["action_schema_sha256"]
        == checkpoint_contract["action_schema_sha256"],
        "native_cadence_contract_metadata_exact": contract["physics_hz"] == 120
        and contract["policy_hz"] == 120
        and contract["repeat_action"] is False,
        "export_outputs_finite": all(np.isfinite(value).all() for value in actual),
        "export_controllers_legal": bool(
            np.all(controller[:, :5] >= -1.0)
            and np.all(controller[:, :5] <= 1.0)
            and np.all(np.isin(np.rint(controller[:, 5:]), (0.0, 1.0)))
        ),
        "offline_actor_p99_within_120hz_frame_budget": latency["p99"]
        < 1000.0 / 120.0,
        "frozen_wisp_hashes_unchanged": wisp_hashes == EXPECTED_WISP_HASHES,
        "no_rlbot_or_rocket_league_launcher_called": True,
        "no_rlbot_or_rocket_league_process_started_stopped_or_inspected": True,
        "production_promotion_authorized": False,
    }
    passed = all(
        value for key, value in checks.items() if key != "production_promotion_authorized"
    ) and checks["production_promotion_authorized"] is False
    checks["passed"] = passed
    result = {
        "schema_version": 1,
        "status": "offline_only_user_prohibited",
        "milestone": "10",
        "boundary_added_simulated_hours": boundary,
        "checkpoint": identity,
        "checkpoint_fresh_reload": reload_parity,
        "export": {
            **metadata,
            "metadata_path": portable_path(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "held_replay_maximum_absolute_errors": errors,
            "offline_single_observation_latency_milliseconds": latency,
        },
        "production_model_hashes": wisp_hashes,
        "live_native_rlbot": {
            "status": "not_run_user_prohibited",
            "technical_native_120hz_cadence_verified": False,
            "native_packet_continuity_verified": False,
            "native_full_pipeline_latency_verified": False,
            "reason": (
                "The operator explicitly prohibited launching, stopping, or inspecting "
                "RLBot or Rocket League while using the game online."
            ),
        },
        "scope": {
            "checkpoint_reload_export_schema_controller_and_cpu_actor_checks_only": True,
            "rlbot_imported_or_launched": False,
            "rocket_league_process_touched": False,
            "wins_and_losses_observed": False,
            "production_promoted": False,
        },
        "checks": checks,
    }
    write_json_atomic(args.output, result)
    if not passed:
        raise RuntimeError(f"M10 offline transfer boundary failed: {checks}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-added-hours", type=int, choices=(25, 100), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup-samples", type=int, default=200)
    parser.add_argument("--latency-samples", type=int, default=3000)
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            args.raw_root
            / boundary_slug(args.boundary_added_hours)
            / "offline_transfer_boundary.json"
        )
    report = run_offline_transfer(args)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
