"""Export and verify one M10 boundary through the native RLBot v5 path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


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


PRODUCTION_PYTHON = REPOSITORY_ROOT / ".venv/Scripts/python.exe"
LIVE_SCRIPT = REPOSITORY_ROOT / "training/scripts/run_m09_scratch_live_smoke.py"
DEFAULT_ARTIFACT_ROOT = REPOSITORY_ROOT / "training/artifacts/milestone10"
DEFAULT_RAW_ROOT = REPOSITORY_ROOT / "training/results/raw/milestone10"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_native(args: argparse.Namespace) -> dict[str, Any]:
    boundary = int(args.boundary_added_hours)
    if boundary not in (25, 100):
        raise ValueError("M10 native checks are authorized at +25h and +100h")
    slug = boundary_slug(boundary)
    checkpoint = args.checkpoint.resolve()
    artifact_directory = (args.artifact_root / slug).resolve()
    raw_directory = (args.raw_root / slug).resolve()
    raw_directory.mkdir(parents=True, exist_ok=True)
    live_output = raw_directory / "native_rlbot_live_smoke.json"

    loaded = load_v9_checkpoint(checkpoint, device="cpu")
    checkpoint_identity = checkpoint_record(checkpoint, manifest=loaded["manifest"])
    reload_parity = verify_checkpoint_reload_parity(
        checkpoint, expected_config=loaded["config"], device="cpu"
    )
    export = export_v9_deployment(checkpoint, artifact_directory)
    model_path = REPOSITORY_ROOT / export["selected_path"]
    metadata_path = REPOSITORY_ROOT / export["metadata_path"]
    command = [
        str(PRODUCTION_PYTHON),
        str(LIVE_SCRIPT),
        "--model",
        str(model_path),
        "--metadata",
        str(metadata_path),
        "--opponent",
        args.opponent,
        "--rival-team",
        str(args.rival_team),
        "--launcher",
        args.launcher,
        "--maximum-records",
        str(args.maximum_records),
        "--smoke-game-seconds",
        str(args.smoke_game_seconds),
        "--warmup-records",
        str(args.warmup_records),
        "--output",
        str(live_output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(360.0, args.smoke_game_seconds * 4.0),
    )
    if not live_output.is_file():
        raise RuntimeError(
            "Native RLBot v5 smoke produced no report; "
            f"exit={completed.returncode}, stdout={completed.stdout[-4000:]}, "
            f"stderr={completed.stderr[-4000:]}"
        )
    live = _json(live_output)
    metadata = export["metadata"]
    contract = metadata["contract"]
    checkpoint_contract = loaded["manifest"]["contract"]
    checks = {
        "checkpoint_fresh_reload_exact": reload_parity["checks"]["passed"],
        "export_source_checkpoint_manifest_exact": metadata["source_checkpoint"][
            "checkpoint_manifest_sha256"
        ]
        == checkpoint_identity["manifest_sha256"],
        "export_source_actor_exact": metadata["source_checkpoint"]["actor_sha256"]
        == checkpoint_identity["actor_sha256"],
        "export_parity_within_1e_5": metadata["export_parity"]["passed"],
        "export_model_used_live": live["model"]["sha256"] == sha256_file(model_path),
        "export_metadata_used_live": live["metadata"]["sha256"]
        == sha256_file(metadata_path),
        "policy_identity_exact": contract["policy_version"]
        == checkpoint_contract["policy_version"],
        "observation_identity_exact": contract["observation_version"]
        == checkpoint_contract["observation_version"]
        and contract["observation_schema_sha256"]
        == checkpoint_contract["observation_schema_sha256"],
        "action_identity_exact": contract["action_version"]
        == checkpoint_contract["action_version"]
        and contract["action_schema_sha256"]
        == checkpoint_contract["action_schema_sha256"],
        "native_120hz_contract_exact": contract["physics_hz"] == 120
        and contract["policy_hz"] == 120
        and contract["repeat_action"] is False,
        "native_live_smoke_process_passed": completed.returncode == 0
        and live["status"] == "passed",
        "native_outputs_finite": live["checks"]["runtime_outputs_finite"],
        "native_controllers_legal": live["checks"]["runtime_controllers_legal"],
        "no_sustained_missed_ticks": live["checks"][
            "no_sustained_post_warmup_frame_gaps"
        ]
        and live["checks"]["post_warmup_gap_fraction_below_half_percent"],
        "native_callback_within_120hz_budget": live["checks"][
            "callback_p99_within_native_frame_budget"
        ],
        "production_not_promoted": live["checks"]["production_not_promoted"],
        "wins_and_losses_excluded_from_technical_pass_fail": True,
    }
    passed = all(checks.values())
    checks["passed"] = passed
    result = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "milestone": "10",
        "boundary_added_simulated_hours": boundary,
        "checkpoint": checkpoint_identity,
        "checkpoint_fresh_reload": reload_parity,
        "export": {
            **metadata,
            "metadata_path": export["metadata_path"],
            "metadata_sha256": export["metadata_sha256"],
        },
        "native_rlbot_v5": {
            "live_report_path": portable_path(live_output),
            "live_report_sha256": sha256_file(live_output),
            "live_report_size_bytes": live_output.stat().st_size,
            "process_exit_code": completed.returncode,
            "opponent_context_only": args.opponent,
            "rival_team": args.rival_team,
            "native_corpus": live["native_corpus"],
            "scratch_runtime": live["scratch_runtime"],
            "natural_match_manifest": live["natural_match_manifest"],
            "technical_checks": live["checks"],
        },
        "production_promotion_authorized": False,
        "checks": checks,
    }
    write_json_atomic(args.output, result)
    if not passed:
        raise RuntimeError(f"M10 native RLBot boundary failed: {checks}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary-added-hours", type=int, choices=(25, 100), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--opponent", choices=("wisp", "nexto"), default="wisp")
    parser.add_argument("--rival-team", type=int, choices=(0, 1), default=0)
    parser.add_argument("--launcher", choices=("steam", "epic", "no-launch"), default="steam")
    parser.add_argument("--maximum-records", type=int, default=2400)
    parser.add_argument("--smoke-game-seconds", type=float, default=28.0)
    parser.add_argument("--warmup-records", type=int, default=120)
    args = parser.parse_args()
    if args.output is None:
        args.output = (
            args.raw_root
            / boundary_slug(args.boundary_added_hours)
            / "native_boundary.json"
        )
    report = run_native(args)
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
