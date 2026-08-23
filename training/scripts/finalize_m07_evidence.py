from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "training/results/milestone07"
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.checkpoint import portable_path  # noqa: E402


REPORT_EXPECTATIONS = {
    "action_parity.json": {"passed"},
    "live_policy_parity.json": {"passed"},
    "observation_domain.json": {"completed"},
    "rlviser_spectator_smoke.json": {None},
    "transfer_matrix.json": {"completed", "completed_with_runtime_anomaly"},
    "transfer_telemetry.json": {"completed", "completed_with_runtime_anomaly"},
    "transition_audit.json": {"completed"},
    "zero_step_export.json": {"passed"},
    "zero_step_runtime_production.json": {None},
    "zero_step_runtime_training.json": {None},
}

LOCAL_ARTIFACTS = (
    "bot/models/POLICY.lt",
    "bot/models/SHARED_HEAD.lt",
    "bot/models/RIVAL_ACTIONS_V1.npy",
    "training/artifacts/bootstrap/wisp_student_expanded_v1.pt",
    "training/artifacts/milestone06/stage_b_020m/candidate_actor.ts",
    "training/artifacts/milestone07/zero_step_actor.ts",
    "training/checkpoints/milestone06/020000016_stage_b_m4p0/PPO_POLICY.pt",
    "training/tools/rlviser/rlviser.exe",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _committed_text_sha256(path: Path) -> str:
    """Hash the LF-normalized bytes Git stores for repository text reports."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_tracks(path: str) -> bool:
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", path],
            cwd=REPOSITORY_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )


def _load_and_validate_reports() -> list[dict[str, Any]]:
    entries = []
    for name, accepted_statuses in sorted(REPORT_EXPECTATIONS.items()):
        path = RESULTS_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing required M07 report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        status = report.get("status")
        if status not in accepted_statuses:
            raise RuntimeError(
                f"Unexpected status for {name}: {status!r}; "
                f"accepted={sorted(str(item) for item in accepted_statuses)}"
            )
        if report.get("production_promoted") is True:
            raise RuntimeError(f"Production promotion recorded unexpectedly in {name}")
        if report.get("production_modified_or_promoted") is True:
            raise RuntimeError(f"Production modification recorded unexpectedly in {name}")
        if name.startswith("zero_step_runtime_"):
            if report.get("finite") is not True or report.get("output_shape") != [64, 158]:
                raise RuntimeError(f"Invalid zero-step runtime evidence in {name}")
        if name == "rlviser_spectator_smoke.json" and report.get(
            "renderer_process_verified"
        ) is not True:
            raise RuntimeError("RLViser spectator smoke did not verify its viewer process")
        entries.append(
            {
                "path": portable_path(path),
                "sha256": _committed_text_sha256(path),
                "size_bytes": path.stat().st_size,
                "status": status,
            }
        )
    return entries


def _artifact_entries() -> list[dict[str, Any]]:
    entries = []
    for relative in LOCAL_ARTIFACTS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required M07 source artifact: {path}")
        entries.append(
            {
                "path": portable_path(path),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "tracked": _git_tracks(relative),
            }
        )
    return entries


def build_manifest() -> dict[str, Any]:
    matrix = json.loads(
        (RESULTS_ROOT / "transfer_matrix.json").read_text(encoding="utf-8")
    )
    live_parity = json.loads(
        (RESULTS_ROOT / "live_policy_parity.json").read_text(encoding="utf-8")
    )
    telemetry = json.loads(
        (RESULTS_ROOT / "transfer_telemetry.json").read_text(encoding="utf-8")
    )
    return {
        "schema_version": 1,
        "status": "passed",
        "purpose": "milestone07_compact_evidence_manifest",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repository": {
            "head_before_evidence_commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "origin": _git("remote", "get-url", "origin"),
            "m06_boundary_is_ancestor": subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    "652395a9f512ce835830bfc5bc3a7cb078f6105e",
                    "HEAD",
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
            ).returncode
            == 0,
        },
        "scope": {
            "serious_ppo_training_resumed": False,
            "stage_c_started": False,
            "optimizer_or_ppo_steps_created": 0,
            "production_modified_or_promoted": False,
            "rlbot_match_modes": sorted(matrix["completed_modes"]),
            "rlbot_completed_matches": sum(
                mode["aggregates"]["overall"]["completed_match_results"]
                for mode in matrix["modes"].values()
            ),
            "runtime_clean_sessions": telemetry["hard_invariants"][
                "runtime_clean_sessions"
            ],
            "live_observation_samples": live_parity["corpus"]["samples"],
        },
        "reports": _load_and_validate_reports(),
        "source_artifacts": _artifact_entries(),
        "validation": {
            "all_required_reports_parsed": True,
            "all_required_source_artifacts_hashed": True,
            "production_promotion_flags_rejected": True,
            "note": "Test commands and remote readback are reported outside this self-hashed manifest.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "evidence_manifest.json",
    )
    args = parser.parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "reports": len(manifest["reports"]),
                "source_artifacts": len(manifest["source_artifacts"]),
                **manifest["scope"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
