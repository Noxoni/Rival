from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import rlbot.managers


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.challenge_v3 import build_report  # noqa: E402
from tools.evidence.probes import FAKE_CHALLENGE_BEHAVIORS  # noqa: E402
from tools.evidence.runner import RAW_EVIDENCE_ROOT, run_fake_challenge_probes  # noqa: E402
from tools.evidence.session import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one coarse M03 treatment candidate against the recorded baseline"
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "evidence"
            / "results"
            / "v3"
            / "controlled_ab.json"
        ),
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--game-speed", type=float, default=1.0)
    parser.add_argument("--launcher", choices=("steam", "epic"), default="steam")
    parser.add_argument("--parameter-version", default="m03-candidate-low0-gap1p5")
    parser.add_argument("--low-threshold", type=float, default=0.0)
    parser.add_argument("--max-logit-gap", type=float, default=1.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "evidence"
            / "results"
            / "v3"
            / "controlled_candidate_low0_gap1p5.json"
        ),
    )
    args = parser.parse_args()
    baseline_report = json.loads(args.baseline_report.read_text(encoding="utf-8"))
    baseline_ids = [
        session["session_id"]
        for session in baseline_report.get("sessions", [])
        if session.get("mode") == "off" and session.get("source") == "controlled_probe"
    ]
    if len(baseline_ids) != len(FAKE_CHALLENGE_BEHAVIORS):
        parser.error(
            "baseline report must contain one off-mode session for every controlled behavior"
        )

    challenge_environment = {
        "RIVAL_CHALLENGE_PARAMETER_VERSION": args.parameter_version,
        "RIVAL_CHALLENGE_LOW_THRESHOLD": str(args.low_threshold),
        "RIVAL_CHALLENGE_MAX_LOGIT_GAP": str(args.max_logit_gap),
        "RIVAL_CHALLENGE_MAX_DEFERRAL_TICKS": "1",
    }
    started = time.monotonic()
    manager = rlbot.managers.MatchManager()
    try:
        manifests = run_fake_challenge_probes(
            repetitions=args.repetitions,
            rival_team=0,
            launcher=args.launcher,
            behaviors=FAKE_CHALLENGE_BEHAVIORS,
            game_speed=args.game_speed,
            challenge_mode="intervene",
            lane_id="controlled-candidate-sequential-lane-1",
            challenge_environment=challenge_environment,
            manager=manager,
        )
    finally:
        manager.shut_down()

    inputs = [RAW_EVIDENCE_ROOT / session_id for session_id in baseline_ids]
    inputs.extend(
        RAW_EVIDENCE_ROOT / str(manifest["session_id"])
        for manifest in manifests
        if manifest.get("session_id")
    )
    report = build_report(inputs)
    report["parameter_attempt"] = {
        "attempt_index": 2,
        "selection_basis": (
            "Initial treatment had zero interventions; three target releases were blocked "
            "only by low-state classification or a 1.349 logit gap with low confidence."
        ),
        "environment": challenge_environment,
        "maximum_deferral_policy_ticks": 1,
        "requested_game_speed": args.game_speed,
        "sequential_or_parallel": "sequential",
        "natural_match_budget_consumed": 0,
        "total_runner_wall_duration_seconds": time.monotonic() - started,
    }
    report["session_status"] = [
        {
            "session_id": manifest.get("session_id"),
            "status": manifest.get("status"),
            "termination_reason": manifest.get("termination_reason"),
            "wall_duration_seconds": manifest.get("wall_duration_seconds"),
            "raw_telemetry": manifest.get("raw_telemetry") or {},
            "runtime_warnings": manifest.get("runtime_warnings") or [],
        }
        for manifest in manifests
    ]
    write_json(args.output, report)
    summary = {
        "output": str(args.output),
        "complete_sessions": sum(
            manifest.get("status") == "complete" for manifest in manifests
        ),
        "paired_gate_inputs": report["paired_controlled"]["paired_gate_inputs"],
        "fake_pressure_aggregate": report["paired_controlled"][
            "fake_pressure_aggregate"
        ],
        "true_commit_aggregate": report["paired_controlled"][
            "true_commit_aggregate"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if all(manifest.get("status") == "complete" for manifest in manifests) else 1


if __name__ == "__main__":
    raise SystemExit(main())
