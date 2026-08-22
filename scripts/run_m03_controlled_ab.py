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
from tools.evidence.runner import (  # noqa: E402
    FAKE_CHALLENGE_BEHAVIORS,
    RAW_EVIDENCE_ROOT,
    run_fake_challenge_probes,
)
from tools.evidence.session import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the original paired M03 controlled baseline/treatment suite"
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--game-speed", type=float, default=1.0)
    parser.add_argument("--launcher", choices=("steam", "epic"), default="steam")
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "evidence"
            / "results"
            / "v3"
            / "controlled_ab.json"
        ),
    )
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("repetitions must be at least one")
    if args.game_speed <= 0.0:
        parser.error("game-speed must be positive")

    started = time.monotonic()
    manager = rlbot.managers.MatchManager()
    manifests = []
    try:
        for mode in ("off", "intervene"):
            manifests.extend(
                run_fake_challenge_probes(
                    repetitions=args.repetitions,
                    rival_team=0,
                    launcher=args.launcher,
                    behaviors=FAKE_CHALLENGE_BEHAVIORS,
                    game_speed=args.game_speed,
                    challenge_mode=mode,
                    lane_id="controlled-sequential-lane-1",
                    manager=manager,
                )
            )
    finally:
        manager.shut_down()

    inputs = [
        RAW_EVIDENCE_ROOT / str(manifest["session_id"])
        for manifest in manifests
        if manifest.get("session_id")
    ]
    report = build_report(inputs)
    report["execution_control"] = {
        "requested_game_speed": args.game_speed,
        "evidence_speed_source": "bounded_speed_integrity_and_fallback_gate",
        "sequential_or_parallel": "sequential",
        "lane_id": "controlled-sequential-lane-1",
        "original_repetitions_per_behavior_per_mode": args.repetitions,
        "behaviors": list(FAKE_CHALLENGE_BEHAVIORS),
        "natural_match_budget_consumed": 0,
        "total_runner_wall_duration_seconds": time.monotonic() - started,
    }
    report["parameter_attempt"] = {
        "attempt_index": 1,
        "parameter_version": "m03-conservative-v1",
        "environment": {
            "RIVAL_CHALLENGE_LOW_THRESHOLD": "0.34",
            "RIVAL_CHALLENGE_HIGH_THRESHOLD": "0.70",
            "RIVAL_CHALLENGE_PRESSURE_DISTANCE": "1900.0",
            "RIVAL_CHALLENGE_PRESSURE_ETA": "1.40",
            "RIVAL_CHALLENGE_PROJECTED_MISS_REFERENCE": "450.0",
            "RIVAL_CHALLENGE_CONTROL_DISTANCE": "650.0",
            "RIVAL_CHALLENGE_MAX_LOGIT_GAP": "0.85",
            "RIVAL_CHALLENGE_MAX_DEFERRAL_TICKS": "1",
        },
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
        "sessions": len(manifests),
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
