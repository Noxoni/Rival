from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import rlbot.managers


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.runner import run_natural_match  # noqa: E402
from tools.evidence.session import utc_now, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bounded packet-observability fallback search after a failed 5x gate"
    )
    parser.add_argument("--launcher", choices=("steam", "epic"), default="steam")
    parser.add_argument("--game-seconds", type=float, default=12.0)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--speeds", nargs="+", type=float, default=[4.0, 3.0, 2.0])
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "evidence"
            / "results"
            / "v3"
            / "speed_fallback_search.json"
        ),
    )
    args = parser.parse_args()

    started = time.monotonic()
    manager = rlbot.managers.MatchManager()
    manifests = []
    try:
        for speed in args.speeds:
            manifests.append(
                run_natural_match(
                    "nexto",
                    rival_team=0,
                    launcher=args.launcher,
                    timeout=args.timeout,
                    game_speed=speed,
                    challenge_mode="off",
                    lane_id="speed-fallback-lane-1",
                    smoke_game_seconds=args.game_seconds,
                    manager=manager,
                )
            )
    finally:
        manager.shut_down()

    candidates = []
    for speed, manifest in zip(args.speeds, manifests):
        execution = manifest.get("execution") or {}
        observed_all = execution.get("observed_game_speed_all_active") or {}
        observed_sustained = execution.get("observed_game_speed_sustained") or {}
        median = observed_sustained.get("median")
        packet_field_pass = bool(
            manifest.get("status") == "complete"
            and execution.get("requested_speed_reached")
            and isinstance(median, (int, float))
            and abs(float(median) - speed) <= 0.20
        )
        candidates.append(
            {
                "requested_speed": speed,
                "session_id": manifest.get("session_id"),
                "status": manifest.get("status"),
                "packet_field_pass": packet_field_pass,
                "packet_game_speed_all_active": observed_all,
                "packet_game_speed_sustained": observed_sustained,
                "effective_game_seconds_per_wall_second": execution.get(
                    "effective_game_seconds_per_wall_second"
                ),
                "decision_records_per_wall_second": execution.get(
                    "decision_records_per_wall_second"
                ),
                "state_setting_apply_count": execution.get(
                    "state_setting_apply_count"
                ),
                "runtime_warnings": manifest.get("runtime_warnings") or [],
                "raw_telemetry": manifest.get("raw_telemetry") or {},
            }
        )
    selected = next(
        (candidate["requested_speed"] for candidate in candidates if candidate["packet_field_pass"]),
        None,
    )
    report = {
        "report_schema_version": 1,
        "generated_utc": utc_now(),
        "gate": "rival-m03-game-speed-fallback-v1",
        "reason": "5x packet match_info.game_speed remained 1.0",
        "natural_match_budget_consumed": 0,
        "candidates": candidates,
        "selected_accelerated_speed": selected,
        "selected_evidence_speed": selected if selected is not None else 1.0,
        "total_wall_duration_seconds": time.monotonic() - started,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
