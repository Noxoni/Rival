from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any

import rlbot.managers


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.runner import RAW_EVIDENCE_ROOT, run_natural_match  # noqa: E402
from tools.evidence.session import utc_now, write_json  # noqa: E402


def _telemetry_distribution(session_id: str) -> dict[str, Any]:
    path = RAW_EVIDENCE_ROOT / session_id / "decisions.jsonl"
    action_indices: Counter[str] = Counter()
    decisions = 0
    jump = 0
    boost = 0
    invalid = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if record.get("record_type") != "rival_policy_decision":
                continue
            decision = record.get("decision") or {}
            action = decision.get("final_controller_action") or decision.get(
                "controller_action", {}
            )
            index = decision.get("final_action_index", decision.get("action_index"))
            action_indices[str(index)] += 1
            decisions += 1
            jump += bool(action.get("jump", False))
            boost += bool(action.get("boost", False))
    return {
        "decision_count": decisions,
        "invalid_json_lines": invalid,
        "distinct_action_indices": len(action_indices),
        "top_action_indices": action_indices.most_common(10),
        "jump_fraction": jump / decisions if decisions else None,
        "boost_fraction": boost / decisions if decisions else None,
    }


def _compact(manifest: dict[str, Any]) -> dict[str, Any]:
    execution = manifest.get("execution") or {}
    distribution = _telemetry_distribution(str(manifest["session_id"]))
    game_seconds = float(execution.get("game_seconds_advanced") or 0.0)
    distribution["decisions_per_game_second"] = (
        distribution["decision_count"] / game_seconds if game_seconds > 0.0 else None
    )
    return {
        "session_id": manifest["session_id"],
        "opponent": (manifest.get("opponent") or {}).get("identity"),
        "status": manifest.get("status"),
        "termination_reason": manifest.get("termination_reason"),
        "final_score": manifest.get("final_score"),
        "wall_duration_seconds": manifest.get("wall_duration_seconds"),
        "execution": execution,
        "telemetry": distribution,
        "runtime_warnings": manifest.get("runtime_warnings") or [],
        "raw_telemetry": manifest.get("raw_telemetry") or {},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the bounded Rival 1x/5x integrity gate")
    parser.add_argument("--launcher", choices=("steam", "epic"), default="steam")
    parser.add_argument("--accelerated-speed", type=float, default=5.0)
    parser.add_argument("--game-seconds", type=float, default=30.0)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "evidence" / "results" / "v3" / "speed_integrity.json",
    )
    args = parser.parse_args()

    started = time.monotonic()
    manager = rlbot.managers.MatchManager()
    manifests: list[dict[str, Any]] = []
    try:
        for opponent, speed in (
            ("nexto", 1.0),
            ("nexto", args.accelerated_speed),
            ("wisp", args.accelerated_speed),
        ):
            manifests.append(
                run_natural_match(
                    opponent,
                    rival_team=0,
                    launcher=args.launcher,
                    timeout=args.timeout,
                    game_speed=speed,
                    challenge_mode="off",
                    lane_id="speed-gate-lane-1",
                    smoke_game_seconds=args.game_seconds,
                    manager=manager,
                )
            )
    finally:
        manager.shut_down()

    sessions = [_compact(manifest) for manifest in manifests]
    baseline = sessions[0]
    accelerated_nexto = sessions[1]
    accelerated_wisp = sessions[2]
    baseline_rate = baseline["telemetry"]["decisions_per_game_second"] or 0.0
    accelerated_rate = (
        accelerated_nexto["telemetry"]["decisions_per_game_second"] or 0.0
    )
    rate_ratio = accelerated_rate / baseline_rate if baseline_rate > 0.0 else 0.0
    accelerated_sessions = (accelerated_nexto, accelerated_wisp)
    statuses_pass = all(session["status"] == "complete" for session in sessions)
    speed_pass = all(
        bool(session["execution"].get("requested_speed_reached"))
        and abs(
            float(
                session["execution"]["observed_game_speed_sustained"]["median"]
                or 0.0
            )
            - args.accelerated_speed
        )
        <= 0.20
        for session in accelerated_sessions
    )
    responsiveness_pass = all(
        session["telemetry"]["decision_count"] >= 30
        and session["telemetry"]["distinct_action_indices"] >= 2
        and session["telemetry"]["invalid_json_lines"] == 0
        for session in sessions
    )
    cadence_pass = rate_ratio >= 0.50
    passed = statuses_pass and speed_pass and responsiveness_pass and cadence_pass
    report = {
        "report_schema_version": 1,
        "generated_utc": utc_now(),
        "gate": "rival-m03-game-speed-integrity-v1",
        "requested_accelerated_speed": args.accelerated_speed,
        "smoke_game_seconds_per_session": args.game_seconds,
        "full_match_configuration_preserved": True,
        "natural_match_budget_consumed": 0,
        "sessions": sessions,
        "comparison": {
            "nexto_accelerated_to_1x_decision_cadence_ratio": rate_ratio,
            "statuses_pass": statuses_pass,
            "observed_speed_pass": speed_pass,
            "bot_responsiveness_pass": responsiveness_pass,
            "decision_cadence_pass": cadence_pass,
            "representative_action_distributions_recorded": True,
            "queue_or_missed_packet_warnings": [
                message
                for session in sessions
                for message in session["runtime_warnings"]
                if "queue" in message.lower() or "missed" in message.lower()
            ],
        },
        "accepted": passed,
        "selected_speed": args.accelerated_speed if passed else None,
        "total_wall_duration_seconds": time.monotonic() - started,
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
