from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable, Mapping

import rlbot.managers


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.runner import RAW_EVIDENCE_ROOT, run_natural_match  # noqa: E402
from tools.evidence.session import utc_now, write_json  # noqa: E402


OPPONENTS = ("nexto", "wisp")
PROTOCOL = "rival-m04p1-natural-batch-v1"


def build_schedule(matches_per_opponent: int) -> list[dict[str, Any]]:
    if matches_per_opponent < 1:
        raise ValueError("matches per opponent must be positive")
    schedule = []
    for repetition in range(matches_per_opponent):
        for opponent_index, opponent in enumerate(OPPONENTS):
            rival_team = (repetition + opponent_index) % 2
            schedule.append(
                {
                    "index": len(schedule) + 1,
                    "opponent": opponent,
                    "opponent_repetition": repetition + 1,
                    "rival_team": rival_team,
                    "rival_side": "blue" if rival_team == 0 else "orange",
                }
            )
    return schedule


def _input_signature(value: Mapping[str, Any]) -> str:
    relevant = {
        key: value.get(key)
        for key in (
            "throttle",
            "steer",
            "pitch",
            "yaw",
            "roll",
            "jump",
            "boost",
            "handbrake",
        )
    }
    return json.dumps(relevant, sort_keys=True, separators=(",", ":"))


def telemetry_health(session_id: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    telemetry_path = RAW_EVIDENCE_ROOT / session_id / "decisions.jsonl"
    actions: Counter[int] = Counter()
    opponent_inputs: Counter[str] = Counter()
    decision_count = 0
    invalid_lines = 0
    schema_versions: Counter[int] = Counter()
    previous_decision_sample: tuple[float, float, str | None, tuple[Any, ...]] | None = None
    in_play_game_seconds = 0.0
    in_play_wall_seconds = 0.0
    in_play_interval_rates: list[float] = []
    if telemetry_path.is_file():
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            schema = record.get("schema_version")
            if isinstance(schema, int):
                schema_versions[schema] += 1
            if record.get("record_type") != "rival_policy_decision":
                continue
            decision_count += 1
            decision = record.get("decision") or {}
            packet = record.get("packet") or {}
            match = packet.get("match") or {}
            phase = (match.get("phase") or {}).get("name")
            scores = tuple(
                item.get("score")
                for item in match.get("scores") or []
                if isinstance(item, dict)
            )
            game_time = decision.get("game_time")
            timestamp_ns = decision.get("timestamp_unix_ns")
            if isinstance(game_time, (int, float)) and isinstance(timestamp_ns, int):
                sample = (float(game_time), timestamp_ns / 1e9, phase, scores)
                if previous_decision_sample is not None:
                    game_delta = sample[0] - previous_decision_sample[0]
                    wall_delta = sample[1] - previous_decision_sample[1]
                    same_active_segment = (
                        previous_decision_sample[2] == "Active"
                        and sample[2] == "Active"
                        and previous_decision_sample[3] == sample[3]
                    )
                    if (
                        same_active_segment
                        and 0.0 < game_delta <= 0.25
                        and 0.0 < wall_delta <= 0.25
                    ):
                        in_play_game_seconds += game_delta
                        in_play_wall_seconds += wall_delta
                        in_play_interval_rates.append(game_delta / wall_delta)
                previous_decision_sample = sample
            action_index = decision.get("final_action_index", decision.get("action_index"))
            if isinstance(action_index, int):
                actions[action_index] += 1
            players = packet.get("players") or []
            for index in packet.get("opponent_indices") or []:
                if isinstance(index, int) and 0 <= index < len(players):
                    opponent_inputs[_input_signature(
                        (players[index] or {}).get("last_input") or {}
                    )] += 1
    zero_signature = _input_signature({})
    execution = manifest.get("execution") or {}
    game_seconds = float(execution.get("game_seconds_advanced") or 0.0)
    end_to_end_speed = execution.get("effective_game_seconds_per_wall_second")
    effective_speed = (
        in_play_game_seconds / in_play_wall_seconds
        if in_play_wall_seconds > 0.0
        else None
    )
    decisions_per_game_second = (
        decision_count / game_seconds if game_seconds > 0.0 else None
    )
    checks = {
        "full_match_complete": (
            manifest.get("status") == "complete"
            and manifest.get("termination_reason") == "match_phase_ended"
        ),
        "effective_5x_progression": (
            isinstance(effective_speed, (int, float))
            and 4.5 <= float(effective_speed) <= 5.5
            and len(in_play_interval_rates) >= 100
        ),
        "decision_cadence_healthy": (
            isinstance(decisions_per_game_second, (int, float))
            and float(decisions_per_game_second) >= 8.0
        ),
        "action_distribution_non_degenerate": len(actions) >= 10,
        "opponent_process_responsive": (
            sum(opponent_inputs.values()) - opponent_inputs.get(zero_signature, 0) > 0
        ),
        "telemetry_valid": (
            telemetry_path.is_file()
            and invalid_lines == 0
            and not (manifest.get("raw_telemetry") or {}).get(
                "invalid_record_count"
            )
        ),
    }
    return {
        "decision_count": decision_count,
        "game_seconds_advanced": game_seconds,
        "decisions_per_game_second": decisions_per_game_second,
        "effective_game_seconds_per_wall_second": effective_speed,
        "sustained_in_play_speed": {
            "method": (
                "weighted game-time/wall-time ratio across consecutive Rival decisions "
                "inside the same Active score segment; reset/goal/kickoff gaps excluded"
            ),
            "accepted_interval_count": len(in_play_interval_rates),
            "game_seconds": in_play_game_seconds,
            "wall_seconds": in_play_wall_seconds,
            "weighted_rate": effective_speed,
            "median_interval_rate": (
                statistics.median(in_play_interval_rates)
                if in_play_interval_rates
                else None
            ),
            "p10_interval_rate": (
                statistics.quantiles(in_play_interval_rates, n=10)[0]
                if len(in_play_interval_rates) >= 10
                else None
            ),
            "p90_interval_rate": (
                statistics.quantiles(in_play_interval_rates, n=10)[-1]
                if len(in_play_interval_rates) >= 10
                else None
            ),
        },
        "end_to_end_game_seconds_per_wall_second": end_to_end_speed,
        "distinct_action_indices": len(actions),
        "top_action_indices": actions.most_common(10),
        "opponent_input_sample_count": sum(opponent_inputs.values()),
        "opponent_nonzero_input_sample_count": (
            sum(opponent_inputs.values()) - opponent_inputs.get(zero_signature, 0)
        ),
        "schema_record_counts": dict(sorted(schema_versions.items())),
        "invalid_json_lines": invalid_lines,
        "checks": checks,
        "accepted": all(checks.values()),
    }


def compact_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    session_id = str(manifest["session_id"])
    opponent = manifest.get("opponent") or {}
    compact_opponent = {
        key: opponent.get(key)
        for key in (
            "key",
            "identity",
            "config_sha256",
            "executable_sha256",
        )
        if opponent.get(key) is not None
    }
    for source_key, target_key in (
        ("config_path", "config_filename"),
        ("executable_path", "executable_filename"),
    ):
        raw_path = opponent.get(source_key)
        if raw_path:
            compact_opponent[target_key] = Path(str(raw_path)).name
    return {
        "session_id": session_id,
        "status": manifest.get("status"),
        "termination_reason": manifest.get("termination_reason"),
        # Raw manifests retain resolved installed-reference paths locally. The
        # committed batch needs identity and hashes, not a workstation path.
        "opponent": compact_opponent,
        "rival_team": manifest.get("rival_team"),
        "team_assignment": manifest.get("team_assignment") or {},
        "final_score": manifest.get("final_score") or {},
        "rival_git_commit": manifest.get("rival_git_commit"),
        "rival_model_sha256": manifest.get("rival_model_sha256") or {},
        "runtime_versions": manifest.get("runtime_versions") or {},
        "wall_duration_seconds": manifest.get("wall_duration_seconds"),
        "raw_telemetry": manifest.get("raw_telemetry") or {},
        "execution": manifest.get("execution") or {},
        "runtime_warnings": manifest.get("runtime_warnings") or [],
        "error": manifest.get("error"),
        "health": telemetry_health(session_id, manifest),
    }


def _summarize(sessions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(sessions)
    effective = [
        float(session["health"]["effective_game_seconds_per_wall_second"])
        for session in values
        if isinstance(
            session["health"].get("effective_game_seconds_per_wall_second"),
            (int, float),
        )
    ]
    cadence = [
        float(session["health"]["decisions_per_game_second"])
        for session in values
        if isinstance(session["health"].get("decisions_per_game_second"), (int, float))
    ]
    return {
        "scheduled_match_count": len(values),
        "complete_match_count": sum(
            session.get("status") == "complete"
            and session.get("termination_reason") == "match_phase_ended"
            for session in values
        ),
        "health_accepted_match_count": sum(
            bool((session.get("health") or {}).get("accepted")) for session in values
        ),
        "decision_count": sum(
            int((session.get("health") or {}).get("decision_count") or 0)
            for session in values
        ),
        "raw_bytes": sum(
            int((session.get("raw_telemetry") or {}).get("bytes") or 0)
            for session in values
        ),
        "wall_duration_seconds": sum(
            float(session.get("wall_duration_seconds") or 0.0) for session in values
        ),
        "effective_speed": {
            "minimum": min(effective) if effective else None,
            "median": statistics.median(effective) if effective else None,
            "maximum": max(effective) if effective else None,
        },
        "decisions_per_game_second": {
            "minimum": min(cadence) if cadence else None,
            "median": statistics.median(cadence) if cadence else None,
            "maximum": max(cadence) if cadence else None,
        },
        "opponents": dict(
            Counter(
                str((session.get("opponent") or {}).get("key") or "unknown")
                for session in values
            )
        ),
        "rival_sides": dict(
            Counter(
                "blue" if int(session.get("rival_team") or 0) == 0 else "orange"
                for session in values
            )
        ),
        "natural_match_budget_kind": "v4.1 optimization throughput",
    }


def _batch_id(phase: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"m04p1-{phase}-{stamp}"


def _batch_report(
    *,
    batch_id: str,
    phase: str,
    adjustment_mode: str,
    parameter_version: str,
    requested_game_speed: float,
    schedule: list[dict[str, Any]],
    sessions: list[dict[str, Any]],
    failed: bool,
    invocation_started: float,
) -> dict[str, Any]:
    return {
        "report_schema_version": 1,
        "generated_utc": utc_now(),
        "protocol": PROTOCOL,
        "batch_id": batch_id,
        "phase": phase,
        "adjustment_mode": adjustment_mode,
        "parameter_version": parameter_version,
        "requested_game_speed": requested_game_speed,
        "packet_game_speed_echo_is_acceptance_oracle": False,
        "match_configuration": {
            "full_five_minute_soccar": True,
            "map": "Stadium_P",
            "normal_boost_gravity_demolition_scoring": True,
            "normal_kickoff_countdowns": True,
            "skip_goal_replays": True,
            "auto_save_replay": False,
            "debug_rendering": "AlwaysOff",
            "performance_monitor": "NeverShow",
            "wait_for_agents": True,
            "existing_match_behavior": "Restart",
            "natural_state_setting_scope": "desired_match_info.game_speed_only",
        },
        "schedule": schedule,
        "sessions": sessions,
        "summary": _summarize(sessions),
        "complete": len(sessions) == len(schedule) and not failed,
        "stopped_on_health_failure": failed,
        "runner_wall_duration_seconds_this_invocation": (
            time.monotonic() - invocation_started
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a balanced full-match Rival v4.1 natural-play batch"
    )
    parser.add_argument("--phase", choices=("baseline", "treatment"), required=True)
    parser.add_argument("--matches-per-opponent", type=int, default=4)
    parser.add_argument("--game-speed", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--launcher", choices=("steam", "epic"), default="steam")
    parser.add_argument(
        "--adjustment-mode",
        choices=("off", "observe", "intervene"),
        default="off",
    )
    parser.add_argument("--parameter-version", default="none")
    parser.add_argument("--batch-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.matches_per_opponent < 1:
        parser.error("matches-per-opponent must be positive")
    if args.game_speed <= 0.0 or args.timeout <= 0.0:
        parser.error("game-speed and timeout must be positive")
    if args.phase == "baseline" and args.adjustment_mode != "off":
        parser.error("baseline phase requires adjustment-mode off")
    if args.phase == "treatment" and args.adjustment_mode == "off":
        parser.error("treatment phase requires observe or intervene mode")
    output = args.output or (
        REPOSITORY_ROOT
        / "evidence"
        / "results"
        / "v4.1"
        / f"natural_{args.phase}_batch.json"
    )
    schedule = build_schedule(args.matches_per_opponent)
    batch_id = args.batch_id or _batch_id(args.phase)
    sessions: list[dict[str, Any]] = []
    if output.is_file():
        if not args.resume:
            parser.error(f"output already exists; use --resume to continue: {output}")
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("protocol") != PROTOCOL:
            parser.error("existing output has the wrong protocol")
        if args.batch_id is None:
            batch_id = str(existing.get("batch_id"))
        if existing.get("batch_id") != batch_id:
            parser.error("--batch-id must match the existing report when resuming")
        sessions = list(existing.get("sessions") or [])
        for session in sessions:
            session_id = str(session["session_id"])
            session["health"] = telemetry_health(session_id, session)
            # Normalize reports produced by an earlier checkpoint of this runner.
            session["opponent"] = compact_manifest(
                {"session_id": session_id, "opponent": session.get("opponent")}
            )["opponent"]
    completed_indices = {
        int(session["schedule_index"])
        for session in sessions
        if session.get("schedule_index") is not None
    }
    started = time.monotonic()
    pending = [item for item in schedule if item["index"] not in completed_indices]
    manager = rlbot.managers.MatchManager() if pending else None
    failed = any(
        not bool((session.get("health") or {}).get("accepted"))
        for session in sessions
    )
    try:
        for item in schedule:
            if item["index"] in completed_indices:
                continue
            assert manager is not None
            overrides = {}
            if args.adjustment_mode != "off":
                overrides = {
                    "RIVAL_NATURAL_ADJUSTMENT_MODE": args.adjustment_mode,
                    "RIVAL_NATURAL_PARAMETER_VERSION": args.parameter_version,
                }
            manifest = run_natural_match(
                item["opponent"],
                rival_team=item["rival_team"],
                launcher=args.launcher,
                timeout=args.timeout,
                game_speed=args.game_speed,
                challenge_mode="off",
                lane_id="m04p1-natural-sequential-lane-1",
                execution_regime="sequential",
                session_version="v4p1",
                session_source="natural_match",
                experiment_milestone="m04.1-natural-play-optimization",
                experiment_metadata={
                    "protocol": PROTOCOL,
                    "batch_id": batch_id,
                    "phase": args.phase,
                    "schedule_index": item["index"],
                    "adjustment_mode": args.adjustment_mode,
                    "parameter_version": args.parameter_version,
                },
                rival_environment_overrides=overrides,
                manager=manager,
            )
            compact = {
                **compact_manifest(manifest),
                "schedule_index": item["index"],
                "opponent_repetition": item["opponent_repetition"],
                "adjustment_mode": args.adjustment_mode,
                "parameter_version": args.parameter_version,
            }
            sessions.append(compact)
            failed = not compact["health"]["accepted"]
            report = _batch_report(
                batch_id=batch_id,
                phase=args.phase,
                adjustment_mode=args.adjustment_mode,
                parameter_version=args.parameter_version,
                requested_game_speed=args.game_speed,
                schedule=schedule,
                sessions=sessions,
                failed=failed,
                invocation_started=started,
            )
            write_json(output, report)
            print(
                json.dumps(
                    {
                        "schedule_index": item["index"],
                        "scheduled": len(schedule),
                        "session_id": compact["session_id"],
                        "score": compact["final_score"],
                        "effective_speed": compact["health"][
                            "effective_game_seconds_per_wall_second"
                        ],
                        "decisions_per_game_second": compact["health"][
                            "decisions_per_game_second"
                        ],
                        "health_accepted": compact["health"]["accepted"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if failed:
                break
    finally:
        if manager is not None:
            manager.shut_down()

    # A resume with a complete schedule has no loop iteration. Rewrite it anyway
    # so health recomputation and evidence-path normalization are persisted.
    write_json(
        output,
        _batch_report(
            batch_id=batch_id,
            phase=args.phase,
            adjustment_mode=args.adjustment_mode,
            parameter_version=args.parameter_version,
            requested_game_speed=args.game_speed,
            schedule=schedule,
            sessions=sessions,
            failed=failed,
            invocation_started=started,
        ),
    )

    final = json.loads(output.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "output": str(output),
                "complete": final["complete"],
                "matches_completed": final["summary"]["complete_match_count"],
                "matches_scheduled": len(schedule),
                "health_accepted": final["summary"]["health_accepted_match_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
