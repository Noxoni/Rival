"""Capture a bounded natural RLBot-v5 stream at native physics cadence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import psutil


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evidence.runner import run_natural_match  # noqa: E402


RESULT_PATH = REPO_ROOT / "training" / "results" / "milestone09" / "gate03_native_capture.json"
RAW_ROOT = REPO_ROOT / "evidence" / "raw"
NATIVE_CORPUS_VERSION = "RivalV9NativePacketCorpusV2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _stop_verified_rocket_league() -> list[dict[str, Any]]:
    processes = []
    records = []
    for process in psutil.process_iter(("pid", "name", "exe")):
        try:
            executable = process.info.get("exe")
            if str(process.info.get("name") or "").lower() != "rocketleague.exe":
                continue
            if not executable:
                continue
            path = Path(executable).resolve()
            if path.name.lower() != "rocketleague.exe":
                continue
            processes.append(process)
            records.append({"pid": process.pid, "path": str(path)})
            process.terminate()
        except (OSError, psutil.Error):
            continue
    if processes:
        _, alive = psutil.wait_procs(processes, timeout=20.0)
        if alive:
            raise RuntimeError(
                "Rocket League did not close for a clean native-corpus launch: "
                + ", ".join(str(process.pid) for process in alive)
            )
        time.sleep(2.0)
    return records


def _summarize(path: Path) -> dict[str, Any]:
    records = 0
    invalid = 0
    frames = []
    versions: Counter[str] = Counter()
    end_record = None
    callback_ns = []
    field_info_goals: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            record_type = value.get("record_type")
            if record_type == "rival_v9_native_packet":
                records += 1
                frames.append(int(value["frame_num"]))
                versions[str(value.get("corpus_version"))] += 1
                callback_ns.append(int(value["callback_wall_ns"]))
                for goal in (value.get("packet") or {}).get("goals", []):
                    metadata = {
                        "team_num": goal.get("team_num"),
                        "location": goal.get("location"),
                        "direction": goal.get("direction"),
                        "width": goal.get("width"),
                        "height": goal.get("height"),
                    }
                    field_info_goals[
                        json.dumps(metadata, sort_keys=True, separators=(",", ":"))
                    ] += 1
            elif record_type == "rival_v9_native_corpus_end":
                end_record = value
    deltas = Counter(
        current - previous for previous, current in zip(frames, frames[1:])
    )
    longest = 0
    current_run = 0
    previous = None
    for frame in frames:
        if previous is None or frame - previous == 1:
            current_run += 1
        else:
            current_run = 1
        longest = max(longest, current_run)
        previous = frame
    ordered_callback = sorted(callback_ns)

    def percentile(fraction: float) -> float | None:
        if not ordered_callback:
            return None
        index = round((len(ordered_callback) - 1) * fraction)
        return ordered_callback[index] / 1e6

    return {
        "version": NATIVE_CORPUS_VERSION,
        "path": path.resolve().relative_to(REPO_ROOT.resolve()).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "records": records,
        "invalid_json_records": invalid,
        "version_counts": dict(sorted(versions.items())),
        "frame_delta_counts": {str(key): value for key, value in sorted(deltas.items())},
        "one_tick_transitions": deltas[1],
        "longest_contiguous_run": longest,
        "first_frame": frames[0] if frames else None,
        "last_frame": frames[-1] if frames else None,
        "end_record_present": end_record is not None,
        "end_record": end_record,
        "callback_wall_milliseconds": {
            "samples": len(callback_ns),
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "maximum": (max(callback_ns) / 1e6 if callback_ns else None),
        },
        "field_info_goal_metadata": [
            {**json.loads(serialized), "samples": samples}
            for serialized, samples in sorted(field_info_goals.items())
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=("wisp", "nexto"), default="wisp")
    parser.add_argument("--rival-team", type=int, choices=(0, 1), default=0)
    parser.add_argument("--maximum-records", type=int, default=6000)
    parser.add_argument("--smoke-game-seconds", type=float, default=70.0)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--launcher", choices=("steam", "epic", "no-launch"), default="steam")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.maximum_records < 4096:
        raise ValueError("Gate 3 native capture requires at least 4096 records")
    if args.smoke_game_seconds < args.maximum_records / 120.0 + 5.0:
        raise ValueError("Smoke window is too short for the requested native record bound")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    capture_root = RAW_ROOT / f"rival-v9-native-corpus-{stamp}"
    native_path = capture_root / "native_packets.jsonl"
    stopped = _stop_verified_rocket_league()
    manifest = run_natural_match(
        args.opponent,
        rival_team=args.rival_team,
        launcher=args.launcher,
        timeout=max(240.0, args.smoke_game_seconds * 3.0),
        game_speed=1.0,
        challenge_mode="off",
        lane_id="m09-gate03-native-corpus",
        execution_regime="sequential_single_match",
        smoke_game_seconds=args.smoke_game_seconds,
        session_version="v9",
        session_source="milestone09_native_observation_corpus",
        experiment_milestone="m09-scratch-policy",
        experiment_metadata={
            "purpose": "observation_and_one_tick_history_parity",
            "score_used_for_gate": False,
            "m08_training_budget_used": False,
            "policy": "frozen_production_wisp_only_for_natural_state_generation",
        },
        rival_environment_overrides={
            "RIVAL_TRANSFER_DIAGNOSTIC_MODE": "1",
            "RIVAL_V9_NATIVE_CORPUS_ENABLED": "1",
            "RIVAL_V9_NATIVE_CORPUS_PATH": str(native_path),
            "RIVAL_V9_NATIVE_CORPUS_MAX_RECORDS": str(args.maximum_records),
            "RIVAL_M08_DUAL_RATE_ENABLED": "0",
            "RIVAL_TICK_SKIP": "8",
        },
    )
    if not native_path.is_file():
        raise FileNotFoundError(f"Native Rival packet corpus was not written: {native_path}")
    corpus = _summarize(native_path)
    checks = {
        "natural_smoke_completed": manifest.get("status") == "complete",
        "native_record_bound_reached": corpus["records"] == args.maximum_records,
        "no_invalid_json": corpus["invalid_json_records"] == 0,
        "exact_corpus_version": corpus["version_counts"]
        == {NATIVE_CORPUS_VERSION: args.maximum_records},
        "end_record_present": corpus["end_record_present"],
        "at_least_4000_one_tick_transitions": corpus["one_tick_transitions"] >= 4000,
        "at_least_4000_contiguous_ticks": corpus["longest_contiguous_run"] >= 4000,
        "native_game_speed_requested": manifest.get("execution", {}).get(
            "requested_game_speed"
        )
        == 1.0,
        "field_info_goals_captured": len(corpus["field_info_goal_metadata"]) == 2,
        "no_manifest_runtime_warnings": not manifest.get("runtime_warnings"),
        "frame_gaps_measured_and_not_interpolated": (
            corpus["one_tick_transitions"]
            + sum(
                count
                for delta, count in corpus["frame_delta_counts"].items()
                if int(delta) != 1
            )
            == max(0, corpus["records"] - 1)
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "schema_version": 2,
        "milestone": 9,
        "gate": 3,
        "purpose": "bounded_native_tick_observation_corpus_capture",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "session_id": manifest.get("session_id"),
        "opponent": args.opponent,
        "rival_team": args.rival_team,
        "rival_side": "blue" if args.rival_team == 0 else "orange",
        "stopped_prior_rocket_league_processes": stopped,
        "native_corpus": corpus,
        "natural_match_manifest": {
            "status": manifest.get("status"),
            "termination_reason": manifest.get("termination_reason"),
            "wall_duration_seconds": manifest.get("wall_duration_seconds"),
            "runtime_warnings": manifest.get("runtime_warnings"),
            "execution": manifest.get("execution"),
        },
        "gate_semantics": {
            "score_used": False,
            "win_loss_used": False,
            "m08_training_budget_used": False,
            "capture_policy": "frozen production Wisp for natural state generation only",
            "native_tick_requirement": "RLBot packet frame_num increments by exactly one",
            "gap_policy": (
                "Every non-unit frame delta is reported; downstream history replay resets "
                "rather than interpolating across a gap."
            ),
        },
        "production_model_hashes": {
            "POLICY.lt": _sha256(REPO_ROOT / "bot" / "models" / "POLICY.lt"),
            "SHARED_HEAD.lt": _sha256(REPO_ROOT / "bot" / "models" / "SHARED_HEAD.lt"),
        },
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
