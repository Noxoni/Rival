"""Run the opt-in Rival v9 export in a native-rate RLBot v5 match smoke."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import psutil


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evidence.runner import run_natural_match  # noqa: E402


DEFAULT_OUTPUT = (
    REPO_ROOT / "training/results/milestone09/gate12_scratch_live_smoke.json"
)
RAW_ROOT = REPO_ROOT / "evidence/raw"
NATIVE_CORPUS_VERSION = "RivalV9NativePacketCorpusV2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _stop_verified_rocket_league() -> list[dict[str, Any]]:
    processes: list[psutil.Process] = []
    records: list[dict[str, Any]] = []
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
            records.append({"pid": process.pid, "name": path.name})
            processes.append(process)
            process.terminate()
        except (OSError, psutil.Error):
            continue
    if processes:
        _, alive = psutil.wait_procs(processes, timeout=20.0)
        if alive:
            raise RuntimeError(
                "Rocket League did not close before the clean Gate 12 launch: "
                + ", ".join(str(process.pid) for process in alive)
            )
        time.sleep(2.0)
    return records


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        return float(ordered[round((len(ordered) - 1) * fraction)])

    return {
        "samples": len(values),
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "maximum": float(ordered[-1]),
    }


def _summarize_native_corpus(path: Path, warmup_records: int) -> dict[str, Any]:
    frames: list[int] = []
    callback_milliseconds: list[float] = []
    invalid = 0
    start = None
    end = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if value.get("record_type") == "rival_v9_native_corpus_start":
                start = value
            elif value.get("record_type") == "rival_v9_native_packet":
                frames.append(int(value["frame_num"]))
                callback_milliseconds.append(float(value["callback_wall_ns"]) / 1e6)
            elif value.get("record_type") == "rival_v9_native_corpus_end":
                end = value
    deltas = [current - prior for prior, current in zip(frames, frames[1:])]
    counts = Counter(deltas)
    longest_contiguous = 1 if frames else 0
    current_contiguous = longest_contiguous
    for delta in deltas:
        if delta == 1:
            current_contiguous += 1
            longest_contiguous = max(longest_contiguous, current_contiguous)
        else:
            current_contiguous = 1
    post_warmup_deltas = deltas[max(0, warmup_records - 1) :]
    maximum_nonunit_run = 0
    current_nonunit_run = 0
    for delta in post_warmup_deltas:
        if delta == 1:
            current_nonunit_run = 0
        else:
            current_nonunit_run += 1
            maximum_nonunit_run = max(maximum_nonunit_run, current_nonunit_run)
    post_warmup_callbacks = callback_milliseconds[warmup_records:]
    return {
        "path": _portable(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "records": len(frames),
        "invalid_json_records": invalid,
        "start_record": start,
        "end_record_present": end is not None,
        "end_record": end,
        "frame_delta_counts": {
            str(delta): count for delta, count in sorted(counts.items())
        },
        "one_tick_transitions": counts[1],
        "longest_contiguous_run": longest_contiguous,
        "post_warmup_maximum_consecutive_nonunit_transitions": maximum_nonunit_run,
        "post_warmup_frame_gap_fraction": (
            sum(delta != 1 for delta in post_warmup_deltas)
            / max(1, len(post_warmup_deltas))
        ),
        "callback_wall_milliseconds_after_warmup": _percentiles(
            post_warmup_callbacks
        ),
        "warmup_records_excluded_from_callback_gate": warmup_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--opponent", choices=("wisp", "nexto"), default="wisp")
    parser.add_argument("--rival-team", type=int, choices=(0, 1), default=0)
    parser.add_argument("--launcher", choices=("steam", "epic", "no-launch"), default="steam")
    parser.add_argument("--maximum-records", type=int, default=2400)
    parser.add_argument("--smoke-game-seconds", type=float, default=28.0)
    parser.add_argument("--warmup-records", type=int, default=120)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reassess-existing", action="store_true")
    return parser.parse_args()


def _reassess_existing(path: Path) -> int:
    """Correct report-only accounting without replaying a completed live smoke."""

    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report["checks"]
    checks.pop("one_decision_per_record_with_bounded_startup_difference", None)
    runtime = report["scratch_runtime"]
    accounted_transitions = sum(
        int(count) for count in runtime["frame_delta_counts"].values()
    )
    derived_active_segments = int(runtime["decision_count"]) - accounted_transitions
    checks["runtime_decision_frame_accounting_consistent"] = bool(
        1 <= derived_active_segments <= 16
        and runtime["duplicate_packets_reused_previous_controller"] == 0
        and runtime["out_of_order_packets"] == 0
    )
    runtime["derived_active_segments"] = derived_active_segments
    report["evidence_reassessment"] = {
        "assessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "match_rerun": False,
        "reason": (
            "The native corpus intentionally stopped at its 2,400-record bound while "
            "the runtime continued through the complete 28-second match window. "
            "Decision accounting must therefore use the runtime's own frame deltas; "
            "decision_count minus transition_count is the number of active segments."
        ),
        "removed_invalid_check": (
            "one_decision_per_record_with_bounded_startup_difference"
        ),
        "replacement_check": "runtime_decision_frame_accounting_consistent",
        "decision_count": runtime["decision_count"],
        "accounted_transitions": accounted_transitions,
        "derived_active_segments": derived_active_segments,
    }
    report["status"] = "passed" if all(checks.values()) else "failed"
    _write_json(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    args = parse_args()
    if args.reassess_existing:
        if not args.output.is_file():
            raise FileNotFoundError(f"Existing Gate 12 report is missing: {args.output}")
        return _reassess_existing(args.output)
    if args.model is None or args.metadata is None:
        raise ValueError("--model and --metadata are required for a live smoke")
    if args.maximum_records < 1200:
        raise ValueError("Gate 12 live smoke requires at least 1200 native packets")
    if args.smoke_game_seconds < args.maximum_records / 120.0 + 5.0:
        raise ValueError("Live smoke duration is too short for the packet bound")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_directory = RAW_ROOT / f"rival-v9-gate12-live-{stamp}"
    native_path = raw_directory / "native_packets.jsonl"
    runtime_path = raw_directory / "scratch_runtime.json"
    stopped = _stop_verified_rocket_league()
    manifest = run_natural_match(
        args.opponent,
        rival_team=args.rival_team,
        launcher=args.launcher,
        timeout=max(240.0, args.smoke_game_seconds * 3.0),
        game_speed=1.0,
        challenge_mode="off",
        lane_id="m09-gate12-native-scratch",
        execution_regime="sequential_single_match",
        smoke_game_seconds=args.smoke_game_seconds,
        session_version="v9",
        session_source="milestone09_gate12_native_scratch",
        experiment_milestone="m09-scratch-policy",
        experiment_metadata={
            "purpose": "native_120hz_export_runtime_smoke",
            "score_used_for_gate": False,
            "win_loss_used_for_gate": False,
            "production_promotion": False,
        },
        rival_environment_overrides={
            "RIVAL_TRANSFER_DIAGNOSTIC_MODE": "1",
            "RIVAL_V9_SCRATCH_MODEL_PATH": str(args.model.resolve()),
            "RIVAL_V9_SCRATCH_METADATA_PATH": str(args.metadata.resolve()),
            "RIVAL_V9_SCRATCH_RUNTIME_EVIDENCE_PATH": str(runtime_path),
            "RIVAL_V9_SCRATCH_RUNTIME_LABEL": "m09_gate12_native_scratch",
            "RIVAL_V9_NATIVE_CORPUS_ENABLED": "1",
            "RIVAL_V9_NATIVE_CORPUS_PATH": str(native_path),
            "RIVAL_V9_NATIVE_CORPUS_MAX_RECORDS": str(args.maximum_records),
            "RIVAL_M08_DUAL_RATE_ENABLED": "0",
            "RIVAL_TICK_SKIP": "1",
        },
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and not runtime_path.is_file():
        time.sleep(0.25)
    if not native_path.is_file():
        raise FileNotFoundError(f"Gate 12 native corpus was not written: {native_path}")
    if not runtime_path.is_file():
        raise FileNotFoundError(
            f"Gate 12 scratch runtime evidence was not written: {runtime_path}"
        )
    corpus = _summarize_native_corpus(native_path, args.warmup_records)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    callback = corpus["callback_wall_milliseconds_after_warmup"]
    pipeline = runtime["pipeline_milliseconds"]
    checks = {
        "natural_smoke_completed": manifest.get("status") == "complete",
        "native_game_speed_requested": manifest.get("execution", {}).get(
            "requested_game_speed"
        )
        == 1.0,
        "native_game_speed_observed": manifest.get("execution", {}).get(
            "requested_speed_reached"
        )
        is True,
        "scratch_opt_in_runtime_recorded": (
            (corpus.get("start_record") or {}).get("metadata", {}).get(
                "policy_runtime_mode"
            )
            == "m09_gate12_native_scratch"
        ),
        "native_record_bound_reached": corpus["records"] == args.maximum_records,
        "native_corpus_complete": corpus["end_record_present"],
        "native_corpus_json_valid": corpus["invalid_json_records"] == 0,
        "long_contiguous_native_run": corpus["longest_contiguous_run"] >= 1200,
        "no_sustained_post_warmup_frame_gaps": corpus[
            "post_warmup_maximum_consecutive_nonunit_transitions"
        ]
        <= 1,
        "post_warmup_gap_fraction_below_half_percent": corpus[
            "post_warmup_frame_gap_fraction"
        ]
        < 0.005,
        "callback_p99_within_native_frame_budget": callback["p99"] < (1000.0 / 120.0),
        "runtime_full_pipeline_p99_below_6ms": pipeline["total"]["p99"] < 6.0,
        "runtime_actor_p99_below_2ms_target": pipeline["actor"]["p99"] < 2.0,
        "runtime_actor_maximum_below_4ms_hard_limit": pipeline["actor"][
            "maximum"
        ]
        < 4.0,
        "runtime_decision_frame_accounting_consistent": bool(
            1
            <= int(runtime["decision_count"])
            - sum(int(count) for count in runtime["frame_delta_counts"].values())
            <= 16
            and runtime["duplicate_packets_reused_previous_controller"] == 0
            and runtime["out_of_order_packets"] == 0
        ),
        "runtime_outputs_finite": runtime["non_finite_outputs"] == 0,
        "runtime_controllers_legal": runtime["illegal_controllers"] == 0,
        "no_python_runtime_warnings": not manifest.get("runtime_warnings"),
        "production_not_promoted": runtime["policy_mode"]
        == "opt_in_scratch_candidate",
    }
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 12,
        "purpose": "opt_in_native_120hz_rlbot_process_smoke",
        "status": "passed" if all(checks.values()) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "model": {
            "path": _portable(args.model),
            "sha256": _sha256(args.model),
            "size_bytes": args.model.stat().st_size,
        },
        "metadata": {
            "path": _portable(args.metadata),
            "sha256": _sha256(args.metadata),
            "size_bytes": args.metadata.stat().st_size,
        },
        "opponent": args.opponent,
        "rival_team": args.rival_team,
        "stopped_prior_rocket_league_processes": stopped,
        "native_corpus": corpus,
        "scratch_runtime": {
            **runtime,
            "raw_path": _portable(runtime_path),
            "raw_sha256": _sha256(runtime_path),
            "raw_size_bytes": runtime_path.stat().st_size,
        },
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
            "game_speed": 1.0,
            "five_x_used_for_native_certification": False,
            "startup_warmup_records_excluded_only_from_latency_and_sustained_gap_checks": args.warmup_records,
            "production_promoted": False,
        },
        "production_model_hashes": {
            "POLICY.lt": _sha256(REPO_ROOT / "bot/models/POLICY.lt"),
            "SHARED_HEAD.lt": _sha256(REPO_ROOT / "bot/models/SHARED_HEAD.lt"),
        },
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
