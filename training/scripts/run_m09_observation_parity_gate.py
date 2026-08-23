"""Run Milestone 09 Gate 3 over broad and native-tick RLBot corpora."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_canonical import (  # noqa: E402
    RivalCanonicalStateV1,
    RLBotCanonicalAdapterV1,
)
from rival_training.v9_observations import (  # noqa: E402
    HISTORY_TICKS,
    RivalObsV1Builder,
    deterministic_intercept_time,
    observation_schema_manifest,
)
from rival_training.v9_rlbot_corpus import (  # noqa: E402
    NATIVE_CORPUS_VERSION,
    SourceAuditAccumulator,
    audit_canonical_against_snapshot,
    packet_coverage,
    snapshot_to_rlbot_sources,
)


RESULT_PATH = TRAINING_ROOT / "results" / "milestone09" / "gate03_observation_parity.json"
M08_REPORT = TRAINING_ROOT / "results" / "milestone08" / "rlbot_005m_native_rate.json"
DEFAULT_NATIVE_REPORT = (
    TRAINING_ROOT / "results" / "milestone09" / "gate03_native_capture.json"
)
AUDIT_NAMES = (
    "match",
    "self_physics",
    "opponent_physics",
    "self_controller",
    "opponent_controller",
    "self_resources",
    "opponent_resources",
    "self_air_dodge",
    "opponent_air_dodge",
    "ball_physics",
    "goals",
    "boost_pads",
    "touch",
)
REQUIRED_COVERAGE = (
    "kickoff",
    "normal_ground_play",
    "low_boost",
    "high_boost",
    "ball_contact",
    "first_jump_hold",
    "first_jump_release",
    "double_jump",
    "directional_dodge",
    "flip_cancel",
    "wall_contact",
    "ceiling_contact",
    "awkward_recovery",
    "late_clock",
)
NATURALLY_AVAILABLE_COVERAGE = (
    "airborne_reset_resource",
    "demolition_respawn",
    "overtime",
)


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


def _relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _broad_sources() -> list[dict[str, Any]]:
    report = json.loads(M08_REPORT.read_text(encoding="utf-8"))
    games = report["modes"]["M8C"]["games"]
    sources = []
    for game in games:
        session_id = str(game["session_id"])
        session_root = REPO_ROOT / "evidence" / "raw" / session_id
        path = session_root / "decisions.jsonl"
        manifest_path = session_root / "session_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sources.append(
            {
                "kind": "historical_natural_breadth",
                "session_id": session_id,
                "path": path,
                "manifest_path": manifest_path,
                "manifest_status": manifest.get("status"),
                "expected_policy_records": int(game["decision_records"]),
                "expected_sha256": manifest["raw_telemetry"]["sha256"],
                "rival_side": game["rival_side"],
                "opponent": game["opponent"],
            }
        )
    return sources


def _native_source(report_path: Path) -> dict[str, Any] | None:
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    corpus = report.get("native_corpus", {})
    raw_path = Path(str(corpus.get("path", "")))
    if not raw_path.is_absolute():
        raw_path = REPO_ROOT / raw_path
    return {
        "kind": "native_tick_history",
        "session_id": report.get("session_id", "unknown"),
        "path": raw_path,
        "capture_report_path": report_path,
        "capture_status": report.get("status"),
        "expected_records": int(corpus.get("records", 0)),
        "expected_sha256": corpus.get("sha256"),
        "rival_side": report.get("rival_side"),
        "opponent": report.get("opponent"),
    }


def _records(source: Mapping[str, Any]) -> Iterable[tuple[dict[str, Any], bytes]]:
    path = Path(source["path"])
    with path.open("rb") as stream:
        for raw in stream:
            record = json.loads(raw)
            kind = source["kind"]
            if kind == "historical_natural_breadth":
                if record.get("record_type") != "rival_policy_decision":
                    continue
            elif record.get("record_type") != "rival_v9_native_packet":
                continue
            yield record, raw


def _field_slices() -> dict[str, slice]:
    return {
        field["name"]: slice(int(field["start"]), int(field["end"]))
        for field in observation_schema_manifest()["fields"]
    }


class CorpusAudit:
    def __init__(self) -> None:
        self.source_audits = {
            name: SourceAuditAccumulator() for name in AUDIT_NAMES
        }
        self.coverage: Counter[str] = Counter()
        self.phases: Counter[str] = Counter()
        self.field_info_goal_metadata: Counter[str] = Counter()
        self.frame_deltas: Counter[int] = Counter()
        self.prediction_ages: Counter[int] = Counter()
        self.samples = 0
        self.breadth_samples = 0
        self.native_samples = 0
        self.bit_identical = 0
        self.bit_mismatches = 0
        self.maximum_replay_abs_error = 0.0
        self.nonfinite_observation_values = 0
        self.history_samples = 0
        self.history_mismatches = 0
        self.one_tick_transitions = 0
        self.longest_contiguous_native_run = 0
        self.motion_delta_mismatches = 0
        self.predictor_value_mismatches = 0
        self.callback_wall_ns: list[int] = []
        self.intercepts = {"self": [], "opponent": []}
        self.pad_inactive_samples = 0
        self.pad_remaining_min = math.inf
        self.pad_remaining_max = -math.inf
        self.dodge_elapsed_max = 0.0
        self.dodge_window_max = 0.0
        self.self_controller_changes = 0
        self.opponent_controller_changes = 0
        self.touch_age_min = {"self": math.inf, "opponent": math.inf}
        self.touch_age_max = {"self": -math.inf, "opponent": -math.inf}
        self.field_extrema = {
            name: {
                "minimum": math.inf,
                "maximum": -math.inf,
                "maximum_abs": 0.0,
                "nonfinite": 0,
                "values": 0,
            }
            for name in _field_slices()
        }

    def observe_output(self, observation: np.ndarray) -> None:
        finite = np.isfinite(observation)
        self.nonfinite_observation_values += int(np.count_nonzero(~finite))
        slices = _field_slices()
        for name, region in slices.items():
            values = observation[region]
            stats = self.field_extrema[name]
            stats["values"] += int(values.size)
            stats["nonfinite"] += int(np.count_nonzero(~np.isfinite(values)))
            finite_values = values[np.isfinite(values)]
            if finite_values.size:
                stats["minimum"] = min(stats["minimum"], float(np.min(finite_values)))
                stats["maximum"] = max(stats["maximum"], float(np.max(finite_values)))
                stats["maximum_abs"] = max(
                    stats["maximum_abs"], float(np.max(np.abs(finite_values)))
                )

    def finalize_extrema(self) -> dict[str, Any]:
        result = {}
        for name, source in self.field_extrema.items():
            result[name] = {
                key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                for key, value in source.items()
            }
        return result


def _phase(snapshot: Mapping[str, Any]) -> str:
    value = (snapshot.get("match") or {}).get("phase")
    if isinstance(value, Mapping):
        return str(value.get("name", "Inactive"))
    return str(value).split(".")[-1]


def _verify_history(
    audit: CorpusAudit,
    builder: RivalObsV1Builder,
    expected_self: deque[np.ndarray],
    expected_opponent: deque[np.ndarray],
) -> None:
    audit.history_samples += 1
    if not np.array_equal(np.stack(builder.self_history), np.stack(expected_self)):
        audit.history_mismatches += 1
    if not np.array_equal(np.stack(builder.opponent_history), np.stack(expected_opponent)):
        audit.history_mismatches += 1


def _process_source(
    source: Mapping[str, Any],
    audit: CorpusAudit,
    *,
    maximum_samples: int,
    breadth_samples: int,
) -> dict[str, Any]:
    path = Path(source["path"])
    if not path.is_file():
        raise FileNotFoundError(f"Missing Gate 3 corpus source: {path}")
    actual_hash = _sha256(path)
    expected_hash = source.get("expected_sha256")
    source_hash_matches = expected_hash in (None, "") or actual_hash == expected_hash
    adapter = RLBotCanonicalAdapterV1()
    builder = RivalObsV1Builder(prediction_refresh_ticks=4)
    independent = RivalObsV1Builder(prediction_refresh_ticks=4)
    zero = np.zeros(8, dtype=np.float32)
    expected_self: deque[np.ndarray] = deque(
        [zero.copy() for _ in range(HISTORY_TICKS)], maxlen=HISTORY_TICKS
    )
    expected_opponent: deque[np.ndarray] = deque(
        [zero.copy() for _ in range(HISTORY_TICKS)], maxlen=HISTORY_TICKS
    )
    previous_frame: int | None = None
    previous_self: np.ndarray | None = None
    previous_opponent: np.ndarray | None = None
    previous_motion: np.ndarray | None = None
    contiguous_run = 0
    records = 0
    source_records_seen = 0
    expected_source_records = int(
        source.get(
            "expected_policy_records"
            if source["kind"] == "historical_natural_breadth"
            else "expected_records",
            0,
        )
    )
    if maximum_samples:
        target_samples = min(maximum_samples, expected_source_records)
    elif source["kind"] == "historical_natural_breadth":
        target_samples = min(breadth_samples, expected_source_records)
    else:
        target_samples = expected_source_records
    if target_samples <= 0:
        selected_indices: set[int] | None = None
    elif target_samples >= expected_source_records:
        selected_indices = None
    elif target_samples == 1:
        selected_indices = {expected_source_records // 2}
    else:
        selected_indices = {
            round(index * (expected_source_records - 1) / (target_samples - 1))
            for index in range(target_samples)
        }
    coverage_witness_indices: dict[str, int] = {}
    if (
        source["kind"] == "historical_natural_breadth"
        and selected_indices is not None
    ):
        for source_index, (record, _raw) in enumerate(_records(source)):
            snapshot = record.get("packet")
            if not isinstance(snapshot, Mapping):
                continue
            for category in packet_coverage(snapshot):
                coverage_witness_indices.setdefault(category, source_index)
        selected_indices.update(coverage_witness_indices.values())
    started = time.perf_counter()

    for source_index, (record, _raw) in enumerate(_records(source)):
        source_records_seen += 1
        if selected_indices is not None and source_index not in selected_indices:
            continue
        snapshot = record.get("packet")
        if not isinstance(snapshot, Mapping):
            raise ValueError(f"Corpus record in {path} has no packet snapshot")
        for goal in snapshot.get("goals", []):
            metadata = {
                "team_num": int(goal.get("team_num", -1)),
                "location": goal.get("location"),
                "direction": goal.get("direction"),
                "width": goal.get("width"),
                "height": goal.get("height"),
            }
            audit.field_info_goal_metadata[
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ] += 1
        packet, field_info, self_index = snapshot_to_rlbot_sources(snapshot)
        frame = int(packet.match_info.frame_num)
        delta = None if previous_frame is None else frame - previous_frame
        if delta is not None:
            audit.frame_deltas[delta] += 1

        native = source["kind"] == "native_tick_history"
        if native and delta not in (None, 1):
            adapter.reset()
            builder.reset()
            independent.reset()
            expected_self = deque(
                [zero.copy() for _ in range(HISTORY_TICKS)], maxlen=HISTORY_TICKS
            )
            expected_opponent = deque(
                [zero.copy() for _ in range(HISTORY_TICKS)], maxlen=HISTORY_TICKS
            )
            previous_motion = None
            contiguous_run = 0
        canonical = adapter.adapt(packet, self_index, field_info)
        audit_canonical_against_snapshot(canonical, snapshot, audit.source_audits)

        runtime_before = builder.export_runtime_state()
        serialized = json.dumps(
            {
                "canonical": canonical.to_payload(),
                "observation_runtime": runtime_before,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        restored_payload = json.loads(serialized)
        restored = RivalCanonicalStateV1.from_payload(restored_payload["canonical"])
        observation = builder.build(canonical)
        independent.load_runtime_state(restored_payload["observation_runtime"])
        reproduced = independent.build(restored)
        if np.array_equal(observation, reproduced):
            audit.bit_identical += 1
        else:
            audit.bit_mismatches += 1
            audit.maximum_replay_abs_error = max(
                audit.maximum_replay_abs_error,
                float(np.max(np.abs(observation - reproduced))),
            )
        if not np.array_equal(builder.prediction_positions, independent.prediction_positions):
            audit.predictor_value_mismatches += 1
        audit.prediction_ages[int(builder.last_timings["prediction_age_ticks"])] += 1
        audit.observe_output(observation)

        audit.coverage.update(packet_coverage(snapshot))
        audit.phases[_phase(snapshot)] += 1
        audit.samples += 1
        records += 1
        if native:
            audit.native_samples += 1
            contiguous_run += 1
            audit.longest_contiguous_native_run = max(
                audit.longest_contiguous_native_run, contiguous_run
            )
            if delta == 1:
                audit.one_tick_transitions += 1
            expected_self.append(canonical.self_car.latest_controller.copy())
            expected_opponent.append(canonical.opponent_car.latest_controller.copy())
            _verify_history(audit, builder, expected_self, expected_opponent)
            motion = builder._motion(canonical)
            expected_delta = (
                np.zeros((3, 6), dtype=np.float32)
                if previous_motion is None
                else motion - previous_motion
            )
            if not np.array_equal(builder.motion_delta, expected_delta):
                audit.motion_delta_mismatches += 1
            previous_motion = motion
            callback_ns = record.get("callback_wall_ns")
            if isinstance(callback_ns, int) and callback_ns >= 0:
                audit.callback_wall_ns.append(callback_ns)
        else:
            audit.breadth_samples += 1

        self_input = canonical.self_car.latest_controller
        opponent_input = canonical.opponent_car.latest_controller
        if previous_self is not None and not np.array_equal(self_input, previous_self):
            audit.self_controller_changes += 1
        if previous_opponent is not None and not np.array_equal(
            opponent_input, previous_opponent
        ):
            audit.opponent_controller_changes += 1
        previous_self = self_input.copy()
        previous_opponent = opponent_input.copy()
        previous_frame = frame

        inactive = canonical.pad_time_until_active[canonical.pad_active < 0.5]
        audit.pad_inactive_samples += int(inactive.size)
        if inactive.size:
            audit.pad_remaining_min = min(
                audit.pad_remaining_min, float(np.min(inactive))
            )
            audit.pad_remaining_max = max(
                audit.pad_remaining_max, float(np.max(inactive))
            )
        audit.dodge_elapsed_max = max(
            audit.dodge_elapsed_max,
            canonical.self_car.dodge_elapsed,
            canonical.opponent_car.dodge_elapsed,
        )
        audit.dodge_window_max = max(
            audit.dodge_window_max,
            canonical.self_car.dodge_window_remaining,
            canonical.opponent_car.dodge_window_remaining,
        )
        for label, age in (
            ("self", canonical.self_touch_age),
            ("opponent", canonical.opponent_touch_age),
        ):
            audit.touch_age_min[label] = min(audit.touch_age_min[label], age)
            audit.touch_age_max[label] = max(audit.touch_age_max[label], age)
        audit.intercepts["self"].append(
            deterministic_intercept_time(canonical.self_car, canonical.ball)
        )
        audit.intercepts["opponent"].append(
            deterministic_intercept_time(canonical.opponent_car, canonical.ball)
        )

    return {
        "kind": source["kind"],
        "session_id": source["session_id"],
        "path": _relative(path),
        "size_bytes": path.stat().st_size,
        "sha256": actual_hash,
        "expected_sha256": expected_hash,
        "source_hash_matches": source_hash_matches,
        "records_audited": records,
        "source_records_seen": source_records_seen,
        "expected_records": expected_source_records,
        "all_expected_records_present": source_records_seen == expected_source_records,
        "sample_selection": (
            "all records"
            if selected_indices is None
            else (
                f"{target_samples} uniform indices plus first natural coverage "
                f"witnesses ({len(selected_indices)} unique total)"
            )
        ),
        "coverage_witness_indices": dict(sorted(coverage_witness_indices.items())),
        "rival_side": source.get("rival_side"),
        "opponent": source.get("opponent"),
        "elapsed_wall_seconds": time.perf_counter() - started,
    }


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "minimum": None, "p50": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": len(values),
        "minimum": int(np.min(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": int(np.max(array)),
    }


def _finite_range(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"samples": 0, "minimum": None, "maximum": None}
    return {"samples": len(values), "minimum": min(values), "maximum": max(values)}


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    observation_schema = observation_schema_manifest()
    sources = _broad_sources()
    native = _native_source(args.native_capture_report)
    if native is not None:
        sources.append(native)
    audit = CorpusAudit()
    source_records = []
    for source in sources:
        source_records.append(
            _process_source(
                source,
                audit,
                maximum_samples=args.max_samples_per_source,
                breadth_samples=args.breadth_samples_per_source,
            )
        )

    source_records_complete = all(
        record["source_hash_matches"]
        and record["all_expected_records_present"]
        for record in source_records
    )
    coverage = {
        name: {
            "samples": audit.coverage[name],
            "observed": audit.coverage[name] > 0,
            "required": name in REQUIRED_COVERAGE,
        }
        for name in (*REQUIRED_COVERAGE, *NATURALLY_AVAILABLE_COVERAGE)
    }
    source_audits = {
        name: value.to_record() for name, value in audit.source_audits.items()
    }
    field_info_goals = [
        {**json.loads(serialized), "samples": samples}
        for serialized, samples in sorted(audit.field_info_goal_metadata.items())
    ]
    callback = _distribution(audit.callback_wall_ns)
    callback["p95_milliseconds"] = (
        None if callback["p95"] is None else callback["p95"] / 1e6
    )
    checks = {
        "four_complete_balanced_m08_breadth_sources": (
            len([value for value in source_records if value["kind"] == "historical_natural_breadth"])
            == 4
            and {value["rival_side"] for value in source_records if value["kind"] == "historical_natural_breadth"}
            == {"blue", "orange"}
        ),
        "broad_replay_sample_has_several_thousand_ticks": audit.breadth_samples >= 3_000,
        "raw_sources_hash_and_record_counts_match": source_records_complete,
        "all_required_natural_categories_observed": all(
            coverage[name]["observed"] for name in REQUIRED_COVERAGE
        ),
        "canonical_serialization_replay_bit_identical": (
            audit.bit_mismatches == 0 and audit.bit_identical == audit.samples
        ),
        "source_adapter_field_audits_pass": all(
            value["passed"] for value in source_audits.values()
        ),
        "native_field_info_goals_audited": source_audits["goals"]["comparisons"] > 0,
        "native_field_info_goal_metadata_captured": len(field_info_goals) == 2,
        "observations_have_no_nonfinite_values": audit.nonfinite_observation_values == 0,
        "native_corpus_version_exact": (
            native is not None
            and json.loads(Path(native["capture_report_path"]).read_text(encoding="utf-8"))
            .get("native_corpus", {})
            .get("version")
            == NATIVE_CORPUS_VERSION
        ),
        "native_history_corpus_has_at_least_4096_ticks": audit.native_samples >= 4096,
        "native_corpus_has_at_least_4000_one_tick_transitions": audit.one_tick_transitions >= 4000,
        "native_corpus_has_long_contiguous_run": audit.longest_contiguous_native_run >= 4000,
        "self_and_opponent_histories_exact": audit.history_mismatches == 0,
        "one_tick_motion_deltas_exact": audit.motion_delta_mismatches == 0,
        "prediction_cache_values_replay_exact": audit.predictor_value_mismatches == 0,
        "both_controller_streams_change_naturally": (
            audit.self_controller_changes > 0 and audit.opponent_controller_changes > 0
        ),
    }
    if args.max_samples_per_source:
        status = "discovery_only"
        return_code = 0
    else:
        status = "passed" if all(checks.values()) else "failed"
        return_code = 0 if status == "passed" else 1
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 3,
        "gate_name": "observation_parity_corpus",
        "observation_schema": {
            "version": observation_schema["observation_version"],
            "float_count": observation_schema["float_count"],
            "schema_sha256": observation_schema["schema_sha256"],
            "builder_source_sha256": observation_schema["builder_source_sha256"],
            "canonical_source_sha256": observation_schema["canonical_source_sha256"],
        },
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator_revision": {
            "revision": 2,
            "initial_dry_run_threshold": 20_000,
            "corrected_several_thousand_threshold": 3_000,
            "reason": (
                "Revision 1 accidentally retained an all-record aspirational threshold after "
                "the prospective uniform-plus-rare-witness sampling contract was selected. "
                "No observed metric or numerical acceptance condition changed."
            ),
        },
        "checks": checks,
        "corpus": {
            "total_samples": audit.samples,
            "historical_breadth_samples": audit.breadth_samples,
            "historical_source_records_verified": sum(
                value["source_records_seen"]
                for value in source_records
                if value["kind"] == "historical_natural_breadth"
            ),
            "native_tick_samples": audit.native_samples,
            "sources": source_records,
            "coverage": coverage,
            "match_phase_counts": dict(sorted(audit.phases.items())),
            "frame_delta_counts": {
                str(key): value for key, value in sorted(audit.frame_deltas.items())
            },
        },
        "serialization_parity": {
            "bit_identical_samples": audit.bit_identical,
            "mismatched_samples": audit.bit_mismatches,
            "maximum_abs_error": audit.maximum_replay_abs_error,
            "serialized_payload_includes": [
                "RivalCanonicalStateV1",
                "RivalObsV1Builder runtime/history/prediction state before build",
            ],
        },
        "source_adapter_audits": source_audits,
        "special_audits": {
            "field_info_goals": {
                "source_audit": source_audits["goals"],
                "captured_runtime_metadata": field_info_goals,
                "physical_opening_contract": {
                    "centers": [[0.0, -5120.0, 321.3875], [0.0, 5120.0, 321.3875]],
                    "width": 1785.51,
                    "height": 642.775,
                    "authority": "https://wiki.rlbot.org/v5/botmaking/useful-game-values/",
                },
                "runtime_semantics": (
                    "The observed RLBot v5 beta FieldInfo dimensions describe a larger goal/scoring "
                    "volume than the documented physical opening. Rival records and audits that "
                    "metadata, but canonical actor goal centers/posts use the shared documented "
                    "physical standard-Soccar opening in both RocketSim and RLBot."
                ),
            },
            "boost_pad_timer_conversion": {
                "inactive_pad_samples": audit.pad_inactive_samples,
                "time_until_active_minimum": (
                    None if not math.isfinite(audit.pad_remaining_min) else audit.pad_remaining_min
                ),
                "time_until_active_maximum": (
                    None if not math.isfinite(audit.pad_remaining_max) else audit.pad_remaining_max
                ),
                "source_audit": source_audits["boost_pads"],
                "rlbot_source_semantics": (
                    "active -> 0; inactive -> max(0, respawn_seconds - elapsed_timer), "
                    "using 4 seconds small and 10 seconds full"
                ),
            },
            "jump_air_dodge_mapping": {
                "self_source_audit": source_audits["self_air_dodge"],
                "opponent_source_audit": source_audits["opponent_air_dodge"],
                "maximum_dodge_elapsed_seconds": audit.dodge_elapsed_max,
                "maximum_dodge_window_remaining_seconds": audit.dodge_window_max,
                "coverage": {
                    key: coverage[key]
                    for key in (
                        "first_jump_hold",
                        "first_jump_release",
                        "double_jump",
                        "directional_dodge",
                        "flip_cancel",
                        "airborne_reset_resource",
                    )
                },
            },
            "controller_histories": {
                "native_samples": audit.history_samples,
                "one_tick_transitions": audit.one_tick_transitions,
                "longest_contiguous_native_run": audit.longest_contiguous_native_run,
                "history_mismatches": audit.history_mismatches,
                "one_tick_motion_delta_mismatches": audit.motion_delta_mismatches,
                "self_controller_changes": audit.self_controller_changes,
                "opponent_controller_changes": audit.opponent_controller_changes,
            },
            "touch": {
                "source_audit": source_audits["touch"],
                "self_age_seconds": _finite_range(
                    [audit.touch_age_min["self"], audit.touch_age_max["self"]]
                ),
                "opponent_age_seconds": _finite_range(
                    [audit.touch_age_min["opponent"], audit.touch_age_max["opponent"]]
                ),
                "ball_contact_samples": audit.coverage["ball_contact"],
            },
            "match": {
                "source_audit": source_audits["match"],
                "phase_counts": dict(sorted(audit.phases.items())),
                "late_clock_samples": audit.coverage["late_clock"],
                "overtime_samples": audit.coverage["overtime"],
            },
            "surface_geometry": {
                "wall_contact_samples": audit.coverage["wall_contact"],
                "ceiling_contact_samples": audit.coverage["ceiling_contact"],
                "awkward_recovery_samples": audit.coverage["awkward_recovery"],
                "self_distance_extrema": audit.field_extrema["self.surface_distances"],
                "opponent_distance_extrema": audit.field_extrema[
                    "opponent.surface_distances"
                ],
            },
            "prediction_cache": {
                "refresh_ticks": 4,
                "age_distribution": {
                    str(key): value for key, value in sorted(audit.prediction_ages.items())
                },
                "replay_value_mismatches": audit.predictor_value_mismatches,
            },
            "intercept_proxy": {
                "self_seconds": _finite_range(audit.intercepts["self"]),
                "opponent_seconds": _finite_range(audit.intercepts["opponent"]),
                "implementation": "shared deterministic_intercept_time pure function",
            },
            "normalization": {
                "nonfinite_values": audit.nonfinite_observation_values,
                "field_extrema": audit.finalize_extrema(),
            },
            "native_callback_latency": callback,
        },
        "commands": {
            "capture": (
                ".venv/Scripts/python.exe "
                "training/scripts/run_m09_native_corpus_capture.py"
            ),
            "gate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_observation_parity_gate.py"
            ),
            "unit_tests": (
                "training/.venv/Scripts/python.exe -m pytest "
                "training/tests/test_v9_rlbot_corpus.py -q"
            ),
        },
        "interpretation": (
            "Technical parity only. Scores and match outcomes are excluded from every Gate 3 "
            "decision. Historical eight-frame telemetry supplies state breadth; only the "
            "separate native corpus is allowed to certify one-tick histories and deltas."
        ),
        "sampling_contract": {
            "historical": (
                "At least 750 uniformly spaced records from each of four complete natural "
                "matches, plus the first naturally observed witness for every coverage "
                "category. All source records are streamed for count verification and the "
                "entire source file hash is verified."
            ),
            "native": "Every record in the bounded native-tick capture is audited.",
            "several_thousand_threshold": 3000,
            "reason": (
                "The handoff prefers several thousand sampled ticks. Replaying every one of "
                "the 21,859 historical records would add repeated predictor work without "
                "increasing the predetermined state-coverage contract."
            ),
        },
    }
    return report, return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-capture-report", type=Path, default=DEFAULT_NATIVE_REPORT)
    parser.add_argument(
        "--max-samples-per-source",
        type=int,
        default=0,
        help="Nonzero bounded discovery run; never emits passing gate evidence.",
    )
    parser.add_argument(
        "--breadth-samples-per-source",
        type=int,
        default=750,
        help="Uniform samples from each complete historical natural match.",
    )
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    parser.add_argument("--no-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_samples_per_source < 0:
        raise ValueError("--max-samples-per-source cannot be negative")
    if args.breadth_samples_per_source < 750:
        raise ValueError("Gate 3 requires at least 750 breadth samples per source")
    report, return_code = run(args)
    if not args.no_write:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
