"""Bounded natural-state RocketSim/RLBot transition audit for Milestone 07."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from rlgym.rocket_league.sim import RocketSimEngine

from .checkpoint import portable_path
from .observation_audit import (
    REPOSITORY_ROOT,
    RLGYM_TO_WISP_PAD_INDICES,
    _car,
    _controller_array,
    _physics,
    _snapshot_by_packet_index,
)
from .teacher import sha256_file


RAW_ROOT = REPOSITORY_ROOT / "evidence/raw"
HORIZONS = (4, 8, 16, 32, 64)


def _rotation(record: dict[str, Any]) -> np.ndarray:
    return _physics(record).rotation_mtx


def _orientation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _vector_error(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def _touch_time(player: dict[str, Any]) -> float | None:
    value = (player.get("latest_touch") or {}).get("game_seconds")
    return float(value) if isinstance(value, (int, float)) else None


def _new_touch(start: dict[str, Any], end: dict[str, Any]) -> bool:
    first = _touch_time(start)
    second = _touch_time(end)
    return second is not None and (first is None or second > first + 1e-4)


def _game_state(record: dict[str, Any], engine: RocketSimEngine):
    packet = record["packet"]
    snapshots = _snapshot_by_packet_index(record)
    state = engine.create_base_state()
    state.tick_count = int(packet["match"]["frame_num"])
    state.ball = _physics(packet["ball"]["physics"])
    state.cars = {
        int(player["packet_index"]): _car(
            player, snapshots.get(int(player["packet_index"]))
        )
        for player in packet["players"]
    }
    wisp_timers = np.asarray(
        [float(item.get("timer") or 0.0) for item in packet["boost_pads"]],
        dtype=np.float32,
    )
    state.boost_pad_timers = np.zeros_like(wisp_timers)
    state.boost_pad_timers[RLGYM_TO_WISP_PAD_INDICES] = wisp_timers
    return state


def _load_active_decisions(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.open("r", encoding="utf-8"):
        record = json.loads(line)
        if record.get("record_type") != "rival_policy_decision":
            continue
        phase = (((record.get("packet") or {}).get("match") or {}).get("phase") or {}).get(
            "name"
        )
        if phase == "Active":
            records.append(record)
    return records


def _score_key(record: dict[str, Any]) -> tuple[int, ...]:
    scores = record["packet"]["match"].get("scores") or []
    return tuple(int(item["score"]) for item in scores)


def _candidate_windows(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    by_frame = {
        int(record["packet"]["match"]["frame_num"]): record for record in records
    }
    result = []
    required = tuple(range(0, max(HORIZONS) + 1, 4))
    for start_frame, start in sorted(by_frame.items()):
        frames = [start_frame + offset for offset in required]
        if not all(frame in by_frame for frame in frames):
            continue
        window = [by_frame[frame] for frame in frames]
        if any(_score_key(item) != _score_key(start) for item in window):
            continue
        elapsed = [float(item["packet"]["match"]["seconds_elapsed"]) for item in window]
        if any(second <= first for first, second in zip(elapsed, elapsed[1:])):
            continue
        result.append(window)
    return result


def _window_category(window: list[dict[str, Any]]) -> str:
    start = window[0]
    grounded = bool((start.get("state") or {}).get("self", {}).get("on_ground"))
    packet = start["packet"]
    self_index = int(packet["self_index"])
    opponent_index = int(packet["opponent_indices"][0])
    actual_touch = any(
        _new_touch(packet["players"][index], window[-1]["packet"]["players"][index])
        for index in (self_index, opponent_index)
    )
    return f"{'grounded' if grounded else 'airborne'}_{'contact' if actual_touch else 'free'}"


def _select_windows(
    sources: list[tuple[str, list[dict[str, Any]]]], maximum: int
) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for session_id, records in sources:
        for window in _candidate_windows(records):
            grouped[_window_category(window)].append((session_id, window))
    result = []
    categories = sorted(grouped)
    per_category = max(1, math.ceil(maximum / max(len(categories), 1)))
    for category in categories:
        values = grouped[category]
        if len(values) <= per_category:
            result.extend(values)
            continue
        indices = np.linspace(0, len(values) - 1, per_category, dtype=int)
        result.extend(values[int(index)] for index in indices)
    result.sort(
        key=lambda item: (
            item[0],
            int(item[1][0]["packet"]["match"]["frame_num"]),
        )
    )
    if len(result) > maximum:
        indices = np.linspace(0, len(result) - 1, maximum, dtype=int)
        result = [result[int(index)] for index in indices]
    return result


def _compare_car(sim_car, live_player: dict[str, Any], live_state: dict[str, Any]) -> dict[str, Any]:
    live_physics = _physics(live_player["physics"])
    return {
        "position_error_uu": _vector_error(sim_car.physics.position, live_physics.position),
        "linear_velocity_error_uu_per_s": _vector_error(
            sim_car.physics.linear_velocity, live_physics.linear_velocity
        ),
        "angular_velocity_error_rad_per_s": _vector_error(
            sim_car.physics.angular_velocity, live_physics.angular_velocity
        ),
        "orientation_error_degrees": _orientation_error_degrees(
            sim_car.physics.rotation_mtx, live_physics.rotation_mtx
        ),
        "boost_error": abs(float(sim_car.boost_amount) - float(live_player["boost"])),
        "on_ground_agreement": bool(sim_car.on_ground) == bool(live_state.get("on_ground")),
        "has_flip_or_jump_agreement": (
            bool(
                sim_car.on_ground
                or (
                    not sim_car.has_flipped
                    and not sim_car.has_double_jumped
                    and float(sim_car.air_time_since_jump) < 1.25
                )
            )
            == bool(live_state.get("has_flip_or_jump"))
        ),
    }


def _compare_ball(sim_ball, live_record: dict[str, Any]) -> dict[str, float]:
    live = _physics(live_record)
    return {
        "position_error_uu": _vector_error(sim_ball.position, live.position),
        "linear_velocity_error_uu_per_s": _vector_error(
            sim_ball.linear_velocity, live.linear_velocity
        ),
        "angular_velocity_error_rad_per_s": _vector_error(
            sim_ball.angular_velocity, live.angular_velocity
        ),
    }


def _run_window(
    engine: RocketSimEngine,
    session_id: str,
    window: list[dict[str, Any]],
) -> dict[str, Any]:
    start = window[0]
    packet = start["packet"]
    self_index = int(packet["self_index"])
    opponent_index = int(packet["opponent_indices"][0])
    sim_state = engine.set_state(_game_state(start, engine), {})
    results = {}
    predicted_touches = {self_index: False, opponent_index: False}
    for segment, record in enumerate(window[:-1]):
        packet_players = record["packet"]["players"]
        previous_self = _controller_array(packet_players[self_index]["last_input"])
        selected_self = _controller_array(record["decision"]["final_controller_action"])
        opponent = _controller_array(packet_players[opponent_index]["last_input"])
        for local_tick in range(4):
            self_controls = previous_self if local_tick == 0 else selected_self
            sim_state = engine.step(
                {
                    self_index: self_controls[None, :],
                    opponent_index: opponent[None, :],
                },
                {},
            )
            for index in predicted_touches:
                predicted_touches[index] |= bool(sim_state.cars[index].ball_touches)
        horizon = (segment + 1) * 4
        if horizon not in HORIZONS:
            continue
        live = window[segment + 1]
        live_packet = live["packet"]
        live_snapshots = _snapshot_by_packet_index(live)
        actual_touches = {
            index: _new_touch(packet["players"][index], live_packet["players"][index])
            for index in (self_index, opponent_index)
        }
        results[str(horizon)] = {
            "self": _compare_car(
                sim_state.cars[self_index],
                live_packet["players"][self_index],
                live_snapshots.get(self_index, {}),
            ),
            "opponent": _compare_car(
                sim_state.cars[opponent_index],
                live_packet["players"][opponent_index],
                live_snapshots.get(opponent_index, {}),
            ),
            "ball": _compare_ball(sim_state.ball, live_packet["ball"]["physics"]),
            "touch_occurrence": {
                "actual_self": actual_touches[self_index],
                "actual_opponent": actual_touches[opponent_index],
                "rocketsim_self": predicted_touches[self_index],
                "rocketsim_opponent": predicted_touches[opponent_index],
                "any_actual_or_simulated": any(actual_touches.values())
                or any(predicted_touches.values()),
                "exact_agreement": actual_touches == predicted_touches,
            },
        }
    return {
        "session_id": session_id,
        "start_frame": int(packet["match"]["frame_num"]),
        "start_seconds_elapsed": float(packet["match"]["seconds_elapsed"]),
        "category": _window_category(window),
        "horizons": results,
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(array.max()),
    }


def _aggregate(samples: list[dict[str, Any]], contact_free: bool) -> dict[str, Any]:
    output = {}
    for horizon in HORIZONS:
        records = []
        for sample in samples:
            record = sample["horizons"].get(str(horizon))
            if record is None:
                continue
            if contact_free and record["touch_occurrence"]["any_actual_or_simulated"]:
                continue
            records.append(record)
        metrics = {}
        for entity in ("self", "opponent", "ball"):
            keys = sorted(records[0][entity]) if records else []
            metrics[entity] = {
                key: (
                    {
                        "count": len(records),
                        "agreement_share": sum(bool(record[entity][key]) for record in records)
                        / max(len(records), 1),
                    }
                    if key.endswith("_agreement")
                    else _distribution([float(record[entity][key]) for record in records])
                )
                for key in keys
            }
        output[str(horizon)] = {
            "windows": len(records),
            **metrics,
            "touch_occurrence_exact_agreement_share": sum(
                record["touch_occurrence"]["exact_agreement"] for record in records
            )
            / max(len(records), 1),
        }
    return output


def build_transition_audit_report(
    matrix_report_path: str | Path,
    *,
    maximum_windows: int = 32,
) -> dict[str, Any]:
    matrix_path = Path(matrix_report_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    t4 = matrix["modes"]["T4"]
    sources = []
    source_records = []
    for game in t4["games"]:
        path = RAW_ROOT / game["session_id"] / "decisions.jsonl"
        records = _load_active_decisions(path)
        sources.append((game["session_id"], records))
        source_records.append(
            {
                "session_id": game["session_id"],
                "path": portable_path(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "active_decisions": len(records),
            }
        )
    selected = _select_windows(sources, maximum_windows)
    if not selected:
        raise RuntimeError("No contiguous natural T4 windows cover 64 physics ticks")
    engine = RocketSimEngine(rlbot_delay=False)
    samples = []
    excluded = []
    for session_id, window in selected:
        try:
            samples.append(_run_window(engine, session_id, window))
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            excluded.append(
                {
                    "session_id": session_id,
                    "start_frame": int(window[0]["packet"]["match"]["frame_num"]),
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
    if not samples:
        raise RuntimeError(f"All selected transition windows failed: {excluded}")
    all_aggregate = _aggregate(samples, contact_free=False)
    free_aggregate = _aggregate(samples, contact_free=True)
    early = free_aggregate["16"]
    materially_large = bool(
        (early["self"]["position_error_uu"]["p95"] or 0.0) > 100.0
        or (early["ball"]["position_error_uu"]["p95"] or 0.0) > 100.0
        or (early["self"]["linear_velocity_error_uu_per_s"]["p95"] or 0.0)
        > 250.0
        or (early["self"]["orientation_error_degrees"]["p95"] or 0.0) > 10.0
    )
    return {
        "schema_version": 1,
        "status": "completed",
        "purpose": "milestone07_short_horizon_transition_audit",
        "matrix_report": portable_path(matrix_path),
        "source_mode": "T4 rejected 20M actor, legacy-only, tick 4",
        "horizons_physics_ticks": list(HORIZONS),
        "fixed_controller_sequence_contract": {
            "rival": (
                "For each observed four-tick policy segment, replay the packet's previous "
                "applied input for tick 1 and the deterministic selected input for ticks 2-4."
            ),
            "opponent": (
                "Hold the opponent packet last_input for each observed four-tick segment; "
                "this is an approximation because opponent outputs are not directly logged."
            ),
            "engine": "RocketSimEngine(rlbot_delay=False), controls set before every physics tick",
            "state_initialization": (
                "Exact observable packet kinematics/boost plus closest RLGym jump/ground flags; Octane hitbox assumed."
            ),
        },
        "selection": {
            "maximum_windows": maximum_windows,
            "selected": len(selected),
            "completed": len(samples),
            "excluded": excluded,
            "categories": dict(
                sorted(
                    {
                        category: sum(sample["category"] == category for sample in samples)
                        for category in {sample["category"] for sample in samples}
                    }.items()
                )
            ),
            "sources": source_records,
        },
        "all_windows": all_aggregate,
        "contact_free_primary": free_aggregate,
        "materiality_rule": {
            "evaluated_at_ticks": 16,
            "thresholds": {
                "self_or_ball_position_p95_uu": 100.0,
                "self_linear_velocity_p95_uu_per_s": 250.0,
                "self_orientation_p95_degrees": 10.0,
            },
            "early_systematic_divergence_large_enough_to_matter": materially_large,
            "interpretation_scope": (
                "A bounded diagnostic threshold, not a claim of RocketSim/Rocket League bit parity."
            ),
        },
        "samples": samples,
        "production_modified_or_promoted": False,
    }


def write_transition_audit_report(
    matrix_report_path: str | Path,
    output_path: str | Path,
    *,
    maximum_windows: int = 32,
) -> dict[str, Any]:
    report = build_transition_audit_report(
        matrix_report_path, maximum_windows=maximum_windows
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
