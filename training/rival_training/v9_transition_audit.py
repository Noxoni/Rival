"""Native-120-Hz RocketSim/RLBot short-horizon transfer audit for Rival v9."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.sim import RocketSimEngine

from .v9_actions import CONTROLLER_FIELDS
from .v9_canonical import (
    AIR_STATES,
    RLBotCanonicalAdapterV1,
    RivalCanonicalStateV1,
    _pad_mapping,
)
from .v9_rlbot_corpus import snapshot_to_rlbot_sources
from .v9_soccar_geometry import ROCKETSIM_PAD_ORB_POSITIONS


HORIZONS = (1, 2, 4, 8, 16, 32)
MAXIMUM_WINDOWS = 64


@dataclass(frozen=True)
class NativeFrameV1:
    record: Mapping[str, Any]
    canonical: RivalCanonicalStateV1

    @property
    def frame_num(self) -> int:
        return int(self.record["frame_num"])

    @property
    def self_index(self) -> int:
        return int(self.record["packet"]["self_index"])

    @property
    def opponent_index(self) -> int:
        return int(self.record["packet"]["opponent_indices"][0])


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _controller(value: Mapping[str, Any] | None) -> np.ndarray:
    source = value or {}
    return np.asarray(
        [float(source.get(field, 0.0)) for field in CONTROLLER_FIELDS],
        dtype=np.float32,
    )


def _physics(source) -> PhysicsObject:
    physics = PhysicsObject()
    physics.position = np.asarray(source.position, dtype=np.float32).copy()
    physics.linear_velocity = np.asarray(
        source.linear_velocity, dtype=np.float32
    ).copy()
    physics.angular_velocity = np.asarray(
        source.angular_velocity, dtype=np.float32
    ).copy()
    physics.rotation_mtx = np.asarray(source.rotation_mtx, dtype=np.float32).copy()
    return physics


def _car(source, team_num: int) -> Car:
    car = Car()
    car.team_num = int(team_num)
    car.hitbox_type = 0  # Octane/Fennec family used by the capture participants.
    car.ball_touches = 0
    car.bump_victim_id = None
    car.demo_respawn_timer = float(source.demo_time_remaining)
    car.on_ground = bool(source.surface_contact)
    car.supersonic_time = 1.0 if source.supersonic else 0.0
    car.boost_amount = float(source.boost)
    car.boost_active_time = 1.0 / 120.0 if source.boosting else 0.0
    car.handbrake = float(source.handbrake)
    air_state = AIR_STATES[int(source.air_state)]
    car.is_jumping = air_state == "Jumping"
    car.has_jumped = bool(source.has_jumped)
    car.is_holding_jump = bool(source.jump_held)
    car.jump_time = float(source.jump_hold_elapsed)
    car.has_flipped = bool(source.has_dodged)
    car.has_double_jumped = bool(source.has_double_jumped)
    if source.surface_contact or not source.has_jumped or car.is_jumping:
        car.air_time_since_jump = 0.0
    elif source.can_dodge:
        car.air_time_since_jump = max(
            0.0, 1.25 - float(source.dodge_window_remaining)
        )
    else:
        car.air_time_since_jump = max(1.25, float(source.air_time))
    car.flip_time = float(source.dodge_elapsed)
    direction = np.asarray(source.dodge_direction, dtype=np.float32)
    car.flip_torque = np.asarray(
        [-direction[1], direction[0], 0.0], dtype=np.float32
    )
    car.is_autoflipping = False
    car.autoflip_timer = 0.0
    car.autoflip_direction = 0.0
    car.physics = _physics(source.physics)
    car._inverted_physics = None
    return car


def _game_state(frame: NativeFrameV1, engine: RocketSimEngine) -> GameState:
    canonical = frame.canonical
    packet_players = frame.record["packet"]["players"]
    teams = {
        int(player["packet_index"]): int(player["team"])
        for player in packet_players
    }
    state = engine.create_base_state()
    state.tick_count = canonical.tick_index
    state.goal_scored = False
    state.config.gravity = abs(float(canonical.gravity_z)) / 650.0
    state.cars = {
        frame.self_index: _car(canonical.self_car, teams[frame.self_index]),
        frame.opponent_index: _car(
            canonical.opponent_car, teams[frame.opponent_index]
        ),
    }
    state.ball = _physics(canonical.ball)
    state._inverted_ball = None
    canonical_to_source = _pad_mapping(ROCKETSIM_PAD_ORB_POSITIONS, False)
    state.boost_pad_timers = np.zeros(34, dtype=np.float32)
    state.boost_pad_timers[canonical_to_source] = canonical.pad_time_until_active
    state._inverted_boost_pad_timers = None
    return state


def _touch_time(frame: NativeFrameV1, packet_index: int) -> float | None:
    touch = frame.record["packet"]["players"][packet_index].get("latest_touch")
    value = None if touch is None else touch.get("game_seconds")
    return float(value) if isinstance(value, (float, int)) else None


def _new_touch(start: NativeFrameV1, end: NativeFrameV1, packet_index: int) -> bool:
    first = _touch_time(start, packet_index)
    second = _touch_time(end, packet_index)
    return second is not None and (first is None or second > first + 1e-6)


def _scores(frame: NativeFrameV1) -> tuple[int, ...]:
    return tuple(
        int(item["score"])
        for item in frame.record["packet"]["match"].get("scores", [])
    )


def load_native_frames(path: Path) -> list[NativeFrameV1]:
    adapter = RLBotCanonicalAdapterV1()
    frames: list[NativeFrameV1] = []
    previous_frame: int | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") != "rival_v9_native_packet":
                continue
            frame_num = int(record["frame_num"])
            if previous_frame is not None and frame_num - previous_frame != 1:
                adapter.reset()
            packet, field_info, self_index = snapshot_to_rlbot_sources(record["packet"])
            canonical = adapter.adapt(packet, self_index, field_info)
            if int(record["packet"]["players"][self_index]["team"]) != 0:
                raise ValueError("Gate 6 native corpus must retain Rival on blue/world frame")
            frames.append(NativeFrameV1(record=record, canonical=canonical))
            previous_frame = frame_num
    return frames


def candidate_windows(frames: Iterable[NativeFrameV1]) -> list[list[NativeFrameV1]]:
    by_frame = {frame.frame_num: frame for frame in frames}
    result: list[list[NativeFrameV1]] = []
    for start_frame, start in sorted(by_frame.items()):
        window = [by_frame.get(start_frame + offset) for offset in range(33)]
        if any(item is None for item in window):
            continue
        concrete = [item for item in window if item is not None]
        if any(_scores(item) != _scores(start) for item in concrete):
            continue
        phases = {
            str(item.record["packet"]["match"]["phase"].get("name"))
            for item in concrete
        }
        if not phases.issubset({"Kickoff", "Active"}):
            continue
        result.append(concrete)
    return result


def _category(window: list[NativeFrameV1]) -> str:
    start = window[0]
    self_position = start.canonical.self_car.physics.position
    surface = bool(start.canonical.self_car.surface_contact)
    near_wall = abs(float(self_position[0])) > 3400 or abs(float(self_position[1])) > 4600
    actual_contact = any(
        _new_touch(start, window[-1], index)
        for index in (start.self_index, start.opponent_index)
    )
    posture = "surface" if surface else ("wall_air" if near_wall else "open_air")
    return f"{posture}_{'contact' if actual_contact else 'free'}"


def select_windows(
    candidates: list[list[NativeFrameV1]], maximum: int = MAXIMUM_WINDOWS
) -> list[list[NativeFrameV1]]:
    grouped: dict[str, list[list[NativeFrameV1]]] = defaultdict(list)
    for window in candidates:
        grouped[_category(window)].append(window)
    if not grouped:
        return []
    selected: list[list[NativeFrameV1]] = []
    per_category = max(1, math.ceil(maximum / len(grouped)))
    for category in sorted(grouped):
        values = grouped[category]
        indices = np.linspace(
            0, len(values) - 1, min(per_category, len(values)), dtype=int
        )
        selected.extend(values[int(index)] for index in indices)
    selected.sort(key=lambda window: window[0].frame_num)
    if len(selected) > maximum:
        indices = np.linspace(0, len(selected) - 1, maximum, dtype=int)
        selected = [selected[int(index)] for index in indices]
    return selected


def _orientation_error_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _physics_error(simulated, live) -> dict[str, float]:
    return {
        "position_uu": float(np.linalg.norm(simulated.position - live.position)),
        "linear_velocity_uu_per_s": float(
            np.linalg.norm(simulated.linear_velocity - live.linear_velocity)
        ),
        "angular_velocity_rad_per_s": float(
            np.linalg.norm(simulated.angular_velocity - live.angular_velocity)
        ),
        "orientation_degrees": _orientation_error_degrees(
            simulated.rotation_mtx, live.rotation_mtx
        ),
    }


def _ball_error(simulated, live) -> dict[str, float]:
    return {
        "position_uu": float(np.linalg.norm(simulated.position - live.position)),
        "linear_velocity_uu_per_s": float(
            np.linalg.norm(simulated.linear_velocity - live.linear_velocity)
        ),
        "angular_velocity_rad_per_s": float(
            np.linalg.norm(simulated.angular_velocity - live.angular_velocity)
        ),
    }


def _applied_controls(frame: NativeFrameV1) -> dict[int, np.ndarray]:
    packet = frame.record["packet"]
    return {
        int(player["packet_index"]): _controller(player.get("last_input"))[None, :]
        for player in packet["players"]
    }


def run_window(engine: RocketSimEngine, window: list[NativeFrameV1]) -> dict[str, Any]:
    start = window[0]
    simulated = engine.set_state(_game_state(start, engine), {})
    initialization = {
        "self": _physics_error(
            simulated.cars[start.self_index].physics,
            start.canonical.self_car.physics,
        ),
        "opponent": _physics_error(
            simulated.cars[start.opponent_index].physics,
            start.canonical.opponent_car.physics,
        ),
        "ball": _ball_error(simulated.ball, start.canonical.ball),
    }
    simulated_touches = {start.self_index: False, start.opponent_index: False}
    horizons: dict[str, Any] = {}
    for tick in range(1, 33):
        # Packet at t+1 exposes the physical rows applied over transition t->t+1.
        simulated = engine.step(_applied_controls(window[tick]), {})
        for index in simulated_touches:
            simulated_touches[index] |= bool(simulated.cars[index].ball_touches)
        if tick not in HORIZONS:
            continue
        live = window[tick]
        actual_touches = {
            index: _new_touch(start, live, index)
            for index in (start.self_index, start.opponent_index)
        }
        demo_discontinuity = any(
            float(player.get("demolished_timeout") or -1.0) > 0.0
            for frame in window[: tick + 1]
            for player in frame.record["packet"]["players"]
        )
        horizons[str(tick)] = {
            "self": _physics_error(
                simulated.cars[start.self_index].physics,
                live.canonical.self_car.physics,
            ),
            "opponent": _physics_error(
                simulated.cars[start.opponent_index].physics,
                live.canonical.opponent_car.physics,
            ),
            "ball": _ball_error(simulated.ball, live.canonical.ball),
            "contact": {
                "actual_self": actual_touches[start.self_index],
                "actual_opponent": actual_touches[start.opponent_index],
                "simulated_self": simulated_touches[start.self_index],
                "simulated_opponent": simulated_touches[start.opponent_index],
                "any_actual_or_simulated": any(actual_touches.values())
                or any(simulated_touches.values()),
                "actual_any": any(actual_touches.values()),
                "demo_discontinuity": demo_discontinuity,
            },
        }
    return {
        "start_frame": start.frame_num,
        "start_seconds_elapsed": start.canonical.seconds_elapsed,
        "category": _category(window),
        "initialization": initialization,
        "horizons": horizons,
    }


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "p50": None,
            "p95": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def aggregate(
    samples: list[dict[str, Any]], *, mode: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS:
        records = [sample["horizons"][str(horizon)] for sample in samples]
        if mode == "contact_free":
            records = [
                record
                for record in records
                if not record["contact"]["any_actual_or_simulated"]
                and not record["contact"]["demo_discontinuity"]
            ]
        elif mode == "actual_contact":
            records = [record for record in records if record["contact"]["actual_any"]]
        elif mode != "all":
            raise ValueError(f"Unknown aggregate mode {mode!r}")
        metrics: dict[str, Any] = {}
        for entity in ("self", "opponent", "ball"):
            keys = records[0][entity].keys() if records else ()
            metrics[entity] = {
                key: _distribution([float(record[entity][key]) for record in records])
                for key in keys
            }
        output[str(horizon)] = {"windows": len(records), **metrics}
    return output


def build_gate06_report(
    raw_path: Path,
    *,
    expected_sha256: str,
    expected_records: int,
    maximum_windows: int = MAXIMUM_WINDOWS,
) -> dict[str, Any]:
    frames = load_native_frames(raw_path)
    candidates = candidate_windows(frames)
    selected = select_windows(candidates, maximum_windows)
    if not selected:
        raise RuntimeError("No native contiguous 32-tick windows are available")
    engine = RocketSimEngine(rlbot_delay=False)
    samples = [run_window(engine, window) for window in selected]
    engine.close()
    contact_free = aggregate(samples, mode="contact_free")
    actual_contact = aggregate(samples, mode="actual_contact")
    all_windows = aggregate(samples, mode="all")
    early = contact_free["4"]
    thresholds = {
        "self_position_p95_uu": 5.0,
        "self_linear_velocity_p95_uu_per_s": 100.0,
        "self_orientation_p95_degrees": 7.5,
        "ball_position_p95_uu": 2.0,
    }
    measured = {
        "self_position_p95_uu": early["self"]["position_uu"]["p95"],
        "self_linear_velocity_p95_uu_per_s": early["self"][
            "linear_velocity_uu_per_s"
        ]["p95"],
        "self_orientation_p95_degrees": early["self"]["orientation_degrees"][
            "p95"
        ],
        "ball_position_p95_uu": early["ball"]["position_uu"]["p95"],
    }
    threshold_pass = all(
        measured[name] is not None and float(measured[name]) <= limit
        for name, limit in thresholds.items()
    )
    checks = {
        "raw_hash_matches_gate3": _sha256(raw_path) == expected_sha256,
        "raw_record_count_matches_gate3": len(frames) == expected_records,
        "at_least_32_selected_windows": len(samples) >= 32,
        "contact_and_contact_free_windows_reported": (
            actual_contact["32"]["windows"] > 0
            and contact_free["32"]["windows"] > 0
        ),
        "at_least_20_contact_free_four_tick_windows": (
            contact_free["4"]["windows"] >= 20
        ),
        "contact_free_four_tick_materiality_thresholds_pass": threshold_pass,
    }
    status = "passed" if all(checks.values()) else "failed"
    return {
        "schema_version": 1,
        "milestone": 9,
        "gate": 6,
        "gate_name": "short_horizon_physics_transfer_audit",
        "status": status,
        "checks": checks,
        "source": {
            "path": raw_path.relative_to(raw_path.parents[3]).as_posix(),
            "sha256": _sha256(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "records": len(frames),
        },
        "contract": {
            "horizons_physics_ticks": list(HORIZONS),
            "physics_hz": 120,
            "engine": "RocketSimEngine(rlbot_delay=False)",
            "controller_replay": (
                "For transition packet t to t+1, apply both cars' packet t+1 "
                "PlayerInfo.last_input rows before one RocketSim physics tick."
            ),
            "state_initialization": (
                "Exact observable kinematics, boost, standard pad cooldowns and closest "
                "canonical jump/dodge state; Octane/Fennec hitbox family."
            ),
        },
        "selection": {
            "candidate_windows": len(candidates),
            "maximum_windows": maximum_windows,
            "selected_windows": len(samples),
            "category_counts": dict(
                sorted(
                    {
                        category: sum(sample["category"] == category for sample in samples)
                        for category in {sample["category"] for sample in samples}
                    }.items()
                )
            ),
        },
        "materiality_gate_at_four_ticks": {
            "thresholds": thresholds,
            "measured": measured,
            "passed": threshold_pass,
            "scope": "contact-free and demo-free windows only",
        },
        "contact_free_primary": contact_free,
        "actual_contact_diagnostic": actual_contact,
        "all_windows_diagnostic": all_windows,
        "samples": samples,
        "interpretation": (
            "Longer-horizon/contact divergence is diagnostic. Gate status is controlled "
            "by the explicit contact-free first-four-tick thresholds, not wins or scores."
        ),
    }
