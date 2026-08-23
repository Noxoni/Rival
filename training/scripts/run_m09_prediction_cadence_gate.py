"""Benchmark RivalObsV1 shared-predictor refresh periods at native 120 Hz."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_actions import (  # noqa: E402
    ACTION_DIM,
    ANALOG_DIM,
    button_combo_to_bits,
)
from rival_training.v9_canonical import RLBotCanonicalAdapterV1  # noqa: E402
from rival_training.v9_environment import (  # noqa: E402
    V9_ENVIRONMENT_VERSION,
    build_v9_diagnostic_env,
)
from rival_training.v9_observations import (  # noqa: E402
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    RivalObsV1Builder,
    observation_schema_manifest,
)
from rival_training.v9_rlbot_corpus import (  # noqa: E402
    snapshot_to_rlbot_sources,
)


RESULT_PATH = (
    TRAINING_ROOT / "results" / "milestone09" / "gate04_prediction_cadence.json"
)
PERIODS = (1, 2, 4)
PHYSICS_HZ = 120
AGENTS = 2
SEED = 20260904


def _percentiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _action_trace(total_ticks: int) -> np.ndarray:
    """Generate one frozen, smooth-ish exploratory controller trace per team."""

    rng = np.random.default_rng(SEED)
    actions = np.zeros((total_ticks, AGENTS, ACTION_DIM), dtype=np.float32)
    block_ticks = 12
    for start in range(0, total_ticks, block_ticks):
        stop = min(total_ticks, start + block_ticks)
        analog = rng.uniform(-1.0, 1.0, size=(AGENTS, ANALOG_DIM)).astype(np.float32)
        # Bias throttle toward motion while retaining reverse/brake coverage.
        analog[:, 0] = rng.choice(
            np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=np.float32), size=AGENTS
        )
        actions[start:stop, :, :ANALOG_DIM] = analog
        for team in range(AGENTS):
            combo = int(rng.integers(0, 8))
            actions[start:stop, team, ANALOG_DIM:] = button_combo_to_bits(combo)
    # Begin with a reproducible ordinary kickoff so the measured window has a
    # naturally moving ball instead of accidentally benchmarking a stationary
    # prediction cache. Both kickoff spawns face the ball by construction.
    kickoff_ticks = min(total_ticks, 300)
    actions[:kickoff_ticks] = 0.0
    actions[:kickoff_ticks, :, 0] = 1.0
    actions[:kickoff_ticks, :, ANALOG_DIM:] = button_combo_to_bits(2)
    return actions


def _actions_by_agent(environment, controller_rows: np.ndarray) -> dict[Any, np.ndarray]:
    return {
        agent: controller_rows[int(environment.state.cars[agent].team_num)].copy()
        for agent in environment.agents
    }


def _observations_by_team(environment, observations: dict[Any, np.ndarray]) -> np.ndarray:
    ordered = np.empty((AGENTS, OBSERVATION_SIZE), dtype=np.float32)
    for agent, observation in observations.items():
        team = int(environment.state.cars[agent].team_num)
        ordered[team] = observation
    return ordered


def _update_state_hash(digest, environment, *, reset_marker: bool = False) -> None:
    digest.update(b"R" if reset_marker else b"T")
    state = environment.state
    digest.update(np.asarray([state.tick_count], dtype="<i8").tobytes())
    for value in (
        state.ball.position,
        state.ball.linear_velocity,
        state.ball.angular_velocity,
    ):
        digest.update(np.asarray(value, dtype="<f4").tobytes())
    cars = sorted(state.cars.values(), key=lambda car: int(car.team_num))
    for car in cars:
        digest.update(np.asarray([car.team_num], dtype="<i4").tobytes())
        for value in (
            car.physics.position,
            car.physics.rotation_mtx,
            car.physics.linear_velocity,
            car.physics.angular_velocity,
        ):
            digest.update(np.asarray(value, dtype="<f4").tobytes())
        digest.update(
            np.asarray(
                [
                    car.boost_amount,
                    car.demo_respawn_timer,
                    car.on_ground,
                    car.has_jumped,
                    car.has_double_jumped,
                    car.has_flipped,
                ],
                dtype="<f4",
            ).tobytes()
        )


def _collect_timings(
    environment,
    observation_seconds: list[float],
    predictor_seconds: list[float],
    ages: Counter[int],
) -> None:
    for timing in environment.shared_info["rival_v9_observation_timings"].values():
        observation_seconds.append(float(timing["observation_seconds"]))
        predictor_seconds.append(float(timing["predictor_seconds"]))
        ages[int(timing["prediction_age_ticks"])] += 1


def _run_period(
    period: int,
    actions: np.ndarray,
    *,
    warmup_ticks: int,
    measured_ticks: int,
) -> tuple[dict[str, Any], np.ndarray]:
    random.seed(SEED)
    np.random.seed(SEED)
    environment = build_v9_diagnostic_env(prediction_refresh_ticks=period)
    observations = environment.reset()
    resets = 0
    try:
        for index in range(warmup_ticks):
            observations, _rewards, terminated, truncated = environment.step(
                _actions_by_agent(environment, actions[index])
            )
            if any(terminated.values()) or any(truncated.values()):
                observations = environment.reset()

        observation_seconds: list[float] = []
        predictor_seconds: list[float] = []
        ages: Counter[int] = Counter()
        captured = np.empty(
            (measured_ticks, AGENTS, OBSERVATION_SIZE), dtype=np.float32
        )
        state_digest = hashlib.sha256()
        nonfinite = 0
        cpu_started = time.process_time()
        wall_started = time.perf_counter()
        for offset in range(measured_ticks):
            action_index = warmup_ticks + offset
            observations, rewards, terminated, truncated = environment.step(
                _actions_by_agent(environment, actions[action_index])
            )
            _collect_timings(
                environment, observation_seconds, predictor_seconds, ages
            )
            ordered = _observations_by_team(environment, observations)
            captured[offset] = ordered
            nonfinite += int(not np.isfinite(ordered).all())
            if any(float(value) != 0.0 for value in rewards.values()):
                raise AssertionError("Gate 4 diagnostic environment emitted nonzero reward")
            _update_state_hash(state_digest, environment)
            if any(terminated.values()) or any(truncated.values()):
                observations = environment.reset()
                resets += 1
                _collect_timings(
                    environment, observation_seconds, predictor_seconds, ages
                )
                _update_state_hash(state_digest, environment, reset_marker=True)
        wall_seconds = time.perf_counter() - wall_started
        process_cpu_seconds = time.process_time() - cpu_started
        agent_steps = measured_ticks * AGENTS
        simulated_seconds = measured_ticks / PHYSICS_HZ
        report = {
            "period_ticks": period,
            "physics_ticks": measured_ticks,
            "agent_steps": agent_steps,
            "simulated_game_seconds": simulated_seconds,
            "wall_seconds": wall_seconds,
            "agent_steps_per_second": agent_steps / wall_seconds,
            "simulated_game_seconds_per_wall_second": simulated_seconds
            / wall_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "single_process_cpu_utilization_percent": 100.0
            * process_cpu_seconds
            / wall_seconds,
            "episode_resets": resets,
            "observation_samples": len(observation_seconds),
            "observation_build_seconds": _percentiles(observation_seconds),
            "observation_build_cpu_total_seconds": float(
                np.sum(observation_seconds)
            ),
            "predictor_seconds": _percentiles(
                [value for value in predictor_seconds if value > 0.0]
            ),
            "predictor_cpu_total_seconds": float(np.sum(predictor_seconds)),
            "predictor_refreshes": int(
                np.count_nonzero(np.asarray(predictor_seconds) > 0.0)
            ),
            "prediction_age_distribution": {
                str(age): int(count) for age, count in sorted(ages.items())
            },
            "nonfinite_observation_ticks": nonfinite,
            "state_trajectory_sha256": state_digest.hexdigest(),
        }
        return report, captured
    finally:
        environment.close()


def _comparison(
    candidate: np.ndarray,
    reference: np.ndarray,
    schema: dict[str, Any],
) -> dict[str, Any]:
    absolute = np.abs(candidate.astype(np.float64) - reference.astype(np.float64))
    prediction_slice = schema["block_slices"]["prediction"]
    prediction = absolute[
        ..., int(prediction_slice["start"]) : int(prediction_slice["end"])
    ]
    age_field = next(field for field in schema["fields"] if field["name"] == "prediction.age")
    mask = np.ones(OBSERVATION_SIZE, dtype=np.bool_)
    mask[int(prediction_slice["start"]) : int(prediction_slice["end"])] = False
    mask[int(age_field["start"]) : int(age_field["end"])] = False
    per_observation_prediction_max = np.max(prediction, axis=-1).reshape(-1)
    return {
        "reference_period_ticks": 1,
        "compared_observations": int(np.prod(candidate.shape[:-1])),
        "all_observation_max_abs": float(np.max(absolute)),
        "prediction_block_max_abs": float(np.max(prediction)),
        "prediction_block_per_observation_max_abs": _percentiles(
            per_observation_prediction_max.tolist()
        ),
        "non_prediction_and_non_age_mismatched_floats": int(
            np.count_nonzero(absolute[..., mask])
        ),
    }


def _native_replay(
    period: int, raw_path: Path
) -> tuple[dict[str, Any], np.ndarray]:
    """Replay one exact natural RLBot sequence through one cache cadence."""

    adapter = RLBotCanonicalAdapterV1()
    builder = RivalObsV1Builder(prediction_refresh_ticks=period)
    observations: list[np.ndarray] = []
    observation_seconds: list[float] = []
    predictor_seconds: list[float] = []
    ages: Counter[int] = Counter()
    last_frame: int | None = None
    discontinuity_resets = 0
    with raw_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") != "rival_v9_native_packet":
                continue
            frame = int(record["frame_num"])
            if last_frame is not None and frame - last_frame != 1:
                adapter.reset()
                builder.reset()
                discontinuity_resets += 1
            packet, field_info, self_index = snapshot_to_rlbot_sources(record["packet"])
            canonical = adapter.adapt(packet, self_index, field_info)
            observation = builder.build(canonical)
            observations.append(observation)
            timing = builder.last_timings
            observation_seconds.append(float(timing["observation_seconds"]))
            predictor_seconds.append(float(timing["predictor_seconds"]))
            ages[int(timing["prediction_age_ticks"])] += 1
            last_frame = frame
    captured = np.asarray(observations, dtype=np.float32)
    report = {
        "period_ticks": period,
        "records": int(captured.shape[0]),
        "discontinuity_resets": discontinuity_resets,
        "nonfinite_observations": int(
            np.count_nonzero(~np.isfinite(captured).all(axis=1))
        ),
        "observation_build_seconds": _percentiles(observation_seconds),
        "predictor_seconds": _percentiles(
            [value for value in predictor_seconds if value > 0.0]
        ),
        "predictor_refreshes": int(
            np.count_nonzero(np.asarray(predictor_seconds) > 0.0)
        ),
        "prediction_age_distribution": {
            str(age): int(count) for age, count in sorted(ages.items())
        },
    }
    return report, captured


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_gate(*, warmup_ticks: int, measured_ticks: int, output: Path) -> int:
    schema = observation_schema_manifest()
    actions = _action_trace(warmup_ticks + measured_ticks)
    results: dict[int, dict[str, Any]] = {}
    # Exercise one sacrificial arena through a contact window so one-time
    # RocketSim process initialization is excluded from all three measurements.
    process_warmup_ticks = min(600, measured_ticks)
    _warmup_report, _warmup_capture = _run_period(
        4,
        actions,
        warmup_ticks=warmup_ticks,
        measured_ticks=process_warmup_ticks,
    )
    del _warmup_report, _warmup_capture
    for period in PERIODS:
        result, capture = _run_period(
            period,
            actions,
            warmup_ticks=warmup_ticks,
            measured_ticks=measured_ticks,
        )
        results[period] = result
        del capture

    capture_report = json.loads(
        (TRAINING_ROOT / "results" / "milestone09" / "gate03_native_capture.json")
        .read_text(encoding="utf-8")
    )
    native_contract = capture_report["native_corpus"]
    native_path = TRAINING_ROOT.parent / native_contract["path"]
    with native_path.open("rb") as native_stream:
        native_sha256 = hashlib.file_digest(native_stream, "sha256").hexdigest()
    native_replays: dict[int, dict[str, Any]] = {}
    native_captures: dict[int, np.ndarray] = {}
    for period in PERIODS:
        native_replays[period], native_captures[period] = _native_replay(
            period, native_path
        )
    native_reference = native_captures[1]
    for period in PERIODS:
        comparison = _comparison(
            native_captures[period], native_reference, schema
        )
        native_replays[period]["fresh_period1_observation_comparison"] = comparison
        results[period]["fresh_period1_observation_comparison"] = comparison

    trajectory_hashes = {
        result["state_trajectory_sha256"] for result in results.values()
    }
    freshness_acceptable = {
        period: (
            result["fresh_period1_observation_comparison"][
                "prediction_block_per_observation_max_abs"
            ]["p95"]
            <= 0.025
            and result["fresh_period1_observation_comparison"][
                "prediction_block_max_abs"
            ]
            <= 0.10
        )
        for period, result in results.items()
    }
    eligible = [period for period in PERIODS if freshness_acceptable[period]]
    fastest = max(eligible, key=lambda period: results[period]["agent_steps_per_second"])
    period1_throughput = results[1]["agent_steps_per_second"]
    fastest_materially_better = (
        results[fastest]["agent_steps_per_second"] >= 1.03 * period1_throughput
    )
    selected = fastest if fastest == 1 or fastest_materially_better else 1
    checks = {
        "all_periods_completed": set(results) == set(PERIODS),
        "all_observations_finite": all(
            result["nonfinite_observation_ticks"] == 0 for result in results.values()
        ),
        "prediction_age_never_exceeds_period_minus_one": all(
            max(int(age) for age in result["prediction_age_distribution"])
            <= period - 1
            for period, result in results.items()
        ),
        "native_corpus_hash_matches_gate3": native_sha256
        == native_contract["sha256"],
        "native_replay_record_counts_match": all(
            replay["records"] == int(native_contract["records"])
            for replay in native_replays.values()
        ),
        "native_replay_observations_finite": all(
            replay["nonfinite_observations"] == 0
            for replay in native_replays.values()
        ),
        "non_prediction_features_bit_identical": all(
            result["fresh_period1_observation_comparison"][
                "non_prediction_and_non_age_mismatched_floats"
            ]
            == 0
            for result in results.values()
        ),
        "selected_period_meets_prospective_freshness_bounds": freshness_acceptable[
            selected
        ],
        "selected_period_is_measured_fastest_eligible_or_period1_fallback": (
            selected == fastest or (selected == 1 and not fastest_materially_better)
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 4,
        "gate_name": "prediction_update_cadence_benchmark",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "contract": {
            "physics_hz": PHYSICS_HZ,
            "agents": AGENTS,
            "warmup_ticks_per_period": warmup_ticks,
            "measured_ticks_per_period": measured_ticks,
            "measured_agent_steps_per_period": measured_ticks * AGENTS,
            "sacrificial_process_warmup_ticks": process_warmup_ticks,
            "prediction_periods_ticks": list(PERIODS),
            "action_trace_seed": SEED,
            "action_trace_sha256": hashlib.sha256(actions.tobytes()).hexdigest(),
            "environment_version": V9_ENVIRONMENT_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "observation_schema_sha256": schema["schema_sha256"],
            "shared_ball_prediction": schema["shared_ball_prediction"],
            "diagnostic_reward": "identically_zero",
            "rlbot_delay": True,
        },
        "prospective_freshness_bounds": {
            "prediction_block_per_observation_max_abs_p95": 0.025,
            "prediction_block_global_max_abs": 0.10,
            "interpretation": (
                "Normalized actor-input bounds chosen before the benchmark; period 4 "
                "can be at most three physics ticks (25 ms) stale."
            ),
        },
        "results": {str(period): results[period] for period in PERIODS},
        "natural_rlbot_freshness_replay": {
            "source": {
                "path": native_contract["path"],
                "sha256": native_sha256,
                "size_bytes": native_path.stat().st_size,
                "records": native_contract["records"],
                "corpus_version": native_contract["version"],
            },
            "periods": {
                str(period): native_replays[period] for period in PERIODS
            },
        },
        "open_loop_environment_trajectory_diagnostic": {
            "unique_hashes": len(trajectory_hashes),
            "hashes_by_period": {
                str(period): results[period]["state_trajectory_sha256"]
                for period in PERIODS
            },
            "gate_use": (
                "Diagnostic only. Cache freshness is compared on the identical frozen "
                "native corpus because independently created RocketSim arenas can have "
                "process-order variation unrelated to observation cadence."
            ),
        },
        "selection": {
            "freshness_acceptable": {
                str(period): acceptable
                for period, acceptable in freshness_acceptable.items()
            },
            "fastest_eligible_period_ticks": fastest,
            "fastest_materially_better_than_period1_by_at_least_3_percent": (
                fastest_materially_better
            ),
            "selected_prediction_refresh_ticks": selected,
            "policy_behavior_comparison": (
                "Not applicable: no meaningful scratch Rival policy exists before the "
                "ordered pre-PPO gates. Actor-input differences were measured on the "
                "identical frozen natural RLBot state sequence; the RocketSim runs use "
                "open-loop actions only for throughput and are not a skill evaluation."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "process_model": "single_process_headless_RocketSim",
        },
        "commands": {
            "gate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_prediction_cadence_gate.py"
            ),
            "unit_tests": (
                "training/.venv/Scripts/python.exe -m pytest "
                "training/tests/test_v9_environment.py -q"
            ),
        },
        "interpretation": (
            "This gate measures the complete canonical adapter plus shared RivalObsV1 "
            "workload at native one-tick environment cadence. It is not PPO throughput, "
            "worker-count evidence, or a policy-skill evaluation."
        ),
    }
    _write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-ticks", type=int, default=300)
    parser.add_argument("--measured-ticks", type=int, default=4800)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup_ticks < 1 or args.measured_ticks < 120:
        raise ValueError("Gate 4 requires positive warmup and at least 120 measured ticks")
    return run_gate(
        warmup_ticks=args.warmup_ticks,
        measured_ticks=args.measured_ticks,
        output=args.output,
    )


if __name__ == "__main__":
    raise SystemExit(main())
