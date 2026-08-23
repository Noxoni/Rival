"""Generate Milestone 09 Gate 2 canonical-observation evidence in both runtimes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_canonical import (  # noqa: E402
    CANONICAL_ADAPTER_VERSION,
    CANONICAL_STATE_VERSION,
    STANDARD_PAD_IS_BIG,
    STANDARD_PAD_POSITIONS,
    RLBotCanonicalAdapterV1,
    RivalCanonicalStateV1,
    RocketSimCanonicalAdapterV1,
)
from rival_training.v9_observations import (  # noqa: E402
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    RivalObsV1Builder,
    observation_schema_manifest,
)
from rival_training.v9_soccar_geometry import (  # noqa: E402
    STANDARD_GOAL_CENTERS,
    STANDARD_GOAL_HEIGHTS,
    STANDARD_GOAL_WIDTHS,
)


SCHEMA_PATH = TRAINING_ROOT / "schemas" / "rival_obs_v1.json"
RESULT_PATH = TRAINING_ROOT / "results" / "milestone09" / "gate02_canonical_schema.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rlbot_physics(flat, position, velocity, angular):
    return flat.Physics(
        location=flat.Vector3(*position),
        rotation=flat.Rotator(0.0, 0.0, 0.0),
        velocity=flat.Vector3(*velocity),
        angular_velocity=flat.Vector3(*angular),
    )


def _rlbot_runtime_smoke() -> dict[str, Any]:
    import rlbot.flat as flat

    self_input = flat.ControllerState(
        throttle=0.25,
        steer=-0.5,
        pitch=0.75,
        yaw=-0.125,
        roll=0.625,
        jump=False,
        boost=True,
        handbrake=False,
    )
    opponent_input = flat.ControllerState(
        throttle=-0.4,
        steer=0.3,
        pitch=-0.2,
        yaw=0.1,
        roll=-0.6,
        jump=False,
        boost=False,
        handbrake=True,
    )
    players = [
        flat.PlayerInfo(
            physics=_rlbot_physics(
                flat,
                (-1000.0, -2000.0, 17.0),
                (400.0, 50.0, 0.0),
                (0.1, -0.2, 0.3),
            ),
            team=0,
            boost=55.0,
            air_state=flat.AirState.OnGround,
            demolished_timeout=-1.0,
            last_input=self_input,
            latest_touch=flat.Touch(
                game_seconds=10.0,
                location=flat.Vector3(100.0, 200.0, 93.0),
                normal=flat.Vector3(0.0, 0.0, 1.0),
                ball_index=0,
            ),
            dodge_timeout=-1.0,
            dodge_dir=flat.Vector2(0.0, 0.0),
        ),
        flat.PlayerInfo(
            physics=_rlbot_physics(
                flat,
                (900.0, 1800.0, 17.0),
                (400.0, 50.0, 0.0),
                (0.1, -0.2, 0.3),
            ),
            team=1,
            boost=55.0,
            air_state=flat.AirState.OnGround,
            demolished_timeout=-1.0,
            last_input=opponent_input,
            dodge_timeout=-1.0,
            dodge_dir=flat.Vector2(0.0, 0.0),
        ),
    ]
    dynamic_pads = [flat.BoostPadState(is_active=True, timer=0.0) for _ in range(34)]
    dynamic_pads[0] = flat.BoostPadState(is_active=False, timer=2.0)
    dynamic_pads[3] = flat.BoostPadState(is_active=False, timer=3.0)
    packet = flat.GamePacket(
        players=players,
        balls=[
            flat.BallInfo(
                physics=_rlbot_physics(
                    flat,
                    (100.0, 200.0, 93.0),
                    (500.0, -100.0, 20.0),
                    (0.0, 0.5, 0.0),
                )
            )
        ],
        boost_pads=dynamic_pads,
        teams=[flat.TeamInfo(team_index=0, score=2), flat.TeamInfo(team_index=1, score=1)],
        match_info=flat.MatchInfo(
            seconds_elapsed=10.0,
            game_time_remaining=290.0,
            frame_num=1200,
            is_overtime=False,
            match_phase=flat.MatchPhase.Active,
            world_gravity_z=-650.0,
            game_speed=1.0,
        ),
    )
    field_info = flat.FieldInfo(
        boost_pads=[
            flat.BoostPad(
                location=flat.Vector3(*position),
                is_full_boost=bool(STANDARD_PAD_IS_BIG[index]),
            )
            for index, position in enumerate(STANDARD_PAD_POSITIONS)
        ],
        goals=[
            flat.GoalInfo(
                team_num=team,
                location=flat.Vector3(*STANDARD_GOAL_CENTERS[team]),
                direction=flat.Vector3(0.0, 1.0 if team == 0 else -1.0, 0.0),
                width=float(STANDARD_GOAL_WIDTHS[team]),
                height=float(STANDARD_GOAL_HEIGHTS[team]),
            )
            for team in range(2)
        ],
    )
    canonical = RLBotCanonicalAdapterV1().adapt(packet, 0, field_info)
    builder = RivalObsV1Builder(
        prediction_refresh_ticks=4,
        collision_mesh_directory=REPO_ROOT / "bot" / "collision_meshes",
    )
    observation = builder.build(canonical)
    serialized = json.dumps(canonical.to_payload(), sort_keys=True)
    restored = RivalCanonicalStateV1.from_payload(json.loads(serialized))
    runtime = builder.export_runtime_state()
    expected = builder.build(restored)
    independent = RivalObsV1Builder(
        prediction_refresh_ticks=4,
        collision_mesh_directory=REPO_ROOT / "bot" / "collision_meshes",
    )
    independent.load_runtime_state(json.loads(json.dumps(runtime)))
    reproduced = independent.build(restored)
    return {
        "status": "passed",
        "runtime": "production_rlbot_v5_virtual_environment",
        "used_generated_rlbot_flat_types": True,
        "canonical_state_version": canonical.version,
        "adapter_version": canonical.adapter_version,
        "observation_version": OBSERVATION_VERSION,
        "observation_shape": list(observation.shape),
        "observation_finite": bool(np.isfinite(observation).all()),
        "serialized_reproduction_bit_identical": bool(np.array_equal(expected, reproduced)),
        "pad_count": len(canonical.pad_positions),
        "pad0_time_until_active": float(canonical.pad_time_until_active[0]),
        "pad3_time_until_active": float(canonical.pad_time_until_active[3]),
        "goal_centers": canonical.goal_centers.tolist(),
        "goal_widths": canonical.goal_widths.tolist(),
        "goal_heights": canonical.goal_heights.tolist(),
        "prediction_refreshed": bool(builder.last_timings["prediction_refreshed"]),
    }


def _training_runtime_smoke() -> dict[str, Any]:
    from rival_training.environment import build_dual_rate_env

    environment = build_dual_rate_env(natural_only=True, seed=20260902)
    try:
        environment.reset()
        state = environment.state
        observations: list[np.ndarray] = []
        parity: list[bool] = []
        goal_contracts: list[dict[str, Any]] = []
        for agent in state.cars:
            adapter = RocketSimCanonicalAdapterV1()
            canonical = adapter.adapt(state, agent, environment.shared_info)
            builder = RivalObsV1Builder(prediction_refresh_ticks=4)
            observation = builder.build(canonical)
            observations.append(observation)
            goal_contracts.append(
                {
                    "centers": canonical.goal_centers.tolist(),
                    "widths": canonical.goal_widths.tolist(),
                    "heights": canonical.goal_heights.tolist(),
                }
            )
            runtime = builder.export_runtime_state()
            serialized = json.dumps(canonical.to_payload(), sort_keys=True)
            restored = RivalCanonicalStateV1.from_payload(json.loads(serialized))
            expected = builder.build(restored)
            independent = RivalObsV1Builder(prediction_refresh_ticks=4)
            independent.load_runtime_state(json.loads(json.dumps(runtime)))
            reproduced = independent.build(restored)
            parity.append(bool(np.array_equal(expected, reproduced)))
        return {
            "status": "passed",
            "runtime": "training_rlgym_rocketsim_virtual_environment",
            "agents": len(state.cars),
            "canonical_state_version": CANONICAL_STATE_VERSION,
            "adapter_version": CANONICAL_ADAPTER_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "observation_shapes": [list(value.shape) for value in observations],
            "goal_contracts": goal_contracts,
            "observations_finite": all(bool(np.isfinite(value).all()) for value in observations),
            "serialized_reproduction_bit_identical": all(parity),
            "shared_builder_module": RivalObsV1Builder.__module__,
            "adapter_modules": [
                RocketSimCanonicalAdapterV1.__module__,
                RLBotCanonicalAdapterV1.__module__,
            ],
        }
    finally:
        environment.close()


def _main_gate() -> int:
    schema = observation_schema_manifest()
    _write_json(SCHEMA_PATH, schema)
    training_smoke = _training_runtime_smoke()
    production_python = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    completed = subprocess.run(
        [str(production_python), str(Path(__file__).resolve()), "--rlbot-runtime-smoke"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"RLBot runtime smoke failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}"
        )
    production_smoke = json.loads(completed.stdout.strip().splitlines()[-1])
    fields = schema["fields"]
    checks = {
        "canonical_version_frozen": schema["canonical_state_version"] == CANONICAL_STATE_VERSION,
        "adapter_version_frozen": schema["canonical_adapter_version"] == CANONICAL_ADAPTER_VERSION,
        "shared_builder_version_frozen": schema["observation_version"] == OBSERVATION_VERSION,
        "schema_size_stable": schema["float_count"] == OBSERVATION_SIZE == 714,
        "schema_contiguous": all(
            field["start"] == (0 if index == 0 else fields[index - 1]["end"])
            for index, field in enumerate(fields)
        ),
        "every_field_documented": all(
            field[attribute]
            for field in fields
            for attribute in (
                "normalization",
                "coordinate_frame",
                "canonical_source",
                "update_cadence",
                "reset_semantics",
            )
        ),
        "all_34_pads_structured": schema["entity_shapes"]["boost_pads"] == [34, 9],
        "six_prediction_horizons_structured": schema["entity_shapes"]["prediction"] == [6, 12],
        "eight_tick_histories_structured": (
            schema["entity_shapes"]["self_controller_history"] == [8, 8]
            and schema["entity_shapes"]["opponent_controller_history"] == [8, 8]
        ),
        "training_runtime_passed": training_smoke["status"] == "passed",
        "production_runtime_passed": production_smoke["status"] == "passed",
        "training_snapshot_bit_identical": training_smoke["serialized_reproduction_bit_identical"],
        "production_snapshot_bit_identical": production_smoke["serialized_reproduction_bit_identical"],
        "same_shared_builder_after_canonicalization": (
            training_smoke["shared_builder_module"] == "rival_training.v9_observations"
            and set(training_smoke["adapter_modules"])
            == {"rival_training.v9_canonical"}
        ),
        "no_running_observation_standardization": schema["running_standardization"] is False,
        "no_state_dependent_x_mirror": "no state-dependent X mirror" in schema["team_frame"],
        "geometry_source_hash_frozen": len(schema["geometry_source_sha256"]) == 64,
        "physical_goal_contract_matches_in_both_runtimes": (
            all(
                contract["centers"] == production_smoke["goal_centers"]
                and contract["widths"] == production_smoke["goal_widths"]
                and contract["heights"] == production_smoke["goal_heights"]
                for contract in training_smoke["goal_contracts"]
            )
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 2,
        "gate_name": "canonical_observation_schema",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "schema": {
            "path": "training/schemas/rival_obs_v1.json",
            "size_bytes": SCHEMA_PATH.stat().st_size,
            "file_sha256": _sha256(SCHEMA_PATH),
            "schema_sha256": schema["schema_sha256"],
            "float_count": schema["float_count"],
            "block_slices": schema["block_slices"],
            "entity_shapes": schema["entity_shapes"],
            "builder_source_sha256": schema["builder_source_sha256"],
            "canonical_source_sha256": schema["canonical_source_sha256"],
            "geometry_source_sha256": schema["geometry_source_sha256"],
            "standard_soccar_geometry": schema["standard_soccar_geometry"],
        },
        "training_runtime_smoke": training_smoke,
        "production_runtime_smoke": production_smoke,
        "commands": {
            "generate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_canonical_gate.py"
            ),
            "unit_tests": (
                "training/.venv/Scripts/python.exe -m pytest "
                "training/tests/test_v9_canonical_observation.py -q"
            ),
        },
        "gate_scope": (
            "Gate 2 freezes the canonical types and generated schema. Broad natural RLBot "
            "corpus/source-field parity remains Gate 3 and is not claimed here."
        ),
    }
    _write_json(RESULT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rlbot-runtime-smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    if arguments.rlbot_runtime_smoke:
        print(json.dumps(_rlbot_runtime_smoke(), sort_keys=True))
        return 0
    return _main_gate()


if __name__ == "__main__":
    raise SystemExit(main())
