from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import tempfile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = REPOSITORY_ROOT / "bot"
sys.path.insert(0, str(BOT_ROOT))

import rlbot.flat as flat  # noqa: E402

from bot import RivalBot  # noqa: E402


def _physics(
    x: float,
    y: float,
    z: float,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> flat.Physics:
    return flat.Physics(
        location=flat.Vector3(x, y, z),
        rotation=flat.Rotator(0.0, 0.0, 0.0),
        velocity=flat.Vector3(vx, vy, vz),
        angular_velocity=flat.Vector3(0.0, 0.0, 0.0),
    )


def build_synthetic_inputs() -> tuple[flat.GamePacket, flat.BallPrediction]:
    ball_physics = _physics(0.0, 0.0, 93.0, 0.0, 250.0, 0.0)
    prediction = flat.BallPrediction(
        [
            flat.PredictionSlice(
                game_seconds=1.0 + tick / 120.0,
                physics=_physics(0.0, 250.0 * tick / 120.0, 93.0, 0.0, 250.0, 0.0),
            )
            for tick in range(600)
        ]
    )
    players = [
        flat.PlayerInfo(
            physics=_physics(-1000.0, -1500.0, 17.0, 500.0, 700.0, 0.0),
            air_state=flat.AirState.OnGround,
            is_supersonic=False,
            is_bot=True,
            name="Rival Dev",
            team=0,
            boost=40.0,
            player_id=1,
            last_input=flat.ControllerState(),
        ),
        flat.PlayerInfo(
            physics=_physics(1000.0, 1500.0, 17.0, -400.0, -600.0, 0.0),
            air_state=flat.AirState.OnGround,
            is_supersonic=False,
            is_bot=True,
            name="Synthetic Opponent",
            team=1,
            boost=55.0,
            player_id=2,
            last_input=flat.ControllerState(),
        ),
    ]
    packet = flat.GamePacket(
        players=players,
        boost_pads=[],
        balls=[flat.BallInfo(physics=ball_physics)],
        match_info=flat.MatchInfo(
            seconds_elapsed=1.0,
            game_time_remaining=299.0,
            is_overtime=False,
            is_unlimited_time=False,
            match_phase=flat.MatchPhase.Active,
            world_gravity_z=-650.0,
            game_speed=1.0,
            frame_num=120,
        ),
        teams=[flat.TeamInfo(team_index=0, score=0), flat.TeamInfo(team_index=1, score=0)],
    )
    return packet, prediction


def main() -> int:
    random.seed(20260822)
    scratch_root = REPOSITORY_ROOT / ".pytest_tmp"
    scratch_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="decision-smoke-", dir=scratch_root) as scratch:
        telemetry_path = Path(scratch) / "decision.jsonl"
        rival = RivalBot()
        rival.index = 0
        rival.team = 0
        rival.name = "Rival Dev"
        rival.initialize()
        rival.telemetry.path = telemetry_path
        rival.telemetry.enabled = True

        packet, prediction = build_synthetic_inputs()
        rival.ball_prediction = prediction
        controller = rival.get_output(packet)
        rival.telemetry.close()

        if rival.last_decision is None or rival.last_tactical_metrics is None:
            print(json.dumps({"status": "fail", "reason": "no decision"}, indent=2))
            return 1

        records = [
            json.loads(line)
            for line in telemetry_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        decision = rival.last_decision
        result = {
            "status": "pass" if len(records) == 1 else "fail",
            "action_index": decision.action_index,
            "controller_action": decision.controller_action.to_record(),
            "controller_output": {
                "throttle": float(controller.throttle),
                "steer": float(controller.steer),
                "pitch": float(controller.pitch),
                "yaw": float(controller.yaw),
                "roll": float(controller.roll),
                "jump": bool(controller.jump),
                "boost": bool(controller.boost),
                "handbrake": bool(controller.handbrake),
            },
            "legal_actions": int(decision.legal_mask.sum().item()),
            "top_action_indices": [
                candidate.action_index for candidate in decision.top_actions
            ],
            "confidence": decision.confidence,
            "margin": decision.margin,
            "eta_method": rival.last_tactical_metrics.eta_method,
            "telemetry_records": len(records),
            "telemetry_schema_version": (
                records[0]["schema_version"] if records else None
            ),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
