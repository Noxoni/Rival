from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

BOT_ROOT = Path(__file__).resolve().parents[2] / "bot"
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from telemetry.native_packet_corpus import NativePacketCorpusLogger  # noqa: E402
from telemetry.packet_snapshot import extract_packet_snapshot  # noqa: E402
from rival_training.v9_canonical import (  # noqa: E402
    STANDARD_PAD_IS_BIG,
    STANDARD_PAD_POSITIONS,
    RLBotCanonicalAdapterV1,
)
from rival_training.v9_rlbot_corpus import (  # noqa: E402
    SourceAuditAccumulator,
    audit_canonical_against_snapshot,
    packet_coverage,
    snapshot_to_rlbot_sources,
)
from rival_training.v9_soccar_geometry import (  # noqa: E402
    STANDARD_GOAL_CENTERS,
    STANDARD_GOAL_HEIGHTS,
    STANDARD_GOAL_WIDTHS,
)


def _vec(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=float(x), y=float(y), z=float(z))


def _physics(position, velocity=(0.0, 0.0, 0.0)):
    return SimpleNamespace(
        location=_vec(*position),
        rotation=SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0),
        velocity=_vec(*velocity),
        angular_velocity=_vec(),
    )


def _controller(**overrides):
    values = {
        "throttle": 0.0,
        "steer": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "roll": 0.0,
        "jump": False,
        "boost": False,
        "handbrake": False,
        "use_item": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _packet(frame: int = 1200):
    self_player = SimpleNamespace(
        name="Rival",
        player_id=1,
        team=0,
        is_bot=True,
        physics=_physics((-1000.0, -2000.0, 17.0), (400.0, 50.0, 0.0)),
        boost=5.0,
        is_supersonic=False,
        last_input=_controller(throttle=1.0),
        latest_touch=SimpleNamespace(
            game_seconds=10.0,
            location=_vec(0.0, 0.0, 93.0),
            normal=_vec(0.0, 0.0, 1.0),
            ball_index=0,
        ),
        air_state="OnGround",
        has_jumped=False,
        has_double_jumped=False,
        has_dodged=False,
        dodge_elapsed=0.0,
        dodge_timeout=-1.0,
        dodge_dir=_vec(),
        demolished_timeout=-1.0,
        score_info=None,
        accolades=[],
    )
    opponent = SimpleNamespace(
        name="Opponent",
        player_id=2,
        team=1,
        is_bot=True,
        physics=_physics((900.0, 1800.0, 17.0), (-300.0, -25.0, 0.0)),
        boost=100.0,
        is_supersonic=False,
        last_input=_controller(steer=-0.5),
        latest_touch=None,
        air_state="OnGround",
        has_jumped=False,
        has_double_jumped=False,
        has_dodged=False,
        dodge_elapsed=0.0,
        dodge_timeout=-1.0,
        dodge_dir=_vec(),
        demolished_timeout=-1.0,
        score_info=None,
        accolades=[],
    )
    pads = [SimpleNamespace(is_active=True, timer=0.0) for _ in range(34)]
    pads[0] = SimpleNamespace(is_active=False, timer=2.0)
    packet = SimpleNamespace(
        players=[self_player, opponent],
        balls=[SimpleNamespace(physics=_physics((0.0, 0.0, 93.0)))],
        boost_pads=pads,
        teams=[
            SimpleNamespace(team_index=0, score=1),
            SimpleNamespace(team_index=1, score=0),
        ],
        match_info=SimpleNamespace(
            seconds_elapsed=10.0,
            game_time_remaining=20.0,
            frame_num=frame,
            match_phase="Active",
            is_overtime=False,
            game_speed=1.0,
            world_gravity_z=-650.0,
        ),
    )
    field_info = SimpleNamespace(
        boost_pads=[
            SimpleNamespace(
                location=_vec(
                    float(position[0]),
                    float(position[1]),
                    8.0 if bool(STANDARD_PAD_IS_BIG[index]) else 0.0820159912109375,
                ),
                is_full_boost=bool(STANDARD_PAD_IS_BIG[index]),
            )
            for index, position in enumerate(STANDARD_PAD_POSITIONS)
        ],
        goals=[
            SimpleNamespace(
                team_num=team,
                location=_vec(*STANDARD_GOAL_CENTERS[team]),
                direction=_vec(0.0, 1.0 if team == 0 else -1.0, 0.0),
                width=float(STANDARD_GOAL_WIDTHS[team]),
                height=float(STANDARD_GOAL_HEIGHTS[team]),
            )
            for team in range(2)
        ],
    )
    return packet, field_info


def test_packet_snapshot_reconstruction_uses_real_canonical_adapter() -> None:
    packet, field_info = _packet()
    snapshot = extract_packet_snapshot(packet, 0, field_info)
    assert len(snapshot["goals"]) == 2
    reconstructed, reconstructed_field, self_index = snapshot_to_rlbot_sources(snapshot)
    expected = RLBotCanonicalAdapterV1().adapt(packet, 0, field_info)
    actual = RLBotCanonicalAdapterV1().adapt(
        reconstructed, self_index, reconstructed_field
    )
    assert actual.to_payload() == expected.to_payload()

    audits = {
        name: SourceAuditAccumulator()
        for name in (
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
    }
    audit_canonical_against_snapshot(actual, snapshot, audits)
    assert all(value.to_record()["passed"] for value in audits.values())
    assert {"normal_ground_play", "low_boost", "ball_contact", "late_clock"} <= packet_coverage(snapshot)


def test_native_corpus_logger_bounds_deduplicates_and_records_timing(tmp_path) -> None:
    packet, field_info = _packet(frame=100)
    path = tmp_path / "native.jsonl"
    logger = NativePacketCorpusLogger(
        path,
        enabled=True,
        maximum_records=2,
        metadata={"test": True},
    )
    assert logger.log(
        packet,
        self_index=0,
        field_info=field_info,
        controller_output=_controller(throttle=1.0),
        callback_started_ns=10,
        callback_finished_ns=25,
    )
    assert not logger.log(
        packet,
        self_index=0,
        field_info=field_info,
        controller_output=_controller(),
        callback_started_ns=30,
        callback_finished_ns=40,
    )
    packet.match_info.frame_num = 102
    assert logger.log(
        packet,
        self_index=0,
        field_info=field_info,
        controller_output=_controller(steer=0.5),
        callback_started_ns=50,
        callback_finished_ns=80,
    )
    logger.close()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [record["record_type"] for record in records] == [
        "rival_v9_native_corpus_start",
        "rival_v9_native_packet",
        "rival_v9_native_packet",
        "rival_v9_native_corpus_end",
    ]
    assert records[1]["callback_wall_ns"] == 15
    assert records[2]["frame_num"] == 102
    assert records[-1]["records"] == 2
    assert records[-1]["duplicates_ignored"] == 1
    assert records[-1]["skipped_source_frames"] == 1
    assert records[-1]["complete_bound_reached"] is True
    assert np.isfinite(records[-1]["wall_seconds"])
