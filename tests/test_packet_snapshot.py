from __future__ import annotations

import rlbot.flat as flat
from types import SimpleNamespace

from telemetry.packet_snapshot import extract_packet_snapshot, extract_player


def _physics() -> flat.Physics:
    return flat.Physics(
        location=flat.Vector3(1.0, 2.0, 3.0),
        rotation=flat.Rotator(0.1, 0.2, 0.3),
        velocity=flat.Vector3(4.0, 5.0, 6.0),
        angular_velocity=flat.Vector3(0.4, 0.5, 0.6),
    )


def test_v5_packet_snapshot_extracts_last_input_touch_and_orientation() -> None:
    rival = flat.PlayerInfo(
        physics=_physics(),
        latest_touch=flat.Touch(
            game_seconds=12.25,
            location=flat.Vector3(10.0, 20.0, 30.0),
            normal=flat.Vector3(0.0, 0.0, 1.0),
            ball_index=0,
        ),
        air_state=flat.AirState.Dodging,
        dodge_timeout=1.25,
        name="Rival Dev",
        team=0,
        boost=17.0,
        player_id=44,
        last_input=flat.ControllerState(
            throttle=1.0,
            steer=-0.25,
            jump=True,
            boost=True,
        ),
        has_jumped=True,
        has_dodged=True,
    )
    opponent = flat.PlayerInfo(
        physics=_physics(),
        air_state=flat.AirState.OnGround,
        name="Opponent",
        team=1,
        player_id=45,
    )
    packet = flat.GamePacket(
        players=[rival, opponent],
        boost_pads=[flat.BoostPadState(is_active=False, timer=4.0)],
        balls=[flat.BallInfo(physics=_physics())],
        match_info=flat.MatchInfo(
            seconds_elapsed=12.5,
            game_time_remaining=287.5,
            match_phase=flat.MatchPhase.Active,
            frame_num=1500,
        ),
        teams=[flat.TeamInfo(0, 1), flat.TeamInfo(1, 2)],
    )
    field_info = flat.FieldInfo(
        boost_pads=[flat.BoostPad(flat.Vector3(100.0, 0.0, 73.0), True)]
    )

    snapshot = extract_packet_snapshot(packet, 0, field_info)

    self_record = snapshot["players"][0]
    assert self_record["name"] == "Rival Dev"
    assert self_record["player_id"] == 44
    assert self_record["last_input"]["jump"] is True
    assert self_record["last_input"]["boost"] is True
    assert self_record["latest_touch"]["game_seconds"] == 12.25
    assert self_record["air_state"]["name"] == "Dodging"
    assert self_record["physics"]["rotation"]["forward"]
    assert self_record["physics"]["rotation"]["up"]
    assert self_record["physics"]["angular_velocity"]["z"] == 0.6
    assert snapshot["opponent_indices"] == [1]
    assert snapshot["boost_pads"][0]["is_full_boost"] is True
    assert snapshot["match"]["scores"] == [
        {"team": 0, "score": 1},
        {"team": 1, "score": 2},
    ]


def test_v5_packet_snapshot_handles_missing_optional_fields() -> None:
    packet = flat.GamePacket(
        players=[flat.PlayerInfo(name="Sparse", team=0)],
        balls=[],
        boost_pads=[],
        match_info=flat.MatchInfo(match_phase=flat.MatchPhase.Active),
        teams=[],
    )

    snapshot = extract_packet_snapshot(packet, 0)

    assert snapshot["ball"] is None
    assert snapshot["opponent_indices"] == []

    sparse_player = extract_player(SimpleNamespace(name="Sparse", team=0), 0)
    assert sparse_player["last_input"] is None
    assert sparse_player["latest_touch"] is None
    assert sparse_player["physics"] is None
