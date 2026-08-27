from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PATH = ROOT / "bot" / "rival2_live" / "runtime.py"
MANIFEST_PATH = ROOT / "bot" / "rival2_live" / "models" / "rival2_gameplay_v2_479.json"
MODEL_PATH = ROOT / "bot" / "rival2_live" / "models" / "rival2_gameplay_v2_479.ts"


def _runtime_module():
    spec = importlib.util.spec_from_file_location("rival2_live_runtime", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _vec(x=0.0, y=0.0, z=0.0):
    return SimpleNamespace(x=x, y=y, z=z)


def _controller(**values):
    defaults = dict(
        throttle=0.0,
        steer=0.0,
        pitch=0.0,
        yaw=0.0,
        roll=0.0,
        jump=False,
        boost=False,
        handbrake=False,
    )
    defaults.update(values)
    return SimpleNamespace(**defaults)


def _packet(manifest, frame=0, phase="Kickoff"):
    positions = manifest["observation"]["canonical_boost_pad_positions"]
    # Stable packet order independent of the canonical six-big-first order.
    order = sorted(range(34), key=lambda i: (positions[i][1], positions[i][0]))
    field_info = SimpleNamespace(
        boost_pads=[
            SimpleNamespace(
                location=_vec(
                    positions[index][0],
                    positions[index][1],
                    8.0 if index < 6 else 0.0820159912109375,
                )
            )
            for index in order
        ]
    )
    def physics(x, y, yaw):
        return SimpleNamespace(
            location=_vec(x, y, 17.0),
            velocity=_vec(),
            angular_velocity=_vec(),
            rotation=SimpleNamespace(pitch=0.0, yaw=yaw, roll=0.0),
        )
    players = [
        SimpleNamespace(
            team=0,
            physics=physics(-2048.0, -2560.0, np.pi / 4),
            boost=33.0,
            air_state="AirState.OnGround",
            has_jumped=False,
            has_double_jumped=False,
            has_dodged=False,
            dodge_elapsed=0.0,
            demolished_timeout=-1.0,
            is_supersonic=False,
            last_input=_controller(),
            latest_touch=None,
        ),
        SimpleNamespace(
            team=1,
            physics=physics(2048.0, 2560.0, -3 * np.pi / 4),
            boost=67.0,
            air_state="AirState.OnGround",
            has_jumped=False,
            has_double_jumped=False,
            has_dodged=False,
            dodge_elapsed=0.0,
            demolished_timeout=-1.0,
            is_supersonic=False,
            last_input=_controller(),
            latest_touch=None,
        ),
    ]
    packet = SimpleNamespace(
        players=players,
        balls=[
            SimpleNamespace(
                physics=SimpleNamespace(
                    location=_vec(0.0, 0.0, 92.75),
                    velocity=_vec(),
                    angular_velocity=_vec(),
                )
            )
        ],
        boost_pads=[SimpleNamespace(is_active=True, timer=0.0) for _ in range(34)],
        match_info=SimpleNamespace(
            match_phase=f"MatchPhase.{phase}",
            frame_num=frame,
            seconds_elapsed=frame / 120.0,
        ),
        teams=[SimpleNamespace(score=0), SimpleNamespace(score=0)],
    )
    return packet, field_info


def test_export_identity_and_action_are_valid():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
    with torch.inference_mode():
        action = model(torch.zeros((4, 182), dtype=torch.float32))
    assert tuple(action.shape) == (4, 8)
    assert torch.isfinite(action).all()
    assert torch.all(action[:, :5].abs() <= 1)
    assert torch.all((action[:, 5:] == 0) | (action[:, 5:] == 1))
    assert manifest["export_parity"]["pass"] is True
    assert manifest["source"]["policy_version"] == 479


def test_live_adapter_maps_pads_and_builds_both_team_observations():
    runtime = _runtime_module()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    packet, field_info = _packet(manifest)
    adapter = runtime.Rival2LiveAdapter(manifest, field_info)
    adapter.reset(packet)
    observation = adapter.observation(packet)
    assert observation.shape == (1, 2, 182)
    assert np.isfinite(observation).all()
    assert sorted(adapter.pad_mapping.tolist()) == list(range(34))
    assert observation[0, 0, -7] == 1.0
    assert observation[0, 1, -7] == 1.0


def test_live_runtime_decides_every_four_unique_physics_frames_and_resets():
    runtime_module = _runtime_module()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    packet, field_info = _packet(manifest, frame=100)
    runtime = runtime_module.Rival2LiveRuntime(MODEL_PATH, MANIFEST_PATH, field_info)
    first = runtime.step(packet, team=0)
    duplicate = runtime.step(packet, team=0)
    assert runtime.decisions == 1
    assert runtime.duplicate_packets == 1
    np.testing.assert_array_equal(first, duplicate)

    for frame in (101, 102, 103):
        packet, _ = _packet(manifest, frame=frame)
        runtime.step(packet, team=0)
    assert runtime.decisions == 1
    packet, _ = _packet(manifest, frame=104)
    runtime.step(packet, team=0)
    assert runtime.decisions == 2

    countdown, _ = _packet(manifest, frame=105, phase="Countdown")
    np.testing.assert_array_equal(runtime.step(countdown, team=0), np.zeros(8))
    kickoff, _ = _packet(manifest, frame=106, phase="Kickoff")
    runtime.step(kickoff, team=0)
    assert runtime.decisions == 3
    assert runtime.adapter.memory.kickoff_indicator[0] == 0


def test_play_launcher_builds_balanced_five_minute_human_match():
    from scripts.play_rival2 import build_match

    for human_team in (0, 1):
        match = build_match(human_team, "steam")
        assert len(match.player_configurations) == 2
        assert [player.team for player in match.player_configurations] == [0, 1]
        human = match.player_configurations[human_team]
        rival = match.player_configurations[1 - human_team]
        assert type(human.variety).__name__ == "Human"
        assert rival.variety.name == "Rival 2"
        assert rival.variety.agent_id == "noxoni/rival2/gameplay-v2-479"
        assert str(match.game_mode).endswith("Soccar")
        assert str(match.mutators.match_length).endswith("FiveMinutes")
