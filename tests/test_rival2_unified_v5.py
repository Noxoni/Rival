from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from test_rival2_live import _packet


ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = ROOT / "bot" / "rival2_unified_v5"
RUNTIME_PATH = BOT_ROOT / "unified_runtime.py"
MODEL_PATH = BOT_ROOT / "models" / "rival2_unified_capability_v5.ts"
MANIFEST_PATH = BOT_ROOT / "models" / "rival2_unified_capability_v5.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "955C93BF538BC913CC2E42F42E3B0EDC4CCDB1065DA9581FB88D84C363B7C216"
)


def _runtime_module():
    spec = importlib.util.spec_from_file_location("rival2_unified_runtime", RUNTIME_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_recurrent_artifact_identity_contract_and_sequence_are_valid():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["format"] == "RIVAL2_RLBOT_RECURRENT_DEPLOY_V1"
    assert manifest["source"]["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert manifest["source"]["runtime_router"] is False
    assert manifest["source"]["task_identifier_input"] is False
    assert manifest["contracts"]["physics_hz"] == 120
    assert manifest["contracts"]["policy_hz"] == 120
    assert manifest["contracts"]["hold_ticks"] == 1
    assert manifest["export_parity"]["pass"] is True
    model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()
    hidden = torch.zeros((1, 2, 256), dtype=torch.float32)
    with torch.inference_mode():
        first, hidden = model(torch.zeros((2, 182)), hidden)
        second, next_hidden = model(torch.ones((2, 182)), hidden)
    assert tuple(first.shape) == (2, 8)
    assert tuple(second.shape) == (2, 8)
    assert tuple(next_hidden.shape) == (1, 2, 256)
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert torch.isfinite(next_hidden).all()
    assert not torch.equal(hidden, torch.zeros_like(hidden))


def test_runtime_advances_hidden_once_per_unique_frame_and_resets():
    runtime_module = _runtime_module()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    packet, field_info = _packet(manifest, frame=100)
    runtime = runtime_module.Rival2UnifiedLiveRuntime(
        MODEL_PATH, MANIFEST_PATH, field_info
    )
    first = runtime.step(packet, team=0)
    first_hidden = runtime.hidden.clone()
    duplicate = runtime.step(packet, team=0)
    assert runtime.decisions == 1
    assert runtime.duplicate_packets == 1
    assert runtime.recurrent_resets == 1
    assert torch.equal(runtime.hidden, first_hidden)
    np.testing.assert_array_equal(first, duplicate)

    packet, _ = _packet(manifest, frame=101)
    runtime.step(packet, team=0)
    assert runtime.decisions == 2
    assert not torch.equal(runtime.hidden, first_hidden)

    countdown, _ = _packet(manifest, frame=102, phase="Countdown")
    np.testing.assert_array_equal(runtime.step(countdown, team=0), np.zeros(8))
    kickoff, _ = _packet(manifest, frame=103, phase="Kickoff")
    runtime.step(kickoff, team=0)
    assert runtime.decisions == 3
    assert runtime.recurrent_resets == 2


def test_unified_launcher_builds_human_match_with_new_agent():
    from scripts.play_rival2_unified_v5 import build_match

    for human_team in (0, 1):
        match = build_match(human_team, "steam")
        assert len(match.player_configurations) == 2
        human = match.player_configurations[human_team]
        rival = match.player_configurations[1 - human_team]
        assert type(human.variety).__name__ == "Human"
        assert rival.variety.name == "Rival 2 Unified V5"
        assert rival.variety.agent_id == "noxoni/rival2/unified-capability-v5"
