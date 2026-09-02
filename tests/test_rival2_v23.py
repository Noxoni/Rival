from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = ROOT / "bot" / "rival2_v23"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _bot_module():
    spec = importlib.util.spec_from_file_location("rival2_v23_bot", BOT_ROOT / "bot.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_side_bundle_selects_exact_120hz_artifacts() -> None:
    bot = _bot_module()
    bundle = json.loads((BOT_ROOT / "deployment_bundle.json").read_text(encoding="utf-8"))
    assert bundle["selector"] == "physical_team_side_before_match"
    assert bundle["source"]["full_match_vs_nexto"] == {
        "wins": 8,
        "losses": 2,
        "goals_for": 159,
        "goals_against": 111,
    }

    for team, side in ((0, "blue"), (1, "orange")):
        model_path, manifest_path, selected = bot._side_paths(team)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert model_path == BOT_ROOT / selected["model"]
        assert manifest["source"]["checkpoint_sha256"] == selected["checkpoint_sha256"]
        assert manifest["contracts"]["observation"] == "RIVAL2_OBS_V2_120HZ"
        assert manifest["contracts"]["action"] == "RIVAL2_ACTION_V2_120HZ"
        assert manifest["contracts"]["physics_hz"] == 120
        assert manifest["contracts"]["policy_hz"] == 120
        assert manifest["contracts"]["hold_ticks"] == 1
        assert manifest["artifact"]["sha256"] == _sha256(model_path)
        assert manifest["export_parity"]["pass"] is True

        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        with torch.inference_mode():
            action = model(torch.zeros((2, 182), dtype=torch.float32))
        assert tuple(action.shape) == (2, 8)
        assert torch.isfinite(action).all()


def test_v23_launcher_builds_balanced_human_match() -> None:
    from scripts.play_rival2_v23 import build_match

    for human_team in (0, 1):
        match = build_match(human_team, "steam")
        assert [player.team for player in match.player_configurations] == [0, 1]
        rival = match.player_configurations[1 - human_team]
        assert rival.variety.name == "Rival 2 V23"
        assert rival.variety.agent_id == "noxoni/rival2/codex-autonomous-v23"
