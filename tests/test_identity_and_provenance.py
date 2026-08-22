from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

import config


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = REPOSITORY_ROOT / "bot"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_rival_has_unique_rlbot_identity_and_no_overrides() -> None:
    with (BOT_ROOT / "rival.bot.toml").open("rb") as stream:
        bot_config = tomllib.load(stream)

    assert bot_config["settings"]["name"] == "Rival Dev"
    assert bot_config["settings"]["agent_id"] == "noxoni/rival/dev-v1"
    assert bot_config["settings"]["agent_id"] != "eastvillage/wisp/v2-75B"
    assert config.RLBOT_AGENT_ID == bot_config["settings"]["agent_id"]
    assert config.STRATEGIC_OVERRIDES_ENABLED is False
    assert ".venv" in bot_config["settings"]["run_command"]


def test_wisp_model_artifacts_match_recorded_release_hashes() -> None:
    expected = {
        "POLICY.lt": (
            7_689_613,
            "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
        ),
        "SHARED_HEAD.lt": (
            5_995_907,
            "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
        ),
    }

    for name, (size, digest) in expected.items():
        path = BOT_ROOT / "models" / name
        assert path.stat().st_size == size
        assert _sha256(path) == digest


def test_upstream_record_and_installed_snapshot_are_complete() -> None:
    reference_root = REPOSITORY_ROOT / "reference_manifests" / "v1"
    installed = json.loads((reference_root / "MANIFEST.json").read_text(encoding="utf-8"))
    upstream = json.loads(
        (reference_root / "UPSTREAM_SOURCES.json").read_text(encoding="utf-8")
    )

    assert installed["selected_sources"]["wisp"].endswith("WispV2")
    assert installed["selected_sources"]["nexto"].endswith("Nexto")
    assert len(installed["snapshots"]["wisp"]) == 4
    assert len(installed["snapshots"]["nexto"]) == 4
    assert upstream["wisp"]["commit"] == "58d4ab18fd0c92529b5ae6582ecf1713a6b1887a"
    assert upstream["nexto_v5_port"]["selected_as_rival_baseline"] is False


def test_nexto_material_is_not_incorporated() -> None:
    tracked_bot_paths = [
        path.relative_to(BOT_ROOT).as_posix().lower()
        for path in BOT_ROOT.rglob("*")
        if path.is_file()
    ]
    assert all("nexto" not in path and "necto" not in path for path in tracked_bot_paths)
    assert (REPOSITORY_ROOT / "third_party" / "wisp" / "LICENSE").is_file()
