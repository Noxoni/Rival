from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tomllib


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_bob_configuration_uses_canonical_bot_and_pyinstaller_spec() -> None:
    with (REPOSITORY_ROOT / "bob.toml").open("rb") as stream:
        bob = tomllib.load(stream)

    assert len(bob["config"]) == 1
    config = bob["config"][0]
    assert config["project_name"] == "RivalDev"
    assert config["bot_configs"] == ["bot/rival.bot.toml"]
    assert config["builder_config"] == {
        "builder_type": "pyinstaller",
        "entry_file": "packaging/rival.spec",
    }


def test_release_spec_charges_required_runtime_data() -> None:
    spec = (REPOSITORY_ROOT / "packaging" / "rival.spec").read_text(
        encoding="utf-8"
    )

    assert 'bot_root / "models"' in spec
    assert 'bot_root / "collision_meshes"' in spec
    assert '"third_party/wisp"' in spec
    assert 'name="RivalDev"' in spec
    assert "exclude_binaries=True" in spec


def test_source_runtime_self_test_passes() -> None:
    process = subprocess.run(
        [sys.executable, "bot.py", "--self-test"],
        cwd=REPOSITORY_ROOT / "bot",
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert process.returncode == 0, process.stdout + process.stderr
    result = json.loads(process.stdout)
    assert result["status"] == "pass"
    assert result["frozen"] is False
    assert result["observation_shape"] == [432]
    assert result["policy_output_shape"] == [90]
    assert result["collision_mesh_count"] == 16
    assert result["selected_action_index"] == result["compatibility_action_index"]


def test_build_requirement_is_pinned() -> None:
    requirements = (REPOSITORY_ROOT / "requirements-build.txt").read_text(
        encoding="utf-8"
    )
    assert "-r requirements.txt" in requirements
    assert "pyinstaller==6.22.2" in requirements
