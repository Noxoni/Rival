from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = REPOSITORY_ROOT / "bot"
CANDIDATE = REPOSITORY_ROOT / "training/artifacts/milestone06/stage_b_020m/candidate_actor.ts"


def _config_probe(overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    for name in tuple(environment):
        if name.startswith("RIVAL_"):
            environment.pop(name)
    environment.update(overrides)
    environment["PYTHONPATH"] = str(BOT_ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import config; "
                "print(config.TICK_SKIP, config.ACTION_DELAY, "
                "config.CANDIDATE_POLICY_ENABLED, "
                "config.CANDIDATE_LEGACY_ONLY, "
                "config.TRANSFER_DIAGNOSTIC_MODE)"
            ),
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_production_defaults_remain_frozen_wisp_tick_8() -> None:
    result = _config_probe({})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "8 7 False False False"


def test_candidate_tick_8_requires_explicit_transfer_diagnostic_mode() -> None:
    base = {
        "RIVAL_CANDIDATE_MODEL_PATH": str(CANDIDATE),
        "RIVAL_TICK_SKIP": "8",
    }
    rejected = _config_probe(base)
    assert rejected.returncode != 0
    assert "RIVAL_TRANSFER_DIAGNOSTIC_MODE" in rejected.stderr

    accepted = _config_probe({**base, "RIVAL_TRANSFER_DIAGNOSTIC_MODE": "1"})
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == "8 7 True False True"


def test_candidate_legacy_only_requires_candidate_and_diagnostic_mode() -> None:
    rejected = _config_probe({"RIVAL_CANDIDATE_LEGACY_ONLY": "1"})
    assert rejected.returncode != 0
    assert "requires both a candidate model" in rejected.stderr

    accepted = _config_probe(
        {
            "RIVAL_CANDIDATE_MODEL_PATH": str(CANDIDATE),
            "RIVAL_TRANSFER_DIAGNOSTIC_MODE": "1",
            "RIVAL_CANDIDATE_LEGACY_ONLY": "1",
            "RIVAL_TICK_SKIP": "4",
        }
    )
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.strip() == "4 3 True True True"


def test_candidate_runtime_label_is_explicit_and_diagnostic_only() -> None:
    rejected = _config_probe({"RIVAL_CANDIDATE_RUNTIME_LABEL": "m07_zero_step"})
    assert rejected.returncode != 0
    assert "requires a candidate model" in rejected.stderr

    environment = {
        "RIVAL_CANDIDATE_MODEL_PATH": str(CANDIDATE),
        "RIVAL_TRANSFER_DIAGNOSTIC_MODE": "1",
        "RIVAL_CANDIDATE_RUNTIME_LABEL": "m07_zero_step",
    }
    accepted = _config_probe(environment)
    assert accepted.returncode == 0, accepted.stderr
    labeled = _config_probe(
        {**environment, "RIVAL_CANDIDATE_RUNTIME_LABEL": "Not Valid"}
    )
    assert labeled.returncode != 0
    assert "lowercase letters" in labeled.stderr


def test_observation_capture_requires_explicit_diagnostic_telemetry() -> None:
    rejected = _config_probe({"RIVAL_DIAGNOSTIC_CAPTURE_OBSERVATIONS": "1"})
    assert rejected.returncode != 0
    assert "RIVAL_TRANSFER_DIAGNOSTIC_MODE" in rejected.stderr

    rejected_without_telemetry = _config_probe(
        {
            "RIVAL_TRANSFER_DIAGNOSTIC_MODE": "1",
            "RIVAL_DIAGNOSTIC_CAPTURE_OBSERVATIONS": "1",
        }
    )
    assert rejected_without_telemetry.returncode != 0
    assert "RIVAL_TELEMETRY_ENABLED" in rejected_without_telemetry.stderr

    accepted = _config_probe(
        {
            "RIVAL_TRANSFER_DIAGNOSTIC_MODE": "1",
            "RIVAL_TELEMETRY_ENABLED": "1",
            "RIVAL_DIAGNOSTIC_CAPTURE_OBSERVATIONS": "1",
            "RIVAL_DIAGNOSTIC_OBSERVATION_STRIDE": "4",
        }
    )
    assert accepted.returncode == 0, accepted.stderr
