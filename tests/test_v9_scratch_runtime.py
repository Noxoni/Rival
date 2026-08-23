from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("RIVAL_"):
            environment.pop(name)
    environment.pop("PYTHONPATH", None)
    return environment


def _config_probe(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,config; print(json.dumps({"
                "'mode':config.POLICY_RUNTIME_MODE,"
                "'tick_skip':config.TICK_SKIP,"
                "'v9':config.V9_SCRATCH_POLICY_ENABLED,"
                "'candidate':config.CANDIDATE_POLICY_ENABLED,"
                "'m08':config.M08_DUAL_RATE_ENABLED,"
                "'policy':config.MODEL_INFO_POLICY.path.name}))"
            ),
        ],
        cwd=REPOSITORY_ROOT / "bot",
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_production_default_remains_frozen_wisp() -> None:
    completed = _config_probe(_clean_environment())
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {
        "mode": "frozen_wisp_production",
        "tick_skip": 8,
        "v9": False,
        "candidate": False,
        "m08": False,
        "policy": "POLICY.lt",
    }


def test_v9_scratch_mode_requires_explicit_diagnostic_opt_in() -> None:
    environment = _clean_environment()
    environment["RIVAL_V9_SCRATCH_MODEL_PATH"] = "candidate.ts"
    completed = _config_probe(environment)
    assert completed.returncode != 0
    assert "opt-in diagnostic-only" in completed.stderr


def test_v9_scratch_mode_is_native_tick_and_does_not_replace_wisp_default_model() -> None:
    environment = _clean_environment()
    environment.update(
        {
            "RIVAL_TRANSFER_DIAGNOSTIC_MODE": "1",
            "RIVAL_V9_SCRATCH_MODEL_PATH": "candidate.ts",
            "RIVAL_V9_SCRATCH_METADATA_PATH": "candidate.metadata.json",
        }
    )
    completed = _config_probe(environment)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout) == {
        "mode": "m09_scratch_candidate_opt_in",
        "tick_skip": 1,
        "v9": True,
        "candidate": False,
        "m08": False,
        "policy": "POLICY.lt",
    }


def test_scratch_runtime_accepts_frozen_m10_training_config_identity() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json,v9_scratch_runtime as runtime; "
                "print(json.dumps(runtime.validate_runtime_constants()))"
            ),
        ],
        cwd=REPOSITORY_ROOT / "bot",
        env=_clean_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    constants = json.loads(completed.stdout)
    assert "RivalM10TrainingConfigV1" in constants[
        "accepted_training_config_versions"
    ]
