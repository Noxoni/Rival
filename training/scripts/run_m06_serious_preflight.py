"""Run the measured full-size Milestone 06 PPO preflight iteration."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.preflight import run_serious_ppo_preflight  # noqa: E402


def main() -> int:
    report = run_serious_ppo_preflight()
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
