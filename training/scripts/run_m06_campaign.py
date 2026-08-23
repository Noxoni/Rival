"""Run or resume one coherent Milestone 06 stage to its safe boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.campaign import run_campaign_stage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("stage_a", "stage_b", "stage_c", "stage_d"), required=True
    )
    parser.add_argument("--appended-offset", type=float, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    report = run_campaign_stage(
        args.stage,
        appended_logit_offset=args.appended_offset,
        resume_directory=args.resume,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["health"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
