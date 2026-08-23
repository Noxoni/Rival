"""Run the M08 mechanics-only PPO campaign to one authorized boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.m08_campaign import run_m08_training_boundary  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=int,
        choices=(500_000, 1_000_000, 2_000_000, 5_000_000),
        required=True,
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--workers", type=int, choices=(48, 56, 64), default=56)
    parser.add_argument("--worker-transition-evidence", type=Path)
    args = parser.parse_args()
    report = run_m08_training_boundary(
        args.target,
        resume_directory=args.resume,
        worker_count=args.workers,
        worker_transition_evidence=args.worker_transition_evidence,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["health"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
