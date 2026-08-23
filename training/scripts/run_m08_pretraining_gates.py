"""Run deterministic Milestone 08 software and fallback gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.rival_training.milestone08_gates import (  # noqa: E402
    write_pretraining_gate_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPOSITORY_ROOT / "training/results/milestone07/transfer_matrix.json",
    )
    parser.add_argument(
        "--observation-report",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "training/results/milestone08/observation_contract_v2.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT / "training/results/milestone08/pretraining_gates.json"
        ),
    )
    args = parser.parse_args()
    report = write_pretraining_gate_report(
        args.matrix, args.observation_report, args.output
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": report["checks"],
                "fallback": report["fallback_invariant"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
