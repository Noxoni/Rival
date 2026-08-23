"""Run the Milestone 08 held-live Wisp432ContractV2 gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.rival_training.observation_audit import (  # noqa: E402
    write_observation_audit_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPOSITORY_ROOT / "training/results/milestone07/transfer_matrix.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "training/results/milestone08/observation_contract_v2.json"
        ),
    )
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    report = write_observation_audit_report(
        args.matrix, args.output, max_samples=args.max_samples
    )
    summary = {
        "status": report["status"],
        "samples": report["corpus"]["samples"],
        "policy": report["frozen_wisp_policy_effect"]["live_vs_training_style"],
        "gate": report["observation_gate"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["observation_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
