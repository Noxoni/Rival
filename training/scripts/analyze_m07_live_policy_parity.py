from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.live_policy_parity import (  # noqa: E402
    build_live_policy_parity_report,
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
        default=REPOSITORY_ROOT / "training/results/milestone07/live_policy_parity.json",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = build_live_policy_parity_report(args.matrix, device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": report["status"],
        "samples": report["corpus"]["samples"],
        "zero_step_vs_frozen": report["zero_step_vs_frozen"],
        "trained_20m_vs_frozen": {
            key: report["trained_20m_vs_frozen"][key]
            for key in (
                "max_abs_first_90_logit_drift",
                "mean_abs_first_90_logit_drift",
                "masked_top1_agreement",
                "disagreement_count",
            )
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
