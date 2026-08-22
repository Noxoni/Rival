"""Compare the training prefix directly with production ``DefaultAction`` rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "bot"))
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from action_parser import DefaultAction  # noqa: E402
from rival_training.wisp_actions import (  # noqa: E402
    action_table_fingerprint,
    build_wisp_action_table,
)


def build_report() -> dict:
    production = np.stack(
        [action.get_np() for action in DefaultAction().actions]
    ).astype(np.float32)
    training = build_wisp_action_table()
    return {
        "schema_version": 1,
        "proof": "direct import of bot/action_parser.py DefaultAction",
        "production_shape": list(production.shape),
        "training_prefix_shape": list(training.shape),
        "exact_array_equal": bool(np.array_equal(production, training)),
        "maximum_absolute_error": float(np.max(np.abs(production - training))),
        "production_sha256": action_table_fingerprint(production),
        "training_prefix_sha256": action_table_fingerprint(training),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "training/results/action_prefix_proof.json",
    )
    args = parser.parse_args()
    report = build_report()
    if not report["exact_array_equal"]:
        raise SystemExit(f"Wisp prefix parity failed: {report}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
