from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.transfer_diagnostics import (  # noqa: E402
    export_zero_step_diagnostic_actor,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor-output",
        type=Path,
        default=REPOSITORY_ROOT / "training/artifacts/milestone07/zero_step_actor.ts",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=REPOSITORY_ROOT / "training/results/milestone07/zero_step_export.json",
    )
    args = parser.parse_args()
    report = export_zero_step_diagnostic_actor(args.actor_output)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.report_output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
