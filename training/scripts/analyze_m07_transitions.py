"""Write the Milestone 07 short-horizon transition audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from training.rival_training.transition_audit import (  # noqa: E402
    write_transition_audit_report,
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
        default=REPOSITORY_ROOT / "training/results/milestone07/transition_audit.json",
    )
    parser.add_argument("--maximum-windows", type=int, default=32)
    args = parser.parse_args()
    report = write_transition_audit_report(
        args.matrix, args.output, maximum_windows=args.maximum_windows
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "selection": report["selection"],
                "contact_free_primary": report["contact_free_primary"],
                "materiality_rule": report["materiality_rule"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
