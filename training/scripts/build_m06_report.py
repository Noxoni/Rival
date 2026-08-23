"""Build compact committed Milestone 06 preflight or stage evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.reporting import (  # noqa: E402
    build_preflight_report,
    build_stage_report,
    write_results_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("preflight")
    stage = subparsers.add_parser("stage")
    stage.add_argument("--summary", type=Path, required=True)
    stage.add_argument("--label", required=True)
    stage.add_argument("--next-stage")
    stage.add_argument("--next-offset", type=float)
    stage.add_argument("--rlbot", type=Path)
    args = parser.parse_args()
    if args.mode == "preflight":
        report = build_preflight_report()
    else:
        report = build_stage_report(
            args.summary,
            label=args.label,
            next_stage=args.next_stage,
            next_offset=args.next_offset,
            rlbot_path=args.rlbot,
        )
    markdown = write_results_markdown()
    print(json.dumps(report, indent=2))
    print(f"wrote {markdown.relative_to(REPOSITORY_ROOT).as_posix()}")
    return 0 if report["status"] not in {"failed", "rejected_at_evaluation_boundary"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
