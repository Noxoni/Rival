from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.natural_v4 import build_natural_analysis, markdown_summary  # noqa: E402
from tools.evidence.session import write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate consequence-aware Rival v4.1 natural telemetry"
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=REPOSITORY_ROOT / "evidence" / "raw",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    report = build_natural_analysis(args.batch, args.raw_root)
    write_json(args.output_json, report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(
        markdown_summary(report),
        encoding="utf-8",
        newline="\n",
    )
    print(
        f"Analyzed {report['aggregate']['match_count']} matches, "
        f"{report['aggregate']['decision_count']} decisions; "
        f"highest priority: {report['highest_priority_pattern']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
