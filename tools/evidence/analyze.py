from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .events import DETECTOR_VERSION, DetectorParameters, detect_events
from .fixtures import curate_top_fixtures
from .io import load_sessions
from .report import markdown_report
from .session import utc_now, write_json


def build_report(inputs: list[Path], params: DetectorParameters | None = None) -> tuple[dict[str, Any], Any]:
    params = params or DetectorParameters()
    sessions = load_sessions(inputs)
    events = detect_events(sessions, params)
    counts = Counter(event["class"] for event in events)
    report = {
        "report_schema_version": 1,
        "generated_utc": utc_now(),
        "candidate_only": True,
        "detector_version": DETECTOR_VERSION,
        "detector_parameters": params.to_record(),
        "session_count": len(sessions),
        "decision_record_count": sum(len(session.decisions) for session in sessions),
        "event_counts": {
            event_class: counts[event_class]
            for event_class in (
                "resource_stressed_aerial",
                "boost_detour_possession_loss",
                "apparent_vs_actual_challenge",
            )
        },
        "sessions": [
            {
                "session_id": session.session_id,
                "source": session.source,
                "opponent": session.opponent,
                "raw_path": str(session.raw_path),
                "raw_sha256": session.raw_sha256,
                "decision_record_count": len(session.decisions),
                "warnings": session.warnings,
            }
            for session in sessions
        ],
        "events": events,
    }
    return report, sessions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze Rival telemetry schema v1/v2")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evidence/reports/current"))
    parser.add_argument("--format", choices=("json", "markdown", "both"), default="both")
    parser.add_argument("--curate", type=Path)
    args = parser.parse_args(argv)

    report, sessions = build_report(args.inputs)
    if args.curate:
        fixture_paths = curate_top_fixtures(report["events"], sessions, args.curate)
        report["curated_fixtures"] = [str(path) for path in fixture_paths]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.format in {"json", "both"}:
        write_json(args.output_dir / "candidate_events.json", report)
    if args.format in {"markdown", "both"}:
        (args.output_dir / "candidate_events.md").write_text(
            markdown_report(report, sessions),
            encoding="utf-8",
            newline="\n",
        )
    print(
        f"Analyzed {report['session_count']} sessions, {report['decision_record_count']} decisions, "
        f"{len(report['events'])} candidate events"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
