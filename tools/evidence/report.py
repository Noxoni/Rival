from __future__ import annotations

from collections import Counter
from typing import Any

from .io import EvidenceSession


def markdown_report(report: dict[str, Any], sessions: list[EvidenceSession]) -> str:
    counts = Counter(event["class"] for event in report["events"])
    lines = [
        "# Rival Milestone 02 Candidate Evidence Report",
        "",
        "> Detector findings are ranking candidates for review, not confirmed gameplay defects.",
        "",
        f"- Detector: `{report['detector_version']}`",
        f"- Sessions: {report['session_count']}",
        f"- Decision records: {report['decision_record_count']}",
        f"- Candidate events: {len(report['events'])}",
        "",
        "## Candidate counts",
        "",
        "| Class | Count |",
        "| --- | ---: |",
    ]
    for event_class in (
        "resource_stressed_aerial",
        "boost_detour_possession_loss",
        "apparent_vs_actual_challenge",
    ):
        lines.append(f"| `{event_class}` | {counts[event_class]} |")

    lines.extend(["", "## Sessions", "", "| Session | Source | Opponent | Decisions | Warnings |", "| --- | --- | --- | ---: | ---: |"])
    for session in sessions:
        lines.append(
            f"| `{session.session_id}` | {session.source} | {session.opponent} | "
            f"{len(session.decisions)} | {len(session.warnings)} |"
        )

    lines.extend(["", "## Highest-ranked candidates", ""])
    for event in report["events"][:10]:
        explanation = "; ".join(event["ranking_explanation"])
        lines.extend(
            [
                f"### {event['event_id']} — `{event['class']}`",
                "",
                f"- Session/time: `{event['session_id']}` at {event['anchor_game_time']:.3f}s",
                f"- Opponent/source: {event['opponent']} / {event['source']}",
                f"- Ranking score: {event['ranking_score']:.3f}",
                f"- Outcome: `{event['outcome'].get('next_touch', 'unknown')}` next touch",
                f"- Why ranked: {explanation}",
                "",
            ]
        )
    if not report["events"]:
        lines.append("No candidate events were detected with the recorded parameters.")
        lines.append("")

    warnings = [
        f"{session.session_id}: {warning}"
        for session in sessions
        for warning in session.warnings
    ]
    lines.extend(["## Integrity warnings", ""])
    if warnings:
        lines.extend(f"- `{warning}`" for warning in warnings)
    else:
        lines.append("No loader integrity warnings.")
    lines.append("")
    return "\n".join(lines)
