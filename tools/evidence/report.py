from __future__ import annotations

from collections import Counter
from typing import Any

from .io import EvidenceSession


def markdown_report(report: dict[str, Any], sessions: list[EvidenceSession]) -> str:
    persisted_counts = Counter(event["class"] for event in report["events"])
    counts = report.get("event_counts", persisted_counts)
    total_count = report.get("total_candidate_event_count", len(report["events"]))
    lines = [
        "# Rival Milestone 02 Candidate Evidence Report",
        "",
        "> Detector findings are ranking candidates for review, not confirmed gameplay defects.",
        "",
        f"- Detector: `{report['detector_version']}`",
        f"- Sessions: {report['session_count']}",
        f"- Decision records: {report['decision_record_count']}",
        f"- Candidate events detected: {total_count}",
        f"- Candidate events persisted: {len(report['events'])}",
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

    lines.extend(["", "## Highest-ranked candidates by class", ""])
    for event_class in (
        "resource_stressed_aerial",
        "boost_detour_possession_loss",
        "apparent_vs_actual_challenge",
    ):
        lines.extend([f"### `{event_class}`", ""])
        class_events = [
            event for event in report["events"] if event["class"] == event_class
        ][:5]
        if not class_events:
            lines.extend(["No candidates detected.", ""])
            continue
        for event in class_events:
            explanation = "; ".join(event["ranking_explanation"])
            lines.extend(
                [
                    f"#### {event['event_id']}",
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
