"""Sign-preserving credit-window diagnostics for M10.9."""

from __future__ import annotations

from typing import Any, Iterable

from .v10_8_credit import (
    WINDOWS,
    aggregate_credit_diagnostics as _aggregate_v10_8,
    credit_assignment_window_diagnostics as _credit_v10_8,
)


def credit_assignment_window_diagnostics(
    *,
    scaled_advantages,
    **kwargs,
) -> dict[str, Any]:
    """Reuse the frozen trajectory cohorts while naming scale-only values exactly."""

    report = _credit_v10_8(
        normalized_advantages=scaled_advantages,
        **kwargs,
    )
    report["version"] = "RivalM10_9CreditAssignmentWindowsV1"
    for cohort in ("successful_first_contact", "failed_timeout_like"):
        for window in WINDOWS:
            metrics = report[cohort][window]["metrics"]
            metrics["scaled_advantage"] = metrics.pop("normalized_advantage")
    return report


def aggregate_credit_diagnostics(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    report = _aggregate_v10_8(rows)
    report["version"] = "RivalM10_9AggregateCreditAssignmentWindowsV1"
    return report
