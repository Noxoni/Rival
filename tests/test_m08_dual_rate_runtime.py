from __future__ import annotations

import numpy as np

from backend.gamestate.action import Action
from dual_rate_runtime import LiveMechanicsWindow


def _row(value: float) -> np.ndarray:
    return np.full(8, value, dtype=np.float32)


def test_live_forced_pass_is_exact_strategic_trace() -> None:
    runtime = LiveMechanicsWindow()
    runtime.begin(0, None)
    strategic = np.stack([_row(1), _row(2), _row(3), _row(4)])
    actual, sources = runtime.trace(strategic)
    assert np.array_equal(actual, strategic)
    assert sources == ("strategic_pass_through",) * 4


def test_live_override_uses_previous_then_three_selected() -> None:
    runtime = LiveMechanicsWindow()
    runtime.begin(0, None)
    runtime.trace(np.stack([_row(1)] * 4))
    decision = runtime.begin(1, Action(_row(7)))
    actual, sources = runtime.trace(np.stack([_row(2)] * 4))
    assert decision.global_action_index == 90
    assert np.array_equal(actual, np.stack([_row(1), _row(7), _row(7), _row(7)]))
    assert sources == (
        "mechanics_delay_previous",
        "mechanics_override",
        "mechanics_override",
        "mechanics_override",
    )


def test_live_strategic_advances_under_override_then_passes_through() -> None:
    runtime = LiveMechanicsWindow()
    runtime.begin(2, Action(_row(8)))
    runtime.trace(np.stack([_row(3)] * 4))
    runtime.begin(0, None)
    strategic = np.stack([_row(4), _row(5), _row(6), _row(7)])
    actual, _ = runtime.trace(strategic)
    assert np.array_equal(actual, strategic)
