"""Exact temporal schedulers and compositor for Milestone 08."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


STRATEGIC_SCHEDULER_VERSION = "RivalStrategicDelay8V1"
MECHANICS_SCHEDULER_VERSION = "RivalMechanicsDelay4V1"
COMPOSITOR_VERSION = "RivalDualRateCompositorV1"
STRATEGIC_TICKS = 8
MECHANICS_TICKS = 4
PASS_INDEX = 0
MECHANICS_ACTION_COUNT = 69
APPENDED_GLOBAL_START = 90
APPENDED_GLOBAL_END = 157


def _controller(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32).reshape(-1)
    if result.shape != (8,):
        raise ValueError(f"Controller row must have shape (8,), got {result.shape}")
    return result.copy()


class StrategicWindowScheduler:
    """Emit the frozen live-Wisp eight-tick delayed window exactly."""

    def __init__(self, initial: np.ndarray | None = None) -> None:
        self.previous = _controller(
            np.zeros(8, dtype=np.float32) if initial is None else initial
        )
        self._pending: deque[np.ndarray] = deque()
        self.decision_count = 0

    def select(self, selected: np.ndarray) -> np.ndarray:
        if self._pending:
            raise RuntimeError("Strategic decision arrived before its prior window ended")
        new = _controller(selected)
        window = np.stack([self.previous] * 5 + [new] * 3)
        self._pending.extend(row.copy() for row in window)
        self.previous = new
        self.decision_count += 1
        return window

    def take(self, ticks: int = MECHANICS_TICKS) -> np.ndarray:
        if ticks < 1 or len(self._pending) < ticks:
            raise RuntimeError(
                f"Strategic scheduler has {len(self._pending)} pending ticks, requested {ticks}"
            )
        return np.stack([self._pending.popleft() for _ in range(ticks)])

    @property
    def pending_ticks(self) -> int:
        return len(self._pending)


@dataclass(frozen=True)
class CompositeWindow:
    controllers: np.ndarray
    sources: tuple[str, ...]
    mechanics_choice: int
    global_action_index: int | None

    @property
    def override_selected(self) -> bool:
        return self.mechanics_choice != PASS_INDEX


class DualRateCompositor:
    """Overlay a four-tick mechanics choice without pausing strategic time."""

    def __init__(self, initial: np.ndarray | None = None) -> None:
        self.last_emitted = _controller(
            np.zeros(8, dtype=np.float32) if initial is None else initial
        )

    def compose(
        self,
        strategic: np.ndarray,
        mechanics_choice: int,
        mechanics_controller: np.ndarray | None,
    ) -> CompositeWindow:
        strategic_rows = np.asarray(strategic, dtype=np.float32)
        if strategic_rows.shape != (MECHANICS_TICKS, 8):
            raise ValueError(
                f"Strategic subwindow must be (4, 8), got {strategic_rows.shape}"
            )
        if mechanics_choice == PASS_INDEX:
            if mechanics_controller is not None:
                raise ValueError("PASS cannot supply a mechanics controller")
            output = strategic_rows.copy()
            sources = ("strategic",) * MECHANICS_TICKS
            global_index = None
        else:
            if not 1 <= mechanics_choice < MECHANICS_ACTION_COUNT:
                raise IndexError(f"Mechanics choice outside [0, 68]: {mechanics_choice}")
            if mechanics_controller is None:
                raise ValueError("An override choice requires a controller row")
            selected = _controller(mechanics_controller)
            output = np.stack([self.last_emitted, selected, selected, selected])
            sources = (
                "mechanics_delay_previous",
                "mechanics_override",
                "mechanics_override",
                "mechanics_override",
            )
            global_index = APPENDED_GLOBAL_START + mechanics_choice - 1
        self.last_emitted = output[-1].copy()
        return CompositeWindow(output, sources, mechanics_choice, global_index)


def dual_rate_metadata() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "compositor_version": COMPOSITOR_VERSION,
        "strategic_scheduler_version": STRATEGIC_SCHEDULER_VERSION,
        "mechanics_scheduler_version": MECHANICS_SCHEDULER_VERSION,
        "strategic_ticks": STRATEGIC_TICKS,
        "strategic_window": ["previous"] * 5 + ["selected"] * 3,
        "mechanics_ticks": MECHANICS_TICKS,
        "mechanics_override_window": [
            "previous_emitted",
            "selected",
            "selected",
            "selected",
        ],
        "mechanics_actions": {
            "count": MECHANICS_ACTION_COUNT,
            "pass": PASS_INDEX,
            "outputs_1_through_68_map_to_global": [
                APPENDED_GLOBAL_START,
                APPENDED_GLOBAL_END,
            ],
        },
    }
