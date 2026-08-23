"""Opt-in live dual-rate compositor used only by Milestone 08 diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.gamestate.action import Action


LIVE_DUAL_RATE_VERSION = "RivalLiveDualRateV1"
PASS_INDEX = 0
MECHANICS_ACTION_COUNT = 69
APPENDED_GLOBAL_START = 90
APPENDED_GLOBAL_END = 157


def _copy(action: Action) -> Action:
    return Action(action.get_np().copy())


@dataclass(frozen=True)
class LiveMechanicsDecision:
    choice: int
    global_action_index: int | None
    previous: Action
    selected: Action | None

    @property
    def override_selected(self) -> bool:
        return self.choice != PASS_INDEX


class LiveMechanicsWindow:
    """Emit [previous, selected, selected, selected] for an override.

    PASS is deliberately represented as no owner: callers continue emitting the
    unmodified strategic row.  This makes disabled and forced-PASS modes share the
    exact production strategic code path rather than a numerical reconstruction.
    """

    def __init__(self) -> None:
        self.last_emitted = Action()
        self.current = LiveMechanicsDecision(PASS_INDEX, None, Action(), None)
        self.position = 0

    def reset(self) -> None:
        self.last_emitted = Action()
        self.current = LiveMechanicsDecision(PASS_INDEX, None, Action(), None)
        self.position = 0

    def begin(self, choice: int, selected: Action | None) -> LiveMechanicsDecision:
        if not 0 <= int(choice) < MECHANICS_ACTION_COUNT:
            raise IndexError(f"Mechanics choice outside [0, 68]: {choice}")
        if choice == PASS_INDEX:
            if selected is not None:
                raise ValueError("PASS cannot provide a mechanics controller")
            global_index = None
        else:
            if selected is None:
                raise ValueError("Mechanics override requires a controller")
            global_index = APPENDED_GLOBAL_START + int(choice) - 1
        self.current = LiveMechanicsDecision(
            int(choice),
            global_index,
            _copy(self.last_emitted),
            None if selected is None else _copy(selected),
        )
        self.position = 0
        return self.current

    def emit(self, strategic: Action) -> tuple[Action, str]:
        if not self.current.override_selected:
            output = _copy(strategic)
            source = "strategic_pass_through"
        elif self.position == 0:
            output = _copy(self.current.previous)
            source = "mechanics_delay_previous"
        else:
            assert self.current.selected is not None
            output = _copy(self.current.selected)
            source = "mechanics_override"
        self.last_emitted = _copy(output)
        self.position = min(self.position + 1, 3)
        return output, source

    def trace(self, strategic_rows: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        values = np.asarray(strategic_rows, dtype=np.float32)
        if values.shape != (4, 8):
            raise ValueError(f"Expected four strategic rows, got {values.shape}")
        outputs = []
        sources = []
        for row in values:
            output, source = self.emit(Action(row))
            outputs.append(output.get_np().copy())
            sources.append(source)
        return np.stack(outputs), tuple(sources)
