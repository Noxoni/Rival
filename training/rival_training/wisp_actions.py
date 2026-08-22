"""Dependency-light definition of the frozen Wisp controller-action table.

This module intentionally depends only on NumPy so the production RLBot environment
can compare its live ``DefaultAction`` rows against the training prefix without
installing RLGym into the production virtual environment.
"""

from __future__ import annotations

import hashlib

import numpy as np


WISP_ACTION_COUNT = 90
CONTROLLER_FIELDS = (
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "jump",
    "boost",
    "handbrake",
)


def build_wisp_action_table() -> np.ndarray:
    """Return Wisp's exact 90 rows in production order as float32."""
    actions: list[list[float]] = []
    binary = (0, 1)
    ternary = (-1, 0, 1)

    for throttle in ternary:
        for steer in ternary:
            for boost in binary:
                for handbrake in binary:
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append(
                        [throttle, steer, 0, steer, 0, 0, boost, handbrake]
                    )

    for pitch in ternary:
        for yaw in ternary:
            for roll in ternary:
                for jump in binary:
                    for boost in binary:
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll and roll == jump and jump == 0:
                            continue
                        handbrake = int(
                            jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        )
                        actions.append(
                            [boost, yaw, pitch, yaw, roll, jump, boost, handbrake]
                        )

    table = np.asarray(actions, dtype=np.float32)
    if table.shape != (WISP_ACTION_COUNT, len(CONTROLLER_FIELDS)):
        raise AssertionError(f"Unexpected Wisp action shape: {table.shape}")
    return table


def action_table_fingerprint(table: np.ndarray) -> str:
    """Hash a table's canonical little-endian float32 row-major representation."""
    canonical = np.ascontiguousarray(table, dtype="<f4")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()
