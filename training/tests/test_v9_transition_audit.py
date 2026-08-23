from __future__ import annotations

import numpy as np

from rival_training.v9_transition_audit import (
    _orientation_error_degrees,
    _physics_error,
    select_windows,
)


class _Physics:
    def __init__(self, position, velocity=(0, 0, 0)) -> None:
        self.position = np.asarray(position, dtype=np.float32)
        self.linear_velocity = np.asarray(velocity, dtype=np.float32)
        self.angular_velocity = np.zeros(3, dtype=np.float32)
        self.rotation_mtx = np.eye(3, dtype=np.float32)


def test_orientation_error_zero_and_quarter_turn() -> None:
    identity = np.eye(3, dtype=np.float32)
    quarter = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    assert _orientation_error_degrees(identity, identity) == 0.0
    assert abs(_orientation_error_degrees(identity, quarter) - 90.0) < 1e-6


def test_physics_error_uses_euclidean_physical_units() -> None:
    first = _Physics((3, 4, 0), (10, 0, 0))
    second = _Physics((0, 0, 0), (0, 0, 0))
    error = _physics_error(first, second)
    assert error["position_uu"] == 5.0
    assert error["linear_velocity_uu_per_s"] == 10.0
    assert error["orientation_degrees"] == 0.0


def test_empty_window_selection_is_safe() -> None:
    assert select_windows([], maximum=32) == []
