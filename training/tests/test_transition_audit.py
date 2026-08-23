import numpy as np

from training.rival_training.transition_audit import _orientation_error_degrees


def test_orientation_error_is_zero_for_identical_basis() -> None:
    basis = np.eye(3, dtype=np.float32)
    assert _orientation_error_degrees(basis, basis) == 0.0


def test_orientation_error_reports_quarter_turn() -> None:
    first = np.eye(3, dtype=np.float32)
    second = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    assert abs(_orientation_error_degrees(first, second) - 90.0) < 1e-6
