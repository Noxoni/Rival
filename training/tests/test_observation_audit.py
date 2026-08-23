import numpy as np

from training.rival_training.observation_audit import (
    FEATURE_GROUPS,
    _align_1v1_opponent_slot,
)


def test_feature_groups_cover_432_contract_once() -> None:
    indices = [index for _, start, end in FEATURE_GROUPS for index in range(start, end)]
    assert indices == list(range(432))


def test_opponent_slot_alignment_moves_whole_block_only() -> None:
    live = np.zeros(432, dtype=np.float32)
    training = np.zeros(432, dtype=np.float32)
    live[279 + 2 * 51 : 279 + 3 * 51] = 2
    training[279 : 279 + 51] = np.arange(1, 52, dtype=np.float32)
    aligned = _align_1v1_opponent_slot(training, live)
    assert np.array_equal(aligned[279 + 2 * 51 : 279 + 3 * 51], np.arange(1, 52))
    assert np.count_nonzero(aligned[279 : 279 + 2 * 51]) == 0
