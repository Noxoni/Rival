from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from action_parser import XMirroredActionParser


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXPANDED_TABLE_SHA256 = (
    "38ed338273ae09736d81d3e7fb69c45d91397e45d50f1ae97101e3737c0ecd20"
)


def test_default_runtime_action_parser_remains_frozen_wisp() -> None:
    parser = XMirroredActionParser()
    assert len(parser.actions) == 90
    assert not parser.allow_all_actions


def test_candidate_runtime_loads_exact_expanded_table_and_enables_all_actions() -> None:
    path = REPOSITORY_ROOT / "bot/models/RIVAL_ACTIONS_V1.npy"
    table = np.load(path, allow_pickle=False)
    logical_hash = hashlib.sha256(
        np.asarray(table, dtype="<f4").tobytes(order="C")
    ).hexdigest()
    parser = XMirroredActionParser(path, allow_all_actions=True)
    assert table.shape == (158, 8)
    assert logical_hash == EXPECTED_EXPANDED_TABLE_SHA256
    assert len(parser.actions) == 158
    assert parser.get_action_mask(None, None).all()
