from __future__ import annotations

import numpy as np
from rlgym.rocket_league.action_parsers import LookupTableAction

from rival_training.actions import (
    RivalActionParser,
    action_metadata,
    build_expanded_action_table,
)
from rival_training.wisp_actions import build_wisp_action_table


def test_exact_wisp_prefix_count_order_and_values() -> None:
    wisp = build_wisp_action_table()
    upstream_lookup = LookupTableAction.make_lookup_table().astype(np.float32)
    expanded = build_expanded_action_table()

    assert wisp.shape == (90, 8)
    assert np.array_equal(wisp, upstream_lookup)
    assert np.array_equal(expanded[:90], wisp)


def test_expanded_table_is_unique_and_fingerprinted() -> None:
    table = build_expanded_action_table()
    metadata = action_metadata()

    assert table.shape == (158, 8)
    assert len(np.unique(table, axis=0)) == 158
    assert metadata["appended_unique_count"] == 68
    assert (
        metadata["wisp_prefix_sha256"]
        == "86baa15c48c42c497f3ea0fe62efeb49e4a8241cb3191957822e453cd2d0b655"
    )
    assert (
        metadata["expanded_table_sha256"]
        == "38ed338273ae09736d81d3e7fb69c45d91397e45d50f1ae97101e3737c0ecd20"
    )


def test_both_cadence_modes_are_available() -> None:
    assert RivalActionParser("legacy8").repeats == 8
    assert RivalActionParser("mechanics4").repeats == 4
