from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from action_parser import XMirroredActionParser
from backend.gamestate.player import Player
from backend.gamestate.team import Team
from backend.gamestate.vec import Vec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_EXPANDED_TABLE_SHA256 = (
    "38ed338273ae09736d81d3e7fb69c45d91397e45d50f1ae97101e3737c0ecd20"
)


def test_default_runtime_action_parser_remains_frozen_wisp() -> None:
    parser = XMirroredActionParser()
    assert len(parser.actions) == 90
    assert not parser.allow_all_actions
    assert not parser.legacy_only


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


def test_m07_legacy_only_candidate_uses_exact_wisp_mask_and_hard_masks_suffix() -> None:
    path = REPOSITORY_ROOT / "bot/models/RIVAL_ACTIONS_V1.npy"
    production = XMirroredActionParser()
    candidate = XMirroredActionParser(path, legacy_only=True)
    player = Player(team=Team.BLUE, pos=Vec(100, 0, 17))
    player.boost = 0
    player.is_on_ground = False
    player.has_flipped = True
    player.has_double_jumped = True

    production_mask = production.get_action_mask(player, None)
    candidate_mask = candidate.get_action_mask(player, None)

    assert len(candidate_mask) == 158
    assert np.array_equal(candidate_mask[:90].numpy(), production_mask.numpy())
    assert not candidate_mask[90:].any()
    with pytest.raises(IndexError, match="legacy-only"):
        candidate.get_action(90, player, None)


def test_m07_legacy_only_parser_preserves_x_mirroring_for_legacy_actions() -> None:
    path = REPOSITORY_ROOT / "bot/models/RIVAL_ACTIONS_V1.npy"
    production = XMirroredActionParser()
    candidate = XMirroredActionParser(path, legacy_only=True)
    regimes = (
        (Team.BLUE, -100.0),
        (Team.BLUE, 100.0),
        (Team.ORANGE, -100.0),
        (Team.ORANGE, 100.0),
    )
    for team, x_position in regimes:
        player = Player(team=team, pos=Vec(x_position, 0, 17))
        for index in range(90):
            assert np.array_equal(
                production.get_action(index, player, None).get_np(),
                candidate.get_action(index, player, None).get_np(),
            )


def test_action_parser_rejects_conflicting_full_and_legacy_masks() -> None:
    path = REPOSITORY_ROOT / "bot/models/RIVAL_ACTIONS_V1.npy"
    with pytest.raises(ValueError, match="mutually exclusive"):
        XMirroredActionParser(path, allow_all_actions=True, legacy_only=True)
