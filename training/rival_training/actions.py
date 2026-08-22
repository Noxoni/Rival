"""Mechanics-capable discrete actions with an immutable Wisp prefix."""

from __future__ import annotations

from typing import Any

import numpy as np
from rlgym.api import ActionParser, AgentID
from rlgym.rocket_league.api import GameState
from rlgym_tools.rocket_league.action_parsers.advanced_lookup_table_action import (
    AdvancedLookupTableAction,
)

from .wisp_actions import (
    CONTROLLER_FIELDS,
    WISP_ACTION_COUNT,
    action_table_fingerprint,
    build_wisp_action_table,
)


ACTION_TABLE_VERSION = "RivalExpandedActionV1"
CADENCE_TICKS = {"legacy8": 8, "mechanics4": 4}


def build_advanced_candidate_table() -> np.ndarray:
    parser = AdvancedLookupTableAction(
        torque_subdivisions=3,
        flip_bins=16,
        include_stalls=True,
    )
    return np.asarray(parser._lookup_table, dtype=np.float32)  # noqa: SLF001


def build_expanded_action_table() -> np.ndarray:
    """Append unique advanced actions after the exact 90-row Wisp table."""
    wisp = build_wisp_action_table()
    advanced = build_advanced_candidate_table()
    seen = {tuple(row.tolist()) for row in wisp}
    appended: list[np.ndarray] = []
    for row in advanced:
        key = tuple(row.tolist())
        if key not in seen:
            appended.append(row)
            seen.add(key)
    table = np.concatenate((wisp, np.asarray(appended, dtype=np.float32)), axis=0)
    if not np.array_equal(table[:WISP_ACTION_COUNT], wisp):
        raise AssertionError("Expanded table changed the frozen Wisp prefix")
    if len(np.unique(table, axis=0)) != len(table):
        raise AssertionError("Expanded action table contains duplicate rows")
    return np.ascontiguousarray(table, dtype=np.float32)


def action_metadata() -> dict[str, Any]:
    wisp = build_wisp_action_table()
    advanced = build_advanced_candidate_table()
    expanded = build_expanded_action_table()
    return {
        "schema_version": 1,
        "action_table_version": ACTION_TABLE_VERSION,
        "controller_fields": list(CONTROLLER_FIELDS),
        "serialization": "row-major little-endian float32 bytes",
        "wisp_prefix_count": len(wisp),
        "advanced_candidate_count": len(advanced),
        "appended_unique_count": len(expanded) - len(wisp),
        "expanded_action_count": len(expanded),
        "wisp_prefix_sha256": action_table_fingerprint(wisp),
        "expanded_table_sha256": action_table_fingerprint(expanded),
        "advanced_parameters": {
            "torque_subdivisions": 3,
            "flip_bins": 16,
            "include_stalls": True,
        },
        "cadence_modes": CADENCE_TICKS,
    }


class RivalActionParser(ActionParser[AgentID, np.ndarray, np.ndarray, GameState, tuple[str, int]]):
    """Discrete action parser with live-Wisp X mirroring and selectable cadence."""

    def __init__(self, cadence: str = "mechanics4") -> None:
        if cadence not in CADENCE_TICKS:
            raise ValueError(f"Unknown cadence {cadence!r}; expected {tuple(CADENCE_TICKS)}")
        self.cadence = cadence
        self.repeats = CADENCE_TICKS[cadence]
        self.lookup_table = build_expanded_action_table()

    def get_action_space(self, agent: AgentID) -> tuple[str, int]:
        return "discrete", len(self.lookup_table)

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        shared_info["previous_actions"] = {
            agent: np.zeros(8, dtype=np.float32) for agent in agents
        }
        shared_info["cadence_ticks"] = self.repeats

    def parse_actions(
        self,
        actions: dict[AgentID, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, np.ndarray]:
        parsed: dict[AgentID, np.ndarray] = {}
        previous = shared_info.setdefault("previous_actions", {})
        for agent, raw_index in actions.items():
            index_array = np.asarray(raw_index).reshape(-1)
            if index_array.size != 1:
                raise ValueError(f"Expected one discrete action for {agent}, got {raw_index}")
            index = int(index_array[0])
            if index < 0 or index >= len(self.lookup_table):
                raise IndexError(f"Action index {index} outside [0, {len(self.lookup_table)})")

            world_action = self.lookup_table[index].copy()
            car = state.cars[agent]
            mirror_x = (car.team_num == 1) != (float(car.physics.position[0]) < 0)
            if mirror_x:
                world_action[[1, 3, 4]] *= -1
            previous[agent] = world_action.copy()
            parsed[agent] = np.repeat(world_action[None, :], self.repeats, axis=0)
        return parsed
