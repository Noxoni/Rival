"""Mechanics-capable discrete actions with an immutable Wisp prefix."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
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
from .dual_rate import (
    APPENDED_GLOBAL_START,
    DualRateCompositor,
    MECHANICS_ACTION_COUNT,
    PASS_INDEX,
    StrategicWindowScheduler,
)
from .observations import Wisp432ContractV2
from .teacher import FrozenWispReference


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


def action_family(index: int, table: np.ndarray | None = None) -> str:
    """Classify appended controls generically for diagnostics, not rewards."""
    if index < WISP_ACTION_COUNT:
        return "legacy_wisp"
    selected_table = build_expanded_action_table() if table is None else table
    row = selected_table[index]
    if row[5] > 0.5:
        return "appended_jump_dodge_control"
    if row[6] > 0.5:
        return "appended_boosted_air_control"
    if np.any(np.abs(row[2:5]) > 0.5):
        return "appended_unboosted_air_control"
    return "appended_ground_recovery_control"


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


def wisp_legal_action_mask(car) -> np.ndarray:
    """Pure-RLGym reproduction of frozen ``DefaultAction.get_action_mask``."""
    table = build_wisp_action_table()
    num_ground_actions = 24
    jump = table[:, 5] > 0.5
    boost = table[:, 6] > 0.5
    ground = np.arange(WISP_ACTION_COUNT) < num_ground_actions
    air = (np.arange(WISP_ACTION_COUNT) > num_ground_actions) & ~jump
    ground_air_alias = ground & (
        (table[:, 0] == table[:, 6])
        & ((table[:, 3] != 0) == (table[:, 7] > 0.5))
    )
    air |= ground_air_alias
    mask = ground.copy() if car.on_ground else air.copy()
    if float(car.boost_amount) == 0.0:
        mask &= ~boost
    has_flip_or_jump = car.on_ground or (
        not car.has_flipped
        and not car.has_double_jumped
        and float(car.air_time_since_jump) < 1.25
    )
    position = np.asarray(car.physics.position)
    up = np.asarray(car.physics.up)
    turtled = (
        not car.on_ground
        and float(up[2]) < -0.8
        and abs(float(car.physics.linear_velocity[2])) < 50.0
        and float(position[2]) < 50.0
    )
    if has_flip_or_jump or turtled:
        mask |= jump
    if not mask.any():
        raise RuntimeError("Frozen Wisp action mask unexpectedly removed every action")
    return mask


class DualRateActionParser(
    ActionParser[AgentID, np.ndarray, np.ndarray, GameState, tuple[str, int]]
):
    """Frozen Wisp strategic policy plus a 69-choice mechanics overlay."""

    def __init__(
        self,
        *,
        mechanics_disabled: bool = False,
        force_pass: bool = False,
        anchor_team: int | None = None,
        seed: int = 20260823,
    ) -> None:
        self.mechanics_disabled = bool(mechanics_disabled)
        self.force_pass = bool(force_pass)
        self.anchor_team = anchor_team
        self.lookup_table = build_expanded_action_table()
        self.strategic_observations = Wisp432ContractV2(seed=seed)
        self.strategic_model = FrozenWispReference().eval()
        for parameter in self.strategic_model.parameters():
            parameter.requires_grad_(False)
        self._strategic: dict[AgentID, StrategicWindowScheduler] = {}
        self._compositors: dict[AgentID, DualRateCompositor] = {}
        self._strategic_indices: dict[AgentID, int] = {}
        self._mechanics_decision = 0

    def get_action_space(self, agent: AgentID) -> tuple[str, int]:
        return "discrete", MECHANICS_ACTION_COUNT

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        self._strategic = {agent: StrategicWindowScheduler() for agent in agents}
        self._compositors = {agent: DualRateCompositor() for agent in agents}
        self._strategic_indices = {agent: -1 for agent in agents}
        self._mechanics_decision = 0
        shared_info["previous_actions"] = {
            agent: np.zeros(8, dtype=np.float32) for agent in agents
        }
        shared_info["cadence_ticks"] = 4
        shared_info["dual_rate_last_decisions"] = {}
        shared_info["dual_rate_last_controllers"] = {}
        self.strategic_observations.reset(agents, initial_state, shared_info)

    @staticmethod
    def _world_action(row: np.ndarray, car) -> np.ndarray:
        result = np.asarray(row, dtype=np.float32).copy()
        mirror_x = (car.team_num == 1) != (float(car.physics.position[0]) < 0)
        if mirror_x:
            result[[1, 3, 4]] *= -1
        return result

    def _select_strategic(
        self,
        agents: list[AgentID],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        observations = self.strategic_observations.build_obs(
            agents, state, shared_info
        )
        batch = torch.from_numpy(np.stack([observations[agent] for agent in agents]))
        with torch.inference_mode():
            logits = self.strategic_model(batch).cpu().numpy()
        table = build_wisp_action_table()
        masks: dict[AgentID, np.ndarray] = {}
        for row_index, agent in enumerate(agents):
            mask = wisp_legal_action_mask(state.cars[agent])
            masks[agent] = mask.copy()
            selected = int(np.argmax(np.where(mask, logits[row_index], -1e10)))
            self._strategic_indices[agent] = selected
            world = self._world_action(table[selected], state.cars[agent])
            self._strategic[agent].select(world)
        shared_info["dual_rate_last_strategic_observations"] = {
            agent: observations[agent].copy() for agent in agents
        }
        shared_info["dual_rate_last_strategic_logits"] = {
            agent: logits[index].copy() for index, agent in enumerate(agents)
        }
        shared_info["dual_rate_last_strategic_masks"] = masks

    def parse_actions(
        self,
        actions: dict[AgentID, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, np.ndarray]:
        agents = list(state.cars)
        if set(actions) != set(agents):
            raise KeyError("Dual-rate parser requires one mechanics choice per agent")
        strategic_decision = self._mechanics_decision % 2 == 0
        if strategic_decision:
            self._select_strategic(agents, state, shared_info)

        parsed: dict[AgentID, np.ndarray] = {}
        decision_metadata: dict[AgentID, dict[str, Any]] = {}
        previous_actions = shared_info.setdefault("previous_actions", {})
        for agent in agents:
            raw = np.asarray(actions[agent]).reshape(-1)
            if raw.size != 1:
                raise ValueError(f"Expected one mechanics choice for {agent}, got {raw}")
            requested = int(raw[0])
            if not 0 <= requested < MECHANICS_ACTION_COUNT:
                raise IndexError(f"Mechanics choice outside [0, 68]: {requested}")
            forced = (
                self.mechanics_disabled
                or self.force_pass
                or (
                    self.anchor_team is not None
                    and state.cars[agent].team_num == self.anchor_team
                )
            )
            choice = PASS_INDEX if forced else requested
            strategic_rows = self._strategic[agent].take(4)
            if choice == PASS_INDEX:
                mechanics_row = None
                global_index = None
            else:
                global_index = APPENDED_GLOBAL_START + choice - 1
                mechanics_row = self._world_action(
                    self.lookup_table[global_index], state.cars[agent]
                )
            composite = self._compositors[agent].compose(
                strategic_rows, choice, mechanics_row
            )
            parsed[agent] = composite.controllers
            previous_actions[agent] = composite.controllers[-1].copy()
            decision_metadata[agent] = {
                "strategic_decision": strategic_decision,
                "strategic_action_index": self._strategic_indices[agent],
                "requested_mechanics_choice": requested,
                "applied_mechanics_choice": choice,
                "global_action_index": global_index,
                "forced_pass": forced,
                "override_selected": choice != PASS_INDEX,
                "tick_sources": list(composite.sources),
            }
        shared_info["dual_rate_last_decisions"] = decision_metadata
        shared_info["dual_rate_last_controllers"] = {
            agent: rows.copy() for agent, rows in parsed.items()
        }
        self._mechanics_decision += 1
        return parsed
