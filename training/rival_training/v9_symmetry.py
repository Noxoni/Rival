"""Episode-stable left/right augmentation for the Rival v9 training path.

Team inversion remains part of the canonical adapters.  This optional second
transform reflects the already-team-normalized world across the YZ plane.  The
mirror bit is sampled only during environment reset and never depends on the
car's current X coordinate.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from rlgym.api import ActionParser, AgentID
from rlgym.rocket_league.api import GameState

from .v9_actions import (
    ACTION_DIM,
    RivalActionV1Parser,
    validate_physical_actions,
)
from .v9_canonical import (
    CanonicalCarV1,
    CanonicalPhysicsV1,
    RivalCanonicalStateV1,
)


SYMMETRY_VERSION = "RivalEpisodeLeftRightSymmetryV1"
WORLD_REFLECTION = np.diag(np.asarray([-1.0, 1.0, 1.0], dtype=np.float32))
AXIAL_REFLECTION = np.diag(np.asarray([1.0, -1.0, -1.0], dtype=np.float32))
LOCAL_BASIS_REFLECTION = np.diag(
    np.asarray([1.0, -1.0, 1.0], dtype=np.float32)
)
CONTROLLER_SIGN = np.asarray(
    [1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float32
)
DODGE_DIRECTION_SIGN = np.asarray([1.0, -1.0], dtype=np.float32)


def mirror_controller(actions: np.ndarray) -> np.ndarray:
    """Reflect throttle/steer/pitch/yaw/roll/buttons; the operation is involutive."""

    validated = validate_physical_actions(np.asarray(actions))
    mirrored = validated * CONTROLLER_SIGN
    if np.asarray(actions).ndim == 1:
        return np.ascontiguousarray(mirrored[0], dtype=np.float32)
    return np.ascontiguousarray(mirrored, dtype=np.float32)


def _mirror_physics(physics: CanonicalPhysicsV1) -> CanonicalPhysicsV1:
    # Position, forward, up, and linear velocity are polar vectors. Angular
    # velocity is axial and therefore receives det(S)*S. The local-right basis
    # column is also negated so the mirrored rotation remains right-handed.
    return CanonicalPhysicsV1(
        position=WORLD_REFLECTION @ physics.position,
        rotation_mtx=(
            WORLD_REFLECTION @ physics.rotation_mtx @ LOCAL_BASIS_REFLECTION
        ),
        linear_velocity=WORLD_REFLECTION @ physics.linear_velocity,
        angular_velocity=AXIAL_REFLECTION @ physics.angular_velocity,
    )


def _mirror_car(car: CanonicalCarV1) -> CanonicalCarV1:
    return replace(
        car,
        physics=_mirror_physics(car.physics),
        dodge_direction=car.dodge_direction * DODGE_DIRECTION_SIGN,
        latest_controller=mirror_controller(car.latest_controller),
    )


def _mirrored_pad_order(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reflected = np.asarray(positions, dtype=np.float32).copy()
    reflected[:, 0] *= -1.0
    order: list[int] = []
    used: set[int] = set()
    for target in np.asarray(positions, dtype=np.float32):
        errors = np.sum((reflected - target) ** 2, axis=1)
        for index in np.argsort(errors):
            candidate = int(index)
            if candidate not in used:
                if float(errors[candidate]) > 1e-6:
                    raise ValueError(
                        "Canonical boost-pad geometry is not left/right symmetric; "
                        f"nearest squared error is {float(errors[candidate])}"
                    )
                order.append(candidate)
                used.add(candidate)
                break
    if len(order) != len(positions):
        raise ValueError("Unable to construct a one-to-one mirrored boost-pad order")
    mapping = np.asarray(order, dtype=np.int64)
    return np.ascontiguousarray(reflected[mapping]), mapping


def mirror_canonical_state(state: RivalCanonicalStateV1) -> RivalCanonicalStateV1:
    """Return an exactly reflected canonical state in canonical pad order."""

    pad_positions, pad_order = _mirrored_pad_order(state.pad_positions)
    goal_centers = np.asarray(state.goal_centers, dtype=np.float32).copy()
    goal_centers[:, 0] *= -1.0
    return replace(
        state,
        self_car=_mirror_car(state.self_car),
        opponent_car=_mirror_car(state.opponent_car),
        ball=_mirror_physics(state.ball),
        goal_centers=goal_centers,
        pad_positions=pad_positions,
        pad_is_big=state.pad_is_big[pad_order],
        pad_active=state.pad_active[pad_order],
        pad_time_until_active=state.pad_time_until_active[pad_order],
    )


class RivalEpisodeSymmetryActionParser(
    ActionParser[AgentID, np.ndarray, np.ndarray, GameState, tuple[str, int]]
):
    """Training-only reflection wrapper around the certified action parser."""

    repeats = 1
    state_dependent_action_mask = False

    def __init__(
        self,
        *,
        mirror_probability: float = 0.5,
        seed: int = 20260908,
        forced_mirror: bool | None = None,
    ) -> None:
        probability = float(mirror_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("mirror_probability must be within [0, 1]")
        self.base = RivalActionV1Parser()
        self.mirror_probability = probability
        self.seed = int(seed)
        self.forced_mirror = forced_mirror
        self.rng = np.random.default_rng(self.seed)
        self.episode_index = 0
        self.mirrored = False

    def get_action_space(self, agent: AgentID) -> tuple[str, int]:
        return self.base.get_action_space(agent)

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        if self.forced_mirror is None:
            self.mirrored = bool(self.rng.random() < self.mirror_probability)
        else:
            self.mirrored = bool(self.forced_mirror)
        self.base.reset(agents, initial_state, shared_info)
        shared_info["rival_v9_symmetry_version"] = SYMMETRY_VERSION
        shared_info["rival_v9_episode_mirror"] = self.mirrored
        shared_info["rival_v9_episode_mirror_index"] = self.episode_index
        shared_info["rival_v9_episode_mirror_probability"] = self.mirror_probability
        shared_info["rival_v9_actor_selected_actions"] = {
            agent: np.zeros(ACTION_DIM, dtype=np.float32) for agent in agents
        }
        shared_info["rival_v9_actor_applied_actions"] = {
            agent: np.zeros(ACTION_DIM, dtype=np.float32) for agent in agents
        }
        self.episode_index += 1

    def parse_actions(
        self,
        actions: dict[AgentID, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, np.ndarray]:
        if bool(shared_info.get("rival_v9_episode_mirror", False)) != self.mirrored:
            raise RuntimeError("Episode mirror bit changed after reset")
        physical_actions = {
            agent: mirror_controller(action) if self.mirrored else np.asarray(action)
            for agent, action in actions.items()
        }
        parsed = self.base.parse_actions(physical_actions, state, shared_info)
        physical_selected = shared_info["rival_v9_selected_actions"]
        physical_applied = shared_info["rival_v9_applied_actions"]
        shared_info["rival_v9_actor_selected_actions"] = {
            agent: (
                mirror_controller(row) if self.mirrored else np.asarray(row).copy()
            )
            for agent, row in physical_selected.items()
        }
        shared_info["rival_v9_actor_applied_actions"] = {
            agent: (
                mirror_controller(row) if self.mirrored else np.asarray(row).copy()
            )
            for agent, row in physical_applied.items()
        }
        return parsed


def symmetry_metadata() -> dict[str, Any]:
    return {
        "version": SYMMETRY_VERSION,
        "axis": "left/right X reflection in canonical team frame",
        "selection": "sampled once at environment reset; episode-stable",
        "state_dependent_mirror": False,
        "controller_signs": CONTROLLER_SIGN.tolist(),
        "polar_vector_signs": np.diag(WORLD_REFLECTION).tolist(),
        "axial_vector_signs": np.diag(AXIAL_REFLECTION).tolist(),
        "rotation_rule": "S_world @ rotation @ diag(1,-1,1)",
        "pad_rule": "reflect X and restore canonical entity order",
    }
