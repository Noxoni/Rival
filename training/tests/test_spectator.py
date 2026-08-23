from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rival_training.spectator import (
    load_spectator_policy,
    resolve_tick_skip,
    spectator_preflight,
)
from rival_training.policy import MechanicsActor


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_spectator_auto_cadence_matches_policy_origin() -> None:
    assert resolve_tick_skip("frozen-wisp", "auto") == 8
    assert resolve_tick_skip("current", "auto") == 4
    assert resolve_tick_skip("actor.ts", "8") == 8


def test_spectator_loads_frozen_and_zero_step_policies() -> None:
    frozen = load_spectator_policy(None)
    zero_step = load_spectator_policy(
        REPOSITORY_ROOT / "training/artifacts/bootstrap/wisp_student_expanded_v1.pt"
    )
    observation = np.zeros(432, dtype=np.float32)

    assert frozen.action_count == 90
    assert zero_step.action_count == 158
    assert frozen.action(observation) == zero_step.action(observation, legacy_only=True)


def test_spectator_preflight_is_one_isolated_renderer_free_environment() -> None:
    report = spectator_preflight(
        "frozen-wisp",
        tick_skip=8,
        device="cpu",
        check_rlviser=False,
        check_binary=False,
    )

    assert report["status"] == "passed"
    assert report["single_environment"] is True
    assert report["separate_process"] is True
    assert report["headless_environment_modified"] is False
    assert report["rlviser_binary"] is None
    assert report["reset_agent_count"] == 2
    assert set(map(tuple, report["step_observation_shapes"].values())) == {(432,)}


def test_spectator_loads_m08_mechanics_export_and_steps_dual_rate(tmp_path) -> None:
    actor = MechanicsActor(seed=8).eval()
    export = tmp_path / "mechanics.ts"
    torch.jit.trace(actor, torch.zeros(1, 432)).save(str(export))

    loaded = load_spectator_policy(export)
    report = spectator_preflight(
        export,
        tick_skip=4,
        device="cpu",
        check_rlviser=False,
        check_binary=False,
    )

    assert loaded.action_count == 69
    assert report["action_count"] == 69
    assert report["control_mode"] == "dual_rate_pass_or_override"
    assert report["tick_skip"] == 4
