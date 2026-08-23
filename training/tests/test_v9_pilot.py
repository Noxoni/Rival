from __future__ import annotations

import gym
import numpy as np
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator

from rival_training.v9_checkpoint import (
    DEFAULT_PILOT_CONFIG_PATH,
    load_m09_config,
    pilot_config_migration_report,
)
from rival_training.v9_curriculum import (
    V9_PILOT_CURRICULUM_WEIGHTS,
    RivalV9PilotCurriculumMutator,
    curriculum_distribution_report,
)
from rival_training.v9_environment import (
    RivalV9PilotGymWrapper,
    build_v9_pilot_env,
)
from rival_training.v9_metrics import (
    V9_PILOT_METRIC_VECTOR_SIZE,
    aggregate_v9_pilot_metrics,
    metric_schema,
)


def test_pilot_config_migration_changes_only_reset_and_metric_contract() -> None:
    source = load_m09_config()
    target = load_m09_config(DEFAULT_PILOT_CONFIG_PATH)
    report = pilot_config_migration_report(source, target)
    assert report["passed"]
    assert report["changed_top_level_keys"] == [
        "config_version",
        "curriculum",
        "environment_version",
    ]
    assert source["policy_version"] == target["policy_version"]
    assert source["ppo"] == target["ppo"]
    assert source["pilot"] == target["pilot"]


def test_pilot_curriculum_is_seeded_finite_and_matches_authorized_mix() -> None:
    engine = RocketSimEngine(rlbot_delay=True)
    try:
        mutator = RivalV9PilotCurriculumMutator(V9_PILOT_CURRICULUM_WEIGHTS, seed=20260913)
        report = curriculum_distribution_report(
            mutator,
            engine.create_base_state,
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            samples=5000,
        )
    finally:
        engine.close()
    assert all(report["counts"][name] > 0 for name in report["counts"])
    for name, expected in V9_PILOT_CURRICULUM_WEIGHTS.items():
        assert abs(report["shares"][name] - expected) < 0.03
    assert report["shares"]["natural"] > 0.5


def test_pilot_metric_transport_is_finite_and_diagnostic_only() -> None:
    wrapper = RivalV9PilotGymWrapper(build_v9_pilot_env(seed=20260913, forced_mirror=False))
    try:
        observations = wrapper.reset()
        assert type(wrapper.action_space) is gym.spaces.Box
        assert wrapper.is_discrete is False
        _, _, _, _, info = wrapper.step(np.zeros((len(observations), 8), dtype=np.float32))
    finally:
        wrapper.close()
    vector = info["state"]
    assert vector.shape == (V9_PILOT_METRIC_VECTOR_SIZE,)
    assert np.isfinite(vector).all()
    report = aggregate_v9_pilot_metrics([vector])
    assert report["finite"]
    assert report["metric_vector_count"] == 1
    assert sum(report["reset_counts"].values()) == 1
    assert report["mechanic_like_detectors_are_diagnostics_only"] is True
    assert metric_schema()["reward_influence"] is False
