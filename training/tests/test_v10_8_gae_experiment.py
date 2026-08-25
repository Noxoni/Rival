from __future__ import annotations

from copy import deepcopy
import json
import math

import numpy as np

from rival_training.v10_8_campaign import (
    ARM_LAMBDAS,
    load_arm_config,
    paired_contract_report,
)
from rival_training.v10_8_credit import (
    BALL_POSITION,
    SELF_FORWARD,
    SELF_POSITION,
    TOUCH_AGE,
    credit_assignment_window_diagnostics,
    gae_physical_time_report,
)
from rival_training.v9_observations import FIELD_SCALE, OBSERVATION_SIZE


GAMMA = 0.9987444968227265


def test_all_arm_configs_are_identical_except_gae_lambda() -> None:
    configs = {arm: load_arm_config(arm) for arm in ARM_LAMBDAS}
    normalized = {}
    for arm, config in configs.items():
        assert config["ppo"]["gae_lambda"] == ARM_LAMBDAS[arm]
        row = deepcopy(config)
        row["ppo"]["gae_lambda"] = None
        normalized[arm] = json.dumps(row, sort_keys=True)
    assert len(set(normalized.values())) == 1
    assert paired_contract_report()["checks"]["passed"]


def test_physical_time_horizons_match_expected_values() -> None:
    expected_half_lives = {"A": 0.4102545607463736, "B": 0.7532943820857415, "C": 2.0}
    expected_products = {
        "A": 0.9860190386615196,
        "B": 0.9923613699785113,
        "C": 0.9971160533345892,
    }
    for arm, gae_lambda in ARM_LAMBDAS.items():
        report = gae_physical_time_report(
            gamma=GAMMA, gae_lambda=gae_lambda, arm=arm
        )
        assert report["checks"]["passed"]
        assert math.isclose(
            report["gamma_times_lambda"], expected_products[arm], abs_tol=1e-15
        )
        assert math.isclose(
            report["half_life_seconds"], expected_half_lives[arm], abs_tol=1e-12
        )
        for seconds in (0.25, 0.5, 1.0, 2.0, 3.0, 5.0):
            expected = expected_products[arm] ** int(seconds * 120)
            assert math.isclose(
                report["retention"][f"{seconds:g}_seconds"],
                expected,
                abs_tol=1e-15,
            )


def test_delayed_reward_advantage_exactly_matches_analytical_retention() -> None:
    for arm, gae_lambda in ARM_LAMBDAS.items():
        report = gae_physical_time_report(
            gamma=GAMMA, gae_lambda=gae_lambda, arm=arm
        )
        probes = report["synthetic_delayed_reward"]["probes"]
        for seconds in (0.5, 1.0, 2.0, 3.0):
            row = probes[f"{seconds:g}_seconds_before_reward"]
            assert row["absolute_error"] <= 1e-6
            assert math.isclose(
                row["measured_advantage"],
                row["analytical_advantage"],
                rel_tol=1e-6,
                abs_tol=1e-7,
            )


def _observations(ticks: int) -> tuple[np.ndarray, np.ndarray]:
    observations = np.zeros((ticks, OBSERVATION_SIZE), dtype=np.float32)
    next_observations = np.zeros_like(observations)
    scale = np.asarray(FIELD_SCALE, dtype=np.float32)
    observations[:, TOUCH_AGE] = 1.0
    next_observations[:, TOUCH_AGE] = 1.0
    observations[:, SELF_FORWARD] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    next_observations[:, SELF_FORWARD] = np.asarray(
        [1.0, 0.0, 0.0], dtype=np.float32
    )
    ball = np.asarray([2000.0, 0.0, 92.75], dtype=np.float32) / scale
    observations[:, BALL_POSITION] = ball
    next_observations[:, BALL_POSITION] = ball
    car_x = np.linspace(0.0, 1000.0, ticks + 1, dtype=np.float32)
    observations[:, SELF_POSITION] = np.column_stack(
        (car_x[:-1], np.zeros(ticks), np.full(ticks, 17.0))
    ) / scale
    next_observations[:, SELF_POSITION] = np.column_stack(
        (car_x[1:], np.zeros(ticks), np.full(ticks, 17.0))
    ) / scale
    return observations, next_observations


def test_credit_windows_cover_success_and_full_timeout_without_changing_gae() -> None:
    success_ticks = 360
    failure_ticks = 1440
    ticks = success_ticks + failure_ticks
    observations, next_observations = _observations(ticks)
    actions = np.zeros((ticks, 8), dtype=np.float32)
    actions[:, 0] = 0.75
    actions[:, 1] = np.linspace(-0.5, 0.5, ticks, dtype=np.float32)
    rewards = np.zeros(ticks, dtype=np.float64)
    rewards[success_ticks - 1] = 10.0
    next_observations[success_ticks - 1, TOUCH_AGE] = 0.0
    terminated = np.zeros(ticks, dtype=np.float32)
    truncated = np.zeros(ticks, dtype=np.float32)
    terminated[success_ticks - 1] = 1.0
    truncated[-1] = 1.0
    advantages = np.linspace(-1.0, 2.0, ticks, dtype=np.float64)
    normalized = (advantages - advantages.mean()) / advantages.std()

    report = credit_assignment_window_diagnostics(
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminated=terminated,
        truncated=truncated,
        advantages=advantages,
        normalized_advantages=normalized,
    )
    assert report["checks"]["passed"]
    assert report["trajectory_counts"]["successful_first_contact_trajectories"] == 1
    assert report["trajectory_counts"]["failed_timeout_like_trajectories"] == 1
    for cohort in ("successful_first_contact", "failed_timeout_like"):
        assert len(report[cohort]) == 4
        for row in report[cohort].values():
            assert row["eligible_trajectories"] == 1
            assert row["metrics"]["raw_advantage"]["samples"] > 0
            assert row["metrics"]["throttle"]["mean"] == 0.75
            assert row["metrics"]["distance_progress_uu"]["mean"] > 0.0
