from __future__ import annotations

import numpy as np

from rival_training.actions import DualRateActionParser
from rival_training.dual_rate import DualRateCompositor, StrategicWindowScheduler
from rival_training.environment import build_dual_rate_env


def _row(value: float) -> np.ndarray:
    return np.full(8, value, dtype=np.float32)


def test_strategic_scheduler_exact_consecutive_windows_and_long_trace() -> None:
    scheduler = StrategicWindowScheduler(_row(1))
    trace = []
    for value in (2, 3, 4, 5):
        window = scheduler.select(_row(value))
        expected = np.stack([_row(value - 1)] * 5 + [_row(value)] * 3)
        assert np.array_equal(window, expected)
        trace.extend(scheduler.take(4))
        trace.extend(scheduler.take(4))
    assert np.array_equal(np.stack(trace[:8]), np.stack([_row(1)] * 5 + [_row(2)] * 3))
    assert scheduler.pending_ticks == 0


def test_mechanics_pass_and_consecutive_override_windows() -> None:
    compositor = DualRateCompositor(_row(1))
    strategic = np.stack([_row(2), _row(2), _row(3), _row(3)])
    passed = compositor.compose(strategic, 0, None)
    assert np.array_equal(passed.controllers, strategic)
    first = compositor.compose(strategic, 1, _row(7))
    assert np.array_equal(first.controllers, np.stack([_row(3), _row(7), _row(7), _row(7)]))
    second = compositor.compose(strategic, 2, _row(8))
    assert np.array_equal(second.controllers, np.stack([_row(7), _row(8), _row(8), _row(8)]))
    returned = compositor.compose(strategic, 0, None)
    assert np.array_equal(returned.controllers, strategic)


def test_disabled_and_forced_pass_long_traces_are_exact() -> None:
    rng = np.random.default_rng(8)
    decisions = [rng.integers(-1, 2, size=8).astype(np.float32) for _ in range(64)]
    disabled_scheduler = StrategicWindowScheduler()
    pass_scheduler = StrategicWindowScheduler()
    disabled_compositor = DualRateCompositor()
    pass_compositor = DualRateCompositor()
    disabled_trace = []
    pass_trace = []
    for selected in decisions:
        disabled_scheduler.select(selected)
        pass_scheduler.select(selected)
        for _ in range(2):
            disabled_trace.extend(
                disabled_compositor.compose(disabled_scheduler.take(4), 0, None).controllers
            )
            pass_trace.extend(
                pass_compositor.compose(pass_scheduler.take(4), 0, None).controllers
            )
    assert np.array_equal(np.stack(disabled_trace), np.stack(pass_trace))


def test_dual_rate_parser_has_exact_mechanics_head_and_frozen_strategic_branch() -> None:
    parser = DualRateActionParser(force_pass=True)
    assert parser.get_action_space("agent") == ("discrete", 69)
    assert all(not parameter.requires_grad for parameter in parser.strategic_model.parameters())


def test_dual_rate_environment_steps_four_explicit_ticks() -> None:
    env = build_dual_rate_env(force_pass=True, natural_only=True)
    try:
        observations = env.reset()
        assert all(space == ("discrete", 69) for space in env.action_spaces.values())
        before = env.state.tick_count
        actions = {agent: np.array([0], dtype=np.int64) for agent in observations}
        next_observations, rewards, terminated, truncated = env.step(actions)
        assert env.state.tick_count - before == 4
        assert all(value.shape == (432,) for value in next_observations.values())
        assert all(np.isfinite(value) for value in rewards.values())
        assert not any(terminated.values())
        assert not any(truncated.values())
    finally:
        env.close()
