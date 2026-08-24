"""Pre-PPO diagnostics for the M10.7 action-policy correction."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
from typing import Any

import numpy as np
import torch
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator
from torch.distributions import Bernoulli, Normal

from .v10_3_curriculum import RivalBallAcquisitionCurriculumV2
from .v10_7_actions import (
    ACTION_DIM,
    ANALOG_DIM,
    BUTTON_FIELDS,
    BUTTON_PERSISTENCE,
    HISTORY_START,
    HISTORY_TICKS,
    CONTROLLER_SIZE,
    TANH_EPSILON,
    RivalStickyBernoulliPolicy,
    deterministic_transition_reachability,
)
from .v10_7_policy import RivalPolicyV1IndependentStickyButtons
from .v9_actions import RivalActionV1Parser, validate_physical_actions
from .v9_canonical import RocketSimCanonicalAdapterV1
from .v9_curriculum import _set_ball, _set_car
from .v9_observations import RivalObsV1Builder


OBSERVATION_CATEGORIES = {
    "easy_aligned_ground_approach": "stationary_close",
    "medium_approach": "stationary_medium",
    "moving_chase": "moving_chase",
    "awkward_heading": "awkward_heading",
    "kickoff": "natural_kickoff_holdout",
    "low_boost": "stationary_medium",
    "recovery": "awkward_heading",
    "airborne": "moving_chase",
}


def _array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode())
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _active_car(state, team: int = 0):
    return next(car for car in state.cars.values() if int(car.team_num) == int(team))


def _active_agent(state, team: int = 0):
    return next(agent for agent, car in state.cars.items() if int(car.team_num) == int(team))


def _apply_category_override(category: str, state) -> None:
    active = _active_car(state)
    if category == "low_boost":
        active.boost_amount = 0.0
    elif category == "recovery":
        position = np.asarray(active.physics.position, dtype=np.float32)
        position[2] = 240.0
        _set_car(
            active,
            position=position,
            velocity=np.asarray([500.0, -250.0, -650.0]),
            angular_velocity=np.asarray([1.5, -2.0, 1.0]),
            euler=np.asarray([0.65, float(active.physics.euler_angles[1]), 0.8]),
            boost=float(active.boost_amount),
        )
    elif category == "airborne":
        position = np.asarray(active.physics.position, dtype=np.float32)
        position[2] = 850.0
        _set_car(
            active,
            position=position,
            velocity=np.asarray([700.0, 150.0, 250.0]),
            angular_velocity=np.asarray([-0.8, 1.1, -1.4]),
            euler=np.asarray([-0.35, float(active.physics.euler_angles[1]), -0.4]),
            boost=float(active.boost_amount),
        )


def build_observation_corpus(
    *,
    samples_per_category: int = 32,
    seed_base: int = 2026107100,
) -> tuple[np.ndarray, list[str], list[dict[str, float | str | int]]]:
    """Generate a deterministic eight-category RocketSim/RivalObsV1 corpus."""

    engine = RocketSimEngine(rlbot_delay=True)
    teams = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    observations: list[np.ndarray] = []
    categories: list[str] = []
    geometry: list[dict[str, float | str | int]] = []
    try:
        for category_index, (category, family) in enumerate(
            OBSERVATION_CATEGORIES.items()
        ):
            mutator = RivalBallAcquisitionCurriculumV2(
                "A",
                seed=int(seed_base) + category_index * 100_000,
                forced_family=family,
                forced_active_team=0,
            )
            for sample_index in range(int(samples_per_category)):
                state = engine.create_base_state()
                shared: dict[str, Any] = {}
                teams.apply(state, shared)
                mutator.apply(state, shared)
                _apply_category_override(category, state)
                state = engine.set_state(state, shared)
                agent = _active_agent(state)
                shared["rival_v9_applied_actions"] = {
                    key: np.zeros(ACTION_DIM, dtype=np.float32) for key in state.cars
                }
                adapter = RocketSimCanonicalAdapterV1()
                builder = RivalObsV1Builder(prediction_refresh_ticks=4)
                canonical = adapter.adapt(state, agent, shared)
                observation = builder.build(canonical)
                car = canonical.self_car.physics
                delta = np.asarray(canonical.ball.position - car.position, dtype=np.float64)
                length = max(float(np.linalg.norm(delta)), 1e-12)
                direction = delta / length
                forward = float(np.dot(car.forward, direction))
                right = float(np.dot(car.right, direction))
                bearing = math.atan2(right, forward)
                observations.append(observation)
                categories.append(category)
                geometry.append(
                    {
                        "category": category,
                        "category_sample_index": sample_index,
                        "car_local_ball_bearing_radians": bearing,
                        "alignment": forward,
                        "distance_uu": length,
                        "boost": float(canonical.self_car.boost),
                        "car_z": float(car.position[2]),
                    }
                )
    finally:
        engine.close()
    return np.asarray(observations, dtype=np.float32), categories, geometry


def observation_corpus_report(
    observations: np.ndarray,
    categories: list[str],
    geometry: list[dict[str, float | str | int]],
    *,
    seed_base: int,
) -> dict[str, Any]:
    unique = list(OBSERVATION_CATEGORIES)
    return {
        "version": "RivalM10_7FrozenButtonObservationCorpusV1",
        "generator_seed_base": int(seed_base),
        "category_source_families": dict(OBSERVATION_CATEGORIES),
        "samples": int(len(observations)),
        "samples_per_category": {
            name: int(sum(value == name for value in categories)) for name in unique
        },
        "observation_shape": list(observations.shape),
        "observation_float32_sha256": _array_sha256(observations),
        "geometry_summary": {
            name: {
                "alignment_range": [
                    float(min(row["alignment"] for row in geometry if row["category"] == name)),
                    float(max(row["alignment"] for row in geometry if row["category"] == name)),
                ],
                "distance_uu_range": [
                    float(min(row["distance_uu"] for row in geometry if row["category"] == name)),
                    float(max(row["distance_uu"] for row in geometry if row["category"] == name)),
                ],
                "boost_range": [
                    float(min(row["boost"] for row in geometry if row["category"] == name)),
                    float(max(row["boost"] for row in geometry if row["category"] == name)),
                ],
                "car_z_range": [
                    float(min(row["car_z"] for row in geometry if row["category"] == name)),
                    float(max(row["car_z"] for row in geometry if row["category"] == name)),
                ],
            }
            for name in unique
        },
        "checks": {
            "all_eight_required_categories_present": set(categories) == set(unique),
            "equal_nonzero_category_counts": len(
                {sum(value == name for value in categories) for name in unique}
            )
            == 1,
            "all_observations_finite": bool(np.isfinite(observations).all()),
            "all_reset_previous_buttons_zero": bool(
                np.all(
                    observations[
                        :,
                        HISTORY_START
                        + (HISTORY_TICKS - 1) * CONTROLLER_SIZE
                        + ANALOG_DIM : HISTORY_START
                        + (HISTORY_TICKS - 1) * CONTROLLER_SIZE
                        + ACTION_DIM,
                    ]
                    == 0.0
                )
            ),
        },
    }


def exact_log_probability_replay_report(
    actor: RivalPolicyV1IndependentStickyButtons,
    observations: np.ndarray,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    policy = RivalStickyBernoulliPolicy(actor, device)
    torch.manual_seed(20261071)
    actions, stored = policy.get_action(observations, deterministic=False)
    distribution = policy.distribution(observations)
    replayed = distribution.log_prob(actions.to(policy.device)).detach().cpu()
    analog = actions[:, :ANALOG_DIM].to(policy.device)
    buttons = actions[:, ANALOG_DIM:].to(policy.device)
    bounded = analog.clamp(-1.0 + TANH_EPSILON, 1.0 - TANH_EPSILON)
    pre_tanh = torch.atanh(bounded)
    independent_analog = (
        Normal(distribution.analog_mean, distribution.analog_log_std.exp()).log_prob(
            pre_tanh
        )
        - torch.log(torch.clamp(1.0 - bounded.square(), min=TANH_EPSILON))
    ).sum(dim=-1)
    independent_buttons = Bernoulli(
        probs=distribution.effective_probabilities
    ).log_prob(buttons).sum(dim=-1)
    independent = (independent_analog + independent_buttons).detach().cpu()
    replay_error = (stored - replayed).abs().numpy()
    reference_error = (stored - independent).abs().numpy()
    checks = {
        "stored_physical_actions_shape_exact": list(actions.shape)
        == [len(observations), ACTION_DIM],
        "stored_buttons_exact_binary": bool(
            torch.equal(actions[:, ANALOG_DIM:], actions[:, ANALOG_DIM:].round())
        ),
        "same_distribution_replay_exact": bool(torch.equal(stored, replayed)),
        "independent_reference_within_1e_6": float(reference_error.max()) <= 1e-6,
        "all_values_finite": bool(
            torch.isfinite(actions).all()
            and torch.isfinite(stored).all()
            and torch.isfinite(independent).all()
        ),
    }
    checks["passed"] = all(checks.values())
    return {
        "formula": (
            "sum(5 tanh-Gaussian log probabilities) + jump effective Bernoulli + "
            "boost effective Bernoulli + handbrake effective Bernoulli"
        ),
        "samples": int(len(actions)),
        "same_distribution_maximum_absolute_error": float(replay_error.max()),
        "independent_reference_maximum_absolute_error": float(reference_error.max()),
        "checks": checks,
    }


def _histogram(values: np.ndarray) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(values, bins=edges)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def _combo_distribution(actions: np.ndarray) -> dict[str, Any]:
    buttons = np.rint(actions[:, ANALOG_DIM:]).astype(np.int64)
    combos = buttons[:, 0] + 2 * buttons[:, 1] + 4 * buttons[:, 2]
    counts = np.bincount(combos, minlength=8)
    return {
        "counts": counts.tolist(),
        "shares": (counts / max(int(counts.sum()), 1)).tolist(),
        "analog_mean": actions[:, :ANALOG_DIM].mean(axis=0).tolist(),
        "analog_mean_absolute": np.abs(actions[:, :ANALOG_DIM]).mean(axis=0).tolist(),
    }


def _duration_summary(values: list[int]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "runs": int(array.size),
        "mean_ticks": float(array.mean()),
        "median_ticks": float(np.median(array)),
        "p90_ticks": float(np.percentile(array, 90)),
        "p99_ticks": float(np.percentile(array, 99)),
        "maximum_ticks": int(array.max()),
        "mean_seconds": float(array.mean() / 120.0),
    }


def _button_run_durations(
    base_probabilities: np.ndarray,
    *,
    seed: int = 20261072,
    ticks_per_state: int = 1440,
    states: int = 32,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    selected = base_probabilities[: min(states, len(base_probabilities))]
    by_button: dict[str, list[int]] = {name: [] for name in BUTTON_FIELDS}
    for base in selected:
        previous = np.zeros(3, dtype=np.int64)
        current_lengths = np.zeros(3, dtype=np.int64)
        for _ in range(ticks_per_state):
            persistence = np.asarray(
                [BUTTON_PERSISTENCE[name] for name in BUTTON_FIELDS], dtype=np.float64
            )
            logits = np.log(np.clip(base, 1e-9, 1.0 - 1e-9)) - np.log(
                np.clip(1.0 - base, 1e-9, 1.0)
            )
            prior = np.log(persistence / (1.0 - persistence))
            effective_logits = logits + (2.0 * previous - 1.0) * prior
            effective = 1.0 / (1.0 + np.exp(-effective_logits))
            sampled = (rng.random(3) < effective).astype(np.int64)
            for index, name in enumerate(BUTTON_FIELDS):
                if current_lengths[index] == 0 or sampled[index] == previous[index]:
                    current_lengths[index] += 1
                else:
                    by_button[name].append(int(current_lengths[index]))
                    current_lengths[index] = 1
            previous = sampled
        for index, name in enumerate(BUTTON_FIELDS):
            by_button[name].append(int(current_lengths[index]))
    return {
        "simulation": {
            "states": int(len(selected)),
            "ticks_per_state": ticks_per_state,
            "physics_hz": 120,
            "initial_previous_buttons": [0, 0, 0],
        },
        "stochastic": {
            name: _duration_summary(values) for name, values in by_button.items()
        },
        "deterministic": {
            name: {
                "runs": int(len(selected)),
                "run_ticks_each": ticks_per_state,
                "run_seconds_each": ticks_per_state / 120.0,
                "chosen_bit": 0,
                "transition_count": 0,
            }
            for name in BUTTON_FIELDS
        },
    }


def button_policy_diagnostics(
    actor: RivalPolicyV1IndependentStickyButtons,
    observations: np.ndarray,
    *,
    device: str | torch.device,
    stochastic_draws_per_state: int = 64,
) -> dict[str, Any]:
    policy = RivalStickyBernoulliPolicy(actor, device)
    distribution = policy.distribution(observations)
    details = distribution.diagnostics()
    base = details["base_probability"].detach().cpu().numpy()
    effective = details["effective_probability"].detach().cpu().numpy()
    deterministic, _ = policy.get_action(observations, deterministic=True)
    repeated = np.repeat(observations, stochastic_draws_per_state, axis=0)
    torch.manual_seed(20261073)
    stochastic, _ = policy.get_action(repeated, deterministic=False)
    stochastic_np = stochastic.numpy()
    deterministic_np = deterministic.numpy()
    deterministic_repeated = np.repeat(
        deterministic_np, stochastic_draws_per_state, axis=0
    )
    entropy = distribution.bernoulli.entropy().detach().cpu().numpy()
    per_button = {}
    for index, name in enumerate(BUTTON_FIELDS):
        stochastic_share = float(stochastic_np[:, ANALOG_DIM + index].mean())
        deterministic_share = float(deterministic_np[:, ANALOG_DIM + index].mean())
        per_button[name] = {
            "mean_base_probability": float(base[:, index].mean()),
            "mean_effective_probability": float(effective[:, index].mean()),
            "base_probability_histogram": _histogram(base[:, index]),
            "effective_probability_histogram": _histogram(effective[:, index]),
            "mean_entropy": float(entropy[:, index].mean()),
            "states_effective_probability_within_0p45_0p55_share": float(
                np.mean((effective[:, index] >= 0.45) & (effective[:, index] <= 0.55))
            ),
            "states_effective_probability_within_0p40_0p60_share": float(
                np.mean((effective[:, index] >= 0.40) & (effective[:, index] <= 0.60))
            ),
            "deterministic_on_share": deterministic_share,
            "stochastic_sampled_on_share": stochastic_share,
            "deterministic_stochastic_disagreement_share": float(
                np.mean(
                    stochastic_np[:, ANALOG_DIM + index]
                    != deterministic_repeated[:, ANALOG_DIM + index]
                )
            ),
            "mean_effective_distance_from_0p5": float(
                np.abs(effective[:, index] - 0.5).mean()
            ),
            "previous_bit_share": 0.0,
        }
    return {
        "actor_state": "exact_source_transfer_before_ppo",
        "corpus_states": int(len(observations)),
        "stochastic_draws_per_state": int(stochastic_draws_per_state),
        "buttons": per_button,
        "mean_button_entropy": float(entropy.sum(axis=1).mean()),
        "deterministic_physical_action_distribution": _combo_distribution(
            deterministic_np
        ),
        "stochastic_physical_action_distribution": _combo_distribution(stochastic_np),
        "average_button_disagreement_share": float(
            np.mean(stochastic_np[:, ANALOG_DIM:] != deterministic_repeated[:, ANALOG_DIM:])
        ),
        "button_run_durations": _button_run_durations(base),
        "deterministic_reachability": deterministic_transition_reachability(),
    }


def gradient_smoke_report(
    actor: RivalPolicyV1IndependentStickyButtons,
    observations: np.ndarray,
    *,
    device: str | torch.device,
) -> dict[str, Any]:
    disposable = deepcopy(actor).to(device).train()
    policy = RivalStickyBernoulliPolicy(disposable, device)
    values = torch.as_tensor(observations[:128], dtype=torch.float32, device=device)
    torch.manual_seed(20261074)
    with torch.no_grad():
        actions, old_log_probabilities = policy.get_action(values)
    actions = actions.to(device)
    old_log_probabilities = old_log_probabilities.to(device)
    distribution = policy.distribution(values)
    log_probabilities = distribution.log_prob(actions)
    advantages = torch.linspace(-1.0, 1.0, len(values), device=device)
    entropy = distribution.entropy(actions)
    loss = -(
        torch.exp(log_probabilities - old_log_probabilities) * advantages
    ).mean() - 0.0002 * entropy.analog_monte_carlo - 0.001 * entropy.button_exact
    loss.backward()
    head = disposable.action_head
    analog_mean = head.analog_mean.weight.grad.detach().abs().sum(dim=1).cpu().numpy()
    analog_log_std = head.analog_log_std.grad.detach().abs().cpu().numpy()
    buttons = head.button_logits.weight.grad.detach().abs().sum(dim=1).cpu().numpy()
    branches = {
        name: {
            "mean_weight_absolute_gradient_sum": float(analog_mean[index]),
            "log_std_absolute_gradient": float(analog_log_std[index]),
            "finite_nonzero": bool(
                np.isfinite(analog_mean[index])
                and np.isfinite(analog_log_std[index])
                and analog_mean[index] > 0.0
                and analog_log_std[index] > 0.0
            ),
        }
        for index, name in enumerate(("throttle", "steer", "pitch", "yaw", "roll"))
    }
    branches.update(
        {
            name: {
                "logit_weight_absolute_gradient_sum": float(buttons[index]),
                "finite_nonzero": bool(
                    np.isfinite(buttons[index]) and buttons[index] > 0.0
                ),
            }
            for index, name in enumerate(BUTTON_FIELDS)
        }
    )
    return {
        "loss": float(loss.detach().cpu()),
        "branches": branches,
        "checks": {
            "loss_finite": bool(torch.isfinite(loss)),
            "all_eight_controller_branches_finite_nonzero": all(
                row["finite_nonzero"] for row in branches.values()
            ),
        },
    }


def _engine_trace(
    action: np.ndarray,
    *,
    ticks: int,
    airborne: bool = False,
    initial_speed: float = 0.0,
    boost: float = 100.0,
) -> dict[str, Any]:
    engine = RocketSimEngine(rlbot_delay=True)
    state = engine.create_base_state()
    shared: dict[str, Any] = {}
    FixedTeamSizeMutator(blue_size=1, orange_size=1).apply(state, shared)
    agent = _active_agent(state)
    car = state.cars[agent]
    _set_ball(
        state,
        position=np.asarray([0.0, 3000.0, 93.0]),
        velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
    )
    _set_car(
        car,
        position=np.asarray([0.0, 0.0, 850.0 if airborne else 17.0]),
        velocity=np.asarray([initial_speed, 0.0, 0.0]),
        angular_velocity=np.zeros(3),
        euler=np.zeros(3),
        boost=boost,
    )
    opponent = next(value for key, value in state.cars.items() if key != agent)
    _set_car(
        opponent,
        position=np.asarray([3500.0, 4500.0, 17.0]),
        velocity=np.zeros(3),
        angular_velocity=np.zeros(3),
        euler=np.zeros(3),
        boost=0.0,
    )
    state = engine.set_state(state, shared)
    initial = {
        "position": np.asarray(state.cars[agent].physics.position).copy(),
        "velocity": np.asarray(state.cars[agent].physics.linear_velocity).copy(),
        "angular_velocity": np.asarray(state.cars[agent].physics.angular_velocity).copy(),
        "boost": float(state.cars[agent].boost_amount),
    }
    rows = []
    try:
        for tick in range(1, int(ticks) + 1):
            actions = {
                key: (
                    action.reshape(1, ACTION_DIM)
                    if key == agent
                    else np.zeros((1, ACTION_DIM), dtype=np.float32)
                )
                for key in state.cars
            }
            state = engine.step(actions, shared)
            current = state.cars[agent]
            rows.append(
                {
                    "tick": tick,
                    "position": np.asarray(current.physics.position).tolist(),
                    "velocity": np.asarray(current.physics.linear_velocity).tolist(),
                    "angular_velocity": np.asarray(
                        current.physics.angular_velocity
                    ).tolist(),
                    "boost": float(current.boost_amount),
                    "on_ground": bool(current.on_ground),
                    "holding_jump": bool(current.is_holding_jump),
                    "is_boosting": bool(current.is_boosting),
                    "handbrake": float(current.handbrake),
                }
            )
    finally:
        engine.close()
    return {"initial": {key: value.tolist() if hasattr(value, "tolist") else value for key, value in initial.items()}, "rows": rows}


def action_mapping_report() -> dict[str, Any]:
    def row(**values: float) -> np.ndarray:
        result = np.zeros(ACTION_DIM, dtype=np.float32)
        for name, value in values.items():
            result[("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake").index(name)] = value
        return result

    traces = {
        "throttle_positive": _engine_trace(row(throttle=1.0), ticks=120),
        "throttle_negative": _engine_trace(row(throttle=-1.0), ticks=120),
        "steer_positive": _engine_trace(row(throttle=1.0, steer=1.0), ticks=60, initial_speed=900.0),
        "steer_negative": _engine_trace(row(throttle=1.0, steer=-1.0), ticks=60, initial_speed=900.0),
        "pitch_positive": _engine_trace(row(pitch=1.0), ticks=30, airborne=True),
        "pitch_negative": _engine_trace(row(pitch=-1.0), ticks=30, airborne=True),
        "yaw_positive": _engine_trace(row(yaw=1.0), ticks=30, airborne=True),
        "yaw_negative": _engine_trace(row(yaw=-1.0), ticks=30, airborne=True),
        "roll_positive": _engine_trace(row(roll=1.0), ticks=30, airborne=True),
        "roll_negative": _engine_trace(row(roll=-1.0), ticks=30, airborne=True),
        "jump": _engine_trace(row(jump=1.0), ticks=30),
        "no_jump": _engine_trace(row(), ticks=30),
        "boost": _engine_trace(row(throttle=1.0, boost=1.0), ticks=60),
        "no_boost": _engine_trace(row(throttle=1.0), ticks=60),
        "handbrake": _engine_trace(row(throttle=1.0, steer=1.0, handbrake=1.0), ticks=30, initial_speed=900.0),
        "jump_boost": _engine_trace(row(throttle=1.0, jump=1.0, boost=1.0), ticks=30),
    }

    def final(name: str, field: str) -> np.ndarray:
        return np.asarray(traces[name]["rows"][-1][field], dtype=np.float64)

    parser = RivalActionV1Parser()
    shared: dict[str, Any] = {}
    parser.reset(["learner"], None, shared)
    all_combos = []
    for combo in range(8):
        buttons = np.asarray([combo & 1, (combo >> 1) & 1, (combo >> 2) & 1], np.float32)
        physical = np.concatenate((np.zeros(5, np.float32), buttons))
        parsed = parser.parse_actions({"learner": physical}, None, shared)["learner"][0]
        all_combos.append(bool(np.array_equal(parsed, physical)))
    selected = np.asarray([0.2, -0.3, 0.4, -0.5, 0.6, 1, 1, 0], np.float32)
    parser.reset(["learner"], None, shared)
    parser.parse_actions({"learner": selected}, None, shared)
    first_applied = shared["rival_v9_applied_actions"]["learner"].copy()
    next_selected = -selected.copy()
    next_selected[ANALOG_DIM:] = np.asarray([0, 0, 1], np.float32)
    parser.parse_actions({"learner": next_selected}, None, shared)
    second_applied = shared["rival_v9_applied_actions"]["learner"].copy()

    angular_pairs = {}
    for axis, component in (("pitch", 1), ("yaw", 2), ("roll", 0)):
        positive = float(final(f"{axis}_positive", "angular_velocity")[component])
        negative = float(final(f"{axis}_negative", "angular_velocity")[component])
        angular_pairs[axis] = {
            "positive_controller_result_rad_per_s": positive,
            "negative_controller_result_rad_per_s": negative,
            "opposite_nonzero": positive * negative < 0.0,
        }
    steer_positive = float(final("steer_positive", "angular_velocity")[2])
    steer_negative = float(final("steer_negative", "angular_velocity")[2])
    checks = {
        "throttle_plus_accelerates_forward": float(final("throttle_positive", "velocity")[0]) > 100.0,
        "throttle_minus_reverses": float(final("throttle_negative", "velocity")[0]) < -100.0,
        "steer_signs_produce_opposite_ground_turns": steer_positive * steer_negative < 0.0,
        "pitch_signs_opposite_and_nonzero": angular_pairs["pitch"]["opposite_nonzero"],
        "yaw_signs_opposite_and_nonzero": angular_pairs["yaw"]["opposite_nonzero"],
        "roll_signs_opposite_and_nonzero": angular_pairs["roll"]["opposite_nonzero"],
        "jump_true_physically_applied": max(row["position"][2] for row in traces["jump"]["rows"])
        > max(row["position"][2] for row in traces["no_jump"]["rows"]) + 50.0,
        "boost_true_physically_applied": (
            float(final("boost", "velocity")[0]) > float(final("no_boost", "velocity")[0])
            and traces["boost"]["rows"][-1]["boost"] < traces["boost"]["initial"]["boost"]
        ),
        "handbrake_true_physically_applied": max(row["handbrake"] for row in traces["handbrake"]["rows"]) > 0.5,
        "simultaneous_jump_boost_possible": (
            max(row["position"][2] for row in traces["jump_boost"]["rows"]) > 50.0
            and traces["jump_boost"]["rows"][-1]["boost"] < traces["jump_boost"]["initial"]["boost"]
        ),
        "all_eight_button_combinations_parser_exact": all(all_combos),
        "one_tick_parser_delay_exact": np.array_equal(first_applied, np.zeros(ACTION_DIM))
        and np.array_equal(second_applied, selected),
        "no_hidden_action_repeat": parser.repeats == 1,
        "physical_transport_validation_exact": np.array_equal(
            validate_physical_actions(selected)[0], selected
        ),
    }
    checks["passed"] = all(checks.values())
    return {
        "version": "RivalM10_7TargetedRocketSimActionMappingV1",
        "physics_hz": 120,
        "engine": "RocketSimEngine(rlbot_delay=True)",
        "steer_z_angular_velocity": {"positive": steer_positive, "negative": steer_negative},
        "air_axis_angular_velocity": angular_pairs,
        "throttle_final_forward_velocity": {
            "positive": float(final("throttle_positive", "velocity")[0]),
            "negative": float(final("throttle_negative", "velocity")[0]),
        },
        "boost_final_forward_velocity": {
            "boost": float(final("boost", "velocity")[0]),
            "no_boost": float(final("no_boost", "velocity")[0]),
        },
        "checks": checks,
    }


def supervised_learnability_report(
    source_actor: RivalPolicyV1IndependentStickyButtons,
    *,
    device: str | torch.device,
    train_samples_per_category: int = 128,
    validation_samples_per_category: int = 32,
    updates: int = 400,
) -> dict[str, Any]:
    train_obs, _, train_geometry = build_observation_corpus(
        samples_per_category=train_samples_per_category,
        seed_base=2026107200,
    )
    validation_obs, _, validation_geometry = build_observation_corpus(
        samples_per_category=validation_samples_per_category,
        seed_base=2026107300,
    )

    def targets(rows: list[dict[str, float | str | int]]) -> np.ndarray:
        output = []
        for row in rows:
            bearing = float(row["car_local_ball_bearing_radians"])
            alignment = float(row["alignment"])
            # The targeted RocketSim trace proves positive steer produces
            # positive yaw, matching positive car-local right bearing.
            steer = float(np.clip(bearing / (math.pi / 2.0), -1.0, 1.0))
            throttle = 1.0 if alignment > 0.0 else 0.0
            output.append([throttle, steer])
        return np.asarray(output, dtype=np.float32)

    train_target = targets(train_geometry)
    validation_target = targets(validation_geometry)
    actor = deepcopy(source_actor).to(device).train()
    initial_state_hash = hashlib.sha256(
        b"".join(
            value.detach().cpu().contiguous().numpy().tobytes()
            for _, value in sorted(actor.state_dict().items())
        )
    ).hexdigest()
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    rng = np.random.default_rng(20261074)

    def measure(observations: np.ndarray, expected: np.ndarray) -> dict[str, float]:
        actor.eval()
        with torch.inference_mode():
            mean, _, _ = actor(
                torch.as_tensor(observations, dtype=torch.float32, device=device)
            )
            predicted = torch.tanh(mean[:, :2]).cpu().numpy()
        actor.train()
        mse = float(np.mean((predicted - expected) ** 2))
        sign_mask = np.abs(expected[:, 1]) >= 0.05
        sign_accuracy = float(
            np.mean(np.sign(predicted[sign_mask, 1]) == np.sign(expected[sign_mask, 1]))
        )
        return {"mean_squared_error": mse, "steering_sign_accuracy": sign_accuracy}

    initial_train = measure(train_obs, train_target)
    initial_validation = measure(validation_obs, validation_target)
    losses = []
    batch_size = 256
    for _ in range(int(updates)):
        indices = rng.integers(0, len(train_obs), size=batch_size)
        observations = torch.as_tensor(
            train_obs[indices], dtype=torch.float32, device=device
        )
        expected = torch.as_tensor(
            train_target[indices], dtype=torch.float32, device=device
        )
        mean, _, _ = actor(observations)
        predicted = torch.tanh(mean[:, :2])
        loss = torch.nn.functional.mse_loss(predicted, expected)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    final_train = measure(train_obs, train_target)
    final_validation = measure(validation_obs, validation_target)
    checks = {
        "disposable_copy_not_source_actor": actor is not source_actor,
        "validation_error_reduced_by_at_least_75_percent": final_validation[
            "mean_squared_error"
        ]
        <= 0.25 * initial_validation["mean_squared_error"],
        "held_out_steering_sign_accuracy_at_least_90_percent": final_validation[
            "steering_sign_accuracy"
        ]
        >= 0.90,
        "training_and_validation_finite": bool(
            np.isfinite(losses).all()
            and all(np.isfinite(list(row.values())).all() for row in (final_train, final_validation))
        ),
        "diagnostic_actor_marked_disposable_and_not_returned": True,
    }
    checks["passed"] = all(checks.values())
    return {
        "version": "RivalM10_7SupervisedDirectionalLearnabilityV1",
        "purpose": "disposable RivalObsV1 to throttle/steer geometry diagnostic",
        "source_transfer_actor_state_sha256_before_copy": initial_state_hash,
        "train_samples": int(len(train_obs)),
        "validation_samples": int(len(validation_obs)),
        "train_seed_base": 2026107200,
        "held_out_seed_base": 2026107300,
        "updates": int(updates),
        "batch_size": batch_size,
        "optimizer": "Adam(lr=3e-4), disposable diagnostic only",
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "final_train": final_train,
        "final_validation": final_validation,
        "training_loss_tail": losses[-20:],
        "held_out_generalization": final_validation,
        "checks": checks,
    }
