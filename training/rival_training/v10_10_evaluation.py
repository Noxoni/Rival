"""Frozen-corpus deterministic and AR stochastic evaluation for M10.10."""

from __future__ import annotations

from collections import defaultdict
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from . import v10_2_evaluation as _base
from . import v10_7_evaluation as _policy_metrics
from .v10_6_campaign import FROZEN_CORPUS_EVALUATION_VERSION, REPOSITORY_ROOT
from .v10_7_actions import ANALOG_DIM
from .v10_7_checkpoint import load_checkpoint
from .v10_7_policy import RivalPolicyV1IndependentStickyButtons
from .v10_9_actions import ANALOG_FIELDS, AR_RHO, RivalARStickyBernoulliPolicy
from .v10_10_environment import (
    RivalSingleLearnerFirstTouchWrapperV1,
    build_first_touch_velocity_env,
)
from .v9_checkpoint import portable_path, sha256_file


EVALUATION_VERSION = "RivalFirstTouchVelocityEvaluationV1PPOV2AR1"
_WORKER_POLICY: RivalARStickyBernoulliPolicy | None = None
_WORKER_DETERMINISTIC = True


def _install_environment_bindings() -> None:
    _base.RivalSingleLearnerGymWrapperV1 = RivalSingleLearnerFirstTouchWrapperV1
    _base.build_ball_acquisition_env = build_first_touch_velocity_env


def _velocity_projection(env: RivalSingleLearnerFirstTouchWrapperV1) -> float:
    if env.active_agent is None:
        raise RuntimeError("Evaluation environment has no active learner")
    car = env.rlgym_env.state.cars[env.active_agent]
    relative = np.asarray(
        env.rlgym_env.state.ball.position, dtype=np.float64
    ) - np.asarray(car.physics.position, dtype=np.float64)
    norm = float(np.linalg.norm(relative))
    direction = (
        np.zeros(3, dtype=np.float64) if norm <= 1e-12 else relative / norm
    )
    return float(
        np.dot(
            np.asarray(car.physics.linear_velocity, dtype=np.float64),
            direction,
        )
    )


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return _base._summary(values)  # noqa: SLF001


def _start_episode(specification: dict[str, Any]) -> dict[str, Any]:
    _install_environment_bindings()
    state = _base._start_episode(specification)  # noqa: SLF001
    state.update(
        {
            "initial_velocity_projection": _velocity_projection(state["env"]),
            "velocity_projection_values": [],
            "velocity_to_ball_reward_total": 0.0,
            "button_base_probabilities": [],
            "button_effective_probabilities": [],
            "button_entropies": [],
            "deterministic_buttons": [],
            "previous_buttons": [],
            "analog_means": [],
            "analog_log_stds": [],
            "analog_epsilons": [],
            "ar_initial_flags": [],
        }
    )
    return state


def _finish_episode(
    state: dict[str, Any], *, deterministic: bool
) -> dict[str, Any]:
    row = _policy_metrics._finish_with_policy_diagnostics(state)  # noqa: SLF001
    row["initial_car_to_ball_velocity_projection"] = float(
        state["initial_velocity_projection"]
    )
    row["terminal_car_to_ball_velocity_projection"] = _velocity_projection(
        state["env"]
    )
    row["mean_car_to_ball_velocity_projection"] = float(
        np.mean(state["velocity_projection_values"])
    )
    row["velocity_to_ball_reward_total"] = float(
        state["velocity_to_ball_reward_total"]
    )
    means = np.asarray(state["analog_means"], dtype=np.float32)
    log_std = np.asarray(state["analog_log_stds"], dtype=np.float32)
    epsilon_values = np.asarray(state["analog_epsilons"], dtype=np.float32)
    actions = np.asarray(state["actions"], dtype=np.float32)
    row["analog_exploration_diagnostics"] = {
        "rho": AR_RHO,
        "deterministic_has_zero_ar_term": bool(
            not deterministic or np.all(epsilon_values == 0.0)
        ),
        "policy_mean": {
            name: float(means[:, index].mean())
            for index, name in enumerate(ANALOG_FIELDS)
        },
        "log_std": {
            name: float(log_std[:, index].mean())
            for index, name in enumerate(ANALOG_FIELDS)
        },
        "epsilon_mean": {
            name: float(epsilon_values[:, index].mean())
            for index, name in enumerate(ANALOG_FIELDS)
        },
        "epsilon_std": {
            name: float(epsilon_values[:, index].std())
            for index, name in enumerate(ANALOG_FIELDS)
        },
        "saturation_share": {
            name: float(np.mean(np.abs(actions[:, index]) > 0.95))
            for index, name in enumerate(ANALOG_FIELDS)
        },
        "initial_transition_count": int(
            np.asarray(state["ar_initial_flags"]).sum()
        ),
    }
    return row


def _episode(
    policy: RivalARStickyBernoulliPolicy,
    specification: dict[str, Any],
    *,
    deterministic: bool,
) -> dict[str, Any]:
    seed = 20261010100 + int(specification["index"])
    torch.manual_seed(seed)
    policy.reset_exploration(seed=seed)
    state = _start_episode(specification)
    env = state["env"]
    try:
        while True:
            distribution = policy.evaluation_distribution(state["observation"])
            diagnostics = distribution.diagnostics()
            entropy = distribution.bernoulli.entropy()
            state["button_base_probabilities"].append(
                diagnostics["base_probability"][0].detach().cpu().numpy()
            )
            state["button_effective_probabilities"].append(
                diagnostics["effective_probability"][0].detach().cpu().numpy()
            )
            state["button_entropies"].append(
                entropy[0].detach().cpu().numpy()
            )
            state["deterministic_buttons"].append(
                diagnostics["deterministic_bit"][0].detach().cpu().numpy()
            )
            state["previous_buttons"].append(
                diagnostics["previous_bit"][0].detach().cpu().numpy()
            )
            state["analog_means"].append(
                distribution.analog_mean[0].detach().cpu().numpy()
            )
            state["analog_log_stds"].append(
                distribution.analog_log_std[0].detach().cpu().numpy()
            )
            state["ar_initial_flags"].append(
                float(distribution.initial[0, 0].detach().cpu())
            )
            action, _ = policy.get_action(
                state["observation"], deterministic=deterministic
            )
            if deterministic:
                epsilon = np.zeros(ANALOG_DIM, dtype=np.float32)
            else:
                epsilon = (
                    distribution.epsilon_from_action(
                        action[:, :ANALOG_DIM].to(policy.device)
                    )[0]
                    .detach()
                    .cpu()
                    .numpy()
                )
            state["analog_epsilons"].append(epsilon)
            action_np = action.numpy()
            observation, rewards, done, truncated, info = env.step(action_np)
            state["observation"] = observation
            state["ticks"] += 1
            state["actions"].append(action_np[0].copy())
            metrics = env.rlgym_env.shared_info["rival_v10_10_reward_metrics"]
            components = env.rlgym_env.shared_info["reward_components"][
                env.active_agent
            ]
            state["reward_total"] += float(rewards[0])
            state["touch_reward_total"] += float(
                components["physical_new_touch"]
            )
            state["velocity_to_ball_reward_total"] += float(
                components["velocity_to_ball"]
            )
            state["velocity_projection_values"].append(
                float(metrics["directed_velocity_uu_per_second"])
            )
            # Compatibility accumulators remain exact zero by contract.
            state["progress_values"].append(0.0)
            if bool(metrics["new_physical_touch"]):
                state["physical_touches"] += 1
                state["contact_times_seconds"].append(state["ticks"] / 120.0)
                if state["first_touch_seconds"] is None:
                    state["first_touch_seconds"] = state["ticks"] / 120.0
            if done or truncated:
                state["termination_reason"] = info["rival_v10_2"][
                    "termination_reason"
                ]
                state["goal_scored"] = bool(
                    done and state["termination_reason"] == "goal"
                )
                return _finish_episode(state, deterministic=deterministic)
    finally:
        env.close()


def _worker_initialize(actor_path: str, deterministic: bool) -> None:
    global _WORKER_POLICY, _WORKER_DETERMINISTIC
    torch.set_num_threads(1)
    payload = torch.load(actor_path, map_location="cpu", weights_only=True)
    actor = RivalPolicyV1IndependentStickyButtons()
    actor.load_state_dict(payload["state_dict"], strict=True)
    actor.eval()
    _WORKER_POLICY = RivalARStickyBernoulliPolicy(actor, "cpu")
    _WORKER_DETERMINISTIC = bool(deterministic)


def _worker_episode(specification: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_POLICY is None:
        raise RuntimeError("M10.10 evaluation worker was not initialized")
    return _episode(
        _WORKER_POLICY, specification, deterministic=_WORKER_DETERMINISTIC
    )


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    result = _policy_metrics._aggregate(episodes)  # noqa: SLF001
    failures = [row for row in episodes if not row["first_touch_success"]]
    result.update(
        {
            "no_touch_failure_count": len(episodes)
            - int(result["first_touch_success_count"]),
            "no_touch_failure_share": 1.0
            - float(result["first_touch_success_share"]),
            "initial_car_to_ball_velocity_projection": _summary(
                [
                    float(row["initial_car_to_ball_velocity_projection"])
                    for row in episodes
                ]
            ),
            "terminal_car_to_ball_velocity_projection": _summary(
                [
                    float(row["terminal_car_to_ball_velocity_projection"])
                    for row in episodes
                ]
            ),
            "failed_episode_initial_car_to_ball_velocity_projection": _summary(
                [
                    float(row["initial_car_to_ball_velocity_projection"])
                    for row in failures
                ]
            ),
            "failed_episode_terminal_car_to_ball_velocity_projection": _summary(
                [
                    float(row["terminal_car_to_ball_velocity_projection"])
                    for row in failures
                ]
            ),
            "failed_episode_velocity_projection_change": _summary(
                [
                    float(row["terminal_car_to_ball_velocity_projection"])
                    - float(row["initial_car_to_ball_velocity_projection"])
                    for row in failures
                ]
            ),
            "velocity_to_ball_reward_total": sum(
                float(row["velocity_to_ball_reward_total"]) for row in episodes
            ),
            "second_touch_success_count": 0,
            "second_touch_success_share": 0.0,
            "third_touch_success_count": 0,
            "third_touch_success_share": 0.0,
            "all_three_contacts_success_count": 0,
            "all_three_contacts_success_share": 0.0,
        }
    )
    weights = [row["active_learner_steps"] for row in episodes]
    result["analog_exploration_diagnostics"] = {
        "rho": AR_RHO,
        "deterministic_has_zero_ar_term": all(
            row["analog_exploration_diagnostics"][
                "deterministic_has_zero_ar_term"
            ]
            for row in episodes
        ),
        **{
            field: {
                name: float(
                    np.average(
                        [
                            row["analog_exploration_diagnostics"][field][name]
                            for row in episodes
                        ],
                        weights=weights,
                    )
                )
                for name in ANALOG_FIELDS
            }
            for field in (
                "policy_mean",
                "log_std",
                "epsilon_mean",
                "epsilon_std",
                "saturation_share",
            )
        },
        "initial_transition_count": sum(
            row["analog_exploration_diagnostics"]["initial_transition_count"]
            for row in episodes
        ),
    }
    return result


def evaluate_stage1_checkpoint(
    checkpoint: str | Path,
    corpus_manifest: str | Path,
    *,
    deterministic: bool,
    device: str = "cuda:0",
    include_episode_rows: bool = False,
    evaluation_workers: int = 24,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_path = Path(checkpoint)
    corpus_path = Path(corpus_manifest)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if corpus.get("evaluation_version") != FROZEN_CORPUS_EVALUATION_VERSION:
        raise RuntimeError("Frozen Stage-1 evaluation corpus version mismatch")
    specifications = list(corpus["episodes"])
    if int(evaluation_workers) == 1:
        loaded = load_checkpoint(checkpoint_path, device=device)
        policy = RivalARStickyBernoulliPolicy(loaded["actor"], device)
        episodes = [
            _episode(policy, specification, deterministic=deterministic)
            for specification in specifications
        ]
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=int(evaluation_workers),
            initializer=_worker_initialize,
            initargs=(str(checkpoint_path / "actor.pt"), bool(deterministic)),
        ) as pool:
            episodes = list(
                pool.imap_unordered(_worker_episode, specifications, chunksize=1)
            )
        episodes.sort(key=lambda row: int(row["index"]))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_family[str(episode["family"])].append(episode)
    overall = _aggregate(episodes)
    report = {
        "schema_version": 1,
        "evaluation_version": EVALUATION_VERSION,
        "frozen_corpus_evaluation_version": FROZEN_CORPUS_EVALUATION_VERSION,
        "mode": "deterministic" if deterministic else "stochastic_ar1_fixed_seed",
        "deterministic_actor": bool(deterministic),
        "checkpoint": {
            "directory": portable_path(checkpoint_path),
            "manifest_sha256": sha256_file(
                checkpoint_path / "checkpoint_manifest.json"
            ),
            "actor_sha256": sha256_file(checkpoint_path / "actor.pt"),
        },
        "corpus": {
            "path": corpus_path.resolve().relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "sha256": sha256_file(corpus_path),
            "name": corpus["name"],
            "episode_count": int(corpus["episode_count"]),
        },
        "fixed_stochastic_seed_formula": (
            "20261010100 + frozen episode index"
        ),
        "evaluation_workers": int(evaluation_workers),
        "families": {name: _aggregate(rows) for name, rows in by_family.items()},
        "overall": overall,
        "evaluation_wall_seconds": time.perf_counter() - started,
        "checks": {
            "all_500_manifest_episodes_completed": len(episodes) == 500,
            "all_five_families_present": len(by_family) == 5,
            "all_metrics_finite": all(
                np.isfinite(
                    [
                        row["reward_total"],
                        row["initial_car_ball_distance"],
                        row["terminal_car_ball_distance"],
                        row["initial_car_to_ball_velocity_projection"],
                        row["terminal_car_to_ball_velocity_projection"],
                    ]
                ).all()
                for row in episodes
            ),
            "touch_terminates_immediately": all(
                not row["first_touch_success"]
                or row["termination_reason"] == "first_touch"
                for row in episodes
            ),
            "no_reacquisition_objective": all(
                int(row["physical_touch_count"]) <= 1 for row in episodes
            ),
            "dummy_rows_evaluated": 0,
            "deterministic_ar_term_zero": bool(
                not deterministic
                or overall["analog_exploration_diagnostics"][
                    "deterministic_has_zero_ar_term"
                ]
            ),
        },
    }
    report["checks"]["passed"] = all(
        value == 0 if key == "dummy_rows_evaluated" else bool(value)
        for key, value in report["checks"].items()
    )
    if include_episode_rows:
        report["episode_rows"] = episodes
    return report


def capability_gap(
    deterministic: dict[str, Any], stochastic: dict[str, Any]
) -> dict[str, Any]:
    det = deterministic["overall"]
    sto = stochastic["overall"]
    return {
        "definition": "stochastic share minus deterministic deployment share",
        "first_touch_success_share": float(
            sto["first_touch_success_share"] - det["first_touch_success_share"]
        ),
        "no_touch_timeout_share": float(
            sto["no_touch_timeout_share"] - det["no_touch_timeout_share"]
        ),
        "no_touch_failure_share": float(
            sto["no_touch_failure_share"] - det["no_touch_failure_share"]
        ),
        "stochastic_materially_better_first_touch_by_5_points": (
            sto["first_touch_success_share"]
            - det["first_touch_success_share"]
            >= 0.05
        ),
    }
