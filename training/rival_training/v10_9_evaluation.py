"""Matched deterministic and fixed-seed AR(1) stochastic evaluation for M10.9."""

from __future__ import annotations

from collections import defaultdict
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from . import v10_7_evaluation as _prior
from .v10_6_campaign import FROZEN_CORPUS_EVALUATION_VERSION, REPOSITORY_ROOT
from .v10_7_actions import ANALOG_DIM
from .v10_7_checkpoint import load_checkpoint
from .v10_7_policy import RivalPolicyV1IndependentStickyButtons
from .v10_9_actions import ANALOG_FIELDS, AR_RHO, RivalARStickyBernoulliPolicy
from .v9_checkpoint import portable_path, sha256_file


EVALUATION_VERSION = "RivalBallAcquisitionEvaluationV8PPOV2AR1"
_WORKER_POLICY: RivalARStickyBernoulliPolicy | None = None
_WORKER_DETERMINISTIC = True


def _episode(
    policy: RivalARStickyBernoulliPolicy,
    specification: dict[str, Any],
    *,
    deterministic: bool,
) -> dict[str, Any]:
    _prior._install_environment_bindings()  # noqa: SLF001
    seed = 2026107900 + int(specification["index"])
    torch.manual_seed(seed)
    policy.reset_exploration(seed=seed)
    state = _prior._base._start_episode(specification)  # noqa: SLF001
    state.update(
        {
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
            action_np = action.numpy()
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
            observation, rewards, done, truncated, info = env.step(action_np)
            state["observation"] = observation
            state["ticks"] += 1
            state["actions"].append(action_np[0].copy())
            metrics = env.rlgym_env.shared_info["rival_v10_2_reward_metrics"]
            state["reward_total"] += float(rewards[0])
            components = env.rlgym_env.shared_info["reward_components"][env.active_agent]
            state["distance_reward_total"] += float(components["distance_progress"])
            state["heading_reward_total"] += float(
                components.get("heading_alignment", 0.0)
            )
            state["touch_reward_total"] += float(components["physical_new_touch"])
            state["progress_values"].append(float(metrics["car_progress_clipped_uu"]))
            state["idle_ticks"] += int(metrics.get("idle_ticks", 0))
            state["idle_seconds"] += float(metrics.get("idle_seconds", 0.0))
            state["idle_penalty_total"] += float(metrics.get("idle_penalty", 0.0))
            if (
                state["first_touch_seconds"] is None
                and not bool(metrics["new_physical_touch"])
                and state["ticks"] > 60
            ):
                state["pre_touch_observed_ticks"] += 1
            if bool(metrics["new_physical_touch"]):
                state["physical_touches"] += 1
                state["contact_times_seconds"].append(state["ticks"] / 120.0)
                if state["first_touch_seconds"] is None:
                    state["first_touch_seconds"] = state["ticks"] / 120.0
            if done or truncated:
                state["termination_reason"] = info["rival_v10_2"][
                    "termination_reason"
                ]
                state["goal_scored"] = bool(done)
                row = _prior._finish_with_policy_diagnostics(state)  # noqa: SLF001
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
        raise RuntimeError("M10.9 evaluation worker was not initialized")
    return _episode(
        _WORKER_POLICY, specification, deterministic=_WORKER_DETERMINISTIC
    )


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    result = _prior._aggregate(episodes)  # noqa: SLF001
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
            "path": corpus_path.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(corpus_path),
            "name": corpus["name"],
            "episode_count": int(corpus["episode_count"]),
        },
        "fixed_stochastic_seed_formula": "2026107900 + frozen episode index",
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
                        row["initial_car_ball_alignment"],
                        row["terminal_car_ball_alignment"],
                    ]
                ).all()
                for row in episodes
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


capability_gap = _prior.capability_gap
