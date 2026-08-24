"""Matched deterministic/stochastic Stage-1 evaluation for M10.7."""

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
from .v10_6_campaign import FROZEN_CORPUS_EVALUATION_VERSION, REPOSITORY_ROOT
from .v10_6_environment import (
    RivalSingleLearnerGymWrapperV5,
    build_ball_acquisition_env,
)
from .v10_7_actions import ANALOG_DIM, BUTTON_FIELDS, RivalStickyBernoulliPolicy
from .v10_7_checkpoint import load_checkpoint
from .v10_7_policy import RivalPolicyV1IndependentStickyButtons
from .v9_checkpoint import portable_path, sha256_file


EVALUATION_VERSION = "RivalBallAcquisitionEvaluationV6ActionPolicy"
_WORKER_POLICY: RivalStickyBernoulliPolicy | None = None
_WORKER_DETERMINISTIC = True


def _install_environment_bindings() -> None:
    _base.RivalSingleLearnerGymWrapperV1 = RivalSingleLearnerGymWrapperV5
    _base.build_ball_acquisition_env = build_ball_acquisition_env


def _histogram(values: np.ndarray) -> dict[str, Any]:
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(values, bins=edges)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def _run_histogram(bits: np.ndarray) -> dict[str, Any]:
    bins = (1, 2, 3, 5, 11, 29, 59, 119, 239)
    labels = (
        "1",
        "2",
        "3-4",
        "5-10",
        "11-28",
        "29-58",
        "59-118",
        "119-238",
        "239_plus",
    )
    counts = {label: 0 for label in labels}
    if not len(bits):
        return {"counts": counts, "runs": 0, "mean_ticks": None, "maximum_ticks": None}
    changes = np.flatnonzero(np.diff(bits) != 0) + 1
    boundaries = np.concatenate(([0], changes, [len(bits)]))
    durations = np.diff(boundaries)
    for duration in durations:
        index = int(np.searchsorted(bins, int(duration), side="left"))
        counts[labels[min(index, len(labels) - 1)]] += 1
    return {
        "counts": counts,
        "runs": int(len(durations)),
        "mean_ticks": float(durations.mean()),
        "maximum_ticks": int(durations.max()),
    }


def _finish_with_policy_diagnostics(state: dict[str, Any]) -> dict[str, Any]:
    row = _base._finish_episode(state)  # noqa: SLF001
    selected = np.asarray(state["actions"], dtype=np.float32)
    applied = np.concatenate(
        (np.zeros((1, 8), dtype=np.float32), selected[:-1]), axis=0
    )
    base = np.asarray(state["button_base_probabilities"], dtype=np.float64)
    effective = np.asarray(state["button_effective_probabilities"], dtype=np.float64)
    entropy = np.asarray(state["button_entropies"], dtype=np.float64)
    row["action_diagnostics"] = {
        "throttle_mean": float(applied[:, 0].mean()),
        "steer_mean": float(applied[:, 1].mean()),
        "mean_absolute_throttle": float(np.abs(applied[:, 0]).mean()),
        "mean_absolute_steer": float(np.abs(applied[:, 1]).mean()),
        **{
            f"{name}_share": float(applied[:, ANALOG_DIM + index].mean())
            for index, name in enumerate(BUTTON_FIELDS)
        },
    }
    row["button_policy_diagnostics"] = {
        name: {
            "mean_base_probability": float(base[:, index].mean()),
            "mean_effective_probability": float(effective[:, index].mean()),
            "base_probability_histogram": _histogram(base[:, index]),
            "effective_probability_histogram": _histogram(effective[:, index]),
            "mean_entropy": float(entropy[:, index].mean()),
            "effective_within_0p45_0p55_share": float(
                np.mean((effective[:, index] >= 0.45) & (effective[:, index] <= 0.55))
            ),
            "effective_within_0p40_0p60_share": float(
                np.mean((effective[:, index] >= 0.40) & (effective[:, index] <= 0.60))
            ),
            "effective_distance_from_0p5_mean": float(
                np.abs(effective[:, index] - 0.5).mean()
            ),
            "deterministic_chosen_share": float(
                np.asarray(state["deterministic_buttons"])[:, index].mean()
            ),
            "previous_bit_share": float(
                np.asarray(state["previous_buttons"])[:, index].mean()
            ),
            "applied_run_durations": _run_histogram(
                applied[:, ANALOG_DIM + index].astype(np.int64)
            ),
        }
        for index, name in enumerate(BUTTON_FIELDS)
    }
    return row


def _episode(
    policy: RivalStickyBernoulliPolicy,
    specification: dict[str, Any],
    *,
    deterministic: bool,
) -> dict[str, Any]:
    _install_environment_bindings()
    torch.manual_seed(2026107500 + int(specification["index"]))
    state = _base._start_episode(specification)  # noqa: SLF001
    state.update(
        {
            "button_base_probabilities": [],
            "button_effective_probabilities": [],
            "button_entropies": [],
            "deterministic_buttons": [],
            "previous_buttons": [],
        }
    )
    env = state["env"]
    try:
        while True:
            distribution = policy.distribution(state["observation"])
            diagnostics = distribution.diagnostics()
            entropy = distribution.bernoulli.entropy()
            state["button_base_probabilities"].append(
                diagnostics["base_probability"][0].detach().cpu().numpy()
            )
            state["button_effective_probabilities"].append(
                diagnostics["effective_probability"][0].detach().cpu().numpy()
            )
            state["button_entropies"].append(entropy[0].detach().cpu().numpy())
            state["deterministic_buttons"].append(
                diagnostics["deterministic_bit"][0].detach().cpu().numpy()
            )
            state["previous_buttons"].append(
                diagnostics["previous_bit"][0].detach().cpu().numpy()
            )
            if deterministic:
                action = distribution.mode().detach().cpu().numpy()
            else:
                action = distribution.sample()[0].detach().cpu().numpy()
            observation, rewards, done, truncated, info = env.step(action)
            state["observation"] = observation
            state["ticks"] += 1
            state["actions"].append(action[0].copy())
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
                state["termination_reason"] = info["rival_v10_2"]["termination_reason"]
                state["goal_scored"] = bool(done)
                return _finish_with_policy_diagnostics(state)
    finally:
        env.close()


def _worker_initialize(actor_path: str, deterministic: bool) -> None:
    global _WORKER_POLICY, _WORKER_DETERMINISTIC
    torch.set_num_threads(1)
    payload = torch.load(actor_path, map_location="cpu", weights_only=True)
    actor = RivalPolicyV1IndependentStickyButtons()
    actor.load_state_dict(payload["state_dict"], strict=True)
    actor.eval()
    _WORKER_POLICY = RivalStickyBernoulliPolicy(actor, "cpu")
    _WORKER_DETERMINISTIC = bool(deterministic)


def _worker_episode(specification: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_POLICY is None:
        raise RuntimeError("M10.7 evaluation worker was not initialized")
    return _episode(
        _WORKER_POLICY, specification, deterministic=_WORKER_DETERMINISTIC
    )


def _weighted_mean(rows: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    weights = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
        weights.append(int(row["active_learner_steps"]))
    return float(np.average(values, weights=weights))


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    result = _base._aggregate(episodes)  # noqa: SLF001
    result["action_diagnostics"] = {
        name: float(np.average(
            [row["action_diagnostics"][name] for row in episodes],
            weights=[row["active_learner_steps"] for row in episodes],
        ))
        for name in (
            "throttle_mean",
            "steer_mean",
            "mean_absolute_throttle",
            "mean_absolute_steer",
            "jump_share",
            "boost_share",
            "handbrake_share",
        )
    }
    result["button_policy_diagnostics"] = {
        name: {
            key: _weighted_mean(
                episodes, ("button_policy_diagnostics", name, key)
            )
            for key in (
                "mean_base_probability",
                "mean_effective_probability",
                "mean_entropy",
                "effective_within_0p45_0p55_share",
                "effective_within_0p40_0p60_share",
                "effective_distance_from_0p5_mean",
                "deterministic_chosen_share",
                "previous_bit_share",
            )
        }
        for name in BUTTON_FIELDS
    }
    result["mean_button_entropy"] = sum(
        result["button_policy_diagnostics"][name]["mean_entropy"]
        for name in BUTTON_FIELDS
    )
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
        raise RuntimeError("Frozen M10.5/M10.6 evaluation corpus version mismatch")
    specifications = list(corpus["episodes"])
    if int(evaluation_workers) == 1:
        loaded = load_checkpoint(checkpoint_path, device=device)
        policy = RivalStickyBernoulliPolicy(loaded["actor"], device)
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
    wall_seconds = time.perf_counter() - started
    report = {
        "schema_version": 1,
        "evaluation_version": EVALUATION_VERSION,
        "frozen_corpus_evaluation_version": FROZEN_CORPUS_EVALUATION_VERSION,
        "mode": "deterministic" if deterministic else "stochastic",
        "deterministic_actor": bool(deterministic),
        "checkpoint": {
            "directory": portable_path(checkpoint_path),
            "manifest_sha256": sha256_file(checkpoint_path / "checkpoint_manifest.json"),
            "actor_sha256": sha256_file(checkpoint_path / "actor.pt"),
        },
        "corpus": {
            "path": corpus_path.resolve().relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": sha256_file(corpus_path),
            "name": corpus["name"],
            "episode_count": int(corpus["episode_count"]),
        },
        "evaluation_workers": int(evaluation_workers),
        "families": {name: _aggregate(rows) for name, rows in by_family.items()},
        "overall": overall,
        "evaluation_wall_seconds": wall_seconds,
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
        },
    }
    report["checks"]["passed"] = (
        report["checks"]["all_500_manifest_episodes_completed"]
        and report["checks"]["all_five_families_present"]
        and report["checks"]["all_metrics_finite"]
        and report["checks"]["dummy_rows_evaluated"] == 0
    )
    if include_episode_rows:
        report["episode_rows"] = episodes
    return report


def capability_gap(
    deterministic: dict[str, Any], stochastic: dict[str, Any]
) -> dict[str, Any]:
    deterministic_overall = deterministic["overall"]
    stochastic_overall = stochastic["overall"]
    fields = (
        "first_touch_success_share",
        "second_touch_success_share",
        "third_touch_success_share",
        "all_three_contacts_success_share",
        "no_touch_timeout_share",
    )
    return {
        "definition": "stochastic share minus deterministic deployment share",
        **{
            field: float(stochastic_overall[field] - deterministic_overall[field])
            for field in fields
        },
        "stochastic_materially_better_first_touch_by_5_points": (
            stochastic_overall["first_touch_success_share"]
            - deterministic_overall["first_touch_success_share"]
            >= 0.05
        ),
    }
