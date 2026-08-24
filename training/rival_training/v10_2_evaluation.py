"""Deterministic Stage-1 acquisition evaluation for Rival v10.2."""

from __future__ import annotations

from collections import defaultdict
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .v10_2_campaign import REPOSITORY_ROOT
from .v10_2_environment import (
    RivalSingleLearnerGymWrapperV1,
    build_ball_acquisition_env,
)
from .v9_actions import RivalHybridPolicy
from .v9_checkpoint import load_v9_checkpoint, portable_path, sha256_file


EVALUATION_VERSION = "RivalBallAcquisitionEvaluationV1"
_WORKER_POLICY: RivalHybridPolicy | None = None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "samples": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _distance(env: RivalSingleLearnerGymWrapperV1) -> float:
    if env.active_agent is None:
        raise RuntimeError("Evaluation environment has no active learner")
    car = env.rlgym_env.state.cars[env.active_agent]
    return float(
        np.linalg.norm(
            np.asarray(car.physics.position, dtype=np.float64)
            - np.asarray(env.rlgym_env.state.ball.position, dtype=np.float64)
        )
    )


def _alignment(env: RivalSingleLearnerGymWrapperV1) -> float:
    if env.active_agent is None:
        raise RuntimeError("Evaluation environment has no active learner")
    car = env.rlgym_env.state.cars[env.active_agent]
    relative = np.asarray(env.rlgym_env.state.ball.position, dtype=np.float64) - np.asarray(
        car.physics.position, dtype=np.float64
    )
    norm = float(np.linalg.norm(relative))
    if norm <= 1e-12:
        return 0.0
    forward = np.asarray(car.physics.forward, dtype=np.float64)
    return float(np.clip(np.dot(forward, relative / norm), -1.0, 1.0))


def _start_episode(specification: dict[str, Any]) -> dict[str, Any]:
    raw = build_ball_acquisition_env(
        phase="A",
        seed=int(specification["environment_seed"]),
        forced_family=str(specification["family"]),
        forced_active_team=int(specification["active_team"]),
        forced_mirror=bool(specification["mirror"]),
    )
    env = RivalSingleLearnerGymWrapperV1(raw)
    observation = env.reset()
    return {
        "specification": specification,
        "env": env,
        "observation": observation,
        "initial_distance": _distance(env),
        "initial_alignment": _alignment(env),
        "first_touch_seconds": None,
        "physical_touches": 0,
        "reward_total": 0.0,
        "distance_reward_total": 0.0,
        "heading_reward_total": 0.0,
        "touch_reward_total": 0.0,
        "idle_ticks": 0,
        "idle_seconds": 0.0,
        "idle_penalty_total": 0.0,
        "pre_touch_observed_ticks": 0,
        "progress_values": [],
        "actions": [],
        "termination_reason": None,
        "goal_scored": False,
        "ticks": 0,
    }


def _finish_episode(state: dict[str, Any]) -> dict[str, Any]:
    specification = state["specification"]
    env = state["env"]
    physical_actions = np.asarray(state["actions"], dtype=np.float32)
    buttons = physical_actions[:, 5:]
    return {
        "index": int(specification["index"]),
        "family": str(specification["family"]),
        "active_team": int(specification["active_team"]),
        "mirror": bool(specification["mirror"]),
        "environment_seed": int(specification["environment_seed"]),
        "first_touch_success": state["first_touch_seconds"] is not None,
        "time_to_first_touch_seconds": state["first_touch_seconds"],
        "physical_touch_count": state["physical_touches"],
        "active_learner_steps": state["ticks"],
        "simulated_seconds": state["ticks"] / 120.0,
        "initial_car_ball_distance": state["initial_distance"],
        "terminal_car_ball_distance": _distance(env),
        "initial_car_ball_alignment": state["initial_alignment"],
        "terminal_car_ball_alignment": _alignment(env),
        "mean_signed_car_progress_uu": float(np.mean(state["progress_values"])),
        "median_signed_car_progress_uu": float(np.median(state["progress_values"])),
        "distance_reward_total": state["distance_reward_total"],
        "heading_reward_total": state["heading_reward_total"],
        "touch_reward_total": state["touch_reward_total"],
        "idle_ticks": state["idle_ticks"],
        "idle_simulated_seconds": state["idle_seconds"],
        "pre_touch_observed_ticks": state["pre_touch_observed_ticks"],
        "pre_touch_idle_share": state["idle_ticks"] / max(state["pre_touch_observed_ticks"], 1),
        "cumulative_idle_penalty": state["idle_penalty_total"],
        "reward_total": state["reward_total"],
        "distance_budget_saturated": bool(
            env.rlgym_env.shared_info["rival_v10_2_reward_metrics"]["distance_budget_saturated"]
        ),
        "goal_scored_reward_neutral": state["goal_scored"],
        "termination_reason": state["termination_reason"],
        "action_diagnostics": {
            "mean_absolute_throttle": float(np.mean(np.abs(physical_actions[:, 0]))),
            "mean_absolute_steer": float(np.mean(np.abs(physical_actions[:, 1]))),
            "jump_share": float(buttons[:, 0].mean()),
            "boost_share": float(buttons[:, 1].mean()),
            "handbrake_share": float(buttons[:, 2].mean()),
        },
    }


def _episode_batch(
    policy: RivalHybridPolicy,
    specifications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    states = [_start_episode(specification) for specification in specifications]
    active = list(range(len(states)))
    completed: list[dict[str, Any]] = []
    try:
        while active:
            observations = np.concatenate(
                [states[index]["observation"] for index in active], axis=0
            )
            actions, _ = policy.get_action(observations, deterministic=True)
            physical_batch = actions.numpy().astype(np.float32, copy=False)
            next_active: list[int] = []
            for row_index, state_index in enumerate(active):
                state = states[state_index]
                env = state["env"]
                physical = physical_batch[row_index : row_index + 1]
                observation, rewards, done, truncated, info = env.step(physical)
                state["observation"] = observation
                state["ticks"] += 1
                state["actions"].append(physical[0].copy())
                metrics = env.rlgym_env.shared_info["rival_v10_2_reward_metrics"]
                state["reward_total"] += float(rewards[0])
                components = env.rlgym_env.shared_info["reward_components"][env.active_agent]
                state["distance_reward_total"] += float(components["distance_progress"])
                state["heading_reward_total"] += float(components.get("heading_alignment", 0.0))
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
                    if state["first_touch_seconds"] is None:
                        state["first_touch_seconds"] = state["ticks"] / 120.0
                if done or truncated:
                    state["termination_reason"] = info["rival_v10_2"]["termination_reason"]
                    state["goal_scored"] = bool(done)
                    completed.append(_finish_episode(state))
                    env.close()
                else:
                    next_active.append(state_index)
            active = next_active
    finally:
        for index in active:
            states[index]["env"].close()
    return sorted(completed, key=lambda row: int(row["index"]))


def _episode(
    policy: RivalHybridPolicy,
    specification: dict[str, Any],
) -> dict[str, Any]:
    return _episode_batch(policy, [specification])[0]


def _evaluation_worker_initialize(actor_path: str) -> None:
    global _WORKER_POLICY
    torch.set_num_threads(1)
    actor = torch.load(actor_path, map_location="cpu", weights_only=True)
    from .v9_policy import RivalPolicyV1

    model = RivalPolicyV1()
    model.load_state_dict(actor["state_dict"], strict=True)
    model.eval()
    _WORKER_POLICY = RivalHybridPolicy(model, "cpu")


def _evaluation_worker_episode(specification: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_POLICY is None:
        raise RuntimeError("Evaluation worker actor was not initialized")
    return _episode(_WORKER_POLICY, specification)


def _aggregate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    success = [row for row in episodes if row["first_touch_success"]]
    failures = [row for row in episodes if not row["first_touch_success"]]
    action_names = (
        "mean_absolute_throttle",
        "mean_absolute_steer",
        "jump_share",
        "boost_share",
        "handbrake_share",
    )
    return {
        "episodes": len(episodes),
        "first_touch_success_count": len(success),
        "first_touch_success_share": len(success) / max(len(episodes), 1),
        "no_touch_timeout_count": sum(
            row["termination_reason"] == "no_touch_timeout" for row in episodes
        ),
        "no_touch_timeout_share": sum(
            row["termination_reason"] == "no_touch_timeout" for row in episodes
        )
        / max(len(episodes), 1),
        "successful_time_to_first_touch_seconds": _summary(
            [float(row["time_to_first_touch_seconds"]) for row in success]
        ),
        "initial_car_ball_distance": _summary(
            [float(row["initial_car_ball_distance"]) for row in episodes]
        ),
        "failed_episode_initial_car_ball_distance": _summary(
            [float(row["initial_car_ball_distance"]) for row in failures]
        ),
        "failed_episode_terminal_car_ball_distance": _summary(
            [float(row["terminal_car_ball_distance"]) for row in failures]
        ),
        "initial_car_ball_alignment": _summary(
            [float(row["initial_car_ball_alignment"]) for row in episodes]
        ),
        "terminal_car_ball_alignment": _summary(
            [float(row["terminal_car_ball_alignment"]) for row in episodes]
        ),
        "failed_episode_initial_car_ball_alignment": _summary(
            [float(row["initial_car_ball_alignment"]) for row in failures]
        ),
        "failed_episode_terminal_car_ball_alignment": _summary(
            [float(row["terminal_car_ball_alignment"]) for row in failures]
        ),
        "failed_episode_alignment_improvement": _summary(
            [
                float(row["terminal_car_ball_alignment"]) - float(row["initial_car_ball_alignment"])
                for row in failures
            ]
        ),
        "physical_touch_count": sum(int(row["physical_touch_count"]) for row in episodes),
        "touches_per_episode": sum(int(row["physical_touch_count"]) for row in episodes)
        / max(len(episodes), 1),
        "active_learner_steps": sum(int(row["active_learner_steps"]) for row in episodes),
        "touches_per_100k_active_learner_steps": 100_000.0
        * sum(int(row["physical_touch_count"]) for row in episodes)
        / max(
            sum(int(row["active_learner_steps"]) for row in episodes),
            1,
        ),
        "distance_reward_total": sum(float(row["distance_reward_total"]) for row in episodes),
        "heading_reward_total": sum(float(row["heading_reward_total"]) for row in episodes),
        "touch_reward_total": sum(float(row["touch_reward_total"]) for row in episodes),
        "idle_ticks": sum(int(row["idle_ticks"]) for row in episodes),
        "idle_simulated_seconds": sum(float(row["idle_simulated_seconds"]) for row in episodes),
        "pre_touch_idle_share": sum(int(row["idle_ticks"]) for row in episodes)
        / max(
            sum(int(row["pre_touch_observed_ticks"]) for row in episodes),
            1,
        ),
        "cumulative_idle_penalty": sum(float(row["cumulative_idle_penalty"]) for row in episodes),
        "dense_budget_saturation_share": sum(
            bool(row["distance_budget_saturated"]) for row in episodes
        )
        / max(len(episodes), 1),
        "mean_signed_car_progress_uu": float(
            np.mean([row["mean_signed_car_progress_uu"] for row in episodes])
        ),
        "median_signed_car_progress_uu": float(
            np.median([row["median_signed_car_progress_uu"] for row in episodes])
        ),
        "goals_reward_neutral": sum(bool(row["goal_scored_reward_neutral"]) for row in episodes),
        "action_diagnostics": {
            name: float(np.mean([row["action_diagnostics"][name] for row in episodes]))
            for name in action_names
        },
    }


def evaluate_stage1_checkpoint(
    checkpoint: str | Path,
    corpus_manifest: str | Path,
    *,
    device: str = "cuda:0",
    include_episode_rows: bool = False,
    environment_batch_size: int = 32,
    evaluation_workers: int = 24,
) -> dict[str, Any]:
    started = time.perf_counter()
    checkpoint_path = Path(checkpoint)
    corpus_path = Path(corpus_manifest)
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    if corpus.get("evaluation_version") != EVALUATION_VERSION:
        raise RuntimeError("Evaluation corpus version mismatch")
    if int(environment_batch_size) <= 0:
        raise ValueError("environment_batch_size must be positive")
    if int(evaluation_workers) <= 0:
        raise ValueError("evaluation_workers must be positive")
    episodes: list[dict[str, Any]] = []
    specifications = list(corpus["episodes"])
    if int(evaluation_workers) == 1:
        loaded = load_v9_checkpoint(checkpoint_path, device=device)
        policy = RivalHybridPolicy(loaded["actor"], device)
        for start in range(0, len(specifications), int(environment_batch_size)):
            episodes.extend(
                _episode_batch(
                    policy,
                    specifications[start : start + int(environment_batch_size)],
                )
            )
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(
            processes=int(evaluation_workers),
            initializer=_evaluation_worker_initialize,
            initargs=(str(checkpoint_path / "actor.pt"),),
        ) as pool:
            episodes = list(
                pool.imap_unordered(
                    _evaluation_worker_episode,
                    specifications,
                    chunksize=1,
                )
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
        "deterministic_actor": True,
        "environment_batch_size": int(environment_batch_size),
        "evaluation_workers": int(evaluation_workers),
        "evaluation_device": "cpu" if int(evaluation_workers) > 1 else device,
        "families": {family: _aggregate(rows) for family, rows in by_family.items()},
        "overall": overall,
        "evaluation_wall_seconds": wall_seconds,
        "aggregate_simulated_game_seconds_per_wall_second": (
            overall["active_learner_steps"] / 120.0 / wall_seconds
        ),
        "checks": {
            "all_manifest_episodes_completed": len(episodes) == int(corpus["episode_count"]),
            "all_five_families_present": len(by_family) == 5,
            "dummy_rows_evaluated": 0,
            "all_metrics_finite": all(
                np.isfinite(
                    [
                        row["reward_total"],
                        row["initial_car_ball_distance"],
                        row["terminal_car_ball_distance"],
                    ]
                ).all()
                for row in episodes
            ),
        },
    }
    report["checks"]["passed"] = (
        report["checks"]["all_manifest_episodes_completed"]
        and report["checks"]["all_five_families_present"]
        and report["checks"]["dummy_rows_evaluated"] == 0
        and report["checks"]["all_metrics_finite"]
    )
    if include_episode_rows:
        report["episode_rows"] = episodes
    return report
