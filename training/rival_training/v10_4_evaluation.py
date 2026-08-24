"""Deterministic Stage-1 V3 evaluation with idle-violation telemetry."""

from __future__ import annotations

from collections import defaultdict
import json
import multiprocessing
from pathlib import Path
import time
from typing import Any

import numpy as np

from . import v10_2_evaluation as _base
from .v10_4_campaign import REPOSITORY_ROOT
from .v10_4_environment import (
    RivalSingleLearnerGymWrapperV3,
    build_ball_acquisition_env,
)
from .v9_actions import RivalHybridPolicy
from .v9_checkpoint import load_v9_checkpoint, portable_path, sha256_file


EVALUATION_VERSION = "RivalBallAcquisitionEvaluationV3"


def _install_v3_environment_bindings() -> None:
    _base.EVALUATION_VERSION = EVALUATION_VERSION
    _base.RivalSingleLearnerGymWrapperV1 = RivalSingleLearnerGymWrapperV3
    _base.build_ball_acquisition_env = build_ball_acquisition_env


def _evaluation_worker_initialize(actor_path: str) -> None:
    _install_v3_environment_bindings()
    _base._evaluation_worker_initialize(actor_path)  # noqa: SLF001


def _evaluation_worker_episode(specification: dict[str, Any]) -> dict[str, Any]:
    return _base._evaluation_worker_episode(specification)  # noqa: SLF001


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
    if int(environment_batch_size) <= 0 or int(evaluation_workers) <= 0:
        raise ValueError("Evaluation batch and worker counts must be positive")
    specifications = list(corpus["episodes"])
    episodes: list[dict[str, Any]] = []
    if int(evaluation_workers) == 1:
        _install_v3_environment_bindings()
        loaded = load_v9_checkpoint(checkpoint_path, device=device)
        policy = RivalHybridPolicy(loaded["actor"], device)
        for start in range(0, len(specifications), int(environment_batch_size)):
            episodes.extend(
                _base._episode_batch(  # noqa: SLF001
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
    overall = _base._aggregate(episodes)  # noqa: SLF001
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
        "families": {
            family: _base._aggregate(rows)  # noqa: SLF001
            for family, rows in by_family.items()
        },
        "overall": overall,
        "evaluation_wall_seconds": wall_seconds,
        "aggregate_simulated_game_seconds_per_wall_second": (
            overall["active_learner_steps"] / 120.0 / wall_seconds
        ),
        "checks": {
            "all_manifest_episodes_completed": len(episodes)
            == int(corpus["episode_count"]),
            "all_five_families_present": len(by_family) == 5,
            "dummy_rows_evaluated": 0,
            "idle_telemetry_present": all(
                all(
                    key in row
                    for key in (
                        "idle_ticks",
                        "idle_simulated_seconds",
                        "pre_touch_idle_share",
                        "cumulative_idle_penalty",
                    )
                )
                for row in episodes
            ),
            "all_metrics_finite": all(
                np.isfinite(
                    [
                        row["reward_total"],
                        row["initial_car_ball_distance"],
                        row["terminal_car_ball_distance"],
                        row["idle_simulated_seconds"],
                        row["cumulative_idle_penalty"],
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
        and report["checks"]["idle_telemetry_present"]
        and report["checks"]["all_metrics_finite"]
    )
    if include_episode_rows:
        report["episode_rows"] = episodes
    return report
