"""Bounded balanced-side comparisons against frozen scratch snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import write_json_atomic  # noqa: E402
from rival_training.v9_actions import RivalHybridPolicy  # noqa: E402
from rival_training.v9_checkpoint import (  # noqa: E402
    checkpoint_contract,
    load_v9_checkpoint,
    portable_path,
    sha256_file,
)
from rival_training.v9_environment import build_v9_pilot_env  # noqa: E402
from rival_training.v9_metrics import (  # noqa: E402
    ACTION_FIELDS,
    CONTINUOUS_FIELDS,
    EVENT_FIELDS,
    V9_PILOT_METRIC_INDEX,
    RivalV9PilotMetricTracker,
)
from rival_training.v9_rewards import COMPONENTS  # noqa: E402


def _seed_environment(environment, seed: int) -> None:
    mutators = getattr(environment.state_mutator, "mutators", ())
    for mutator in mutators:
        method = getattr(mutator, "seed", None)
        if callable(method):
            method(seed)
    action_seed = getattr(environment.action_parser, "seed", None)
    if callable(action_seed):
        action_seed(seed + 100_000)


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise FloatingPointError("Frozen comparison metrics require finite samples")
    return {
        "samples": int(len(array)),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
        "cumulative": float(array.sum()),
        "cumulative_absolute": float(np.abs(array).sum()),
    }


def _empty_policy_metrics() -> dict[str, Any]:
    return {
        "continuous": {name: [] for name in CONTINUOUS_FIELDS},
        "events": {name: 0 for name in EVENT_FIELDS},
        "rewards": {name: [] for name in COMPONENTS},
        "actions": [],
        "goals": 0,
        "concessions": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
    }


def _append_slot_metrics(
    destination: dict[str, Any], vector: np.ndarray, slot: str
) -> None:
    for name in CONTINUOUS_FIELDS:
        destination["continuous"][name].append(
            float(vector[V9_PILOT_METRIC_INDEX[f"{slot}.{name}"]])
        )
    for name in EVENT_FIELDS:
        destination["events"][name] += int(
            round(float(vector[V9_PILOT_METRIC_INDEX[f"{slot}.{name}"]]))
        )
    for name in COMPONENTS:
        destination["rewards"][name].append(
            float(vector[V9_PILOT_METRIC_INDEX[f"{slot}.reward.{name}"]])
        )
    destination["actions"].append(
        [float(vector[V9_PILOT_METRIC_INDEX[f"{slot}.{name}"]]) for name in ACTION_FIELDS]
    )


def _finalize_policy_metrics(values: dict[str, Any]) -> dict[str, Any]:
    actions = np.asarray(values["actions"], dtype=np.float64)
    analog = actions[:, :5]
    buttons = np.rint(actions[:, 5:]).astype(np.int64)
    combo = buttons[:, 0] + 2 * buttons[:, 1] + 4 * buttons[:, 2]
    combo_counts = np.bincount(combo, minlength=8)
    samples = int(len(actions))
    event_rates = {
        name: float(count * 100_000.0 / max(samples, 1))
        for name, count in values["events"].items()
    }
    return {
        "agent_metric_samples": samples,
        "record": {
            "wins": int(values["wins"]),
            "losses": int(values["losses"]),
            "draws": int(values["draws"]),
            "goals": int(values["goals"]),
            "concessions": int(values["concessions"]),
            "goal_differential": int(values["goals"] - values["concessions"]),
        },
        "movement_and_recovery": {
            name: _summary(rows) for name, rows in values["continuous"].items()
        },
        "event_counts": values["events"],
        "event_rates_per_100k_agent_steps": event_rates,
        "reward_components": {
            name: _summary(rows) for name, rows in values["rewards"].items()
        },
        "actions": {
            "analog": {
                name: {
                    **_summary(analog[:, index].tolist()),
                    "absolute_over_0_95_share": float(
                        np.mean(np.abs(analog[:, index]) > 0.95)
                    ),
                }
                for index, name in enumerate(
                    ("throttle", "steer", "pitch", "yaw", "roll")
                )
            },
            "button_combo_counts": combo_counts.tolist(),
            "button_combo_shares": (
                combo_counts / max(int(combo_counts.sum()), 1)
            ).tolist(),
            "marginal_button_shares": {
                name: float(buttons[:, index].mean())
                for index, name in enumerate(("jump", "boost", "handbrake"))
            },
        },
        "behavior_signature": {
            "mean_speed": float(np.mean(values["continuous"]["movement_speed"])),
            "mean_planar_speed": float(
                np.mean(values["continuous"]["movement_planar_speed"])
            ),
            "mean_distance_to_ball": float(
                np.mean(values["continuous"]["movement_distance_to_ball"])
            ),
            "mean_boost": float(np.mean(values["continuous"]["movement_boost"])),
            "touches_per_100k_agent_steps": event_rates["touch_event"],
            "aerial_touches_per_100k_agent_steps": event_rates["aerial_touch_event"],
            "first_jumps_per_100k_agent_steps": event_rates["first_jump_event"],
            "dodges_per_100k_agent_steps": event_rates[
                "dodge_or_double_jump_event"
            ],
            "recovery_landings_per_100k_agent_steps": event_rates[
                "recovery_landing_like_event"
            ],
        },
        "finite": bool(np.isfinite(actions).all()),
    }


def _checkpoint_identity(path: Path, loaded: dict[str, Any]) -> dict[str, Any]:
    state = loaded["trainer_state"]
    return {
        "directory": portable_path(path),
        "manifest_sha256": sha256_file(path / "checkpoint_manifest.json"),
        "actor_sha256": sha256_file(path / "actor.pt"),
        "cumulative_agent_steps": int(state["cumulative_agent_steps"]),
        "simulated_game_hours": float(state["simulated_game_hours"]),
        "contract": checkpoint_contract(loaded["config"]),
    }


def _compare(
    candidate_path: Path,
    reference_path: Path,
    *,
    label: str,
    seed: int,
    seed_pairs: int,
    maximum_ticks: int,
    device: str,
) -> dict[str, Any]:
    candidate_loaded = load_v9_checkpoint(candidate_path, device=device)
    reference_loaded = load_v9_checkpoint(reference_path, device=device)
    candidate_policy = RivalHybridPolicy(candidate_loaded["actor"].eval(), device)
    reference_policy = RivalHybridPolicy(reference_loaded["actor"].eval(), device)
    policies = {"candidate": candidate_policy, "reference": reference_policy}
    metrics = {"candidate": _empty_policy_metrics(), "reference": _empty_policy_metrics()}
    episodes: list[dict[str, Any]] = []
    environment = build_v9_pilot_env(seed=seed, forced_mirror=False)
    tracker = RivalV9PilotMetricTracker()
    started = time.perf_counter()
    try:
        for pair in range(seed_pairs):
            episode_seed = seed + pair
            for candidate_team in (0, 1):
                _seed_environment(environment, episode_seed)
                observations = environment.reset()
                tracker.reset(environment.state)
                ticks = 0
                winner: str | None = None
                for _ in range(maximum_ticks):
                    agents = list(observations)
                    action_map = {}
                    slot_policy: dict[str, str] = {}
                    for agent in agents:
                        team = int(environment.state.cars[agent].team_num)
                        slot = "blue" if team == 0 else "orange"
                        policy_name = "candidate" if team == candidate_team else "reference"
                        slot_policy[slot] = policy_name
                        action, _ = policies[policy_name].get_action(
                            observations[agent][None, :], deterministic=True
                        )
                        action_map[agent] = action[0].numpy()
                    observations, _, terminated, truncated = environment.step(action_map)
                    vector = tracker.build(environment.state, environment.shared_info)
                    for slot in ("blue", "orange"):
                        _append_slot_metrics(metrics[slot_policy[slot]], vector, slot)
                    ticks += 1
                    if bool(environment.state.goal_scored):
                        scoring_team = int(environment.state.scoring_team)
                        winner = (
                            "candidate" if scoring_team == candidate_team else "reference"
                        )
                    if any(terminated.values()) or any(truncated.values()):
                        break
                loser = None if winner is None else (
                    "reference" if winner == "candidate" else "candidate"
                )
                if winner is None:
                    metrics["candidate"]["draws"] += 1
                    metrics["reference"]["draws"] += 1
                else:
                    metrics[winner]["wins"] += 1
                    metrics[winner]["goals"] += 1
                    metrics[loser]["losses"] += 1
                    metrics[loser]["concessions"] += 1
                episodes.append(
                    {
                        "seed_pair": pair,
                        "seed": episode_seed,
                        "candidate_team": candidate_team,
                        "ticks": ticks,
                        "winner": winner or "draw",
                    }
                )
    finally:
        environment.close()
    finalized = {name: _finalize_policy_metrics(rows) for name, rows in metrics.items()}
    checks = {
        "balanced_sides": sum(row["candidate_team"] == 0 for row in episodes)
        == sum(row["candidate_team"] == 1 for row in episodes),
        "fixed_seed_pairs_completed": len(episodes) == 2 * seed_pairs,
        "candidate_metrics_finite": finalized["candidate"]["finite"],
        "reference_metrics_finite": finalized["reference"]["finite"],
        "scores_context_only": True,
    }
    checks["passed"] = all(checks.values())
    if not checks["passed"]:
        raise RuntimeError(f"Frozen comparison failed: {checks}")
    return {
        "label": label,
        "candidate": _checkpoint_identity(candidate_path, candidate_loaded),
        "reference": _checkpoint_identity(reference_path, reference_loaded),
        "protocol": {
            "version": "RivalM10FrozenSnapshotComparisonV1",
            "base_seed": seed,
            "seed_pairs": seed_pairs,
            "balanced_side_episodes": 2 * seed_pairs,
            "maximum_ticks_per_episode": maximum_ticks,
            "policy_mode": "deterministic_tanh_mean_and_button_argmax",
            "one_goal_or_fixed_tick_cap": True,
            "wins_and_scores_are_context_not_a_technical_gate": True,
        },
        "episodes": episodes,
        "metrics": finalized,
        "wall_seconds": time.perf_counter() - started,
        "checks": checks,
    }


def _reference(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--reference must be LABEL=CHECKPOINT_PATH")
    label, raw_path = value.split("=", 1)
    if not label or not all(character.isalnum() or character in "-_" for character in label):
        raise ValueError(f"Invalid frozen reference label: {label!r}")
    return label, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--seed-pairs", type=int, default=6)
    parser.add_argument("--max-ticks-per-episode", type=int, default=2400)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    references = [_reference(value) for value in args.reference]
    if len({label for label, _ in references}) != len(references):
        raise ValueError("Frozen reference labels must be unique")
    comparisons = [
        _compare(
            args.candidate.resolve(),
            path.resolve(),
            label=label,
            seed=args.seed,
            seed_pairs=args.seed_pairs,
            maximum_ticks=args.max_ticks_per_episode,
            device=args.device,
        )
        for label, path in references
    ]
    result = {
        "schema_version": 1,
        "status": "passed",
        "comparison_version": "RivalM10FrozenSnapshotComparisonV1",
        "comparisons": comparisons,
        "checks": {
            "all_comparisons_passed": all(row["checks"]["passed"] for row in comparisons),
            "wins_and_scores_excluded_from_technical_pass_fail": True,
        },
    }
    result["checks"]["passed"] = all(result["checks"].values())
    write_json_atomic(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
