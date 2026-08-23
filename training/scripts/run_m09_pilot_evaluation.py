"""Run a fixed, deterministic headless behavior evaluation for Gate 13."""

from __future__ import annotations

import argparse
import hashlib
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

from rival_training.v9_actions import RivalHybridPolicy  # noqa: E402
from rival_training.v9_checkpoint import (  # noqa: E402
    load_v9_checkpoint,
    portable_path,
    sha256_file,
)
from rival_training.v9_environment import build_v9_pilot_env  # noqa: E402
from rival_training.v9_metrics import (  # noqa: E402
    EVENT_FIELDS,
    RivalV9PilotMetricTracker,
    aggregate_v9_pilot_metrics,
)


def _seed_environment(environment, seed: int) -> None:
    mutators = getattr(environment.state_mutator, "mutators", ())
    for mutator in mutators:
        seed_method = getattr(mutator, "seed", None)
        if callable(seed_method):
            seed_method(seed)
    action_seed = getattr(environment.action_parser, "seed", None)
    if callable(action_seed):
        action_seed(seed + 100_000)


def _actor_fingerprint(loaded: dict[str, Any]) -> dict[str, Any]:
    actor = loaded["actor"].to("cpu").eval()
    observations = np.asarray(loaded["reload_observations"], dtype=np.float32)
    with torch.inference_mode():
        mean, log_std, button_logits = actor(torch.from_numpy(observations))
        parameters = torch.cat((mean, log_std.expand_as(mean), button_logits), dim=-1).numpy()
        deterministic, _ = RivalHybridPolicy(actor, "cpu").get_action(
            observations, deterministic=True
        )
    parameter_vector = torch.nn.utils.parameters_to_vector(actor.parameters()).detach()
    return {
        "held_observation_count": int(len(observations)),
        "distribution_parameter_shape": list(parameters.shape),
        "distribution_parameter_sha256": hashlib.sha256(
            np.ascontiguousarray(parameters).tobytes()
        ).hexdigest(),
        "deterministic_action_sha256": hashlib.sha256(
            np.ascontiguousarray(deterministic.numpy()).tobytes()
        ).hexdigest(),
        "parameter_l2_norm": float(torch.linalg.vector_norm(parameter_vector)),
        "all_finite": bool(
            np.isfinite(parameters).all() and bool(torch.isfinite(deterministic).all())
        ),
    }


def _behavior_signature(metrics: dict[str, Any]) -> dict[str, float]:
    continuous = metrics["movement_and_recovery"]
    events = metrics["event_counts"]
    agent_steps = max(int(metrics["agent_metric_samples"]), 1)

    def mean_across_teams(name: str) -> float:
        return float(
            np.mean(
                [
                    continuous[f"blue.{name}"]["mean"],
                    continuous[f"orange.{name}"]["mean"],
                ]
            )
        )

    def event_rate(name: str) -> float:
        count = sum(int(events[f"{side}.{name}"]) for side in ("blue", "orange"))
        return float(count * 100_000.0 / agent_steps)

    mechanic_like_names = [
        name
        for name in EVENT_FIELDS
        if name.endswith("_like_event")
        or name
        in {
            "first_jump_event",
            "dodge_or_double_jump_event",
            "directional_dodge_event",
        }
    ]
    mechanic_total = sum(
        int(events[f"{side}.{name}"]) for side in ("blue", "orange") for name in mechanic_like_names
    )
    return {
        "mean_speed": mean_across_teams("movement_speed"),
        "mean_planar_speed": mean_across_teams("movement_planar_speed"),
        "mean_distance_to_ball": mean_across_teams("movement_distance_to_ball"),
        "mean_boost": mean_across_teams("movement_boost"),
        "airborne_distance_per_agent_step": mean_across_teams("movement_airborne_distance_step"),
        "touches_per_100k_agent_steps": event_rate("touch_event"),
        "aerial_touches_per_100k_agent_steps": event_rate("aerial_touch_event"),
        "first_jumps_per_100k_agent_steps": event_rate("first_jump_event"),
        "dodges_per_100k_agent_steps": event_rate("dodge_or_double_jump_event"),
        "recovery_landings_per_100k_agent_steps": event_rate("recovery_landing_like_event"),
        "mechanic_like_events_per_100k_agent_steps": float(
            mechanic_total * 100_000.0 / agent_steps
        ),
        "air_roll_active_share": float(
            sum(events[f"{side}.air_roll_active_tick"] for side in ("blue", "orange")) / agent_steps
        ),
        "aerial_possession_like_share": float(
            sum(events[f"{side}.aerial_possession_like_tick"] for side in ("blue", "orange"))
            / agent_steps
        ),
    }


def _summary(values: list[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array) or not np.isfinite(array).all():
        raise FloatingPointError("Fixed action diagnostics require finite samples")
    return {
        "samples": int(len(array)),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "minimum": float(array.min()),
        "p01": float(np.percentile(array, 1)),
        "p50": float(np.percentile(array, 50)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def _button_run_lengths(
    sequences: list[np.ndarray], button_index: int, state: int
) -> list[float]:
    lengths: list[float] = []
    for sequence in sequences:
        values = np.rint(sequence[:, button_index]).astype(np.int64)
        start = 0
        while start < len(values):
            end = start + 1
            while end < len(values) and values[end] == values[start]:
                end += 1
            if int(values[start]) == state:
                lengths.append(float(end - start))
            start = end
    return lengths or [0.0]


def _fixed_action_diagnostics(
    sequences: list[np.ndarray],
    *,
    learned_log_std: np.ndarray,
    button_entropy_samples: list[float],
) -> dict[str, Any]:
    actions = np.concatenate(sequences, axis=0)
    analog = actions[:, :5]
    buttons = np.rint(actions[:, 5:]).astype(np.int64)
    combos = buttons[:, 0] + 2 * buttons[:, 1] + 4 * buttons[:, 2]
    combo_counts = np.bincount(combos, minlength=8)
    names = ("throttle", "steer", "pitch", "yaw", "roll")
    button_names = ("jump", "boost", "handbrake")
    changes = sum(
        int(np.any(np.abs(np.diff(sequence, axis=0)) > 1e-6, axis=1).sum())
        for sequence in sequences
    )
    elapsed_agent_seconds = sum(len(sequence) for sequence in sequences) / 120.0
    return {
        "schema_version": 1,
        "action_semantics": "policy_selected_RivalActionV1_before_one_tick_transport",
        "samples": int(len(actions)),
        "analog": {
            name: {
                **_summary(analog[:, index]),
                "absolute_over_0_95_share": float(
                    np.mean(np.abs(analog[:, index]) > 0.95)
                ),
            }
            for index, name in enumerate(names)
        },
        "learned_analog_log_std": np.asarray(learned_log_std, dtype=np.float64).tolist(),
        "learned_analog_std": np.exp(
            np.asarray(learned_log_std, dtype=np.float64)
        ).tolist(),
        "button_combo_counts": combo_counts.tolist(),
        "button_combo_shares": (combo_counts / max(int(combo_counts.sum()), 1)).tolist(),
        "button_categorical_entropy": _summary(button_entropy_samples),
        "standalone_button_activation_rates": {
            name: float(buttons[:, index].mean())
            for index, name in enumerate(button_names)
        },
        "consecutive_button_state_run_lengths_ticks": {
            name: {
                "held": _summary(_button_run_lengths(sequences, 5 + index, 1)),
                "released": _summary(_button_run_lengths(sequences, 5 + index, 0)),
            }
            for index, name in enumerate(button_names)
        },
        "controller_change_rate_per_agent_second": float(
            changes / max(elapsed_agent_seconds, 1e-12)
        ),
        "all_finite": True,
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    loaded = load_v9_checkpoint(args.checkpoint, device=args.device)
    actor = loaded["actor"].to(args.device).eval()
    policy = RivalHybridPolicy(actor, args.device)
    fingerprint = _actor_fingerprint(loaded)
    vectors: list[np.ndarray] = []
    action_sequences: list[np.ndarray] = []
    button_entropy_samples: list[float] = []
    episode_reports: list[dict[str, Any]] = []
    tracker = RivalV9PilotMetricTracker()
    environment = build_v9_pilot_env(seed=args.seed, forced_mirror=False)
    started = time.perf_counter()
    total_ticks = 0
    try:
        for episode in range(int(args.episodes)):
            episode_seed = int(args.seed) + episode
            _seed_environment(environment, episode_seed)
            observations = environment.reset()
            tracker.reset(environment.state)
            episode_actions: list[list[np.ndarray]] = [[], []]
            terminated_reason = "fixed_tick_cap"
            ticks = 0
            for _ in range(int(args.max_ticks_per_episode)):
                agents = list(observations)
                batch = np.stack([observations[agent] for agent in agents])
                with torch.inference_mode():
                    distribution = policy.distribution(batch)
                    actions = distribution.mode().detach().cpu()
                    button_entropy_samples.extend(
                        distribution.categorical.entropy().detach().cpu().numpy().tolist()
                    )
                action_map = {agent: actions[index].numpy() for index, agent in enumerate(agents)}
                for index in range(len(agents)):
                    episode_actions[index].append(actions[index].numpy().copy())
                observations, _, terminated, truncated = environment.step(action_map)
                vectors.append(tracker.build(environment.state, environment.shared_info))
                ticks += 1
                total_ticks += 1
                if any(terminated.values()) or any(truncated.values()):
                    terminated_reason = (
                        "goal" if any(terminated.values()) else "environment_truncation"
                    )
                    break
            action_sequences.extend(
                np.asarray(rows, dtype=np.float32) for rows in episode_actions if rows
            )
            episode_reports.append(
                {
                    "episode": episode,
                    "seed": episode_seed,
                    "environment_ticks": ticks,
                    "agent_steps": 2 * ticks,
                    "end_reason": terminated_reason,
                }
            )
    finally:
        environment.close()
    metrics = aggregate_v9_pilot_metrics(vectors)
    fixed_action_diagnostics = _fixed_action_diagnostics(
        action_sequences,
        learned_log_std=actor.action_head.analog_log_std.detach().cpu().numpy(),
        button_entropy_samples=button_entropy_samples,
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "evaluation_version": "RivalScratchPilotFixedEvaluationV1",
        "checkpoint": {
            "directory": portable_path(args.checkpoint),
            "manifest_sha256": sha256_file(Path(args.checkpoint) / "checkpoint_manifest.json"),
            "actor_sha256": sha256_file(Path(args.checkpoint) / "actor.pt"),
            "cumulative_agent_steps": int(loaded["trainer_state"]["cumulative_agent_steps"]),
            "simulated_game_hours": float(loaded["trainer_state"]["simulated_game_hours"]),
        },
        "fixed_protocol": {
            "seed": int(args.seed),
            "episodes": int(args.episodes),
            "maximum_ticks_per_episode": int(args.max_ticks_per_episode),
            "policy_mode": "deterministic_tanh_mean_and_button_argmax",
            "opponent": "same_current_scratch_actor_self_play",
            "episode_seed_rule": "base_seed_plus_episode_index",
            "scores_are_recorded_but_not_used_as_a_technical_gate": True,
            "evaluation_agent_steps_are_not_training_experience": True,
        },
        "actor_fingerprint": fingerprint,
        "environment_ticks": total_ticks,
        "agent_steps_evaluated": total_ticks * 2,
        "wall_seconds": time.perf_counter() - started,
        "episodes": episode_reports,
        "metrics": metrics,
        "fixed_action_diagnostics": fixed_action_diagnostics,
        "behavior_signature": _behavior_signature(metrics),
        "checks": {
            "actor_outputs_finite": fingerprint["all_finite"],
            "metric_transport_finite": metrics["finite"],
            "all_requested_episodes_completed": len(episode_reports) == int(args.episodes),
            "no_parser_or_domain_failure": True,
            "fixed_action_diagnostics_finite": fixed_action_diagnostics["all_finite"],
            "scores_excluded_from_pass_fail": True,
        },
    }
    report["checks"]["passed"] = all(report["checks"].values())
    if not report["checks"]["passed"]:
        raise RuntimeError(f"Fixed pilot evaluation failed: {report['checks']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-ticks-per-episode", type=int, default=2400)
    args = parser.parse_args()
    report = run_evaluation(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
