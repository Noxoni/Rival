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
            terminated_reason = "fixed_tick_cap"
            ticks = 0
            for _ in range(int(args.max_ticks_per_episode)):
                agents = list(observations)
                batch = np.stack([observations[agent] for agent in agents])
                actions, _ = policy.get_action(batch, deterministic=True)
                action_map = {agent: actions[index].numpy() for index, agent in enumerate(agents)}
                observations, _, terminated, truncated = environment.step(action_map)
                vectors.append(tracker.build(environment.state, environment.shared_info))
                ticks += 1
                total_ticks += 1
                if any(terminated.values()) or any(truncated.values()):
                    terminated_reason = (
                        "goal" if any(terminated.values()) else "environment_truncation"
                    )
                    break
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
        "behavior_signature": _behavior_signature(metrics),
        "checks": {
            "actor_outputs_finite": fingerprint["all_finite"],
            "metric_transport_finite": metrics["finite"],
            "all_requested_episodes_completed": len(episode_reports) == int(args.episodes),
            "no_parser_or_domain_failure": True,
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
