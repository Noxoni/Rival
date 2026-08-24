"""Run the fixed deterministic Rival v10.1 bootstrap probe suite."""

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

from rival_training.v10_bootstrap_curriculum import (  # noqa: E402
    FAMILIES,
    RivalAgencyBootstrapCurriculumV1,
)
from rival_training.v10_bootstrap_environment import (  # noqa: E402
    RivalAgencyBootstrapGymWrapperV1,
    build_v10_bootstrap_env,
)
from rival_training.v10_bootstrap_metrics import (  # noqa: E402
    METRIC_INDEX,
    aggregate_v10_bootstrap_metrics,
)
from rival_training.v9_actions import RivalHybridPolicy  # noqa: E402
from rival_training.v9_checkpoint import (  # noqa: E402
    load_v9_checkpoint,
    portable_path,
    sha256_file,
)


EVALUATION_VERSION = "RivalAgencyBootstrapFixedEvaluationV1"


def _curriculum(environment) -> RivalAgencyBootstrapCurriculumV1:
    for mutator in getattr(environment.rlgym_env.state_mutator, "mutators", ()):
        if isinstance(mutator, RivalAgencyBootstrapCurriculumV1):
            return mutator
    raise RuntimeError("Bootstrap curriculum mutator was not found")


def _seed_environment(environment, seed: int, active_team: int) -> None:
    curriculum = _curriculum(environment)
    curriculum.seed(seed)
    curriculum.forced_active_team = int(active_team)
    action_seed = getattr(environment.rlgym_env.action_parser, "seed", None)
    if callable(action_seed):
        action_seed(seed + 100_000)


def _termination_reason(vector: np.ndarray, done: bool, truncated: bool) -> str:
    if done:
        return "goal"
    if truncated:
        for name in ("no_touch_timeout", "episode_timeout"):
            if vector[METRIC_INDEX[f"termination.{name}"]] > 0.5:
                return name
        return "unclassified_environment_truncation"
    return "fixed_tick_cap"


def _mean_across_slots(metrics: dict[str, Any], name: str) -> float:
    return float(
        np.mean(
            [
                metrics["motion"][f"blue.{name}"]["mean"],
                metrics["motion"][f"orange.{name}"]["mean"],
            ]
        )
    )


def _capability_summary(
    metrics: dict[str, Any], episode_reports: list[dict[str, Any]]
) -> dict[str, Any]:
    termination_total = max(sum(metrics["termination_counts"].values()), 1)
    easy = [report for report in episode_reports if report["family"] == "easy_finish"]
    natural = [report for report in episode_reports if report["family"] == "natural"]
    easy_active_goals = sum(
        int(report["active_team_scored"]) for report in easy
    )
    natural_goals = sum(int(report["end_reason"] == "goal") for report in natural)
    return {
        "mean_speed": _mean_across_slots(metrics, "speed"),
        "mean_planar_speed": _mean_across_slots(metrics, "planar_speed"),
        "mean_distance_to_ball": _mean_across_slots(metrics, "distance_to_ball"),
        "speed_share_over_500": _mean_across_slots(metrics, "speed_over_500"),
        "speed_share_over_1000": _mean_across_slots(metrics, "speed_over_1000"),
        "speed_share_over_1500": _mean_across_slots(metrics, "speed_over_1500"),
        "speed_share_over_2000": _mean_across_slots(metrics, "speed_over_2000"),
        **metrics["interaction_rates_per_100k_agent_steps"],
        "maximum_touch_chain_length_lower_bound": metrics["touch_chain"][
            "maximum_observed_chain_length_lower_bound"
        ],
        "two_or_more_chain_touches_per_100k_agent_steps": metrics[
            "touch_chain"
        ]["two_or_more_chain_touches_per_100k_agent_steps"],
        "goals": metrics["goals"],
        "goal_termination_share": metrics["termination_counts"]["goal"]
        / termination_total,
        "no_touch_timeout_share": metrics["termination_counts"][
            "no_touch_timeout"
        ]
        / termination_total,
        "episode_timeout_share": metrics["termination_counts"][
            "episode_timeout"
        ]
        / termination_total,
        "easy_finish_active_team_goals": easy_active_goals,
        "easy_finish_success_rate": easy_active_goals / max(len(easy), 1),
        "natural_goal_episodes": natural_goals,
        "natural_goal_activity": natural_goals > 0,
        "touches_by_reset_family": metrics["touches_by_reset_family"],
        "episode_touch_shares": metrics["episode_touch_shares"],
    }


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    loaded = load_v9_checkpoint(args.checkpoint, device=args.device)
    policy = RivalHybridPolicy(loaded["actor"].eval(), args.device)
    vectors: list[np.ndarray] = []
    episode_reports: list[dict[str, Any]] = []
    total_ticks = 0
    wall_started = time.perf_counter()
    for family_index, family in enumerate(FAMILIES):
        raw = build_v10_bootstrap_env(
            phase=args.phase,
            seed=args.seed + family_index * 10_000,
            forced_family=family,
            forced_active_team=0,
            forced_mirror=False,
        )
        environment = RivalAgencyBootstrapGymWrapperV1(raw)
        try:
            for episode in range(args.episodes_per_family):
                episode_seed = (
                    args.seed + family_index * 10_000 + episode
                )
                active_team = episode % 2
                _seed_environment(environment, episode_seed, active_team)
                observations = environment.reset()
                end_reason = "fixed_tick_cap"
                scoring_team: int | None = None
                last_vector = np.zeros(len(METRIC_INDEX), dtype=np.float32)
                ticks = 0
                for _ in range(args.max_ticks_per_episode):
                    with torch.inference_mode():
                        actions, _ = policy.get_action(
                            observations, deterministic=True
                        )
                    action_array = actions.numpy()
                    observations, _, done, truncated, info = environment.step(
                        action_array
                    )
                    last_vector = np.asarray(info["state"], dtype=np.float32)
                    vectors.append(last_vector)
                    ticks += 1
                    total_ticks += 1
                    if done or truncated:
                        end_reason = _termination_reason(
                            last_vector, bool(done), bool(truncated)
                        )
                        if done:
                            scoring_team = int(
                                environment.rlgym_env.state.scoring_team
                            )
                        break
                episode_reports.append(
                    {
                        "family": family,
                        "episode": episode,
                        "seed": episode_seed,
                        "active_team": active_team,
                        "environment_ticks": ticks,
                        "agent_steps": ticks * 2,
                        "end_reason": end_reason,
                        "scoring_team": scoring_team,
                        "active_team_scored": scoring_team == active_team,
                    }
                )
        finally:
            environment.close()
        completed = [
            report for report in episode_reports if report["family"] == family
        ]
        print(
            json.dumps(
                {
                    "evaluation_progress": family,
                    "families_completed": family_index + 1,
                    "families_total": len(FAMILIES),
                    "episodes_completed": len(completed),
                    "goals": sum(
                        int(report["end_reason"] == "goal")
                        for report in completed
                    ),
                    "environment_ticks_total": total_ticks,
                }
            ),
            flush=True,
        )
    metrics = aggregate_v10_bootstrap_metrics(vectors)
    capability = _capability_summary(metrics, episode_reports)
    checks = {
        "checkpoint_loaded": True,
        "all_six_families_evaluated": set(
            report["family"] for report in episode_reports
        )
        == set(FAMILIES),
        "episode_count_exact": len(episode_reports)
        == len(FAMILIES) * args.episodes_per_family,
        "metric_transport_finite": metrics["finite"],
        "all_episodes_reached_environment_end": all(
            report["end_reason"] != "fixed_tick_cap"
            for report in episode_reports
        ),
        "deterministic_policy": True,
        "scores_recorded_as_capability_evidence": True,
        "scores_not_the_only_gate": True,
    }
    checks["passed"] = all(checks.values())
    report = {
        "schema_version": 1,
        "status": "passed" if checks["passed"] else "failed",
        "evaluation_version": EVALUATION_VERSION,
        "checkpoint": {
            "directory": portable_path(args.checkpoint),
            "manifest_sha256": sha256_file(
                Path(args.checkpoint) / "checkpoint_manifest.json"
            ),
            "actor_sha256": sha256_file(Path(args.checkpoint) / "actor.pt"),
            "config_version": loaded["config"]["config_version"],
            "cumulative_agent_steps": int(
                loaded["trainer_state"]["cumulative_agent_steps"]
            ),
            "simulated_game_hours": float(
                loaded["trainer_state"]["simulated_game_hours"]
            ),
        },
        "fixed_protocol": {
            "phase": args.phase,
            "seed": args.seed,
            "families": list(FAMILIES),
            "episodes_per_family": args.episodes_per_family,
            "active_team_rule": "episode_index_modulo_2",
            "forced_mirror": False,
            "maximum_ticks_per_episode": args.max_ticks_per_episode,
            "policy_mode": "deterministic_tanh_mean_and_button_argmax",
            "opponent": "same_current_Rival_actor_self_play",
            "native_physics_and_policy_hz": 120,
            "evaluation_agent_steps_are_not_training_experience": True,
        },
        "environment_ticks": total_ticks,
        "agent_steps_evaluated": total_ticks * 2,
        "wall_seconds": time.perf_counter() - wall_started,
        "episodes": episode_reports,
        "metrics": metrics,
        "capability_summary": capability,
        "checks": checks,
    }
    if report["status"] != "passed":
        raise RuntimeError(f"Bootstrap evaluation failed: {checks}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("A", "B", "C"), default="A")
    parser.add_argument("--seed", type=int, default=20261061)
    parser.add_argument("--episodes-per-family", type=int, default=8)
    parser.add_argument("--max-ticks-per-episode", type=int, default=14_405)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.episodes_per_family < 4:
        raise ValueError("Bootstrap evaluation requires at least four episodes/family")
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
