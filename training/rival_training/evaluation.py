"""Deterministic natural headless evaluation against the frozen Wisp teacher."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .actions import action_family, build_expanded_action_table
from .campaign import load_campaign_state, make_campaign_ppo
from .checkpoint import load_actor_checkpoint, portable_path
from .config import REPOSITORY_ROOT, canonical_config_sha256, load_milestone06_config
from .environment import build_campaign_env
from .policy import StudentDiscretePolicy, normalize_bootstrap_actor_for_prior
from .rewards import MECHANICS_METRICS, V2_COMPONENTS
from .teacher import FrozenWispReference, sha256_file


def load_evaluation_policy(
    *,
    checkpoint_directory: str | Path | None,
    bootstrap_offset: float | None,
    device: str,
) -> tuple[StudentDiscretePolicy, dict[str, Any]]:
    config = load_milestone06_config()
    if checkpoint_directory is not None:
        directory = Path(checkpoint_directory)
        state = load_campaign_state(directory)
        offset = float(state["action_exploration_prior"]["appended_logit_offset"])
        ppo = make_campaign_ppo(
            config,
            device=device,
            appended_logit_offset=offset,
        )
        ppo.load_from(str(directory))
        policy = ppo.policy.eval()
        source = {
            "kind": "full_campaign_checkpoint",
            "directory": portable_path(directory),
            "state": state,
            "files": {
                path.name: {
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(directory.iterdir())
                if path.is_file()
            },
        }
        return policy, source
    if bootstrap_offset is None:
        raise ValueError("Bootstrap evaluation requires an explicit calibrated offset")
    path = REPOSITORY_ROOT / "training/artifacts/bootstrap/wisp_student_expanded_v1.pt"
    actor, metadata = load_actor_checkpoint(path, "cpu")
    normalize_bootstrap_actor_for_prior(actor)
    policy = StudentDiscretePolicy(
        actor,
        device,
        appended_logit_offset=float(bootstrap_offset),
    ).eval()
    return policy, {
        "kind": "frozen_bootstrap_with_checkpointed_prior",
        "path": portable_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "metadata": metadata,
        "appended_logit_offset": float(bootstrap_offset),
    }


def _component_summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "mean": 0.0,
            "absolute_mean": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
            "cumulative_signed": 0.0,
            "cumulative_absolute": 0.0,
        }
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "absolute_mean": float(np.abs(array).mean()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "cumulative_signed": float(array.sum()),
        "cumulative_absolute": float(np.abs(array).sum()),
    }


@torch.inference_mode()
def evaluate_frozen_wisp(
    policy: StudentDiscretePolicy,
    source: dict[str, Any],
    *,
    games: int,
    seed: int,
    device: str = "cuda:0",
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if games < 2 or games % 2:
        raise ValueError("Headless evaluation requires an even balanced game count")
    config = load_milestone06_config()
    teacher = FrozenWispReference().to(device).eval()
    policy = policy.to(device).eval()
    table = build_expanded_action_table()
    env = build_campaign_env("stage_a", seed=seed, natural_only=True)
    curriculum = env.state_mutator.mutators[-1]
    curriculum.seed(seed)
    wins = losses = ties = goals_for = goals_against = 0
    student_action_counts = np.zeros(158, dtype=np.int64)
    family_counts: Counter[str] = Counter()
    appended_probability_mass: list[float] = []
    reward_values = {name: [] for name in V2_COMPONENTS}
    mechanic_totals = Counter()
    touches = 0
    game_records = []
    started = time.perf_counter()
    try:
        for game_index in range(games):
            observations = env.reset()
            student_team = game_index % 2
            student_agent = next(
                agent
                for agent in observations
                if env.state.cars[agent].team_num == student_team
            )
            teacher_agent = next(agent for agent in observations if agent != student_agent)
            teacher_action = 0
            decisions = 0
            while True:
                student_obs = torch.from_numpy(observations[student_agent]).to(
                    device
                ).unsqueeze(0)
                student_logits = policy.logits(student_obs).squeeze(0)
                student_action = int(student_logits.argmax().item())
                probabilities = torch.softmax(student_logits, dim=-1)
                appended_probability_mass.append(
                    float(probabilities[90:].sum().item())
                )
                # Frozen Wisp acts every eight physics ticks while the student
                # acts every four. Cache its action for every second env decision.
                if decisions % 2 == 0:
                    teacher_obs = torch.from_numpy(observations[teacher_agent]).to(
                        device
                    ).unsqueeze(0)
                    teacher_action = int(teacher(teacher_obs).argmax(dim=-1).item())
                actions = {
                    student_agent: np.array([student_action], dtype=np.int64),
                    teacher_agent: np.array([teacher_action], dtype=np.int64),
                }
                observations, _, terminated, truncated = env.step(actions)
                decisions += 1
                student_action_counts[student_action] += 1
                family_counts[action_family(student_action, table)] += 1
                touches += int(env.state.cars[student_agent].ball_touches > 0)
                for name, value in env.shared_info["reward_components"][
                    student_agent
                ].items():
                    reward_values[name].append(float(value))
                for name, value in env.shared_info["mechanics_metrics"][
                    student_agent
                ].items():
                    mechanic_totals[name] += float(value)
                if any(terminated.values()) or any(truncated.values()):
                    goal = bool(env.state.goal_scored)
                    if goal and env.state.scoring_team == student_team:
                        wins += 1
                        goals_for += 1
                        outcome = "win"
                    elif goal:
                        losses += 1
                        goals_against += 1
                        outcome = "loss"
                    else:
                        ties += 1
                        outcome = "tie"
                    game_records.append(
                        {
                            "game": game_index + 1,
                            "student_team": student_team,
                            "student_side": "blue" if student_team == 0 else "orange",
                            "outcome": outcome,
                            "decisions": decisions,
                            "terminated_by_goal": goal,
                        }
                    )
                    break
    finally:
        env.close()

    component_audit = {
        name: _component_summary(values) for name, values in reward_values.items()
    }
    shaping_absolute = sum(
        float(record["cumulative_absolute"])
        for name, record in component_audit.items()
        if name != "outcome"
    )
    for name, record in component_audit.items():
        record["share_of_total_absolute_shaping"] = (
            0.0
            if name == "outcome" or shaping_absolute <= 0
            else float(record["cumulative_absolute"] / shaping_absolute)
        )
    sampled_actions = int(student_action_counts.sum())
    win_rate = wins / games
    appended_share = float(student_action_counts[90:].sum() / max(sampled_actions, 1))
    health_checks = {
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for value in (
                win_rate,
                appended_share,
                *mechanic_totals.values(),
                *appended_probability_mass,
            )
        ),
        "balanced_sides": sum(item["student_team"] == 0 for item in game_records)
        == sum(item["student_team"] == 1 for item in game_records),
        "appended_share_below_rejection_gate": appended_share
        <= float(config["evaluation"]["maximum_appended_action_share_for_health"]),
    }
    baseline_comparison = None
    if baseline is not None:
        baseline_win_rate = float(baseline["outcomes"]["win_rate"])
        win_rate_drop = baseline_win_rate - win_rate
        baseline_comparison = {
            "baseline_win_rate": baseline_win_rate,
            "candidate_win_rate": win_rate,
            "win_rate_drop": win_rate_drop,
            "maximum_allowed_drop": float(
                config["evaluation"]["headless_collapse_win_rate_drop"]
            ),
            "not_collapsed": win_rate_drop
            <= float(config["evaluation"]["headless_collapse_win_rate_drop"]),
        }
        health_checks["frozen_wisp_performance_not_collapsed"] = baseline_comparison[
            "not_collapsed"
        ]
    health_checks["passed"] = all(health_checks.values())
    return {
        "schema_version": 1,
        "status": "passed" if health_checks["passed"] else "rejected",
        "evaluation": "deterministic_headless_frozen_wisp",
        "episode_definition": "natural kickoff until first goal or configured truncation",
        "games": games,
        "balanced_sides": True,
        "student_cadence_ticks": 4,
        "teacher_cadence_ticks": 8,
        "deterministic": True,
        "seed": seed,
        "config_sha256": canonical_config_sha256(config),
        "student_source": source,
        "action_exploration_prior": policy.prior_state(),
        "outcomes": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": win_rate,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_differential": goals_for - goals_against,
        },
        "actions": {
            "sampled_action_count": sampled_actions,
            "appended_action_count": int(student_action_counts[90:].sum()),
            "appended_action_share": appended_share,
            "mean_appended_probability_mass": float(
                np.mean(appended_probability_mass)
            ),
            "full_action_counts": student_action_counts.tolist(),
            "action_family_counts": dict(sorted(family_counts.items())),
        },
        "possession": {"student_touch_proxy_count": touches},
        "reward_contribution_audit": component_audit,
        "mechanics_recovery": {
            name: float(mechanic_totals[name]) for name in MECHANICS_METRICS
        },
        "game_records": game_records,
        "baseline_comparison": baseline_comparison,
        "health": health_checks,
        "wall_seconds": time.perf_counter() - started,
    }
