"""Deterministic dual-rate evaluation against a forced-PASS frozen-Wisp anchor."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from .actions import action_family
from .config import (
    REPOSITORY_ROOT,
    canonical_config_sha256,
    load_milestone08_config,
)
from .environment import build_dual_rate_env
from .m08_campaign import frozen_strategic_proof, load_m08_state, make_m08_ppo
from .m08_metrics import M08_CONTEXTS, mechanics_context
from .mechanics import load_mechanics_actor
from .policy import MechanicsDiscretePolicy
from .rewards import MECHANICS_METRICS, V2_COMPONENTS
from .teacher import sha256_file


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_m08_evaluation_policy(
    checkpoint_directory: str | Path | None,
    *,
    device: str,
) -> tuple[MechanicsDiscretePolicy, dict[str, Any]]:
    config = load_milestone08_config()
    if checkpoint_directory is None:
        path = (
            REPOSITORY_ROOT
            / "training/artifacts/milestone08/mechanics_initial_v1.pt"
        )
        actor, metadata = load_mechanics_actor(path, device="cpu")
        return MechanicsDiscretePolicy(actor, device).eval(), {
            "kind": "calibrated_zero_step_mechanics_actor",
            "path": _portable(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "metadata": metadata,
        }
    directory = Path(checkpoint_directory).resolve()
    state = load_m08_state(directory)
    ppo = make_m08_ppo(config, device=device)
    ppo.load_from(str(directory))
    return ppo.policy.eval(), {
        "kind": "m08_full_ppo_checkpoint",
        "directory": _portable(directory),
        "state": state,
        "files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(directory.iterdir())
            if path.is_file()
        },
    }


def _summary(values: list[float]) -> dict[str, float | int]:
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
def evaluate_m08_frozen_anchor(
    policy: MechanicsDiscretePolicy,
    source: dict[str, Any],
    *,
    games: int,
    seed: int,
    device: str = "cuda:0",
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if games < 2 or games % 2:
        raise ValueError("M08 headless evaluation requires an even game count")
    config = load_milestone08_config()
    policy = policy.to(device).eval()
    env = build_dual_rate_env(seed=seed, natural_only=True)
    curriculum = env.state_mutator.mutators[-1]
    curriculum.seed(seed)
    wins = losses = ties = goals_for = goals_against = 0
    action_counts = np.zeros(69, dtype=np.int64)
    probability_pass: list[float] = []
    entropies: list[float] = []
    reward_values = {name: [] for name in V2_COMPONENTS}
    mechanics_totals = Counter()
    context_counts = Counter()
    family_counts = Counter()
    touches = 0
    override_windows = 0
    useful_touches_after_override = 0
    goals_for_after_override = 0
    goals_against_after_override = 0
    game_records = []
    started = time.perf_counter()
    try:
        for game_index in range(games):
            observations = env.reset()
            candidate_team = game_index % 2
            candidate_agent = next(
                agent
                for agent in observations
                if env.state.cars[agent].team_num == candidate_team
            )
            anchor_agent = next(
                agent for agent in observations if agent != candidate_agent
            )
            decisions = 0
            recent_override = 0
            game_overrides = 0
            game_touches_after_override = 0
            while True:
                tensor = torch.from_numpy(observations[candidate_agent]).to(
                    device
                ).unsqueeze(0)
                logits = policy.logits(tensor).squeeze(0)
                probabilities = torch.softmax(logits, dim=-1)
                choice = int(logits.argmax().item())
                probability_pass.append(float(probabilities[0].item()))
                entropies.append(
                    float(
                        (-(probabilities * torch.log(torch.clamp(probabilities, 1e-12))))
                        .sum()
                        .item()
                    )
                )
                action_counts[choice] += 1
                if choice != 0:
                    global_index = 89 + choice
                    family_counts[action_family(global_index)] += 1
                    context_counts[
                        mechanics_context(env.state.cars[candidate_agent], env.state)
                    ] += 1
                    recent_override = 30
                    override_windows += 1
                    game_overrides += 1
                actions = {
                    candidate_agent: np.asarray([choice], dtype=np.int64),
                    anchor_agent: np.asarray([0], dtype=np.int64),
                }
                observations, _, terminated, truncated = env.step(actions)
                decisions += 1
                touched = int(env.state.cars[candidate_agent].ball_touches > 0)
                touches += touched
                if recent_override > 0 and touched:
                    useful_touches_after_override += 1
                    game_touches_after_override += 1
                for name, value in env.shared_info["reward_components"][
                    candidate_agent
                ].items():
                    reward_values[name].append(float(value))
                for name, value in env.shared_info["mechanics_metrics"][
                    candidate_agent
                ].items():
                    mechanics_totals[name] += float(value)
                if any(terminated.values()) or any(truncated.values()):
                    goal = bool(env.state.goal_scored)
                    if goal and env.state.scoring_team == candidate_team:
                        wins += 1
                        goals_for += 1
                        outcome = "win"
                        if recent_override > 0:
                            goals_for_after_override += 1
                    elif goal:
                        losses += 1
                        goals_against += 1
                        outcome = "loss"
                        if recent_override > 0:
                            goals_against_after_override += 1
                    else:
                        ties += 1
                        outcome = "tie"
                    game_records.append(
                        {
                            "game": game_index + 1,
                            "candidate_team": candidate_team,
                            "candidate_side": (
                                "blue" if candidate_team == 0 else "orange"
                            ),
                            "outcome": outcome,
                            "decisions": decisions,
                            "terminated_by_goal": goal,
                            "deterministic_override_count": game_overrides,
                            "touches_after_override": game_touches_after_override,
                        }
                    )
                    break
                recent_override = max(0, recent_override - 1)
    finally:
        env.close()

    component_audit = {
        name: _summary(values) for name, values in reward_values.items()
    }
    shaping_absolute = sum(
        float(item["cumulative_absolute"])
        for name, item in component_audit.items()
        if name != "outcome"
    )
    for name, item in component_audit.items():
        item["share_of_total_absolute_shaping"] = (
            0.0
            if name == "outcome" or shaping_absolute <= 0.0
            else float(item["cumulative_absolute"] / shaping_absolute)
        )
    sampled_actions = int(action_counts.sum())
    override_count = int(action_counts[1:].sum())
    override_rate = override_count / max(sampled_actions, 1)
    win_rate = wins / games
    strategic = frozen_strategic_proof(config)
    baseline_comparison = None
    if baseline is not None:
        baseline_comparison = {
            "baseline_win_rate": float(baseline["outcomes"]["win_rate"]),
            "candidate_win_rate": win_rate,
            "win_rate_change": win_rate
            - float(baseline["outcomes"]["win_rate"]),
            "baseline_goal_differential": int(
                baseline["outcomes"]["goal_differential"]
            ),
            "candidate_goal_differential": goals_for - goals_against,
            "goal_differential_change": goals_for
            - goals_against
            - int(baseline["outcomes"]["goal_differential"]),
        }
    health = {
        "all_metrics_finite": all(
            math.isfinite(float(value))
            for value in (
                win_rate,
                override_rate,
                *probability_pass,
                *entropies,
                *mechanics_totals.values(),
            )
        ),
        "balanced_sides": sum(item["candidate_team"] == 0 for item in game_records)
        == sum(item["candidate_team"] == 1 for item in game_records),
        "override_rate_bounded": override_rate
        <= float(config["evaluation"]["maximum_sampled_override_share"]),
        "strategic_branch_unchanged": strategic["all_unchanged"],
    }
    health["passed"] = all(health.values())
    return {
        "schema_version": 1,
        "status": "passed" if health["passed"] else "rejected",
        "evaluation": "m08_deterministic_dual_rate_vs_forced_pass_frozen_wisp",
        "episode_definition": "natural kickoff until first goal or truncation",
        "games": games,
        "balanced_sides": True,
        "strategic_cadence_ticks": 8,
        "mechanics_cadence_ticks": 4,
        "deterministic": True,
        "seed": seed,
        "config_sha256": canonical_config_sha256(config),
        "candidate_source": source,
        "mechanics_prior": policy.prior_state(),
        "strategic_branch": strategic,
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
            "decision_count": sampled_actions,
            "pass_count": int(action_counts[0]),
            "override_count": override_count,
            "deterministic_pass_rate": float(action_counts[0] / max(sampled_actions, 1)),
            "deterministic_override_rate": override_rate,
            "mean_pass_probability": float(np.mean(probability_pass)),
            "mean_override_probability": float(1.0 - np.mean(probability_pass)),
            "mean_policy_entropy": float(np.mean(entropies)),
            "full_mechanics_action_counts": action_counts.tolist(),
            "appended_action_family_counts": dict(sorted(family_counts.items())),
        },
        "override_contexts": {
            "counts": {name: int(context_counts[name]) for name in M08_CONTEXTS},
            "shares": {
                name: float(context_counts[name] / max(override_count, 1))
                for name in M08_CONTEXTS
            },
        },
        "short_window_after_override": {
            "window_mechanics_decisions": 30,
            "override_windows_started": override_windows,
            "useful_touch_records": useful_touches_after_override,
            "goals_for": goals_for_after_override,
            "goals_against": goals_against_after_override,
        },
        "possession_touch": {"candidate_touch_proxy_count": touches},
        "reward_contribution_audit": component_audit,
        "mechanics_recovery_resource": {
            name: float(mechanics_totals[name]) for name in MECHANICS_METRICS
        },
        "game_records": game_records,
        "baseline_comparison": baseline_comparison,
        "health": health,
        "wall_seconds": time.perf_counter() - started,
        "production_promoted": False,
    }
