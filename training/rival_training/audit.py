"""Bounded Reward V2 contribution and curriculum distribution audits."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator

from .checkpoint import load_actor_checkpoint
from .config import REPOSITORY_ROOT, load_milestone06_config
from .curriculum import (
    CURRICULUM_FAMILIES,
    RivalCurriculumMutator,
    curriculum_distribution_smoke,
)
from .environment import build_campaign_env
from .policy import StudentDiscretePolicy, normalize_bootstrap_actor_for_prior
from .rewards import MECHANICS_METRICS, V2_COMPONENTS


class _Accumulator:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.values = {name: [] for name in names}

    def add(self, values: dict[str, float]) -> None:
        for name in self.values:
            self.values[name].append(float(values.get(name, 0.0)))

    def report(self) -> dict[str, Any]:
        result = {}
        for name, values in self.values.items():
            array = np.asarray(values, dtype=np.float64)
            result[name] = {
                "count": int(len(array)),
                "mean": float(array.mean()) if len(array) else 0.0,
                "absolute_mean": float(np.abs(array).mean()) if len(array) else 0.0,
                "minimum": float(array.min()) if len(array) else 0.0,
                "maximum": float(array.max()) if len(array) else 0.0,
                "cumulative_signed": float(array.sum()),
                "cumulative_absolute": float(np.abs(array).sum()),
            }
        return result


def run_reward_v2_rollout_audit(
    *,
    mode: str,
    decisions: int,
    seed: int,
    appended_logit_offset: float,
) -> dict[str, Any]:
    if mode not in {"natural_wisp", "weighted_uniform_random"}:
        raise ValueError(f"Unknown Reward V2 audit mode {mode}")
    env = build_campaign_env(
        "stage_a",
        seed=seed,
        natural_only=mode == "natural_wisp",
    )
    rng = np.random.default_rng(seed)
    policy = None
    if mode == "natural_wisp":
        actor_path = (
            REPOSITORY_ROOT / "training/artifacts/bootstrap/wisp_student_expanded_v1.pt"
        )
        actor, _ = load_actor_checkpoint(actor_path, "cpu")
        normalize_bootstrap_actor_for_prior(actor)
        policy = StudentDiscretePolicy(
            actor,
            "cpu",
            appended_logit_offset=appended_logit_offset,
        ).eval()
    components = _Accumulator(V2_COMPONENTS)
    mechanics = _Accumulator(MECHANICS_METRICS)
    action_counts = np.zeros(158, dtype=np.int64)
    episodes = 1
    goal_terminations = 0
    truncations = 0
    observations = env.reset()
    try:
        for _ in range(decisions):
            agents = list(observations)
            if policy is None:
                selected = rng.integers(0, 158, size=len(agents))
            else:
                batch = np.stack([observations[agent] for agent in agents])
                with torch.inference_mode():
                    actions, _ = policy.get_action(batch, deterministic=False)
                selected = np.asarray(actions, dtype=np.int64)
            action_dict = {
                agent: np.array([int(selected[index])], dtype=np.int64)
                for index, agent in enumerate(agents)
            }
            for value in selected:
                action_counts[int(value)] += 1
            observations, rewards, terminated, truncated = env.step(action_dict)
            if not all(math.isfinite(float(value)) for value in rewards.values()):
                raise FloatingPointError("Non-finite Reward V2 audit reward")
            for agent in agents:
                components.add(env.shared_info["reward_components"][agent])
                mechanics.add(env.shared_info["mechanics_metrics"][agent])
            if any(terminated.values()) or any(truncated.values()):
                goal_terminations += int(any(terminated.values()))
                truncations += int(any(truncated.values()))
                episodes += 1
                observations = env.reset()
        reset_counts = {
            name: int(env.shared_info.get("curriculum_reset_counts", {}).get(name, 0))
            for name in CURRICULUM_FAMILIES
        }
    finally:
        env.close()
    component_report = components.report()
    shaping_absolute = sum(
        float(item["cumulative_absolute"])
        for name, item in component_report.items()
        if name != "outcome"
    )
    for name, item in component_report.items():
        item["share_of_total_absolute_shaping"] = (
            0.0
            if name == "outcome" or shaping_absolute <= 0
            else float(item["cumulative_absolute"] / shaping_absolute)
        )
    return {
        "mode": mode,
        "decisions": decisions,
        "agent_steps": decisions * 2,
        "episodes": episodes,
        "goal_terminations": goal_terminations,
        "truncations": truncations,
        "reward_components": component_report,
        "mechanics_recovery": mechanics.report(),
        "actions": {
            "full_action_counts": action_counts.tolist(),
            "appended_action_count": int(action_counts[90:].sum()),
            "appended_action_share": float(
                action_counts[90:].sum() / max(action_counts.sum(), 1)
            ),
        },
        "curriculum_reset_counts": reset_counts,
        "all_finite": True,
    }


def run_curriculum_audit(*, samples_per_stage: int, seed: int) -> dict[str, Any]:
    config = load_milestone06_config()
    engine = RocketSimEngine(rlbot_delay=True)
    team_size = FixedTeamSizeMutator(blue_size=1, orange_size=1)
    reports = {}
    try:
        for stage_index, stage in enumerate(config["stages"]):
            mutator = RivalCurriculumMutator(
                stage["curriculum_weights"], seed=seed + stage_index
            )
            report = curriculum_distribution_smoke(
                mutator,
                engine.create_base_state,
                team_size,
                samples=samples_per_stage,
            )
            deviations = {
                name: abs(
                    report["shares"][name] - report["configured_weights"][name]
                )
                for name in CURRICULUM_FAMILIES
            }
            report["maximum_absolute_share_error"] = max(deviations.values())
            report["natural_majority_observed"] = report["shares"]["natural"] >= 0.73
            report["passed"] = bool(
                report["maximum_absolute_share_error"] <= 0.02
                and report["natural_majority_observed"]
            )
            reports[stage["name"]] = report
    finally:
        engine.close()
    return {
        "samples_per_stage": samples_per_stage,
        "stages": reports,
        "passed": all(report["passed"] for report in reports.values()),
    }


def reward_audit_health(report: dict[str, Any]) -> dict[str, Any]:
    mechanics_max = max(
        abs(float(item["minimum"]))
        for rollout in report["rollouts"]
        for name, item in rollout["reward_components"].items()
        if name == "mechanics_resource"
    )
    mechanics_max = max(
        mechanics_max,
        max(
            abs(float(item["maximum"]))
            for rollout in report["rollouts"]
            for name, item in rollout["reward_components"].items()
            if name == "mechanics_resource"
        ),
    )
    recovery_max = max(
        abs(float(item[bound]))
        for rollout in report["rollouts"]
        for name, item in rollout["reward_components"].items()
        if name == "recovery"
        for bound in ("minimum", "maximum")
    )
    mechanics_shares = [
        float(rollout["reward_components"]["mechanics_resource"][
            "share_of_total_absolute_shaping"
        ])
        for rollout in report["rollouts"]
    ]
    checks = {
        "all_rollouts_finite": all(item["all_finite"] for item in report["rollouts"]),
        "mechanics_step_cap_respected": mechanics_max <= 0.0200001,
        "recovery_step_scale_bounded": recovery_max <= 0.0300001,
        "mechanics_not_dominant": max(mechanics_shares) <= 0.25,
    }
    checks["passed"] = all(checks.values())
    return checks
