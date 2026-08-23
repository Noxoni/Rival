from __future__ import annotations

import json

import numpy as np
import torch
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator

from rival_training.config import (
    MILESTONE06_CONFIG_PATH,
    canonical_config_sha256,
    load_milestone06_config,
)
from rival_training.curriculum import RivalCurriculumMutator, curriculum_distribution_smoke
from rival_training.deploy import make_exact_policy_export
from rival_training.environment import CampaignGymWrapper, build_campaign_env
from rival_training.metrics import (
    CAMPAIGN_METRIC_VECTOR_SIZE,
    aggregate_campaign_metrics,
)
from rival_training.policy import (
    StudentDiscretePolicy,
    materialize_effective_actor,
    normalize_bootstrap_actor_for_prior,
)
from rival_training.rewards import MECHANICS_METRICS, V2_COMPONENTS, reward_v2_metadata
from rival_training.teacher import build_wisp_student


def test_m06_config_round_trips_and_preserves_cadence_horizon() -> None:
    config = load_milestone06_config()
    reparsed = json.loads(MILESTONE06_CONFIG_PATH.read_text(encoding="utf-8"))
    assert reparsed == config
    assert canonical_config_sha256(reparsed) == canonical_config_sha256(config)
    assert config["ppo"]["gamma"] == 0.99**0.5
    assert config["campaign_ceiling_agent_steps"] == 100_000_000
    assert config["environment"]["workers"] == 56
    assert all(stage["curriculum_weights"]["natural"] >= 0.75 for stage in config["stages"])


def test_checkpointed_appended_prior_is_ppo_consistent_and_materializable() -> None:
    actor = build_wisp_student()
    observations = torch.randn(32, 432, device="cuda")
    with torch.no_grad():
        bootstrap_logits = actor.to("cuda")(observations)
    normalize_bootstrap_actor_for_prior(actor.to("cpu"))
    policy = StudentDiscretePolicy(actor, "cuda", appended_logit_offset=-12.0)
    with torch.no_grad():
        assert torch.equal(bootstrap_logits, policy.logits(observations))
    actions, rollout_log_probs = policy.get_action(observations)
    backprop_log_probs, entropy = policy.get_backprop_data(
        observations, actions.to("cuda").view(-1, 1)
    )
    assert torch.allclose(
        rollout_log_probs,
        backprop_log_probs.detach().cpu().view(-1),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.isfinite(entropy)
    materialized = materialize_effective_actor(policy).to("cuda")
    with torch.no_grad():
        assert torch.equal(policy.logits(observations), materialized(observations))


def test_exact_policy_export_preserves_prior_addition_order() -> None:
    actor = build_wisp_student()
    normalize_bootstrap_actor_for_prior(actor)
    policy = StudentDiscretePolicy(actor, "cpu", appended_logit_offset=-6.0)
    exported = make_exact_policy_export(policy)
    observations = torch.randn(
        64,
        432,
        generator=torch.Generator(device="cpu").manual_seed(20260831),
    )
    with torch.inference_mode():
        assert torch.equal(policy.logits(observations), exported(observations))


def test_stage_a_curriculum_distribution_is_seeded_and_majority_natural() -> None:
    config = load_milestone06_config()
    stage = config["stages"][0]
    engine = RocketSimEngine(rlbot_delay=True)
    try:
        mutator = RivalCurriculumMutator(stage["curriculum_weights"], seed=123)
        report = curriculum_distribution_smoke(
            mutator,
            engine.create_base_state,
            FixedTeamSizeMutator(blue_size=1, orange_size=1),
            samples=2000,
        )
    finally:
        engine.close()
    assert report["shares"]["natural"] >= 0.85
    assert all(report["counts"][name] > 0 for name in report["counts"])


def test_reward_v2_environment_and_components_remain_finite() -> None:
    env = build_campaign_env("stage_a", seed=7)
    rng = np.random.default_rng(7)
    try:
        observations = env.reset()
        for _ in range(128):
            actions = {
                agent: np.array([rng.integers(0, 158)], dtype=np.int64)
                for agent in observations
            }
            observations, rewards, terminated, truncated = env.step(actions)
            assert all(np.isfinite(value).all() for value in observations.values())
            assert all(np.isfinite(value) for value in rewards.values())
            assert all(
                tuple(values) == V2_COMPONENTS
                for values in env.shared_info["reward_components"].values()
            )
            assert all(
                tuple(values) == MECHANICS_METRICS
                for values in env.shared_info["mechanics_metrics"].values()
            )
            if any(terminated.values()) or any(truncated.values()):
                observations = env.reset()
    finally:
        env.close()
    metadata = reward_v2_metadata()
    assert metadata["outcome_goal"] == 10.0
    assert metadata["mechanics_absolute_step_cap"] == 0.02
    assert not metadata["named_mechanic_rewards_enabled"]


def test_campaign_metric_transport_and_aggregation() -> None:
    wrapped = CampaignGymWrapper(build_campaign_env("stage_a", seed=9))
    try:
        observations = wrapped.reset()
        actions = np.zeros(len(observations), dtype=np.int64)
        _, _, _, _, info = wrapped.step(actions)
        vector = info["state"]
    finally:
        wrapped.close()
    assert vector.shape == (CAMPAIGN_METRIC_VECTOR_SIZE,)
    report = aggregate_campaign_metrics([vector])
    assert report["metric_vector_count"] == 1
    assert tuple(report["reward_components"]) == V2_COMPONENTS
    assert report["curriculum_reset_counts"]["natural"] in (0, 1)
