from __future__ import annotations

import json

import numpy as np
import torch

from rival_training.config import canonical_config_sha256, load_milestone08_config
from rival_training.m08_campaign import (
    M08_CHECKPOINT_FORMAT,
    M08_STATE_FILE,
    _validate_worker_transition,
    make_m08_ppo,
    verify_m08_checkpoint,
)
from rival_training.mechanics import mechanics_state_sha256
from rival_training.policy import (
    MechanicsActor,
    MechanicsDiscretePolicy,
    calibrate_mechanics_pass_prior,
)


def test_m08_config_is_bounded_and_natural_majority() -> None:
    config = load_milestone08_config()
    assert config["campaign_ceiling_agent_steps"] == 5_000_000
    assert config["boundaries_agent_steps"] == [500_000, 1_000_000, 2_000_000, 5_000_000]
    assert config["environment"]["curriculum_weights"]["natural"] >= 0.80
    assert config["production_promotion_authorized"] is False


def test_mechanics_actor_is_separate_69_output_rlgym_policy() -> None:
    actor = MechanicsActor(seed=8)
    policy = MechanicsDiscretePolicy(actor, "cpu")
    observations = torch.randn(17, 432)
    logits = policy.logits(observations)
    assert logits.shape == (17, 69)
    actions, log_probs = policy.get_action(observations)
    assert actions.shape == (17,)
    assert log_probs.shape == (17,)
    selected, entropy = policy.get_backprop_data(observations, actions[:, None])
    assert selected.shape == (17, 1)
    assert torch.isfinite(entropy)
    assert all(parameter.requires_grad for parameter in actor.parameters())


def test_mechanics_pass_prior_is_measured_and_deterministic() -> None:
    rng = np.random.default_rng(8)
    observations = rng.normal(size=(2048, 432)).astype(np.float32)
    actor = MechanicsActor(seed=8)
    before_hash = mechanics_state_sha256(actor)
    report = calibrate_mechanics_pass_prior(
        actor,
        observations,
        target_override_probability=0.03,
    )
    assert abs(report["mean_override_probability"] - 0.03) < 1e-5
    assert report["deterministic_pass_rate"] >= 0.995
    assert mechanics_state_sha256(actor) != before_hash


def test_full_ppo_checkpoint_reloads_exact_mechanics_logits(tmp_path) -> None:
    config = load_milestone08_config()
    ppo = make_m08_ppo(config, device="cpu")
    ppo.save_to(str(tmp_path))
    state = {
        "format": M08_CHECKPOINT_FORMAT,
        "schema_version": 1,
        "config_sha256": canonical_config_sha256(config),
        "campaign_id": config["campaign_id"],
        "completed_iterations": 0,
        "cumulative_agent_steps": 0,
        "cumulative_model_updates": 0,
        "worker_count": config["environment"]["workers"],
    }
    (tmp_path / M08_STATE_FILE).write_text(
        json.dumps(state) + "\n", encoding="utf-8"
    )

    proof = verify_m08_checkpoint(
        tmp_path,
        ppo,
        config,
        require_optimizer_state=False,
    )

    assert proof["fresh_instance"] is True
    assert proof["exact_logits"] is True
    assert proof["max_abs_logit_error"] == 0.0
    assert proof["state_file_parse_passed"] is True


def test_worker_transition_requires_exact_prospective_evidence(tmp_path) -> None:
    restored = {"worker_count": 64, "cumulative_agent_steps": 499_748}
    evidence_path = tmp_path / "worker_transition.json"
    evidence_path.write_text(
        json.dumps(
            {
                "status": "authorized",
                "source_checkpoint_agent_steps": 499_748,
                "from_worker_count": 64,
                "to_worker_count": 56,
                "failed_launch_collected_agent_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    transition = _validate_worker_transition(
        evidence_path,
        restored=restored,
        requested_workers=56,
    )

    assert transition is not None
    assert transition["from_worker_count"] == 64
    assert transition["to_worker_count"] == 56
    assert all(transition["checks"].values())
