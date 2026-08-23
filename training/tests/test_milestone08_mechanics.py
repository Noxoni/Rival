from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from rival_training.config import (
    REPOSITORY_ROOT,
    canonical_config_sha256,
    load_milestone08_config,
)
from rival_training.m08_campaign import (
    M08_CHECKPOINT_FORMAT,
    M08_STATE_FILE,
    _validate_mechanics_usage_adjustment,
    _validate_worker_transition,
    make_m08_ppo,
    verify_m08_checkpoint,
)
from rival_training.m08_deployment_candidate import export_m08_candidate
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


def test_mechanics_usage_adjustment_is_hash_bound_and_applied_once(tmp_path) -> None:
    config = load_milestone08_config()
    ppo = make_m08_ppo(config, device="cpu")
    ppo.save_to(str(tmp_path))
    restored = {
        "format": M08_CHECKPOINT_FORMAT,
        "schema_version": 1,
        "config_sha256": canonical_config_sha256(config),
        "campaign_id": config["campaign_id"],
        "completed_iterations": 40,
        "cumulative_agent_steps": 1_999_776,
        "cumulative_model_updates": 111,
        "worker_count": 56,
        "mechanics_usage_adjustment_history": [],
    }
    state_path = tmp_path / M08_STATE_FILE
    state_path.write_text(json.dumps(restored) + "\n", encoding="utf-8")
    files = {
        path.name: {
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(tmp_path.iterdir())
        if path.is_file()
    }
    source_bias = float(ppo.policy.actor.output_layer.bias[0].item())
    delta = -1.25
    evidence_path = tmp_path.parent / "usage_adjustment.json"
    evidence_path.write_text(
        json.dumps(
            {
                "status": "authorized",
                "config_sha256": canonical_config_sha256(config),
                "source_checkpoint": {
                    "directory": tmp_path.resolve()
                    .relative_to(REPOSITORY_ROOT.resolve())
                    .as_posix(),
                    "agent_steps": 1_999_776,
                    "files": files,
                },
                "target_mean_override_probability": 0.10,
                "pass_bias_adjustment": {
                    "source_bias": source_bias,
                    "source_mean_override_probability": 0.03,
                    "delta": delta,
                    "adjusted_bias": source_bias + delta,
                },
                "before": {"mean_override_probability": 0.03},
                "after": {"mean_override_probability": 0.10},
                "sampled_audit": {"override_rate": 0.101},
            }
        ),
        encoding="utf-8",
    )

    applied = _validate_mechanics_usage_adjustment(
        evidence_path,
        restored=restored,
        source_directory=tmp_path,
        policy=ppo.policy,
        config=config,
    )

    assert applied is not None
    assert applied["pass_bias_delta"] == delta
    assert applied["optimizer_state_preserved"] is True
    assert abs(float(ppo.policy.actor.output_layer.bias[0].item()) - (source_bias + delta)) < 1e-6

    restored["mechanics_usage_adjustment_history"] = [applied]
    try:
        _validate_mechanics_usage_adjustment(
            evidence_path,
            restored=restored,
            source_directory=tmp_path,
            policy=ppo.policy,
            config=config,
        )
    except ValueError as exc:
        assert "source_not_previously_adjusted" in str(exc)
    else:
        raise AssertionError("A mechanics usage adjustment was applied twice")


def test_m08_candidate_export_requires_safe_label() -> None:
    try:
        export_m08_candidate("missing", label="unsafe label")
    except ValueError as exc:
        assert "label" in str(exc).lower()
    else:
        raise AssertionError("Unsafe candidate label was accepted")
