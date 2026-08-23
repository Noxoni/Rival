from __future__ import annotations

import json
from pathlib import Path

import torch

from rival_training.m10_campaign import (
    DEFAULT_M10_CONFIG_PATH,
    M09_FINAL_STEPS,
    boundary_slug,
    load_m10_config,
    m10_config_migration_report,
    nominal_boundary_steps,
    prune_rolling_checkpoints,
    save_checkpoint_atomic,
    verify_checkpoint_reload_parity,
)
from rival_training.v9_checkpoint import DEFAULT_PILOT_CONFIG_PATH, load_m09_config
from rival_training.v9_deployment import export_v9_deployment
from rival_training.v9_observations import observation_schema_manifest
from rival_training.v9_policy import RivalCriticV1, RivalPolicyV1


def _initialized_checkpoint_state(config: dict) -> tuple:
    actor = RivalPolicyV1()
    critic = RivalCriticV1()
    actor_optimizer = torch.optim.Adam(
        actor.parameters(), lr=float(config["ppo"]["actor_learning_rate"])
    )
    critic_optimizer = torch.optim.Adam(
        critic.parameters(), lr=float(config["ppo"]["critic_learning_rate"])
    )
    observations = torch.zeros((32, observation_schema_manifest()["float_count"]))
    mean, log_std, logits = actor(observations)
    (mean.mean() + log_std.mean() + logits.mean()).backward()
    actor_optimizer.step()
    critic(observations).mean().backward()
    critic_optimizer.step()
    return actor, critic, actor_optimizer, critic_optimizer, observations.numpy()


def test_m10_config_changes_only_campaign_metadata() -> None:
    source = load_m09_config(DEFAULT_PILOT_CONFIG_PATH)
    target = load_m10_config(DEFAULT_M10_CONFIG_PATH)
    report = m10_config_migration_report(source, target)
    assert report["passed"]
    assert report["changed_top_level_keys"] == [
        "campaign",
        "campaign_id",
        "config_version",
        "pilot",
    ]
    assert report["checks"]["all_learning_semantics_exact"]
    assert source["ppo"] == target["ppo"]
    assert source["backend"] == target["backend"]
    assert source["curriculum"] == target["curriculum"]


def test_m10_boundary_targets_are_additional_to_exact_m09_checkpoint() -> None:
    assert boundary_slug(5) == "plus-005h"
    assert boundary_slug(100) == "plus-100h"
    assert nominal_boundary_steps(5) == M09_FINAL_STEPS + 5 * 864000
    assert nominal_boundary_steps(100) == 88_080_214


def test_atomic_checkpoint_reload_and_two_state_rolling_retention(tmp_path: Path) -> None:
    config = load_m10_config()
    actor, critic, actor_optimizer, critic_optimizer, observations = (
        _initialized_checkpoint_state(config)
    )
    rolling = tmp_path / "rolling"
    state = {
        "completed_iterations": 10,
        "cumulative_agent_steps": 1_872_214,
        "cumulative_model_updates": 39,
        "simulated_game_seconds": 1_872_214 / 240.0,
        "simulated_game_hours": 1_872_214 / 864000.0,
        "worker_count": 56,
        "clean_boundary": True,
        "partial_experience_buffer_records": 0,
        "production_promotion_authorized": False,
    }
    for offset in range(3):
        destination = rolling / str(state["cumulative_agent_steps"] + offset)
        record = save_checkpoint_atomic(
            destination,
            actor=actor,
            critic=critic,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            trainer_state=state,
            config=config,
            reload_observations=observations,
        )
        assert record["clean_boundary"]
    removed = prune_rolling_checkpoints(rolling, keep=2)
    assert len(removed) == 1
    survivors = sorted(path.name for path in rolling.iterdir())
    assert survivors == ["1872215", "1872216"]
    parity = verify_checkpoint_reload_parity(
        rolling / "1872216", expected_config=config, device="cpu"
    )
    assert parity["checks"]["passed"]
    assert max(parity["maximum_absolute_error"].values()) == 0.0
    manifest = json.loads(
        (rolling / "1872216" / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["directory"].endswith("1872216")
    exported = export_v9_deployment(rolling / "1872216", tmp_path / "export")
    assert exported["metadata"]["export_parity"]["passed"]
    assert (
        exported["metadata"]["contract"]["training_config_version"]
        == "RivalM10TrainingConfigV1"
    )
