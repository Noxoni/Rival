"""Materialize a campaign checkpoint for opt-in RLBot candidate evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actions import action_metadata, build_expanded_action_table
from .campaign import load_campaign_state, make_campaign_ppo
from .checkpoint import portable_path, save_actor_checkpoint
from .config import REPOSITORY_ROOT, canonical_config_sha256, load_milestone06_config
from .deploy import export_torchscript, make_exact_policy_export
from .policy import materialize_effective_actor
from .teacher import sha256_file
from .wisp_actions import action_table_fingerprint


def export_campaign_candidate(
    checkpoint_directory: str | Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Bake the checkpointed prior and create a dormant, opt-in candidate export."""
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in label):
        raise ValueError("Candidate label must use lowercase letters, digits, dash, or underscore")
    config = load_milestone06_config()
    config_hash = canonical_config_sha256(config)
    source = Path(checkpoint_directory).resolve()
    state = load_campaign_state(source)
    if state["config_sha256"] != config_hash:
        raise RuntimeError("Candidate checkpoint config does not match Milestone 06")
    offset = float(state["action_exploration_prior"]["appended_logit_offset"])
    ppo = make_campaign_ppo(config, device="cpu", appended_logit_offset=offset)
    ppo.load_from(str(source))
    effective_actor = materialize_effective_actor(ppo.policy)
    exact_export = make_exact_policy_export(ppo.policy)

    artifact_directory = REPOSITORY_ROOT / "training/artifacts/milestone06" / label
    actor_path = artifact_directory / "candidate_actor.pt"
    torchscript_path = artifact_directory / "candidate_actor.ts"
    actor_manifest = save_actor_checkpoint(
        actor_path,
        effective_actor,
        {
            "source_campaign_checkpoint": portable_path(source),
            "source_campaign_state_sha256": sha256_file(
                source / "RIVAL_CAMPAIGN_STATE.json"
            ),
            "cumulative_agent_steps": int(state["cumulative_agent_steps"]),
            "cumulative_model_updates": int(state["cumulative_model_updates"]),
            "appended_prior_baked_into_actor": offset,
            "production_promoted": False,
        },
    )
    torchscript_manifest = export_torchscript(exact_export, torchscript_path)

    action_table = build_expanded_action_table()
    action_table_path = REPOSITORY_ROOT / "bot/models/RIVAL_ACTIONS_V1.npy"
    action_table_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(action_table_path, action_table, allow_pickle=False)
    reloaded_table = np.load(action_table_path, allow_pickle=False)
    metadata = action_metadata()
    table_parity = {
        "path": portable_path(action_table_path),
        "file_sha256": sha256_file(action_table_path),
        "logical_float32_sha256": action_table_fingerprint(reloaded_table),
        "expected_logical_float32_sha256": metadata["expanded_table_sha256"],
        "shape": list(reloaded_table.shape),
        "exact_reload": bool(np.array_equal(action_table, reloaded_table)),
    }
    table_parity["passed"] = bool(
        table_parity["exact_reload"]
        and table_parity["logical_float32_sha256"]
        == table_parity["expected_logical_float32_sha256"]
    )

    sample = torch.randn(
        256,
        432,
        generator=torch.Generator(device="cpu").manual_seed(20260831),
    )
    scripted = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    with torch.inference_mode():
        policy_logits = ppo.policy.logits(sample)
        effective_logits = effective_actor(sample)
        exact_export_logits = exact_export(sample)
        scripted_logits = scripted(sample)
    parity = {
        "checkpoint_policy_to_baked_actor_exact": bool(
            torch.equal(policy_logits, effective_logits)
        ),
        "checkpoint_policy_to_baked_actor_allclose_1e-6": bool(
            torch.allclose(policy_logits, effective_logits, atol=1e-6, rtol=1e-6)
        ),
        "checkpoint_policy_to_exact_export_exact": bool(
            torch.equal(policy_logits, exact_export_logits)
        ),
        "exact_export_to_torchscript_exact": bool(
            torch.equal(exact_export_logits, scripted_logits)
        ),
        "checkpoint_policy_to_baked_actor_max_abs_error": float(
            (policy_logits - effective_logits).abs().max().item()
        ),
        "checkpoint_policy_to_exact_export_max_abs_error": float(
            (policy_logits - exact_export_logits).abs().max().item()
        ),
        "exact_export_to_torchscript_max_abs_error": float(
            (exact_export_logits - scripted_logits).abs().max().item()
        ),
        "finite": bool(torch.isfinite(scripted_logits).all().item()),
    }
    parity["passed"] = all(
        parity[key]
        for key in (
            "checkpoint_policy_to_baked_actor_allclose_1e-6",
            "checkpoint_policy_to_exact_export_exact",
            "exact_export_to_torchscript_exact",
            "finite",
        )
    )
    if not table_parity["passed"] or not parity["passed"]:
        raise RuntimeError(
            f"Candidate deployment parity failed: table={table_parity}, logits={parity}"
        )

    return {
        "schema_version": 1,
        "status": "passed",
        "label": label,
        "config_sha256": config_hash,
        "source_checkpoint": portable_path(source),
        "source_campaign_state": state,
        "actor_checkpoint": actor_manifest,
        "torchscript_export": torchscript_manifest,
        "action_table": table_parity,
        "logit_parity": parity,
        "runtime_mode": "opt_in_candidate_only",
        "production_default": "frozen_wisp_unchanged",
        "rlbot_environment": {
            "RIVAL_CANDIDATE_MODEL_PATH": torchscript_manifest["path"],
            "RIVAL_CANDIDATE_ACTION_TABLE_PATH": table_parity["path"],
            "RIVAL_TICK_SKIP": "4",
        },
        "production_promoted": False,
    }


def write_candidate_export_report(
    checkpoint_directory: str | Path,
    *,
    label: str,
    output: str | Path,
) -> dict[str, Any]:
    report = export_campaign_candidate(checkpoint_directory, label=label)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
