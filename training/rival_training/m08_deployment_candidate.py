"""Export a Milestone 08 mechanics checkpoint for opt-in live evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actions import action_metadata, build_expanded_action_table
from .checkpoint import portable_path
from .config import (
    REPOSITORY_ROOT,
    canonical_config_sha256,
    load_milestone08_config,
)
from .m08_campaign import frozen_strategic_proof, load_m08_state, make_m08_ppo
from .mechanics import export_mechanics_torchscript, mechanics_state_sha256
from .teacher import sha256_file
from .wisp_actions import action_table_fingerprint, build_wisp_action_table


def export_m08_candidate(
    checkpoint_directory: str | Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Create and independently reload a dormant 69-logit mechanics export."""
    if not label or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in label
    ):
        raise ValueError(
            "Candidate label must use lowercase letters, digits, dash, or underscore"
        )

    config = load_milestone08_config()
    config_hash = canonical_config_sha256(config)
    source = Path(checkpoint_directory).resolve()
    state = load_m08_state(source)
    if state["config_sha256"] != config_hash:
        raise RuntimeError("Candidate checkpoint config does not match Milestone 08")

    ppo = make_m08_ppo(config, device="cpu")
    ppo.load_from(str(source))
    actor = ppo.policy.actor.to("cpu").eval()
    actor_state_hash = mechanics_state_sha256(actor)

    artifact_directory = (
        REPOSITORY_ROOT / "training/artifacts/milestone08" / label
    )
    torchscript_path = artifact_directory / "mechanics_actor.ts"
    torchscript = export_mechanics_torchscript(actor, torchscript_path)

    generator = torch.Generator(device="cpu").manual_seed(20260903)
    sample = torch.randn(256, 432, generator=generator)
    loaded = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    with torch.inference_mode():
        checkpoint_logits = ppo.policy.logits(sample)
        actor_logits = actor(sample)
        exported_logits = loaded(sample)
    parity = {
        "sample_count": int(len(sample)),
        "input_shape": list(sample.shape),
        "output_shape": list(exported_logits.shape),
        "checkpoint_policy_to_actor_exact": bool(
            torch.equal(checkpoint_logits, actor_logits)
        ),
        "actor_to_torchscript_exact": bool(
            torch.equal(actor_logits, exported_logits)
        ),
        "actor_to_torchscript_max_abs_error": float(
            (actor_logits - exported_logits).abs().max().item()
        ),
        "all_logits_finite": bool(torch.isfinite(exported_logits).all().item()),
    }
    parity["passed"] = all(
        parity[key]
        for key in (
            "checkpoint_policy_to_actor_exact",
            "actor_to_torchscript_exact",
            "all_logits_finite",
        )
    )

    expanded = build_expanded_action_table()
    wisp = build_wisp_action_table()
    action_table_path = REPOSITORY_ROOT / "bot/models/RIVAL_ACTIONS_V1.npy"
    if not action_table_path.is_file():
        raise FileNotFoundError(
            "M08 live evaluation requires the existing expanded action-table artifact: "
            f"{action_table_path}"
        )
    reloaded_table = np.load(action_table_path, allow_pickle=False)
    metadata = action_metadata()
    action_table = {
        "path": portable_path(action_table_path),
        "size_bytes": action_table_path.stat().st_size,
        "file_sha256": sha256_file(action_table_path),
        "shape": list(reloaded_table.shape),
        "exact_generated_reload": bool(np.array_equal(expanded, reloaded_table)),
        "wisp_prefix_exact": bool(
            np.array_equal(reloaded_table[: len(wisp)], wisp)
        ),
        "wisp_prefix_sha256": action_table_fingerprint(
            reloaded_table[: len(wisp)]
        ),
        "expanded_table_sha256": action_table_fingerprint(reloaded_table),
        "expected_wisp_prefix_sha256": config["frozen_fingerprints"][
            "wisp_prefix_sha256"
        ],
        "expected_expanded_table_sha256": config["frozen_fingerprints"][
            "expanded_table_sha256"
        ],
        "mechanics_mapping": {
            "pass_choice": 0,
            "choice_1_global_action": 90,
            "choice_68_global_action": 157,
        },
    }
    action_table["passed"] = all(
        (
            action_table["exact_generated_reload"],
            action_table["wisp_prefix_exact"],
            action_table["wisp_prefix_sha256"]
            == action_table["expected_wisp_prefix_sha256"],
            action_table["expanded_table_sha256"]
            == action_table["expected_expanded_table_sha256"],
        )
    )

    strategic = frozen_strategic_proof(config)
    gates = {
        "source_checkpoint_config_exact": state["config_sha256"] == config_hash,
        "mechanics_output_count_exact": parity["output_shape"] == [256, 69],
        "randomized_export_reload_exact": parity["passed"],
        "action_table_and_prefix_exact": action_table["passed"],
        "strategic_branch_unchanged": strategic["all_unchanged"],
        "production_promotion_forbidden": not config[
            "production_promotion_authorized"
        ],
    }
    if not all(gates.values()):
        raise RuntimeError(f"M08 candidate export gate failed: {gates}")

    return {
        "schema_version": 1,
        "status": "passed",
        "purpose": "milestone08_opt_in_dual_rate_rlbot_candidate",
        "label": label,
        "config_sha256": config_hash,
        "source_checkpoint": portable_path(source),
        "source_state": state,
        "source_files": {
            path.name: {
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(source.iterdir())
            if path.is_file()
        },
        "mechanics_actor_state_sha256": actor_state_hash,
        "torchscript_export": torchscript,
        "randomized_logit_parity": parity,
        "action_contract": metadata,
        "action_table": action_table,
        "strategic_branch": strategic,
        "cadence_contract": {
            "strategic_ticks": 8,
            "mechanics_ticks": 4,
            "strategic_schedule": [
                "previous",
                "previous",
                "previous",
                "previous",
                "previous",
                "new",
                "new",
                "new",
            ],
            "mechanics_schedule": ["previous", "new", "new", "new"],
        },
        "runtime_mode": "opt_in_candidate_only",
        "rlbot_environment": {
            "RIVAL_M08_DUAL_RATE_ENABLED": "1",
            "RIVAL_M08_MECHANICS_FORCE_PASS": "0",
            "RIVAL_M08_MECHANICS_MODEL_PATH": torchscript["path"],
            "RIVAL_M08_ACTION_TABLE_PATH": action_table["path"],
            "RIVAL_M08_MECHANICS_DETERMINISTIC": "1",
            "RIVAL_TICK_SKIP": "8",
        },
        "gates": gates,
        "production_default": "frozen_wisp_unchanged",
        "production_promoted": False,
    }


def write_m08_candidate_export_report(
    checkpoint_directory: str | Path,
    *,
    label: str,
    output: str | Path,
) -> dict[str, Any]:
    report = export_m08_candidate(checkpoint_directory, label=label)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
