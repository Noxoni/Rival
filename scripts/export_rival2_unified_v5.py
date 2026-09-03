"""Export the validated RivalSim unified V5 policy for recurrent RLBot play."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = ROOT / "bot/rival2_v23/models/rival2_v23_blue.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "955C93BF538BC913CC2E42F42E3B0EDC4CCDB1065DA9581FB88D84C363B7C216"
)
EXPECTED_CHECKPOINT_FORMAT = "RIVAL2_UNIFIED_CAPABILITY_CHECKPOINT_V5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def git_commit_for_path(repository: Path, path: Path) -> str:
    relative = path.resolve().relative_to(repository.resolve()).as_posix()
    return subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=repository,
        text=True,
    ).strip()


class DeterministicUnifiedPolicy(nn.Module):
    """Deployment action plus recurrent state for one 120 Hz policy step."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, observation: torch.Tensor, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actor, _value, next_hidden = self.model(observation, hidden)
        analog = torch.tanh(actor[:, :5])
        buttons = (torch.sigmoid(actor[:, 10:13]) >= 0.5).to(actor.dtype)
        return torch.cat((analog, buttons), dim=1), next_hidden


def load_checkpoint(rivalsim_root: Path, checkpoint: Path) -> tuple[Any, dict[str, Any]]:
    sys.path.insert(0, str(rivalsim_root))
    from rivalsim.rival2_unified_policy import (
        Rival2UnifiedActorCritic,
        Rival2UnifiedPolicyConfig,
    )

    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "unified V5 checkpoint identity mismatch: "
            f"{checkpoint_sha256} != {EXPECTED_CHECKPOINT_SHA256}"
        )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != EXPECTED_CHECKPOINT_FORMAT:
        raise RuntimeError(f"unsupported checkpoint format: {payload.get('format')}")
    if payload.get("runtime_router") is not False:
        raise RuntimeError("unified V5 checkpoint unexpectedly enables a runtime router")
    if payload.get("task_identifier_input") is not False:
        raise RuntimeError("unified V5 checkpoint unexpectedly uses a task identifier")
    config = Rival2UnifiedPolicyConfig(**payload["policy_config"])
    if payload.get("policy_config_sha256") != config.content_hash:
        raise RuntimeError("unified V5 policy configuration hash mismatch")
    contracts = payload["contracts"]
    hashes = payload["contract_hashes"]
    for name, key in (
        (contracts["observation"], "observation_sha256"),
        (contracts["action"], "action_sha256"),
    ):
        if hashes.get(name) != contracts[key]:
            raise RuntimeError(f"unified V5 contract hash mismatch for {name}")
    model = Rival2UnifiedActorCritic(config).eval()
    model.load_state_dict(payload["model"], strict=True)
    return model, payload


def export(
    rivalsim_root: Path,
    checkpoint: Path,
    output: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    model, payload = load_checkpoint(rivalsim_root, checkpoint)
    wrapper = DeterministicUnifiedPolicy(model).eval()
    config = model.config
    observation = torch.zeros((1, config.obs_dim), dtype=torch.float32)
    hidden = torch.zeros(
        (config.context_layers, 1, config.context_hidden_dim), dtype=torch.float32
    )
    traced = torch.jit.trace(wrapper, (observation, hidden), check_trace=True)
    frozen = torch.jit.freeze(traced)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(frozen, output)
    deployed = torch.jit.load(str(output), map_location="cpu").eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(202609036101)
    parity_observations = torch.randn(
        (512, config.obs_dim), generator=generator, dtype=torch.float32
    )
    parity_hidden = torch.randn(
        (config.context_layers, 512, config.context_hidden_dim),
        generator=generator,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        expected_action, expected_hidden = wrapper(parity_observations, parity_hidden)
        actual_action, actual_hidden = deployed(parity_observations, parity_hidden)
    analog_error = float(
        torch.max(torch.abs(expected_action[:, :5] - actual_action[:, :5])).item()
    )
    buttons_exact = bool(torch.equal(expected_action[:, 5:], actual_action[:, 5:]))
    hidden_error = float(torch.max(torch.abs(expected_hidden - actual_hidden)).item())
    if analog_error != 0.0 or hidden_error != 0.0 or not buttons_exact:
        raise RuntimeError(
            "unified recurrent export parity failed: "
            f"analog={analog_error}, hidden={hidden_error}, buttons={buttons_exact}"
        )

    sequence_hidden_expected = parity_hidden[:, :8].clone()
    sequence_hidden_actual = sequence_hidden_expected.clone()
    sequence_action_error = 0.0
    sequence_hidden_error = 0.0
    with torch.inference_mode():
        for _ in range(256):
            step_observation = torch.randn(
                (8, config.obs_dim), generator=generator, dtype=torch.float32
            )
            expected_step, sequence_hidden_expected = wrapper(
                step_observation, sequence_hidden_expected
            )
            actual_step, sequence_hidden_actual = deployed(
                step_observation, sequence_hidden_actual
            )
            sequence_action_error = max(
                sequence_action_error,
                float(torch.max(torch.abs(expected_step - actual_step)).item()),
            )
            sequence_hidden_error = max(
                sequence_hidden_error,
                float(
                    torch.max(
                        torch.abs(sequence_hidden_expected - sequence_hidden_actual)
                    ).item()
                ),
            )
    if sequence_action_error != 0.0 or sequence_hidden_error != 0.0:
        raise RuntimeError(
            "unified recurrent sequence parity failed: "
            f"action={sequence_action_error}, hidden={sequence_hidden_error}"
        )

    reference = json.loads(REFERENCE_MANIFEST.read_text(encoding="utf-8"))
    contracts = payload["contracts"]
    if (
        reference["contracts"]["observation_schema_sha256"]
        != contracts["observation_sha256"]
    ):
        raise RuntimeError("live observation adapter contract does not match unified V5")
    if reference["contracts"]["action_contract_sha256"] != contracts["action_sha256"]:
        raise RuntimeError("live action adapter contract does not match unified V5")
    manifest = {
        "schema_version": 1,
        "format": "RIVAL2_RLBOT_RECURRENT_DEPLOY_V1",
        "source": {
            "rivalsim_commit": git_head(rivalsim_root),
            "checkpoint_commit": git_commit_for_path(rivalsim_root, checkpoint),
            "checkpoint_path": checkpoint.relative_to(rivalsim_root).as_posix(),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_format": payload["format"],
            "model_tensor_sha256": payload["model_tensor_sha256"],
            "selected_supervised_step": payload["accepted_supervised_steps"],
            "cumulative_supervised_steps": payload["cumulative_supervised_steps"],
            "reward_version": payload["reward_version"],
            "policy_config_sha256": payload["policy_config_sha256"],
            "runtime_router": payload["runtime_router"],
            "task_identifier_input": payload["task_identifier_input"],
        },
        "contracts": {
            "observation": contracts["observation"],
            "observation_schema_sha256": contracts["observation_sha256"],
            "action": contracts["action"],
            "action_contract_sha256": contracts["action_sha256"],
            "controller_fields": reference["contracts"]["controller_fields"],
            "physics_hz": contracts["physics_hz"],
            "policy_hz": contracts["policy_hz"],
            "hold_ticks": 1,
            "deployment_action": "deterministic tanh(mean), sigmoid(logit)>=0.5",
        },
        "recurrent": {
            "input_hidden_shape": [
                config.context_layers,
                "batch",
                config.context_hidden_dim,
            ],
            "initial_hidden": "zeros at match, kickoff, score, or frame reset",
            "advance": "exactly once per unique RLBot physics packet",
            "duplicate_packet": "hold action and hidden state",
            "runtime_router": False,
        },
        "observation": reference["observation"],
        "artifact": {
            "path": output.name,
            "size_bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
        "export_parity": {
            "seed": 202609036101,
            "single_step_observations": 512,
            "sequence_batch": 8,
            "sequence_ticks": 256,
            "analog_max_abs_error": analog_error,
            "buttons_exact": buttons_exact,
            "hidden_max_abs_error": hidden_error,
            "sequence_action_max_abs_error": sequence_action_error,
            "sequence_hidden_max_abs_error": sequence_hidden_error,
            "pass": True,
        },
        "live_packet_qualifications": reference["live_packet_qualifications"],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rivalsim-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = export(
        args.rivalsim_root.resolve(),
        args.checkpoint.resolve(),
        args.output.resolve(),
        args.manifest.resolve(),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
