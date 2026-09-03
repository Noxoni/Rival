"""Export the single RivalSim official checkpoint as an RLBot bundle."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
V23_TEMPLATE = ROOT / "bot/rival2_v23/models/rival2_v23_blue.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


class DeterministicPolicy(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        actor, _value = self.model(observation)
        analog = torch.tanh(actor[:, :5])
        buttons = (torch.sigmoid(actor[:, 10:13]) >= 0.5).to(actor.dtype)
        return torch.cat((analog, buttons), dim=1)


def export_component(
    name: str,
    component: dict[str, Any],
    output_dir: Path,
    template: dict[str, Any],
    official_sha256: str,
    official_commit: str,
    actor_critic: type[nn.Module],
    policy_config: type[Any],
) -> dict[str, Any]:
    config = policy_config(**component["policy_config"])
    if config.content_hash != component["policy_config_hash"]:
        raise RuntimeError(f"{name} policy configuration hash mismatch")
    model = actor_critic(config)
    model.load_state_dict(component["model"], strict=True)
    wrapper = DeterministicPolicy(model.eval()).eval()
    example = torch.zeros((1, 182), dtype=torch.float32)
    artifact = output_dir / "models" / f"{name}.ts"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    traced = torch.jit.trace(wrapper, example, check_trace=True)
    torch.jit.save(torch.jit.freeze(traced), artifact)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(2_026_090_311)
    observations = torch.randn((4096, 182), generator=generator)
    with torch.inference_mode():
        expected = wrapper(observations)
        observed = torch.jit.load(str(artifact), map_location="cpu").eval()(observations)
    analog_error = float((expected[:, :5] - observed[:, :5]).abs().max())
    buttons_exact = bool(torch.equal(expected[:, 5:], observed[:, 5:]))
    if analog_error != 0.0 or not buttons_exact:
        raise RuntimeError(f"{name} TorchScript export parity failed")

    manifest = deepcopy(template)
    manifest["source"] = {
        "rivalsim_commit": official_commit,
        "checkpoint_path": "checkpoints/rival2/official_v1/rival2_official_v1.pt",
        "checkpoint_sha256": official_sha256,
        "component": name,
        "component_source_path": component["source_path"],
        "component_source_sha256": component["source_sha256"],
        "policy_config_hash": component["policy_config_hash"],
    }
    manifest["artifact"] = {
        "path": artifact.name,
        "size_bytes": artifact.stat().st_size,
        "sha256": sha256_file(artifact),
    }
    manifest["export_parity"] = {
        "seed": 2_026_090_311,
        "observations": 4096,
        "analog_max_abs_error": analog_error,
        "buttons_exact": buttons_exact,
        "pass": True,
    }
    manifest_path = artifact.with_suffix(".json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "model": artifact.relative_to(output_dir).as_posix(),
        "manifest": manifest_path.relative_to(output_dir).as_posix(),
        "artifact": manifest["artifact"],
        "source": manifest["source"],
        "export_parity": manifest["export_parity"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rivalsim-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "bot/rival2_official"
    )
    args = parser.parse_args()
    rivalsim_root = args.rivalsim_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    expected_sha256 = args.expected_sha256.upper()
    observed_sha256 = sha256_file(checkpoint)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            f"official checkpoint identity mismatch: {observed_sha256} != {expected_sha256}"
        )
    sys.path.insert(0, str(rivalsim_root))
    from rivalsim.rival2_official_bundle_v1 import OFFICIAL_BUNDLE_V1_FORMAT
    from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != OFFICIAL_BUNDLE_V1_FORMAT:
        raise RuntimeError("unsupported Rival official checkpoint")
    if not all(
        payload["router_config"][name] is False
        for name in (
            "automatic_aerial_enabled",
            "automatic_recovery_enabled",
            "automatic_offensive_demo_enabled",
        )
    ):
        raise RuntimeError("playable official export requires fail-closed routing")
    template = json.loads(V23_TEMPLATE.read_text(encoding="utf-8"))
    if template["contracts"]["hold_ticks"] != 1:
        raise RuntimeError("official deployment requires native 120 Hz control")
    output_dir.mkdir(parents=True, exist_ok=True)
    components = {
        name: export_component(
            name,
            component,
            output_dir,
            template,
            observed_sha256,
            payload["rivalsim_commit"],
            Rival2ActorCritic,
            Rival2PolicyConfig,
        )
        for name, component in sorted(payload["components"].items())
    }
    bundle = {
        "format": "RIVAL2_RLBOT_OFFICIAL_BUNDLE_V1",
        "source": {
            "checkpoint_path": "checkpoints/rival2/official_v1/rival2_official_v1.pt",
            "checkpoint_sha256": observed_sha256,
            "rivalsim_commit": payload["rivalsim_commit"],
        },
        "contracts": payload["contract_hashes"],
        "physics_hz": 120,
        "policy_hz": 120,
        "components": components,
        "active_default": {"blue": "base_blue", "orange": "base_orange"},
        "router_config": payload["router_config"],
        "specialist_status": {
            "embedded": ["aerial", "capability_blue", "capability_orange"],
            "automatic_takeover": False,
            "reason": (
                "whole-match candidates with specialist takeover failed the "
                "frozen Nexto physical gate; the playable path is fail-closed"
            ),
        },
    }
    bundle_path = output_dir / "deployment_bundle.json"
    bundle_path.write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(bundle, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
