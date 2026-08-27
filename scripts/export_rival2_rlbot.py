"""Export a RivalSim Rival 2 checkpoint as a compact deterministic RLBot policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _git_head(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def _load_rivalsim(root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(root))
    from rivalsim.kernels.boost_pad import SOCCAR_PAD_POSITIONS
    from rivalsim.rival2_contracts import (
        ACTION_CONTRACT_HASH,
        ACTION_NAMES,
        AIR_TIME_SCALE,
        ANGULAR_SPEED_SCALE,
        BALL_LINEAR_SPEED_SCALE,
        BOOSTING_TIME_SCALE,
        BOOST_SCALE,
        CAR_LINEAR_SPEED_SCALE,
        DEMO_TIMER_SCALE,
        EPISODE_AGE_SCALE_TICKS,
        FLIP_TIME_SCALE,
        JUMP_TIME_SCALE,
        NO_TOUCH_AGE_SCALE_TICKS,
        OBSERVATION_SCHEMA_HASH,
        OBS_DIM,
        ORANGE_PAD_REMAP,
        POSITION_SCALE,
        STICKY_TICK_SCALE,
        SUPERSONIC_TIME_SCALE,
        TIME_SINCE_BOOSTED_SCALE,
    )
    from rivalsim.rival2_policy import Rival2ActorCritic, Rival2PolicyConfig

    return {
        "SOCCAR_PAD_POSITIONS": SOCCAR_PAD_POSITIONS,
        "ACTION_CONTRACT_HASH": ACTION_CONTRACT_HASH,
        "ACTION_NAMES": ACTION_NAMES,
        "AIR_TIME_SCALE": AIR_TIME_SCALE,
        "ANGULAR_SPEED_SCALE": ANGULAR_SPEED_SCALE,
        "BALL_LINEAR_SPEED_SCALE": BALL_LINEAR_SPEED_SCALE,
        "BOOSTING_TIME_SCALE": BOOSTING_TIME_SCALE,
        "BOOST_SCALE": BOOST_SCALE,
        "CAR_LINEAR_SPEED_SCALE": CAR_LINEAR_SPEED_SCALE,
        "DEMO_TIMER_SCALE": DEMO_TIMER_SCALE,
        "EPISODE_AGE_SCALE_TICKS": EPISODE_AGE_SCALE_TICKS,
        "FLIP_TIME_SCALE": FLIP_TIME_SCALE,
        "JUMP_TIME_SCALE": JUMP_TIME_SCALE,
        "NO_TOUCH_AGE_SCALE_TICKS": NO_TOUCH_AGE_SCALE_TICKS,
        "OBSERVATION_SCHEMA_HASH": OBSERVATION_SCHEMA_HASH,
        "OBS_DIM": OBS_DIM,
        "ORANGE_PAD_REMAP": ORANGE_PAD_REMAP,
        "POSITION_SCALE": POSITION_SCALE,
        "STICKY_TICK_SCALE": STICKY_TICK_SCALE,
        "SUPERSONIC_TIME_SCALE": SUPERSONIC_TIME_SCALE,
        "TIME_SINCE_BOOSTED_SCALE": TIME_SINCE_BOOSTED_SCALE,
        "Rival2ActorCritic": Rival2ActorCritic,
        "Rival2PolicyConfig": Rival2PolicyConfig,
    }


class DeterministicRival2Policy(nn.Module):
    """Deployment-only deterministic action defined by RIVAL2_ACTION_V1."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        actor, _value = self.model(observation)
        analog = torch.tanh(actor[:, :5])
        buttons = (torch.sigmoid(actor[:, 10:13]) >= 0.5).to(actor.dtype)
        return torch.cat((analog, buttons), dim=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rivalsim-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()

    rivalsim_root = args.rivalsim_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    manifest_path = args.manifest.resolve()
    symbols = _load_rivalsim(rivalsim_root)

    checkpoint_sha = _sha256(checkpoint)
    if (
        args.expected_checkpoint_sha256
        and checkpoint_sha != args.expected_checkpoint_sha256.upper()
    ):
        raise RuntimeError(
            f"checkpoint SHA-256 mismatch: {checkpoint_sha} != "
            f"{args.expected_checkpoint_sha256.upper()}"
        )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = symbols["Rival2PolicyConfig"](**payload["policy_config"])
    if payload.get("policy_config_hash") != config.content_hash:
        raise RuntimeError("checkpoint policy configuration hash mismatch")
    model = symbols["Rival2ActorCritic"](config)
    model.load_state_dict(payload["model"])
    model.eval()
    wrapper = DeterministicRival2Policy(model).eval()
    example = torch.zeros((1, symbols["OBS_DIM"]), dtype=torch.float32)
    traced = torch.jit.trace(wrapper, example, check_trace=True)
    frozen = torch.jit.freeze(traced)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(frozen, output)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026082701)
    parity_observations = torch.randn(
        (4096, symbols["OBS_DIM"]), generator=generator, dtype=torch.float32
    )
    with torch.inference_mode():
        reference = wrapper(parity_observations)
        exported = torch.jit.load(str(output), map_location="cpu").eval()(
            parity_observations
        )
    analog_error = float(torch.max(torch.abs(reference[:, :5] - exported[:, :5])).item())
    button_exact = bool(torch.equal(reference[:, 5:], exported[:, 5:]))
    if analog_error != 0.0 or not button_exact:
        raise RuntimeError(
            f"export parity failed: analog max error={analog_error}, "
            f"buttons exact={button_exact}"
        )

    observation = {
        "dimension": symbols["OBS_DIM"],
        "position_scale": list(symbols["POSITION_SCALE"]),
        "car_linear_speed_scale": symbols["CAR_LINEAR_SPEED_SCALE"],
        "ball_linear_speed_scale": symbols["BALL_LINEAR_SPEED_SCALE"],
        "angular_speed_scale": symbols["ANGULAR_SPEED_SCALE"],
        "boost_scale": symbols["BOOST_SCALE"],
        "demo_timer_scale": symbols["DEMO_TIMER_SCALE"],
        "jump_time_scale": symbols["JUMP_TIME_SCALE"],
        "air_time_scale": symbols["AIR_TIME_SCALE"],
        "flip_time_scale": symbols["FLIP_TIME_SCALE"],
        "boosting_time_scale": symbols["BOOSTING_TIME_SCALE"],
        "time_since_boosted_scale": symbols["TIME_SINCE_BOOSTED_SCALE"],
        "supersonic_time_scale": symbols["SUPERSONIC_TIME_SCALE"],
        "sticky_tick_scale": symbols["STICKY_TICK_SCALE"],
        "episode_age_scale_ticks": symbols["EPISODE_AGE_SCALE_TICKS"],
        "no_touch_age_scale_ticks": symbols["NO_TOUCH_AGE_SCALE_TICKS"],
        "orange_pad_remap": list(symbols["ORANGE_PAD_REMAP"]),
        "canonical_boost_pad_positions": np.asarray(
            symbols["SOCCAR_PAD_POSITIONS"], dtype=np.float32
        ).tolist(),
        "canonical_boost_pad_durations": [10.0] * 6 + [4.0] * 28,
    }
    manifest = {
        "schema_version": 1,
        "format": "RIVAL2_RLBOT_DEPLOY_V1",
        "source": {
            "rivalsim_commit": _git_head(rivalsim_root),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "policy_version": int(payload["policy_version"]),
            "iteration": int(payload["iteration"]),
            "total_agent_samples": int(payload["total_agent_samples"]),
            "reward_version": payload["reward_version"],
            "policy_config_hash": config.content_hash,
        },
        "contracts": {
            "observation": "RIVAL2_OBS_V1",
            "observation_schema_sha256": symbols["OBSERVATION_SCHEMA_HASH"],
            "action": "RIVAL2_ACTION_V1",
            "action_contract_sha256": symbols["ACTION_CONTRACT_HASH"],
            "controller_fields": list(symbols["ACTION_NAMES"]),
            "physics_hz": 120,
            "policy_hz": 30,
            "hold_ticks": 4,
            "deployment_action": "deterministic tanh(mean), sigmoid(logit)>=0.5",
        },
        "observation": observation,
        "artifact": {
            "path": output.name,
            "size_bytes": output.stat().st_size,
            "sha256": _sha256(output),
        },
        "export_parity": {
            "seed": 2026082701,
            "observations": 4096,
            "analog_max_abs_error": analog_error,
            "buttons_exact": button_exact,
            "pass": analog_error == 0.0 and button_exact,
        },
        "live_packet_qualifications": {
            "boost_pad_mapping": (
                "canonical pads are matched to live FieldInfo by unique horizontal "
                "center within the measured 2-unit Soccar coordinate delta because "
                "RLBot reports rendered pad coordinates rather than RivalSim's pickup "
                "trigger coordinates"
            ),
            "individual_wheel_contacts": (
                "RLBot v5 exposes authoritative aggregate AirState.OnGround but not "
                "four individual wheel-contact bits; live deployment broadcasts the "
                "aggregate state to the four frozen observation fields"
            ),
            "internal_vehicle_timers": (
                "jump/air/boost/supersonic timers omitted by live packets are maintained "
                "from authoritative packet transitions and frame_num at 120 Hz"
            ),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
