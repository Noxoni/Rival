"""Targeted fidelity gate for the Rival 2 RLBot live observation/policy port."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = ROOT / "bot" / "rival2_live"
sys.path.insert(0, str(LIVE_ROOT))

from runtime import LiveMemory, Rival2LiveAdapter  # noqa: E402


def _vec(value: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(x=float(value[0]), y=float(value[1]), z=float(value[2]))


def _controller(row: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(
        throttle=float(row[0]),
        steer=float(row[1]),
        pitch=float(row[2]),
        yaw=float(row[3]),
        roll=float(row[4]),
        jump=bool(row[5]),
        boost=bool(row[6]),
        handbrake=bool(row[7]),
    )


def _accepted_memory(source: LiveMemory, memory_type: type) -> object:
    target = memory_type.create(1)
    for name in (
        "episode_ticks",
        "no_touch_ticks",
        "kickoff_indicator",
        "touch_event",
        "demoed_event",
        "previous_action",
        "time_since_boosted",
        "sticky_ticks",
        "previous_demoed",
    ):
        getattr(target, name)[:] = getattr(source, name)
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rivalsim-root", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=LIVE_ROOT / "models" / "rival2_gameplay_v2_479.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=LIVE_ROOT / "models" / "rival2_gameplay_v2_479.ts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "rival2_live_adapter.json",
    )
    parser.add_argument("--states", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026082702)
    args = parser.parse_args()

    rivalsim_root = args.rivalsim_root.resolve()
    sys.path.insert(0, str(rivalsim_root))
    from rivalsim.rocketsim_adapter import (
        FrozenRivalPolicy,
        RocketSimBatchState,
        RocketSimRivalMemory,
        build_rival2_observation,
    )

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rng = np.random.default_rng(args.seed)
    canonical_positions = np.asarray(
        manifest["observation"]["canonical_boost_pad_positions"], dtype=np.float32
    )
    packet_order = rng.permutation(34)

    def live_pad_position(index: int) -> np.ndarray:
        x, y, _z = canonical_positions[index]
        if abs(float(y)) == 2300.0:
            y += 2.0 if y > 0 else -2.0
        return np.asarray(
            (x, y, 8.0 if index < 6 else 0.0820159912109375),
            dtype=np.float32,
        )

    field_info = SimpleNamespace(
        boost_pads=[
            SimpleNamespace(location=_vec(live_pad_position(index)))
            for index in packet_order
        ]
    )
    adapter = Rival2LiveAdapter(manifest, field_info)
    live_rows = []
    accepted_rows = []
    block_max = {
        "ball": 0.0,
        "self_car": 0.0,
        "opponent_car": 0.0,
        "relative": 0.0,
        "boost_pads": 0.0,
        "previous_action": 0.0,
        "lifecycle": 0.0,
    }
    blocks = {
        "ball": slice(0, 9),
        "self_car": slice(9, 48),
        "opponent_car": slice(48, 87),
        "relative": slice(87, 99),
        "boost_pads": slice(99, 167),
        "previous_action": slice(167, 175),
        "lifecycle": slice(175, 182),
    }
    phases = ("OnGround", "Jumping", "DoubleJumping", "Dodging", "InAir")

    for sample in range(args.states):
        memory = LiveMemory.create()
        memory.episode_ticks[0] = rng.integers(0, 5401)
        memory.no_touch_ticks[0] = rng.integers(0, 1801)
        memory.kickoff_indicator[0] = rng.integers(0, 2)
        memory.touch_event[0] = rng.integers(0, 2, size=2)
        memory.demoed_event[0] = rng.integers(0, 2, size=2)
        memory.previous_action[0, :, :5] = rng.uniform(-1, 1, size=(2, 5))
        memory.previous_action[0, :, 5:] = rng.integers(0, 2, size=(2, 3))
        memory.time_since_boosted[0] = rng.uniform(0, 1.5, size=2)
        memory.sticky_ticks[0] = rng.integers(0, 4, size=2)
        memory.jump_time[0] = rng.uniform(0, 0.2, size=2)
        memory.air_time[0] = rng.uniform(0, 2.0, size=2)
        memory.air_time_since_jump[0] = rng.uniform(0, 1.5, size=2)
        memory.boosting_time[0] = rng.uniform(0, 0.2, size=2)
        memory.supersonic_time[0] = rng.uniform(0, 1.5, size=2)
        adapter.memory = memory

        players = []
        for team in (0, 1):
            phase = phases[int(rng.integers(0, len(phases)))]
            on_ground = phase == "OnGround"
            has_jumped = bool(rng.integers(0, 2)) and not on_ground
            has_double = bool(rng.integers(0, 2)) and has_jumped
            has_dodged = bool(rng.integers(0, 2)) and has_jumped and not has_double
            control = rng.uniform(-1, 1, size=8).astype(np.float32)
            control[5:] = rng.integers(0, 2, size=3)
            players.append(
                SimpleNamespace(
                    team=team,
                    physics=SimpleNamespace(
                        location=_vec(
                            np.asarray(
                                (
                                    rng.uniform(-4000, 4000),
                                    rng.uniform(-5000, 5000),
                                    rng.uniform(17, 1800),
                                ),
                                dtype=np.float32,
                            )
                        ),
                        velocity=_vec(rng.uniform(-2300, 2300, size=3)),
                        angular_velocity=_vec(rng.uniform(-5.5, 5.5, size=3)),
                        rotation=SimpleNamespace(
                            pitch=float(rng.uniform(-1.5, 1.5)),
                            yaw=float(rng.uniform(-np.pi, np.pi)),
                            roll=float(rng.uniform(-np.pi, np.pi)),
                        ),
                    ),
                    boost=float(rng.uniform(0, 100)),
                    air_state=f"AirState.{phase}",
                    has_jumped=has_jumped,
                    has_double_jumped=has_double,
                    has_dodged=has_dodged,
                    dodge_elapsed=float(rng.uniform(0, 0.95)),
                    demolished_timeout=(
                        float(rng.uniform(0.01, 3.0))
                        if rng.random() < 0.05
                        else -1.0
                    ),
                    is_supersonic=bool(rng.integers(0, 2)),
                    last_input=_controller(control),
                    latest_touch=None,
                )
            )
        if sample % 2:
            players.reverse()

        canonical_active = rng.integers(0, 2, size=34).astype(bool)
        canonical_elapsed = np.asarray(
            [
                0.0
                if active
                else rng.uniform(0, manifest["observation"]["canonical_boost_pad_durations"][i])
                for i, active in enumerate(canonical_active)
            ],
            dtype=np.float32,
        )
        packet_pads = [None] * 34
        for packet_index, canonical_index in enumerate(packet_order):
            packet_pads[packet_index] = SimpleNamespace(
                is_active=bool(canonical_active[canonical_index]),
                timer=float(canonical_elapsed[canonical_index]),
            )
        ball = SimpleNamespace(
            physics=SimpleNamespace(
                location=_vec(
                    np.asarray(
                        (
                            rng.uniform(-4000, 4000),
                            rng.uniform(-5100, 5100),
                            rng.uniform(93, 1900),
                        ),
                        dtype=np.float32,
                    )
                ),
                velocity=_vec(rng.uniform(-5800, 5800, size=3)),
                angular_velocity=_vec(rng.uniform(-5.5, 5.5, size=3)),
            )
        )
        packet = SimpleNamespace(players=players, balls=[ball], boost_pads=packet_pads)
        live = adapter.observation(packet)
        live_state = adapter._state(packet)
        accepted_state = RocketSimBatchState(
            **{
                field: getattr(live_state, field)
                for field in live_state.__dataclass_fields__
            },
            flip_rel_torque=np.zeros((1, 2, 3), dtype=np.float32),
        )
        accepted_memory = _accepted_memory(memory, RocketSimRivalMemory)
        accepted = build_rival2_observation(accepted_state, accepted_memory)
        error = np.abs(live - accepted)
        for name, block in blocks.items():
            block_max[name] = max(block_max[name], float(error[..., block].max()))
        live_rows.append(live.reshape(2, 182))
        accepted_rows.append(accepted.reshape(2, 182))

    live_observation = np.concatenate(live_rows, axis=0)
    accepted_observation = np.concatenate(accepted_rows, axis=0)
    observation_max = float(np.max(np.abs(live_observation - accepted_observation)))
    source_policy = FrozenRivalPolicy(
        manifest["source"]["checkpoint_path"], device="cpu", stochastic=False
    )
    source_action = source_policy.act(live_observation)
    exported_model = torch.jit.load(str(args.model), map_location="cpu").eval()
    with torch.inference_mode():
        exported_action = exported_model(torch.from_numpy(live_observation)).numpy()
    action_error = np.abs(source_action - exported_action)
    analog_max = float(action_error[:, :5].max())
    button_exact = float(
        np.all(source_action[:, 5:] == exported_action[:, 5:], axis=1).mean()
    )
    passed = observation_max == 0.0 and analog_max <= 1e-6 and button_exact == 1.0
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "PASS_GREEN" if passed else "FAIL_RED",
        "corpus": {
            "seed": args.seed,
            "physical_states": args.states,
            "team_observations": args.states * 2,
            "packet_player_order_swapped_states": args.states // 2,
            "all_five_air_states": True,
            "boost_pad_order_permuted": True,
        },
        "identity": {
            "checkpoint_sha256": manifest["source"]["checkpoint_sha256"],
            "policy_version": manifest["source"]["policy_version"],
            "artifact_sha256": manifest["artifact"]["sha256"],
            "observation_contract_sha256": manifest["contracts"][
                "observation_schema_sha256"
            ],
            "action_contract_sha256": manifest["contracts"][
                "action_contract_sha256"
            ],
        },
        "observation_parity": {
            "overall_max_abs_error": observation_max,
            "block_max_abs_error": block_max,
        },
        "deterministic_action_parity": {
            "analog_max_abs_error": analog_max,
            "binary_button_exact_fraction": button_exact,
        },
        "live_packet_qualifications": manifest["live_packet_qualifications"],
        "gates": {
            "observation_exact": observation_max == 0.0,
            "analog_action_max_abs_le_1e6": analog_max <= 1e-6,
            "binary_buttons_exact": button_exact == 1.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
