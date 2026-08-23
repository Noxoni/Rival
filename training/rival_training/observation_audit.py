"""Live-RLBot to closest-RocketSim observation audit for Milestone 07."""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.sim import RocketSimEngine

from .checkpoint import portable_path
from .observations import (
    PLAYER_OBSERVATION_SIZE,
    RLGYM_TO_WISP_PAD_INDICES,
    WispCompatibleObs,
)
from .teacher import FrozenWispReference, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPOSITORY_ROOT / "evidence/raw"

# These ranges are mutually exclusive and cover the exact 432-value contract.
FEATURE_GROUPS: tuple[tuple[str, int, int], ...] = (
    ("ball_kinematics", 0, 9),
    ("goal_relative_vectors", 9, 15),
    ("kickoff_flag", 15, 16),
    ("ball_prediction_horizons", 16, 64),
    ("boost_pad_availability_timers", 64, 98),
    ("close_pad_relative_positions", 98, 108),
    ("previous_action", 108, 116),
    ("wall_distances", 116, 119),
    ("landing_normal", 119, 122),
    ("score_differential", 122, 123),
    ("self_turn_touch_handbrake", 123, 126),
    ("self_car_block", 126, 177),
    ("teammate_padding_blocks", 177, 279),
    ("opponent_blocks", 279, 432),
)

# Overlapping drill-down ranges answer the specifically named audit questions.
TARGETED_GROUPS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("self_eta", (174,)),
    ("opponent_eta_slots", (327, 378, 429)),
    ("touch_and_handbrake", (124, 125)),
    ("self_jump_dodge_ground_state", tuple(range(166, 172))),
    (
        "opponent_jump_dodge_ground_state",
        tuple(
            index
            for start in (279, 330, 381)
            for index in range(start + 40, start + 46)
        ),
    ),
)


def _array3(record: dict[str, Any] | None) -> np.ndarray:
    value = record or {}
    return np.asarray(
        [value.get("x") or 0.0, value.get("y") or 0.0, value.get("z") or 0.0],
        dtype=np.float32,
    )


def _physics(record: dict[str, Any]) -> PhysicsObject:
    physics = PhysicsObject()
    physics.position = _array3(record.get("position"))
    physics.linear_velocity = _array3(record.get("velocity"))
    physics.angular_velocity = _array3(record.get("angular_velocity"))
    rotation = record.get("rotation") or {}
    forward = _array3(rotation.get("forward"))
    up = _array3(rotation.get("up"))
    if float(np.linalg.norm(forward)) < 1e-6:
        forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if float(np.linalg.norm(up)) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    forward /= np.linalg.norm(forward)
    up -= forward * float(np.dot(forward, up))
    up /= max(float(np.linalg.norm(up)), 1e-7)
    right = np.cross(up, forward)
    right /= max(float(np.linalg.norm(right)), 1e-7)
    physics.rotation_mtx = np.column_stack((forward, right, up)).astype(np.float32)
    return physics


def _controller_array(record: dict[str, Any] | None) -> np.ndarray:
    value = record or {}
    return np.asarray(
        [
            value.get("throttle") or 0.0,
            value.get("steer") or 0.0,
            value.get("pitch") or 0.0,
            value.get("yaw") or 0.0,
            value.get("roll") or 0.0,
            float(bool(value.get("jump", False))),
            float(bool(value.get("boost", False))),
            float(bool(value.get("handbrake", False))),
        ],
        dtype=np.float32,
    )


def _snapshot_by_packet_index(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    packet = record["packet"]
    result: dict[int, dict[str, Any]] = {}
    self_index = int(packet["self_index"])
    state = record.get("state") or {}
    if isinstance(state.get("self"), dict):
        result[self_index] = state["self"]
    opponent_indices = list(packet.get("opponent_indices") or [])
    if opponent_indices and isinstance(state.get("opponent"), dict):
        result[int(opponent_indices[0])] = state["opponent"]
    return result


def _car(
    packet_player: dict[str, Any],
    live_snapshot: dict[str, Any] | None,
) -> Car:
    snapshot = live_snapshot or {}
    car = Car()
    car.team_num = int(packet_player["team"])
    car.hitbox_type = 0
    car.ball_touches = int(packet_player.get("latest_touch") is not None)
    car.bump_victim_id = None
    demolished = float(packet_player.get("demolished_timeout") or -1.0)
    car.demo_respawn_timer = max(0.0, demolished)
    on_ground = bool(snapshot.get("on_ground", False))
    car.wheels_with_contact = (on_ground,) * 4
    car.supersonic_time = float(bool(packet_player.get("is_supersonic")))
    car.boost_amount = float(packet_player.get("boost") or 0.0)
    last_input = packet_player.get("last_input") or {}
    car.boost_active_time = 1.0 / 120.0 if last_input.get("boost") else 0.0
    car.handbrake = float(bool(last_input.get("handbrake")))
    car.is_jumping = bool(snapshot.get("is_jumping", False))
    car.has_jumped = bool(packet_player.get("has_jumped", False))
    car.is_holding_jump = bool(last_input.get("jump", False))
    car.jump_time = 0.0
    has_live_flip = bool(snapshot.get("has_flip_or_jump", True))
    car.has_flipped = bool(packet_player.get("has_dodged", False))
    car.has_double_jumped = bool(packet_player.get("has_double_jumped", False))
    if not has_live_flip and not car.has_flipped and not car.has_double_jumped:
        car.has_double_jumped = True
    car.air_time_since_jump = 0.0 if has_live_flip else 2.0
    car.flip_time = max(0.0, float(packet_player.get("dodge_elapsed") or 0.0))
    car.flip_torque = np.zeros(3, dtype=np.float32)
    car.is_autoflipping = False
    car.autoflip_timer = 0.0
    car.autoflip_direction = 0.0
    car.physics = _physics(packet_player["physics"])
    car._inverted_physics = None
    return car


def closest_rocketsim_observation(
    record: dict[str, Any],
    builder: WispCompatibleObs,
    sample_index: int,
) -> np.ndarray:
    """Map an exact packet snapshot into the closest supported RLGym state."""
    packet = record["packet"]
    self_index = int(packet["self_index"])
    snapshots = _snapshot_by_packet_index(record)
    state = GameState()
    state.tick_count = sample_index
    state.goal_scored = False
    state.config = None
    state.cars = {
        int(player["packet_index"]): _car(
            player, snapshots.get(int(player["packet_index"]))
        )
        for player in packet["players"]
    }
    state.ball = _physics(packet["ball"]["physics"])
    state._inverted_ball = None
    wisp_timers = np.asarray(
        [float(item.get("timer") or 0.0) for item in packet["boost_pads"]],
        dtype=np.float32,
    )
    rlgym_timers = np.zeros_like(wisp_timers)
    rlgym_timers[RLGYM_TO_WISP_PAD_INDICES] = wisp_timers
    state.boost_pad_timers = rlgym_timers
    state._inverted_boost_pad_timers = None
    previous = _controller_array(
        ((record.get("state") or {}).get("self") or {}).get("previous_action")
    )
    shared_info = {
        "score_diff": int((record.get("state") or {}).get("score_diff", 0)),
        "previous_actions": {self_index: previous},
    }
    prediction = builder._ball_prediction(state)  # noqa: SLF001
    result = builder._build_one(  # noqa: SLF001
        self_index, state, shared_info, prediction
    )
    return _align_1v1_opponent_slot(result, np.asarray(
        record["diagnostic"]["live_observation_432"], dtype=np.float32
    ))


def _align_1v1_opponent_slot(training: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Remove irrelevant RNG slot order while preserving each 51-value block."""
    result = np.asarray(training, dtype=np.float32).copy()
    live_blocks = live[279:432].reshape(3, PLAYER_OBSERVATION_SIZE)
    training_blocks = result[279:432].reshape(3, PLAYER_OBSERVATION_SIZE).copy()
    live_slot = int(np.argmax(np.linalg.norm(live_blocks, axis=1)))
    training_slot = int(np.argmax(np.linalg.norm(training_blocks, axis=1)))
    if live_slot != training_slot:
        training_blocks[[live_slot, training_slot]] = training_blocks[
            [training_slot, live_slot]
        ]
        result[279:432] = training_blocks.reshape(-1)
    return result


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None, "minimum": None, "maximum": None}
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "minimum": float(finite.min()),
        "maximum": float(finite.max()),
    }


def _error_metrics(live: np.ndarray, training: np.ndarray) -> dict[str, Any]:
    difference = np.abs(live - training)
    meaningful = (np.abs(live) > 1e-7) & (np.abs(training) > 1e-7)
    if meaningful.any():
        sign_disagreement = float(
            (np.sign(live[meaningful]) != np.sign(training[meaningful])).mean()
        )
    else:
        sign_disagreement = None
    flat_live = live.reshape(-1).astype(np.float64)
    flat_training = training.reshape(-1).astype(np.float64)
    correlation = None
    if flat_live.std() > 1e-12 and flat_training.std() > 1e-12:
        correlation = float(np.corrcoef(flat_live, flat_training)[0, 1])
    lower = live.min(axis=0, keepdims=True)
    upper = live.max(axis=0, keepdims=True)
    outside = (training < lower - 1e-7) | (training > upper + 1e-7)
    return {
        "values": int(difference.size),
        "mean_abs_error": float(difference.mean()),
        "p50_abs_error": float(np.percentile(difference, 50)),
        "p95_abs_error": float(np.percentile(difference, 95)),
        "p99_abs_error": float(np.percentile(difference, 99)),
        "max_abs_error": float(difference.max()),
        "sign_disagreement_share_where_both_nonzero": sign_disagreement,
        "pearson_correlation_flattened": correlation,
        "training_outside_live_observed_per_feature_range_share": float(outside.mean()),
        "live_distribution": _distribution(live),
        "training_distribution": _distribution(training),
    }


def _batched_logits(model: torch.nn.Module, observations: np.ndarray) -> torch.Tensor:
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(observations), 512):
            tensor = torch.from_numpy(observations[start : start + 512])
            chunks.append(model(tensor)[:, :90].cpu())
    return torch.cat(chunks)


def _policy_metrics(logits: torch.Tensor, masks: np.ndarray) -> dict[str, np.ndarray]:
    mask = torch.from_numpy(masks)
    probabilities = torch.softmax(logits.masked_fill(~mask, -1e10), dim=-1)
    values, indices = torch.topk(probabilities, 2, dim=-1)
    return {
        "top1": indices[:, 0].numpy(),
        "confidence": values[:, 0].numpy(),
        "margin": (values[:, 0] - values[:, 1]).numpy(),
    }


def _load_records(
    matrix: dict[str, Any], max_samples: int | None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    sources = []
    for game in matrix["modes"]["P0"]["games"]:
        path = RAW_ROOT / game["session_id"] / "decisions.jsonl"
        count = 0
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("record_type") != "rival_policy_decision":
                    continue
                if not (record.get("diagnostic") or {}).get("live_observation_432"):
                    continue
                records.append(record)
                count += 1
                if max_samples is not None and len(records) >= max_samples:
                    break
        sources.append(
            {
                "session_id": game["session_id"],
                "samples": count,
                "path": portable_path(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    return records, sources


def _indices(start: int, end: int) -> tuple[int, ...]:
    return tuple(range(start, end))


def _ablation_metrics(
    model: torch.nn.Module,
    live: np.ndarray,
    training: np.ndarray,
    masks: np.ndarray,
    live_logits: torch.Tensor,
    training_logits: torch.Tensor,
    indices: Iterable[int],
) -> dict[str, Any]:
    selected = tuple(indices)
    training_plus_live = training.copy()
    training_plus_live[:, selected] = live[:, selected]
    live_plus_training = live.copy()
    live_plus_training[:, selected] = training[:, selected]
    tpl_logits = _batched_logits(model, training_plus_live)
    lpt_logits = _batched_logits(model, live_plus_training)
    live_policy = _policy_metrics(live_logits, masks)
    training_policy = _policy_metrics(training_logits, masks)
    tpl_policy = _policy_metrics(tpl_logits, masks)
    lpt_policy = _policy_metrics(lpt_logits, masks)
    baseline_disagreement = training_policy["top1"] != live_policy["top1"]
    recovered = baseline_disagreement & (tpl_policy["top1"] == live_policy["top1"])
    return {
        "training_plus_live_group": {
            "top1_changed_from_training_share": float(
                (tpl_policy["top1"] != training_policy["top1"]).mean()
            ),
            "top1_agreement_with_live": float(
                (tpl_policy["top1"] == live_policy["top1"]).mean()
            ),
            "baseline_disagreements_recovered": int(recovered.sum()),
            "mean_abs_first_90_logit_change": float(
                (tpl_logits - training_logits).abs().mean().item()
            ),
            "max_abs_first_90_logit_change": float(
                (tpl_logits - training_logits).abs().max().item()
            ),
        },
        "live_plus_training_group": {
            "top1_changed_from_live_share": float(
                (lpt_policy["top1"] != live_policy["top1"]).mean()
            ),
            "mean_abs_first_90_logit_change": float(
                (lpt_logits - live_logits).abs().mean().item()
            ),
            "max_abs_first_90_logit_change": float(
                (lpt_logits - live_logits).abs().max().item()
            ),
        },
    }


def build_observation_audit_report(
    matrix_report_path: str | Path,
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    matrix_path = Path(matrix_report_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    records, sources = _load_records(matrix, max_samples)
    if not records:
        raise RuntimeError("No exact live observation records were found")
    # Initialize the packaged soccar collision meshes before BallPredictor.
    transition_engine = RocketSimEngine(rlbot_delay=True)
    builder = WispCompatibleObs(seed=20260822)
    live = np.asarray(
        [record["diagnostic"]["live_observation_432"] for record in records],
        dtype=np.float32,
    )
    masks = np.asarray(
        [record["decision"]["legal_mask"][:90] for record in records], dtype=bool
    )
    training = np.stack(
        [
            closest_rocketsim_observation(record, builder, index + 1)
            for index, record in enumerate(records)
        ]
    )
    if live.shape != training.shape or live.shape[1:] != (432,):
        raise AssertionError(f"Observation shapes differ: {live.shape} vs {training.shape}")
    if not np.isfinite(live).all() or not np.isfinite(training).all():
        raise FloatingPointError("Observation audit produced non-finite values")

    model = FrozenWispReference().eval()
    live_logits = _batched_logits(model, live)
    training_logits = _batched_logits(model, training)
    live_policy = _policy_metrics(live_logits, masks)
    training_policy = _policy_metrics(training_logits, masks)
    feature_report: dict[str, Any] = {}
    for name, start, end in FEATURE_GROUPS:
        feature_report[name] = {
            "range": [start, end],
            **_error_metrics(live[:, start:end], training[:, start:end]),
        }
    targeted_report = {
        name: {
            "indices": list(indices),
            **_error_metrics(live[:, indices], training[:, indices]),
        }
        for name, indices in TARGETED_GROUPS
    }

    ablations = {}
    for name, start, end in FEATURE_GROUPS:
        ablations[name] = _ablation_metrics(
            model,
            live,
            training,
            masks,
            live_logits,
            training_logits,
            _indices(start, end),
        )
    targeted_ablations = {
        name: _ablation_metrics(
            model,
            live,
            training,
            masks,
            live_logits,
            training_logits,
            indices,
        )
        for name, indices in TARGETED_GROUPS
    }
    sensitivity_ranking = sorted(
        (
            {
                "feature_group": name,
                **metrics["training_plus_live_group"],
            }
            for name, metrics in ablations.items()
        ),
        key=lambda item: (
            item["top1_changed_from_training_share"],
            item["mean_abs_first_90_logit_change"],
        ),
        reverse=True,
    )
    logit_difference = (live_logits - training_logits).abs()
    baseline = {
        "masked_top1_agreement": float(
            (live_policy["top1"] == training_policy["top1"]).mean()
        ),
        "disagreement_count": int(
            (live_policy["top1"] != training_policy["top1"]).sum()
        ),
        "mean_abs_first_90_logit_error": float(logit_difference.mean().item()),
        "max_abs_first_90_logit_error": float(logit_difference.max().item()),
        "live_confidence": _distribution(live_policy["confidence"]),
        "training_style_confidence": _distribution(training_policy["confidence"]),
        "live_margin": _distribution(live_policy["margin"]),
        "training_style_margin": _distribution(training_policy["margin"]),
    }
    return {
        "schema_version": 1,
        "status": "completed",
        "purpose": "milestone07_observation_domain_audit",
        "matrix_report": portable_path(matrix_path),
        "corpus": {
            "description": "exact live P0 tensors plus same-packet closest RLGym/RocketSim reconstruction",
            "samples": len(records),
            "live_shape": list(live.shape),
            "training_shape": list(training.shape),
            "finite": True,
            "sources": sources,
        },
        "conversion_contract": {
            "physical_source": "exact RLBot v5 packet snapshot at the live inference decision",
            "live_tensor": "exact 432 floats consumed by frozen production Wisp",
            "training_tensor": "WispCompatible432RLGymV1 from a packet-mapped RLGym GameState",
            "ball_prediction": (
                "RocketSim BallPredictor recomputed at ticks 22/66/198/594 from the "
                "same current ball state; RLBot prediction slices were not copied"
            ),
            "slot_alignment": (
                "the one non-padding opponent block is moved to the live tensor's random "
                "slot; the 51-value player block is not altered"
            ),
            "limitations": [
                "Packet fields are mapped without a RocketSim settling step.",
                "RLBot air/jump flags are mapped to the closest RLGym Car flags.",
                "Live arena-SDF landing normal and cached Wisp ETA remain intentionally different approximations.",
            ],
        },
        "whole_observation": _error_metrics(live, training),
        "feature_groups": feature_report,
        "targeted_state_groups": targeted_report,
        "frozen_wisp_policy_effect": {
            "live_vs_training_style": baseline,
            "group_substitution_ablations": ablations,
            "targeted_state_substitution_ablations": targeted_ablations,
            "sensitivity_ranking": sensitivity_ranking,
        },
        "production_modified_or_promoted": False,
        "diagnostic_transition_engine_initialized": type(transition_engine).__name__,
    }


def write_observation_audit_report(
    matrix_report_path: str | Path,
    output_path: str | Path,
    *,
    max_samples: int | None = None,
) -> dict[str, Any]:
    report = build_observation_audit_report(
        matrix_report_path, max_samples=max_samples
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
