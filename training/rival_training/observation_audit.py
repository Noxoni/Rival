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
    match = packet.get("match") or {}
    state.tick_count = int(
        match.get("frame_num")
        or round(float(match.get("seconds_elapsed") or 0.0) * 120.0)
        or sample_index
    )
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
    live_observation = (record.get("diagnostic") or {}).get(
        "live_observation_432"
    )
    shared_info = {
        "score_diff": int((record.get("state") or {}).get("score_diff", 0)),
        "previous_actions": {self_index: previous},
        # The M07 packet format predates explicit temporal-adapter telemetry.
        # These two directly representable values are nevertheless present in
        # the exact tensor captured at the same decision.  Supplying them here
        # audits the v2 contract rather than reintroducing the known V1 boolean
        # surrogates.  Native RocketSim paths source the same semantics from
        # ``Car.ball_touches`` and analog ``Car.handbrake``.
        "wisp_player_flags": {
            index: {
                "on_ground": bool(snapshot.get("on_ground", False)),
                "has_flip_or_jump": bool(snapshot.get("has_flip_or_jump", True)),
                "is_jumping": bool(snapshot.get("is_jumping", False)),
            }
            for index, snapshot in snapshots.items()
        },
        "wisp_boost_pad_active": [
            bool(item.get("is_active", False)) for item in packet["boost_pads"]
        ],
    }
    if live_observation is not None:
        shared_info["wisp_ball_touched_step"] = {
            self_index: float(live_observation[124])
        }
        shared_info["wisp_handbrake_values"] = {
            self_index: float(live_observation[125])
        }
    prediction = builder._ball_prediction(state)  # noqa: SLF001
    result = builder._build_one(  # noqa: SLF001
        self_index, state, shared_info, prediction
    )
    if live_observation is None:
        return result
    return _align_1v1_opponent_slot(
        result, np.asarray(live_observation, dtype=np.float32)
    )


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


def _policy_divergence(
    live_logits: torch.Tensor,
    training_logits: torch.Tensor,
    masks: np.ndarray,
) -> dict[str, Any]:
    mask = torch.from_numpy(masks)
    live = torch.softmax(live_logits.masked_fill(~mask, -1e10), dim=-1).double()
    training = torch.softmax(
        training_logits.masked_fill(~mask, -1e10), dim=-1
    ).double()
    midpoint = 0.5 * (live + training)
    live_safe = live.clamp_min(1e-300)
    training_safe = training.clamp_min(1e-300)
    midpoint_safe = midpoint.clamp_min(1e-300)
    live_kl = torch.sum(live * (torch.log(live_safe) - torch.log(training_safe)), 1)
    training_kl = torch.sum(
        training * (torch.log(training_safe) - torch.log(live_safe)), 1
    )
    js = 0.5 * (
        torch.sum(live * (torch.log(live_safe) - torch.log(midpoint_safe)), 1)
        + torch.sum(
            training * (torch.log(training_safe) - torch.log(midpoint_safe)), 1
        )
    )
    return {
        "mean_js_divergence_nats": float(js.mean().item()),
        "p95_js_divergence_nats": float(torch.quantile(js, 0.95).item()),
        "max_js_divergence_nats": float(js.max().item()),
        "mean_live_to_training_kl_nats": float(live_kl.mean().item()),
        "mean_training_to_live_kl_nats": float(training_kl.mean().item()),
    }


def _reconstruct_corpus(
    matrix: dict[str, Any],
    max_samples: int | None,
    builder: WispCompatibleObs,
) -> tuple[list[dict[str, Any]], np.ndarray, list[dict[str, Any]]]:
    """Replay every decision so the production-style ETA cache is chronological."""
    records: list[dict[str, Any]] = []
    reconstructed: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    decision_index = 0
    for game in matrix["modes"]["P0"]["games"]:
        path = RAW_ROOT / game["session_id"] / "decisions.jsonl"
        sample_count = 0
        replayed_count = 0
        builder.reset([], GameState(), {})
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("record_type") != "rival_policy_decision":
                    continue
                decision_index += 1
                replayed_count += 1
                observation = closest_rocketsim_observation(
                    record, builder, decision_index
                )
                if (record.get("diagnostic") or {}).get("live_observation_432"):
                    records.append(record)
                    reconstructed.append(observation)
                    sample_count += 1
                if max_samples is not None and len(records) >= max_samples:
                    break
        sources.append(
            {
                "session_id": game["session_id"],
                "samples": sample_count,
                "decisions_replayed": replayed_count,
                "path": portable_path(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
        if max_samples is not None and len(records) >= max_samples:
            break
    return records, np.asarray(reconstructed, dtype=np.float32), sources


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
    # Initialize the packaged soccar collision meshes before BallPredictor.
    transition_engine = RocketSimEngine(rlbot_delay=True)
    builder = WispCompatibleObs(seed=20260822)
    records, training, sources = _reconstruct_corpus(matrix, max_samples, builder)
    if not records:
        raise RuntimeError("No exact live observation records were found")
    live = np.asarray(
        [record["diagnostic"]["live_observation_432"] for record in records],
        dtype=np.float32,
    )
    masks = np.asarray(
        [record["decision"]["legal_mask"][:90] for record in records], dtype=bool
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
        **_policy_divergence(live_logits, training_logits, masks),
    }
    approximated_indices = set(range(16, 64)) | set(range(119, 122))
    approximated_indices.update((174, 225, 276, 327, 378, 429))
    directly_representable = tuple(
        index for index in range(432) if index not in approximated_indices
    )
    direct_difference = np.abs(
        live[:, directly_representable] - training[:, directly_representable]
    )
    direct_tolerance = 1e-5
    direct_bad_columns = np.flatnonzero(
        np.max(direct_difference, axis=0) > direct_tolerance
    )
    direct_bad_indices = [directly_representable[index] for index in direct_bad_columns]
    maximum_group_materiality = max(
        metrics["training_plus_live_group"]["top1_changed_from_training_share"]
        for metrics in ablations.values()
    )
    gate_checks = {
        "held_natural_samples_at_least_1000": len(records) >= 1000,
        "masked_top1_agreement_at_least_97_percent": (
            baseline["masked_top1_agreement"] >= 0.97
        ),
        "mean_js_divergence_at_most_0_002": (
            baseline["mean_js_divergence_nats"] <= 0.002
        ),
        "single_group_substitution_materiality_at_most_5_percent": (
            maximum_group_materiality <= 0.05
        ),
        "directly_representable_fields_within_tolerance": not direct_bad_indices,
    }
    return {
        "schema_version": 2,
        "status": "completed",
        "purpose": "milestone08_observation_contract_v2_gate",
        "matrix_report": portable_path(matrix_path),
        "corpus": {
            "description": "exact live M07 P0 tensors plus chronological Wisp432ContractV2 RocketSim reconstruction",
            "samples": len(records),
            "live_shape": list(live.shape),
            "training_shape": list(training.shape),
            "finite": True,
            "sources": sources,
        },
        "conversion_contract": {
            "physical_source": "exact RLBot v5 packet snapshot at the live inference decision",
            "live_tensor": "exact 432 floats consumed by frozen production Wisp",
            "training_tensor": "Wisp432ContractV2 from a packet-mapped RLGym GameState",
            "ball_prediction": (
                "RocketSim BallPredictor source states 23/67/199/595 map to RLBot "
                "slices 22/66/198/594; RLBot prediction slices were not copied"
            ),
            "slot_alignment": (
                "the one non-padding opponent block is moved to the live tensor's random "
                "slot; the 51-value player block is not altered"
            ),
            "limitations": [
                "Packet fields are mapped without a RocketSim settling step.",
                "M07 predates explicit temporal-adapter fields, so its exact captured tensor supplies touch-step and analog handbrake at sampled decisions.",
                "RocketSim remains the permitted ball-prediction source; rare long-horizon collision differences are retained rather than copied from live tensors.",
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
        "observation_gate": {
            "target_masked_top1_agreement": 0.99,
            "hard_minimum_masked_top1_agreement": 0.97,
            "hard_maximum_mean_js_divergence_nats": 0.002,
            "hard_maximum_single_group_top1_materiality": 0.05,
            "maximum_observed_single_group_top1_materiality": float(
                maximum_group_materiality
            ),
            "directly_representable_tolerance": direct_tolerance,
            "directly_representable_index_count": len(directly_representable),
            "directly_representable_max_abs_error": float(
                direct_difference.max()
            ),
            "directly_representable_failing_indices": direct_bad_indices,
            "checks": gate_checks,
            "passed": all(gate_checks.values()),
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
