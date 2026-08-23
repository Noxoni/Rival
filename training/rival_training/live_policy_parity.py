"""Same-live-observation policy comparison for Milestone 07."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import portable_path
from .teacher import FrozenWispReference, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPOSITORY_ROOT / "evidence/raw"
ZERO_STEP_EXPORT = REPOSITORY_ROOT / "training/artifacts/milestone07/zero_step_actor.ts"
TRAINED_EXPORT = (
    REPOSITORY_ROOT / "training/artifacts/milestone06/stage_b_020m/candidate_actor.ts"
)


def _percentile(values: np.ndarray, q: float) -> float | None:
    return None if values.size == 0 else float(np.percentile(values, q))


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "minimum": None,
            "p50": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "minimum": float(finite.min()),
        "p50": _percentile(finite, 50),
        "p95": _percentile(finite, 95),
        "p99": _percentile(finite, 99),
        "maximum": float(finite.max()),
    }


def _load_live_corpus(
    matrix_report: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    p0 = matrix_report["modes"]["P0"]
    observations: list[list[float]] = []
    masks: list[list[bool]] = []
    context: list[dict[str, Any]] = []
    sources = []
    for game in p0["games"]:
        session_id = game["session_id"]
        telemetry_path = RAW_ROOT / session_id / "decisions.jsonl"
        session_samples = 0
        invalid = 0
        decision_index = 0
        with telemetry_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    invalid += 1
                    continue
                if record.get("record_type") != "rival_policy_decision":
                    continue
                diagnostic = record.get("diagnostic") or {}
                live = diagnostic.get("live_observation_432")
                if live is None:
                    decision_index += 1
                    continue
                legal = (record.get("decision") or {}).get("legal_mask")
                if not isinstance(live, list) or len(live) != 432:
                    raise ValueError(
                        f"Invalid live observation in {session_id} line {line_number}"
                    )
                if not isinstance(legal, list) or len(legal) < 90:
                    raise ValueError(f"Invalid legal mask in {session_id} line {line_number}")
                observations.append([float(value) for value in live])
                masks.append([bool(value) for value in legal[:90]])
                state = record.get("state") or {}
                tactical = record.get("tactical_metrics") or {}
                packet_match = ((record.get("packet") or {}).get("match") or {})
                context.append(
                    {
                        "session_id": session_id,
                        "decision_index": decision_index,
                        "logged_action_index": int(record["decision"]["action_index"]),
                        "ball_height": float(tactical.get("ball_height", 0.0)),
                        "ball_distance": float(tactical.get("ball_distance", 0.0)),
                        "self_boost": float(tactical.get("self_boost", 0.0)),
                        "self_airborne": bool(tactical.get("self_airborne", False)),
                        "opponent_airborne": bool(
                            tactical.get("opponent_airborne", False)
                        ),
                        "score_diff": int(state.get("score_diff", 0)),
                        "phase": str((packet_match.get("phase") or {}).get("name")),
                        "kickoff_feature": float(live[15]),
                    }
                )
                session_samples += 1
                decision_index += 1
        sources.append(
            {
                "session_id": session_id,
                "opponent": game["opponent"],
                "rival_side": game["rival_side"],
                "samples": session_samples,
                "invalid_json_records": invalid,
                "raw_path": portable_path(telemetry_path),
                "raw_sha256": sha256_file(telemetry_path),
                "raw_size_bytes": telemetry_path.stat().st_size,
            }
        )
    observation_array = np.asarray(observations, dtype=np.float32)
    mask_array = np.asarray(masks, dtype=bool)
    if observation_array.ndim != 2 or observation_array.shape[1:] != (432,):
        raise RuntimeError(f"Live corpus has invalid shape {observation_array.shape}")
    if mask_array.shape != (len(observation_array), 90) or not mask_array.any(axis=1).all():
        raise RuntimeError(f"Live legal-mask corpus has invalid shape/content {mask_array.shape}")
    if not np.isfinite(observation_array).all():
        raise RuntimeError("Live corpus contains non-finite observations")
    return observation_array, mask_array, context, sources


def _masked_policy_metrics(logits: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
    masked = logits.masked_fill(~masks, -1e10)
    probabilities = torch.softmax(masked, dim=-1)
    top_values, top_indices = torch.topk(probabilities, k=2, dim=-1)
    return {
        "probabilities": probabilities,
        "top1": top_indices[:, 0],
        "confidence": top_values[:, 0],
        "margin": top_values[:, 0] - top_values[:, 1],
    }


def _divergence(
    first: torch.Tensor, second: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    epsilon = 1e-12
    p = first.clamp_min(epsilon)
    q = second.clamp_min(epsilon)
    middle = ((p + q) * 0.5).clamp_min(epsilon)
    kl_p_q = (p * (p.log() - q.log())).sum(dim=-1)
    kl_q_p = (q * (q.log() - p.log())).sum(dim=-1)
    js = 0.5 * (
        (p * (p.log() - middle.log())).sum(dim=-1)
        + (q * (q.log() - middle.log())).sum(dim=-1)
    )
    return kl_p_q, kl_q_p, js


def _counter(counter: Counter[Any], limit: int = 30) -> list[dict[str, Any]]:
    return [
        {
            "from": int(key[0]) if isinstance(key, tuple) else None,
            "to": int(key[1]) if isinstance(key, tuple) else int(key),
            "count": int(count),
        }
        for key, count in counter.most_common(limit)
    ]


def _action_summary(
    actions: np.ndarray,
    contexts: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(int(value) for value in actions)
    sequential = Counter()
    previous_by_session: dict[str, int] = {}
    changes = comparisons = 0
    for action, context in zip(actions, contexts):
        session = context["session_id"]
        previous = previous_by_session.get(session)
        if previous is not None:
            sequential[(previous, int(action))] += 1
            comparisons += 1
            changes += int(previous != int(action))
        previous_by_session[session] = int(action)
    total = len(actions)
    entropy = -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )
    return {
        "sampled_decisions": total,
        "unique_actions": len(counts),
        "entropy_nats": entropy,
        "top_action_counts": _counter(counts, 20),
        "stride_sampled_action_change_share": changes / max(comparisons, 1),
        "top_stride_sampled_transitions": _counter(sequential, 30),
    }


def _region_labels(context: dict[str, Any]) -> dict[str, str]:
    ball_height = context["ball_height"]
    boost = context["self_boost"]
    distance = context["ball_distance"]
    return {
        "ball_height": "low" if ball_height < 250 else "mid" if ball_height < 800 else "high",
        "ball_distance": "close" if distance < 800 else "mid" if distance < 2500 else "far",
        "self_boost": "empty" if boost <= 1 else "low" if boost < 34 else "medium" if boost < 67 else "high",
        "self_airborne": str(context["self_airborne"]).lower(),
        "opponent_airborne": str(context["opponent_airborne"]).lower(),
        "score_state": "behind" if context["score_diff"] < 0 else "tied" if context["score_diff"] == 0 else "ahead",
        "phase": context["phase"],
        "kickoff": str(context["kickoff_feature"] > 0.5).lower(),
    }


def _region_report(
    contexts: list[dict[str, Any]],
    agreement: np.ndarray,
    js: np.ndarray,
    mean_abs_logit_drift: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[int]]] = {}
    for index, context in enumerate(contexts):
        for family, label in _region_labels(context).items():
            grouped.setdefault(family, {}).setdefault(label, []).append(index)
    return {
        family: [
            {
                "region": label,
                "samples": len(indices),
                "top1_agreement": float(agreement[indices].mean()),
                "mean_js_divergence": float(js[indices].mean()),
                "mean_abs_first_90_logit_drift": float(
                    mean_abs_logit_drift[indices].mean()
                ),
            }
            for label, indices in sorted(labels.items())
        ]
        for family, labels in sorted(grouped.items())
    }


def build_live_policy_parity_report(
    matrix_report_path: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    matrix_path = Path(matrix_report_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    observations, masks, contexts, sources = _load_live_corpus(matrix)
    selected_device = torch.device(device)
    reference = FrozenWispReference().to(selected_device).eval()
    zero = torch.jit.load(str(ZERO_STEP_EXPORT), map_location=selected_device).eval()
    trained = torch.jit.load(str(TRAINED_EXPORT), map_location=selected_device).eval()
    observation_tensor = torch.from_numpy(observations).to(selected_device)
    mask_tensor = torch.from_numpy(masks).to(selected_device)
    with torch.inference_mode():
        frozen_logits = reference(observation_tensor)
        zero_logits = zero(observation_tensor)[:, :90]
        trained_logits = trained(observation_tensor)[:, :90]
        frozen_metrics = _masked_policy_metrics(frozen_logits, mask_tensor)
        zero_metrics = _masked_policy_metrics(zero_logits, mask_tensor)
        trained_metrics = _masked_policy_metrics(trained_logits, mask_tensor)
        zero_difference = (frozen_logits - zero_logits).abs()
        trained_difference = (frozen_logits - trained_logits).abs()
        zero_kl_fz, zero_kl_zf, zero_js = _divergence(
            frozen_metrics["probabilities"], zero_metrics["probabilities"]
        )
        trained_kl_ft, trained_kl_tf, trained_js = _divergence(
            frozen_metrics["probabilities"], trained_metrics["probabilities"]
        )

    def cpu(name: dict[str, torch.Tensor], key: str) -> np.ndarray:
        return name[key].detach().cpu().numpy()

    frozen_top1 = cpu(frozen_metrics, "top1")
    zero_top1 = cpu(zero_metrics, "top1")
    trained_top1 = cpu(trained_metrics, "top1")
    logged = np.asarray([item["logged_action_index"] for item in contexts])
    zero_agreement = zero_top1 == frozen_top1
    trained_agreement = trained_top1 == frozen_top1
    trained_js_array = trained_js.detach().cpu().numpy()
    per_sample_drift = trained_difference.mean(dim=-1).detach().cpu().numpy()
    cross_transitions = Counter(
        (int(first), int(second)) for first, second in zip(frozen_top1, trained_top1)
    )
    disagreement_transitions = Counter(
        (int(first), int(second))
        for first, second in zip(frozen_top1, trained_top1)
        if first != second
    )
    zero_parity = {
        "max_abs_first_90_logit_error": float(zero_difference.max().item()),
        "mean_abs_first_90_logit_error": float(zero_difference.mean().item()),
        "first_90_logits_exact": bool(torch.equal(frozen_logits, zero_logits)),
        "first_90_logits_allclose_atol_1e-6_rtol_1e-6": bool(
            torch.allclose(frozen_logits, zero_logits, atol=1e-6, rtol=1e-6)
        ),
        "masked_top1_agreement": float(zero_agreement.mean()),
        "frozen_recomputed_vs_logged_top1_agreement": float(
            (frozen_top1 == logged).mean()
        ),
        "kl_frozen_to_zero": _distribution(zero_kl_fz.detach().cpu().numpy()),
        "kl_zero_to_frozen": _distribution(zero_kl_zf.detach().cpu().numpy()),
        "js_divergence": _distribution(zero_js.detach().cpu().numpy()),
    }
    zero_parity["passed"] = all(
        (
            zero_parity["first_90_logits_allclose_atol_1e-6_rtol_1e-6"],
            zero_parity["masked_top1_agreement"] == 1.0,
            zero_parity["frozen_recomputed_vs_logged_top1_agreement"] == 1.0,
        )
    )
    report = {
        "schema_version": 1,
        "status": "passed" if zero_parity["passed"] else "failed",
        "purpose": "milestone07_same_live_observation_policy_parity",
        "matrix_report": portable_path(matrix_path),
        "device": str(selected_device),
        "corpus": {
            "description": "sampled exact 432 tensors used by fresh P0 RLBot decisions",
            "observation_capture_version": "RivalM07LiveObservationV1",
            "shape": list(observations.shape),
            "samples": len(observations),
            "finite": bool(np.isfinite(observations).all()),
            "legal_mask_shape": list(masks.shape),
            "sources": sources,
        },
        "models": {
            "frozen_wisp": {
                "policy_sha256": sha256_file(REPOSITORY_ROOT / "bot/models/POLICY.lt"),
                "shared_head_sha256": sha256_file(
                    REPOSITORY_ROOT / "bot/models/SHARED_HEAD.lt"
                ),
            },
            "zero_step": {
                "path": portable_path(ZERO_STEP_EXPORT),
                "sha256": sha256_file(ZERO_STEP_EXPORT),
            },
            "trained_20m": {
                "path": portable_path(TRAINED_EXPORT),
                "sha256": sha256_file(TRAINED_EXPORT),
            },
        },
        "zero_step_vs_frozen": zero_parity,
        "trained_20m_vs_frozen": {
            "max_abs_first_90_logit_drift": float(trained_difference.max().item()),
            "mean_abs_first_90_logit_drift": float(trained_difference.mean().item()),
            "per_sample_mean_abs_logit_drift": _distribution(per_sample_drift),
            "masked_top1_agreement": float(trained_agreement.mean()),
            "disagreement_count": int((~trained_agreement).sum()),
            "kl_frozen_to_trained": _distribution(
                trained_kl_ft.detach().cpu().numpy()
            ),
            "kl_trained_to_frozen": _distribution(
                trained_kl_tf.detach().cpu().numpy()
            ),
            "js_divergence": _distribution(trained_js_array),
            "confidence": {
                "frozen": _distribution(cpu(frozen_metrics, "confidence")),
                "trained": _distribution(cpu(trained_metrics, "confidence")),
                "trained_minus_frozen": _distribution(
                    cpu(trained_metrics, "confidence")
                    - cpu(frozen_metrics, "confidence")
                ),
            },
            "top1_top2_margin": {
                "frozen": _distribution(cpu(frozen_metrics, "margin")),
                "trained": _distribution(cpu(trained_metrics, "margin")),
                "trained_minus_frozen": _distribution(
                    cpu(trained_metrics, "margin") - cpu(frozen_metrics, "margin")
                ),
            },
            "cross_policy_action_mapping_top30": _counter(cross_transitions, 30),
            "disagreement_action_mapping_top30": _counter(
                disagreement_transitions, 30
            ),
            "region_breakdown": _region_report(
                contexts, trained_agreement, trained_js_array, per_sample_drift
            ),
        },
        "action_frequencies": {
            "frozen_wisp": _action_summary(frozen_top1, contexts),
            "zero_step": _action_summary(zero_top1, contexts),
            "trained_20m": _action_summary(trained_top1, contexts),
        },
        "match_testing_gate": {
            "zero_step_live_parity_required": True,
            "passed": zero_parity["passed"],
            "student_match_testing_authorized": zero_parity["passed"],
        },
    }
    if not zero_parity["passed"]:
        raise RuntimeError(f"Zero-step failed live parity gate: {zero_parity}")
    return report
