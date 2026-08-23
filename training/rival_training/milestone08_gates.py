"""Machine-readable pre-training gates for the Milestone 08 architecture."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actions import build_expanded_action_table
from .checkpoint import portable_path
from .dual_rate import StrategicWindowScheduler, dual_rate_metadata
from .environment import build_dual_rate_env
from .live_policy_parity import _load_live_corpus
from .teacher import FrozenWispReference, sha256_file, verify_teacher_hashes
from .wisp_actions import action_table_fingerprint, build_wisp_action_table


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ZERO_STEP_EXPORT = (
    REPOSITORY_ROOT / "training/artifacts/milestone07/zero_step_actor.ts"
)


def _logits(model: torch.nn.Module, observations: np.ndarray) -> torch.Tensor:
    values = []
    with torch.inference_mode():
        for start in range(0, len(observations), 512):
            values.append(model(torch.from_numpy(observations[start : start + 512])).cpu())
    return torch.cat(values)


def _parity_case(
    reference: torch.nn.Module,
    candidate: torch.nn.Module,
    observations: np.ndarray,
    masks: np.ndarray | None,
) -> dict[str, Any]:
    expected = _logits(reference, observations)[:, :90]
    actual = _logits(candidate, observations)[:, :90]
    difference = (expected - actual).abs()
    if masks is None:
        expected_top = expected.argmax(1)
        actual_top = actual.argmax(1)
    else:
        mask = torch.from_numpy(masks)
        expected_top = expected.masked_fill(~mask, -1e10).argmax(1)
        actual_top = actual.masked_fill(~mask, -1e10).argmax(1)
    return {
        "samples": len(observations),
        "max_abs_first_90_logit_error": float(difference.max().item()),
        "mean_abs_first_90_logit_error": float(difference.mean().item()),
        "allclose_atol_1e_6_rtol_1e_6": bool(
            torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
        ),
        "top1_agreement": float((expected_top == actual_top).float().mean().item()),
    }


def _array_digest(values: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(np.ascontiguousarray(value, dtype="<f4").tobytes())
    return digest.hexdigest()


def _temporal_proof() -> dict[str, Any]:
    scheduler = StrategicWindowScheduler(np.full(8, -1, dtype=np.float32))
    windows = []
    trace = []
    previous = -1
    exact = True
    for selected in range(12):
        row = np.full(8, selected, dtype=np.float32)
        window = scheduler.select(row)
        expected = np.stack(
            [np.full(8, previous, dtype=np.float32)] * 5 + [row] * 3
        )
        exact &= bool(np.array_equal(window, expected))
        windows.append(
            {
                "previous": previous,
                "selected": selected,
                "symbols": [previous] * 5 + [selected] * 3,
                "exact": bool(np.array_equal(window, expected)),
            }
        )
        trace.extend(scheduler.take(4))
        trace.extend(scheduler.take(4))
        previous = selected
    return {
        "decisions": len(windows),
        "physics_ticks": len(trace),
        "windows": windows,
        "trace_sha256": _array_digest(trace),
        "exact": exact and scheduler.pending_ticks == 0,
    }


def _fallback_proof(windows: int = 256) -> dict[str, Any]:
    disabled = build_dual_rate_env(
        natural_only=True, mechanics_disabled=True, seed=20260823
    )
    forced = build_dual_rate_env(
        natural_only=True, force_pass=True, seed=20260823
    )
    controller_rows: list[np.ndarray] = []
    maximum_observation_error = 0.0
    controller_exact = True
    strategic_indices_exact = True
    strategic_observations_exact = True
    strategic_logits_exact = True
    strategic_masks_exact = True
    synchronized_resets = 0
    strategic_decisions = 0
    try:
        disabled_obs = disabled.reset()
        forced_obs = forced.reset()
        for _ in range(windows):
            for agent in disabled_obs:
                maximum_observation_error = max(
                    maximum_observation_error,
                    float(np.max(np.abs(disabled_obs[agent] - forced_obs[agent]))),
                )
            disabled_actions = {
                agent: np.array([0], dtype=np.int64) for agent in disabled_obs
            }
            forced_actions = {
                agent: np.array([0], dtype=np.int64) for agent in forced_obs
            }
            disabled_obs, _, disabled_done, disabled_truncated = disabled.step(
                disabled_actions
            )
            forced_obs, _, forced_done, forced_truncated = forced.step(forced_actions)
            disabled_rows = disabled.shared_info["dual_rate_last_controllers"]
            forced_rows = forced.shared_info["dual_rate_last_controllers"]
            for agent in disabled_rows:
                controller_exact &= bool(
                    np.array_equal(disabled_rows[agent], forced_rows[agent])
                )
                controller_rows.extend(disabled_rows[agent])
            disabled_meta = disabled.shared_info["dual_rate_last_decisions"]
            forced_meta = forced.shared_info["dual_rate_last_decisions"]
            for agent in disabled_meta:
                strategic_indices_exact &= (
                    disabled_meta[agent]["strategic_action_index"]
                    == forced_meta[agent]["strategic_action_index"]
                )
                if disabled_meta[agent]["strategic_decision"]:
                    strategic_decisions += 1
            if any(item["strategic_decision"] for item in disabled_meta.values()):
                for key, accumulator in (
                    ("dual_rate_last_strategic_observations", "observations"),
                    ("dual_rate_last_strategic_logits", "logits"),
                    ("dual_rate_last_strategic_masks", "masks"),
                ):
                    left = disabled.shared_info[key]
                    right = forced.shared_info[key]
                    exact = all(np.array_equal(left[agent], right[agent]) for agent in left)
                    if accumulator == "observations":
                        strategic_observations_exact &= exact
                    elif accumulator == "logits":
                        strategic_logits_exact &= exact
                    else:
                        strategic_masks_exact &= exact
            done_equal = disabled_done == forced_done
            truncated_equal = disabled_truncated == forced_truncated
            if not done_equal or not truncated_equal:
                raise RuntimeError("Fallback environments diverged in episode boundaries")
            if any(disabled_done.values()) or any(disabled_truncated.values()):
                disabled_obs = disabled.reset()
                forced_obs = forced.reset()
                synchronized_resets += 1
    finally:
        disabled.close()
        forced.close()
    checks = {
        "controller_trace_exact": controller_exact,
        "mechanics_observation_trace_exact": maximum_observation_error == 0.0,
        "strategic_action_indices_exact": strategic_indices_exact,
        "strategic_observations_exact": strategic_observations_exact,
        "first_90_logits_exact": strategic_logits_exact,
        "legal_masks_exact": strategic_masks_exact,
    }
    return {
        "mechanics_windows": windows,
        "physics_ticks_per_agent": windows * 4,
        "strategic_agent_decisions": strategic_decisions,
        "synchronized_resets": synchronized_resets,
        "controller_trace_sha256": _array_digest(controller_rows),
        "maximum_observation_abs_error": maximum_observation_error,
        "checks": checks,
        "passed": all(checks.values()),
    }


def build_pretraining_gate_report(
    matrix_path: str | Path,
    observation_report_path: str | Path,
) -> dict[str, Any]:
    matrix_file = Path(matrix_path)
    observation_file = Path(observation_report_path)
    matrix = json.loads(matrix_file.read_text(encoding="utf-8"))
    observation_report = json.loads(observation_file.read_text(encoding="utf-8"))
    live, masks, _, sources = _load_live_corpus(matrix)
    random_observations = np.random.default_rng(20260823).standard_normal(
        (4096, 432), dtype=np.float32
    )
    reference = FrozenWispReference().eval()
    zero_step = torch.jit.load(str(ZERO_STEP_EXPORT), map_location="cpu").eval()
    randomized_parity = _parity_case(
        reference, zero_step, random_observations, None
    )
    held_parity = _parity_case(reference, zero_step, live, masks)
    temporal = _temporal_proof()
    fallback = _fallback_proof()
    wisp_table = build_wisp_action_table()
    expanded_table = build_expanded_action_table()
    table_gate = {
        "wisp_shape": list(wisp_table.shape),
        "expanded_shape": list(expanded_table.shape),
        "wisp_prefix_exact": bool(np.array_equal(wisp_table, expanded_table[:90])),
        "wisp_sha256": action_table_fingerprint(wisp_table),
        "expanded_sha256": action_table_fingerprint(expanded_table),
    }
    checks = {
        "teacher_hashes": verify_teacher_hashes()["all_match"],
        "randomized_first_90_logit_parity": randomized_parity[
            "allclose_atol_1e_6_rtol_1e_6"
        ],
        "held_first_90_logit_parity": held_parity[
            "allclose_atol_1e_6_rtol_1e_6"
        ],
        "observation_contract_v2": bool(observation_report["observation_gate"]["passed"]),
        "strategic_temporal_schedule": temporal["exact"],
        "mechanics_disabled_forced_pass_fallback": fallback["passed"],
        "action_prefix": table_gate["wisp_prefix_exact"],
    }
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "purpose": "milestone08_pretraining_software_gates",
        "checks": checks,
        "passed": all(checks.values()),
        "teacher_hashes": verify_teacher_hashes(),
        "zero_step_export": {
            "path": portable_path(ZERO_STEP_EXPORT),
            "sha256": sha256_file(ZERO_STEP_EXPORT),
            "size_bytes": ZERO_STEP_EXPORT.stat().st_size,
        },
        "randomized_logit_parity": randomized_parity,
        "held_live_logit_parity": {
            **held_parity,
            "sources": sources,
        },
        "observation_gate": {
            "path": portable_path(observation_file),
            "sha256": sha256_file(observation_file),
            "summary": observation_report["observation_gate"],
        },
        "temporal_scheduler": temporal,
        "fallback_invariant": fallback,
        "action_table": table_gate,
        "dual_rate_contract": dual_rate_metadata(),
        "production_modified_or_promoted": False,
    }


def write_pretraining_gate_report(
    matrix_path: str | Path,
    observation_report_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    report = build_pretraining_gate_report(matrix_path, observation_report_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
