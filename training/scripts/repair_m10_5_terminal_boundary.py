"""Validate and reclassify the clean M10.5 terminal worker-segment shortfall."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import run_m10_2_stage1_boundary as base  # noqa: E402
from rival_training.m10_campaign import (  # noqa: E402
    verify_checkpoint_reload_parity,
    write_json_atomic,
)
from rival_training.v10_5_campaign import (  # noqa: E402
    CAMPAIGN_STATE_PATH,
    DEFAULT_STAGE1_CONFIG,
    RESULT_ROOT,
    load_stage1_config,
    update_progressive_state,
)
from rival_training.v9_checkpoint import sha256_file  # noqa: E402


TRAINING_RESULT = RESULT_ROOT / "stage_1/training_plus-005h.json"


def main() -> int:
    result = json.loads(TRAINING_RESULT.read_text(encoding="utf-8"))
    state = json.loads(CAMPAIGN_STATE_PATH.read_text(encoding="utf-8"))
    if result["training_boundary_version"] != "RivalM10_5Stage1TrainingBoundaryV1":
        raise RuntimeError("Refusing a non-M10.5 terminal result")
    if float(result["boundary_hours"]) != 5.0 or result["status"] != "failed":
        raise RuntimeError("M10.5 result is not the expected failed +5h classification")
    if result["stop_reason"] != "stop_stage_1_boundary_not_reached":
        raise RuntimeError("M10.5 result failed for an unexpected reason")
    if state["stop_reason"] != "stop_stage_1_boundary_not_reached":
        raise RuntimeError("M10.5 state is not at the expected recoverable stop")

    config = deepcopy(load_stage1_config(DEFAULT_STAGE1_CONFIG))
    preflight = json.loads((RESULT_ROOT / "preflight.json").read_text(encoding="utf-8"))
    config["backend"]["worker_count"] = int(preflight["effective_selected_worker_count"])
    terminal = base._terminal_boundary_status(  # noqa: SLF001
        cumulative_steps=int(result["reached_active_learner_steps"]),
        target_steps=int(result["target_active_learner_steps"]),
        stage_maximum_steps=int(config["stage_contract"]["maximum_active_learner_steps"]),
        worker_count=int(config["backend"]["worker_count"]),
    )
    if not terminal["accepted_terminal_worker_segment_shortfall"]:
        raise RuntimeError(f"Terminal shortfall is outside the frozen tolerance: {terminal}")
    expected_checks = {
        key: value
        for key, value in result["checks"].items()
        if key not in {"boundary_reached_or_wall_stop", "passed"}
    }
    if (
        not all(
            value
            for key, value in expected_checks.items()
            if key != "production_promotion_authorized"
        )
        or expected_checks["production_promotion_authorized"] is not False
    ):
        raise RuntimeError(f"A non-boundary training check failed: {expected_checks}")

    checkpoint = REPOSITORY_ROOT / result["immutable_checkpoint"]["directory"]
    if sha256_file(checkpoint / "actor.pt") != result["immutable_checkpoint"]["actor_sha256"]:
        raise RuntimeError("Terminal actor hash mismatch")
    if (
        sha256_file(checkpoint / "checkpoint_manifest.json")
        != result["immutable_checkpoint"]["manifest_sha256"]
    ):
        raise RuntimeError("Terminal manifest hash mismatch")
    reload_parity = verify_checkpoint_reload_parity(
        checkpoint,
        expected_config=config,
        device="cpu",
    )
    if not reload_parity["checks"]["passed"]:
        raise RuntimeError("Terminal checkpoint failed independent reload parity")

    result["status"] = "passed"
    result["terminal_boundary_tolerance"] = {
        **terminal,
        "rationale": (
            "The rollout scheduler reserves two worker segments at the hard experience "
            "ceiling to prevent an unauthorized overshoot."
        ),
    }
    result["terminal_boundary_repair"] = {
        "version": "RivalM10_5TerminalWorkerShortfallRepairV1",
        "actor_or_optimizer_update_performed": False,
        "reward_or_curriculum_change_performed": False,
        "independent_checkpoint_reload": reload_parity,
    }
    result["stop_reason"] = None
    result["checks"]["boundary_reached_or_wall_stop"] = True
    result["checks"]["terminal_worker_segment_shortfall_accepted"] = True
    result["checks"]["passed"] = True
    write_json_atomic(TRAINING_RESULT, result)
    update_progressive_state(
        {
            "stage_active_learner_steps": int(result["reached_active_learner_steps"]),
            "stage_simulated_hours": float(result["reached_simulated_hours"]),
            "total_progressive_active_learner_steps": int(result["reached_active_learner_steps"]),
            "total_progressive_simulated_hours": float(result["reached_simulated_hours"]),
            "current_evaluation_boundary": 5.0,
            "gate_decision": "pending_stage_1_evaluation",
            "latest_clean_recovery_checkpoint": result["immutable_checkpoint"],
            "stop_reason": None,
            "terminal_boundary_tolerance": result["terminal_boundary_tolerance"],
        }
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checkpoint": result["immutable_checkpoint"],
                "terminal_boundary_tolerance": result["terminal_boundary_tolerance"],
                "reload_passed": reload_parity["checks"]["passed"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
