"""Evaluate one M10.8 GAE arm on the frozen deterministic/stochastic corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import write_json_atomic  # noqa: E402
from rival_training.v10_7_checkpoint import load_checkpoint  # noqa: E402
from rival_training.v10_7_diagnostics import (  # noqa: E402
    build_observation_corpus,
    button_policy_diagnostics,
)
from rival_training.v10_7_evaluation import (  # noqa: E402
    capability_gap,
    evaluate_stage1_checkpoint,
)
from rival_training.v10_8_campaign import (  # noqa: E402
    ARM_LAMBDAS,
    CORPUS_ROOT,
    GATE_CORPUS_FILENAME,
    RESULT_ROOT,
    arm_slug,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _boundary_slug(hours: float) -> str:
    return {0.5: "plus-000p5h", 1.0: "plus-001h"}[float(hours)]


def compact_capability(report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    initial_distance = overall["failed_episode_initial_car_ball_distance"]
    terminal_distance = overall["failed_episode_terminal_car_ball_distance"]
    initial_alignment = overall["failed_episode_initial_car_ball_alignment"]
    terminal_alignment = overall["failed_episode_terminal_car_ball_alignment"]
    return {
        "first_contact_success": overall["first_touch_success_share"],
        "second_contact_success": overall["second_touch_success_share"],
        "third_contact_success": overall["third_touch_success_share"],
        "all_three_success": overall["all_three_contacts_success_share"],
        "no_touch_rate": overall["no_touch_timeout_share"],
        "timing_seconds": {
            "reset_to_first": overall["successful_time_to_first_touch_seconds"],
            "first_to_second": overall[
                "successful_time_first_to_second_touch_seconds"
            ],
            "second_to_third": overall[
                "successful_time_second_to_third_touch_seconds"
            ],
        },
        "failed_trajectory": {
            "initial_distance": initial_distance,
            "terminal_distance": terminal_distance,
            "distance_change_terminal_minus_initial_mean": (
                float(terminal_distance["mean"]) - float(initial_distance["mean"])
            ),
            "initial_alignment": initial_alignment,
            "terminal_alignment": terminal_alignment,
            "alignment_change_terminal_minus_initial_mean": (
                float(terminal_alignment["mean"])
                - float(initial_alignment["mean"])
            ),
        },
        "actions": overall["action_diagnostics"],
        "button_policy_diagnostics": overall["button_policy_diagnostics"],
        "mean_button_entropy": overall["mean_button_entropy"],
        "reward_components": {
            "heading": overall["heading_reward_total"],
            "distance": overall["distance_reward_total"],
            "acquisition_time": overall["cumulative_acquisition_time_penalty"],
            "touch": overall["touch_reward_total"],
        },
    }


def _initial_row(preflight: dict[str, Any]) -> dict[str, Any]:
    capability = preflight["initialization_capability"]
    return {
        "boundary_hours": 0.0,
        "deterministic": compact_capability(
            {"overall": capability["deterministic_overall"]}
        ),
        "stochastic": compact_capability(
            {"overall": capability["stochastic_overall"]}
        ),
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    arm = str(args.arm).upper()
    training = _read(args.training_result)
    if training["arm"] != arm or training["gae_lambda"] != ARM_LAMBDAS[arm]:
        raise RuntimeError("Training result does not match requested M10.8 arm")
    boundary = float(training["boundary_hours"])
    slug = _boundary_slug(boundary)
    arm_directory = arm_slug(arm)
    result_root = RESULT_ROOT / "arms" / arm_directory
    checkpoint = REPOSITORY_ROOT / training["immutable_checkpoint"]["directory"]
    corpus = CORPUS_ROOT / GATE_CORPUS_FILENAME
    deterministic = evaluate_stage1_checkpoint(
        checkpoint,
        corpus,
        deterministic=True,
        evaluation_workers=int(args.evaluation_workers),
    )
    stochastic = evaluate_stage1_checkpoint(
        checkpoint,
        corpus,
        deterministic=False,
        evaluation_workers=int(args.evaluation_workers),
    )
    deterministic_path = result_root / f"evaluation_{slug}_deterministic.json"
    stochastic_path = result_root / f"evaluation_{slug}_stochastic.json"
    write_json_atomic(deterministic_path, deterministic)
    write_json_atomic(stochastic_path, stochastic)
    det_compact = compact_capability(deterministic)
    sto_compact = compact_capability(stochastic)
    gap = capability_gap(deterministic, stochastic)

    loaded = load_checkpoint(checkpoint, device=args.device)
    observations, _, _ = build_observation_corpus(
        samples_per_category=32, seed_base=2026107100
    )
    frozen_observation_buttons = button_policy_diagnostics(
        loaded["actor"],
        observations,
        device=args.device,
        stochastic_draws_per_state=64,
    )
    preflight = _read(RESULT_ROOT / "preflight.json")
    history = [_initial_row(preflight)]
    if args.previous_boundary is not None:
        previous = _read(args.previous_boundary)
        history = previous["learning_curve"]
    history.append(
        {
            "boundary_hours": boundary,
            "deterministic": det_compact,
            "stochastic": sto_compact,
        }
    )
    deterministic_curve = [
        float(row["deterministic"]["first_contact_success"]) for row in history
    ]
    stochastic_curve = [
        float(row["stochastic"]["first_contact_success"]) for row in history
    ]
    monotonicity = {
        "deterministic_first_contact_non_decreasing": all(
            right >= left
            for left, right in zip(deterministic_curve, deterministic_curve[1:])
        ),
        "stochastic_first_contact_non_decreasing": all(
            right >= left
            for left, right in zip(stochastic_curve, stochastic_curve[1:])
        ),
        "deterministic_gain_from_initialization": deterministic_curve[-1]
        - deterministic_curve[0],
        "stochastic_gain_from_initialization": stochastic_curve[-1]
        - stochastic_curve[0],
    }
    decision = (
        "stop_at_plus_1h_for_controlled_gae_comparison"
        if boundary == 1.0
        else "continue_same_arm_to_plus_1h"
    )
    result = {
        "schema_version": 1,
        "boundary_result_version": "RivalM10_8GAEArmBoundaryResultV1",
        "arm": arm,
        "gae_lambda": ARM_LAMBDAS[arm],
        "boundary_hours": boundary,
        "checkpoint": training["immutable_checkpoint"],
        "training_result": args.training_result.resolve()
        .relative_to(REPOSITORY_ROOT)
        .as_posix(),
        "evaluation_reports": {
            "deterministic": deterministic_path.relative_to(
                REPOSITORY_ROOT
            ).as_posix(),
            "stochastic": stochastic_path.relative_to(REPOSITORY_ROOT).as_posix(),
        },
        "deterministic_capability": det_compact,
        "stochastic_capability": sto_compact,
        "stochastic_vs_deterministic_capability_gap": gap,
        "frozen_observation_button_diagnostics": frozen_observation_buttons,
        "training_diagnostics": {
            "iterations": training["iterations"],
            "aggregate_credit_assignment": training[
                "aggregate_credit_assignment"
            ],
        },
        "learning_curve": history,
        "monotonicity": monotonicity,
        "decision": decision,
        "next_authorized_boundary_hours": None if boundary == 1.0 else 1.0,
        "stage_2_authorized": False,
        "production_promotion_authorized": False,
        "checks": {
            "deterministic_500_episode_evaluation_passed": deterministic["checks"][
                "passed"
            ],
            "stochastic_500_episode_evaluation_passed": stochastic["checks"][
                "passed"
            ],
            "same_checkpoint_and_corpus": deterministic["checkpoint"]
            == stochastic["checkpoint"]
            and deterministic["corpus"] == stochastic["corpus"],
            "training_boundary_passed": training["checks"]["passed"],
            "credit_diagnostics_passed": training["aggregate_credit_assignment"][
                "checks"
            ]["passed"],
            "stage_2_authorized": False,
            "production_promotion_authorized": False,
        },
    }
    result["checks"]["passed"] = all(
        value
        for key, value in result["checks"].items()
        if key not in {"stage_2_authorized", "production_promotion_authorized"}
    )
    output = args.output or result_root / f"boundary_{slug}.json"
    write_json_atomic(output, result)
    write_json_atomic(
        result_root / "campaign_state.json",
        {
            "format": "rival-m10-8-arm-state-v1",
            "arm": arm,
            "lambda": ARM_LAMBDAS[arm],
            "phase": "complete" if boundary == 1.0 else "ready_next_boundary",
            "boundary_hours": boundary,
            "latest_clean_recovery_checkpoint": training["immutable_checkpoint"],
            "decision": decision,
            "next_authorized_boundary_hours": None if boundary == 1.0 else 1.0,
            "stage_2_authorized": False,
            "production_promotion_authorized": False,
        },
    )
    if not result["checks"]["passed"]:
        raise RuntimeError(f"M10.8 arm {arm} evaluation failed: {result['checks']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARM_LAMBDAS), required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--previous-boundary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--evaluation-workers", type=int, default=24)
    args = parser.parse_args()
    report = finalize(args)
    print(
        json.dumps(
            {
                "arm": report["arm"],
                "boundary_hours": report["boundary_hours"],
                "decision": report["decision"],
                "deterministic_first_contact_success": report[
                    "deterministic_capability"
                ]["first_contact_success"],
                "stochastic_first_contact_success": report[
                    "stochastic_capability"
                ]["first_contact_success"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
