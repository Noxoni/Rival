"""Evaluate deterministic/stochastic M10.7 capability at one boundary."""

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
from rival_training.v10_7_campaign import (  # noqa: E402
    CORPUS_ROOT,
    GATE_CORPUS_FILENAME,
    RESULT_ROOT,
)
from rival_training.v10_7_checkpoint import load_checkpoint  # noqa: E402
from rival_training.v10_7_diagnostics import (  # noqa: E402
    build_observation_corpus,
    button_policy_diagnostics,
)
from rival_training.v10_7_evaluation import (  # noqa: E402
    capability_gap,
    evaluate_stage1_checkpoint,
)


CAMPAIGN_STATE = RESULT_ROOT / "stage1_campaign_state.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _slug(hours: float) -> str:
    return {0.5: "plus-000p5h", 1.0: "plus-001h", 2.5: "plus-002p5h"}[
        float(hours)
    ]


def _compact_capability(report: dict[str, Any]) -> dict[str, Any]:
    overall = report["overall"]
    return {
        "first_contact_success": overall["first_touch_success_share"],
        "second_contact_success": overall["second_touch_success_share"],
        "third_contact_success": overall["third_touch_success_share"],
        "all_three_success": overall["all_three_contacts_success_share"],
        "no_touch_rate": overall["no_touch_timeout_share"],
        "time_to_first_contact": overall["successful_time_to_first_touch_seconds"],
        "time_first_to_second_contact": overall[
            "successful_time_first_to_second_touch_seconds"
        ],
        "time_second_to_third_contact": overall[
            "successful_time_second_to_third_touch_seconds"
        ],
        "failed_initial_distance": overall[
            "failed_episode_initial_car_ball_distance"
        ],
        "failed_terminal_distance": overall[
            "failed_episode_terminal_car_ball_distance"
        ],
        "failed_initial_alignment": overall[
            "failed_episode_initial_car_ball_alignment"
        ],
        "failed_terminal_alignment": overall[
            "failed_episode_terminal_car_ball_alignment"
        ],
        "cumulative_heading_reward": overall["heading_reward_total"],
        "cumulative_distance_reward": overall["distance_reward_total"],
        "cumulative_acquisition_time_penalty": overall[
            "cumulative_acquisition_time_penalty"
        ],
        "action_diagnostics": overall["action_diagnostics"],
        "button_policy_diagnostics": overall["button_policy_diagnostics"],
        "mean_button_entropy": overall["mean_button_entropy"],
    }


def _history(previous: Path | None) -> list[dict[str, Any]]:
    source = _read(RESULT_ROOT / "preflight.json")["source_transfer_capability"]
    source_row = {
        "boundary_hours": 0.0,
        "deterministic_first_contact_success": source[
            "deterministic_overall"
        ]["first_touch_success_share"],
        "stochastic_first_contact_success": source["stochastic_overall"][
            "first_touch_success_share"
        ],
    }
    if previous is None:
        return [source_row]
    rows = list(_read(previous)["learning_curve"])
    if not rows or float(rows[0]["boundary_hours"]) != 0.0:
        rows.insert(0, source_row)
    return rows


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    training = _read(args.training_result)
    boundary = float(training["boundary_hours"])
    slug = _slug(boundary)
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
    deterministic_path = RESULT_ROOT / "stage_1" / f"evaluation_{slug}_deterministic.json"
    stochastic_path = RESULT_ROOT / "stage_1" / f"evaluation_{slug}_stochastic.json"
    write_json_atomic(deterministic_path, deterministic)
    write_json_atomic(stochastic_path, stochastic)
    gap = capability_gap(deterministic, stochastic)

    loaded = load_checkpoint(checkpoint, device=args.device)
    observations, _, _ = build_observation_corpus(
        samples_per_category=32, seed_base=2026107100
    )
    corpus_buttons = button_policy_diagnostics(
        loaded["actor"],
        observations,
        device=args.device,
        stochastic_draws_per_state=64,
    )
    history = _history(args.previous_boundary)
    history.append(
        {
            "boundary_hours": boundary,
            "deterministic_first_contact_success": deterministic["overall"][
                "first_touch_success_share"
            ],
            "stochastic_first_contact_success": stochastic["overall"][
                "first_touch_success_share"
            ],
        }
    )
    deterministic_curve = [
        float(row["deterministic_first_contact_success"]) for row in history
    ]
    stochastic_curve = [
        float(row["stochastic_first_contact_success"]) for row in history
    ]
    monotonic = {
        "deterministic_first_contact_non_decreasing": all(
            right >= left for left, right in zip(deterministic_curve, deterministic_curve[1:])
        ),
        "stochastic_first_contact_non_decreasing": all(
            right >= left for left, right in zip(stochastic_curve, stochastic_curve[1:])
        ),
        "deterministic_first_contact_gain_from_source": deterministic_curve[-1]
        - deterministic_curve[0],
        "stochastic_first_contact_gain_from_source": stochastic_curve[-1]
        - stochastic_curve[0],
    }
    monotonic["clearly_and_monotonically_improving"] = (
        monotonic["deterministic_first_contact_non_decreasing"]
        and monotonic["stochastic_first_contact_non_decreasing"]
        and monotonic["deterministic_first_contact_gain_from_source"] >= 0.05
        and not gap["stochastic_materially_better_first_touch_by_5_points"]
    )
    if boundary >= 2.5:
        decision = "stop_at_plus_2p5h_for_action_policy_evidence_review"
        next_boundary = None
    elif gap["stochastic_materially_better_first_touch_by_5_points"]:
        decision = "continue_to_next_boundary_with_action_policy_gap_flag"
        next_boundary = 1.0 if boundary == 0.5 else 2.5
    else:
        decision = "continue_to_next_authorized_stage1_boundary"
        next_boundary = 1.0 if boundary == 0.5 else 2.5
    result = {
        "schema_version": 1,
        "boundary_result_version": "RivalM10_7Stage1BoundaryResultV1",
        "stage": 1,
        "boundary_hours": boundary,
        "checkpoint": training["immutable_checkpoint"],
        "training_result": args.training_result.resolve()
        .relative_to(REPOSITORY_ROOT)
        .as_posix(),
        "evaluation_reports": {
            "deterministic": deterministic_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "stochastic": stochastic_path.relative_to(REPOSITORY_ROOT).as_posix(),
        },
        "deterministic_capability": _compact_capability(deterministic),
        "stochastic_capability": _compact_capability(stochastic),
        "stochastic_vs_deterministic_capability_gap": gap,
        "frozen_observation_button_diagnostics": corpus_buttons,
        "learning_curve": history,
        "monotonicity": monotonic,
        "decision": decision,
        "next_authorized_boundary_hours": next_boundary,
        "stage_2_authorized": False,
        "production_promotion_authorized": False,
        "checks": {
            "deterministic_500_episode_evaluation_passed": deterministic["checks"][
                "passed"
            ],
            "stochastic_500_episode_evaluation_passed": stochastic["checks"][
                "passed"
            ],
            "same_checkpoint_and_corpus": (
                deterministic["checkpoint"] == stochastic["checkpoint"]
                and deterministic["corpus"] == stochastic["corpus"]
            ),
            "button_diagnostics_all_required_fields_present": all(
                all(
                    key in corpus_buttons["buttons"][name]
                    for key in (
                        "mean_base_probability",
                        "mean_effective_probability",
                        "mean_entropy",
                        "deterministic_on_share",
                        "stochastic_sampled_on_share",
                        "deterministic_stochastic_disagreement_share",
                    )
                )
                for name in ("jump", "boost", "handbrake")
            ),
            "stage_2_authorized": False,
            "production_promotion_authorized": False,
        },
    }
    result["checks"]["passed"] = (
        all(
            value
            for key, value in result["checks"].items()
            if key not in {"stage_2_authorized", "production_promotion_authorized"}
        )
        and result["checks"]["stage_2_authorized"] is False
        and result["checks"]["production_promotion_authorized"] is False
    )
    output = args.output or RESULT_ROOT / "stage_1" / f"boundary_{slug}.json"
    write_json_atomic(output, result)
    write_json_atomic(
        CAMPAIGN_STATE,
        {
            "format": "rival-m10-7-stage1-state-v1",
            "campaign_id": "rival-v10-7-stage1-action-policy-correction",
            "current_phase": "complete" if boundary >= 2.5 else "ready_next_boundary",
            "current_boundary_hours": boundary,
            "latest_clean_recovery_checkpoint": training["immutable_checkpoint"],
            "gate_decision": decision,
            "next_authorized_boundary_hours": next_boundary,
            "stage_2_authorized": False,
            "production_promotion_authorized": False,
        },
    )
    if not result["checks"]["passed"]:
        raise RuntimeError(f"M10.7 boundary evaluation failed: {result['checks']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
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
                "boundary_hours": report["boundary_hours"],
                "decision": report["decision"],
                "deterministic_first_contact_success": report[
                    "deterministic_capability"
                ]["first_contact_success"],
                "stochastic_first_contact_success": report["stochastic_capability"][
                    "first_contact_success"
                ],
                "capability_gap": report[
                    "stochastic_vs_deterministic_capability_gap"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
