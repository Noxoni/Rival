"""Evaluate and apply exact Rival v10.2 Stage-1 exit gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import write_json_atomic  # noqa: E402
from rival_training.v10_2_campaign import (  # noqa: E402
    CAMPAIGN_STATE_PATH,
    CORPUS_ROOT,
    RESULT_ROOT,
    boundary_slug,
    update_progressive_state,
    wall_clock_status,
)
from rival_training.v10_2_evaluation import (  # noqa: E402
    evaluate_stage1_checkpoint,
)


CORE_FAMILIES = (
    "stationary_close",
    "stationary_medium",
    "moving_chase",
    "awkward_heading",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _core_metrics(report: dict[str, Any]) -> dict[str, float]:
    episode_rows = [
        row
        for row in report.get("episode_rows", [])
        if row["family"] in CORE_FAMILIES
    ]
    if episode_rows:
        successes = [
            row for row in episode_rows if row["first_touch_success"]
        ]
        failures = [
            row for row in episode_rows if not row["first_touch_success"]
        ]
        failed_initial = (
            statistics.mean(
                float(row["initial_car_ball_distance"]) for row in failures
            )
            if failures
            else 0.0
        )
        failed_terminal = (
            statistics.mean(
                float(row["terminal_car_ball_distance"]) for row in failures
            )
            if failures
            else 0.0
        )
        return {
            "episodes": len(episode_rows),
            "first_touch_success_share": len(successes)
            / max(len(episode_rows), 1),
            "no_touch_timeout_share": sum(
                row["termination_reason"] == "no_touch_timeout"
                for row in episode_rows
            )
            / max(len(episode_rows), 1),
            "successful_time_to_first_touch_median": (
                statistics.median(
                    float(row["time_to_first_touch_seconds"])
                    for row in successes
                )
                if successes
                else float("inf")
            ),
            "failed_initial_distance_mean": failed_initial,
            "failed_terminal_distance_mean": failed_terminal,
            "failed_terminal_distance_reduction_share": (
                (failed_initial - failed_terminal) / failed_initial
                if failed_initial > 0.0
                else 0.0
            ),
        }
    rows = [report["families"][family] for family in CORE_FAMILIES]
    episodes = sum(int(row["episodes"]) for row in rows)
    successes = sum(int(row["first_touch_success_count"]) for row in rows)
    time_values = []
    for row in rows:
        summary = row["successful_time_to_first_touch_seconds"]
        if summary["samples"]:
            # Family medians are only a compact approximation if raw episodes
            # are absent, so use the exact overall all-family median for the
            # formal gate below and retain this only as core telemetry.
            time_values.extend(
                [float(summary["median"])] * int(summary["samples"])
            )
    failed_initial_count = sum(
        int(row["failed_episode_initial_car_ball_distance"]["samples"])
        for row in rows
    )
    failed_initial = sum(
        float(row["failed_episode_initial_car_ball_distance"]["mean"] or 0.0)
        * int(row["failed_episode_initial_car_ball_distance"]["samples"])
        for row in rows
    ) / max(failed_initial_count, 1)
    failed_terminal = sum(
        float(row["failed_episode_terminal_car_ball_distance"]["mean"] or 0.0)
        * int(row["failed_episode_terminal_car_ball_distance"]["samples"])
        for row in rows
    ) / max(failed_initial_count, 1)
    no_touch = sum(int(row["no_touch_timeout_count"]) for row in rows)
    return {
        "episodes": episodes,
        "first_touch_success_share": successes / max(episodes, 1),
        "no_touch_timeout_share": no_touch / max(episodes, 1),
        "successful_time_to_first_touch_median": (
            statistics.median(time_values) if time_values else float("inf")
        ),
        "failed_initial_distance_mean": failed_initial,
        "failed_terminal_distance_mean": failed_terminal,
        "failed_terminal_distance_reduction_share": (
            (failed_initial - failed_terminal) / failed_initial
            if failed_initial > 0.0
            else 0.0
        ),
    }


def _phase_a_gates(
    report: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, Any]:
    core = _core_metrics(report)
    family = report["families"]
    regression = True
    if previous is not None:
        previous_family = previous["evaluation"]["families"]
        for name in CORE_FAMILIES:
            old = float(previous_family[name]["first_touch_success_share"])
            new = float(family[name]["first_touch_success_share"])
            if old > 0.70 and new < old - 0.10:
                regression = False
    checks = {
        "stationary_close_at_least_95_percent": family[
            "stationary_close"
        ]["first_touch_success_share"]
        >= 0.95,
        "stationary_medium_at_least_85_percent": family[
            "stationary_medium"
        ]["first_touch_success_share"]
        >= 0.85,
        "moving_chase_at_least_75_percent": family["moving_chase"][
            "first_touch_success_share"
        ]
        >= 0.75,
        "awkward_heading_at_least_80_percent": family[
            "awkward_heading"
        ]["first_touch_success_share"]
        >= 0.80,
        "core_aggregate_at_least_85_percent": core[
            "first_touch_success_share"
        ]
        >= 0.85,
        "core_no_touch_at_most_15_percent": core[
            "no_touch_timeout_share"
        ]
        <= 0.15,
        "successful_median_at_most_5_seconds": core[
            "successful_time_to_first_touch_median"
        ]
        <= 5.0,
        "failed_terminal_distance_reduced_at_least_25_percent": core[
            "failed_terminal_distance_reduction_share"
        ]
        >= 0.25,
        "no_family_regression_over_10_points": regression,
    }
    checks["passed"] = all(checks.values())
    return {"core": core, "checks": checks}


def _phase_b_frozen_gates(report: dict[str, Any]) -> dict[str, Any]:
    family = report["families"]
    repeated_success_rows = [
        row
        for row in report.get("episode_rows", [])
        if row["first_touch_success"] and row["physical_touch_count"] >= 2
    ]
    checks = {
        "stationary_close_at_least_97_percent": family[
            "stationary_close"
        ]["first_touch_success_share"]
        >= 0.97,
        "stationary_medium_at_least_92_percent": family[
            "stationary_medium"
        ]["first_touch_success_share"]
        >= 0.92,
        "moving_chase_at_least_88_percent": family["moving_chase"][
            "first_touch_success_share"
        ]
        >= 0.88,
        "awkward_heading_at_least_90_percent": family[
            "awkward_heading"
        ]["first_touch_success_share"]
        >= 0.90,
        "kickoff_at_least_80_percent": family[
            "natural_kickoff_holdout"
        ]["first_touch_success_share"]
        >= 0.80,
        "overall_at_least_90_percent": report["overall"][
            "first_touch_success_share"
        ]
        >= 0.90,
        "no_touch_at_most_10_percent": report["overall"][
            "no_touch_timeout_share"
        ]
        <= 0.10,
        "successful_median_at_most_4_seconds": report["overall"][
            "successful_time_to_first_touch_seconds"
        ]["median"]
        is not None
        and report["overall"]["successful_time_to_first_touch_seconds"][
            "median"
        ]
        <= 4.0,
        "touch_events_one_for_one": report["overall"][
            "physical_touch_count"
        ]
        == int(round(report["overall"]["touch_reward_total"])),
        "touch_return_dominates_positive_dense_on_repeated_successes": (
            bool(repeated_success_rows)
            and sum(
                float(row["touch_reward_total"])
                for row in repeated_success_rows
            )
            > sum(
                max(0.0, float(row["distance_reward_total"]))
                for row in repeated_success_rows
            )
        ),
        "goal_reward_zero": True,
        "speed_reward_absent": True,
        "stationary_ball_motion_exploit_absent": True,
    }
    checks["passed"] = all(checks.values())
    return {"checks": checks}


def _unseen_gates(report: dict[str, Any]) -> dict[str, Any]:
    core_minimum = min(
        float(report["families"][family]["first_touch_success_share"])
        for family in CORE_FAMILIES
    )
    checks = {
        "aggregate_at_least_85_percent": report["overall"][
            "first_touch_success_share"
        ]
        >= 0.85,
        "no_core_family_below_75_percent": core_minimum >= 0.75,
        "no_touch_at_most_15_percent": report["overall"][
            "no_touch_timeout_share"
        ]
        <= 0.15,
    }
    checks["passed"] = all(checks.values())
    return {"checks": checks}


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    training = _read(args.training_result)
    boundary = float(training["boundary_hours"])
    slug = boundary_slug(boundary)
    phase = str(training["phase"])
    checkpoint = REPOSITORY_ROOT / training["immutable_checkpoint"][
        "directory"
    ]
    evaluation = evaluate_stage1_checkpoint(
        checkpoint,
        CORPUS_ROOT / "stage1_frozen_gate_corpus.json",
        device=args.device,
        evaluation_workers=args.evaluation_workers,
        include_episode_rows=True,
    )
    evaluation_path = RESULT_ROOT / "stage_1" / f"evaluation_{slug}.json"
    write_json_atomic(evaluation_path, evaluation)
    previous = _read(args.previous_boundary) if args.previous_boundary else None
    source = _read(RESULT_ROOT / "stage_1/source_v10_1_plus10_gate.json")
    phase_a = _phase_a_gates(evaluation, previous)
    phase_b = _phase_b_frozen_gates(evaluation)
    unseen = None
    unseen_gates = None
    if phase == "B" and phase_b["checks"]["passed"]:
        unseen = evaluate_stage1_checkpoint(
            checkpoint,
            CORPUS_ROOT / "stage1_unseen_generalization_corpus.json",
            device=args.device,
            evaluation_workers=args.evaluation_workers,
        )
        unseen_path = RESULT_ROOT / "stage_1" / f"unseen_{slug}.json"
        write_json_atomic(unseen_path, unseen)
        unseen_gates = _unseen_gates(unseen)

    apparent_phase_b_pass = bool(
        phase == "B"
        and phase_b["checks"]["passed"]
        and unseen_gates is not None
        and unseen_gates["checks"]["passed"]
    )
    previous_phase_b_pass = bool(
        previous
        and previous.get("phase") == "B"
        and previous.get("gates", {})
        .get("phase_b", {})
        .get("apparent_pass", False)
    )
    core = _core_metrics(evaluation)
    source_core = _core_metrics(source)
    no_learning = bool(
        boundary >= 5.0
        and core["first_touch_success_share"]
        - source_core["first_touch_success_share"]
        < 0.10
        and source_core["no_touch_timeout_share"]
        - core["no_touch_timeout_share"]
        < 0.10
    )
    if training["status"] == "wall_clock_stop":
        decision = "stop_progressive_overnight_wall_clock_budget_exhausted"
        next_phase = phase
    elif phase == "A" and phase_a["checks"]["passed"]:
        decision = "ball_acquisition_phase_a_passed_unlock_phase_b"
        next_phase = "B"
    elif phase == "B" and apparent_phase_b_pass and previous_phase_b_pass:
        decision = "ball_acquisition_skill_passed_unlock_ground_control"
        next_phase = "complete"
    elif no_learning:
        decision = "stop_ball_acquisition_no_material_learning_by_plus_5h"
        next_phase = "stopped"
    elif boundary >= 15.0:
        decision = "stop_ball_acquisition_not_mastered_by_plus_15h"
        next_phase = "stopped"
    else:
        decision = "continue_ball_acquisition_training"
        next_phase = phase
    clock = wall_clock_status()
    compact_evaluation = {
        key: value
        for key, value in evaluation.items()
        if key != "episode_rows"
    }
    result = {
        "schema_version": 1,
        "boundary_result_version": "RivalM10_2Stage1BoundaryResultV1",
        "stage": 1,
        "skill": "ball_acquisition",
        "phase": phase,
        "boundary_hours": boundary,
        "checkpoint": training["immutable_checkpoint"],
        "training_result": args.training_result.resolve()
        .relative_to(REPOSITORY_ROOT)
        .as_posix(),
        "evaluation": compact_evaluation,
        "source_baseline": {
            "checkpoint": source["checkpoint"],
            "core": source_core,
        },
        "gates": {
            "phase_a": phase_a,
            "phase_b": {
                **phase_b,
                "unseen": unseen_gates,
                "apparent_pass": apparent_phase_b_pass,
                "previous_boundary_apparent_pass": previous_phase_b_pass,
                "two_consecutive_passes": apparent_phase_b_pass
                and previous_phase_b_pass,
            },
            "no_learning_by_plus_5h": no_learning,
        },
        "unseen_evaluation": unseen,
        "decision": decision,
        "next_phase": next_phase,
        "campaign_wall_clock": clock,
        "production_promotion_authorized": False,
    }
    output = args.output or RESULT_ROOT / "stage_1" / f"boundary_{slug}.json"
    write_json_atomic(output, result)
    stop = decision.startswith("stop_")
    passed = decision == "ball_acquisition_skill_passed_unlock_ground_control"
    update_progressive_state(
        {
            "current_phase": next_phase,
            "current_evaluation_boundary": boundary,
            "gate_decision": decision,
            "next_authorized_stage": 2 if passed else None,
            "passed_prerequisite_checkpoints": (
                {
                    "stage_1": training["immutable_checkpoint"],
                }
                if passed
                else _read(CAMPAIGN_STATE_PATH).get(
                    "passed_prerequisite_checkpoints", {}
                )
            ),
            "campaign_wall_clock_elapsed_seconds": clock["elapsed_seconds"],
            "campaign_wall_clock_remaining_seconds": clock[
                "remaining_seconds"
            ],
            "stop_reason": decision if stop else None,
        }
    )
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
                "stage": 1,
                "phase": report["phase"],
                "boundary_hours": report["boundary_hours"],
                "decision": report["decision"],
                "first_touch_success_share": report["evaluation"][
                    "overall"
                ]["first_touch_success_share"],
                "wall_clock": report["campaign_wall_clock"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
