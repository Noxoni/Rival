"""Build the side-by-side M10.8 GAE comparison after all six boundaries exist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import psutil


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import write_json_atomic  # noqa: E402
from rival_training.v10_7_checkpoint import (  # noqa: E402
    verify_checkpoint,
    verify_reload_parity,
)
from rival_training.v10_8_campaign import (  # noqa: E402
    ARM_LAMBDAS,
    RESULT_ROOT,
    arm_slug,
    load_arm_config,
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_credit(boundary: dict[str, Any], cohort: str, window: str) -> float | None:
    return boundary["training_diagnostics"]["aggregate_credit_assignment"][cohort][
        window
    ]["metrics"]["raw_advantage"]["mean"]


def _training_processes() -> list[dict[str, Any]]:
    needles = ("run_m10_8_preflight.py", "run_m10_8_stage1_boundary.py")
    current = psutil.Process(os.getpid())
    excluded = {current.pid, *(parent.pid for parent in current.parents())}
    rows = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if process.pid not in excluded and any(needle in command for needle in needles):
            rows.append({"pid": int(process.pid), "command": command})
    return rows


def _arm_report(arm: str) -> dict[str, Any]:
    root = RESULT_ROOT / "arms" / arm_slug(arm)
    half = _read(root / "boundary_plus-000p5h.json")
    final = _read(root / "boundary_plus-001h.json")
    deterministic_curve = [
        float(row["deterministic"]["first_contact_success"])
        for row in final["learning_curve"]
    ]
    stochastic_curve = [
        float(row["stochastic"]["first_contact_success"])
        for row in final["learning_curve"]
    ]
    det = final["deterministic_capability"]
    sto = final["stochastic_capability"]
    training = final["training_diagnostics"]["iterations"]
    last = training[-1]
    checkpoint_path = REPOSITORY_ROOT / final["checkpoint"]["directory"]
    config = load_arm_config(arm)
    verify_checkpoint(checkpoint_path, expected_config=config)
    reload = verify_reload_parity(
        checkpoint_path, expected_config=config, device="cpu"
    )
    return {
        "arm": arm,
        "lambda": ARM_LAMBDAS[arm],
        "checkpoint": final["checkpoint"],
        "checkpoint_reload_parity": reload,
        "initialization": final["learning_curve"][0],
        "plus_0p5h": {
            "deterministic": half["deterministic_capability"],
            "stochastic": half["stochastic_capability"],
            "capability_gap": half[
                "stochastic_vs_deterministic_capability_gap"
            ],
        },
        "plus_1h": {
            "deterministic": det,
            "stochastic": sto,
            "capability_gap": final[
                "stochastic_vs_deterministic_capability_gap"
            ],
        },
        "monotonicity": {
            "deterministic_curve": deterministic_curve,
            "stochastic_curve": stochastic_curve,
            "deterministic_non_decreasing": all(
                right >= left
                for left, right in zip(
                    deterministic_curve, deterministic_curve[1:]
                )
            ),
            "stochastic_non_decreasing": all(
                right >= left
                for left, right in zip(stochastic_curve, stochastic_curve[1:])
            ),
        },
        "credit_assignment_at_plus_1h": {
            window: {
                "successful_first_contact_raw_advantage": _mean_credit(
                    final, "successful_first_contact", window
                ),
                "failed_timeout_like_raw_advantage": _mean_credit(
                    final, "failed_timeout_like", window
                ),
            }
            for window in final["training_diagnostics"][
                "aggregate_credit_assignment"
            ]["successful_first_contact"]
        },
        "final_training_health": {
            "advantage": last["gae"]["advantage"],
            "normalized_advantage": last["gae"]["normalized_advantage"],
            "returns": last["gae"]["return"],
            "critic_explained_variance": last["ppo"][
                "explained_variance_before_update"
            ],
            "critic_loss": last["ppo"]["critic_loss"],
            "actor_loss": last["ppo"]["actor_loss"],
            "approximate_kl": last["ppo"]["approximate_kl"],
            "clip_fraction": last["ppo"]["clip_fraction"],
            "actor_update_magnitude": last["ppo"]["actor_update_magnitude"],
            "critic_update_magnitude": last["ppo"]["critic_update_magnitude"],
            "analog_log_std": last["ppo"]["analog_log_std"],
            "analog_std": last["ppo"]["analog_std"],
            "controller_branch_gradients": last["ppo"][
                "controller_branch_gradient_absolute_sums"
            ],
        },
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    preflight = _read(RESULT_ROOT / "preflight.json")
    arms = {arm: _arm_report(arm) for arm in ARM_LAMBDAS}
    ranked = sorted(
        ARM_LAMBDAS,
        key=lambda arm: (
            arms[arm]["plus_1h"]["deterministic"]["first_contact_success"],
            arms[arm]["plus_1h"]["deterministic"]["all_three_success"],
            arms[arm]["plus_1h"]["deterministic"]["second_contact_success"],
            -arms[arm]["plus_1h"]["deterministic"]["failed_trajectory"][
                "distance_change_terminal_minus_initial_mean"
            ],
        ),
        reverse=True,
    )
    best = ranked[0]
    control_final = arms["A"]["plus_1h"]["deterministic"][
        "first_contact_success"
    ]
    best_final = arms[best]["plus_1h"]["deterministic"]["first_contact_success"]
    longer_clear_gain = best in {"B", "C"} and best_final - control_final >= 0.03
    longer_monotonic = best in {"B", "C"} and arms[best]["monotonicity"][
        "deterministic_non_decreasing"
    ]
    credit_horizon_supported = longer_clear_gain and longer_monotonic
    if credit_horizon_supported:
        conclusion = "supported_as_material_limitation"
        carry_forward = best
    else:
        conclusion = "rejected_as_primary_remaining_failure"
        carry_forward = None
    if (
        arms["B"]["plus_1h"]["deterministic"]["first_contact_success"]
        > arms["A"]["plus_1h"]["deterministic"]["first_contact_success"]
        and arms["C"]["plus_1h"]["deterministic"]["first_contact_success"]
        < arms["B"]["plus_1h"]["deterministic"]["first_contact_success"]
        and credit_horizon_supported
    ):
        carry_forward = "B"

    processes = _training_processes()
    checks = {
        "preflight_passed": preflight["checks"]["passed"],
        "all_six_boundaries_passed": all(
            _read(
                RESULT_ROOT
                / "arms"
                / arm_slug(arm)
                / f"boundary_{slug}.json"
            )["checks"]["passed"]
            for arm in ARM_LAMBDAS
            for slug in ("plus-000p5h", "plus-001h")
        ),
        "all_final_reload_parity_exact": all(
            row["checkpoint_reload_parity"]["checks"]["passed"]
            for row in arms.values()
        ),
        "all_arms_stopped_at_one_hour": all(
            row["checkpoint"]["simulated_game_hours"] >= 1.0
            and row["checkpoint"]["simulated_game_hours"] <= 1.0003
            for row in arms.values()
        ),
        "no_training_processes_remain": not processes,
        "stage_2_authorized": False,
        "production_promotion_authorized": False,
    }
    checks["passed"] = all(
        value
        for key, value in checks.items()
        if key not in {"stage_2_authorized", "production_promotion_authorized"}
    )
    report = {
        "schema_version": 1,
        "closeout_version": "RivalM10_8GAECreditAssignmentComparisonV1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "gae_physical_time_horizons": preflight["gae_physical_time_proofs"],
        "comparison_priority": [
            "deterministic_first_contact_capability",
            "reacquisition_capability",
            "monotonic_improvement",
            "failed_trajectories_move_closer",
            "stable_throttle_and_steering",
            "earlier_reward_credit",
        ],
        "arms": arms,
        "capability_ranking": ranked,
        "decision_rule_evidence": {
            "best_arm_by_capability_priority": best,
            "best_minus_control_deterministic_first_contact": best_final
            - control_final,
            "clear_gain_threshold": 0.03,
            "best_longer_arm_monotonic": longer_monotonic,
        },
        "credit_horizon_conclusion": conclusion,
        "lambda_to_carry_forward": (
            None if carry_forward is None else ARM_LAMBDAS[carry_forward]
        ),
        "arm_to_carry_forward": carry_forward,
        "next_investigation_if_rejected": [
            "critic_value_accuracy",
            "advantage_normalization",
            "ppo_update_stability",
            "actor_learning_rate_and_update_magnitude",
            "reward_component_credit_attribution",
        ],
        "stage_2_authorized": False,
        "production_promotion_authorized": False,
        "remaining_processes": processes,
        "checks": checks,
    }
    write_json_atomic(args.output, report)
    if not checks["passed"]:
        raise RuntimeError(f"M10.8 comparative closeout failed: {checks}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "final_comparison.json"
    )
    args = parser.parse_args()
    report = finalize(args)
    print(
        json.dumps(
            {
                "status": "passed",
                "capability_ranking": report["capability_ranking"],
                "credit_horizon_conclusion": report[
                    "credit_horizon_conclusion"
                ],
                "arm_to_carry_forward": report["arm_to_carry_forward"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
