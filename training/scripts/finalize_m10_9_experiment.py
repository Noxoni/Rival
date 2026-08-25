"""Build the immutable M10.9 PPO V2 comparative closeout."""

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
from rival_training.v10_9_campaign import (  # noqa: E402
    BOUNDARY_HOURS,
    RESULT_ROOT,
    load_stage1_config,
)


M10_8_FINAL = (
    REPOSITORY_ROOT / "training/results/milestone10_8/final_comparison.json"
)
BOUNDARY_SLUGS = {
    0.25: "plus-000p25h",
    0.5: "plus-000p5h",
    1.0: "plus-001h",
}


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training_processes() -> list[dict[str, Any]]:
    needles = ("run_m10_9_preflight.py", "run_m10_9_stage1_boundary.py")
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


def _capability(overall: dict[str, Any]) -> dict[str, Any]:
    initial_distance = overall["failed_episode_initial_car_ball_distance"]
    terminal_distance = overall["failed_episode_terminal_car_ball_distance"]
    initial_alignment = overall["failed_episode_initial_car_ball_alignment"]
    terminal_alignment = overall["failed_episode_terminal_car_ball_alignment"]
    return {
        "episodes": int(overall["episodes"]),
        "first_contact_success": float(overall["first_touch_success_share"]),
        "second_contact_success": float(overall["second_touch_success_share"]),
        "third_contact_success": float(overall["third_touch_success_share"]),
        "all_three_success": float(overall["all_three_contacts_success_share"]),
        "no_touch_rate": float(overall["no_touch_timeout_share"]),
        "failed_trajectory": {
            "initial_distance_mean": float(initial_distance["mean"]),
            "terminal_distance_mean": float(terminal_distance["mean"]),
            "distance_change_terminal_minus_initial_mean": float(
                terminal_distance["mean"] - initial_distance["mean"]
            ),
            "initial_alignment_mean": float(initial_alignment["mean"]),
            "terminal_alignment_mean": float(terminal_alignment["mean"]),
            "alignment_change_terminal_minus_initial_mean": float(
                terminal_alignment["mean"] - initial_alignment["mean"]
            ),
        },
        "timing_seconds": {
            "reset_to_first": overall["successful_time_to_first_touch_seconds"],
            "first_to_second": overall[
                "successful_time_first_to_second_touch_seconds"
            ],
            "second_to_third": overall[
                "successful_time_second_to_third_touch_seconds"
            ],
        },
        "controller": overall["action_diagnostics"],
        "reward_components": {
            "touch": float(overall["touch_reward_total"]),
            "heading": float(overall["heading_reward_total"]),
            "distance": float(overall["distance_reward_total"]),
            "acquisition_time": float(
                overall["cumulative_acquisition_time_penalty"]
            ),
        },
        "analog_exploration": overall["analog_exploration_diagnostics"],
        "button_policy": overall["button_policy_diagnostics"],
    }


def _iteration_health(iteration: dict[str, Any]) -> dict[str, Any]:
    actor = iteration["ppo"]["actor"]
    critic = iteration["ppo"]["critic"]
    replay = iteration["rollout_log_probability_reproduction"]
    return {
        "iteration": int(iteration["iteration"]),
        "cumulative_agent_steps": int(iteration["cumulative_agent_steps"]),
        "raw_advantage": iteration["gae"]["raw_advantage"],
        "scaled_advantage": iteration["gae"]["scaled_advantage"],
        "raw_scaled_sign_agreement": bool(
            iteration["gae"]["raw_scaled_sign_agreement"]
        ),
        "actor": {
            "epochs_authorized": int(actor["epochs_authorized"]),
            "epochs_executed": int(actor["epochs_executed"]),
            "optimizer_steps": int(actor["optimizer_steps"]),
            "kl_stopped_early": bool(actor["kl_stopped_early"]),
            "approximate_kl": actor["approximate_kl"],
            "clip_fraction": actor["clip_fraction"],
            "actor_loss": actor["actor_loss"],
            "update_magnitude": float(actor["actor_update_magnitude"]),
            "controller_branch_gradients": actor[
                "controller_branch_gradient_absolute_sums"
            ],
        },
        "critic": {
            "epochs_executed": int(critic["epochs_executed"]),
            "optimizer_steps": int(critic["optimizer_steps"]),
            "held_out_ev_before": float(
                critic["before"]["held_out"]["explained_variance"]
            ),
            "held_out_ev_after": float(
                critic["after"]["held_out"]["explained_variance"]
            ),
            "held_out_ev_improvement": float(critic["held_out_ev_improvement"]),
            "held_out_loss_before": float(critic["before"]["held_out"]["loss"]),
            "held_out_loss_after": float(critic["after"]["held_out"]["loss"]),
            "held_out_loss_improvement": float(
                critic["held_out_loss_improvement"]
            ),
            "update_magnitude": float(iteration["ppo"]["critic_update_magnitude"]),
        },
        "log_probability_replay": replay,
        "ar_exploration": {
            "rho": float(iteration["exploration"]["rho"]),
            "maximum_analytical_vs_measured_deviation": float(
                iteration["exploration"][
                    "maximum_analytical_vs_measured_deviation"
                ]
            ),
            "epsilon_autocorrelation": iteration["exploration"][
                "epsilon_autocorrelation"
            ],
            "deterministic_minus_tanh_mean_max_abs": float(
                iteration["exploration"][
                    "deterministic_minus_tanh_mean_max_abs"
                ]
            ),
        },
        "analog_log_std": iteration["ppo"]["analog_log_std"],
        "analog_std": iteration["ppo"]["analog_std"],
        "checks": iteration["health"],
    }


def _credit(boundary: dict[str, Any]) -> dict[str, Any]:
    aggregate = boundary["aggregate_credit_assignment"]
    return {
        cohort: {
            window: {
                metric: values["metrics"][metric]
                for metric in (
                    "raw_advantage",
                    "scaled_advantage",
                    "throttle",
                    "steer_magnitude",
                    "heading_improvement",
                    "distance_progress_uu",
                )
            }
            for window, values in aggregate[cohort].items()
        }
        for cohort in ("successful_first_contact", "failed_timeout_like")
    }


def _historical_m10_8_arm_c() -> dict[str, Any]:
    report = _read(M10_8_FINAL)
    arm = report["arms"]["C"]
    return {
        "source": M10_8_FINAL.relative_to(REPOSITORY_ROOT).as_posix(),
        "lambda": float(arm["lambda"]),
        "initial": arm["initialization"],
        "plus_0p5h": arm["plus_0p5h"],
        "plus_1h": arm["plus_1h"],
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    config = load_stage1_config(args.config)
    preflight = _read(RESULT_ROOT / "preflight.json")
    init_det = _read(RESULT_ROOT / "initialization_deterministic.json")
    init_sto = _read(RESULT_ROOT / "initialization_stochastic_ar1.json")
    boundaries = {
        hours: _read(RESULT_ROOT / f"training_{BOUNDARY_SLUGS[hours]}.json")
        for hours in BOUNDARY_HOURS
    }
    curve = {
        "initial": {
            "hours": 0.0,
            "deterministic": _capability(init_det["overall"]),
            "stochastic": _capability(init_sto["overall"]),
        }
    }
    for hours, boundary in boundaries.items():
        curve[BOUNDARY_SLUGS[hours]] = {
            "hours": hours,
            "deterministic": _capability(
                boundary["evaluation"]["deterministic_overall"]
            ),
            "stochastic": _capability(
                boundary["evaluation"]["stochastic_overall"]
            ),
            "stochastic_minus_deterministic": boundary["evaluation"][
                "stochastic_vs_deterministic_gap"
            ],
        }

    final = boundaries[1.0]
    final_checkpoint = REPOSITORY_ROOT / final["immutable_checkpoint"]["directory"]
    verify_checkpoint(final_checkpoint, expected_config=config)
    reload = verify_reload_parity(
        final_checkpoint, expected_config=config, device="cpu"
    )
    processes = _training_processes()
    deterministic_first_curve = [
        curve[key]["deterministic"]["first_contact_success"]
        for key in ("initial", "plus-000p25h", "plus-000p5h", "plus-001h")
    ]
    deterministic_all_three_curve = [
        curve[key]["deterministic"]["all_three_success"]
        for key in ("initial", "plus-000p25h", "plus-000p5h", "plus-001h")
    ]
    historical = _historical_m10_8_arm_c()
    m10_8_final = historical["plus_1h"]["deterministic"]
    m10_9_final = curve["plus-001h"]["deterministic"]
    final_delta = {
        metric: m10_9_final[metric] - m10_8_final[metric]
        for metric in (
            "first_contact_success",
            "second_contact_success",
            "third_contact_success",
            "all_three_success",
            "no_touch_rate",
        )
    }
    final_delta["failed_distance_change_uu"] = (
        m10_9_final["failed_trajectory"][
            "distance_change_terminal_minus_initial_mean"
        ]
        - m10_8_final["failed_trajectory"][
            "distance_change_terminal_minus_initial_mean"
        ]
    )

    all_iterations = [
        iteration
        for boundary in boundaries.values()
        for iteration in boundary["iterations"]
    ]
    iteration_health = [_iteration_health(row) for row in all_iterations]
    failed_negative_windows_stay_negative = all(
        values["metrics"]["scaled_advantage"]["mean"] < 0.0
        for values in final["aggregate_credit_assignment"][
            "failed_timeout_like"
        ].values()
        if values["metrics"]["raw_advantage"]["mean"] < 0.0
    )
    critic_demonstrated_learning = all(
        row["critic"]["held_out_ev_improvement"] > 0.0
        and row["critic"]["held_out_loss_improvement"] > 0.0
        for row in iteration_health
    )
    checks = {
        "preflight_passed": preflight["checks"]["passed"],
        "all_boundaries_passed": all(
            boundary["checks"]["passed"] for boundary in boundaries.values()
        ),
        "all_advantage_signs_preserved": all(
            row["raw_scaled_sign_agreement"] for row in iteration_health
        ),
        "failed_negative_window_means_stay_negative": (
            failed_negative_windows_stay_negative
        ),
        "critic_demonstrated_held_out_learning": critic_demonstrated_learning,
        "all_rollout_log_probability_replays_passed": all(
            row["log_probability_replay"]["same_policy_replay"]["passed"]
            and row["log_probability_replay"]["independent_formula"]["passed"]
            for row in iteration_health
        ),
        "all_ar_diagnostics_passed": all(
            row["checks"]["passed"] for row in iteration_health
        ),
        "final_checkpoint_reload_exact": reload["checks"]["passed"],
        "stopped_at_one_hour": 1.0
        <= float(final["reached_simulated_hours"])
        <= 1.0003,
        "no_training_processes_remain": not processes,
        "stage_2_authorized": False,
        "production_promotion_authorized": False,
    }
    checks["passed"] = all(
        value
        for key, value in checks.items()
        if key not in {"stage_2_authorized", "production_promotion_authorized"}
    )
    monotonic_first = all(
        right >= left
        for left, right in zip(
            deterministic_first_curve, deterministic_first_curve[1:]
        )
    )
    monotonic_all_three = all(
        right >= left
        for left, right in zip(
            deterministic_all_three_curve, deterministic_all_three_curve[1:]
        )
    )
    capability_improved_materially = (
        monotonic_first
        and monotonic_all_three
        and m10_9_final["first_contact_success"]
        > m10_8_final["first_contact_success"]
        and m10_9_final["failed_trajectory"][
            "distance_change_terminal_minus_initial_mean"
        ]
        < 0.0
    )
    report = {
        "schema_version": 1,
        "closeout_version": "RivalM10_9PPOV2ComparisonV1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "exact_ppo_v2": {
            "gamma": float(config["ppo"]["gamma"]),
            "gae_lambda": float(config["ppo"]["gae_lambda"]),
            "advantage_scaling": config["ppo"]["advantage_scaling"],
            "actor": {
                "learning_rate": float(config["ppo"]["actor_learning_rate"]),
                "epochs": int(config["ppo"]["actor_epochs"]),
                "minibatch_agent_steps": int(
                    config["ppo"]["actor_minibatch_agent_steps"]
                ),
                "clip_range": float(config["ppo"]["clip_range"]),
                "max_gradient_norm": float(config["ppo"]["max_gradient_norm"]),
                "kl_stop_threshold": float(
                    config["ppo"]["actor_kl_stop_threshold"]
                ),
            },
            "critic": {
                "learning_rate": float(config["ppo"]["critic_learning_rate"]),
                "epochs": int(config["ppo"]["critic_epochs"]),
                "minibatch_agent_steps": int(
                    config["ppo"]["critic_minibatch_agent_steps"]
                ),
                "loss": config["ppo"]["critic_loss"],
                "held_out_agent_steps": int(
                    config["ppo"]["critic_validation_agent_steps"]
                ),
            },
            "analog_exploration": config["analog_exploration"],
            "native_controller_hz": int(config["time_base"]["physics_hz"]),
            "action_repeat": False,
        },
        "paired_initialization": preflight["paired_initialization"],
        "gae_physical_time_horizon": preflight["gae_physical_time_proof"],
        "preflight_critic_learnability": preflight[
            "disposable_real_rollout_and_critic"
        ]["critic_learnability"],
        "advantage_sign_evidence": {
            "preflight": preflight["advantage_sign_evidence"],
            "all_campaign_iterations_preserved_sign": checks[
                "all_advantage_signs_preserved"
            ],
            "final_credit_windows": _credit(final),
            "failed_negative_window_means_stay_negative": (
                failed_negative_windows_stay_negative
            ),
        },
        "ar_exploration_evidence": {
            "preflight_synthetic": preflight["ar_synthetic_evidence"],
            "offline_log_probability_replay": preflight[
                "offline_log_probability_replay"
            ],
            "campaign_iterations": [row["ar_exploration"] for row in iteration_health],
        },
        "capability_curve": curve,
        "campaign_iteration_health": iteration_health,
        "historical_benchmark": {
            "m10_8_arm_c": historical,
            "m10_9_minus_m10_8_at_plus_1h": final_delta,
        },
        "monotonicity": {
            "deterministic_first_contact_curve": deterministic_first_curve,
            "deterministic_first_contact_non_decreasing": monotonic_first,
            "deterministic_all_three_curve": deterministic_all_three_curve,
            "deterministic_all_three_non_decreasing": monotonic_all_three,
        },
        "ppo_v2_materially_improved_native_continuous_control_learning": (
            capability_improved_materially
        ),
        "exact_conclusion": (
            "PPO V2 fixed the measured advantage-sign, critic-learning, update-"
            "schedule, likelihood-accounting, and correlated-exploration mechanics, "
            "but deterministic Stage-1 ball acquisition did not improve "
            "monotonically and finished below M10.8 Arm C on first contact and "
            "all-three reacquisition. The PPO V2 formulation therefore did not "
            "materially improve native continuous-control learning in this run."
        ),
        "next_investigation": [
            "state_value_decomposition",
            "policy_architecture",
            "per_component_advantage_attribution",
            "off_policy_replay_or_alternative_actor_critic",
            "ppo_suitability_for_native_120hz_continuous_control",
        ],
        "final_checkpoint": final["immutable_checkpoint"],
        "final_checkpoint_reload_parity": reload,
        "remaining_processes": processes,
        "stage_2_authorized": False,
        "production_promotion_authorized": False,
        "checks": checks,
    }
    write_json_atomic(args.output, report)
    if not checks["passed"]:
        raise RuntimeError(f"M10.9 comparative closeout failed: {checks}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "training/configs/milestone10_9_stage1.json",
    )
    parser.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "final_comparison.json"
    )
    args = parser.parse_args()
    report = finalize(args)
    print(
        json.dumps(
            {
                "status": "passed",
                "material_improvement": report[
                    "ppo_v2_materially_improved_native_continuous_control_learning"
                ],
                "deterministic_first_contact_curve": report["monotonicity"][
                    "deterministic_first_contact_curve"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
