"""Compact committed evidence reports for the Milestone 06 campaign."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import REPOSITORY_ROOT, canonical_config_sha256, load_milestone06_config


RESULTS_ROOT = REPOSITORY_ROOT / "training/results/milestone06"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _optional_read(path: Path) -> dict[str, Any] | None:
    return _read(path) if path.is_file() else None


def _checkpoint_compact(manifest: dict[str, Any]) -> dict[str, Any]:
    state = manifest["trainer_state"]
    return {
        "directory": manifest["directory"],
        "files": manifest["files"],
        "cumulative_agent_steps": state["cumulative_agent_steps"],
        "cumulative_model_updates": state["cumulative_model_updates"],
        "stage": state["stage"],
        "worker_count": state["worker_count"],
        "action_exploration_prior": state["action_exploration_prior"],
        "prior_history": state["prior_history"],
        "fresh_reload_proof": manifest["fresh_reload_proof"],
    }


def _million_boundary_history(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for item in history:
        selected[int(item["cumulative_agent_steps"]) // 1_000_000] = item
    return [selected[key] for key in sorted(selected)]


def build_preflight_report() -> dict[str, Any]:
    config = load_milestone06_config()
    throughput = _read(RESULTS_ROOT / "throughput_sweep.json")
    calibration = _read(RESULTS_ROOT / "action_prior_calibration.json")
    audit = _read(RESULTS_ROOT / "reward_curriculum_audit.json")
    baseline = _read(RESULTS_ROOT / "headless_wisp_preflight.json")
    serious = _read(RESULTS_ROOT / "serious_ppo_preflight.json")
    candidate_export = _read(RESULTS_ROOT / "candidate_export_preflight.json")
    candidate_runtime = _read(RESULTS_ROOT / "candidate_runtime_smoke_preflight.json")
    production_runtime = _read(RESULTS_ROOT / "deployment_runtime_production.json")
    training_runtime = _read(RESULTS_ROOT / "deployment_runtime_training.json")
    verification = _read(RESULTS_ROOT / "preflight_verification.json")
    candidate_rows = []
    for result in throughput["results"]:
        candidate_rows.append(
            {
                "workers": result["workers"],
                "stable": result["stable"],
                "agent_steps_per_second": result[
                    "sustained_agent_steps_per_second_mean"
                ],
                "simulated_game_seconds_per_second": result[
                    "aggregate_simulated_game_seconds_per_second_mean"
                ],
                "cpu_percent": result["resources"]["cpu_percent"],
                "gpu_utilization_percent": result["resources"][
                    "gpu_utilization_percent"
                ],
                "environment_decision_latency_ms": result[
                    "rollout_inference_latency"
                ]["mean_environment_decision_latency_ms"],
                "worker_rss_mib": result["resources"]["worker_rss_mib"],
                "worker_process_health": result["worker_process_health"],
            }
        )
    checks = {
        "throughput_sweep_passed": throughput["status"] == "passed",
        "action_prior_calibration_passed": calibration["status"] == "passed",
        "reward_curriculum_audit_passed": audit["status"] == "passed",
        "headless_baseline_passed": baseline["health"]["passed"],
        "serious_ppo_iteration_passed": serious["status"] == "passed",
        "checkpoint_reload_passed": serious["checkpoint"]["fresh_reload_proof"][
            "exact_logits"
        ],
        "candidate_export_passed": candidate_export["status"] == "passed",
        "candidate_runtime_smoke_passed": candidate_runtime["status"] == "passed",
        "training_runtime_export_load_passed": training_runtime["finite"],
        "production_runtime_export_load_passed": production_runtime["finite"],
        "production_default_unchanged": not candidate_runtime[
            "production_default_replaced"
        ],
        "verification_matrix_passed": verification["status"] == "passed",
    }
    checks["passed"] = all(checks.values())
    report = {
        "schema_version": 1,
        "status": "passed" if checks["passed"] else "failed",
        "boundary": "preflight_0m",
        "campaign_agent_steps": 0,
        "campaign_model_updates": 0,
        "preflight_steps_excluded_from_campaign": serious["timing"][
            "collected_agent_steps"
        ],
        "config_sha256": canonical_config_sha256(config),
        "ppo_configuration": config["ppo"],
        "environment": config["environment"],
        "throughput_sweep": {
            "selection_rule": throughput["selection_rule"],
            "stop_reason": throughput["stop_reason"],
            "candidates": candidate_rows,
            "selected_worker_count": throughput["selected_worker_count"],
            "selected_agent_steps_per_second": throughput[
                "selected_sustained_agent_steps_per_second"
            ],
            "selected_simulated_game_seconds_per_second": throughput[
                "selected_aggregate_simulated_game_seconds_per_second"
            ],
            "serious_ppo_iteration_timing": throughput["ppo_iteration_timing"],
        },
        "action_prior_calibration": {
            "natural_observations": calibration["natural_observations"],
            "candidate_results": calibration["candidate_results"],
            "selection_rule": calibration["selection_rule"],
            "selected_appended_logit_offset": calibration[
                "selected_appended_logit_offset"
            ],
        },
        "reward_v2_audit_health": audit["reward_v2_contribution_audit"]["health"],
        "curriculum_distribution": {
            name: {
                "configured": item["configured_weights"],
                "observed": item["shares"],
                "passed": item["passed"],
            }
            for name, item in audit["curriculum_distribution_audit"]["stages"].items()
        },
        "headless_frozen_wisp_baseline": {
            "games": baseline["games"],
            "outcomes": baseline["outcomes"],
            "actions": {
                key: baseline["actions"][key]
                for key in (
                    "sampled_action_count",
                    "appended_action_count",
                    "appended_action_share",
                    "mean_appended_probability_mass",
                )
            },
            "reward_contribution_audit": baseline["reward_contribution_audit"],
            "mechanics_recovery": baseline["mechanics_recovery"],
            "health": baseline["health"],
        },
        "serious_ppo_iteration": {
            "timing": serious["timing"],
            "actions": serious["actions"],
            "rollout_metrics": serious["rollout_metrics"],
            "ppo": serious["ppo"],
            "checkpoint": _checkpoint_compact(serious["checkpoint"]),
        },
        "deployment": {
            "candidate_export": candidate_export,
            "candidate_runtime_smoke": candidate_runtime,
            "training_runtime": training_runtime,
            "production_runtime": production_runtime,
            "production_policy": "frozen_wisp_unchanged",
        },
        "checks": checks,
        "verification": verification,
        "exact_stage_a_command": (
            "training/.venv/Scripts/python.exe training/scripts/run_m06_campaign.py "
            "--stage stage_a --appended-offset -6"
        ),
        "outcome": "preflight_passed_campaign_not_started",
        "production_promoted": False,
    }
    _write(RESULTS_ROOT / "stage_preflight.json", report)
    return report


def build_stage_report(
    summary_path: str | Path,
    *,
    label: str,
    next_stage: str | None,
    next_offset: float | None,
    rlbot_path: str | Path | None = None,
    diagnostics_path: str | Path | None = None,
    completion_outcome: str | None = None,
    final_verification_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_milestone06_config()
    summary = _read(Path(summary_path))
    evaluations = []
    for reference in summary["headless_evaluations"]:
        full = _read(REPOSITORY_ROOT / reference["path"])
        evaluations.append(
            {
                **reference,
                "actions": full["actions"],
                "reward_contribution_audit": full["reward_contribution_audit"],
                "baseline_comparison": full["baseline_comparison"],
            }
        )
    rlbot = _read(Path(rlbot_path)) if rlbot_path is not None else None
    diagnostics = (
        _read(Path(diagnostics_path)) if diagnostics_path is not None else None
    )
    final_verification = (
        _read(Path(final_verification_path))
        if final_verification_path is not None
        else None
    )
    if completion_outcome not in {None, "healthy_candidate", "rejected_rollback"}:
        raise ValueError(f"Unknown completion outcome {completion_outcome!r}")
    if completion_outcome is not None and next_stage is not None:
        raise ValueError("A completed campaign cannot authorize a next stage")
    if completion_outcome == "rejected_rollback":
        if rlbot is None or diagnostics is None:
            raise ValueError("A rollback outcome requires RLBot and diagnostic evidence")
        if diagnostics["campaign_decision"]["outcome"] != "rejected_rollback":
            raise ValueError("Diagnostic evidence does not support a rollback outcome")
    latest = summary["latest_checkpoint"]
    next_prior_decision = None
    if next_stage is not None:
        if next_offset is None:
            raise ValueError("A next stage requires an explicit health-gated offset")
        if not summary["health"]["passed"]:
            raise ValueError("Cannot advance the action prior after a failed stage gate")
        calibration = _read(RESULTS_ROOT / "action_prior_calibration.json")
        calibrated_candidate = next(
            candidate
            for candidate in calibration["candidate_results"]
            if candidate["appended_logit_offset"] == next_offset
        )
        if not calibrated_candidate["safe_minority_exploration"]:
            raise ValueError(
                f"Requested next-stage offset {next_offset:g} failed calibration"
            )
        latest_headless = evaluations[-1] if evaluations else None
        next_prior_decision = {
            "decision": "advance_at_clean_health-passed_stage_boundary",
            "next_stage": next_stage,
            "selected_appended_logit_offset": next_offset,
            "calibrated_candidate": calibrated_candidate,
            "stage_health_passed": summary["health"]["passed"],
            "aggregate_stage_appended_action_share": summary["aggregate_actions"][
                "appended_action_share"
            ],
            "maximum_health_appended_action_share": config["evaluation"][
                "maximum_appended_action_share_for_health"
            ],
            "latest_headless_baseline_comparison": (
                latest_headless["baseline_comparison"]
                if latest_headless is not None
                else None
            ),
            "rationale": (
                "The completed stage passed every numerical and headless health gate; "
                "the requested next offset was separately safe in natural-observation "
                "calibration, and the campaign action share remained controlled."
            ),
        }
        exact_resume = (
            "training/.venv/Scripts/python.exe training/scripts/run_m06_campaign.py "
            f"--stage {next_stage} --appended-offset {next_offset:g} "
            f"--resume {latest['directory']}"
        )
    else:
        exact_resume = summary["exact_same_stage_resume_command"]
    stage_names = [stage["name"] for stage in config["stages"]]
    stage_index = stage_names.index(summary["stage"])
    possible_next_stage = (
        stage_names[stage_index + 1] if stage_index + 1 < len(stage_names) else None
    )
    resume_if_new_authority = None
    if completion_outcome is not None and possible_next_stage is not None:
        resume_if_new_authority = (
            "training/.venv/Scripts/python.exe training/scripts/run_m06_campaign.py "
            f"--stage {possible_next_stage} "
            f"--appended-offset {summary['action_exploration_prior']['appended_logit_offset']:g} "
            f"--resume {latest['directory']}"
        )
        exact_resume = None

    candidate_export = _optional_read(RESULTS_ROOT / f"candidate_export_{label}.json")
    candidate_runtime = _optional_read(
        RESULTS_ROOT / f"candidate_runtime_smoke_{label}.json"
    )
    training_runtime = _optional_read(
        RESULTS_ROOT / f"deployment_runtime_training_{label}.json"
    )
    production_runtime = _optional_read(
        RESULTS_ROOT / f"deployment_runtime_production_{label}.json"
    )
    completion_status = (
        "rejected_at_evaluation_boundary"
        if completion_outcome == "rejected_rollback"
        else summary["status"]
    )
    compact_diagnostics = None
    if diagnostics is not None:
        compact_diagnostics = {
            "path": Path(diagnostics_path).as_posix(),
            "status": diagnostics["status"],
            "comparison": diagnostics["comparison"],
            "diagnosis": diagnostics["diagnosis"],
            "campaign_decision": diagnostics["campaign_decision"],
            "execution_note": diagnostics["execution_note"],
        }
    deployment_boundary = None
    if candidate_export is not None:
        deployment_boundary = {
            "candidate_export": candidate_export,
            "candidate_runtime_smoke": candidate_runtime,
            "training_runtime": training_runtime,
            "production_runtime": production_runtime,
            "exact_export_command": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/export_m06_candidate.py "
                f"--checkpoint {latest['directory']} --label {candidate_export['label']} "
                f"--output training/results/milestone06/candidate_export_{label}.json"
            ),
            "exact_rlbot_evaluation_command": (
                ".venv/Scripts/python.exe "
                "training/scripts/run_m06_rlbot_stage_eval.py "
                f"--export-report training/results/milestone06/candidate_export_{label}.json "
                f"--games 8 --output training/results/milestone06/rlbot_{label}.json"
            ),
            "opt_in_candidate_environment": candidate_export["rlbot_environment"],
            "deployment_status": (
                "rejected_do_not_promote"
                if completion_outcome == "rejected_rollback"
                else "candidate_only_not_promoted"
            ),
        }
    report = {
        "schema_version": 1,
        "status": completion_status,
        "boundary": label,
        "stage": summary["stage"],
        "cumulative_agent_steps": summary["cumulative_agent_steps"],
        "cumulative_model_updates": summary["cumulative_model_updates"],
        "config_sha256": canonical_config_sha256(config),
        "worker_count": summary["worker_count"],
        "ppo_configuration": config["ppo"],
        "curriculum_configuration": next(
            stage["curriculum_weights"]
            for stage in config["stages"]
            if stage["name"] == summary["stage"]
        ),
        "action_exploration_prior": summary["action_exploration_prior"],
        "prior_history": summary["prior_history"],
        "latest_checkpoint": _checkpoint_compact(latest),
        "latest_ppo": summary["latest_iteration"]["ppo"],
        "latest_iteration_timing": {
            key: summary["latest_iteration"][key]
            for key in (
                "collected_agent_steps",
                "collection_seconds",
                "agent_steps_per_second",
                "iteration_wall_seconds",
            )
        },
        "aggregate_actions": summary["aggregate_actions"],
        "action_share_history_approximately_each_1m": _million_boundary_history(
            summary["action_share_history"]
        ),
        "ppo_history_approximately_each_1m": _million_boundary_history(
            summary["ppo_history"]
        ),
        "reward_contribution_audit": summary["aggregate_rollout_metrics"][
            "reward_components"
        ],
        "mechanics_recovery": summary["aggregate_rollout_metrics"][
            "mechanics_recovery"
        ],
        "curriculum_distribution": {
            "counts": summary["aggregate_rollout_metrics"]["curriculum_reset_counts"],
            "shares": summary["aggregate_rollout_metrics"]["curriculum_reset_shares"],
        },
        "headless_frozen_wisp_evaluations": evaluations,
        "rlbot_evaluation": rlbot,
        "rlbot_diagnostics": compact_diagnostics,
        "deployment_boundary": deployment_boundary,
        "health": summary["health"],
        "next_stage_prior_decision": next_prior_decision,
        "exact_resume_command": exact_resume,
        "resume_command_if_new_authority": resume_if_new_authority,
        "campaign_outcome": (
            None
            if completion_outcome is None
            else {
                "healthy_candidate": "healthy candidate",
                "rejected_rollback": "rejected/rollback",
            }[completion_outcome]
        ),
        "campaign_ceiling_agent_steps": config["campaign_ceiling_agent_steps"],
        "unused_ceiling_agent_steps": (
            config["campaign_ceiling_agent_steps"]
            - summary["cumulative_agent_steps"]
        ),
        "remaining_authorized_agent_steps": (
            0
            if completion_outcome is not None
            else config["campaign_ceiling_agent_steps"]
            - summary["cumulative_agent_steps"]
        ),
        "best_checkpoint": {
            "basis": (
                "highest 100-game headless frozen-Wisp result among trained boundaries; "
                "retained for research but rejected for deployment by the 20M RLBot battery"
                if completion_outcome == "rejected_rollback"
                else "latest health-passed trained boundary"
            ),
            "checkpoint": _checkpoint_compact(latest),
        },
        "best_verified_deployment_policy": (
            "frozen_wisp_v4.1_4-4_goal_differential_plus_5"
        ),
        "final_promotion_battery": (
            "not_run_candidate_failed_ordinary_20m_boundary"
            if completion_outcome == "rejected_rollback"
            else "not_run"
        ),
        "final_verification": final_verification,
        "production_policy": "frozen_wisp_unchanged",
        "production_promoted": False,
    }
    _write(RESULTS_ROOT / f"stage_{label}.json", report)
    return report


def write_results_markdown() -> Path:
    config = load_milestone06_config()
    preflight = _read(RESULTS_ROOT / "stage_preflight.json")
    stage_paths = sorted(RESULTS_ROOT.glob("stage_*m.json"))
    stages = [_read(path) for path in stage_paths]
    lines = [
        "# Milestone 06 Results",
        "",
        "Milestone 06 is Rival's first serious staged RLGym/RocketSim training campaign. "
        "Production remains the frozen Wisp policy unless the explicit final promotion "
        "battery is earned.",
        "",
        "## Fixed campaign configuration",
        "",
        f"- Campaign ceiling: {config['campaign_ceiling_agent_steps']:,} agent-steps.",
        f"- Measured worker count: {config['environment']['workers']}.",
        f"- Student cadence: {config['environment']['cadence_ticks']} physics ticks at "
        f"{config['environment']['physics_tick_rate_hz']} Hz.",
        f"- PPO iteration/buffer/batch/minibatch: "
        f"{config['ppo']['agent_steps_per_iteration']:,} / "
        f"{config['ppo']['experience_buffer_size']:,} / "
        f"{config['ppo']['batch_size']:,} / "
        f"{config['ppo']['minibatch_size']:,}.",
        f"- Actor/critic learning rates: {config['ppo']['policy_learning_rate']:g} / "
        f"{config['ppo']['critic_learning_rate']:g}.",
        f"- Gamma / GAE lambda: {config['ppo']['gamma']:.15f} / "
        f"{config['ppo']['gae_lambda']:g}.",
        "",
        "## Preflight evidence",
        "",
        f"Status: `{preflight['status']}`. The 24–64 sustained sweep selected "
        f"**{preflight['throughput_sweep']['selected_worker_count']} workers** at "
        f"{preflight['throughput_sweep']['selected_agent_steps_per_second']:.2f} "
        "agent-steps/sec. The measured Stage A appended-action offset is "
        f"`{preflight['action_prior_calibration']['selected_appended_logit_offset']:g}`.",
        "",
        f"The frozen-Wisp headless baseline was "
        f"{preflight['headless_frozen_wisp_baseline']['outcomes']['wins']}-"
        f"{preflight['headless_frozen_wisp_baseline']['outcomes']['losses']}-"
        f"{preflight['headless_frozen_wisp_baseline']['outcomes']['ties']} over "
        f"{preflight['headless_frozen_wisp_baseline']['games']} balanced games. "
        "Reward/curriculum audits, a full-size PPO iteration, fresh optimizer reload, "
        "exact policy export, and production-runtime loading all passed.",
        "",
        "## Training boundaries",
        "",
    ]
    if not stages:
        lines.append("No campaign steps have been counted yet.")
        lines.append("")
    for stage in stages:
        latest_eval = (
            stage["headless_frozen_wisp_evaluations"][-1]
            if stage["headless_frozen_wisp_evaluations"]
            else None
        )
        lines.extend(
            [
                f"### {stage['boundary']}",
                "",
                f"- Status: `{stage['status']}`; cumulative steps/updates: "
                f"{stage['cumulative_agent_steps']:,} / "
                f"{stage['cumulative_model_updates']:,}.",
                f"- Aggregate appended-action share: "
                f"{stage['aggregate_actions']['appended_action_share']:.4%}.",
            ]
        )
        if latest_eval is not None:
            outcomes = latest_eval["outcomes"]
            lines.append(
                f"- Latest 100-game headless Wisp result: {outcomes['wins']}-"
                f"{outcomes['losses']}-{outcomes['ties']}, goal differential "
                f"{outcomes['goal_differential']:+d}; health "
                f"`{'passed' if latest_eval['health']['passed'] else 'failed'}`."
            )
        if stage["rlbot_evaluation"] is not None:
            aggregate = stage["rlbot_evaluation"]["aggregates"]["overall"]
            lines.append(
                f"- RLBot stage context: {aggregate['wins']}-"
                f"{aggregate['losses']}-{aggregate['ties']}, goal differential "
                f"{aggregate['goal_differential']:+d}."
            )
        if stage.get("rlbot_diagnostics") is not None:
            diagnosis = stage["rlbot_diagnostics"]["diagnosis"]
            lines.append(
                "- RLBot telemetry integrity: "
                f"`{'passed' if diagnosis['runtime_integrity_passed'] else 'failed'}`; "
                f"transfer verdict: `{diagnosis['gameplay_transfer_verdict']}`."
            )
        if stage.get("campaign_outcome") is not None:
            lines.append(f"- Campaign outcome: **{stage['campaign_outcome']}**.")
        if stage["exact_resume_command"] is not None:
            resume_text = stage["exact_resume_command"]
        elif stage.get("resume_command_if_new_authority") is not None:
            resume_text = (
                "none authorized; new authority would be required before using "
                f"{stage['resume_command_if_new_authority']}"
            )
        else:
            resume_text = "none; health gate stopped campaign"
        lines.extend(
            [
                f"- Production: frozen Wisp unchanged. Resume: "
                f"`{resume_text}`",
                "",
            ]
        )
    completed_stage = next(
        (stage for stage in reversed(stages) if stage.get("campaign_outcome") is not None),
        None,
    )
    lines.extend(
        [
            "## Promotion state",
            "",
            (
                "Milestone 06 ended as **rejected/rollback** at the 20M clean boundary. "
                "The candidate failed the ordinary eight-game RLBot boundary, so the final "
                "16-game promotion battery was not run and production remains frozen Wisp."
                if completed_stage is not None
                and completed_stage["campaign_outcome"] == "rejected/rollback"
                else "No trained checkpoint is promoted by training-step count alone. A final "
                "candidate must pass the governed 16-game RLBot battery, deployment parity, "
                "and aggregate gameplay/mechanics review. Until then, the production Rival "
                "policy remains frozen Wisp."
            ),
            "",
        ]
    )
    destination = REPOSITORY_ROOT / "docs/MILESTONE_06_RESULTS.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination
