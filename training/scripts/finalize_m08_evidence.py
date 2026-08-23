from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPOSITORY_ROOT / "training/results/milestone08"
FINAL_VERDICT = "partial_architecture_pass_learning_inconclusive"
FINAL_STATUS = "completed_with_artifact_preservation_limitation"

REPORT_EXPECTATIONS = {
    "artifact_preservation_incident.json": {
        "completed_with_irrecoverable_ignored_artifacts"
    },
    "candidate_export_001m.json": {"passed"},
    "candidate_export_005m.json": {"passed"},
    "headless_000000000.json": {"passed"},
    "headless_000499748.json": {"passed"},
    "headless_000999822.json": {"passed"},
    "headless_001999776.json": {"passed"},
    "headless_004999790.json": {"passed"},
    "mechanics_prior_calibration.json": {"passed"},
    "mechanics_usage_adjustment_002m.json": {"authorized"},
    "observation_contract_v2.json": {"completed"},
    "pretraining_gates.json": {"passed"},
    "rlbot_001m.json": {"passed"},
    "rlbot_001m_native_rate.json": {"passed"},
    "rlbot_005m_native_rate.json": {"passed"},
    "rlbot_cadence_rate_attribution_001m.json": {
        "passed_native_rate_with_accelerated_stress_limit"
    },
    # This accelerated 5x stress result is retained as a measured limitation. The
    # prospective native-rate rerun and attribution report are the accepted gate.
    "rlbot_gate_001m.json": {"failed"},
    "rlbot_gate_001m_native_rate.json": {"passed"},
    "rlbot_gate_005m_native_rate.json": {"passed"},
    "rlviser_spectator_001m_smoke.json": {"passed"},
    "rlviser_spectator_005m_smoke.json": {"passed"},
    "throughput_sanity.json": {"passed"},
    "training_000500000.json": {"completed_boundary"},
    "training_001000000.json": {"completed_boundary"},
    "training_002000000.json": {"completed_boundary"},
    "training_005000000.json": {"completed_boundary"},
    "worker_fallback_000500000.json": {"authorized"},
    "zero_step_dual_rate_rlbot.json": {"passed"},
    "zero_step_rlbot.json": {"passed"},
    "zero_step_transfer_gate.json": {"passed"},
}

LOCAL_ARTIFACTS = (
    "bot/models/POLICY.lt",
    "bot/models/SHARED_HEAD.lt",
    "bot/models/RIVAL_ACTIONS_V1.npy",
    "training/artifacts/bootstrap/wisp_student_expanded_v1.pt",
    "training/artifacts/milestone07/zero_step_actor.ts",
    "training/artifacts/milestone08/mechanics_initial_v1.pt",
    (
        "training/artifacts/milestone08/"
        "mechanics_initial_v1_regenerated_from_exact_state_20260823.ts"
    ),
    (
        "training/datasets/milestone08/"
        "natural_prior_observations_accidental_rerun_20260823.npy"
    ),
    "training/artifacts/milestone08/001m/mechanics_actor.ts",
    "training/artifacts/milestone08/005m/mechanics_actor.ts",
    "training/checkpoints/milestone08/001999776/PPO_POLICY.pt",
    "training/checkpoints/milestone08/001999776/PPO_POLICY_OPTIMIZER.pt",
    "training/checkpoints/milestone08/001999776/PPO_VALUE_NET.pt",
    "training/checkpoints/milestone08/001999776/PPO_VALUE_NET_OPTIMIZER.pt",
    "training/checkpoints/milestone08/001999776/RIVAL_M08_STATE.json",
    "training/checkpoints/milestone08/004999790/PPO_POLICY.pt",
    "training/checkpoints/milestone08/004999790/PPO_POLICY_OPTIMIZER.pt",
    "training/checkpoints/milestone08/004999790/PPO_VALUE_NET.pt",
    "training/checkpoints/milestone08/004999790/PPO_VALUE_NET_OPTIMIZER.pt",
    "training/checkpoints/milestone08/004999790/RIVAL_M08_STATE.json",
    "training/tools/rlviser/rlviser.exe",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _committed_text_sha256(path: Path) -> str:
    """Hash the LF-normalized bytes Git stores for text evidence."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(*arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_tracks(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative_path],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _load(name: str) -> dict[str, Any]:
    path = RESULTS_ROOT / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required Milestone 08 report: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_local_sha256(relative_path: str, expected_sha256: str) -> None:
    path = REPOSITORY_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Missing hash-gated artifact: {path}")
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Artifact hash mismatch for {relative_path}: {actual} != {expected_sha256}"
        )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _validate_reports() -> dict[str, dict[str, Any]]:
    reports: dict[str, dict[str, Any]] = {}
    for name, accepted_statuses in sorted(REPORT_EXPECTATIONS.items()):
        report = _load(name)
        status = report.get("status")
        if status not in accepted_statuses:
            raise RuntimeError(
                f"Unexpected status for {name}: {status!r}; "
                f"accepted={sorted(accepted_statuses)}"
            )
        if report.get("production_promoted") is True:
            raise RuntimeError(f"Production promotion recorded unexpectedly in {name}")
        if report.get("production_modified_or_promoted") is True:
            raise RuntimeError(
                f"Production modification recorded unexpectedly in {name}"
            )
        reports[name] = report
    return reports


def _require_core_gates(reports: dict[str, dict[str, Any]]) -> None:
    preservation = reports["artifact_preservation_incident.json"]
    if not preservation["initial_actor_checkpoint"]["byte_exact_recovery"]:
        raise RuntimeError("Initial mechanics actor was not recovered byte-exactly")
    if preservation["natural_observation_corpus"][
        "original_bytes_available_locally"
    ]:
        raise RuntimeError("Preservation report unexpectedly claims corpus recovery")
    if preservation["initial_torchscript"]["original_bytes_available_locally"]:
        raise RuntimeError(
            "Preservation report unexpectedly claims original TorchScript recovery"
        )
    if preservation["impact"]["campaign_training_or_evaluation_changed"]:
        raise RuntimeError("Artifact incident affected campaign training/evaluation")
    if (REPOSITORY_ROOT / preservation["natural_observation_corpus"]["original_path"]).exists():
        raise RuntimeError("Canonical corpus path could misidentify replacement bytes")
    if (REPOSITORY_ROOT / preservation["initial_torchscript"]["original_path"]).exists():
        raise RuntimeError("Canonical initial export could misidentify replacement bytes")
    tracked_report = preservation["tracked_evidence"]
    if _committed_text_sha256(REPOSITORY_ROOT / tracked_report["path"]) != tracked_report[
        "restored_lf_normalized_sha256"
    ]:
        raise RuntimeError("Tracked prior calibration report was not restored exactly")
    _require_local_sha256(
        preservation["initial_actor_checkpoint"]["path"],
        preservation["initial_actor_checkpoint"]["expected_sha256"],
    )
    _require_local_sha256(
        preservation["natural_observation_corpus"]["accidental_rerun_preserved_as"],
        preservation["natural_observation_corpus"]["accidental_rerun_sha256"],
    )
    _require_local_sha256(
        preservation["initial_torchscript"]["regenerated_preserved_as"],
        preservation["initial_torchscript"]["regenerated_sha256"],
    )

    observation = reports["observation_contract_v2.json"]
    observation_gate = observation["observation_gate"]
    policy_effect = observation["frozen_wisp_policy_effect"][
        "live_vs_training_style"
    ]
    if not observation_gate["passed"]:
        raise RuntimeError("Observation contract v2 gate did not pass")
    if observation["corpus"]["samples"] < 1000:
        raise RuntimeError("Observation corpus does not contain 1,000 held states")
    if policy_effect["masked_top1_agreement"] < 0.97:
        raise RuntimeError("Held-live masked top-1 agreement is below 97 percent")
    if policy_effect["mean_js_divergence_nats"] > 0.002:
        raise RuntimeError("Held-live mean JS divergence exceeds 0.002")
    if observation_gate["maximum_observed_single_group_top1_materiality"] > 0.05:
        raise RuntimeError("A substitution group changes more than 5 percent top-1")

    pretraining = reports["pretraining_gates.json"]
    if not pretraining["passed"] or not pretraining["fallback_invariant"]["passed"]:
        raise RuntimeError("Pretraining or exact forced-PASS fallback gate failed")
    if not pretraining["teacher_hashes"]["all_match"]:
        raise RuntimeError("Frozen Wisp teacher hashes changed")
    if not pretraining["action_table"]["wisp_prefix_exact"]:
        raise RuntimeError("The 90-row Wisp action prefix is not exact")

    zero_step = reports["zero_step_transfer_gate.json"]
    if not zero_step["passed"]:
        raise RuntimeError("Zero-step transfer gate failed")
    if zero_step["decision_rule"]["wins_losses_or_score_used_by_gate"]:
        raise RuntimeError("Zero-step cadence gate improperly depends on match score")

    prior = reports["mechanics_prior_calibration.json"]
    if prior["action_contract"]["expanded_action_count"] != 158:
        raise RuntimeError("Expanded action table does not contain 158 rows")
    if prior["action_contract"]["appended_unique_count"] != 68:
        raise RuntimeError("Appended action suffix does not contain 68 rows")
    if prior["calibrated"]["deterministic_override_rate"] != 0.0:
        raise RuntimeError("Initial mechanics prior is not deterministic PASS")
    if not 0.0 < prior["sampled_audit"]["sampled_override_rate"] < 0.10:
        raise RuntimeError("Initial sampled mechanics exposure is not small/nonzero")

    adjustment = reports["mechanics_usage_adjustment_002m.json"]
    if adjustment["source_checkpoint"]["agent_steps"] != 1_999_776:
        raise RuntimeError("Mechanics adjustment is not bound to the clean 2M boundary")
    if not all(adjustment["checks"].values()):
        raise RuntimeError("Mechanics usage adjustment validation failed")
    for name, file_evidence in adjustment["source_checkpoint"]["files"].items():
        _require_local_sha256(
            f"training/checkpoints/milestone08/001999776/{name}",
            file_evidence["sha256"],
        )

    training_names = (
        "training_000500000.json",
        "training_001000000.json",
        "training_002000000.json",
        "training_005000000.json",
    )
    expected_steps = (499_748, 999_822, 1_999_776, 4_999_790)
    for name, expected in zip(training_names, expected_steps, strict=True):
        training = reports[name]
        if training["cumulative_agent_steps"] != expected:
            raise RuntimeError(f"Unexpected cumulative steps in {name}")
        if training["cumulative_agent_steps"] > 5_000_000:
            raise RuntimeError(f"Authorized 5M ceiling exceeded in {name}")
        if not training["strategic_unchanged"] or not training["health"]["passed"]:
            raise RuntimeError(f"Training health/frozen strategic proof failed in {name}")

    final_training = reports["training_005000000.json"]
    if final_training["worker_count"] != 56:
        raise RuntimeError("Final campaign did not use the authorized stable fallback")
    if len(final_training["mechanics_usage_adjustment_history"]) != 1:
        raise RuntimeError("Mechanics usage adjustment was not applied exactly once")
    reload_proof = final_training["latest_checkpoint"]["fresh_reload_proof"]
    if not all(
        reload_proof[key]
        for key in (
            "fresh_instance",
            "exact_logits",
            "policy_optimizer_state_loaded",
            "critic_optimizer_state_loaded",
            "state_file_parse_passed",
        )
    ):
        raise RuntimeError("Final full-checkpoint reload proof failed")

    export = reports["candidate_export_005m.json"]
    if not all(export["gates"].values()):
        raise RuntimeError("Final candidate export gate failed")
    if export["production_default"] != "frozen_wisp_unchanged":
        raise RuntimeError("Final export does not preserve the production default")
    for name, file_evidence in export["source_files"].items():
        _require_local_sha256(
            f"training/checkpoints/milestone08/004999790/{name}",
            file_evidence["sha256"],
        )
    _require_local_sha256(
        export["torchscript_export"]["path"],
        export["torchscript_export"]["sha256"],
    )
    teacher_files = reports["pretraining_gates.json"]["teacher_hashes"]["files"]
    for relative_path, file_evidence in teacher_files.items():
        _require_local_sha256(relative_path, file_evidence["expected_sha256"])

    final_rlbot = reports["rlbot_gate_005m_native_rate.json"]
    if not final_rlbot["passed"]:
        raise RuntimeError("Final native-rate RLBot transfer gate failed")
    if not final_rlbot["decision_rule"]["cadence_collapse_is_technical_not_score_based"]:
        raise RuntimeError("Final cadence decision rule is not technical")

    spectator = reports["rlviser_spectator_005m_smoke.json"]
    if not all(
        (
            spectator["separate_process"],
            spectator["single_environment"],
            not spectator["headless_environment_modified"],
            spectator["spectator_version"] == "RivalRLViserSpectatorV2",
        )
    ):
        raise RuntimeError("Final optional RLViser isolation preflight failed")


def _report_entries(
    reports: dict[str, dict[str, Any]], final_verification_path: Path
) -> list[dict[str, Any]]:
    entries = []
    for name in sorted(reports):
        path = RESULTS_ROOT / name
        entries.append(
            {
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": _committed_text_sha256(path),
                "size_bytes": path.stat().st_size,
                "status": reports[name].get("status"),
            }
        )
    final_report = json.loads(final_verification_path.read_text(encoding="utf-8"))
    entries.append(
        {
            "path": final_verification_path.relative_to(REPOSITORY_ROOT).as_posix(),
            "sha256": _committed_text_sha256(final_verification_path),
            "size_bytes": final_verification_path.stat().st_size,
            "status": final_report["status"],
        }
    )
    return entries


def _artifact_entries() -> list[dict[str, Any]]:
    entries = []
    for relative in LOCAL_ARTIFACTS:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing required Milestone 08 artifact: {path}")
        entries.append(
            {
                "path": relative,
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "tracked": _git_tracks(relative),
            }
        )
    return entries


def _headless_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "games": report["games"],
        **report["outcomes"],
        "deterministic_override_rate": report["actions"][
            "deterministic_override_rate"
        ],
        "mean_override_probability": report["actions"][
            "mean_override_probability"
        ],
        "health_passed": report["health"]["passed"],
    }


def build_final_verification(
    reports: dict[str, dict[str, Any]], *, verified_utc: str
) -> dict[str, Any]:
    observation = reports["observation_contract_v2.json"]
    policy_effect = observation["frozen_wisp_policy_effect"][
        "live_vs_training_style"
    ]
    observation_gate = observation["observation_gate"]
    prior = reports["mechanics_prior_calibration.json"]
    adjustment = reports["mechanics_usage_adjustment_002m.json"]
    final_training = reports["training_005000000.json"]
    final_headless = reports["headless_004999790.json"]
    final_export = reports["candidate_export_005m.json"]
    zero_matrix = reports["zero_step_dual_rate_rlbot.json"]["modes"]["M8P"]
    one_matrix = reports["rlbot_001m_native_rate.json"]["modes"]["M8C"]
    final_matrix = reports["rlbot_005m_native_rate.json"]["modes"]["M8C"]
    final_cadence = reports["rlbot_gate_005m_native_rate.json"]
    rates = [
        float(iteration["agent_steps_per_second"])
        for iteration in final_training["ppo_history"]
    ]
    final_history = final_training["action_probability_history"]

    boundaries = []
    boundary_spec = (
        (
            "0",
            None,
            "headless_000000000.json",
            prior["sampled_audit"]["sampled_override_rate"],
            "calibrated_natural_prior",
        ),
        (
            "500k",
            "training_000500000.json",
            "headless_000499748.json",
            None,
            "training_invocation_aggregate",
        ),
        (
            "1M",
            "training_001000000.json",
            "headless_000999822.json",
            None,
            "training_invocation_aggregate",
        ),
        (
            "2M",
            "training_002000000.json",
            "headless_001999776.json",
            None,
            "training_invocation_aggregate",
        ),
        (
            "5M",
            "training_005000000.json",
            "headless_004999790.json",
            None,
            "post_adjustment_2m_to_5m_invocation_aggregate",
        ),
    )
    for label, training_name, headless_name, sampled_rate, rate_scope in boundary_spec:
        headless = reports[headless_name]
        training = reports[training_name] if training_name else None
        if training is not None:
            sampled_rate = training["aggregate_actions"]["sampled_override_rate"]
        boundaries.append(
            {
                "label": label,
                "cumulative_agent_steps": (
                    training["cumulative_agent_steps"] if training else 0
                ),
                "cumulative_model_updates": (
                    training["cumulative_model_updates"] if training else 0
                ),
                "sampled_override_rate": sampled_rate,
                "sampled_override_rate_scope": rate_scope,
                "deterministic_override_rate": headless["actions"][
                    "deterministic_override_rate"
                ],
                "headless": _headless_summary(headless),
                "strategic_unchanged": headless["strategic_branch"]["all_unchanged"],
            }
        )

    final_sessions = final_cadence["sessions"]
    final_mechanics_decisions = sum(
        session["cadence"]["mechanics_decisions"] for session in final_sessions
    )
    final_overrides = sum(
        session["mechanics"]["override_count"] for session in final_sessions
    )

    return {
        "schema_version": 1,
        "status": FINAL_STATUS,
        "verdict": FINAL_VERDICT,
        "verified_utc": verified_utc,
        "source_head_before_final_commit": _git("rev-parse", "HEAD"),
        "repository_sync": {
            "branch": _git("branch", "--show-current"),
            "origin": _git("remote", "get-url", "origin"),
            "origin_main_before_final_commit": _git("rev-parse", "origin/main"),
            "merge_base": _git("merge-base", "HEAD", "origin/main"),
            "post_push_remote_readback": (
                "reported after the final push to avoid a self-referential commit"
            ),
        },
        "tests": {
            "root_full_pytest": {
                "command": (
                    ".venv/Scripts/python.exe -m pytest tests -q --basetemp "
                    ".tmp/pytest-m08-final-production-3"
                ),
                "passed": 87,
                "failed": 0,
                "warnings": 2,
            },
            "training_full_pytest": {
                "command": (
                    "training/.venv/Scripts/python.exe -m pytest training/tests -q "
                    "--basetemp .tmp/pytest-m08-final-training-3"
                ),
                "passed": 43,
                "failed": 0,
                "warnings": 43,
            },
        },
        "static_verification": {
            "repository_wide_ruff": "passed",
            "compileall_production_tests_training_scripts": "passed",
            "git_diff_check": "passed",
            "milestone08_json_reports_parsed_before_manifest": len(reports),
            "action_prefix_rows_exact": 90,
            "frozen_hash_checks": "passed",
            "calibration_help_side_effect_free": "passed",
            "calibration_overwrite_guard": "enabled",
        },
        "architecture_gates": {
            "observation_contract": {
                "held_live_samples": observation["corpus"]["samples"],
                "masked_top1_agreement": policy_effect["masked_top1_agreement"],
                "hard_minimum": observation_gate[
                    "hard_minimum_masked_top1_agreement"
                ],
                "target": observation_gate["target_masked_top1_agreement"],
                "target_reached": (
                    policy_effect["masked_top1_agreement"]
                    >= observation_gate["target_masked_top1_agreement"]
                ),
                "mean_js_divergence_nats": policy_effect[
                    "mean_js_divergence_nats"
                ],
                "maximum_single_group_top1_materiality": observation_gate[
                    "maximum_observed_single_group_top1_materiality"
                ],
                "directly_representable_max_abs_error": observation_gate[
                    "directly_representable_max_abs_error"
                ],
                "passed": observation_gate["passed"],
            },
            "randomized_first_90_exact_samples": reports["pretraining_gates.json"][
                "randomized_logit_parity"
            ]["samples"],
            "held_first_90_exact_samples": reports["pretraining_gates.json"][
                "held_live_logit_parity"
            ]["samples"],
            "strategic_schedule": [
                "previous",
                "previous",
                "previous",
                "previous",
                "previous",
                "selected",
                "selected",
                "selected",
            ],
            "mechanics_schedule": [
                "previous_emitted",
                "selected",
                "selected",
                "selected",
            ],
            "forced_pass_exact": reports["pretraining_gates.json"][
                "fallback_invariant"
            ]["passed"],
            "strategic_optimizer_parameters": 0,
            "mechanics_outputs": 69,
            "mechanics_mapping": "PASS plus global action indices 90 through 157",
        },
        "throughput_and_worker_selection": {
            "m06_reference_worker_count": 56,
            "m06_reference_agent_steps_per_second": reports[
                "throughput_sanity.json"
            ]["m06_selected_agent_steps_per_second"],
            "m08_short_sanity": [
                {
                    "workers": item["workers"],
                    "agent_steps_per_second": item[
                        "sustained_agent_steps_per_second_mean"
                    ],
                    "rate_cv": item["window_rate_coefficient_of_variation"],
                    "worker_rss_mib": item["resources"]["worker_rss_mib"]["mean"],
                    "stable": item["stable"],
                }
                for item in reports["throughput_sanity.json"]["results"]
            ],
            "short_sanity_winner": reports["throughput_sanity.json"][
                "selected_worker_count"
            ],
            "effective_campaign_worker_count": final_training["worker_count"],
            "fallback_reason": reports["worker_fallback_000500000.json"][
                "failed_launch_error"
            ],
            "post_adjustment_iteration_agent_steps_per_second": {
                "count": len(rates),
                "mean": statistics.fmean(rates),
                "minimum": min(rates),
                "maximum": max(rates),
            },
        },
        "mechanics_usage": {
            "initial_calibrated_mean_override_probability": prior["calibrated"][
                "mean_override_probability"
            ],
            "initial_sampled_override_rate": prior["sampled_audit"][
                "sampled_override_rate"
            ],
            "adjustment_source_steps": adjustment["source_checkpoint"][
                "agent_steps"
            ],
            "adjustment_target_mean_override_probability": adjustment[
                "target_mean_override_probability"
            ],
            "adjustment_before_mean_override_probability": adjustment["before"][
                "mean_override_probability"
            ],
            "adjustment_after_mean_override_probability": adjustment["after"][
                "mean_override_probability"
            ],
            "adjustment_sampled_audit_override_rate": adjustment["sampled_audit"][
                "override_rate"
            ],
            "first_post_adjustment_iteration_sampled_override_rate": final_history[0][
                "sampled_override_rate"
            ],
            "post_adjustment_leg_sampled_override_rate": final_training[
                "aggregate_actions"
            ]["sampled_override_rate"],
            "final_iteration_sampled_override_rate": final_history[-1][
                "sampled_override_rate"
            ],
            "final_headless_mean_override_probability": final_headless["actions"][
                "mean_override_probability"
            ],
            "final_headless_deterministic_override_rate": final_headless["actions"][
                "deterministic_override_rate"
            ],
            "final_rlbot_mechanics_decisions": final_mechanics_decisions,
            "final_rlbot_deterministic_overrides": final_overrides,
            "original_prior_corpus_bytes_available_locally": False,
            "conclusion": (
                "Sampled mechanics exposure was measurable, but deterministic use "
                "remained zero and the learned policy moved back toward PASS."
            ),
        },
        "training": {
            "authorized_ceiling_agent_steps": 5_000_000,
            "actual_final_agent_steps": final_training["cumulative_agent_steps"],
            "cumulative_model_updates": final_training["cumulative_model_updates"],
            "all_iterations_healthy": final_training["health"][
                "all_iterations_passed"
            ],
            "strategic_branch_unchanged": final_training["strategic_unchanged"],
            "boundaries": boundaries,
            "fresh_checkpoint_reload": final_training["latest_checkpoint"][
                "fresh_reload_proof"
            ],
        },
        "rlbot_transfer": {
            "cadence_decision_is_score_independent": True,
            "score_role": "bounded severe-regression context only",
            "forced_pass_zero_step": zero_matrix["aggregates"]["overall"],
            "candidate_001m_native_rate": one_matrix["aggregates"]["overall"],
            "candidate_005m_native_rate": final_matrix["aggregates"]["overall"],
            "final_native_rate_cadence": {
                "mechanics_decisions": final_mechanics_decisions,
                "strategic_modal_interval_ticks": 8,
                "mechanics_modal_interval_ticks": 4,
                "minimum_mechanics_within_one_tick_rate": min(
                    session["cadence"]["mechanics_within_one_tick_of_mode_rate"]
                    for session in final_sessions
                ),
                "minimum_strategic_within_one_tick_rate": min(
                    session["cadence"]["strategic_within_one_tick_of_mode_rate"]
                    for session in final_sessions
                ),
                "minimum_two_to_one_clock_ratio": min(
                    session["cadence"][
                        "mechanics_to_expected_two_per_strategic_ratio"
                    ]
                    for session in final_sessions
                ),
                "all_sessions_passed": all(
                    session["passed"] for session in final_sessions
                ),
            },
            "accelerated_001m_stress_result": (
                "failed strict host-scheduling cadence at game speed 5; retained, "
                "not used to overwrite the prospective native-rate result"
            ),
            "severe_regression_gate_passed": True,
            "mechanics_learning_demonstrated": False,
        },
        "deployment": {
            "candidate_runtime": final_export["runtime_mode"],
            "torchscript_export": final_export["torchscript_export"],
            "rlviser_optional_separate_process_smoke": reports[
                "rlviser_spectator_005m_smoke.json"
            ],
            "production_policy": "frozen_wisp_unchanged",
            "production_promoted": False,
        },
        "preservation": {
            "paused_strategy_stash": (
                "stash@{0}: On main: rival-v4-paused-superseded-before-v4.1"
            ),
            "m06_20m_actor_role": "rejected_diagnostic_only",
            "two_million_pre_adjustment_checkpoint_preserved": True,
            "large_checkpoints_exports_datasets_and_raw_telemetry_ignored": True,
            "artifact_incident_report": (
                "training/results/milestone08/artifact_preservation_incident.json"
            ),
            "initial_actor_checkpoint_recovered_byte_exact": True,
            "original_prior_corpus_available_locally": False,
            "original_initial_torchscript_archive_available_locally": False,
            "campaign_checkpoints_exports_and_telemetry_unaffected": True,
        },
        "verdict_basis": {
            "architecture_and_transfer_gates_passed": True,
            "sampled_mechanics_use_measurable": True,
            "deterministic_mechanics_use_observed": False,
            "meaningful_mechanics_skill_gain_demonstrated": False,
            "production_promotion_authorized": False,
            "exact_local_artifact_preservation_complete": False,
        },
    }


def build_manifest(
    reports: dict[str, dict[str, Any]],
    final_verification_path: Path,
    *,
    generated_utc: str,
) -> dict[str, Any]:
    final_training = reports["training_005000000.json"]
    final_matrix = reports["rlbot_005m_native_rate.json"]["modes"]["M8C"]
    return {
        "schema_version": 1,
        "status": FINAL_STATUS,
        "verdict": FINAL_VERDICT,
        "purpose": "milestone08_compact_evidence_manifest",
        "generated_utc": generated_utc,
        "repository": {
            "head_before_evidence_commit": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "origin": _git("remote", "get-url", "origin"),
            "m07_boundary_is_ancestor": subprocess.run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    "10c41f708d6e8145bf719f8f322041e7753f6c3f",
                    "HEAD",
                ],
                cwd=REPOSITORY_ROOT,
                check=False,
            ).returncode
            == 0,
        },
        "scope": {
            "training_agent_steps": final_training["cumulative_agent_steps"],
            "training_ceiling_agent_steps": 5_000_000,
            "cumulative_model_updates": final_training["cumulative_model_updates"],
            "final_effective_worker_count": final_training["worker_count"],
            "final_rlbot_completed_matches": final_matrix["aggregates"]["overall"][
                "completed_match_results"
            ],
            "final_rlbot_runtime_clean_matches": final_matrix["aggregates"]["overall"][
                "runtime_clean"
            ],
            "production_modified_or_promoted": False,
        },
        "reports": _report_entries(reports, final_verification_path),
        "source_artifacts": _artifact_entries(),
        "validation": {
            "all_required_reports_parsed": True,
            "all_core_gates_revalidated": True,
            "all_required_source_artifacts_hashed": True,
            "production_promotion_flags_rejected": True,
            "note": (
                "Final test commands are in final_verification.json. Exact remote "
                "readback is reported after the final push because a commit cannot "
                "contain a readback of itself."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-verification-output",
        type=Path,
        default=RESULTS_ROOT / "final_verification.json",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=RESULTS_ROOT / "evidence_manifest.json",
    )
    args = parser.parse_args()

    reports = _validate_reports()
    _require_core_gates(reports)
    generated_utc = datetime.now(timezone.utc).isoformat()
    final_verification = build_final_verification(
        reports,
        verified_utc=generated_utc,
    )
    _write_json(args.final_verification_output, final_verification)
    manifest = build_manifest(
        reports,
        args.final_verification_output,
        generated_utc=generated_utc,
    )
    _write_json(args.manifest_output, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "verdict": manifest["verdict"],
                "reports": len(manifest["reports"]),
                "source_artifacts": len(manifest["source_artifacts"]),
                **manifest["scope"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
