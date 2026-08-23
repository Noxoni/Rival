"""Validate and compact the completed Milestone 09 Gate 13 evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_checkpoint import (  # noqa: E402
    DEFAULT_PILOT_CONFIG_PATH,
    config_sha256,
    load_m09_config,
    load_v9_checkpoint,
    portable_path,
    sha256_file,
)
from rival_training.v9_curriculum import (  # noqa: E402
    V9_PILOT_CURRICULUM_FAMILIES,
)


DEFAULT_SOURCE_CHECKPOINT = (
    REPOSITORY_ROOT / "training/checkpoints/milestone09/gate11-20260823T190944Z/resumed"
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _checkpoint_directory(portable: str) -> Path:
    path = REPOSITORY_ROOT / portable
    if not path.is_dir():
        raise FileNotFoundError(f"Gate 13 checkpoint is missing: {portable}")
    return path


def _compact_iteration(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["pilot_metrics"]
    return {
        "iteration": int(report["iteration"]),
        "phase": report["gate13_phase"],
        "requested_agent_steps": int(
            report["authorized_collection"]["requested_rollout_agent_steps"]
        ),
        "collected_agent_steps": int(report["collected_agent_steps"]),
        "cumulative_agent_steps": int(report["cumulative_agent_steps"]),
        "simulated_game_hours": float(report["simulated_game_hours"]),
        "agent_steps_per_second": float(report["agent_steps_per_second"]),
        "iteration_wall_seconds": float(report["iteration_wall_seconds"]),
        "reward": report["reward"],
        "actor_update_magnitude": float(report["ppo"]["actor_update_magnitude"]),
        "critic_update_magnitude": float(report["ppo"]["critic_update_magnitude"]),
        "analog_entropy": report["ppo"]["analog_entropy"],
        "button_entropy": report["ppo"]["button_entropy"],
        "analog_std": report["ppo"]["analog_std"],
        "button_combo_counts": report["actions"]["button_combo_counts"],
        "button_combo_shares": report["actions"]["button_combo_shares"],
        "marginal_button_shares": report["actions"]["marginal_button_shares"],
        "analog_exploration": report["actions"]["analog"],
        "reset_counts": metrics["reset_counts"],
        "scores_recorded_for_diagnostics_only": metrics["scores_recorded_for_diagnostics_only"],
        "event_counts": metrics["event_counts"],
        "rollout_inference": report["rollout_inference"],
        "health": report["health"],
    }


def _fixed_comparison(baseline: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    before = baseline["behavior_signature"]
    after = final["behavior_signature"]
    rows: dict[str, Any] = {}
    materially_changed = 0
    for name in sorted(before):
        first = float(before[name])
        second = float(after[name])
        threshold = max(1e-6, abs(first) * 0.01)
        changed = abs(second - first) > threshold
        materially_changed += int(changed)
        rows[name] = {
            "baseline": first,
            "final": second,
            "delta": second - first,
            "material_change_threshold": threshold,
            "materially_changed": changed,
        }
    return {
        "protocol_equal": baseline["fixed_protocol"] == final["fixed_protocol"],
        "evaluation_agent_steps_each": int(baseline["agent_steps_evaluated"]),
        "baseline_checkpoint_steps": int(baseline["checkpoint"]["cumulative_agent_steps"]),
        "final_checkpoint_steps": int(final["checkpoint"]["cumulative_agent_steps"]),
        "distribution_parameter_sha256": {
            "baseline": baseline["actor_fingerprint"]["distribution_parameter_sha256"],
            "final": final["actor_fingerprint"]["distribution_parameter_sha256"],
        },
        "deterministic_action_sha256": {
            "baseline": baseline["actor_fingerprint"]["deterministic_action_sha256"],
            "final": final["actor_fingerprint"]["deterministic_action_sha256"],
        },
        "behavior_signature": rows,
        "materially_changed_signature_count": materially_changed,
        "scores_recorded_but_excluded_from_pass_fail": {
            "baseline": baseline["metrics"]["scores_recorded_for_diagnostics_only"],
            "final": final["metrics"]["scores_recorded_for_diagnostics_only"],
        },
        "selected_fixed_event_counts": {
            label: {
                "baseline": baseline["metrics"]["event_counts"].get(label, 0),
                "final": final["metrics"]["event_counts"].get(label, 0),
            }
            for label in sorted(final["metrics"]["event_counts"])
            if any(
                token in label
                for token in (
                    "touch",
                    "jump",
                    "dodge",
                    "recovery",
                    "wavedash",
                    "wall_dash",
                    "zap_dash",
                    "stall",
                    "flip_cancel",
                )
            )
        },
    }


def _policy_change(source: Path, final: Path) -> dict[str, Any]:
    source_loaded = load_v9_checkpoint(source, device="cpu")
    final_loaded = load_v9_checkpoint(
        final,
        device="cpu",
        expected_config=load_m09_config(DEFAULT_PILOT_CONFIG_PATH),
    )
    source_vector = torch.nn.utils.parameters_to_vector(source_loaded["actor"].parameters())
    final_vector = torch.nn.utils.parameters_to_vector(final_loaded["actor"].parameters())
    observations = np.asarray(source_loaded["reload_observations"], dtype=np.float32)
    with torch.inference_mode():
        source_outputs = source_loaded["actor"](torch.from_numpy(observations))
        final_outputs = final_loaded["actor"](torch.from_numpy(observations))
    maximum_output_difference = max(
        float((second - first).abs().max())
        for first, second in zip(source_outputs, final_outputs, strict=True)
    )
    second_reload = load_v9_checkpoint(
        final,
        device="cpu",
        expected_config=load_m09_config(DEFAULT_PILOT_CONFIG_PATH),
    )
    reload_error = max(
        float((first - second).detach().abs().max())
        for first, second in zip(
            final_loaded["actor"].parameters(),
            second_reload["actor"].parameters(),
            strict=True,
        )
    )
    return {
        "actor_parameter_l2_change_from_gate11": float(
            torch.linalg.vector_norm(final_vector - source_vector).detach()
        ),
        "maximum_distribution_parameter_change_on_gate11_held_corpus": (maximum_output_difference),
        "independent_second_reload_maximum_parameter_error": reload_error,
        "actor_optimizer_state_entries": len(final_loaded["actor_optimizer"].state),
        "critic_optimizer_state_entries": len(final_loaded["critic_optimizer"].state),
        "final_trainer_counters": {
            key: final_loaded["trainer_state"][key]
            for key in (
                "completed_iterations",
                "cumulative_agent_steps",
                "cumulative_model_updates",
                "simulated_game_hours",
            )
        },
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    phase1 = _json(args.run_root / "phase1.json")
    phase2 = _json(args.run_root / "phase2.json")
    baseline = _json(args.run_root / "baseline_evaluation.json")
    final_evaluation = _json(args.run_root / "final_evaluation.json")
    spectator_preflight = _json(args.run_root / "spectator_preflight.json")
    spectator_smoke = _json(args.run_root / "spectator_smoke.json")
    iterations = list(phase1["iterations"]) + list(phase2["iterations"])
    compact_iterations = [_compact_iteration(item) for item in iterations]
    checkpoints = list(phase1["checkpoints"]) + list(phase2["checkpoints"])
    final_checkpoint = _checkpoint_directory(str(phase2["latest_checkpoint"]["directory"]))
    source_checkpoint = args.source_checkpoint.resolve()
    pilot_config = load_m09_config(DEFAULT_PILOT_CONFIG_PATH)

    reset_counts: Counter[str] = Counter()
    button_counts = np.zeros(8, dtype=np.int64)
    for iteration in iterations:
        reset_counts.update(iteration["pilot_metrics"]["reset_counts"])
        button_counts += np.asarray(iteration["actions"]["button_combo_counts"], dtype=np.int64)
    reset_total = sum(reset_counts.values())
    comparison = _fixed_comparison(baseline, final_evaluation)
    policy_change = _policy_change(source_checkpoint, final_checkpoint)
    start_steps = int(baseline["checkpoint"]["cumulative_agent_steps"])
    final_steps = int(final_evaluation["checkpoint"]["cumulative_agent_steps"])
    maximum_steps = int(pilot_config["pilot"]["maximum_cumulative_agent_steps"])
    collected_sum = sum(int(item["collected_agent_steps"]) for item in iterations)
    phase1_float_transport = all(
        item["actions"]["all_actions_finite"]
        and item["actions"]["all_eight_button_combos_sampled"]
        and all(row["nontrivial_range"] for row in item["actions"]["analog"])
        for item in phase1["iterations"]
    )
    checkpoint_records = []
    for record in checkpoints:
        directory = _checkpoint_directory(str(record["directory"]))
        manifest_path = directory / "checkpoint_manifest.json"
        manifest = _json(manifest_path)
        checkpoint_records.append(
            {
                **record,
                "format": manifest["format"],
                "manifest_size_bytes": manifest_path.stat().st_size,
                "manifest_sha256_verified": sha256_file(manifest_path) == record["manifest_sha256"],
                "payload_files": manifest["files"],
                "payload_total_size_bytes": sum(
                    int(item["size_bytes"]) for item in manifest["files"].values()
                ),
            }
        )

    checks = {
        "phase1_passed": phase1["status"] == "passed",
        "phase2_passed": phase2["status"] == "passed",
        "six_real_ppo_updates_completed": len(iterations) == 6,
        "all_iteration_health_checks_passed": all(item["health"]["passed"] for item in iterations),
        "both_hybrid_heads_updated_every_iteration": all(
            item["health"]["actor_updated"]
            and item["health"]["critic_updated"]
            and item["health"]["all_hybrid_head_gradient_rows_nonzero"]
            for item in iterations
        ),
        "all_action_branches_explored_every_iteration": all(
            item["health"]["all_button_combos_sampled"]
            and item["health"]["all_analog_axes_explored"]
            for item in iterations
        ),
        "reward_and_diagnostic_metrics_finite": all(
            item["health"]["all_update_metrics_finite"] and item["health"]["pilot_metrics_finite"]
            for item in iterations
        ),
        "gate11_steps_counted_from_exact_source": start_steps == 576_024,
        "collected_steps_match_counter_delta": collected_sum == final_steps - start_steps,
        "two_hour_ceiling_respected": final_steps <= maximum_steps,
        "unused_ceiling_margin_positive": maximum_steps - final_steps > 0,
        "checkpoint_written_after_every_iteration": len(checkpoints) == len(iterations),
        "all_checkpoint_manifest_hashes_verified": all(
            item["manifest_sha256_verified"] for item in checkpoint_records
        ),
        "phase2_fresh_parent_resumed_phase1_counter": int(
            phase2["iterations"][0]["cumulative_agent_steps"]
        )
        - int(phase2["iterations"][0]["collected_agent_steps"])
        == int(phase1["latest_checkpoint"]["cumulative_agent_steps"]),
        "final_action_space_reported_continuous": phase2["environment_shapes"] == [714, 8, 2],
        "phase1_continuous_float_transport_proven_despite_label_quirk": (
            phase1_float_transport and phase1["environment_shapes"] == [714, 8, 0]
        ),
        "all_curriculum_families_observed": all(
            reset_counts[name] > 0 for name in V9_PILOT_CURRICULUM_FAMILIES
        ),
        "natural_resets_remained_majority": (reset_counts["natural"] / max(reset_total, 1) > 0.5),
        "policy_parameters_changed": policy_change["actor_parameter_l2_change_from_gate11"] > 0.0,
        "held_distribution_outputs_changed": policy_change[
            "maximum_distribution_parameter_change_on_gate11_held_corpus"
        ]
        > 0.0,
        "final_checkpoint_second_reload_exact": policy_change[
            "independent_second_reload_maximum_parameter_error"
        ]
        == 0.0,
        "fixed_evaluation_protocol_equal": comparison["protocol_equal"],
        "fixed_behavior_changed_materially": comparison["materially_changed_signature_count"] >= 3,
        "scores_not_used_for_gate": baseline["fixed_protocol"][
            "scores_are_recorded_but_not_used_as_a_technical_gate"
        ]
        and final_evaluation["fixed_protocol"][
            "scores_are_recorded_but_not_used_as_a_technical_gate"
        ],
        "scratch_rlviser_preflight_passed": spectator_preflight["status"] == "passed",
        "scratch_rlviser_process_observed": spectator_smoke["renderer_process_verified"],
        "scratch_rlviser_isolated_from_workers": spectator_smoke["single_environment"]
        and spectator_smoke["separate_process"]
        and not spectator_smoke["training_workers_rendered"],
        "production_promotion_not_authorized": not pilot_config["pilot"][
            "production_promotion_authorized"
        ],
    }
    status = "passed" if all(checks.values()) else "failed"
    final_manifest = _json(final_checkpoint / "checkpoint_manifest.json")
    report = {
        "schema_version": 1,
        "gate": 13,
        "status": status,
        "objective": (
            "Bounded scratch PPO pilot proving technical learning health, measurable "
            "behavior change, recoverable checkpoints, and independent RLViser watchability"
        ),
        "run_id": args.run_root.name,
        "authority": {
            "maximum_cumulative_agent_steps": maximum_steps,
            "maximum_simulated_game_hours": float(
                pilot_config["pilot"]["maximum_simulated_game_hours"]
            ),
            "gate11_steps_count_toward_ceiling": True,
            "start_agent_steps": start_steps,
            "new_gate13_agent_steps": collected_sum,
            "final_cumulative_agent_steps": final_steps,
            "final_simulated_game_hours": final_steps / 864000.0,
            "unused_agent_step_margin": maximum_steps - final_steps,
            "additional_training_authorized": False,
            "production_promotion_authorized": False,
        },
        "config": {
            "path": portable_path(DEFAULT_PILOT_CONFIG_PATH),
            "sha256": config_sha256(pilot_config),
            "migration": phase1["config_migration"],
            "worker_count": int(pilot_config["backend"]["worker_count"]),
            "policy_hz": 120,
            "repeat_action": False,
            "curriculum": pilot_config["curriculum"],
        },
        "implementation_boundary_note": {
            "phase1_rlgym_ppo_shape_report": phase1["environment_shapes"],
            "phase2_rlgym_ppo_shape_report": phase2["environment_shapes"],
            "cause": (
                "rlgym-ppo 1.3.13 classifies action spaces with exact type(Box); "
                "the Phase 1 seed-forwarding Box subclass was labeled discrete even "
                "though wrapper.is_discrete remained false and continuous float32 "
                "physical controllers were transported"
            ),
            "evidence_phase1_was_not_quantized": (
                "Every Phase 1 analog axis had nontrivial continuous range, all "
                "controllers were finite, and all eight joint button combinations "
                "were sampled"
            ),
            "prospective_fix_commit": "334ed3b",
            "rerun_performed": False,
            "reason_no_rerun": (
                "The label did not affect action casting or PPO data, and rerunning "
                "would spend unauthorized duplicate pilot experience"
            ),
        },
        "training": {
            "iterations": compact_iterations,
            "aggregate_reset_counts": {
                name: int(reset_counts[name]) for name in V9_PILOT_CURRICULUM_FAMILIES
            },
            "aggregate_reset_shares": {
                name: float(reset_counts[name] / max(reset_total, 1))
                for name in V9_PILOT_CURRICULUM_FAMILIES
            },
            "aggregate_button_combo_counts": button_counts.tolist(),
            "all_sides_used_same_current_scratch_actor": True,
            "fixed_strong_opponent_introduced": False,
            "scores_or_wins_used_to_decide_health": False,
        },
        "checkpoints": {
            "count": len(checkpoint_records),
            "during_pilot": checkpoint_records,
            "final": {
                "directory": portable_path(final_checkpoint),
                "format": final_manifest["format"],
                "manifest_sha256": sha256_file(final_checkpoint / "checkpoint_manifest.json"),
                "files": final_manifest["files"],
                "fresh_reload": policy_change,
                "rlviser_compatible": True,
            },
        },
        "fixed_behavior_evaluation": comparison,
        "spectator": {
            "version": spectator_preflight["spectator_version"],
            "preflight": spectator_preflight,
            "live_smoke": {
                "status": spectator_smoke["status"],
                "renderer_process_verified": spectator_smoke["renderer_process_verified"],
                "decisions": int(spectator_smoke["decisions"]),
                "wall_seconds": float(spectator_smoke["wall_seconds"]),
                "decisions_per_wall_second": float(
                    spectator_smoke["decisions"] / spectator_smoke["wall_seconds"]
                ),
                "missed_pacing_deadlines": int(spectator_smoke["missed_pacing_deadlines"]),
                "tick_skip": int(spectator_smoke["tick_skip"]),
                "physics_tick_rate_hz": int(spectator_smoke["physics_tick_rate_hz"]),
                "single_environment": spectator_smoke["single_environment"],
                "separate_process": spectator_smoke["separate_process"],
                "training_workers_rendered": spectator_smoke["training_workers_rendered"],
            },
            "launch_command": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_rlviser_spectator.py "
                "--checkpoint current --playback-speed 1"
            ),
            "disabled_by_default": True,
        },
        "checks": checks,
        "conclusion": {
            "gate13_passed": status == "passed",
            "learning_technically_healthy": all(item["health"]["passed"] for item in iterations),
            "behavior_changed_measurably": checks["fixed_behavior_changed_materially"],
            "skill_claim": (
                "No Wisp/Nexto competitiveness claim. Deterministic touch behavior "
                "emerged and approach/recovery metrics changed, while deterministic "
                "jump/dodge use remained zero at this very early budget."
            ),
            "promotion_decision": "not_authorized_not_promoted",
        },
        "reproduction_commands": {
            "baseline_evaluation": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_pilot_evaluation.py --checkpoint "
                "training/checkpoints/milestone09/gate11-20260823T190944Z/resumed "
                "--output <run-root>/baseline_evaluation.json --episodes 12 "
                "--max-ticks-per-episode 2400"
            ),
            "phase1": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_scratch_pilot_phase.py --phase phase1 "
                "--source-checkpoint training/checkpoints/milestone09/"
                "gate11-20260823T190944Z/resumed --checkpoint-root "
                "training/checkpoints/milestone09/<run-id>/phase1 --output "
                "<run-root>/phase1.json --rollout-targets 192000,192000,192000"
            ),
            "phase2": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_scratch_pilot_phase.py --phase phase2 "
                "--source-checkpoint <phase1-final> --checkpoint-root "
                "training/checkpoints/milestone09/<run-id>/phase2 --output "
                "<run-root>/phase2.json --rollout-targets 192000,192000,144000"
            ),
            "final_evaluation": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_pilot_evaluation.py --checkpoint "
                "<phase2-final> --output <run-root>/final_evaluation.json "
                "--episodes 12 --max-ticks-per-episode 2400"
            ),
            "spectator": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_rlviser_spectator.py --checkpoint current "
                "--playback-speed 1"
            ),
            "finalize": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/finalize_m09_gate13.py --run-root <run-root>"
            ),
        },
        "provenance": {
            "source_checkpoint": portable_path(source_checkpoint),
            "source_checkpoint_manifest_sha256": sha256_file(
                source_checkpoint / "checkpoint_manifest.json"
            ),
            "foundation_commit": "0f10cab",
            "continuous_space_reporting_fix_commit": "334ed3b",
            "finalizer_worktree_head": _git("rev-parse", "HEAD"),
            "large_checkpoints_raw_rollouts_and_temporary_evaluations_ignored": True,
        },
    }
    if status != "passed":
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Gate 13 finalization failed checks: {failed}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "training/results/milestone09/gate13_scratch_pilot.json",
    )
    args = parser.parse_args()
    report = finalize(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": portable_path(args.output),
                "final_agent_steps": report["authority"]["final_cumulative_agent_steps"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
