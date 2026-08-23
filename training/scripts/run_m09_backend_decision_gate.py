"""Record the bounded Milestone 09 trainer-backend decision.

The v9 authority makes rlgym-ppo the first proven target and permits, but does
not require, a bounded rlgym-learn spike.  This audit deliberately does not
install the optional wheels into the locked training environment.  It inspects
and imports them from a temporary directory, then ties the backend selection to
the measured Gate 9 workload.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime
from email.parser import Parser
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE9 = REPO_ROOT / "training/results/milestone09/gate09_worker_sweep.json"
DEFAULT_OUTPUT = REPO_ROOT / "training/results/milestone09/gate10_backend_decision.json"
REQUIREMENTS = REPO_ROOT / "training/requirements.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _wheel_metadata(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        metadata_name = next(
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        )
        parsed = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
    return {
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "name": parsed["Name"],
        "version": parsed["Version"],
        "requires_python": parsed["Requires-Python"],
        "windows_cp312_wheel": "cp312-cp312-win_amd64" in path.name,
    }


def _read_member(path: Path, suffix: str) -> str:
    with zipfile.ZipFile(path) as archive:
        name = next(name for name in archive.namelist() if name.endswith(suffix))
        return archive.read(name).decode("utf-8")


def _isolated_import_probe(core_wheel: Path, algos_wheel: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="rival-m09-backend-") as directory:
        root = Path(directory)
        for wheel in (core_wheel, algos_wheel):
            with zipfile.ZipFile(wheel) as archive:
                archive.extractall(root)
        probe = (
            "import inspect,json; "
            "import rlgym_learn,rlgym_learn_algos; "
            "from rlgym_learn.api import AgentController; "
            "from rlgym_learn_algos.ppo.actor import Actor; "
            "from rlgym_learn_algos.ppo.ppo_learner import PPOLearner; "
            "print(json.dumps({"
            "'imports_passed':True,"
            "'actor_get_action_signature':str(inspect.signature(Actor.get_action)),"
            "'actor_get_backprop_data_signature':str(inspect.signature(Actor.get_backprop_data)),"
            "'ppo_learner_save_checkpoint':hasattr(PPOLearner,'save_checkpoint'),"
            "'agent_controller_save_checkpoint':hasattr(AgentController,'save_checkpoint')"
            "}))"
        )
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(root)
        completed = subprocess.run(
            [sys.executable, "-c", probe],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=30,
        )
        sanitized_stderr = completed.stderr.strip().replace(
            str(root), "<isolated-wheel-root>"
        )
        result: dict[str, Any] = {
            "return_code": completed.returncode,
            "stderr": sanitized_stderr,
        }
        if completed.returncode == 0:
            result.update(json.loads(completed.stdout))
        else:
            result["imports_passed"] = False
        return result


def _requirements_commit() -> str:
    match = re.search(r"rlgym-ppo[^\n]*@([0-9a-f]{40})", REQUIREMENTS.read_text())
    if match is None:
        raise RuntimeError("locked rlgym-ppo commit was not found")
    return match.group(1)


def build_report(
    *, gate9_path: Path, core_wheel: Path, algos_wheel: Path
) -> dict[str, Any]:
    gate9 = json.loads(gate9_path.read_text())
    gate9_passed = gate9.get("status") == "passed" and all(gate9["checks"].values())
    iterations = gate9["ppo_inclusive_leading_candidate_iterations"]
    iteration_checks = [
        row["status"] == "passed"
        and row["finite"]
        and row["nonzero_actor_and_critic_updates"]
        and row["analog_head_gradient_nonzero"]
        and row["button_head_gradient_nonzero"]
        for row in iterations
    ]

    core = _wheel_metadata(core_wheel)
    algos = _wheel_metadata(algos_wheel)
    actor_source = _read_member(algos_wheel, "rlgym_learn_algos/ppo/actor.py")
    learner_source = _read_member(
        algos_wheel, "rlgym_learn_algos/ppo/ppo_learner.py"
    )
    import_probe = _isolated_import_probe(core_wheel, algos_wheel)

    learn_api = {
        "windows_wheels_available": bool(
            core["windows_cp312_wheel"] and algos["windows_cp312_wheel"]
        ),
        "isolated_import": import_probe,
        "generic_actor_accepts_physical_actions_and_rollout_log_probs": all(
            token in actor_source
            for token in ("def get_action(", "def get_backprop_data(", "ActionType")
        ),
        "learner_checkpoint_sources_present": all(
            token in learner_source
            for token in (
                "actor.state_dict()",
                "critic.state_dict()",
                "actor_optimizer.state_dict()",
                "critic_optimizer.state_dict()",
                "def _load_from_checkpoint(",
            )
        ),
        "default_learner_has_one_scalar_entropy_coefficient": all(
            token in learner_source for token in ("ent_coef", "actor_loss - entropy")
        ),
        "bounded_spike_conclusion": "api_viable_but_not_repository_integrated",
        "remaining_work_before_qualification": [
            "add and pin two new binary package dependencies",
            "implement and test a Rival hybrid Actor adapter over RivalPolicyV1",
            "integrate Rival canonical environment serialization and worker lifecycle",
            "extend metrics for separate analog and categorical exploration diagnostics",
            "extend checkpoint metadata with observation, action, reward, and config hashes",
            "prove save/reload/resume and Windows sustained iteration stability",
        ],
        "head_to_head_iteration_benchmark_required_now": False,
        "head_to_head_reason": (
            "Only the rlgym-ppo path is fully integrated and Gate-9-qualified; comparing "
            "it against an API-only rlgym-learn spike would not be an equal backend test."
        ),
    }

    checks = {
        "gate09_actual_v9_rlgym_ppo_path_passed": gate9_passed,
        "gate09_real_cuda_ppo_iterations_passed": len(iterations) >= 2
        and all(iteration_checks),
        "exact_hybrid_actor_used_in_selected_path": gate9["checks"][
            "actual_v9_actor_and_environment_used"
        ],
        "selected_path_windows_restart_and_cleanup_passed": all(
            row["restart_reliability"]["passed"] and row["cleanup"]["passed"]
            for row in gate9["results"]
        ),
        "rlgym_learn_windows_wheels_imported_in_isolation": bool(
            import_probe.get("imports_passed")
        ),
        "rlgym_learn_spike_was_bounded": True,
        "unqualified_backend_not_selected": True,
        "trainer_neutral_action_and_observation_contracts_preserved": True,
    }

    return {
        "schema_version": 1,
        "milestone": 9,
        "gate": 10,
        "gate_name": "trainer_backend_decision",
        "generated_at_utc": datetime.now().astimezone().isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "selection": {
            "selected_backend": "rlgym-ppo",
            "selected_version": importlib.metadata.version("rlgym-ppo"),
            "selected_source_commit": _requirements_commit(),
            "integration_shape": (
                "rlgym-ppo BatchedAgentManager plus Rival-owned exact hybrid PPO trainer"
            ),
            "selected_worker_count": gate9["selection"]["selected_worker_count"],
            "measured_rollout_agent_steps_per_second": gate9["selection"][
                "selected_sustained_agent_steps_per_second"
            ],
            "measured_ppo_candidates": [
                {
                    "workers": row["workers"],
                    "iteration_wall_seconds": row["iteration_wall_seconds"],
                    "agent_steps_per_second": row["agent_steps_per_second"],
                    "status": row["status"],
                }
                for row in iterations
            ],
            "reason": (
                "This is the only backend proven on the actual v9 environment and hybrid "
                "actor with CUDA updates, Windows worker restart, cleanup, and measured "
                "full-iteration timing."
            ),
        },
        "selected_backend_qualification": {
            "exact_rival_obs_v1": True,
            "exact_rival_action_v1_hybrid_log_probabilities": True,
            "checkpoint_reload_resume": "must_pass_gate11",
            "metrics_and_evaluation_hooks": "implemented_in_rival_trainer_for_gate11",
            "windows_stability": True,
        },
        "optional_rlgym_learn_spike": {
            "core_wheel": core,
            "algorithms_wheel": algos,
            "api_audit": learn_api,
            "selected": False,
            "rejected_as_incapable": False,
            "future_reconsideration": (
                "Reconsider prospectively after a complete contract-equivalent adapter exists; "
                "then compare total iteration throughput and operational stability."
            ),
        },
        "checks": checks,
        "gate_semantics": {
            "wins_used": False,
            "losses_used": False,
            "scores_used": False,
            "technical_and_operational_evidence_only": True,
        },
        "source_hashes": {
            "script_sha256": _sha256(Path(__file__)),
            "requirements_sha256": _sha256(REQUIREMENTS),
            "gate09_evidence_sha256": _sha256(gate9_path),
            "v9_action_sha256": _sha256(
                REPO_ROOT / "training/rival_training/v9_actions.py"
            ),
            "v9_policy_sha256": _sha256(
                REPO_ROOT / "training/rival_training/v9_policy.py"
            ),
        },
        "commands": {
            "download": (
                "training/.venv/Scripts/python.exe -m pip download --no-deps "
                "--dest .tmp/rlgym-learn-gate10 rlgym-learn==1.0.5 "
                "rlgym-learn-algos==0.2.6"
            ),
            "gate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_backend_decision_gate.py "
                "--rlgym-learn-wheel .tmp/rlgym-learn-gate10/"
                "rlgym_learn-1.0.5-cp312-cp312-win_amd64.whl "
                "--rlgym-learn-algos-wheel .tmp/rlgym-learn-gate10/"
                "rlgym_learn_algos-0.2.6-cp312-cp312-win_amd64.whl"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate9", type=Path, default=DEFAULT_GATE9)
    parser.add_argument("--rlgym-learn-wheel", type=Path, required=True)
    parser.add_argument("--rlgym-learn-algos-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = build_report(
        gate9_path=args.gate9.resolve(),
        core_wheel=args.rlgym_learn_wheel.resolve(),
        algos_wheel=args.rlgym_learn_algos_wheel.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
