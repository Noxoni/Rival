"""Coordinate Milestone 09 Gate 12 export, CPU probe, and live evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_checkpoint import sha256_file  # noqa: E402
from rival_training.v9_deployment import (  # noqa: E402
    canonical_source_sha256,
    export_v9_deployment,
)


DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "training/checkpoints/milestone09/gate11-20260823T190944Z/resumed"
)
DEFAULT_ARTIFACTS = REPO_ROOT / "training/artifacts/milestone09/gate12"
DEFAULT_CPU_PROBE = (
    REPO_ROOT / "training/results/milestone09/gate12_cpu_runtime_probe.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "training/results/milestone09/gate12_export_live_inference.json"
)
DEFAULT_NATIVE_CORPUS = (
    REPO_ROOT
    / "evidence/raw/rival-v9-native-corpus-20260823T163832Z/native_packets.jsonl"
)
GATE11_EVIDENCE = (
    REPO_ROOT / "training/results/milestone09/gate11_hybrid_ppo.json"
)


def _portable(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _reuse_export(directory: Path) -> dict[str, Any]:
    """Finalize against the exact artifact already exercised by the live smoke."""

    metadata_path = directory.resolve() / "rival_v9_scratch.metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_path = REPO_ROOT / metadata["artifact"]["path"]
    modern_path = REPO_ROOT / metadata["selection"]["torch_export_candidate"]["path"]
    reference_path = REPO_ROOT / metadata["held_corpus"]["path"]
    for path in (selected_path, modern_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"Gate 12 export artifact is missing: {path}")
    return {
        "metadata": metadata,
        "metadata_path": _portable(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "selected_path": _portable(selected_path),
        "modern_path": _portable(modern_path),
        "reference_path": _portable(reference_path),
    }


def _clean_production_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("RIVAL_"):
            environment.pop(name)
    environment.pop("PYTHONPATH", None)
    return environment


def _production_default_probe() -> dict[str, Any]:
    code = (
        "import json,config; "
        "print(json.dumps({'mode':config.POLICY_RUNTIME_MODE,"
        "'tick_skip':config.TICK_SKIP,"
        "'v9_enabled':config.V9_SCRATCH_POLICY_ENABLED,"
        "'candidate_enabled':config.CANDIDATE_POLICY_ENABLED,"
        "'m08_enabled':config.M08_DUAL_RATE_ENABLED,"
        "'policy':config.MODEL_INFO_POLICY.path.name,"
        "'shared_head':config.MODEL_INFO_SHARED_HEAD.path.name}))"
    )
    completed = subprocess.run(
        [str(REPO_ROOT / ".venv/Scripts/python.exe"), "-c", code],
        cwd=REPO_ROOT / "bot",
        env=_clean_production_environment(),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return json.loads(completed.stdout)


def _run_cpu_probe(
    export: dict[str, Any],
    native_corpus: Path,
    output: Path,
) -> dict[str, Any]:
    command = [
        str(REPO_ROOT / ".venv/Scripts/python.exe"),
        str(REPO_ROOT / "training/scripts/run_m09_export_runtime_probe.py"),
        "--model",
        str(REPO_ROOT / export["selected_path"]),
        "--metadata",
        str(REPO_ROOT / export["metadata_path"]),
        "--torch-export",
        str(REPO_ROOT / export["modern_path"]),
        "--reference",
        str(REPO_ROOT / export["reference_path"]),
        "--native-corpus",
        str(native_corpus),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_clean_production_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Gate 12 fresh CPU runtime probe failed:\n"
            + completed.stdout[-4000:]
            + completed.stderr[-4000:]
        )
    return json.loads(output.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--artifact-directory", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--native-corpus", type=Path, default=DEFAULT_NATIVE_CORPUS)
    parser.add_argument("--cpu-probe-output", type=Path, default=DEFAULT_CPU_PROBE)
    parser.add_argument("--live-smoke", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gate11 = json.loads(GATE11_EVIDENCE.read_text(encoding="utf-8"))
    if gate11.get("status") != "passed" or not all(gate11["checks"].values()):
        raise RuntimeError("Gate 11 must pass before Gate 12 export")
    export = (
        _reuse_export(args.artifact_directory)
        if args.live_smoke is not None
        else export_v9_deployment(args.checkpoint, args.artifact_directory)
    )
    cpu_probe = _run_cpu_probe(export, args.native_corpus, args.cpu_probe_output)
    production = _production_default_probe()
    live_smoke = (
        None
        if args.live_smoke is None
        else json.loads(args.live_smoke.read_text(encoding="utf-8"))
    )
    selected_path = REPO_ROOT / export["selected_path"]
    metadata_path = REPO_ROOT / export["metadata_path"]
    live_artifact_matches = bool(
        live_smoke is not None
        and live_smoke.get("model", {}).get("sha256") == sha256_file(selected_path)
        and live_smoke.get("metadata", {}).get("sha256")
        == sha256_file(metadata_path)
    )
    checks = {
        "gate11_passed_before_export": True,
        "held_corpus_export_parity_passed": export["metadata"]["export_parity"][
            "passed"
        ],
        "fresh_cpu_runtime_probe_passed": cpu_probe.get("status") == "passed"
        and all(cpu_probe["checks"].values()),
        "selected_actor_p99_below_2ms_target": cpu_probe["actor_benchmark"][
            "selected_torchscript"
        ]["milliseconds"]["p99"]
        < 2.0,
        "selected_actor_maximum_below_4ms_hard_limit": cpu_probe[
            "actor_benchmark"
        ]["selected_torchscript"]["milliseconds"]["maximum"]
        < 4.0,
        "full_pipeline_p99_below_6ms": cpu_probe["full_pipeline"][
            "external_observation_to_controller_milliseconds"
        ]["p99"]
        < 6.0,
        "live_native_rate_smoke_supplied": live_smoke is not None,
        "live_native_rate_smoke_passed": live_smoke is not None
        and live_smoke.get("status") == "passed"
        and all(live_smoke["checks"].values()),
        "live_smoke_used_exact_export": live_artifact_matches,
        "production_default_is_frozen_wisp": production
        == {
            "mode": "frozen_wisp_production",
            "tick_skip": 8,
            "v9_enabled": False,
            "candidate_enabled": False,
            "m08_enabled": False,
            "policy": "POLICY.lt",
            "shared_head": "SHARED_HEAD.lt",
        },
        "production_promotion_not_authorized": gate11["config"]["pilot"][
            "production_promotion_authorized"
        ]
        is False,
    }
    status = "passed" if all(checks.values()) else (
        "live_smoke_pending"
        if live_smoke is None and all(
            value
            for key, value in checks.items()
            if key not in {
                "live_native_rate_smoke_supplied",
                "live_native_rate_smoke_passed",
                "live_smoke_used_exact_export",
            }
        )
        else "failed"
    )
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 12,
        "gate_name": "export_and_native_120hz_live_inference",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "source_checkpoint": export["metadata"]["source_checkpoint"],
        "export": {
            **export["metadata"],
            "metadata_path": export["metadata_path"],
            "metadata_sha256": export["metadata_sha256"],
        },
        "fresh_cpu_runtime_probe": cpu_probe,
        "native_rlbot_process_smoke": live_smoke,
        "production_default_probe": production,
        "gate_semantics": {
            "score_used": False,
            "win_loss_used": False,
            "five_x_used_for_native_certification": False,
            "production_promoted": False,
            "training_agent_steps_consumed": 0,
            "selected_format": "TorchScript frozen optimize_for_inference",
            "format_decision_basis": (
                "Both formats met parity; the selected TorchScript seam had lower "
                "fresh-process Windows CPU p99/maximum latency and simpler eval loading."
            ),
        },
        "production_model_hashes": {
            "POLICY.lt": sha256_file(REPO_ROOT / "bot/models/POLICY.lt"),
            "SHARED_HEAD.lt": sha256_file(REPO_ROOT / "bot/models/SHARED_HEAD.lt"),
        },
        "source_hashes": {
            "gate_script_sha256": _source_hash(Path(__file__)),
            "deployment_source_sha256": canonical_source_sha256(
                REPO_ROOT / "training/rival_training/v9_deployment.py"
            ),
            "cpu_probe_source_sha256": _source_hash(
                REPO_ROOT / "training/scripts/run_m09_export_runtime_probe.py"
            ),
            "live_smoke_source_sha256": _source_hash(
                REPO_ROOT / "training/scripts/run_m09_scratch_live_smoke.py"
            ),
            "live_runtime_source_sha256": _source_hash(
                REPO_ROOT / "bot/v9_scratch_runtime.py"
            ),
            "bot_source_sha256": _source_hash(REPO_ROOT / "bot/bot.py"),
            "config_source_sha256": _source_hash(REPO_ROOT / "bot/config.py"),
            "gate11_evidence_sha256": sha256_file(GATE11_EVIDENCE),
        },
        "commands": {
            "offline_export_and_probe": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_export_gate.py"
            ),
            "live_smoke": (
                ".venv/Scripts/python.exe "
                "training/scripts/run_m09_scratch_live_smoke.py "
                "--model training/artifacts/milestone09/gate12/rival_v9_scratch.ts "
                "--metadata training/artifacts/milestone09/gate12/"
                "rival_v9_scratch.metadata.json"
            ),
            "finalize_with_live_smoke": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_export_gate.py --live-smoke "
                "training/results/milestone09/gate12_scratch_live_smoke.json"
            ),
        },
    }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "passed" else (2 if status == "live_smoke_pending" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
