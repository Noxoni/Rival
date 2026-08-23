"""Fresh CPU-production-environment probe for the Rival v9 export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
for source in (REPO_ROOT / "bot", REPO_ROOT / "training"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from rival_training.v9_rlbot_corpus import snapshot_to_rlbot_sources  # noqa: E402
from v9_scratch_runtime import (  # noqa: E402
    RivalV9ScratchRuntime,
    validate_runtime_constants,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _portable(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty timing series")
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        return float(ordered[round((len(ordered) - 1) * fraction)])

    return {
        "samples": len(values),
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "maximum": float(ordered[-1]),
    }


def _as_arrays(outputs: Any) -> tuple[np.ndarray, ...]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
        raise RuntimeError("Deployment export did not return four tensors")
    return tuple(
        np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        for value in outputs
    )


def _reference(reference: np.lib.npyio.NpzFile) -> tuple[np.ndarray, ...]:
    return tuple(
        np.ascontiguousarray(reference[name], dtype=np.float32)
        for name in ("analog_mean", "analog_log_std", "button_logits", "controller")
    )


def _maximum_errors(
    expected: tuple[np.ndarray, ...],
    actual: tuple[np.ndarray, ...],
) -> dict[str, float]:
    return {
        name: float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        for name, left, right in zip(
            ("analog_mean", "analog_log_std", "button_logits", "controller"),
            expected,
            actual,
        )
    }


def _legal_controllers(values: np.ndarray) -> bool:
    return bool(
        values.ndim == 2
        and values.shape[1] == 8
        and np.isfinite(values).all()
        and np.all(values[:, :5] >= -1.0)
        and np.all(values[:, :5] <= 1.0)
        and np.all(np.isin(values[:, 5:], (0.0, 1.0)))
    )


def _actor_benchmark(
    model: Any,
    observations: torch.Tensor,
    *,
    samples: int,
    warmup: int,
) -> dict[str, Any]:
    timings: list[float] = []
    outputs = None
    with torch.inference_mode():
        for index in range(warmup + samples):
            row = observations[index % len(observations) : index % len(observations) + 1]
            started = time.perf_counter_ns()
            outputs = model(row)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if index >= warmup:
                timings.append(elapsed)
    arrays = _as_arrays(outputs)
    return {
        "milliseconds": _stats(timings),
        "all_outputs_finite": all(np.isfinite(value).all() for value in arrays),
        "warmup_calls_excluded": warmup,
    }


def _load_corpus(path: Path, maximum: int) -> list[tuple[Any, Any, int]]:
    sources: list[tuple[Any, Any, int]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            value = json.loads(line)
            if value.get("record_type") != "rival_v9_native_packet":
                continue
            sources.append(snapshot_to_rlbot_sources(value["packet"]))
            if len(sources) >= maximum:
                break
    if len(sources) != maximum:
        raise RuntimeError(f"Expected {maximum} native packet records, got {len(sources)}")
    return sources


def _pipeline_probe(
    model_path: Path,
    metadata_path: Path,
    native_corpus: Path,
    *,
    samples: int,
    warmup: int,
) -> dict[str, Any]:
    runtime = RivalV9ScratchRuntime(model_path, metadata_path)
    sources = _load_corpus(native_corpus, samples + warmup)
    external_total: list[float] = []
    for index, (packet, field_info, self_index) in enumerate(sources):
        started = time.perf_counter_ns()
        runtime.step(packet, self_index=self_index, field_info=field_info)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        if index >= warmup:
            external_total.append(elapsed)
    runtime_summary = runtime.summary()
    component_timings = {
        name: _stats(values[warmup:])
        for name, values in runtime.pipeline_milliseconds.items()
    }
    return {
        "samples": samples,
        "warmup_packets_excluded": warmup,
        "external_observation_to_controller_milliseconds": _stats(external_total),
        "component_milliseconds": component_timings,
        "runtime": runtime_summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--torch-export", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native-corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--actor-samples", type=int, default=3000)
    parser.add_argument("--pipeline-samples", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.set_num_threads(1)
    reference = np.load(args.reference, allow_pickle=False)
    observations_array = np.ascontiguousarray(
        reference["observations"], dtype=np.float32
    )
    expected = _reference(reference)
    observations = torch.from_numpy(observations_array)

    runtime = RivalV9ScratchRuntime(args.model, args.metadata)
    with torch.inference_mode():
        selected_outputs = _as_arrays(runtime.model(observations))
    selected_errors = _maximum_errors(expected, selected_outputs)

    modern = torch.export.load(args.torch_export).module()
    modern_rows: list[list[np.ndarray]] = [[], [], [], []]
    with torch.inference_mode():
        for index in range(len(observations)):
            outputs = _as_arrays(modern(observations[index : index + 1]))
            for target, value in zip(modern_rows, outputs):
                target.append(value)
    modern_outputs = tuple(np.concatenate(rows, axis=0) for rows in modern_rows)
    modern_errors = _maximum_errors(expected, modern_outputs)

    selected_actor = _actor_benchmark(
        runtime.model,
        observations,
        samples=args.actor_samples,
        warmup=args.warmup,
    )
    modern_actor = _actor_benchmark(
        modern,
        observations,
        samples=args.actor_samples,
        warmup=args.warmup,
    )
    pipeline = _pipeline_probe(
        args.model,
        args.metadata,
        args.native_corpus,
        samples=args.pipeline_samples,
        warmup=args.warmup,
    )
    checks = {
        "selected_export_finite": all(
            np.isfinite(value).all() for value in selected_outputs
        ),
        "modern_export_finite": all(
            np.isfinite(value).all() for value in modern_outputs
        ),
        "selected_export_parity_at_1e_5": max(selected_errors.values()) <= 1e-5,
        "modern_export_parity_at_1e_5": max(modern_errors.values()) <= 1e-5,
        "selected_controllers_legal": _legal_controllers(selected_outputs[3]),
        "modern_controllers_legal": _legal_controllers(modern_outputs[3]),
        "actor_cpu_p99_below_2ms_target": selected_actor["milliseconds"]["p99"]
        < 2.0,
        "actor_cpu_maximum_below_4ms_hard_limit": selected_actor["milliseconds"][
            "maximum"
        ]
        < 4.0,
        "full_pipeline_cpu_p99_below_6ms": pipeline[
            "external_observation_to_controller_milliseconds"
        ]["p99"]
        < 6.0,
        "runtime_no_nonfinite_outputs": pipeline["runtime"]["non_finite_outputs"]
        == 0,
        "runtime_no_illegal_controllers": pipeline["runtime"]["illegal_controllers"]
        == 0,
        "runtime_contract_exact": False,
    }
    # The metadata contract intentionally has two extra descriptive fields and
    # no runtime-version field.  Compare the actual frozen keys explicitly.
    metadata_contract = json.loads(args.metadata.read_text(encoding="utf-8"))[
        "contract"
    ]
    runtime_contract = validate_runtime_constants()
    checks["runtime_contract_exact"] = all(
        metadata_contract.get(key) == value
        for key, value in runtime_contract.items()
        if key != "runtime_version"
    )
    report = {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "failed",
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "platform": platform.platform(),
            "executable_role": "repository CPU-only production virtual environment",
        },
        "artifacts": {
            "selected": {
                "path": _portable(args.model),
                "sha256": _sha256(args.model),
                "size_bytes": args.model.stat().st_size,
            },
            "metadata": {
                "path": _portable(args.metadata),
                "sha256": _sha256(args.metadata),
                "size_bytes": args.metadata.stat().st_size,
            },
            "torch_export_candidate": {
                "path": _portable(args.torch_export),
                "sha256": _sha256(args.torch_export),
                "size_bytes": args.torch_export.stat().st_size,
            },
            "held_reference": {
                "path": _portable(args.reference),
                "sha256": _sha256(args.reference),
                "size_bytes": args.reference.stat().st_size,
            },
            "native_corpus": {
                "path": _portable(args.native_corpus),
                "sha256": _sha256(args.native_corpus),
                "size_bytes": args.native_corpus.stat().st_size,
            },
        },
        "held_observations": len(observations_array),
        "parity_tolerance": 1e-5,
        "selected_maximum_absolute_errors": selected_errors,
        "modern_maximum_absolute_errors": modern_errors,
        "actor_benchmark": {
            "selected_torchscript": selected_actor,
            "torch_export_candidate": modern_actor,
        },
        "full_pipeline": pipeline,
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
