"""Verify a materialized Milestone 06 actor export in the invoking runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_export(path: Path, expected_sha256: str | None) -> dict[str, object]:
    source = path.resolve()
    actual_hash = _sha256_file(source)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise RuntimeError("Candidate TorchScript hash does not match expected value")
    module = torch.jit.load(str(source), map_location="cpu").eval()
    sample = torch.randn(
        64,
        432,
        generator=torch.Generator(device="cpu").manual_seed(20260830),
    )
    with torch.inference_mode():
        output = module(sample)
    try:
        portable = source.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        portable = str(source)
    report: dict[str, object] = {
        "path": portable,
        "sha256": actual_hash,
        "torch_version": torch.__version__,
        "input_shape": list(sample.shape),
        "output_shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all().item()),
    }
    if report["output_shape"] != [64, 158] or not report["finite"]:
        raise RuntimeError(f"Candidate export runtime verification failed: {report}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("torchscript", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--runtime-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = _verify_export(args.torchscript, args.sha256)
    report["runtime_label"] = args.runtime_label
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
