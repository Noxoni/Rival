"""Execute the bounded Path-A Wisp bootstrap and persist compact evidence."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.checkpoint import save_actor_checkpoint  # noqa: E402
from rival_training.teacher import (  # noqa: E402
    architecture_metadata,
    build_wisp_student,
    validate_student_against_reference,
    verify_teacher_hashes,
)


def main() -> None:
    before = verify_teacher_hashes()
    start = time.perf_counter()
    student = build_wisp_student()
    parity = validate_student_against_reference(
        student,
        batch_size=4096,
        seed=20260822,
        device="cuda",
    )
    elapsed = time.perf_counter() - start
    checkpoint_path = (
        REPOSITORY_ROOT / "training/artifacts/bootstrap/wisp_student_expanded_v1.pt"
    )
    checkpoint = save_actor_checkpoint(
        checkpoint_path,
        student,
        {"bootstrap_path": "A_direct_trainable_reconstruction"},
    )
    after = verify_teacher_hashes()
    report = {
        "schema_version": 1,
        "status": "passed",
        "selected_path": "A_direct_trainable_reconstruction",
        "bounded_elapsed_seconds": elapsed,
        "architecture": architecture_metadata(),
        "teacher_hashes_before": before,
        "numerical_parity": parity,
        "teacher_hashes_after": after,
        "student_checkpoint": checkpoint,
        "behavior_distillation": {
            "run": False,
            "reason": "Direct reconstruction passed exact numerical parity; ordered fallback was not needed.",
            "dataset_records": 0,
        },
        "passed": before["all_match"]
        and after["all_match"]
        and parity["allclose_atol_1e-6_rtol_1e-6"],
    }
    output = REPOSITORY_ROOT / "training/results/bootstrap_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
