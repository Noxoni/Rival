"""Run Milestone 09 Gate 6 short-horizon physics transfer audit."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_transition_audit import build_gate06_report  # noqa: E402


CAPTURE_REPORT = (
    TRAINING_ROOT / "results" / "milestone09" / "gate03_native_capture.json"
)
RESULT_PATH = TRAINING_ROOT / "results" / "milestone09" / "gate06_transition_audit.json"


def main() -> int:
    capture = json.loads(CAPTURE_REPORT.read_text(encoding="utf-8"))
    source = capture["native_corpus"]
    report = build_gate06_report(
        REPO_ROOT / source["path"],
        expected_sha256=source["sha256"],
        expected_records=int(source["records"]),
    )
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["commands"] = {
        "gate": (
            "training/.venv/Scripts/python.exe "
            "training/scripts/run_m09_transition_gate.py"
        ),
        "unit_tests": (
            "training/.venv/Scripts/python.exe -m pytest "
            "training/tests/test_v9_transition_audit.py -q"
        ),
    }
    report["gate_semantics"] = {
        "score_used": False,
        "win_loss_used": False,
        "training_budget_used": False,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": report["checks"],
                "selection": report["selection"],
                "materiality_gate_at_four_ticks": report[
                    "materiality_gate_at_four_ticks"
                ],
                "contact_free_primary": report["contact_free_primary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
