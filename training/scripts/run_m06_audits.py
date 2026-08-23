"""Run Reward V2 contribution and broad-curriculum preflight audits."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.audit import (  # noqa: E402
    reward_audit_health,
    run_curriculum_audit,
    run_reward_v2_rollout_audit,
)
from rival_training.config import (  # noqa: E402
    canonical_config_sha256,
    load_milestone06_config,
)


def main() -> int:
    config = load_milestone06_config()
    calibration_path = (
        REPOSITORY_ROOT / "training/results/milestone06/action_prior_calibration.json"
    )
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    offset = float(calibration["selected_appended_logit_offset"])
    started = time.perf_counter()
    reward_report = {
        "rollouts": [
            run_reward_v2_rollout_audit(
                mode="natural_wisp",
                decisions=5000,
                seed=int(config["seeds"]["audit"]),
                appended_logit_offset=offset,
            ),
            run_reward_v2_rollout_audit(
                mode="weighted_uniform_random",
                decisions=5000,
                seed=int(config["seeds"]["audit"]) + 1,
                appended_logit_offset=offset,
            ),
        ]
    }
    reward_report["health"] = reward_audit_health(reward_report)
    curriculum_report = run_curriculum_audit(
        samples_per_stage=10000,
        seed=int(config["seeds"]["audit"]),
    )
    report = {
        "schema_version": 1,
        "status": (
            "passed"
            if reward_report["health"]["passed"] and curriculum_report["passed"]
            else "failed"
        ),
        "config_sha256": canonical_config_sha256(config),
        "selected_appended_logit_offset": offset,
        "reward_v2_contribution_audit": reward_report,
        "curriculum_distribution_audit": curriculum_report,
        "wall_seconds": time.perf_counter() - started,
    }
    output = REPOSITORY_ROOT / "training/results/milestone06/reward_curriculum_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
