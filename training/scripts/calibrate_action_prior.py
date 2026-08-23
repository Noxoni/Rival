"""Calibrate Stage A's appended-action prior on natural RocketSim observations."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.checkpoint import load_actor_checkpoint  # noqa: E402
from rival_training.config import (  # noqa: E402
    canonical_config_sha256,
    load_milestone06_config,
)
from rival_training.exploration import (  # noqa: E402
    calibrate_appended_offsets,
    collect_natural_observations,
)
from rival_training.policy import normalize_bootstrap_actor_for_prior  # noqa: E402


def main() -> int:
    config = load_milestone06_config()
    exploration = config["action_exploration"]
    sweep_path = REPOSITORY_ROOT / "training/results/milestone06/throughput_sweep.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    worker_count = int(sweep["selected_worker_count"])
    bootstrap_path = (
        REPOSITORY_ROOT / "training/artifacts/bootstrap/wisp_student_expanded_v1.pt"
    )
    started = time.perf_counter()
    observations, observation_report = collect_natural_observations(
        bootstrap_path,
        worker_count=worker_count,
        target_agent_observations=int(
            exploration["calibration_natural_agent_observations"]
        ),
        seed=int(config["seeds"]["calibration"]),
    )
    actor, _ = load_actor_checkpoint(bootstrap_path, "cpu")
    normalize_bootstrap_actor_for_prior(actor)
    calibration = calibrate_appended_offsets(
        actor,
        observations,
        candidate_offsets=[float(value) for value in exploration["candidate_offsets"]],
        minimum_probability_mass=float(
            exploration["minimum_appended_probability_mass"]
        ),
        maximum_probability_mass=float(
            exploration["maximum_appended_probability_mass"]
        ),
        minimum_sampled_share=float(
            exploration["minimum_sampled_appended_share"]
        ),
        maximum_deterministic_share=float(
            exploration["maximum_deterministic_appended_share"]
        ),
        minimum_legacy_top1_retention=float(
            exploration["minimum_legacy_top1_retention"]
        ),
        seed=int(config["seeds"]["calibration"]),
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "config_sha256": canonical_config_sha256(config),
        "worker_count": worker_count,
        "natural_observations": observation_report,
        "gates": {
            key: exploration[key]
            for key in (
                "minimum_appended_probability_mass",
                "maximum_appended_probability_mass",
                "minimum_sampled_appended_share",
                "maximum_deterministic_appended_share",
                "minimum_legacy_top1_retention",
            )
        },
        **calibration,
        "wall_seconds": time.perf_counter() - started,
    }
    output = REPOSITORY_ROOT / "training/results/milestone06/action_prior_calibration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
