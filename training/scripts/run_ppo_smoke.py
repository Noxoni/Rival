"""Run the bounded PPO save/reload/resume proof at the measured worker count."""

from __future__ import annotations

import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.environment import make_gym_env_mechanics4  # noqa: E402
from rival_training.ppo_smoke import run_bounded_ppo_smoke  # noqa: E402


def main() -> None:
    throughput_path = REPOSITORY_ROOT / "training/results/throughput_report.json"
    throughput = json.loads(throughput_path.read_text(encoding="utf-8"))
    worker_count = int(throughput["selected_worker_count"])
    report = run_bounded_ppo_smoke(
        make_gym_env_mechanics4,
        worker_count=worker_count,
        checkpoint_directory=REPOSITORY_ROOT / "training/checkpoints/milestone05_smoke",
        actor_checkpoint_path=REPOSITORY_ROOT
        / "training/artifacts/ppo/milestone05_smoke_actor.pt",
    )
    output = REPOSITORY_ROOT / "training/results/ppo_smoke_report.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
