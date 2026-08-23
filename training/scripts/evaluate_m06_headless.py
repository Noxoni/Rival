"""Run the deterministic Milestone 06 frozen-Wisp headless gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.config import load_milestone06_config  # noqa: E402
from rival_training.evaluation import (  # noqa: E402
    evaluate_frozen_wisp,
    load_evaluation_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap-offset", type=float)
    parser.add_argument("--games", type=int)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_milestone06_config()
    policy, source_record = load_evaluation_policy(
        checkpoint_directory=args.checkpoint,
        bootstrap_offset=args.bootstrap_offset if args.bootstrap else None,
        device="cuda:0",
    )
    baseline = (
        json.loads(args.baseline.read_text(encoding="utf-8"))
        if args.baseline is not None
        else None
    )
    report = evaluate_frozen_wisp(
        policy,
        source_record,
        games=int(args.games or config["evaluation"]["headless_games"]),
        seed=int(config["seeds"]["evaluation"]),
        baseline=baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["health"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
