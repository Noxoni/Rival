"""Run deterministic M08 dual-rate evaluation at a campaign boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "training"))

from rival_training.config import load_milestone08_config  # noqa: E402
from rival_training.m08_evaluation import (  # noqa: E402
    evaluate_m08_frozen_anchor,
    load_m08_evaluation_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--games", type=int)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_milestone08_config()
    policy, source = load_m08_evaluation_policy(
        args.checkpoint,
        device="cuda:0",
    )
    baseline = (
        None
        if args.baseline is None
        else json.loads(args.baseline.read_text(encoding="utf-8"))
    )
    report = evaluate_m08_frozen_anchor(
        policy,
        source,
        games=int(
            args.games or config["evaluation"]["headless_games_per_boundary"]
        ),
        seed=int(config["seeds"]["evaluation"]),
        baseline=baseline,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["health"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
