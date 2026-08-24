"""Run one deterministic frozen-corpus Rival v10.2 Stage-1 evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


TRAINING_ROOT = Path(__file__).resolve().parents[1]
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.m10_campaign import write_json_atomic  # noqa: E402
from rival_training.v10_2_evaluation import (  # noqa: E402
    evaluate_stage1_checkpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--include-episode-rows", action="store_true")
    parser.add_argument("--environment-batch-size", type=int, default=32)
    parser.add_argument("--evaluation-workers", type=int, default=24)
    args = parser.parse_args()
    report = evaluate_stage1_checkpoint(
        args.checkpoint,
        args.corpus,
        device=args.device,
        include_episode_rows=args.include_episode_rows,
        environment_batch_size=args.environment_batch_size,
        evaluation_workers=args.evaluation_workers,
    )
    write_json_atomic(args.output, report)
    print(
        json.dumps(
            {
                "status": "passed" if report["checks"]["passed"] else "failed",
                "checkpoint": report["checkpoint"],
                "corpus": report["corpus"],
                "overall": report["overall"],
                "evaluation_wall_seconds": report["evaluation_wall_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["checks"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
