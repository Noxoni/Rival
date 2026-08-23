"""Launch or preflight the independent native-120-Hz scratch spectator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_spectator import (  # noqa: E402
    run_scratch_spectator,
    scratch_spectator_preflight,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Watch a scratch Rival v9 checkpoint in one isolated RLViser "
            "RocketSim environment. This never renders training workers."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="current",
        help="current or a Rival v9 checkpoint directory",
    )
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--duration-seconds", type=float, default=0.0)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--status-interval-seconds", type=float, default=1.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load the exact checkpoint/env/renderer seam and step once headlessly.",
    )
    args = parser.parse_args()
    report = (
        scratch_spectator_preflight(args.checkpoint, device=args.device)
        if args.check
        else run_scratch_spectator(
            args.checkpoint,
            seed=args.seed,
            playback_speed=args.playback_speed,
            duration_seconds=args.duration_seconds,
            max_episodes=args.max_episodes,
            device=args.device,
            status_interval_seconds=args.status_interval_seconds,
        )
    )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
