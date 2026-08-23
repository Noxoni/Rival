from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.spectator import (  # noqa: E402
    resolve_tick_skip,
    run_spectator,
    spectator_preflight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch one selected Rival policy play in an isolated RLViser process."
    )
    parser.add_argument(
        "--checkpoint",
        default="frozen-wisp",
        help=(
            "frozen-wisp, current, a campaign checkpoint directory, a portable actor "
            "checkpoint, or a TorchScript actor export"
        ),
    )
    parser.add_argument(
        "--opponent",
        choices=("frozen-wisp", "selected"),
        default="frozen-wisp",
    )
    parser.add_argument("--team", choices=("blue", "orange"), default="blue")
    parser.add_argument("--tick-skip", choices=("auto", "4", "8"), default="auto")
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="Wall-clock viewer bound; zero runs until Ctrl+C or --max-episodes.",
    )
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Load dependencies/policy and step once without opening the viewer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tick_skip = resolve_tick_skip(args.checkpoint, args.tick_skip)
    if args.check:
        report = spectator_preflight(
            args.checkpoint,
            tick_skip=tick_skip,
            device=args.device,
        )
    else:
        report = run_spectator(
            args.checkpoint,
            opponent_mode=args.opponent,
            selected_team=0 if args.team == "blue" else 1,
            tick_skip=tick_skip,
            legacy_only=args.legacy_only,
            seed=args.seed,
            playback_speed=args.playback_speed,
            duration_seconds=args.duration_seconds,
            max_episodes=args.max_episodes,
            device=args.device,
        )
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
