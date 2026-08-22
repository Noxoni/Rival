from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import rlbot.managers


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.probes import FAKE_CHALLENGE_BEHAVIORS  # noqa: E402
from tools.evidence.runner import (  # noqa: E402
    describe_match_configuration,
    run_fake_challenge_probes,
    run_natural_match,
    run_resource_aerial_probes,
)


def _team(value: str) -> int:
    return {"blue": 0, "orange": 1}[value]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rival RLBot v5 evidence runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover = subparsers.add_parser("describe", help="validate and describe a match")
    discover.add_argument("--opponent", choices=("nexto", "wisp"), required=True)
    discover.add_argument("--rival-team", choices=("blue", "orange"), default="blue")
    discover.add_argument("--game-speed", type=float, default=1.0)

    natural = subparsers.add_parser("natural", help="run complete natural matches")
    natural.add_argument("--opponent", choices=("nexto", "wisp"), required=True)
    natural.add_argument("--rival-team", choices=("blue", "orange"), default="blue")
    natural.add_argument("--count", type=int, default=1)
    natural.add_argument("--launcher", choices=("steam", "epic", "no-launch"), default="steam")
    natural.add_argument("--timeout", type=float, default=900.0)
    natural.add_argument("--game-speed", type=float, default=1.0)
    natural.add_argument(
        "--challenge-mode",
        choices=("off", "observe", "intervene"),
        default="off",
    )

    speed_smoke = subparsers.add_parser(
        "speed-smoke", help="run a bounded non-budgeted game-speed integrity window"
    )
    speed_smoke.add_argument("--opponent", choices=("nexto", "wisp"), required=True)
    speed_smoke.add_argument("--rival-team", choices=("blue", "orange"), default="blue")
    speed_smoke.add_argument("--launcher", choices=("steam", "epic"), default="steam")
    speed_smoke.add_argument("--timeout", type=float, default=180.0)
    speed_smoke.add_argument("--game-speed", type=float, required=True)
    speed_smoke.add_argument("--game-seconds", type=float, default=30.0)

    fake = subparsers.add_parser("probe-fake", help="run controlled fake-challenge probes")
    fake.add_argument("--rival-team", choices=("blue", "orange"), default="blue")
    fake.add_argument("--repetitions", type=int, default=5)
    fake.add_argument("--behavior", choices=FAKE_CHALLENGE_BEHAVIORS, action="append")
    fake.add_argument("--launcher", choices=("steam", "epic", "no-launch"), default="steam")
    fake.add_argument("--game-speed", type=float, default=1.0)
    fake.add_argument(
        "--challenge-mode",
        choices=("off", "observe", "intervene"),
        default="off",
    )

    aerial = subparsers.add_parser("probe-aerial", help="run resource-aerial grid")
    aerial.add_argument("--rival-team", choices=("blue", "orange"), default="blue")
    aerial.add_argument("--launcher", choices=("steam", "epic", "no-launch"), default="steam")
    aerial.add_argument("--game-speed", type=float, default=1.0)

    args = parser.parse_args()
    if args.command == "describe":
        result: object = describe_match_configuration(
            opponent=args.opponent,
            rival_team=_team(args.rival_team),
            state_setting=args.game_speed != 1.0,
            requested_game_speed=args.game_speed,
        )
    elif args.command == "natural":
        manager = rlbot.managers.MatchManager()
        initial_team = _team(args.rival_team)
        try:
            result = [
                run_natural_match(
                    args.opponent,
                    rival_team=initial_team if index % 2 == 0 else 1 - initial_team,
                    launcher=args.launcher,
                    timeout=args.timeout,
                    game_speed=args.game_speed,
                    challenge_mode=args.challenge_mode,
                    manager=manager,
                )
                for index in range(args.count)
            ]
        finally:
            manager.shut_down()
    elif args.command == "speed-smoke":
        result = run_natural_match(
            args.opponent,
            rival_team=_team(args.rival_team),
            launcher=args.launcher,
            timeout=args.timeout,
            game_speed=args.game_speed,
            challenge_mode="off",
            smoke_game_seconds=args.game_seconds,
        )
    elif args.command == "probe-fake":
        result = run_fake_challenge_probes(
            repetitions=args.repetitions,
            rival_team=_team(args.rival_team),
            launcher=args.launcher,
            behaviors=args.behavior or FAKE_CHALLENGE_BEHAVIORS,
            game_speed=args.game_speed,
            challenge_mode=args.challenge_mode,
        )
    else:
        result = run_resource_aerial_probes(
            rival_team=_team(args.rival_team),
            launcher=args.launcher,
            game_speed=args.game_speed,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if isinstance(result, list):
        return 0 if all(item.get("status") == "complete" for item in result) else 1
    return 0 if args.command == "describe" or result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
