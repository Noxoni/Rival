from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.runner import describe_match_configuration  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate references and print a portable Rival match description"
    )
    parser.add_argument("--opponent", choices=("nexto", "wisp"), required=True)
    parser.add_argument("--rival-team", choices=("blue", "orange"), default="blue")
    args = parser.parse_args()
    record = describe_match_configuration(
        opponent=args.opponent,
        rival_team=0 if args.rival_team == "blue" else 1,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
