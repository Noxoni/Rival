from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE = "4f2b21c00e2fcb7108ab1006fd950b066fbd0484"
EXACT_FILES = (
    "bot/action_parser.py",
    "bot/obs_builder.py",
    "bot/backend/model.py",
    "bot/backend/rocketsim_adapter.py",
    "bot/policy/decision.py",
    "bot/policy/inspector.py",
)


def _git_show(revision: str, path: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _policy_core(value: bytes) -> bytes:
    text = value.decode("utf-8")
    match = re.search(
        r"        # Build obs.*?        opponents =",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError("Could not locate RivalBot.update_action policy core")
    return match.group(0).encode("utf-8")


def verify(baseline: str = BASELINE) -> list[str]:
    failures: list[str] = []
    for relative in EXACT_FILES:
        expected = _git_show(baseline, relative)
        actual = (REPOSITORY_ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        if _digest(actual) != _digest(expected.replace(b"\r\n", b"\n")):
            failures.append(relative)

    expected_core = _policy_core(_git_show(baseline, "bot/bot.py"))
    actual_core = _policy_core((REPOSITORY_ROOT / "bot" / "bot.py").read_bytes())
    if _digest(actual_core.replace(b"\r\n", b"\n")) != _digest(
        expected_core.replace(b"\r\n", b"\n")
    ):
        failures.append("bot/bot.py:update_action_policy_core")

    config_text = (REPOSITORY_ROOT / "bot" / "config.py").read_text(encoding="utf-8")
    required = (
        "TICK_SKIP = 8",
        "ACTION_DELAY = TICK_SKIP - 1",
        "DETERMINISTIC = True",
        "STRATEGIC_OVERRIDES_ENABLED = False",
    )
    failures.extend(f"bot/config.py:{value}" for value in required if value not in config_text)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify frozen Milestone 01 policy surfaces")
    parser.add_argument("--baseline", default=BASELINE)
    args = parser.parse_args()
    failures = verify(args.baseline)
    if failures:
        print("POLICY FREEZE FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"POLICY FREEZE PASS against {args.baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
