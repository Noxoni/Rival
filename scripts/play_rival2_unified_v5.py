"""Launch an offline human-vs-Rival-Unified-V5 RLBot match."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import rlbot.config
import rlbot.flat as flat
import rlbot.managers


ROOT = Path(__file__).resolve().parents[1]
RIVAL_CONFIG = ROOT / "bot" / "rival2_unified_v5" / "rival2.bot.toml"


def _launcher(name: str) -> flat.Launcher:
    return {
        "steam": flat.Launcher.Steam,
        "epic": flat.Launcher.Epic,
        "no-launch": flat.Launcher.NoLaunch,
    }[name]


def build_match(human_team: int, launcher: str) -> flat.MatchConfiguration:
    rival_team = 1 - human_team
    human = flat.PlayerConfiguration(flat.Human(), human_team, 0)
    rival = rlbot.config.load_player_config(RIVAL_CONFIG, rival_team)
    players = [human, rival] if human_team == 0 else [rival, human]
    return flat.MatchConfiguration(
        launcher=_launcher(launcher),
        auto_start_agents=True,
        wait_for_agents=True,
        game_map_upk="Stadium_P",
        player_configurations=players,
        script_configurations=[],
        game_mode=flat.GameMode.Soccar,
        skip_replays=True,
        instant_start=False,
        mutators=flat.MutatorSettings(
            match_length=flat.MatchLengthMutator.FiveMinutes,
            max_score=flat.MaxScoreMutator.Unlimited,
            overtime=flat.OvertimeMutator.Unlimited,
            game_speed=flat.GameSpeedMutator.Default,
            boost_amount=flat.BoostAmountMutator.NormalBoost,
            boost_strength=flat.BoostStrengthMutator.One,
            gravity=flat.GravityMutator.Default,
            demolish=flat.DemolishMutator.Default,
        ),
        existing_match_behavior=flat.ExistingMatchBehavior.Restart,
        enable_rendering=flat.DebugRendering.AlwaysOff,
        enable_state_setting=False,
        auto_save_replay=False,
        freeplay=False,
        performance_monitor=flat.PerformanceMonitor.NeverShow,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play a five-minute offline 1v1 against Rival 2 Unified V5"
    )
    parser.add_argument("--human-team", choices=("blue", "orange"), default="blue")
    parser.add_argument(
        "--launcher", choices=("steam", "epic", "no-launch"), default="steam"
    )
    args = parser.parse_args()
    human_team = 0 if args.human_team == "blue" else 1
    manager = rlbot.managers.MatchManager()
    try:
        manager.start_match(
            build_match(human_team, args.launcher),
            wait_for_start=True,
            ensure_server_started=True,
        )
        print(
            "Rival 2 Unified V5 match started: "
            f"human={'Blue' if human_team == 0 else 'Orange'}, "
            f"Rival={'Orange' if human_team == 0 else 'Blue'}"
        )
        while True:
            packet = manager.packet
            if packet is not None:
                phase = str(packet.match_info.match_phase).split(".")[-1]
                if phase == "Ended":
                    return 0
            time.sleep(0.1)
    except KeyboardInterrupt:
        return 0
    finally:
        manager.stop_match()


if __name__ == "__main__":
    raise SystemExit(main())
