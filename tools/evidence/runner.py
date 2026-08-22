from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Iterable

import rlbot.config
import rlbot.flat as flat
import rlbot.managers

from .probes import (
    FAKE_CHALLENGE_BEHAVIORS,
    FakeChallengeParameters,
    ResourceAerialParameters,
    default_resource_aerial_grid,
    fake_challenge_state,
    resource_aerial_state,
)
from .references import REPOSITORY_ROOT, discover_reference, sha256_file
from .session import (
    build_session_metadata,
    make_session_id,
    summarize_telemetry,
    utc_now,
    write_json,
)


RAW_EVIDENCE_ROOT = REPOSITORY_ROOT / "evidence" / "raw"
RIVAL_CONFIG = REPOSITORY_ROOT / "bot" / "rival.bot.toml"
PROBE_CONFIG = REPOSITORY_ROOT / "probes" / "controlled_opponent" / "probe.bot.toml"


def _environment(values: dict[str, str]) -> list[flat.EnvironmentVariable]:
    return [flat.EnvironmentVariable(name, value) for name, value in sorted(values.items())]


def _custom_player_with_environment(
    config_path: Path,
    team: int,
    environment: dict[str, str],
) -> flat.PlayerConfiguration:
    loaded = rlbot.config.load_player_config(config_path, team)
    custom = loaded.variety
    if not isinstance(custom, flat.CustomBot):
        raise TypeError(f"Expected CustomBot in {config_path}, got {type(custom).__name__}")
    merged = {
        str(item.name): str(item.value)
        for item in (getattr(custom, "environment", None) or [])
    }
    merged.update(environment)
    return flat.PlayerConfiguration(
        flat.CustomBot(
            custom.name,
            custom.root_dir,
            custom.run_command,
            custom.loadout,
            custom.agent_id,
            custom.hivemind,
            _environment(merged),
        ),
        team,
        loaded.player_id,
    )


def build_match_configuration(
    *,
    rival_team: int,
    opponent_config: Path,
    launcher: flat.Launcher = flat.Launcher.Steam,
    state_setting: bool = False,
    opponent_environment: dict[str, str] | None = None,
    rival_environment: dict[str, str] | None = None,
    auto_save_replay: bool = True,
) -> flat.MatchConfiguration:
    if rival_team not in (0, 1):
        raise ValueError("rival_team must be 0 (blue) or 1 (orange)")
    rival = _custom_player_with_environment(
        RIVAL_CONFIG,
        rival_team,
        rival_environment or {},
    )
    if opponent_environment is None:
        opponent = rlbot.config.load_player_config(opponent_config, 1 - rival_team)
    else:
        opponent = _custom_player_with_environment(
            opponent_config,
            1 - rival_team,
            opponent_environment,
        )
    return flat.MatchConfiguration(
        launcher=launcher,
        auto_start_agents=True,
        wait_for_agents=True,
        game_map_upk="Stadium_P",
        player_configurations=[rival, opponent],
        script_configurations=[],
        game_mode=flat.GameMode.Soccar,
        skip_replays=True,
        instant_start=state_setting,
        mutators=flat.MutatorSettings(
            match_length=flat.MatchLengthMutator.FiveMinutes,
            game_speed=flat.GameSpeedMutator.Default,
            boost_amount=flat.BoostAmountMutator.NormalBoost,
            boost_strength=flat.BoostStrengthMutator.One,
            gravity=flat.GravityMutator.Default,
            demolish=flat.DemolishMutator.Default,
        ),
        existing_match_behavior=flat.ExistingMatchBehavior.Restart,
        enable_rendering=flat.DebugRendering.OffByDefault,
        enable_state_setting=state_setting,
        auto_save_replay=auto_save_replay,
        freeplay=False,
        performance_monitor=flat.PerformanceMonitor.ShowWhenSuboptimal,
    )


def describe_match_configuration(
    *,
    opponent: str,
    rival_team: int,
    state_setting: bool = False,
) -> dict[str, Any]:
    reference = discover_reference(opponent)
    return {
        "opponent": reference.to_record(),
        "rival_config": str(RIVAL_CONFIG),
        "rival_team": rival_team,
        "opponent_team": 1 - rival_team,
        "game_mode": "Soccar",
        "map": "Stadium_P",
        "match_length": "FiveMinutes",
        "game_speed": "Default",
        "boost_amount": "NormalBoost",
        "boost_strength": "One",
        "gravity": "Default",
        "demolish": "Default",
        "state_setting": state_setting,
        "installed_reference_mutation": False,
    }


def _launcher(value: str) -> flat.Launcher:
    choices = {
        "steam": flat.Launcher.Steam,
        "epic": flat.Launcher.Epic,
        "no-launch": flat.Launcher.NoLaunch,
    }
    try:
        return choices[value]
    except KeyError as exc:
        raise ValueError(f"Unknown launcher {value!r}") from exc


def _replay_files() -> set[Path]:
    candidates = [
        Path.home() / "Documents" / "My Games" / "Rocket League" / "TAGame" / "Demos",
        Path.home()
        / "OneDrive"
        / "Documents"
        / "My Games"
        / "Rocket League"
        / "TAGame"
        / "Demos",
    ]
    return {
        path.resolve()
        for directory in candidates
        if directory.is_dir()
        for path in directory.glob("*.replay")
    }


def _scores(packet: Any) -> dict[str, int | None]:
    teams = list(getattr(packet, "teams", None) or [])
    return {
        "blue": int(teams[0].score) if len(teams) > 0 else None,
        "orange": int(teams[1].score) if len(teams) > 1 else None,
    }


def _phase_name(packet: Any) -> str:
    info = getattr(packet, "match_info", None)
    phase = getattr(info, "match_phase", None)
    return str(phase).split(".")[-1]


def _wait_for_active_packet(manager: rlbot.managers.MatchManager, timeout: float = 90.0) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        packet = manager.packet
        if packet is not None and _phase_name(packet) in {"Countdown", "Kickoff", "Active"}:
            return packet
        time.sleep(0.05)
    raise TimeoutError("RLBot produced no active packet within the startup timeout")


def _wait_game_seconds(
    manager: rlbot.managers.MatchManager,
    anchor: float,
    duration: float,
    wall_timeout: float = 30.0,
) -> Any:
    deadline = time.monotonic() + wall_timeout
    while time.monotonic() < deadline:
        packet = manager.packet
        if packet is not None:
            elapsed = float(packet.match_info.seconds_elapsed)
            if elapsed >= anchor + duration:
                return packet
        time.sleep(0.03)
    raise TimeoutError(f"Probe window did not advance {duration:.2f} game seconds")


def _finalize_manifest(
    session_dir: Path,
    metadata: dict[str, Any],
    *,
    status: str,
    final_score: dict[str, int | None],
    termination_reason: str,
    started_wall: float,
    replay_before: set[Path],
    schedule: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    telemetry_path = session_dir / "decisions.jsonl"
    new_replays = sorted(_replay_files() - replay_before, key=lambda path: path.stat().st_mtime)
    replay_records = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in new_replays
    ]
    manifest = {
        **metadata,
        "end_timestamp_utc": utc_now(),
        "status": status,
        "termination_reason": termination_reason,
        "final_score": final_score,
        "wall_duration_seconds": round(time.monotonic() - started_wall, 3),
        "raw_telemetry": summarize_telemetry(telemetry_path),
        "replays": replay_records,
        "schedule": schedule or [],
        "error": error,
    }
    write_json(session_dir / "session_manifest.json", manifest)
    return manifest


def run_natural_match(
    opponent_key: str,
    *,
    rival_team: int,
    launcher: str = "steam",
    timeout: float = 900.0,
    manager: rlbot.managers.MatchManager | None = None,
) -> dict[str, Any]:
    reference = discover_reference(opponent_key)
    session_id = make_session_id("natural", reference.key, rival_team)
    session_dir = RAW_EVIDENCE_ROOT / session_id
    telemetry_path = session_dir / "decisions.jsonl"
    metadata_path = session_dir / "session_start.json"
    match_record = describe_match_configuration(
        opponent=opponent_key,
        rival_team=rival_team,
        state_setting=False,
    )
    metadata = build_session_metadata(
        session_id=session_id,
        source="natural_match",
        opponent=reference,
        rival_team=rival_team,
        match=match_record,
        telemetry_path=telemetry_path,
    )
    write_json(metadata_path, metadata)
    rival_environment = {
        "RIVAL_TELEMETRY_ENABLED": "1",
        "RIVAL_TELEMETRY_INCLUDE_LOGITS": "0",
        "RIVAL_TELEMETRY_PATH": str(telemetry_path),
        "RIVAL_SESSION_METADATA_PATH": str(metadata_path),
    }
    config = build_match_configuration(
        rival_team=rival_team,
        opponent_config=reference.config_path,
        launcher=_launcher(launcher),
        rival_environment=rival_environment,
        state_setting=False,
        auto_save_replay=True,
    )

    owned_manager = manager is None
    manager = manager or rlbot.managers.MatchManager()
    replay_before = _replay_files()
    started_wall = time.monotonic()
    last_score = {"blue": None, "orange": None}
    last_phase = "none"
    termination = "unknown"
    status = "failed"
    error: str | None = None
    try:
        print(f"START {session_id} Rival {'blue' if rival_team == 0 else 'orange'} vs {reference.identity}", flush=True)
        manager.start_match(config, wait_for_start=True, ensure_server_started=True)
        _wait_for_active_packet(manager)
        deadline = time.monotonic() + timeout
        next_status = time.monotonic()
        while time.monotonic() < deadline:
            packet = manager.packet
            if packet is None:
                time.sleep(0.05)
                continue
            phase = _phase_name(packet)
            last_score = _scores(packet)
            if phase != last_phase or time.monotonic() >= next_status:
                remaining = float(packet.match_info.game_time_remaining)
                print(
                    f"STATUS {session_id} phase={phase} remaining={remaining:.1f} score={last_score['blue']}-{last_score['orange']}",
                    flush=True,
                )
                last_phase = phase
                next_status = time.monotonic() + 30.0
            if phase == "Ended":
                termination = "match_phase_ended"
                status = "complete"
                break
            time.sleep(0.05)
        else:
            termination = "wall_timeout"
            raise TimeoutError(f"Natural match exceeded {timeout:.0f}s wall timeout")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"ERROR {session_id} {error}", flush=True)
    finally:
        try:
            manager.stop_match()
        except Exception as exc:
            error = error or f"stop_match:{type(exc).__name__}:{exc}"
        time.sleep(2.0)
        manifest = _finalize_manifest(
            session_dir,
            metadata,
            status=status,
            final_score=last_score,
            termination_reason=termination,
            started_wall=started_wall,
            replay_before=replay_before,
            error=error,
        )
        if owned_manager:
            manager.shut_down()
    print(f"END {session_id} status={status} score={last_score['blue']}-{last_score['orange']}", flush=True)
    return manifest


def _run_probe_session(
    *,
    family: str,
    behavior: str,
    cases: Iterable[FakeChallengeParameters | ResourceAerialParameters],
    rival_team: int,
    launcher: str,
    manager: rlbot.managers.MatchManager | None = None,
) -> dict[str, Any]:
    cases = list(cases)
    session_id = make_session_id(f"probe-{family}", behavior, rival_team)
    session_dir = RAW_EVIDENCE_ROOT / session_id
    telemetry_path = session_dir / "decisions.jsonl"
    metadata_path = session_dir / "session_start.json"
    probe_opponent = {
        "key": "controlled_probe",
        "identity": f"Controlled probe ({behavior})",
        "config_path": str(PROBE_CONFIG),
        "config_sha256": sha256_file(PROBE_CONFIG),
        "executable_path": None,
        "executable_sha256": None,
    }
    case_records = [case.to_record() for case in cases]
    match_record = {
        "game_mode": "Soccar",
        "map": "Stadium_P",
        "match_length": "FiveMinutes",
        "game_speed": "Default",
        "boost_amount": "NormalBoost",
        "gravity": "Default",
        "state_setting": True,
        "installed_reference_mutation": False,
    }
    metadata = build_session_metadata(
        session_id=session_id,
        source="controlled_probe",
        opponent=probe_opponent,
        rival_team=rival_team,
        match=match_record,
        telemetry_path=telemetry_path,
        probe={"family": family, "behavior": behavior, "cases": case_records},
    )
    write_json(metadata_path, metadata)
    rival_environment = {
        "RIVAL_TELEMETRY_ENABLED": "1",
        "RIVAL_TELEMETRY_INCLUDE_LOGITS": "0",
        "RIVAL_TELEMETRY_PATH": str(telemetry_path),
        "RIVAL_SESSION_METADATA_PATH": str(metadata_path),
    }
    opponent_environment = {
        "RIVAL_PROBE_BEHAVIOR": behavior,
        "RIVAL_PROBE_ABORT_TIME": str(
            getattr(next(iter(cases), None), "abort_time", 0.65)
        ),
    }
    config = build_match_configuration(
        rival_team=rival_team,
        opponent_config=PROBE_CONFIG,
        launcher=_launcher(launcher),
        rival_environment=rival_environment,
        opponent_environment=opponent_environment,
        state_setting=True,
        auto_save_replay=False,
    )

    owned_manager = manager is None
    manager = manager or rlbot.managers.MatchManager()
    replay_before = _replay_files()
    started_wall = time.monotonic()
    last_score = {"blue": None, "orange": None}
    schedule: list[dict[str, Any]] = []
    status = "failed"
    termination = "unknown"
    error: str | None = None
    try:
        print(f"START {session_id} controlled {family}/{behavior}", flush=True)
        manager.start_match(config, wait_for_start=True, ensure_server_started=True)
        packet = _wait_for_active_packet(manager)
        for index, case in enumerate(cases):
            if isinstance(case, FakeChallengeParameters):
                cars, balls = fake_challenge_state(case, rival_team)
            else:
                cars, balls = resource_aerial_state(case, rival_team)
            manager.set_game_state(cars=cars, balls=balls)
            time.sleep(0.20)
            packet = manager.packet or packet
            anchor = float(packet.match_info.seconds_elapsed)
            entry = {
                "index": index,
                "start_game_time": anchor,
                "parameters": case.to_record(),
            }
            packet = _wait_game_seconds(manager, anchor, float(case.window_seconds))
            entry["end_game_time"] = float(packet.match_info.seconds_elapsed)
            entry["score_after"] = _scores(packet)
            schedule.append(entry)
            last_score = _scores(packet)
            print(
                f"PROBE {session_id} {index + 1}/{len(case_records)} time={anchor:.2f}-{entry['end_game_time']:.2f}",
                flush=True,
            )
        status = "complete"
        termination = "controlled_probe_schedule_complete"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"ERROR {session_id} {error}", flush=True)
    finally:
        try:
            manager.stop_match()
        except Exception as exc:
            error = error or f"stop_match:{type(exc).__name__}:{exc}"
        time.sleep(2.0)
        manifest = _finalize_manifest(
            session_dir,
            metadata,
            status=status,
            final_score=last_score,
            termination_reason=termination,
            started_wall=started_wall,
            replay_before=replay_before,
            schedule=schedule,
            error=error,
        )
        if owned_manager:
            manager.shut_down()
    print(f"END {session_id} status={status} probes={len(schedule)}", flush=True)
    return manifest


def run_fake_challenge_probes(
    *,
    repetitions: int = 5,
    rival_team: int = 0,
    launcher: str = "steam",
    behaviors: Iterable[str] = FAKE_CHALLENGE_BEHAVIORS,
) -> list[dict[str, Any]]:
    results = []
    manager = rlbot.managers.MatchManager()
    try:
        for behavior in behaviors:
            cases = [
                FakeChallengeParameters(behavior=behavior, repetition=index + 1)
                for index in range(repetitions)
            ]
            results.append(
                _run_probe_session(
                    family="fake_challenge",
                    behavior=behavior,
                    cases=cases,
                    rival_team=rival_team,
                    launcher=launcher,
                    manager=manager,
                )
            )
    finally:
        manager.shut_down()
    return results


def run_resource_aerial_probes(
    *,
    rival_team: int = 0,
    launcher: str = "steam",
    cases: Iterable[ResourceAerialParameters] | None = None,
) -> dict[str, Any]:
    grid = list(cases or default_resource_aerial_grid())
    return _run_probe_session(
        family="resource_aerial",
        behavior="shadow",
        cases=grid,
        rival_team=rival_team,
        launcher=launcher,
    )
