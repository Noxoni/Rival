from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import statistics
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


@dataclass
class GameSpeedMonitor:
    requested: float
    tolerance: float = 0.15
    minimum_reapply_wall_seconds: float = 0.50
    observed_all: list[float] = field(default_factory=list)
    observed_sustained: list[float] = field(default_factory=list)
    first_game_time: float | None = None
    last_game_time: float | None = None
    first_observed_wall: float | None = None
    last_observed_wall: float | None = None
    apply_count: int = 0
    reached_requested_speed: bool = False
    _last_apply_wall: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.requested) or self.requested <= 0.0:
            raise ValueError("requested game speed must be a positive finite number")

    def observe(
        self,
        manager: rlbot.managers.MatchManager,
        packet: Any,
        *,
        allow_state_setting: bool,
    ) -> None:
        info = getattr(packet, "match_info", None)
        if info is None or _phase_name(packet) not in {"Countdown", "Kickoff", "Active"}:
            return
        game_time = float(info.seconds_elapsed)
        observed = float(getattr(info, "game_speed", 1.0))
        if math.isfinite(observed):
            self.observed_all.append(observed)
            if abs(observed - self.requested) <= self.tolerance:
                self.reached_requested_speed = True
            if self.reached_requested_speed:
                self.observed_sustained.append(observed)
        if self.first_game_time is None:
            self.first_game_time = game_time
            self.first_observed_wall = time.monotonic()
        self.last_game_time = game_time
        self.last_observed_wall = time.monotonic()
        now = time.monotonic()
        if (
            allow_state_setting
            and abs(observed - self.requested) > self.tolerance
            and (
                self._last_apply_wall is None
                or now - self._last_apply_wall >= self.minimum_reapply_wall_seconds
            )
        ):
            manager.set_game_state(
                match_info=flat.DesiredMatchInfo(game_speed=self.requested)
            )
            self.apply_count += 1
            self._last_apply_wall = now

    def to_record(self, wall_duration_seconds: float) -> dict[str, Any]:
        def stats(values: list[float]) -> dict[str, float | int | None]:
            if not values:
                return {"samples": 0, "minimum": None, "median": None, "maximum": None}
            return {
                "samples": len(values),
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
            }

        advanced = (
            0.0
            if self.first_game_time is None or self.last_game_time is None
            else max(0.0, self.last_game_time - self.first_game_time)
        )
        active_wall = (
            0.0
            if self.first_observed_wall is None or self.last_observed_wall is None
            else max(0.0, self.last_observed_wall - self.first_observed_wall)
        )
        return {
            "requested_game_speed": self.requested,
            "requested_speed_reached": self.reached_requested_speed,
            "state_setting_apply_count": self.apply_count,
            "observed_game_speed_all_active": stats(self.observed_all),
            "observed_game_speed_sustained": stats(self.observed_sustained),
            "game_seconds_advanced": advanced,
            "session_wall_duration_seconds": wall_duration_seconds,
            "active_observation_wall_seconds": active_wall,
            "effective_game_seconds_per_wall_second": (
                advanced / active_wall if active_wall > 0.0 else None
            ),
        }


class RuntimeWarningCapture(logging.Handler):
    """Capture relevant Python-side RLBot warnings without hiding console output."""

    KEYWORDS = ("queue", "missed", "packet", "disconnect", "socket", "restart")

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        if any(keyword in message.lower() for keyword in self.KEYWORDS):
            self.messages.append(message)


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
    instant_start: bool | None = None,
    opponent_environment: dict[str, str] | None = None,
    rival_environment: dict[str, str] | None = None,
    auto_save_replay: bool = False,
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
        instant_start=state_setting if instant_start is None else instant_start,
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
        enable_state_setting=state_setting,
        auto_save_replay=auto_save_replay,
        freeplay=False,
        performance_monitor=flat.PerformanceMonitor.NeverShow,
    )


def describe_match_configuration(
    *,
    opponent: str,
    rival_team: int,
    state_setting: bool = False,
    requested_game_speed: float = 1.0,
    instant_start: bool = False,
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
        "game_speed_mutator": "Default",
        "requested_game_speed": requested_game_speed,
        "boost_amount": "NormalBoost",
        "boost_strength": "One",
        "gravity": "Default",
        "demolish": "Default",
        "state_setting": state_setting,
        "state_setting_scope": (
            "desired_match_info.game_speed_only" if requested_game_speed != 1.0 else "none"
        ),
        "skip_replays": True,
        "auto_save_replay": False,
        "enable_rendering": "AlwaysOff",
        "performance_monitor": "NeverShow",
        "auto_start_agents": True,
        "wait_for_agents": True,
        "instant_start": instant_start,
        "existing_match_behavior": "Restart",
        "freeplay": False,
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
    speed_monitor: GameSpeedMonitor | None = None,
) -> Any:
    deadline = time.monotonic() + wall_timeout
    while time.monotonic() < deadline:
        packet = manager.packet
        if packet is not None:
            if speed_monitor is not None:
                speed_monitor.observe(
                    manager,
                    packet,
                    allow_state_setting=True,
                )
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
    execution: dict[str, Any] | None = None,
    runtime_warnings: list[str] | None = None,
) -> dict[str, Any]:
    telemetry_path = session_dir / "decisions.jsonl"
    new_replays = sorted(_replay_files() - replay_before, key=lambda path: path.stat().st_mtime)
    replay_records = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in new_replays
    ]
    wall_duration = round(time.monotonic() - started_wall, 3)
    telemetry_summary = summarize_telemetry(telemetry_path)
    decision_count = int(
        telemetry_summary["record_counts"].get("rival_policy_decision", 0)
    )
    execution_record = dict(execution or {})
    execution_record["decision_records_per_wall_second"] = (
        decision_count / wall_duration if wall_duration > 0.0 else None
    )
    manifest = {
        **metadata,
        "end_timestamp_utc": utc_now(),
        "status": status,
        "termination_reason": termination_reason,
        "final_score": final_score,
        "wall_duration_seconds": wall_duration,
        "raw_telemetry": telemetry_summary,
        "replays": replay_records,
        "schedule": schedule or [],
        "error": error,
        "execution": execution_record,
        "runtime_warnings": list(runtime_warnings or []),
    }
    write_json(session_dir / "session_manifest.json", manifest)
    return manifest


def run_natural_match(
    opponent_key: str,
    *,
    rival_team: int,
    launcher: str = "steam",
    timeout: float = 900.0,
    game_speed: float = 1.0,
    challenge_mode: str = "off",
    lane_id: str = "lane-1",
    execution_regime: str = "sequential",
    smoke_game_seconds: float | None = None,
    session_version: str = "v3",
    session_source: str | None = None,
    experiment_milestone: str = "m03-challenge-calibration",
    experiment_metadata: dict[str, Any] | None = None,
    rival_environment_overrides: dict[str, str] | None = None,
    manager: rlbot.managers.MatchManager | None = None,
) -> dict[str, Any]:
    reference = discover_reference(opponent_key)
    rival_environment_overrides = dict(rival_environment_overrides or {})
    invalid_environment = sorted(
        key for key in rival_environment_overrides if not key.startswith("RIVAL_")
    )
    if invalid_environment:
        raise ValueError(
            "Rival environment overrides must use RIVAL_* names: "
            + ", ".join(invalid_environment)
        )
    source_key = "speed-smoke" if smoke_game_seconds is not None else "natural"
    source = "speed_integrity_smoke" if smoke_game_seconds is not None else "natural_match"
    session_id = make_session_id(
        source_key,
        reference.key,
        rival_team,
        milestone=session_version,
    )
    session_dir = RAW_EVIDENCE_ROOT / session_id
    telemetry_path = session_dir / "decisions.jsonl"
    metadata_path = session_dir / "session_start.json"
    accelerated = abs(game_speed - 1.0) > 1e-6
    match_record = describe_match_configuration(
        opponent=opponent_key,
        rival_team=rival_team,
        state_setting=accelerated,
        requested_game_speed=game_speed,
        instant_start=False,
    )
    metadata = build_session_metadata(
        session_id=session_id,
        source=session_source or source,
        opponent=reference,
        rival_team=rival_team,
        match=match_record,
        telemetry_path=telemetry_path,
        experiment_milestone=experiment_milestone,
    )
    metadata["challenge_calibration"] = {"mode": challenge_mode}
    metadata["execution_request"] = {
        "lane_id": lane_id,
        "requested_game_speed": game_speed,
        "smoke_game_seconds": smoke_game_seconds,
    }
    if experiment_metadata:
        metadata["natural_play_experiment"] = dict(experiment_metadata)
    write_json(metadata_path, metadata)
    rival_environment = {
        "RIVAL_TELEMETRY_ENABLED": "1",
        "RIVAL_TELEMETRY_INCLUDE_LOGITS": "0",
        "RIVAL_TELEMETRY_PATH": str(telemetry_path),
        "RIVAL_SESSION_METADATA_PATH": str(metadata_path),
        "RIVAL_CHALLENGE_CALIBRATION_MODE": challenge_mode,
    }
    rival_environment.update(rival_environment_overrides)
    config = build_match_configuration(
        rival_team=rival_team,
        opponent_config=reference.config_path,
        launcher=_launcher(launcher),
        rival_environment=rival_environment,
        state_setting=accelerated,
        instant_start=False,
        auto_save_replay=False,
    )

    owned_manager = manager is None
    manager = manager or rlbot.managers.MatchManager()
    replay_before = _replay_files()
    started_wall = time.monotonic()
    speed_monitor = GameSpeedMonitor(game_speed)
    warning_capture = RuntimeWarningCapture()
    logging.getLogger().addHandler(warning_capture)
    last_score = {"blue": None, "orange": None}
    last_phase = "none"
    termination = "unknown"
    status = "failed"
    error: str | None = None
    try:
        print(f"START {session_id} Rival {'blue' if rival_team == 0 else 'orange'} vs {reference.identity}", flush=True)
        manager.start_match(config, wait_for_start=True, ensure_server_started=True)
        first_packet = _wait_for_active_packet(manager)
        speed_monitor.observe(
            manager,
            first_packet,
            allow_state_setting=accelerated,
        )
        smoke_anchor = float(first_packet.match_info.seconds_elapsed)
        deadline = time.monotonic() + timeout
        next_status = time.monotonic()
        while time.monotonic() < deadline:
            packet = manager.packet
            if packet is None:
                time.sleep(0.05)
                continue
            phase = _phase_name(packet)
            speed_monitor.observe(
                manager,
                packet,
                allow_state_setting=accelerated,
            )
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
            if smoke_game_seconds is not None and (
                float(packet.match_info.seconds_elapsed) - smoke_anchor
                >= smoke_game_seconds
            ):
                termination = "speed_integrity_window_complete"
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
        wall_duration = time.monotonic() - started_wall
        server_process = getattr(manager, "rlbot_server_process", None)
        execution = speed_monitor.to_record(wall_duration)
        execution.update(
            {
                "lane_id": lane_id,
                "rlbot_server_port": getattr(manager, "rlbot_server_port", None),
                "rlbot_server_pid": getattr(server_process, "pid", None),
                "sequential_or_parallel": execution_regime,
                "natural_match_clock": "FiveMinutes",
                "natural_state_setting_scope": (
                    "desired_match_info.game_speed_only" if accelerated else "none"
                ),
            }
        )
        manifest = _finalize_manifest(
            session_dir,
            metadata,
            status=status,
            final_score=last_score,
            termination_reason=termination,
            started_wall=started_wall,
            replay_before=replay_before,
            error=error,
            execution=execution,
            runtime_warnings=warning_capture.messages,
        )
        logging.getLogger().removeHandler(warning_capture)
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
    game_speed: float,
    challenge_mode: str,
    lane_id: str = "lane-1",
    challenge_environment: dict[str, str] | None = None,
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
        "game_speed_mutator": "Default",
        "requested_game_speed": game_speed,
        "boost_amount": "NormalBoost",
        "gravity": "Default",
        "state_setting": True,
        "state_setting_scope": "controlled_state_and_desired_match_info.game_speed",
        "skip_replays": True,
        "auto_save_replay": False,
        "enable_rendering": "AlwaysOff",
        "performance_monitor": "NeverShow",
        "instant_start": True,
        "existing_match_behavior": "Restart",
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
    challenge_environment = challenge_environment or {}
    invalid_environment = sorted(
        key for key in challenge_environment if not key.startswith("RIVAL_CHALLENGE_")
    )
    if invalid_environment:
        raise ValueError(
            "controlled challenge overrides must use RIVAL_CHALLENGE_* names: "
            + ", ".join(invalid_environment)
        )
    metadata["challenge_calibration"] = {
        "mode": challenge_mode,
        "environment_overrides": dict(sorted(challenge_environment.items())),
    }
    metadata["execution_request"] = {
        "lane_id": lane_id,
        "requested_game_speed": game_speed,
    }
    write_json(metadata_path, metadata)
    rival_environment = {
        "RIVAL_TELEMETRY_ENABLED": "1",
        "RIVAL_TELEMETRY_INCLUDE_LOGITS": "0",
        "RIVAL_TELEMETRY_PATH": str(telemetry_path),
        "RIVAL_SESSION_METADATA_PATH": str(metadata_path),
        "RIVAL_CHALLENGE_CALIBRATION_MODE": challenge_mode,
    }
    rival_environment.update(challenge_environment)
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
        instant_start=True,
        auto_save_replay=False,
    )

    owned_manager = manager is None
    manager = manager or rlbot.managers.MatchManager()
    replay_before = _replay_files()
    started_wall = time.monotonic()
    speed_monitor = GameSpeedMonitor(game_speed)
    warning_capture = RuntimeWarningCapture()
    logging.getLogger().addHandler(warning_capture)
    last_score = {"blue": None, "orange": None}
    schedule: list[dict[str, Any]] = []
    status = "failed"
    termination = "unknown"
    error: str | None = None
    try:
        print(f"START {session_id} controlled {family}/{behavior}", flush=True)
        manager.start_match(config, wait_for_start=True, ensure_server_started=True)
        packet = _wait_for_active_packet(manager)
        speed_monitor.observe(manager, packet, allow_state_setting=True)
        for index, case in enumerate(cases):
            if isinstance(case, FakeChallengeParameters):
                cars, balls = fake_challenge_state(case, rival_team)
            else:
                cars, balls = resource_aerial_state(case, rival_team)
            manager.set_game_state(cars=cars, balls=balls)
            time.sleep(min(0.20, 0.25 / game_speed))
            packet = manager.packet or packet
            speed_monitor.observe(manager, packet, allow_state_setting=True)
            anchor = float(packet.match_info.seconds_elapsed)
            entry = {
                "index": index,
                "start_game_time": anchor,
                "parameters": case.to_record(),
            }
            packet = _wait_game_seconds(
                manager,
                anchor,
                float(case.window_seconds),
                speed_monitor=speed_monitor,
            )
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
        wall_duration = time.monotonic() - started_wall
        server_process = getattr(manager, "rlbot_server_process", None)
        execution = speed_monitor.to_record(wall_duration)
        execution.update(
            {
                "lane_id": lane_id,
                "rlbot_server_port": getattr(manager, "rlbot_server_port", None),
                "rlbot_server_pid": getattr(server_process, "pid", None),
                "sequential_or_parallel": "sequential",
                "controlled_state_setting": True,
            }
        )
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
            execution=execution,
            runtime_warnings=warning_capture.messages,
        )
        logging.getLogger().removeHandler(warning_capture)
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
    game_speed: float = 1.0,
    challenge_mode: str = "off",
    lane_id: str = "lane-1",
    challenge_environment: dict[str, str] | None = None,
    manager: rlbot.managers.MatchManager | None = None,
) -> list[dict[str, Any]]:
    results = []
    owned_manager = manager is None
    manager = manager or rlbot.managers.MatchManager()
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
                    game_speed=game_speed,
                    challenge_mode=challenge_mode,
                    lane_id=lane_id,
                    challenge_environment=challenge_environment,
                    manager=manager,
                )
            )
    finally:
        if owned_manager:
            manager.shut_down()
    return results


def run_resource_aerial_probes(
    *,
    rival_team: int = 0,
    launcher: str = "steam",
    cases: Iterable[ResourceAerialParameters] | None = None,
    game_speed: float = 1.0,
    challenge_mode: str = "off",
) -> dict[str, Any]:
    grid = list(cases or default_resource_aerial_grid())
    return _run_probe_session(
        family="resource_aerial",
        behavior="shadow",
        cases=grid,
        rival_team=rival_team,
        launcher=launcher,
        game_speed=game_speed,
        challenge_mode=challenge_mode,
    )
