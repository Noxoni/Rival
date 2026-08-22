from __future__ import annotations

import argparse
from collections.abc import Callable
import json
from pathlib import Path
import shutil
import sys
import threading
import time
from typing import Any

import psutil
import rlbot.managers
from rlbot.utils import gateway


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tools.evidence.references import sha256_file  # noqa: E402
from tools.evidence.runner import run_natural_match  # noqa: E402
from tools.evidence.session import utc_now, write_json  # noqa: E402


class IsolatedMatchManager(rlbot.managers.MatchManager):
    """Match manager whose process lookup honors a renamed Windows server image."""

    @property
    def _rlbot_server_name(self) -> str:
        if self.rlbot_server_path is None:
            return super()._rlbot_server_name
        return self.rlbot_server_path.name


def _processes_named(name: str) -> list[psutil.Process]:
    matches = []
    for process in psutil.process_iter(["name"]):
        try:
            if (process.info.get("name") or "").casefold() == name.casefold():
                matches.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matches


def _rocket_league_pids() -> list[int]:
    return sorted(process.pid for process in _processes_named("RocketLeague.exe"))


def _child_snapshot(server: psutil.Process) -> list[dict[str, Any]]:
    children = []
    try:
        descendants = server.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return children
    for child in descendants:
        try:
            environment = child.environ()
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            environment = {}
        try:
            name = child.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            name = "unavailable"
        children.append(
            {
                "pid": child.pid,
                "name": name,
                "rlbot_server_port": environment.get("RLBOT_SERVER_PORT"),
            }
        )
    return sorted(children, key=lambda item: int(item["pid"]))


def _launch_server(
    source: Path,
    destination: Path,
) -> tuple[IsolatedMatchManager, psutil.Process]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    manager = IsolatedMatchManager(destination)
    manager.ensure_server_started()
    deadline = time.monotonic() + 10.0
    server: psutil.Process | None = None
    while time.monotonic() < deadline:
        server, _ = gateway.find_server_process(destination.name)
        if server is not None:
            break
        time.sleep(0.1)
    if server is None:
        raise RuntimeError(f"Could not identify isolated server process {destination.name}")
    manager.rlbot_server_process = server
    bind_deadline = time.monotonic() + 10.0
    while (
        server.is_running()
        and gateway.is_port_accessible(manager.rlbot_server_port)
        and time.monotonic() < bind_deadline
    ):
        time.sleep(0.05)
    if not server.is_running():
        raise RuntimeError(f"Isolated server exited before binding {destination.name}")
    if gateway.is_port_accessible(manager.rlbot_server_port):
        raise TimeoutError(
            f"Isolated server did not bind port {manager.rlbot_server_port}"
        )
    return manager, server


def _compact_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    previous_signature: object = None
    for sample in samples:
        signature = (
            tuple(sample.get("rocket_league_pids") or []),
            tuple(
                (
                    lane.get("lane_id"),
                    lane.get("server_alive"),
                    tuple(
                        (
                            child.get("pid"),
                            child.get("name"),
                            child.get("rlbot_server_port"),
                        )
                        for child in lane.get("children") or []
                    ),
                )
                for lane in sample.get("lanes") or []
            ),
        )
        if signature != previous_signature:
            compact.append(sample)
            previous_signature = signature
    if samples and compact and compact[-1] is not samples[-1]:
        compact.append(samples[-1])
    return compact


def _lane_worker(
    *,
    barrier: threading.Barrier,
    result: dict[str, Any],
    result_key: str,
    manager: IsolatedMatchManager,
    lane_id: str,
    opponent: str,
    rival_team: int,
    game_seconds: float,
    timeout: float,
) -> None:
    try:
        barrier.wait(timeout=10.0)
        result[result_key] = run_natural_match(
            opponent,
            rival_team=rival_team,
            launcher="steam",
            timeout=timeout,
            game_speed=1.0,
            challenge_mode="off",
            lane_id=lane_id,
            execution_regime="parallel",
            smoke_game_seconds=game_seconds,
            manager=manager,
        )
    except BaseException as exc:
        result[result_key] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _safe_call(action: Callable[[], Any]) -> str | None:
    try:
        action()
    except BaseException as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One bounded two-lane RLBot/Rocket League isolation capability test"
    )
    parser.add_argument("--game-seconds", type=float, default=8.0)
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument(
        "--server-source",
        type=Path,
        default=(Path.home() / "AppData" / "Local" / "RLBot5" / "bin" / "RLBotServer.exe"),
    )
    parser.add_argument(
        "--runtime-root",
        type=Path,
        default=REPOSITORY_ROOT / ".pytest_tmp" / "m03-concurrency-capability",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "evidence"
            / "results"
            / "v3"
            / "concurrency_capability.json"
        ),
    )
    parser.add_argument(
        "--compact-existing",
        action="store_true",
        help="compact an already generated report without performing another attempt",
    )
    parser.add_argument(
        "--native-console-observation",
        action="append",
        default=[],
        help="exact native-server observation to preserve while compacting",
    )
    args = parser.parse_args()
    if args.compact_existing:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        samples = existing.get("monitor_samples") or []
        existing["monitor_sample_count_raw"] = len(samples)
        existing["monitor_samples"] = _compact_samples(samples)
        existing["monitor_sample_count_committed"] = len(
            existing["monitor_samples"]
        )
        existing["native_console_observations"] = list(
            args.native_console_observation
        )
        existing["harness_revision_note"] = (
            "The bounded attempt exposed a server-port startup race. The committed harness "
            "now waits for listener readiness, but the experiment was not repeated because "
            "the handoff permits exactly one concurrency attempt."
        )
        write_json(args.output, existing)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "monitor_sample_count_raw": len(samples),
                    "monitor_sample_count_committed": len(
                        existing["monitor_samples"]
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.game_seconds <= 0.0 or args.timeout <= 0.0:
        parser.error("game-seconds and timeout must be positive")
    source = args.server_source.resolve()
    if not source.is_file():
        parser.error(f"RLBotServer was not found at {source}")

    preexisting_servers = sorted(
        process.pid
        for process in psutil.process_iter(["name"])
        if (process.info.get("name") or "").casefold().startswith("rlbotserver")
    )
    preexisting_rocket_league = _rocket_league_pids()
    if preexisting_servers or preexisting_rocket_league:
        report = {
            "report_schema_version": 1,
            "generated_utc": utc_now(),
            "gate": "rival-m03-two-lane-capability-v1",
            "attempt_count": 0,
            "supported": False,
            "blocked_before_attempt": True,
            "reason": "preexisting RLBotServer or Rocket League process would make isolation ambiguous",
            "preexisting_rlbot_server_pids": preexisting_servers,
            "preexisting_rocket_league_pids": preexisting_rocket_league,
            "natural_match_budget_consumed": 0,
        }
        write_json(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2

    started = time.monotonic()
    managers: list[IsolatedMatchManager] = []
    servers: list[psutil.Process] = []
    results: dict[str, Any] = {}
    cleanup_errors: list[str] = []
    monitor_samples: list[dict[str, Any]] = []
    timed_out = False
    launch_error: str | None = None
    threads: list[threading.Thread] = []
    lane_specs = [
        ("lane-1", "nexto", 0, "RLBotServerM03Lane1.exe"),
        ("lane-2", "wisp", 1, "RLBotServerM03Lane2.exe"),
    ]
    try:
        for lane_id, _opponent, _team, image_name in lane_specs:
            manager, server = _launch_server(
                source,
                args.runtime_root.resolve() / lane_id / image_name,
            )
            managers.append(manager)
            servers.append(server)
        barrier = threading.Barrier(3)
        for index, ((lane_id, opponent, rival_team, _), manager) in enumerate(
            zip(lane_specs, managers)
        ):
            thread = threading.Thread(
                target=_lane_worker,
                kwargs={
                    "barrier": barrier,
                    "result": results,
                    "result_key": lane_id,
                    "manager": manager,
                    "lane_id": lane_id,
                    "opponent": opponent,
                    "rival_team": rival_team,
                    "game_seconds": args.game_seconds,
                    "timeout": args.timeout,
                },
                name=f"m03-{lane_id}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        barrier.wait(timeout=10.0)
        deadline = time.monotonic() + args.timeout
        while any(thread.is_alive() for thread in threads) and time.monotonic() < deadline:
            monitor_samples.append(
                {
                    "wall_offset_seconds": time.monotonic() - started,
                    "rocket_league_pids": _rocket_league_pids(),
                    "lanes": [
                        {
                            "lane_id": lane_specs[index][0],
                            "server_pid": server.pid,
                            "server_alive": server.is_running(),
                            "children": _child_snapshot(server),
                        }
                        for index, server in enumerate(servers)
                    ],
                }
            )
            time.sleep(0.25)
        timed_out = any(thread.is_alive() for thread in threads)
    except BaseException as exc:
        launch_error = f"{type(exc).__name__}: {exc}"
    finally:
        if timed_out:
            for manager in managers:
                message = _safe_call(manager.stop_match)
                if message:
                    cleanup_errors.append(f"stop_match: {message}")
        for thread in threads:
            thread.join(timeout=5.0)
        for manager in managers:
            message = _safe_call(manager.shut_down)
            if message:
                cleanup_errors.append(f"shut_down: {message}")

    lane_records = []
    for index, (lane_id, opponent, rival_team, image_name) in enumerate(lane_specs):
        manifest = results.get(lane_id) or {}
        server = servers[index] if index < len(servers) else None
        manager = managers[index] if index < len(managers) else None
        lane_records.append(
            {
                "lane_id": lane_id,
                "opponent": opponent,
                "rival_team": rival_team,
                "server_image": image_name,
                "server_image_sha256": sha256_file(source),
                "server_pid": server.pid if server is not None else None,
                "server_port": getattr(manager, "rlbot_server_port", None),
                "session_id": manifest.get("session_id"),
                "status": manifest.get("status"),
                "error": manifest.get("error"),
                "termination_reason": manifest.get("termination_reason"),
                "telemetry": manifest.get("raw_telemetry") or {},
                "execution": manifest.get("execution") or {},
            }
        )

    simultaneous_rocket_league_pids = max(
        (len(sample["rocket_league_pids"]) for sample in monitor_samples),
        default=0,
    )
    unique_ports = {lane["server_port"] for lane in lane_records if lane["server_port"]}
    unique_servers = {lane["server_pid"] for lane in lane_records if lane["server_pid"]}
    unique_sessions = {lane["session_id"] for lane in lane_records if lane["session_id"]}
    server_isolation = len(unique_ports) == 2 and len(unique_servers) == 2
    match_completion = all(lane["status"] == "complete" for lane in lane_records)
    telemetry_isolation = len(unique_sessions) == 2 and all(
        lane["telemetry"].get("exists") and not lane["telemetry"].get("invalid_record_count")
        for lane in lane_records
    )
    rocket_league_isolation = simultaneous_rocket_league_pids >= 2
    reasons = []
    if launch_error:
        reasons.append(launch_error)
    if timed_out:
        reasons.append("one or both lanes exceeded the bounded capability-test deadline")
    if not server_isolation:
        reasons.append("two distinct RLBotServer processes and ports were not verified")
    if not rocket_league_isolation:
        reasons.append("two simultaneous Rocket League processes were not observed")
    if not match_completion:
        reasons.append("both smoke windows did not complete independently")
    if not telemetry_isolation:
        reasons.append("two independent valid telemetry sessions were not verified")
    if cleanup_errors:
        reasons.append("one or more cleanup operations reported an error")
    supported = not reasons
    report = {
        "report_schema_version": 1,
        "generated_utc": utc_now(),
        "gate": "rival-m03-two-lane-capability-v1",
        "attempt_count": 1,
        "bounded_timeout_seconds": args.timeout,
        "smoke_game_seconds_per_lane": args.game_seconds,
        "natural_match_budget_consumed": 0,
        "supported": supported,
        "checks": {
            "server_process_and_port_isolation": server_isolation,
            "rocket_league_process_isolation": rocket_league_isolation,
            "independent_smoke_completion": match_completion,
            "telemetry_session_isolation": telemetry_isolation,
            "maximum_simultaneous_rocket_league_processes": simultaneous_rocket_league_pids,
        },
        "rejection_reasons": reasons,
        "lanes": lane_records,
        "monitor_samples": monitor_samples,
        "timed_out": timed_out,
        "cleanup_errors": cleanup_errors,
        "total_wall_duration_seconds": time.monotonic() - started,
        "fallback_execution": "sequential" if not supported else "parallel-concurrency-2",
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if supported else 2


if __name__ == "__main__":
    raise SystemExit(main())
