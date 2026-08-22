from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

from .references import REPOSITORY_ROOT, ReferenceBot, sha256_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def make_session_id(source: str, opponent: str, rival_team: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    side = "blue" if rival_team == 0 else "orange"
    return f"rival-v2-{source}-{opponent}-{side}-{stamp}-{uuid4().hex[:8]}"


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def model_hashes() -> dict[str, str]:
    model_root = REPOSITORY_ROOT / "bot" / "models"
    return {
        path.name: sha256_file(path)
        for path in sorted(model_root.glob("*.lt"))
    }


def runtime_versions() -> dict[str, str | None]:
    return {
        "python": sys.version.split()[0],
        "rlbot": _package_version("rlbot"),
        "torch": _package_version("torch"),
        "rocketsim": _package_version("rocketsim"),
    }


def build_session_metadata(
    *,
    session_id: str,
    source: str,
    opponent: ReferenceBot | dict[str, Any],
    rival_team: int,
    match: dict[str, Any],
    telemetry_path: Path,
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    opponent_record = opponent.to_record() if isinstance(opponent, ReferenceBot) else opponent
    return {
        "session_id": session_id,
        "source": source,
        "rival_git_commit": git_commit(),
        "rival_model_sha256": model_hashes(),
        "opponent": opponent_record,
        "rival_team": rival_team,
        "team_assignment": {
            "rival": "blue" if rival_team == 0 else "orange",
            "opponent": "orange" if rival_team == 0 else "blue",
        },
        "match": match,
        "probe": probe,
        "runtime_versions": runtime_versions(),
        "telemetry_configuration": {
            "schema_version": 2,
            "include_logits": False,
            "decision_path": str(telemetry_path),
            "deterministic_policy": True,
            "strategic_overrides_enabled": False,
        },
        "start_timestamp_utc": utc_now(),
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def summarize_telemetry(path: Path) -> dict[str, Any]:
    counts: dict[str, int] = {}
    invalid = 0
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                record_type = str(record.get("record_type", "unknown"))
                counts[record_type] = counts.get(record_type, 0) + 1
            except (json.JSONDecodeError, AttributeError):
                invalid += 1
    return {
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "sha256": sha256_file(path) if path.is_file() else None,
        "record_counts": counts,
        "invalid_record_count": invalid,
    }
