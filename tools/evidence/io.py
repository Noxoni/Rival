from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .references import sha256_file


@dataclass
class EvidenceSession:
    session_id: str
    raw_path: Path
    raw_sha256: str
    metadata: dict[str, Any]
    manifest: dict[str, Any]
    decisions: list[dict[str, Any]]
    warnings: list[str]

    @property
    def opponent(self) -> str:
        opponent = self.metadata.get("opponent", {})
        if isinstance(opponent, dict):
            return str(opponent.get("identity") or opponent.get("key") or "unknown")
        return str(opponent or "unknown")

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", "unknown"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def discover_jsonl(inputs: Iterable[Path]) -> list[Path]:
    result: set[Path] = set()
    for value in inputs:
        path = value.resolve()
        if path.is_file() and path.suffix.lower() == ".jsonl":
            result.add(path)
        elif path.is_dir():
            result.update(candidate.resolve() for candidate in path.rglob("*.jsonl"))
    return sorted(result)


def load_session(path: Path) -> EvidenceSession:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            warnings.append(f"invalid_json_line:{line_number}:{exc.msg}")
            continue
        if not isinstance(record, dict):
            warnings.append(f"non_object_line:{line_number}")
            continue
        schema = record.get("schema_version")
        if schema not in (1, 2, 3):
            warnings.append(f"unsupported_schema:{line_number}:{schema}")
            continue
        record["_raw_line"] = line_number
        records.append(record)

    start = next(
        (record for record in records if record.get("record_type") == "rival_session_start"),
        {},
    )
    metadata = dict(start.get("metadata") or {})
    manifest = _read_json(path.parent / "session_manifest.json")
    if manifest:
        metadata = {**metadata, **{key: value for key, value in manifest.items() if key not in {"schedule"}}}
    session_id = str(
        start.get("session_id")
        or manifest.get("session_id")
        or next((record.get("session_id") for record in records if record.get("session_id")), None)
        or f"legacy-v1-{path.stem}"
    )
    decisions = [
        record
        for record in records
        if record.get("record_type") == "rival_policy_decision"
    ]
    for index, decision in enumerate(decisions):
        decision["_decision_index"] = index
        decision.setdefault("session_id", session_id)

    previous_time: float | None = None
    seen_ticks: set[Any] = set()
    for decision in decisions:
        tick = (decision.get("decision") or {}).get("tick")
        game_time = (decision.get("decision") or {}).get("game_time")
        if tick in seen_ticks:
            warnings.append(f"duplicate_decision_tick:{tick}")
        seen_ticks.add(tick)
        if isinstance(game_time, (int, float)):
            if previous_time is not None and game_time + 0.1 < previous_time:
                warnings.append(f"non_monotonic_game_time:{previous_time}:{game_time}")
            previous_time = float(game_time)
    if not decisions:
        warnings.append("no_decision_records")

    return EvidenceSession(
        session_id=session_id,
        raw_path=path,
        raw_sha256=sha256_file(path),
        metadata=metadata,
        manifest=manifest,
        decisions=decisions,
        warnings=warnings,
    )


def load_sessions(inputs: Iterable[Path]) -> list[EvidenceSession]:
    return [load_session(path) for path in discover_jsonl(inputs)]
