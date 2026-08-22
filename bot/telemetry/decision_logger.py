from __future__ import annotations

import json
from pathlib import Path
import threading
from datetime import datetime, timezone
from typing import Any, Mapping
from uuid import uuid4

from analysis.tactical_metrics import TacticalMetrics
from policy.decision import PolicyDecision


class DecisionTelemetryLogger:
    """Append one machine-readable JSON object per model decision tick.

    Disabled mode is deliberately inert: it creates neither directories nor files.
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = False,
        include_logits: bool = False,
        session_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.include_logits = bool(include_logits)
        supplied = dict(session_metadata or {})
        self.session_id = str(supplied.pop("session_id", f"adhoc-{uuid4()}"))
        self.session_metadata = supplied
        self._stream = None
        self._lock = threading.Lock()
        self._started = False
        self._finalized = False
        self._decision_count = 0
        self._last_score: dict[str, int | None] | None = None

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def _ensure_open(self) -> None:
        if self._stream is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", newline="\n", buffering=1)

    def _write_record(self, record: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            dict(record),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._ensure_open()
        assert self._stream is not None
        self._stream.write(encoded + "\n")
        self._stream.flush()

    def start(self) -> bool:
        if not self.enabled or self._started:
            return False
        with self._lock:
            if self._started:
                return False
            self._write_record(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "record_type": "rival_session_start",
                    "session_id": self.session_id,
                    "timestamp_utc": self._utc_now(),
                    "metadata": self.session_metadata,
                    "telemetry": {"include_logits": self.include_logits},
                }
            )
            self._started = True
        return True

    def log(
        self,
        decision: PolicyDecision,
        tactical_metrics: TacticalMetrics,
        state: Mapping[str, Any],
        runtime: Mapping[str, Any],
        direct_packet: Mapping[str, Any] | None = None,
    ) -> bool:
        if not self.enabled:
            return False

        record = {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": "rival_policy_decision",
            "session_id": self.session_id,
            "decision": decision.to_record(include_logits=self.include_logits),
            "tactical_metrics": tactical_metrics.to_record(),
            "state": dict(state),
            "packet": None if direct_packet is None else dict(direct_packet),
            "runtime": dict(runtime),
        }

        with self._lock:
            if not self._started:
                self._write_record(
                    {
                        "schema_version": self.SCHEMA_VERSION,
                        "record_type": "rival_session_start",
                        "session_id": self.session_id,
                        "timestamp_utc": self._utc_now(),
                        "metadata": self.session_metadata,
                        "telemetry": {"include_logits": self.include_logits},
                    }
                )
                self._started = True
            self._write_record(record)
            self._decision_count += 1
        return True

    def observe_final_score(self, blue: int | None, orange: int | None) -> None:
        self._last_score = {"blue": blue, "orange": orange}

    def finalize(
        self,
        *,
        termination_reason: str = "logger_closed",
        final_score: Mapping[str, int | None] | None = None,
        replay_path: str | None = None,
    ) -> bool:
        if not self.enabled or self._finalized or not self._started:
            return False
        with self._lock:
            if self._finalized:
                return False
            self._write_record(
                {
                    "schema_version": self.SCHEMA_VERSION,
                    "record_type": "rival_session_end",
                    "session_id": self.session_id,
                    "timestamp_utc": self._utc_now(),
                    "termination_reason": termination_reason,
                    "final_score": dict(final_score or self._last_score or {}),
                    "replay_path": replay_path,
                    "decision_record_count": self._decision_count,
                }
            )
            self._finalized = True
        return True

    def close(self, *, finalize: bool = True) -> None:
        if finalize:
            self.finalize()
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def __enter__(self) -> "DecisionTelemetryLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
