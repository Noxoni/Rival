from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Mapping

from analysis.tactical_metrics import TacticalMetrics
from policy.decision import PolicyDecision


class DecisionTelemetryLogger:
    """Append one machine-readable JSON object per model decision tick.

    Disabled mode is deliberately inert: it creates neither directories nor files.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = False,
        include_logits: bool = False,
    ) -> None:
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.include_logits = bool(include_logits)
        self._stream = None
        self._lock = threading.Lock()

    def _ensure_open(self) -> None:
        if self._stream is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", newline="\n", buffering=1)

    def log(
        self,
        decision: PolicyDecision,
        tactical_metrics: TacticalMetrics,
        state: Mapping[str, Any],
        runtime: Mapping[str, Any],
    ) -> bool:
        if not self.enabled:
            return False

        record = {
            "schema_version": self.SCHEMA_VERSION,
            "record_type": "rival_policy_decision",
            "decision": decision.to_record(include_logits=self.include_logits),
            "tactical_metrics": tactical_metrics.to_record(),
            "state": dict(state),
            "runtime": dict(runtime),
        }
        encoded = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

        with self._lock:
            self._ensure_open()
            assert self._stream is not None
            self._stream.write(encoded + "\n")
            self._stream.flush()
        return True

    def close(self) -> None:
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None

    def __enter__(self) -> "DecisionTelemetryLogger":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
