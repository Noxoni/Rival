"""Bounded, opt-in native-tick packet recorder for Rival v9 diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Mapping

from .packet_snapshot import extract_packet_snapshot


NATIVE_CORPUS_VERSION = "RivalV9NativePacketCorpusV2"


def _controller_record(controller: Any) -> dict[str, Any]:
    return {
        "throttle": float(getattr(controller, "throttle", 0.0)),
        "steer": float(getattr(controller, "steer", 0.0)),
        "pitch": float(getattr(controller, "pitch", 0.0)),
        "yaw": float(getattr(controller, "yaw", 0.0)),
        "roll": float(getattr(controller, "roll", 0.0)),
        "jump": bool(getattr(controller, "jump", False)),
        "boost": bool(getattr(controller, "boost", False)),
        "handbrake": bool(getattr(controller, "handbrake", False)),
    }


class NativePacketCorpusLogger:
    """Write at most one record per unique RLBot physics frame.

    The logger is inert unless explicitly enabled.  It stops writing after the
    configured bound, while retaining counters for duplicate callbacks and
    skipped source frames.  Batches are flushed once per simulated second so a
    stopped diagnostic remains recoverable without forcing a disk flush on
    every 120-Hz callback.
    """

    SCHEMA_VERSION = 2

    def __init__(
        self,
        path: str | Path,
        *,
        enabled: bool = False,
        maximum_records: int = 6000,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if maximum_records < 1:
            raise ValueError("Native packet corpus maximum_records must be positive")
        self.path = Path(path)
        self.enabled = bool(enabled)
        self.maximum_records = int(maximum_records)
        self.metadata = dict(metadata or {})
        self._stream = None
        self._last_frame: int | None = None
        self._records = 0
        self._duplicates = 0
        self._skipped_frames = 0
        self._out_of_order = 0
        self._started_ns: int | None = None

    def _ensure_open(self) -> None:
        if self._stream is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a", encoding="utf-8", newline="\n", buffering=262144)
        self._started_ns = time.perf_counter_ns()
        self._write(
            {
                "schema_version": self.SCHEMA_VERSION,
                "record_type": "rival_v9_native_corpus_start",
                "corpus_version": NATIVE_CORPUS_VERSION,
                "maximum_records": self.maximum_records,
                "metadata": self.metadata,
            }
        )
        self._stream.flush()

    def _write(self, value: Mapping[str, Any]) -> None:
        assert self._stream is not None
        self._stream.write(
            json.dumps(dict(value), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            + "\n"
        )

    @staticmethod
    def _frame(packet: Any) -> int:
        info = getattr(packet, "match_info", None)
        frame = getattr(info, "frame_num", None)
        if frame is not None:
            return int(frame)
        elapsed = float(getattr(info, "seconds_elapsed", 0.0))
        return int(round(elapsed * 120.0))

    def log(
        self,
        packet: Any,
        *,
        self_index: int,
        field_info: Any,
        controller_output: Any,
        callback_started_ns: int,
        callback_finished_ns: int,
    ) -> bool:
        if not self.enabled or self._records >= self.maximum_records:
            return False
        frame = self._frame(packet)
        if self._last_frame is not None:
            delta = frame - self._last_frame
            if delta == 0:
                self._duplicates += 1
                return False
            if delta < 0:
                self._out_of_order += 1
            elif delta > 1:
                self._skipped_frames += delta - 1
        self._ensure_open()
        self._write(
            {
                "schema_version": self.SCHEMA_VERSION,
                "record_type": "rival_v9_native_packet",
                "corpus_version": NATIVE_CORPUS_VERSION,
                "sequence": self._records,
                "frame_num": frame,
                "callback_started_ns": int(callback_started_ns),
                "callback_finished_ns": int(callback_finished_ns),
                "callback_wall_ns": int(callback_finished_ns - callback_started_ns),
                "controller_output": _controller_record(controller_output),
                "packet": extract_packet_snapshot(packet, self_index, field_info),
            }
        )
        self._records += 1
        self._last_frame = frame
        if self._records % 120 == 0:
            assert self._stream is not None
            self._stream.flush()
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "corpus_version": NATIVE_CORPUS_VERSION,
            "records": self._records,
            "maximum_records": self.maximum_records,
            "duplicates_ignored": self._duplicates,
            "skipped_source_frames": self._skipped_frames,
            "out_of_order_frames": self._out_of_order,
            "complete_bound_reached": self._records >= self.maximum_records,
            "wall_seconds": (
                None
                if self._started_ns is None
                else (time.perf_counter_ns() - self._started_ns) / 1e9
            ),
        }

    def close(self) -> None:
        if self._stream is None:
            return
        self._write(
            {
                "schema_version": self.SCHEMA_VERSION,
                "record_type": "rival_v9_native_corpus_end",
                **self.summary(),
            }
        )
        self._stream.flush()
        self._stream.close()
        self._stream = None
