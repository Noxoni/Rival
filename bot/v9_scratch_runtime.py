"""Opt-in native-120-Hz deployment runtime for RivalPolicyV1.

The normal Rival bot never imports this module.  It is loaded only when an
explicit scratch export is supplied through the v9 diagnostic environment
variables.  Canonicalization and observation construction are imported from
the same ``rival_training`` modules used by RocketSim training; there is no
second live-only feature implementation.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import rlbot.flat
import torch


_BOT_ROOT = Path(__file__).resolve().parent
_REPOSITORY_ROOT = _BOT_ROOT.parent
_TRAINING_ROOT = _REPOSITORY_ROOT / "training"
if _TRAINING_ROOT.is_dir() and str(_TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAINING_ROOT))

from rival_training.v9_canonical import (  # noqa: E402
    CANONICAL_ADAPTER_VERSION,
    CANONICAL_STATE_VERSION,
    RLBotCanonicalAdapterV1,
)
from rival_training.v9_observations import (  # noqa: E402
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    RivalObsV1Builder,
    observation_schema_manifest,
)


RUNTIME_VERSION = "RivalV9ScratchRLBotRuntimeV1"
EXPORT_FORMAT = "rival-v9-torchscript-deterministic-controller-v1"
POLICY_VERSION = "RivalPolicyV1"
ACTION_VERSION = "RivalActionV1"
ACTION_SCHEMA_SHA256 = (
    "0121360ac73546911cc04dd6971ab5c53d1629c82589c00c45cb6b298a8f4163"
)
TRAINING_CONFIG_VERSION = "RivalM09TrainingConfigV1"
REWARD_VERSION = "RivalScratchRewardV1"
REWARD_SCHEDULE_VERSION = "RivalScratchRewardScheduleV1"
PREDICTION_REFRESH_TICKS = 1
CONTROLLER_FIELDS = (
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "jump",
    "boost",
    "handbrake",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _phase_name(packet: Any) -> str:
    match = getattr(packet, "match_info", None)
    return str(getattr(match, "match_phase", "Inactive")).split(".")[-1]


def _frame_number(packet: Any) -> int:
    match = getattr(packet, "match_info", None)
    frame = getattr(match, "frame_num", None)
    if frame is not None:
        return int(frame)
    seconds = float(getattr(match, "seconds_elapsed", 0.0))
    return int(round(seconds * 120.0))


def _percentiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "samples": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "maximum": None,
        }
    ordered = sorted(values)

    def pick(fraction: float) -> float:
        return float(ordered[round((len(ordered) - 1) * fraction)])

    return {
        "samples": len(values),
        "p50": pick(0.50),
        "p95": pick(0.95),
        "p99": pick(0.99),
        "maximum": float(ordered[-1]),
    }


class RivalV9ScratchRuntime:
    """Load a versioned export and emit one physical controller per packet."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path,
        *,
        runtime_evidence_path: str | Path | None = None,
        collision_mesh_directory: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.metadata_path = Path(metadata_path).expanduser().resolve()
        self.runtime_evidence_path = (
            None
            if runtime_evidence_path is None
            else Path(runtime_evidence_path).expanduser().resolve()
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Rival v9 scratch export is missing: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(
                f"Rival v9 scratch export metadata is missing: {self.metadata_path}"
            )
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self._verify_contract()
        self.model = torch.jit.load(str(self.model_path), map_location="cpu").eval()
        with torch.inference_mode():
            warmup = torch.zeros((1, OBSERVATION_SIZE), dtype=torch.float32)
            for _ in range(16):
                self._validate_outputs(self.model(warmup))
        self.adapter = RLBotCanonicalAdapterV1()
        meshes = (
            Path(collision_mesh_directory)
            if collision_mesh_directory is not None
            else _BOT_ROOT / "collision_meshes"
        )
        self.observation_builder = RivalObsV1Builder(
            prediction_refresh_ticks=PREDICTION_REFRESH_TICKS,
            collision_mesh_directory=meshes,
        )
        self.zero_controller = rlbot.flat.ControllerState()
        self.last_controller = self.zero_controller
        self.last_controller_row = np.zeros(8, dtype=np.float32)
        self.last_frame: int | None = None
        self.last_active = False
        self.decision_index = 0
        self.duplicate_packets = 0
        self.out_of_order_packets = 0
        self.reset_count = 0
        self.frame_deltas: Counter[int] = Counter()
        self.pipeline_milliseconds: dict[str, list[float]] = {
            "canonical_adapter": [],
            "observation": [],
            "actor": [],
            "controller": [],
            "total": [],
        }
        self.non_finite_outputs = 0
        self.illegal_controllers = 0
        self.button_combo_counts: Counter[int] = Counter()
        self.last_parameters: dict[str, list[float] | int] | None = None

    def _verify_contract(self) -> None:
        metadata = self.metadata
        artifact = metadata.get("artifact", {})
        contract = metadata.get("contract", {})
        expected = {
            "export_format": EXPORT_FORMAT,
            "policy_version": POLICY_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "observation_schema_sha256": observation_schema_manifest()[
                "schema_sha256"
            ],
            "observation_size": OBSERVATION_SIZE,
            "action_version": ACTION_VERSION,
            "action_schema_sha256": ACTION_SCHEMA_SHA256,
            "canonical_state_version": CANONICAL_STATE_VERSION,
            "canonical_adapter_version": CANONICAL_ADAPTER_VERSION,
            "training_config_version": TRAINING_CONFIG_VERSION,
            "reward_version": REWARD_VERSION,
            "reward_schedule_version": REWARD_SCHEDULE_VERSION,
            "physics_hz": 120,
            "policy_hz": 120,
            "prediction_refresh_ticks": PREDICTION_REFRESH_TICKS,
            "repeat_action": False,
        }
        mismatches = {
            key: {"metadata": contract.get(key), "runtime": value}
            for key, value in expected.items()
            if contract.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"Rival v9 export contract mismatch: {mismatches}")
        actual_artifact = {
            "size_bytes": self.model_path.stat().st_size,
            "sha256": _sha256(self.model_path),
        }
        recorded_artifact = {
            "size_bytes": artifact.get("size_bytes"),
            "sha256": artifact.get("sha256"),
        }
        if actual_artifact != recorded_artifact:
            raise RuntimeError(
                "Rival v9 export artifact hash/size mismatch: "
                f"expected {recorded_artifact}, got {actual_artifact}"
            )
        if tuple(contract.get("controller_fields", ())) != CONTROLLER_FIELDS:
            raise RuntimeError("Rival v9 controller-field order mismatch")

    def reset(self) -> None:
        self.adapter.reset()
        self.observation_builder.reset()
        self.last_controller = self.zero_controller
        self.last_controller_row = np.zeros(8, dtype=np.float32)
        self.last_frame = None
        self.last_active = False
        self.reset_count += 1

    @staticmethod
    def _validate_outputs(outputs: Any) -> tuple[torch.Tensor, ...]:
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 4:
            raise RuntimeError("Rival v9 export must return four tensors")
        mean, log_std, logits, controller = outputs
        expected_shapes = ((1, 5), (1, 5), (1, 8), (1, 8))
        for name, value, shape in zip(
            ("analog_mean", "analog_log_std", "button_logits", "controller"),
            (mean, log_std, logits, controller),
            expected_shapes,
        ):
            if tuple(value.shape) != shape:
                raise RuntimeError(
                    f"Rival v9 export {name} shape {tuple(value.shape)} != {shape}"
                )
            if not bool(torch.isfinite(value).all().item()):
                raise FloatingPointError(f"Rival v9 export {name} is non-finite")
        return mean, log_std, logits, controller

    @staticmethod
    def _legal_controller(row: np.ndarray) -> bool:
        return bool(
            row.shape == (8,)
            and np.isfinite(row).all()
            and np.all(row[:5] >= -1.0)
            and np.all(row[:5] <= 1.0)
            and np.all(np.isin(row[5:], (0.0, 1.0)))
        )

    @staticmethod
    def _to_controller(row: np.ndarray) -> rlbot.flat.ControllerState:
        return rlbot.flat.ControllerState(
            float(row[0]),
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            bool(row[5] > 0.5),
            bool(row[6] > 0.5),
            bool(row[7] > 0.5),
        )

    def step(
        self,
        packet: Any,
        *,
        self_index: int,
        field_info: Any | None,
    ) -> rlbot.flat.ControllerState:
        active = _phase_name(packet) in {"Countdown", "Kickoff", "Active"}
        has_ball = bool(getattr(packet, "balls", None))
        if not active or not has_ball:
            self.last_active = False
            self.last_controller = self.zero_controller
            self.last_controller_row.fill(0.0)
            return self.last_controller

        frame = _frame_number(packet)
        if not self.last_active:
            self.adapter.reset()
            self.observation_builder.reset()
            self.last_frame = None
        self.last_active = True
        if self.last_frame is not None:
            delta = frame - self.last_frame
            if delta == 0:
                self.duplicate_packets += 1
                return self.last_controller
            if delta < 0:
                self.out_of_order_packets += 1
                self.adapter.reset()
                self.observation_builder.reset()
            else:
                self.frame_deltas[delta] += 1
        self.last_frame = frame

        started = time.perf_counter_ns()
        canonical = self.adapter.adapt(packet, self_index, field_info)
        adapted = time.perf_counter_ns()
        observation = self.observation_builder.build(canonical)
        observed = time.perf_counter_ns()
        with torch.inference_mode():
            outputs = self.model(torch.from_numpy(observation).unsqueeze(0))
        inferred = time.perf_counter_ns()
        try:
            mean, log_std, logits, controller = self._validate_outputs(outputs)
        except FloatingPointError:
            self.non_finite_outputs += 1
            raise
        row = np.ascontiguousarray(controller[0].detach().cpu().numpy(), dtype=np.float32)
        if not self._legal_controller(row):
            self.illegal_controllers += 1
            raise RuntimeError(f"Rival v9 export emitted an illegal controller: {row}")
        result = self._to_controller(row)
        finished = time.perf_counter_ns()

        combo = int(row[5]) + 2 * int(row[6]) + 4 * int(row[7])
        self.button_combo_counts[combo] += 1
        self.last_parameters = {
            "analog_mean": mean[0].detach().cpu().tolist(),
            "analog_log_std": log_std[0].detach().cpu().tolist(),
            "button_logits": logits[0].detach().cpu().tolist(),
            "button_combo": combo,
        }
        self.last_controller = result
        self.last_controller_row = row
        self.decision_index += 1
        for name, left, right in (
            ("canonical_adapter", started, adapted),
            ("observation", adapted, observed),
            ("actor", observed, inferred),
            ("controller", inferred, finished),
            ("total", started, finished),
        ):
            self.pipeline_milliseconds[name].append((right - left) / 1e6)
        return result

    def summary(self) -> dict[str, Any]:
        positive_deltas = {
            str(delta): count for delta, count in sorted(self.frame_deltas.items())
        }
        missed_source_frames = sum(
            max(0, delta - 1) * count for delta, count in self.frame_deltas.items()
        )
        return {
            "schema_version": 1,
            "runtime_version": RUNTIME_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_mode": "opt_in_scratch_candidate",
            "model": {
                "sha256": _sha256(self.model_path),
                "size_bytes": self.model_path.stat().st_size,
                "metadata_sha256": _sha256(self.metadata_path),
            },
            "decision_count": self.decision_index,
            "duplicate_packets_reused_previous_controller": self.duplicate_packets,
            "out_of_order_packets": self.out_of_order_packets,
            "frame_delta_counts": positive_deltas,
            "missed_source_frames": missed_source_frames,
            "reset_count": self.reset_count,
            "non_finite_outputs": self.non_finite_outputs,
            "illegal_controllers": self.illegal_controllers,
            "button_combo_counts": {
                str(combo): self.button_combo_counts.get(combo, 0)
                for combo in range(8)
            },
            "pipeline_milliseconds": {
                name: _percentiles(values)
                for name, values in self.pipeline_milliseconds.items()
            },
            "timing_contract": {
                "physics_hz": 120,
                "policy_hz": 120,
                "one_decision_per_unique_active_packet": True,
                "duplicate_packet_behavior": "reuse previous controller exactly",
                "missed_packet_behavior": (
                    "RLBot retains the previously returned controller; no missing frame "
                    "is fabricated or interpolated"
                ),
            },
        }

    def finalize(self) -> dict[str, Any]:
        report = self.summary()
        if self.runtime_evidence_path is not None:
            self.runtime_evidence_path.parent.mkdir(parents=True, exist_ok=True)
            self.runtime_evidence_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        return report


def validate_runtime_constants() -> dict[str, Any]:
    """Expose the deploy-side frozen contract for unit/fresh-process probes."""

    return {
        "runtime_version": RUNTIME_VERSION,
        "export_format": EXPORT_FORMAT,
        "policy_version": POLICY_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "observation_schema_sha256": observation_schema_manifest()["schema_sha256"],
        "observation_size": OBSERVATION_SIZE,
        "action_version": ACTION_VERSION,
        "action_schema_sha256": ACTION_SCHEMA_SHA256,
        "canonical_state_version": CANONICAL_STATE_VERSION,
        "canonical_adapter_version": CANONICAL_ADAPTER_VERSION,
        "training_config_version": TRAINING_CONFIG_VERSION,
        "reward_version": REWARD_VERSION,
        "reward_schedule_version": REWARD_SCHEDULE_VERSION,
        "prediction_refresh_ticks": PREDICTION_REFRESH_TICKS,
        "physics_hz": 120,
        "policy_hz": 120,
        "repeat_action": False,
        "controller_fields": list(CONTROLLER_FIELDS),
    }
