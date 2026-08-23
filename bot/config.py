import os
import json
from pathlib import Path
from typing import Any

from backend.model import ActivationType, ModelInfo

RLBOT_AGENT_ID = "noxoni/rival/dev-v1"  # Must match rival.bot.toml

MAX_PLAYERS_PER_TEAM = 3

_BASE_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _BASE_DIR / "models"


def _environment_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _environment_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        converted = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    return converted


def _environment_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


_candidate_model_raw = os.environ.get("RIVAL_CANDIDATE_MODEL_PATH", "").strip()
CANDIDATE_POLICY_ENABLED = bool(_candidate_model_raw)
_candidate_runtime_label_raw = os.environ.get(
    "RIVAL_CANDIDATE_RUNTIME_LABEL", ""
).strip()
CANDIDATE_MODEL_PATH = (
    Path(_candidate_model_raw).expanduser().resolve()
    if CANDIDATE_POLICY_ENABLED
    else None
)
CANDIDATE_ACTION_TABLE_PATH = (
    Path(
        os.environ.get(
            "RIVAL_CANDIDATE_ACTION_TABLE_PATH",
            str(_MODELS_DIR / "RIVAL_ACTIONS_V1.npy"),
        )
    )
    .expanduser()
    .resolve()
    if CANDIDATE_POLICY_ENABLED
    else None
)

if CANDIDATE_POLICY_ENABLED:
    MODEL_INFO_POLICY = ModelInfo(CANDIDATE_MODEL_PATH, ActivationType.RELU)
    MODEL_INFO_SHARED_HEAD = None
    POLICY_RUNTIME_MODE = (
        _candidate_runtime_label_raw or "milestone06_trained_candidate"
    )
else:
    MODEL_INFO_POLICY = ModelInfo(_MODELS_DIR / "POLICY.lt", ActivationType.RELU)
    MODEL_INFO_SHARED_HEAD = ModelInfo(_MODELS_DIR / "SHARED_HEAD.lt", ActivationType.RELU)
    POLICY_RUNTIME_MODE = "frozen_wisp_production"

TRANSFER_DIAGNOSTIC_MODE = _environment_flag(
    "RIVAL_TRANSFER_DIAGNOSTIC_MODE", False
)
CANDIDATE_LEGACY_ONLY = _environment_flag("RIVAL_CANDIDATE_LEGACY_ONLY", False)
M08_DUAL_RATE_ENABLED = _environment_flag("RIVAL_M08_DUAL_RATE_ENABLED", False)
M08_MECHANICS_FORCE_PASS = _environment_flag("RIVAL_M08_MECHANICS_FORCE_PASS", False)
_m08_mechanics_model_raw = os.environ.get(
    "RIVAL_M08_MECHANICS_MODEL_PATH", ""
).strip()
M08_MECHANICS_MODEL_PATH = (
    Path(_m08_mechanics_model_raw).expanduser().resolve()
    if _m08_mechanics_model_raw
    else None
)
M08_MECHANICS_MODEL_INFO = (
    ModelInfo(M08_MECHANICS_MODEL_PATH, ActivationType.RELU)
    if M08_MECHANICS_MODEL_PATH is not None
    else None
)
M08_ACTION_TABLE_PATH = (
    Path(
        os.environ.get(
            "RIVAL_M08_ACTION_TABLE_PATH",
            str(_MODELS_DIR / "RIVAL_ACTIONS_V1.npy"),
        )
    )
    .expanduser()
    .resolve()
    if M08_DUAL_RATE_ENABLED
    else None
)
_m08_runtime_label_raw = os.environ.get(
    "RIVAL_M08_RUNTIME_LABEL", ""
).strip()
M08_MECHANICS_DETERMINISTIC = _environment_flag(
    "RIVAL_M08_MECHANICS_DETERMINISTIC", True
)
if M08_DUAL_RATE_ENABLED:
    if not TRANSFER_DIAGNOSTIC_MODE:
        raise ValueError(
            "Milestone 08 dual-rate runtime requires explicit "
            "RIVAL_TRANSFER_DIAGNOSTIC_MODE"
        )
    if CANDIDATE_POLICY_ENABLED:
        raise ValueError(
            "Milestone 08 dual-rate runtime cannot be combined with the rejected "
            "monolithic candidate path"
        )
    if not M08_MECHANICS_FORCE_PASS and M08_MECHANICS_MODEL_PATH is None:
        raise ValueError(
            "Milestone 08 dual-rate runtime needs a mechanics model unless forced PASS"
        )
    if _m08_runtime_label_raw and any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
        for character in _m08_runtime_label_raw
    ):
        raise ValueError(
            "RIVAL_M08_RUNTIME_LABEL must use lowercase letters, digits, dash, "
            "or underscore"
        )
    POLICY_RUNTIME_MODE = _m08_runtime_label_raw or (
        "m08_dual_rate_force_pass"
        if M08_MECHANICS_FORCE_PASS
        else "m08_dual_rate_candidate"
    )
if _candidate_runtime_label_raw and not (
    CANDIDATE_POLICY_ENABLED and TRANSFER_DIAGNOSTIC_MODE
):
    raise ValueError(
        "RIVAL_CANDIDATE_RUNTIME_LABEL requires a candidate model and explicit "
        "RIVAL_TRANSFER_DIAGNOSTIC_MODE"
    )
if _candidate_runtime_label_raw and any(
    character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
    for character in _candidate_runtime_label_raw
):
    raise ValueError(
        "RIVAL_CANDIDATE_RUNTIME_LABEL must use lowercase letters, digits, dash, "
        "or underscore"
    )

TICK_SKIP = int(
    os.environ.get(
        "RIVAL_TICK_SKIP",
        "8" if M08_DUAL_RATE_ENABLED else "4" if CANDIDATE_POLICY_ENABLED else "8",
    )
)
if M08_DUAL_RATE_ENABLED and TICK_SKIP != 8:
    raise ValueError("Milestone 08 dual-rate strategic clock must remain tick skip 8")
if CANDIDATE_POLICY_ENABLED and TRANSFER_DIAGNOSTIC_MODE:
    if TICK_SKIP not in {4, 8}:
        raise ValueError(
            "Milestone 07 transfer diagnostics require RIVAL_TICK_SKIP=4 or 8"
        )
elif CANDIDATE_POLICY_ENABLED and TICK_SKIP != 4:
    raise ValueError(
        "Candidate deployment requires RIVAL_TICK_SKIP=4 unless explicit "
        "RIVAL_TRANSFER_DIAGNOSTIC_MODE is enabled"
    )
if not CANDIDATE_POLICY_ENABLED and TICK_SKIP != 8:
    raise ValueError("Frozen Wisp production deployment requires tick skip 8")
if CANDIDATE_LEGACY_ONLY and not (
    CANDIDATE_POLICY_ENABLED and TRANSFER_DIAGNOSTIC_MODE
):
    raise ValueError(
        "RIVAL_CANDIDATE_LEGACY_ONLY requires both a candidate model and explicit "
        "RIVAL_TRANSFER_DIAGNOSTIC_MODE"
    )
ACTION_DELAY = TICK_SKIP - 1

MODEL_DEVICE = "cpu"

# Infer bot deterministically (stochastic otherwise)
DETERMINISTIC = True

ROCKETSIM_COLLISION_DIR = _BASE_DIR / "collision_meshes"


POLICY_TOP_N = int(os.environ.get("RIVAL_POLICY_TOP_N", "5"))
TELEMETRY_ENABLED = _environment_flag("RIVAL_TELEMETRY_ENABLED", False)
TELEMETRY_INCLUDE_LOGITS = _environment_flag(
    "RIVAL_TELEMETRY_INCLUDE_LOGITS", False
)
DIAGNOSTIC_CAPTURE_OBSERVATIONS = _environment_flag(
    "RIVAL_DIAGNOSTIC_CAPTURE_OBSERVATIONS", False
)
DIAGNOSTIC_OBSERVATION_STRIDE = _environment_int(
    "RIVAL_DIAGNOSTIC_OBSERVATION_STRIDE", 8
)
if DIAGNOSTIC_CAPTURE_OBSERVATIONS and not TRANSFER_DIAGNOSTIC_MODE:
    raise ValueError(
        "RIVAL_DIAGNOSTIC_CAPTURE_OBSERVATIONS requires explicit "
        "RIVAL_TRANSFER_DIAGNOSTIC_MODE"
    )
if DIAGNOSTIC_CAPTURE_OBSERVATIONS and not TELEMETRY_ENABLED:
    raise ValueError(
        "RIVAL_DIAGNOSTIC_CAPTURE_OBSERVATIONS requires RIVAL_TELEMETRY_ENABLED"
    )
if DIAGNOSTIC_OBSERVATION_STRIDE < 1:
    raise ValueError("RIVAL_DIAGNOSTIC_OBSERVATION_STRIDE must be at least 1")
TELEMETRY_PATH = Path(
    os.environ.get(
        "RIVAL_TELEMETRY_PATH",
        str(_BASE_DIR.parent / "telemetry" / "rival_decisions.jsonl"),
    )
).expanduser()


def _load_session_metadata() -> dict[str, Any]:
    raw_path = os.environ.get("RIVAL_SESSION_METADATA_PATH")
    if not raw_path:
        return {}
    path = Path(raw_path).expanduser()
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("RIVAL_SESSION_METADATA_PATH must contain a JSON object")
    return value


TELEMETRY_SESSION_METADATA = _load_session_metadata()

CHALLENGE_CALIBRATION_MODE = os.environ.get(
    "RIVAL_CHALLENGE_CALIBRATION_MODE", "off"
).strip().lower()
if CHALLENGE_CALIBRATION_MODE not in {"off", "observe", "intervene"}:
    raise ValueError(
        "RIVAL_CHALLENGE_CALIBRATION_MODE must be off, observe, or intervene; "
        f"got {CHALLENGE_CALIBRATION_MODE!r}"
    )

# The environment surface is intentionally small and named so every controlled
# parameter attempt can be reconstructed without changing the frozen Wisp policy.
CHALLENGE_PARAMETER_VERSION = os.environ.get(
    "RIVAL_CHALLENGE_PARAMETER_VERSION", "m03-conservative-v1"
).strip()
if not CHALLENGE_PARAMETER_VERSION:
    raise ValueError("RIVAL_CHALLENGE_PARAMETER_VERSION cannot be empty")
CHALLENGE_LOW_THRESHOLD = _environment_float("RIVAL_CHALLENGE_LOW_THRESHOLD", 0.34)
CHALLENGE_HIGH_THRESHOLD = _environment_float("RIVAL_CHALLENGE_HIGH_THRESHOLD", 0.70)
CHALLENGE_PRESSURE_DISTANCE = _environment_float(
    "RIVAL_CHALLENGE_PRESSURE_DISTANCE", 1900.0
)
CHALLENGE_PRESSURE_ETA = _environment_float("RIVAL_CHALLENGE_PRESSURE_ETA", 1.40)
CHALLENGE_PROJECTED_MISS_REFERENCE = _environment_float(
    "RIVAL_CHALLENGE_PROJECTED_MISS_REFERENCE", 450.0
)
CHALLENGE_CONTROL_DISTANCE = _environment_float(
    "RIVAL_CHALLENGE_CONTROL_DISTANCE", 650.0
)
CHALLENGE_MAX_LOGIT_GAP = _environment_float(
    "RIVAL_CHALLENGE_MAX_LOGIT_GAP", 0.85
)
CHALLENGE_MAX_DEFERRAL_TICKS = _environment_int(
    "RIVAL_CHALLENGE_MAX_DEFERRAL_TICKS", 1
)

NATURAL_ADJUSTMENT_MODE = os.environ.get(
    "RIVAL_NATURAL_ADJUSTMENT_MODE", "off"
).strip().lower()
if NATURAL_ADJUSTMENT_MODE not in {"off", "observe", "intervene"}:
    raise ValueError(
        "RIVAL_NATURAL_ADJUSTMENT_MODE must be off, observe, or intervene; "
        f"got {NATURAL_ADJUSTMENT_MODE!r}"
    )
NATURAL_PARAMETER_VERSION = os.environ.get(
    "RIVAL_NATURAL_PARAMETER_VERSION", "none"
).strip()
if not NATURAL_PARAMETER_VERSION:
    raise ValueError("RIVAL_NATURAL_PARAMETER_VERSION cannot be empty")
if (
    NATURAL_ADJUSTMENT_MODE != "off"
    and NATURAL_PARAMETER_VERSION != "m04p1-low-resource-aerial-v1"
):
    raise ValueError(
        "active natural adjustment requires parameter version "
        "'m04p1-low-resource-aerial-v1'; got "
        f"{NATURAL_PARAMETER_VERSION!r}"
    )
if NATURAL_ADJUSTMENT_MODE != "off" and CHALLENGE_CALIBRATION_MODE != "off":
    raise ValueError(
        "natural adjustment and challenge calibration cannot be active together"
    )

# Milestone 01 is measurement-only. This is intentionally not environment-toggleable.
STRATEGIC_OVERRIDES_ENABLED = False
