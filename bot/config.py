import os
import json
from pathlib import Path
from typing import Any

from backend.model import ActivationType, ModelInfo

RLBOT_AGENT_ID = "noxoni/rival/dev-v1"  # Must match rival.bot.toml

TICK_SKIP = 8
ACTION_DELAY = TICK_SKIP - 1
MAX_PLAYERS_PER_TEAM = 3

_BASE_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _BASE_DIR / "models"

MODEL_INFO_POLICY = ModelInfo(_MODELS_DIR / "POLICY.lt", ActivationType.RELU)
MODEL_INFO_SHARED_HEAD = ModelInfo(_MODELS_DIR / "SHARED_HEAD.lt", ActivationType.RELU)
#                         ^ Set to None if you don't have a shared head export

MODEL_DEVICE = "cpu"

# Infer bot deterministically (stochastic otherwise)
DETERMINISTIC = True

ROCKETSIM_COLLISION_DIR = _BASE_DIR / "collision_meshes"


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


POLICY_TOP_N = int(os.environ.get("RIVAL_POLICY_TOP_N", "5"))
TELEMETRY_ENABLED = _environment_flag("RIVAL_TELEMETRY_ENABLED", False)
TELEMETRY_INCLUDE_LOGITS = _environment_flag(
    "RIVAL_TELEMETRY_INCLUDE_LOGITS", False
)
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

# Milestone 01 is measurement-only. This is intentionally not environment-toggleable.
STRATEGIC_OVERRIDES_ENABLED = False
