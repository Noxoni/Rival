"""Configuration helpers for the committed Milestone 05 setup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRAINING_ROOT.parent
DEFAULT_CONFIG_PATH = TRAINING_ROOT / "configs" / "milestone05.json"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve_repo_path(relative_path: str) -> Path:
    path = (REPOSITORY_ROOT / relative_path).resolve()
    path.relative_to(REPOSITORY_ROOT.resolve())
    return path
