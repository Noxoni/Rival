from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "reference_manifests" / "v1" / "MANIFEST.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReferenceBot:
    key: str
    identity: str
    root: Path
    config_path: Path
    executable_path: Path
    config_sha256: str
    executable_sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "identity": self.identity,
            "root": str(self.root),
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "executable_path": str(self.executable_path),
            "executable_sha256": self.executable_sha256,
        }


def _resolve_root(manifest: dict[str, Any], key: str) -> Path:
    recorded = Path(manifest["selected_sources"][key])
    if recorded.is_dir():
        return recorded

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        botpack = Path(local_app_data) / "RLBot5" / "bots"
        suffix = recorded.name
        candidates = list(botpack.glob(f"**/{suffix}"))
        if len(candidates) == 1:
            return candidates[0]
    raise FileNotFoundError(f"Installed {key} root was not found: {recorded}")


def discover_reference(
    key: str,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    validate: bool = True,
) -> ReferenceBot:
    key = key.lower()
    if key not in {"nexto", "wisp"}:
        raise ValueError(f"Unsupported reference bot {key!r}; expected nexto or wisp")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = _resolve_root(manifest, key)
    snapshots = manifest["snapshots"][key]

    config_entry = next(
        entry
        for entry in snapshots
        if entry["path"].lower().endswith(".toml") and "loadout" not in entry["path"].lower()
    )
    executable_entry = next(
        entry for entry in snapshots if entry["path"].lower().endswith(".exe")
    )

    if validate:
        failures: list[str] = []
        for entry in snapshots:
            path = root / Path(entry["path"])
            if not path.is_file():
                failures.append(f"missing:{entry['path']}")
                continue
            if path.stat().st_size != int(entry["size"]):
                failures.append(f"size:{entry['path']}")
                continue
            if sha256_file(path) != entry["sha256"]:
                failures.append(f"sha256:{entry['path']}")
        if failures:
            raise RuntimeError(
                f"Installed {key} reference does not match the read-only manifest: "
                + ", ".join(failures)
            )

    return ReferenceBot(
        key=key,
        identity="Nexto" if key == "nexto" else "Wisp v2-75B",
        root=root,
        config_path=root / Path(config_entry["path"]),
        executable_path=root / Path(executable_entry["path"]),
        config_sha256=config_entry["sha256"],
        executable_sha256=executable_entry["sha256"],
    )
