"""RLBot v5 entry point for the official fail-closed Rival capability bundle."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import rlbot.flat
import rlbot.managers
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "rival2_live"))
from runtime import Rival2LiveRuntime  # noqa: E402

AGENT_ID = "noxoni/rival2/official-v1"
BUNDLE = ROOT / "deployment_bundle.json"


def _bundle() -> dict:
    value = json.loads(BUNDLE.read_text(encoding="utf-8"))
    if value.get("format") != "RIVAL2_RLBOT_OFFICIAL_BUNDLE_V1":
        raise RuntimeError("unsupported Rival official deployment bundle")
    if value["specialist_status"]["automatic_takeover"] is not False:
        raise RuntimeError("official play build must be fail-closed")
    return value


def _active(team: int) -> tuple[Path, Path, dict]:
    if team not in (0, 1):
        raise RuntimeError(f"invalid Rival team index: {team}")
    bundle = _bundle()
    side = "blue" if team == 0 else "orange"
    name = bundle["active_default"][side]
    component = bundle["components"][name]
    return ROOT / component["model"], ROOT / component["manifest"], component


def _controller(row: np.ndarray) -> rlbot.flat.ControllerState:
    return rlbot.flat.ControllerState(
        throttle=float(row[0]),
        steer=float(row[1]),
        pitch=float(row[2]),
        yaw=float(row[3]),
        roll=float(row[4]),
        jump=bool(row[5] >= 0.5),
        boost=bool(row[6] >= 0.5),
        handbrake=bool(row[7] >= 0.5),
    )


class Rival2OfficialBot(rlbot.managers.Bot):
    def __init__(self):
        super().__init__(AGENT_ID)
        self.runtime: Rival2LiveRuntime | None = None
        self.component: str | None = None

    def initialize(self):
        torch.set_num_threads(1)
        model, manifest, component = _active(int(self.team))
        self.component = component["source"]["component"]
        self.runtime = Rival2LiveRuntime(model, manifest, self.field_info)
        self.logger.info(
            "Rival official ready: component=%s checkpoint=%s",
            self.component,
            component["source"]["checkpoint_sha256"],
        )

    def get_output(self, packet: rlbot.flat.GamePacket) -> rlbot.flat.ControllerState:
        if self.runtime is None:
            return rlbot.flat.ControllerState()
        return _controller(self.runtime.step(packet, self.team))

    def retire(self):
        if self.runtime is not None:
            summary = self.runtime.summary()
            summary["active_component"] = self.component
            self.logger.info("Rival official runtime summary: %s", summary)


def self_test() -> int:
    bundle = _bundle()
    results = {}
    for name, component in bundle["components"].items():
        model_path = ROOT / component["model"]
        manifest_path = ROOT / component["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source"]["checkpoint_sha256"] != bundle["source"]["checkpoint_sha256"]:
            raise RuntimeError(f"{name} source checkpoint mismatch")
        if manifest["contracts"]["hold_ticks"] != 1:
            raise RuntimeError(f"{name} is not a 120 Hz deployment")
        if component["artifact"]["sha256"] != manifest["artifact"]["sha256"]:
            raise RuntimeError(f"{name} artifact manifest mismatch")
        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        with torch.inference_mode():
            action = model(torch.zeros((1, 182), dtype=torch.float32))
        row = action[0].numpy()
        Rival2LiveRuntime._validate_action(row)
        results[name] = {
            "sha256": component["artifact"]["sha256"],
            "finite": bool(torch.isfinite(action).all()),
            "controller": row.tolist(),
        }
    print(
        json.dumps(
            {
                "status": "pass",
                "source": bundle["source"],
                "active_default": bundle["active_default"],
                "specialist_status": bundle["specialist_status"],
                "components": results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        raise SystemExit(self_test())
    Rival2OfficialBot().run(wants_ball_predictions=False)
