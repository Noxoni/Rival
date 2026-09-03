"""RLBot v5 entry point for the single-network Rival Unified V5 policy."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import rlbot.flat
import rlbot.managers
import torch

from unified_runtime import Rival2UnifiedLiveRuntime


AGENT_ID = "noxoni/rival2/unified-capability-v5"
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "rival2_unified_capability_v5.ts"
MANIFEST = ROOT / "models" / "rival2_unified_capability_v5.json"
EXPECTED_CHECKPOINT_SHA256 = (
    "955C93BF538BC913CC2E42F42E3B0EDC4CCDB1065DA9581FB88D84C363B7C216"
)


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


class Rival2UnifiedV5Bot(rlbot.managers.Bot):
    def __init__(self):
        super().__init__(AGENT_ID)
        self.runtime: Rival2UnifiedLiveRuntime | None = None

    def initialize(self):
        torch.set_num_threads(1)
        self.runtime = Rival2UnifiedLiveRuntime(MODEL, MANIFEST, self.field_info)
        source = self.runtime.manifest["source"]
        if source["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
            raise RuntimeError("Rival unified V5 source checkpoint identity mismatch")
        self.logger.info(
            "Rival Unified V5 ready: checkpoint=%s router=%s",
            source["checkpoint_sha256"],
            source["runtime_router"],
        )

    def get_output(self, packet: rlbot.flat.GamePacket) -> rlbot.flat.ControllerState:
        if self.runtime is None:
            return rlbot.flat.ControllerState()
        return _controller(self.runtime.step(packet, self.team))

    def retire(self):
        if self.runtime is not None:
            self.logger.info("Rival Unified V5 runtime summary: %s", self.runtime.summary())


def self_test() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["source"]["checkpoint_sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Rival unified V5 source checkpoint identity mismatch")
    if manifest["source"]["runtime_router"] is not False:
        raise RuntimeError("Rival unified V5 unexpectedly uses a runtime router")
    if manifest["contracts"]["hold_ticks"] != 1:
        raise RuntimeError("Rival unified V5 is not deployed at 120 Hz")
    model = torch.jit.load(str(MODEL), map_location="cpu").eval()
    hidden_shape = manifest["recurrent"]["input_hidden_shape"]
    hidden = torch.zeros((int(hidden_shape[0]), 1, int(hidden_shape[2])))
    controllers = []
    with torch.inference_mode():
        for _ in range(16):
            action, hidden = model(torch.zeros((1, 182)), hidden)
            row = action[0].numpy()
            controllers.append(row.tolist())
            from runtime import Rival2LiveRuntime

            Rival2LiveRuntime._validate_action(row)
    print(
        json.dumps(
            {
                "status": "pass",
                "source": manifest["source"],
                "artifact": manifest["artifact"],
                "recurrent": manifest["recurrent"],
                "finite_hidden": bool(torch.isfinite(hidden).all().item()),
                "first_controller": controllers[0],
                "last_controller": controllers[-1],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        raise SystemExit(self_test())
    Rival2UnifiedV5Bot().run(wants_ball_predictions=False)
