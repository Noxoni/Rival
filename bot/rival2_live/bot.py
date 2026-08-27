"""RLBot v5 entry point for the frozen Rival 2 policy."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import rlbot.flat
import rlbot.managers
import torch

from runtime import Rival2LiveRuntime


AGENT_ID = "noxoni/rival2/gameplay-v2-479"
ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "rival2_gameplay_v2_479.ts"
MANIFEST = ROOT / "models" / "rival2_gameplay_v2_479.json"


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


class Rival2Bot(rlbot.managers.Bot):
    def __init__(self):
        super().__init__(AGENT_ID)
        self.runtime: Rival2LiveRuntime | None = None

    def initialize(self):
        torch.set_num_threads(1)
        self.runtime = Rival2LiveRuntime(MODEL, MANIFEST, self.field_info)
        source = self.runtime.manifest["source"]
        self.logger.info(
            "Rival 2 ready: policy=%s samples=%s checkpoint=%s",
            source["policy_version"],
            source["total_agent_samples"],
            source["checkpoint_sha256"],
        )

    def get_output(self, packet: rlbot.flat.GamePacket) -> rlbot.flat.ControllerState:
        if self.runtime is None:
            return rlbot.flat.ControllerState()
        return _controller(self.runtime.step(packet, self.team))

    def retire(self):
        if self.runtime is not None:
            self.logger.info("Rival 2 runtime summary: %s", self.runtime.summary())


def self_test() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model = torch.jit.load(str(MODEL), map_location="cpu").eval()
    observation = torch.zeros((1, manifest["observation"]["dimension"]), dtype=torch.float32)
    with torch.inference_mode():
        action = model(observation)
    row = action[0].numpy()
    Rival2LiveRuntime._validate_action(row)
    print(
        json.dumps(
            {
                "status": "pass",
                "source": manifest["source"],
                "artifact": manifest["artifact"],
                "output_shape": list(action.shape),
                "finite": bool(torch.isfinite(action).all().item()),
                "controller": row.tolist(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        raise SystemExit(self_test())
    Rival2Bot().run(wants_ball_predictions=False)
