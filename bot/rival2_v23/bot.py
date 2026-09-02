"""RLBot v5 entry point for the frozen Codex Autonomous V23 Rival policy bundle."""

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


AGENT_ID = "noxoni/rival2/codex-autonomous-v23"
BUNDLE = ROOT / "deployment_bundle.json"


def _load_bundle() -> dict:
    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    if bundle.get("format") != "RIVAL2_RLBOT_SIDE_SPECIALIZED_BUNDLE_V1":
        raise RuntimeError("unsupported Rival V23 deployment bundle")
    if bundle.get("selector") != "physical_team_side_before_match":
        raise RuntimeError("Rival V23 requires fixed physical-team-side selection")
    return bundle


def _side_paths(team: int) -> tuple[Path, Path, dict]:
    if team not in (0, 1):
        raise RuntimeError(f"invalid Rival team index: {team}")
    bundle = _load_bundle()
    side = "blue" if team == 0 else "orange"
    selected = bundle["sides"][side]
    return ROOT / selected["model"], ROOT / selected["manifest"], selected


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


class Rival2V23Bot(rlbot.managers.Bot):
    def __init__(self):
        super().__init__(AGENT_ID)
        self.runtime: Rival2LiveRuntime | None = None
        self.selected_side: str | None = None

    def initialize(self):
        torch.set_num_threads(1)
        model, manifest, selected = _side_paths(int(self.team))
        self.selected_side = "blue" if int(self.team) == 0 else "orange"
        self.runtime = Rival2LiveRuntime(model, manifest, self.field_info)
        source = self.runtime.manifest["source"]
        if source["checkpoint_sha256"] != selected["checkpoint_sha256"]:
            raise RuntimeError("Rival V23 source checkpoint identity mismatch")
        self.logger.info(
            "Rival V23 ready: side=%s policy=%s checkpoint=%s",
            self.selected_side,
            source["policy_version"],
            source["checkpoint_sha256"],
        )

    def get_output(self, packet: rlbot.flat.GamePacket) -> rlbot.flat.ControllerState:
        if self.runtime is None:
            return rlbot.flat.ControllerState()
        return _controller(self.runtime.step(packet, self.team))

    def retire(self):
        if self.runtime is not None:
            summary = self.runtime.summary()
            summary["selected_side"] = self.selected_side
            self.logger.info("Rival V23 runtime summary: %s", summary)


def self_test() -> int:
    results = {}
    for team in (0, 1):
        side = "blue" if team == 0 else "orange"
        model_path, manifest_path, selected = _side_paths(team)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source"]["checkpoint_sha256"] != selected["checkpoint_sha256"]:
            raise RuntimeError(f"{side} checkpoint identity mismatch")
        if manifest["contracts"]["hold_ticks"] != 1:
            raise RuntimeError(f"{side} policy is not deployed at 120 Hz")
        model = torch.jit.load(str(model_path), map_location="cpu").eval()
        observation = torch.zeros(
            (1, manifest["observation"]["dimension"]), dtype=torch.float32
        )
        with torch.inference_mode():
            action = model(observation)
        row = action[0].numpy()
        Rival2LiveRuntime._validate_action(row)
        results[side] = {
            "source": manifest["source"],
            "artifact": manifest["artifact"],
            "output_shape": list(action.shape),
            "finite": bool(torch.isfinite(action).all().item()),
            "controller": row.tolist(),
        }
    print(json.dumps({"status": "pass", "sides": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        raise SystemExit(self_test())
    Rival2V23Bot().run(wants_ball_predictions=False)
