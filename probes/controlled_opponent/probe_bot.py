from __future__ import annotations

import os

import rlbot.flat as flat
import rlbot.managers


class ControlledProbeBot(rlbot.managers.Bot):
    def __init__(self) -> None:
        super().__init__("noxoni/rival/controlled-probe-v2")
        self.behavior = os.environ.get("RIVAL_PROBE_BEHAVIOR", "shadow")
        self.abort_time = float(os.environ.get("RIVAL_PROBE_ABORT_TIME", "0.65"))
        self._phase_start: float | None = None
        self._previous_y: float | None = None

    def get_output(self, packet: flat.GamePacket) -> flat.ControllerState:
        if self.index >= len(packet.players):
            return flat.ControllerState()
        now = float(packet.match_info.seconds_elapsed)
        y = float(packet.players[self.index].physics.location.y)
        if self._phase_start is None or (
            self._previous_y is not None and abs(y - self._previous_y) > 300.0
        ):
            self._phase_start = now
        self._previous_y = y
        elapsed = now - self._phase_start

        if self.behavior == "true_commit":
            return flat.ControllerState(throttle=1.0, boost=elapsed < 1.4)
        if self.behavior == "boost_then_brake":
            if elapsed < self.abort_time:
                return flat.ControllerState(throttle=1.0, boost=True)
            return flat.ControllerState(throttle=-1.0)
        if self.behavior == "boost_then_veer":
            if elapsed < self.abort_time:
                return flat.ControllerState(throttle=1.0, boost=True)
            return flat.ControllerState(throttle=0.7, steer=1.0, handbrake=True)
        if self.behavior == "jump_fake":
            return flat.ControllerState(
                throttle=0.75,
                jump=0.08 <= elapsed <= 0.20,
                pitch=-0.4 if elapsed > 0.10 else 0.0,
            )
        if self.behavior == "delayed_challenge":
            if elapsed < 1.0:
                return flat.ControllerState(throttle=0.0)
            return flat.ControllerState(throttle=1.0, boost=elapsed < 1.5)
        return flat.ControllerState(throttle=0.15)


if __name__ == "__main__":
    ControlledProbeBot().run()
