from time import sleep

try:
    import torch
except ImportError:
    import sys
    print(__file__)
    sys.path.insert(0, "../../torch-archive")
    import torch

import math
from typing import Optional

from RocketSim import Angle

from analysis.tactical_metrics import build_state_snapshot, compute_tactical_metrics
from utils import clip
import config
import rlbot.flat
import rlbot.managers
from backend import rlbot_conversion
from backend.gamestate import common_values
from backend.gamestate.action import Action
from backend.model import ModelSet
from backend.rocketsim_adapter import RocketSimStateAdapter
from action_parser import XMirroredActionParser
from eta import rough_eta
from obs_builder import CustomObs
from policy.decision import PolicyDecision
from policy.inspector import PolicyInspector
from telemetry.decision_logger import DecisionTelemetryLogger


class RivalBot(rlbot.managers.Bot):
    def __init__(self):
        super().__init__(config.RLBOT_AGENT_ID)

        self.tick_count = 0
        """ RLBot-reported frame number (may be jittery) """

        self.ticks = -1
        """ Internal tick counter aligned to 120 Hz using seconds_elapsed (matches C++). """

        self.update_action_flag = True
        """ Whether to compute a new action on this frame. """

        self.old_action = Action()
        """ Action chosen last step, applied UNTIL the action delay """

        self.new_action = Action()
        """ Action chosen this step, applied AFTER the action delay """

        self.obs_builder = CustomObs()
        self.action_parser = XMirroredActionParser()

        self.models: ModelSet | None = None
        """ Model storage"""

        self.state_adapter: RocketSimStateAdapter | None = None
        self.prev_actions: list[Action] = []
        self.prev_time: float | None = None
        self.cached_state = None

        self.policy_tick = 0
        self.policy_inspector = PolicyInspector(config.POLICY_TOP_N)
        self.last_decision: PolicyDecision | None = None
        self.last_tactical_metrics = None
        self.telemetry = DecisionTelemetryLogger(
            config.TELEMETRY_PATH,
            enabled=config.TELEMETRY_ENABLED,
            include_logits=config.TELEMETRY_INCLUDE_LOGITS,
        )

    def initialize(self):
        self.logger.info(f"{self.name} index {self.index} team {self.team}: Initializing...")

        # Prevent thread-fighting with multiple instances
        torch.set_num_threads(1)

        # Load models
        shared_head_info = config.MODEL_INFO_SHARED_HEAD
        if shared_head_info is not None and not shared_head_info.path.exists():
            self.logger.warning(
                "Shared head model missing at %s; continuing without it",
                shared_head_info.path,
            )
            shared_head_info = None

        try:
            self.models = ModelSet(
                config.MODEL_INFO_POLICY,
                shared_head_info,
                device=config.MODEL_DEVICE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Failed to load model artifacts: {exc}. Ensure TorchScript exports are in '{config.MODEL_INFO_POLICY.path.parent}'."
            ) from exc

        self.state_adapter = RocketSimStateAdapter(config.ROCKETSIM_COLLISION_DIR)

        self.logger.info(
            "Rival policy inspection ready (top_n=%d, telemetry=%s, logits=%s)",
            config.POLICY_TOP_N,
            config.TELEMETRY_ENABLED,
            config.TELEMETRY_INCLUDE_LOGITS,
        )
        print("Rival initialization complete!")

    def update_action(
        self,
        state,
        player,
        score_diff: int,
        game_time: float | None,
        seconds_remaining: float | None,
    ):
        # Build obs (use torch tensor to avoid np->torch conversion overhead)
        device = config.MODEL_DEVICE
        obs = self.obs_builder.build_obs(player, state, self.ball_prediction, score_diff).get_tensor(device=device)

        # Build action mask
        action_mask = self.action_parser.get_action_mask(player, state)

        # Parse action
        if self.models is None:
            raise RuntimeError(
                "ModelSet has not been initialized before action inference"
            )

        inference = self.models.infer_policy(obs, action_mask)
        action_idx = inference.select_action(config.DETERMINISTIC)
        parsed_action = self.action_parser.get_action(action_idx, player, state)
        decision = self.policy_inspector.inspect(
            inference,
            action_idx,
            parsed_action,
            lambda index: self.action_parser.get_action(index, player, state),
            tick=self.policy_tick,
            game_time=game_time,
        )

        # Apply
        self.old_action = self.new_action
        self.new_action = parsed_action
        self.last_decision = decision

        opponents = [other for other in state.players if other.team != player.team]
        opponent = (
            min(opponents, key=lambda other: other.pos.dist_sq(state.ball.pos))
            if opponents
            else None
        )
        eta_self_ball = rough_eta(player, self.ball_prediction)
        eta_opponent_ball = (
            None if opponent is None else rough_eta(opponent, self.ball_prediction)
        )
        tactical_metrics = compute_tactical_metrics(
            state,
            player,
            opponent,
            parsed_action,
            score_diff=score_diff,
            seconds_remaining=seconds_remaining,
            eta_self_ball=eta_self_ball,
            eta_opponent_ball=eta_opponent_ball,
            eta_method="wisp_rough_eta_ball_prediction",
        )
        self.last_tactical_metrics = tactical_metrics

        if self.telemetry.enabled:
            state_snapshot = build_state_snapshot(
                state,
                player,
                opponent,
                score_diff=score_diff,
                game_time=game_time,
                seconds_remaining=seconds_remaining,
                boost_locations=common_values.BOOST_LOCATIONS,
            )
            try:
                self.telemetry.log(
                    decision,
                    tactical_metrics,
                    state_snapshot,
                    {
                        "deterministic": config.DETERMINISTIC,
                        "strategic_overrides_enabled": config.STRATEGIC_OVERRIDES_ENABLED,
                        "tick_skip": config.TICK_SKIP,
                        "action_delay": config.ACTION_DELAY,
                        "tick_window": self.ticks,
                        "update_action_flag": self.update_action_flag,
                    },
                )
            except (OSError, TypeError, ValueError) as exc:
                self.logger.warning("Disabling Rival telemetry after write failure: %s", exc)
                self.telemetry.enabled = False

        self.policy_tick += 1

    def get_output(self, packet: rlbot.flat.GamePacket) -> rlbot.flat.ControllerState:
        # Detect if the game phase is something we don't care about playing in
        is_active_game_phase = packet.match_info.match_phase in [
            rlbot.flat.MatchPhase.Countdown,
            rlbot.flat.MatchPhase.Kickoff,
            rlbot.flat.MatchPhase.Active,
        ]
        ball_exists = len(packet.balls) > 0
        if (not is_active_game_phase) or (not ball_exists):
            return self.celebrate(packet) or rlbot.flat.ControllerState()  # No output given

        player_count = len(packet.players)
        if player_count == 0:
            self.prev_actions = []
            return rlbot.flat.ControllerState()

        seconds_elapsed = (
            packet.match_info.seconds_elapsed if packet.match_info else None
        )
        if seconds_elapsed is not None:
            # Reset on timer rewind
            if self.prev_time is not None and seconds_elapsed + 1e-3 < self.prev_time:
                if self.state_adapter is not None:
                    self.state_adapter.reset()
                self.prev_actions = [Action() for _ in range(player_count)]
                self.old_action = Action()
                self.new_action = Action()
                self.ticks = -1
                self.update_action_flag = True
                self.policy_tick = 0
                self.last_decision = None

            # Compute tick delta like C++
            if self.prev_time is not None:
                delta_time = seconds_elapsed - self.prev_time
                ticks_elapsed = int(round(delta_time * 120.0))
                self.ticks = self.ticks + ticks_elapsed if self.ticks >= 0 else -1
            self.prev_time = seconds_elapsed

        if len(self.prev_actions) != player_count:
            self.prev_actions = [Action() for _ in range(player_count)]

        if self.state_adapter is None:
            raise RuntimeError("RocketSimStateAdapter not initialized")

        state = self.state_adapter.build(packet, self.prev_actions)
        self.cached_state = state

        if self.index >= len(state.players):
            return rlbot.flat.ControllerState()

        player_state = state.players[self.index]

        # Determine if we should update/apply action using the same logic as C++
        if self.update_action_flag:
            self.update_action_flag = False
            score_diff = packet.teams[self.team].score - packet.teams[1 - self.team].score
            seconds_remaining = getattr(packet.match_info, "game_time_remaining", None)
            self.update_action(
                state,
                player_state,
                score_diff,
                None if seconds_elapsed is None else float(seconds_elapsed),
                None if seconds_remaining is None else float(seconds_remaining),
            )

        # Apply action once we reach the action delay (one earlier due to RLBot delay)
        apply_new_action = (self.ticks >= (config.ACTION_DELAY - 1)) or (
            self.ticks == -1
        )

        action_for_rlbot = self.new_action if apply_new_action else self.old_action
        controller_state = rlbot_conversion.convert_action_to_controller_state(
            action_for_rlbot
        )

        if self.index < len(self.prev_actions):
            self.prev_actions[self.index] = Action(action_for_rlbot.get_np().copy())

        # After sending the action, manage tick window
        if self.ticks >= config.TICK_SKIP or self.ticks == -1:
            self.ticks = 0
            self.update_action_flag = True

        return controller_state

    def celebrate(self, packet: rlbot.flat.GamePacket) -> Optional[rlbot.flat.ControllerState]:
        phase = packet.match_info.match_phase

        if phase == rlbot.flat.MatchPhase.GoalScored:
            t = packet.match_info.seconds_elapsed
            phys = packet.players[self.index].physics
            steering = math.sin(7.2 * t + 1.3 * self.index)
            rot_mat = Angle(phys.rotation.yaw, phys.rotation.pitch, phys.rotation.roll).as_rot_mat()
            pitch = clip(3 * rot_mat.up.z + 3 * float(rot_mat.forward.z < 0), -1, 1)
            boost = phys.location.z < 1550 and rot_mat.forward.z > 0
            ctrls = rlbot.flat.ControllerState(1., steering, pitch, 0, 1., False, boost, False)
            return ctrls

        if phase == rlbot.flat.MatchPhase.Ended:
            t = packet.match_info.seconds_elapsed
            steering = math.sin(7.2 * t + 1.3 * self.index)
            boost = abs(steering) >= 1.5
            ctrls = rlbot.flat.ControllerState(-1., steering, 0, 0, 0, False, boost, False)
            return ctrls

        return None


if __name__ == "__main__":
    try:
        RivalBot().run()
    except Exception as e:
        print(e)
        while True:
            sleep(1)
