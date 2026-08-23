import hashlib
import json
from pathlib import Path
import sys
from time import sleep

try:
    import torch
except ImportError:
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
from strategy import (
    ChallengeCalibrationController,
    ChallengeCalibrationDecision,
    ChallengeCalibrationMode,
    ChallengeCalibrationParameters,
    ChallengeCommitmentParameters,
    ChallengeSample,
    NaturalAdjustmentController,
    NaturalAdjustmentDecision,
    NaturalAdjustmentMode,
    NaturalAdjustmentParameters,
    NaturalAdjustmentSample,
)
from telemetry.decision_logger import DecisionTelemetryLogger
from telemetry.packet_snapshot import extract_packet_snapshot


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
        self.action_parser = XMirroredActionParser(
            config.CANDIDATE_ACTION_TABLE_PATH,
            allow_all_actions=(
                config.CANDIDATE_POLICY_ENABLED
                and not config.CANDIDATE_LEGACY_ONLY
            ),
            legacy_only=config.CANDIDATE_LEGACY_ONLY,
        )

        self.models: ModelSet | None = None
        """ Model storage"""

        self.state_adapter: RocketSimStateAdapter | None = None
        self.prev_actions: list[Action] = []
        self.prev_time: float | None = None
        self.cached_state = None

        self.policy_tick = 0
        self.policy_inspector = PolicyInspector(config.POLICY_TOP_N)
        self.last_decision: PolicyDecision | None = None
        commitment_parameters = ChallengeCommitmentParameters(
            low_threshold=config.CHALLENGE_LOW_THRESHOLD,
            high_threshold=config.CHALLENGE_HIGH_THRESHOLD,
            pressure_distance=config.CHALLENGE_PRESSURE_DISTANCE,
            pressure_eta=config.CHALLENGE_PRESSURE_ETA,
            projected_miss_reference=config.CHALLENGE_PROJECTED_MISS_REFERENCE,
        )
        calibration_parameters = ChallengeCalibrationParameters(
            version=config.CHALLENGE_PARAMETER_VERSION,
            commitment=commitment_parameters,
            control_distance=config.CHALLENGE_CONTROL_DISTANCE,
            maximum_logit_gap=config.CHALLENGE_MAX_LOGIT_GAP,
            maximum_deferral_policy_ticks=config.CHALLENGE_MAX_DEFERRAL_TICKS,
        )
        self.challenge_calibration = ChallengeCalibrationController(
            config.CHALLENGE_CALIBRATION_MODE,
            calibration_parameters,
        )
        self.last_challenge_decision: ChallengeCalibrationDecision | None = None
        natural_parameters = NaturalAdjustmentParameters(
            version=config.NATURAL_PARAMETER_VERSION,
        )
        self.natural_adjustment = NaturalAdjustmentController(
            config.NATURAL_ADJUSTMENT_MODE,
            natural_parameters,
        )
        self.last_natural_decision: NaturalAdjustmentDecision | None = None
        self.last_tactical_metrics = None
        session_metadata = {
            **config.TELEMETRY_SESSION_METADATA,
            "policy_runtime": {
                "mode": config.POLICY_RUNTIME_MODE,
                "candidate_enabled": config.CANDIDATE_POLICY_ENABLED,
                "transfer_diagnostic_mode": config.TRANSFER_DIAGNOSTIC_MODE,
                "legacy_only": config.CANDIDATE_LEGACY_ONLY,
                "tick_skip": config.TICK_SKIP,
                "action_count": len(self.action_parser.actions),
                "observation_capture": config.DIAGNOSTIC_CAPTURE_OBSERVATIONS,
                "observation_capture_stride": config.DIAGNOSTIC_OBSERVATION_STRIDE,
            },
            "challenge_calibration": {
                "mode": self.challenge_calibration.mode.value,
                "treatment_enabled": (
                    self.challenge_calibration.mode
                    is ChallengeCalibrationMode.INTERVENE
                ),
                "parameters": calibration_parameters.to_record(),
            },
            "natural_adjustment": {
                "mode": self.natural_adjustment.mode.value,
                "treatment_enabled": (
                    self.natural_adjustment.mode
                    is NaturalAdjustmentMode.INTERVENE
                ),
                "parameters": natural_parameters.to_record(),
            },
        }
        self.telemetry = DecisionTelemetryLogger(
            config.TELEMETRY_PATH,
            enabled=config.TELEMETRY_ENABLED,
            include_logits=config.TELEMETRY_INCLUDE_LOGITS,
            session_metadata=session_metadata,
        )
        self._latest_packet_score: dict[str, int | None] | None = None

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
            "Rival policy inspection ready (mode=%s, actions=%d, tick_skip=%d, "
            "top_n=%d, telemetry=%s, logits=%s)",
            config.POLICY_RUNTIME_MODE,
            len(self.action_parser.actions),
            config.TICK_SKIP,
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
        packet: rlbot.flat.GamePacket | None = None,
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
        live_state_needed = bool(
            packet is not None
            and (
                self.challenge_calibration.mode is not ChallengeCalibrationMode.OFF
                or self.natural_adjustment.mode is not NaturalAdjustmentMode.OFF
            )
        )
        sample = (
            ChallengeSample.from_packet(
                packet,
                self.index,
                tactical_metrics,
            )
            if live_state_needed and packet is not None
            else None
        )
        calibration_decision = self.challenge_calibration.evaluate(
            decision,
            sample,
            lambda index: self.action_parser.get_action(index, player, state),
        )
        self.last_challenge_decision = calibration_decision
        if calibration_decision.final_action_index != decision.action_index:
            self.new_action = self.action_parser.get_action(
                calibration_decision.final_action_index,
                player,
                state,
            )
            tactical_metrics = compute_tactical_metrics(
                state,
                player,
                opponent,
                self.new_action,
                score_diff=score_diff,
                seconds_remaining=seconds_remaining,
                eta_self_ball=eta_self_ball,
                eta_opponent_ball=eta_opponent_ball,
                eta_method="wisp_rough_eta_ball_prediction",
            )
        natural_sample = NaturalAdjustmentSample.from_live(sample, tactical_metrics)
        natural_decision = self.natural_adjustment.evaluate(
            decision,
            natural_sample,
            lambda index: self.action_parser.get_action(index, player, state),
        )
        self.last_natural_decision = natural_decision
        if natural_decision.final_action_index != decision.action_index:
            self.new_action = self.action_parser.get_action(
                natural_decision.final_action_index,
                player,
                state,
            )
            tactical_metrics = compute_tactical_metrics(
                state,
                player,
                opponent,
                self.new_action,
                score_diff=score_diff,
                seconds_remaining=seconds_remaining,
                eta_self_ball=eta_self_ball,
                eta_opponent_ball=eta_opponent_ball,
                eta_method="wisp_rough_eta_ball_prediction",
            )
        self.last_tactical_metrics = tactical_metrics

        if self.telemetry.enabled:
            diagnostic = None
            if (
                config.DIAGNOSTIC_CAPTURE_OBSERVATIONS
                and self.policy_tick % config.DIAGNOSTIC_OBSERVATION_STRIDE == 0
            ):
                diagnostic = {
                    "capture_version": "RivalM07LiveObservationV1",
                    "live_observation_432": obs.detach().cpu().tolist(),
                }
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
                        "challenge_calibration_mode": self.challenge_calibration.mode.value,
                        "natural_adjustment_mode": self.natural_adjustment.mode.value,
                        "transfer_diagnostic_mode": config.TRANSFER_DIAGNOSTIC_MODE,
                        "candidate_legacy_only": config.CANDIDATE_LEGACY_ONLY,
                        "tick_skip": config.TICK_SKIP,
                        "action_delay": config.ACTION_DELAY,
                        "tick_window": self.ticks,
                        "update_action_flag": self.update_action_flag,
                    },
                    (
                        None
                        if packet is None
                        else extract_packet_snapshot(
                            packet, self.index, getattr(self, "field_info", None)
                        )
                    ),
                    calibration_decision,
                    natural_decision,
                    diagnostic,
                )
            except (OSError, TypeError, ValueError) as exc:
                self.logger.warning("Disabling Rival telemetry after write failure: %s", exc)
                self.telemetry.enabled = False

        self.policy_tick += 1

    def get_output(self, packet: rlbot.flat.GamePacket) -> rlbot.flat.ControllerState:
        if len(packet.teams) >= 2:
            self._latest_packet_score = {
                "blue": int(packet.teams[0].score),
                "orange": int(packet.teams[1].score),
            }
            self.telemetry.observe_final_score(
                self._latest_packet_score["blue"],
                self._latest_packet_score["orange"],
            )
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
                self.last_challenge_decision = None
                self.last_natural_decision = None
                self.challenge_calibration.reset("timer_rewind")
                self.natural_adjustment.reset("timer_rewind")

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
                packet,
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

    def retire(self):
        self.telemetry.finalize(
            termination_reason="rlbot_retired",
            final_score=self._latest_packet_score,
        )
        self.telemetry.close(finalize=False)
        super().retire()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_self_test() -> int:
    """Verify a source or frozen runtime without connecting to RLBot."""
    expected_hashes = {
        "POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
        "SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
    }

    # Initializing the adapter proves the packaged RocketSim module and collision
    # meshes are both loadable. ModelSet then exercises the real Wisp artifacts.
    RocketSimStateAdapter(config.ROCKETSIM_COLLISION_DIR)
    models = ModelSet(
        config.MODEL_INFO_POLICY,
        config.MODEL_INFO_SHARED_HEAD,
        device="cpu",
    )
    observation = torch.linspace(-1.0, 1.0, 432, dtype=torch.float32)
    legal_mask = torch.ones(90, dtype=torch.bool)
    inference = models.infer_policy(observation, legal_mask)
    selected = inference.select_action(deterministic=True)
    compatibility_selected = models.get_action(
        observation, legal_mask, deterministic=True
    )

    model_paths = {
        "POLICY.lt": config.MODEL_INFO_POLICY.path,
        "SHARED_HEAD.lt": config.MODEL_INFO_SHARED_HEAD.path,
    }
    model_hashes = {name: _sha256(path) for name, path in model_paths.items()}
    collision_mesh_count = len(
        list(config.ROCKETSIM_COLLISION_DIR.glob("soccar/*.cmf"))
    )
    passed = all(
        (
            tuple(observation.shape) == (432,),
            tuple(inference.raw_logits.shape) == (90,),
            bool(torch.isfinite(inference.raw_logits).all().item()),
            selected == compatibility_selected,
            model_hashes == expected_hashes,
            collision_mesh_count == 16,
        )
    )
    result = {
        "status": "pass" if passed else "fail",
        "frozen": bool(getattr(sys, "frozen", False)),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "model_base": str(config.MODEL_INFO_POLICY.path.parent),
        "collision_mesh_count": collision_mesh_count,
        "observation_shape": list(observation.shape),
        "policy_output_shape": list(inference.raw_logits.shape),
        "finite_logits": bool(torch.isfinite(inference.raw_logits).all().item()),
        "selected_action_index": selected,
        "compatibility_action_index": compatibility_selected,
        "models": model_hashes,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        raise SystemExit(run_self_test())

    try:
        RivalBot().run()
    except Exception:
        # A frozen release must exit with a useful traceback so RLBotGUI can show
        # the actual failure. Preserve Wisp's keep-open behavior for source runs.
        if getattr(sys, "frozen", False):
            raise
        import traceback

        traceback.print_exc()
        while True:
            sleep(1)
