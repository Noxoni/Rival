"""Preflight the Stage-1-only turn/approach/first-touch restart."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import run_m10_2_preflight as base  # noqa: E402
from rival_training import v10_5_campaign as campaign  # noqa: E402
from rival_training.v10_3_curriculum import curriculum_reset_audit  # noqa: E402
from rival_training.v10_5_environment import (  # noqa: E402
    RivalSingleLearnerGymWrapperV4,
    build_ball_acquisition_env,
    make_ball_acquisition_phase_a_env,
)
from rival_training.v10_5_evaluation import evaluate_stage1_checkpoint  # noqa: E402
from rival_training.v10_5_reward import (  # noqa: E402
    BallAcquisitionTransitionV4,
    RivalBallAcquisitionRewardKernelV4,
    ball_acquisition_reward_metadata,
    reward_truth_table_v4,
)


USER_AUTHORITY = {
    "scope": "stage_1_only",
    "restart_from": "exact_m10_1_plus_10h_actor_weights_only",
    "fresh_critic_and_actor_critic_optimizers": True,
    "purpose": ["turn_toward_ball", "move_toward_ball", "first_physical_touch"],
    "reward": {
        "first_physical_touch": 10.0,
        "later_touches": 0.0,
        "heading_delta_scale": 1.5,
        "heading_positive_cap": 3.0,
        "heading_negative_cap": -3.0,
        "approach_positive_cap": 3.0,
        "moving_away_cap": -3.0,
        "idle_grace_seconds": 0.5,
        "idle_speed_threshold_uu_per_second": 80.0,
        "idle_penalty_per_simulated_second": -1.4,
        "full_idle_penalty": -16.1,
        "maximum_positive_episode_reward": 16.0,
    },
    "stage_2_through_4_authorized": False,
    "production_promotion_authorized": False,
}


def _install_bindings() -> None:
    base.CAMPAIGN_STATE_PATH = campaign.CAMPAIGN_STATE_PATH
    base.CORPUS_ROOT = campaign.CORPUS_ROOT
    base.DEFAULT_STAGE1_CONFIG = campaign.DEFAULT_STAGE1_CONFIG
    base.RESULT_ROOT = campaign.RESULT_ROOT
    base.SOURCE_ACTOR_SHA256 = campaign.SOURCE_ACTOR_SHA256
    base.SOURCE_CHECKPOINT = campaign.SOURCE_CHECKPOINT
    base.SOURCE_MANIFEST_SHA256 = campaign.SOURCE_MANIFEST_SHA256
    base.actor_only_stage_transfer = campaign.actor_only_stage_transfer
    base.build_stage1_corpus_manifests = campaign.build_stage1_corpus_manifests
    base.config_identity = campaign.config_identity
    base.initialize_progressive_state = campaign.initialize_progressive_state
    base.load_stage1_config = campaign.load_stage1_config
    base.update_progressive_state = campaign.update_progressive_state
    base.curriculum_reset_audit = curriculum_reset_audit
    base.RivalSingleLearnerGymWrapperV1 = RivalSingleLearnerGymWrapperV4
    base.build_ball_acquisition_env = build_ball_acquisition_env
    base.evaluate_stage1_checkpoint = evaluate_stage1_checkpoint
    base.BallAcquisitionTransitionV1 = BallAcquisitionTransitionV4
    base.RivalBallAcquisitionRewardKernelV1 = RivalBallAcquisitionRewardKernelV4
    base.ball_acquisition_reward_metadata = ball_acquisition_reward_metadata
    base._reward_truth_table = reward_truth_table_v4
    base.PHASE_A_ENV_FACTORY = make_ball_acquisition_phase_a_env
    base.DEFAULT_OUTPUT = campaign.RESULT_ROOT / "preflight.json"
    base.DEFAULT_DISPOSABLE_ROOT = REPOSITORY_ROOT / "training/checkpoints/milestone10_5/preflight"
    base.PREFLIGHT_VERSION = "RivalM10_5Stage1OnlyPreflightV1"
    base.DISPOSABLE_STATE_KEY = "v10_5_disposable_preflight"
    base.MILESTONE_LABEL = "Milestone 10.5 Stage 1 only"
    base.GATE_CORPUS_FILENAME = campaign.GATE_CORPUS_FILENAME
    base.UNSEEN_CORPUS_FILENAME = campaign.UNSEEN_CORPUS_FILENAME


def main() -> int:
    _install_bindings()
    original = base.run_preflight

    def run_with_user_authority(args):
        report = original(args)
        report["user_authority"] = USER_AUTHORITY
        base.write_json_atomic(args.output, report)
        campaign.update_progressive_state({"user_authority": USER_AUTHORITY})
        return report

    base.run_preflight = run_with_user_authority
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
