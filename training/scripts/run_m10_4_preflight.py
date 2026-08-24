"""Preflight the user-authorized Stage-1-only V3 reward restart."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import run_m10_2_preflight as base  # noqa: E402
from rival_training import v10_4_campaign as campaign  # noqa: E402
from rival_training.v10_3_curriculum import curriculum_reset_audit  # noqa: E402
from rival_training.v10_4_environment import (  # noqa: E402
    RivalSingleLearnerGymWrapperV3,
    build_ball_acquisition_env,
    make_ball_acquisition_phase_a_env,
)
from rival_training.v10_4_evaluation import evaluate_stage1_checkpoint  # noqa: E402
from rival_training.v10_4_reward import (  # noqa: E402
    BallAcquisitionTransitionV3,
    RivalBallAcquisitionRewardKernelV3,
    ball_acquisition_reward_metadata,
    reward_truth_table_v3,
)


USER_AUTHORITY = {
    "scope": "stage_1_only",
    "restart_from": "exact_m10_1_plus_10h_actor_weights_only",
    "reward": {
        "physical_new_touch": 1.0,
        "toward_ball_episode_cap": 0.75,
        "away_ball_episode_cap": -0.75,
        "idle_grace_seconds": 0.5,
        "first_stationary_tick_after_grace": -0.80,
        "idle_penalty_repeat_mode": "one_shot_capped_per_episode",
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
    base.RivalSingleLearnerGymWrapperV1 = RivalSingleLearnerGymWrapperV3
    base.build_ball_acquisition_env = build_ball_acquisition_env
    base.evaluate_stage1_checkpoint = evaluate_stage1_checkpoint
    base.BallAcquisitionTransitionV1 = BallAcquisitionTransitionV3
    base.RivalBallAcquisitionRewardKernelV1 = RivalBallAcquisitionRewardKernelV3
    base.ball_acquisition_reward_metadata = ball_acquisition_reward_metadata
    base._reward_truth_table = reward_truth_table_v3
    base.PHASE_A_ENV_FACTORY = make_ball_acquisition_phase_a_env
    base.DEFAULT_OUTPUT = campaign.RESULT_ROOT / "preflight.json"
    base.DEFAULT_DISPOSABLE_ROOT = (
        REPOSITORY_ROOT / "training/checkpoints/milestone10_4/preflight"
    )
    base.PREFLIGHT_VERSION = "RivalM10_4Stage1OnlyPreflightV1"
    base.DISPOSABLE_STATE_KEY = "v10_4_disposable_preflight"
    base.MILESTONE_LABEL = "Milestone 10.4 Stage 1 only"
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
