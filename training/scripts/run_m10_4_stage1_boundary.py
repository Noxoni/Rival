"""Run one recoverable Stage-1-only V3 training boundary."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import run_m10_2_stage1_boundary as base  # noqa: E402
from rival_training import v10_4_campaign as campaign  # noqa: E402
from rival_training.v10_4_environment import (  # noqa: E402
    BALL_ACQUISITION_ENV_FACTORY_BY_PHASE,
)


def main() -> int:
    base.CAMPAIGN_STATE_PATH = campaign.CAMPAIGN_STATE_PATH
    base.DEFAULT_STAGE1_CONFIG = campaign.DEFAULT_STAGE1_CONFIG
    base.RESULT_ROOT = campaign.RESULT_ROOT
    base.SOURCE_CHECKPOINT = campaign.SOURCE_CHECKPOINT
    base.actor_only_stage_transfer = campaign.actor_only_stage_transfer
    base.boundary_ppo_batch_agent_steps = campaign.boundary_ppo_batch_agent_steps
    base.boundary_slug = campaign.boundary_slug
    base.load_stage1_config = campaign.load_stage1_config
    base.nominal_stage1_steps = campaign.nominal_stage1_steps
    base.start_real_campaign_clock = campaign.start_real_campaign_clock
    base.update_progressive_state = campaign.update_progressive_state
    base.wall_clock_status = campaign.wall_clock_status
    base.BALL_ACQUISITION_ENV_FACTORY_BY_PHASE = BALL_ACQUISITION_ENV_FACTORY_BY_PHASE
    base.DEFAULT_CHECKPOINT_ROOT = (
        REPOSITORY_ROOT / "training/checkpoints/milestone10_4/stage_1"
    )
    base.SOURCE_STATE_KEY = "v10_4_source_checkpoint"
    base.ACTIVE_BOUNDARY_STATE_KEY = "v10_4_active_boundary_hours"
    base.COMPLETED_BOUNDARY_STATE_KEY = "v10_4_completed_boundary_hours"
    base.TRAINING_BOUNDARY_VERSION = "RivalM10_4Stage1TrainingBoundaryV1"
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
