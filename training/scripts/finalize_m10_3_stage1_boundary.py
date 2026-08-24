"""Evaluate and decide one Rival v10.3 Stage-1 V2 boundary."""

from __future__ import annotations

from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import finalize_m10_2_stage1_boundary as base  # noqa: E402
from rival_training import v10_3_campaign as campaign  # noqa: E402
from rival_training.v10_3_evaluation import (  # noqa: E402
    evaluate_stage1_checkpoint,
)


def main() -> int:
    base.CAMPAIGN_STATE_PATH = campaign.CAMPAIGN_STATE_PATH
    base.CORPUS_ROOT = campaign.CORPUS_ROOT
    base.RESULT_ROOT = campaign.RESULT_ROOT
    base.boundary_slug = campaign.boundary_slug
    base.update_progressive_state = campaign.update_progressive_state
    base.wall_clock_status = campaign.wall_clock_status
    base.evaluate_stage1_checkpoint = evaluate_stage1_checkpoint
    base.GATE_CORPUS_FILENAME = campaign.GATE_CORPUS_FILENAME
    base.UNSEEN_CORPUS_FILENAME = campaign.UNSEEN_CORPUS_FILENAME
    base.BOUNDARY_RESULT_VERSION = "RivalM10_3Stage1BoundaryResultV1"
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
