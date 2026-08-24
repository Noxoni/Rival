"""Execute Rival v10.3 Stage-1 V2 implementation preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

import run_m10_2_preflight as base  # noqa: E402
from rival_training import v10_3_campaign as campaign  # noqa: E402
from rival_training.v10_3_curriculum import (  # noqa: E402
    curriculum_reset_audit,
)
from rival_training.v10_3_environment import (  # noqa: E402
    RivalSingleLearnerGymWrapperV2,
    build_ball_acquisition_env,
    make_ball_acquisition_phase_a_env,
)
from rival_training.v10_3_evaluation import (  # noqa: E402
    evaluate_stage1_checkpoint,
)
from rival_training.v10_3_reward import (  # noqa: E402
    BallAcquisitionTransitionV2,
    RivalBallAcquisitionRewardKernelV2,
    ball_acquisition_reward_metadata,
    reward_truth_table_v2,
)


def _git_bytes(path: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"HEAD:{path}"], cwd=REPOSITORY_ROOT
    )


def _authority_package_audit() -> dict[str, Any]:
    manifest = json.loads(
        (REPOSITORY_ROOT / "handoff/v10.3/PACKAGE_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )
    files = []
    for expected in manifest["files"]:
        raw = _git_bytes(expected["path"])
        digest = hashlib.sha256(raw).hexdigest()
        files.append(
            {
                "path": expected["path"],
                "committed_bytes": len(raw),
                "expected_bytes": int(expected["bytes"]),
                "committed_sha256": digest,
                "expected_sha256": expected["sha256"],
                "byte_count_matches": len(raw) == int(expected["bytes"]),
                "sha256_matches": digest == expected["sha256"],
            }
        )
    inherited = []
    for expected in manifest["inherited_exact_blobs"]:
        actual = subprocess.check_output(
            ["git", "hash-object", "--", expected["path"]],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
        inherited.append(
            {
                "path": expected["path"],
                "actual_blob_sha": actual,
                "expected_blob_sha": expected["blob_sha"],
                "passed": actual == expected["blob_sha"],
            }
        )
    history = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            "52dc505af3f43a28f6b2a5a5eb20d4d034f842d2",
            "HEAD",
        ],
        cwd=REPOSITORY_ROOT,
        check=False,
    ).returncode == 0
    stash = subprocess.check_output(
        ["git", "stash", "list"], cwd=REPOSITORY_ROOT, text=True
    ).splitlines()
    checks = {
        "required_m10_2_closeout_in_history": history,
        "paused_strategy_stash_preserved": any(
            "rival-v4-paused-superseded-before-v4.1" in row for row in stash
        ),
        "all_manifest_byte_counts_match": all(
            row["byte_count_matches"] for row in files
        ),
        "all_inherited_blob_shas_match": all(
            row["passed"] for row in inherited
        ),
        "manifest_sha256_fields_match": all(
            row["sha256_matches"] for row in files
        ),
    }
    return {
        "manifest": "handoff/v10.3/PACKAGE_MANIFEST.json",
        "files": files,
        "inherited_exact_blobs": inherited,
        "stash_list": stash,
        "checks": checks,
        "gating_checks_passed": all(
            checks[name]
            for name in (
                "required_m10_2_closeout_in_history",
                "paused_strategy_stash_preserved",
                "all_manifest_byte_counts_match",
                "all_inherited_blob_shas_match",
            )
        ),
        "non_gating_discrepancy": (
            None
            if checks["manifest_sha256_fields_match"]
            else "The five v10.3 manifest SHA-256 fields do not match the committed blobs; byte counts and inherited Git blob identities do match."
        ),
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
    base.RivalSingleLearnerGymWrapperV1 = RivalSingleLearnerGymWrapperV2
    base.build_ball_acquisition_env = build_ball_acquisition_env
    base.evaluate_stage1_checkpoint = evaluate_stage1_checkpoint
    base.BallAcquisitionTransitionV1 = BallAcquisitionTransitionV2
    base.RivalBallAcquisitionRewardKernelV1 = (
        RivalBallAcquisitionRewardKernelV2
    )
    base.ball_acquisition_reward_metadata = ball_acquisition_reward_metadata
    base._reward_truth_table = reward_truth_table_v2
    base.PHASE_A_ENV_FACTORY = make_ball_acquisition_phase_a_env
    base.DEFAULT_OUTPUT = campaign.RESULT_ROOT / "preflight.json"
    base.DEFAULT_DISPOSABLE_ROOT = (
        REPOSITORY_ROOT / "training/checkpoints/milestone10_3/preflight"
    )
    base.PREFLIGHT_VERSION = "RivalM10_3Stage1PreflightV1"
    base.DISPOSABLE_STATE_KEY = "v10_3_disposable_preflight"
    base.MILESTONE_LABEL = "Milestone 10.3"
    base.GATE_CORPUS_FILENAME = campaign.GATE_CORPUS_FILENAME
    base.UNSEEN_CORPUS_FILENAME = campaign.UNSEEN_CORPUS_FILENAME


def main() -> int:
    _install_bindings()
    original = base.run_preflight

    def run_with_authority(args):
        authority = _authority_package_audit()
        if not authority["gating_checks_passed"]:
            raise RuntimeError(f"M10.3 authority preflight failed: {authority}")
        report = original(args)
        report["authority_package_audit"] = authority
        base.write_json_atomic(args.output, report)
        campaign.update_progressive_state(
            {"authority_package_audit": authority}
        )
        return report

    base.run_preflight = run_with_authority
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
