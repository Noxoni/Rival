"""Isolated process phase for the Milestone 09 Gate 11 coordinator."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "training"))

from rival_training.v9_checkpoint import (  # noqa: E402
    load_v9_checkpoint,
    save_v9_checkpoint,
)
from rival_training.v9_policy import RivalPolicyV1  # noqa: E402
from rival_training.v9_trainer import RivalV9PPOTrainer  # noqa: E402


def _deterministic_parameters(
    actor: RivalPolicyV1, observations: np.ndarray
) -> np.ndarray:
    actor = actor.to("cpu").eval()
    with torch.inference_mode():
        mean, log_std, button_logits = actor(
            torch.as_tensor(observations, dtype=torch.float32)
        )
        output = torch.cat((mean, log_std.expand_as(mean), button_logits), dim=-1)
    return np.ascontiguousarray(output.numpy(), dtype=np.float32)


def _compact(iteration: dict[str, Any]) -> dict[str, Any]:
    return {
        "iteration": iteration["iteration"],
        "phase": iteration["phase"],
        "agent_steps": iteration["cumulative_agent_steps"],
        "agent_steps_per_second": iteration["agent_steps_per_second"],
        "actor_update": iteration["ppo"]["actor_update_magnitude"],
        "critic_update": iteration["ppo"]["critic_update_magnitude"],
        "health": iteration["health"]["passed"],
    }


def run_phase(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config_snapshot.read_text(encoding="utf-8"))
    if args.phase == "pre":
        trainer = RivalV9PPOTrainer(config, device=args.device)
        restored_counters_exact: bool | None = None
        prior_reports: list[dict[str, Any]] = []
    else:
        if args.source_checkpoint is None:
            raise ValueError("Resume phase requires --source-checkpoint")
        loaded = load_v9_checkpoint(
            args.source_checkpoint, device=args.device, expected_config=config
        )
        trainer = RivalV9PPOTrainer(
            config,
            device=args.device,
            actor=loaded["actor"],
            critic=loaded["critic"],
            actor_optimizer=loaded["actor_optimizer"],
            critic_optimizer=loaded["critic_optimizer"],
            trainer_state=loaded["trainer_state"],
        )
        restored = trainer.trainer_state()
        expected = loaded["trainer_state"]
        restored_counters_exact = all(
            int(restored[key]) == int(expected[key])
            for key in (
                "completed_iterations",
                "cumulative_agent_steps",
                "cumulative_model_updates",
            )
        )
        if not restored_counters_exact:
            raise RuntimeError("Trainer counters changed during phase restore")
        prior_reports = list(expected.get("gate11_iteration_reports", []))

    shapes = trainer.start_workers()
    reports: list[dict[str, Any]] = []
    held_observations: np.ndarray | None = None
    try:
        for _ in range(int(args.iterations)):
            iteration, held_observations = trainer.run_iteration()
            iteration["phase"] = (
                "before_fresh_reload" if args.phase == "pre" else "after_fresh_reload"
            )
            reports.append(iteration)
            print(json.dumps(_compact(iteration)), flush=True)
        if held_observations is None:
            raise RuntimeError("Phase did not collect held observations")
        deterministic = _deterministic_parameters(trainer.actor, held_observations)
        trainer_state = trainer.trainer_state()
        trainer_state["gate11_iteration_reports"] = prior_reports + reports
        manifest = save_v9_checkpoint(
            args.checkpoint,
            actor=trainer.actor,
            critic=trainer.critic,
            actor_optimizer=trainer.actor_optimizer,
            critic_optimizer=trainer.critic_optimizer,
            trainer_state=trainer_state,
            config=config,
            reload_observations=held_observations,
        )
    finally:
        cleanup = trainer.cleanup()
    if not cleanup["passed"]:
        raise RuntimeError(f"Phase worker cleanup failed: {cleanup}")
    return {
        "schema_version": 1,
        "phase": args.phase,
        "status": "passed",
        "environment_shapes": shapes,
        "iterations": reports,
        "restored_counters_exact": restored_counters_exact,
        "checkpoint_manifest": manifest,
        "cleanup": cleanup,
        "expected_output_shape": list(deterministic.shape),
        "expected_output_sha256": hashlib.sha256(deterministic.tobytes()).hexdigest(),
        "expected_output_float32_base64": base64.b64encode(
            deterministic.tobytes()
        ).decode("ascii"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre", "resume"), required=True)
    parser.add_argument("--config-snapshot", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_phase(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"phase": args.phase, "status": report["status"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
