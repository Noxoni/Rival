"""Generate the compact Milestone 09 RivalActionV1 gate evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_actions import (  # noqa: E402
    ACTION_DIM,
    ANALOG_DIM,
    BUTTON_COMBO_COUNT,
    TANH_EPSILON,
    RivalActionHeadV1,
    RivalActionV1Parser,
    RivalHybridDistribution,
    RivalHybridPolicy,
    action_metadata,
    button_bits_to_combo,
    button_combo_to_bits,
)


SCHEMA_PATH = TRAINING_ROOT / "schemas" / "rival_action_v1.json"
RESULT_PATH = TRAINING_ROOT / "results" / "milestone09" / "gate01_action_contract.json"


class _GateActor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(13, 32), nn.SiLU())
        self.head = RivalActionHeadV1(32)

    def forward(self, observations: torch.Tensor):
        return self.head(self.encoder(observations))


def _sha256_bytes(values: bytes) -> str:
    return hashlib.sha256(values).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_report() -> tuple[dict, dict]:
    torch.manual_seed(20260901)
    np.random.seed(20260901)
    metadata = action_metadata()

    button_round_trips = [
        button_bits_to_combo(button_combo_to_bits(combo))
        for combo in range(BUTTON_COMBO_COUNT)
    ]

    parser = RivalActionV1Parser()
    shared: dict = {}
    parser.reset(["gate-agent"], None, shared)
    seeded_rows = np.random.uniform(-1.0, 1.0, size=(4096, ACTION_DIM)).astype(np.float32)
    seeded_rows[:, ANALOG_DIM:] = np.random.randint(
        0, 2, size=(len(seeded_rows), ACTION_DIM - ANALOG_DIM)
    )
    parser_max_error = 0.0
    for row in seeded_rows:
        output = parser.parse_actions({"gate-agent": row}, None, shared)["gate-agent"]
        parser_max_error = max(parser_max_error, float(np.max(np.abs(output[0] - row))))

    means = torch.randn(4096, ANALOG_DIM, dtype=torch.float64) * 0.4
    log_std = torch.linspace(-1.2, 0.3, ANALOG_DIM, dtype=torch.float64)
    logits = torch.randn(4096, BUTTON_COMBO_COUNT, dtype=torch.float64)
    distribution = RivalHybridDistribution(means, log_std, logits)
    analog = torch.tanh(torch.randn(4096, ANALOG_DIM, dtype=torch.float64) * 1.5)
    combos = torch.arange(4096) % BUTTON_COMBO_COUNT
    bits = torch.stack(
        ((combos & 1), ((combos >> 1) & 1), ((combos >> 2) & 1)), dim=-1
    ).to(torch.float64)
    physical = torch.cat((analog, bits), dim=-1)
    bounded = analog.clamp(-1 + TANH_EPSILON, 1 - TANH_EPSILON)
    pre_tanh = torch.atanh(bounded)
    reference = (
        Normal(means, log_std.exp()).log_prob(pre_tanh)
        - torch.log(torch.clamp(1 - bounded.square(), min=TANH_EPSILON))
    ).sum(-1) + Categorical(logits=logits).log_prob(combos)
    implementation = distribution.log_prob(physical)
    log_probability_max_error = float((implementation - reference).abs().max())

    actor = _GateActor()
    policy = RivalHybridPolicy(actor, "cpu")
    observations = torch.randn(1024, 13)
    sampled_actions, rollout_log_probs = policy.get_action(observations)
    backprop_log_probs, entropy = policy.get_backprop_data(observations, sampled_actions)
    reproduction_max_error = float(
        (rollout_log_probs - backprop_log_probs.detach()).abs().max()
    )
    advantages = torch.linspace(-1.0, 1.0, len(observations))
    objective = -(
        torch.exp(backprop_log_probs - rollout_log_probs) * advantages
    ).mean() - 0.001 * entropy
    objective.backward()
    mean_gradient_by_axis = actor.head.analog_mean.weight.grad.abs().sum(dim=1)
    log_std_gradient = actor.head.analog_log_std.grad.abs()
    button_gradient = actor.head.button_logits.weight.grad.abs().sum()

    random_buttons = sampled_actions[:, ANALOG_DIM:].numpy()
    sampled_combos = np.asarray(
        [button_bits_to_combo(row) for row in random_buttons], dtype=np.int64
    )
    analog_values = sampled_actions[:, :ANALOG_DIM].numpy()
    checks = {
        "one_policy_action_per_physics_tick": parser.repeats == 1,
        "physical_transport_shape_8": sampled_actions.shape[1] == ACTION_DIM,
        "analog_axes_bounded": bool(np.all(np.abs(analog_values) <= 1.0)),
        "analog_continuous_not_quantized": all(
            len(np.unique(analog_values[:, index])) > 900 for index in range(ANALOG_DIM)
        ),
        "all_button_combos_round_trip": button_round_trips == list(range(8)),
        "all_button_combos_sampled": set(sampled_combos.tolist()) == set(range(8)),
        "steer_yaw_independent": not np.array_equal(analog_values[:, 1], analog_values[:, 3]),
        "parser_byte_float_identical": parser_max_error == 0.0,
        "independent_log_probability_match": log_probability_max_error <= 1e-10,
        "rollout_backprop_log_probability_match": reproduction_max_error <= 1e-6,
        "finite_objective": bool(torch.isfinite(objective).item()),
        "all_analog_mean_axes_receive_gradient": bool(torch.all(mean_gradient_by_axis > 0)),
        "all_log_std_axes_receive_gradient": bool(torch.all(log_std_gradient > 0)),
        "button_head_receives_gradient": float(button_gradient) > 0,
        "finite_gradients": all(
            bool(torch.isfinite(parameter.grad).all().item())
            for parameter in actor.parameters()
            if parameter.grad is not None
        ),
        "state_dependent_action_mask_absent": metadata["state_dependent_action_mask"] is False,
        "lookup_table_absent": metadata["lookup_table"] is False,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 1,
        "gate_name": "action_contract_implementation",
        "status": "passed" if passed else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": 20260901,
        "checks": checks,
        "measurements": {
            "parser_rows": len(seeded_rows),
            "parser_max_abs_error": parser_max_error,
            "distribution_rows": len(physical),
            "independent_log_probability_max_abs_error": log_probability_max_error,
            "rollout_rows": len(sampled_actions),
            "rollout_backprop_log_probability_max_abs_error": reproduction_max_error,
            "analog_sample_min": analog_values.min(axis=0).tolist(),
            "analog_sample_max": analog_values.max(axis=0).tolist(),
            "analog_saturation_fraction_abs_gt_0_95": (
                np.abs(analog_values) > 0.95
            ).mean(axis=0).tolist(),
            "button_combo_counts": {
                str(combo): int(np.count_nonzero(sampled_combos == combo))
                for combo in range(BUTTON_COMBO_COUNT)
            },
            "analog_mean_gradient_l1_by_axis": mean_gradient_by_axis.tolist(),
            "analog_log_std_gradient_abs_by_axis": log_std_gradient.tolist(),
            "button_head_gradient_l1": float(button_gradient),
            "mixed_entropy": float(entropy.detach()),
            "branch_entropy": policy.last_entropy,
        },
        "contract": metadata,
        "commands": {
            "generate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_action_gate.py"
            ),
            "tests": (
                "training/.venv/Scripts/python.exe -m pytest "
                "training/tests/test_v9_actions.py -q"
            ),
        },
    }
    return metadata, report


def main() -> int:
    metadata, report = build_report()
    _write_json(SCHEMA_PATH, metadata)
    schema_bytes = SCHEMA_PATH.read_bytes()
    report["action_schema_artifact"] = {
        "path": str(SCHEMA_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        "size_bytes": len(schema_bytes),
        "sha256": _sha256_bytes(schema_bytes),
    }
    _write_json(RESULT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
