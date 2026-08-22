from __future__ import annotations

import torch

from rival_training.policy import StudentDiscretePolicy
from rival_training.teacher import (
    build_wisp_student,
    validate_student_against_reference,
    verify_teacher_hashes,
)


def test_teacher_hashes_and_direct_reconstruction_parity() -> None:
    assert verify_teacher_hashes()["all_match"]
    student = build_wisp_student()
    report = validate_student_against_reference(
        student, batch_size=128, seed=42, device="cuda"
    )
    assert report["allclose_atol_1e-6_rtol_1e-6"]
    assert report["max_abs_logit_error"] <= 1e-5
    assert report["teacher_student_top1_agreement"] == 1.0
    assert report["appended_action_selection_rate"] == 0.0


def test_student_policy_is_trainable_and_rlgym_ppo_compatible() -> None:
    policy = StudentDiscretePolicy(build_wisp_student(), "cuda")
    observations = torch.randn(8, 432, device="cuda")
    actions, log_probabilities = policy.get_action(observations)
    assert tuple(actions.shape) == (8,)
    assert tuple(log_probabilities.shape) == (8,)
    selected_log_probs, entropy = policy.get_backprop_data(
        observations, actions.to("cuda").view(-1, 1)
    )
    loss = -selected_log_probs.mean() - 0.001 * entropy
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )
    assert any(parameter.grad is not None for parameter in policy.parameters())
