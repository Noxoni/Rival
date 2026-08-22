"""Bounded direct reconstruction of the frozen Wisp TorchScript artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import resolve_repo_path


EXPECTED_TEACHER_HASHES = {
    "bot/models/POLICY.lt": "1bd600a15f43106645de84b42379fe9ae404ecfb509dc21a2e309480ea17ebf7",
    "bot/models/SHARED_HEAD.lt": "3f7b6b363a72d7ceaba3cdb58bc13e1ae95e07b041b5e94a326c7045bebd7e42",
}
TEACHER_INPUT_SIZE = 432
TEACHER_ACTION_COUNT = 90
EXPANDED_ACTION_COUNT = 158


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_teacher_hashes() -> dict[str, Any]:
    files = {}
    all_match = True
    for relative_path, expected in EXPECTED_TEACHER_HASHES.items():
        path = resolve_repo_path(relative_path)
        actual = sha256_file(path)
        matches = actual == expected
        all_match &= matches
        files[relative_path] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
            "matches": matches,
            "size_bytes": path.stat().st_size,
        }
    if not all_match:
        raise RuntimeError(f"Frozen Wisp teacher hash mismatch: {files}")
    return {"all_match": all_match, "files": files}


def reconstruct_sequential_from_artifact(path: str | Path) -> nn.Sequential:
    """Rebuild transparent Linear/LayerNorm/ReLU children as trainable modules."""
    module = torch.jit.load(str(path), map_location="cpu")
    result = nn.Sequential()
    for child in list(module.modules())[1:]:
        named_parameters = list(child.named_parameters())
        if len(named_parameters) == 2:
            names = [item[0] for item in named_parameters]
            if names != ["weight", "bias"]:
                raise RuntimeError(f"Unexpected parameter names in {path}: {names}")
            weight, bias = (item[1] for item in named_parameters)
            if weight.ndim == 2 and bias.ndim == 1:
                layer: nn.Module = nn.Linear(weight.shape[1], weight.shape[0])
            elif weight.ndim == 1 and bias.ndim == 1:
                layer = nn.LayerNorm(weight.shape[0])
            else:
                raise RuntimeError(
                    f"Unsupported Wisp parameter shapes {weight.shape}/{bias.shape}"
                )
            layer.load_state_dict(child.state_dict())
            result.append(layer)
        elif len(named_parameters) == 0:
            result.append(nn.ReLU())
        else:
            raise RuntimeError(
                f"Unsupported Wisp child with {len(named_parameters)} parameters"
            )
    result.eval()
    return result


class FrozenWispReference(nn.Module):
    """Faithful ordinary-PyTorch view of the two frozen artifact graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.shared = reconstruct_sequential_from_artifact(
            resolve_repo_path("bot/models/SHARED_HEAD.lt")
        )
        self.policy = reconstruct_sequential_from_artifact(
            resolve_repo_path("bot/models/POLICY.lt")
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy(self.shared(observations))


class WispStudentActor(nn.Module):
    """Explicit trainable Wisp architecture with an expanded output head."""

    def __init__(self, action_count: int = EXPANDED_ACTION_COUNT) -> None:
        super().__init__()
        self.action_count = action_count
        self.shared = nn.Sequential(
            nn.Linear(432, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.policy(self.shared(observations))


def _copy_module(source: nn.Module, destination: nn.Module) -> None:
    destination.load_state_dict(source.state_dict())


def build_wisp_student(
    action_count: int = EXPANDED_ACTION_COUNT,
    *,
    appended_bias: float = -12.0,
) -> WispStudentActor:
    """Copy every teacher parameter and conservatively initialize appended rows."""
    if action_count < TEACHER_ACTION_COUNT:
        raise ValueError("Student action count cannot remove Wisp actions")
    verify_teacher_hashes()
    reference = FrozenWispReference()
    student = WispStudentActor(action_count)

    if len(reference.shared) != len(student.shared):
        raise RuntimeError("Shared-head reconstruction does not match explicit architecture")
    for source, destination in zip(reference.shared, student.shared):
        _copy_module(source, destination)

    if len(reference.policy) != len(student.policy):
        raise RuntimeError("Policy reconstruction does not match explicit architecture")
    for source, destination in zip(reference.policy[:-1], student.policy[:-1]):
        _copy_module(source, destination)

    teacher_output = reference.policy[-1]
    student_output = student.policy[-1]
    if not isinstance(teacher_output, nn.Linear) or not isinstance(student_output, nn.Linear):
        raise RuntimeError("Wisp output head was not a Linear layer")
    with torch.no_grad():
        student_output.weight.zero_()
        student_output.bias.fill_(appended_bias)
        student_output.weight[:TEACHER_ACTION_COUNT].copy_(teacher_output.weight)
        student_output.bias[:TEACHER_ACTION_COUNT].copy_(teacher_output.bias)
    return student


def validate_student_against_reference(
    student: WispStudentActor,
    *,
    batch_size: int = 4096,
    seed: int = 20260822,
    device: str = "cuda",
) -> dict[str, Any]:
    """Numerically gate original-logit parity on a large randomized batch."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    observations = torch.randn(batch_size, TEACHER_INPUT_SIZE, generator=generator)
    reference = FrozenWispReference().to(device).eval()
    student = student.to(device).eval()
    observations = observations.to(device)
    with torch.no_grad():
        expected = reference(observations)
        all_student_logits = student(observations)
        actual = all_student_logits[:, :TEACHER_ACTION_COUNT]
        difference = (expected - actual).abs()
        reference_top1 = expected.argmax(dim=-1)
        student_top1 = all_student_logits.argmax(dim=-1)
        appended_rate = float(
            (student_top1 >= TEACHER_ACTION_COUNT).float().mean().item()
        )
    report = {
        "batch_size": batch_size,
        "seed": seed,
        "device": device,
        "reference_shape": list(expected.shape),
        "student_prefix_shape": list(actual.shape),
        "max_abs_logit_error": float(difference.max().item()),
        "mean_abs_logit_error": float(difference.mean().item()),
        "allclose_atol_1e-6_rtol_1e-6": bool(
            torch.allclose(expected, actual, atol=1e-6, rtol=1e-6)
        ),
        "teacher_student_top1_agreement": float(
            (reference_top1 == student_top1).float().mean().item()
        ),
        "appended_action_selection_rate": appended_rate,
    }
    if not report["allclose_atol_1e-6_rtol_1e-6"]:
        raise RuntimeError(f"Student failed Wisp numerical parity gate: {report}")
    return report


def architecture_metadata(action_count: int = EXPANDED_ACTION_COUNT) -> dict[str, Any]:
    actor = WispStudentActor(action_count)
    return {
        "schema_version": 1,
        "bootstrap_path": "A_direct_trainable_reconstruction",
        "teacher_input_size": TEACHER_INPUT_SIZE,
        "teacher_action_count": TEACHER_ACTION_COUNT,
        "student_action_count": action_count,
        "shared_layers": [432, 1024, 1024],
        "policy_layers": [1024, 1024, 512, 512, 128, action_count],
        "hidden_normalization": "LayerNorm after each hidden Linear",
        "activation": "ReLU",
        "parameter_count": sum(parameter.numel() for parameter in actor.parameters()),
        "appended_initialization": "zero weights and bias -12.0",
    }
