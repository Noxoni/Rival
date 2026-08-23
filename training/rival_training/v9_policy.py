"""Schema-driven scratch actor and critic for RivalPolicyV1.

No Wisp/Nexto module, action table, or parameter enters this graph.  Logical
observation blocks are located from the generated ``RivalObsV1`` manifest so
the policy cannot silently drift to hand-maintained flat indices.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import time
from typing import Any

import torch
from torch import nn

from .v9_actions import RivalActionHeadV1, RivalHybridPolicy
from .v9_observations import observation_schema_manifest


POLICY_VERSION = "RivalPolicyV1"
CRITIC_VERSION = "RivalCriticV1"
POLICY_INITIALIZATION_VERSION = "RivalPolicyInitializationV1"


def _schema_layout() -> dict[str, Any]:
    manifest = observation_schema_manifest()
    blocks = manifest["block_slices"]
    core_names = (
        "match_control",
        "self_car",
        "opponent_car",
        "ball_goal",
        "motion_delta",
    )
    return {
        "schema_sha256": manifest["schema_sha256"],
        "observation_size": int(manifest["float_count"]),
        "core_slices": tuple(
            (int(blocks[name]["start"]), int(blocks[name]["end"]))
            for name in core_names
        ),
        "pad_slice": (
            int(blocks["boost_pads"]["start"]),
            int(blocks["boost_pads"]["end"]),
        ),
        "prediction_slice": (
            int(blocks["prediction"]["start"]),
            int(blocks["prediction"]["end"]),
        ),
        "history_slice": (
            int(blocks["controller_history"]["start"]),
            int(blocks["controller_history"]["end"]),
        ),
        "pad_shape": tuple(manifest["entity_shapes"]["boost_pads"]),
        "prediction_shape": tuple(manifest["entity_shapes"]["prediction"]),
        "self_history_shape": tuple(
            manifest["entity_shapes"]["self_controller_history"]
        ),
        "opponent_history_shape": tuple(
            manifest["entity_shapes"]["opponent_controller_history"]
        ),
        "prediction_horizon_ticks": tuple(
            manifest["shared_ball_prediction"]["horizon_ticks"]
        ),
    }


SCHEMA_LAYOUT = _schema_layout()


def _hidden_linear(linear: nn.Linear, *, gain: float = math.sqrt(2.0)) -> None:
    nn.init.orthogonal_(linear.weight, gain=gain)
    nn.init.zeros_(linear.bias)


class RivalStructuredEncoderV1(nn.Module):
    """Core/entity/prediction/history encoders plus the 512-wide fusion trunk."""

    output_width = 512

    def __init__(self) -> None:
        super().__init__()
        core_width = sum(end - start for start, end in SCHEMA_LAYOUT["core_slices"])
        pad_count, pad_features = SCHEMA_LAYOUT["pad_shape"]
        prediction_count, prediction_features = SCHEMA_LAYOUT["prediction_shape"]
        history_ticks, controller_features = SCHEMA_LAYOUT["self_history_shape"]
        if SCHEMA_LAYOUT["opponent_history_shape"] != (
            history_ticks,
            controller_features,
        ):
            raise ValueError("Self/opponent controller history shapes must match")
        self.core_width = core_width
        self.pad_count = pad_count
        self.pad_features = pad_features
        self.prediction_count = prediction_count
        self.prediction_features = prediction_features
        self.history_ticks = history_ticks
        self.controller_features = controller_features

        self.core_encoder = nn.Sequential(
            nn.LayerNorm(core_width),
            nn.Linear(core_width, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 384),
            nn.SiLU(),
        )

        self.pad_encoder = nn.Sequential(
            nn.Linear(pad_features, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.pad_query = nn.Linear(384, 64)
        self.pad_projection = nn.Sequential(
            nn.LayerNorm(64 * 3),
            nn.Linear(64 * 3, 128),
            nn.SiLU(),
        )

        # Two explicit normalized horizon coordinates are appended to each row.
        self.prediction_encoder = nn.Sequential(
            nn.Linear(prediction_features + 2, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.prediction_query = nn.Linear(384, 64)
        self.prediction_temporal_logits = nn.Parameter(torch.zeros(prediction_count))
        horizons = torch.as_tensor(
            SCHEMA_LAYOUT["prediction_horizon_ticks"], dtype=torch.float32
        )
        horizon_features = torch.stack(
            (
                horizons / horizons.max(),
                torch.log1p(horizons) / torch.log1p(horizons.max()),
            ),
            dim=-1,
        )
        self.register_buffer("prediction_horizon_features", horizon_features)

        self.history_conv1 = nn.Conv1d(
            controller_features * 2, 64, kernel_size=3, padding=1
        )
        self.history_conv2 = nn.Conv1d(64, 96, kernel_size=3, padding=1)
        self.history_projection = nn.Sequential(
            nn.LayerNorm(96 * 3),
            nn.Linear(96 * 3, 128),
            nn.SiLU(),
        )

        self.fusion = nn.Sequential(
            nn.LayerNorm(384 + 128 * 3),
            nn.Linear(384 + 128 * 3, 768),
            nn.SiLU(),
            nn.Linear(768, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
        )
        self._initialize_hidden_layers()

    def _initialize_hidden_layers(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                _hidden_linear(module)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)
        # Attention queries begin at a smaller scale than feature transforms.
        nn.init.normal_(self.pad_query.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.pad_query.bias)
        nn.init.normal_(self.prediction_query.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.prediction_query.bias)

    @staticmethod
    def _slice(values: torch.Tensor, bounds: tuple[int, int]) -> torch.Tensor:
        return values[..., bounds[0] : bounds[1]]

    def _core(self, observations: torch.Tensor) -> torch.Tensor:
        values = torch.cat(
            [self._slice(observations, bounds) for bounds in SCHEMA_LAYOUT["core_slices"]],
            dim=-1,
        )
        return self.core_encoder(values)

    def _pads(self, observations: torch.Tensor, core: torch.Tensor) -> torch.Tensor:
        raw = self._slice(observations, SCHEMA_LAYOUT["pad_slice"])
        entities = raw.reshape(-1, self.pad_count, self.pad_features)
        encoded = self.pad_encoder(entities)
        query = self.pad_query(core).unsqueeze(1)
        weights = torch.softmax((encoded * query).sum(dim=-1) / 8.0, dim=1)
        attended = (weights.unsqueeze(-1) * encoded).sum(dim=1)
        mean = encoded.mean(dim=1)
        maximum = encoded.max(dim=1).values
        return self.pad_projection(torch.cat((attended, mean, maximum), dim=-1))

    def _prediction(
        self, observations: torch.Tensor, core: torch.Tensor
    ) -> torch.Tensor:
        raw = self._slice(observations, SCHEMA_LAYOUT["prediction_slice"])
        rows = raw.reshape(-1, self.prediction_count, self.prediction_features)
        horizon = self.prediction_horizon_features.unsqueeze(0).expand(
            rows.shape[0], -1, -1
        )
        encoded = self.prediction_encoder(torch.cat((rows, horizon), dim=-1))
        query = self.prediction_query(core).unsqueeze(1)
        attention = torch.softmax((encoded * query).sum(dim=-1) / 8.0, dim=1)
        attended = (attention.unsqueeze(-1) * encoded).sum(dim=1)
        temporal = torch.softmax(self.prediction_temporal_logits, dim=0)
        chronological = (encoded * temporal.view(1, -1, 1)).sum(dim=1)
        return torch.cat((attended, chronological), dim=-1)

    def _history(self, observations: torch.Tensor) -> torch.Tensor:
        raw = self._slice(observations, SCHEMA_LAYOUT["history_slice"])
        one_history = self.history_ticks * self.controller_features
        self_rows = raw[..., :one_history].reshape(
            -1, self.history_ticks, self.controller_features
        )
        opponent_rows = raw[..., one_history:].reshape(
            -1, self.history_ticks, self.controller_features
        )
        sequence = torch.cat((self_rows, opponent_rows), dim=-1).transpose(1, 2)
        encoded = torch.nn.functional.silu(self.history_conv1(sequence))
        encoded = torch.nn.functional.silu(self.history_conv2(encoded))
        newest = encoded[..., -1]
        mean = encoded.mean(dim=-1)
        maximum = encoded.max(dim=-1).values
        return self.history_projection(torch.cat((newest, mean, maximum), dim=-1))

    def logical_embeddings(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        if observations.ndim != 2 or observations.shape[-1] != SCHEMA_LAYOUT[
            "observation_size"
        ]:
            raise ValueError(
                "RivalPolicyV1 expects batched RivalObsV1 shape "
                f"(N, {SCHEMA_LAYOUT['observation_size']}), got {tuple(observations.shape)}"
            )
        core = self._core(observations)
        return {
            "core": core,
            "pads": self._pads(observations, core),
            "prediction": self._prediction(observations, core),
            "history": self._history(observations),
        }

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        embeddings = self.logical_embeddings(observations)
        fused = torch.cat(
            (
                embeddings["core"],
                embeddings["pads"],
                embeddings["prediction"],
                embeddings["history"],
            ),
            dim=-1,
        )
        return self.fusion(fused)


class RivalPolicyV1(nn.Module):
    """Scratch hybrid actor: structured encoder plus RivalActionV1 heads."""

    def __init__(self, *, seed: int = 20260909) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.encoder = RivalStructuredEncoderV1()
            self.action_head = RivalActionHeadV1(self.encoder.output_width)

    def forward(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.action_head(self.encoder(observations))


class RivalCriticV1(nn.Module):
    """Independent same-observation critic with no actor parameter sharing."""

    def __init__(self, *, seed: int = 20260910) -> None:
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.encoder = RivalStructuredEncoderV1()
            self.value_head = nn.Linear(self.encoder.output_width, 1)
            nn.init.normal_(self.value_head.weight, mean=0.0, std=0.005)
            nn.init.zeros_(self.value_head.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.value_head(self.encoder(observations))


class InstrumentedRivalHybridPolicy(RivalHybridPolicy):
    """Central rollout policy with low-overhead batch/latency evidence."""

    def __init__(self, actor: RivalPolicyV1, device: str | torch.device) -> None:
        super().__init__(actor, device)
        self.inference_samples: list[dict[str, float | int]] = []

    @torch.no_grad()
    def get_action(
        self,
        observations,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(observations.shape[0])
        started = time.perf_counter()
        actions, log_probabilities = super().get_action(
            observations, deterministic=deterministic
        )
        elapsed = time.perf_counter() - started
        self.inference_samples.append(
            {
                "batch_size": batch_size,
                "wall_seconds": elapsed,
                "per_agent_microseconds": elapsed * 1e6 / max(batch_size, 1),
            }
        )
        return actions, log_probabilities

    def drain_inference_samples(self) -> list[dict[str, float | int]]:
        samples = self.inference_samples
        self.inference_samples = []
        return samples


def make_rival_policy(
    device: str | torch.device = "cuda:0",
    *,
    seed: int = 20260909,
) -> RivalHybridPolicy:
    return RivalHybridPolicy(RivalPolicyV1(seed=seed), device)


def make_instrumented_rival_policy(
    device: str | torch.device = "cuda:0",
    *,
    seed: int = 20260909,
) -> InstrumentedRivalHybridPolicy:
    return InstrumentedRivalHybridPolicy(RivalPolicyV1(seed=seed), device)


def _source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def trainable_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def policy_metadata() -> dict[str, Any]:
    actor = RivalPolicyV1()
    critic = RivalCriticV1()
    return {
        "policy_version": POLICY_VERSION,
        "critic_version": CRITIC_VERSION,
        "initialization_version": POLICY_INITIALIZATION_VERSION,
        "observation_schema_sha256": SCHEMA_LAYOUT["schema_sha256"],
        "observation_size": SCHEMA_LAYOUT["observation_size"],
        "actor_parameter_count": trainable_parameter_count(actor),
        "critic_parameter_count": trainable_parameter_count(critic),
        "actor_critic_parameters_shared": False,
        "recurrent": False,
        "batch_norm": False,
        "logical_widths": {
            "core": 384,
            "pads": 128,
            "prediction": 128,
            "history": 128,
            "fusion_input": 768,
            "fusion_output": 512,
        },
        "entity_shapes": {
            "pads": list(SCHEMA_LAYOUT["pad_shape"]),
            "prediction": list(SCHEMA_LAYOUT["prediction_shape"]),
            "self_history": list(SCHEMA_LAYOUT["self_history_shape"]),
            "opponent_history": list(SCHEMA_LAYOUT["opponent_history_shape"]),
        },
        "pad_pooling": "core-conditioned learned-query attention plus mean/max",
        "prediction_pooling": (
            "core-conditioned attention plus ordered learned temporal weights"
        ),
        "history_encoder": "two 1D temporal convolutions; newest/mean/max projection",
        "hidden_initialization": "orthogonal linear; Kaiming convolution; zero bias",
        "actor_head_initialization": (
            "near-zero analog/button weights; no-buttons logit bias 0.5; log_std -0.5"
        ),
        "source_sha256": _source_sha256(),
    }
