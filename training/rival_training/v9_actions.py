"""RivalActionV1 native 120-Hz controller and hybrid PPO distribution.

This module is the complete scratch-policy action boundary.  Legacy Wisp lookup
tables remain in :mod:`rival_training.actions`; nothing in this module imports or
quantizes through them.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rlgym.api import ActionParser, AgentID
from rlgym.rocket_league.api import GameState
from torch import nn
from torch.distributions import Categorical, Normal


ACTION_VERSION = "RivalActionV1"
ACTION_SCHEMA_VERSION = 1
PHYSICS_HZ = 120
POLICY_HZ = 120
TIMING_VERSION = "RivalOneTickDelayV1"
CONTROLLER_FIELDS = (
    "throttle",
    "steer",
    "pitch",
    "yaw",
    "roll",
    "jump",
    "boost",
    "handbrake",
)
ANALOG_FIELDS = CONTROLLER_FIELDS[:5]
BUTTON_FIELDS = CONTROLLER_FIELDS[5:]
ANALOG_DIM = len(ANALOG_FIELDS)
BUTTON_COMBO_COUNT = 8
ACTION_DIM = len(CONTROLLER_FIELDS)
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.0
TANH_EPSILON = 1e-6


def button_combo_to_bits(combo: int) -> np.ndarray:
    """Decode ``jump + 2*boost + 4*handbrake`` into three float32 bits."""
    index = int(combo)
    if not 0 <= index < BUTTON_COMBO_COUNT:
        raise ValueError(f"Button combo must be in [0, 8), got {combo!r}")
    return np.asarray(
        [index & 1, (index >> 1) & 1, (index >> 2) & 1], dtype=np.float32
    )


def button_bits_to_combo(buttons: np.ndarray | list[float] | tuple[float, ...]) -> int:
    """Encode exact jump/boost/handbrake bits into the canonical combo index."""
    values = np.asarray(buttons, dtype=np.float32).reshape(-1)
    if values.shape != (3,):
        raise ValueError(f"Expected three button values, got shape {values.shape}")
    rounded = np.rint(values)
    if not np.array_equal(values, rounded) or np.any((rounded < 0) | (rounded > 1)):
        raise ValueError(f"Button values must be exact binary values, got {values.tolist()}")
    return int(rounded[0] + 2 * rounded[1] + 4 * rounded[2])


def _torch_button_bits(combo: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.stack(
        (
            combo.bitwise_and(1),
            combo.bitwise_right_shift(1).bitwise_and(1),
            combo.bitwise_right_shift(2).bitwise_and(1),
        ),
        dim=-1,
    ).to(dtype=dtype)


def _torch_button_combo(actions: torch.Tensor) -> torch.Tensor:
    buttons = actions[..., ANALOG_DIM:]
    rounded = buttons.round()
    if not bool(torch.allclose(buttons, rounded, atol=0.0, rtol=0.0)):
        raise ValueError("Stored RivalActionV1 button values must be exact 0/1 bits")
    if not bool(torch.all((rounded >= 0) & (rounded <= 1))):
        raise ValueError("Stored RivalActionV1 button values must be in {0, 1}")
    integer = rounded.to(dtype=torch.long)
    return integer[..., 0] + 2 * integer[..., 1] + 4 * integer[..., 2]


def validate_physical_actions(actions: np.ndarray) -> np.ndarray:
    """Validate one or more physical controller rows without changing values."""
    values = np.asarray(actions)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != ACTION_DIM:
        raise ValueError(f"Expected physical controller shape (N, 8), got {values.shape}")
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"Controller actions must be numeric, got {values.dtype}")
    if not np.isfinite(values).all():
        raise ValueError("Controller actions must all be finite")
    if np.any(values[:, :ANALOG_DIM] < -1.0) or np.any(values[:, :ANALOG_DIM] > 1.0):
        raise ValueError("Analog controller axes must be within [-1, 1]")
    buttons = values[:, ANALOG_DIM:]
    if not np.array_equal(buttons, np.rint(buttons)) or np.any((buttons < 0) | (buttons > 1)):
        raise ValueError("Controller button values must be exact 0/1 bits")
    return np.ascontiguousarray(values, dtype=np.float32)


class RivalActionV1Parser(
    ActionParser[AgentID, np.ndarray, np.ndarray, GameState, tuple[str, int]]
):
    """Map one native controller row to exactly one 1/120-second physics tick."""

    repeats = 1
    state_dependent_action_mask = False

    def get_action_space(self, agent: AgentID) -> tuple[str, int]:
        del agent
        return "continuous", ACTION_DIM

    def reset(
        self,
        agents: list[AgentID],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> None:
        del initial_state
        shared_info["previous_actions"] = {
            agent: np.zeros(ACTION_DIM, dtype=np.float32) for agent in agents
        }
        shared_info["cadence_ticks"] = 1
        shared_info["rival_action_version"] = ACTION_VERSION
        shared_info["rival_action_state_mask"] = False

    def parse_actions(
        self,
        actions: dict[AgentID, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, np.ndarray]:
        del state
        parsed: dict[AgentID, np.ndarray] = {}
        previous = shared_info.setdefault("previous_actions", {})
        for agent, raw_action in actions.items():
            controller = validate_physical_actions(np.asarray(raw_action))
            if controller.shape != (1, ACTION_DIM):
                raise ValueError(
                    f"RivalActionV1 requires one controller row per policy step; got "
                    f"{controller.shape} for {agent!r}"
                )
            previous[agent] = controller[0].copy()
            parsed[agent] = controller
        shared_info["rival_action_last_applied"] = {
            agent: rows[0].copy() for agent, rows in parsed.items()
        }
        return parsed


@dataclass(frozen=True)
class HybridEntropy:
    analog_monte_carlo: torch.Tensor
    button_exact: torch.Tensor

    @property
    def mixed(self) -> torch.Tensor:
        return self.analog_monte_carlo + self.button_exact


class RivalHybridDistribution:
    """Five tanh-Gaussian axes plus one joint eight-way button categorical."""

    def __init__(
        self,
        analog_mean: torch.Tensor,
        analog_log_std: torch.Tensor,
        button_logits: torch.Tensor,
    ) -> None:
        if analog_mean.shape[-1] != ANALOG_DIM:
            raise ValueError(f"Expected five analog means, got {analog_mean.shape}")
        if button_logits.shape[-1] != BUTTON_COMBO_COUNT:
            raise ValueError(f"Expected eight button logits, got {button_logits.shape}")
        bounded_log_std = torch.clamp(analog_log_std, LOG_STD_MIN, LOG_STD_MAX)
        if bounded_log_std.ndim == 1:
            bounded_log_std = bounded_log_std.expand_as(analog_mean)
        if bounded_log_std.shape != analog_mean.shape:
            raise ValueError(
                f"Analog log-std shape {bounded_log_std.shape} does not match means "
                f"{analog_mean.shape}"
            )
        self.analog_mean = analog_mean
        self.analog_log_std = bounded_log_std
        self.button_logits = button_logits
        self.normal = Normal(analog_mean, bounded_log_std.exp())
        self.categorical = Categorical(logits=button_logits)

    @staticmethod
    def _atanh(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = actions.clamp(-1.0 + TANH_EPSILON, 1.0 - TANH_EPSILON)
        pre_tanh = 0.5 * (torch.log1p(bounded) - torch.log1p(-bounded))
        return bounded, pre_tanh

    def analog_log_prob(self, analog_actions: torch.Tensor) -> torch.Tensor:
        bounded, pre_tanh = self._atanh(analog_actions)
        base = self.normal.log_prob(pre_tanh)
        log_jacobian = torch.log(torch.clamp(1.0 - bounded.square(), min=TANH_EPSILON))
        return (base - log_jacobian).sum(dim=-1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != ACTION_DIM:
            raise ValueError(f"Expected RivalActionV1 shape (..., 8), got {actions.shape}")
        analog = actions[..., :ANALOG_DIM]
        combo = _torch_button_combo(actions)
        return self.analog_log_prob(analog) + self.categorical.log_prob(combo)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        pre_tanh = self.normal.rsample()
        analog = torch.tanh(pre_tanh)
        combo = self.categorical.sample()
        actions = torch.cat(
            (analog, _torch_button_bits(combo, dtype=analog.dtype)), dim=-1
        )
        # Recompute from the stored physical row.  This is the value PPO must
        # reproduce after serialization, not a private pre-squash shortcut.
        return actions, self.log_prob(actions)

    def mode(self) -> torch.Tensor:
        analog = torch.tanh(self.analog_mean)
        combo = self.button_logits.argmax(dim=-1)
        return torch.cat(
            (analog, _torch_button_bits(combo, dtype=analog.dtype)), dim=-1
        )

    def entropy(self, stored_actions: torch.Tensor) -> HybridEntropy:
        analog_estimate = -self.analog_log_prob(stored_actions[..., :ANALOG_DIM]).mean()
        button_exact = self.categorical.entropy().mean()
        return HybridEntropy(analog_estimate, button_exact)


class RivalActionHeadV1(nn.Module):
    """Versioned hybrid controller head shared by training and deployment actors."""

    def __init__(
        self,
        embedding_size: int,
        *,
        initial_log_std: float = -0.5,
        no_buttons_bias: float = 0.5,
    ) -> None:
        super().__init__()
        self.analog_mean = nn.Linear(int(embedding_size), ANALOG_DIM)
        self.button_logits = nn.Linear(int(embedding_size), BUTTON_COMBO_COUNT)
        self.analog_log_std = nn.Parameter(
            torch.full((ANALOG_DIM,), float(initial_log_std), dtype=torch.float32)
        )
        nn.init.normal_(self.analog_mean.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.analog_mean.bias)
        nn.init.normal_(self.button_logits.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.button_logits.bias)
        with torch.no_grad():
            self.button_logits.bias[0] = float(no_buttons_bias)

    def forward(
        self, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = self.analog_mean(embedding)
        log_std = torch.clamp(self.analog_log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std, self.button_logits(embedding)


class RivalHybridPolicy(nn.Module):
    """rlgym-ppo-compatible wrapper for a scratch actor returning hybrid parameters."""

    def __init__(self, actor: nn.Module, device: str | torch.device) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.action_dim = ACTION_DIM
        self.last_entropy: dict[str, float] = {}

    def distribution(
        self, observations: np.ndarray | torch.Tensor
    ) -> RivalHybridDistribution:
        values = torch.as_tensor(
            observations, dtype=torch.float32, device=self.device
        )
        mean, log_std, button_logits = self.actor(values)
        return RivalHybridDistribution(mean, log_std, button_logits)

    @torch.no_grad()
    def get_action(
        self,
        observations: np.ndarray | torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        if deterministic:
            actions = distribution.mode()
            log_probabilities = distribution.log_prob(actions)
        else:
            actions, log_probabilities = distribution.sample()
        return actions.detach().cpu(), log_probabilities.detach().cpu()

    def get_backprop_data(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(observations)
        physical_actions = actions.to(self.device, dtype=torch.float32)
        log_probabilities = distribution.log_prob(physical_actions)
        entropy = distribution.entropy(physical_actions)
        self.last_entropy = {
            "analog_monte_carlo": float(entropy.analog_monte_carlo.detach().cpu()),
            "button_exact": float(entropy.button_exact.detach().cpu()),
            "mixed": float(entropy.mixed.detach().cpu()),
        }
        return log_probabilities, entropy.mixed


def _source_sha256() -> str:
    # Git normalizes tracked Python to LF while Windows working trees may use
    # CRLF.  Hash the canonical LF source so checkpoint metadata survives a
    # clean checkout without pretending line-ending conversion changed code.
    canonical = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest()


def action_metadata() -> dict[str, Any]:
    source_hash = _source_sha256()
    return {
        "schema_version": ACTION_SCHEMA_VERSION,
        "version": ACTION_VERSION,
        "controller_field_order": list(CONTROLLER_FIELDS),
        "physics_hz": PHYSICS_HZ,
        "policy_hz": POLICY_HZ,
        "policy_decisions_per_physics_tick": 1,
        "repeat_action": False,
        "transport_shape": [ACTION_DIM],
        "analog": {
            "fields": list(ANALOG_FIELDS),
            "distribution": "tanh_squashed_diagonal_gaussian",
            "bounds": [-1.0, 1.0],
            "log_std_bounds": [LOG_STD_MIN, LOG_STD_MAX],
            "tanh_jacobian_log_probability": True,
        },
        "buttons": {
            "fields": list(BUTTON_FIELDS),
            "distribution": "joint_categorical",
            "combo_count": BUTTON_COMBO_COUNT,
            "encoding": "jump + 2*boost + 4*handbrake",
        },
        "state_dependent_action_mask": False,
        "timing_delay_version": TIMING_VERSION,
        "stored_action": "physical_controller_float32",
        "lookup_table": False,
        "hidden_controller_synthesis": False,
        "parser_source_sha256": source_hash,
        "policy_action_head_source_sha256": source_hash,
    }
