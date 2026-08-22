from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import rlbot.flat as flat


FAKE_CHALLENGE_BEHAVIORS = (
    "true_commit",
    "boost_then_brake",
    "boost_then_veer",
    "jump_fake",
    "delayed_challenge",
)


@dataclass(frozen=True)
class FakeChallengeParameters:
    behavior: str
    repetition: int
    rival_boost: float = 45.0
    opponent_boost: float = 55.0
    separation: float = 1750.0
    lateral_offset: float = 100.0
    challenger_speed: float = 650.0
    abort_time: float = 0.65
    window_seconds: float = 4.0

    def __post_init__(self) -> None:
        if self.behavior not in FAKE_CHALLENGE_BEHAVIORS:
            raise ValueError(f"Unknown fake-challenge behavior: {self.behavior}")

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceAerialParameters:
    case_id: str
    rival_boost: float
    ball_height: float
    ball_distance: float
    ball_speed: float
    opponent_pressure: str
    field_y: float
    ground_alternative: bool
    window_seconds: float = 4.0

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def default_resource_aerial_grid() -> list[ResourceAerialParameters]:
    return [
        ResourceAerialParameters("low-near-lowpressure", 8, 500, 900, 100, "low", -1800, True),
        ResourceAerialParameters("low-far-highpressure", 8, 900, 1700, 250, "high", -1000, True),
        ResourceAerialParameters("low-high-mediumpressure", 12, 1250, 1400, 0, "medium", -500, False),
        ResourceAerialParameters("mid-near-highpressure", 28, 650, 850, 150, "high", -1500, True),
        ResourceAerialParameters("mid-far-lowpressure", 28, 1000, 1800, 200, "low", -500, False),
        ResourceAerialParameters("mid-high-mediumpressure", 35, 1450, 1500, -100, "medium", 0, False),
        ResourceAerialParameters("high-near-highpressure", 70, 700, 900, 200, "high", -1200, True),
        ResourceAerialParameters("high-far-lowpressure", 70, 1200, 1900, 250, "low", 300, False),
    ]


def _physics(
    x: float,
    y: float,
    z: float,
    *,
    yaw: float,
    vx: float = 0.0,
    vy: float = 0.0,
    vz: float = 0.0,
) -> flat.DesiredPhysics:
    return flat.DesiredPhysics(
        location=flat.Vector3Partial(x, y, z),
        rotation=flat.RotatorPartial(0.0, yaw, 0.0),
        velocity=flat.Vector3Partial(vx, vy, vz),
        angular_velocity=flat.Vector3Partial(0.0, 0.0, 0.0),
    )


def fake_challenge_state(
    params: FakeChallengeParameters,
    rival_team: int,
) -> tuple[dict[int, flat.DesiredCarState], dict[int, flat.DesiredBallState]]:
    direction = 1.0 if rival_team == 0 else -1.0
    rival_y = -1500.0 * direction
    rival_yaw = math.pi / 2 if direction > 0 else -math.pi / 2
    ball_y = rival_y + 380.0 * direction
    opponent_y = ball_y + params.separation * direction
    opponent_yaw = -rival_yaw
    cars = {
        0: flat.DesiredCarState(
            _physics(0.0, rival_y, 17.0, yaw=rival_yaw, vy=430.0 * direction),
            params.rival_boost,
        ),
        1: flat.DesiredCarState(
            _physics(
                params.lateral_offset,
                opponent_y,
                17.0,
                yaw=opponent_yaw,
                vy=-params.challenger_speed * direction,
            ),
            params.opponent_boost,
        ),
    }
    balls = {
        0: flat.DesiredBallState(
            _physics(0.0, ball_y, 110.0, yaw=0.0, vy=400.0 * direction)
        )
    }
    return cars, balls


def resource_aerial_state(
    params: ResourceAerialParameters,
    rival_team: int,
) -> tuple[dict[int, flat.DesiredCarState], dict[int, flat.DesiredBallState]]:
    direction = 1.0 if rival_team == 0 else -1.0
    rival_y = params.field_y * direction
    rival_yaw = math.pi / 2 if direction > 0 else -math.pi / 2
    ball_y = rival_y + params.ball_distance * direction
    pressure_distance = {"high": 700.0, "medium": 1300.0, "low": 2200.0}[
        params.opponent_pressure
    ]
    opponent_y = ball_y + pressure_distance * direction
    cars = {
        0: flat.DesiredCarState(
            _physics(0.0, rival_y, 17.0, yaw=rival_yaw, vy=500.0 * direction),
            params.rival_boost,
        ),
        1: flat.DesiredCarState(
            _physics(
                150.0,
                opponent_y,
                17.0,
                yaw=-rival_yaw,
                vy=-650.0 * direction,
            ),
            50.0,
        ),
    }
    balls = {
        0: flat.DesiredBallState(
            _physics(
                0.0,
                ball_y,
                params.ball_height,
                yaw=0.0,
                vy=params.ball_speed * direction,
            )
        )
    }
    return cars, balls
