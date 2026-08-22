from __future__ import annotations

from tools.evidence.probes import (
    FakeChallengeParameters,
    default_resource_aerial_grid,
    fake_challenge_state,
    resource_aerial_state,
)
from tools.evidence.references import discover_reference, sha256_file
from tools.evidence.runner import build_match_configuration, describe_match_configuration


def test_reference_discovery_validates_without_modifying_botpack() -> None:
    reference = discover_reference("nexto")
    before_config = sha256_file(reference.config_path)
    before_executable = sha256_file(reference.executable_path)

    description = describe_match_configuration(opponent="nexto", rival_team=0)

    assert description["game_speed"] == "Default"
    assert description["match_length"] == "FiveMinutes"
    assert description["installed_reference_mutation"] is False
    assert sha256_file(reference.config_path) == before_config
    assert sha256_file(reference.executable_path) == before_executable


def test_match_configuration_injects_only_rival_session_environment() -> None:
    reference = discover_reference("nexto")
    config = build_match_configuration(
        rival_team=1,
        opponent_config=reference.config_path,
        rival_environment={"RIVAL_TELEMETRY_ENABLED": "1"},
    )

    assert config.player_configurations[0].team == 1
    assert config.player_configurations[1].team == 0
    assert str(config.mutators.match_length).endswith("FiveMinutes")
    assert str(config.mutators.game_speed).endswith("Default")
    rival_environment = {
        item.name: item.value for item in config.player_configurations[0].variety.environment
    }
    opponent_environment = config.player_configurations[1].variety.environment
    assert rival_environment["RIVAL_TELEMETRY_ENABLED"] == "1"
    assert not opponent_environment


def test_controlled_probe_state_generation_is_parameterized_and_serializable() -> None:
    true_commit = FakeChallengeParameters("true_commit", 1)
    veer = FakeChallengeParameters("boost_then_veer", 2, lateral_offset=300.0)
    true_cars, true_ball = fake_challenge_state(true_commit, rival_team=0)
    veer_cars, _ = fake_challenge_state(veer, rival_team=0)

    assert true_commit.to_record()["behavior"] == "true_commit"
    assert true_ball[0].physics.location.z == 110.0
    assert true_cars[1].physics.location.x != veer_cars[1].physics.location.x

    grid = default_resource_aerial_grid()
    assert len(grid) >= 8
    assert len({case.rival_boost for case in grid}) >= 3
    assert len({case.ball_height for case in grid}) >= 3
    cars, balls = resource_aerial_state(grid[0], rival_team=1)
    assert cars[0].boost_amount == grid[0].rival_boost
    assert balls[0].physics.location.z == grid[0].ball_height
