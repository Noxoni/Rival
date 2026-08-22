from __future__ import annotations

from types import SimpleNamespace

import rlbot.flat as flat

from tools.evidence.probes import (
    FakeChallengeParameters,
    default_resource_aerial_grid,
    fake_challenge_state,
    resource_aerial_state,
)
from tools.evidence.references import discover_reference, sha256_file
from tools.evidence.runner import (
    GameSpeedMonitor,
    build_match_configuration,
    describe_match_configuration,
)


def test_reference_discovery_validates_without_modifying_botpack() -> None:
    reference = discover_reference("nexto")
    before_config = sha256_file(reference.config_path)
    before_executable = sha256_file(reference.executable_path)

    description = describe_match_configuration(opponent="nexto", rival_team=0)

    assert description["game_speed_mutator"] == "Default"
    assert description["requested_game_speed"] == 1.0
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
    assert str(config.mutators.max_score).endswith("Unlimited")
    assert str(config.mutators.overtime).endswith("Unlimited")
    assert config.skip_replays is True
    assert config.auto_save_replay is False
    assert config.enable_rendering == flat.DebugRendering.AlwaysOff
    assert config.performance_monitor == flat.PerformanceMonitor.NeverShow
    assert config.auto_start_agents is True
    assert config.wait_for_agents is True
    assert config.instant_start is False
    assert config.existing_match_behavior == flat.ExistingMatchBehavior.Restart
    assert config.enable_state_setting is False
    assert config.freeplay is False
    rival_environment = {
        item.name: item.value for item in config.player_configurations[0].variety.environment
    }
    opponent_environment = config.player_configurations[1].variety.environment
    assert rival_environment["RIVAL_TELEMETRY_ENABLED"] == "1"
    assert not opponent_environment


def test_accelerated_natural_config_changes_only_state_setting_capability() -> None:
    reference = discover_reference("nexto")
    config = build_match_configuration(
        rival_team=0,
        opponent_config=reference.config_path,
        state_setting=True,
        instant_start=False,
    )

    assert config.enable_state_setting is True
    assert config.instant_start is False
    assert config.mutators.game_speed == flat.GameSpeedMutator.Default
    assert config.mutators.boost_amount == flat.BoostAmountMutator.NormalBoost
    assert config.mutators.boost_strength == flat.BoostStrengthMutator.One
    assert config.mutators.gravity == flat.GravityMutator.Default
    assert config.mutators.demolish == flat.DemolishMutator.Default


def test_controlled_probe_config_keeps_its_instant_state_setting() -> None:
    config = build_match_configuration(
        rival_team=0,
        opponent_config=discover_reference("nexto").config_path,
        state_setting=True,
        instant_start=True,
    )

    assert config.enable_state_setting is True
    assert config.instant_start is True
    assert config.auto_save_replay is False
    assert config.skip_replays is True


def test_speed_monitor_uses_only_desired_match_info_game_speed() -> None:
    calls: list[dict] = []
    manager = SimpleNamespace(
        set_game_state=lambda **kwargs: calls.append(kwargs)
    )
    packet = SimpleNamespace(
        match_info=SimpleNamespace(
            match_phase=flat.MatchPhase.Active,
            seconds_elapsed=10.0,
            game_speed=1.0,
        )
    )
    monitor = GameSpeedMonitor(5.0)

    monitor.observe(manager, packet, allow_state_setting=True)

    assert len(calls) == 1
    assert set(calls[0]) == {"match_info"}
    assert calls[0]["match_info"].game_speed == 5.0
    assert monitor.apply_count == 1


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
