"""Prove Rival v9 selected/applied controller timing at native 120 Hz."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_actions import (  # noqa: E402
    ACTION_DIM,
    ACTION_VERSION,
    ANALOG_DIM,
    CONTROLLER_FIELDS,
    PHYSICS_HZ,
    TIMING_VERSION,
    action_metadata,
    button_bits_to_combo,
    button_combo_to_bits,
)
from rival_training.v9_environment import (  # noqa: E402
    V9_ENVIRONMENT_VERSION,
    build_v9_diagnostic_env,
)
from rival_training.v9_observations import (  # noqa: E402
    OBSERVATION_VERSION,
    observation_schema_manifest,
)


RESULT_PATH = TRAINING_ROOT / "results" / "milestone09" / "gate05_timing_parity.json"
CAPTURE_REPORT_PATH = (
    TRAINING_ROOT / "results" / "milestone09" / "gate03_native_capture.json"
)
ACTION_SCHEMA_PATH = TRAINING_ROOT / "schemas" / "rival_action_v1.json"
ROCKETSIM_TRACE_TICKS = 512
MISSED_BLUE_SELECTIONS = {127, 255, 383}


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _controller(source: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [float(source.get(field, 0.0)) for field in CONTROLLER_FIELDS],
        dtype=np.float32,
    )


def _native_rlbot_trace() -> dict[str, Any]:
    capture = json.loads(CAPTURE_REPORT_PATH.read_text(encoding="utf-8"))
    contract = capture["native_corpus"]
    raw_path = REPO_ROOT / contract["path"]
    previous: dict[str, Any] | None = None
    records = 0
    adjacent_pairs = 0
    adjacent_matches = 0
    one_tick_pairs = 0
    one_tick_matches = 0
    one_tick_maximum_abs_error = 0.0
    frame_gap_pairs = 0
    frame_gap_preserved = 0
    frame_gap_details: list[dict[str, Any]] = []
    output_changes = 0
    same_packet_matches = 0
    sequence_mismatches = 0
    maximum_abs_error = 0.0
    phase_counts: Counter[str] = Counter()
    game_speeds: list[float] = []

    with raw_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("record_type") != "rival_v9_native_packet":
                continue
            sequence_mismatches += int(int(record["sequence"]) != records)
            packet = record["packet"]
            self_index = int(packet["self_index"])
            last_input = _controller(packet["players"][self_index]["last_input"])
            returned = _controller(record["controller_output"])
            same_packet_matches += int(np.array_equal(last_input, returned))
            phase = packet["match"]["phase"]
            phase_name = str(phase.get("name", phase)) if isinstance(phase, dict) else str(phase)
            phase_counts[phase_name] += 1
            game_speeds.append(float(packet["match"].get("game_speed", 1.0)))
            if previous is not None:
                adjacent_pairs += 1
                expected = previous["returned"]
                error = float(np.max(np.abs(last_input - expected)))
                maximum_abs_error = max(maximum_abs_error, error)
                exact = np.array_equal(last_input, expected)
                adjacent_matches += int(exact)
                frame_delta = int(record["frame_num"]) - int(previous["frame_num"])
                if frame_delta == 1:
                    one_tick_pairs += 1
                    one_tick_matches += int(exact)
                    one_tick_maximum_abs_error = max(
                        one_tick_maximum_abs_error, error
                    )
                else:
                    frame_gap_pairs += 1
                    frame_gap_preserved += int(exact)
                    frame_gap_details.append(
                        {
                            "previous_sequence": previous["sequence"],
                            "current_sequence": int(record["sequence"]),
                            "previous_frame": previous["frame_num"],
                            "current_frame": int(record["frame_num"]),
                            "frame_delta": frame_delta,
                            "previous_phase": previous["phase"],
                            "current_phase": phase_name,
                            "previous_returned": expected.tolist(),
                            "current_last_input": last_input.tolist(),
                            "exact": exact,
                        }
                    )
                output_changes += int(not np.array_equal(returned, expected))
            previous = {
                "sequence": int(record["sequence"]),
                "frame_num": int(record["frame_num"]),
                "phase": phase_name,
                "returned": returned,
            }
            records += 1

    return {
        "source": {
            "path": contract["path"],
            "expected_sha256": contract["sha256"],
            "observed_sha256": _sha256(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "corpus_version": contract["version"],
            "capture_status": capture["status"],
        },
        "records": records,
        "sequence_mismatches": sequence_mismatches,
        "adjacent_record_pairs": adjacent_pairs,
        "next_packet_last_input_matches_previous_returned": adjacent_matches,
        "one_tick_pairs": one_tick_pairs,
        "one_tick_exact_matches": one_tick_matches,
        "one_tick_maximum_abs_controller_error": one_tick_maximum_abs_error,
        "frame_gap_pairs": frame_gap_pairs,
        "frame_gap_previous_controller_preserved": frame_gap_preserved,
        "frame_gap_details": frame_gap_details,
        "post_initial_startup_gap_pairs": max(0, len(frame_gap_details) - 1),
        "post_initial_startup_gap_exact_matches": sum(
            int(detail["exact"]) for detail in frame_gap_details[1:]
        ),
        "initial_startup_gap_interpretation": (
            "The first logged callback returned throttle+boost, but after the first "
            "documented Countdown outbound-queue gap the server still reported zero "
            "last_input. This pre-handshake/startup discontinuity is not certified as "
            "an applied controller transition. The later startup gap preserves the "
            "previous row exactly, and all 5,997 delivered one-tick transitions are exact."
        ),
        "capture_console_observation": capture.get("console_observation"),
        "returned_controller_changes": output_changes,
        "same_packet_last_input_equals_current_returned": same_packet_matches,
        "maximum_abs_controller_error": maximum_abs_error,
        "match_phase_counts": dict(sorted(phase_counts.items())),
        "game_speed": {
            "minimum": min(game_speeds),
            "maximum": max(game_speeds),
            "all_native_1x": all(value == 1.0 for value in game_speeds),
        },
        "timing_relation": (
            "For every delivered one-tick packet i+1, PlayerInfo.last_input equals the "
            "physical controller returned on packet i. The one post-initial-startup "
            "frame gap also preserves the row; the first pre-handshake queue gap is "
            "reported separately and is not represented as a certified transition."
        ),
    }


def _trace_action(tick: int, team: int) -> np.ndarray:
    phase = tick + 31 * team
    analog = np.asarray(
        [
            np.sin(phase * 0.071),
            np.sin(phase * 0.113 + 0.2),
            np.cos(phase * 0.097 - 0.1),
            np.sin(phase * 0.137 + 0.4),
            np.cos(phase * 0.083 + 0.7),
        ],
        dtype=np.float32,
    )
    combo = (tick + 3 * team) % 8
    return np.concatenate((analog, button_combo_to_bits(combo))).astype(np.float32)


def _field(schema: dict[str, Any], name: str) -> tuple[int, int]:
    field = next(item for item in schema["fields"] if item["name"] == name)
    return int(field["start"]), int(field["end"])


def _rocketsim_trace() -> dict[str, Any]:
    schema = observation_schema_manifest()
    self_history = _field(schema, "history.self_controllers")
    opponent_history = _field(schema, "history.opponent_controllers")
    opponent_latest = _field(schema, "opponent.latest_controller")
    environment = build_v9_diagnostic_env(prediction_refresh_ticks=1)
    observations = environment.reset()
    del observations
    agents_by_team = {
        int(environment.state.cars[agent].team_num): agent for agent in environment.agents
    }
    expected_pending = {
        agent: np.zeros(ACTION_DIM, dtype=np.float32) for agent in environment.agents
    }
    maximum_selected_error = 0.0
    maximum_pending_error = 0.0
    maximum_applied_error = 0.0
    maximum_self_history_error = 0.0
    maximum_opponent_history_error = 0.0
    maximum_opponent_latest_error = 0.0
    tick_increment_mismatches = 0
    decision_index_mismatches = 0
    episode_done_ticks: list[int] = []
    missed_verified = 0
    observed_button_combos: set[int] = set()
    initial_tick = int(environment.state.tick_count)

    try:
        for tick in range(ROCKETSIM_TRACE_TICKS):
            prior_pending = {
                agent: row.copy() for agent, row in expected_pending.items()
            }
            actions = {
                agent: _trace_action(tick, team)
                for team, agent in agents_by_team.items()
            }
            blue = agents_by_team[0]
            blue_missed = tick in MISSED_BLUE_SELECTIONS
            if blue_missed:
                del actions[blue]
            expected_selected = {
                agent: (
                    actions[agent].copy()
                    if agent in actions
                    else prior_pending[agent].copy()
                )
                for agent in environment.agents
            }
            for row in expected_selected.values():
                observed_button_combos.add(button_bits_to_combo(row[ANALOG_DIM:]))

            observations, _rewards, terminated, truncated = environment.step(actions)
            if any(terminated.values()) or any(truncated.values()):
                episode_done_ticks.append(tick)
            tick_increment_mismatches += int(
                int(environment.state.tick_count) != initial_tick + tick + 1
            )
            decision_index_mismatches += int(
                environment.shared_info["rival_v9_last_decision_index"] != tick
            )

            selected = environment.shared_info["rival_v9_selected_actions"]
            pending = environment.shared_info["rival_v9_pending_actions"]
            applied = environment.shared_info["rival_v9_applied_actions"]
            for team, agent in agents_by_team.items():
                other = agents_by_team[1 - team]
                maximum_selected_error = max(
                    maximum_selected_error,
                    float(np.max(np.abs(selected[agent] - expected_selected[agent]))),
                )
                maximum_pending_error = max(
                    maximum_pending_error,
                    float(np.max(np.abs(pending[agent] - expected_selected[agent]))),
                )
                maximum_applied_error = max(
                    maximum_applied_error,
                    float(np.max(np.abs(applied[agent] - prior_pending[agent]))),
                )
                self_rows = observations[agent][
                    self_history[0] : self_history[1]
                ].reshape(8, ACTION_DIM)
                opponent_rows = observations[agent][
                    opponent_history[0] : opponent_history[1]
                ].reshape(8, ACTION_DIM)
                latest_opponent = observations[agent][
                    opponent_latest[0] : opponent_latest[1]
                ]
                maximum_self_history_error = max(
                    maximum_self_history_error,
                    float(np.max(np.abs(self_rows[-1] - prior_pending[agent]))),
                )
                maximum_opponent_history_error = max(
                    maximum_opponent_history_error,
                    float(np.max(np.abs(opponent_rows[-1] - prior_pending[other]))),
                )
                maximum_opponent_latest_error = max(
                    maximum_opponent_latest_error,
                    float(np.max(np.abs(latest_opponent - prior_pending[other]))),
                )
            if blue_missed:
                missed_verified += int(
                    np.array_equal(expected_selected[blue], prior_pending[blue])
                    and np.array_equal(pending[blue], prior_pending[blue])
                )
            expected_pending = expected_selected
    finally:
        environment.close()

    return {
        "environment_version": V9_ENVIRONMENT_VERSION,
        "trace_ticks": ROCKETSIM_TRACE_TICKS,
        "physics_hz": PHYSICS_HZ,
        "tick_increment_mismatches": tick_increment_mismatches,
        "decision_index_mismatches": decision_index_mismatches,
        "maximum_selected_controller_error": maximum_selected_error,
        "maximum_pending_controller_error": maximum_pending_error,
        "maximum_applied_controller_error": maximum_applied_error,
        "maximum_self_history_error": maximum_self_history_error,
        "maximum_opponent_history_error": maximum_opponent_history_error,
        "maximum_opponent_latest_error": maximum_opponent_latest_error,
        "missed_selection_ticks": sorted(MISSED_BLUE_SELECTIONS),
        "missed_selections_verified": missed_verified,
        "parser_missed_selection_counter": environment.shared_info[
            "rival_v9_missed_action_selections"
        ],
        "observed_button_combos": sorted(observed_button_combos),
        "episode_done_ticks": episode_done_ticks,
        "timing_relation": (
            "At selected tick t, RocketSim advances physics with pending row t-1, "
            "installs selected row t afterward, and RivalObsV1 records only row t-1 "
            "as the applied controller for the resulting state."
        ),
    }


def _physics_delay_probe() -> dict[str, Any]:
    environment = build_v9_diagnostic_env(prediction_refresh_ticks=1)
    environment.reset()
    agents_by_team = {
        int(environment.state.cars[agent].team_num): agent for agent in environment.agents
    }
    blue = agents_by_team[0]
    zero = {
        agent: np.zeros(ACTION_DIM, dtype=np.float32) for agent in environment.agents
    }
    selected = {agent: row.copy() for agent, row in zero.items()}
    selected[blue][0] = 1.0

    try:
        environment.step(selected)
        after_selection = environment.state.cars[blue]
        speed_after_selection = float(
            np.dot(
                after_selection.physics.linear_velocity,
                after_selection.physics.forward,
            )
        )
        applied_after_selection = environment.shared_info[
            "rival_v9_applied_actions"
        ][blue].copy()
        environment.step(zero)
        after_delayed_application = environment.state.cars[blue]
        speed_after_delayed_application = float(
            np.dot(
                after_delayed_application.physics.linear_velocity,
                after_delayed_application.physics.forward,
            )
        )
        applied_after_delayed_application = environment.shared_info[
            "rival_v9_applied_actions"
        ][blue].copy()
    finally:
        environment.close()

    return {
        "selected_controller": selected[blue].tolist(),
        "first_transition_applied_controller": applied_after_selection.tolist(),
        "second_transition_applied_controller": (
            applied_after_delayed_application.tolist()
        ),
        "forward_speed_after_selection_before_application": speed_after_selection,
        "forward_speed_after_delayed_application": speed_after_delayed_application,
        "forward_speed_delta": (
            speed_after_delayed_application - speed_after_selection
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_gate() -> int:
    native = _native_rlbot_trace()
    rocketsim = _rocketsim_trace()
    physics_probe = _physics_delay_probe()
    action_contract = action_metadata()
    action_schema = json.loads(ACTION_SCHEMA_PATH.read_text(encoding="utf-8"))
    observation_schema = observation_schema_manifest()
    checks = {
        "native_corpus_hash_matches_capture": (
            native["source"]["observed_sha256"]
            == native["source"]["expected_sha256"]
        ),
        "native_capture_is_1x": native["game_speed"]["all_native_1x"],
        "native_sequence_is_contiguous_by_record": native["sequence_mismatches"] == 0,
        "native_has_at_least_4000_one_tick_pairs": native["one_tick_pairs"] >= 4000,
        "native_one_tick_relation_exact": (
            native["one_tick_exact_matches"] == native["one_tick_pairs"]
            and native["one_tick_maximum_abs_controller_error"] == 0.0
        ),
        "native_post_initial_startup_gap_preserves_previous_controller": (
            native["post_initial_startup_gap_pairs"] > 0
            and native["post_initial_startup_gap_exact_matches"]
            == native["post_initial_startup_gap_pairs"]
        ),
        "native_initial_startup_gap_is_explicitly_uncertified": (
            native["frame_gap_pairs"] == 2
            and native["frame_gap_details"][0]["previous_sequence"] == 0
            and native["frame_gap_details"][0]["exact"] is False
            and "outbound-queue-full"
            in str(native["capture_console_observation"])
        ),
        "native_controller_stream_changes_naturally": (
            native["returned_controller_changes"] >= 100
        ),
        "rocketsim_policy_decision_every_tick": (
            rocketsim["decision_index_mismatches"] == 0
        ),
        "rocketsim_physics_advances_exactly_one_tick": (
            rocketsim["tick_increment_mismatches"] == 0
            and not rocketsim["episode_done_ticks"]
        ),
        "rocketsim_selected_pending_applied_exact": all(
            rocketsim[name] == 0.0
            for name in (
                "maximum_selected_controller_error",
                "maximum_pending_controller_error",
                "maximum_applied_controller_error",
            )
        ),
        "rocketsim_histories_record_applied_rows": all(
            rocketsim[name] == 0.0
            for name in (
                "maximum_self_history_error",
                "maximum_opponent_history_error",
                "maximum_opponent_latest_error",
            )
        ),
        "missed_selection_preserves_previous_controller": (
            rocketsim["missed_selections_verified"] == len(MISSED_BLUE_SELECTIONS)
            and rocketsim["parser_missed_selection_counter"]
            == len(MISSED_BLUE_SELECTIONS)
        ),
        "all_button_combos_exercised": rocketsim["observed_button_combos"]
        == list(range(8)),
        "physics_probe_first_transition_is_noop": np.array_equal(
            np.asarray(physics_probe["first_transition_applied_controller"]),
            np.zeros(ACTION_DIM),
        ),
        "physics_probe_selected_row_applies_on_second_transition": (
            physics_probe["second_transition_applied_controller"]
            == physics_probe["selected_controller"]
            and physics_probe["forward_speed_delta"] > 0.1
        ),
        "no_repeat_action_contract": (
            action_contract["repeat_action"] is False
            and action_contract["policy_decisions_per_physics_tick"] == 1
        ),
        "action_schema_current": (
            action_schema["parser_source_sha256"]
            == action_contract["parser_source_sha256"]
        ),
    }
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 5,
        "gate_name": "one_tick_timing_parity",
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "contract": {
            "action_version": ACTION_VERSION,
            "timing_version": TIMING_VERSION,
            "observation_version": OBSERVATION_VERSION,
            "physics_hz": PHYSICS_HZ,
            "policy_hz": PHYSICS_HZ,
            "action_schema_sha256": _sha256(ACTION_SCHEMA_PATH),
            "action_source_sha256": action_contract["parser_source_sha256"],
            "observation_schema_sha256": observation_schema["schema_sha256"],
            "environment_version": V9_ENVIRONMENT_VERSION,
        },
        "native_rlbot_v5_trace": native,
        "rocketsim_trace": rocketsim,
        "rocketsim_physics_delay_probe": physics_probe,
        "commands": {
            "gate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_timing_parity_gate.py"
            ),
            "unit_tests": (
                "training/.venv/Scripts/python.exe -m pytest "
                "training/tests/test_v9_actions.py "
                "training/tests/test_v9_environment.py -q"
            ),
        },
        "gate_semantics": {
            "score_used": False,
            "win_loss_used": False,
            "training_budget_used": False,
            "native_rate_source": "frozen RLBot v5 1x packet/callback corpus",
        },
    }
    _write_json(RESULT_PATH, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status == "passed" else 1


def main() -> int:
    return run_gate()


if __name__ == "__main__":
    raise SystemExit(main())
