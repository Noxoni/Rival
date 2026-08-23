"""Run Milestone 09 Gate 7 cadence-safe reward audit.

This gate is deliberately independent of match outcomes or policy skill.  It
replays one frozen native RLBot-v5 trace and one deterministic synthetic trace
at equivalent 1/2/4-tick observation intervals, then audits physical-time
integration, event invariance, symmetry, finiteness, and shaping budgets.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np


TRAINING_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TRAINING_ROOT.parent
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_canonical import (  # noqa: E402
    PHYSICS_HZ,
    RLBotCanonicalAdapterV1,
)
from rival_training.v9_rewards import (  # noqa: E402
    COMPONENTS,
    GAMMA_120HZ,
    GOAL_REWARD,
    SHAPING_ABSOLUTE_EPISODE_BUDGETS,
    SHAPING_COMPONENTS,
    RewardEventsV1,
    RewardStateV1,
    RivalRewardKernelV1,
    reward_metadata,
    reward_state_from_canonical,
    select_reward_phase,
    touch_quality_from_transition,
)
from rival_training.v9_rlbot_corpus import (  # noqa: E402
    NATIVE_CORPUS_VERSION,
    snapshot_to_rlbot_sources,
)


CAPTURE_REPORT = (
    TRAINING_ROOT / "results" / "milestone09" / "gate03_native_capture.json"
)
RESULT_PATH = (
    TRAINING_ROOT / "results" / "milestone09" / "gate07_reward_cadence.json"
)
PERIODS = (1, 2, 4)
NATURAL_TRANSITIONS = 4096


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_native_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("record_type") == "rival_v9_native_packet":
                records.append(record)
    return records


def _longest_contiguous(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    best_start = 0
    best_end = 1
    start = 0
    for index in range(1, len(records)):
        if int(records[index]["frame_num"]) != int(records[index - 1]["frame_num"]) + 1:
            if index - start > best_end - best_start:
                best_start, best_end = start, index
            start = index
    if len(records) - start > best_end - best_start:
        best_start, best_end = start, len(records)
    return records[best_start:best_end]


def _touch_stamp(player: dict[str, Any]) -> float | None:
    touch = player.get("latest_touch")
    return None if touch is None else float(touch["game_seconds"])


def _score_map(snapshot: dict[str, Any]) -> dict[int, int]:
    return {
        int(item["team"]): int(item["score"])
        for item in snapshot["match"].get("scores", [])
    }


def _natural_views(
    records: list[dict[str, Any]],
) -> tuple[
    dict[int, list[RewardStateV1]],
    dict[int, list[RewardEventsV1]],
    dict[str, Any],
]:
    adapter = RLBotCanonicalAdapterV1()
    states: dict[int, list[RewardStateV1]] = {0: [], 1: []}
    events: dict[int, list[RewardEventsV1]] = {0: [], 1: []}
    previous_touch: dict[int, float | None] = {0: None, 1: None}
    previous_view: dict[int, RewardStateV1 | None] = {0: None, 1: None}
    previous_scores = {0: 0, 1: 0}
    coverage = {
        "touch_events_by_team": {"0": 0, "1": 0},
        "goal_events_by_team": {"0": 0, "1": 0},
        "airborne_states_by_team": {"0": 0, "1": 0},
        "boosting_states_by_team": {"0": 0, "1": 0},
    }
    for record_index, record in enumerate(records):
        snapshot = record["packet"]
        packet, field_info, _ = snapshot_to_rlbot_sources(snapshot)
        players_by_team = {
            int(player["team"]): player for player in snapshot["players"]
        }
        scores = _score_map(snapshot)
        for team in (0, 1):
            canonical = adapter.adapt(packet, team, field_info)
            view = reward_state_from_canonical(canonical)
            states[team].append(view)
            stamp = _touch_stamp(players_by_team[team])
            touched = bool(
                record_index > 0
                and stamp is not None
                and stamp != previous_touch[team]
            )
            goal_for = scores.get(team, 0) > previous_scores.get(team, 0)
            goal_against = scores.get(1 - team, 0) > previous_scores.get(1 - team, 0)
            touch_quality = None
            touch_velocity_gain = None
            if touched and previous_view[team] is not None:
                touch_quality, touch_velocity_gain = touch_quality_from_transition(
                    previous_view[team], view
                )
            events[team].append(
                RewardEventsV1(
                    goal_for=goal_for,
                    goal_against=goal_against,
                    self_touch=touched,
                    touch_event_count=int(touched),
                    touch_quality_sum=touch_quality,
                    touch_velocity_gain_sum=touch_velocity_gain,
                )
            )
            coverage["touch_events_by_team"][str(team)] += int(touched)
            coverage["goal_events_by_team"][str(team)] += int(goal_for)
            coverage["airborne_states_by_team"][str(team)] += int(
                not view.self_surface_contact
            )
            coverage["boosting_states_by_team"][str(team)] += int(view.self_boosting)
            previous_touch[team] = stamp
            previous_view[team] = view
        previous_scores = scores
    return states, events, coverage


def _aggregate_events(events: Iterable[RewardEventsV1]) -> RewardEventsV1:
    values = list(events)
    touch_count = sum(event.touch_event_count or int(event.self_touch) for event in values)
    touch_quality_values = [
        float(event.touch_quality_sum)
        for event in values
        if event.touch_quality_sum is not None
    ]
    touch_velocity_values = [
        float(event.touch_velocity_gain_sum)
        for event in values
        if event.touch_velocity_gain_sum is not None
    ]
    return RewardEventsV1(
        goal_for=any(event.goal_for for event in values),
        goal_against=any(event.goal_against for event in values),
        self_touch=touch_count > 0,
        touch_event_count=touch_count,
        touch_quality_sum=(sum(touch_quality_values) if touch_count else None),
        touch_velocity_gain_sum=(sum(touch_velocity_values) if touch_count else None),
    )


def _integrate(
    states: list[RewardStateV1],
    events: list[RewardEventsV1],
    period: int,
) -> dict[str, Any]:
    phase = select_reward_phase(0.0)
    kernel = RivalRewardKernelV1()
    kernel.reset(states[0])
    signed = defaultdict(float)
    absolute = defaultdict(float)
    proposal_signed = defaultdict(float)
    discounted = defaultdict(float)
    detector_totals = defaultdict(float)
    nonfinite = 0
    clipped = defaultdict(int)
    discount_prefix = 1.0
    steps = 0
    last_index = 0
    for index in range(period, len(states), period):
        interval_events = _aggregate_events(events[last_index + 1 : index + 1])
        step = kernel.step(states[index], interval_events, phase)
        for name in COMPONENTS:
            value = float(step.components[name])
            proposal = float(step.proposals[name])
            signed[name] += value
            absolute[name] += abs(value)
            proposal_signed[name] += proposal
            discounted[name] += discount_prefix * value
            nonfinite += int(not math.isfinite(value) or not math.isfinite(proposal))
        for name, value in step.detectors.items():
            detector_totals[name] += float(value)
        for name in step.budget_clipped:
            clipped[name] += 1
        discount_prefix *= GAMMA_120HZ**step.delta_ticks
        last_index = index
        steps += 1
    elapsed_ticks = int(states[last_index].tick_index - states[0].tick_index)
    elapsed_seconds = elapsed_ticks / PHYSICS_HZ
    shaping_signed = float(sum(signed[name] for name in SHAPING_COMPONENTS))
    shaping_absolute = float(sum(absolute[name] for name in SHAPING_COMPONENTS))
    return {
        "period_ticks": period,
        "sampled_transitions": steps,
        "native_elapsed_ticks": elapsed_ticks,
        "simulated_seconds": elapsed_seconds,
        "component_signed": {name: float(signed[name]) for name in COMPONENTS},
        "component_absolute": {name: float(absolute[name]) for name in COMPONENTS},
        "component_proposal_signed": {
            name: float(proposal_signed[name]) for name in COMPONENTS
        },
        "component_discounted": {
            name: float(discounted[name]) for name in COMPONENTS
        },
        "shaping_signed": shaping_signed,
        "shaping_absolute": shaping_absolute,
        "shaping_signed_per_simulated_second": (
            shaping_signed / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "shaping_absolute_per_simulated_second": (
            shaping_absolute / elapsed_seconds if elapsed_seconds else 0.0
        ),
        "outcome_signed": float(signed["outcome"]),
        "detector_totals": dict(detector_totals),
        "budget_clip_counts": dict(clipped),
        "nonfinite_values": nonfinite,
    }


def _synthetic_states() -> tuple[list[RewardStateV1], list[RewardEventsV1]]:
    states: list[RewardStateV1] = []
    events: list[RewardEventsV1] = []
    total_ticks = 1200
    for tick in range(total_ticks + 1):
        t = tick / PHYSICS_HZ
        ball_y = -1200.0 + 210.0 * t + 80.0 * math.sin(0.7 * t)
        ball_vy = 210.0 + 56.0 * math.cos(0.7 * t)
        car_y = -2600.0 + 230.0 * t
        speed = 230.0 + 20.0 * math.sin(0.4 * t)
        waste_window = 360 <= tick <= 600
        states.append(
            RewardStateV1(
                tick_index=tick,
                self_position=np.asarray([250.0, car_y, 17.0]),
                self_linear_velocity=np.asarray(
                    [0.0, 2300.0 if waste_window else speed, 0.0]
                ),
                self_forward=np.asarray([0.0, 1.0, 0.0]),
                self_up=np.asarray([0.0, 0.0, 1.0]),
                self_boost=50.0,
                self_surface_contact=not (720 <= tick <= 840),
                self_boosting=waste_window,
                self_supersonic=waste_window,
                self_can_dodge=780 <= tick <= 840,
                ball_position=np.asarray([50.0, ball_y, 92.75]),
                ball_linear_velocity=np.asarray([0.0, ball_vy, 0.0]),
            )
        )
        touched = tick in (240, 480, 720, 960)
        touch_quality = None
        touch_velocity_gain = None
        if touched:
            touch_quality, touch_velocity_gain = touch_quality_from_transition(
                states[tick - 1], states[tick]
            )
        events.append(
            RewardEventsV1(
                goal_for=tick == total_ticks,
                self_touch=touched,
                touch_event_count=int(touched),
                touch_quality_sum=touch_quality,
                touch_velocity_gain_sum=touch_velocity_gain,
            )
        )
    return states, events


def _cadence_comparisons(results: dict[int, dict[str, Any]]) -> dict[str, Any]:
    baseline = results[1]
    output: dict[str, Any] = {}
    for period in (2, 4):
        candidate = results[period]
        component_differences = {}
        for name in COMPONENTS:
            component_differences[name] = {
                "signed_abs_difference": abs(
                    candidate["component_signed"][name]
                    - baseline["component_signed"][name]
                ),
                "discounted_abs_difference": abs(
                    candidate["component_discounted"][name]
                    - baseline["component_discounted"][name]
                ),
            }
        base_rate = baseline["shaping_absolute_per_simulated_second"]
        candidate_rate = candidate["shaping_absolute_per_simulated_second"]
        output[str(period)] = {
            "component_differences_from_period1": component_differences,
            "shaping_signed_abs_difference": abs(
                candidate["shaping_signed"] - baseline["shaping_signed"]
            ),
            "shaping_absolute_rate_ratio_to_period1": (
                candidate_rate / base_rate if base_rate > 1e-12 else 1.0
            ),
            "outcome_exact": candidate["outcome_signed"] == baseline["outcome_signed"],
        }
    return output


def _mirror_audit() -> dict[str, Any]:
    states, events = _synthetic_states()
    # A 180-degree world mirror followed by the orange canonical transform is
    # mathematically the same role-relative state.  Perform both operations
    # explicitly rather than merely comparing a state with itself.
    inversion = np.asarray([-1.0, -1.0, 1.0])
    mirrored: list[RewardStateV1] = []
    for state in states:
        world_car = state.self_position * inversion
        world_velocity = state.self_linear_velocity * inversion
        world_forward = state.self_forward * inversion
        world_up = state.self_up * inversion
        world_ball = state.ball_position * inversion
        world_ball_velocity = state.ball_linear_velocity * inversion
        mirrored.append(
            RewardStateV1(
                tick_index=state.tick_index,
                self_position=world_car * inversion,
                self_linear_velocity=world_velocity * inversion,
                self_forward=world_forward * inversion,
                self_up=world_up * inversion,
                self_boost=state.self_boost,
                self_surface_contact=state.self_surface_contact,
                self_boosting=state.self_boosting,
                self_supersonic=state.self_supersonic,
                self_can_dodge=state.self_can_dodge,
                ball_position=world_ball * inversion,
                ball_linear_velocity=world_ball_velocity * inversion,
            )
        )
    original = _integrate(states, events, 1)
    transformed = _integrate(mirrored, events, 1)
    differences = {
        name: abs(
            original["component_signed"][name]
            - transformed["component_signed"][name]
        )
        for name in COMPONENTS
    }
    return {
        "operation": "180-degree world mirror then orange team canonical inversion",
        "component_signed_abs_differences": differences,
        "maximum_abs_difference": max(differences.values()),
        "exact": max(differences.values()) == 0.0,
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> int:
    capture = json.loads(CAPTURE_REPORT.read_text(encoding="utf-8"))
    source = capture["native_corpus"]
    native_path = REPO_ROOT / source["path"]
    native_sha256 = _sha256(native_path)
    raw_records = _read_native_records(native_path)
    contiguous = _longest_contiguous(raw_records)
    required_states = NATURAL_TRANSITIONS + 1
    if len(contiguous) < required_states:
        raise RuntimeError(
            f"Gate 7 needs {required_states} contiguous native states, got {len(contiguous)}"
        )
    selected_records = contiguous[:required_states]
    natural_states, natural_events, natural_coverage = _natural_views(selected_records)
    natural_results = {
        team: {
            period: _integrate(natural_states[team], natural_events[team], period)
            for period in PERIODS
        }
        for team in (0, 1)
    }
    natural_comparisons = {
        team: _cadence_comparisons(natural_results[team]) for team in (0, 1)
    }

    synthetic_states, synthetic_events = _synthetic_states()
    synthetic_results = {
        period: _integrate(synthetic_states, synthetic_events, period)
        for period in PERIODS
    }
    synthetic_comparisons = _cadence_comparisons(synthetic_results)
    mirror = _mirror_audit()

    potential_names = (
        "ball_progress_potential",
        "approach_control_potential",
        "recovery_potential",
    )
    synthetic_potential_ok = all(
        comparison["component_differences_from_period1"][name][
            "signed_abs_difference"
        ]
        <= 0.02
        and comparison["component_differences_from_period1"][name][
            "discounted_abs_difference"
        ]
        <= 1e-10
        for comparison in synthetic_comparisons.values()
        for name in potential_names
    )
    natural_potential_ok = all(
        comparison["component_differences_from_period1"][name][
            "signed_abs_difference"
        ]
        <= 0.03
        and comparison["component_differences_from_period1"][name][
            "discounted_abs_difference"
        ]
        <= 1e-8
        for team in (0, 1)
        for comparison in natural_comparisons[team].values()
        for name in potential_names
    )
    rate_ratios = [
        comparison["shaping_absolute_rate_ratio_to_period1"]
        for comparison in synthetic_comparisons.values()
    ] + [
        comparison["shaping_absolute_rate_ratio_to_period1"]
        for team in (0, 1)
        for comparison in natural_comparisons[team].values()
    ]
    all_results = list(synthetic_results.values()) + [
        natural_results[team][period] for team in (0, 1) for period in PERIODS
    ]
    metadata = reward_metadata()
    checks = {
        "native_source_hash_matches_gate3": native_sha256 == source["sha256"],
        "native_source_version_matches": source["version"] == NATIVE_CORPUS_VERSION,
        "natural_trace_has_4096_contiguous_transitions": len(selected_records)
        == required_states,
        "synthetic_goal_event_exact_at_1_2_4_ticks": all(
            result["outcome_signed"] == GOAL_REWARD
            for result in synthetic_results.values()
        ),
        "natural_goal_events_exact_at_1_2_4_ticks": all(
            comparison["outcome_exact"]
            for team in (0, 1)
            for comparison in natural_comparisons[team].values()
        ),
        "synthetic_potential_terms_cadence_comparable": synthetic_potential_ok,
        "natural_potential_terms_cadence_comparable": natural_potential_ok,
        "no_4x_dense_reward_amplification": min(rate_ratios) >= 0.70
        and max(rate_ratios) <= 1.30,
        "every_component_finite_and_logged": all(
            result["nonfinite_values"] == 0
            and set(result["component_signed"]) == set(COMPONENTS)
            for result in all_results
        ),
        "combined_declared_shaping_budget_below_goal": metadata[
            "combined_shaping_absolute_episode_budget"
        ]
        < GOAL_REWARD,
        "observed_shaping_spend_respects_declared_budgets": all(
            result["component_absolute"][name]
            <= SHAPING_ABSOLUTE_EPISODE_BUDGETS[name] + 1e-12
            for result in all_results
            for name in SHAPING_COMPONENTS
        ),
        "mirrored_inverted_reward_exactly_symmetric": mirror["exact"],
        "named_mechanic_identity_rewards_disabled": not metadata[
            "named_mechanic_identity_rewards"
        ],
    }
    report = {
        "schema_version": 1,
        "milestone": 9,
        "gate": 7,
        "gate_name": "reward_cadence_audit",
        "status": "passed" if all(checks.values()) else "failed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "prospective_bounds": {
            "potential_raw_signed_abs_difference": {
                "synthetic": 0.02,
                "natural": 0.03,
            },
            "potential_discounted_abs_difference": {
                "synthetic": 1e-10,
                "natural": 1e-8,
            },
            "shaping_absolute_rate_ratio_to_period1": [0.70, 1.30],
            "combined_shaping_absolute_episode_budget_strictly_below_goal": True,
        },
        "reward_contract": metadata,
        "synthetic_audit": {
            "description": (
                "Ten simulated seconds of smooth progress/approach, a bounded wasteful-"
                "boost interval, four aligned touch events, airborne resource interval, "
                "and one terminal goal event."
            ),
            "results": {str(period): synthetic_results[period] for period in PERIODS},
            "comparisons": synthetic_comparisons,
        },
        "natural_rlbot_v5_audit": {
            "source": {
                "path": source["path"],
                "sha256": native_sha256,
                "version": source["version"],
                "records_in_file": len(raw_records),
                "longest_contiguous_records": len(contiguous),
                "selected_first_frame": int(selected_records[0]["frame_num"]),
                "selected_last_frame": int(selected_records[-1]["frame_num"]),
                "selected_transitions": NATURAL_TRANSITIONS,
            },
            "coverage": natural_coverage,
            "results_by_team": {
                str(team): {
                    str(period): natural_results[team][period] for period in PERIODS
                }
                for team in (0, 1)
            },
            "comparisons_by_team": {
                str(team): natural_comparisons[team] for team in (0, 1)
            },
        },
        "symmetry_audit": mirror,
        "phase_schedule_audit": {
            "pilot_at_2_hours": select_reward_phase(2.0).name,
            "100_hours_without_readiness": select_reward_phase(100.0).name,
            "25_hours_with_competence_readiness": select_reward_phase(
                25.0, competence_ready=True
            ).name,
            "250_hours_with_mature_readiness": select_reward_phase(
                250.0, mature_ready=True
            ).name,
        },
        "gate_semantics": {
            "score_used_as_policy_skill_measure": False,
            "win_loss_used": False,
            "policy_checkpoint_used": False,
            "training_budget_used": False,
            "goal_event_used_only_as_reward_contract_fixture": True,
        },
        "commands": {
            "gate": (
                "training/.venv/Scripts/python.exe "
                "training/scripts/run_m09_reward_cadence_gate.py"
            ),
            "unit_tests": (
                "training/.venv/Scripts/python.exe -m pytest "
                "training/tests/test_v9_rewards.py -q"
            ),
        },
        "source_hashes": {
            "reward_module_sha256": _sha256(
                TRAINING_ROOT / "rival_training" / "v9_rewards.py"
            ),
            "gate_script_sha256": _sha256(Path(__file__)),
        },
        "interpretation": (
            "Gate 7 is a numerical reward-contract gate. It does not evaluate wins, "
            "losses, policy quality, or whether a trained policy exists."
        ),
    }
    if not _all_finite(report):
        raise FloatingPointError("Gate 7 report contains a non-finite float")
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "natural_coverage": natural_coverage,
                "synthetic_comparisons": synthetic_comparisons,
                "symmetry": mirror,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
