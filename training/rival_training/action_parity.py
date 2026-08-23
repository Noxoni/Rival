"""Milestone 07 spatial and temporal action-function parity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np

from .actions import RivalActionParser, build_expanded_action_table
from .checkpoint import portable_path
from .teacher import sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BOT_ROOT = REPOSITORY_ROOT / "bot"
PRODUCTION_SITE_PACKAGES = REPOSITORY_ROOT / ".venv/Lib/site-packages"
if str(PRODUCTION_SITE_PACKAGES) not in sys.path:
    # The training environment intentionally excludes RLBot. Append only as a
    # fallback so its RLBot flatbuffer module is available to production types
    # without overriding the training environment's NumPy/Torch/RLGym stack.
    sys.path.append(str(PRODUCTION_SITE_PACKAGES))
if str(BOT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOT_ROOT))

from action_parser import DefaultAction, XMirroredActionParser  # noqa: E402
from backend.gamestate.team import Team  # noqa: E402


ACTION_TABLE_PATH = (
    REPOSITORY_ROOT / "bot/models/RIVAL_ACTIONS_V1.npy"
)


@dataclass(frozen=True)
class MirrorCase:
    name: str
    team_num: int
    x: float

    @property
    def mirror_x(self) -> bool:
        return (self.team_num == 1) != (self.x < 0)


MIRROR_CASES = (
    MirrorCase("blue_negative_x", 0, -1000.0),
    MirrorCase("blue_positive_x", 0, 1000.0),
    MirrorCase("orange_negative_x", 1, -1000.0),
    MirrorCase("orange_positive_x", 1, 1000.0),
)


def _production_action(parser: DefaultAction, index: int, case: MirrorCase) -> np.ndarray:
    player = SimpleNamespace(
        team=Team.ORANGE if case.team_num == 1 else Team.BLUE,
        pos=SimpleNamespace(x=case.x),
    )
    return parser.get_action(index, player, SimpleNamespace()).get_np().astype(np.float32)


def _training_action(parser: RivalActionParser, index: int, case: MirrorCase) -> np.ndarray:
    agent = "rival"
    car = SimpleNamespace(
        team_num=case.team_num,
        physics=SimpleNamespace(position=np.array([case.x, 0.0, 17.0], dtype=np.float32)),
    )
    state = SimpleNamespace(cars={agent: car})
    shared_info: dict[str, Any] = {}
    parsed = parser.parse_actions(
        {agent: np.array([index], dtype=np.int64)}, state, shared_info
    )[agent]
    if parsed.shape != (parser.repeats, 8):
        raise AssertionError(f"Unexpected repeated action shape {parsed.shape}")
    if not np.all(parsed == parsed[0]):
        raise AssertionError("Training parser changed a controller inside its repeat window")
    return parsed[0]


def _temporal_window(tick_skip: int) -> dict[str, Any]:
    action_delay = tick_skip - 1
    live_cycle_ticks = list(range(1, tick_skip + 1))
    live_applied = [
        "new" if tick >= action_delay - 1 else "previous"
        for tick in live_cycle_ticks
    ]
    rocketsim_applied = ["previous"] + ["new"] * (tick_skip - 1)
    mismatch_ticks = [
        tick
        for tick, (live, sim) in enumerate(
            zip(live_applied, rocketsim_applied), start=1
        )
        if live != sim
    ]
    return {
        "tick_skip": tick_skip,
        "configured_live_action_delay": action_delay,
        "steady_state_cycle_ticks": live_cycle_ticks,
        "live_wisp_applied_action": live_applied,
        "rocketsim_rlbot_delay_applied_action": rocketsim_applied,
        "live_previous_action_ticks": live_applied.count("previous"),
        "live_new_action_ticks": live_applied.count("new"),
        "rocketsim_previous_action_ticks": rocketsim_applied.count("previous"),
        "rocketsim_new_action_ticks": rocketsim_applied.count("new"),
        "mismatch_ticks": mismatch_ticks,
        "exact_window_match": not mismatch_ticks,
        "interpretation": (
            "RocketSimEngine(rlbot_delay=True) advances physics once before setting each "
            "row's controls. The live loop selects at steady-state tick_window=1 and "
            "continues the previous action until tick_window >= ACTION_DELAY - 1."
        ),
    }


def build_action_parity_report() -> dict[str, Any]:
    artifact_table = np.load(ACTION_TABLE_PATH, allow_pickle=False).astype(np.float32)
    generated_table = build_expanded_action_table()
    if not np.array_equal(artifact_table, generated_table):
        raise RuntimeError("M06 artifact action table differs from the generated table")

    production_frozen = XMirroredActionParser()
    production_candidate = XMirroredActionParser(
        ACTION_TABLE_PATH,
        legacy_only=True,
    )
    training = RivalActionParser(cadence="mechanics4")
    comparisons = []
    maximum_error = 0.0
    exact_rows = 0
    total_rows = 0
    for case in MIRROR_CASES:
        case_exact = 0
        case_maximum = 0.0
        for index in range(90):
            frozen = _production_action(production_frozen, index, case)
            candidate = _production_action(production_candidate, index, case)
            simulated = _training_action(training, index, case)
            error = max(
                float(np.max(np.abs(frozen - candidate))),
                float(np.max(np.abs(frozen - simulated))),
            )
            case_maximum = max(case_maximum, error)
            maximum_error = max(maximum_error, error)
            exact = bool(
                np.array_equal(frozen, candidate)
                and np.array_equal(frozen, simulated)
            )
            case_exact += int(exact)
            exact_rows += int(exact)
            total_rows += 1
        comparisons.append(
            {
                "case": case.name,
                "team_num": case.team_num,
                "x": case.x,
                "mirror_x": case.mirror_x,
                "legacy_rows_compared": 90,
                "exact_rows": case_exact,
                "maximum_abs_controller_error": case_maximum,
            }
        )

    appended_rejected = False
    try:
        _production_action(production_candidate, 90, MIRROR_CASES[0])
    except IndexError:
        appended_rejected = True
    temporal = {str(ticks): _temporal_window(ticks) for ticks in (4, 8)}
    spatial_passed = exact_rows == total_rows and maximum_error == 0.0
    return {
        "schema_version": 1,
        "status": "passed" if spatial_passed and appended_rejected else "failed",
        "purpose": "milestone07_action_function_parity",
        "controller_fields": [
            "throttle",
            "steer",
            "pitch",
            "yaw",
            "roll",
            "jump",
            "boost",
            "handbrake",
        ],
        "action_table": {
            "path": portable_path(ACTION_TABLE_PATH),
            "sha256": sha256_file(ACTION_TABLE_PATH),
            "shape": list(artifact_table.shape),
            "generated_table_exact": True,
        },
        "spatial_controller_parity": {
            "production_paths": [
                "frozen DefaultAction/XMirroredActionParser",
                "158-action diagnostic candidate with legacy-only guard",
            ],
            "training_path": "RivalActionParser",
            "legacy_rows_per_case": 90,
            "comparisons": comparisons,
            "total_row_case_comparisons": total_rows,
            "exact_row_case_comparisons": exact_rows,
            "maximum_abs_controller_error": maximum_error,
            "appended_index_90_rejected_by_legacy_only_deployment": appended_rejected,
            "passed": spatial_passed and appended_rejected,
        },
        "temporal_action_function": {
            "source_derived_not_physics_outcome": True,
            "tick_4": temporal["4"],
            "tick_8": temporal["8"],
            "finding": (
                "The four-tick live and RocketSim repeat/delay windows are exact in "
                "steady state. The eight-tick windows differ on ticks 2-5: RocketSim "
                "has already applied the new action while live Wisp is still applying "
                "the previous action."
            ),
        },
        "production_modified_or_promoted": False,
    }


def write_action_parity_report(path: str | Path) -> dict[str, Any]:
    report = build_action_parity_report()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report
