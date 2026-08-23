"""Scrape and freeze the RLBot-v5 standard-Soccar authority evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.request import Request, urlopen

import numpy as np


TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from rival_training.v9_canonical import _pad_mapping  # noqa: E402
from rival_training.v9_soccar_geometry import (  # noqa: E402
    FLATBUFFER_SCHEMA_URL,
    GAME_DATA_URL,
    ROCKETSIM_PAD_ORB_POSITIONS,
    STANDARD_PAD_POSITIONS,
    USEFUL_GAME_VALUES_URL,
    geometry_authority_manifest,
)


RESULT_PATH = (
    TRAINING_ROOT / "results" / "milestone09" / "rlbot_v5_geometry_authority.json"
)
NATIVE_CAPTURE_PATH = (
    TRAINING_ROOT / "results" / "milestone09" / "gate03_native_capture.json"
)
RAW_SCHEMA_URL = (
    "https://raw.githubusercontent.com/RLBot/flatbuffers-schema/main/schema/gamedata.fbs"
)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.fragments: list[str] = []

    def handle_data(self, data: str) -> None:
        self.fragments.append(data)

    def text(self) -> str:
        return " ".join(" ".join(self.fragments).split())


def _fetch(url: str) -> tuple[bytes, dict[str, Any]]:
    request = Request(url, headers={"User-Agent": "Rival-M09-authority-audit/1"})
    with urlopen(request, timeout=30) as response:
        body = response.read()
        record = {
            "requested_url": url,
            "resolved_url": response.geturl(),
            "http_status": int(response.status),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
    return body, record


def _number_after(text: str, label: str) -> float:
    match = re.search(re.escape(label) + r"[^0-9-]*(-?[0-9][0-9,.]*)", text)
    if match is None:
        raise ValueError(f"Unable to scrape numeric value after {label!r}")
    return float(match.group(1).replace(",", ""))


def _scrape_pad_table(text: str) -> np.ndarray:
    matches = re.findall(
        r"([0-9]+):\s*\[\s*([-0-9.]+),\s*([-0-9.]+),\s*([-0-9.]+)\]",
        text,
    )
    if len(matches) != 34:
        raise ValueError(f"Expected 34 RLBot pad rows, scraped {len(matches)}")
    rows = sorted(
        (
            int(index),
            (float(x), float(y), float(z)),
        )
        for index, x, y, z in matches
    )
    if [index for index, _position in rows] != list(range(34)):
        raise ValueError("Scraped boost-pad indices are not the contiguous range 0..33")
    return np.asarray([position for _index, position in rows], dtype=np.float64)


def main() -> int:
    page_bytes, page_source = _fetch(USEFUL_GAME_VALUES_URL)
    schema_bytes, schema_source = _fetch(RAW_SCHEMA_URL)

    parser = _TextExtractor()
    parser.feed(page_bytes.decode("utf-8", errors="replace"))
    page_text = parser.text()
    schema_text = schema_bytes.decode("utf-8")

    scraped_values = {
        "floor_z": _number_after(page_text, "Floor:"),
        "side_wall_abs_x": _number_after(page_text, "Side wall: x="),
        "side_wall_length": _number_after(page_text, "Side wall length:"),
        "back_wall_abs_y": _number_after(page_text, "Back wall: y="),
        "back_wall_length": _number_after(page_text, "Back wall length:"),
        "ceiling_z": _number_after(page_text, "Ceiling: z="),
        "goal_height": _number_after(page_text, "Goal height: z="),
        "goal_half_width": _number_after(page_text, "Goal center-to-post:"),
        "goal_depth": _number_after(page_text, "Goal depth:"),
        "corner_wall_length": _number_after(page_text, "Corner wall length:"),
        "corner_plane_axis_intercept": _number_after(
            page_text, "The corner planes intersect the axes at"
        ),
        "wall_bottom_ramp_radius_approximate": _number_after(
            page_text, "Wall bottom ramp radius: Aprox."
        ),
        "standard_gravity_magnitude": _number_after(page_text, "Gravity:"),
        "ball_radius": _number_after(page_text, "Radius:"),
        "ball_max_speed": _number_after(page_text, "Max speed:"),
        "car_max_speed": _number_after(page_text, "Max car speed (boosting):"),
        "car_supersonic_threshold": _number_after(
            page_text, "Supersonic speed threshold:"
        ),
        "car_max_no_boost_speed": _number_after(
            page_text, "Max driving speed (forward and backward) with no boost:"
        ),
        "car_max_angular_speed": _number_after(
            page_text, "Maximum car angular velocity:"
        ),
    }
    scraped_pads = _scrape_pad_table(page_text)
    manifest = geometry_authority_manifest()
    frozen_values = {
        key: float(manifest["standard_soccar"].get(key))
        for key in (
            "floor_z",
            "side_wall_abs_x",
            "side_wall_length",
            "back_wall_abs_y",
            "back_wall_length",
            "ceiling_z",
            "goal_height",
            "goal_half_width",
            "goal_depth",
            "corner_wall_length",
            "corner_plane_axis_intercept",
            "wall_bottom_ramp_radius_approximate",
        )
    }
    frozen_values.update(
        {
            key: float(manifest["physics_values_used_by_rival"][key])
            for key in (
                "standard_gravity_magnitude",
                "ball_radius",
                "ball_max_speed",
                "car_max_speed",
                "car_supersonic_threshold",
                "car_max_no_boost_speed",
                "car_max_angular_speed",
            )
        }
    )

    value_errors = {
        key: abs(float(scraped_values[key]) - float(frozen_values[key]))
        for key in frozen_values
    }
    pad_errors = np.abs(scraped_pads - STANDARD_PAD_POSITIONS.astype(np.float64))
    source_map = _pad_mapping(ROCKETSIM_PAD_ORB_POSITIONS, False)
    mapped_orbs = ROCKETSIM_PAD_ORB_POSITIONS[source_map]
    source_xy_errors = np.linalg.norm(
        mapped_orbs[:, :2].astype(np.float64) - scraped_pads[:, :2], axis=1
    )
    native_capture = (
        json.loads(NATIVE_CAPTURE_PATH.read_text(encoding="utf-8"))
        if NATIVE_CAPTURE_PATH.is_file()
        else None
    )
    live_field_info_goals = (
        []
        if native_capture is None
        else native_capture.get("native_corpus", {}).get(
            "field_info_goal_metadata", []
        )
    )

    checks = {
        "useful_values_http_200": page_source["http_status"] == 200,
        "schema_http_200": schema_source["http_status"] == 200,
        "all_scraped_values_match_frozen_constants": max(value_errors.values()) <= 1e-9,
        "all_34_scraped_pad_rows_match_canonical_table": float(np.max(pad_errors)) <= 1e-5,
        "rlgym_source_table_maps_uniquely_to_rlbot_order": sorted(source_map.tolist())
        == list(range(34)),
        "rlgym_source_xy_differences_are_bounded": float(np.max(source_xy_errors)) <= 2.0,
        "field_info_exposes_goals": all(
            token in schema_text
            for token in (
                "table GoalInfo",
                "location: Vector3 (required);",
                "direction: Vector3 (required);",
                "width: float;",
                "height: float;",
                "goals: [GoalInfo] (required);",
            )
        ),
        "field_info_exposes_boost_pads": all(
            token in schema_text
            for token in (
                "table BoostPad",
                "boost_pads: [BoostPad] (required);",
            )
        ),
        "curved_ramp_is_not_claimed_exact": (
            "not circular" in page_text and manifest["standard_soccar"]["wall_bottom_ramp_exact"] is False
        ),
        "live_field_info_goal_metadata_captured": len(live_field_info_goals) == 2,
    }
    result = {
        "milestone": 9,
        "purpose": "RLBot v5 standard-Soccar geometry and runtime FieldInfo authority audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "sources": {
            "useful_game_values": page_source,
            "game_data_documentation": GAME_DATA_URL,
            "flatbuffer_schema_documentation": FLATBUFFER_SCHEMA_URL,
            "flatbuffer_schema_raw": schema_source,
        },
        "scraped_values": scraped_values,
        "frozen_value_absolute_errors": value_errors,
        "boost_pad_audit": {
            "scraped_rows": int(scraped_pads.shape[0]),
            "canonical_max_abs_coordinate_error": float(np.max(pad_errors)),
            "rlgym_to_rlbot_index_map": source_map.tolist(),
            "rlgym_to_rlbot_max_xy_error": float(np.max(source_xy_errors)),
            "representation_note": (
                "RLBot canonical positions use FieldInfo pickup anchors; RLGym source positions use "
                "orb centers and a different order. Identity is mapped by bounded XY distance."
            ),
        },
        "goal_authority": {
            "wiki_defines_standard_opening_and_depth": True,
            "wiki_does_not_publish_every_runtime_goal_tuple": True,
            "runtime_schema": "FieldInfo.goals: team_num, location, direction, width, height",
            "semantic_caveat": (
                "FieldInfo is captured and audited, but a live v5 beta may expose a larger "
                "goal/scoring volume rather than the physical opening/post dimensions."
            ),
            "live_stadium_p_field_info": live_field_info_goals,
            "live_observation_conclusion": (
                "The captured v5 beta values are approximately 1920 x 752 at z=312, "
                "so Rival keeps them as runtime goal/scoring-volume evidence and does not "
                "substitute them for the documented 1785.51 x 642.775 physical opening."
            ),
            "rival_standard_goal_centers_are_derived": manifest[
                "derived_standard_goal_centers"
            ],
        },
        "clearance_policy": {
            "exact_from_wiki": "floor, ceiling, planar walls, 45-degree corner plane, rectangular goal opening/depth",
            "not_exact_from_wiki": "curved bottom ramps, rounded posts, full collision mesh",
            "rival_feature_claim": manifest["scope_limit"],
        },
        "authority_manifest": manifest,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": result["status"], "path": str(RESULT_PATH), "checks": checks}, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
