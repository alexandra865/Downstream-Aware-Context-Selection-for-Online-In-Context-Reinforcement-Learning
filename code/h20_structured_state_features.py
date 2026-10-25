#!/usr/bin/env python3
"""Observable current/history control-state features for H20 predictors."""

from __future__ import annotations

import math
import re
from typing import Any, Sequence

import numpy as np

from frozen_prompt_parser_v2 import prompt_snapshot, safe_float


QUERY_STRUCTURED_FEATURES = (
    "speed", "speed_missing",
    "acceleration", "acceleration_missing",
    "lane_index", "lane_index_missing",
    "lane_count", "lane_count_missing",
    "nearby_vehicle_count", "nearby_vehicle_count_missing",
    "minimum_distance", "minimum_distance_missing",
    "minimum_conflict_time", "minimum_conflict_time_missing",
    "lane_position", "lane_position_missing",
    "lane_length", "lane_length_missing",
    "distance_to_stop_line", "distance_to_stop_line_missing",
    "junction", "nav_left", "nav_right", "nav_straight",
    "traffic_light_none", "traffic_light_red", "traffic_light_yellow",
    "traffic_light_green",
)

HISTORY_STRUCTURED_FEATURES = (
    "recency", "token_count",
    "historical_action_1", "historical_action_2", "historical_action_3",
    "historical_action_4", "historical_action_8", "historical_reward",
    *QUERY_STRUCTURED_FEATURES,
    "history_change_score", "history_change_missing",
)

ACTION_IDS = (1, 2, 3, 4, 8)


def _first(pattern: str, text: str) -> tuple[float, bool]:
    match = re.search(pattern, text, flags=re.I)
    if not match:
        return math.nan, False
    value = safe_float(match.group(1))
    return value, math.isfinite(value)


def observable_snapshot(prompt: Any, navigation: Any = "") -> dict[str, Any]:
    """Parse only values visible in the exact policy prompt."""
    text = str(prompt or "")
    snapshot = dict(prompt_snapshot(text, navigation))
    acceleration, acceleration_observed = _first(
        r"###\s*Your current state:[\s\S]*?acceleration is\s*"
        r"(-?\d+(?:\.\d+)?)\s*m/s\^?2",
        text,
    )
    lane_position, lane_position_observed = _first(
        r"###\s*Your current state:[\s\S]*?lane position is\s*"
        r"(-?\d+(?:\.\d+)?)\s*m",
        text,
    )
    lane_length, lane_length_observed = _first(
        r"length of the current lane is\s*(-?\d+(?:\.\d+)?)\s*m",
        text,
    )
    stop_line, stop_line_observed = _first(
        r"stop line at the junction is\s*(-?\d+(?:\.\d+)?)\s*meters?",
        text,
    )
    light_match = re.search(
        r"traffic light is currently\s+(red|yellow|green)\b", text, flags=re.I
    )
    light = light_match.group(1).lower() if light_match else "none"
    snapshot.update({
        "acceleration": acceleration,
        "acceleration_observed": acceleration_observed,
        "lane_position": lane_position,
        "lane_position_observed": lane_position_observed,
        "lane_length": lane_length,
        "lane_length_observed": lane_length_observed,
        "distance_to_stop_line": stop_line,
        "distance_to_stop_line_observed": stop_line_observed,
        "traffic_light_state": light,
    })
    return snapshot


def _scaled(snapshot: dict[str, Any], value: str, observed: str,
            scale: float, clip: float | None = None) -> tuple[float, float]:
    present = bool(snapshot[observed])
    if not present:
        return 0.0, 1.0
    result = float(snapshot[value]) / float(scale)
    if clip is not None:
        result = max(-float(clip), min(float(clip), result))
    return result, 0.0


def query_structured_features(prompt: Any, navigation: Any = "") -> np.ndarray:
    snapshot = observable_snapshot(prompt, navigation)
    values = []
    for value, observed, scale, clip in (
        ("speed_from_prompt", "speed_observed", 20.0, 3.0),
        ("acceleration", "acceleration_observed", 10.0, 3.0),
        ("lane_index", "lane_index_observed", 8.0, 2.0),
        ("lane_count", "lane_count_observed", 8.0, 2.0),
        ("nearby_count", "nearby_count_observed", 10.0, 3.0),
        ("minimum_distance", "minimum_distance_observed", 100.0, 5.0),
        ("minimum_conflict_time", "minimum_conflict_time_observed", 20.0, 3.0),
        ("lane_position", "lane_position_observed", 200.0, 5.0),
        ("lane_length", "lane_length_observed", 200.0, 5.0),
        ("distance_to_stop_line", "distance_to_stop_line_observed", 200.0, 5.0),
    ):
        values.extend(_scaled(snapshot, value, observed, scale, clip))
    light = str(snapshot["traffic_light_state"])
    values.extend((
        float(snapshot["junction"]), float(snapshot["nav_left"]),
        float(snapshot["nav_right"]), float(snapshot["nav_straight"]),
        float(light == "none"), float(light == "red"),
        float(light == "yellow"), float(light == "green"),
    ))
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (len(QUERY_STRUCTURED_FEATURES),) or not np.all(np.isfinite(array)):
        raise RuntimeError("Invalid observable query structured vector")
    return array


def history_structured_features(prompts: Sequence[str], actions: Sequence[int],
                                rewards: Sequence[Any]) -> np.ndarray:
    if not (len(prompts) == len(actions) == len(rewards) == 20):
        raise ValueError("Expected aligned chronological H20 history")
    rows = []
    previous: np.ndarray | None = None
    for index, (prompt, action, reward) in enumerate(zip(prompts, actions, rewards)):
        action_value = int(action)
        if action_value not in ACTION_IDS:
            raise ValueError(f"Unknown historical action: {action_value}")
        state = query_structured_features(prompt, "")
        observed_values = state[0:20:2]
        observed_missing = state[1:20:2]
        comparable = observed_missing < 0.5
        changes = []
        if previous is not None:
            previous_values, previous_missing = previous[0:20:2], previous[1:20:2]
            valid = comparable & (previous_missing < 0.5)
            if np.any(valid):
                changes = np.abs(observed_values[valid] - previous_values[valid]).tolist()
        previous = state
        reward_value = safe_float(reward)
        if not math.isfinite(reward_value):
            raise ValueError("Historical reward must be finite")
        row = [
            index / 19.0,
            max(1, len(str(prompt).split())) / 512.0,
            *[float(action_value == candidate) for candidate in ACTION_IDS],
            reward_value / 10.0,
            *state.tolist(),
            float(np.mean(changes)) if changes else 0.0,
            float(not bool(changes)),
        ]
        rows.append(row)
    array = np.asarray(rows, dtype=np.float32)
    if array.shape != (20, len(HISTORY_STRUCTURED_FEATURES)) or not np.all(np.isfinite(array)):
        raise RuntimeError("Invalid observable history structured matrix")
    return array
