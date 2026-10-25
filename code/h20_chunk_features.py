"""Observable structured history features for H20 deletion scoring."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from frozen_prompt_parser_v2 import prompt_snapshot
from h20_action_space import HISTORICAL_ACTION_IDS as ACTION_IDS


def extract_chunk_structured_features(
    prompts: Sequence[str], actions: Sequence[int], rewards: Sequence[Any]
) -> np.ndarray:
    if not (len(prompts) == len(actions) == len(rewards) == 20):
        raise ValueError("Expected aligned 20-chunk history")
    rows = []
    previous: dict[str, Any] | None = None
    for index, (prompt, action, reward) in enumerate(zip(prompts, actions, rewards)):
        text = str(prompt)
        snapshot = prompt_snapshot(text)
        action_value = int(action)
        if action_value not in ACTION_IDS:
            raise ValueError(f"Unknown historical action: {action_value}")
        action_one_hot = [float(action_value == candidate) for candidate in ACTION_IDS]
        speed_observed = bool(snapshot["speed_observed"])
        lane_observed = bool(snapshot["lane_index_observed"])
        nearby_observed = bool(snapshot["nearby_count_observed"])
        conflict_observed = bool(snapshot["minimum_conflict_time_observed"])
        speed = float(snapshot["speed_from_prompt"]) if speed_observed else 0.0
        lane = float(snapshot["lane_index"]) if lane_observed else 0.0
        nearby = float(snapshot["nearby_count"]) if nearby_observed else 0.0
        conflict = float(snapshot["minimum_conflict_time"]) if conflict_observed else 0.0
        changes = []
        if previous is not None:
            for key, observed_key, scale in (
                ("speed_from_prompt", "speed_observed", 10.0),
                ("lane_index", "lane_index_observed", 4.0),
                ("nearby_count", "nearby_count_observed", 10.0),
                ("minimum_conflict_time", "minimum_conflict_time_observed", 10.0),
            ):
                if snapshot[observed_key] and previous[observed_key]:
                    changes.append(min(1.0, abs(float(snapshot[key]) - float(previous[key])) / scale))
        history_change = float(np.mean(changes)) if changes else 0.0
        history_change_missing = not bool(changes)
        previous = snapshot
        try:
            reward_value = float(reward)
        except (TypeError, ValueError):
            raise ValueError("Historical reward must be numeric")
        if not math.isfinite(reward_value):
            raise ValueError("Historical reward must be finite")
        rows.append([
            index / 19.0,
            len(text.split()) / 512.0,
            *action_one_hot,
            reward_value / 10.0,
            speed / 20.0,
            float(not speed_observed),
            lane / 8.0,
            float(not lane_observed),
            nearby / 10.0,
            float(not nearby_observed),
            min(conflict, 20.0) / 20.0 if conflict_observed else 0.0,
            float(not conflict_observed),
            history_change,
            float(history_change_missing),
        ])
    return np.asarray(rows, dtype=np.float32)
