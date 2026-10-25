#!/usr/bin/env python3
"""Canonical H20 chronology shared by selection and labeling code."""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from typing import Any


RAW_HISTORY_ORDER = "newest_to_oldest"
CANONICAL_HISTORY_ORDER = "oldest_to_newest"
VALID_ACTION_IDS = {1, 2, 3, 4, 8}


def _finite_reward(value: Any) -> bool:
    try:
        return bool(str(value).strip()) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def canonical_history(record: dict[str, Any], size: int = 20) -> list[dict[str, Any]]:
    declared = str(record.get("history_storage_order", RAW_HISTORY_ORDER))
    if declared != RAW_HISTORY_ORDER:
        raise ValueError(f"Unsupported raw history order: {declared!r}")
    prompts = list(record.get("previous_prompts") or [])
    decisions = list(record.get("previous_decisions") or [])
    rewards = list(record.get("rewards") or [])
    if not (len(prompts) == len(decisions) == len(rewards)):
        raise ValueError(
            "History buffers are misaligned: "
            f"prompts={len(prompts)} decisions={len(decisions)} rewards={len(rewards)}"
        )
    raw = []
    for raw_index, (prompt, decision, reward) in enumerate(
        zip(prompts[:size], decisions[:size], rewards[:size])
    ):
        if isinstance(decision, bool) or not isinstance(decision, (Integral, str)):
            raise ValueError(f"Invalid history action at raw index {raw_index}")
        try:
            action = int(decision)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid history action at raw index {raw_index}") from exc
        if not str(prompt or "").strip() or action not in VALID_ACTION_IDS or not _finite_reward(reward):
            raise ValueError(f"Invalid history chunk at raw index {raw_index}")
        raw.append({
            "raw_index": raw_index,
            "prompt": str(prompt),
            "decision": action,
            "reward": str(reward),
        })
    chronological = list(reversed(raw))
    for canonical_index, item in enumerate(chronological):
        item["canonical_index"] = canonical_index
        item["index"] = canonical_index
    return chronological


def compact_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def canonical_state_history_hash(record: dict[str, Any], size: int = 20) -> str:
    contexts = canonical_history(record, size=size)
    payload = {
        "history_order": CANONICAL_HISTORY_ORDER,
        "current_prompt": compact_text(record.get("current_short_prompt")),
        "history_prompts": [compact_text(item["prompt"]) for item in contexts],
        "history_actions": [item["decision"] for item in contexts],
        "history_rewards": [item["reward"] for item in contexts],
        "navigation": compact_text(record.get("navigation")),
        "available_actions": compact_text(record.get("actions")),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
