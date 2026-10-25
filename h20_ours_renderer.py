#!/usr/bin/env python3
"""Learned order-guided redundancy compression (LORC) for driving.

Repeated static fields are omitted from duplicate history positions.
The ScienceWorld renderer uses explicit textual references.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from h20_scene_parser import (
    DYNAMIC_SECTIONS, FIELDS, HISTORY_INTRO, STATIC_SECTIONS, _value, parse_scene,
)


def compose_lorc_prompt(
    short_prompt: str,
    selected_raw_indices: list[int],
    selected_prompts: list[Any],
    selected_decisions: list[Any],
    selected_rewards: list[Any],
    deletion_positions_by_raw_index: dict[int, int],
    *,
    return_audit: bool = False,
):
    lengths = {
        len(selected_raw_indices), len(selected_prompts),
        len(selected_decisions), len(selected_rewards),
    }
    if len(lengths) != 1:
        raise RuntimeError("Selected history buffers are misaligned")
    if selected_raw_indices != sorted(selected_raw_indices):
        raise RuntimeError("Selected raw indices must use newest-to-oldest order")
    if set(deletion_positions_by_raw_index) != set(selected_raw_indices):
        raise RuntimeError("LORC requires a deletion-order position for every selected history")

    active = parse_scene(short_prompt)
    parsed = {
        raw: parse_scene(scene)
        for raw, scene in zip(selected_raw_indices, selected_prompts, strict=True)
    }
    omitted: set[tuple[int, str]] = set()
    current_value_omissions = 0
    changed_value_duplicate_omissions = 0
    repeated_value_groups = 0
    repeated_changed_group_sizes: list[int] = []

    if active.parse_ok and all(scene.parse_ok for scene in parsed.values()):
        for field in FIELDS:
            current = _value(active, field)
            groups: dict[str, list[int]] = defaultdict(list)
            for raw, scene in parsed.items():
                value = _value(scene, field)
                if value:
                    groups[value].append(raw)
            for value, members in groups.items():
                if current and value == current:
                    for raw in members:
                        omitted.add((raw, field))
                    current_value_omissions += len(members)
                    continue
                retained_index = max(
                    members, key=lambda raw: (deletion_positions_by_raw_index[raw], -raw)
                )
                if len(members) >= 2:
                    repeated_value_groups += 1
                    repeated_changed_group_sizes.append(len(members))
                for raw in members:
                    if raw != retained_index:
                        omitted.add((raw, field))
                        changed_value_duplicate_omissions += 1

    chunks: list[str] = []
    correctness_failures = 0
    emitted_changed_values: dict[tuple[str, str], int] = defaultdict(int)
    for raw, scene, decision, reward in reversed(list(zip(
        selected_raw_indices, selected_prompts,
        selected_decisions, selected_rewards, strict=True,
    ))):
        item = parsed[raw]
        header = f"History age: {int(raw)} decisions ago\n"
        footer = f"Action: {decision}\nReward: {reward}\n\n"
        if not active.parse_ok or not item.parse_ok:
            text = header + item.original
            if text and not text.endswith("\n"):
                text += "\n"
            chunks.append(text + footer)
            continue

        retained = [item.prefix] if item.prefix else []
        for field, block in item.sections:
            if field in DYNAMIC_SECTIONS or field not in STATIC_SECTIONS:
                retained.append(block)
            elif (raw, field) not in omitted:
                retained.append(block)
                value = _value(item, field)
                if value and value != _value(active, field):
                    emitted_changed_values[(field, value)] += 1
        if item.weather and (raw, "Weather") not in omitted:
            retained.append(item.weather)
            if item.weather != active.weather:
                emitted_changed_values[("Weather", item.weather)] += 1

        text = header + "".join(retained)
        if text and not text.endswith("\n"):
            text += "\n"
        text += footer
        for field in DYNAMIC_SECTIONS:
            value = item.section_map().get(field)
            if value and value not in text:
                correctness_failures += 1
        chunks.append(text)

    if active.parse_ok and all(scene.parse_ok for scene in parsed.values()):
        expected_changed = {
            (field, _value(scene, field))
            for field in FIELDS
            for scene in parsed.values()
            if _value(scene, field) and _value(scene, field) != _value(active, field)
        }
        if set(emitted_changed_values) != expected_changed:
            correctness_failures += 1
        if any(count != 1 for count in emitted_changed_values.values()):
            correctness_failures += 1

    prompt = str(short_prompt) + HISTORY_INTRO + "".join(chunks)
    if not prompt.startswith(str(short_prompt)):
        correctness_failures += 1
    audit = {
        "current_value_omissions": current_value_omissions,
        "changed_value_duplicate_omissions": changed_value_duplicate_omissions,
        "repeated_value_groups": repeated_value_groups,
        "repeated_changed_group_sizes": repeated_changed_group_sizes,
        "correctness_failures": correctness_failures,
        "new_bytes_added": 0,
        "chunk_order_modifications": 0,
    }
    return (prompt, audit) if return_audit else prompt
