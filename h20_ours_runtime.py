#!/usr/bin/env python3
"""Runtime for extracting retained-history priorities from Nested-Borda selection."""

from __future__ import annotations

import h20_direct_kl_nested_borda_runtime as BASE


_LAST_DELETION_POSITIONS_RAW: dict[int, int] | None = None


def current_deletion_positions_raw() -> dict[int, int]:
    if _LAST_DELETION_POSITIONS_RAW is None:
        raise RuntimeError("Nested-Borda selection must run before retrieving retained-history priorities")
    return dict(_LAST_DELETION_POSITIONS_RAW)


def retained_deletion_positions_raw(result) -> dict[int, int]:
    if not result.steps or result.steps[0].get("stage") != "one_shot_full_h20_borda_ordering":
        raise RuntimeError("Nested-Borda deletion order is missing")
    priorities = {
        int(row["candidate_chunk_index"]): int(
            row["one_shot_deletion_order_zero_based"]
        )
        for row in result.steps[0].get("candidate_scores", [])
    }
    selected = tuple(int(value) for value in result.canonical_indices)
    if set(selected) - set(priorities):
        raise RuntimeError("Selected histories are missing Nested-Borda priorities")
    return {19 - candidate: priorities[candidate] for candidate in selected}


class FinalAdaptiveH20Selector(BASE.FinalAdaptiveH20Selector):
    def select(self, record):
        global _LAST_DELETION_POSITIONS_RAW
        result = super().select(record)
        _LAST_DELETION_POSITIONS_RAW = retained_deletion_positions_raw(result)
        return result
