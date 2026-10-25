#!/usr/bin/env python3
"""Fixed-K structured-state nearest-history retrieval for StateSim."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE = Path(__file__).resolve().parent
CODE = SOURCE / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from h20_structured_state_features import (  # noqa: E402
    QUERY_STRUCTURED_FEATURES,
    query_structured_features,
)


DEFAULT_ARTIFACT = HERE / "results/h20_statesim_baseline_v1/statesim_feature_scaler.json"


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


class StateSimSelector:
    def __init__(self, artifact_path: Path | str = DEFAULT_ARTIFACT) -> None:
        self.artifact_path = Path(artifact_path).resolve()
        artifact = json.loads(self.artifact_path.read_text())
        unhashed = dict(artifact)
        expected = str(unhashed.pop("artifact_sha256"))
        if canonical_hash(unhashed) != expected:
            raise RuntimeError("StateSim scaler artifact hash mismatch")
        if artifact["feature_names"] != list(QUERY_STRUCTURED_FEATURES):
            raise RuntimeError("StateSim feature schema changed")
        if int(artifact["evaluation_states_used"]) != 0:
            raise RuntimeError("Evaluation states entered StateSim scaler")
        self.artifact = artifact
        self.mean = np.asarray(artifact["feature_mean"], dtype=np.float64)
        self.std = np.asarray(artifact["feature_std"], dtype=np.float64)
        if self.mean.shape != (len(QUERY_STRUCTURED_FEATURES),) or self.std.shape != self.mean.shape:
            raise RuntimeError("StateSim scaler shape mismatch")
        if not (np.isfinite(self.mean).all() and np.isfinite(self.std).all()):
            raise RuntimeError("StateSim scaler statistics must be finite")
        if np.any(self.std <= 0):
            raise RuntimeError("StateSim scaler standard deviations must be positive")

    def vector(self, prompt: Any) -> np.ndarray:
        # Navigation is intentionally excluded to keep current/history symmetric.
        raw = query_structured_features(prompt, "").astype(np.float64)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scaled = (raw - self.mean) / self.std
        if not np.isfinite(scaled).all():
            raise RuntimeError("Non-finite StateSim normalized features")
        return scaled

    def select(self, current_prompt: Any, history_prompts: list[Any], k: int) -> tuple[list[int], dict]:
        if len(history_prompts) != 20 or k not in {13, 14, 15}:
            raise ValueError("StateSim requires H20 and fixed K in {13,14,15}")
        current = self.vector(current_prompt)
        histories = np.asarray([self.vector(prompt) for prompt in history_prompts])
        with np.errstate(over="ignore", invalid="ignore"):
            distances = np.linalg.norm(histories - current[None, :], axis=1)
        if not np.isfinite(distances).all():
            raise RuntimeError("Non-finite StateSim history distances")
        # Buffers are newest-to-oldest; lower raw index is the more recent tie-break.
        ranked = sorted(range(20), key=lambda raw: (float(distances[raw]), raw))
        selected_rank_order = ranked[:k]
        selected_chronological_buffer_order = sorted(selected_rank_order)
        exact_tie_pairs = sum(
            distances[left] == distances[right]
            for left in range(20) for right in range(left + 1, 20)
        )
        identical_pairs = sum(
            np.array_equal(histories[left], histories[right])
            for left in range(20) for right in range(left + 1, 20)
        )
        cutoff = float(distances[selected_rank_order[-1]])
        cutoff_tied = [raw for raw in range(20) if float(distances[raw]) == cutoff]
        audit_rows = [{
            "raw_index_newest_zero": int(raw),
            "history_age_decisions": int(raw + 1),
            "distance": float(distances[raw]),
            "similarity_rank_zero_based": int(rank),
            "selected": bool(raw in selected_rank_order),
        } for rank, raw in enumerate(ranked)]
        return selected_chronological_buffer_order, {
            "stage": "statesim_fixed_k_structured_l2",
            "K": int(k),
            "selected_raw_indices_similarity_order": selected_rank_order,
            "selected_raw_indices_buffer_order": selected_chronological_buffer_order,
            "selected_history_ages_chronological": [
                int(raw + 1) for raw in reversed(selected_chronological_buffer_order)
            ],
            "cutoff_distance": cutoff,
            "cutoff_tie_size": len(cutoff_tied),
            "cutoff_tie_raw_indices": cutoff_tied,
            "exact_distance_tie_pairs": int(exact_tie_pairs),
            "identical_feature_vector_pairs": int(identical_pairs),
            "candidate_scores": audit_rows,
        }
