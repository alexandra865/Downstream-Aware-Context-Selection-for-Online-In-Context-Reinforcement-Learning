#!/usr/bin/env python3
"""Nested-Borda deletion ranking and adaptive history selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

import h20_selector_runtime as BASE
from h20_action_space import ACTION_IDS


class FinalAdaptiveH20Selector(BASE.FinalAdaptiveH20Selector):
    """Generate one nested family from Full-H20 and select only its depth."""

    @staticmethod
    def _all_candidate_borda(member_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ranks = np.empty_like(member_scores, dtype=np.int64)
        for member in range(member_scores.shape[0]):
            order = np.argsort(member_scores[member], kind="mergesort")
            ranks[member, order] = np.arange(member_scores.shape[1])
        return ranks, ranks.sum(axis=0)

    def _single_edge_predictions(
        self, arrays: dict[str, Any], parent: tuple[int, ...], candidate: int
    ) -> np.ndarray:
        if candidate not in parent:
            raise RuntimeError("Nested deletion candidate is absent from parent")
        status = np.zeros((1, 20), dtype=np.int64)
        status[0, list(parent)] = 1
        status[0, candidate] = 2
        legal = np.asarray([[
            action in arrays["legal"] for action in ACTION_IDS
        ]])
        output = []
        for model, scaler in self.members:
            query = arrays["query"][None, :]
            chunks = arrays["chunks"][None, :, :]
            structured = arrays["structured"][None, :, :]
            query, chunks, structured = scaler.arrays(query, chunks, structured)
            with torch.no_grad():
                output.append(float(model(
                    query_embedding=torch.tensor(query, device=self.device),
                    chunk_embeddings=torch.tensor(chunks, device=self.device),
                    chunk_structured=torch.tensor(structured, device=self.device),
                    slot_status=torch.tensor(status, dtype=torch.long, device=self.device),
                    legal_action_mask=torch.tensor(
                        legal, dtype=torch.bool, device=self.device
                    ),
                ).cpu().numpy()[0]))
        return np.asarray(output, dtype=np.float64)

    @staticmethod
    def _mask(selected: tuple[int, ...]) -> str:
        retained = set(selected)
        return "".join("1" if index in retained else "0" for index in range(20))

    def select(self, record: dict[str, Any]) -> BASE.AdaptiveSelection:
        arrays = self._arrays(record)
        full_parent = tuple(range(20))
        initial_scores = self._member_predictions(arrays, full_parent)
        if not np.isfinite(initial_scores).all():
            raise RuntimeError("Non-finite predictor outputs in Full-H20 ordering panel")
        initial_means = initial_scores.mean(axis=0)
        initial_stds = initial_scores.std(axis=0)
        if not (np.isfinite(initial_means).all() and np.isfinite(initial_stds).all()):
            raise RuntimeError("Non-finite predictor statistics in Full-H20 ordering panel")
        ranks, rank_sums = self._all_candidate_borda(initial_scores)
        deletion_order = tuple(sorted(
            full_parent,
            key=lambda candidate: (
                int(rank_sums[candidate]),
                float(initial_means[candidate]),
                -int(arrays["tokens"][candidate]),
                int(candidate),
            ),
        ))

        ordering_panel = []
        for candidate in full_parent:
            ordering_panel.append({
                "candidate_chunk_index": candidate,
                "child_mask": self._mask(tuple(
                    index for index in full_parent if index != candidate
                )),
                "base_predicted_mean_log1p_directional_kl": float(
                    initial_means[candidate]
                ),
                "predicted_std_log1p_directional_kl": float(initial_stds[candidate]),
                "member_predictions_log1p_directional_kl": [
                    float(value) for value in initial_scores[:, candidate]
                ],
                "member_ordinal_ranks_zero_based": [
                    int(value) for value in ranks[:, candidate]
                ],
                "borda_rank_sum": int(rank_sums[candidate]),
                "one_shot_deletion_order_zero_based": int(
                    deletion_order.index(candidate)
                ),
                "candidate_token_count": int(arrays["tokens"][candidate]),
            })

        selected = full_parent
        path = []
        for depth, candidate in enumerate(deletion_order[:20 - self.minimum_k], start=1):
            parent = selected
            child = tuple(index for index in parent if index != candidate)
            child_k = len(child)
            if depth == 1:
                member_values = initial_scores[:, candidate].astype(np.float64)
                score_source = "reused_full_h20_one_delete_panel"
            else:
                member_values = self._single_edge_predictions(
                    arrays, parent, candidate
                )
                score_source = "single_fixed_nested_edge"
            if not np.isfinite(member_values).all():
                raise RuntimeError(
                    f"Non-finite predictor outputs at nested depth {depth} "
                    f"(K={len(parent)} to K={child_k})"
                )
            mean = float(member_values.mean())
            std = float(member_values.std())
            if not np.isfinite([mean, std]).all():
                raise RuntimeError(
                    f"Non-finite predictor statistics at nested depth {depth} "
                    f"(K={len(parent)} to K={child_k})"
                )
            path.append({
                "current_K": len(parent),
                "current_mask": self._mask(parent),
                "child_K": child_k,
                "child_mask": self._mask(child),
                "selected_candidate": int(candidate),
                "nested_depth": depth,
                "base_predicted_mean_log1p_directional_kl": mean,
                "predicted_std_log1p_directional_kl": std,
                "member_predictions_log1p_directional_kl": [
                    float(value) for value in member_values
                ],
                "predicted_mean_feasible": bool(mean <= self.threshold),
                "score_source": score_source,
            })
            selected = child

        feasible = [row for row in path if row["predicted_mean_feasible"]]
        if feasible:
            chosen = min(feasible, key=lambda row: int(row["child_K"]))
            selected_mask = str(chosen["child_mask"])
            selected_indices = tuple(
                index for index, bit in enumerate(selected_mask) if bit == "1"
            )
            selected_k = int(chosen["child_K"])
            selected_score = float(
                chosen["base_predicted_mean_log1p_directional_kl"]
            )
        else:
            chosen = None
            selected_indices = full_parent
            selected_mask = "1" * 20
            selected_k = 20
            selected_score = None

        steps = ({
            "selection_method": "borda_nested_selection",
            "stage": "one_shot_full_h20_borda_ordering",
            "current_K": 20,
            "current_mask": "1" * 20,
            "candidate_scores": ordering_panel,
            "deletion_order": list(deletion_order),
            "candidate_edges_evaluated": 20,
            "decision": "ORDER_ONLY",
        }, *({
            **row,
            "selection_method": "borda_nested_selection",
            "stage": "fixed_nested_family_depth_scoring",
            "candidate_scores": [],
            "selected_base_prediction": row[
                "base_predicted_mean_log1p_directional_kl"
            ],
            "selected_adapted_prediction": row[
                "base_predicted_mean_log1p_directional_kl"
            ],
            "decision": (
                "SELECTED_DEPTH" if chosen is row else "SCORED_NOT_SELECTED"
            ),
        } for row in path))
        return BASE.AdaptiveSelection(
            canonical_indices=selected_indices,
            selected_mask=selected_mask,
            selected_K=selected_k,
            selected_upper_bound=selected_score,
            threshold=self.threshold,
            steps=tuple(steps),
        )
