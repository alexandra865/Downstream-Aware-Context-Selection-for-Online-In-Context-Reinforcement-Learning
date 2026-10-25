"""Shared scaling, action parsing and result types for H20 inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from h20_action_space import ACTION_IDS


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()


def parse_legal_actions(text: str) -> tuple[str, ...]:
    fields = re.findall(r"(?i)\bAction_id[ \t]*:[ \t]*([^\r\n]*)", str(text))
    values = tuple(dict.fromkeys(field.strip() for field in fields))
    if not values or any(value not in ACTION_IDS for value in values):
        raise ValueError("Could not recover displayed legal actions")
    return values


class CheckpointScaler:
    def __init__(self, payload: dict[str, Any]):
        self.query = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in payload["query"].items()
        }
        self.context = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in payload["context"].items()
        }
        self._validate_statistics(self.query, "Query")
        self._validate_statistics(self.context, "Context")

    @staticmethod
    def _validate_statistics(spec: dict[str, np.ndarray], name: str) -> int:
        try:
            median, mean, std = (
                np.asarray(spec[key]) for key in ("median", "mean", "std")
            )
        except KeyError as exc:
            raise ValueError(f"{name} scaler requires median, mean and std") from exc
        if (mean.ndim != 1 or mean.size == 0
                or median.shape != mean.shape or std.shape != mean.shape):
            raise ValueError(f"{name} scaler statistics must be matching nonempty vectors")
        if not all(np.isfinite(value).all() for value in (median, mean, std)):
            raise ValueError(f"{name} scaler statistics must be finite")
        if np.any(std <= 0):
            raise ValueError(f"{name} scaler std must be strictly positive")
        return int(mean.size)

    @staticmethod
    def apply(values: np.ndarray, spec: dict[str, np.ndarray]) -> np.ndarray:
        features = CheckpointScaler._validate_statistics(spec, "Input")
        if values.ndim == 0 or values.shape[-1] != features:
            raise ValueError(f"Scaler input must have {features} features in its last dimension")
        filled = np.where(np.isnan(values), spec["median"], values)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scaled = ((filled - spec["mean"]) / spec["std"]).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled predictor features must be finite")
        return scaled

    def arrays(self, query: np.ndarray, chunks: np.ndarray,
               structured: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaled_query = self.apply(query, self.query)
        joined = np.concatenate((chunks, structured), axis=-1)
        joined = self.apply(joined.reshape(-1, joined.shape[-1]), self.context).reshape(joined.shape)
        split = query.shape[-1]
        return scaled_query, joined[:, :, :split], joined[:, :, split:]


@dataclass(frozen=True)
class AdaptiveSelection:
    canonical_indices: tuple[int, ...]
    selected_mask: str
    selected_K: int
    selected_upper_bound: float | None
    threshold: float
    steps: tuple[dict[str, Any], ...]
