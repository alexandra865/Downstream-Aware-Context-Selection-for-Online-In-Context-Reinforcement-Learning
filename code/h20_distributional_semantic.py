#!/usr/bin/env python3

from __future__ import annotations

import os
import base64
import time
from typing import Sequence

import numpy as np
import requests


class FrozenOpenAIEmbeddingEncoder:
    def __init__(
        self,
        model: str,
        dimensions: int,
        endpoint: str,
        timeout_seconds: float,
        maximum_attempts: int,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.model = str(model)
        self.dimensions = int(dimensions)
        self.endpoint = str(endpoint)
        if self.endpoint != "https://api.openai.com/v1/embeddings":
            raise ValueError("Embedding endpoint must be https://api.openai.com/v1/embeddings")
        self.timeout_seconds = float(timeout_seconds)
        self.maximum_attempts = int(maximum_attempts)
        self.api_key_env = str(api_key_env)
        if self.model != "text-embedding-3-large":
            raise ValueError("The formal distributional encoder is fixed to text-embedding-3-large")
        if self.dimensions != 3072:
            raise ValueError("Formal text-embedding-3-large output must remain 3072-dimensional")
        if self.maximum_attempts < 1:
            raise ValueError("maximum_attempts must be positive")

    def _api_key(self) -> str:
        value = str(os.environ.get(self.api_key_env, "")).strip()
        if not value:
            raise RuntimeError(f"Missing {self.api_key_env} for OpenAI embeddings")
        return value

    def _request(self, texts: list[str]) -> np.ndarray:
        payload = {
            "model": self.model,
            "dimensions": self.dimensions,
            "encoding_format": "base64",
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key()}",
            "Content-Type": "application/json",
        }
        last_error = ""
        for attempt in range(1, self.maximum_attempts + 1):
            try:
                response = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                )
                if response.status_code == 200:
                    body = response.json()
                    if str(body.get("model", "")) != self.model:
                        raise RuntimeError(
                            f"Embedding model mismatch: {body.get('model')!r}"
                        )
                    data = sorted(body.get("data", []), key=lambda row: int(row["index"]))
                    if len(data) != len(texts):
                        raise RuntimeError("Embedding response row count mismatch")
                    decoded = []
                    for row in data:
                        raw = base64.b64decode(str(row["embedding"]), validate=True)
                        vector = np.frombuffer(raw, dtype="<f4")
                        decoded.append(vector)
                    values = np.asarray(decoded, dtype=np.float32)
                    if values.shape != (len(texts), self.dimensions):
                        raise RuntimeError(f"Embedding shape mismatch: {values.shape}")
                    if not np.isfinite(values).all():
                        raise RuntimeError("Embedding response contains nonfinite values")
                    norms = np.linalg.norm(values, axis=1, keepdims=True)
                    if np.any(norms <= 0):
                        raise RuntimeError("Embedding response contains a zero vector")
                    return values / norms
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
                if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                    break
            except (requests.RequestException, ValueError, RuntimeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.maximum_attempts:
                time.sleep(float(2 ** (attempt - 1)))
        raise RuntimeError(
            "OpenAI embedding request failed; hard-fail policy forbids fallback: "
            + last_error
        )

    def encode(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        values = list(map(str, texts))
        if not values:
            return np.empty((0, self.dimensions), dtype=np.float32)
        outputs = []
        for start in range(0, len(values), int(batch_size)):
            outputs.append(self._request(values[start:start + int(batch_size)]))
        return np.concatenate(outputs, axis=0)
