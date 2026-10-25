"""Artifact loading and decision-gap inference for Nested-Borda selection."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from h20_action_space import ACTION_IDS
from h20_direct_action_kl_model import DirectActionKLModel
from h20_predictor_support import (
    AdaptiveSelection, CheckpointScaler, canonical_hash, parse_legal_actions, sha256_file,
)
from h20_distributional_semantic import FrozenOpenAIEmbeddingEncoder
from h20_history_contract import canonical_history
from h20_chunk_features import extract_chunk_structured_features


ARTIFACT_NAME = "h20_selector_artifact.json"


class FinalAdaptiveH20Selector:
    """Load the predictor ensemble used by Nested-Borda history selection."""

    def __init__(self, artifact_dir: Path | str, device: str = "cpu") -> None:
        artifact_location = Path(artifact_dir).resolve()
        artifact_path = (
            artifact_location if artifact_location.is_file()
            else artifact_location / ARTIFACT_NAME
        )
        self.artifact_dir = artifact_path.parent
        self.artifact = json.loads(artifact_path.read_text())
        unhashed = dict(self.artifact)
        expected = unhashed.pop("artifact_sha256", "")
        if canonical_hash(unhashed) != expected:
            raise RuntimeError("Decision gap predictor artifact self-hash mismatch")
        if self.artifact.get("formal_deployment_certified") is not False:
            raise RuntimeError(
                "Expected a research-only selector artifact with "
                "formal_deployment_certified=False"
            )
        self.mode = str(self.artifact["mode"])
        if self.mode != "static":
            raise RuntimeError(f"Only static selector mode is supported: {self.mode}")
        self.minimum_k = int(self.artifact["minimum_K"])
        self.threshold_kl = float(self.artifact["threshold_directional_KL"])
        self.threshold = float(self.artifact["threshold_log1p_directional_KL"])
        if not np.isfinite([self.threshold_kl, self.threshold]).all():
            raise RuntimeError("KL thresholds must be finite")
        if self.minimum_k != 10 or self.threshold_kl != 0.05:
            raise RuntimeError("Adaptive history selection requires minimum_K=10 and KL threshold=0.05")
        if not np.isclose(
            self.threshold, np.log1p(self.threshold_kl), rtol=1e-9, atol=1e-12
        ):
            raise RuntimeError("Log KL threshold must equal log1p of the raw KL threshold")
        if bool(self.artifact["adapter"]["enabled"]):
            raise RuntimeError("Static selector requires adapter.enabled=False")
        if self.artifact.get("support_examples", []):
            raise RuntimeError("Static selector must not contain support examples")

        ranking_path = ROOT / self.artifact["ranking_artifact_path"]
        if sha256_file(ranking_path) != self.artifact["ranking_artifact_file_sha256"]:
            raise RuntimeError("Predictor ranking artifact file hash mismatch")
        ranking = json.loads(ranking_path.read_text())
        ranking_unhashed = dict(ranking)
        ranking_expected = ranking_unhashed.pop("artifact_sha256", "")
        if canonical_hash(ranking_unhashed) != ranking_expected:
            raise RuntimeError("Predictor ranking artifact self-hash mismatch")
        if ranking.get("deployable_stop_rule_present") is not False:
            raise RuntimeError("Unexpected deployable STOP rule in ranking artifact")

        self.device = torch.device(device)
        model_dir = ROOT / ranking["model_dir"]
        self.members = []
        for filename, expected_hash in sorted(ranking["checkpoint_sha256"].items()):
            path = model_dir / "checkpoints" / filename
            if sha256_file(path) != expected_hash:
                raise RuntimeError(f"Predictor checkpoint hash mismatch: {filename}")
            payload = torch.load(path, map_location=self.device, weights_only=True)
            if payload.get("primary_target") != "log1p_directional_kl_pre_reasoning_direct_action":
                raise RuntimeError("Wrong Direct-KL checkpoint target semantics")
            spec = payload["model_config"]
            model = DirectActionKLModel(
                payload["embedding_dim"], payload["structured_dim"],
                spec["hidden_dim"], spec["transformer_layers"],
                spec["attention_heads"], spec["dropout"],
            ).to(self.device)
            model.load_state_dict(payload["model_state_dict"])
            model.eval()
            self.members.append((model, CheckpointScaler(payload["scaler"])))
        if len(self.members) != 5:
            raise RuntimeError("Decision gap predictor requires five ensemble members")
        encoder = self.artifact["semantic_encoder"]
        self.semantic = FrozenOpenAIEmbeddingEncoder(
            encoder["model"], int(encoder["dimensions"]), encoder["endpoint"],
            float(encoder["timeout_seconds"]), int(encoder["maximum_attempts"]),
        )

    def _arrays(self, record: dict[str, Any]) -> dict[str, Any]:
        history = canonical_history(record)
        if len(history) != 20:
            raise ValueError("Adaptive history selection requires H20")
        encoder = self.artifact["semantic_encoder"]
        query_text = encoder["query_prefix"] + "\n".join((
            str(record["current_short_prompt"]), str(record["navigation"]),
            str(record["actions"]),
        ))
        prompts = [str(item["prompt"]) for item in history]
        embedded = self.semantic.encode(
            [query_text, *[encoder["chunk_prefix"] + prompt for prompt in prompts]],
            int(encoder["runtime_request_batch_size"]),
        )
        return {
            "query": embedded[0].astype(np.float32),
            "chunks": embedded[1:].astype(np.float32),
            "structured": extract_chunk_structured_features(
                prompts, [int(item["decision"]) for item in history],
                [item["reward"] for item in history],
            ).astype(np.float32),
            "tokens": np.asarray([max(1, len(prompt.split())) for prompt in prompts]),
            "legal": parse_legal_actions(str(record["actions"])),
        }

    def _member_predictions(
        self, arrays: dict[str, Any], parent: tuple[int, ...]
    ) -> np.ndarray:
        count = len(parent)
        status = np.zeros((count, 20), dtype=np.int64)
        for row, candidate in enumerate(parent):
            status[row, list(parent)] = 1
            status[row, candidate] = 2
        legal = np.asarray([
            [action in arrays["legal"] for action in ACTION_IDS]
            for _ in parent
        ])
        output = []
        for model, scaler in self.members:
            query = np.repeat(arrays["query"][None, :], count, axis=0)
            chunks = np.repeat(arrays["chunks"][None, :, :], count, axis=0)
            structured = np.repeat(
                arrays["structured"][None, :, :], count, axis=0
            )
            query, chunks, structured = scaler.arrays(
                query, chunks, structured
            )
            with torch.no_grad():
                output.append(model(
                    query_embedding=torch.tensor(query, device=self.device),
                    chunk_embeddings=torch.tensor(chunks, device=self.device),
                    chunk_structured=torch.tensor(structured, device=self.device),
                    slot_status=torch.tensor(
                        status, dtype=torch.long, device=self.device
                    ),
                    legal_action_mask=torch.tensor(
                        legal, dtype=torch.bool, device=self.device
                    ),
                ).cpu().numpy())
        return np.stack(output)
