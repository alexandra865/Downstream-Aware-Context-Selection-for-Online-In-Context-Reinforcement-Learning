#!/usr/bin/env python3
"""Candidate-marked predictor for proposal-faithful direct-action KL."""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from h20_action_space import ACTION_IDS


class DirectActionKLModel(nn.Module):
    """Predict Full-H20-to-subset log1p KL using deletion-mask encoding."""

    def __init__(self, embedding_dim: int, structured_dim: int,
                 hidden_dim: int = 64, layers: int = 1,
                 heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.structured_dim = int(structured_dim)
        self.hidden_dim = int(hidden_dim)
        self.query_projection = nn.Linear(embedding_dim, hidden_dim)
        self.chunk_projection = nn.Linear(embedding_dim + structured_dim, hidden_dim)
        self.slot_status_embedding = nn.Embedding(3, hidden_dim)
        self.position_embedding = nn.Embedding(21, hidden_dim)
        self.legal_action_projection = nn.Linear(len(ACTION_IDS), hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=4 * hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(5 * hidden_dim)
        self.log1p_kl_raw = nn.Linear(5 * hidden_dim, 1)

    def forward(self, query_embedding: Tensor, chunk_embeddings: Tensor,
                chunk_structured: Tensor, slot_status: Tensor,
                legal_action_mask: Tensor) -> Tensor:
        batch = query_embedding.shape[0]
        if chunk_embeddings.shape != (batch, 20, self.embedding_dim):
            raise ValueError("chunk_embeddings must be [batch,20,embedding_dim]")
        if chunk_structured.shape != (batch, 20, self.structured_dim):
            raise ValueError("chunk_structured shape mismatch")
        if slot_status.shape != (batch, 20) or slot_status.dtype != torch.long:
            raise ValueError("slot_status must be int64 [batch,20]")
        if not torch.all((slot_status >= 0) & (slot_status <= 2)):
            raise ValueError("slot_status values must be 0, 1, or 2")
        if not torch.all((slot_status == 2).sum(dim=1) == 1):
            raise ValueError("Each row must have exactly one deletion candidate")
        if legal_action_mask.shape != (batch, len(ACTION_IDS)):
            raise ValueError("legal_action_mask must be [batch,5]")
        if not torch.all(legal_action_mask.any(dim=1)):
            raise ValueError("Each row must contain at least one legal action")

        parent_visible = (slot_status > 0).unsqueeze(-1).to(chunk_embeddings.dtype)
        chunk_input = torch.cat((
            chunk_embeddings * parent_visible,
            chunk_structured * parent_visible,
        ), dim=-1)
        chunks = self.chunk_projection(chunk_input)
        chunks = chunks + self.slot_status_embedding(slot_status)
        query = self.query_projection(query_embedding).unsqueeze(1)
        sequence = torch.cat((query, chunks), dim=1)
        positions = torch.arange(21, device=sequence.device).unsqueeze(0)
        encoded = self.encoder(sequence + self.position_embedding(positions))

        candidate_index = (slot_status == 2).to(torch.int64).argmax(dim=1)
        candidate = encoded[
            torch.arange(batch, device=encoded.device), candidate_index + 1
        ]
        parent = (slot_status > 0).to(encoded.dtype).unsqueeze(-1)
        child = (slot_status == 1).to(encoded.dtype).unsqueeze(-1)
        parent_mean = (encoded[:, 1:] * parent).sum(1) / parent.sum(1).clamp_min(1.0)
        child_mean = (encoded[:, 1:] * child).sum(1) / child.sum(1).clamp_min(1.0)
        legal = self.legal_action_projection(legal_action_mask.to(encoded.dtype))
        hidden = self.output_norm(torch.cat((
            encoded[:, 0], candidate, parent_mean, child_mean, legal,
        ), dim=-1))
        return F.softplus(self.log1p_kl_raw(hidden).squeeze(-1))

