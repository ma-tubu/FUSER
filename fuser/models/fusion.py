from __future__ import annotations

import torch
from torch import nn

from .attention import DIRECTIONS, SixDirectedCrossAttention
from .encoders import MODALITIES, masked_sequence_mean


class MultimodalFusion(nn.Module):
    """Fuse three unimodal residuals and six reliability-weighted interactions."""

    def __init__(
        self,
        input_dim: int = 128,
        attention_dim: int = 128,
        heads: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        output_activation: str = "identity",
    ):
        super().__init__()
        if output_activation not in ("identity", "tanh"):
            raise ValueError("Prediction activation must be identity or tanh")
        self.attention = SixDirectedCrossAttention(input_dim, attention_dim, heads, dropout)
        self.residual = nn.ModuleDict(
            {
                m: (
                    nn.Identity()
                    if input_dim == attention_dim
                    else nn.Linear(input_dim, attention_dim, bias=False)
                )
                for m in MODALITIES
            }
        )
        self.regressor = nn.Sequential(
            nn.Linear(9 * attention_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        if output_activation == "tanh":
            self.regressor.append(nn.Tanh())

    def forward(self, refined, masks, reliability):
        branches = self.attention(refined, masks, reliability)
        vectors = [self.residual[m](masked_sequence_mean(refined[m], masks[m])) for m in MODALITIES]
        vectors.extend(
            masked_sequence_mean(branches[f"{target}_from_{source}"], masks[target])
            for target, source in DIRECTIONS
        )
        return self.regressor(torch.cat(vectors, -1)).squeeze(-1)
