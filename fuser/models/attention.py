from __future__ import annotations

import math

import torch
from torch import nn

from .encoders import MODALITIES


DIRECTIONS = (
    ("text", "audio"),
    ("audio", "text"),
    ("text", "vision"),
    ("vision", "text"),
    ("audio", "vision"),
    ("vision", "audio"),
)


class SixDirectedCrossAttention(nn.Module):
    def __init__(self, input_dim: int, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        if heads < 1 or dim < 1 or dim % heads:
            raise ValueError("attention.dim must be positive and divisible by heads")
        self.dim, self.heads = dim, heads
        self.query = nn.ModuleDict({m: nn.Linear(input_dim, dim, bias=False) for m in MODALITIES})
        self.key = nn.ModuleDict({m: nn.Linear(input_dim, dim, bias=False) for m in MODALITIES})
        self.value = nn.ModuleDict({m: nn.Linear(input_dim, dim, bias=False) for m in MODALITIES})
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.heads, self.dim // self.heads).transpose(1, 2)

    def forward(self, sequences: dict, masks: dict, reliability: dict) -> dict:
        clean = {m: sequences[m].masked_fill(~masks[m].bool().unsqueeze(-1), 0) for m in MODALITIES}
        q = {m: self._split_heads(self.query[m](clean[m])) for m in MODALITIES}
        k = {m: self._split_heads(self.key[m](clean[m])) for m in MODALITIES}
        v = {m: self._split_heads(self.value[m](clean[m])) for m in MODALITIES}
        result = {}
        for target, source in DIRECTIONS:
            batch_size, length = sequences[target].shape[:2]
            out = sequences[target].new_zeros((batch_size, length, self.dim))
            active = masks[source].any(1) & masks[target].any(1)
            if active.any():
                logits = q[target][active] @ k[source][active].transpose(-1, -2)
                logits = logits / math.sqrt(self.dim // self.heads)
                logits = logits.masked_fill(
                    ~masks[source][active, None, None, :].bool(), float("-inf")
                )
                attention = self.dropout(torch.softmax(logits, dim=-1))
                z = (attention @ v[source][active]).transpose(1, 2).reshape(-1, length, self.dim)
                out[active] = z
            r = reliability[source]
            if r.shape not in ((batch_size,), (batch_size, 1)):
                raise ValueError("Source reliability must be [B] or [B,1]")
            if not torch.isfinite(r).all() or (r < 0).any() or (r > 1).any():
                raise ValueError("Source reliability must be finite in [0,1]")
            out = out * r.reshape(batch_size, 1, 1)
            result[f"{target}_from_{source}"] = out.masked_fill(
                ~masks[target].bool().unsqueeze(-1), 0
            )
        return result
