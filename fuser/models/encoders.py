from __future__ import annotations

import math

import torch
from torch import nn


MODALITIES = ("text", "audio", "vision")


def zero_invalid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return x.masked_fill(~mask.unsqueeze(-1), 0.0)


def masked_sequence_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.unsqueeze(-1).to(x.dtype)
    clean = x.masked_fill(~mask.unsqueeze(-1), 0.0)
    return clean.sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TextEncoder(nn.Module):
    """Project cached linguistic features without changing token positions."""

    def __init__(self, config: dict, hidden_dim: int):
        super().__init__()
        input_dim = int(config["cached_dim"])
        dropout = float(config.get("dropout", 0.1))
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        mask = batch["attention_mask"].bool()
        lexical_mask = batch.get("lexical_mask", mask).bool() & mask
        states = batch["text_features"].masked_fill(~mask.unsqueeze(-1), 0.0)
        if states.ndim != 3 or states.shape[:2] != mask.shape:
            raise ValueError("Text features and masks must have shapes [B,T,D] and [B,T]")
        if not torch.isfinite(states).all():
            raise ValueError("Linguistic features must be finite at valid positions")
        local = zero_invalid(self.projection(states), mask)
        return {
            "sequence": local,
            "local_sequence": local,
            "pooled": masked_sequence_mean(local, lexical_mask),
            "mask": mask,
            "lexical_mask": lexical_mask,
            "available": lexical_mask.any(dim=1),
            "position_index": torch.arange(mask.shape[1], device=mask.device).expand_as(mask),
        }


class _TemporalEncoder(nn.Module):
    """Shared Conv1D + Transformer implementation used by audio and vision."""

    def __init__(self, input_dim: int, hidden_dim: int, config: dict):
        super().__init__()
        dropout = float(config.get("dropout", 0.1))
        kernel = int(config.get("conv_kernel", 3))
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("conv_kernel must be odd for length preservation")
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel, padding=kernel // 2, stride=1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        max_len = int(config.get("max_position", 1024))
        positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim)
        )
        position = torch.zeros(max_len, hidden_dim)
        position[:, 0::2] = torch.sin(positions * scale)
        position[:, 1::2] = torch.cos(positions * scale[: position[:, 1::2].shape[1]])
        self.register_buffer("position", position, persistent=False)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=int(config.get("nhead", 4)),
            dim_feedforward=int(config.get("ffn_dim", 256)),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, int(config.get("transformer_layers", 2)), enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        available: torch.Tensor,
        position_index: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        mask, available = mask.bool(), available.bool()
        if x.ndim != 3 or x.shape[:2] != mask.shape or available.shape != x.shape[:1]:
            raise ValueError("Temporal inputs, masks and availability have incompatible shapes")
        mask = mask & available[:, None]
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        if not torch.isfinite(x).all():
            raise ValueError("Temporal features must be finite at valid positions")
        batch_size, length = mask.shape
        if length > self.position.shape[0]:
            raise ValueError(
                f"Sequence length {length} exceeds positional limit {self.position.shape[0]}"
            )
        hidden_dim = self.input[0].out_features
        sequence = x.new_zeros((batch_size, length, hidden_dim))
        local_sequence = torch.zeros_like(sequence)
        active = torch.nonzero(available & mask.any(dim=1), as_tuple=False).flatten()
        if active.numel():
            active_x, active_mask = x[active], mask[active]
            h0 = zero_invalid(self.input(active_x), active_mask)
            conv = self.conv(h0.transpose(1, 2)).transpose(1, 2)
            local = zero_invalid(h0 + self.dropout(self.activation(conv)), active_mask)
            contextual_input = zero_invalid(
                local + self.position[:length].to(local.dtype), active_mask
            )
            contextual = self.transformer(contextual_input, src_key_padding_mask=~active_mask)
            sequence[active] = zero_invalid(self.output_norm(contextual), active_mask)
            local_sequence[active] = local
        effective_available = available & mask.any(dim=1)
        if position_index is None:
            position_index = torch.arange(length, device=x.device).expand(batch_size, length)
        return {
            "sequence": sequence,
            "local_sequence": local_sequence,
            "pooled": masked_sequence_mean(sequence, mask & effective_available.unsqueeze(1)),
            "mask": mask,
            "available": effective_available,
            "position_index": position_index,
        }


class AudioEncoder(_TemporalEncoder):
    """Audio Conv1D + Transformer encoder."""


class VisionEncoder(_TemporalEncoder):
    """Vision Conv1D + Transformer encoder."""
