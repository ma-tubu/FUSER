from __future__ import annotations

import math

import torch
from torch import nn

from .encoders import MODALITIES


class SelectiveRefinement(nn.Module):
    """Bounded residual refinement from four audit diagnostics and a modality embedding."""

    def __init__(
        self,
        alpha_max: float = 0.3,
        embedding_dim: int = 4,
        hidden_dim: int = 16,
        initial_bias: float = -2.0,
    ):
        super().__init__()
        if not math.isfinite(alpha_max) or not 0 <= alpha_max <= 1:
            raise ValueError("alpha_max must be finite and in [0,1]")
        self.alpha_max = float(alpha_max)
        self.embedding = nn.Embedding(len(MODALITIES), embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(4 + embedding_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        nn.init.constant_(self.network[-1].bias, float(initial_bias))
        self.register_buffer("diagnostic_mean", torch.zeros(len(MODALITIES), 4))
        self.register_buffer("diagnostic_std", torch.ones(len(MODALITIES), 4))

    @torch.no_grad()
    def fit_normalization(self, diagnostics: dict, eligibility: dict, split: str = "train"):
        if split != "train":
            raise ValueError("Diagnostic normalization must use training audits")
        for index, modality in enumerate(MODALITIES):
            values = diagnostics[modality].to(self.diagnostic_mean)
            if (
                values.ndim != 2
                or values.shape[-1] != 4
                or eligibility[modality].shape != values.shape[:1]
            ):
                raise ValueError("Normalization expects diagnostics [B,4] and eligibility [B]")
            valid = eligibility[modality].bool().to(values.device) & torch.isfinite(values).all(1)
            self.diagnostic_mean[index].zero_()
            self.diagnostic_std[index].fill_(1.0)
            if valid.any():
                self.diagnostic_mean[index].copy_(values[valid].mean(0))
                self.diagnostic_std[index].copy_(
                    values[valid].std(0, unbiased=False).clamp_min(1e-6)
                )

    def forward(self, modality, h, h_cf, support, diagnostics, eligible):
        if modality not in MODALITIES or h.ndim != 3 or h_cf.shape != h.shape:
            raise ValueError("Refinement expects a known modality and matching [B,T,D] features")
        if (
            support.shape != h.shape[:2]
            or diagnostics.shape != (len(h), 4)
            or eligible.shape != h.shape[:1]
        ):
            raise ValueError("Refinement supports or audit diagnostics have incompatible shapes")
        index = MODALITIES.index(modality)
        usable = eligible.bool() & support.bool().any(1) & torch.isfinite(diagnostics).all(1)
        usable &= torch.isfinite(h_cf).flatten(1).all(1)
        q = torch.where(usable[:, None], diagnostics, self.diagnostic_mean[index])
        q = (q - self.diagnostic_mean[index]) / self.diagnostic_std[index]
        indices = torch.full((len(h),), index, dtype=torch.long, device=h.device)
        logits = self.network(torch.cat((q, self.embedding(indices)), -1)).squeeze(-1)
        alpha = torch.where(usable, self.alpha_max * logits.sigmoid(), torch.zeros_like(logits))
        active = support.bool() & usable[:, None]
        difference = torch.where(active[:, :, None], h_cf - h, torch.zeros_like(h))
        refined = h + alpha[:, None, None] * difference
        return refined, alpha
