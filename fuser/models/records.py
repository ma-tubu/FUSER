from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EvidenceUnit:
    evidence_id: str
    sample_id: str
    video_id: str | None
    modality: str
    input_indices: list[int]
    h_indices: list[int] | None
    scale_id: str
    origin: str
    e_x: torch.Tensor
    e_early: torch.Tensor | None
    e_h: torch.Tensor | None
    r_x: torch.Tensor
    r_h: torch.Tensor | None
    editable_indices: list[int] | None = None
    x_segment: torch.Tensor | None = None
    context_x: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)
    scope: str = "local"
    r_early: torch.Tensor | None = None
    g_x: torch.Tensor | None = None
    g_early: torch.Tensor | None = None
    g_h: torch.Tensor | None = None


@dataclass
class FactorEntry:
    factor_id: str
    modality: str
    scope: str
    feature_source: str
    scale_id: str | None
    medoid: torch.Tensor
    representative_ids: list[str]
    unique_sample_support: int
    unique_video_support: int
    normalization_id: str
    presence_threshold: float
    association_score: float | None = None
    audit_metadata: dict = field(default_factory=dict)


@dataclass
class FactorMatch:
    sample_id: str
    factor_id: str
    modality: str
    scope: str
    status: str
    score: float
    input_indices: list[int] | None
    h_indices: list[int] | None
    evidence_id: str | None
