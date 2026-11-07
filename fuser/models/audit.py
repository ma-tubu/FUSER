from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .evidence import EvidenceExtractor
from .factors import FactorBank
from .records import EvidenceUnit


@dataclass
class AuditResult:
    h_cf: torch.Tensor
    mask_h: torch.Tensor
    diagnostics: torch.Tensor
    eligible: bool
    prediction_shift: float = 0.0


def partial_intervention(x, donor_segment, indices, eta):
    if x.ndim != 2 or donor_segment.ndim != 2:
        raise ValueError("Intervention inputs must have shape [T,D]")
    if not math.isfinite(eta) or not 0 <= eta <= 1:
        raise ValueError("Intervention strength must be finite and in [0,1]")
    if not indices or len(indices) != len(set(indices)) or indices != sorted(indices):
        raise ValueError("The target support must contain ordered, unique positions")
    if indices[0] < 0 or indices[-1] >= len(x) or donor_segment.shape != x[indices].shape:
        raise ValueError("Reference features and target support are incompatible")
    x_cf = x.clone()
    donor = donor_segment.to(x)
    x_cf[indices] = x[indices] + eta * (donor - x[indices])
    mask = torch.zeros(len(x), device=x.device, dtype=torch.bool)
    mask[indices] = True
    return x_cf, mask


def structural_validity(x, x_cf, h, h_cf, support, valid_mask, score_before, score_after, config):
    """Return structural admissibility, valid-position coverage and off-target drift."""
    if x.shape != x_cf.shape or h.shape != h_cf.shape or x.ndim != 2 or h.ndim != 2:
        raise ValueError("Original and counterfactual features must have matching shapes")
    if support.shape != x.shape[:1] or valid_mask.shape != support.shape or len(h) != len(x):
        raise ValueError("Input and hidden supports require a verified identity position mapping")
    valid, support = valid_mask.bool(), support.bool()
    if not valid.any() or not support.any() or (support & ~valid).any():
        return False, 0.0, 0.0
    coverage = float(support.sum() / valid.sum())
    if not all(torch.isfinite(value).all() for value in (x, x_cf, h, h_cf)):
        return False, coverage, 0.0
    outside = ~support
    input_error = float((x_cf[outside] - x[outside]).abs().max()) if outside.any() else 0.0
    hidden_outside = valid & ~support
    drift = (
        float((h_cf[hidden_outside] - h[hidden_outside]).square().mean().sqrt())
        if hidden_outside.any()
        else 0.0
    )
    changed = bool((x_cf[support] != x[support]).any())
    target_changed = (
        score_after is not None
        and math.isfinite(score_before)
        and math.isfinite(score_after)
        and score_before - score_after > float(config.get("min_target_reduction", 0.0))
    )
    admissible = (
        changed
        and target_changed
        and input_error <= float(config.get("max_off_target_input_error", 0.0))
        and coverage <= float(config.get("max_coverage", 0.5))
        and drift <= float(config.get("max_off_target_hidden_drift", 0.25))
    )
    return bool(admissible), coverage, drift


class ReferencePool:
    """In-memory training references for context-compatible local interventions."""

    def __init__(self, config: dict | None = None):
        self.config = dict(config or {})
        self.units: dict[tuple[str, int], list[EvidenceUnit]] = {}
        self.fitted = False

    def fit(self, evidence: list[EvidenceUnit], split: str = "train"):
        if split != "train":
            raise ValueError("Reference pools must be fitted on training evidence")
        maximum = int(self.config.get("max_per_actual_length", 4096))
        if maximum < 1:
            raise ValueError("The reference-pool size must be positive")
        groups = {}
        for unit in evidence:
            if (
                unit.scope != "local"
                or unit.x_segment is None
                or unit.context_x is None
                or unit.video_id is None
            ):
                continue
            if not torch.isfinite(unit.x_segment).all() or not torch.isfinite(unit.context_x).all():
                continue
            groups.setdefault((unit.modality, len(unit.input_indices)), []).append(unit)
        self.units = {}
        for key, units in groups.items():
            units = sorted(units, key=lambda u: hashlib.sha256(u.evidence_id.encode()).digest())
            selected, seen = [], set()
            for unit in units:
                if unit.sample_id not in seen:
                    selected.append(unit)
                    seen.add(unit.sample_id)
                if len(selected) == maximum:
                    break
            self.units[key] = selected
        self.fitted = True
        return self

    def state_dict(self):
        return {
            "config": dict(self.config),
            "fitted": self.fitted,
            "groups": [
                {"modality": m, "length": n, "units": [asdict(u) for u in units]}
                for (m, n), units in self.units.items()
            ],
        }

    def load_state_dict(self, state):
        self.config = dict(state["config"])
        self.fitted = bool(state["fitted"])
        self.units = {
            (group["modality"], group["length"]): [EvidenceUnit(**u) for u in group["units"]]
            for group in state["groups"]
        }
        return self

    def retrieve(self, target, entry, bank, x, valid_mask):
        if not self.fitted:
            raise RuntimeError("Fit the training reference pool before auditing")
        if target.video_id is None or target.context_x is None:
            return None
        weak_margin = float(self.config.get("weak_margin", 0.05))
        top_k = int(self.config.get("boundary_top_k", 16))
        boundary_weight = float(self.config.get("boundary_weight", 0.1))
        if top_k < 1 or weak_margin < 0 or boundary_weight < 0:
            raise ValueError("Invalid reference-retrieval parameters")
        candidates = [
            unit
            for unit in self.units.get((target.modality, len(target.input_indices)), [])
            if unit.sample_id != target.sample_id
            and unit.video_id != target.video_id
            and bank.score(entry, unit) < entry.presence_threshold - weak_margin
        ]
        if not candidates:
            return None
        context = target.context_x.float().cpu()
        similarities = torch.tensor(
            [
                float(F.cosine_similarity(unit.context_x.float().cpu(), context, dim=0, eps=1e-8))
                for unit in candidates
            ]
        )
        indices = similarities.topk(min(top_k, len(candidates))).indices.tolist()
        best, best_score = None, -float("inf")
        first, last = target.input_indices[0], target.input_indices[-1]
        for index in indices:
            unit = candidates[index]
            donor = unit.x_segment.to(x)
            differences = []
            if first > 0 and valid_mask[first - 1]:
                differences.append(torch.linalg.vector_norm(donor[0] - x[first - 1]))
            if last + 1 < len(x) and valid_mask[last + 1]:
                differences.append(torch.linalg.vector_norm(donor[-1] - x[last + 1]))
            boundary = float(torch.stack(differences).mean()) if differences else 0.0
            score = float(similarities[index]) - boundary_weight * boundary
            if score > best_score:
                best, best_score = unit, score
        return best


class FactorAuditor:
    """Frozen-encoder, label-free utilization auditing for one factor per modality."""

    def __init__(
        self,
        extractor: EvidenceExtractor,
        bank: FactorBank,
        references: ReferencePool,
        config: dict,
    ):
        self.extractor, self.bank, self.references = extractor, bank, references
        self.config = dict(config)
        tau = float(config.get("tau_u", 0.5))
        eta = float(config.get("eta", 0.2))
        if not math.isfinite(tau) or tau <= 0 or not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("Auditing requires positive tau_u and eta in [0,1]")

    @torch.no_grad()
    def audit(
        self,
        sample_id,
        video_id,
        modality,
        x,
        valid_mask,
        observed_mask,
        encoded,
        prediction,
        reencode,
        predict,
    ):
        h = encoded["sequence"]
        result = AuditResult(
            h.clone(), torch.zeros(len(h), device=h.device, dtype=torch.bool), h.new_zeros(4), False
        )
        if not self.bank.fitted or not self.references.fitted:
            raise RuntimeError("Fit the factor bank and training references before auditing")
        if len(x) != len(h):
            raise ValueError("This encoder must preserve positions within each modality")
        if not encoded["available"] or not torch.isfinite(h).all():
            return result
        units = self.extractor.extract(
            sample_id, video_id, modality, x, valid_mask, observed_mask, encoded, "identity"
        )
        matches = [
            row
            for row in self.bank.match(units)
            if row.status == "present" and row.scope == "local"
        ]
        if not matches:
            return result
        match = max(matches, key=lambda row: row.score)
        target = next(unit for unit in units if unit.evidence_id == match.evidence_id)
        entry = self.bank.entry(match.factor_id)
        donor = self.references.retrieve(target, entry, self.bank, x, valid_mask)
        if donor is None:
            return result
        x_cf, mask = partial_intervention(
            x, donor.x_segment, target.input_indices, float(self.config.get("eta", 0.2))
        )
        cf_encoded = reencode(x_cf)
        cf_units = self.extractor.extract(
            sample_id, video_id, modality, x_cf, valid_mask, observed_mask, cf_encoded, "identity"
        )
        same = next(
            (
                unit
                for unit in cf_units
                if unit.scope == "local"
                and unit.scale_id == target.scale_id
                and unit.input_indices == target.input_indices
            ),
            None,
        )
        score_after = self.bank.score(entry, same) if same is not None else None
        admissible, coverage, drift = structural_validity(
            x,
            x_cf,
            h,
            cf_encoded["sequence"],
            mask,
            valid_mask,
            match.score,
            score_after,
            self.config.get("validity", {}),
        )
        prediction_cf = float(predict(cf_encoded))
        shift = prediction_cf - float(prediction)
        if not admissible or not math.isfinite(shift):
            return result
        utilization = math.tanh(abs(shift) / float(self.config.get("tau_u", 0.5)))
        return AuditResult(
            cf_encoded["sequence"].detach(),
            mask,
            h.new_tensor([utilization, match.score, coverage, drift]),
            True,
            shift,
        )
