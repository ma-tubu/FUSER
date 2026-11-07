from __future__ import annotations

import torch
from torch import nn

from .audit import FactorAuditor, ReferencePool
from .encoders import MODALITIES
from .evidence import EvidenceExtractor
from .factors import FactorBank
from .fusion import MultimodalFusion
from .refinement import SelectiveRefinement
from .reliability import PostRefinementReliability


class FUSER(nn.Module):
    """Selective factor refinement followed by post-calibrated directed fusion."""

    INPUT_FIELDS = (
        "text_features",
        "attention_mask",
        "lexical_mask",
        "audio",
        "audio_mask",
        "audio_available",
        "audio_observed_mask",
        "vision",
        "vision_mask",
        "vision_available",
        "vision_observed_mask",
    )

    def __init__(
        self,
        backbone,
        auditor: FactorAuditor | None = None,
        alpha_max: float = 0.3,
        embedding_dim: int = 4,
        gate_hidden: int = 16,
        gate_bias: float = -2.0,
        attention_dim: int = 128,
        heads: int = 4,
        fusion_hidden: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.backbone.set_phase("frozen")
        self.auditor = auditor
        self.refinement = SelectiveRefinement(alpha_max, embedding_dim, gate_hidden, gate_bias)
        self.post_reliability = PostRefinementReliability(backbone)
        self.fusion = MultimodalFusion(
            backbone.hidden_dim,
            attention_dim,
            heads,
            fusion_hidden,
            dropout,
            backbone.output_activation,
        )
        self.phase = "fusion"
        self.set_phase("fusion")

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.refinement.train(mode and self.phase in ("gate_warmup", "refinement"))
        self.post_reliability.train(mode and self.phase == "calibration")
        self.fusion.train(mode and self.phase in ("initialization", "refinement", "fusion"))
        return self

    def set_phase(self, phase: str):
        if phase not in ("initialization", "gate_warmup", "refinement", "calibration", "fusion"):
            raise ValueError("Unknown FUSER optimization phase")
        self.phase = phase
        self.requires_grad_(False)
        if phase in ("initialization", "gate_warmup", "refinement", "calibration"):
            self.post_reliability.calibration_fitted.fill_(False)
        if phase in ("gate_warmup", "refinement"):
            self.refinement.requires_grad_(True)
        if phase in ("initialization", "refinement", "fusion"):
            self.fusion.requires_grad_(True)
        elif phase == "calibration":
            self.post_reliability.heads.requires_grad_(True)
        self.train(self.training)

    @torch.no_grad()
    def complete_calibration(self):
        """Call after fitting post-refinement heads with the converged gate frozen."""
        if self.phase != "calibration":
            raise RuntimeError("Complete the calibration phase before marking its heads fitted")
        self.post_reliability.calibration_fitted.fill_(True)

    @torch.no_grad()
    def fit_diagnostic_normalization(self, audits: dict, split: str = "train"):
        self.refinement.fit_normalization(
            {m: audits[m]["diagnostics"] for m in MODALITIES},
            {m: audits[m]["eligible"] for m in MODALITIES},
            split,
        )
        self.post_reliability.calibration_fitted.fill_(False)

    @torch.no_grad()
    def audit_inputs(self, inputs: dict, raw: dict | None = None):
        if self.auditor is None:
            raise RuntimeError("Supply a fitted factor auditor or precomputed audits")
        batch = {key: inputs[key] for key in self.INPUT_FIELDS if key in inputs}
        raw = self.backbone(batch) if raw is None else raw
        size = len(raw["pred"]["text"])
        sample_ids, group_ids = inputs.get("sample_ids"), inputs.get("group_ids")
        if (
            sample_ids is None
            or group_ids is None
            or len(sample_ids) != size
            or len(group_ids) != size
        ):
            raise ValueError("Online reference exclusion requires sample_ids and source group_ids")
        audits = {}
        for modality in MODALITIES:
            rows = []
            for index in range(size):
                one = {key: value[index : index + 1] for key, value in batch.items()}
                encoded = {key: value[index] for key, value in raw["features"][modality].items()}
                x_key = "text_features" if modality == "text" else modality

                def reencode(x_cf):
                    edited = dict(one)
                    edited[x_key] = x_cf.unsqueeze(0)
                    return {
                        key: value[0]
                        for key, value in self.backbone.encode_modality(modality, edited).items()
                    }

                def predict(state):
                    return self.backbone.predict_modality(
                        modality, {"pooled": state["pooled"].unsqueeze(0)}
                    ).item()

                valid = encoded.get("lexical_mask", encoded["mask"])
                observed = one.get(f"{modality}_observed_mask")
                observed = observed[0] if observed is not None else valid
                rows.append(
                    self.auditor.audit(
                        sample_ids[index],
                        group_ids[index],
                        modality,
                        one[x_key][0],
                        valid,
                        observed,
                        encoded,
                        raw["pred"][modality][index],
                        reencode,
                        predict,
                    )
                )
            audits[modality] = {
                "reference_h": raw["features"][modality]["sequence"].detach().clone(),
                "h_cf": torch.stack([row.h_cf for row in rows]),
                "mask_h": torch.stack([row.mask_h for row in rows]),
                "diagnostics": torch.stack([row.diagnostics for row in rows]),
                "eligible": torch.tensor(
                    [row.eligible for row in rows], device=raw["pred"][modality].device
                ),
            }
        return audits

    def forward(self, inputs: dict, audits: dict | None = None):
        with torch.no_grad():
            raw = self.backbone({key: inputs[key] for key in self.INPUT_FIELDS if key in inputs})
        masks = {
            m: raw["features"][m]["mask"].bool() & raw["available"][m][:, None] for m in MODALITIES
        }
        refined, alpha = {}, {}
        if self.phase == "initialization":
            refined = {m: raw["features"][m]["sequence"] for m in MODALITIES}
            alpha = {m: raw["pred"][m].new_zeros(len(raw["pred"][m])) for m in MODALITIES}
        else:
            audits = self.audit_inputs(inputs, raw) if audits is None else audits
            for m in MODALITIES:
                h = raw["features"][m]["sequence"]
                audit = audits[m]
                reference = audit["reference_h"].to(h)
                if reference.shape != h.shape or not torch.allclose(
                    reference, h, atol=1e-5, rtol=1e-5
                ):
                    raise ValueError(
                        "Audits do not correspond to the current frozen encoder representations"
                    )
                support = audit["mask_h"].to(h.device).bool()
                if support.shape != masks[m].shape:
                    raise ValueError("Audit supports do not match this modality's sequence length")
                refined[m], alpha[m] = self.refinement(
                    m,
                    h,
                    audit["h_cf"].to(h).detach(),
                    support & masks[m],
                    audit["diagnostics"].to(h).detach(),
                    audit["eligible"].to(h.device).bool() & raw["available"][m],
                )
        post = self.post_reliability(self.backbone, raw["features"], refined)
        output = {
            "refined": refined,
            "alpha": alpha,
            "masks": masks,
            "available": raw["available"],
            "post_error": post["error"],
            "post_unimodal": post["unimodal"],
            "post_reliability": post["reliability"],
            "raw_reliability": raw["reliability"],
        }
        if self.phase != "calibration":
            if self.phase == "fusion":
                if not bool(self.post_reliability.calibration_fitted):
                    raise RuntimeError(
                        "Final fusion requires fitted post-refinement reliability heads"
                    )
                reliability = post["reliability"]
            else:
                reliability = raw["reliability"]
            output["prediction"] = self.fusion(refined, masks, reliability)
        return output

    def get_extra_state(self):
        if self.auditor is None:
            return None
        return {
            "extractor": self.auditor.extractor.config,
            "bank": self.auditor.bank.state_dict(),
            "references": self.auditor.references.state_dict(),
            "audit": self.auditor.config,
        }

    def set_extra_state(self, state):
        if state is None:
            self.auditor = None
        else:
            self.auditor = FactorAuditor(
                EvidenceExtractor(state["extractor"]),
                FactorBank(state["bank"]["config"]).load_state_dict(state["bank"]),
                ReferencePool().load_state_dict(state["references"]),
                state["audit"],
            )
