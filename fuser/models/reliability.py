from __future__ import annotations

import copy

import torch
from torch import nn

from .encoders import MODALITIES, masked_sequence_mean


class PostRefinementReliability(nn.Module):
    """Separate error estimators calibrated on the representations used by fusion."""

    def __init__(self, backbone):
        super().__init__()
        self.heads = copy.deepcopy(backbone.error_heads)
        self.register_buffer("calibration_fitted", torch.tensor(False))

    def forward(self, backbone, encoded, refined):
        errors, reliability, predictions = {}, {}, {}
        for m in MODALITIES:
            pool_mask = encoded[m].get("lexical_mask", encoded[m]["mask"])
            pooled = masked_sequence_mean(refined[m].detach(), pool_mask)
            with torch.no_grad():
                prediction = backbone.predict_modality(m, {"pooled": pooled}).squeeze(-1)
            error = self.heads[m](pooled).sigmoid().squeeze(-1)
            errors[m], predictions[m] = error, prediction
            reliability[m] = (1.0 - error).masked_fill(~encoded[m]["available"], 0.0)
        return {"error": errors, "reliability": reliability, "unimodal": predictions}
