from __future__ import annotations

import math

import torch

from ..models import FUSER, UnimodalBackbone


def build_optimizer(model: UnimodalBackbone | FUSER, config: dict):
    """Build AdamW for the currently selected optimization phase."""
    phase = model.phase
    rates = config["learning_rates"][phase]
    weight_decay = float(config["weight_decay"])
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Weight decay must be finite and nonnegative")
    if isinstance(model, UnimodalBackbone) and phase == "base":
        encoders = [model.text_encoder, model.audio_encoder, model.vision_encoder]
        groups = [
            {
                "params": [
                    p for module in encoders for p in module.parameters() if p.requires_grad
                ],
                "lr": float(rates["encoders"]),
            },
            {
                "params": [p for p in model.pred_heads.parameters() if p.requires_grad],
                "lr": float(rates["predictors"]),
            },
        ]
    else:
        groups = [
            {"params": [p for p in model.parameters() if p.requires_grad], "lr": float(rates)}
        ]
    if any(not group["params"] for group in groups):
        raise ValueError("Select a trainable model phase before creating the optimizer")
    if any(not math.isfinite(group["lr"]) or group["lr"] <= 0 for group in groups):
        raise ValueError("Learning rates must be finite and positive")
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
