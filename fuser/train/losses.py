from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..models.encoders import MODALITIES


def normalized_error_targets(predictions, labels, tau_e: float):
    """Targets may be formed from training-only out-of-fold or held-out predictions."""
    if not math.isfinite(tau_e) or tau_e <= 0:
        raise ValueError("tau_e must be finite and positive")
    if labels.ndim != 1 or any(p.shape != labels.shape for p in predictions.values()):
        raise ValueError("Labels and unimodal predictions must have shape [B]")
    return {
        m: ((predictions[m].detach() - labels.detach()).abs() / tau_e).tanh() for m in MODALITIES
    }


def reliability_loss(errors, targets, available):
    losses = []
    for m in MODALITIES:
        if (
            errors[m].ndim != 1
            or targets[m].shape != errors[m].shape
            or available[m].shape != errors[m].shape
        ):
            raise ValueError("Errors, normalized targets and availability must have shape [B]")
        valid = available[m].bool()
        squared = (errors[m] - targets[m].detach()).square()
        losses.append(squared[valid].mean() if valid.any() else errors[m].sum() * 0.0)
    return torch.stack(losses).mean()


def unimodal_loss(output, labels, beta: float = 0.5):
    if not math.isfinite(beta) or beta <= 0 or labels.ndim != 1:
        raise ValueError("Use a positive Smooth-L1 beta and labels [B]")
    losses = []
    for m in MODALITIES:
        prediction = output["pred"][m]
        if prediction.shape != labels.shape:
            raise ValueError("Unimodal predictions and labels must have shape [B]")
        valid = output["available"][m].bool()
        loss = F.smooth_l1_loss(prediction, labels, beta=beta, reduction="none")
        losses.append(loss[valid].mean() if valid.any() else prediction.sum() * 0.0)
    return torch.stack(losses).mean()


def refinement_loss(output, labels, beta: float = 0.5, lambda_alpha: float = 0.001):
    if not math.isfinite(beta) or beta <= 0 or not math.isfinite(lambda_alpha) or lambda_alpha < 0:
        raise ValueError("Use positive beta and nonnegative lambda_alpha")
    if labels.ndim != 1 or output["prediction"].shape != labels.shape:
        raise ValueError("Predictions and labels must have shape [B]")
    task = F.smooth_l1_loss(output["prediction"], labels, beta=beta)
    penalty = torch.stack([output["alpha"][m].square().mean() for m in MODALITIES]).mean()
    return task + lambda_alpha * penalty
