from __future__ import annotations

import math

import torch

from ..models import FUSER, UnimodalBackbone
from ..utils.tensors import move_to_device
from .losses import normalized_error_targets, refinement_loss, reliability_loss, unimodal_loss


class Trainer:
    """Optimize one selected phase using caller-supplied feature batches."""

    def __init__(self, model: UnimodalBackbone | FUSER, optimizer, config: dict):
        self.model, self.optimizer = model, optimizer
        self.phase = model.phase
        self.loss_config = dict(config["loss"])
        self.accumulation_steps = int(config.get("accumulation_steps", 1))
        self.max_grad_norm = float(config.get("max_grad_norm", 1.0))
        if (
            self.accumulation_steps < 1
            or not math.isfinite(self.max_grad_norm)
            or self.max_grad_norm <= 0
        ):
            raise ValueError("Accumulation steps and gradient clipping must be positive")
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        trainable = {id(p) for p in model.parameters() if p.requires_grad}
        if not trainable or optimized != trainable:
            raise ValueError(
                "The optimizer must contain exactly the current phase's trainable parameters"
            )
        self.pending_batches = 0
        self.pending_samples = 0
        self.optimizer.zero_grad(set_to_none=True)

    def _check_phase(self):
        if self.model.phase != self.phase:
            raise RuntimeError("Create a new optimizer and Trainer after changing model phase")

    def train_batch(
        self,
        inputs: dict,
        labels: torch.Tensor,
        audits: dict | None = None,
        error_targets: dict | None = None,
    ) -> dict:
        self._check_phase()
        if labels.ndim != 1 or not len(labels) or not torch.isfinite(labels).all():
            raise ValueError("Training labels must be a nonempty finite tensor [B]")
        if error_targets is not None and self.phase not in ("error", "calibration"):
            raise ValueError("External normalized error targets are only used for error estimation")
        device = next(self.model.parameters()).device
        inputs, labels = move_to_device(inputs, device), labels.to(device)
        audits = move_to_device(audits, device)
        error_targets = move_to_device(error_targets, device)
        self.model.train()
        beta = float(self.loss_config.get("beta", 0.5))
        tau_e = float(self.loss_config.get("tau_e", 1.0))
        if isinstance(self.model, UnimodalBackbone):
            output = self.model(inputs)
            if self.phase == "base":
                loss = unimodal_loss(output, labels, beta)
            elif self.phase == "error":
                targets = (
                    normalized_error_targets(output["pred"], labels, tau_e)
                    if error_targets is None
                    else error_targets
                )
                loss = reliability_loss(output["error"], targets, output["available"])
            else:
                raise ValueError("The frozen backbone cannot be optimized")
        else:
            output = self.model(inputs, audits)
            if self.phase == "calibration":
                targets = (
                    normalized_error_targets(output["post_unimodal"], labels, tau_e)
                    if error_targets is None
                    else error_targets
                )
                loss = reliability_loss(output["post_error"], targets, output["available"])
            else:
                loss = refinement_loss(
                    output, labels, beta, float(self.loss_config.get("lambda_alpha", 0.001))
                )
        if not torch.isfinite(loss):
            raise FloatingPointError("The training objective is not finite")
        size = len(labels)
        (loss * size).backward()
        self.pending_batches += 1
        self.pending_samples += size
        stepped = self.flush() if self.pending_batches == self.accumulation_steps else False
        return {"loss": float(loss.detach()), "samples": size, "optimizer_step": stepped}

    def flush(self) -> bool:
        """Apply accumulated gradients, including a partial final accumulation window."""
        self._check_phase()
        if not self.pending_batches:
            return False
        parameters = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        for parameter in parameters:
            parameter.grad.div_(self.pending_samples)
        torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm, error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending_batches = self.pending_samples = 0
        return True
