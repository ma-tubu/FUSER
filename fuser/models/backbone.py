from __future__ import annotations

import torch
from torch import nn

from .encoders import AudioEncoder, MODALITIES, MLPHead, TextEncoder, VisionEncoder


class UnimodalBackbone(nn.Module):
    """Linguistic, acoustic and visual encoders with independent sentiment and error heads."""

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden_dim: int = 128,
        dropout: float = 0.1,
        predictor_hidden: int = 64,
        error_hidden: int = 64,
        layers: int = 2,
        heads: int = 4,
        ffn_dim: int = 256,
        conv_kernel: int = 3,
        max_position: int = 1024,
        output_activation: str = "identity",
    ):
        super().__init__()
        if set(input_dims) != set(MODALITIES) or any(int(d) < 1 for d in input_dims.values()):
            raise ValueError("Provide positive input dimensions for text, audio and vision")
        if hidden_dim < 1 or heads < 1 or hidden_dim % heads or layers < 1 or max_position < 1:
            raise ValueError("Invalid hidden dimension, attention heads, layers or position limit")
        if output_activation not in ("identity", "tanh"):
            raise ValueError("Prediction activation must be identity or tanh")
        self.hidden_dim = int(hidden_dim)
        self.input_dims = {m: int(input_dims[m]) for m in MODALITIES}
        self.output_activation = output_activation
        self.phase = "base"
        self.text_encoder = TextEncoder(
            {"cached_dim": input_dims["text"], "dropout": dropout}, hidden_dim
        )
        av = {
            "dropout": dropout,
            "transformer_layers": layers,
            "nhead": heads,
            "ffn_dim": ffn_dim,
            "conv_kernel": conv_kernel,
            "max_position": max_position,
        }
        self.audio_encoder = AudioEncoder(input_dims["audio"], hidden_dim, av)
        self.vision_encoder = VisionEncoder(input_dims["vision"], hidden_dim, av)
        self.pred_heads = nn.ModuleDict(
            {m: MLPHead(hidden_dim, predictor_hidden, dropout=dropout) for m in MODALITIES}
        )
        self.error_heads = nn.ModuleDict(
            {m: MLPHead(hidden_dim, error_hidden, dropout=dropout) for m in MODALITIES}
        )
        if output_activation == "tanh":
            for head in self.pred_heads.values():
                head.net.append(nn.Tanh())
        self.set_phase("base")

    def train(self, mode: bool = True):
        super().train(mode)
        for encoder in (
            self.text_encoder,
            self.audio_encoder,
            self.vision_encoder,
            self.pred_heads,
        ):
            encoder.train(mode and self.phase == "base")
        self.error_heads.train(mode and self.phase == "error")
        return self

    def set_phase(self, phase: str):
        if phase not in ("base", "error", "frozen"):
            raise ValueError("Backbone phase must be base, error or frozen")
        self.phase = phase
        self.requires_grad_(False)
        if phase == "base":
            for module in (
                self.text_encoder,
                self.audio_encoder,
                self.vision_encoder,
                self.pred_heads,
            ):
                module.requires_grad_(True)
        elif phase == "error":
            self.error_heads.requires_grad_(True)
        self.train(self.training)

    def encode_modality(self, modality, inputs):
        if modality not in MODALITIES:
            raise ValueError("Unknown modality")
        if modality == "text":
            return self.text_encoder(inputs)
        x, mask = inputs[modality], inputs[f"{modality}_mask"].bool()
        available = inputs.get(f"{modality}_available", mask.any(1)).bool()
        encoder = self.audio_encoder if modality == "audio" else self.vision_encoder
        return encoder(x, mask, available)

    def predict_modality(self, modality, encoded):
        return self.pred_heads[modality](encoded["pooled"])

    def forward(self, inputs):
        encoded = {m: self.encode_modality(m, inputs) for m in MODALITIES}
        pred, err, reliability = {}, {}, {}
        for m, state in encoded.items():
            pred[m] = self.predict_modality(m, state).squeeze(-1)
            err[m] = self.error_heads[m](state["pooled"].detach()).sigmoid().squeeze(-1)
            reliability[m] = (1.0 - err[m]).masked_fill(~state["available"], 0.0)
        return {
            "features": encoded,
            "pred": pred,
            "error": err,
            "reliability": reliability,
            "available": {m: encoded[m]["available"] for m in MODALITIES},
        }
