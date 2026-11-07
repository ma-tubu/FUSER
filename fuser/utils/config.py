from __future__ import annotations

import copy
import json
from pathlib import Path

from ..models import (
    FUSER,
    EvidenceExtractor,
    FactorAuditor,
    FactorBank,
    ReferencePool,
    UnimodalBackbone,
)


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    return config


def build_backbone(input_dims: dict[str, int], config: dict) -> UnimodalBackbone:
    return UnimodalBackbone(input_dims, **copy.deepcopy(config["backbone"]))


def build_model(
    backbone: UnimodalBackbone, config: dict, auditor: FactorAuditor | None = None
) -> FUSER:
    options = {**copy.deepcopy(config["refinement"]), **copy.deepcopy(config["fusion"])}
    return FUSER(backbone, auditor, **options)


def build_auditor(
    training_evidence: list, config: dict, prototype_budget: int | dict, split: str = "train"
) -> FactorAuditor:
    extractor = EvidenceExtractor(copy.deepcopy(config["evidence"]))
    bank_config = {**copy.deepcopy(config["factor_bank"]), "clusters_per_group": prototype_budget}
    bank = FactorBank(bank_config).fit(training_evidence, split)
    references = ReferencePool(copy.deepcopy(config["references"])).fit(training_evidence, split)
    return FactorAuditor(extractor, bank, references, copy.deepcopy(config["audit"]))
