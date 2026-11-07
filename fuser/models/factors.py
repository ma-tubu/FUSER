from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .records import EvidenceUnit, FactorEntry, FactorMatch


def _setting(value, modality, default):
    return value.get(modality, value.get("default", default)) if isinstance(value, dict) else value


def _feature(unit: EvidenceUnit, source: str) -> torch.Tensor:
    if unit.scope == "global":
        value = {"X": unit.g_x, "early": unit.g_early, "H": unit.g_h}.get(source)
    else:
        value = {"X": unit.e_x, "early": unit.e_early, "H": unit.e_h}.get(source)
    if value is None:
        raise ValueError(f"Evidence does not contain the requested {source} descriptor")
    return value.detach().float().cpu()


@dataclass
class FeatureNormalizer:
    mean: torch.Tensor
    std: torch.Tensor

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        values = values.detach().float().cpu()
        return F.normalize((values - self.mean) / self.std, dim=-1, eps=1e-8)


def _kmedoids(values: torch.Tensor, k: int, seed: int, iterations: int):
    """Cosine-distance alternating K-medoids with observed representatives."""
    n = len(values)
    k = min(k, n)
    distance = (1.0 - values @ values.T).clamp_min(0.0)
    medoids = [random.Random(seed).randrange(n)]
    while len(medoids) < k:
        nearest = distance[:, medoids].min(dim=1).values
        nearest[medoids] = -1.0
        medoids.append(int(nearest.argmax()))
    for _ in range(iterations):
        assignment = distance[:, medoids].argmin(dim=1)
        updated = []
        for cluster in range(k):
            members = (assignment == cluster).nonzero(as_tuple=False).flatten()
            if not len(members):
                updated.append(medoids[cluster])
            else:
                costs = distance[members][:, members].sum(dim=1)
                updated.append(int(members[costs.argmin()]))
        if updated == medoids:
            break
        medoids = updated
    return medoids


class FactorBank:
    """Training-only prototype banks, grouped by modality and evidence scope."""

    def __init__(self, config: dict):
        self.config = dict(config)
        self.entries: list[FactorEntry] = []
        self.normalizers: dict[str, FeatureNormalizer] = {}
        self.fitted = False

    @staticmethod
    def _key(modality, scope, source):
        return f"{modality}:{scope}:{source}"

    def fit(self, evidence: list[EvidenceUnit], split: str = "train"):
        if split != "train":
            raise ValueError("Factor banks must be fitted on training evidence")
        if "clusters_per_group" not in self.config:
            raise ValueError("Specify the prototype budget with clusters_per_group")
        quantile = float(self.config.get("presence_similarity_quantile", 0.1))
        cap = int(self.config.get("max_windows_per_sample", 4))
        maximum = int(self.config.get("max_fit_evidence", 5000))
        iterations = int(self.config.get("kmedoids_iterations", 20))
        minimum = int(self.config.get("min_unique_sample_support", 5))
        min_videos = int(self.config.get("min_unique_video_support", 2))
        if not 0 <= quantile <= 1 or min(cap, maximum, iterations, minimum) < 1 or min_videos < 0:
            raise ValueError("Invalid factor-bank support, quantile or fitting budget")
        groups = {}
        for unit in evidence:
            selected = self.config.get("selected_scales", {}).get(unit.modality)
            if selected is not None and unit.scope == "local" and unit.scale_id not in selected:
                continue
            groups.setdefault((unit.modality, unit.scope), []).append(unit)
        self.entries, self.normalizers, self.fitted = [], {}, False
        for offset, ((modality, scope), units) in enumerate(sorted(groups.items())):
            source = _setting(self.config.get("feature_source", "early"), modality, "early")
            k = _setting(self.config["clusters_per_group"], modality, None)
            if k is None:
                raise ValueError("Specify a prototype budget for every fitted modality")
            k = int(k)
            if k < 1:
                raise ValueError("The number of prototypes must be positive")
            raw = torch.stack([_feature(unit, source) for unit in units])
            if not torch.isfinite(raw).all():
                raise ValueError("Factor descriptors must be finite")
            key = self._key(modality, scope, source)
            normalizer = FeatureNormalizer(raw.mean(0), raw.std(0, unbiased=False).clamp_min(1e-6))
            self.normalizers[key] = normalizer
            values = normalizer.transform(raw)
            by_sample = {}
            for index, unit in enumerate(units):
                by_sample.setdefault(unit.sample_id, []).append(index)
            rng = random.Random(int(self.config.get("seed", 0)) + offset)
            indices = []
            for choices in by_sample.values():
                rng.shuffle(choices)
                indices.extend(choices[:cap])
            rng.shuffle(indices)
            indices = sorted(indices[:maximum])
            medoid_ids = _kmedoids(
                values[indices], k, int(self.config.get("seed", 0)) + offset, iterations
            )
            medoids = values[[indices[i] for i in medoid_ids]]
            assignment = (values @ medoids.T).argmax(dim=1)
            threshold = _setting(self.config.get("presence_threshold", {}), modality, None)
            if threshold is None:
                threshold = float(torch.quantile((values @ medoids.T).max(dim=1).values, quantile))
            threshold = float(threshold)
            if not -1 <= threshold <= 1:
                raise ValueError("Cosine matching thresholds must be in [-1,1]")
            for cluster, medoid in enumerate(medoids):
                members = (assignment == cluster).nonzero(as_tuple=False).flatten().tolist()
                samples = {units[i].sample_id for i in members}
                videos = {units[i].video_id for i in members if units[i].video_id is not None}
                if not members or len(samples) < minimum or len(videos) < min_videos:
                    continue
                self.entries.append(
                    FactorEntry(
                        factor_id=f"{modality}:{scope}:{cluster}",
                        modality=modality,
                        scope=scope,
                        feature_source=source,
                        scale_id=None,
                        medoid=medoid.clone(),
                        representative_ids=[units[indices[medoid_ids[cluster]]].evidence_id],
                        unique_sample_support=len(samples),
                        unique_video_support=len(videos),
                        normalization_id=key,
                        presence_threshold=threshold,
                    )
                )
        self.fitted = True
        return self

    def score(self, entry: FactorEntry, unit: EvidenceUnit) -> float:
        if not self.fitted:
            raise RuntimeError("Fit the factor bank before scoring evidence")
        key = self._key(entry.modality, entry.scope, entry.feature_source)
        return float(
            self.normalizers[key].transform(_feature(unit, entry.feature_source)) @ entry.medoid
        )

    def match(self, evidence: list[EvidenceUnit]) -> list[FactorMatch]:
        if not self.fitted:
            raise RuntimeError("Fit the factor bank before matching evidence")
        if len({unit.sample_id for unit in evidence}) > 1:
            raise ValueError("Factor localization expects evidence from one sample")
        matches = []
        for entry in self.entries:
            candidates = [
                u for u in evidence if u.modality == entry.modality and u.scope == entry.scope
            ]
            if not candidates:
                continue
            key = self._key(entry.modality, entry.scope, entry.feature_source)
            values = self.normalizers[key].transform(
                torch.stack([_feature(u, entry.feature_source) for u in candidates])
            )
            scores = values @ entry.medoid
            best = int(scores.argmax())
            unit, score = candidates[best], float(scores[best])
            matches.append(
                FactorMatch(
                    unit.sample_id,
                    entry.factor_id,
                    entry.modality,
                    entry.scope,
                    "present" if score >= entry.presence_threshold else "absent",
                    score,
                    unit.input_indices,
                    unit.h_indices,
                    unit.evidence_id,
                )
            )
        return sorted(matches, key=lambda row: (row.status != "present", -row.score, row.factor_id))

    def entry(self, factor_id: str) -> FactorEntry:
        return next(entry for entry in self.entries if entry.factor_id == factor_id)

    def state_dict(self) -> dict:
        return {
            "config": dict(self.config),
            "entries": [asdict(entry) for entry in self.entries],
            "normalizers": {key: asdict(value) for key, value in self.normalizers.items()},
            "fitted": self.fitted,
        }

    def load_state_dict(self, state: dict):
        self.config = dict(state["config"])
        self.entries = [FactorEntry(**value) for value in state["entries"]]
        self.normalizers = {
            key: FeatureNormalizer(**value) for key, value in state["normalizers"].items()
        }
        self.fitted = bool(state["fitted"])
        return self
