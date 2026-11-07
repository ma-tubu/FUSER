from __future__ import annotations

import torch

from .records import EvidenceUnit


def _masked_mean(x: torch.Tensor, indices: list[int]) -> torch.Tensor:
    return x[indices].mean(dim=0)


def _contiguous_runs(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    runs, current = [], [indices[0]]
    for index in indices[1:]:
        if index == current[-1] + 1:
            current.append(index)
        else:
            runs.append(current)
            current = [index]
    runs.append(current)
    return runs


def _windows(run: list[int], width: int, stride: int) -> list[list[int]]:
    if width < 1 or stride < 1:
        raise ValueError("Window width and stride must be positive")
    if not run:
        return []
    if len(run) <= width:
        return [run]
    starts = list(range(0, len(run) - width + 1, stride))
    tail = len(run) - width
    if starts[-1] != tail:
        starts.append(tail)
    return [run[start : start + width] for start in starts]


class EvidenceExtractor:
    """Local multiscale descriptors and global summaries with explicit supports."""

    def __init__(self, config: dict):
        self.config = config

    def extract(
        self,
        sample_id: str,
        video_id: str | None,
        modality: str,
        x: torch.Tensor,
        valid_mask: torch.Tensor,
        observed_mask: torch.Tensor | None,
        encoded: dict[str, torch.Tensor],
        position_mapping: str | None,
    ) -> list[EvidenceUnit]:
        if x.ndim != 2 or valid_mask.ndim != 1 or len(x) != len(valid_mask):
            raise ValueError("Evidence extraction expects x=[T,D] and valid_mask=[T]")
        usable = valid_mask.bool().clone()
        if observed_mask is not None:
            usable &= observed_mask.bool()
        positions = torch.nonzero(usable, as_tuple=False).flatten().tolist()
        if not positions:
            return []
        h = encoded["sequence"]
        if position_mapping != "identity" or len(h) != len(x):
            raise ValueError("Evidence supports require verified position-preserving encoders")
        if "position_index" in encoded:
            expected = torch.arange(len(x), device=encoded["position_index"].device)
            if not torch.equal(encoded["position_index"], expected):
                raise ValueError("Encoder position indices do not match the input sequence")
        early = encoded.get("local_sequence")
        h_mask = encoded["mask"].bool()
        can_map = position_mapping == "identity" and len(h) == len(x)
        if can_map and (len(h_mask) != len(x) or not bool(h_mask[positions].all())):
            raise ValueError("Encoded mask does not cover usable input positions")
        if can_map and early is not None and len(early) != len(x):
            raise ValueError("Early feature length differs from identity mapping")
        scales = self.config["scales"][modality]
        strides = self.config.get("strides", {})
        global_x = _masked_mean(x, positions)
        global_h = _masked_mean(h, positions) if can_map else None
        global_early = _masked_mean(early, positions) if early is not None and can_map else None
        evidence, seen = [], set()
        proposals = []
        for run in _contiguous_runs(positions):
            for width in scales:
                stride = int(strides.get(modality, {}).get(str(width), max(1, int(width) // 2)))
                for window in _windows(run, int(width), stride):
                    key = tuple(window)
                    if key in seen:
                        continue
                    seen.add(key)
                    proposals.append((window, f"w{width}", "sliding"))
        for window, scale_id, origin in proposals:
            h_indices = window if can_map else None
            e_x = _masked_mean(x, window)
            e_h = _masked_mean(h, h_indices) if h_indices is not None else None
            e_early = (
                _masked_mean(early, h_indices)
                if early is not None and h_indices is not None
                else None
            )
            outside = [index for index in positions if index not in set(window)]
            context = _masked_mean(x, outside) if outside else global_x
            evidence.append(
                EvidenceUnit(
                    evidence_id=f"{sample_id}:{modality}:{scale_id}:{window[0]}-{window[-1] + 1}",
                    sample_id=sample_id,
                    video_id=video_id,
                    modality=modality,
                    input_indices=window,
                    h_indices=h_indices,
                    scale_id=scale_id,
                    origin=origin,
                    e_x=e_x.detach().cpu(),
                    e_early=e_early.detach().cpu() if e_early is not None else None,
                    e_h=e_h.detach().cpu() if e_h is not None else None,
                    r_x=(e_x - global_x).detach().cpu(),
                    r_h=(
                        (e_h - global_h).detach().cpu()
                        if e_h is not None and global_h is not None
                        else None
                    ),
                    editable_indices=window if can_map else None,
                    x_segment=x[window].detach().cpu(),
                    context_x=context.detach().cpu(),
                    meta={
                        "position_mapping": position_mapping,
                        "actual_length": len(window),
                        "usable_positions": len(positions),
                    },
                    r_early=(
                        (e_early - global_early).detach().cpu() if e_early is not None else None
                    ),
                )
            )
        if modality in self.config.get("global", {}).get("modalities", []) and self.config.get(
            "global", {}
        ).get("enabled", False):

            def descriptor(value):
                return (
                    torch.cat((value[positions].mean(0), value[positions].std(0, unbiased=False)))
                    .detach()
                    .cpu()
                )

            evidence.append(
                EvidenceUnit(
                    evidence_id=f"{sample_id}:{modality}:global",
                    sample_id=sample_id,
                    video_id=video_id,
                    modality=modality,
                    input_indices=positions,
                    h_indices=positions if can_map else None,
                    scale_id="global",
                    origin="global_statistics",
                    scope="global",
                    e_x=global_x.detach().cpu(),
                    e_early=global_early.detach().cpu() if global_early is not None else None,
                    e_h=global_h.detach().cpu() if global_h is not None else None,
                    r_x=torch.zeros_like(global_x).cpu(),
                    r_h=torch.zeros_like(global_h).cpu() if global_h is not None else None,
                    g_x=descriptor(x),
                    g_early=descriptor(early) if global_early is not None else None,
                    g_h=descriptor(h) if global_h is not None else None,
                    meta={
                        "position_mapping": position_mapping,
                        "diagnostic_only": True,
                        "actual_length": len(positions),
                        "usable_positions": len(positions),
                    },
                )
            )
        return evidence
