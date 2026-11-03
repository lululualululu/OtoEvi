"""Multi-depth Visual Memory and the causal report decoder (Eqs. 2-3)."""

import math

import torch
from torch.nn import functional as F

from ._initialization import SemanticCaptioner as _CanonicalInitialization


class VisualMemoryDecoder(_CanonicalInitialization):
    """50 base tokens plus eight multi-depth evidence tokens; no text prefix."""

    @staticmethod
    def _latent_structure(tokens: torch.Tensor) -> torch.Tensor:
        batch, count, width = tokens.shape
        side = int(math.sqrt(count))
        if side * side != count:
            return tokens - tokens.mean(dim=1, keepdim=True)
        grid = tokens.transpose(1, 2).reshape(batch, width, side, side)
        neighbourhood = F.avg_pool2d(
            grid, kernel_size=3, stride=1, padding=1, count_include_pad=False
        )
        return (grid - neighbourhood).flatten(2).transpose(1, 2)

    def _structured_multiscale_tokens(self, features: dict) -> torch.Tensor:
        raw = features["multiscale"]
        if raw.shape[1] not in {1, 4}:
            raise ValueError("expected one or four visual feature levels")
        levels = []
        for index in range(raw.shape[1]):
            projection = (
                self.visual_proj
                if raw.shape[1] == 1 or index == 3
                else self.early_scale_proj[index]
            )
            level = projection(raw[:, index])
            level = level + self.spatial_pos.unsqueeze(0)
            scale_index = 3 if raw.shape[1] == 1 else index
            level = level + self.scale_embedding[scale_index].view(1, 1, -1)
            structure = self._latent_structure(level)
            structure = self.structure_projection(self.structure_norm(structure))
            level = level + torch.sigmoid(self.structure_gate) * structure
            levels.append(level)
        queries = F.normalize(self.evidence_queries, dim=-1)
        selected = []
        keep = max(1, min(self.retained_per_scale, levels[0].shape[1]))
        for level in levels:
            relevance = torch.einsum(
                "bnd,qd->bnq", F.normalize(level, dim=-1), queries
            ).amax(dim=-1)
            indices = relevance.topk(keep, dim=1).indices
            batch_index = torch.arange(level.shape[0], device=level.device)[:, None]
            selected.append(level[batch_index, indices])
        return torch.cat(selected, dim=1)

    def build_memory(self, features):
        global_token = self.global_proj(features["global"])[:, None]
        if not self.use_memory:
            return global_token, None
        patches = (
            self.visual_proj(features["multiscale"][:, -1]) + self.spatial_pos[None]
        )
        base = self.encoder(torch.cat((global_token, patches), 1))
        visual = self._structured_multiscale_tokens(features)
        queries = self.evidence_queries[None].expand(len(visual), -1, -1)
        queries = queries + self.query_from_global(features["global"])[:, None]
        evidence, _ = self.evidence_attention(
            queries, visual, visual, need_weights=False
        )
        evidence = self.evidence_norm(queries + evidence)
        evidence = self.evidence_norm(evidence + self.evidence_ffn(evidence))
        evidence = torch.sigmoid(self.evidence_gate) * evidence
        return torch.cat((base, evidence), 1), None
