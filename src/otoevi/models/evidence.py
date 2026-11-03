"""Per-view evidence slots and sparse causal reads."""

from __future__ import annotations

import math

import torch
from torch import nn


def sparsemax(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Projection onto the probability simplex; exact zeros without Top-k."""
    shifted = values - values.amax(dim=dim, keepdim=True)
    ordered, _ = shifted.sort(dim=dim, descending=True)
    count = values.shape[dim]
    ranks = torch.arange(1, count + 1, device=values.device, dtype=values.dtype)
    shape = [1] * values.ndim
    shape[dim] = count
    ranks = ranks.reshape(shape)
    cumulative = ordered.cumsum(dim)
    support = 1 + ranks * ordered > cumulative
    support_size = support.sum(dim=dim, keepdim=True).clamp_min(1)
    threshold = (cumulative.gather(dim, support_size - 1) - 1) / support_size.to(
        values.dtype
    )
    return (shifted - threshold).clamp_min(0)


class EvidenceSlotResampler(nn.Module):
    def __init__(self, width: int = 256, slots: int = 12, heads: int = 4):
        super().__init__()
        self.slots = slots
        self.queries = nn.Parameter(torch.randn(slots, width) / math.sqrt(width))
        self.global_condition = nn.Linear(512, width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.norm = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )

    def forward(
        self,
        tokens: torch.Tensor,
        token_valid: torch.Tensor,
        view_global: torch.Tensor,
        view_mask: torch.Tensor,
    ):
        batch, views, length, width = tokens.shape
        flat_mask = view_mask.flatten()
        indices = flat_mask.nonzero().flatten()
        source = tokens.reshape(batch * views, length, width)[indices]
        valid = token_valid.reshape(batch * views, length)[indices]
        condition = view_global.reshape(batch * views, -1)[indices]
        queries = self.queries[None].expand(len(indices), -1, -1)
        queries = queries + self.global_condition(condition).unsqueeze(1)
        update, _ = self.attention(
            queries, source, source, key_padding_mask=~valid, need_weights=False
        )
        slots = self.norm(queries + update)
        slots = self.norm(slots + self.ffn(slots))
        packed = slots.new_zeros(batch * views, self.slots, width)
        packed.index_copy_(0, indices, slots)
        packed = packed.reshape(batch, views * self.slots, width)
        slot_valid = (
            view_mask[:, :, None].expand(batch, views, self.slots).reshape(batch, -1)
        )
        return (packed, slot_valid)


class SparseEvidenceReader(nn.Module):
    def __init__(self, width: int = 256, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.output = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def prepare(self, tokens: torch.Tensor, valid: torch.Tensor):
        batch, length, width = tokens.shape
        key, value = self.key_value(tokens).chunk(2, -1)
        key = key.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)
        value = value.reshape(batch, length, self.heads, self.head_width).transpose(
            1, 2
        )
        return (key, value, valid[:, None, None, :])

    def forward(self, hidden: torch.Tensor, context):
        key, value, valid = context
        normalized = self.query_norm(hidden)
        query = (
            self.query(normalized)
            .reshape(hidden.shape[0], hidden.shape[1], self.heads, self.head_width)
            .transpose(1, 2)
        )
        scores = (
            query.float() @ key.float().transpose(-2, -1) / math.sqrt(self.head_width)
        )
        scores = scores.masked_fill(~valid, -10000.0)
        weights = sparsemax(scores, -1).to(value.dtype)
        update = (weights @ value).transpose(1, 2).reshape_as(hidden)
        return hidden + self.gate(normalized).sigmoid() * self.output(update)
