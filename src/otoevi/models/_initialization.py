"""Historical constructor order for seed-compatible initialization only.

Temporary retired heads consume the original RNG draws, then are discarded by
OtoEvi. They have no forward method, training objective, checkpoint storage or
inference role in this release. See docs/reproducibility.md.
"""

from __future__ import annotations

import math

import torch
from torch import nn

CLIP_VIT_B32_VISION_PARAMETERS = 87849216
VARIANTS = {"sparse_anchors"}


class LowRankDynamicPrefix(nn.Module):
    def __init__(self, global_dim: int, width: int, length: int = 8, rank: int = 8):
        super().__init__()
        self.length = length
        self.rank = rank
        self.base = nn.Parameter(torch.randn(length, width) / math.sqrt(width))
        self.basis = nn.Parameter(
            torch.randn(rank, length, width) / math.sqrt(width * rank)
        )
        self.coefficients = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, width),
            nn.GELU(),
            nn.Linear(width, rank),
        )
        self.film = nn.Linear(global_dim, width * 2)
        nn.init.zeros_(self.coefficients[-1].weight)
        nn.init.zeros_(self.coefficients[-1].bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)


class _BaseCaptioner(nn.Module):
    VARIANTS = {"visual_evidence"}

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        num_tags: int,
        variant: str,
        d_model: int = 256,
        nhead: int = 8,
        layers: int = 2,
        max_len: int = 90,
        evidence_slots: int = 8,
        refine_slots: int = 4,
        contrastive_weight: float = 0.05,
        local_contrastive_weight: float = 0.0,
        tag_weight: float = 0.15,
        initial_loss_weight: float = 0.35,
        visual_dim: int = 768,
        global_dim: int = 512,
        vision_parameters: int = CLIP_VIT_B32_VISION_PARAMETERS,
    ):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError("unsupported captioner variant: {}".format(variant))
        self.variant = "visual_evidence"
        self.pad_id = pad_id
        self.d_model = d_model
        self.max_len = max_len
        self.contrastive_weight = contrastive_weight
        self.local_contrastive_weight = local_contrastive_weight
        self.tag_weight = tag_weight
        self.initial_loss_weight = initial_loss_weight
        self.visual_dim = visual_dim
        self.global_dim = global_dim
        self.vision_parameters = vision_parameters
        self.visual_proj = nn.Linear(visual_dim, d_model)
        self.global_proj = nn.Linear(global_dim, d_model)
        self.spatial_pos = nn.Parameter(torch.randn(49, d_model) * 0.01)
        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.text_pos = nn.Parameter(torch.randn(max_len, d_model) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, d_model * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, d_model * 4, dropout=0.1, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.output = nn.Linear(d_model, vocab_size)
        self.early_scale_proj = nn.ModuleList(
            [nn.Linear(visual_dim, d_model) for _ in range(3)]
        )
        self.scale_embedding = nn.Parameter(torch.randn(4, d_model) * 0.01)
        self.scale_gate = nn.Linear(global_dim, 4)
        self.evidence_queries = nn.Parameter(
            torch.randn(evidence_slots, d_model) / math.sqrt(d_model)
        )
        self.query_from_global = nn.Linear(global_dim, d_model)
        self.evidence_attention = nn.MultiheadAttention(
            d_model, nhead, batch_first=True
        )
        self.evidence_norm = nn.LayerNorm(d_model)
        self.evidence_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.evidence_gate = nn.Parameter(torch.tensor(-3.0))
        self.state_mixer = nn.Identity()
        self.tag_head = nn.Linear(d_model, num_tags)
        self.visual_align = nn.Linear(d_model, d_model, bias=False)
        self.text_align = nn.Linear(d_model, d_model, bias=False)
        self.uses_refinement = False


class _PrefixCaptioner(_BaseCaptioner):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        num_tags: int,
        variant: str,
        d_model: int = 256,
        nhead: int = 8,
        layers: int = 2,
        max_len: int = 90,
        evidence_slots: int = 8,
        refine_slots: int = 4,
        contrastive_weight: float = 0.05,
        local_contrastive_weight: float = 0.0,
        tag_weight: float = 0.1,
        initial_loss_weight: float = 0.35,
        visual_dim: int = 768,
        global_dim: int = 512,
        vision_parameters: int = CLIP_VIT_B32_VISION_PARAMETERS,
        prefix_length: int = 8,
        prefix_rank: int = 8,
        retained_per_scale: int = 24,
        phrase_alignment_weight: float = 0.02,
        phrase_chunks: int = 6,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"unsupported anchor-memory variant: {variant}")
        super().__init__(
            vocab_size=vocab_size,
            pad_id=pad_id,
            num_tags=num_tags,
            variant="visual_evidence",
            d_model=d_model,
            nhead=nhead,
            layers=layers,
            max_len=max_len,
            evidence_slots=evidence_slots,
            refine_slots=refine_slots,
            contrastive_weight=contrastive_weight,
            local_contrastive_weight=local_contrastive_weight,
            tag_weight=tag_weight,
            initial_loss_weight=initial_loss_weight,
            visual_dim=visual_dim,
            global_dim=global_dim,
            vision_parameters=vision_parameters,
        )
        self.variant = "sparse_anchors"
        self.num_tags = num_tags
        self.evidence_slots = evidence_slots
        self.retained_per_scale = retained_per_scale
        self.phrase_alignment_weight = 0.0
        self.adaptive_token_weight = 0.0
        self.phrase_chunks = phrase_chunks
        self.uses_semantic_injection = False
        self.uses_retrieval = False
        self.uses_refinement = False
        self.structure_norm = nn.LayerNorm(d_model)
        self.structure_projection = nn.Linear(d_model, d_model, bias=False)
        self.structure_gate = nn.Parameter(torch.tensor(-2.5))
        nn.init.zeros_(self.structure_projection.weight)
        self.dynamic_prefix = LowRankDynamicPrefix(
            global_dim, d_model, prefix_length, prefix_rank
        )
        self.prefix_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.prefix_norm = nn.LayerNorm(d_model)
        self.prefix_gate = nn.Parameter(torch.tensor(-2.5))
        self.phrase_visual = nn.Linear(d_model, d_model, bias=False)
        self.phrase_text = nn.Linear(d_model, d_model, bias=False)
        self.alignment_temperature = nn.Parameter(torch.tensor(math.log(1 / 0.1)))
        self.uses_semantic_memory = False
        self.uses_anchor_memory = True
        self.anchor_topk = 8
        self.anchor_tag_head = nn.Linear(d_model, num_tags)
        self.anchor_embeddings = nn.Parameter(
            torch.randn(num_tags, d_model) / math.sqrt(d_model)
        )
        self.anchor_gate = nn.Parameter(torch.tensor(-1.5))
        self.anchor_loss_weight = 0.05
        self.uses_clause_state = False


class SemanticCaptioner(_PrefixCaptioner):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        num_tags: int,
        anchor_topk: int,
        d_model: int = 256,
        layers: int = 5,
        max_len: int = 90,
        visual_dim: int = 512,
        global_dim: int = 512,
        balanced_loss_weight: float = 0.08,
    ):
        variant = "sparse_anchors" if anchor_topk <= 8 else "dense_anchors"
        super().__init__(
            vocab_size=vocab_size,
            pad_id=pad_id,
            num_tags=num_tags,
            variant=variant,
            d_model=d_model,
            layers=layers,
            max_len=max_len,
            visual_dim=visual_dim,
            global_dim=global_dim,
            vision_parameters=0,
            tag_weight=0.0,
        )
        self.anchor_topk = min(anchor_topk, num_tags)
        self.anchor_loss_weight = 0.0
        self.balanced_loss_weight = balanced_loss_weight
        self.register_buffer("tag_positive_weights", torch.ones(num_tags))
