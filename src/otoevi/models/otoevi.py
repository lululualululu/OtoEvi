"""OtoEvi: visual evidence-guided report generation from otoscopic images."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

from .evidence import EvidenceSlotResampler, SparseEvidenceReader
from .memory import VisualMemoryDecoder
from .vision import DINOv3Vision

ASSETS = Path("weights/dinov3")


class _EvidenceGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        num_tags: int = 384,
        *,
        checkpoint: Path = ASSETS / "model.safetensors",
        max_len: int = 90,
        decoder_layers: int = 5,
    ):
        super().__init__()
        self.vision = DINOv3Vision(checkpoint)
        self.captioner = VisualMemoryDecoder(
            vocab_size,
            pad_id,
            num_tags,
            min(8, num_tags),
            layers=decoder_layers,
            max_len=max_len,
        )
        self.local_projections = nn.ModuleList(
            [
                nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 256), nn.LayerNorm(256))
                for _ in range(2)
            ]
        )
        self.sparse_readers = nn.ModuleDict()
        with torch.random.fork_rng(devices=[]):
            self.slot_resampler = EvidenceSlotResampler()
            for layer in range(max(0, decoder_layers - 2), decoder_layers):
                self.sparse_readers[str(layer)] = SparseEvidenceReader()

    def decode(self, inputs: torch.Tensor, memory: torch.Tensor, contexts):
        captioner = self.captioner
        hidden = (
            captioner.token_embed(inputs) + captioner.text_pos[: inputs.shape[1]][None]
        )
        causal = torch.ones(
            inputs.shape[1], inputs.shape[1], dtype=torch.bool, device=inputs.device
        ).triu(1)
        for index, layer in enumerate(captioner.decoder.layers):
            hidden = layer(
                hidden,
                memory,
                tgt_mask=causal,
                tgt_key_padding_mask=inputs.eq(captioner.pad_id),
            )
            if str(index) in contexts:
                hidden = self.sparse_readers[str(index)](hidden, contexts[str(index)])
        if captioner.decoder.norm is not None:
            hidden = captioner.decoder.norm(hidden)
        return captioner.output(hidden)

    def forward(
        self, images, input_ids, targets=None, tag_targets=None, view_mask=None
    ):
        if targets is not None or tag_targets is not None:
            raise ValueError(
                "Use training.objective.objective for the full CCL objective"
            )
        memory, contexts = self.encode(images, view_mask)
        return {"logits": self.decode(input_ids, memory, contexts)}

    @torch.inference_mode()
    def generate(self, images, bos_id, eos_id, max_new_tokens=90, view_mask=None):
        memory, contexts = self.encode(images, view_mask)
        tokens = torch.full(
            (images.shape[0], 1), bos_id, dtype=torch.long, device=images.device
        )
        ended = torch.zeros(images.shape[0], dtype=torch.bool, device=images.device)
        for _ in range(max_new_tokens):
            next_token = self.decode(tokens, memory, contexts)[:, -1].argmax(-1)
            next_token = next_token.masked_fill(ended, eos_id)
            tokens = torch.cat([tokens, next_token[:, None]], 1)
            ended |= next_token.eq(eos_id)
            if ended.all():
                break
        return tokens[:, 1:]


class OtoEvi(_EvidenceGenerator):
    """One deployable model; text projections are training-only auxiliaries."""

    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        num_tags: int = 384,
        *,
        checkpoint: Path = ASSETS / "model.safetensors",
        max_len: int = 90,
        decoder_layers: int = 5,
    ):
        super().__init__(
            vocab_size,
            pad_id,
            num_tags,
            checkpoint=checkpoint,
            max_len=max_len,
            decoder_layers=decoder_layers,
        )
        self.clause_visual_projection = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 256)
        )
        self.clause_text_projection = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 256)
        )
        self.fov_global_projection = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, 512), nn.GELU()
        )
        self.fov_global_output = nn.Linear(512, 512, bias=False)
        nn.init.zeros_(self.fov_global_output.weight)
        # Preserve seed-compatible construction, discard every retired parameter.
        from .compatibility import discard_retired_modules

        discard_retired_modules(self)
        self.use_memory = self.use_reading = self.use_coverage = True
        self.coordinate_gradients = True
        self.captioner.use_memory = True
        self.variant = "full"
        self.vision.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        self.vision.backbone.eval()
        return self

    def encode_evidence(
        self,
        images: torch.Tensor,
        view_mask: Optional[torch.Tensor] = None,
        with_aux=True,
    ):
        features = self.vision(images, view_mask)
        features["global"] = features["global_feature"]
        memory, _ = self.captioner.build_memory(features)
        if not (self.use_reading or (self.use_coverage and with_aux)):
            return memory, {}, None, None
        dense = features["dense"]
        batch, views, _, native, _ = dense.shape
        tokens = torch.cat(
            [
                projection(dense[:, :, index])
                for index, projection in enumerate(self.local_projections)
            ],
            2,
        )
        valid = (
            features["view_mask"][:, :, None, None]
            & features["spatial_valid"][:, :, None, :]
        )
        valid = valid.expand(batch, views, 2, native).reshape(batch, views, 2 * native)
        slots, slot_valid = self.slot_resampler(
            tokens, valid, features["view_global"], features["view_mask"]
        )
        contexts = (
            {
                key: reader.prepare(slots, slot_valid)
                for key, reader in self.sparse_readers.items()
            }
            if self.use_reading
            else {}
        )
        return (memory, contexts, slots, slot_valid)

    def encode(self, images: torch.Tensor, view_mask: Optional[torch.Tensor] = None):
        memory, contexts, _, _ = self.encode_evidence(images, view_mask, with_aux=False)
        return (memory, contexts)

    def clause_embeddings(self, clause_tokens: torch.Tensor, clause_mask: torch.Tensor):
        embedded = self.captioner.token_embed(clause_tokens).detach()
        pooled = (embedded * clause_mask[:, :, :, None]).sum(2) / clause_mask.sum(2)[
            :, :, None
        ].clamp_min(1)
        return self.clause_text_projection(pooled)

    def parameter_audit(self):
        parameters = dict(self.named_parameters())
        total = sum(p.numel() for p in parameters.values())
        training_prefixes = (
            "clause_visual_projection.",
            "clause_text_projection.",
            "captioner.visual_align.",
            "captioner.text_align.",
        )
        training_only = sum(
            p.numel() for n, p in parameters.items() if n.startswith(training_prefixes)
        )
        return dict(
            model="OtoEvi",
            variant=self.variant,
            checkpoint_total_parameters=total,
            trainable_parameters=sum(
                p.numel() for p in parameters.values() if p.requires_grad
            ),
            training_only_parameters=training_only,
            visual_tower_parameters=sum(
                p.numel() for p in self.vision.backbone.parameters()
            ),
            generation_memory_tokens=58 if self.use_memory else 1,
            local_reading=self.use_reading,
            clause_coverage=self.use_coverage,
            under_100m=total < 100000000,
            count_note="Stored ablation weights include disabled branches; not an active inference cost estimate.",
        )
