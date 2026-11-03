"""Frozen DINOv3 features and real-view/FOV masks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import timm
import torch
from torch import nn
from torch.nn import functional as F


class DINOv3Vision(nn.Module):
    """Return report means plus native per-view dense features and FOV masks."""

    def __init__(self, checkpoint: Path):
        super().__init__()
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(
                "Supply the DINOv3 ViT-S/16 pretrained checkpoint: " + str(checkpoint)
            )
        self.backbone = timm.create_model(
            "vit_small_patch16_dinov3",
            pretrained=True,
            pretrained_cfg_overlay={"file": str(checkpoint)},
            num_classes=0,
            img_size=384,
        )
        self.backbone.requires_grad_(False)
        self.levels = (2, 5, 8, 11)
        self.level_projections = nn.ModuleList(
            [nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 512)) for _ in self.levels]
        )
        self.global_projection = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, 512))
        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    def forward(self, images: torch.Tensor, view_mask: Optional[torch.Tensor] = None):
        if images.ndim != 5:
            raise ValueError("Expected [batch, views, channels, height, width]")
        batch, views, channels, height, width = images.shape
        if view_mask is None:
            view_mask = torch.ones(batch, views, dtype=torch.bool, device=images.device)
        if (
            view_mask.shape != (batch, views)
            or view_mask.dtype != torch.bool
            or (not view_mask.any(1).all())
        ):
            raise ValueError("Every report needs at least one real view")
        valid_indices = view_mask.flatten().nonzero().flatten()
        valid_images = images.reshape(batch * views, channels, height, width)[
            valid_indices
        ]
        final, intermediates = self.backbone.forward_intermediates(
            valid_images,
            indices=list(self.levels),
            return_prefix_tokens=True,
            norm=True,
            output_fmt="NCHW",
            stop_early=False,
        )
        grids = [item[0] for item in intermediates]
        compact = []
        for grid, projection in zip(grids, self.level_projections):
            pooled = F.adaptive_avg_pool2d(grid, 7).flatten(2).transpose(1, 2)
            compact.append(projection(pooled))
        compact = torch.stack(compact, 1)
        global_feature = self.global_projection(final[:, 0])
        dense = torch.stack(
            [
                grids[-2].flatten(2).transpose(1, 2),
                grids[-1].flatten(2).transpose(1, 2),
            ],
            1,
        )
        native_tokens = dense.shape[2]
        raw = (
            valid_images * self.image_std.to(valid_images.dtype)
            + self.image_mean.to(valid_images.dtype)
        ).clamp(0, 1)
        pixel_foreground = raw.amax(1, keepdim=True).gt(0.04).float()
        spatial_valid = F.adaptive_avg_pool2d(pixel_foreground, 24).flatten(1).gt(0.25)
        # An empty real view must not change the mask of another real view.
        spatial_valid = spatial_valid | ~spatial_valid.any(dim=1, keepdim=True)
        packed_multiscale = compact.new_zeros(batch * views, 4, 49, 512)
        packed_global = global_feature.new_zeros(batch * views, 512)
        packed_dense = dense.new_zeros(batch * views, 2, native_tokens, 384)
        packed_spatial = torch.zeros(
            batch * views, native_tokens, dtype=torch.bool, device=images.device
        )
        packed_multiscale.index_copy_(0, valid_indices, compact)
        packed_global.index_copy_(0, valid_indices, global_feature)
        packed_dense.index_copy_(0, valid_indices, dense)
        packed_spatial.index_copy_(0, valid_indices, spatial_valid)
        weights = view_mask.to(global_feature.dtype)
        denominator = weights.sum(1).clamp_min(1)
        return {
            "multiscale": (
                packed_multiscale.reshape(batch, views, 4, 49, 512)
                * weights[:, :, None, None, None]
            ).sum(1)
            / denominator[:, None, None, None],
            "global_feature": (
                packed_global.reshape(batch, views, 512) * weights[:, :, None]
            ).sum(1)
            / denominator[:, None],
            "view_global": packed_global.reshape(batch, views, 512),
            "dense": packed_dense.reshape(batch, views, 2, native_tokens, 384),
            "spatial_valid": packed_spatial.reshape(batch, views, native_tokens),
            "view_mask": view_mask,
        }
