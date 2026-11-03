"""Masked multiview collation and reproducible report-level sampling."""

import torch
from torch.utils.data import DataLoader, default_collate

from .clauses import candidate_mask


def collate_reports(items):
    most = max(len(item["images"]) for item in items)
    rows = []
    for item in items:
        row = {
            k: v
            for k, v in item.items()
            if k not in ("images", "view_mask", "pair_features")
        }
        count = len(item["images"])
        images = item["images"].new_zeros((most,) + tuple(item["images"].shape[1:]))
        images[:count] = item["images"]
        row.update(images=images, view_mask=torch.arange(most) < count)
        rows.append(row)
    batch = default_collate(rows)
    batch["pair_allowed"] = candidate_mask([item["pair_features"] for item in items])
    return batch


def build_loader(dataset, batch_size, workers, seed, epoch=0, training=False):
    order = (
        torch.randperm(
            len(dataset), generator=torch.Generator().manual_seed(seed + epoch)
        ).tolist()
        if training
        else list(range(len(dataset)))
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=order,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_reports,
        generator=torch.Generator().manual_seed(seed + epoch + 100000),
    )
