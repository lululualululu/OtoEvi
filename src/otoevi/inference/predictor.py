"""Image-to-report inference, independent of command-line parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from otoevi.data.text import decode
from otoevi.data.transforms import dino_transform

from .checkpoints import load_model


def predict_report(
    *,
    checkpoint: Path,
    vision_checkpoint: Path,
    images: Sequence[Path],
    device: str = "cpu",
    variant: str | None = None,
) -> str:
    """Load a checkpoint and generate one report without a reference caption."""
    image_paths = list(images)
    if not 1 <= len(image_paths) <= 2:
        raise ValueError("Supply one or two real images from the same report")
    model, vocabulary, _ = load_model(checkpoint, vision_checkpoint, device, variant)
    transform, values = dino_transform(), []
    for path in image_paths:
        with Image.open(path) as image:
            values.append(transform(image.convert("RGB")))
    images = torch.stack(values)[None].to(device)
    mask = torch.ones(images.shape[:2], dtype=torch.bool, device=device)
    with torch.inference_mode():
        tokens = model.generate(
            images, vocabulary["stoi"]["<bos>"], vocabulary["stoi"]["<eos>"], 90, mask
        )
    return decode(tokens.cpu(), vocabulary)[0]
