"""Load OtoEvi checkpoints and validate active model parameters."""

from pathlib import Path

import torch

from otoevi.models.ablations import VARIANTS, AblationOtoEvi
from otoevi.models.compatibility import active_state_dict

CHECKPOINT_VARIANT_ALIASES = dict(
    zip(
        (
            "H000_base",
            "H100_A_only",
            "H010_B_only",
            "H001_D1_only",
            "H110_AB",
            "H101_AD1",
            "H011_BD1",
            "H111_ABD1",
        ),
        (
            "base",
            "memory_only",
            "reading_only",
            "coverage_only",
            "memory_reading",
            "memory_coverage",
            "reading_coverage",
            "full",
        ),
    )
)
CHECKPOINT_VARIANT_ALIASES["D1_clause_coverage"] = "full"


def checkpoint_variant(config, override=None):
    selected = override or config.get("variant", "full")
    selected = CHECKPOINT_VARIANT_ALIASES.get(selected, selected)
    if selected not in VARIANTS:
        raise ValueError("Unsupported OtoEvi checkpoint variant: " + selected)
    return selected


def load_model(checkpoint, vision_checkpoint, device="cpu", variant=None):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    vocabulary = saved["vocabulary"]
    tags = saved.get("tags", [])
    config = saved.get("config", {})
    selected = checkpoint_variant(config, variant)
    model = AblationOtoEvi(
        len(vocabulary["tokens"]),
        vocabulary["stoi"]["<pad>"],
        config.get("initialization_bank_size", len(tags) or 384),
        selected,
        checkpoint=Path(vision_checkpoint),
        max_len=config.get("max_len", 90),
        decoder_layers=config.get("decoder_layers", 5),
    )
    model.load_state_dict(active_state_dict(saved["model"]), strict=True)
    return model.to(device).eval(), vocabulary, []
