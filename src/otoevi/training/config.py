"""Validated options for the fixed-budget training workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from otoevi.models.ablations import VARIANTS


@dataclass(frozen=True)
class TrainConfig:
    """Paths and run options; the model's published recipe stays in the engine."""

    train: Path
    val: Path
    image_root: Path
    vision_checkpoint: Path
    output: Path
    variant: str = "full"
    seed: int = 42
    epochs: int = 15
    batch_size: int = 32
    effective_batch: int = 64
    initialization_bank_size: int = 384
    workers: int = 4
    device: str = "cuda:0"
    resume: bool = False

    def __post_init__(self) -> None:
        for field in ("train", "val", "image_root", "vision_checkpoint", "output"):
            object.__setattr__(self, field, Path(getattr(self, field)))

    def validate(self) -> None:
        """Reject invalid budgets and accidental reuse of a populated run folder."""
        if self.variant not in VARIANTS:
            raise ValueError(f"Unknown variant: {self.variant}")
        if (
            self.epochs < 2
            or self.batch_size < 1
            or self.effective_batch < 1
            or self.effective_batch % self.batch_size
        ):
            raise ValueError(
                "epochs must be >=2; positive batch-size must divide effective-batch"
            )
        if self.initialization_bank_size < 1 or self.workers < 0:
            raise ValueError(
                "initialization-bank-size must be positive; workers must be nonnegative"
            )
        if self.output.exists() and any(self.output.iterdir()) and not self.resume:
            raise ValueError(
                "Output already contains files; choose a new directory or use --resume"
            )
