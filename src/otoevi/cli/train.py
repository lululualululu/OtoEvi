"""Train OtoEvi with the fixed-budget report-generation protocol."""

import argparse
import json
from pathlib import Path

from otoevi.models.ablations import VARIANTS
from otoevi.training.config import TrainConfig
from otoevi.training.engine import train


def main(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args(argv)
    defaults = json.loads(known.config.read_text()) if known.config else {}
    unknown = set(defaults) - set(TrainConfig.__dataclass_fields__)
    if unknown:
        pre.error("Unknown config keys: " + str(sorted(unknown)))
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default=TrainConfig.variant)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument(
        "--effective-batch", type=int, default=TrainConfig.effective_batch
    )
    parser.add_argument(
        "--initialization-bank-size",
        type=int,
        default=TrainConfig.initialization_bank_size,
        help="Legacy initialization bank size only; not a trained semantic head (paper: 384)",
    )
    parser.add_argument("--workers", type=int, default=TrainConfig.workers)
    parser.add_argument("--device", default=TrainConfig.device)
    parser.add_argument("--resume", action="store_true")
    for action in parser._actions:
        if action.dest in defaults:
            action.required = False
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    del args.config
    try:
        config = TrainConfig(**vars(args))
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    train(config)


if __name__ == "__main__":
    main()
