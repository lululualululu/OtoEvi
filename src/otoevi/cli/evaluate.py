"""Evaluate generated reports and matched/shuffled/blank visual controls."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--train",
        type=Path,
        required=True,
        help="Training manifest for template strata",
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("matched", "shuffled", "blank"),
        default=("matched", "shuffled", "blank"),
    )
    parser.add_argument("--variant", default=None)
    args = parser.parse_args(argv)
    from otoevi.evaluation.runner import evaluate

    result = evaluate(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
