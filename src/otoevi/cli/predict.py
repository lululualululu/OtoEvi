"""Generate a report from one or two views belonging to the same examination."""

import argparse
from pathlib import Path

from otoevi.inference.predictor import predict_report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vision-checkpoint", type=Path, required=True)
    parser.add_argument("--images", nargs="+", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", default=None)
    args = parser.parse_args(argv)
    if not 1 <= len(args.images) <= 2:
        parser.error("Supply one or two real images from the same report")
    print(predict_report(**vars(args)))


if __name__ == "__main__":
    main()
