#!/usr/bin/env python3
"""Score saved predictions, optionally against a different reference manifest."""

import argparse
import gzip
import json
from pathlib import Path

from otoevi.evaluation.metrics import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--references", type=Path)
    parser.add_argument(
        "--clinical", action="store_true", help="Chinese ten-concept rules only"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    opener = gzip.open if args.predictions.suffix == ".gz" else open
    with opener(args.predictions, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or len({r["report_id"] for r in rows}) != len(rows):
        raise ValueError("Predictions must have unique report IDs and be nonempty")
    if args.references:
        refs = json.loads(args.references.read_text())
        mapping = {r["report_id"]: r["caption"] for r in refs}
        if len(mapping) != len(refs) or set(mapping) != {r["report_id"] for r in rows}:
            raise ValueError("Reference/prediction cohorts differ")
        rows = [dict(r, reference=mapping[r["report_id"]]) for r in rows]
    result = score(rows, clinical=args.clinical)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
