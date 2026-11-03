#!/usr/bin/env python3
"""Rebuild the paper's fixed public-dataset manifests from user-supplied TSVs."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def prepare(dataset, annotations, output):
    spec = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs/splits" / (dataset + ".json")
        ).read_text()
    )
    captions = {}
    for filename in annotations:
        for line in Path(filename).read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            name, caption = line.split("\t", 1)
            if name in captions and captions[name] != caption:
                raise ValueError("Conflicting captions for " + name)
            captions[name] = caption
    rebuilt, originals = {}, []
    for split, rows in spec["splits"].items():
        rebuilt[split] = []
        for row in rows:
            candidates = [captions[f] for f in row["filenames"] if f in captions]
            source = next(
                (s for s in candidates if digest(s) == row["source_caption_sha256"]),
                None,
            )
            if source is None:
                raise ValueError(
                    "Missing or different source caption: " + row["report_id"]
                )
            target = (
                "".join(
                    source[p[0] : p[1]] if isinstance(p, list) else p
                    for p in row["extractive_edits"]
                )
                if "extractive_edits" in row
                else source
            )
            if digest(target) != row["caption_sha256"]:
                raise ValueError("Caption reconstruction failed: " + row["report_id"])
            item = dict(
                report_id=row["report_id"],
                patient_id=None,
                filenames=row["filenames"],
                caption=target,
            )
            rebuilt[split].append(item)
            if split == "test":
                originals.append(dict(item, caption=source))
    output = Path(output)
    if output.exists() and any(output.glob("*.json")):
        raise FileExistsError("Choose an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in rebuilt.items():
        (output / (split + ".json")).write_text(
            json.dumps(rows, ensure_ascii=False, indent=2)
        )
    (output / "test_original.json").write_text(
        json.dumps(originals, ensure_ascii=False, indent=2)
    )
    print(json.dumps({split: len(rows) for split, rows in rebuilt.items()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("iu_xray", "peir_gross"))
    parser.add_argument("--annotations", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(**vars(args))
