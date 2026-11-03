"""Evaluate a checkpoint on report manifests, including visual controls."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from otoevi.data.clauses import TrainClauseStatistics
from otoevi.data.dataset import ReportDataset
from otoevi.data.loading import build_loader
from otoevi.data.manifests import load_manifest
from otoevi.data.text import decode, normalize_caption, word_tokens
from otoevi.inference.checkpoints import load_model
from otoevi.utils.files import save_json

from .metrics import score


def evaluate(
    *,
    checkpoint: Path,
    vision_checkpoint: Path,
    manifest: Path,
    train: Path,
    image_root: Path,
    output: Path,
    device: str = "cuda:0",
    batch_size: int = 32,
    workers: int = 4,
    modes: tuple[str, ...] = ("matched", "shuffled", "blank"),
    variant: str | None = None,
) -> dict:
    """Write per-report predictions and return aggregate and template-stratum scores."""
    output = Path(output)
    entries, training_entries = load_manifest(manifest), load_manifest(train)
    model, vocabulary, tags = load_model(checkpoint, vision_checkpoint, device, variant)
    statistics = TrainClauseStatistics(training_entries)
    templates = {normalize_caption(e["caption"]) for e in training_entries}
    output.mkdir(parents=True, exist_ok=True)
    result = {}
    for mode in modes:
        dataset = ReportDataset(
            entries, image_root, vocabulary, tags, statistics, visual_mode=mode
        )
        rows = []
        with torch.inference_mode():
            for batch in build_loader(dataset, batch_size, workers, 42):
                tokens = model.generate(
                    batch["images"].to(device),
                    vocabulary["stoi"]["<bos>"],
                    vocabulary["stoi"]["<eos>"],
                    90,
                    batch["view_mask"].to(device),
                )
                rows.extend(
                    dict(
                        report_id=i,
                        file=f,
                        reference=r,
                        prediction=(
                            ("" if vocabulary["mode"] == "character" else " ").join(
                                word_tokens(p)[:90]
                            )
                            or "<empty>"
                        ),
                        visual_source_report_id=d,
                    )
                    for i, f, r, p, d in zip(
                        batch["report_id"],
                        batch["filename"],
                        batch["caption"],
                        decode(tokens.cpu(), vocabulary),
                        batch["visual_source_report_id"],
                    )
                )
        path = output / (mode + "_predictions.jsonl")
        path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
            encoding="utf-8",
        )
        result[mode] = score(rows, clinical=vocabulary.get("mode") == "character")
        unseen = [r for r in rows if normalize_caption(r["reference"]) not in templates]
        result[mode]["unseen_templates"] = (
            score(unseen, clinical=vocabulary.get("mode") == "character")
            if unseen
            else None
        )
    result["parameters"] = model.parameter_audit()
    result["metric_note"] = (
        "Corpus BLEU/ROUGE-L/CIDEr; Chinese character / English regex-word tokens. Clinical values are Chinese rule proxies only."
    )
    save_json(output / "metrics.json", result)
    return result
