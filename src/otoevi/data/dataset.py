"""Report-level images and text targets, including visual control modes."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from .manifests import patient_id
from .text import normalize_caption, word_tokens
from .transforms import dino_transform


class ReportDataset(Dataset):
    def __init__(
        self,
        entries,
        image_root,
        vocabulary,
        tags,
        statistics,
        training=False,
        visual_mode="matched",
        max_len=90,
    ):
        if visual_mode not in ("matched", "shuffled", "blank"):
            raise ValueError(visual_mode)
        self.entries = entries
        self.image_root = Path(image_root)
        self.vocabulary = vocabulary
        self.tags = tags
        self.statistics = statistics
        self.training = training
        self.visual_mode = visual_mode
        self.max_len = max_len
        self.transform = dino_transform(training)
        self.donors = list(range(len(entries)))
        if visual_mode == "shuffled":
            if len(entries) < 2:
                raise ValueError("Shuffled evaluation needs at least two reports")
            self.donors = [(i + 1) % len(entries) for i in range(len(entries))]
        stoi = vocabulary["stoi"]
        self.punctuation = {
            stoi[p]
            for p in (".", ",", ";", "!", "?", "。", "，", "；", "！", "？", "、")
            if p in stoi
        }
        self.special = {stoi[p] for p in ("<pad>", "<bos>", "<eos>")}

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        donor = self.entries[self.donors[index]]
        images = []
        for name in donor["filenames"][:2]:
            with Image.open(self.image_root / name) as image:
                images.append(self.transform(image.convert("RGB")))
        images = torch.stack(images)
        if self.visual_mode == "blank":
            images.zero_()
        stoi = self.vocabulary["stoi"]
        encoded = [stoi.get(t, stoi["<unk>"]) for t in word_tokens(entry["caption"])][
            : self.max_len - 1
        ]
        inputs, targets = [stoi["<bos>"]] + encoded, encoded + [stoi["<eos>"]]
        inputs += [stoi["<pad>"]] * (self.max_len - len(inputs))
        targets += [stoi["<pad>"]] * (self.max_len - len(targets))
        normalized = normalize_caption(entry["caption"])
        token_set = set(word_tokens(entry["caption"]))
        tag_targets = [
            float(
                tag in normalized
                if any("\u4e00" <= c <= "\u9fff" for c in tag)
                else tag in token_set
            )
            for tag in self.tags
        ]
        clauses, current = [], []
        for value in inputs:
            if value in self.special:
                continue
            if value in self.punctuation:
                if current:
                    clauses.append(current)
                    current = []
            else:
                current.append(value)
        if current:
            clauses.append(current)
        tokens = torch.full((10, 32), stoi["<pad>"], dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for row, clause in enumerate(clauses[:10]):
            clause = clause[:32]
            tokens[row, : len(clause)] = torch.tensor(clause)
            mask[row, : len(clause)] = True
        if not mask.any():
            raise ValueError("Report needs a nonempty clause")
        return dict(
            images=images,
            view_mask=torch.ones(len(images), dtype=torch.bool),
            inputs=torch.tensor(inputs),
            targets=torch.tensor(targets),
            tag_targets=torch.tensor(tag_targets, dtype=torch.float32),
            report_id=entry["report_id"],
            filename=entry["filenames"][0],
            caption=entry["caption"],
            visual_source_report_id=donor["report_id"],
            ce_weights=self.statistics.token_weights(entry["caption"], self.max_len),
            pair_features=self.statistics.features(entry["caption"], patient_id(entry)),
            clause_tokens=tokens,
            clause_mask=mask,
            clause_valid=mask.any(-1),
        )
