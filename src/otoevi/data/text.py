"""Chinese/English report vocabulary fitted on training data only."""

from __future__ import annotations

import re
from collections import Counter

import torch

SPECIAL = ("<pad>", "<bos>", "<eos>", "<unk>")
PUNCTUATION = re.compile(r"^[^a-z0-9]+$")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\w\s]")
CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")


def word_tokens(text: str) -> list[str]:
    if CJK_PATTERN.search(text):
        return [character for character in text if not character.isspace()]
    return TOKEN_PATTERN.findall(text.lower())


def build_vocabulary(
    entries: list[dict], train_indices: list[int], minimum_frequency: int
) -> dict:
    counts = Counter()
    for index in train_indices:
        counts.update(word_tokens(entries[index]["caption"]))
    body = [
        token
        for token, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count >= minimum_frequency
    ]
    tokens = list(SPECIAL) + body
    mode = (
        "character"
        if any(
            (CJK_PATTERN.search(entries[index]["caption"]) for index in train_indices)
        )
        else "word"
    )
    return {
        "tokens": tokens,
        "stoi": {token: index for index, token in enumerate(tokens)},
        "mode": mode,
    }


def decode(token_rows: torch.Tensor, vocabulary: dict) -> list[str]:
    tokens = vocabulary["tokens"]
    eos = vocabulary["stoi"]["<eos>"]
    outputs = []
    for row in token_rows.tolist():
        words = []
        for token_id in row:
            if token_id == eos:
                break
            token = tokens[token_id] if 0 <= token_id < len(tokens) else "<unk>"
            if token not in SPECIAL:
                words.append(token)
        separator = "" if vocabulary.get("mode") == "character" else " "
        outputs.append(separator.join(words).strip() or "<empty>")
    return outputs


def normalize_caption(caption: str) -> str:
    return " ".join(caption.lower().split())
