"""Train-only clause statistics and report-pair exclusions."""

import math
from collections import Counter

import numpy as np
import torch

from .text import word_tokens

BOUNDARIES = frozenset(("，", ",", "。", ".", ";", "；", "!", "！", "?", "？"))


def clause_spans(tokens):
    start, result = (0, [])
    for index, token in enumerate(tokens):
        if token in BOUNDARIES:
            if any((t not in BOUNDARIES for t in tokens[start : index + 1])):
                result.append((start, index + 1))
            start = index + 1
    if start < len(tokens):
        result.append((start, len(tokens)))
    return result


def clause_keys(tokens):
    return [
        tuple((t for t in tokens[a:b] if t not in BOUNDARIES))
        for a, b in clause_spans(tokens)
    ]


class TrainClauseStatistics:
    def __init__(self, entries):
        if not entries:
            raise ValueError("Training corpus must be nonempty")
        self.documents = len(entries)
        self.df = Counter()
        for entry in entries:
            self.df.update(set(clause_keys(word_tokens(entry["caption"]))))
        self.median_df = float(np.median(list(self.df.values())))

    def idf(self, key):
        return math.log((self.documents + 1) / (self.df.get(key, 0) + 1)) + 1

    def token_weights(self, caption, max_len):
        tokens = word_tokens(caption)[: max_len - 1]
        weights = np.ones(len(tokens) + 1, dtype=np.float32)
        for a, b in clause_spans(tokens):
            key = tuple((t for t in tokens[a:b] if t not in BOUNDARIES))
            value = (self.median_df / max(1, self.df.get(key, 0))) ** 0.25
            weights[a:b] = min(2.0, max(0.5, value))
        weights /= weights.mean()
        result = torch.zeros(max_len, dtype=torch.float32)
        result[: len(weights)] = torch.from_numpy(weights)
        return result

    def features(self, caption, patient):
        tokens = tuple(word_tokens(caption))
        clauses = set(clause_keys(tokens))
        grams = set((tuple(tokens[i : i + 3]) for i in range(max(0, len(tokens) - 2))))
        if not grams:
            grams = {tokens}
        return dict(
            tokens=tokens,
            clauses={key: self.idf(key) for key in clauses},
            grams=grams,
            patient=patient,
        )

    def audit(self):
        return dict(
            training_reports=self.documents,
            unique_clauses=len(self.df),
            median_clause_df=self.median_df,
            raw_weight_limits=[0.5, 2.0],
            statistics_source="train only",
            medical_classes_defined=False,
        )


def candidate_mask(features):
    """Unknown matches are excluded, not made clinical negatives or positives."""
    n = len(features)
    allowed = torch.ones(n, n, dtype=torch.bool)
    for i in range(n):
        for j in range(i):
            left, right = (features[i], features[j])
            if left["tokens"] == right["tokens"]:
                continue
            a, b = (left["clauses"], right["clauses"])
            union, common = (a.keys() | b.keys(), a.keys() & b.keys())
            overlap = sum((a[k] for k in common)) / max(
                1e-12, sum((a.get(k, b.get(k)) for k in union))
            )
            gu = left["grams"] | right["grams"]
            gram_overlap = len(left["grams"] & right["grams"]) / max(1, len(gu))
            same_patient = (
                left["patient"] is not None and left["patient"] == right["patient"]
            )
            if same_patient or overlap >= 0.5 or gram_overlap >= 0.7:
                allowed[i, j] = allowed[j, i] = False
    return allowed
