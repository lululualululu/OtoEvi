"""Corpus caption scores, historical scores and clinical rule proxies."""

import contextlib
import io

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.rouge.rouge import Rouge

from otoevi.data.text import word_tokens

from .ten_concept_evaluator import score as clinical_score


def score(rows, clinical=False):
    references = {
        i: [" ".join(word_tokens(row["reference"]))] for i, row in enumerate(rows)
    }
    predictions = {
        i: [" ".join(word_tokens(row["prediction"]))] for i, row in enumerate(rows)
    }
    with contextlib.redirect_stdout(io.StringIO()):
        bleu, _ = Bleu(4).compute_score(references, predictions)
        cider, _ = Cider().compute_score(references, predictions)
        rouge, _ = Rouge().compute_score(references, predictions)
    clinical_result = clinical_score(rows)[0] if clinical else None
    return dict(
        reports=len(rows),
        corpus_bleu=dict(zip(("B1", "B2", "B3", "B4"), bleu)),
        rouge_l=rouge,
        cider=cider,
        clinical_proxy=clinical_result,
        unique_predictions=len({r["prediction"] for r in rows}),
    )
