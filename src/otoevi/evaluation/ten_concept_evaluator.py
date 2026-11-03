"""Frozen reference-agreement proxy, with explicit negatives and laterality.

Unmentioned findings are not confirmed absent; unmatched positives are reference
mismatches. These rules do not constitute clinician-adjudicated accuracy.
"""

import re

import numpy as np

from ._clinical_rules import BINARY_RULES, extract_states, normalize_clinical_text

NAMES = tuple(BINARY_RULES)
# Only direct negation of a named finding; broad "normal morphology" is excluded.
NEGATIVE_RULES = {
    name: tuple(
        p
        for p in rule["negative"]
        if any(w in p for w in ("未见", "没有", "无")) and "皮肤" not in p
    )
    for name, rule in BINARY_RULES.items()
}
NEGATIVE_RULES["hyperemia"] = (r"(?:未见|无|没有)(?:明显)?(?:充血|发红|红肿)",)
NEGATIVE_RULES["clouding"] = (r"(?:未见|无|没有)(?:明显)?(?:浑浊|混浊)",)


def explicit_negative(text, name):
    text = normalize_clinical_text(text)
    found = any(re.search(p, text) for p in NEGATIVE_RULES[name])
    # A contradictory positive assertion cannot qualify as an unambiguous negative.
    cleaned = text
    for pattern in NEGATIVE_RULES[name]:
        cleaned = re.sub(pattern, "", cleaned)
    positive = any(re.search(p, cleaned) for p in BINARY_RULES[name]["terms"])
    return found and not positive


def ratio(a, b):
    return a / b if b else None


def score(rows):
    counts = np.zeros((len(rows), len(NAMES), 3), dtype=np.int32)  # TP, FP, FN
    negative = {
        n: dict(
            support=0,
            reproduced_negative=0,
            contradictory_positive=0,
            conflict=0,
            unconfirmed_or_omitted=0,
        )
        for n in NAMES
    }
    side = dict(
        reference_single_side=0,
        correct=0,
        opposite=0,
        conflict=0,
        omitted=0,
        reference_unknown=0,
        reference_ambiguous=0,
    )
    unknown_added = {n: 0 for n in NAMES}
    for i, row in enumerate(rows):
        ref, pred = extract_states(row["reference"]), extract_states(row["prediction"])
        for j, name in enumerate(NAMES):
            r, p = ref[name] == "positive", pred[name] == "positive"
            counts[i, j] = (r and p, not r and p, r and not p)
            unknown_added[name] += ref[name] == "unknown" and p
            if explicit_negative(row["reference"], name):
                item = negative[name]
                item["support"] += 1
                if explicit_negative(row["prediction"], name):
                    item["reproduced_negative"] += 1
                elif pred[name] == "positive":
                    item["contradictory_positive"] += 1
                elif pred[name] == "conflict":
                    item["conflict"] += 1
                else:
                    item["unconfirmed_or_omitted"] += 1
        if ref["laterality"] in ("left", "right"):
            side["reference_single_side"] += 1
            p = pred["laterality"]
            side[
                "correct"
                if p == ref["laterality"]
                else "opposite"
                if p in ("left", "right")
                else "conflict"
                if p == "conflict"
                else "omitted"
            ] += 1
        else:
            side[
                "reference_unknown"
                if ref["laterality"] == "unknown"
                else "reference_ambiguous"
            ] += 1
    totals = counts.sum(0)
    concepts = {}
    for name, (tp, fp, fn) in zip(NAMES, totals.tolist()):
        neg = negative[name]
        neg["negative_reproduction_rate"] = ratio(
            neg["reproduced_negative"], neg["support"]
        )
        neg["positive_contradiction_rate"] = ratio(
            neg["contradictory_positive"], neg["support"]
        )
        concepts[name] = dict(
            support=tp + fn,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=ratio(tp, tp + fp),
            recall=ratio(tp, tp + fn),
            f1=ratio(2 * tp, 2 * tp + fp + fn) or 0.0,
            prediction_positive_reference_unknown=unknown_added[name],
        )
    macro, micro = f1(totals)
    side["opposite_rate"] = ratio(side["opposite"], side["reference_single_side"])
    side["any_side_failure_rate"] = ratio(
        side["opposite"] + side["conflict"] + side["omitted"],
        side["reference_single_side"],
    )
    return dict(
        reports=len(rows),
        macro_f1=macro,
        micro_f1=micro,
        concepts=concepts,
        explicit_negative=negative,
        laterality=side,
    ), counts


def f1(totals, supported=None):
    tp, fp, fn = totals.T
    denom = 2 * tp + fp + fn
    values = np.divide(
        2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0
    )
    supported = (tp + fn > 0) if supported is None else supported
    return float(values[supported].mean()) if supported.any() else 0.0, float(
        2 * tp.sum() / denom.sum()
    ) if denom.sum() else 0.0
