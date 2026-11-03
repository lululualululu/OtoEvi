"""Generation and report-level semantic supervision."""

import torch
from torch.nn import functional as F


def multi_positive_loss(scores, positives, allowed=None):
    scores = scores.float()
    if allowed is None:
        allowed = torch.ones_like(positives)
    if scores.shape != positives.shape or allowed.shape != scores.shape:
        raise ValueError("Pair masks must match scores")
    if not positives.any(1).all() or not (allowed | ~positives).all():
        raise ValueError("Every row needs a retained positive")

    def direction(s, p, a):
        return (
            s.masked_fill(~a, -torch.inf).logsumexp(-1)
            - s.masked_fill(~p, -torch.inf).logsumexp(-1)
        ).mean()

    return 0.5 * (
        direction(scores, positives, allowed)
        + direction(scores.T, positives.T, allowed.T)
    )


def auxiliary_loss(captioner, memory, inputs, tags, allowed=None):
    pooled_visual = memory.mean(1)
    mask = inputs.ne(captioner.pad_id).unsqueeze(-1)
    pooled_text = (captioner.token_embed(inputs) * mask).sum(1) / mask.sum(1).clamp_min(
        1
    )
    visual = F.normalize(captioner.visual_align(pooled_visual), dim=-1)
    text = F.normalize(captioner.text_align(pooled_text), dim=-1)
    scores = visual @ text.T / 0.07
    positives = inputs[:, None].eq(inputs[None, :]).all(-1)
    if allowed is not None:
        allowed = allowed | positives
    contrast = multi_positive_loss(scores, positives, allowed)
    allowed = torch.ones_like(positives) if allowed is None else allowed
    negatives = allowed & ~positives
    stats = dict(
        positive_pair_fraction=float(positives.float().mean().detach()),
        excluded_pair_fraction=float((~allowed).float().mean().detach()),
        weak_negative_fraction=float(negatives.float().mean().detach()),
        no_negative_row_fraction=float((~negatives.any(1)).float().mean().detach()),
    )
    return (
        captioner.contrastive_weight * contrast,
        contrast,
        stats,
    )


def generation_losses(logits, targets, pad_id, weights):
    raw = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.flatten(),
        ignore_index=pad_id,
        reduction="none",
    ).reshape_as(targets)
    valid = targets.ne(pad_id)
    ce = raw.sum() / valid.sum().clamp_min(1)
    weighted = (raw * weights * valid).sum() / valid.sum().clamp_min(1)
    return (ce, 0.5 * ce + 0.5 * weighted)
