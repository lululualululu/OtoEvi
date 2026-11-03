"""Clause Coverage Learning: exact-positive coverage and pooled reverse loss."""

import torch
from torch.nn import functional as F


def clause_pair_masks(clause_tokens, clause_mask, clause_valid, report_pair_allowed):
    batch, clauses, length = clause_tokens.shape
    flat_tokens = clause_tokens.reshape(batch * clauses, length)
    flat_mask = clause_mask.reshape(batch * clauses, length)
    flat_valid = clause_valid.reshape(batch * clauses)
    owner = (
        torch.arange(batch, device=clause_tokens.device)[:, None]
        .expand(batch, clauses)
        .reshape(-1)
    )
    exact = (
        flat_tokens[:, None].eq(flat_tokens[None, :])
        | ~(flat_mask[:, None] | flat_mask[None, :])
    ).all(-1)
    exact &= flat_mask[:, None].eq(flat_mask[None, :]).all(-1)
    exact &= flat_valid[:, None] & flat_valid[None, :]
    membership = F.one_hot(owner, batch).T.bool() & flat_valid[None]
    positives = (membership.float() @ exact.float()).gt(0)
    allowed = report_pair_allowed[:, owner] & flat_valid[None]
    allowed |= positives
    return (positives, allowed, owner, flat_valid)


def slot_clause_loss(
    model,
    slots,
    slot_valid,
    clause_tokens,
    clause_mask,
    clause_valid,
    report_pair_allowed,
    temperature=0.07,
):
    positives, allowed, owner, flat_valid = clause_pair_masks(
        clause_tokens, clause_mask, clause_valid, report_pair_allowed
    )
    text = model.clause_embeddings(clause_tokens, clause_mask).reshape(
        -1, slots.shape[-1]
    )
    visual = F.normalize(model.clause_visual_projection(slots), dim=-1)
    text = F.normalize(text, dim=-1)
    similarity = torch.einsum("bkd,nd->bkn", visual.float(), text.float())
    similarity = similarity.masked_fill(~slot_valid[:, :, None], -torch.inf)
    scores = temperature * torch.logsumexp(similarity / temperature, 1)
    scores -= temperature * slot_valid.sum(1).float().log()[:, None]
    scores = scores.masked_fill(~flat_valid[None], -torch.inf)
    unique = unique_positive_mask(clause_tokens, clause_mask, clause_valid, positives)
    loss = coverage_loss(scores / temperature, positives, allowed, unique)
    negatives = allowed & ~positives
    return (
        loss,
        {
            "clauses": int(flat_valid.sum().detach()),
            "positive_fraction": float(
                positives[:, flat_valid].float().mean().detach()
            ),
            "excluded_fraction": float(
                (~allowed[:, flat_valid]).float().mean().detach()
            ),
            "no_negative_row_fraction": float(
                (~negatives.any(1)).float().mean().detach()
            ),
        },
    )


def unique_positive_mask(tokens, mask, valid, positives):
    count, length = tokens.shape[1:]
    flat, flat_mask = tokens.reshape(-1, length), mask.reshape(-1, length)
    same = (flat[:, None].eq(flat[None]) | ~(flat_mask[:, None] | flat_mask[None])).all(
        -1
    ) & flat_mask[:, None].eq(flat_mask[None]).all(-1)
    same &= valid.flatten()[:, None] & valid.flatten()[None]
    # An exact clause has the same positive label in every duplicate column.
    first = ~torch.tril(same, diagonal=-1).any(1)
    return positives & first[None]


def coverage_loss(scores, positives, allowed, unique):
    denominator = scores.masked_fill(~allowed, -torch.inf).logsumexp(-1)
    positive_mean = scores.masked_fill(~unique, 0.0).sum(-1) / unique.sum(-1).clamp_min(
        1
    )
    forward = (denominator - positive_mean).mean()
    active = positives.any(0)
    reverse_scores = scores.T[active]
    reverse_pos, reverse_allowed = positives.T[active], allowed.T[active]
    reverse = (
        reverse_scores.masked_fill(~reverse_allowed, -torch.inf).logsumexp(-1)
        - reverse_scores.masked_fill(~reverse_pos, -torch.inf).logsumexp(-1)
    ).mean()
    return 0.5 * (forward + reverse)
