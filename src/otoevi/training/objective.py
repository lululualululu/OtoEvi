"""Paper Eq. 10 and shared-slot gradient coordination (Eq. 11)."""

from .alignment import slot_clause_loss
from .generation import auxiliary_loss, generation_losses
from .gradients import protect_slot_gradient


def objective(model, batch, weight=1.0, protect=True):
    memory, contexts, slots, valid = model.encode_evidence(
        batch["images"], batch["view_mask"], with_aux=True
    )
    logits = model.decode(batch["inputs"], memory, contexts)
    ce, _ = generation_losses(
        logits, batch["targets"], model.captioner.pad_id, batch["ce_weights"]
    )
    auxiliary, contrast, pairs = auxiliary_loss(
        model.captioner, memory, batch["inputs"], None
    )
    main = ce + auxiliary
    raw, stats = ce.new_zeros(()), {}
    if model.use_coverage:
        raw, stats = slot_clause_loss(
            model,
            slots,
            valid,
            batch["clause_tokens"],
            batch["clause_mask"],
            batch["clause_valid"],
            batch["pair_allowed"],
        )
    clause = 0.015 * raw
    gradients = None
    if protect and model.training and model.use_coverage and model.coordinate_gradients:
        connected_main = main if model.use_reading else main + slots.sum() * 0.0
        gradients = protect_slot_gradient(
            connected_main * weight, clause * weight, slots
        )
    return dict(
        loss=main + clause,
        scaled_loss=(main + clause) * weight,
        ce=ce,
        aux=auxiliary,
        contrast=contrast,
        clause_raw=raw,
        clause=clause,
        memory=memory,
        slots=slots,
        contexts=contexts,
        pair_statistics=pairs,
        clause_statistics=stats,
        slot_gradient_statistics=gradients,
    )
