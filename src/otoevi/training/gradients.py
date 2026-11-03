"""Coordinate the main and clause gradients at the shared evidence slots."""

import torch


def protect_slot_gradient(main, clause, slots):
    (main_gradient,) = torch.autograd.grad(main, slots, retain_graph=True)
    (clause_gradient,) = torch.autograd.grad(clause, slots, retain_graph=True)
    gm = main_gradient.float().flatten(1)
    ga = clause_gradient.float().flatten(1)
    dot = (gm * ga).sum(1, keepdim=True)
    norm2 = gm.square().sum(1, keepdim=True)
    projected = (
        ga
        - torch.where(dot < 0, dot / norm2.clamp_min(1e-20), torch.zeros_like(dot)) * gm
    )
    main_norm = gm.norm(dim=1, keepdim=True)
    auxiliary_norm = projected.norm(dim=1, keepdim=True)
    scale = torch.minimum(
        torch.ones_like(main_norm), 0.5 * main_norm / auxiliary_norm.clamp_min(1e-20)
    )
    scale = torch.where(main_norm > 1e-10, scale, torch.ones_like(scale))
    protected = projected * scale
    desired = (gm + protected).reshape_as(slots).to(slots.dtype).detach()
    slots.register_hook(lambda incoming: desired)
    cosine = (gm * ga).sum(1) / (gm.norm(dim=1) * ga.norm(dim=1)).clamp_min(1e-20)
    active = main_norm.squeeze(1) > 1e-10
    raw_ratio = ga.norm(dim=1) / main_norm.squeeze(1).clamp_min(1e-20)
    protected_ratio = protected.norm(dim=1) / main_norm.squeeze(1).clamp_min(1e-20)
    return {
        "raw_cosine": float(cosine.mean().detach()),
        "raw_aux_to_main_norm": float(raw_ratio[active].mean().detach())
        if active.any()
        else None,
        "protected_aux_to_main_norm": float(protected_ratio[active].mean().detach())
        if active.any()
        else None,
        "main_gradient_zero_fraction": float((~active).float().mean().detach()),
        "conflict_fraction": float(dot.lt(0).float().mean().detach()),
    }
