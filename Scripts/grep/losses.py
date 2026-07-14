import torch

EPS = 1e-30


def evidence_kl_loss(u: torch.Tensor, evidence: list[list[int]]) -> torch.Tensor | None:
    idx = [i for i, ev in enumerate(evidence) if ev]
    if not idx:
        return None
    n_sent = u.size(1)
    losses = []
    for i in idx:
        ev = [s for s in evidence[i] if s < n_sent]
        if not ev:
            continue
        v = torch.zeros(n_sent, device=u.device, dtype=u.dtype)
        v[ev] = 1.0 / len(ev)
        ui = u[i].clamp_min(EPS)
        vi = v.clamp_min(EPS)
        losses.append((v * (torch.log(vi) - torch.log(ui))).sum())
    if not losses:
        return None
    return torch.stack(losses).mean()
