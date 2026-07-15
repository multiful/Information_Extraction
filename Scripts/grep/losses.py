"""Auxiliary losses for GREP (Evidence Extraction + Global Relation Prediction).

Re-implemented from Zhang, Yan & Cheng, "Document-Level Relation Extraction
with Global Relations and Entity Pair Reasoning" (GREP), ACL Findings 2025.
The relation-classification loss (Eq 20) is unchanged from ATLOP, so it is not
reimplemented here -- import `Scripts.atlop.losses.ATLoss` instead.
"""

import torch

EPS = 1e-30


def evidence_kl_loss(u: torch.Tensor, evidence: list[list[int]]) -> torch.Tensor | None:
    """Evidence Extraction loss (paper Eq 16) for one document.

    u: (n_pairs, n_sent) -- per-pair predicted evidence distribution over
       sentences (`GREPModel.sentence_attention`'s output).
    evidence: length n_pairs, gold evidence sentence ids per pair (empty list
       = no gold evidence for that pair, e.g. Na pairs -- excluded from the loss,
       matching the DREEAM-style convention the paper follows).

    Returns mean KL(v || u) over pairs with gold evidence, or None if the doc
    has no such pairs.

    Note: the paper's Eq 16 is written as `L_evi = sum v*(log u - log v)`,
    which is the *negative* of KL(v||u); minimizing it as written would push
    u's mass *away* from the gold sentences. That's almost certainly a sign
    typo (the surrounding text says the goal is to minimize the KL
    divergence), so this minimizes the standard KL(v||u) = sum v*(log v - log u).
    """
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
