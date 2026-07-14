"""Adaptive Thresholding Loss (ATLoss) for ATLOP.

Re-implemented from the description in Zhou et al., "Document-Level Relation
Extraction with Adaptive Thresholding and Localized Context Pooling" (AAAI 2021)
and the reference repo wzhouad/ATLOP (losses.py). The repo's license is
unspecified, so this is a clean re-implementation, not a copy.

Class index 0 is the *threshold* class (TH) — it doubles as DocRED's "Na"
(no-relation) label, which lines up with `data.docred_io.build_rel2id`
(Na -> 0, the 96 real P-codes -> 1..96).

Idea: instead of a single global 0.5 threshold, the model learns a per-example
threshold as class 0's logit. A relation r is predicted iff logit_r > logit_TH.
The loss pushes every gold positive above TH and every negative below it, which
handles DocRED's ~97% NA imbalance structurally.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """logits, labels: (num_pairs, num_classes). labels is multi-hot with
        class 0 = Na/TH. Returns a scalar (mean over pairs)."""
        # TH label is always class 0.
        th_label = torch.zeros_like(labels, dtype=torch.float)
        th_label[:, 0] = 1.0
        labels = labels.clone().float()
        labels[:, 0] = 0.0  # class 0 is handled by TH, not as a real positive

        p_mask = labels + th_label   # positive relations + TH slot
        n_mask = 1 - labels          # everything that is NOT a gold positive

        # Part 1: rank each gold positive class above the TH class.
        #   mask out non-(positive/TH) logits so softmax is over {positives, TH}.
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Part 2: rank the TH class above every negative class.
        #   mask out gold-positive logits so softmax is over {negatives, TH}.
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        loss = loss1 + loss2
        return loss.mean()

    def get_label(self, logits: torch.Tensor, num_labels: int = -1) -> torch.Tensor:
        """Turn logits into a multi-hot prediction (num_pairs, num_classes).

        A class is predicted iff its logit exceeds the TH (class 0) logit.
        `num_labels` optionally caps the number of positive relations per pair
        (top-k among those above threshold). If nothing beats TH, the pair is
        predicted as Na (class 0 set to 1)."""
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits)
        mask = logits > th_logit
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        # Rows with no relation above threshold -> predict Na (class 0).
        output[:, 0] = (output.sum(1) == 0).to(logits)
        return output


def evidence_loss(sent_attns: list[torch.Tensor], evidence: list[list[list[int]]]) -> torch.Tensor:
    """DREEAM-style evidence-guided attention loss.

    Re-implemented from the evidence-supervision idea in Ma et al., "DREEAM:
    Guiding Attention with Evidence for Improving Document-Level Relation
    Extraction" (ACL 2023): supervise the localized-context attention itself
    (not just the final relation logits) so it concentrates on the gold
    evidence sentences, rather than only on whichever tokens already scored
    highest.

    sent_attns[i]: (n_pair, n_sent) attention mass per sentence, for doc i's
        pairs in hts order (from DocREModel.get_hrt).
    evidence[i][k]: gold evidence sentence ids for pair k of doc i; empty for
        pairs with no evidence supervision (Na pairs, or splits without gold
        evidence like train_distant), which are skipped entirely.

    For each supervised pair, the target is a uniform distribution over its
    gold evidence sentences; the loss is the cross-entropy between that target
    and the model's per-sentence attention mass. Averaged over every
    supervised pair in the batch; 0.0 (no gradient) if none exist.
    """
    losses = []
    for attn, doc_evidence in zip(sent_attns, evidence):
        if attn.size(1) == 0:
            continue
        for k, evi in enumerate(doc_evidence):
            if not evi:
                continue
            target = attn.new_zeros(attn.size(1))
            target[evi] = 1.0 / len(evi)
            losses.append(-(target * torch.log(attn[k] + 1e-30)).sum())
    if not losses:
        return sent_attns[0].sum() * 0.0 if sent_attns else torch.tensor(0.0)
    return torch.stack(losses).mean()
