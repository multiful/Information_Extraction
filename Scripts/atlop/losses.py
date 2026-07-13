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


class PUATLoss(ATLoss):
    """PU(positive-unlabeled)-style variant of ATLoss for train_distant.

    Motivation (measured on this dataset, see Scripts/models/EXPERIMENTS.md and
    the track-3 problem-definition analysis): re-running the distant labeling
    function (Wikidata fact matching) over dev shows 62.2% of human-annotated
    relations come out as "Na" — i.e. most Na labels in train_distant are
    *unlabeled*, not confirmed negatives. Under plain ATLoss, every such pair
    feeds Part 2 ("rank TH above every negative") with a false target, actively
    training the threshold to sit above true relations.

    Fix, kept deliberately minimal (same simplification A/B-validated on the
    track-1 model — TTM-RE 2024's core intuition without its nnPU risk
    estimator): down-weight Part 2 by `na_weight` for pairs whose distant label
    is all-Na. Pairs with >=1 distant positive are untouched, and Part 1 is
    untouched. na_weight=1.0 is exactly ATLoss."""

    def __init__(self, na_weight: float = 0.5):
        super().__init__()
        self.na_weight = na_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        th_label = torch.zeros_like(labels, dtype=torch.float)
        th_label[:, 0] = 1.0
        labels = labels.clone().float()
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Down-weight the TH-ranking term only where the distant label is
        # all-Na (an *unlabeled* pair, not a trusted negative).
        is_na = labels.sum(1) == 0
        w2 = torch.where(is_na, torch.full_like(loss2, self.na_weight),
                         torch.ones_like(loss2))
        return (loss1 + w2 * loss2).mean()
