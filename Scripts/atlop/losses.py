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


def evidence_contrastive_loss(rs_list: list[torch.Tensor], sent_reprs: list[torch.Tensor],
                               evidence: list[list[list[int]]], tau: float = 0.1) -> torch.Tensor:
    """InfoNCE evidence-contrastive loss for the GAT+MLP+sigmoid pipeline.

    Unlike `evidence_loss` above (which supervises the LCP *attention* directly
    via cross-entropy), this contrasts the pair's context representation `rs`
    against per-sentence *embeddings*: gold evidence sentences of a pair are
    positives, every other sentence in the same document is a negative. This
    is the supervised multi-positive InfoNCE form (Khosla et al. 2020):
        L = -log( sum_{p in pos} exp(sim_p/tau) / sum_{a in pos+neg} exp(sim_a/tau) )
    with cosine similarity between `rs` and sentence embeddings.

    rs_list[i]: (n_pair_i, hidden) pair-context vectors (ATLOP LCP output,
        `rs` from DocREModel.get_hrt), doc i's pairs in hts order.
    sent_reprs[i]: (n_sent_i, hidden) per-sentence embeddings for doc i, e.g.
        mean-pooled token hidden states over each sentence's token span.
    evidence[i][k]: gold evidence sentence ids for pair k of doc i; empty for
        pairs with no evidence supervision, which are skipped entirely (as is
        any doc with fewer than 2 sentences, since there's no negative left
        to contrast against).
    tau: softmax temperature.

    Averaged over every supervised pair in the batch; 0.0 (no gradient) if
    none exist.
    """
    losses = []
    for rs, sents, doc_evidence in zip(rs_list, sent_reprs, evidence):
        n_sent = sents.size(0)
        if n_sent < 2 or rs.size(0) == 0:
            continue
        sim = F.cosine_similarity(rs.unsqueeze(1), sents.unsqueeze(0), dim=-1) / tau  # (n_pair, n_sent)
        for k, evi in enumerate(doc_evidence):
            valid = [s for s in evi if 0 <= s < n_sent]
            if not valid or len(valid) >= n_sent:
                continue  # no supervision, or no negatives left to contrast
            pos_mask = sim.new_zeros(n_sent, dtype=torch.bool)
            pos_mask[valid] = True
            row = sim[k]
            losses.append(torch.logsumexp(row, dim=0) - torch.logsumexp(row[pos_mask], dim=0))
    if not losses:
        return rs_list[0].sum() * 0.0 if rs_list else torch.tensor(0.0)
    return torch.stack(losses).mean()


class BCEEvidenceContrastiveLoss(nn.Module):
    """Loss for the BERT -> ATLOP LCP -> entity-pair rep -> 2-layer edge-featured
    GAT -> 2-layer MLP classifier -> sigmoid pipeline:

        Loss = BCEWithLogitsLoss + 0.2 x Evidence Contrastive Loss

    This pipeline ends in a plain sigmoid, not ATLoss's learned per-example
    threshold, so class 0 (Na) can't be a TH logit to rank against -- it is
    excluded from the BCE target entirely and only used at prediction time:
    a pair is Na iff no real class (1..num_labels-1) clears `threshold`.
    """

    def __init__(self, evi_weight: float = 0.2, tau: float = 0.1, pos_weight: torch.Tensor = None):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.evi_weight = evi_weight
        self.tau = tau

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                rs_list: list[torch.Tensor] = None, sent_reprs: list[torch.Tensor] = None,
                evidence: list[list[list[int]]] = None) -> torch.Tensor:
        labels = labels.clone().float()
        labels[:, 0] = 0.0  # class 0 = Na, inferred by absence, not a direct BCE target
        loss = self.bce(logits[:, 1:], labels[:, 1:])
        if rs_list is not None and sent_reprs is not None and evidence is not None:
            loss = loss + self.evi_weight * evidence_contrastive_loss(rs_list, sent_reprs, evidence, tau=self.tau)
        return loss

    def get_label(self, logits: torch.Tensor, num_labels: int = -1, threshold: float = 0.5) -> torch.Tensor:
        """Multi-hot prediction from sigmoid probabilities (no TH class):
        class r (r>=1) is predicted iff sigmoid(logit_r) > threshold; if none
        clear it, the pair is predicted Na (class 0 set to 1). Mirrors
        ATLoss.get_label's contract so it's a drop-in for DocREModel-style
        `preds = loss_fnt.get_label(logits, num_labels=...)` call sites."""
        probs = torch.sigmoid(logits)
        mask = probs > threshold
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = mask & (logits >= top_v.unsqueeze(1))
        mask[:, 0] = False  # class 0 is never a direct sigmoid prediction
        output = mask.float()
        output[:, 0] = (output.sum(1) == 0).to(logits)
        return output
