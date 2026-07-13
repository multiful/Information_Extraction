"""dk's personal RE model: RoBERTa encoder + Mention Average Pooling entity
representation + Localized Context Pooling + Adaptive Thresholding loss,
plus an optional bucketed sentence-distance embedding.

Localized Context Pooling and Adaptive Thresholding are the two techniques
ATLOP (Zhou et al., AAAI 2021) is built around -- see PRD.md section 3. This
combines them with a simpler Mention Average Pooling entity representation
(no entity markers, no logsumexp pooling), so it is close to but not
identical to a full ATLOP reproduction -- intentional, since it's being
compared against the team's separate ATLOP track.

The per-pair computation (localized context, head/tail projection, classify)
is fully vectorized across all of a document's candidate pairs in one batched
op each, instead of a Python loop over pairs -- that loop was the actual
runtime bottleneck (~380 pairs/doc, each doing several tiny tensor ops) and
was one of two causes of MPS out-of-memory crashes (many small
variably-shaped allocations fragment its caching allocator). The second
cause -- HF's eager attention fallback on every layer when
output_attentions=True is requested -- is addressed in encode() below by
only recomputing the last layer's attention manually, leaving the rest on
fast SDPA (see encode()'s docstring for the mechanism and verification).

New file. Does not modify any shared module under data/.
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaATLOPLiteModel(nn.Module):
    def __init__(self, model_name: str = "roberta-base", num_class: int = 97, hidden_dim: int = 256,
                 dropout: float = 0.1, use_dist_embedding: bool = False,
                 num_dist_buckets: int = 6, dist_emb_dim: int = 20):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        emb_size = self.encoder.config.hidden_size  # 768 for roberta-base

        # pair representation = [entity_emb; localized_context] -> proj -> tanh, for head and tail
        # separately (ATLOP-style asymmetric head/tail extractors), then concat (+ optional
        # distance embedding) -> classifier.
        # RoBERTa itself already has internal dropout (HF default), but head_proj/tail_proj/
        # classifier are freshly-initialized task-specific layers being fully fine-tuned on a
        # small doc count -- the most overfitting-prone part of this model -- so they get their
        # own dropout too. Automatically inert during model.eval().
        self.head_proj = nn.Linear(emb_size * 2, hidden_dim)
        self.tail_proj = nn.Linear(emb_size * 2, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

        # Optional: bucketed min sentence-distance between head/tail entities. Merges the
        # PRD's separate "Distance Embedding" and "Sentence Position Embedding" ideas into one
        # feature (bucket 0 = same sentence, higher buckets = further apart) since both describe
        # the same underlying signal. See dk_pairs.py::compute_dist_buckets for bucket boundaries.
        self.use_dist_embedding = use_dist_embedding
        classifier_in = hidden_dim * 2
        if use_dist_embedding:
            self.dist_embedding = nn.Embedding(num_dist_buckets, dist_emb_dim)
            classifier_in += dist_emb_dim
        self.classifier = nn.Linear(classifier_in, num_class)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Only the last layer's attention is ever used (see _entity_attention). Requesting
        output_attentions=True at the model level makes HF fall back from fast SDPA to eager
        attention on EVERY layer (SDPA can't return attention weights) -- confirmed via
        transformers 4.57.6 source (RobertaSdpaSelfAttention.forward: falls back to
        super().forward() per-call whenever output_attentions=True is passed to it) that this
        fallback happens independently per layer, not as a single model-wide switch. That means
        requesting attentions only forces the *specific layers asked*, so leaving the other 11
        layers on fast SDPA and manually recomputing just the last layer's attention in eager
        mode (from its own query/key weights) gives the same result at ~1/12th the eager-mode
        memory footprint -- verified numerically identical to the old output_attentions=True
        path (max diff ~3e-6 on a real forward pass). This is the fix for the MPS OOM that
        vectorizing the pair loop alone didn't solve (eager attention on MPS, not the loop, was
        the actual second cause)."""
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=False,
            output_hidden_states=True,
        )
        sequence_output = outputs.last_hidden_state  # (B, L, H)
        extended_mask = self.encoder.get_extended_attention_mask(attention_mask, input_ids.shape)
        last_layer = self.encoder.encoder.layer[-1]
        _, attention = last_layer.attention(
            outputs.hidden_states[-2], attention_mask=extended_mask, output_attentions=True
        )  # (B, num_heads, L, L)
        return sequence_output, attention

    @staticmethod
    def _mention_average(sequence_output_doc: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
        """spans: (start, end) subword ranges for one entity's mentions. Mean over all
        their tokens. Falls back to the CLS (position 0) vector for the rare case
        (~0.5-0.8% of docs, see EDA) where every mention of this entity fell outside
        a truncated document -- avoids crashing on an empty span list."""
        vecs = [sequence_output_doc[s:e].mean(dim=0) for s, e in spans if e > s]
        if not vecs:
            return sequence_output_doc[0]
        return torch.stack(vecs, dim=0).mean(dim=0)

    def _entity_embeddings(self, sequence_output_doc: torch.Tensor, entity_pos_doc: list) -> torch.Tensor:
        """entity_pos_doc: list[entity][mention] = (start, end). Returns (num_entities, H).
        Looped per-entity (~19.5/doc on average, see EDA) -- ~20x fewer iterations than the
        per-pair loop this file used to have, so left as-is rather than also vectorizing."""
        return torch.stack(
            [self._mention_average(sequence_output_doc, mentions) for mentions in entity_pos_doc],
            dim=0,
        )

    @staticmethod
    def _entity_attention(attention_doc: torch.Tensor, entity_pos_doc: list) -> torch.Tensor:
        """Average attention-from-token over an entity's mentions and over heads.
        attention_doc: (num_heads, L, L) for one doc. Returns (num_entities, L) --
        how much each entity 'attends to' each sequence position."""
        head_avg = attention_doc.mean(dim=0)  # (L, L)
        entity_att = []
        for mentions in entity_pos_doc:
            rows = [head_avg[s:e].mean(dim=0) for s, e in mentions if e > s]
            entity_att.append(head_avg[0] if not rows else torch.stack(rows, dim=0).mean(dim=0))
        return torch.stack(entity_att, dim=0)  # (num_entities, L)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                entity_pos: list, hts: list, dist_buckets: Optional[list] = None) -> list[torch.Tensor]:
        """
        entity_pos:   list[doc] of list[entity] of list[(start,end)]  -- ragged, per doc
        hts:          list[doc] of list[(h_idx, t_idx)]               -- candidate pairs per doc
        dist_buckets: list[doc] of list[int], same length/order as hts[doc] -- required iff
                      use_dist_embedding=True (from dk_pairs.py::compute_dist_buckets)
        Returns: list[doc] of Tensor(num_pairs_doc, num_class) logits
        """
        sequence_output, attention = self.encode(input_ids, attention_mask)
        all_logits = []
        for i in range(input_ids.size(0)):
            seq_i = sequence_output[i]  # (L, H)
            att_i = attention[i]         # (num_heads, L, L)

            if not hts[i]:
                all_logits.append(torch.zeros(0, self.classifier.out_features, device=input_ids.device))
                continue

            ent_emb = self._entity_embeddings(seq_i, entity_pos[i])  # (num_ent, H)
            ent_att = self._entity_attention(att_i, entity_pos[i])    # (num_ent, L)

            # --- everything below is vectorized across all P pairs of this doc at once ---
            h_idx = torch.tensor([h for h, _ in hts[i]], device=input_ids.device)
            t_idx = torch.tensor([t for _, t in hts[i]], device=input_ids.device)

            h_emb, t_emb = ent_emb[h_idx], ent_emb[t_idx]  # (P, H) each
            h_att, t_att = ent_att[h_idx], ent_att[t_idx]  # (P, L) each

            joint = h_att * t_att                                          # (P, L)
            joint = joint / (joint.sum(dim=-1, keepdim=True) + 1e-10)
            context = torch.matmul(joint, seq_i)                           # (P, L) @ (L, H) -> (P, H)

            h_vec = self.dropout(self.activation(self.head_proj(torch.cat([h_emb, context], dim=-1))))
            t_vec = self.dropout(self.activation(self.tail_proj(torch.cat([t_emb, context], dim=-1))))
            pair_repr = torch.cat([h_vec, t_vec], dim=-1)                  # (P, 2*hidden_dim)

            if self.use_dist_embedding:
                buckets = torch.tensor(dist_buckets[i], device=input_ids.device)
                pair_repr = torch.cat([pair_repr, self.dist_embedding(buckets)], dim=-1)

            all_logits.append(self.classifier(pair_repr))  # (P, num_class)
        return all_logits


class ATLoss(nn.Module):
    """Adaptive Thresholding loss (Zhou et al., AAAI 2021 ATLOP). Reimplemented
    from the mechanism described in PRD.md section 3 (two masked log-softmax
    terms), not copied from any external repo. Class index 0 is the learned
    TH ("no relation") class -- inputs never set it as a positive label."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.clone()
        labels[:, 0] = 0.0  # TH column is never itself a positive target

        th_label = torch.zeros_like(labels)
        th_label[:, 0] = 1.0

        p_mask = labels + th_label  # term 1 keeps: true positives + TH
        n_mask = 1 - labels          # term 2 keeps: everything except true positives (incl. TH)

        # rank true positive relations above TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(torch.log_softmax(logit1, dim=-1) * labels).sum(dim=-1)

        # rank TH above all negative (non-positive) relations
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(torch.log_softmax(logit2, dim=-1) * th_label).sum(dim=-1)

        return (loss1 + loss2).mean()

    @staticmethod
    def get_label(logits: torch.Tensor, num_labels: int = 4) -> torch.Tensor:
        """Decode predicted positive classes: logit > TH(class 0) logit, capped
        at the top `num_labels` scoring classes per pair (set num_labels<=0 to
        disable the cap)."""
        th_logit = logits[:, 0].unsqueeze(1)
        mask = logits > th_logit
        mask[:, 0] = False
        if num_labels > 0:
            top_v, _ = torch.topk(logits, min(num_labels, logits.size(-1)), dim=-1)
            floor = top_v[:, -1].unsqueeze(1)
            mask = mask & (logits >= floor)
        return mask


class PUATLoss(ATLoss):
    """Approximate PU (positive-unlabeled)-learning adaptation of ATLoss, for
    training on train_distant. Inspired by TTM-RE (2024)'s use of a PU loss to
    handle distant supervision's core problem -- pairs distant matching labeled
    "Na" aren't confirmed negatives, they're *unlabeled* (some are true relations
    Wikidata/entity-linking just missed).

    This is NOT a faithful reproduction of TTM-RE's actual nnPU risk estimator
    (which needs a separately-estimated class prior and a non-negative risk
    correction term) or its Token Turing Machine memory module -- just the core
    intuition, cheaply: down-weight the "rank TH above everything else" loss
    term specifically for pairs with zero positive labels (distant-Na pairs),
    since we shouldn't fully trust that label. Pairs with >=1 distant-positive
    label are left untouched (false positives are a smaller concern than missed
    positives in distant RE, so only the Na side is discounted)."""

    def __init__(self, na_weight: float = 0.5):
        super().__init__()
        self.na_weight = na_weight

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.clone()
        labels[:, 0] = 0.0

        th_label = torch.zeros_like(labels)
        th_label[:, 0] = 1.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(torch.log_softmax(logit1, dim=-1) * labels).sum(dim=-1)

        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(torch.log_softmax(logit2, dim=-1) * th_label).sum(dim=-1)

        is_distant_na = labels.sum(dim=-1) == 0
        weight2 = torch.where(
            is_distant_na, torch.full_like(loss2, self.na_weight), torch.ones_like(loss2)
        )

        return (loss1 + weight2 * loss2).mean()
