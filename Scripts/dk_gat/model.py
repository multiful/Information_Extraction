"""dk EGAT model: BERT encoder + ATLOP-style entity/context pooling +
2-layer Edge-featured GAT + 2-layer MLP classifier, trained with
Adaptive Thresholding (ATLoss / PUATLoss) + 0.2 x Evidence Contrastive Loss
(InfoNCE over the document's sentences).

Implements the user-specified proposed architecture (see README.md in this
folder for the full pipeline doc and the interpretation decisions):

  encoder      bert-base-cased (eager attention -- LCP needs attention maps),
               long docs handled by Scripts/atlop/long_input.process_long_input
  entity emb   mention token-span average -> mean over mentions (768)
  pair context ATLOP Localized Context Pooling from last-layer attention (768)
  EGAT         2 layers over the entity graph; attention score
               a^T [W h_i || W h_j || W_e e_ij] per head, edge embedding 32-d
               (edge category 8 + distance bucket 8 + type_i 8 + type_j 8);
               residual + LayerNorm each layer, GELU+dropout after layer 1 only
  pair repr    Linear([g_h ; g_t ; c_ht], 2304 -> 768)
  classifier   LayerNorm -> Linear(768,768) -> GELU -> Dropout -> Linear(768,97)
  loss         Adaptive Thresholding (class 0 = learned TH/Na, see
               Scripts/atlop/losses.py) -- train_gat.py injects PUATLoss
               (na_weight=0.7, TTM-RE-inspired) for the distant stage and
               plain ATLoss for annotated fine-tune, exactly like
               Scripts/atlop/train_re.py's distant_mode=pretrain flow
               + 0.2 x InfoNCE(c_ht vs sentence embeddings, positives = gold
               evidence sentences) for pairs that have evidence annotations
               (train_distant has none -> the term is silently skipped there)

Note: this file originally used BCEWithLogitsLoss + a dev-threshold sweep.
Switched to Adaptive Thresholding after a real run showed it measurably
underperforming (dev F1 24.77 after distant pretrain on 20k docs, vs 43.15
for RoBERTa+LCP+ATLoss on the same 20k subset, see
Scripts/models/EXPERIMENTS.md experiment 2) -- BCE + post-hoc threshold only
handles DocRED's 97% NA imbalance at decision time, not during training,
so positive-class logits stayed undertrained. Adaptive Thresholding bakes
the NA-vs-relation decision into the loss itself.

Prediction: ATLoss.get_label(logits) -- a relation is emitted iff its logit
exceeds the learned per-pair TH (class 0) logit, capped at top-4. No global
threshold to sweep.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from Scripts.atlop.long_input import process_long_input  # noqa: E402
from Scripts.atlop.losses import ATLoss                    # noqa: E402
from Scripts.dk_gat.preprocess_gat import (               # noqa: E402
    EDGE_CATS, NUM_DIST_BUCKETS, NUM_ENTITY_TYPES,
)

EDGE_EMB_DIM = 32  # 4 x 8


class EdgeFeaturedGATLayer(nn.Module):
    """One EGAT layer: multi-head attention over graph neighbors where the
    score also sees the edge embedding -- alpha_ij = softmax_j over
    LeakyReLU(a^T [W h_i || W h_j || e_ij]) -- followed by residual + LayerNorm
    (+ optional GELU/dropout, layer-1 only per the spec)."""

    def __init__(self, dim: int = 768, edge_dim: int = EDGE_EMB_DIM,
                 num_heads: int = 4, dropout: float = 0.1, final_gelu: bool = True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj = nn.Linear(dim, dim)
        self.att = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim + edge_dim))
        nn.init.xavier_uniform_(self.att)
        self.leaky = nn.LeakyReLU(0.2)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.final_gelu = final_gelu

    def forward(self, x: torch.Tensor, edge_emb: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x (N, dim) / edge_emb (N, N, edge_dim) / adj (N, N) bool (self-loops included)
        n = x.size(0)
        h = self.proj(x).view(n, self.num_heads, self.head_dim)
        hi = h.unsqueeze(1).expand(n, n, self.num_heads, self.head_dim)
        hj = h.unsqueeze(0).expand(n, n, self.num_heads, self.head_dim)
        e = edge_emb.unsqueeze(2).expand(n, n, self.num_heads, edge_emb.size(-1))
        scores = self.leaky((torch.cat([hi, hj, e], dim=-1) * self.att).sum(-1))  # (N, N, H)
        scores = scores.masked_fill(~adj.unsqueeze(-1), -1e30)
        alpha = self.dropout(torch.softmax(scores, dim=1))
        out = torch.einsum("ijh,jhd->ihd", alpha, h).reshape(n, -1)
        out = self.norm(x + out)  # residual + LayerNorm
        if self.final_gelu:
            out = self.dropout(F.gelu(out))
        return out


class DocREGATModel(nn.Module):
    def __init__(self, config, encoder, num_labels: int = 97, num_heads: int = 4,
                 dropout: float = 0.1, evidence_weight: float = 0.2,
                 evidence_tau: float = 0.1, offset: int = 1, loss_fnt: nn.Module = None):
        super().__init__()
        self.config = config
        self.encoder = encoder
        hidden = config.hidden_size
        self.offset = offset
        self.num_labels = num_labels
        self.evidence_weight = evidence_weight
        self.evidence_tau = evidence_tau

        self.cat_emb = nn.Embedding(EDGE_CATS, 8)
        self.dist_emb = nn.Embedding(NUM_DIST_BUCKETS, 8)
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, 8)

        self.gat1 = EdgeFeaturedGATLayer(hidden, EDGE_EMB_DIM, num_heads, dropout, final_gelu=True)
        self.gat2 = EdgeFeaturedGATLayer(hidden, EDGE_EMB_DIM, num_heads, dropout, final_gelu=False)

        self.pair_proj = nn.Linear(hidden * 3, hidden)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )
        # Adaptive Thresholding (ATLoss / PUATLoss), injected so train_gat can swap in
        # PUATLoss(na_weight=0.7) for the distant stage and plain ATLoss for annotated
        # fine-tune, exactly like Scripts/atlop/train_re.py -- NOT BCEWithLogitsLoss.
        # BCE + post-hoc threshold sweep only fixes the NA imbalance at decision time,
        # not during training, and measurably underperformed on this dataset (dev F1
        # 24.77 with BCE vs 43.15 for RoBERTa+LCP+ATLoss on the same 20k distant subset,
        # see Scripts/models/EXPERIMENTS.md experiment 2) -- adaptive thresholding builds
        # the NA-vs-relation decision into the loss itself instead of a fixed/swept cutoff.
        self.loss_fnt = loss_fnt if loss_fnt is not None else ATLoss()

    def encode(self, input_ids, attention_mask):
        start_tokens = [self.config.cls_token_id]
        end_tokens = [self.config.sep_token_id]
        return process_long_input(self.encoder, input_ids, attention_mask, start_tokens, end_tokens)

    def _entities(self, seq_i, att_i, entity_pos_i):
        """Per-entity embedding (mention token-span average, mean over mentions)
        and attention distribution (for LCP). seq_i (L, H), att_i (heads, L, L)."""
        c = seq_i.size(0)
        head_avg = att_i.mean(0)  # (L, L)
        embs, atts = [], []
        for mentions in entity_pos_i:
            m_embs, m_atts = [], []
            for start, end in mentions:
                s, e = start + self.offset, min(end + self.offset, c)
                if s < c and e > s:
                    m_embs.append(seq_i[s:e].mean(0))
                    m_atts.append(head_avg[s:e].mean(0))
            if m_embs:
                embs.append(torch.stack(m_embs).mean(0))
                atts.append(torch.stack(m_atts).mean(0))
            else:  # every mention truncated away -> CLS fallback
                embs.append(seq_i[0])
                atts.append(head_avg[0])
        return torch.stack(embs), torch.stack(atts)  # (N, H), (N, L)

    def _edge_embeddings(self, edge_cat, edge_dist, entity_types, device):
        cat = self.cat_emb(torch.as_tensor(edge_cat, dtype=torch.long, device=device))
        dist = self.dist_emb(torch.as_tensor(edge_dist, dtype=torch.long, device=device))
        types = self.type_emb(torch.as_tensor(entity_types, dtype=torch.long, device=device))
        n = types.size(0)
        ti = types.unsqueeze(1).expand(n, n, 8)
        tj = types.unsqueeze(0).expand(n, n, 8)
        return torch.cat([cat, dist, ti, tj], dim=-1)  # (N, N, 32)

    def _evidence_loss(self, context, hts_i, evidence_i, sent_spans_i, seq_i):
        """InfoNCE: pull each pair's LCP context toward its gold evidence
        sentences, against the document's other sentences. Skipped when the doc
        has no evidence annotations (e.g. train_distant)."""
        if not evidence_i:
            return None
        c = seq_i.size(0)
        sent_embs = []
        for s, e in sent_spans_i:
            s, e = s + self.offset, min(e + self.offset, c)
            sent_embs.append(seq_i[s:e].mean(0) if e > s else seq_i[0])
        sent_embs = F.normalize(torch.stack(sent_embs), dim=-1)  # (S, H)

        pair_pos = {ht: k for k, ht in enumerate(hts_i)}
        losses = []
        for ht, ev in evidence_i.items():
            k = pair_pos.get(ht)
            if k is None:
                continue
            ev = [s for s in ev if s < len(sent_spans_i)]
            if not ev:
                continue
            ctx = F.normalize(context[k], dim=-1)
            sims = sent_embs @ ctx / self.evidence_tau  # (S,)
            log_denom = torch.logsumexp(sims, dim=0)
            log_num = torch.logsumexp(sims[torch.as_tensor(ev, device=sims.device)], dim=0)
            losses.append(log_denom - log_num)
        if not losses:
            return None
        return torch.stack(losses).mean()

    def forward(self, input_ids, attention_mask, batch_features, labels=None):
        """batch_features: the raw per-doc feature dicts (entity_pos, hts,
        edge_cat, edge_dist, adj, entity_types, sent_spans, evidence).
        Returns (loss, logits) in training, (logits,) otherwise; logits are
        concatenated over the batch in hts order."""
        sequence_output, attention = self.encode(input_ids, attention_mask)
        device = input_ids.device

        all_logits, ev_losses = [], []
        for i, f in enumerate(batch_features):
            seq_i, att_i = sequence_output[i], attention[i]
            ent_emb, ent_att = self._entities(seq_i, att_i, f["entity_pos"])

            edge_emb = self._edge_embeddings(f["edge_cat"], f["edge_dist"], f["entity_types"], device)
            adj = torch.as_tensor(f["adj"], dtype=torch.bool, device=device)
            g = self.gat2(self.gat1(ent_emb, edge_emb, adj), edge_emb, adj)  # (N, H)

            ht = torch.as_tensor(f["hts"], dtype=torch.long, device=device).reshape(-1, 2)
            h_att = ent_att[ht[:, 0]]
            t_att = ent_att[ht[:, 1]]
            joint = h_att * t_att
            joint = joint / (joint.sum(1, keepdim=True) + 1e-30)
            context = joint @ seq_i  # (P, H) Localized Context Pooling

            pair = torch.cat([g[ht[:, 0]], g[ht[:, 1]], context], dim=-1)
            logits = self.classifier(self.pair_proj(pair))
            all_logits.append(logits)

            if labels is not None:
                ev = self._evidence_loss(context, f["hts"], f["evidence"], f["sent_spans"], seq_i)
                if ev is not None:
                    ev_losses.append(ev)

        logits = torch.cat(all_logits, dim=0)
        if labels is None:
            return (logits,)
        loss = self.loss_fnt(logits, labels.to(logits))
        if ev_losses:
            loss = loss + self.evidence_weight * torch.stack(ev_losses).mean()
        return loss, logits
