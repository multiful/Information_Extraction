"""GREP relation-extraction model.

Re-implemented from Zhang, Yan & Cheng, "Document-Level Relation Extraction
with Global Relations and Entity Pair Reasoning" (GREP), ACL Findings 2025
(https://aclanthology.org/2025.findings-acl.1002/, code:
https://github.com/yanyi74/GREP). Builds on the team's ATLOP reimplementation
(`Scripts.atlop`): GREP's document encoding (marker insertion, log-sum-exp
entity pooling, Localized Context Pooling -- paper Eq 1-4) is *identical* to
ATLOP's, so it's reused (`process_long_input`) rather than reimplemented from
scratch; `get_entity_and_pair_context` below is `DocREModel.get_hrt` with one
addition: it also returns the raw per-pair token-attention `q` (paper's
q^(s,o), Eq 3), which ATLOP computes but discards -- GREP's Evidence
Extraction module needs it.

On top of that shared encoding, GREP adds three modules (paper Fig. 2):
  1. EntityPairGraph        -- Eq 5-14: build a graph over (head,tail) pairs
     (edge k->j when k's tail == j's head) and run a small GAT+GCN stack so
     pairs can exchange information before the final classifier.
  2. Evidence Extraction    -- Eq 15-16: sum q^(s,o) over each sentence's
     tokens to get a per-sentence distribution u^(s,o), trained with a KL
     loss against the gold evidence sentences (`Scripts.grep.losses`).
  3. Global Relation Prediction -- Eq 17-19: a document-level multi-label
     classifier on the [CLS] token, whose (sigmoid) output is added to every
     pair's logits.

Implementation choices the paper leaves unspecified (see also README.md):
  - f^(s,o) (Eq 7) node-feature dim: ATLOP's grouped-bilinear trick produces
    an `emb_size*block_size`-dim vector; this is projected down to
    `node_dim` (default = encoder hidden size) before entering the graph.
  - Eq 9 as literally written aggregates `W^l f_j^{l-1}` (the *target* node's
    own features) inside the neighbor sum, which -- since attention weights
    sum to 1 -- collapses to just `W^l f_j^{l-1}` regardless of neighbors,
    making the attention a no-op. Implemented instead as standard GAT
    (aggregate `W^l f_k^{l-1}`, the *neighbor's* features), which is almost
    certainly what was intended.
  - Graph attention (Eq 8) head count: unspecified; 4 heads, Q/K shared
    across graph layers, W^l per layer.
  - Evidence-sentence selection for the Inference Fusion pseudo-document
    (Sec 4.6, used in `Scripts/grep/train_grep.py`): a sentence is kept if
    its u^(s,o) score is at least that pair's own mean sentence score
    (paper doesn't specify a threshold).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Scripts.atlop.losses import ATLoss
from Scripts.atlop.long_input import process_long_input
from Scripts.grep.losses import evidence_kl_loss


def build_pair_adjacency(hts: list[tuple[int, int]], n_ent: int, device) -> torch.Tensor:
    """adj[j, k] = True iff pair k is a graph-neighbor of pair j, i.e. k's
    tail == j's head (so the path k -> j is (a,b)=k then (b,c)=j, Sec 4.2)."""
    n = len(hts)
    pair_index = {ht: i for i, ht in enumerate(hts)}
    adj = torch.zeros(n, n, dtype=torch.bool, device=device)
    for j, (s, _o) in enumerate(hts):
        for r in range(n_ent):
            if r == s:
                continue
            k = pair_index.get((r, s))
            if k is not None:
                adj[j, k] = True
    return adj


class EntityPairGraph(nn.Module):
    """GAT+GCN over entity-pair nodes (paper Eq 8-9, `graph_layers` layers)."""

    def __init__(self, node_dim: int, num_layers: int = 2, num_heads: int = 4):
        super().__init__()
        assert node_dim % num_heads == 0, "node_dim must be divisible by num_heads"
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.layer_w = nn.ModuleList([nn.Linear(node_dim, node_dim) for _ in range(num_layers)])

    def forward(self, node_feats: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """node_feats: (n_pairs, node_dim). adj: (n_pairs, n_pairs) bool.
        Returns updated (n_pairs, node_dim)."""
        n, h, d = node_feats.size(0), self.num_heads, self.head_dim
        f = node_feats
        for layer in range(self.num_layers):
            q = self.q_proj(f).view(n, h, d)
            k = self.k_proj(f).view(n, h, d)
            scores = torch.einsum("jhd,khd->jkh", q, k) / (d ** 0.5)   # (n, n, h)
            scores = scores.masked_fill(~adj.unsqueeze(-1), float("-inf"))
            alpha = torch.softmax(scores, dim=1)
            alpha = torch.nan_to_num(alpha, nan=0.0)  # isolated node (no neighbors): zero contribution
            wf = self.layer_w[layer](f).view(n, h, d)
            agg = torch.einsum("jkh,khd->jhd", alpha, wf).reshape(n, -1)
            f = F.gelu(agg) + f  # Eq 9 residual
        return f


class GREPModel(nn.Module):
    def __init__(self, config, encoder, emb_size: int = 768, block_size: int = 64,
                 num_labels: int = 97, offset: int = 1, node_dim: int | None = None,
                 graph_layers: int = 2, graph_heads: int = 4,
                 alpha: float = 0.1, beta: float = 0.1):
        super().__init__()
        assert emb_size % block_size == 0, "emb_size must be divisible by block_size"
        self.config = config
        self.encoder = encoder
        self.hidden_size = config.hidden_size
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels
        self.offset = offset
        self.node_dim = node_dim or config.hidden_size
        self.alpha = alpha
        self.beta = beta
        self.loss_fnt = ATLoss()

        # Eq 5-7: initial pair embedding -> graph node feature.
        self.init_head_extractor = nn.Linear(2 * self.hidden_size, emb_size)
        self.init_tail_extractor = nn.Linear(2 * self.hidden_size, emb_size)
        self.node_proj = nn.Linear(emb_size * block_size, self.node_dim)

        self.graph = EntityPairGraph(self.node_dim, num_layers=graph_layers, num_heads=graph_heads)

        # Eq 10-14: final pair embedding -> relation logits.
        self.final_head_extractor = nn.Linear(2 * self.hidden_size + self.node_dim, emb_size)
        self.final_tail_extractor = nn.Linear(2 * self.hidden_size + self.node_dim, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, num_labels)

        # Eq 17: global relation prediction (96 real relations; TH/class-0 excluded).
        self.doc_classifier = nn.Linear(self.hidden_size, num_labels - 1)

    def encode(self, input_ids, attention_mask):
        start_tokens = [self.config.cls_token_id]
        end_tokens = [self.config.sep_token_id]
        return process_long_input(self.encoder, input_ids, attention_mask, start_tokens, end_tokens)

    def _group_bilinear(self, zs: torch.Tensor, zt: torch.Tensor) -> torch.Tensor:
        """ATLOP-style grouped bilinear: (n, emb_size) x (n, emb_size) -> (n, emb_size*block_size)."""
        b1 = zs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = zt.view(-1, self.emb_size // self.block_size, self.block_size)
        return (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)

    def get_entity_and_pair_context(self, sequence_output, attention, entity_pos, hts):
        """`DocREModel.get_hrt` (Scripts/atlop/re_model.py), extended to also
        return the raw per-pair token-attention `q` (Eq 3) needed by the
        Evidence Extraction module. Returns per-doc lists (kept separate,
        rather than concatenated like ATLOP does, so the graph/evidence
        modules can operate per-document)."""
        offset = self.offset
        _, num_heads, _, c = attention.size()
        hss, tss, rss, qs = [], [], [], []

        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for mentions in entity_pos[i]:
                if len(mentions) > 1:
                    m_emb, m_att = [], []
                    for start, _end in mentions:
                        if start + offset < c:
                            m_emb.append(sequence_output[i, start + offset])
                            m_att.append(attention[i, :, start + offset])
                    if m_emb:
                        e_emb = torch.logsumexp(torch.stack(m_emb, dim=0), dim=0)
                        e_att = torch.stack(m_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(num_heads, c).to(attention)
                else:
                    start, _end = mentions[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.hidden_size).to(sequence_output)
                        e_att = torch.zeros(num_heads, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)   # (n_ent, hidden)
            entity_atts = torch.stack(entity_atts, dim=0)   # (n_ent, heads, seq)

            ht_i = torch.as_tensor(hts[i], dtype=torch.long,
                                   device=sequence_output.device).reshape(-1, 2)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            q = (h_att * t_att).mean(1)                       # Eq 3, averaged over heads
            q = q / (q.sum(1, keepdim=True) + 1e-30)          # Eq 3, renormalized -> (n_pair, seq)
            rs = torch.matmul(q, sequence_output[i])          # Eq 4 -> (n_pair, hidden)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
            qs.append(q)

        return hss, rss, tss, qs

    def sentence_attention(self, q_i: torch.Tensor, sent_pos_i: list[tuple[int, int]]) -> torch.Tensor:
        """Eq 15: sum a pair's token-attention `q_i` over each sentence's
        tokens. sent_pos_i uses the same pre-[CLS]-offset coordinates as
        entity_pos (Scripts/atlop/preprocess.py). Returns u_i: (n_pair, n_sent)."""
        offset = self.offset
        seq_len = q_i.size(1)
        sums = []
        for start, end in sent_pos_i:
            lo, hi = start + offset, min(end + offset, seq_len)
            if hi <= lo:
                sums.append(torch.zeros(q_i.size(0), device=q_i.device, dtype=q_i.dtype))
            else:
                sums.append(q_i[:, lo:hi].sum(dim=1))
        return torch.stack(sums, dim=1)

    def forward(self, input_ids, attention_mask, entity_pos, hts, sent_pos=None,
                evidence=None, doc_rel_labels=None, labels=None, use_evidence_loss=True):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hss, rss, tss, qs = self.get_entity_and_pair_context(sequence_output, attention, entity_pos, hts)

        all_logits, all_u, doc_probs = [], [], []
        for i in range(len(entity_pos)):
            hs, rs, ts, q = hss[i], rss[i], tss[i], qs[i]
            n_ent = len(entity_pos[i])

            # Eq 5-7: initial pair embedding -> graph node feature.
            zs0 = torch.tanh(self.init_head_extractor(torch.cat([hs, rs], dim=1)))
            zt0 = torch.tanh(self.init_tail_extractor(torch.cat([ts, rs], dim=1)))
            f0 = self.node_proj(self._group_bilinear(zs0, zt0))

            # Eq 8-9: entity pair graph reasoning.
            adj = build_pair_adjacency(hts[i], n_ent, f0.device)
            f_update = self.graph(f0, adj)

            # Eq 10-14: final pair embedding -> relation logits p^(s,o).
            zs = torch.tanh(self.final_head_extractor(torch.cat([hs, f_update, rs], dim=1)))
            zt = torch.tanh(self.final_tail_extractor(torch.cat([ts, f_update, rs], dim=1)))
            pair_logits = self.bilinear(self._group_bilinear(zs, zt))

            # Eq 17 + 19: global relation prediction, fused into pair logits.
            cls_hidden = sequence_output[i, 0]
            doc_prob_i = torch.sigmoid(self.doc_classifier(cls_hidden))   # (num_labels-1,)
            doc_prob_padded = F.pad(doc_prob_i, (1, 0))                    # TH slot -> 0
            fused_logits = pair_logits + doc_prob_padded.unsqueeze(0)

            all_logits.append(fused_logits)
            doc_probs.append(doc_prob_i)
            all_u.append(self.sentence_attention(q, sent_pos[i]) if sent_pos is not None else None)

        logits = torch.cat(all_logits, dim=0)
        preds = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        # `all_u` (per-doc, ragged) and raw `logits` (flat across docs, like
        # `preds`) are exposed for Scripts/grep/train_grep.py's Inference
        # Fusion phase (Sec 4.6), which needs continuous scores + evidence
        # distributions, not just the thresholded prediction.
        output = (preds, all_u, logits)

        if labels is not None:
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels, dtype=torch.float)
            labels = labels.to(dtype=torch.float, device=logits.device)
            loss = self.loss_fnt(logits, labels)

            if doc_rel_labels is not None:
                doc_losses = []
                for i, dp in enumerate(doc_probs):
                    target = torch.zeros(self.num_labels - 1, device=logits.device, dtype=dp.dtype)
                    for r in doc_rel_labels[i]:
                        target[r - 1] = 1.0  # doc_rel_labels are 1..96 (class 0/TH excluded)
                    doc_losses.append(F.binary_cross_entropy(dp.clamp(1e-6, 1 - 1e-6), target))
                loss = loss + self.alpha * torch.stack(doc_losses).mean()

            if use_evidence_loss and evidence is not None:
                evi_losses = [
                    l for u_i, ev_i in zip(all_u, evidence)
                    if u_i is not None and (l := evidence_kl_loss(u_i, ev_i)) is not None
                ]
                if evi_losses:
                    loss = loss + self.beta * torch.stack(evi_losses).mean()

            output = (loss,) + output

        return output
