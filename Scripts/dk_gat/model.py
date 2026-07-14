"""dk EGAT model: BERT encoder + ATLOP-style entity/context pooling +
2-layer Edge-featured GAT over a heterogeneous Entity+Sentence graph +
2-layer MLP classifier, trained with Adaptive Thresholding (ATLoss /
PUATLoss) + 0.2 x Evidence Contrastive Loss (InfoNCE over the document's
sentences).

Implements the user-specified proposed architecture (see README.md in this
folder for the full pipeline doc and the interpretation decisions):

  encoder      bert-base-cased (eager attention -- LCP needs attention maps),
               long docs handled by Scripts/atlop/long_input.process_long_input
  entity emb   mention token-span average -> mean over mentions (768)
  sentence emb mean over each sentence's own token span (768) -- same pooling
               shape as entity emb so both can sit in one node-feature matrix
  EGAT         2 layers over the heterogeneous graph (entity nodes + sentence
               nodes, see Scripts/dk_gat/preprocess_gat.py's module docstring
               for the edge scheme); attention score
               a^T [W h_i || W h_j || W_e e_ij] per head, edge embedding 32-d
               (edge category 8 + distance bucket 8 + type_i 8 + type_j 8);
               residual + LayerNorm each layer, GELU+dropout after layer 1
               only. Only the entity rows of the GAT output are kept
               afterwards -- sentence nodes exist purely to let two entities
               that never co-occur directly exchange information after 2
               layers by routing through a sentence node they both touch
               (e.g. Steve Jobs -[S1]- Apple -[S2]- California), which an
               entity-only graph couldn't do without a direct or same-type
               edge between them.
  Jump         element-wise MAX over {input embedding, layer-1 output,
  Knowledge    layer-2 output} instead of only using the last layer -- lets
               the classifier draw on 0/1/2-hop information per node rather
               than a fixed blend. Adds zero new parameters (deliberately --
               see forward()'s comment for why, given everything else added
               the same day already grew parameter count on a fixed-size
               annotated set).
  pair context ATLOP Localized Context Pooling from last-layer attention (768)
  pair repr    Linear([g_h ; g_t ; g_h*g_t ; c_ht], 3072 -> 768) -- the added
               element-wise product term lets the classifier see multiplicative
               head/tail feature interactions directly, not just their
               concatenation (standard relation-classification trick, akin to
               the bilinear/DistMult-style interaction ATLOP's own grouped
               bilinear classifier uses, but explicit here as a single term
               instead of a block-bilinear expansion). Optional (use_abs_diff,
               off by default): also append |g_h - g_t| (InferSent-style) --
               a magnitude-of-disagreement signal the product term doesn't
               capture. Both this and use_bilinear_classifier are alternative/
               additional ways of enriching the pair representation; only used
               when use_bilinear_classifier is off (that path bypasses pair_proj
               entirely).
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
the NA-vs-relation decision into the loss itself. The heterogeneous graph
and the g_h*g_t interaction term were added afterwards, once this fix was
confirmed to recover dev F1 into the expected range (46.58 distant-only,
see Scripts/dk_gat/README.md) -- kept as a separate, later change so the
two effects (loss fix vs. graph/interaction upgrade) aren't conflated.

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
                 evidence_tau: float = 0.1, offset: int = 1, loss_fnt: nn.Module = None,
                 use_jk: bool = True, use_gated_fusion: bool = False,
                 use_bilinear_classifier: bool = False, use_abs_diff: bool = False):
        super().__init__()
        self.config = config
        self.encoder = encoder
        hidden = config.hidden_size
        self.offset = offset
        self.num_labels = num_labels
        self.evidence_weight = evidence_weight
        self.evidence_tau = evidence_tau
        # Jump Knowledge on/off -- kept toggleable (not just documented) so it can be
        # A/B tested on CPU before being trusted as the pushed default, per the
        # project's "one variable at a time, validate before keeping" convention.
        self.use_jk = use_jk
        # Gated Fusion: learned per-dimension gate blending the GAT-refined entity
        # embedding with the original (pre-GAT) BERT entity embedding, instead of
        # JK's parameter-free max. Motivation: GAT message-passing can dilute a
        # strong entity representation by averaging it with less-informative
        # neighbors (e.g. "Obama" pulled toward "Apple"/"Company"/a sentence node);
        # a learned gate lets the model decide, per entity, how much to trust the
        # graph-refined signal vs. fall back to BERT's own representation, rather
        # than a fixed element-wise max or residual ratio. Takes priority over
        # use_jk when both would apply (see forward()) -- it's meant to supersede
        # JK's combination step, not stack with it, since both address the same
        # "how much of the original embedding to keep" question. Off by default
        # until A/B tested against the current use_jk=True default.
        self.use_gated_fusion = use_gated_fusion
        # ATLOP-style grouped bilinear classifier, as an alternative to the
        # concat+interaction+MLP pair representation below. Motivation: training
        # curves look healthy (steady loss decrease, no instability), so the gap
        # vs baseline may be a classifier-capacity bottleneck rather than a GAT/
        # graph problem -- ATLOP's grouped bilinear captures many more head/tail
        # cross-terms (block-wise outer products) than a single elementwise
        # g_h*g_t term. Supersedes (not stacks with) the interaction term, since
        # both serve the same "capture multiplicative interactions" purpose.
        self.use_bilinear_classifier = use_bilinear_classifier
        # abs(g_h - g_t): standard InferSent-style interaction term alongside the
        # existing g_h*g_t product. Product captures "both large/both small";
        # abs-diff captures magnitude of disagreement between head/tail features --
        # a different signal (e.g. two entities of very different "size"/salience
        # in context). Ignored when use_bilinear_classifier is on (that path doesn't
        # use the concat pair_proj at all). Off by default -- A/B test first.
        self.use_abs_diff = use_abs_diff
        if use_bilinear_classifier:
            self.emb_size = hidden
            self.block_size = 64
            assert self.emb_size % self.block_size == 0
            self.head_extractor = nn.Linear(hidden * 2, self.emb_size)
            self.tail_extractor = nn.Linear(hidden * 2, self.emb_size)
            self.bilinear = nn.Linear(self.emb_size * self.block_size, num_labels)
        if use_gated_fusion:
            self.gate_proj = nn.Linear(hidden * 2, hidden)

        self.cat_emb = nn.Embedding(EDGE_CATS, 8)
        self.dist_emb = nn.Embedding(NUM_DIST_BUCKETS, 8)
        self.type_emb = nn.Embedding(NUM_ENTITY_TYPES, 8)

        self.gat1 = EdgeFeaturedGATLayer(hidden, EDGE_EMB_DIM, num_heads, dropout, final_gelu=True)
        self.gat2 = EdgeFeaturedGATLayer(hidden, EDGE_EMB_DIM, num_heads, dropout, final_gelu=False)

        # [g_h ; g_t ; g_h*g_t ; c_ht] -> MLP path -- only built when NOT using the
        # bilinear classifier (mutually exclusive, see use_bilinear_classifier
        # above; no dead/unused params either way).
        if not use_bilinear_classifier:
            pair_dim = hidden * (5 if use_abs_diff else 4)
            self.pair_proj = nn.Linear(pair_dim, hidden)
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

    def _sentence_embeddings(self, seq_i, sent_spans_i):
        """Per-sentence embedding (token-span average) -- same pooling shape as
        _entities' mention pooling, so entity and sentence node features can
        sit in one (N_ent+N_sent, H) matrix for the heterogeneous GAT. Also
        reused (L2-normalized) as the sentence-embedding side of the evidence
        contrastive loss, computed once per doc instead of twice."""
        c = seq_i.size(0)
        embs = []
        for s, e in sent_spans_i:
            s, e = s + self.offset, min(e + self.offset, c)
            embs.append(seq_i[s:e].mean(0) if e > s else seq_i[0])
        return torch.stack(embs)  # (S, H)

    def _edge_embeddings(self, edge_cat, edge_dist, entity_types, device):
        cat = self.cat_emb(torch.as_tensor(edge_cat, dtype=torch.long, device=device))
        dist = self.dist_emb(torch.as_tensor(edge_dist, dtype=torch.long, device=device))
        types = self.type_emb(torch.as_tensor(entity_types, dtype=torch.long, device=device))
        n = types.size(0)
        ti = types.unsqueeze(1).expand(n, n, 8)
        tj = types.unsqueeze(0).expand(n, n, 8)
        return torch.cat([cat, dist, ti, tj], dim=-1)  # (N, N, 32)

    def _evidence_loss(self, context, hts_i, evidence_i, sent_embs, num_sents):
        """InfoNCE: pull each pair's LCP context toward its gold evidence
        sentences, against the document's other sentences. Skipped when the doc
        has no evidence annotations (e.g. train_distant). sent_embs: raw (not
        yet normalized) per-sentence embeddings from _sentence_embeddings,
        shared with the graph-node features so this isn't computed twice."""
        if not evidence_i:
            return None
        sent_embs = F.normalize(sent_embs, dim=-1)  # (S, H)

        pair_pos = {ht: k for k, ht in enumerate(hts_i)}
        losses = []
        for ht, ev in evidence_i.items():
            k = pair_pos.get(ht)
            if k is None:
                continue
            ev = [s for s in ev if s < num_sents]
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
        edge_cat, edge_dist, adj, entity_types, sent_spans, evidence,
        num_entities, num_sentences -- edge_cat/edge_dist/adj/entity_types are
        now sized (num_entities+num_sentences)^2, see preprocess_gat.py).
        Returns (loss, logits) in training, (logits,) otherwise; logits are
        concatenated over the batch in hts order."""
        sequence_output, attention = self.encode(input_ids, attention_mask)
        device = input_ids.device

        all_logits, ev_losses = [], []
        for i, f in enumerate(batch_features):
            seq_i, att_i = sequence_output[i], attention[i]
            n_ent = f["num_entities"]
            ent_emb, ent_att = self._entities(seq_i, att_i, f["entity_pos"])
            sent_emb = self._sentence_embeddings(seq_i, f["sent_spans"])
            node_emb = torch.cat([ent_emb, sent_emb], dim=0)  # (n_ent+n_sent, H)

            edge_emb = self._edge_embeddings(f["edge_cat"], f["edge_dist"], f["entity_types"], device)
            adj = torch.as_tensor(f["adj"], dtype=torch.bool, device=device)
            # Jump Knowledge: element-wise max over {input, layer-1, layer-2} instead of
            # only using the last layer's output. Each GAT layer already has its own
            # residual (x + out), so some "jumping" happens internally, but that only
            # gives a fixed blend of the previous layer into the next -- JK gives the
            # classifier direct, unmixed access to each hop's own representation so it
            # can lean on 0/1/2-hop information per-node rather than a fixed ratio.
            # Max (not concat+Linear) deliberately adds zero new parameters -- with the
            # heterogeneous graph and the pair interaction term added earlier today
            # already increasing parameter count on a fixed 3,053-doc annotated set,
            # this keeps the "did it help" signal attributable to the JK idea itself
            # rather than to extra capacity.
            h1 = self.gat1(node_emb, edge_emb, adj)
            h2 = self.gat2(h1, edge_emb, adj)
            if self.use_gated_fusion:
                # Learned per-dimension gate between the original entity embedding and
                # the GAT-refined one -- entity rows only (sentence nodes have no
                # "original identity" worth preserving this way, they're just
                # message-passing scaffolding). Supersedes JK's combination step.
                ent_orig = node_emb[:n_ent]
                ent_gat = h2[:n_ent]
                gate = torch.sigmoid(self.gate_proj(torch.cat([ent_orig, ent_gat], dim=-1)))
                g = gate * ent_gat + (1 - gate) * ent_orig
            elif self.use_jk:
                g_all = torch.stack([node_emb, h1, h2], dim=0).max(dim=0).values  # (n_ent+n_sent, H)
                g = g_all[:n_ent]  # sentence nodes only mediate message passing, drop them here
            else:
                g = h2[:n_ent]  # pre-JK behavior: last GAT layer output only

            ht = torch.as_tensor(f["hts"], dtype=torch.long, device=device).reshape(-1, 2)
            h_att = ent_att[ht[:, 0]]
            t_att = ent_att[ht[:, 1]]
            joint = h_att * t_att
            joint = joint / (joint.sum(1, keepdim=True) + 1e-30)
            context = joint @ seq_i  # (P, H) Localized Context Pooling

            g_h, g_t = g[ht[:, 0]], g[ht[:, 1]]
            if self.use_bilinear_classifier:
                # ATLOP-style grouped bilinear: head/tail each get their own
                # [entity; context] projection, then a block-wise outer product
                # captures many more cross-terms than a single g_h*g_t. Replaces
                # the interaction term entirely (see __init__ comment).
                hs = torch.tanh(self.head_extractor(torch.cat([g_h, context], dim=-1)))
                ts = torch.tanh(self.tail_extractor(torch.cat([g_t, context], dim=-1)))
                b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
                b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
                bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
                logits = self.bilinear(bl)
            else:
                parts = [g_h, g_t, g_h * g_t]
                if self.use_abs_diff:
                    parts.append(torch.abs(g_h - g_t))
                parts.append(context)
                pair = torch.cat(parts, dim=-1)  # (P, 4H or 5H)
                logits = self.classifier(self.pair_proj(pair))
            all_logits.append(logits)

            if labels is not None:
                ev = self._evidence_loss(context, f["hts"], f["evidence"], sent_emb, f["num_sentences"])
                if ev is not None:
                    ev_losses.append(ev)

        logits = torch.cat(all_logits, dim=0)
        if labels is None:
            return (logits,)
        loss = self.loss_fnt(logits, labels.to(logits))
        if ev_losses:
            loss = loss + self.evidence_weight * torch.stack(ev_losses).mean()
        return loss, logits
