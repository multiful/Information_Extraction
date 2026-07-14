"""[통합 모델 재구축] 이미지 파이프라인 그대로의 DocRE 모델 (DocREModelFull).

    Input Document
         │
    BERT-base Encoder
         │
    ├────────────────────────────┬──────────────────────────┐
    ▼                            ▼
  [GAT (Node-level)]           [DREAM (Doc-level)]
   Entity/Mention/Sentence      Document Reasoning
   이종 그래프, 구조적 이웃 통합    증거 문장 탐색 (Evidence)
    │                            │
    ▼                            │
  [ATLOP Localized Context]      │
   GAT 강화 엔티티 표현 활용        │
   Entity Pair h(i,j)_GAT        │
    └───────────┬────────────────┘
                ▼
      [Attention / Gated Fusion]  ── 구조 정보 ⊕ 증거 정보
                ▼
        2-Layer MLP Classifier
                ▼
      AT-Loss  +  λ·Evidence Contrastive Loss

기존 통합버전(entity-pair 그래프 + DREEAM 게이트)과는 무관하게, 위 틀 안에서
관계 연결성/성능을 노린 새 설계다. baseline 파일(re_model.py)은 무수정 —
process_long_input / ATLoss 만 import 로 재사용하고, 인코더는 self.encoder 로
같은 키(`encoder.*`)를 써서 baseline 체크포인트로 warm-start 가능하다.

핵심 아이디어
-------------
1. Node-level 이종 그래프 (GAT).
   문서마다 Mention·Entity·Sentence 노드를 만들고 6종 타입 엣지로 잇는다
   (mention-entity 소속 / mention-sentence 포함 / entity-sentence / sentence-
   sentence 인접 / mention-mention coref(같은 엔티티) / mention-mention 공기(같은
   문장)). GAT 로 전파하면 한 엔티티의 흩어진 멘션·문맥 문장이 서로를 읽어,
   ATLOP 이 독립 분류하던 (h,t) 쌍이 문서 구조를 반영한 표현을 갖는다
   (multi-hop·coreference 연결성).

2. ATLOP Localized Context (GAT 강화 엔티티).
   GAT 로 강화된 엔티티 노드를 head/tail 로 쓰고, 원 attention 곱(ATLOP LCP)으로
   문맥 벡터를 뽑아 쌍 표현 h(i,j)_GAT 를 만든다.

3. DREAM (Doc-level) 증거 분기.
   문장 노드에 문서 수준 self-attention(reasoning)을 한 번 더 얹고, 쌍 쿼리로
   문장을 스코어링해 증거 분포 p_evi 와 증거 문맥 c_evi 를 얻는다. p_evi 는 gold
   evidence 로 contrastive 지도학습된다(아래 4).

4. Attention/Gated Fusion → 2-Layer MLP.
   구조 표현(h(i,j)_GAT)과 증거 표현(c_evi)을 학습형 게이트로 섞고 2-layer MLP 로
   분류한다.

5. AT-Loss + λ·Evidence Contrastive Loss.
   관계 손실은 주입된 loss_fnt(distant=PUATLoss, annotated=ATLoss). 증거 손실은
   문장에 대한 multi-positive InfoNCE(gold evidence 문장=양성) — annotated 에만
   evidence 가 있으므로 distant 단계에서는 자동으로 0.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph_layers import HeteroGATLayer
from .long_input import process_long_input
from .losses import ATLoss

# heterogeneous edge types (symmetric, self-loops excluded)
ME = 0        # mention <-> entity   (mention belongs to entity)
MS = 1        # mention <-> sentence (mention located in sentence)
ES = 2        # entity  <-> sentence (entity has a mention in sentence)
SS = 3        # sentence<-> sentence (adjacent sentences)
MM_COREF = 4  # mention <-> mention  (same entity -> coreference)
MM_COOC = 5   # mention <-> mention  (same sentence -> co-occurrence)
N_HETERO_EDGES = 6


class DocREModelFull(nn.Module):
    """Rebuilt integrated model following the PNG pipeline (BERT node-GAT +
    DREAM evidence + gated fusion + MLP head, AT-Loss + evidence contrastive)."""

    def __init__(self, config, encoder, emb_size: int = 768, block_size: int = 64,
                 num_labels: int = 97, offset: int = 1, loss_fnt=None,
                 graph_layers: int = 2, graph_dim: int = 256, graph_heads: int = 4,
                 graph_dropout: float = 0.1, evi_lambda: float = 0.1,
                 mlp_dropout: float = 0.2):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.hidden_size = config.hidden_size
        # injectable so train_full can swap PUATLoss in for distant pretraining.
        self.loss_fnt = loss_fnt if loss_fnt is not None else ATLoss()

        self.emb_size = emb_size
        self.block_size = block_size          # kept for CLI compat (MLP head unused)
        self.num_labels = num_labels
        self.offset = offset                  # skip leading [CLS] when indexing markers
        self.graph_dim = graph_dim
        self.evi_lambda = evi_lambda

        H = config.hidden_size

        # --- (1) node input projections + node-type embeddings -> graph_dim ---
        self.mention_in = nn.Linear(H, graph_dim)
        self.entity_in = nn.Linear(H, graph_dim)
        self.sent_in = nn.Linear(H, graph_dim)
        self.node_type_emb = nn.Parameter(torch.zeros(3, graph_dim))  # [mention, entity, sent]

        # --- (1) heterogeneous node GAT ---
        self.hetero_gnn = nn.ModuleList(
            HeteroGATLayer(graph_dim, N_HETERO_EDGES, heads=graph_heads, dropout=graph_dropout)
            for _ in range(graph_layers)
        )

        # --- (2) ATLOP localized context -> struct pair representation ---
        self.head_ext = nn.Linear(graph_dim + H, emb_size)
        self.tail_ext = nn.Linear(graph_dim + H, emb_size)
        self.struct_proj = nn.Linear(2 * emb_size, emb_size)

        # --- (3) DREAM doc-level sentence reasoning + evidence attention ---
        self.sent_reason = nn.MultiheadAttention(
            graph_dim, graph_heads, dropout=graph_dropout, batch_first=True)
        self.sent_norm = nn.LayerNorm(graph_dim)
        self.evi_q = nn.Linear(2 * graph_dim, emb_size)
        self.evi_k = nn.Linear(graph_dim, emb_size)
        self.evi_v = nn.Linear(graph_dim, emb_size)

        # --- (4) gated fusion of struct (GAT) and evidence (DREAM) ---
        self.fuse_gate = nn.Linear(2 * emb_size, emb_size)

        # --- (4) 2-layer MLP classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(emb_size, num_labels),
        )

    # ------------------------------------------------------------------ encode
    def encode(self, input_ids, attention_mask):
        start_tokens = [self.config.cls_token_id]
        end_tokens = [self.config.sep_token_id]
        return process_long_input(
            self.encoder, input_ids, attention_mask, start_tokens, end_tokens
        )

    # --------------------------------------------------- per-document node graph
    def _doc_nodes(self, seq_i, att_i, entity_pos_i, sent_pos_i):
        """Build one document's heterogeneous node features + typed adjacency.

        Returns
          x0         : (N, graph_dim) initial node features [mentions|entities|sents]
          adj        : (N_HETERO_EDGES, N, N) bool typed adjacency
          entity_att : (n_e, heads, c) per-entity token attention (for LCP)
          counts     : (n_m, n_e, n_s)
        Node order: mentions [0:n_m), entities [n_m:n_m+n_e), sentences [.. :N).
        """
        device = seq_i.device
        H = self.hidden_size
        num_heads, c, _ = att_i.size()
        offset = self.offset

        # sentence token spans in +offset space, clamped to the encoded length
        sent_spans = []
        for (st, en) in sent_pos_i:
            a, b = st + offset, min(en + offset, c)
            sent_spans.append((a, b))
        n_s = len(sent_spans)

        def which_sent(tok):
            for s, (a, b) in enumerate(sent_spans):
                if a <= tok < b:
                    return s
            return 0

        mention_emb, mention2entity, mention2sent = [], [], []
        entity_emb, entity_att, entity2sents = [], [], []
        for e, mentions in enumerate(entity_pos_i):
            m_emb_list, m_att_list, sents_of_e = [], [], set()
            for (start, _end) in mentions:
                pos = start + offset
                if pos < c:
                    emb = seq_i[pos]
                    at = att_i[:, pos]
                    s = which_sent(pos)
                else:
                    emb = seq_i.new_zeros(H)
                    at = att_i.new_zeros(num_heads, c)
                    s = 0
                mention_emb.append(emb)
                mention2entity.append(e)
                mention2sent.append(s)
                sents_of_e.add(s)
                m_emb_list.append(emb)
                m_att_list.append(at)
            if m_emb_list:
                e_emb = torch.logsumexp(torch.stack(m_emb_list, 0), 0)
                e_att = torch.stack(m_att_list, 0).mean(0)
            else:
                e_emb = seq_i.new_zeros(H)
                e_att = att_i.new_zeros(num_heads, c)
            entity_emb.append(e_emb)
            entity_att.append(e_att)
            entity2sents.append(sents_of_e)

        n_m, n_e = len(mention_emb), len(entity_emb)

        mention_emb = torch.stack(mention_emb, 0) if n_m else seq_i.new_zeros(0, H)
        entity_emb = torch.stack(entity_emb, 0)
        entity_att = torch.stack(entity_att, 0)
        sent_emb = torch.stack(
            [seq_i[a:b].mean(0) if a < b else seq_i.new_zeros(H) for (a, b) in sent_spans], 0
        ) if n_s else seq_i.new_zeros(0, H)

        # node features (+ type embedding)
        xm = self.mention_in(mention_emb) + self.node_type_emb[0]
        xe = self.entity_in(entity_emb) + self.node_type_emb[1]
        xs = self.sent_in(sent_emb) + self.node_type_emb[2]
        x0 = torch.cat([xm, xe, xs], 0)

        # typed adjacency
        N = n_m + n_e + n_s
        e_off, s_off = n_m, n_m + n_e
        adj = torch.zeros(N_HETERO_EDGES, N, N, dtype=torch.bool, device=device)
        for m in range(n_m):
            e, s = mention2entity[m], mention2sent[m]
            adj[ME, m, e_off + e] = adj[ME, e_off + e, m] = True
            adj[MS, m, s_off + s] = adj[MS, s_off + s, m] = True
        if n_m:
            me = torch.tensor(mention2entity, device=device)
            ms = torch.tensor(mention2sent, device=device)
            eye_m = torch.eye(n_m, dtype=torch.bool, device=device)
            adj[MM_COREF, :n_m, :n_m] = (me.unsqueeze(0) == me.unsqueeze(1)) & ~eye_m
            adj[MM_COOC, :n_m, :n_m] = (ms.unsqueeze(0) == ms.unsqueeze(1)) & ~eye_m
        for e in range(n_e):
            for s in entity2sents[e]:
                adj[ES, e_off + e, s_off + s] = adj[ES, s_off + s, e_off + e] = True
        for s in range(n_s - 1):
            adj[SS, s_off + s, s_off + s + 1] = adj[SS, s_off + s + 1, s_off + s] = True

        return x0, adj, entity_att, (n_m, n_e, n_s)

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids, attention_mask, entity_pos, hts, sent_pos,
                labels=None, evidence=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)

        all_logits, evi_scores_list = [], []
        for i in range(len(entity_pos)):
            seq_i = sequence_output[i]
            x0, adj, entity_att, (n_m, n_e, n_s) = self._doc_nodes(
                seq_i, attention[i], entity_pos[i], sent_pos[i]
            )

            # (1) heterogeneous node GAT
            x = x0
            for layer in self.hetero_gnn:
                x = layer(x, adj)
            ent_enh = x[n_m: n_m + n_e]                     # (n_e, graph_dim)
            sent_enh = x[n_m + n_e:]                        # (n_s, graph_dim)

            # (3) DREAM: doc-level sentence self-attention (reasoning)
            sr, _ = self.sent_reason(sent_enh.unsqueeze(0), sent_enh.unsqueeze(0),
                                     sent_enh.unsqueeze(0))
            sent_reasoned = self.sent_norm(sent_enh + sr.squeeze(0))   # (n_s, graph_dim)

            ht = torch.as_tensor(hts[i], dtype=torch.long, device=seq_i.device).reshape(-1, 2)
            if ht.numel() == 0:
                evi_scores_list.append(None)
                continue
            h_enh = ent_enh.index_select(0, ht[:, 0])       # (n_pair, graph_dim)
            t_enh = ent_enh.index_select(0, ht[:, 1])

            # (2) ATLOP localized context on the ORIGINAL entity attentions
            h_att = entity_att.index_select(0, ht[:, 0])    # (n_pair, heads, c)
            t_att = entity_att.index_select(0, ht[:, 1])
            ht_att = (h_att * t_att).mean(1)                # (n_pair, c)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)
            context = ht_att @ seq_i                        # (n_pair, H)

            head_ctx = torch.tanh(self.head_ext(torch.cat([h_enh, context], -1)))
            tail_ctx = torch.tanh(self.tail_ext(torch.cat([t_enh, context], -1)))
            z_struct = self.struct_proj(torch.cat([head_ctx, tail_ctx], -1))  # (n_pair, emb)

            # (3) DREAM evidence attention over reasoned sentences
            q = self.evi_q(torch.cat([h_enh, t_enh], -1))   # (n_pair, emb)
            k = self.evi_k(sent_reasoned)                   # (n_s, emb)
            v = self.evi_v(sent_reasoned)                   # (n_s, emb)
            evi_scores = q @ k.t() / (self.emb_size ** 0.5)  # (n_pair, n_s)
            p_evi = torch.softmax(evi_scores, dim=-1)
            c_evi = p_evi @ v                               # (n_pair, emb)

            # (4) gated fusion + MLP head
            gate = torch.sigmoid(self.fuse_gate(torch.cat([z_struct, c_evi], -1)))
            z_fused = gate * z_struct + (1.0 - gate) * c_evi
            all_logits.append(self.classifier(z_fused))
            evi_scores_list.append(evi_scores)

        if not all_logits:
            # degenerate batch (no valid pairs); return empty consistent shapes
            logits = sequence_output.new_zeros(0, self.num_labels)
        else:
            logits = torch.cat(all_logits, 0)

        preds = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        output = (preds,)
        if labels is not None:
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels, dtype=torch.float)
            labels = labels.to(dtype=torch.float, device=logits.device)
            loss = self.loss_fnt(logits, labels)
            if evidence is not None and self.evi_lambda > 0:
                loss = loss + self.evi_lambda * self._evidence_contrastive(
                    evi_scores_list, evidence)
            output = (loss,) + output
        return output

    # --------------------------------------------------- evidence contrastive
    def _evidence_contrastive(self, evi_scores_list, evidence):
        """Multi-positive InfoNCE over sentences: for each pair carrying gold
        evidence, pull its evidence sentences' logits up against all sentences
        (softmax denominator). Pairs/docs without evidence (all of train_distant)
        contribute nothing, so the term is auto-disabled there.

        evi_scores_list is aligned with `evidence` document-by-document (None for
        docs that produced no pairs)."""
        num = None
        count = 0
        for scores, doc_evi in zip(evi_scores_list, evidence):
            if scores is None or scores.numel() == 0:
                continue
            n_pair, n_sent = scores.shape
            gold = scores.new_zeros(n_pair, n_sent)
            has = scores.new_zeros(n_pair, dtype=torch.bool)
            for pi, sids in enumerate(doc_evi):
                valid = [s for s in sids if 0 <= s < n_sent]
                if valid:
                    gold[pi, valid] = 1.0
                    has[pi] = True
            if not bool(has.any()):
                continue
            logp = F.log_softmax(scores, dim=-1)
            n_gold = gold.sum(1).clamp(min=1.0)
            per_pair = -(logp * gold).sum(1) / n_gold       # (n_pair,)
            term = per_pair[has].sum()
            num = term if num is None else num + term
            count += int(has.sum())
        if count > 0:
            return num / count
        # nothing to supervise -> zero tensor tied to a param for a valid graph
        return self.evi_q.weight.sum() * 0.0
