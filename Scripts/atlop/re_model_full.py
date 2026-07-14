"""[통합 모델] 세 개선을 하나로 합친 DocRE 파이프라인 (DocREModelFull).

PNG 파이프라인 그대로:
  Document → RoBERTa Encoder → Entity Marker + logsumexp Pooling
    → Evidence-guided Local Context (DREEAM 아이디어)      … 문제 ② 정보 유실
    → Entity Pair Representation
    → Entity Pair Graph + GAT (GREP 아이디어)              … 문제 ① multi-hop
    → Relation Classifier
    → PU-inspired ATLoss (TTM-RE, w=0.7)                   … 문제 ③ 임계값 고도화
    → Relation Triples (h, r, t)

baseline 파일(re_model.py)은 무수정 — 상속으로 인코딩·logsumexp 엔티티 풀링·
grouped bilinear를 물려받고, 아래 세 가지만 얹는다.

1. Evidence-guided Local Context (문제 ②)
   ATLOP의 Localized Context Pooling은 head·tail attention을 곱해 문맥 토큰
   가중치 q_prod를 만든다 — 근거 토큰이 한쪽에서 낮은 attention을 받으면 곱이
   0에 가까워져 정보가 유실된다. DREEAM처럼 q_prod를 문장 단위로 합쳐 evidence
   분포 p_evi를 만들고, 그것을 문장→토큰으로 되뿌린 q_evi를 학습형 게이트 g로
   q_prod에 섞는다: q_final = (1-g)·q_prod + g·q_evi. 근거 문장의 토큰 질량이
   보강되어 유실을 막는다. p_evi는 gold evidence 문장으로 지도학습(evidence loss)
   되어 실제 근거에 집중하도록 유도된다(annotated에만 evidence가 있으므로 distant
   단계에서는 evidence loss가 자동으로 비활성).

2. Entity Pair Graph + GAT (문제 ①) — re_model_gat.py와 동일한 GREP 구조.
   엔티티를 공유하는 쌍끼리 연결(same-head/same-tail/bridge)한 그래프에서 GAT로
   전파, (A,C) 노드가 전제 (A,B)·(B,C)를 읽어 조합 추론을 가능케 한다. graph_out은
   zero-init 잔차 헤드라 학습 시작점이 baseline 경로와 동일.

3. PU-inspired ATLoss (문제 ③) — losses.PUATLoss(na_weight=0.7). distant의 가짜
   Na가 임계값을 오염시키는 것을 막는다. loss_fnt 주입으로 갈아끼우며(train_full이
   distant 단계에만 PUATLoss(0.7), annotated 단계는 ATLoss — annotated의 Na는 gold),
   손실은 PUATLoss = loss1 + w·loss2 (w=0.7) 형태.
"""

import torch
import torch.nn as nn

from .graph_layers import PairGATLayer, build_pair_adjacency
from .re_model import DocREModel


class DocREModelFull(DocREModel):
    def __init__(self, config, encoder, emb_size: int = 768, block_size: int = 64,
                 num_labels: int = 97, offset: int = 1, loss_fnt=None,
                 graph_layers: int = 2, graph_dim: int = 256, graph_heads: int = 4,
                 graph_dropout: float = 0.1, evi_lambda: float = 0.1):
        super().__init__(config, encoder, emb_size=emb_size, block_size=block_size,
                         num_labels=num_labels, offset=offset, loss_fnt=loss_fnt)
        # (1) evidence-guided context: learnable blend gate g = sigmoid(evi_gate).
        # init -2.0 -> g≈0.12, so training starts close to the pure product
        # context (near baseline) and learns how much evidence mass to add.
        self.evi_gate = nn.Parameter(torch.tensor(-2.0))
        self.evi_lambda = evi_lambda

        # (2) entity-pair graph + GAT (GREP), identical wiring to re_model_gat.
        self.graph_in = nn.Linear(2 * emb_size, graph_dim)
        self.graph_gnn = nn.ModuleList(
            PairGATLayer(graph_dim, heads=graph_heads, dropout=graph_dropout)
            for _ in range(graph_layers)
        )
        self.graph_out = nn.Linear(graph_dim, num_labels)
        nn.init.zeros_(self.graph_out.weight)
        nn.init.zeros_(self.graph_out.bias)

    def get_hrt_evidence(self, sequence_output, attention, entity_pos, hts, sent_pos):
        """Base get_hrt + DREEAM evidence-guided context. Returns
        (hss, rss, tss) concatenated across the batch, and evi_list = per-doc
        (n_pair, n_sent) evidence distributions p_evi for the evidence loss."""
        offset = self.offset
        _, num_heads, _, c = attention.size()
        g = torch.sigmoid(self.evi_gate)
        hss, tss, rss, evi_list = [], [], [], []

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

            entity_embs = torch.stack(entity_embs, dim=0)
            entity_atts = torch.stack(entity_atts, dim=0)

            ht_i = torch.as_tensor(hts[i], dtype=torch.long,
                                   device=sequence_output.device).reshape(-1, 2)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)                       # (n_pair, c)
            q_prod = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)

            # sentence membership matrix M (n_sent, c) in +offset token space
            n_sent = len(sent_pos[i])
            M = torch.zeros(n_sent, c, device=sequence_output.device, dtype=q_prod.dtype)
            for s, (st, en) in enumerate(sent_pos[i]):
                a, b = st + offset, min(en + offset, c)
                if a < b:
                    M[s, a:b] = 1.0

            e_sent = q_prod @ M.t()                                # (n_pair, n_sent)
            p_evi = e_sent / (e_sent.sum(1, keepdim=True) + 1e-30)
            # evidence context: spread each sentence's weight over its tokens
            M_rownorm = M / M.sum(1, keepdim=True).clamp(min=1.0)
            q_evi = p_evi @ M_rownorm                              # (n_pair, c)

            q_final = (1.0 - g) * q_prod + g * q_evi
            q_final = q_final / (q_final.sum(1, keepdim=True) + 1e-30)
            rs = q_final @ sequence_output[i]                      # (n_pair, hidden)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
            evi_list.append(p_evi)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss, evi_list

    def _evidence_loss(self, evi_list, evidence):
        """Cross-entropy between p_evi and the gold evidence distribution
        (uniform over a pair's evidence sentences), averaged over pairs that
        actually carry evidence. Pairs/docs without evidence contribute nothing
        (so train_distant, whose evidence is all empty, is skipped)."""
        num = evi_list[0].new_zeros(())
        count = evi_list[0].new_zeros(())
        for p_evi, doc_evi in zip(evi_list, evidence):
            if p_evi.numel() == 0:
                continue
            n_pair, n_sent = p_evi.shape
            gold = p_evi.new_zeros(n_pair, n_sent)
            has = p_evi.new_zeros(n_pair, dtype=torch.bool)
            for pi, sids in enumerate(doc_evi):
                valid = [s for s in sids if 0 <= s < n_sent]
                if valid:
                    gold[pi, valid] = 1.0
                    has[pi] = True
            if has.any():
                gold = gold / gold.sum(1, keepdim=True).clamp(min=1.0)
                logp = torch.log(p_evi.clamp(min=1e-30))
                ce = -(gold * logp).sum(1)
                num = num + ce[has].sum()
                count = count + has.sum()
        if count > 0:
            return num / count
        return num  # zero tensor, keeps graph/device consistent

    def forward(self, input_ids, attention_mask, entity_pos, hts, sent_pos,
                labels=None, evidence=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts, evi_list = self.get_hrt_evidence(
            sequence_output, attention, entity_pos, hts, sent_pos
        )

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))

        # relation classifier: grouped bilinear (baseline path)
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        # entity pair graph + GAT (per document), zero-init residual to logits
        gx = torch.tanh(self.graph_in(torch.cat([hs, ts], dim=1)))
        refined, start = [], 0
        for doc_hts in hts:
            m = len(doc_hts)
            if m == 0:
                continue
            x = gx[start: start + m]
            ht = torch.as_tensor(doc_hts, dtype=torch.long, device=gx.device).reshape(-1, 2)
            adj = build_pair_adjacency(ht)
            for layer in self.graph_gnn:
                x = layer(x, adj)
            refined.append(x)
            start += m
        if refined:
            logits = logits + self.graph_out(torch.cat(refined, dim=0))

        preds = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        output = (preds,)
        if labels is not None:
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels, dtype=torch.float)
            labels = labels.to(dtype=torch.float, device=logits.device)
            loss = self.loss_fnt(logits, labels)
            if evidence is not None and self.evi_lambda > 0:
                loss = loss + self.evi_lambda * self._evidence_loss(evi_list, evidence)
            output = (loss,) + output
        return output
