"""[개선] Baseline(ATLOP) + DREAM(evidence-guided) 기법.

비교 결과 baseline(re_model.DocREModel)이 F1 최고였다. 그래서 baseline의 강점
(logsumexp 엔티티 풀링 · Localized Context Pooling · grouped bilinear)은 **그대로
두고**, 정보 추출을 돕는 **DREAM(DREEAM식 evidence-guided context) 기법 하나만**
얹는다. baseline 파일(re_model.py)은 한 줄도 수정하지 않고 상속으로 구현.

핵심 아이디어 (DREEAM, Ma et al. 2023 의 evidence-guided 아이디어)
----------------------------------------------------------------
ATLOP의 Localized Context Pooling은 head·tail attention을 곱해 문맥 토큰 가중치
q_prod를 만든다. 그런데 근거 토큰이 한쪽에서 낮은 attention을 받으면 곱이 0에
가까워져 **근거 정보가 유실**된다. DREAM은:

  1. q_prod를 **문장 단위로 합산**해 근거 분포 p_evi(문장별 근거 정도)를 만들고,
  2. 그 p_evi를 다시 문장→토큰으로 되뿌린 q_evi 를 만든 뒤,
  3. 학습형 게이트 g로 섞는다:  q_final = (1-g)·q_prod + g·q_evi
     → 근거 문장의 토큰 질량이 보강되어 유실을 막는다.

p_evi 는 gold evidence 문장으로 **지도학습**된다(evidence loss). evidence는
annotated(및 Re-DocRED revised)에만 있으므로, evidence가 없는 배치(train_distant
등)에서는 evidence loss가 자동으로 0이 된다.

설계 포인트
-----------
- **분류기·풀링은 baseline 그대로** (bilinear head 유지 = baseline의 강점 보존).
  DREAM은 문맥 벡터 rs 를 만드는 과정만 바꾼다.
- 게이트 g = sigmoid(evi_gate), init −2.0 → g≈0.12. 시작 시 거의 순수 q_prod
  (≈baseline)에서 출발해 evidence를 얼마나 더할지 학습한다.
- 추가 파라미터는 **evi_gate(스칼라) 하나뿐**. 나머지 이름은 baseline과 동일 →
  `results/atlop.pt`로 warm-start하면 시작 출력이 baseline과 사실상 동일하고
  거기서 evidence 보강만 학습한다.
- forward 시그니처가 sent_pos·evidence 를 추가로 받으므로, 학습은 baseline
  전처리가 아니라 preprocess_full.build_features_full(sent_pos·evidence 포함)을
  쓰는 진입점(train_full 계열)이 필요하다.
"""

import torch
import torch.nn as nn

from .re_model import DocREModel


class DocREModelDREAM(DocREModel):
    def __init__(self, config, encoder, emb_size: int = 768, block_size: int = 64,
                 num_labels: int = 97, offset: int = 1, loss_fnt=None,
                 evi_lambda: float = 0.1):
        super().__init__(config, encoder, emb_size=emb_size, block_size=block_size,
                         num_labels=num_labels, offset=offset, loss_fnt=loss_fnt)
        # 학습형 blend 게이트 g = sigmoid(evi_gate). init −2.0 -> g≈0.12
        # (시작 시 거의 순수 product context = baseline 경로).
        self.evi_gate = nn.Parameter(torch.tensor(-2.0))
        self.evi_lambda = evi_lambda

    def get_hrt_evidence(self, sequence_output, attention, entity_pos, hts, sent_pos):
        """baseline get_hrt + DREEAM evidence-guided context.

        Returns (hss, rss, tss) 를 배치 전체에 대해 concat한 것과, evidence loss용
        per-doc 근거 분포 리스트 evi_list(각 (n_pair, n_sent))."""
        offset = self.offset
        _, num_heads, _, c = attention.size()
        g = torch.sigmoid(self.evi_gate)
        hss, tss, rss, evi_list = [], [], [], []

        for i in range(len(entity_pos)):
            # --- 엔티티 임베딩/어텐션 풀링 (baseline get_hrt와 동일) ---
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

            # --- Localized Context Pooling: product attention q_prod ---
            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])   # (n_pair, heads, c)
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)                          # (n_pair, c)
            q_prod = ht_att / (ht_att.sum(1, keepdim=True) + 1e-30)

            # --- DREAM: q_prod -> 문장 근거 분포 p_evi -> 토큰 q_evi -> blend ---
            n_sent = len(sent_pos[i])
            M = torch.zeros(n_sent, c, device=sequence_output.device, dtype=q_prod.dtype)
            for s, (st, en) in enumerate(sent_pos[i]):
                a, b = st + offset, min(en + offset, c)
                if a < b:
                    M[s, a:b] = 1.0

            e_sent = q_prod @ M.t()                                  # (n_pair, n_sent)
            p_evi = e_sent / (e_sent.sum(1, keepdim=True) + 1e-30)   # 근거 분포
            M_rownorm = M / M.sum(1, keepdim=True).clamp(min=1.0)    # 문장→토큰 균등 분배
            q_evi = p_evi @ M_rownorm                                # (n_pair, c)

            q_final = (1.0 - g) * q_prod + g * q_evi
            q_final = q_final / (q_final.sum(1, keepdim=True) + 1e-30)
            rs = q_final @ sequence_output[i]                        # (n_pair, hidden)

            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
            evi_list.append(p_evi)

        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss, evi_list

    def _evidence_loss(self, evi_list, evidence):
        """p_evi 와 gold evidence(문장 균등 분포) 사이 cross-entropy. 근거가 실제로
        있는 pair에서만 계산하고 평균낸다. evidence가 비어 있는 문서/pair(예:
        train_distant)는 기여하지 않으므로 자동으로 건너뛴다."""
        num = None
        count = 0
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
            if not bool(has.any()):
                continue
            gold = gold / gold.sum(1, keepdim=True).clamp(min=1.0)
            logp = torch.log(p_evi.clamp(min=1e-30))
            ce = -(gold * logp).sum(1)                               # (n_pair,)
            term = ce[has].sum()
            num = term if num is None else num + term
            count += int(has.sum())
        if count > 0:
            return num / count
        return self.evi_gate.sum() * 0.0  # 근거 없음 -> 그래프/디바이스 맞춘 0

    def forward(self, input_ids, attention_mask, entity_pos, hts, sent_pos,
                labels=None, evidence=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts, evi_list = self.get_hrt_evidence(
            sequence_output, attention, entity_pos, hts, sent_pos
        )

        # --- baseline 분류 경로 그대로 (head/tail extractor + grouped bilinear) ---
        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

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
