"""[개선 2] ATLOP + Localized Context Pooling + Entity Pair Graph + GAT.

baseline(re_model.DocREModel)은 한 줄도 수정하지 않는다 -- 상속으로 인코딩,
logsumexp 엔티티 풀링, Localized Context Pooling(LCP), grouped bilinear를
전부 그대로 물려받고, 그 위에 graph-attention 전파 단계만 얹는다.

개선 1(re_model_gcn.py)과의 관계: 노드 특징과 그래프 구조(엔티티를 공유하는
쌍끼리 연결, same-head/same-tail/bridge 3타입)는 완전히 동일하다. 차이는
이웃 집계 방식 하나 --

  개선 1 GCN : 타입별 고정 평균 (어느 이웃이든 같은 무게)
  개선 2 GAT : multi-head attention으로 "어떤 이웃 쌍을 읽을지"를 쌍마다
               학습. 엣지 타입별 learnable bias가 있어 head가 bridge 엣지
               (multi-hop 체인 (a,b)-(b,c))에 특화될 수 있다.

LCP가 이름에 들어가는 이유: 노드 특징이 LCP 문맥 벡터가 녹아 있는
post-extractor 쌍 표현이라, GAT의 attention은 "이 쌍의 국소 문맥으로 보아
어떤 이웃 쌍의 증거를 끌어올까"를 문맥 조건부로 정한다 -- 토큰 수준에서
LCP가 하던 일(관련 문맥 선택)을 쌍(pair) 수준으로 확장한 구조.

파라미터 구성은 baseline의 엄격한 superset(이름 동일 + graph_* 추가)이고
graph_out은 zero-init이라, results/atlop.pt로 warm-start하면 시작 시점
출력이 baseline과 정확히 일치한다 (smoke_test_graph.py에서 검증).
"""

import torch
import torch.nn as nn

from .graph_layers import PairGATLayer, build_pair_adjacency
from .re_model import DocREModel


class DocREModelGAT(DocREModel):
    def __init__(self, config, encoder, emb_size: int = 768, block_size: int = 64,
                 num_labels: int = 97, offset: int = 1, loss_fnt=None,
                 graph_layers: int = 2, graph_dim: int = 256, graph_heads: int = 4,
                 graph_dropout: float = 0.1):
        super().__init__(config, encoder, emb_size=emb_size, block_size=block_size,
                         num_labels=num_labels, offset=offset, loss_fnt=loss_fnt)
        # pair-node feature: post-extractor head/tail reps (LCP 문맥 포함)
        self.graph_in = nn.Linear(2 * emb_size, graph_dim)
        self.graph_gnn = nn.ModuleList(
            PairGATLayer(graph_dim, heads=graph_heads, dropout=graph_dropout)
            for _ in range(graph_layers)
        )
        # zero-init residual head: at init the model IS the baseline.
        self.graph_out = nn.Linear(graph_dim, num_labels)
        nn.init.zeros_(self.graph_out.weight)
        nn.init.zeros_(self.graph_out.bias)

    def forward(self, input_ids, attention_mask, entity_pos, hts, labels=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))

        # baseline path -- DocREModel.forward와 동일한 grouped bilinear
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)

        # graph path -- 문서 단위로 분리 실행 (다른 문서의 쌍끼리 섞이면 안 됨)
        g = torch.tanh(self.graph_in(torch.cat([hs, ts], dim=1)))
        refined, start = [], 0
        for doc_hts in hts:
            m = len(doc_hts)
            if m == 0:
                continue
            x = g[start: start + m]
            ht = torch.as_tensor(doc_hts, dtype=torch.long, device=g.device).reshape(-1, 2)
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
            output = (loss,) + output
        return output
