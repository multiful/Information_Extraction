"""[개선 1] GREP: ATLOP + Entity Pair Graph + relational GCN.

baseline(re_model.DocREModel)은 한 줄도 수정하지 않는다 -- 상속으로 인코딩,
logsumexp 엔티티 풀링, Localized Context Pooling, grouped bilinear를 전부
그대로 물려받고, 그 위에 entity-pair graph 전파 단계만 얹는다.

겨냥하는 약점 (model.ipynb 테스트 1): ATLOP은 (h, t) 쌍을 서로 독립으로
분류하므로 A→B, B→C가 문서에 명시돼도 조합해야 나오는 A→C를 잡을 경로가
없다. 여기서는 쌍 표현들을 노드로 하는 그래프(graph_layers.py)를 만들어,
엔티티를 공유하는 쌍끼리 relational GCN으로 정보를 교환한다 -- (a,c) 노드는
한 layer 만에 same-head 이웃 (a,b)와 same-tail 이웃 (b,c)를 모두 읽는다.

GCN의 성격: 이웃 집계가 엣지 타입별 "고정 평균"(타입별 가중치 행렬만 학습).
개선 2(re_model_gat.py)는 같은 그래프에서 집계 가중치 자체를 attention으로
학습한다는 점만 다르다 -- 노드 특징/그래프 구조가 동일해서 깨끗한 ablation.

파라미터 구성은 baseline의 엄격한 superset(이름 동일 + graph_* 추가)이고
graph_out(그래프 → logit 잔차 헤드)은 zero-init이라, results/atlop.pt로
warm-start하면 시작 시점 출력이 baseline과 정확히 일치한다 -- 그래프가
도움이 되는 방향으로만 벗어나며 학습된다 (smoke_test_graph.py에서 검증).
"""

import torch
import torch.nn as nn

from .graph_layers import PairGCNLayer, build_pair_adjacency
from .re_model import DocREModel


class DocREModelGCN(DocREModel):
    def __init__(self, config, encoder, emb_size: int = 768, block_size: int = 64,
                 num_labels: int = 97, offset: int = 1, loss_fnt=None,
                 graph_layers: int = 2, graph_dim: int = 256, graph_dropout: float = 0.1):
        super().__init__(config, encoder, emb_size=emb_size, block_size=block_size,
                         num_labels=num_labels, offset=offset, loss_fnt=loss_fnt)
        # pair-node feature: post-extractor head/tail reps -- Localized Context
        # Pooling의 문맥 벡터가 [h;r]/[t;r] extractor를 통해 이미 녹아 있다.
        self.graph_in = nn.Linear(2 * emb_size, graph_dim)
        self.graph_gnn = nn.ModuleList(
            PairGCNLayer(graph_dim, dropout=graph_dropout) for _ in range(graph_layers)
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
