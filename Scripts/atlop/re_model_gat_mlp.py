"""BERT -> ATLOP LCP -> Entity Pair Rep -> 2-layer edge-featured GAT ->
2-layer MLP classifier -> Sigmoid, trained with
Loss = BCEWithLogitsLoss + 0.2 x Evidence Contrastive Loss (losses.BCEEvidenceContrastiveLoss).

Differs from re_model_gat.DocREModelGAT in the classifier head: that model
keeps the baseline's grouped-bilinear + ATLoss and adds the GAT as a zero-init
residual on top. Here the GAT output IS the only classifier input (no
bilinear, no residual) and it feeds a small MLP instead, so the pipeline
matches this architecture end to end rather than augmenting the baseline.

Reuses, unmodified: DocREModel.encode / get_hrt (BERT + logsumexp entity
pooling + Localized Context Pooling) and graph_layers.PairGATLayer /
build_pair_adjacency (same same-head/same-tail/bridge typed pair graph as the
other GAT variant). self.bilinear is inherited from DocREModel but unused --
kept only so encode/get_hrt don't need re-deriving; it never receives
gradient since forward never reads it.
"""

import torch
import torch.nn as nn

from .graph_layers import PairGATLayer, build_pair_adjacency
from .losses import BCEEvidenceContrastiveLoss
from .re_model import DocREModel


class DocREModelGATMLP(DocREModel):
    def __init__(self, config, encoder, emb_size: int = 768, block_size: int = 64,
                 num_labels: int = 97, offset: int = 1,
                 graph_layers: int = 2, graph_dim: int = 256, graph_heads: int = 4,
                 graph_dropout: float = 0.1, mlp_hidden: int = 256, mlp_dropout: float = 0.1,
                 evi_weight: float = 0.2, tau: float = 0.1, threshold: float = 0.5):
        super().__init__(config, encoder, emb_size=emb_size, block_size=block_size,
                         num_labels=num_labels, offset=offset,
                         loss_fnt=BCEEvidenceContrastiveLoss(evi_weight=evi_weight, tau=tau))
        self.threshold = threshold
        self.graph_dim = graph_dim

        # entity pair representation -> GAT node feature
        self.graph_in = nn.Linear(2 * emb_size, graph_dim)
        self.graph_gnn = nn.ModuleList(
            PairGATLayer(graph_dim, heads=graph_heads, dropout=graph_dropout)
            for _ in range(graph_layers)
        )
        self.classifier = nn.Sequential(
            nn.Linear(graph_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, num_labels),
        )

    def _pool_sent_reprs(self, sequence_output: torch.Tensor,
                          sent_pos: list[list[tuple[int, int]]]) -> list[torch.Tensor]:
        """Per-sentence embedding = mean-pooled token hidden states over the
        sentence's token span (+offset to skip [CLS], same convention as
        entity_pos). Independent of any pair's attention -- this is the plain
        sentence representation the evidence-contrastive loss contrasts pair
        context vectors against."""
        offset = self.offset
        c = sequence_output.size(1)
        out = []
        for i, spans in enumerate(sent_pos):
            if not spans:
                out.append(sequence_output.new_zeros((0, self.hidden_size)))
                continue
            reprs = []
            for st, en in spans:
                a, b = min(st + offset, c), min(en + offset, c)
                reprs.append(sequence_output[i, a:b].mean(0) if a < b
                             else sequence_output.new_zeros(self.hidden_size))
            out.append(torch.stack(reprs, dim=0))
        return out

    def forward(self, input_ids, attention_mask, entity_pos, hts, sent_pos=None,
                labels=None, evidence=None):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        node = torch.tanh(self.graph_in(torch.cat([hs, ts], dim=1)))

        # 2-layer edge-featured GAT over the entity-pair graph, per document
        # (pairs from different docs must never attend to each other).
        refined, start = [], 0
        for doc_hts in hts:
            m = len(doc_hts)
            if m == 0:
                continue
            x = node[start: start + m]
            ht = torch.as_tensor(doc_hts, dtype=torch.long, device=node.device).reshape(-1, 2)
            adj = build_pair_adjacency(ht)
            for layer in self.graph_gnn:
                x = layer(x, adj)
            refined.append(x)
            start += m
        x_all = torch.cat(refined, dim=0) if refined else node

        logits = self.classifier(x_all)

        preds = self.loss_fnt.get_label(logits, num_labels=self.num_labels, threshold=self.threshold)
        output = (preds,)
        if labels is not None:
            if not torch.is_tensor(labels):
                labels = torch.as_tensor(labels, dtype=torch.float)
            labels = labels.to(dtype=torch.float, device=logits.device)

            rs_list, sent_reprs = None, None
            if evidence is not None and sent_pos is not None:
                rs_list, start = [], 0
                for doc_hts in hts:
                    m = len(doc_hts)
                    rs_list.append(rs[start: start + m])
                    start += m
                sent_reprs = self._pool_sent_reprs(sequence_output, sent_pos)

            loss = self.loss_fnt(logits, labels, rs_list=rs_list, sent_reprs=sent_reprs, evidence=evidence)
            output = (loss,) + output
        return output
