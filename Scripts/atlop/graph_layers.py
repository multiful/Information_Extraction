"""Entity-pair graph layers for the ATLOP graph variants (개선 1 GCN / 개선 2 GAT).

Graph definition (per document): one node per ordered entity pair (h, t) --
exactly one node per row of the document's `hts`. Two pair-nodes are connected
when they share an entity, and the share pattern is kept as 3 typed edge sets
so a layer can weight them differently:

  type 0  same-head : (a,b)-(a,c)              h_i == h_j
  type 1  same-tail : (a,c)-(b,c)              t_i == t_j
  type 2  bridge    : (a,b)-(b,c), (c,a)-(a,b) t_i == h_j  or  h_i == t_j

Why this targets the multi-hop weakness (model.ipynb 테스트 1): for the target
pair (a,c), its same-head neighbors include premise (a,b) and its same-tail
neighbors include premise (b,c), so ONE propagation layer already lets the
(a,c) node read both premises of an a->b->c chain; the bridge edges connect
the two premises to each other. ATLOP alone classifies every pair
independently and has no such path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

N_EDGE_TYPES = 3


def build_pair_adjacency(ht: torch.Tensor) -> torch.Tensor:
    """ht: (m, 2) long tensor of (head_entity, tail_entity) per pair-node.
    Returns a bool (3, m, m) typed adjacency; self-connections are removed
    (self information is handled inside each layer)."""
    h, t = ht[:, 0], ht[:, 1]
    same_head = h.unsqueeze(1) == h.unsqueeze(0)
    same_tail = t.unsqueeze(1) == t.unsqueeze(0)
    bridge = (t.unsqueeze(1) == h.unsqueeze(0)) | (h.unsqueeze(1) == t.unsqueeze(0))
    adj = torch.stack([same_head, same_tail, bridge])
    eye = torch.eye(ht.size(0), dtype=torch.bool, device=ht.device)
    return adj & ~eye


class PairGCNLayer(nn.Module):
    """Relational GCN layer over the typed pair graph (개선 1).

    x' = LayerNorm(x + Dropout(ReLU(W_self x + sum_k mean_{j in N_k(i)} x_j W_k)))

    Each edge type gets its own weight matrix (R-GCN style with 3 relations);
    neighbor aggregation is a fixed mean per type -- the "GCN" contrast to the
    learned attention weights of PairGATLayer.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.type_lins = nn.ModuleList(
            nn.Linear(dim, dim, bias=False) for _ in range(N_EDGE_TYPES)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = self.self_lin(x)
        for k, lin in enumerate(self.type_lins):
            a = adj[k].to(x.dtype)
            a = a / a.sum(1, keepdim=True).clamp(min=1.0)  # row-mean over type-k neighbors
            out = out + a @ lin(x)
        return self.norm(x + self.dropout(F.relu(out)))


class PairGATLayer(nn.Module):
    """Multi-head graph-attention layer over the same typed pair graph (개선 2).

    Dot-product attention masked to graph neighbors (+ self), plus a learnable
    additive bias per (edge type, head) so heads can specialize -- e.g. one
    head up-weighting bridge edges for multi-hop chains. Where the GCN layer
    averages neighbors with fixed weights, here the model decides per pair
    which neighbor pairs matter.
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0, "graph_dim must be divisible by graph_heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.type_bias = nn.Parameter(torch.zeros(N_EDGE_TYPES, heads))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        m = x.size(0)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(m, self.heads, self.head_dim).transpose(0, 1)   # (H, m, d_h)
        k = k.view(m, self.heads, self.head_dim).transpose(0, 1)
        v = v.view(m, self.heads, self.head_dim).transpose(0, 1)

        scores = q @ k.transpose(-2, -1) / self.head_dim ** 0.5    # (H, m, m)
        scores = scores + torch.einsum("kmn,kh->hmn", adj.to(x.dtype), self.type_bias)
        allowed = adj.any(0) | torch.eye(m, dtype=torch.bool, device=x.device)
        scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))

        att = self.dropout(torch.softmax(scores, dim=-1))
        out = (att @ v).transpose(0, 1).reshape(m, -1)             # (m, dim)
        return self.norm(x + self.dropout(self.out(out)))
