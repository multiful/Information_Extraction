"""CPU wiring check for the two improvement models (개선 1 GCN / 개선 2 GAT).

The baseline files (re_model.py / train_re.py) are untouched; the variants
live in re_model_gcn.py / re_model_gat.py and train via train_graph.py.
Like smoke_test.py this proves correctness of the plumbing, NOT accuracy:
a tiny random-init BERT on a few dev docs. Three checks:

  1. adjacency sanity  - for pairs of a 3-entity doc, the (a,c) node is linked
                         to premise (a,b) via same-head, to premise (b,c) via
                         same-tail, and the premises to each other via bridge
                         (the multi-hop path 테스트 1 needs).
  2. forward/backward  - loss + preds for both variants, every param reachable.
  3. zero-init parity  - a graph model warm-started from a baseline state_dict
                         (strict=False) reproduces the baseline loss/preds
                         EXACTLY, because graph_out is zero-initialized. This
                         is what makes --init_ckpt results/atlop.pt a safe
                         starting point.

Run:  python -m Scripts.atlop.smoke_test_graph
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, BertConfig, BertModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_dataset import DocREDataset            # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES     # noqa: E402
from Scripts.atlop.graph_layers import build_pair_adjacency  # noqa: E402
from Scripts.atlop.preprocess import build_features       # noqa: E402
from Scripts.atlop.re_model import DocREModel             # noqa: E402
from Scripts.atlop.re_model_gat import DocREModelGAT      # noqa: E402
from Scripts.atlop.re_model_gcn import DocREModelGCN      # noqa: E402
from Scripts.atlop.train_re import make_collate_fn        # noqa: E402

N_DOCS = 4
EMB_SIZE = 64
BLOCK_SIZE = 16
GRAPH_DIM = 32
GRAPH_HEADS = 4


def check_adjacency():
    # 3 entities a=0, b=1, c=2 -> ordered pairs in preprocess.py order:
    # (0,1) (0,2) (1,0) (1,2) (2,0) (2,1); node ids     0     1     2     3     4     5
    ht = torch.tensor([[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]])
    adj = build_pair_adjacency(ht)
    same_head, same_tail, bridge = adj[0], adj[1], adj[2]
    ac, ab, bc = 1, 0, 3  # node ids of pairs (0,2), (0,1), (1,2)
    assert same_head[ac, ab], "(a,c) must see premise (a,b) via same-head"
    assert same_tail[ac, bc], "(a,c) must see premise (b,c) via same-tail"
    assert bridge[ab, bc], "premises (a,b)-(b,c) must be bridge-linked (t==h)"
    assert not adj[:, ac, ac].any(), "self-connections must be removed"
    print("[adjacency] multi-hop connectivity OK "
          "((a,c) reaches both premises in one layer)")


def tiny_models(tokenizer, graph_type):
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=EMB_SIZE, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=1300,
        attn_implementation="eager",
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    torch.manual_seed(0)
    base = DocREModel(config, BertModel(config, add_pooling_layer=False),
                      emb_size=EMB_SIZE, block_size=BLOCK_SIZE, num_labels=NUM_CLASSES)
    torch.manual_seed(1)  # different init on purpose; parity comes from load_state_dict
    common = dict(emb_size=EMB_SIZE, block_size=BLOCK_SIZE, num_labels=NUM_CLASSES,
                  graph_layers=2, graph_dim=GRAPH_DIM)
    if graph_type == "gcn":
        graph = DocREModelGCN(config, BertModel(config, add_pooling_layer=False), **common)
    else:
        graph = DocREModelGAT(config, BertModel(config, add_pooling_layer=False),
                              graph_heads=GRAPH_HEADS, **common)
    return base, graph


def main():
    check_adjacency()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    rel2id = build_rel2id()
    docs = [DocREDataset("dev")[i] for i in range(N_DOCS)]
    features = build_features(docs, tokenizer, rel2id, show_progress=False)
    collate = make_collate_fn(tokenizer.pad_token_id)
    batch = collate(features)
    kwargs = dict(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        entity_pos=batch["entity_pos"], hts=batch["hts"], labels=batch["labels"],
    )

    for graph_type in ("gcn", "gat"):
        base, graph = tiny_models(tokenizer, graph_type)

        # 2) forward/backward on the variant
        graph.train()
        loss, preds = graph(**kwargs)
        assert preds.shape == (sum(len(f["hts"]) for f in features), NUM_CLASSES)
        loss.backward()
        n_graph_params = sum(1 for n, _ in graph.named_parameters() if "graph" in n)
        no_grad = [n for n, p in graph.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert not no_grad, f"params without grad: {no_grad}"
        print(f"[{graph_type}] forward/backward OK  loss={loss.item():.4f}  "
              f"graph params={n_graph_params} (all reachable)")
        graph.zero_grad(set_to_none=True)

        # 3) zero-init parity with the baseline after warm-start
        missing, unexpected = graph.load_state_dict(base.state_dict(), strict=False)
        assert not unexpected, f"unexpected keys: {unexpected}"
        assert all("graph" in k for k in missing), f"non-graph keys missing: {missing}"
        base.eval(), graph.eval()
        with torch.no_grad():
            b_loss, b_preds = base(**kwargs)
            g_loss, g_preds = graph(**kwargs)
        assert torch.equal(b_preds, g_preds), "preds diverge at zero-init"
        assert torch.allclose(b_loss, g_loss, atol=1e-6), \
            f"loss diverges at zero-init: {b_loss.item()} vs {g_loss.item()}"
        print(f"[{graph_type}] zero-init parity OK  "
              f"(baseline loss {b_loss.item():.6f} == variant loss {g_loss.item():.6f})")

    print("\nGRAPH SMOKE TEST PASSED - both variants wired, warm-start starts at baseline.")


if __name__ == "__main__":
    main()
