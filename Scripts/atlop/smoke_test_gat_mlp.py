"""CPU wiring check for re_model_gat_mlp.DocREModelGATMLP.

Random-init tiny BERT on a few dev docs -- proves the whole pipeline wires up
(BERT -> LCP -> entity-pair rep -> edge-featured GAT -> MLP -> sigmoid, loss =
BCE + evi_weight x evidence-contrastive), NOT accuracy. Checks:

  1. fwd/bwd     - loss finite, every param reached (incl. graph + classifier).
  2. get_label   - multi-hot predictions shaped (total_pairs, NUM_CLASSES),
                   every row has >=1 class set (real relation(s) or Na).
  3. evi_weight  - turning it up changes the loss vs off (the evidence
                   contrastive term is actually contributing).

Run:  python -m Scripts.atlop.smoke_test_gat_mlp
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, BertConfig, BertModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_dataset import DocREDataset            # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES     # noqa: E402
from Scripts.atlop.preprocess_full import build_features_full  # noqa: E402
from Scripts.atlop.re_model_gat_mlp import DocREModelGATMLP     # noqa: E402
from Scripts.atlop.train_full import make_collate_fn      # noqa: E402

N_DOCS = 4
EMB_SIZE = 64
BLOCK_SIZE = 16
GRAPH_DIM = 32


def tiny_model(tokenizer, evi_weight):
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=EMB_SIZE, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=1300,
        attn_implementation="eager",
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    torch.manual_seed(0)
    return DocREModelGATMLP(config, BertModel(config, add_pooling_layer=False),
                            emb_size=EMB_SIZE, block_size=BLOCK_SIZE, num_labels=NUM_CLASSES,
                            graph_layers=2, graph_dim=GRAPH_DIM, graph_heads=4,
                            mlp_hidden=GRAPH_DIM, evi_weight=evi_weight)


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    rel2id = build_rel2id()
    docs = [DocREDataset("dev")[i] for i in range(N_DOCS)]
    features = build_features_full(docs, tokenizer, rel2id, show_progress=False)

    n_evi = sum(1 for e in features[0]["evidence"] if e)
    print(f"[features] doc0: pairs={len(features[0]['hts'])} pairs_with_evidence={n_evi}")
    assert n_evi > 0, "expected some dev pairs to carry gold evidence"

    collate = make_collate_fn(tokenizer.pad_token_id)
    batch = collate(features)
    kwargs = dict(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        entity_pos=batch["entity_pos"], hts=batch["hts"], sent_pos=batch["sent_pos"],
    )
    total_pairs = sum(len(f["hts"]) for f in features)

    # 1) fwd/bwd, all params reached
    model = tiny_model(tokenizer, evi_weight=0.2)
    model.train()
    loss, preds = model(labels=batch["labels"], evidence=batch["evidence"], **kwargs)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    # bilinear is inherited from DocREModel but unused here (classifier head is
    # the GAT+MLP path, not grouped bilinear) -- excluded from this check on purpose.
    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None and "bilinear" not in n]
    assert not no_grad, f"params without grad: {no_grad}"
    print(f"[fwd/bwd] loss={loss.item():.4f}  all reachable params have grad (bilinear excluded, unused)")

    # 2) get_label shape + Na-or-relation invariant
    assert preds.shape == (total_pairs, NUM_CLASSES)
    assert (preds.sum(1) >= 1).all(), "every pair must predict >=1 class (relation or Na)"
    print(f"[get_label] preds shape={tuple(preds.shape)} OK")

    # 3) evidence-contrastive term actually contributes
    m_on = tiny_model(tokenizer, evi_weight=0.5)
    m_off = tiny_model(tokenizer, evi_weight=0.0)
    m_off.load_state_dict(m_on.state_dict())  # identical weights
    m_on.eval(), m_off.eval()
    with torch.no_grad():
        loss_on = m_on(labels=batch["labels"], evidence=batch["evidence"], **kwargs)[0]
        loss_off = m_off(labels=batch["labels"], evidence=batch["evidence"], **kwargs)[0]
    assert loss_on.item() != loss_off.item(), "evidence contrastive loss did not change the total"
    print(f"[evi_weight] 0.5 vs 0.0: {loss_on.item():.4f} vs {loss_off.item():.4f} "
          f"(delta={loss_on.item() - loss_off.item():+.4f})")

    print("\nGAT+MLP SMOKE TEST PASSED - BCE + evidence-contrastive wired end-to-end.")


if __name__ == "__main__":
    main()
