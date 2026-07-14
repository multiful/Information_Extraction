"""CPU wiring check for the integrated model (re_model_full.DocREModelFull).

Random-init tiny BERT on a few dev docs — proves the whole pipeline wires up,
NOT accuracy. Checks:

  1. features      - build_features_full adds sent_pos (covers the token axis)
                     and evidence (dev docs carry gold evidence sentences).
  2. evidence dist - per-pair p_evi is a proper distribution over sentences
                     (>=0, sums to ~1).
  3. fwd/bwd       - relation loss + evidence loss finite, every param reached.
  4. evidence loss - turning evi_lambda on changes the loss vs off (the DREEAM
                     supervision term is actually contributing).

Run:  python -m Scripts.atlop.smoke_test_full
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
from Scripts.atlop.re_model_full import DocREModelFull    # noqa: E402
from Scripts.atlop.train_full import make_collate_fn      # noqa: E402

N_DOCS = 4
EMB_SIZE = 64
BLOCK_SIZE = 16
GRAPH_DIM = 32


def tiny_model(tokenizer, evi_lambda):
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=EMB_SIZE, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=1300,
        attn_implementation="eager",
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    torch.manual_seed(0)
    return DocREModelFull(config, BertModel(config, add_pooling_layer=False),
                          emb_size=EMB_SIZE, block_size=BLOCK_SIZE, num_labels=NUM_CLASSES,
                          graph_layers=2, graph_dim=GRAPH_DIM, graph_heads=4,
                          evi_lambda=evi_lambda)


def main():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    rel2id = build_rel2id()
    docs = [DocREDataset("dev")[i] for i in range(N_DOCS)]
    features = build_features_full(docs, tokenizer, rel2id, show_progress=False)

    f0 = features[0]
    n_sent = len(docs[0]["sents"])
    assert len(f0["sent_pos"]) == n_sent, "sent_pos must have one span per sentence"
    n_evi = sum(1 for e in f0["evidence"] if e)
    print(f"[features] doc0: sents={n_sent} pairs={len(f0['hts'])} "
          f"sent_pos={len(f0['sent_pos'])} pairs_with_evidence={n_evi}")
    assert n_evi > 0, "expected some dev pairs to carry gold evidence"

    collate = make_collate_fn(tokenizer.pad_token_id)
    batch = collate(features)
    kwargs = dict(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
        entity_pos=batch["entity_pos"], hts=batch["hts"], sent_pos=batch["sent_pos"],
    )

    # 2) evidence distribution sanity via the model's internal method
    model = tiny_model(tokenizer, evi_lambda=0.1)
    model.eval()
    with torch.no_grad():
        seq, att = model.encode(batch["input_ids"], batch["attention_mask"])
        _, _, _, evi_list = model.get_hrt_evidence(
            seq, att, batch["entity_pos"], batch["hts"], batch["sent_pos"])
    p0 = evi_list[0]
    assert (p0 >= 0).all(), "p_evi has negatives"
    sums = p0.sum(1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4), \
        f"p_evi must sum to 1 per pair, got {sums[:3]}"
    print(f"[evidence dist] p_evi shape={tuple(p0.shape)} sums≈1 OK "
          f"(gate g={torch.sigmoid(model.evi_gate).item():.3f})")

    # 3) forward/backward, all params reached
    model.train()
    loss, preds = model(labels=batch["labels"], evidence=batch["evidence"], **kwargs)
    assert preds.shape == (sum(len(f["hts"]) for f in features), NUM_CLASSES)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"params without grad: {no_grad}"
    print(f"[fwd/bwd] loss={loss.item():.4f}  all params have grad "
          f"(preds {tuple(preds.shape)})")

    # 4) evidence term actually contributes
    m_on = tiny_model(tokenizer, evi_lambda=0.5)
    m_off = tiny_model(tokenizer, evi_lambda=0.0)
    m_off.load_state_dict(m_on.state_dict())  # identical weights
    m_on.eval(), m_off.eval()
    with torch.no_grad():
        loss_on = m_on(labels=batch["labels"], evidence=batch["evidence"], **kwargs)[0]
        loss_off = m_off(labels=batch["labels"], evidence=batch["evidence"], **kwargs)[0]
    assert loss_on.item() != loss_off.item(), "evidence loss did not change the total"
    print(f"[evidence loss] evi_lambda 0.5 vs 0.0: {loss_on.item():.4f} vs "
          f"{loss_off.item():.4f}  (Δ={loss_on.item() - loss_off.item():+.4f})")

    print("\nFULL SMOKE TEST PASSED - DREEAM-LCP + GREP-GAT + PU wired end-to-end.")


if __name__ == "__main__":
    main()
