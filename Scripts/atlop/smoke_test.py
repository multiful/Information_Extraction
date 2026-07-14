"""End-to-end correctness check for the ATLOP pipeline on CPU.

This does NOT measure accuracy — it uses a tiny RANDOM-init BERT (no pretrained
download) on a handful of dev docs to prove every stage wires up correctly:
preprocessing + `*` markers -> encoder -> log-sum-exp entity pooling ->
localized context pooling -> bilinear classifier -> ATLoss (forward+backward) ->
get_label -> common prediction format -> shared scorer.

Run:  python -m Scripts.atlop.smoke_test
Real F1 numbers come from train_re.py with a real pretrained encoder on GPU.
"""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, BertConfig, BertModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_dataset import DocREDataset          # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES   # noqa: E402
from Scripts.atlop.preprocess import build_features     # noqa: E402
from Scripts.atlop.re_model import DocREModel           # noqa: E402
from Scripts.atlop.train_re import make_collate_fn, predict  # noqa: E402
from Scripts.eval.scorer import evaluate                # noqa: E402

N_DOCS = 6
EMB_SIZE = 64
BLOCK_SIZE = 16


def main():
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    rel2id = build_rel2id()
    id2rel = {v: k for k, v in rel2id.items()}

    docs = [DocREDataset("dev")[i] for i in range(N_DOCS)]
    print(f"[data] {len(docs)} dev docs, entities per doc: "
          f"{[len(d['vertexSet']) for d in docs]}")

    features = build_features(docs, tokenizer, rel2id, show_progress=False)
    f0 = features[0]
    print(f"[preprocess] doc0: input_ids={len(f0['input_ids'])} "
          f"entities={f0['num_entities']} pairs={len(f0['hts'])} "
          f"labels(sparse pos-id lists, first 3)={f0['labels'][:3]}")
    # sanity: a marker '*' (id 115) should surround the first mention's first token
    star_id = tokenizer.convert_tokens_to_ids("*")
    first_start = f0["entity_pos"][0][0][0] + 1  # +1 for [CLS]
    assert f0["input_ids"][first_start] == star_id, "start marker not at recorded position"
    print(f"[preprocess] marker check OK (token at entity0/mention0 start is '*')")

    # tiny random encoder — offline, no pretrained weights
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=EMB_SIZE, num_hidden_layers=2, num_attention_heads=4,
        intermediate_size=128, max_position_embeddings=1300,
        attn_implementation="eager",  # required for output_attentions=True
    )
    # add_pooling_layer=False: we only use last_hidden_state + attentions, so the
    # pooler would sit unused (and gradient-less). Dropping it keeps this check clean.
    encoder = BertModel(config, add_pooling_layer=False)
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = DocREModel(config, encoder, emb_size=EMB_SIZE, block_size=BLOCK_SIZE,
                       num_labels=NUM_CLASSES)

    collate = make_collate_fn(tokenizer.pad_token_id)
    batch = collate(features[:3])

    # forward + loss + backward
    loss, preds = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        entity_pos=batch["entity_pos"],
        hts=batch["hts"],
        labels=batch["labels"],
    )
    print(f"[model] loss={loss.item():.4f}  preds shape={tuple(preds.shape)}")
    assert preds.shape[1] == NUM_CLASSES
    assert preds.shape[0] == sum(len(f["hts"]) for f in features[:3])
    loss.backward()
    grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
    print(f"[model] backward OK, {sum(grads)}/{len(grads)} params have grads")
    assert all(grads), "some params did not receive gradients"

    # DREEAM evidence-guided attention: sent_pos/evidence wiring + backward.
    model.zero_grad()
    n_sent = [len(f["sent_pos"]) for f in features[:3]]
    n_evi_pairs = sum(1 for f in features[:3] for evi in f["evidence"] if evi)
    print(f"[dreeam] sentences per doc={n_sent}, pairs with gold evidence={n_evi_pairs}")
    loss_evi, preds_evi = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        entity_pos=batch["entity_pos"],
        hts=batch["hts"],
        labels=batch["labels"],
        sent_pos=batch["sent_pos"],
        evidence=batch["evidence"],
        evi_lambda=0.1,
    )
    # (predictions will differ slightly from the base run above due to encoder
    # dropout noise between forward passes, not the evidence loss itself)
    assert preds_evi.shape == preds.shape
    print(f"[dreeam] loss_with_evidence={loss_evi.item():.4f} (base={loss.item():.4f})")
    loss_evi.backward()
    grads_evi = [p.grad is not None for p in model.parameters() if p.requires_grad]
    print(f"[dreeam] backward OK, {sum(grads_evi)}/{len(grads_evi)} params have grads")
    assert all(grads_evi), "some params did not receive gradients with evidence loss enabled"

    # predict -> common format -> scorer
    from torch.utils.data import DataLoader
    loader = DataLoader(features, batch_size=2, shuffle=False, collate_fn=collate)
    predictions = predict(model, loader, id2rel, torch.device("cpu"))
    print(f"[predict] {len(predictions)} predicted relations, sample: "
          f"{predictions[0] if predictions else '(none — random weights)'}")

    metrics = evaluate(predictions, docs, docs)  # train_docs=docs just to exercise Ign path
    print(f"[scorer] F1={metrics['f1'] * 100:.2f} Ign_F1={metrics['ign_f1'] * 100:.2f} "
          f"P={metrics['precision'] * 100:.2f} R={metrics['recall'] * 100:.2f} "
          f"gold={metrics['num_gold']} submitted={metrics['num_submitted']}")

    print("\nSMOKE TEST PASSED - pipeline is wired end-to-end.")


if __name__ == "__main__":
    main()
