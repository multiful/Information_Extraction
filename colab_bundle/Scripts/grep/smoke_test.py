"""End-to-end correctness check for the GREP pipeline on CPU.

Mirrors Scripts/atlop/smoke_test.py: a tiny RANDOM-init BERT (no pretrained
download) on a handful of dev docs, proving every stage wires up correctly --
extended preprocessing (sent_pos/evidence/doc_rel_labels) -> encoder ->
Entity Pair Reasoning graph -> Evidence Extraction / Global Relation
Prediction losses (forward+backward, both with and without the evidence
loss) -> Inference Fusion (pseudo-document construction + fused decode) ->
common prediction format -> shared scorer.

Run:  python -m Scripts.grep.smoke_test
Real F1 numbers come from train_grep.py with a real pretrained encoder on GPU.
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
from Scripts.eval.scorer import evaluate                # noqa: E402
from Scripts.grep.re_model import GREPModel              # noqa: E402
from Scripts.grep.train_grep import (                    # noqa: E402
    inference_fusion, make_collate_fn, predict,
)

N_DOCS = 6
EMB_SIZE = 64
BLOCK_SIZE = 16
NODE_DIM = 32
GRAPH_HEADS = 4


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
    print(f"[preprocess] doc0: sent_pos={len(f0['sent_pos'])} sentences, "
          f"evidence(first 3)={f0['evidence'][:3]}, doc_rel_labels={f0['doc_rel_labels']}")
    assert len(f0["sent_pos"]) == len(docs[0]["sents"])
    assert len(f0["evidence"]) == len(f0["hts"])

    def build_model():
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=EMB_SIZE, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=128, max_position_embeddings=1300,
            attn_implementation="eager",
        )
        encoder = BertModel(config, add_pooling_layer=False)
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        return GREPModel(config, encoder, emb_size=EMB_SIZE, block_size=BLOCK_SIZE,
                         num_labels=NUM_CLASSES, node_dim=NODE_DIM, graph_heads=GRAPH_HEADS)

    collate = make_collate_fn(tokenizer.pad_token_id)
    batch = collate(features[:3])

    for use_evidence_loss in (True, False):
        model = build_model()
        loss, preds, all_u, logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            entity_pos=batch["entity_pos"],
            hts=batch["hts"],
            sent_pos=batch["sent_pos"],
            evidence=batch["evidence"],
            doc_rel_labels=batch["doc_rel_labels"],
            labels=batch["labels"],
            use_evidence_loss=use_evidence_loss,
        )
        print(f"[model use_evidence_loss={use_evidence_loss}] loss={loss.item():.4f} "
              f"preds shape={tuple(preds.shape)} logits shape={tuple(logits.shape)} "
              f"u shapes={[tuple(u.shape) for u in all_u]}")
        assert logits.shape == preds.shape
        assert preds.shape[1] == NUM_CLASSES
        assert preds.shape[0] == sum(len(f["hts"]) for f in features[:3])
        assert len(all_u) == 3
        loss.backward()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        print(f"[model use_evidence_loss={use_evidence_loss}] backward OK, "
              f"{sum(grads)}/{len(grads)} params have grads")
        assert all(grads), "some params did not receive gradients"

    # predict() -> common format -> scorer (random weights, not a real score)
    model_full = build_model()
    model_no_evi = build_model()
    from torch.utils.data import DataLoader
    loader = DataLoader(features, batch_size=2, shuffle=False, collate_fn=collate)
    predictions = predict(model_full, loader, id2rel, torch.device("cpu"))
    print(f"[predict] {len(predictions)} predicted relations, sample: "
          f"{predictions[0] if predictions else '(none — random weights)'}")
    metrics = evaluate(predictions, docs, docs)
    print(f"[scorer] F1={metrics['f1'] * 100:.2f} Ign_F1={metrics['ign_f1'] * 100:.2f}")

    # Inference Fusion (Eq 22) -- exercises pseudo-document construction end-to-end.
    fused_preds, gamma_used, fused_metrics = inference_fusion(
        model_full, model_no_evi, docs, features, tokenizer, rel2id, id2rel,
        torch.device("cpu"), gamma=0.0, sweep=False, ign_docs=docs,
    )
    print(f"[inference_fusion] gamma={gamma_used} fused_preds={len(fused_preds)} "
          f"F1={fused_metrics['f1'] * 100:.2f}")

    print("\nSMOKE TEST PASSED - GREP pipeline is wired end-to-end.")


if __name__ == "__main__":
    main()
