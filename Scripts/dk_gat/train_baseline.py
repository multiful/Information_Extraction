"""Train / evaluate the plain ATLOP-reproduction baseline (no GAT, see
Scripts/dk_gat/model_baseline.py) on DocRED.

Exact architectural reproduction of the validated baseline (BERT-base-cased,
Sliding Window, Entity Marker + logsumexp Pooling, Localized Context Pooling,
[Entity;Context]->Linear->Tanh, Grouped Bilinear, ATLoss, Adaptive
Thresholding -- dev F1 61.71 / Ign F1 59.86 on the original
train_distant+train_annotated/dev split, see Scripts/dk_gat/README.md).
Self-contained inside Scripts/dk_gat -- Scripts/atlop is a teammate's track,
not imported from (see README.md's jurisdiction note); only the dataset
differs from that original run, not the model/training logic.

Two-stage flow mirrors Scripts/atlop/train_re.py's paper order: distant
pretrain -> annotated fine-tune, both with plain ATLoss (no PU loss --
the validated baseline recipe didn't use it).

Run from the project root:

    # quick CPU sanity run
    python -m Scripts.dk_gat.train_baseline --limit_docs 6 --epochs 1 --distant_epochs 1

    # exact baseline reproduction (original named splits, GPU)
    python -m Scripts.dk_gat.train_baseline --distant_limit 20000 --distant_epochs 1 \
        --epochs 15 --run_name atlop_baseline_dkgat --save_model --seed 66

    # revised-data run (no distant stage), matching the dk_gat GAT track's
    # current data config -- only the model differs from that run
    python -m Scripts.dk_gat.train_baseline \
        --train_split docred_data/data/train_revised.json \
        --dev_split docred_data/data/dev_revised.json \
        --test_file docred_data/data/test_revised.json \
        --distant_epochs 0 --epochs 20 \
        --run_name baseline_revised --save_model --seed 66
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data import docred_io                              # noqa: E402
from data.docred_dataset import DocREDataset            # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES     # noqa: E402
from Scripts.atlop.losses import ATLoss                  # noqa: E402
from Scripts.dk_gat.model_baseline import DocREBaselineModel        # noqa: E402
from Scripts.dk_gat.preprocess_baseline import build_baseline_features  # noqa: E402
from Scripts.eval.scorer import evaluate                  # noqa: E402

RESULTS_DIR = ROOT / "results"


def load_docs(split_or_path: str) -> list:
    """Named split (data.docred_io.SPLITS) via DocREDataset, or a path
    (absolute, or relative to the project root) to a DocRED-format json file
    otherwise -- see Scripts/dk_gat/train_gat.py's load_docs for the same
    convention."""
    if split_or_path in docred_io.SPLITS:
        return list(DocREDataset(split_or_path))
    path = Path(split_or_path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_collate_fn(pad_token_id: int):
    def collate(features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = torch.full((len(features), max_len), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(features), max_len), dtype=torch.long)
        for i, f in enumerate(features):
            n = len(f["input_ids"])
            input_ids[i, :n] = torch.tensor(f["input_ids"], dtype=torch.long)
            attention_mask[i, :n] = 1
        pos_lists = [ids for f in features for ids in f["labels"]]
        labels = torch.zeros((len(pos_lists), NUM_CLASSES), dtype=torch.float)
        for i, ids in enumerate(pos_lists):
            labels[i, ids] = 1.0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "entity_pos": [f["entity_pos"] for f in features],
            "hts": [f["hts"] for f in features],
            "labels": labels,
            "features": features,
        }
    return collate


@torch.no_grad()
def predict(model, loader, id2rel, device) -> list:
    model.eval()
    out = []
    for batch in loader:
        preds = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            entity_pos=batch["entity_pos"],
            hts=batch["hts"],
        )[0].cpu().numpy()
        idx = 0
        for f in batch["features"]:
            n = len(f["hts"])
            doc_preds = preds[idx: idx + n]
            idx += n
            for (h, t), row in zip(f["hts"], doc_preds):
                for r in range(1, NUM_CLASSES):
                    if row[r] == 1:
                        out.append({"title": f["title"], "h_idx": h, "t_idx": t, "r": id2rel[r]})
    model.train()
    return out


def build_optim_sched(model, args, total_steps):
    new_layers = ("head_extractor", "tail_extractor", "bilinear")
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in new_layers)],
         "lr": args.encoder_lr},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in new_layers)],
         "lr": args.classifier_lr},
    ]
    optimizer = torch.optim.AdamW(grouped, eps=1e-6)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


def run_stage(model, loader, args, device, epochs, stage, dev_loader, dev_docs, ign_docs, id2rel):
    total_steps = max(1, len(loader) * epochs)
    optimizer, scheduler = build_optim_sched(model, args, total_steps)
    metrics, preds = None, None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader):
            loss = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                entity_pos=batch["entity_pos"],
                hts=batch["hts"],
                labels=batch["labels"],
            )[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if (step + 1) % args.log_every == 0:
                print(f"  [{stage}] epoch {epoch} step {step + 1}/{len(loader)} "
                      f"loss {running / (step + 1):.4f}", flush=True)

        preds = predict(model, dev_loader, id2rel, device)
        metrics = evaluate(preds, dev_docs, ign_docs)
        print(f"[{stage} | epoch {epoch}] train_loss={running / max(1, len(loader)):.4f} "
              f"dev_F1={metrics['f1'] * 100:.2f} Ign_F1={metrics['ign_f1'] * 100:.2f} "
              f"(P={metrics['precision'] * 100:.2f} R={metrics['recall'] * 100:.2f})", flush=True)
    return metrics, preds


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device}", flush=True)

    rel2id = build_rel2id()
    id2rel = {v: k for k, v in rel2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    collate = make_collate_fn(tokenizer.pad_token_id)

    train_docs = load_docs(args.train_split)
    dev_docs = load_docs(args.dev_split)
    if args.limit_docs > 0:
        train_docs = train_docs[: args.limit_docs]
        dev_docs = dev_docs[: args.limit_docs]
    print(f"[data] train={len(train_docs)} dev={len(dev_docs)} docs", flush=True)

    dev_loader = DataLoader(build_baseline_features(dev_docs, tokenizer, rel2id),
                            batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    train_loader = DataLoader(build_baseline_features(train_docs, tokenizer, rel2id),
                              batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=NUM_CLASSES)
    encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config,
                                        attn_implementation="eager")
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = DocREBaselineModel(config, encoder, emb_size=args.emb_size,
                               block_size=args.block_size, num_labels=NUM_CLASSES).to(device)

    if args.distant_epochs > 0:
        distant_docs = load_docs(args.distant_split)
        cap = args.limit_docs if args.limit_docs > 0 else args.distant_limit
        if cap > 0:
            distant_docs = distant_docs[:cap]
        print(f"[stage 1] distant pretrain on {len(distant_docs)} docs "
              f"({args.distant_epochs} epoch(s))", flush=True)
        distant_loader = DataLoader(build_baseline_features(distant_docs, tokenizer, rel2id),
                                    batch_size=args.distant_batch_size, shuffle=True,
                                    collate_fn=collate)
        metrics, preds = run_stage(model, distant_loader, args, device, args.distant_epochs,
                                   "distant-pretrain", dev_loader, dev_docs, train_docs, id2rel)
        del distant_loader, distant_docs
        if args.save_model:
            RESULTS_DIR.mkdir(exist_ok=True)
            stage1_ckpt = RESULTS_DIR / f"{args.run_name}_stage1.pt"
            torch.save(model.state_dict(), stage1_ckpt)
            print(f"[saved] {stage1_ckpt}  (distant-pretrain checkpoint)", flush=True)

    if args.epochs > 0:
        print(f"[stage 2] annotated fine-tune on {len(train_docs)} docs ({args.epochs} epoch(s))",
              flush=True)
        metrics, preds = run_stage(model, train_loader, args, device, args.epochs,
                                   "annotated-finetune", dev_loader, dev_docs, train_docs, id2rel)
    else:
        print("[stage 2] skipped (--epochs 0) -- reporting stage-1 metrics/predictions", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    pred_path = RESULTS_DIR / f"{args.run_name}_dev_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as fp:
        json.dump(preds, fp, ensure_ascii=False)
    print(f"[saved] {pred_path}  ({len(preds)} predicted relations)", flush=True)

    if args.save_model:
        ckpt = RESULTS_DIR / f"{args.run_name}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"[saved] {ckpt}", flush=True)

    if args.test_file:
        test_docs = load_docs(args.test_file)
        test_loader = DataLoader(build_baseline_features(test_docs, tokenizer, rel2id),
                                 batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
        test_preds = predict(model, test_loader, id2rel, device)
        test_pred_path = RESULTS_DIR / f"{args.run_name}_test_predictions.json"
        with open(test_pred_path, "w", encoding="utf-8") as fp:
            json.dump(test_preds, fp, ensure_ascii=False)
        print(f"[saved] {test_pred_path}  ({len(test_preds)} predicted relations)", flush=True)
        if any(doc.get("labels") for doc in test_docs):
            test_metrics = evaluate(test_preds, test_docs, train_docs)
            print(f"[test] F1={test_metrics['f1'] * 100:.2f} "
                  f"Ign_F1={test_metrics['ign_f1'] * 100:.2f} "
                  f"(P={test_metrics['precision'] * 100:.2f} R={test_metrics['recall'] * 100:.2f})",
                  flush=True)

    return metrics


def build_argparser():
    p = argparse.ArgumentParser(description="Plain ATLOP-reproduction baseline (no GAT) on DocRED")
    p.add_argument("--model_name_or_path", default="bert-base-cased")
    p.add_argument("--run_name", default="atlop_baseline_dkgat")
    p.add_argument("--train_split", default="train_annotated",
                   help="named split (data/docred_io.SPLITS) or a path (absolute, or relative "
                        "to the project root) to a DocRED-format json file")
    p.add_argument("--dev_split", default="dev", help="named split or json path")
    p.add_argument("--distant_split", default="train_distant", help="named split or json path")
    p.add_argument("--test_file", default=None,
                   help="optional json path for a held-out split to run final triple "
                        "prediction on after training (F1/Ign F1 also printed if labeled)")
    p.add_argument("--epochs", type=int, default=15, help="annotated fine-tune epochs")
    p.add_argument("--distant_epochs", type=int, default=1, help="0 = skip distant stage")
    p.add_argument("--distant_limit", type=int, default=20000)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--distant_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--encoder_lr", type=float, default=5e-5)
    p.add_argument("--classifier_lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--emb_size", type=int, default=768)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--limit_docs", type=int, default=0, help="cap all splits; for quick runs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
