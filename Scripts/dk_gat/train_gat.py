"""Train / evaluate the dk EGAT model on DocRED.

Two-stage flow mirroring Scripts/atlop/train_re.py (distant pretrain ->
annotated fine-tune), with the differences dictated by this model's design:
BCEWithLogitsLoss + sigmoid decoding, so the NA imbalance is handled by a
dev-set threshold sweep (PRD section-2 requirement) instead of adaptive
thresholding; plus the 0.2-weighted evidence contrastive loss, which is
active only on splits that carry evidence annotations (train_annotated) and
silently inert on train_distant.

Run from the project root:

    # quick CPU sanity run
    python -m Scripts.dk_gat.train_gat --limit_docs 6 --epochs 1 --distant_epochs 1

    # full run (Colab A100): distant 20k x 1 epoch -> annotated 15 epochs
    # (matches Scripts/atlop baseline's schedule exactly -- only the
    # architecture differs, so the comparison isolates that one variable)
    python -m Scripts.dk_gat.train_gat --distant_limit 20000 --distant_epochs 1 \
        --epochs 15 --run_name dk_gat --save_model --seed 66
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

from data.docred_dataset import DocREDataset            # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES     # noqa: E402
from Scripts.dk_gat.model import DocREGATModel            # noqa: E402
from Scripts.dk_gat.preprocess_gat import build_gat_features  # noqa: E402
from Scripts.eval.scorer import evaluate                  # noqa: E402

RESULTS_DIR = ROOT / "results"
THRESHOLDS = [round(0.1 + 0.05 * i, 2) for i in range(17)]  # 0.10 .. 0.90


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
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "features": features}
    return collate


@torch.no_grad()
def collect_probs(model, loader, device):
    """Sigmoid probabilities for every pair, kept per-doc for threshold sweep."""
    model.eval()
    out = []  # (title, hts, probs ndarray (P, 97))
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                       batch["features"])[0]
        probs = torch.sigmoid(logits).cpu().numpy()
        idx = 0
        for f in batch["features"]:
            n = len(f["hts"])
            out.append((f["title"], f["hts"], probs[idx: idx + n]))
            idx += n
    return out


def probs_to_preds(doc_probs, id2rel, threshold: float):
    preds = []
    for title, hts, probs in doc_probs:
        for (h, t), row in zip(hts, probs):
            for r in range(1, NUM_CLASSES):  # skip Na column
                if row[r] > threshold:
                    preds.append({"title": title, "h_idx": h, "t_idx": t, "r": id2rel[r]})
    return preds


def sweep_eval(model, dev_loader, dev_docs, ign_docs, id2rel, device):
    """Dev evaluation with a threshold sweep; returns (best_metrics, best_preds,
    best_threshold)."""
    doc_probs = collect_probs(model, dev_loader, device)
    best = (None, None, None)
    for th in THRESHOLDS:
        preds = probs_to_preds(doc_probs, id2rel, th)
        if not preds:
            continue
        m = evaluate(preds, dev_docs, ign_docs)
        if best[0] is None or m["f1"] > best[0]["f1"]:
            best = (m, preds, th)
    if best[0] is None:  # no threshold produced any prediction
        best = ({"f1": 0.0, "ign_f1": 0.0, "precision": 0.0, "recall": 0.0}, [], 0.5)
    return best


def run_stage(model, loader, args, device, epochs, stage, dev_loader, dev_docs, ign_docs, id2rel):
    total_steps = max(1, len(loader) * epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    metrics, preds, th = None, None, None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader):
            loss, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                            batch["features"], labels=batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if (step + 1) % args.log_every == 0:
                print(f"  [{stage}] epoch {epoch} step {step + 1}/{len(loader)} "
                      f"loss {running / (step + 1):.4f}", flush=True)
        metrics, preds, th = sweep_eval(model, dev_loader, dev_docs, ign_docs, id2rel, device)
        print(f"[{stage} | epoch {epoch}] train_loss={running / max(1, len(loader)):.4f} "
              f"dev_F1={metrics['f1'] * 100:.2f} Ign_F1={metrics['ign_f1'] * 100:.2f} "
              f"(P={metrics['precision'] * 100:.2f} R={metrics['recall'] * 100:.2f} "
              f"threshold={th})", flush=True)
    return metrics, preds


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device}", flush=True)

    rel2id = build_rel2id()
    id2rel = {v: k for k, v in rel2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    collate = make_collate_fn(tokenizer.pad_token_id)

    train_docs = list(DocREDataset("train_annotated"))
    dev_docs = list(DocREDataset("dev"))
    if args.limit_docs > 0:
        train_docs = train_docs[: args.limit_docs]
        dev_docs = dev_docs[: args.limit_docs]
    print(f"[data] train={len(train_docs)} dev={len(dev_docs)} docs", flush=True)

    dev_loader = DataLoader(build_gat_features(dev_docs, tokenizer, rel2id),
                            batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
    train_loader = DataLoader(build_gat_features(train_docs, tokenizer, rel2id),
                              batch_size=args.train_batch_size, shuffle=True, collate_fn=collate)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=NUM_CLASSES)
    encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config,
                                        attn_implementation="eager")
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = DocREGATModel(config, encoder, num_labels=NUM_CLASSES,
                          num_heads=args.gat_heads, dropout=args.dropout,
                          evidence_weight=args.evidence_weight).to(device)

    if args.distant_epochs > 0:
        distant_docs = list(DocREDataset("train_distant"))
        cap = args.limit_docs if args.limit_docs > 0 else args.distant_limit
        if cap > 0:
            distant_docs = distant_docs[:cap]
        print(f"[stage 1] distant pretrain on {len(distant_docs)} docs "
              f"({args.distant_epochs} epoch(s))", flush=True)
        distant_loader = DataLoader(build_gat_features(distant_docs, tokenizer, rel2id),
                                    batch_size=args.distant_batch_size, shuffle=True,
                                    collate_fn=collate)
        run_stage(model, distant_loader, args, device, args.distant_epochs,
                  "distant-pretrain", dev_loader, dev_docs, train_docs, id2rel)
        del distant_loader, distant_docs
        if args.save_model:
            RESULTS_DIR.mkdir(exist_ok=True)
            p = RESULTS_DIR / f"{args.run_name}_stage1.pt"
            torch.save(model.state_dict(), p)
            print(f"[saved] {p}", flush=True)

    print(f"[stage 2] annotated fine-tune on {len(train_docs)} docs ({args.epochs} epoch(s))",
          flush=True)
    metrics, preds = run_stage(model, train_loader, args, device, args.epochs,
                               "annotated-finetune", dev_loader, dev_docs, train_docs, id2rel)

    RESULTS_DIR.mkdir(exist_ok=True)
    pred_path = RESULTS_DIR / f"{args.run_name}_dev_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as fp:
        json.dump(preds, fp, ensure_ascii=False)
    print(f"[saved] {pred_path}  ({len(preds)} predicted relations)", flush=True)
    if args.save_model:
        ckpt = RESULTS_DIR / f"{args.run_name}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"[saved] {ckpt}", flush=True)
    return metrics


def build_argparser():
    p = argparse.ArgumentParser(description="dk EGAT model on DocRED")
    p.add_argument("--model_name_or_path", default="bert-base-cased")
    p.add_argument("--run_name", default="dk_gat")
    p.add_argument("--epochs", type=int, default=15,
                   help="annotated fine-tune epochs (matches Scripts/atlop baseline's 15 "
                        "for a controlled architecture-only comparison)")
    p.add_argument("--distant_epochs", type=int, default=1,
                   help="0 = skip distant stage (matches Scripts/atlop baseline's 1)")
    p.add_argument("--distant_limit", type=int, default=20000)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--distant_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--gat_heads", type=int, default=4)
    p.add_argument("--evidence_weight", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--limit_docs", type=int, default=0, help="cap all splits; for quick runs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
