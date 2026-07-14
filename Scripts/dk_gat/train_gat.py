"""Train / evaluate the dk EGAT model on DocRED.

Two-stage flow mirroring Scripts/atlop/train_re.py (distant pretrain ->
annotated fine-tune): Adaptive Thresholding, with PUATLoss(na_weight=0.7)
swapped in for the distant stage only (train_distant's Na labels are
distant-supervision noise, not confirmed negatives -- see
Scripts/atlop/PU_THRESHOLD_EXPERIMENT.md) and plain ATLoss for annotated
fine-tune (its Na labels are gold). Plus the 0.2-weighted evidence
contrastive loss, active only on splits with evidence annotations
(train_annotated) and silently inert on train_distant.

Note: this file originally used BCEWithLogitsLoss + a dev threshold sweep.
Switched after a real run showed it measurably underperforming (dev F1
24.77 after distant pretrain on 20k docs, vs 43.15 for RoBERTa+LCP+ATLoss
on the same subset) -- see model.py's module docstring for the diagnosis.

Run from the project root:

    # quick CPU sanity run
    python -m Scripts.dk_gat.train_gat --limit_docs 6 --epochs 1 --distant_epochs 1

    # full run (Colab A100): distant 20k x 1 epoch -> annotated 15 epochs
    # (matches Scripts/atlop baseline's schedule exactly -- only the
    # architecture differs, so the comparison isolates that one variable)
    python -m Scripts.dk_gat.train_gat --distant_limit 20000 --distant_epochs 1 \
        --epochs 15 --use_pu_loss --na_weight 0.7 \
        --run_name dk_gat --save_model --seed 66
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
from Scripts.atlop.losses import ATLoss, PUATLoss          # noqa: E402
from Scripts.dk_gat.model import DocREGATModel            # noqa: E402
from Scripts.dk_gat.preprocess_gat import build_gat_features  # noqa: E402
from Scripts.eval.scorer import evaluate                  # noqa: E402

RESULTS_DIR = ROOT / "results"


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
def predict(model, loader, id2rel, device):
    """Adaptive-Thresholding decode: a relation is emitted iff its logit beats
    the pair's own learned TH (class 0) logit -- no global threshold, matches
    Scripts/atlop/train_re.py's predict()."""
    model.eval()
    out = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                       batch["features"])[0]
        # get_label is inherited unchanged by PUATLoss from ATLoss, so this works
        # regardless of which loss_fnt is currently attached to the model.
        mask = model.loss_fnt.get_label(logits, num_labels=-1).cpu().numpy()
        idx = 0
        for f in batch["features"]:
            n = len(f["hts"])
            doc_mask = mask[idx: idx + n]
            idx += n
            for (h, t), row in zip(f["hts"], doc_mask):
                for r in range(1, NUM_CLASSES):
                    if row[r] == 1:
                        out.append({"title": f["title"], "h_idx": h, "t_idx": t, "r": id2rel[r]})
    model.train()
    return out


def run_stage(model, loader, args, device, epochs, stage, dev_loader, dev_docs, ign_docs, id2rel,
             best_ckpt_path=None):
    """best_ckpt_path: if given, save model.state_dict() there every time a new
    best dev F1 is seen (separate from the caller's own final-epoch save) --
    epoch-to-epoch dev F1 isn't monotonic (we've observed real dips, e.g.
    epoch 6 59.85 -> epoch 7 59.57 on a real run), so whichever epoch happens
    to be last isn't guaranteed to be the best one actually reached."""
    total_steps = max(1, len(loader) * epochs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    metrics, preds = None, None
    best_f1, best_epoch = -1.0, -1
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
        preds = predict(model, dev_loader, id2rel, device)
        metrics = evaluate(preds, dev_docs, ign_docs)
        is_best = metrics["f1"] > best_f1
        print(f"[{stage} | epoch {epoch}] train_loss={running / max(1, len(loader)):.4f} "
              f"dev_F1={metrics['f1'] * 100:.2f} Ign_F1={metrics['ign_f1'] * 100:.2f} "
              f"(P={metrics['precision'] * 100:.2f} R={metrics['recall'] * 100:.2f})"
              f"{'  <- new best' if is_best else ''}", flush=True)
        if is_best:
            best_f1, best_epoch = metrics["f1"], epoch
            if best_ckpt_path is not None:
                torch.save(model.state_dict(), best_ckpt_path)
    if best_ckpt_path is not None:
        print(f"[{stage}] best epoch = {best_epoch} (dev_F1={best_f1 * 100:.2f}), "
              f"saved to {best_ckpt_path}", flush=True)
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
        if args.use_pu_loss:
            model.loss_fnt = PUATLoss(na_weight=args.na_weight)
            print(f"[stage 1] loss = PUATLoss(na_weight={args.na_weight})", flush=True)
        print(f"[stage 1] distant pretrain on {len(distant_docs)} docs "
              f"({args.distant_epochs} epoch(s))", flush=True)
        distant_loader = DataLoader(build_gat_features(distant_docs, tokenizer, rel2id),
                                    batch_size=args.distant_batch_size, shuffle=True,
                                    collate_fn=collate)
        RESULTS_DIR.mkdir(exist_ok=True)
        stage1_best = RESULTS_DIR / f"{args.run_name}_stage1_best.pt" if args.save_model else None
        run_stage(model, distant_loader, args, device, args.distant_epochs,
                  "distant-pretrain", dev_loader, dev_docs, train_docs, id2rel,
                  best_ckpt_path=stage1_best)
        del distant_loader, distant_docs
        if args.save_model:
            p = RESULTS_DIR / f"{args.run_name}_stage1.pt"
            torch.save(model.state_dict(), p)
            print(f"[saved] {p}  (final distant epoch, not necessarily best -- "
                  f"see {stage1_best} for that)", flush=True)
        # annotated's Na labels are gold, not distant-supervision noise -- always
        # plain ATLoss for stage 2, matching Scripts/atlop/train_re.py.
        model.loss_fnt = ATLoss()

    print(f"[stage 2] annotated fine-tune on {len(train_docs)} docs ({args.epochs} epoch(s))",
          flush=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    best_ckpt = RESULTS_DIR / f"{args.run_name}_best.pt" if args.save_model else None
    metrics, preds = run_stage(model, train_loader, args, device, args.epochs,
                               "annotated-finetune", dev_loader, dev_docs, train_docs, id2rel,
                               best_ckpt_path=best_ckpt)

    RESULTS_DIR.mkdir(exist_ok=True)
    pred_path = RESULTS_DIR / f"{args.run_name}_dev_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as fp:
        json.dump(preds, fp, ensure_ascii=False)
    print(f"[saved] {pred_path}  ({len(preds)} predicted relations, final epoch)", flush=True)
    if args.save_model:
        ckpt = RESULTS_DIR / f"{args.run_name}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"[saved] {ckpt}  (final epoch -- see {best_ckpt} for the best-dev-F1 epoch)",
              flush=True)
        if best_ckpt is not None and best_ckpt.exists():
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            best_preds = predict(model, dev_loader, id2rel, device)
            best_pred_path = RESULTS_DIR / f"{args.run_name}_best_dev_predictions.json"
            with open(best_pred_path, "w", encoding="utf-8") as fp:
                json.dump(best_preds, fp, ensure_ascii=False)
            best_metrics = evaluate(best_preds, dev_docs, train_docs)
            print(f"[best checkpoint] dev_F1={best_metrics['f1'] * 100:.2f} "
                  f"Ign_F1={best_metrics['ign_f1'] * 100:.2f} -- "
                  f"saved {best_pred_path} ({len(best_preds)} predicted relations)", flush=True)
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
    p.add_argument("--use_pu_loss", action="store_true",
                   help="PUATLoss for the distant stage instead of plain ATLoss "
                        "(na_weight=0.7 default was swept and validated on Scripts/atlop -- "
                        "recommended on; matches Scripts/atlop/train_re.py's flag)")
    p.add_argument("--na_weight", type=float, default=0.7,
                   help="PUATLoss down-weight for distant all-Na pairs' TH-ranking term")
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--limit_docs", type=int, default=0, help="cap all splits; for quick runs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
