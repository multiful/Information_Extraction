"""Train / evaluate the ATLOP baseline on DocRED (PRD track 2).

Two-stage flow; the stage order is selected with --distant_mode:
  pretrain (default — the original paper's order)
           stage 1  pretrain on noisy train_distant
           stage 2  fine-tune on clean train_annotated (hyperparams mirror
                    wzhouad/ATLOP's canonical DocRED config: bert-base-cased,
                    ATLoss, differential LRs)
  denoise  (team's earlier recipe, kept for comparison)
           stage 1  train on train_annotated — this model doubles as teacher
           stage 2  continue on train_distant AFTER dropping the distant
                    positive labels the teacher disagrees with
  none     train on train_annotated only
  eval     dev F1 / Ign F1 after every epoch via the shared Scripts.eval.scorer,
           predictions saved in the team's common format — directly comparable
           to the track-1 models.

--evi_lambda > 0 additionally enables the DREEAM evidence-guided attention loss
(Ma et al., ACL 2023): the localized-context attention is supervised against
gold evidence sentences (annotated/dev only; train_distant has no evidence, so
that stage is unaffected). Default 0.0 reproduces the pre-DREEAM baseline
exactly, since sent_pos/evidence are only forwarded to the model when enabled.

Run from the project root, e.g.

    # full run, paper order (GPU / Colab recommended — this repo's torch is CPU-only)
    python -m Scripts.atlop.train_re --epochs 30 --distant_limit 20000 --save_model

    # same, with DREEAM evidence-guided attention enabled
    python -m Scripts.atlop.train_re --epochs 30 --distant_limit 20000 --evi_lambda 0.1 --save_model

    # quick CPU sanity run on a handful of docs
    python -m Scripts.atlop.train_re --limit_docs 8 --epochs 1 --distant_epochs 1
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

from data.docred_dataset import DocREDataset          # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES   # noqa: E402
from Scripts.atlop.losses import ATLoss, PUATLoss       # noqa: E402
from Scripts.atlop.preprocess import build_features     # noqa: E402
from Scripts.atlop.re_model import DocREModel           # noqa: E402
from Scripts.eval.scorer import evaluate                # noqa: E402

RESULTS_DIR = ROOT / "results"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_collate_fn(pad_token_id: int):
    """Pad input_ids to the batch max; keep entity_pos/hts ragged; expand the
    sparse positive-id label lists into one dense (total_pairs, NUM_CLASSES)
    multi-hot float tensor in hts order."""
    def collate(features: list[dict]) -> dict:
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
            "sent_pos": [f["sent_pos"] for f in features],
            "evidence": [f["evidence"] for f in features],
            "labels": labels,
            "features": features,
        }
    return collate


@torch.no_grad()
def predict(model, loader, id2rel, device) -> list[dict]:
    """Run the model over a loader and emit predictions in the common format:
    [{"title","h_idx","t_idx","r"}], one row per predicted relation (r = P-code,
    class 0 / Na skipped)."""
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
                for r in range(1, NUM_CLASSES):  # skip class 0 (Na/TH)
                    if row[r] == 1:
                        out.append({"title": f["title"], "h_idx": h, "t_idx": t, "r": id2rel[r]})
    return out


@torch.no_grad()
def denoise_features(model, loader, device) -> tuple[int, int]:
    """Filter distant-supervision labels with the current (annotated-trained)
    model acting as teacher. A distant positive label r on a pair is KEPT only
    if the teacher also ranks r above its adaptive threshold; every other
    positive is treated as distant noise and dropped (the pair falls back to
    Na). Mutates the underlying feature dicts in place (collate passes them
    through by reference). Returns (kept, dropped) positive-label counts."""
    model.eval()
    kept = dropped = 0
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
            rows = preds[idx: idx + n]
            idx += n
            new_labels = []
            for ids, row in zip(f["labels"], rows):
                pos = [r for r in ids if r != 0]
                keep = [r for r in pos if row[r] == 1]
                kept += len(keep)
                dropped += len(pos) - len(keep)
                new_labels.append(keep if keep else [0])
            f["labels"] = new_labels
    return kept, dropped


def build_optim_sched(model, args, total_steps):
    """Fresh optimizer + linear-warmup scheduler. Built per stage so the distant
    pretrain and the annotated fine-tune each get their own schedule.
    Differential LR: pretrained encoder vs. freshly-initialized head."""
    new_layers = ("extractor", "bilinear")
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


def run_stage(model, loader, args, device, epochs, stage, dev_loader, dev_docs,
              ign_docs, id2rel):
    """Train `model` on `loader` for `epochs`, evaluating on dev after each epoch.
    Returns (last_metrics, last_predictions). `ign_docs` = train_annotated, used
    only for the Ign-F1 fact filter, regardless of which split we train on."""
    total_steps = max(1, len(loader) * epochs)
    optimizer, scheduler = build_optim_sched(model, args, total_steps)
    metrics, preds = None, None
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for step, batch in enumerate(loader):
            # sent_pos/evidence are only forwarded when the DREEAM evidence-guided
            # attention loss is enabled, so --evi_lambda 0 (the default) reproduces
            # the exact old code path/behavior.
            evi_kwargs = {}
            if args.evi_lambda > 0:
                evi_kwargs = {
                    "sent_pos": batch["sent_pos"],
                    "evidence": batch["evidence"],
                    "evi_lambda": args.evi_lambda,
                }
            loss = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                entity_pos=batch["entity_pos"],
                hts=batch["hts"],
                labels=batch["labels"],
                **evi_kwargs,
            )[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            if (step + 1) % args.log_every == 0:
                print(f"  [{stage}] epoch {epoch} step {step + 1}/{len(loader)} "
                      f"loss {running / (step + 1):.4f}")

        preds = predict(model, dev_loader, id2rel, device)
        metrics = evaluate(preds, dev_docs, ign_docs)
        print(f"[{stage} | epoch {epoch}] train_loss={running / max(1, len(loader)):.4f} "
              f"dev_F1={metrics['f1'] * 100:.2f} Ign_F1={metrics['ign_f1'] * 100:.2f} "
              f"(P={metrics['precision'] * 100:.2f} R={metrics['recall'] * 100:.2f})")
    return metrics, preds


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device}")

    rel2id = build_rel2id()
    id2rel = {v: k for k, v in rel2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    collate = make_collate_fn(tokenizer.pad_token_id)

    # train_annotated (fine-tune stage) + dev (evaluation).
    train_docs = list(DocREDataset(args.train_split))
    dev_docs = list(DocREDataset(args.dev_split))
    if args.limit_docs > 0:
        train_docs = train_docs[: args.limit_docs]
        dev_docs = dev_docs[: args.limit_docs]
    print(f"[data] train={len(train_docs)} dev={len(dev_docs)} docs")

    dev_features = build_features(dev_docs, tokenizer, rel2id)
    dev_loader = DataLoader(dev_features, batch_size=args.eval_batch_size,
                            shuffle=False, collate_fn=collate)
    train_features = build_features(train_docs, tokenizer, rel2id)
    train_loader = DataLoader(train_features, batch_size=args.train_batch_size,
                              shuffle=True, collate_fn=collate)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=NUM_CLASSES)
    # eager attention is required so the encoder returns attention weights
    # (localized context pooling needs them); sdpa/flash do not.
    encoder = AutoModel.from_pretrained(
        args.model_name_or_path, config=config, attn_implementation="eager"
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = DocREModel(config, encoder, emb_size=args.emb_size,
                       block_size=args.block_size, num_labels=NUM_CLASSES).to(device)

    def load_distant():
        cap = args.limit_docs if args.limit_docs > 0 else args.distant_limit
        docs = list(DocREDataset(args.distant_split))
        if cap > 0:
            docs = docs[:cap]
        return docs, build_features(docs, tokenizer, rel2id)

    if args.distant_mode == "pretrain":
        # Paper order: pretrain on the big noisy distant split first, then
        # fine-tune on the clean human-annotated split.
        distant_docs, distant_features = load_distant()
        print(f"[stage 1] distant pretrain on {len(distant_docs)} docs "
              f"({args.distant_epochs} epoch(s))")
        if args.use_pu_loss:
            # PU loss only makes sense on distant data (its Na labels are
            # unlabeled, not gold negatives) -- swapped back before stage 2.
            model.loss_fnt = PUATLoss(na_weight=args.na_weight)
            print(f"[stage 1] loss = PUATLoss(na_weight={args.na_weight})")
        distant_loader = DataLoader(distant_features, batch_size=args.distant_batch_size,
                                    shuffle=True, collate_fn=collate)
        metrics, preds = run_stage(model, distant_loader, args, device, args.distant_epochs,
                                   "distant-pretrain", dev_loader, dev_docs, train_docs, id2rel)
        del distant_features, distant_loader, distant_docs
        if args.save_model:
            RESULTS_DIR.mkdir(exist_ok=True)
            stage1_ckpt = RESULTS_DIR / f"{args.run_name}_stage1.pt"
            torch.save(model.state_dict(), stage1_ckpt)
            print(f"[saved] {stage1_ckpt}  (distant-pretrain checkpoint)")
        model.loss_fnt = ATLoss()

        if args.epochs > 0:
            print(f"[stage 2] annotated fine-tune on {len(train_docs)} docs ({args.epochs} epoch(s))")
            metrics, preds = run_stage(model, train_loader, args, device, args.epochs,
                                       "annotated-finetune", dev_loader, dev_docs, train_docs, id2rel)
        else:
            print("[stage 2] skipped (--epochs 0) -- reporting stage-1 metrics/predictions")
    else:
        # Team recipe: supervised training on the clean annotated split first.
        # In denoise mode this model doubles as the teacher for stage 2.
        print(f"[stage 1] annotated train on {len(train_docs)} docs ({args.epochs} epoch(s))")
        metrics, preds = run_stage(model, train_loader, args, device, args.epochs,
                                   "annotated-train", dev_loader, dev_docs, train_docs, id2rel)

        # Optional stage 2: continue on train_distant AFTER stripping the
        # labels the stage-1 teacher disagrees with (self-training-style denoising).
        if args.distant_mode == "denoise":
            distant_docs, distant_features = load_distant()
            print(f"[stage 2] distant denoise+train on {len(distant_docs)} docs "
                  f"({args.distant_epochs} epoch(s))")
            denoise_loader = DataLoader(distant_features, batch_size=args.eval_batch_size,
                                        shuffle=False, collate_fn=collate)
            kept, dropped = denoise_features(model, denoise_loader, device)
            total = kept + dropped
            print(f"[stage 2] denoise: kept {kept}/{total} distant positive labels "
                  f"({dropped} dropped as noise)")
            distant_loader = DataLoader(distant_features, batch_size=args.distant_batch_size,
                                        shuffle=True, collate_fn=collate)
            metrics, preds = run_stage(model, distant_loader, args, device, args.distant_epochs,
                                       "distant-denoised", dev_loader, dev_docs, train_docs, id2rel)
            del distant_features, distant_loader, denoise_loader, distant_docs

    RESULTS_DIR.mkdir(exist_ok=True)
    pred_path = RESULTS_DIR / f"{args.run_name}_dev_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as fp:
        json.dump(preds, fp, ensure_ascii=False)
    print(f"[saved] {pred_path}  ({len(preds)} predicted relations)")

    if args.save_model:
        ckpt = RESULTS_DIR / f"{args.run_name}.pt"
        torch.save(model.state_dict(), ckpt)
        print(f"[saved] {ckpt}")

    return metrics


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ATLOP baseline on DocRED")
    p.add_argument("--model_name_or_path", default="bert-base-cased")
    p.add_argument("--train_split", default="train_annotated", help="stage-1 train split")
    p.add_argument("--dev_split", default="dev")
    p.add_argument("--run_name", default="atlop")
    p.add_argument("--epochs", type=int, default=30, help="annotated train/fine-tune epochs")
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--distant_mode", choices=["pretrain", "denoise", "none"], default="pretrain",
                   help="'pretrain' = paper order (distant pretrain -> annotated fine-tune), "
                        "'denoise' = team recipe (annotated train -> teacher-denoised distant), "
                        "'none' = annotated only")
    p.add_argument("--distant_split", default="train_distant")
    p.add_argument("--distant_epochs", type=int, default=1)
    p.add_argument("--distant_batch_size", type=int, default=4)
    p.add_argument("--distant_limit", type=int, default=0,
                   help="cap distant docs (0 = all 101,873; use e.g. 20000 to bound RAM/time on Colab)")
    p.add_argument("--use_pu_loss", action="store_true",
                   help="use losses.PUATLoss for the distant pretrain stage (pretrain mode only; "
                        "annotated fine-tune always uses plain ATLoss since its Na labels are gold)")
    p.add_argument("--na_weight", type=float, default=0.5,
                   help="PUATLoss down-weight for the TH-ranking term on distant all-Na pairs")
    p.add_argument("--encoder_lr", type=float, default=5e-5)
    p.add_argument("--classifier_lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--emb_size", type=int, default=768)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--evi_lambda", type=float, default=0.0,
                   help="weight of the DREEAM evidence-guided attention loss "
                        "(0 = off, reproduces the pre-DREEAM baseline exactly; "
                        "~0.1 enables it). No effect on splits without gold "
                        "evidence (e.g. train_distant), which stay Na/unsupervised.")
    p.add_argument("--limit_docs", type=int, default=0, help="cap train/dev docs (0 = all); for quick runs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)
