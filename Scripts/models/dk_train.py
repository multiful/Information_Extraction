"""Training script for dk's RoBERTa + Localized Context Pooling + Adaptive
Thresholding model. Can train on either train_annotated or train_distant
(--train_split), and can resume/continue from a checkpoint (--init_checkpoint)
-- so the standard "distant pretrain -> annotated finetune" pipeline is just
two invocations:
    python Scripts/models/dk_train.py --train_split train_distant \
        --save_name distant_pretrain.pt ...
    python Scripts/models/dk_train.py --train_split train_annotated \
        --init_checkpoint Scripts/models/dk_checkpoints/distant_pretrain.pt \
        --save_name final.pt ...

Checkpoints go to dk_checkpoints/<save_name>, dev predictions to
predictions/<save_name minus .pt>.json. Redirect stdout yourself to whatever
logs/<name>.log you want (this script doesn't write its own log file).

New file. Does not modify any shared module under data/.

Usage:
    python Scripts/models/dk_train.py --max_train_docs 20 --max_dev_docs 20 \
        --epochs 1 --eval_every 20        # smoke test
    python Scripts/models/dk_train.py                                        # full run, train_annotated

Note on evaluation: the F1 below is a simplified exact-triple-match metric
for quick iteration, NOT the official Ign-F1 scorer described in PRD.md
section 4. Once the team's shared scorer exists, re-score this script's
dev predictions with that instead for the actual ATLOP comparison.

Note on --device mps: previously OOM'd within ~400-450 steps from two
independent causes, both now fixed: (1) the per-pair Python loop fragmenting
MPS's caching allocator -- fixed by vectorizing dk_model.py's forward pass;
(2) HF's eager-attention fallback -- fixed in dk_model.py's encode() by only
recomputing the last layer's attention manually instead of requesting
output_attentions=True for the whole stack. A third, separate cause was
diagnosed after that: torch.mps.driver_allocated_memory() grew linearly
(~6.7GB -> 32GB over 400 steps) while torch.mps.current_allocated_memory()
(PyTorch's own tracked tensors) stayed flat at ~2GB -- i.e. a Metal-driver-
level leak, not a Python/tensor-reference leak. Root cause: every doc has a
different token length (data/tokenization.py pads nothing, only truncates),
so MPSGraph was compiling and caching a new graph per distinct input shape,
and that cache was never evicted. Fixed by padding every doc to a fixed
length on MPS only (to_device_batch's pad_to arg) -- verified via a 450-step
standalone run that driver_allocated_memory then stays flat (~3.75GB) for
the whole run with no OOM. CPU is left unpadded (unaffected by this, and
padding would waste FLOPs on short docs there for no benefit).
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from data.docred_dataset import DocREDataset
from data.tokenization import load_tokenizer, tokenize_document

from dk_model import ATLoss, PUATLoss, RobertaATLOPLiteModel
from dk_pairs import ID2REL, build_pairs_and_labels, compute_dist_buckets

CKPT_DIR = Path(__file__).resolve().parent / "dk_checkpoints"
PRED_DIR = Path(__file__).resolve().parent / "predictions"
CKPT_DIR.mkdir(exist_ok=True)
PRED_DIR.mkdir(exist_ok=True)


def prepare_example(doc: dict, tokenizer, max_length: int) -> dict:
    tok = tokenize_document(doc, tokenizer, max_length)
    num_entities = len(tok["entity_pos"])
    hts, labels = build_pairs_and_labels(doc, num_entities)
    dist_buckets = compute_dist_buckets(doc, hts)
    return {
        "title": doc["title"],
        "input_ids": tok["input_ids"],
        "attention_mask": tok["attention_mask"],
        "entity_pos": tok["entity_pos"],
        "hts": hts,
        "labels": labels,
        "dist_buckets": dist_buckets,
    }


def to_device_batch(ex: dict, device: torch.device, pad_token_id: int = None,
                     pad_to: int = None) -> tuple[torch.Tensor, torch.Tensor]:
    """pad_token_id/pad_to: only used on MPS (see module docstring) -- pads every doc's
    input_ids/attention_mask to a fixed length so the encoder always sees the same shape.
    Padding is masked out by attention_mask and doesn't change results; it's purely to
    stop MPSGraph from compiling+caching a new graph per distinct sequence length."""
    ids, mask = ex["input_ids"], ex["attention_mask"]
    if device.type == "mps" and pad_to is not None and len(ids) < pad_to:
        pad_n = pad_to - len(ids)
        ids = ids + [pad_token_id] * pad_n
        mask = mask + [0] * pad_n
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask], dtype=torch.long, device=device)
    return input_ids, attention_mask


@torch.no_grad()
def evaluate(model: RobertaATLOPLiteModel, dev_examples: list[dict], device: torch.device,
             num_labels: int = 4, pad_token_id: int = None, pad_to: int = None) -> list[dict]:
    model.eval()
    preds = []
    for i, ex in enumerate(dev_examples):
        if not ex["hts"]:
            continue
        input_ids, attention_mask = to_device_batch(ex, device, pad_token_id, pad_to)
        logits = model(input_ids, attention_mask, [ex["entity_pos"]], [ex["hts"]], [ex["dist_buckets"]])[0]
        mask = ATLoss.get_label(logits, num_labels=num_labels)
        for (h, t), row in zip(ex["hts"], mask):
            for rid in row.nonzero(as_tuple=True)[0].tolist():
                preds.append({"title": ex["title"], "h_idx": h, "t_idx": t, "r": ID2REL[rid]})
        if device.type == "mps" and i % 50 == 0:
            torch.mps.empty_cache()
    model.train()
    return preds


def dev_gold_triples(dev_docs: list[dict]) -> set[tuple]:
    gold = set()
    for doc in dev_docs:
        for lab in doc.get("labels", []):
            gold.add((doc["title"], lab["h"], lab["t"], lab["r"]))
    return gold


def f1_score(preds: list[dict], gold_triples: set[tuple]) -> tuple[float, float, float]:
    pred_set = {(p["title"], p["h_idx"], p["t_idx"], p["r"]) for p in preds}
    tp = len(pred_set & gold_triples)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_triples) if gold_triples else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_split", type=str, default="train_annotated",
                         choices=["train_annotated", "train_distant"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_docs", type=int, default=-1,
                         help="-1 = full split; use to sample train_distant for a quicker run")
    parser.add_argument("--max_dev_docs", type=int, default=-1, help="-1 = full dev")
    parser.add_argument("--eval_every", type=int, default=200, help="training steps between dev evals")
    parser.add_argument("--device", type=str, default=None, choices=["mps", "cpu"],
                         help="default: mps if available. Used to OOM within ~400-450 steps from "
                              "three separate causes (allocator fragmentation, eager-attention "
                              "fallback, and a driver-level graph cache leak from variable doc "
                              "lengths) -- all fixed now, see module docstring. cpu remains "
                              "available and unaffected if mps still misbehaves.")
    parser.add_argument("--init_checkpoint", type=str, default=None,
                         help="resume/continue training from this checkpoint instead of a fresh model")
    parser.add_argument("--save_name", type=str, default="best.pt",
                         help="checkpoint filename under dk_checkpoints/ (e.g. best_distant.pt for a "
                              "stage-1 pretrain run, so it doesn't clobber the stage-2 result)")
    parser.add_argument("--patience", type=int, default=2,
                         help="stop early after this many dev evals with no F1 improvement (<=0 disables)")
    parser.add_argument("--use_dist_embedding", action="store_true",
                         help="add the bucketed sentence-distance embedding (PRD Distance/Sentence "
                              "Position Embedding ideas). Off by default for A/B comparison.")
    parser.add_argument("--use_pu_loss", action="store_true",
                         help="use PUATLoss (approximate PU-learning loss, see dk_model.py) instead of "
                              "plain ATLoss -- only makes sense with --train_split train_distant, since "
                              "it down-weights trust in distant-labeled Na pairs specifically")
    parser.add_argument("--na_weight", type=float, default=0.5,
                         help="PUATLoss: down-weight factor for distant-Na pairs' TH-ranking loss term")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=123,
                         help="seed for --max_train_docs random sampling (kept separate from "
                              "--seed so the same subset can be reused across A/B comparisons "
                              "that vary --seed for model init/training)")
    args = parser.parse_args()

    if args.use_pu_loss and args.train_split != "train_distant":
        print(f"warning: --use_pu_loss with --train_split {args.train_split} -- PU loss is only "
              f"meaningful for train_distant (train_annotated's Na labels are gold, not unlabeled)")

    torch.manual_seed(args.seed)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device, "| train_split:", args.train_split, "| use_dist_embedding:", args.use_dist_embedding)

    tokenizer = load_tokenizer("roberta-base")
    pad_token_id = tokenizer.pad_token_id
    pad_to = args.max_length if device.type == "mps" else None
    train_raw = DocREDataset(args.train_split)
    dev_raw = DocREDataset("dev")

    n_train = len(train_raw) if args.max_train_docs < 0 else min(args.max_train_docs, len(train_raw))
    n_dev = len(dev_raw) if args.max_dev_docs < 0 else min(args.max_dev_docs, len(dev_raw))

    if args.max_train_docs < 0:
        train_indices = range(n_train)
    else:
        # random (seeded, reproducible) sample instead of just the first n_train docs --
        # matters for train_distant, which isn't necessarily randomly ordered
        train_indices = random.Random(args.sample_seed).sample(range(len(train_raw)), n_train)

    print(f"preparing {n_train} {args.train_split} docs, {n_dev} dev docs ...")
    train_examples = [prepare_example(train_raw[i], tokenizer, args.max_length) for i in train_indices]
    dev_examples = [prepare_example(dev_raw[i], tokenizer, args.max_length) for i in range(n_dev)]
    gold_triples = dev_gold_triples([dev_raw[i] for i in range(n_dev)])

    model = RobertaATLOPLiteModel(use_dist_embedding=args.use_dist_embedding).to(device)
    if args.init_checkpoint:
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print("resumed from", args.init_checkpoint)
    loss_fn = PUATLoss(na_weight=args.na_weight) if args.use_pu_loss else ATLoss()
    print("loss:", type(loss_fn).__name__)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, args.epochs * len(train_examples))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    save_path = CKPT_DIR / args.save_name

    best_f1 = -1.0
    if args.init_checkpoint:
        preds = evaluate(model, dev_examples, device, pad_token_id=pad_token_id, pad_to=pad_to)
        _, _, best_f1 = f1_score(preds, gold_triples)
        print(f"[dev @ resume] F1={best_f1:.4f}  <- new epochs need to beat this to save a new checkpoint")
    step = 0
    evals_without_improvement = 0
    stop_early = False
    for epoch in range(args.epochs):
        if stop_early:
            break
        for ex in train_examples:
            if not ex["hts"]:
                continue
            input_ids, attention_mask = to_device_batch(ex, device, pad_token_id, pad_to)
            labels = torch.tensor(ex["labels"], dtype=torch.float, device=device)

            logits = model(input_ids, attention_mask, [ex["entity_pos"]], [ex["hts"]], [ex["dist_buckets"]])[0]
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            # MPS's caching allocator doesn't reliably reclaim memory from many small
            # variably-shaped allocations; the pair loop that used to cause this is now
            # vectorized, but this stays as a cheap safety net.
            if device.type == "mps" and step % 20 == 0:
                torch.mps.empty_cache()

            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            if step % args.eval_every == 0:
                preds = evaluate(model, dev_examples, device, pad_token_id=pad_token_id, pad_to=pad_to)
                p, r, f1 = f1_score(preds, gold_triples)
                print(f"  [dev @ step {step}] P={p:.4f} R={r:.4f} F1={f1:.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    evals_without_improvement = 0
                    torch.save(model.state_dict(), save_path)
                else:
                    evals_without_improvement += 1
                    if 0 < args.patience <= evals_without_improvement:
                        print(f"  early stopping: {evals_without_improvement} evals with no "
                              f"improvement over best F1={best_f1:.4f}")
                        stop_early = True
                        break

    preds = evaluate(model, dev_examples, device, pad_token_id=pad_token_id, pad_to=pad_to)
    p, r, f1 = f1_score(preds, gold_triples)
    print(f"[final dev] P={p:.4f} R={r:.4f} F1={f1:.4f} (best during training: {best_f1:.4f})")
    if f1 > best_f1:
        torch.save(model.state_dict(), save_path)
    print("checkpoint:", save_path)

    out_path = PRED_DIR / f"{args.save_name.replace('.pt', '')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
