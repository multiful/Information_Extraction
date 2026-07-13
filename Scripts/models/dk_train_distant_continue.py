"""Stage 2: continue training the dk model on the noise-filtered train_distant
sample (produced by dk_filter_distant.py), initialized from the stage-1
(train_annotated) checkpoint.

New file. Does not modify any shared module under data/ or dk_train.py
(reuses its helper functions via import).

Usage (run only after dk_filter_distant.py has written dk_filtered_distant.json):
    python Scripts/models/dk_train_distant_continue.py --epochs 2
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from data.docred_dataset import DocREDataset
from data.tokenization import load_tokenizer

from dk_model import ATLoss, RobertaATLOPLiteModel
from dk_train import (
    CKPT_DIR,
    dev_gold_triples,
    evaluate,
    f1_score,
    prepare_example,
    to_device_batch,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_checkpoint", type=str, default=str(CKPT_DIR / "best.pt"),
                         help="stage-1 (train_annotated) checkpoint to continue from")
    parser.add_argument("--filtered_data", type=str,
                         default=str(Path(__file__).resolve().parent / "dk_filtered_distant.json"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5, help="lower than stage-1 -- fine-tuning, not from scratch")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_dev_docs", type=int, default=-1)
    parser.add_argument("--eval_every", type=int, default=0, help="0 = once per epoch")
    parser.add_argument("--device", type=str, default="cpu", choices=["mps", "cpu"])
    args = parser.parse_args()

    device = torch.device(args.device)
    print("device:", device)

    tokenizer = load_tokenizer("roberta-base")

    with open(args.filtered_data, encoding="utf-8") as f:
        distant_docs = json.load(f)
    print(f"loaded {len(distant_docs)} filtered train_distant docs from {args.filtered_data}")

    dev_raw = DocREDataset("dev")
    n_dev = len(dev_raw) if args.max_dev_docs < 0 else min(args.max_dev_docs, len(dev_raw))
    dev_examples = [prepare_example(dev_raw[i], tokenizer, args.max_length) for i in range(n_dev)]
    gold_triples = dev_gold_triples([dev_raw[i] for i in range(n_dev)])

    print("preparing filtered train_distant examples ...")
    train_examples = [prepare_example(doc, tokenizer, args.max_length) for doc in distant_docs]

    model = RobertaATLOPLiteModel().to(device)
    model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
    print("initialized from", args.init_checkpoint)

    # baseline: what did the stage-1 model already score on dev, before this continued training?
    preds = evaluate(model, dev_examples, device)
    _, _, stage1_f1 = f1_score(preds, gold_triples)
    print(f"[dev before stage-2] F1={stage1_f1:.4f}  <- compare against this")

    loss_fn = ATLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = max(1, args.epochs * len(train_examples))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    eval_every = args.eval_every if args.eval_every > 0 else len(train_examples)

    best_f1 = stage1_f1  # stage-1's dev F1 is the bar stage-2 needs to beat
    step = 0
    for epoch in range(args.epochs):
        for ex in train_examples:
            if not ex["hts"]:
                continue
            input_ids, attention_mask = to_device_batch(ex, device)
            labels = torch.tensor(ex["labels"], dtype=torch.float, device=device)

            logits = model(input_ids, attention_mask, [ex["entity_pos"]], [ex["hts"]])[0]
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if device.type == "mps" and step % 20 == 0:
                torch.mps.empty_cache()
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

            if step % eval_every == 0:
                preds = evaluate(model, dev_examples, device)
                p, r, f1 = f1_score(preds, gold_triples)
                print(f"  [dev @ step {step}] P={p:.4f} R={r:.4f} F1={f1:.4f}")
                if f1 > best_f1:
                    best_f1 = f1
                    torch.save(model.state_dict(), CKPT_DIR / "best_distant.pt")
                    print("  -> new best, saved to dk_checkpoints/best_distant.pt")

    print(f"[stage-2 done] best dev F1 during stage-2: {best_f1:.4f} "
          f"(stage-1-only baseline was {stage1_f1:.4f})")
    if best_f1 <= stage1_f1:
        print("  note: stage-2 did not beat stage-1 -- best_distant.pt was never written; "
              "dk_checkpoints/best.pt (stage-1) remains the better checkpoint.")


if __name__ == "__main__":
    main()
