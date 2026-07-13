"""Sample docs from train_distant and filter out likely-noisy relation labels
using the model already trained on train_annotated (self-training style
agreement filter): a distant (h,t,r) label is kept only if the model's own
prediction for that pair also includes r; otherwise it's dropped as noise.

Writes a new JSON file (same per-doc schema as DocRED: title/sents/vertexSet/
labels) with filtered `labels`, for dk_train_distant_continue.py to train on.

New file. Does not modify any shared module under data/ or dk_train.py
(reuses its helper functions via import).

Usage (run only after dk_train.py has produced a checkpoint):
    python Scripts/models/dk_filter_distant.py --sample_size 5000
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

from data.docred_dataset import DocREDataset
from data.tokenization import load_tokenizer

from dk_model import ATLoss, RobertaATLOPLiteModel
from dk_pairs import REL2ID
from dk_train import prepare_example, to_device_batch

CKPT_DIR = Path(__file__).resolve().parent / "dk_checkpoints"


@torch.no_grad()
def filter_doc(model, doc, tokenizer, max_length, device, num_labels):
    """Returns a new doc dict with only model-corroborated distant labels kept.
    Also returns (num_kept, num_total) for stats."""
    ex = prepare_example(doc, tokenizer, max_length)
    if not ex["hts"]:
        return {**doc, "labels": []}, 0, 0

    input_ids, attention_mask = to_device_batch(ex, device)
    logits = model(input_ids, attention_mask, [ex["entity_pos"]], [ex["hts"]])[0]
    pred_mask = ATLoss.get_label(logits, num_labels=num_labels)  # (num_pairs, num_class) bool

    # (h, t) -> set of relation ids the model itself predicts for that pair
    pred_by_pair = {}
    for (h, t), row in zip(ex["hts"], pred_mask):
        rids = set(row.nonzero(as_tuple=True)[0].tolist())
        rids.discard(0)  # TH/Na is not a relation
        pred_by_pair[(h, t)] = rids

    kept_labels = []
    num_total = len(doc.get("labels", []))
    for lab in doc.get("labels", []):
        rid = REL2ID[lab["r"]]
        if rid in pred_by_pair.get((lab["h"], lab["t"]), set()):
            kept_labels.append(lab)

    return {**doc, "labels": kept_labels}, len(kept_labels), num_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(CKPT_DIR / "best.pt"))
    parser.add_argument("--sample_size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_labels", type=int, default=4, help="ATLoss.get_label top-k cap")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu", choices=["mps", "cpu"],
                         help="default cpu -- see dk_train.py's MPS OOM note")
    parser.add_argument("--output", type=str,
                         default=str(Path(__file__).resolve().parent / "dk_filtered_distant.json"))
    args = parser.parse_args()

    device = torch.device(args.device)
    print("device:", device)

    tokenizer = load_tokenizer("roberta-base")
    model = RobertaATLOPLiteModel().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()
    print("loaded checkpoint:", args.checkpoint)

    distant = DocREDataset("train_distant")
    random.seed(args.seed)
    sample_idx = random.sample(range(len(distant)), min(args.sample_size, len(distant)))
    print(f"sampled {len(sample_idx)} / {len(distant)} train_distant docs (seed={args.seed})")

    filtered_docs = []
    total_kept, total_orig = 0, 0
    for n, i in enumerate(sample_idx):
        doc = distant[i]
        new_doc, kept, orig = filter_doc(model, doc, tokenizer, args.max_length, device, args.num_labels)
        filtered_docs.append(new_doc)
        total_kept += kept
        total_orig += orig
        if (n + 1) % 200 == 0:
            print(f"  filtered {n + 1}/{len(sample_idx)} docs "
                  f"(running: kept {total_kept}/{total_orig} = {total_kept / max(1, total_orig):.1%})")

    print(f"done. kept {total_kept}/{total_orig} distant labels "
          f"({total_kept / max(1, total_orig):.1%}) -- rest treated as noise and dropped.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(filtered_docs, f, ensure_ascii=False)
    print("wrote", args.output)


if __name__ == "__main__":
    main()
