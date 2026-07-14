import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_dataset import DocREDataset          # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES   # noqa: E402
from Scripts.atlop.losses import ATLoss                 # noqa: E402
from Scripts.atlop.preprocess import build_features     # noqa: E402
from Scripts.eval.scorer import evaluate                # noqa: E402
from Scripts.grep.re_model import GREPModel              # noqa: E402

RESULTS_DIR = ROOT / "results"
GAMMA_GRID = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_collate_fn(pad_token_id: int):
    """Like Scripts.atlop.train_re.make_collate_fn, plus the GREP-only fields
    (sent_pos/evidence/doc_rel_labels) that Scripts/atlop/preprocess.py now
    emits alongside the ATLOP fields."""
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
            "doc_rel_labels": [f["doc_rel_labels"] for f in features],
            "labels": labels,
            "features": features,
        }
    return collate


@torch.no_grad()
def predict(model, loader, id2rel, device) -> list[dict]:
    """Common-format predictions ([{"title","h_idx","t_idx","r"}]), same
    contract as Scripts.atlop.train_re.predict."""
    model.eval()
    out = []
    for batch in loader:
        preds = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            entity_pos=batch["entity_pos"],
            hts=batch["hts"],
            sent_pos=batch["sent_pos"],
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
    return out


def build_optim_sched(model, args, total_steps):
    """Differential LR: pretrained encoder vs. every GREP-specific module
    (graph, extractors, bilinear, doc_classifier) at classifier_lr."""
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
         "lr": args.classifier_lr},
        {"params": [p for n, p in model.named_parameters() if n.startswith("encoder.")],
         "lr": args.encoder_lr},
    ]
    optimizer = torch.optim.AdamW(grouped, eps=1e-6)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


def run_stage(model, loader, args, device, epochs, stage, dev_loader, dev_docs,
              ign_docs, id2rel, use_evidence_loss):
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
                sent_pos=batch["sent_pos"],
                evidence=batch["evidence"],
                doc_rel_labels=batch["doc_rel_labels"],
                labels=batch["labels"],
                use_evidence_loss=use_evidence_loss,
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



def build_pseudo_doc(doc: dict, kept_sent_ids: set[int]) -> tuple[dict, dict[int, int]]:
    """Restrict a raw DocREDataset doc to `kept_sent_ids`, re-indexing
    sentences and dropping entities with no surviving mentions. Returns
    (pseudo_doc, entity_map) where entity_map maps original vertexSet index
    -> pseudo-doc vertexSet index (entities with no surviving mention are
    absent from the map)."""
    kept_sorted = sorted(kept_sent_ids)
    remap_sent = {old: new for new, old in enumerate(kept_sorted)}
    sents = [doc["sents"][sid] for sid in kept_sorted]

    vertex_set, entity_map = [], {}
    for old_idx, entity in enumerate(doc["vertexSet"]):
        mentions = [
            {**m, "sent_id": remap_sent[m["sent_id"]]}
            for m in entity if m["sent_id"] in kept_sent_ids
        ]
        if mentions:
            entity_map[old_idx] = len(vertex_set)
            vertex_set.append(mentions)

    pseudo_doc = {"title": f"{doc['title']}__pseudo", "sents": sents,
                  "vertexSet": vertex_set, "labels": []}
    return pseudo_doc, entity_map


@torch.no_grad()
def collect_full_pass(model_full, dev_features, collate, device):
    """One forward pass of `model_full` over dev, kept per-document (not
    flattened) so Inference Fusion can align pairs 1:1 with each doc."""
    model_full.eval()
    loader = DataLoader(dev_features, batch_size=1, shuffle=False, collate_fn=collate)
    entries = []
    for batch in loader:
        preds, all_u, logits = model_full(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            entity_pos=batch["entity_pos"],
            hts=batch["hts"],
            sent_pos=batch["sent_pos"],
        )
        entries.append({"feature": batch["features"][0], "logits": logits.cpu(), "u": all_u[0].cpu()})
    return entries


@torch.no_grad()
def align_pseudo_logits(entry, raw_doc, model_no_evi, tokenizer, rel2id, collate, device):
    """Builds the pseudo-document from `entry`'s predicted evidence sentences,
    re-infers with `model_no_evi`, and aligns its pair logits back onto
    `entry`'s original (head, tail) pair order.

    Returns (logits_pseudo: (n_pair, num_labels), fallback: (n_pair,) bool --
    True where no pseudo-doc counterpart exists, so Eq 22 fusion should be
    skipped for that pair (use logits_full alone)."""
    f = entry["feature"]
    hts = f["hts"]
    num_labels = model_no_evi.num_labels
    logits_pseudo = torch.zeros(len(hts), num_labels)
    fallback = torch.ones(len(hts), dtype=torch.bool)

    preds_full = ATLoss().get_label(entry["logits"], num_labels=num_labels)
    positive_idx = [i for i in range(len(hts)) if preds_full[i, 1:].any()]
    if not positive_idx:
        return logits_pseudo, fallback

    kept_sents: set[int] = set()
    for i in positive_idx:
        u_row = entry["u"][i]
        thr = u_row.mean()
        kept_sents.update((u_row >= thr).nonzero(as_tuple=True)[0].tolist())
    if not kept_sents:
        return logits_pseudo, fallback

    pseudo_doc, entity_map = build_pseudo_doc(raw_doc, kept_sents)
    if len(pseudo_doc["vertexSet"]) < 2:
        return logits_pseudo, fallback

    pf = build_features([pseudo_doc], tokenizer, rel2id, show_progress=False)[0]
    batch = collate([pf])
    _, _, p_logits = model_no_evi(
        input_ids=batch["input_ids"].to(device),
        attention_mask=batch["attention_mask"].to(device),
        entity_pos=batch["entity_pos"],
        hts=batch["hts"],
    )
    p_logits = p_logits.cpu()
    pair_index_pseudo = {ht: i for i, ht in enumerate(pf["hts"])}

    for i, (h, t) in enumerate(hts):
        if h in entity_map and t in entity_map:
            key = (entity_map[h], entity_map[t])
            if key in pair_index_pseudo:
                logits_pseudo[i] = p_logits[pair_index_pseudo[key]]
                fallback[i] = False
    return logits_pseudo, fallback


def inference_fusion(model_full, model_no_evi, dev_docs, dev_features, tokenizer,
                      rel2id, id2rel, device, gamma: float, sweep: bool,
                      ign_docs) -> tuple[list[dict], float, dict]:
    """Eq 22. Returns (predictions, gamma_used, dev_metrics)."""
    collate = make_collate_fn(tokenizer.pad_token_id)
    entries = collect_full_pass(model_full, dev_features, collate, device)

    cached = []
    for entry, raw_doc in zip(tqdm(entries, desc="inference-fusion pseudo-doc"), dev_docs):
        logits_pseudo, fallback = align_pseudo_logits(
            entry, raw_doc, model_no_evi, tokenizer, rel2id, collate, device)
        cached.append((entry, logits_pseudo, fallback))

    def decode(gamma_val: float) -> list[dict]:
        preds = []
        for entry, logits_pseudo, fallback in cached:
            fused = entry["logits"].clone()
            use = ~fallback
            fused[use] = entry["logits"][use] + logits_pseudo[use] - gamma_val
            row_preds = ATLoss().get_label(fused, num_labels=model_full.num_labels)
            f = entry["feature"]
            for (h, t), row in zip(f["hts"], row_preds):
                for r in range(1, model_full.num_labels):
                    if row[r] == 1:
                        preds.append({"title": f["title"], "h_idx": h, "t_idx": t, "r": id2rel[r]})
        return preds

    if not sweep:
        preds = decode(gamma)
        return preds, gamma, evaluate(preds, dev_docs, ign_docs)

    best_gamma, best_preds, best_metrics = gamma, None, {"f1": -1.0}
    for g in GAMMA_GRID:
        preds = decode(g)
        metrics = evaluate(preds, dev_docs, ign_docs)
        print(f"  [gamma sweep] gamma={g:+.1f} dev_F1={metrics['f1'] * 100:.2f}")
        if metrics["f1"] > best_metrics["f1"]:
            best_gamma, best_preds, best_metrics = g, preds, metrics
    return best_preds, best_gamma, best_metrics


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device}")

    rel2id = build_rel2id()
    id2rel = {v: k for k, v in rel2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    collate = make_collate_fn(tokenizer.pad_token_id)

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

    def build_model():
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=NUM_CLASSES)
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path, config=config, attn_implementation="eager"
        )
        config.cls_token_id = tokenizer.cls_token_id
        config.sep_token_id = tokenizer.sep_token_id
        return GREPModel(config, encoder, emb_size=args.emb_size, block_size=args.block_size,
                          num_labels=NUM_CLASSES, node_dim=args.node_dim,
                          graph_layers=args.graph_layers, graph_heads=args.graph_heads,
                          alpha=args.alpha, beta=args.beta).to(device)

    print(f"[stage 1] model_full (alpha={args.alpha}, beta={args.beta}) "
          f"on {len(train_docs)} docs ({args.epochs} epoch(s))")
    model_full = build_model()
    metrics_full, _ = run_stage(model_full, train_loader, args, device, args.epochs,
                                "grep-full", dev_loader, dev_docs, train_docs, id2rel,
                                use_evidence_loss=True)

    print(f"[stage 2] model_no_evi (alpha={args.alpha}, beta=0, evidence loss off) "
          f"on {len(train_docs)} docs ({args.epochs} epoch(s))")
    model_no_evi = build_model()
    run_stage(model_no_evi, train_loader, args, device, args.epochs,
              "grep-no-evi", dev_loader, dev_docs, train_docs, id2rel,
              use_evidence_loss=False)

    print("[stage 3] inference fusion (Eq 22)")
    fused_preds, gamma_used, fused_metrics = inference_fusion(
        model_full, model_no_evi, dev_docs, dev_features, tokenizer, rel2id, id2rel,
        device, gamma=args.gamma, sweep=args.sweep_gamma, ign_docs=train_docs,
    )
    print(f"[fused | gamma={gamma_used:+.2f}] dev_F1={fused_metrics['f1'] * 100:.2f} "
          f"Ign_F1={fused_metrics['ign_f1'] * 100:.2f} "
          f"(P={fused_metrics['precision'] * 100:.2f} R={fused_metrics['recall'] * 100:.2f})")
    print(f"[reference] model_full alone: dev_F1={metrics_full['f1'] * 100:.2f} "
          f"Ign_F1={metrics_full['ign_f1'] * 100:.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    pred_path = RESULTS_DIR / f"{args.run_name}_dev_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as fp:
        json.dump(fused_preds, fp, ensure_ascii=False)
    print(f"[saved] {pred_path}  ({len(fused_preds)} predicted relations)")

    if args.save_model:
        torch.save(model_full.state_dict(), RESULTS_DIR / f"{args.run_name}_full.pt")
        torch.save(model_no_evi.state_dict(), RESULTS_DIR / f"{args.run_name}_no_evi.pt")
        print(f"[saved] {RESULTS_DIR / f'{args.run_name}_full.pt'}, "
              f"{RESULTS_DIR / f'{args.run_name}_no_evi.pt'}")

    return {"model_full": metrics_full, "fused": fused_metrics}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GREP (Zhang, Yan & Cheng, ACL Findings 2025) on DocRED")
    p.add_argument("--model_name_or_path", default="bert-base-cased")
    p.add_argument("--train_split", default="train_annotated")
    p.add_argument("--dev_split", default="dev")
    p.add_argument("--run_name", default="grep")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--encoder_lr", type=float, default=5e-5)
    p.add_argument("--classifier_lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--emb_size", type=int, default=768)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--node_dim", type=int, default=0, help="graph node feature dim (0 = encoder hidden size)")
    p.add_argument("--graph_layers", type=int, default=2)
    p.add_argument("--graph_heads", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.1, help="Global Relation Prediction loss weight")
    p.add_argument("--beta", type=float, default=0.1, help="Evidence Extraction loss weight (model_full only)")
    p.add_argument("--gamma", type=float, default=0.0, help="Inference Fusion offset (Eq 22)")
    p.add_argument("--sweep_gamma", action="store_true", help="pick gamma from a small dev-F1 grid search")
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--limit_docs", type=int, default=0, help="cap train/dev docs (0 = all); for quick runs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if args.node_dim == 0:
        args.node_dim = None
    train(args)
