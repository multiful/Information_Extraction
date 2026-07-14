"""Train / evaluate re_model_gat_mlp.DocREModelGATMLP:
BERT -> ATLOP LCP -> entity-pair rep -> 2-layer edge-featured GAT -> 2-layer
MLP classifier -> sigmoid, loss = BCEWithLogitsLoss + evi_weight x evidence
contrastive loss (losses.BCEEvidenceContrastiveLoss).

Reuses train_re's set_seed / build_argparser and train_full's make_collate_fn
/ predict (this model's forward signature -- entity_pos, hts, sent_pos,
labels, evidence -- matches what those already assume). preprocess is reused
unmodified too: build_features_full already emits the sent_pos/evidence this
model's evidence-contrastive loss needs.

--use_pu_loss / --na_weight / --evi_lambda from the shared argparser do NOT
apply here (PUATLoss ranks classes against a TH logit, which this sigmoid
head doesn't have; DREEAM's CE evidence loss is superseded by evi_weight's
InfoNCE version) -- loss_fnt stays BCEEvidenceContrastiveLoss(evi_weight, tau)
for every stage, no swapping.

Run from the project root, e.g.

    # full run (Colab GPU)
    python -m Scripts.atlop.train_gat_mlp --epochs 15 --distant_limit 20000 \
        --distant_epochs 1 --evi_weight 0.2 --tau 0.1 --save_model

    # CPU mini sanity run
    python -m Scripts.atlop.train_gat_mlp --limit_docs 4 --epochs 1 \
        --distant_mode none --train_batch_size 2 --eval_batch_size 2
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import get_linear_schedule_with_warmup

from data.docred_dataset import DocREDataset            # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES     # noqa: E402
from Scripts.atlop.preprocess_full import build_features_full  # noqa: E402
from Scripts.atlop.re_model_gat_mlp import DocREModelGATMLP     # noqa: E402
from Scripts.atlop.train_full import make_collate_fn, predict  # noqa: E402
from Scripts.atlop.train_re import build_argparser, set_seed   # noqa: E402
from Scripts.eval.scorer import evaluate                 # noqa: E402

RESULTS_DIR = ROOT / "results"


def build_optim_sched(model, args, total_steps):
    # freshly-initialized layers get classifier LR; pretrained encoder gets encoder LR.
    # ("bilinear" is inherited-but-unused so which group it lands in doesn't matter.)
    new_layers = ("extractor", "bilinear", "graph", "classifier")
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
    """train_full.run_stage, but calling this file's build_optim_sched (its
    fresh-layer names differ: `classifier`, not `evi_gate`) -- kept as its own
    copy rather than a shared import, same reason train_full.py doesn't share
    train_re.py's version."""
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
                labels=batch["labels"],
                evidence=batch["evidence"],
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
    print(f"[device] {device}  [model] GAT+MLP+Sigmoid "
          f"(BCE + {args.evi_weight} x evidence-contrastive, tau={args.tau})")

    if args.run_name == "atlop":
        args.run_name = "atlop_gat_mlp"
        print(f"[run_name] auto-set to {args.run_name}")

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

    dev_features = build_features_full(dev_docs, tokenizer, rel2id)
    dev_loader = DataLoader(dev_features, batch_size=args.eval_batch_size,
                            shuffle=False, collate_fn=collate)
    train_features = build_features_full(train_docs, tokenizer, rel2id)
    train_loader = DataLoader(train_features, batch_size=args.train_batch_size,
                              shuffle=True, collate_fn=collate)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=NUM_CLASSES)
    encoder = AutoModel.from_pretrained(
        args.model_name_or_path, config=config, attn_implementation="eager"
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = DocREModelGATMLP(
        config, encoder, emb_size=args.emb_size, block_size=args.block_size, num_labels=NUM_CLASSES,
        graph_layers=args.graph_layers, graph_dim=args.graph_dim, graph_heads=args.graph_heads,
        graph_dropout=args.graph_dropout, mlp_hidden=args.mlp_hidden, mlp_dropout=args.mlp_dropout,
        evi_weight=args.evi_weight, tau=args.tau, threshold=args.threshold,
    ).to(device)

    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init] warm-start from {args.init_ckpt} "
              f"(missing={len(missing)} fresh params, unexpected={len(unexpected)})")

    def load_distant():
        cap = args.limit_docs if args.limit_docs > 0 else args.distant_limit
        docs = list(DocREDataset(args.distant_split))
        if cap > 0:
            docs = docs[:cap]
        return docs, build_features_full(docs, tokenizer, rel2id)

    if args.distant_mode == "pretrain":
        distant_docs, distant_features = load_distant()
        print(f"[stage 1] distant pretrain on {len(distant_docs)} docs "
              f"({args.distant_epochs} epoch(s))")
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

        if args.epochs > 0:
            print(f"[stage 2] annotated fine-tune on {len(train_docs)} docs ({args.epochs} epoch(s))")
            metrics, preds = run_stage(model, train_loader, args, device, args.epochs,
                                       "annotated-finetune", dev_loader, dev_docs, train_docs, id2rel)
        else:
            print("[stage 2] skipped (--epochs 0) -- reporting stage-1 metrics/predictions")
    else:
        print(f"[stage 1] annotated train on {len(train_docs)} docs ({args.epochs} epoch(s))")
        metrics, preds = run_stage(model, train_loader, args, device, args.epochs,
                                   "annotated-train", dev_loader, dev_docs, train_docs, id2rel)

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


def build_gat_mlp_argparser():
    p = build_argparser()
    p.description = "ATLOP-LCP + entity-pair edge-featured GAT + MLP classifier " \
                     "(sigmoid + BCE + evidence-contrastive) on DocRED"
    p.add_argument("--graph_layers", type=int, default=2, help="pair-graph propagation layers")
    p.add_argument("--graph_dim", type=int, default=256, help="pair-graph node feature dim")
    p.add_argument("--graph_heads", type=int, default=4, help="GAT attention heads")
    p.add_argument("--graph_dropout", type=float, default=0.1)
    p.add_argument("--mlp_hidden", type=int, default=256, help="relation classifier MLP hidden dim")
    p.add_argument("--mlp_dropout", type=float, default=0.1)
    p.add_argument("--evi_weight", type=float, default=0.2,
                   help="weight of the InfoNCE evidence-contrastive loss (0 disables it)")
    p.add_argument("--tau", type=float, default=0.1, help="evidence-contrastive InfoNCE temperature")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="sigmoid probability threshold for a positive relation (no adaptive TH class)")
    p.add_argument("--init_ckpt", default="", help="warm-start state_dict (strict=False)")
    return p


if __name__ == "__main__":
    train(build_gat_mlp_argparser().parse_args())
