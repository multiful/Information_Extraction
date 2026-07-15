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

    # revised-data run (no distant stage): train/dev/test are ad-hoc json
    # files instead of the named train_annotated/dev/train_distant splits --
    # --distant_epochs 0 skips the distant stage entirely (it's just not
    # invoked, no distant_revised file needed), and --test_file runs a final
    # triple-prediction pass (+ F1/Ign F1 if the file has labels) after
    # training, without affecting early stopping/checkpoint selection.
    python -m Scripts.dk_gat.train_gat \
        --train_split docred_data/data/train_revised.json \
        --dev_split docred_data/data/dev_revised.json \
        --test_file docred_data/data/test_revised.json \
        --distant_epochs 0 --epochs 20 \
        --run_name dk_gat_revised --save_model --seed 66
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
from Scripts.atlop.losses import ATLoss, PUATLoss          # noqa: E402
from Scripts.dk_gat.model import DocREGATModel            # noqa: E402
from Scripts.dk_gat.preprocess_gat import build_gat_features  # noqa: E402
from Scripts.eval.scorer import evaluate                  # noqa: E402

RESULTS_DIR = ROOT / "results"


def load_docs(split_or_path: str) -> list:
    """`split_or_path`: a named split from data.docred_io.SPLITS (loaded via
    DocREDataset, current behavior), or -- for anything else -- a path
    (absolute, or relative to the project root) to a DocRED-format json file
    (list of {title, sents, vertexSet, labels}), loaded directly. Lets
    --train_split/--dev_split/--distant_split/--test_file point at ad-hoc
    files (e.g. docred_data/data/train_revised.json) without touching the
    shared data/docred_io.py SPLITS list."""
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
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "features": features}
    return collate


@torch.no_grad()
def predict(model, loader, id2rel, device, evidence_fusion=False, evidence_fusion_top_k=3):
    """Adaptive-Thresholding decode: a relation is emitted iff its logit beats
    the pair's own learned TH (class 0) logit -- no global threshold, matches
    Scripts/atlop/train_re.py's predict().

    evidence_fusion: EIDER-style inference-time fusion (see model.py's
    DocREGATModel.forward docstring) -- no new parameters, so it's valid to
    turn on for ANY already-trained checkpoint, not just future runs. Passed
    here (not baked into forward()'s default) so every dev-eval call during
    training (run_stage) can optionally use it too, letting early-stopping
    and best-checkpoint selection reflect the fused score if enabled."""
    model.eval()
    out = []
    for batch in loader:
        logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                       batch["features"], evidence_fusion=evidence_fusion,
                       evidence_fusion_top_k=evidence_fusion_top_k)[0]
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


def build_param_groups(model, lr, weight_decay, layerwise_lr_decay):
    """Flat LR (layerwise_lr_decay=1.0, default -- current behavior, single
    param group) or BERT layer-wise decay: each encoder layer's LR is
    lr * decay^(depth from the top layer), embeddings get one extra decay
    step, and everything outside model.encoder (GAT/classifier/edge-embeddings
    -- the task-specific head) always trains at the full lr regardless of
    decay. Pooler (unused in forward()) is simply never assigned a group, so
    it never trains -- harmless since it has no gradient anyway."""
    if layerwise_lr_decay >= 1.0:
        return [{"params": list(model.parameters()), "lr": lr, "weight_decay": weight_decay}]
    bert = model.encoder
    layers = list(bert.encoder.layer)
    num_layers = len(layers)
    groups = [{"params": list(bert.embeddings.parameters()),
               "lr": lr * (layerwise_lr_decay ** (num_layers + 1)),
               "weight_decay": weight_decay}]
    for depth, layer in enumerate(layers):
        groups.append({"params": list(layer.parameters()),
                        "lr": lr * (layerwise_lr_decay ** (num_layers - depth)),
                        "weight_decay": weight_decay})
    encoder_param_ids = {id(p) for p in bert.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    groups.append({"params": head_params, "lr": lr, "weight_decay": weight_decay})
    return groups


def run_stage(model, loader, args, device, epochs, stage, dev_loader, dev_docs, ign_docs, id2rel,
             best_ckpt_path=None, lr=None, freeze_encoder_epochs=0, evidence_start_epoch=0,
             early_stop_patience=0, na_weight_schedule=None, evidence_fusion=False,
             evidence_fusion_top_k=3):
    """best_ckpt_path: if given, save model.state_dict() there every time a new
    best dev F1 is seen (separate from the caller's own final-epoch save) --
    epoch-to-epoch dev F1 isn't monotonic (we've observed real dips, e.g.
    epoch 6 59.85 -> epoch 7 59.57 on a real run), so whichever epoch happens
    to be last isn't guaranteed to be the best one actually reached.

    lr: overrides args.lr for this stage (used for --lr2, stage 2 only);
    defaults to args.lr when not given. freeze_encoder_epochs/
    evidence_start_epoch/early_stop_patience: see their --help text in
    build_argparser -- all default to 0/disabled, i.e. current behavior.

    na_weight_schedule: optional (start, end) tuple -- linearly anneals
    model.loss_fnt.na_weight from `start` at step 0 to `end` at the last
    training step of *this stage* (step-level, not epoch-level, since stage 1
    is only 1 epoch in the controlled schedule -- an epoch-level schedule
    would never move). Only applied when model.loss_fnt actually has a
    na_weight attribute (i.e. PUATLoss is attached); silently skipped
    otherwise (e.g. if called for stage 2, which uses plain ATLoss)."""
    lr = args.lr if lr is None else lr
    total_steps = max(1, len(loader) * epochs)
    param_groups = build_param_groups(model, lr, args.weight_decay, args.layerwise_lr_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps)
    metrics, preds = None, None
    best_f1, best_epoch, no_improve = -1.0, -1, 0
    base_evidence_weight = model.evidence_weight
    apply_na_schedule = na_weight_schedule is not None and hasattr(model.loss_fnt, "na_weight")
    global_step = 0
    for epoch in range(epochs):
        if freeze_encoder_epochs > 0:
            should_freeze = epoch < freeze_encoder_epochs
            for p in model.encoder.parameters():
                p.requires_grad = not should_freeze
            if epoch == 0 or epoch == freeze_encoder_epochs:
                print(f"  [{stage}] encoder {'frozen' if should_freeze else 'unfrozen'} "
                      f"(epoch {epoch})", flush=True)
        if evidence_start_epoch > 0:
            model.evidence_weight = base_evidence_weight if epoch >= evidence_start_epoch else 0.0
        model.train()
        running = 0.0
        for step, batch in enumerate(loader):
            if apply_na_schedule:
                start, end = na_weight_schedule
                frac = global_step / max(1, total_steps - 1)
                model.loss_fnt.na_weight = start + (end - start) * frac
            loss, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                            batch["features"], labels=batch["labels"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running += loss.item()
            global_step += 1
            if (step + 1) % args.log_every == 0:
                na_w_msg = f" na_weight={model.loss_fnt.na_weight:.3f}" if apply_na_schedule else ""
                print(f"  [{stage}] epoch {epoch} step {step + 1}/{len(loader)} "
                      f"loss {running / (step + 1):.4f}{na_w_msg}", flush=True)
        preds = predict(model, dev_loader, id2rel, device,
                        evidence_fusion=evidence_fusion, evidence_fusion_top_k=evidence_fusion_top_k)
        metrics = evaluate(preds, dev_docs, ign_docs)
        is_best = metrics["f1"] > best_f1
        print(f"[{stage} | epoch {epoch}] train_loss={running / max(1, len(loader)):.4f} "
              f"dev_F1={metrics['f1'] * 100:.2f} Ign_F1={metrics['ign_f1'] * 100:.2f} "
              f"(P={metrics['precision'] * 100:.2f} R={metrics['recall'] * 100:.2f})"
              f"{'  <- new best' if is_best else ''}", flush=True)
        if is_best:
            best_f1, best_epoch, no_improve = metrics["f1"], epoch, 0
            if best_ckpt_path is not None:
                torch.save(model.state_dict(), best_ckpt_path)
        else:
            no_improve += 1
            if early_stop_patience > 0 and no_improve >= early_stop_patience:
                print(f"[{stage}] early stopping at epoch {epoch} "
                      f"(no dev F1 improvement for {no_improve} epochs, best={best_f1 * 100:.2f} "
                      f"@ epoch {best_epoch})", flush=True)
                break
    if freeze_encoder_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = True
    if evidence_start_epoch > 0:
        model.evidence_weight = base_evidence_weight
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

    train_docs = load_docs(args.train_split)
    dev_docs = load_docs(args.dev_split)
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
                          evidence_weight=args.evidence_weight,
                          use_jk=not args.no_jk,
                          use_gated_fusion=args.use_gated_fusion,
                          use_bilinear_classifier=args.use_bilinear_classifier,
                          use_abs_diff=args.use_abs_diff,
                          use_metapath_attention=args.use_metapath_attention,
                          use_pair_graph=args.use_pair_graph,
                          pair_graph_layers=args.pair_graph_layers,
                          pair_graph_dim=args.pair_graph_dim,
                          pair_graph_heads=args.pair_graph_heads).to(device)

    stage1_metrics, stage1_preds = None, None
    if args.distant_epochs > 0:
        distant_docs = load_docs(args.distant_split)
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
        na_weight_schedule = ((args.na_weight_start, args.na_weight)
                              if args.curriculum_na_weight else None)
        if args.curriculum_na_weight and not args.use_pu_loss:
            print("[warning] --curriculum_na_weight has no effect without --use_pu_loss "
                  "(plain ATLoss has no na_weight to anneal)", flush=True)
        # freeze_encoder_epochs is NOT applied here (stage 1) even if requested --
        # distant_epochs defaults to 1 (matching the atlop baseline's controlled
        # schedule), so "freeze the first N epochs" would freeze the ENTIRE distant
        # stage. That defeats distant pretraining's whole point (adapting the encoder
        # to noisy-but-large in-domain data before the small annotated set) and, on a
        # real run (dk_gat_v2, 2026-07-14), measurably dragged down stage-1 dev F1
        # (37.44 vs the prior 50.08 without freezing) and every following annotated
        # epoch tracked ~3+ F1 behind the equivalent un-frozen run. Only apply the
        # freeze to stage 2 below, where "first 1 of 15 epochs" is a small, sensible
        # warm-up cost instead of "100% of the stage."
        stage1_metrics, stage1_preds = run_stage(
            model, distant_loader, args, device, args.distant_epochs,
            "distant-pretrain", dev_loader, dev_docs, train_docs, id2rel,
            best_ckpt_path=stage1_best, freeze_encoder_epochs=0,
            evidence_start_epoch=args.evidence_start_epoch,
            early_stop_patience=args.early_stop_patience,
            na_weight_schedule=na_weight_schedule)
        del distant_loader, distant_docs
        if args.save_model:
            p = RESULTS_DIR / f"{args.run_name}_stage1.pt"
            torch.save(model.state_dict(), p)
            print(f"[saved] {p}  (final distant epoch, not necessarily best -- "
                  f"see {stage1_best} for that)", flush=True)
        # annotated's Na labels are gold, not distant-supervision noise -- always
        # plain ATLoss for stage 2, matching Scripts/atlop/train_re.py.
        model.loss_fnt = ATLoss()

    if args.epochs > 0:
        print(f"[stage 2] annotated fine-tune on {len(train_docs)} docs ({args.epochs} epoch(s))",
              flush=True)
        RESULTS_DIR.mkdir(exist_ok=True)
        best_ckpt = RESULTS_DIR / f"{args.run_name}_best.pt" if args.save_model else None
        metrics, preds = run_stage(model, train_loader, args, device, args.epochs,
                                   "annotated-finetune", dev_loader, dev_docs, train_docs, id2rel,
                                   best_ckpt_path=best_ckpt, lr=args.lr2,
                                   freeze_encoder_epochs=args.freeze_encoder_epochs,
                                   evidence_start_epoch=args.evidence_start_epoch,
                                   early_stop_patience=args.early_stop_patience,
                                   evidence_fusion=args.evidence_fusion,
                                   evidence_fusion_top_k=args.evidence_fusion_top_k)
    else:
        # --epochs 0: quick distant-only screening run (e.g. na_weight/gat_heads
        # sweeps) -- matches Scripts/atlop/train_re.py's same convention. Falls
        # back to stage 1's metrics/predictions instead of leaving them None
        # (which used to crash json.dump/len below).
        print("[stage 2] skipped (--epochs 0) -- reporting stage-1 metrics/predictions",
              flush=True)
        metrics, preds = stage1_metrics, stage1_preds
        best_ckpt = None

    if preds is None:
        # both --distant_epochs 0 and --epochs 0 -- nothing was actually trained/evaluated.
        print("[warning] no stage ran (--distant_epochs 0 and --epochs 0) -- nothing to save",
              flush=True)
        return metrics
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
            best_preds = predict(model, dev_loader, id2rel, device,
                                 evidence_fusion=args.evidence_fusion,
                                 evidence_fusion_top_k=args.evidence_fusion_top_k)
            best_pred_path = RESULTS_DIR / f"{args.run_name}_best_dev_predictions.json"
            with open(best_pred_path, "w", encoding="utf-8") as fp:
                json.dump(best_preds, fp, ensure_ascii=False)
            best_metrics = evaluate(best_preds, dev_docs, train_docs)
            print(f"[best checkpoint] dev_F1={best_metrics['f1'] * 100:.2f} "
                  f"Ign_F1={best_metrics['ign_f1'] * 100:.2f} -- "
                  f"saved {best_pred_path} ({len(best_preds)} predicted relations)", flush=True)

    if args.test_file:
        # Runs on whatever weights the model currently holds -- the best-dev-F1
        # checkpoint if --save_model produced one (loaded in-place just above),
        # otherwise the final-epoch weights. No training/early-stopping signal
        # is derived from this split.
        test_docs = load_docs(args.test_file)
        test_loader = DataLoader(build_gat_features(test_docs, tokenizer, rel2id),
                                 batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)
        test_preds = predict(model, test_loader, id2rel, device,
                             evidence_fusion=args.evidence_fusion,
                             evidence_fusion_top_k=args.evidence_fusion_top_k)
        test_pred_path = RESULTS_DIR / f"{args.run_name}_test_predictions.json"
        with open(test_pred_path, "w", encoding="utf-8") as fp:
            json.dump(test_preds, fp, ensure_ascii=False)
        used_best = args.save_model and best_ckpt is not None and best_ckpt.exists()
        print(f"[saved] {test_pred_path}  ({len(test_preds)} predicted relations, "
              f"{'best' if used_best else 'final'} checkpoint)", flush=True)
        if any(doc.get("labels") for doc in test_docs):
            test_metrics = evaluate(test_preds, test_docs, train_docs)
            print(f"[test] F1={test_metrics['f1'] * 100:.2f} "
                  f"Ign_F1={test_metrics['ign_f1'] * 100:.2f} "
                  f"(P={test_metrics['precision'] * 100:.2f} R={test_metrics['recall'] * 100:.2f})",
                  flush=True)
    return metrics


def build_argparser():
    p = argparse.ArgumentParser(description="dk EGAT model on DocRED")
    p.add_argument("--model_name_or_path", default="bert-base-cased")
    p.add_argument("--run_name", default="dk_gat")
    p.add_argument("--train_split", default="train_annotated",
                   help="named split (data/docred_io.SPLITS) or a path (absolute, or relative "
                        "to the project root) to a DocRED-format json file, for the annotated "
                        "fine-tune stage -- e.g. docred_data/data/train_revised.json")
    p.add_argument("--dev_split", default="dev",
                   help="named split or json path used for dev eval / early stopping / "
                        "best-checkpoint selection")
    p.add_argument("--distant_split", default="train_distant",
                   help="named split or json path for the distant pretrain stage (only loaded "
                        "if --distant_epochs > 0)")
    p.add_argument("--test_file", default=None,
                   help="optional json path for a held-out split to run final triple "
                        "prediction on after training completes (e.g. "
                        "docred_data/data/test_revised.json) -- no effect on training or "
                        "checkpoint selection. If the docs have a labels key, F1/Ign F1 are "
                        "also printed")
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
    p.add_argument("--no_jk", action="store_true",
                   help="disable Jump Knowledge (max over input/layer1/layer2) and fall back "
                        "to last-GAT-layer-only output -- for A/B testing JK itself before "
                        "trusting it as the default")
    p.add_argument("--use_gated_fusion", action="store_true",
                   help="learned per-dim gate blending GAT-refined entity embedding with the "
                        "original pre-GAT one, instead of JK's max (supersedes --no_jk when "
                        "set). Off by default -- A/B test before trusting as the default")
    p.add_argument("--use_bilinear_classifier", action="store_true",
                   help="ATLOP-style grouped bilinear classifier (head/tail extractors + "
                        "block-wise outer product) instead of concat+g_h*g_t+MLP -- replaces "
                        "the interaction term entirely, doesn't stack with it. Off by default "
                        "-- A/B test before trusting as the default")
    p.add_argument("--use_abs_diff", action="store_true",
                   help="append |g_h - g_t| to the pair representation (InferSent-style), "
                        "alongside the existing g_h*g_t term. Ignored when "
                        "--use_bilinear_classifier is set. Off by default -- A/B test first")
    p.add_argument("--use_metapath_attention", action="store_true",
                   help="separate GAT attention vector per edge category (self / "
                        "entity-entity same-sentence / entity-entity mention-overlap / "
                        "entity-sentence) instead of one shared vector for all edges -- see "
                        "EdgeFeaturedGATLayer's docstring in model.py. Off by default -- "
                        "unlike the other use_* flags this has NOT been distant-screened yet, "
                        "only CPU smoke-tested for crash-freedom; treat as unvalidated")
    p.add_argument("--use_pair_graph", action="store_true",
                   help="second graph stage over (h,t) pair-nodes (not entities), directly "
                        "targeting A->B,B->C=>A->C composition via same-head/same-tail/"
                        "bridge-succ/bridge-pred edges + relation-conditioned messages (this "
                        "pair's own provisional logits fed into its node feature) -- see "
                        "model.py module docstring. zero-init residual head, so it can only "
                        "help vs the no-pair-graph model. Off by default -- new, CPU "
                        "smoke-tested only, not distant-screened yet")
    p.add_argument("--pair_graph_layers", type=int, default=2,
                   help="pair-graph propagation layers (use_pair_graph only)")
    p.add_argument("--pair_graph_dim", type=int, default=256,
                   help="pair-graph node feature dim (use_pair_graph only)")
    p.add_argument("--pair_graph_heads", type=int, default=4,
                   help="pair-graph attention heads (use_pair_graph only)")
    p.add_argument("--evidence_fusion", action="store_true",
                   help="EIDER-style (Xie et al., Findings ACL 2022) inference-time fusion: "
                        "average the normal full-document prediction with a second prediction "
                        "made from an evidence-only LCP context (evidence sentences predicted "
                        "unsupervised from the model's own context/sentence similarity, no "
                        "gold evidence needed -- works at test time too). No new trainable "
                        "parameters -- cheap approximation reusing this forward pass's cached "
                        "encoder output instead of a second BERT pass over a reconstructed "
                        "document, see model.py's DocREGATModel.forward docstring. Applied "
                        "during every annotated-stage dev eval (so early-stop/best-checkpoint "
                        "selection reflects it) and the final best-checkpoint prediction. Off "
                        "by default -- new, CPU smoke-tested only")
    p.add_argument("--evidence_fusion_top_k", type=int, default=3,
                   help="number of predicted-evidence sentences kept per pair for "
                        "--evidence_fusion (capped at the doc's actual sentence count)")
    p.add_argument("--evidence_weight", type=float, default=0.2)
    p.add_argument("--evidence_start_epoch", type=int, default=0,
                   help="curriculum: evidence contrastive loss is added only from this "
                        "within-stage epoch onward (0 = active from the start, current "
                        "behavior). Motivation: early epochs the LCP context is still noisy, "
                        "so pulling it toward evidence sentences may fight the main ATLoss "
                        "signal before the model has learned basic entity/relation cues")
    p.add_argument("--freeze_encoder_epochs", type=int, default=0,
                   help="freeze the BERT encoder's parameters for this many epochs at the "
                        "start of ANNOTATED fine-tune only (0 = never freeze, current "
                        "behavior) -- lets the GAT/classifier head warm up on a stable "
                        "pretrained representation before the encoder itself starts moving. "
                        "NOT applied to the distant stage regardless of this value -- "
                        "distant_epochs defaults to 1, so freezing there would freeze the "
                        "entire stage and defeat its purpose (measured real cost: stage-1 "
                        "dev F1 37.44 vs 50.08 without freezing, dk_gat_v2 2026-07-14)")
    p.add_argument("--lr2", type=float, default=None,
                   help="separate learning rate for stage 2 (annotated fine-tune); defaults "
                        "to --lr (current behavior, unset) if not given. Fine-tune stages "
                        "often want a lower LR than pretrain -- keep this as an explicit A/B "
                        "knob instead of silently changing --lr's default, since --lr is also "
                        "what stage 1 uses and changing it would break the controlled "
                        "architecture-only comparison against Scripts/atlop baseline")
    p.add_argument("--layerwise_lr_decay", type=float, default=1.0,
                   help="BERT layer-wise LR decay factor (1.0 = disabled/uniform LR, current "
                        "behavior). Each encoder layer's LR is base_lr * decay^(depth from "
                        "top); embeddings get one extra decay step. Typical values 0.8-0.95 "
                        "-- lower layers move less, reducing catastrophic forgetting of "
                        "pretrained representations during fine-tune")
    p.add_argument("--early_stop_patience", type=int, default=0,
                   help="stop a stage early if dev F1 doesn't improve for this many "
                        "consecutive epochs (0 = disabled, current behavior -- always runs "
                        "the full epoch count)")
    p.add_argument("--use_pu_loss", action="store_true",
                   help="PUATLoss for the distant stage instead of plain ATLoss "
                        "(na_weight=0.7 default was swept and validated on Scripts/atlop -- "
                        "recommended on; matches Scripts/atlop/train_re.py's flag)")
    p.add_argument("--na_weight", type=float, default=0.7,
                   help="PUATLoss down-weight for distant all-Na pairs' TH-ranking term -- "
                        "also the *end* value of the curriculum when --curriculum_na_weight "
                        "is set")
    p.add_argument("--curriculum_na_weight", action="store_true",
                   help="anneal PUATLoss na_weight linearly from --na_weight_start (step 0) "
                        "to --na_weight (last step) over the distant stage, instead of holding "
                        "it fixed at --na_weight throughout. Motivation: at step 0 the GAT/"
                        "classifier heads are randomly initialized and can't yet tell a "
                        "mislabeled distant Na pair from a real one, so trusting the distant "
                        "labels fully (na_weight closer to 1.0, i.e. plain ATLoss behavior) "
                        "gives a cleaner bootstrapping signal; as training progresses toward "
                        "the validated na_weight=0.7, the down-weighting of likely-noisy Na "
                        "labels phases in. No-op (with a printed warning) unless --use_pu_loss "
                        "is also set. Requires --distant_epochs > 0. Off by default -- "
                        "unvalidated (new this session, only CPU smoke-tested)")
    p.add_argument("--na_weight_start", type=float, default=1.0,
                   help="starting na_weight for --curriculum_na_weight (1.0 = fully trust "
                        "distant Na labels at step 0, i.e. plain ATLoss)")
    p.add_argument("--seed", type=int, default=66)
    p.add_argument("--limit_docs", type=int, default=0, help="cap all splits; for quick runs")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
