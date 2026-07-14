"""Train / evaluate the integrated model (re_model_full.DocREModelFull).

baseline 학습 스크립트(train_re.py)는 무수정. 그 헬퍼(set_seed / build_argparser
/ ATLoss / PUATLoss)를 import로 재사용하고, 통합 모델 전용 진입점을 따로 둔다.
forward 시그니처가 sent_pos·evidence를 추가로 받으므로 collate / predict /
run_stage를 여기서 다시 정의한다 (train_graph.py와 같은 이유).

기본값이 PNG 파이프라인에 맞춰져 있다:
  --model_name_or_path roberta-base   (RoBERTa Encoder)
  --use_pu_loss (기본 켜짐) --na_weight 0.7   (TTM-RE PU, w=0.7 — distant 단계만)
  --evi_lambda 0.1                      (DREEAM evidence 지도학습 가중치)

PU는 distant 사전학습 단계에만 적용된다(annotated의 Na는 gold이므로 fine-tune은
표준 ATLoss). evidence loss는 evidence가 있는 pair에서만 계산되므로 distant
단계에서는 자동으로 0이 된다.

Run from the project root, e.g.

    # 풀 학습 (Colab A100, baseline과 동일 레시피)
    python -m Scripts.atlop.train_full --epochs 15 --distant_limit 20000 \
        --distant_epochs 1 --eval_batch_size 32 --save_model
    # -> results/atlop_full.pt

    # CPU mini sanity run
    python -m Scripts.atlop.train_full --limit_docs 4 --epochs 1 \
        --distant_mode none --train_batch_size 2 --eval_batch_size 2
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers import get_linear_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_dataset import DocREDataset            # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES     # noqa: E402
from Scripts.atlop.losses import ATLoss, PUATLoss         # noqa: E402
from Scripts.atlop.preprocess_full import build_features_full  # noqa: E402
from Scripts.atlop.re_model_full import DocREModelFull    # noqa: E402
from Scripts.atlop.train_re import build_argparser, set_seed   # noqa: E402
from Scripts.eval.scorer import evaluate                  # noqa: E402

RESULTS_DIR = ROOT / "results"


def make_collate_fn(pad_token_id: int):
    """make_collate_fn(train_re) + sent_pos / evidence passthrough."""
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
            "sent_pos": [f["sent_pos"] for f in features],
            "hts": [f["hts"] for f in features],
            "evidence": [f["evidence"] for f in features],
            "labels": labels,
            "features": features,
        }
    return collate


@torch.no_grad()
def predict(model, loader, id2rel, device) -> list[dict]:
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
    # freshly-initialized layers (incl. graph + evidence gate) get classifier LR
    new_layers = ("extractor", "bilinear", "graph", "evi_gate")
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
    print(f"[device] {device}  [model] full (DREEAM-LCP + GREP-GAT + PU)")

    if args.run_name == "atlop":
        args.run_name = "atlop_full"
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
    model = DocREModelFull(config, encoder, emb_size=args.emb_size,
                           block_size=args.block_size, num_labels=NUM_CLASSES,
                           graph_layers=args.graph_layers, graph_dim=args.graph_dim,
                           graph_heads=args.graph_heads, graph_dropout=args.graph_dropout,
                           evi_lambda=args.evi_lambda).to(device)

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
        if args.use_pu_loss:
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
        model.loss_fnt = ATLoss()  # annotated Na is gold -> plain ATLoss

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


def build_full_argparser():
    p = build_argparser()
    p.description = "Integrated DocRE model (DREEAM-LCP + GREP-GAT + PU ATLoss) on DocRED"
    p.add_argument("--graph_layers", type=int, default=2, help="pair-graph propagation layers")
    p.add_argument("--graph_dim", type=int, default=256, help="pair-graph node feature dim")
    p.add_argument("--graph_heads", type=int, default=4, help="GAT attention heads")
    p.add_argument("--graph_dropout", type=float, default=0.1)
    p.add_argument("--evi_lambda", type=float, default=0.1,
                   help="weight of the DREEAM evidence-supervision loss (0 disables it)")
    p.add_argument("--init_ckpt", default="", help="warm-start state_dict (strict=False)")
    # PNG 파이프라인 기본값: RoBERTa 인코더 + PU(w=0.7) 켜짐
    p.set_defaults(model_name_or_path="roberta-base", use_pu_loss=True, na_weight=0.7)
    return p


if __name__ == "__main__":
    train(build_full_argparser().parse_args())
