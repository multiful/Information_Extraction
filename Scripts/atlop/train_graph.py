"""Train / evaluate the graph improvement models on DocRED.

baseline 학습 스크립트(train_re.py)는 한 줄도 수정하지 않고 그대로 둔다 --
여기서는 그 스크립트의 헬퍼(set_seed / make_collate_fn / predict /
denoise_features / build_argparser)를 import로 재사용만 하면서, 개선 모델
전용 진입점을 따로 제공한다.

  --model gcn   [개선 1] re_model_gcn.DocREModelGCN  (Entity Pair Graph + GCN)
  --model gat   [개선 2] re_model_gat.DocREModelGAT  (LCP + Entity Pair Graph + GAT)

--distant_mode(pretrain/denoise/none) 등 나머지 인자와 학습 흐름은
train_re.py와 동일하므로 baseline과 같은 레시피로 돌려 공정하게 비교할 수
있다. run_stage/build_optim_sched만 여기 다시 두는 이유: 새로 추가된
graph_* 파라미터는 인코더 LR(5e-5)이 아니라 분류기 LR(1e-4) 그룹에 넣어야
하는데, train_re.py를 수정하지 않기로 했기 때문.

Run from the project root, e.g.

    # 같은 레시피 재학습 (baseline과 정면 비교; GPU/Colab)
    python -m Scripts.atlop.train_graph --model gcn --epochs 15 \
        --distant_limit 20000 --distant_epochs 1 --eval_batch_size 32 --save_model
    python -m Scripts.atlop.train_graph --model gat --epochs 15 \
        --distant_limit 20000 --distant_epochs 1 --eval_batch_size 32 --save_model

    # 빠른 대안: 학습된 baseline에서 warm-start 후 annotated만 추가 fine-tune
    # (graph head가 zero-init이라 시작점이 정확히 baseline. 이 경우 대조군으로
    #  baseline도 같은 에폭만큼 추가 fine-tune해서 비교할 것 -- README 참고)
    python -m Scripts.atlop.train_graph --model gat --init_ckpt results/atlop.pt \
        --distant_mode none --epochs 5 --save_model

    # CPU mini sanity run
    python -m Scripts.atlop.train_graph --model gcn --limit_docs 4 --epochs 1 \
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
from Scripts.atlop.preprocess import build_features       # noqa: E402
from Scripts.atlop.re_model_gat import DocREModelGAT      # noqa: E402
from Scripts.atlop.re_model_gcn import DocREModelGCN      # noqa: E402
from Scripts.atlop.train_re import (                       # noqa: E402
    RESULTS_DIR, build_argparser, denoise_features, make_collate_fn, predict, set_seed,
)
from Scripts.eval.scorer import evaluate                  # noqa: E402


def build_model(args, config, encoder):
    common = dict(emb_size=args.emb_size, block_size=args.block_size,
                  num_labels=NUM_CLASSES, graph_layers=args.graph_layers,
                  graph_dim=args.graph_dim, graph_dropout=args.graph_dropout)
    if args.model == "gcn":
        return DocREModelGCN(config, encoder, **common)
    return DocREModelGAT(config, encoder, graph_heads=args.graph_heads, **common)


def build_optim_sched(model, args, total_steps):
    """train_re.build_optim_sched와 동일하되 graph_* 파라미터를 분류기 LR
    그룹에 포함한다 (freshly-initialized layers get the higher LR)."""
    new_layers = ("extractor", "bilinear", "graph")
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
    """train_re.run_stage와 동일한 흐름 (이 파일의 build_optim_sched를 쓰도록
    여기 재정의)."""
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
                labels=batch["labels"],
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
    print(f"[device] {device}  [model] {args.model}")

    # baseline 산출물(results/atlop*)을 덮어쓰지 않도록 기본 run_name 분리
    if args.run_name == "atlop":
        args.run_name = f"atlop_{args.model}"
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

    dev_features = build_features(dev_docs, tokenizer, rel2id)
    dev_loader = DataLoader(dev_features, batch_size=args.eval_batch_size,
                            shuffle=False, collate_fn=collate)
    train_features = build_features(train_docs, tokenizer, rel2id)
    train_loader = DataLoader(train_features, batch_size=args.train_batch_size,
                              shuffle=True, collate_fn=collate)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=NUM_CLASSES)
    # eager attention: localized context pooling needs attention weights.
    encoder = AutoModel.from_pretrained(
        args.model_name_or_path, config=config, attn_implementation="eager"
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = build_model(args, config, encoder).to(device)

    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init] warm-start from {args.init_ckpt} "
              f"(missing={len(missing)} fresh graph params, unexpected={len(unexpected)})")
        if unexpected:
            print(f"[init][warn] unexpected keys (ckpt/model mismatch?): {unexpected[:5]}")

    def load_distant():
        cap = args.limit_docs if args.limit_docs > 0 else args.distant_limit
        docs = list(DocREDataset(args.distant_split))
        if cap > 0:
            docs = docs[:cap]
        return docs, build_features(docs, tokenizer, rel2id)

    if args.distant_mode == "pretrain":
        distant_docs, distant_features = load_distant()
        print(f"[stage 1] distant pretrain on {len(distant_docs)} docs "
              f"({args.distant_epochs} epoch(s))")
        if args.use_pu_loss:
            # train_re.py와 동일: PU loss는 distant 단계에만, stage 2 전에 원복.
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
        print(f"[stage 1] annotated train on {len(train_docs)} docs ({args.epochs} epoch(s))")
        metrics, preds = run_stage(model, train_loader, args, device, args.epochs,
                                   "annotated-train", dev_loader, dev_docs, train_docs, id2rel)

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


def build_graph_argparser():
    """train_re.build_argparser에 그래프 전용 인자만 얹는다."""
    p = build_argparser()
    p.description = "ATLOP graph improvement models (GCN / GAT) on DocRED"
    p.add_argument("--model", choices=["gcn", "gat"], required=True,
                   help="gcn = improvement 1 (entity-pair graph + relational GCN); "
                        "gat = improvement 2 (LCP + entity-pair graph + graph attention)")
    p.add_argument("--graph_layers", type=int, default=2, help="pair-graph propagation layers")
    p.add_argument("--graph_dim", type=int, default=256, help="pair-graph node feature dim")
    p.add_argument("--graph_heads", type=int, default=4, help="attention heads (gat only)")
    p.add_argument("--graph_dropout", type=float, default=0.1)
    p.add_argument("--init_ckpt", default="",
                   help="warm-start state_dict, e.g. results/atlop.pt; loaded strict=False so "
                        "graph params stay freshly initialized (zero-init head = starts at baseline)")
    return p


if __name__ == "__main__":
    train(build_graph_argparser().parse_args())
