"""[개선] Baseline + DREAM(evidence-guided) 학습 진입점 (best-ckpt + early stopping).

re_model_dream.DocREModelDREAM 을 학습/평가한다. DREAM은 sent_pos·evidence를
쓰므로 preprocess_full.build_features_full 로 전처리하고, sent_pos/evidence를
넘겨주는 collate/predict/build_optim_sched/run_stage(train_full의 헬퍼)를 그대로
import해 재사용한다. 기존 파일은 한 줄도 수정하지 않는다.

메인(annotated/revised) 학습 단계에 두 가지를 얹는다:
  1. best-checkpoint tracking : dev_F1 최고 epoch에서만 체크포인트/예측 저장
  2. early stopping           : dev_F1이 --patience(기본 5) epoch 개선 없으면 중단

Re-DocRED(revised) 예 (baseline과 동일 데이터/레시피 + DREAM):

    python -m Scripts.atlop.train_dream_es \
      --train_split train_revised --dev_split dev_revised --distant_mode none \
      --epochs 30 --patience 5 --eval_batch_size 32 \
      --run_name atlop_dream_revised --save_model
    # -> results/atlop_dream_revised.pt              (best-epoch 가중치)
    # -> results/atlop_dream_revised_dev_predictions.json  (best-epoch 예측)

    # baseline에서 warm-start(evi_gate만 fresh, 시작점 ≈ baseline):
    #   ... --init_ckpt results/atlop.pt
"""

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_dataset import DocREDataset            # noqa: E402
from data.docred_io import build_rel2id, NUM_CLASSES     # noqa: E402
from Scripts.atlop.losses import ATLoss, PUATLoss         # noqa: E402
from Scripts.atlop.preprocess_full import build_features_full  # noqa: E402
from Scripts.atlop.re_model_dream import DocREModelDREAM  # noqa: E402
from Scripts.atlop.train_full import (                    # noqa: E402
    build_optim_sched, make_collate_fn, predict, run_stage,
)
from Scripts.atlop.train_re import RESULTS_DIR, build_argparser, set_seed  # noqa: E402
from Scripts.eval.scorer import evaluate                  # noqa: E402


def run_stage_best_es(model, loader, args, device, epochs, stage, dev_loader,
                      dev_docs, ign_docs, id2rel, patience, ckpt_path):
    """train_full.run_stage와 같은 흐름(sent_pos·evidence 전달) + best-checkpoint
    tracking + early stopping. dev_F1 기준 최고 epoch의 (metrics, preds)를 반환하고,
    ckpt_path가 주어지면 best 갱신 시마다 그 state_dict를 저장한다."""
    total_steps = max(1, len(loader) * epochs)
    optimizer, scheduler = build_optim_sched(model, args, total_steps)

    best = {"f1": -1.0, "epoch": -1, "metrics": None, "preds": None}
    no_improve = 0
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
        f1 = metrics["f1"]
        improved = f1 > best["f1"]
        tag = " *best*" if improved else f"  (no-improve {no_improve + 1}/{patience})"
        print(f"[{stage} | epoch {epoch}] train_loss={running / max(1, len(loader)):.4f} "
              f"dev_F1={f1 * 100:.2f} Ign_F1={metrics['ign_f1'] * 100:.2f} "
              f"(P={metrics['precision'] * 100:.2f} R={metrics['recall'] * 100:.2f}){tag}")

        if improved:
            best.update(f1=f1, epoch=epoch, metrics=metrics, preds=preds)
            no_improve = 0
            if ckpt_path is not None:
                torch.save(model.state_dict(), ckpt_path)
                print(f"  [best] checkpoint 갱신 -> {ckpt_path.name} (dev_F1={f1 * 100:.2f})")
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f"[early-stop] dev_F1 {patience} epoch 동안 개선 없음 -> 중단 "
                      f"(best dev_F1={best['f1'] * 100:.2f} @ epoch {best['epoch']})")
                break

    print(f"[best] dev_F1={best['f1'] * 100:.2f} / "
          f"Ign_F1={best['metrics']['ign_f1'] * 100:.2f} @ epoch {best['epoch']}")
    return best["metrics"], best["preds"]


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[device] {device}  [model] dream (baseline + evidence-guided; "
          f"+best-ckpt +early-stop patience={args.patience}, evi_lambda={args.evi_lambda})")

    if args.run_name == "atlop":
        args.run_name = "atlop_dream"
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
    print(f"[data] train={len(train_docs)} ({args.train_split}) "
          f"dev={len(dev_docs)} ({args.dev_split}) docs")

    dev_features = build_features_full(dev_docs, tokenizer, rel2id)
    dev_loader = DataLoader(dev_features, batch_size=args.eval_batch_size,
                            shuffle=False, collate_fn=collate)
    train_features = build_features_full(train_docs, tokenizer, rel2id)
    train_loader = DataLoader(train_features, batch_size=args.train_batch_size,
                              shuffle=True, collate_fn=collate)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=NUM_CLASSES)
    # eager attention: localized context pooling needs attention weights.
    encoder = AutoModel.from_pretrained(
        args.model_name_or_path, config=config, attn_implementation="eager"
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = DocREModelDREAM(config, encoder, emb_size=args.emb_size,
                            block_size=args.block_size, num_labels=NUM_CLASSES,
                            evi_lambda=args.evi_lambda).to(device)

    if args.init_ckpt:
        state = torch.load(args.init_ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init] warm-start from {args.init_ckpt} "
              f"(missing={len(missing)} fresh [evi_gate 등], unexpected={len(unexpected)})")

    RESULTS_DIR.mkdir(exist_ok=True)
    ckpt_path = RESULTS_DIR / f"{args.run_name}.pt" if args.save_model else None

    # (선택) distant 사전학습 — best/ES 없이 train_full.run_stage 그대로 (PU는 이 단계만)
    if args.distant_mode == "pretrain":
        cap = args.limit_docs if args.limit_docs > 0 else args.distant_limit
        distant_docs = list(DocREDataset(args.distant_split))
        if cap > 0:
            distant_docs = distant_docs[:cap]
        distant_features = build_features_full(distant_docs, tokenizer, rel2id)
        print(f"[stage 1] distant pretrain on {len(distant_docs)} docs "
              f"({args.distant_epochs} epoch(s))")
        if args.use_pu_loss:
            model.loss_fnt = PUATLoss(na_weight=args.na_weight)
            print(f"[stage 1] loss = PUATLoss(na_weight={args.na_weight})")
        distant_loader = DataLoader(distant_features, batch_size=args.distant_batch_size,
                                    shuffle=True, collate_fn=collate)
        run_stage(model, distant_loader, args, device, args.distant_epochs,
                  "distant-pretrain", dev_loader, dev_docs, train_docs, id2rel)
        del distant_features, distant_loader, distant_docs
        model.loss_fnt = ATLoss()  # annotated Na는 gold -> 표준 ATLoss
    elif args.distant_mode == "denoise":
        print("[warn] --distant_mode denoise 는 이 스크립트에서 미지원 -> 메인 단독 학습으로 진행")

    # 메인 학습 (annotated / revised) — best-checkpoint + early stopping
    stage = "annotated-finetune" if args.distant_mode == "pretrain" else "annotated-train"
    print(f"[main] {stage} on {len(train_docs)} docs "
          f"(최대 {args.epochs} epoch, patience {args.patience})")
    metrics, preds = run_stage_best_es(
        model, train_loader, args, device, args.epochs, stage,
        dev_loader, dev_docs, train_docs, id2rel, args.patience, ckpt_path)

    pred_path = RESULTS_DIR / f"{args.run_name}_dev_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as fp:
        json.dump(preds, fp, ensure_ascii=False)
    print(f"[saved] {pred_path}  (best-epoch 예측, {len(preds)} relations)")
    if ckpt_path is not None:
        print(f"[saved] {ckpt_path}  (best-epoch 체크포인트)")
    return metrics


def build_dream_argparser():
    """train_re.build_argparser + DREAM/best-ES 인자."""
    p = build_argparser()
    p.description = "Baseline + DREAM(evidence-guided) 모델 + best-checkpoint tracking + early stopping"
    p.add_argument("--evi_lambda", type=float, default=0.1,
                   help="evidence 지도학습 손실 가중치 (0=DREAM 지도 끔)")
    p.add_argument("--patience", type=int, default=5,
                   help="early-stopping patience (dev_F1 미개선 epoch 수, 0=끔)")
    p.add_argument("--init_ckpt", default="",
                   help="warm-start state_dict (예: results/atlop.pt, strict=False)")
    return p


if __name__ == "__main__":
    train(build_dream_argparser().parse_args())
