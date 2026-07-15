"""[추론] DREAM(revised) 모델로 지정 split(기본 test_revised)에서 관계 triple 추출.

최종 채택 모델(re_model_dream.DocREModelDREAM, train_revised로 학습)의 체크포인트를
불러와, 각 문서의 모든 엔티티 쌍을 분류해 예측된 관계 triple을 뽑는다. 학습이 아니라
추론만 하므로 labels/evidence는 쓰지 않는다(test에 labels가 있어도 무시).

출력 (`--out`, 기본 results/<ckpt이름>_<split>_triples.json): 문서별 triple 리스트
  {"title", "head","h_idx", "relation","r", "tail","t_idx", "evidence_pred"?}
  - head/tail : vertexSet 엔티티의 대표 이름 / relation : P-code를 사람이 읽는 이름으로 해석

예:
    python -m Scripts.atlop.extract_triples \
      --ckpt results/atlop_dream_revised.pt --split test_revised \
      --model_name_or_path bert-base-cased \
      --out results/atlop_dream_revised_test_triples.json
    # (labels 있는 split이면 F1도 보고 싶을 때) --eval  붙이면 dev/test F1 계산
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.docred_dataset import DocREDataset            # noqa: E402
from data.docred_io import build_rel2id, load_rel_info, NUM_CLASSES  # noqa: E402
from Scripts.atlop.preprocess_full import build_features_full  # noqa: E402
from Scripts.atlop.re_model_dream import DocREModelDREAM  # noqa: E402
from Scripts.atlop.train_full import make_collate_fn, predict  # noqa: E402

RESULTS_DIR = ROOT / "results"


def entity_name(vertex_set, idx):
    """vertexSet 엔티티의 대표 이름 (가장 흔한 mention name)."""
    if idx < 0 or idx >= len(vertex_set) or not vertex_set[idx]:
        return f"<E{idx}>"
    names = [m.get("name", "") for m in vertex_set[idx] if m.get("name")]
    if not names:
        return f"<E{idx}>"
    # 가장 자주 등장한 이름을 대표로
    return max(set(names), key=names.count)


def build_argparser():
    p = argparse.ArgumentParser(description="DREAM 모델로 관계 triple 추출")
    p.add_argument("--ckpt", default="",
                   help="학습된 DocREModelDREAM 체크포인트 (예: results/atlop_dream_revised.pt). "
                        "비우면 미학습 모델 -> triple은 무의미(배선 확인용)")
    p.add_argument("--split", default="test_revised", help="추출 대상 split")
    p.add_argument("--model_name_or_path", default="bert-base-cased", help="인코더(로컬 경로 가능)")
    p.add_argument("--emb_size", type=int, default=768)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--limit_docs", type=int, default=0, help="문서 수 제한(0=전체)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="", help="출력 json 경로(비우면 자동 지정)")
    p.add_argument("--eval", action="store_true",
                   help="split에 gold labels가 있으면 F1/Ign F1도 계산(ign 필터 = --ign_split)")
    p.add_argument("--ign_split", default="train_revised", help="Ign F1의 train fact 필터 split")
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    print(f"[device] {device}  [split] {args.split}  [ckpt] {args.ckpt or '(none, 미학습)'}")

    rel2id = build_rel2id()
    id2rel = {v: k for k, v in rel2id.items()}
    rel_info = load_rel_info()  # P-code -> 사람이 읽는 관계 이름

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    collate = make_collate_fn(tokenizer.pad_token_id)

    docs = list(DocREDataset(args.split))
    if args.limit_docs > 0:
        docs = docs[: args.limit_docs]
    print(f"[data] {len(docs)} docs")
    features = build_features_full(docs, tokenizer, rel2id)
    loader = DataLoader(features, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)

    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=NUM_CLASSES)
    encoder = AutoModel.from_pretrained(
        args.model_name_or_path, config=config, attn_implementation="eager"
    )
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = DocREModelDREAM(config, encoder, emb_size=args.emb_size,
                            block_size=args.block_size, num_labels=NUM_CLASSES).to(device)

    if args.ckpt:
        state = torch.load(args.ckpt, map_location="cpu")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[ckpt] loaded {args.ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            print(f"[ckpt][warn] missing keys (학습값 없이 init됨): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    else:
        print("[ckpt][warn] 체크포인트 없음 -> 미학습 모델. 추출 triple은 의미 없음(배선 확인용).")

    # 예측: [{title, h_idx, t_idx, r(P-code)}]  (train_full.predict = threshold 위 관계)
    preds = predict(model, loader, id2rel, device)

    # 사람이 읽는 triple로 보강
    doc_by_title = {d["title"]: d for d in docs}
    triples = []
    for pr in preds:
        d = doc_by_title.get(pr["title"])
        vs = d["vertexSet"] if d else []
        rcode = pr["r"]
        triples.append({
            "title": pr["title"],
            "head": entity_name(vs, pr["h_idx"]), "h_idx": pr["h_idx"],
            "relation": rel_info.get(rcode, rcode), "r": rcode,
            "tail": entity_name(vs, pr["t_idx"]), "t_idx": pr["t_idx"],
        })

    out = Path(args.out) if args.out else RESULTS_DIR / f"{Path(args.ckpt).stem or 'dream'}_{args.split}_triples.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(triples, fp, ensure_ascii=False, indent=1)
    n_docs_with = len({t["title"] for t in triples})
    print(f"[saved] {out}  ({len(triples)} triples over {n_docs_with}/{len(docs)} docs)")
    for t in triples[:5]:
        print(f"  ({t['head']}) -[{t['relation']}]-> ({t['tail']})   [{t['title']}]")

    # (선택) gold labels가 있으면 F1도
    if args.eval:
        from Scripts.eval.scorer import evaluate  # noqa: E402
        if not docs[0].get("labels"):
            print("[eval] 이 split에 gold labels가 없어 F1 계산 생략")
        else:
            ign_docs = list(DocREDataset(args.ign_split))
            metrics = evaluate(preds, docs, ign_docs)
            print(f"[eval] {args.split}: F1={metrics['f1'] * 100:.2f} Ign_F1={metrics['ign_f1'] * 100:.2f} "
                  f"(P={metrics['precision'] * 100:.2f} R={metrics['recall'] * 100:.2f})")


if __name__ == "__main__":
    main()
