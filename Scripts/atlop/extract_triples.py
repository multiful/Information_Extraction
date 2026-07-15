"""[추론] DREAM(revised) 모델로 지정 split(기본 test_revised)에서 관계 triple 추출.

최종 채택 모델(re_model_dream.DocREModelDREAM, train_revised 학습)의 체크포인트를
불러와 각 문서의 엔티티 쌍을 분류하고, 예측된 관계 triple을 **KG/시각화용 리치
메타데이터** 형식으로 뽑는다.

출력 triple 1건 형식:
  {
    "head":     {"id":"E10", "name":"AirAsia Zest", "type":"ORG"},
    "relation": {"id":"R31", "name":"headquarters_location", "code":"P159"},
    "tail":     {"id":"E12", "name":"Pasay City", "type":"LOC"},
    "confidence": 0.97,                       # sigmoid(logit_rel - logit_TH)
    "source": {"document_id":"AirAsia Zest",  # Re-DocRED title
               "sentence_id":[0],
               "is_revised": true},           # gold에 있으면 True(인적정제), 없으면 False(모델추론)
    "evidence": ["...문장 원문..."]
  }

- head/tail.id = "E"+vertexSet 인덱스, .type = 엔티티 대표 타입
- relation.id  = "R"+rel2id(1..96), .name = P-code의 사람이 읽는 이름(snake_case), .code = 원본 P-code
- confidence   = ATLOP 적응형 임계값 대비 신뢰도
- is_revised   = 예측 triple이 gold(test_revised labels)에 있으면 True + gold evidence,
                 없으면 False + 모델 예측 evidence(p_evi 최상위 문장)

예:
    python -m Scripts.atlop.extract_triples \
      --ckpt results/atlop_dream_revised.pt --split test_revised \
      --model_name_or_path bert-base-cased --eval \
      --out results/atlop_dream_revised_test_revised_triples_v2.json
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
from Scripts.atlop.train_full import make_collate_fn      # noqa: E402

RESULTS_DIR = ROOT / "results"


def _repr_field(vertex_set, idx, field, default):
    """엔티티의 대표 name/type (가장 흔한 값)."""
    if idx < 0 or idx >= len(vertex_set) or not vertex_set[idx]:
        return default
    vals = [m.get(field) for m in vertex_set[idx] if m.get(field)]
    return max(set(vals), key=vals.count) if vals else default


def readable_rel(rel_info, pcode):
    """P-code -> 사람/LLM이 읽기 쉬운 snake_case 관계 이름."""
    name = rel_info.get(pcode, pcode)
    return name.strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


@torch.no_grad()
def infer_logits(model, loader, device):
    """모델 forward를 재현해 문서별 (logits, p_evi, hts)를 반환한다. logits로부터
    triple 예측 + confidence를, p_evi로부터 모델 예측 evidence를 얻기 위함.
    loader가 shuffle=False라 결과는 docs 순서와 일치한다."""
    model.eval()
    out = []
    for batch in loader:
        seq, att = model.encode(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        hs, rs, ts, evi_list = model.get_hrt_evidence(
            seq, att, batch["entity_pos"], batch["hts"], batch["sent_pos"])
        hs = torch.tanh(model.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(model.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, model.emb_size // model.block_size, model.block_size)
        b2 = ts.view(-1, model.emb_size // model.block_size, model.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, model.emb_size * model.block_size)
        logits = model.bilinear(bl).cpu()                 # (total_pairs, C)
        idx = 0
        for doc_hts, p_evi in zip(batch["hts"], evi_list):
            n = len(doc_hts)
            out.append((logits[idx: idx + n], p_evi.cpu(), doc_hts))
            idx += n
    return out


def build_argparser():
    p = argparse.ArgumentParser(description="DREAM 모델로 관계 triple(리치 메타데이터) 추출")
    p.add_argument("--ckpt", default="",
                   help="학습된 DocREModelDREAM 체크포인트 (비우면 미학습 -> 배선 확인용)")
    p.add_argument("--split", default="test_revised")
    p.add_argument("--model_name_or_path", default="bert-base-cased")
    p.add_argument("--emb_size", type=int, default=768)
    p.add_argument("--block_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--limit_docs", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="", help="출력 json (비우면 자동, _v2 접미)")
    p.add_argument("--eval", action="store_true", help="gold labels 있으면 F1/Ign F1도 계산")
    p.add_argument("--ign_split", default="train_revised", help="Ign F1 train fact 필터")
    return p


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)
    print(f"[device] {device}  [split] {args.split}  [ckpt] {args.ckpt or '(none)'}")

    rel2id = build_rel2id()
    id2rel = {v: k for k, v in rel2id.items()}
    rel_info = load_rel_info()

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
        args.model_name_or_path, config=config, attn_implementation="eager")
    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    model = DocREModelDREAM(config, encoder, emb_size=args.emb_size,
                            block_size=args.block_size, num_labels=NUM_CLASSES).to(device)
    if args.ckpt:
        missing, unexpected = model.load_state_dict(torch.load(args.ckpt, map_location="cpu"), strict=False)
        print(f"[ckpt] loaded {args.ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")
        if missing:
            print(f"[ckpt][warn] missing: {missing[:5]}")
    else:
        print("[ckpt][warn] 체크포인트 없음 -> 미학습 모델(triple 무의미, 배선 확인용)")

    doc_results = infer_logits(model, loader, device)

    triples = []
    eval_preds = []  # 공통 포맷(스코어러용): {title,h_idx,t_idx,r}
    for doc, (logits, p_evi, doc_hts) in zip(docs, doc_results):
        vs = doc["vertexSet"]
        sents = doc["sents"]
        title = doc["title"]
        # gold: (h,t,r_id) -> evidence 문장 집합
        gold = {}
        for lab in doc.get("labels", []):
            gold.setdefault((lab["h"], lab["t"], rel2id[lab["r"]]), set()).update(lab.get("evidence", []))

        for pi, (h, t) in enumerate(doc_hts):
            row = logits[pi]
            th = row[0].item()
            for r in range(1, NUM_CLASSES):
                if row[r].item() <= th:            # 임계값 아래 -> 예측 아님
                    continue
                pcode = id2rel[r]
                conf = torch.sigmoid(row[r] - row[0]).item()
                is_rev = (h, t, r) in gold
                if is_rev:
                    sent_ids = sorted(s for s in gold[(h, t, r)] if 0 <= s < len(sents))
                else:  # 모델 예측 evidence: p_evi 최상위 문장
                    sent_ids = [int(p_evi[pi].argmax())] if p_evi.numel() else []
                    sent_ids = [s for s in sent_ids if 0 <= s < len(sents)]
                triples.append({
                    "head": {"id": f"E{h}", "name": _repr_field(vs, h, "name", f"E{h}"),
                             "type": _repr_field(vs, h, "type", "MISC")},
                    "relation": {"id": f"R{r}", "name": readable_rel(rel_info, pcode), "code": pcode},
                    "tail": {"id": f"E{t}", "name": _repr_field(vs, t, "name", f"E{t}"),
                             "type": _repr_field(vs, t, "type", "MISC")},
                    "confidence": round(conf, 4),
                    "source": {"document_id": title, "sentence_id": sent_ids, "is_revised": is_rev},
                    "evidence": [" ".join(sents[s]) for s in sent_ids],
                })
                eval_preds.append({"title": title, "h_idx": h, "t_idx": t, "r": pcode})

    out = Path(args.out) if args.out else \
        RESULTS_DIR / f"{Path(args.ckpt).stem or 'dream'}_{args.split}_triples_v2.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as fp:
        json.dump(triples, fp, ensure_ascii=False, indent=1)
    n_rev = sum(1 for t in triples if t["source"]["is_revised"])
    print(f"[saved] {out}  ({len(triples)} triples | 인적정제일치 {n_rev} / 모델추론 {len(triples) - n_rev})")
    for t in triples[:5]:
        print(f"  ({t['head']['name']}) -[{t['relation']['name']}]-> ({t['tail']['name']}) "
              f"conf={t['confidence']}")

    if args.eval:
        from Scripts.eval.scorer import evaluate  # noqa: E402
        if not docs[0].get("labels"):
            print("[eval] gold labels 없음 -> F1 생략")
        else:
            ign_docs = list(DocREDataset(args.ign_split))
            m = evaluate(eval_preds, docs, ign_docs)
            print(f"[eval] {args.split}: F1={m['f1'] * 100:.2f} Ign_F1={m['ign_f1'] * 100:.2f} "
                  f"(P={m['precision'] * 100:.2f} R={m['recall'] * 100:.2f})")


if __name__ == "__main__":
    main()
