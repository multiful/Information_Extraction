"""1단계 Ground Truth triple을 요청된 JSON 스키마로 원본 DocRED에서 직접 추출.

Neo4j에는 (전역 병합된 개체 + 문서 목록)만 적재되어 있어 evidence 문장 텍스트가
없으므로, 이 스크립트는 원본 docred_data/data/*.json에서 문서 단위 raw triple을
그대로 뽑는다 (entity id = 문서 내 vertexSet 인덱스, 전역 병합 없음).

스키마:
{
  "head": {"id": "E<idx>", "name": str, "type": str},
  "relation": {"id": "P..", "name": str},
  "tail": {"id": "E<idx>", "name": str, "type": str},
  "confidence": 1.0,
  "source": {"document_id": str, "sentence_id": [int, ...]},
  "evidence": [str, ...],
  "evidence_source": "annotated" | "inferred_cooccurrence" | "unresolved_multihop"
}

DocRED 원본 라벨 중 일부(train_annotated 1,421개/38,180개, dev 487개/12,275개)는
evidence가 비어 있음. 이 경우 head/tail이 같은 문장에 함께 언급되는 문장이
있으면 그 문장(들)을 evidence로 추론해서 채우고 "inferred_cooccurrence"로
표시함. 함께 언급되는 문장이 아예 없으면(여러 문장에 걸친 multi-hop 추론이
필요한 경우) 억지로 채우지 않고 evidence를 비워둔 채 "unresolved_multihop"로
표시함 — 근거 없는 추론으로 데이터 정합성을 해치지 않기 위함.

사용법:
    python Scripts/kg/export_triples.py --splits train_annotated dev --out triples.jsonl
    python Scripts/kg/export_triples.py --preview 5   # 파일 저장 없이 미리보기만
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "docred_data" / "data"


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def load_rel_info():
    with open(DATA_DIR / "rel_info.json", encoding="utf-8") as f:
        return json.load(f)


def cluster_name_type(cluster):
    names = [m["name"] for m in cluster]
    types = [m["type"] for m in cluster]
    name = Counter(names).most_common(1)[0][0]
    type_ = Counter(types).most_common(1)[0][0]
    return name, type_


def sentence_text(sent_tokens):
    return " ".join(sent_tokens)


def build_records(splits, rel_info):
    for split in splits:
        docs = load_split(split)
        for doc in docs:
            title = doc["title"]
            sents = doc["sents"]

            vertex_meta = [cluster_name_type(c) for c in doc["vertexSet"]]
            mention_sents = [
                sorted(set(m["sent_id"] for m in cluster)) for cluster in doc["vertexSet"]
            ]

            for label in doc.get("labels", []):
                h_idx, t_idx = label["h"], label["t"]
                h_name, h_type = vertex_meta[h_idx]
                t_name, t_type = vertex_meta[t_idx]
                relation_id = label["r"]
                evidence_sent_ids = label.get("evidence", [])
                evidence_source = "annotated"

                if not evidence_sent_ids:
                    cooccur = sorted(set(mention_sents[h_idx]) & set(mention_sents[t_idx]))
                    if cooccur:
                        evidence_sent_ids = cooccur
                        evidence_source = "inferred_cooccurrence"
                    else:
                        evidence_source = "unresolved_multihop"

                yield {
                    "head": {"id": f"E{h_idx}", "name": h_name, "type": h_type},
                    "relation": {
                        "id": relation_id,
                        "name": rel_info.get(relation_id, relation_id),
                    },
                    "tail": {"id": f"E{t_idx}", "name": t_name, "type": t_type},
                    "confidence": 1.0,
                    "source": {"document_id": title, "sentence_id": evidence_sent_ids},
                    "evidence": [
                        sentence_text(sents[sid])
                        for sid in evidence_sent_ids
                        if sid < len(sents)
                    ],
                    "evidence_source": evidence_source,
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train_annotated", "dev"])
    parser.add_argument("--out", type=str, default=None, help="JSONL 저장 경로")
    parser.add_argument(
        "--preview", type=int, default=0, help="파일 저장 없이 N개만 stdout 미리보기"
    )
    args = parser.parse_args()

    rel_info = load_rel_info()
    records = build_records(args.splits, rel_info)

    if args.preview:
        for i, rec in enumerate(records):
            if i >= args.preview:
                break
            print(json.dumps(rec, ensure_ascii=False, indent=2))
        return

    out_path = Path(args.out) if args.out else ROOT / "triples.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n}개 triple을 {out_path}에 저장했습니다.")


if __name__ == "__main__":
    main()
