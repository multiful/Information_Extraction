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
  "source": {"document_id": str, "sentence_id": [int, ...], "is_revised": bool},
  "evidence": [str, ...],
  "evidence_source": "annotated" | "inferred_cooccurrence" | "unresolved_multihop"
}

evidence 보완 로직은 docred_common.resolve_evidence 참고. `source.is_revised`는
docred_common.is_revised_split 참고 — split 이름이 `_revised`로 끝나면 True.

기본 split은 train_revised/dev_revised(Re-DocRED 재정제본)이며, 원본
train_annotated/dev는 이걸로 완전히 대체됨 (같은 문서에 대한 상위호환 라벨이라
같이 적재하면 revised가 제거한 오라벨이 남는 문제가 있어 교체를 택함).

사용법:
    python Scripts/kg/export_triples.py --out triples.jsonl
    python Scripts/kg/export_triples.py --preview 5   # 파일 저장 없이 미리보기만
"""

import argparse
import json
from pathlib import Path

from docred_common import ROOT, is_revised_split, iter_doc_records, load_rel_info, resolve_evidence


def build_records(splits, rel_info):
    for split, doc, vertex_meta, mention_sents in iter_doc_records(splits):
        title = doc["title"]
        sents = doc["sents"]

        for label in doc.get("labels", []):
            h_idx, t_idx = label["h"], label["t"]
            h_name, h_type = vertex_meta[h_idx]
            t_name, t_type = vertex_meta[t_idx]
            relation_id = label["r"]

            evidence_sent_ids, evidence_texts, evidence_source = resolve_evidence(
                label, mention_sents[h_idx], mention_sents[t_idx], sents
            )

            yield {
                "head": {"id": f"E{h_idx}", "name": h_name, "type": h_type},
                "relation": {
                    "id": relation_id,
                    "name": rel_info.get(relation_id, relation_id),
                },
                "tail": {"id": f"E{t_idx}", "name": t_name, "type": t_type},
                "confidence": 1.0,
                "source": {
                    "document_id": title,
                    "sentence_id": evidence_sent_ids,
                    "is_revised": is_revised_split(split),
                },
                "evidence": evidence_texts,
                "evidence_source": evidence_source,
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train_revised", "dev_revised"])
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
