"""2단계: 모델(ATLOP+DREAM)이 test_revised 문서에서 예측한 triple을 Neo4j에 적재.

1단계(load_ground_truth.py)가 만든 그래프에 얹는 방식이라 기존 노드/엣지를
삭제하지 않는다 (1단계의 `MATCH ()-[r]->() DELETE r` 같은 초기화 없음).
test_revised 문서는 1단계 대상(train_revised/dev_revised)과 겹치지 않으므로
엣지 MERGE 키(document)도 충돌하지 않는다.

개체 노드는 1단계와 같은 전역 병합 정책을 따른다 — 예측 파일의 entity id(E0 등)는
문서 내 vertexSet 인덱스이므로, 해당 클러스터의 canonical 이름/타입으로
`global_entity_id`를 만들어 기존 그래프의 동일 개체 노드에 그대로 병합되게 한다
(예측 파일의 name 필드는 별칭일 수 있어 그대로 쓰면 중복 노드가 생김).

evidence 보완: 예측 파일에 evidence가 비어있는 triple은 원본 test_revised 문서의
vertexSet mention 위치로 추론해서 채운다.
- head/tail이 같은 문장에 함께 언급되면 그 문장들 (`inferred_cooccurrence`)
- 함께 언급된 문장이 없으면(multi-hop) head/tail 각각이 언급된 문장의 합집합
  (`inferred_mention_union`)
- 파일에 이미 있던 evidence는 `model_provided`

엣지는 1단계와 같은 스키마에 `model`/`filter_band`/`filter_action` 속성이 추가되고,
`split`이 `_revised`로 끝나지 않으므로 `is_revised`는 False로 떨어진다 (1단계
ground truth와 구분하는 플래그 — docred_common.is_revised_split 참고).

사용법:
    python Scripts/kg/load_predictions.py <예측 JSON 경로> --dry-run
    python Scripts/kg/load_predictions.py <예측 JSON 경로>
    # evidence 채운 JSON만 뽑고 싶으면:
    python Scripts/kg/load_predictions.py <예측 JSON 경로> --dry-run \
        --write-filled results/..._evidence_filled.json
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from docred_common import (
    ROOT,
    cluster_canonical,
    global_entity_id,
    load_rel_info,
    load_split,
    sentence_text,
)
from load_ground_truth import (
    chunked,
    edge_merge_query,
    entity_merge_query,
    relation_type_name,
    ENTITY_LABEL,
)

SPLIT = "test_revised_pred"  # `_revised`로 끝나지 않음 → is_revised=False
MODEL = "atlop_dream"
DOC_SPLIT = "test_revised"  # 예측 대상 문서가 들어있는 DocRED split
BATCH_SIZE = 500


def vertex_index(entity_id):
    """예측 파일의 entity id("E12") -> vertexSet 인덱스(12)."""
    return int(entity_id.lstrip("E"))


def resolve_pred_evidence(triple, doc):
    """(sentence_ids, evidence_texts, evidence_source)를 반환.

    파일에 evidence_source가 이미 있으면(예: --write-filled로 한 번 채워둔 사본을
    재적재하는 경우) 그 출처를 그대로 신뢰한다 — 재계산하면 inferred_* 였던
    것도 evidence가 채워진 상태라 전부 model_provided로 잘못 뭉개진다.
    evidence_source가 없고 evidence만 있으면(원본 예측 파일) model_provided로
    간주하고, evidence도 없으면 mention 위치로 추론해서 채운다."""
    if triple.get("evidence_source"):
        return (
            triple["source"].get("sentence_id", []),
            triple["evidence"],
            triple["evidence_source"],
        )
    if triple.get("evidence"):
        return (
            triple["source"].get("sentence_id", []),
            triple["evidence"],
            "model_provided",
        )

    sents = doc["sents"]
    h_sents = {m["sent_id"] for m in doc["vertexSet"][vertex_index(triple["head"]["id"])]}
    t_sents = {m["sent_id"] for m in doc["vertexSet"][vertex_index(triple["tail"]["id"])]}

    cooccur = sorted(h_sents & t_sents)
    if cooccur:
        sent_ids, source = cooccur, "inferred_cooccurrence"
    else:
        sent_ids, source = sorted(h_sents | t_sents), "inferred_mention_union"

    texts = [sentence_text(sents[sid]) for sid in sent_ids if sid < len(sents)]
    return sent_ids, texts, source


def build_graph(preds, docs, rel_info):
    """entities: entity_id -> {name, type, aliases}
    edges: 1단계와 같은 스키마 + model/filter_band/filter_action."""
    entities = {}
    edges = []
    ev_source_counts = {}

    for triple in preds:
        doc = docs[triple["source"]["document_id"]]

        entity_ids = {}
        for role in ("head", "tail"):
            cluster = doc["vertexSet"][vertex_index(triple[role]["id"])]
            name, type_ = cluster_canonical(cluster)
            entity_id = global_entity_id(name, type_)
            ent = entities.setdefault(
                entity_id, {"name": name, "type": type_, "aliases": set()}
            )
            ent["aliases"].update(m["name"] for m in cluster)
            entity_ids[role] = entity_id

        relation_id = triple["relation"]["code"]
        sent_ids, texts, source = resolve_pred_evidence(triple, doc)
        ev_source_counts[source] = ev_source_counts.get(source, 0) + 1

        edges.append(
            {
                "head_id": entity_ids["head"],
                "tail_id": entity_ids["tail"],
                "relation_id": relation_id,
                "relation_name": rel_info.get(relation_id, triple["relation"]["name"]),
                "confidence": triple["confidence"],
                "split": SPLIT,
                "document": doc["title"],
                "sentence_id": sent_ids,
                "evidence": texts,
                "evidence_source": source,
                "is_revised": False,
                "model": MODEL,
                "filter_band": triple.get("filter", {}).get("band"),
                "filter_action": triple.get("filter", {}).get("action"),
            }
        )

    return entities, edges, ev_source_counts


def pred_edge_merge_query(type_name):
    """1단계 edge_merge_query와 같은 MERGE 키(document)에 예측 전용 속성 추가."""
    base = edge_merge_query(type_name)
    return base + ",\n    r.model = row.model,\n    r.filter_band = row.filter_band,\n    r.filter_action = row.filter_action\n"


def write_filled_json(preds, docs, path):
    """원본 예측 JSON에 evidence를 채워넣은 사본을 저장 (스키마 동일,
    evidence_source 필드만 추가)."""
    out = []
    for triple in preds:
        doc = docs[triple["source"]["document_id"]]
        sent_ids, texts, source = resolve_pred_evidence(triple, doc)
        filled = dict(triple)
        filled["source"] = dict(triple["source"], sentence_id=sent_ids)
        filled["evidence"] = texts
        filled["evidence_source"] = source
        out.append(filled)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"evidence 채운 JSON 저장: {path} ({len(out)}개 triple)")


def load_into_neo4j(entity_rows_by_type, edge_rows_by_type, batch_size):
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    driver.verify_connectivity()

    with driver.session(database=os.environ.get("NEO4J_DATABASE")) as session:
        total_entities = 0
        for type_label, rows in entity_rows_by_type.items():
            query = entity_merge_query(type_label)
            for batch in chunked(rows, batch_size):
                session.run(query, rows=batch)
            total_entities += len(rows)
        print(f"엔티티 적재 완료: {total_entities}개 ({len(entity_rows_by_type)}개 개체 타입)")

        total_edges = 0
        for type_name, rows in edge_rows_by_type.items():
            query = pred_edge_merge_query(type_name)
            for batch in chunked(rows, batch_size):
                session.run(query, rows=batch)
            total_edges += len(rows)
        print(f"관계 적재 완료: {total_edges}개 ({len(edge_rows_by_type)}개 관계 타입)")

    driver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="모델 예측 triple JSON 경로")
    parser.add_argument("--dry-run", action="store_true", help="Neo4j에 연결하지 않고 집계만 출력")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--write-filled",
        type=Path,
        help="evidence를 채운 예측 JSON 사본을 저장할 경로",
    )
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    with open(args.predictions, encoding="utf-8") as f:
        preds = json.load(f)
    docs = {d["title"]: d for d in load_split(DOC_SPLIT)}
    rel_info = load_rel_info()

    missing = {t["source"]["document_id"] for t in preds} - set(docs)
    if missing:
        raise SystemExit(f"{DOC_SPLIT}에 없는 문서 {len(missing)}개: {sorted(missing)[:5]} ...")

    entities, edges, ev_source_counts = build_graph(preds, docs, rel_info)

    entity_rows = {}
    for entity_id, ent in entities.items():
        row = {"id": entity_id, "name": ent["name"], "type": ent["type"], "aliases": sorted(ent["aliases"])}
        entity_rows.setdefault(ent["type"], []).append(row)
    edge_rows = {}
    for edge in edges:
        edge_rows.setdefault(relation_type_name(edge["relation_name"]), []).append(edge)

    print(f"예측 triple 수: {len(preds)} (문서 {len({e['document'] for e in edges})}개)")
    print(f"고유 개체 수 (전역 병합 후): {len(entities)} ({len(entity_rows)}개 개체 타입)")
    print(f"관계(엣지) 수: {len(edges)} ({len(edge_rows)}개 관계 타입)")
    print(f"evidence 출처별: {ev_source_counts}")

    if args.write_filled:
        write_filled_json(preds, docs, args.write_filled)

    if args.dry_run:
        print("--dry-run: Neo4j에 적재하지 않고 종료합니다.")
        return

    load_into_neo4j(entity_rows, edge_rows, args.batch_size)


if __name__ == "__main__":
    main()
