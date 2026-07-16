"""1단계: train_revised + dev_revised(Re-DocRED 재정제본)의 사람 검증 정답
triple을 Neo4j에 적재.

원래는 원본 DocRED의 train_annotated + dev를 썼으나, Re-DocRED가 같은 3,053개
문서에 대해 누락된 관계를 보강하고 오라벨을 정리한 상위호환 재정제본을 내놓으면서
이걸로 완전히 교체함 (둘을 같이 적재하면 revised가 지운 오라벨이 그래프에
남는 문제가 있어 병행 적재 대신 교체를 택함). dev는 998개 문서에서 500개로
줄어드는 대신(나머지 498개는 test_revised로 분리됨, 이번 적재 대상 아님)
문서당 관계 밀도는 원본보다 훨씬 높아짐.

각 엣지는 `is_revised` 속성을 갖는다 (split이 `_revised`로 끝나면 True —
docred_common.is_revised_split 참고). 지금은 SPLITS 전부가 revised라 항상
True지만, 향후 2단계(모델 예측 triple, 낮은 confidence)가 추가되면 그쪽은
False로 자연히 구분됨.

개체 노드는 (정규화된 이름, type) 기준으로 문서 간 전역 병합한다 (DocRED는 문서 간
entity linking을 제공하지 않으므로, 동일 표기+동일 type을 같은 개체로 취급하는
근사치임 — 동명이인 등은 잘못 병합될 수 있음).

관계 엣지는 반대로 문서 간에 병합하지 않는다 — 같은 (head, tail, relation)
triple이 여러 문서에서 나오면 문서 수만큼 별도 엣지를 만든다. 각 엣지가
confidence/document/sentence_id/evidence/evidence_source를 온전한 속성으로
가지도록 하기 위함 (LangGraph 등에서 "이 관계의 근거는?"을 물었을 때 문서별로
바로 꺼내 쓸 수 있게). evidence 보완 로직은 docred_common.resolve_evidence 참고.

관계는 rel_info.json의 relation_name을 슬러그화(UPPER_SNAKE_CASE)해 Neo4j
관계 타입 자체로 사용한다 (예: "country" -> :COUNTRY). 이렇게 해야 Neo4j
Browser/Bloom에서 별도 caption 설정 없이도 관계 이름이 바로 라벨로 보인다.

개체 노드도 마찬가지로 공통 라벨(`:ZEntity`)에 더해 DocRED type(PER/ORG/LOC/TIME/NUM/MISC)을
보조 라벨로 추가한다 (예: `(:ZEntity:PER)`). Bloom/Browser는 라벨 기준으로 노드를
분류·색칠하는데, 다중 라벨 노드는 알파벳순으로 정렬된 첫 라벨(주로 공통 라벨)을
기준으로 스타일을 고르는 것으로 보여, 공통 라벨 이름을 `Entity`가 아니라 `ZEntity`로
지어서 PER/LOC/... 보다 항상 알파벳순으로 뒤에 오도록 함 (타입 라벨이 먼저 오게).

사용법:
    python Scripts/kg/load_ground_truth.py --dry-run   # DB 연결 없이 집계만 확인
    python Scripts/kg/load_ground_truth.py              # 실제 적재
"""

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from docred_common import (
    ROOT,
    global_entity_id,
    is_revised_split,
    iter_doc_records,
    load_rel_info,
    resolve_evidence,
)

SPLITS = ["train_revised", "dev_revised"]
BATCH_SIZE = 500
ENTITY_LABEL = "ZEntity"


def relation_type_name(relation_name):
    """"country of citizenship" -> "COUNTRY_OF_CITIZENSHIP" (Neo4j 관계 타입용)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", relation_name).strip("_").upper()
    if not slug or slug[0].isdigit():
        slug = f"REL_{slug}"
    return slug


def build_graph(splits, rel_info):
    """entities: entity_id -> {name, type, aliases}
    edges: (head_id, tail_id, relation_id, document)별로 하나씩, 문서 간 병합 없음."""
    entities = {}
    edges = []

    for split, doc, vertex_meta, mention_sents, sent_entities in iter_doc_records(splits):
        title = doc["title"]
        sents = doc["sents"]
        vertex_to_entity_id = []

        for i, cluster in enumerate(doc["vertexSet"]):
            name, type_ = vertex_meta[i]
            entity_id = global_entity_id(name, type_)
            ent = entities.setdefault(
                entity_id, {"name": name, "type": type_, "aliases": set()}
            )
            ent["aliases"].update(m["name"] for m in cluster)
            vertex_to_entity_id.append(entity_id)

        for label in doc.get("labels", []):
            h_idx, t_idx = label["h"], label["t"]
            relation_id = label["r"]
            evidence_sent_ids, evidence_texts, evidence_source = resolve_evidence(
                label, h_idx, t_idx, mention_sents, sent_entities, sents
            )

            edges.append(
                {
                    "head_id": vertex_to_entity_id[h_idx],
                    "tail_id": vertex_to_entity_id[t_idx],
                    "relation_id": relation_id,
                    "relation_name": rel_info.get(relation_id, relation_id),
                    "confidence": 1.0,
                    "split": split,
                    "document": title,
                    "sentence_id": evidence_sent_ids,
                    "evidence": evidence_texts,
                    "evidence_source": evidence_source,
                    "is_revised": is_revised_split(split),
                }
            )

    return entities, edges


def to_entity_rows(entities):
    """DocRED type(예: PER)별로 그룹지어 반환 — 보조 라벨은 관계 타입과 마찬가지로
    쿼리 문자열에 직접 넣어야 하므로 타입별로 배치를 나눈다."""
    by_type = {}
    for entity_id, ent in entities.items():
        row = {
            "id": entity_id,
            "name": ent["name"],
            "type": ent["type"],
            "aliases": sorted(ent["aliases"]),
        }
        by_type.setdefault(ent["type"], []).append(row)
    return by_type


def to_edge_rows(edges):
    """관계 타입(예: COUNTRY)별로 그룹지어 반환 — Cypher 관계 타입은 파라미터로
    넘길 수 없어 쿼리 문자열에 직접 넣어야 하므로, 타입별로 배치를 나눈다."""
    by_type = {}
    for edge in edges:
        type_name = relation_type_name(edge["relation_name"])
        by_type.setdefault(type_name, []).append(edge)
    return by_type


def chunked(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


TYPE_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def entity_merge_query(type_label):
    if not TYPE_NAME_RE.match(type_label):
        raise ValueError(f"안전하지 않은 개체 타입 라벨: {type_label!r}")
    return f"""
UNWIND $rows AS row
MERGE (e:{ENTITY_LABEL} {{id: row.id}})
SET e.name = row.name, e.type = row.type, e.aliases = row.aliases
SET e:{type_label}
"""


def edge_merge_query(type_name):
    """document를 MERGE 매칭 키에 포함시켜 문서별로 별도 엣지를 만든다
    (재실행해도 같은 (head, tail, type, document) 조합은 중복 생성되지 않음)."""
    if not TYPE_NAME_RE.match(type_name):
        raise ValueError(f"안전하지 않은 관계 타입 이름: {type_name!r}")
    return f"""
UNWIND $rows AS row
MATCH (h:{ENTITY_LABEL} {{id: row.head_id}})
MATCH (t:{ENTITY_LABEL} {{id: row.tail_id}})
MERGE (h)-[r:{type_name} {{document: row.document}}]->(t)
SET
    r.relation_id = row.relation_id,
    r.relation_name = row.relation_name,
    r.confidence = row.confidence,
    r.split = row.split,
    r.sentence_id = row.sentence_id,
    r.evidence = row.evidence,
    r.evidence_source = row.evidence_source,
    r.is_revised = row.is_revised
"""


def load_into_neo4j(entity_rows_by_type, edge_rows_by_type, batch_size):
    from neo4j import GraphDatabase

    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()

    with driver.session(database=database) as session:
        # 구 라벨(:Entity)로 적재된 노드가 있으면 새 공통 라벨(:ZEntity)로 이전
        session.run(f"MATCH (e:Entity) WHERE NOT e:{ENTITY_LABEL} SET e:{ENTITY_LABEL} REMOVE e:Entity")
        session.run("DROP CONSTRAINT entity_id_unique IF EXISTS")
        session.run(
            f"CREATE CONSTRAINT {ENTITY_LABEL.lower()}_id_unique IF NOT EXISTS "
            f"FOR (e:{ENTITY_LABEL}) REQUIRE e.id IS UNIQUE"
        )
        # 앱 seed 조회용 fulltext 인덱스 (name/aliases). toLower CONTAINS 풀스캔을
        # 인덱스 조회로 대체한다 (data/streamlit/main.py find_seed_entities).
        session.run(
            f"CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS "
            f"FOR (e:{ENTITY_LABEL}) ON EACH [e.name, e.aliases]"
        )

        total_entities = 0
        for type_label, rows in entity_rows_by_type.items():
            query = entity_merge_query(type_label)
            for batch in chunked(rows, batch_size):
                session.run(query, rows=batch)
            total_entities += len(rows)
        print(f"엔티티 적재 완료: {total_entities}개 ({len(entity_rows_by_type)}개 개체 타입)")

        # 구 스키마(문서 간 병합된 엣지, :RELATION 단일 타입 등)가 남아있으면 전부 제거하고 재적재
        session.run("MATCH ()-[r]->() DELETE r")

        total_edges = 0
        for type_name, rows in edge_rows_by_type.items():
            query = edge_merge_query(type_name)
            for batch in chunked(rows, batch_size):
                session.run(query, rows=batch)
            total_edges += len(rows)
        print(f"관계 적재 완료: {total_edges}개 ({len(edge_rows_by_type)}개 관계 타입)")

    driver.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Neo4j에 연결하지 않고 집계 결과만 출력",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")

    rel_info = load_rel_info()
    entities, edges = build_graph(SPLITS, rel_info)
    entity_rows = to_entity_rows(entities)
    edge_rows = to_edge_rows(edges)

    total_entities = sum(len(rows) for rows in entity_rows.values())
    total_edges = sum(len(rows) for rows in edge_rows.values())
    print(f"대상 split: {SPLITS}")
    print(f"고유 개체 수 (전역 병합 후): {total_entities} ({len(entity_rows)}개 개체 타입)")
    print(f"관계(엣지) 수 (문서별 비병합): {total_edges} ({len(edge_rows)}개 관계 타입)")

    if args.dry_run:
        print("--dry-run: Neo4j에 적재하지 않고 종료합니다.")
        return

    load_into_neo4j(entity_rows, edge_rows, args.batch_size)


if __name__ == "__main__":
    main()
