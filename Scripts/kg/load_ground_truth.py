"""1단계: train_annotated + dev의 사람 annotate 정답 triple을 Neo4j에 confidence=1.0으로 적재.

개체 노드는 (정규화된 이름, type) 기준으로 문서 간 전역 병합한다 (DocRED는 문서 간
entity linking을 제공하지 않으므로, 동일 표기+동일 type을 같은 개체로 취급하는
근사치임 — 동명이인 등은 잘못 병합될 수 있음).

관계는 rel_info.json의 relation_name을 슬러그화(UPPER_SNAKE_CASE)해 Neo4j
관계 타입 자체로 사용한다 (예: "country" -> :COUNTRY). 이렇게 해야 Neo4j
Browser/Bloom에서 별도 caption 설정 없이도 관계 이름이 바로 라벨로 보인다.

사용법:
    python Scripts/kg/load_ground_truth.py --dry-run   # DB 연결 없이 집계만 확인
    python Scripts/kg/load_ground_truth.py              # 실제 적재
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "docred_data" / "data"

SPLITS = ["train_annotated", "dev"]
BATCH_SIZE = 500


def normalize_name(name):
    return " ".join(name.split())


def relation_type_name(relation_name):
    """"country of citizenship" -> "COUNTRY_OF_CITIZENSHIP" (Neo4j 관계 타입용)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", relation_name).strip("_").upper()
    if not slug or slug[0].isdigit():
        slug = f"REL_{slug}"
    return slug


def load_split(name):
    with open(DATA_DIR / f"{name}.json", encoding="utf-8") as f:
        return json.load(f)


def load_rel_info():
    with open(DATA_DIR / "rel_info.json", encoding="utf-8") as f:
        return json.load(f)


def build_graph(splits, rel_info):
    """전체 split을 순회하며 전역 개체/관계 딕셔너리를 만든다."""
    entities = {}  # entity_id -> {"name": str, "type": str, "aliases": set}
    edges = {}  # (head_id, tail_id, relation_id) -> {"relation_name": str, "sources": set}

    for split in splits:
        docs = load_split(split)
        for doc in docs:
            title = doc["title"]
            vertex_to_entity_id = []

            for cluster in doc["vertexSet"]:
                names = [normalize_name(m["name"]) for m in cluster]
                types = [m["type"] for m in cluster]
                canonical_name = Counter(names).most_common(1)[0][0]
                entity_type = Counter(types).most_common(1)[0][0]
                entity_id = f"{canonical_name}::{entity_type}"

                ent = entities.setdefault(
                    entity_id,
                    {"name": canonical_name, "type": entity_type, "aliases": set()},
                )
                ent["aliases"].update(names)
                vertex_to_entity_id.append(entity_id)

            for label in doc.get("labels", []):
                relation_id = label["r"]
                head_id = vertex_to_entity_id[label["h"]]
                tail_id = vertex_to_entity_id[label["t"]]
                key = (head_id, tail_id, relation_id)

                edge = edges.setdefault(
                    key,
                    {
                        "relation_name": rel_info.get(relation_id, relation_id),
                        "sources": set(),
                    },
                )
                edge["sources"].add(f"{split}::{title}")

    return entities, edges


def to_entity_rows(entities):
    return [
        {
            "id": entity_id,
            "name": ent["name"],
            "type": ent["type"],
            "aliases": sorted(ent["aliases"]),
        }
        for entity_id, ent in entities.items()
    ]


def to_edge_rows(edges):
    """관계 타입(예: COUNTRY)별로 그룹지어 반환 — Cypher 관계 타입은 파라미터로
    넘길 수 없어 쿼리 문자열에 직접 넣어야 하므로, 타입별로 배치를 나눈다."""
    by_type = {}
    for (head_id, tail_id, relation_id), edge in edges.items():
        type_name = relation_type_name(edge["relation_name"])
        row = {
            "head_id": head_id,
            "tail_id": tail_id,
            "relation_id": relation_id,
            "relation_name": edge["relation_name"],
            "sources": sorted(edge["sources"]),
        }
        by_type.setdefault(type_name, []).append(row)
    return by_type


def chunked(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


ENTITY_MERGE_QUERY = """
UNWIND $rows AS row
MERGE (e:Entity {id: row.id})
SET e.name = row.name, e.type = row.type, e.aliases = row.aliases
"""

TYPE_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def edge_merge_query(type_name):
    if not TYPE_NAME_RE.match(type_name):
        raise ValueError(f"안전하지 않은 관계 타입 이름: {type_name!r}")
    return f"""
UNWIND $rows AS row
MATCH (h:Entity {{id: row.head_id}})
MATCH (t:Entity {{id: row.tail_id}})
MERGE (h)-[r:{type_name}]->(t)
ON CREATE SET
    r.relation_id = row.relation_id,
    r.relation_name = row.relation_name,
    r.confidence = 1.0,
    r.sources = row.sources
ON MATCH SET
    r.sources = r.sources + [x IN row.sources WHERE NOT x IN r.sources]
"""


def load_into_neo4j(entity_rows, edge_rows_by_type, batch_size):
    from neo4j import GraphDatabase

    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()

    with driver.session(database=database) as session:
        session.run(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS "
            "FOR (e:Entity) REQUIRE e.id IS UNIQUE"
        )

        for batch in chunked(entity_rows, batch_size):
            session.run(ENTITY_MERGE_QUERY, rows=batch)
        print(f"엔티티 적재 완료: {len(entity_rows)}개")

        # 구 스키마(:RELATION 단일 타입)로 적재된 엣지가 남아있으면 제거
        session.run("MATCH ()-[r:RELATION]->() DELETE r")

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

    total_edges = sum(len(rows) for rows in edge_rows.values())
    print(f"대상 split: {SPLITS}")
    print(f"고유 개체 수 (전역 병합 후): {len(entity_rows)}")
    print(f"고유 관계(triple) 수 (전역 병합 후): {total_edges} ({len(edge_rows)}개 관계 타입)")

    if args.dry_run:
        print("--dry-run: Neo4j에 적재하지 않고 종료합니다.")
        return

    load_into_neo4j(entity_rows, edge_rows, args.batch_size)


if __name__ == "__main__":
    main()
