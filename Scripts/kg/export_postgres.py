"""Neo4j에 적재된 1단계 Ground Truth 그래프를 PostgreSQL(Supabase)용
schema.sql + entities.csv + relations.csv로 내보낸다.

- entities: 개체 노드 (id, name, type, aliases). aliases는 JSONB(문자열 배열)로 저장.
- relations: 관계 엣지, 문서별로 병합하지 않은 그대로 (head_id/tail_id는
  entities.id를 참조하는 FK). sentence_id/evidence도 JSONB로 저장.

사용법:
    python Scripts/kg/export_postgres.py

적재 방법 (Supabase):
    1. Supabase SQL Editor에서 schema.sql 실행 (테이블 생성)
    2. Table Editor에서 entities.csv -> entities 테이블, relations.csv ->
       relations 테이블 순서로 Import (entities가 relations의 FK 대상이라 먼저)
    또는 psql:
        psql "$SUPABASE_DB_URL" -f schema.sql
        psql "$SUPABASE_DB_URL" -c "\\copy entities FROM 'entities.csv' WITH (FORMAT csv, HEADER true)"
        psql "$SUPABASE_DB_URL" -c "\\copy relations FROM 'relations.csv' WITH (FORMAT csv, HEADER true)"
"""

import csv
import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    aliases JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS relations (
    id BIGSERIAL PRIMARY KEY,
    head_id TEXT NOT NULL REFERENCES entities(id),
    tail_id TEXT NOT NULL REFERENCES entities(id),
    relation_id TEXT NOT NULL,
    relation_name TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    split TEXT NOT NULL,
    document TEXT NOT NULL,
    sentence_id JSONB NOT NULL,
    evidence JSONB NOT NULL,
    evidence_source TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_relations_head_id ON relations(head_id);
CREATE INDEX IF NOT EXISTS idx_relations_tail_id ON relations(tail_id);
CREATE INDEX IF NOT EXISTS idx_relations_relation_id ON relations(relation_id);
CREATE INDEX IF NOT EXISTS idx_relations_confidence ON relations(confidence);
""".strip()


def main():
    from neo4j import GraphDatabase

    load_dotenv(ROOT / ".env")
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()

    schema_path = ROOT / "schema.sql"
    entities_path = ROOT / "entities.csv"
    relations_path = ROOT / "relations.csv"

    schema_path.write_text(SCHEMA_SQL + "\n", encoding="utf-8")

    with driver.session(database=database) as session:
        result = session.run("MATCH (e:ZEntity) RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases")
        n_entities = 0
        with open(entities_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "name", "type", "aliases"])
            for record in result:
                writer.writerow(
                    [
                        record["id"],
                        record["name"],
                        record["type"],
                        json.dumps(record["aliases"], ensure_ascii=False),
                    ]
                )
                n_entities += 1

        result = session.run(
            """
            MATCH (h:ZEntity)-[r]->(t:ZEntity)
            RETURN
                h.id AS head_id, t.id AS tail_id,
                r.relation_id AS relation_id, r.relation_name AS relation_name,
                r.confidence AS confidence, r.split AS split, r.document AS document,
                r.sentence_id AS sentence_id, r.evidence AS evidence, r.evidence_source AS evidence_source
            """
        )
        n_relations = 0
        with open(relations_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "head_id", "tail_id", "relation_id", "relation_name",
                    "confidence", "split", "document",
                    "sentence_id", "evidence", "evidence_source",
                ]
            )
            for record in result:
                writer.writerow(
                    [
                        record["head_id"], record["tail_id"],
                        record["relation_id"], record["relation_name"],
                        record["confidence"], record["split"], record["document"],
                        json.dumps(record["sentence_id"], ensure_ascii=False),
                        json.dumps(record["evidence"], ensure_ascii=False),
                        record["evidence_source"],
                    ]
                )
                n_relations += 1

    driver.close()
    print(f"schema.sql -> {schema_path}")
    print(f"entities.csv: {n_entities}개 -> {entities_path}")
    print(f"relations.csv: {n_relations}개 -> {relations_path}")


if __name__ == "__main__":
    main()
