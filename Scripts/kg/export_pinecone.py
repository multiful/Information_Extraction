"""Neo4j에 적재된 1단계 Ground Truth 그래프를 Pinecone upsert용 JSONL로 내보낸다.

임베딩은 이 스크립트가 직접 만들지 않는다 (어떤 임베딩 모델/API 키를 쓸지는
사용자 쪽에서 결정할 부분). 대신 Pinecone에 upsert하기 딱 맞는 형태로
{id, text, metadata}를 한 줄씩 뽑아두면, 사용자가 text를 임베딩해서
그대로 index.upsert(vectors=[{"id":.., "values":embedding, "metadata":..}, ...])
하면 된다.

- text: evidence 문장을 이어붙인 것. evidence가 비어있는 경우
  (evidence_source == "unresolved_multihop")는 "{head} {relation} {tail}"
  형태로 대체 문장을 만들어 항상 임베딩 가능한 text가 있도록 함.
- metadata: Pinecone 메타데이터 제약(문자열/숫자/불리언/문자열 리스트만 허용,
  중첩 객체 불가)에 맞춰 평평하게 구성. sentence_id는 문자열 리스트로 변환.

사용법:
    python Scripts/kg/export_pinecone.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent


def build_text(head_name, relation_name, tail_name, evidence, evidence_source):
    if evidence:
        return " ".join(evidence)
    return f"{head_name} {relation_name} {tail_name}"


def main():
    from neo4j import GraphDatabase

    load_dotenv(ROOT / ".env")
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()

    out_path = ROOT / "pinecone_upsert.jsonl"

    query = """
    MATCH (h:ZEntity)-[r]->(t:ZEntity)
    RETURN
        h.id AS head_id, h.name AS head_name, h.type AS head_type,
        r.relation_id AS relation_id, r.relation_name AS relation_name,
        t.id AS tail_id, t.name AS tail_name, t.type AS tail_type,
        r.confidence AS confidence, r.split AS split, r.document AS document,
        r.sentence_id AS sentence_id, r.evidence AS evidence, r.evidence_source AS evidence_source,
        r.is_revised AS is_revised
    """

    n = 0
    with driver.session(database=database) as session:
        result = session.run(query)
        with open(out_path, "w", encoding="utf-8") as f:
            for record in result:
                vector_id = (
                    f"{record['split']}::{record['document']}::"
                    f"{record['head_id']}::{record['relation_id']}::{record['tail_id']}"
                )
                text = build_text(
                    record["head_name"],
                    record["relation_name"],
                    record["tail_name"],
                    record["evidence"],
                    record["evidence_source"],
                )
                row = {
                    "id": vector_id,
                    "text": text,
                    "metadata": {
                        "head_id": record["head_id"],
                        "head_name": record["head_name"],
                        "head_type": record["head_type"],
                        "relation_id": record["relation_id"],
                        "relation_name": record["relation_name"],
                        "tail_id": record["tail_id"],
                        "tail_name": record["tail_name"],
                        "tail_type": record["tail_type"],
                        "confidence": record["confidence"],
                        "split": record["split"],
                        "document": record["document"],
                        "sentence_id": [str(s) for s in record["sentence_id"]],
                        "evidence_source": record["evidence_source"],
                        "is_revised": record["is_revised"],
                    },
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1

    driver.close()
    print(f"pinecone_upsert.jsonl: {n}개 -> {out_path}")


if __name__ == "__main__":
    main()
