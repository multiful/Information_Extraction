"""Neo4j에 실제로 적재된 1단계 Ground Truth 그래프를 triples.csv 한 파일로 내보낸다.

Bloom/Browser 색상 표시 문제와 무관하게 Excel/pandas 등에서 바로 열어볼 수
있도록, DB에 있는 그대로(head/relation/tail + confidence + 출처)를 한 줄에
triple 하나씩 뽑는다.

사용법:
    python Scripts/kg/export_csv.py
"""

import csv
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent.parent


def main():
    from neo4j import GraphDatabase

    load_dotenv(ROOT / ".env")
    uri = os.environ["NEO4J_URI"]
    username = os.environ["NEO4J_USERNAME"]
    password = os.environ["NEO4J_PASSWORD"]
    database = os.environ.get("NEO4J_DATABASE")

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()

    out_path = ROOT / "triples.csv"

    query = """
    MATCH (h:ZEntity)-[r]->(t:ZEntity)
    RETURN
        h.id AS head_id, h.name AS head_name, h.type AS head_type,
        r.relation_id AS relation_id, r.relation_name AS relation_name, type(r) AS relation_type,
        t.id AS tail_id, t.name AS tail_name, t.type AS tail_type,
        r.confidence AS confidence, r.sources AS sources
    """

    with driver.session(database=database) as session:
        result = session.run(query)
        with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "head_id", "head_name", "head_type",
                    "relation_id", "relation_name", "relation_type",
                    "tail_id", "tail_name", "tail_type",
                    "confidence", "sources",
                ]
            )
            n = 0
            for record in result:
                writer.writerow(
                    [
                        record["head_id"], record["head_name"], record["head_type"],
                        record["relation_id"], record["relation_name"], record["relation_type"],
                        record["tail_id"], record["tail_name"], record["tail_type"],
                        record["confidence"], "; ".join(record["sources"]),
                    ]
                )
                n += 1

    driver.close()
    print(f"triples.csv: {n}개 -> {out_path}")


if __name__ == "__main__":
    main()
