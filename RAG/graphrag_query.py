"""Neo4j 지식그래프에서 실제 multi-hop traversal로 질문에 답하는 GraphRAG 조회 스크립트.

RAG/demo_compare.py로 informationextraction(Pinecone, relation triple을 벡터로
쪼개서 top-k 검색)과 informationrag(naive RAG, 문장 단위)를 비교했더니, Pinecone
쪽은 실제 그래프 순회가 아니라 "잘게 쪼갠 naive RAG"에 불과해서 문서 내/문서 간
멀티홉 질문 다수를 못 풀었다. 이 스크립트는 대신 Neo4j에 직접 Cypher로 N-hop을
따라가 서브그래프를 가져오고, 그 사실들을 근거로 LLM이 답하게 한다 (진짜 그래프
추론).

파이프라인:
    1. LLM으로 질문에서 엔티티 멘션 추출 (예: "Cirit을 만드는 회사..." -> ["Cirit"])
    2. 멘션마다 entities.name/aliases 부분일치로 그래프의 seed 노드 탐색
    3. seed에서 최대 MAX_HOPS까지 양방향 관계를 펼쳐 서브그래프(사실 목록) 수집
    4. 서브그래프 사실만 근거로 LLM이 한국어로 답변 (근거 부족하면 '모름')

API 키/DB 자격증명은 .env(레포 루트)에서 읽는다. .env는 .gitignore에 등록되어 있음.

사용법:
    python RAG/graphrag_query.py
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CHAT_MODEL = "gpt-5.4-mini"
SEEDS_PER_MENTION = 3
MAX_HOPS = 3
SUBGRAPH_FACT_LIMIT_PER_HOP = 200
# hop 확장 시 이 값보다 연결이 많은 노드(국가/대륙 등 범용 개체)는 다음 hop의 확장
# 출발점에서 제외한다. 예: Cirit -> Turkey(연결 294개) -> ... 로 뻗으면 Cirit ->
# Roketsan(연결 13개) -> Ankara 같이 실제 필요한 경로가 무관한 사실 수백 개에 묻혀
# LIMIT 안에 못 들어간다.
HUB_DEGREE_THRESHOLD = 40

openai_client = OpenAI()
driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
)
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")


def extract_entities(question):
    """질문에서 그래프 탐색의 출발점이 될 엔티티 멘션을 LLM으로 추출."""
    prompt = (
        "다음 질문에서 지식그래프 탐색의 시작점이 될 고유명사 엔티티(회사/기관/인물/"
        "지명 등)만 JSON 배열로 추출하세요. 다른 설명 없이 배열만 출력하세요.\n"
        f"질문: {question}"
    )
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def find_seed_entities(session, mention):
    """엔티티 이름/별칭에 (대소문자 무시) 부분일치하는 노드를 찾는다."""
    query = """
    MATCH (e:ZEntity)
    WHERE toLower(e.name) CONTAINS toLower($mention)
       OR any(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($mention))
    RETURN e.id AS id, e.name AS name, e.type AS type
    LIMIT $limit
    """
    return list(session.run(query, mention=mention, limit=SEEDS_PER_MENTION))


def expand_subgraph(session, seed_ids):
    """seed 노드에서 MAX_HOPS까지 관계를 BFS로 가져온다.

    매 hop마다 그 hop에서 새로 만난 노드 중 허브 노드(연결이 HUB_DEGREE_THRESHOLD
    초과)는 다음 hop의 확장 출발점에서 제외한다 — 안 그러면 국가/대륙처럼 연결이
    수백 개인 노드를 지나가는 순간 무관한 사실이 쏟아져 정작 필요한 경로가
    LIMIT 안에 못 들어간다.
    """
    if not seed_ids:
        return []

    facts = []
    visited = set(seed_ids)
    frontier = seed_ids

    for _ in range(MAX_HOPS):
        rows = list(session.run(
            """
            MATCH (h:ZEntity)-[r]-(n:ZEntity)
            WHERE h.id IN $frontier
            RETURN DISTINCT startNode(r).name AS head, type(r) AS relation, endNode(r).name AS tail,
                   r.evidence AS evidence,
                   n.id AS next_id, COUNT { (n)--() } AS next_degree
            LIMIT $limit
            """,
            frontier=frontier, limit=SUBGRAPH_FACT_LIMIT_PER_HOP,
        ))
        if not rows:
            break

        # relation triple만으로는 DocRED의 고정된 96개 relation 스키마에 없는 세부
        # 정보(예: "어느 위원회가 세웠는지")를 놓친다. Neo4j 관계에 저장된 evidence
        # 문장을 같이 실어서, 구조화된 grpah 순회 + 원문 근거를 모두 활용한다.
        facts.extend({
            "head": r["head"], "relation": r["relation"], "tail": r["tail"],
            "evidence": " ".join(r["evidence"]) if r["evidence"] else None,
        } for r in rows)

        frontier = [
            r["next_id"] for r in rows
            if r["next_id"] not in visited and r["next_degree"] <= HUB_DEGREE_THRESHOLD
        ]
        visited.update(frontier)
        if not frontier:
            break

    return facts


def _format_fact(f):
    line = f"- {f['head']} -[{f['relation']}]-> {f['tail']}"
    if f.get("evidence"):
        line += f" (근거: {f['evidence']})"
    return line


def answer_with_subgraph(question, facts):
    if not facts:
        context_block = "(그래프에서 관련 사실을 찾지 못함)"
    else:
        context_block = "\n".join(_format_fact(f) for f in facts)

    prompt = (
        "아래는 지식그래프에서 가져온 사실(head -[relation]-> tail) 목록이고, 일부는 "
        "근거 문장이 같이 달려 있습니다. 이 사실들만 근거로 질문에 한국어로 답하세요. "
        "여러 사실을 연결해 추론해도 됩니다. 근거가 불충분하면 반드시 '모름'이라고만 "
        "답하세요.\n\n"
        f"사실 목록:\n{context_block}\n\n질문: {question}"
    )
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def answer_question(question, verbose=True):
    mentions = extract_entities(question)
    with driver.session(database=NEO4J_DATABASE) as session:
        seed_ids = []
        seed_names = []
        for mention in mentions:
            for row in find_seed_entities(session, mention):
                seed_ids.append(row["id"])
                seed_names.append(f"{row['name']} ({row['id']})")

        facts = expand_subgraph(session, seed_ids)

    answer = answer_with_subgraph(question, facts)

    if verbose:
        print(f"질문: {question}")
        print(f"  추출된 엔티티 멘션: {mentions}")
        print(f"  seed 노드: {seed_names}")
        print(f"  서브그래프 사실 수: {len(facts)}")
        print(f"  답변: {answer}")
        print()

    return answer, seed_names, facts


if __name__ == "__main__":
    demo_questions = [
        "Cirit을 만드는 회사의 본사는 어느 도시에 있어?",
        "Roketsan은 어느 나라 회사야?",
        "Health Sciences Centre가 있는 도시에 있는 대학교는 몇 년에 설립됐어?",
        "Outotec의 필터가 만들어지는 도시는 어느 지역에 속해있어?",
        "AirAsia Zest의 본사는 어디였어?",
    ]
    for q in demo_questions:
        answer_question(q)

    driver.close()
