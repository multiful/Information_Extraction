"""베이스라인 GraphRAG vs 고도화 GraphRAG vs naive RAG 성능 비교 평가.

비교 대상 3개 시스템:
    - baseline : RAG/graphrag_query.py (이 세션의 고도화 작업 이전 원본 -- CONTAINS
      부분일치 엔티티 링킹, MAX_HOPS=3, relation-aware 라우팅/리랭킹/다수결/evidence
      폴백 없음)
    - advanced : GraphRAG/graphrag_query.py (현재 버전 -- Exact/Alias/Word Boundary/
      Fuzzy 캐스케이드, relation-aware 라우팅, 임베딩 리랭킹, 3회 샘플링+다수결,
      Graph-grounded Evidence Retrieval 폴백)
    - naive_rag: Pinecone informationrag 인덱스(문장 단위 청크), RAG/demo_compare.py와
      동일한 방식(질문 영어 번역 -> 임베딩 -> top-5 검색 -> LLM 답변)

지표:
    검색(retrieval) -- Precision/Recall/F1/nDCG/MAP: 각 시스템이 답변 생성 직전에
    실제로 사용한 컨텍스트 집합(GraphRAG는 리랭킹 후 fact 리스트, naive RAG는 top-5
    청크)을 대상으로, RAG/demo_compare.py의 QUERIES 중 정답(expected)이 있는 질문만
    사용(집계/요약형 질문은 "정답 집합"을 명확히 정의할 수 없어 제외). gold(정답
    근거)는 GOLD_TRIPLES에 (head, relation, tail)을 사람이 직접 지정 -- 처음엔 seed
    엔티티에서 자동으로 tail 이름이 기대 답변 문자열에 포함되는 엣지를 추출하는
    방식을 시도했으나, 실측으로 두 가지 실패 사례를 발견해 폐기함: (1) 허브 노드
    가지치기 없이 2-hop을 펼치면 "Turkey"처럼 연결이 수백 개인 노드 하나 때문에
    무관한 엣지 298개가 gold로 오염됨, (2) Cypher 무방향 매치(`-[r]-`)에서
    startNode/endNode가 실제 탐색 방향과 반대로 나올 수 있어 tail만 확인하면
    head 쪽에 답이 있는 엣지를 놓침(예: "Ninoy Aquino International Airport
    -[LOCATED_IN...]-> Pasay City"). 소규모(9문항) 평가라 자동화보다 사람이
    Neo4j에 직접 질의해 검증한 고정 triple을 쓰는 게 더 신뢰할 수 있다고 판단.

    생성(generation) -- 충실도/Context Recall/Context Precision/Answer Relevance:
    RAGAS 표준 정의를 참고해 자체 LLM-judge 프롬프트로 근사 계산(질문당 4개 지표를
    1번의 통합 JSON 호출로 묶어 비용 절감 -- ragas 패키지의 다단계 알고리즘을 그대로
    재현한 것은 아니라 근사치이며, 보고서에는 "custom LLM-judge, RAGAS 정의 준용"으로
    기술할 것).

사용법:
    python GraphRAG/eval_compare.py --resolve-only   # gold fact 자동 추출 결과만 점검(LLM 호출 없음)
    python GraphRAG/eval_compare.py                  # 전체 비교 실행(LLM 호출 발생)
"""

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "GraphRAG"))  # advanced 모듈의 `from cache import ...` 상대 import 해결용
sys.path.insert(0, str(ROOT / "RAG"))       # naive RAG 함수 재사용용

from cache import cache_key, cached  # noqa: E402  (sys.path 조작 이후 import)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


baseline_gq = _load_module(ROOT / "RAG" / "graphrag_query.py", "baseline_gq")
advanced_gq = _load_module(ROOT / "GraphRAG" / "graphrag_query.py", "advanced_gq")
import demo_compare as naive_mod  # noqa: E402  (RAG/demo_compare.py -- embed_queries/search_naive/answer_with_context 재사용)

CHAT_MODEL = advanced_gq.CHAT_MODEL
TOP_K_NAIVE = naive_mod.TOP_K

EVAL_QUESTIONS = [q for q in naive_mod.QUERIES if q[3] is not None]

# 질문별 gold (head, relation, tail) -- Neo4j에 직접 질의해 사람이 확인한 고정
# triple(위 모듈 docstring 참고). Roketsan의 "어느 위원회가 세웠는지"처럼 96개
# 관계 스키마에 없는 세부 정보는 INCEPTION 엣지의 evidence 텍스트 안에 이미
# 포함돼 있어(problem 5, 스키마 밖 사실) 별도 gold triple이 필요 없음. AirAsia
# Zest 합병처럼 아예 구조화된 relation이 없는 사실은 실제로 병합 정보를 담고
# 있는 엣지(HEADQUARTERS_LOCATION/COUNTRY, evidence 텍스트에 "Asian Spirit"/
# "merged" 포함 확인됨)를 그대로 gold로 씀.
GOLD_TRIPLES = {
    "Roketsan은 어느 나라 회사야?": [
        ("Roketsan", "COUNTRY", "Turkey"),
        ("Roketsan", "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", "Turkey"),
    ],
    "University of Manitoba는 몇 년에 설립됐어?": [
        ("University of Manitoba", "INCEPTION", "1877"),
    ],
    "Lappeenranta는 어느 나라 도시야?": [
        ("Lappeenranta", "COUNTRY", "Finland"),
        ("Lappeenranta", "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", "Finland"),
    ],
    "AirAsia Zest의 본사는 어디였어?": [
        ("AirAsia Zest", "HEADQUARTERS_LOCATION", "Pasay City"),
    ],
    "Cirit을 만드는 회사의 본사는 어느 도시에 있어?": [
        ("Cirit", "MANUFACTURER", "Roketsan"),
        ("Cirit", "DEVELOPER", "Roketsan"),
        ("Roketsan", "HEADQUARTERS_LOCATION", "Ankara"),
    ],
    "Roketsan은 몇 년에 설립됐고, 어느 위원회가 세웠어?": [
        ("Roketsan", "INCEPTION", "1988"),
    ],
    "AirAsia Zest는 어느 항공사들이 합쳐져 만들어졌고, 언제 그렇게 됐어?": [
        ("AirAsia Zest", "HEADQUARTERS_LOCATION", "Pasay City"),
        ("AirAsia Zest", "COUNTRY", "Philippines"),
    ],
    "Health Sciences Centre가 있는 도시에 있는 대학교는 몇 년에 설립됐어?": [
        ("Health Sciences Centre", "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", "Winnipeg"),
        ("University of Manitoba", "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", "Winnipeg"),
        ("University of Manitoba", "INCEPTION", "1877"),
    ],
    "Outotec의 필터가 만들어지는 도시는 어느 지역에 속해있어?": [
        ("Outotec", "HEADQUARTERS_LOCATION", "Lappeenranta"),
        ("Lappeenranta", "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", "South Karelia"),
    ],
}


def resolve_gold_facts(session, question, expected_answer):
    """GOLD_TRIPLES에 지정된 (head, relation, tail)마다 Neo4j에서 실제 엣지를 찾아
    evidence 텍스트를 채워 반환(naive RAG 관련성 판정에 evidence 텍스트가 필요)."""
    gold = []
    for head, relation, tail in GOLD_TRIPLES[question]:
        rows = list(session.run(
            f"""
            MATCH (h:ZEntity {{name: $head}})-[r:{relation}]-(t:ZEntity {{name: $tail}})
            RETURN r.evidence AS evidence
            LIMIT 1
            """,
            head=head, tail=tail,
        ))
        evidence = " ".join(rows[0]["evidence"]) if rows and rows[0]["evidence"] else None
        gold.append({"head": head, "relation": relation, "tail": tail, "evidence": evidence})
    return gold


def _fact_key(f):
    return (f["head"].lower(), f["relation"], f["tail"].lower())


# ---------------------------------------------------------------------------
# 검색(retrieval) 지표
# ---------------------------------------------------------------------------

def _matched_gold_indices_graphrag(retrieved_facts, gold_facts):
    """retrieved_facts의 각 항목이 gold_facts 중 몇 번째와 매치되는지 인덱스로 추적
    (같은 gold를 몇 번 맞혔는지가 아니라 "찾았는지 여부"를 구분하기 위함). exact
    (head,relation,tail) 매치를 우선 쓰고, 구조화된 gold가 없는 질문(evidence만
    있는 gold, 예: 합병)은 evidence 텍스트 완전 일치로 대체."""
    matched = []
    for f in retrieved_facts:
        idx = None
        fkey = _fact_key(f)
        for i, g in enumerate(gold_facts):
            if _fact_key(g) == fkey or (f.get("evidence") and g.get("evidence") and f["evidence"] == g["evidence"]):
                idx = i
                break
        matched.append(idx)
    return matched


def _normalize_sentence(text):
    """gold evidence(Neo4j에 저장된 DocRED 토큰을 공백으로 join한 형태, 예: "1877 ,
    it was Western Canada 's first university .")와 naive RAG 청크(Pinecone에
    자연스러운 원문 그대로 색인된 형태, 예: "1877, it was Western Canada's first
    university.")는 같은 문장인데 구두점 앞 공백 유무가 달라 그냥 whitespace만
    정규화하면 서로 substring 매치가 안 됨(실측: naive RAG 검색 결과 지표가 명백히
    정답을 맞힌 경우에도 전부 0으로 나오는 버그를 이걸로 발견) -- 구두점 앞 공백도
    제거해야 함."""
    text = re.sub(r"\s+([,.'\)\]:;!?%])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _matched_gold_indices_naive(chunks, gold_facts):
    gold_norm = [_normalize_sentence(g["evidence"]) for g in gold_facts if g.get("evidence")]
    matched = []
    for c in chunks:
        text_norm = _normalize_sentence(c["text"])
        idx = None
        for i, ge in enumerate(gold_norm):
            if text_norm in ge or ge in text_norm:
                idx = i
                break
        matched.append(idx)
    return matched


def compute_retrieval_metrics(matched_indices, n_gold):
    """matched_indices: 검색 순위(rank) 순서대로, 각 항목이 매치된 gold 인덱스(없으면
    None). Precision은 raw label 기준(중복 사실을 여러 번 반환하면 노이즈로 집계돼
    낮아짐 -- dedup 없는 baseline이 같은 사실을 반복 반환하는 문제를 정확히 반영하기
    위해 일부러 원본 리스트 길이로 나눔). Recall/AP/nDCG는 "몇 개의 서로 다른 gold
    항목을 찾았는지" 기준(같은 gold를 중복으로 여러 번 맞혀도 1개만 인정) -- 안
    그러면 중복 반환이 많을수록 값이 1.0을 넘어가는 버그가 생김(baseline 실측에서
    Recall=11.00, AP=6.59까지 나온 걸 발견해 수정)."""
    if not matched_indices:
        return dict(precision=0.0, recall=0.0, f1=0.0, ndcg=0.0, ap=0.0)

    precision = sum(1 for idx in matched_indices if idx is not None) / len(matched_indices)

    seen = set()
    first_hit_labels = []  # 각 gold는 최초로 맞힌 위치에서만 1, 그 뒤 중복은 0(recall/AP/nDCG용)
    for idx in matched_indices:
        if idx is not None and idx not in seen:
            seen.add(idx)
            first_hit_labels.append(1)
        else:
            first_hit_labels.append(0)

    recall = len(seen) / n_gold if n_gold else 0.0

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(first_hit_labels))
    n_ones = min(n_gold, len(first_hit_labels))
    ideal = [1] * n_ones + [0] * (len(first_hit_labels) - n_ones)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    ndcg = dcg / idcg if idcg else 0.0

    hits, precisions = 0, []
    for i, rel in enumerate(first_hit_labels):
        if rel:
            hits += 1
            precisions.append(hits / (i + 1))
    ap = sum(precisions) / n_gold if (precisions and n_gold) else 0.0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(precision=precision, recall=recall, f1=f1, ndcg=ndcg, ap=ap)


# ---------------------------------------------------------------------------
# 생성(generation) 지표 -- 커스텀 LLM-judge (RAGAS 정의 준용, 통합 1회 호출)
# ---------------------------------------------------------------------------

def judge_generation_metrics(question, expected_answer, context_texts, generated_answer, system_name):
    context_block = "\n".join(f"- {c}" for c in context_texts) if context_texts else "(컨텍스트 없음)"

    def compute():
        prompt = (
            "당신은 RAG(검색 증강 생성) 시스템의 출력을 평가하는 채점자입니다. "
            "아래 질문/기대 답변(참고용)/검색된 컨텍스트/생성된 답변을 보고 4개 지표를 "
            "각각 0.0~1.0 사이 실수로 채점하세요.\n\n"
            "- faithfulness(충실도): 생성된 답변에 담긴 주장들 중 검색된 컨텍스트로 "
            "뒷받침되는 비율. 컨텍스트에 없는 내용을 지어냈으면 낮게.\n"
            "- context_precision: 검색된 컨텍스트 중 실제로 질문에 답하는 데 쓸모 "
            "있었던 항목의 비율(관련 없는 노이즈가 많으면 낮게).\n"
            "- context_recall: 기대 답변을 만드는 데 필요한 정보가 검색된 컨텍스트에 "
            "얼마나 포함돼 있었는지(컨텍스트만으로 기대 답변을 재구성할 수 있으면 1.0).\n"
            "- answer_relevance: 생성된 답변이 실제로 질문에서 물은 것에 직접적으로 "
            "답하는 정도('모름'이나 동문서답이면 낮게, 단 컨텍스트가 진짜 부족해서 "
            "정직하게 '모름'이라 한 경우는 answer_relevance를 0.5 정도로 중립 처리).\n\n"
            f"질문: {question}\n"
            f"기대 답변(참고용): {expected_answer}\n"
            f"검색된 컨텍스트:\n{context_block}\n\n"
            f"생성된 답변: {generated_answer}\n\n"
            '반드시 JSON 객체 하나만 출력하세요(다른 설명 없이): '
            '{"faithfulness": 0.0, "context_precision": 0.0, "context_recall": 0.0, "answer_relevance": 0.0}'
        )
        resp = advanced_gq.openai_client.chat.completions.create(
            model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)

    key = cache_key("eval_gen_metrics_v1", CHAT_MODEL, system_name, question, generated_answer, context_block)
    try:
        return cached(key, compute)
    except json.JSONDecodeError:
        return {"faithfulness": None, "context_precision": None, "context_recall": None, "answer_relevance": None}


# ---------------------------------------------------------------------------
# 시스템별 실행 래퍼 -- (답변, 컨텍스트 텍스트 리스트, retrieval 지표용 원본 아이템 리스트) 반환
# ---------------------------------------------------------------------------

def run_baseline(question):
    answer, _seed_names, facts = baseline_gq.answer_question(question, verbose=False)
    context_texts = [baseline_gq._format_fact(f) for f in facts]
    return answer, context_texts, facts


def run_advanced(question):
    answer, _seed_names, facts, _route = advanced_gq.answer_question(question, verbose=False)
    context_texts = [advanced_gq._format_fact(f) for f in facts]
    return answer, context_texts, facts


def run_naive(question, query_en):
    """RAG/demo_compare.py의 search_naive()를 그대로 안 쓰고 직접 재구현 --
    일부 Pinecone 벡터에 metadata.title이 없어 KeyError가 나는 걸 실측 발견(데이터
    스트림릿 앱에서 이미 한 번 겪은 것과 같은 문제, 2026-07-16). 원본 데모 스크립트는
    이 세션의 수정 범위 밖(건드리지 않음)이라 평가 하니스 안에서만 방어적으로 처리."""
    vector = naive_mod.embed_queries([query_en])[0]
    res = naive_mod.naive_index.query(vector=vector, top_k=TOP_K_NAIVE, include_metadata=True)
    chunks = [
        {"text": m["metadata"].get("text", ""), "title": m["metadata"].get("title", "(제목 없음)")}
        for m in res["matches"] if m.get("metadata", {}).get("text")
    ]
    contexts = [f"[{c['title']}] {c['text']}" for c in chunks]
    answer = naive_mod.answer_with_context(question, contexts)
    return answer, contexts, chunks


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main(resolve_only):
    with advanced_gq.driver.session(database=advanced_gq.NEO4J_DATABASE) as session:
        gold_map = {}
        for category, question, query_en, expected in EVAL_QUESTIONS:
            gold = resolve_gold_facts(session, question, expected)
            gold_map[question] = gold
            print(f"[{category}] {question}")
            print(f"  기대 답변: {expected}")
            print(f"  gold facts ({len(gold)}개):")
            for g in gold:
                print(f"    - {g['head']} -[{g['relation']}]-> {g['tail']}")
            print()

    if resolve_only:
        return

    results = {"baseline": [], "advanced": [], "naive_rag": []}

    for category, question, query_en, expected in EVAL_QUESTIONS:
        gold = gold_map[question]
        n_gold = max(len(gold), 1)
        print("=" * 100)
        print(f"[{category}] {question}  (기대 답변: {expected})")

        # --- baseline ---
        b_answer, b_context_texts, b_facts = run_baseline(question)
        b_matched = _matched_gold_indices_graphrag(b_facts, gold)
        b_ret = compute_retrieval_metrics(b_matched, n_gold)
        b_gen = judge_generation_metrics(question, expected, b_context_texts, b_answer, "baseline")
        print(f"  [baseline] 답변: {b_answer}")
        print(f"  [baseline] {b_ret} gen={b_gen}")
        results["baseline"].append({
            "p": b_ret["precision"], "r": b_ret["recall"], "f1": b_ret["f1"],
            "ndcg": b_ret["ndcg"], "ap": b_ret["ap"], **b_gen,
        })

        # --- advanced ---
        a_answer, a_context_texts, a_facts = run_advanced(question)
        a_matched = _matched_gold_indices_graphrag(a_facts, gold)
        a_ret = compute_retrieval_metrics(a_matched, n_gold)
        a_gen = judge_generation_metrics(question, expected, a_context_texts, a_answer, "advanced")
        print(f"  [advanced] 답변: {a_answer}")
        print(f"  [advanced] {a_ret} gen={a_gen}")
        results["advanced"].append({
            "p": a_ret["precision"], "r": a_ret["recall"], "f1": a_ret["f1"],
            "ndcg": a_ret["ndcg"], "ap": a_ret["ap"], **a_gen,
        })

        # --- naive rag ---
        n_answer, n_context_texts, n_chunks = run_naive(question, query_en)
        n_matched = _matched_gold_indices_naive(n_chunks, gold)
        n_ret = compute_retrieval_metrics(n_matched, n_gold)
        n_gen = judge_generation_metrics(question, expected, n_context_texts, n_answer, "naive_rag")
        print(f"  [naive_rag] 답변: {n_answer}")
        print(f"  [naive_rag] {n_ret} gen={n_gen}")
        results["naive_rag"].append({
            "p": n_ret["precision"], "r": n_ret["recall"], "f1": n_ret["f1"],
            "ndcg": n_ret["ndcg"], "ap": n_ret["ap"], **n_gen,
        })
        print()

    print("=" * 100)
    print("최종 평균 (질문 수:", len(EVAL_QUESTIONS), ")")
    summary = {}
    for system, rows in results.items():
        avg = {}
        for metric in ["p", "r", "f1", "ndcg", "ap", "faithfulness", "context_precision", "context_recall", "answer_relevance"]:
            vals = [row[metric] for row in rows if row.get(metric) is not None]
            avg[metric] = sum(vals) / len(vals) if vals else None
        summary[system] = avg
        print(f"  {system}: {avg}")

    with open(ROOT / "GraphRAG" / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"per_question": results, "summary": summary, "n_questions": len(EVAL_QUESTIONS)}, f, ensure_ascii=False, indent=2)
    print("\n결과 저장: GraphRAG/eval_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolve-only", action="store_true", help="gold fact 자동 추출 결과만 출력하고 종료(LLM 호출 없음)")
    args = parser.parse_args()
    main(resolve_only=args.resolve_only)

    baseline_gq.driver.close()
    advanced_gq.driver.close()
