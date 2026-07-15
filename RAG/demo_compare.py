"""naive RAG(informationrag) vs GraphRAG(informationextraction) 데모 비교 스크립트.

같은 질의를 두 Pinecone 인덱스에 각각 벡터 검색하고, 검색된 컨텍스트만 근거로
LLM(OpenAI gpt-5.4-mini)이 답하게 한 뒤 결과를 나란히 비교 출력한다.
- naive RAG: 문장 단위 청크(title, sentence)를 그대로 검색
- GraphRAG: relation triple 단위 청크(head/relation/tail + evidence)를 검색

질의는 한국어지만 색인된 본문은 전부 영어라, 검색용 쿼리는 영어로 번역해 임베딩한다
(같은 언어끼리 검색해야 벡터 유사도가 잘 나옴). 최종 답변은 한국어로 받는다.

사용법:
    python RAG/demo_compare.py
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIMENSIONS = 512
CHAT_MODEL = "gpt-5.4-mini"
TOP_K = 5

openai_client = OpenAI()
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
naive_index = pc.Index("informationrag")
graph_index = pc.Index("informationextraction")

# (카테고리, 한국어 질문, 검색용 영어 질의, 참고용 기대 답변 또는 None)
QUERIES = [
    ("① 단일 홉", "Roketsan은 어느 나라 회사야?",
     "Which country is Roketsan a company from?", "Turkey"),
    ("① 단일 홉", "University of Manitoba는 몇 년에 설립됐어?",
     "What year was the University of Manitoba founded?", "1877"),
    ("① 단일 홉", "Lappeenranta는 어느 나라 도시야?",
     "Which country is the city of Lappeenranta in?", "Finland"),
    ("① 단일 홉", "AirAsia Zest의 본사는 어디였어?",
     "Where was AirAsia Zest headquartered?", "Ninoy Aquino International Airport, Manila"),

    ("② 문서 내 멀티홉", "Cirit을 만드는 회사의 본사는 어느 도시에 있어?",
     "In which city is the headquarters of the company that makes Cirit?", "Ankara"),
    ("② 문서 내 멀티홉", "Roketsan은 몇 년에 설립됐고, 어느 위원회가 세웠어?",
     "What year was Roketsan founded, and which committee established it?",
     "1988, Turkey's Defense Industry Executive Committee (SSIK)"),
    ("② 문서 내 멀티홉", "AirAsia Zest는 어느 항공사들이 합쳐져 만들어졌고, 언제 그렇게 됐어?",
     "Which airlines merged to form AirAsia Zest, and when did that happen?",
     "Asian Spirit and Zest Air; rebranded as AirAsia Zest after the alliance"),

    ("③ 문서 간 멀티홉", "Health Sciences Centre가 있는 도시에 있는 대학교는 몇 년에 설립됐어?",
     "What year was the university located in the same city as the Health Sciences Centre founded?",
     "1877 (Winnipeg -> University of Manitoba)"),
    ("③ 문서 간 멀티홉", "Outotec의 필터가 만들어지는 도시는 어느 지역에 속해있어?",
     "Which region does the city where Outotec's filters are manufactured belong to?",
     "South Karelia (Lappeenranta)"),
    ("③ 문서 간 멀티홉", "Manila을 언급하는 문서들 중에 항공사 관련 문서는 뭐가 있어?",
     "Among documents that mention Manila, which ones are related to airlines?", None),
    ("③ 문서 간 멀티홉", "Philippines과 관련된 문서들을 모아서 요약해줘",
     "Gather and summarize the documents related to the Philippines.", None),

    ("④ 집계/스캔형", "1988년에 설립된 조직이 뭐가 있어?",
     "What organizations were founded in 1988?", None),
    ("④ 집계/스캔형", "본사가 Ankara에 있는 회사는?",
     "Which companies are headquartered in Ankara?", None),
]


def embed_queries(texts):
    response = openai_client.embeddings.create(
        input=texts, model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS
    )
    return [d.embedding for d in response.data]


def search_naive(vector):
    res = naive_index.query(vector=vector, top_k=TOP_K, include_metadata=True)
    matches = res["matches"]
    contexts = [f"[{m['metadata']['title']}] {m['metadata']['text']}" for m in matches]
    labels = [f"{m['metadata']['title']} (score={m['score']:.3f})" for m in matches]
    return contexts, labels


def search_graph(vector):
    res = graph_index.query(vector=vector, top_k=TOP_K, include_metadata=True)
    matches = res["matches"]
    contexts = []
    labels = []
    for m in matches:
        md = m["metadata"]
        contexts.append(
            f"{md['head_name']} -[{md['relation_name']}]-> {md['tail_name']} "
            f"(문서: {md['document']}, 근거: {md['text']})"
        )
        labels.append(
            f"{md['head_name']} -[{md['relation_name']}]-> {md['tail_name']} "
            f"(문서: {md['document']}, score={m['score']:.3f})"
        )
    return contexts, labels


def answer_with_context(question, contexts):
    context_block = "\n".join(f"- {c}" for c in contexts) if contexts else "(검색 결과 없음)"
    prompt = (
        "아래 컨텍스트만 근거로 질문에 한국어로 답하세요. "
        "컨텍스트에 답이 없거나 근거가 불충분하면 반드시 '모름'이라고만 답하세요.\n\n"
        f"컨텍스트:\n{context_block}\n\n질문: {question}"
    )
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def main():
    query_vectors = embed_queries([q[2] for q in QUERIES])

    for (category, question_ko, query_en, expected), vector in zip(QUERIES, query_vectors):
        naive_contexts, naive_labels = search_naive(vector)
        graph_contexts, graph_labels = search_graph(vector)

        naive_answer = answer_with_context(question_ko, naive_contexts)
        graph_answer = answer_with_context(question_ko, graph_contexts)

        print("=" * 100)
        print(f"[{category}] {question_ko}")
        if expected:
            print(f"  기대 답변: {expected}")
        print(f"  naive RAG 검색된 문서: {naive_labels}")
        print(f"  naive RAG 답변       : {naive_answer}")
        print(f"  GraphRAG 검색된 triple: {graph_labels}")
        print(f"  GraphRAG 답변         : {graph_answer}")
        print()


if __name__ == "__main__":
    main()
