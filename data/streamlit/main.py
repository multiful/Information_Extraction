"""Trace — Knowledge Graph Workbench

자연어 질문 -> LLM으로 의도/엔티티/관계타입 추출 -> Neo4j 지식그래프(DocRED)에서
엔티티 링킹 및 라우팅(속성 스캔/1-hop 직행/멀티홉 BFS) -> 리랭킹 -> 반복
샘플링+다수결로 답변을 생성하는 GraphRAG 데모 앱. 같은 질문을 Pinecone 기반
naive RAG(문장 단위 벡터 검색)로도 조회해 나란히 비교한다.

**핵심 로직은 이 파일에서 재구현하지 않고 `GraphRAG/graphrag_query.py`를 그대로
가져와 씀** — 그 파일에 이미 구현·검증된 파이프라인(Exact->Alias->Word Boundary->
Fuzzy 엔티티 링킹 캐스케이드, relation-aware 라우팅, 중복 제거+ORDER BY로 결정적인
BFS, 임베딩 리랭킹, temperature=0 + Graph-grounded Evidence Retrieval 폴백 +
반복 샘플링(x3)/LLM 클러스터링 다수결)를 이 앱에서 다시 손으로 짜면 두 파이프라인이
갈라져서 버그가 각자 따로 생긴다. 이 파일은 그 파이프라인 호출 + 결과를 UI에
표시하기 위한 부가 정보(엔티티 매칭, 엣지 confidence/evidence_source, 그래프
시각화용 타입 등) 조회, 그리고 naive RAG 비교 패널만 담당한다.

'생각 과정'은 LLM의 원문 추론을 그대로 노출하지 않고, 실제 실행된 처리
단계(질의 분석 -> 엔티티 링킹 -> 라우팅 -> 리랭킹 -> 반복 샘플링+다수결)를
실제 수치와 함께 감사 가능한(auditable) 형태로 보여준다.

실행:
    streamlit run data/streamlit/main.py

필요 환경변수 (레포 루트 `.env`): OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME,
NEO4J_PASSWORD, NEO4J_DATABASE(선택), PINECONE_API_KEY(naive RAG 비교용).
"""

import os
import re
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from pinecone import Pinecone
from pyvis.network import Network

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

# Streamlit Cloud엔 .env가 없다 — 대시보드 Secrets를 환경변수로 승격해
# 아래 os.environ[...] 경로가 로컬(.env)/클라우드(secrets) 모두에서 동작하게 한다.
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # secrets 미설정(로컬 .env만 쓰는 경우)이면 무시

# GraphRAG/graphrag_query.py를 모듈로 가져오기 위해 경로 추가 (그 폴더가 자체
# 상대 import를 쓰므로 -- `from cache import ...` -- sys.path에 폴더 자체를 넣어야 함)
sys.path.insert(0, str(ROOT / "GraphRAG"))
import graphrag_query as gq  # noqa: E402  (sys.path 조작 이후에 import해야 함)

CHAT_MODEL = gq.CHAT_MODEL
MAX_QUERY_CHARS = 600

# naive RAG(Pinecone) 비교 패널 설정 -- RAG/load_naive_rag.py가 적재할 때 쓴
# 값과 정확히 맞춰야 벡터 차원이 일치한다.
RAG_INDEX_NAME = "informationrag"
RAG_EMBED_MODEL = "text-embedding-3-small"
RAG_EMBED_DIMENSIONS = 512
RAG_TOP_K = 5

ROUTE_LABELS_KO = {
    "1hop": "① 1-hop 직행 조회",
    "1hop_fallback_bfs": "① 1-hop 실패 -> ③ 멀티홉 BFS 폴백",
    "property_scan": "②' 속성 전역 스캔",
    "bfs": "③ 멀티홉 BFS",
    "no_seed": "매칭된 엔티티 없음",
}

TYPE_COLORS = {
    "PER": "#c97b63",
    "ORG": "#d9a441",
    "LOC": "#8fae94",
    "TIME": "#7a9cc6",
    "NUM": "#a68bc9",
    "MISC": "#a8a397",
}
TYPE_LABELS_KO = {
    "PER": "인물", "ORG": "기관/조직", "LOC": "장소", "TIME": "시간", "NUM": "수치", "MISC": "기타",
}
EVIDENCE_SOURCE_KO = {
    "annotated": "원문 근거(gold)",
    "inferred_bridge": "브리징 추론",
    "inferred_cooccurrence": "문장 동시 등장 추론",
    "inferred_mention_union": "멘션 통합 추론",
    "unresolved_multihop": "멀티홉 (근거 미확정)",
    "model_provided": "모델 예측",
}

SUGGESTED_QUESTIONS = [
    "AirAsia Zest는 어느 항공사들이 합쳐져 만들어졌고, 언제 그렇게 됐어?",
    "Roketsan은 몇 년에 설립됐고, 어느 위원회가 세웠어?",
    "Outotec의 필터가 만들어지는 도시는 어느 지역에 속해있어?",
]


# ---------------------------------------------------------------------------
# 리소스 (캐시) -- Neo4j/OpenAI 클라이언트는 graphrag_query 모듈이 import 시점에
# 이미 하나 만들어 갖고 있으므로(gq.driver/gq.openai_client) 여기서 새로 만들지
# 않고 재사용한다 (같은 인스턴스로 연결 두 벌 열지 않기 위함).
# ---------------------------------------------------------------------------

@st.cache_resource
def get_pinecone_index():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(RAG_INDEX_NAME)


# ---------------------------------------------------------------------------
# GraphRAG 파이프라인 (graphrag_query.py 재사용 + UI용 부가 정보 조회)
# ---------------------------------------------------------------------------

def _fetch_viz_metadata(session, facts):
    """facts(head/relation/tail 이름만 있음)에 시각화·근거원장에 필요한 노드 타입 +
    엣지 evidence_source를 보강 조회한다. graphrag_query.py의 facts 자체엔 이 정보가
    없음(그 파이프라인은 애초에 confidence를 안 씀 -- 문제 9가 폐기된 이유와 같은
    맥락) -- 표시 전용 부가 조회이므로 여기서만 함. confidence는 조회하지 않는다 --
    거의 모든 엣지가 1.0(gold/가공 데이터 기본값)이라 "신뢰도 100%"로만 찍혀 UI에
    실질적 정보를 안 줌(사용자 피드백으로 제거, 2026-07-16)."""
    names = sorted({f["head"] for f in facts} | {f["tail"] for f in facts})
    node_types = {}
    if names:
        for r in session.run(
            "MATCH (e:ZEntity) WHERE e.name IN $names RETURN DISTINCT e.name AS name, e.type AS type",
            names=names,
        ):
            node_types.setdefault(r["name"], r["type"])

    edge_meta = {}
    if facts:
        triples = [{"h": f["head"], "rel": f["relation"], "t": f["tail"]} for f in facts]
        for r in session.run(
            """
            UNWIND $triples AS tr
            MATCH (h:ZEntity {name: tr.h})-[rel]->(t:ZEntity {name: tr.t})
            WHERE type(rel) = tr.rel
            RETURN tr.h AS h, tr.rel AS rel, tr.t AS t, rel.evidence_source AS evidence_source
            LIMIT 2000
            """,
            triples=triples,
        ):
            key = (r["h"], r["rel"], r["t"])
            edge_meta.setdefault(key, {"evidence_source": r["evidence_source"]})
    return node_types, edge_meta


def run_graphrag_analysis(question):
    """GraphRAG/graphrag_query.py의 실제 파이프라인을 그대로 호출:
        ① extract_entities() -- entities/relation_type/value/entity_type 동시 추출
        ② find_seed_entities() -- Exact->Alias->Word Boundary->Fuzzy 캐스케이드
        ③ 라우팅 -- property_scan / relation_lookup(1-hop) / expand_subgraph(멀티홉 BFS)
        ④ rerank_facts() -- 사실 80개 초과 시 임베딩 유사도로 상위 80개만
        ⑤ majority_vote_answer() -- answer_with_subgraph()를 3회 반복 샘플링
           (내부적으로 "모름"이면 Graph-grounded Evidence Retrieval 폴백도 자동 적용됨)
           -> LLM 클러스터링으로 다수결
    """
    parsed = gq.extract_entities(question)
    mentions = parsed["entities"]
    relation_type = parsed["relation_type"]
    value = parsed["value"]
    entity_type = parsed["entity_type"]

    with gq.driver.session(database=gq.NEO4J_DATABASE) as session:
        entity_results = []
        seed_ids = []
        for mention in mentions:
            rows = gq.find_seed_entities(session, mention)
            matches = [{"id": r["id"], "name": r["name"], "type": r["type"]} for r in rows]
            entity_results.append({"mention": mention, "matches": matches})
            seed_ids.extend(m["id"] for m in matches)

        if not seed_ids:
            if relation_type and value:
                facts = gq.property_scan(session, relation_type, value, entity_type)
                route = "property_scan"
            else:
                facts = []
                route = "no_seed"
        elif relation_type:
            facts = gq.relation_lookup(session, seed_ids, relation_type)
            route = "1hop"
            if not facts:
                facts = gq.expand_subgraph(session, seed_ids)
                route = "1hop_fallback_bfs"
        else:
            facts = gq.expand_subgraph(session, seed_ids)
            route = "bfs"

        n_before_rerank = len(facts)
        facts = gq.rerank_facts(question, facts)

        node_types, edge_meta = _fetch_viz_metadata(session, facts)

    # graphrag_query.answer_question()과 동일한 규칙: 흔들림이 실측된 라우팅(bfs,
    # 1hop_fallback_bfs)에서만 3회 샘플링+다수결을 적용하고, 원래도 안정적인
    # 1hop/property_scan은 단발 호출로 비용을 아낀다. no_seed는 facts가 항상 []라
    # LLM을 불러도 결과가 뻔히 "모름"으로 고정되므로 호출 자체를 생략하고 원인이
    # 분명한 고정 응답을 준다(2026-07-16, 사용자 피드백 -- "삼성" 같은 질문이 실제로는
    # 그래프에 엔티티가 있는데도 한국어 멘션이 안 풀려 이 경로를 탄 사례 발견).
    if route == "no_seed":
        answer = gq.ENTITY_NOT_FOUND_ANSWER
        votes = None
        agree_count = None
    elif route in ("bfs", "1hop_fallback_bfs"):
        answer, votes, agree_count = gq.majority_vote_answer(question, facts)
    else:
        answer = gq.answer_with_subgraph(question, facts)
        votes = None
        agree_count = None

    edges = []
    for f in facts:
        meta = edge_meta.get((f["head"], f["relation"], f["tail"]), {})
        edges.append({
            "src_name": f["head"], "relation": f["relation"], "dst_name": f["tail"],
            "evidence": f.get("evidence"),
            "evidence_source": meta.get("evidence_source"),
        })
    nodes = {name: {"name": name, "type": node_types.get(name, "MISC")}
             for name in ({e["src_name"] for e in edges} | {e["dst_name"] for e in edges})}

    return {
        "question": question,
        "entities": mentions,
        "relation_type": relation_type,
        "value": value,
        "entity_type": entity_type,
        "entity_results": entity_results,
        "seed_ids": set(seed_ids),
        "route": route,
        "n_before_rerank": n_before_rerank,
        "nodes": nodes,
        "edges": edges,
        "answer": answer,
        "votes": votes,
        "agree_count": agree_count,
    }


# ---------------------------------------------------------------------------
# naive RAG(Pinecone) 비교 파이프라인
# ---------------------------------------------------------------------------

def translate_to_english(question):
    """색인된 본문이 전부 영어라(RAG/load_naive_rag.py), 벡터 검색 전에 질문을
    영어로 번역한다 -- RAG/demo_compare.py가 이미 확립한 방식(같은 언어끼리
    검색해야 벡터 유사도가 잘 나옴)."""
    resp = gq.openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{
            "role": "user",
            "content": f"다음 한국어 질문을 자연스러운 영어 질문 한 문장으로만 번역하세요 "
                       f"(다른 설명 없이 번역문만):\n{question}",
        }],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def run_naive_rag_analysis(question):
    """Pinecone(informationrag) 문장 단위 벡터 검색 + LLM 답변. GraphRAG처럼
    관계로 구조화하지 않고 문장 청크를 그대로 검색하므로, 단일홉 질의는 잘 찾지만
    문장 간 연결이 필요한 멀티홉 질의는 못 찾는 걸 GraphRAG와 나란히 비교해 보여준다."""
    index = get_pinecone_index()
    query_en = translate_to_english(question)

    resp = gq.openai_client.embeddings.create(
        input=[query_en], model=RAG_EMBED_MODEL, dimensions=RAG_EMBED_DIMENSIONS,
    )
    vector = resp.data[0].embedding

    result = index.query(vector=vector, top_k=RAG_TOP_K, include_metadata=True)
    chunks = [
        {
            "title": m.get("metadata", {}).get("title", "(제목 없음)"),
            "text": m.get("metadata", {}).get("text", ""),
            "score": m["score"],
        }
        for m in result["matches"] if m.get("metadata", {}).get("text")
    ]

    context_block = "\n".join(f"- [{c['title']}] {c['text']}" for c in chunks) if chunks else "(검색 결과 없음)"
    prompt = (
        "아래 컨텍스트만 근거로 질문에 한국어로 답하세요. 컨텍스트에 답이 없거나 "
        "근거가 불충분하면 반드시 '모름'이라고만 답하세요.\n\n"
        f"컨텍스트:\n{context_block}\n\n질문: {question}"
    )
    answer_resp = gq.openai_client.chat.completions.create(
        model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
    )

    return {
        "question": question,
        "query_en": query_en,
        "chunks": chunks,
        "answer": answer_resp.choices[0].message.content.strip(),
    }


# ---------------------------------------------------------------------------
# 그래프 시각화
# ---------------------------------------------------------------------------

def build_graph_html(nodes, edges, seed_names):
    net = Network(
        height="440px", width="100%", bgcolor="#131c11", font_color="#eee8d8",
        directed=True, cdn_resources="remote",
    )
    net.barnes_hut(gravity=-3500, central_gravity=0.25, spring_length=130, spring_strength=0.02)

    for name, info in nodes.items():
        is_seed = name in seed_names
        color = TYPE_COLORS.get(info["type"], "#a8a397")
        net.add_node(
            name,
            label=info["name"],
            title=f"{info['name']} ({TYPE_LABELS_KO.get(info['type'], info['type'])})",
            color={"background": color, "border": "#f5f1e6" if is_seed else color},
            borderWidth=3 if is_seed else 1,
            shape="dot",
            size=26 if is_seed else 17,
            font={"color": "#f5f1e6", "size": 13},
        )

    seen_pairs = set()
    for e in edges:
        pair = (e["src_name"], e["dst_name"], e["relation"])
        if pair in seen_pairs or e["src_name"] not in nodes or e["dst_name"] not in nodes:
            continue
        seen_pairs.add(pair)
        net.add_edge(
            e["src_name"], e["dst_name"],
            label=e["relation"],
            color="#5c6b57",
            font={"size": 10, "color": "#c9c2a8", "strokeWidth": 0},
            arrows="to",
        )

    return net.generate_html(notebook=False)


# ---------------------------------------------------------------------------
# 스타일
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #12190f 0%, #161f13 100%);
            color: #eee8d8;
        }
        [data-testid="stHeader"] { background: transparent; }
        h1, h2, h3, h4 { color: #f3efe1 !important; }
        p, span, label, div { color: #eee8d8; }

        .trace-topbar {
            display: flex; justify-content: space-between; align-items: center;
            padding: 8px 4px 20px 4px; border-bottom: 1px solid #33402c; margin-bottom: 8px;
        }
        .trace-logo { font-weight: 800; letter-spacing: 2px; font-size: 1.1rem; color: #f3efe1; }
        .trace-logo .sub { font-weight: 400; color: #9aa38f; font-size: 0.8rem; margin-left: 8px; }
        .trace-status { font-size: 0.75rem; color: #9aa38f; }
        .trace-status .dot { color: #7fbf7f; }

        .section-label {
            display: flex; align-items: center; gap: 8px;
            font-size: 0.78rem; letter-spacing: 1.5px; color: #b7c2a8; text-transform: uppercase;
            margin-bottom: 6px;
        }
        .badge {
            background: #d9a441; color: #1c2418; font-weight: 700;
            border-radius: 4px; padding: 1px 7px; font-size: 0.72rem;
        }
        .badge.rag { background: #7a9cc6; }

        .hero-title { font-size: 2.4rem; line-height: 1.25; font-weight: 700; color: #f3efe1; margin: 4px 0 10px 0; }
        .hero-title .accent { color: #d9a441; border-bottom: 3px solid #d9a441; }
        .hero-sub { color: #a9b39c; font-size: 0.95rem; max-width: 560px; }

        .pipeline-box { font-size: 0.85rem; }
        .pipeline-item { display: flex; gap: 8px; padding: 4px 0; color: #cdd5c1; }
        .pipeline-item .n { color: #d9a441; font-weight: 700; }

        .card-cream {
            background: #efe8d5; color: #1c2418; border-radius: 12px;
            padding: 22px 26px; margin: 10px 0 18px 0;
        }
        .card-cream * { color: #1c2418; }
        .muted { color: #6b7362 !important; font-size: 0.8rem; }

        .stat-row {
            display: flex; gap: 0; border: 1px solid #33402c; border-radius: 10px;
            overflow: hidden; margin-bottom: 18px;
        }
        .stat-cell {
            flex: 1; padding: 14px 18px; border-right: 1px solid #33402c;
        }
        .stat-cell:last-child { border-right: none; }
        .stat-cell .label { font-size: 0.72rem; color: #9aa38f; text-transform: uppercase; letter-spacing: 1px; }
        .stat-cell .value { font-size: 1.6rem; font-weight: 700; color: #f3efe1; }
        .stat-cell .value .unit { font-size: 0.85rem; color: #9aa38f; font-weight: 400; margin-left: 4px; }

        .panel {
            background: #1a2417; border: 1px solid #33402c; border-radius: 12px;
            padding: 18px 20px; margin-bottom: 16px; height: 100%;
        }
        .panel * { color: inherit; }

        .entity-chip {
            display: inline-block; background: #d9a441; color: #1c2418 !important;
            font-weight: 700; font-size: 0.7rem; border-radius: 4px; padding: 2px 7px; margin-right: 8px;
        }
        .entity-item { padding: 10px 0; border-bottom: 1px solid #2a3524; }
        .entity-item:last-child { border-bottom: none; }
        .entity-name { font-weight: 700; color: #f3efe1; }
        .entity-quote { font-size: 0.78rem; color: #9aa38f; }

        .step-item { display: flex; gap: 10px; padding: 8px 0; font-size: 0.85rem; color: #cdd5c1; }
        .step-num { color: #d9a441; font-weight: 700; min-width: 18px; }

        .evidence-item {
            border: 1px solid #33402c; border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
            background: #171f14;
        }
        .evidence-id { color: #d9a441; font-weight: 700; font-size: 0.75rem; }
        .evidence-triple { font-weight: 700; color: #f3efe1; margin: 4px 0; }
        .evidence-quote { color: #a9b39c; font-size: 0.82rem; font-style: italic; }
        .evidence-tag {
            display: inline-block; background: #2a3524; color: #cdd5c1 !important;
            font-size: 0.68rem; border-radius: 4px; padding: 2px 7px; margin-top: 6px; margin-right: 6px;
        }

        .vote-item {
            border: 1px solid #33402c; border-radius: 8px; padding: 10px 14px; margin-bottom: 6px;
            font-size: 0.82rem; background: #171f14;
        }
        .vote-item.chosen { border-color: #d9a441; background: #24230f; }
        .vote-tag {
            display: inline-block; font-size: 0.65rem; font-weight: 700; border-radius: 4px;
            padding: 1px 6px; margin-bottom: 4px;
        }
        .vote-tag.chosen { background: #d9a441; color: #1c2418; }
        .vote-tag.other { background: #33402c; color: #cdd5c1; }

        .compare-col-title {
            font-weight: 800; font-size: 1rem; letter-spacing: 1px; margin-bottom: 10px;
            display: flex; align-items: center; gap: 8px;
        }

        /* st.container(key=...)가 생성하는 wrapper(stVerticalBlock 자체)에 카드 스타일 적용 */
        div.st-key-query_console {
            background: #efe8d5; border-radius: 12px; padding: 22px 26px 14px 26px; margin: 10px 0 18px 0;
            gap: 0.6rem !important;
        }
        div.st-key-query_console textarea {
            background: #f8f5ec !important; color: #1c2418 !important; border: 1px solid #cfc6a8 !important;
        }
        div.st-key-query_console p { color: #6b7362 !important; margin: 0 !important; }
        div.st-key-chip_row button {
            background: transparent !important; color: #eee8d8 !important; border: 1px solid #4a5842 !important;
            font-weight: 500; font-size: 0.78rem; white-space: normal; text-align: left !important;
            justify-content: flex-start !important; margin-bottom: 6px;
        }
        div.st-key-chip_row button p { text-align: left !important; }
        div.st-key-chip_row button:hover { border-color: #d9a441 !important; color: #d9a441 !important; }

        div.st-key-history_row button {
            background: #171f14 !important; color: #cdd5c1 !important; border: 1px solid #33402c !important;
            text-align: left !important; justify-content: flex-start !important; white-space: pre-line !important;
            font-weight: 500; line-height: 1.5; padding: 10px 14px;
        }
        div.st-key-history_row button:hover { border-color: #d9a441 !important; }
        div.st-key-history_row button p { text-align: left !important; }

        div.stButton > button {
            background: #d9a441; color: #1c2418; border: none; font-weight: 700; border-radius: 8px;
        }
        div.stButton > button:hover { background: #e8b658; color: #1c2418; }

        button[data-baseweb="tab"] {
            background: transparent !important; color: #9aa38f !important; font-weight: 700 !important;
            flex: 1 1 0 !important; justify-content: center !important;
        }
        button[data-baseweb="tab"] p { color: inherit !important; font-size: 0.95rem !important; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #d9a441 !important; }
        button[data-baseweb="tab"][aria-selected="true"] p { color: #d9a441 !important; }
        div[data-baseweb="tab-highlight"] { background-color: #d9a441 !important; }
        div[data-baseweb="tab-border"] { background-color: #33402c !important; }
        div[data-baseweb="tab-list"] { gap: 24px !important; width: 100% !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 렌더링
# ---------------------------------------------------------------------------

def _md_bold_to_html(text):
    """LLM 답변에 마크다운 굵게(**...**)가 섞여 나올 때가 있는데, 이 텍스트를
    커스텀 CSS 카드(raw HTML 블록) 안에 그대로 넣으면 스트림릿 마크다운 파서가
    안 먹혀 별표가 그대로 보인다 -- <b> 태그로 직접 변환."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)


def render_header(neo4j_online):
    # 정상 연결(기본 상태)일 땐 상태 표시를 아예 안 보여준다 -- 항상 켜져 있는
    # "ONLINE" 배지는 매번 같은 값이라 실질 정보가 없다는 사용자 피드백(2026-07-16).
    # 실제로 끊겼을 때만(드문, 행동이 필요한 상태) 눈에 띄게 알린다.
    status_html = (
        '<div class="trace-status" style="color:#e07a5f;">🔴 KNOWLEDGE GRAPH OFFLINE</div>'
        if not neo4j_online else ""
    )
    st.markdown(
        f"""
        <div class="trace-topbar">
            <div class="trace-logo">TRACE <span class="sub">| KG WORKBENCH</span></div>
            {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown('<div class="section-label"><span class="badge">01</span> QUERY INTELLIGENCE CONSOLE</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([2.2, 1])
    with col1:
        st.markdown(
            '<div class="hero-title">질문을 <span class="accent">경로</span>로,<br>'
            '답변을 <span class="accent">증거</span>로.</div>'
            '<div class="hero-sub">자연어 질문을 GraphRAG(관계 그래프 탐색)와 naive RAG'
            '(문장 벡터 검색)로 동시에 답변해 비교합니다.</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="pipeline-box">
            <div class="muted" style="letter-spacing:1px;">GRAPHRAG PIPELINE</div>
            <div class="pipeline-item"><span class="n">01</span> 질의 분석(엔티티/관계/값)</div>
            <div class="pipeline-item"><span class="n">02</span> 엔티티 링킹 캐스케이드</div>
            <div class="pipeline-item"><span class="n">03</span> 라우팅(스캔/1-hop/멀티홉)</div>
            <div class="pipeline-item"><span class="n">04</span> 시맨틱 리랭킹</div>
            <div class="pipeline-item"><span class="n">05</span> 3회 샘플링 + 다수결</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _set_query(sq):
    st.session_state.query_input = sq


def render_query_console():
    with st.container(key="query_console"):
        query = st.text_area(
            "질문 입력", key="query_input", height=70, max_chars=MAX_QUERY_CHARS,
            label_visibility="collapsed", placeholder="질문을 입력하세요 (예: AirAsia Zest의 본사는 어디야?)",
        )
        st.markdown(f'<p style="text-align:right; margin:0;">{len(query)} / {MAX_QUERY_CHARS}</p>', unsafe_allow_html=True)
        run_clicked = st.button("분석 실행 →", type="primary")

        st.markdown('<p style="margin-top:10px;">추천 질문</p>', unsafe_allow_html=True)
        with st.container(key="chip_row"):
            for i, sq in enumerate(SUGGESTED_QUESTIONS):
                st.button(f"{i+1} · {sq}", key=f"chip_{i}", on_click=_set_query, args=(sq,), use_container_width=True)
    return query, run_clicked


def render_graphrag_panel(result):
    st.markdown(
        '<div class="compare-col-title">🕸️ GraphRAG <span class="muted">관계 그래프 탐색</span></div>',
        unsafe_allow_html=True,
    )

    route_label = ROUTE_LABELS_KO.get(result["route"], result["route"])
    st.markdown(
        f"""
        <div class="card-cream">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="font-size:0.75rem; font-weight:700;">다수결 최종 답변</div>
                <div class="evidence-tag" style="background:#d9a441;">{route_label}</div>
            </div>
            <div style="font-size:1.15rem; font-weight:700; margin:10px 0 6px 0;">{_md_bold_to_html(result['answer'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_entities = len(result["seed_ids"])
    n_nodes = len(result["nodes"])
    n_edges = len(result["edges"])
    votes = result["votes"]
    if votes is not None:
        # agree_count는 majority_vote_answer()의 LLM 클러스터링 결과(핵심 사실
        # 기준 그룹 크기)를 그대로 쓴다 -- 문자열 단순 비교로 세면 "터키의 SSİK가
        # 세웠습니다" vs "Turkey의 SSİK가 세웠습니다"처럼 표현만 다른 같은 사실이
        # 별개로 카운트돼 실제보다 낮은 일치도가 표시된다(2026-07-16, 사용자가
        # 3표가 사실상 같은 내용인데 1/3로 뜨는 걸 발견해 수정).
        consensus_value = f'{result["agree_count"]}<span class="unit">/{len(votes)}표 일치</span>'
    else:
        consensus_value = '<span style="font-size:1rem;">직접 조회</span>'
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-cell"><div class="label">매칭 엔티티</div><div class="value">{n_entities}<span class="unit">개</span></div></div>
            <div class="stat-cell"><div class="label">다수결 일치도</div><div class="value">{consensus_value}</div></div>
            <div class="stat-cell"><div class="label">최종 사실</div><div class="value">{n_edges}<span class="unit">개</span></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label" style="margin-top:6px;">KNOWLEDGE GRAPH MAPPING</div>', unsafe_allow_html=True)
    if result["nodes"]:
        seed_names = {m["name"] for e in result["entity_results"] for m in e["matches"]}
        html = build_graph_html(result["nodes"], result["edges"], seed_names)
        components.html(html, height=380, scrolling=False)
    else:
        st.markdown('<div class="panel muted">탐색된 그래프가 없습니다.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">AUDITABLE ANALYSIS PATH</div>', unsafe_allow_html=True)
    value_suffix = f", 값={result['value']}" if result["value"] else ""
    steps = [
        f'엔티티 {result["entities"] or "없음"}, 관계타입 {result["relation_type"] or "-"}'
        f'{value_suffix}로 질의를 분석했습니다.',
        (
            f'{n_entities}개 엔티티를 Exact→Alias→Word Boundary→Fuzzy 캐스케이드로 링킹했습니다.'
            if n_entities else '그래프에서 매칭되는 엔티티를 찾지 못했습니다.'
        ),
        f'"{route_label}" 경로로 라우팅해 사실 {result["n_before_rerank"]}개를 수집했습니다.',
        (
            f'사실이 80개를 초과해 임베딩 리랭킹으로 상위 {n_edges}개만 사용했습니다.'
            if result["n_before_rerank"] > n_edges else '사실 수가 리랭킹 임계값 이하라 전부 사용했습니다.'
        ),
        (
            '같은 질문을 3회 반복 생성(temperature=0)한 뒤, "모름"이면 evidence 원문 전용 '
            '재시도를 거치고, LLM 클러스터링으로 다수결 답변을 채택했습니다.'
            if result["votes"] is not None else
            '엔티티 링킹이 아무것도 찾지 못하고 속성 스캔으로 이어갈 관계/값도 없어, '
            '결과가 뻔한 답변 생성 호출은 생략하고 바로 안내 문구를 반환했습니다.'
            if result["route"] == "no_seed" else
            '이 경로(1-hop 직접 조회/속성 스캔)는 원래도 답이 안정적이라, 비용 절약을 위해 '
            '3회 샘플링 없이 단발 호출로 답을 생성했습니다.'
        ),
    ]
    steps_html = '<div class="panel">'
    for i, s in enumerate(steps, 1):
        steps_html += f'<div class="step-item"><span class="step-num">{i:02d}</span><span>{s}</span></div>'
    steps_html += '</div>'
    st.markdown(steps_html, unsafe_allow_html=True)

    if result["votes"] is not None:
        st.markdown('<div class="section-label">3회 샘플링 결과</div>', unsafe_allow_html=True)
        votes_html = ""
        for i, v in enumerate(result["votes"], 1):
            chosen = v == result["answer"]
            votes_html += (
                f'<div class="vote-item {"chosen" if chosen else ""}">'
                f'<span class="vote-tag {"chosen" if chosen else "other"}">샘플 {i}{" · 채택" if chosen else ""}</span>'
                f'<div>{_md_bold_to_html(v)}</div></div>'
            )
        st.markdown(votes_html, unsafe_allow_html=True)

    st.markdown(
        f'<div class="section-label"><span class="badge">04</span> EVIDENCE LEDGER '
        f'<span class="muted" style="margin-left:auto;">{n_edges} sources</span></div>',
        unsafe_allow_html=True,
    )
    if not result["edges"]:
        st.markdown('<div class="panel muted">수집된 근거가 없습니다.</div>', unsafe_allow_html=True)
    else:
        # 사실이 많은 질문(허브 인접 질문 등)은 근거가 수십 개까지 나올 수 있어
        # 페이지 전체 스크롤에 묻히기 쉽다 -- 자체 스크롤 영역으로 묶어 페이지
        # 스크롤과 분리(2026-07-16, UX 개선).
        with st.container(height=480, border=False):
            for i, e in enumerate(result["edges"], 1):
                quote = e["evidence"] or "(근거 문장 없음)"
                tag = EVIDENCE_SOURCE_KO.get(e["evidence_source"], e["evidence_source"] or "")
                st.markdown(
                    f"""
                    <div class="evidence-item">
                        <div class="evidence-id">E{i:02d}</div>
                        <div class="evidence-triple">{e['src_name']} → {e['relation']} → {e['dst_name']}</div>
                        <div class="evidence-quote">"{quote}"</div>
                        {f'<span class="evidence-tag">{tag}</span>' if tag else ''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_rag_panel(result):
    st.markdown(
        '<div class="compare-col-title">📄 Naive RAG <span class="muted">문장 벡터 검색</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="card-cream">
            <div style="font-size:0.75rem; font-weight:700;">답변</div>
            <div style="font-size:1.15rem; font-weight:700; margin:10px 0 6px 0;">{_md_bold_to_html(result['answer'])}</div>
            <div class="muted">검색어(영문 번역): {result['query_en']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-label" style="margin-top:6px;">RETRIEVED CHUNKS '
        f'<span class="muted" style="margin-left:auto;">top-{RAG_TOP_K}</span></div>',
        unsafe_allow_html=True,
    )
    if not result["chunks"]:
        st.markdown('<div class="panel muted">검색된 문장이 없습니다.</div>', unsafe_allow_html=True)
    else:
        for i, c in enumerate(result["chunks"], 1):
            st.markdown(
                f"""
                <div class="evidence-item">
                    <div class="evidence-id">C{i:02d} · score {c['score']:.3f}</div>
                    <div class="evidence-quote">[{c['title']}] "{c['text']}"</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_result(item):
    st.markdown('<div class="section-label" style="margin-top:8px;">ANALYSIS COMPLETE</div>', unsafe_allow_html=True)
    st.markdown('<h2 style="margin:0;">분석 결과</h2>', unsafe_allow_html=True)
    # 기본은 GraphRAG 탭만 보이게(st.tabs는 첫 탭이 기본 선택) -- naive RAG 결과는
    # 이미 계산되어 있고(run_naive_rag_analysis가 GraphRAG와 함께 실행됨), "Naive RAG"
    # 탭을 누를 때만 화면에 나타난다. 두 탭 모두 각각 단일 패널만 보여준다(비교는
    # 탭을 오가며 확인).
    tab_graph, tab_rag = st.tabs(["🕸️ GraphRAG", "📄 Naive RAG"])
    with tab_graph:
        render_graphrag_panel(item["graphrag"])
    with tab_rag:
        render_rag_panel(item["rag"])


ROUTE_ICONS = {
    "1hop": "⚡", "1hop_fallback_bfs": "🕸️", "property_scan": "🔍",
    "bfs": "🕸️", "no_seed": "❔",
}


def render_history():
    if not st.session_state.get("history"):
        return
    st.markdown('<div class="section-label" style="margin-top:24px;">🕒 최근 분석</div>', unsafe_allow_html=True)
    with st.container(key="history_row"):
        cols = st.columns(2)
        for i, item in enumerate(reversed(st.session_state.history[-6:])):
            with cols[i % 2]:
                graphrag = item["graphrag"]
                icon = ROUTE_ICONS.get(graphrag["route"], "🕸️")
                answer_preview = graphrag["answer"].replace("**", "").replace("\n", " ")
                if len(answer_preview) > 50:
                    answer_preview = answer_preview[:50] + "…"
                clicked = st.button(
                    f'{icon} {graphrag["question"]}\n\n{answer_preview}',
                    key=f'history_{item["ts"]}',
                    use_container_width=True,
                )
                if clicked:
                    st.session_state.active_result = item
                    st.rerun()


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Trace — Knowledge Graph Workbench", page_icon="🕸️", layout="wide")
    inject_css()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_result" not in st.session_state:
        st.session_state.active_result = None

    try:
        gq.driver.verify_connectivity()
        neo4j_online = True
    except Exception:
        neo4j_online = False

    render_header(neo4j_online)
    render_hero()
    query, run_clicked = render_query_console()

    if run_clicked and query.strip():
        if not neo4j_online:
            st.error(
                "Neo4j에 연결할 수 없습니다. 로컬은 .env, Streamlit Cloud는 앱 Secrets에 "
                "NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD를 설정했는지 확인하세요."
            )
        else:
            with st.spinner("GraphRAG 그래프 탐색 + naive RAG 벡터 검색을 동시에 진행하는 중..."):
                graphrag_result = run_graphrag_analysis(query.strip())
                rag_result = run_naive_rag_analysis(query.strip())
            item = {"graphrag": graphrag_result, "rag": rag_result, "ts": len(st.session_state.history)}
            st.session_state.history.append(item)
            st.session_state.active_result = item

    if st.session_state.active_result:
        render_result(st.session_state.active_result)

    render_history()


if __name__ == "__main__":
    main()
