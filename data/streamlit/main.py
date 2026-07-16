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
from cache import cache_key, cached  # noqa: E402

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

# 위 ROUTE_LABELS_KO는 기술 상세용(라우팅 로직을 정확히 감사하려는 사용자용) --
# 일반 사용자에게 보여주는 요약 문장에는 이 쉬운 설명을 대신 쓴다(2026-07-16,
# "너무 어렵고 난해하다"는 피드백으로 추가).
ROUTE_PLAIN_KO = {
    "1hop": "바로 연결된 정보에서 찾았어요",
    "1hop_fallback_bfs": "바로 연결된 정보엔 없어서, 몇 단계를 더 건너가며 찾았어요",
    "property_scan": "조건에 맞는 대상을 그래프 전체에서 찾았어요",
    "bfs": "여러 단계를 거쳐 연결된 정보까지 넓게 찾았어요",
    "no_seed": "질문에 맞는 대상을 찾지 못했어요",
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
    검색해야 벡터 유사도가 잘 나옴).

    graphrag_query.py의 cached()로 감쌈 -- temperature=0이라 같은 질문은 항상
    같은 번역이 나오는데, 이 파일만 그 캐싱 관례를 안 따르고 있었음(2026-07-16,
    감사에서 발견). 추천 질문 칩을 데모 중 다시 눌러도 API를 또 부르지 않는다."""
    def compute():
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

    return cached(cache_key("st_translate_to_english", CHAT_MODEL, question), compute)


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

    def compute_answer():
        prompt = (
            "아래 컨텍스트만 근거로 질문에 한국어로 답하세요. 컨텍스트에 답이 없거나 "
            "근거가 불충분하면 반드시 '모름'이라고만 답하세요.\n\n"
            f"컨텍스트:\n{context_block}\n\n질문: {question}"
        )
        resp = gq.openai_client.chat.completions.create(
            model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        return resp.choices[0].message.content.strip()

    # translate_to_english()와 같은 이유로 캐싱 -- context_block을 키에 포함시켜
    # (질문은 같아도) 검색된 청크가 바뀌면 캐시를 새로 씀.
    answer = cached(cache_key("st_naive_rag_answer", CHAT_MODEL, question, context_block), compute_answer)

    return {
        "question": question,
        "query_en": query_en,
        "chunks": chunks,
        "answer": answer,
    }


# ---------------------------------------------------------------------------
# 그래프 시각화
# ---------------------------------------------------------------------------

def build_graph_html(nodes, edges, seed_names):
    net = Network(
        height="440px", width="100%", bgcolor="#131c11", font_color="#eee8d8",
        directed=True,
        # "remote"였던 걸 "in_line"으로 -- vis-network.js를 생성된 HTML 안에 그대로
        # 박아 넣어서 CDN 접속이 필요 없다. 이 앱은 교수/평가자 앞 발표에서 쓰이는데
        # (PRODUCT.md), 발표 장소 네트워크가 CDN을 막거나 느리면 그래프 패널 전체가
        # 조용히 안 뜨는 위험이 있었음(2026-07-16, 감사에서 발견).
        cdn_resources="in_line",
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

    html = net.generate_html(notebook=False)
    # pyvis 기본 동작은 그래프가 다 안정된(stabilized) 뒤에도 barnes_hut 물리
    # 시뮬레이션을 끄지 않고 계속 돌린다 -- 화면이 이미 멈춰 보여도 탭이 열려있는
    # 동안 CPU를 계속 씀(2026-07-16, 감사에서 발견). pyvis는 이 이벤트에 파이썬
    # API가 없어서, 생성된 HTML에 vis-network의 stabilizationIterationsDone
    # 이벤트를 직접 붙이는 스크립트를 덧붙인다 (템플릿의 전역 `network` 변수를 그대로 씀).
    html = html.replace(
        "</body>",
        "<script>"
        "(function(){"
        "var n=0;"
        "var t=setInterval(function(){"
        "n++;"
        "if (typeof network !== 'undefined') {"
        "clearInterval(t);"
        "network.once('stabilizationIterationsDone', function(){ network.setOptions({physics:false}); });"
        # pyvis 내부 변수명이 언젠가 바뀌거나 스크립트 에러로 network가 끝까지
        # 안 잡히는 경우까지 대비해 10초(100회) 뒤엔 그냥 포기하고 멈춤
        # (재감사에서 발견 -- 원래는 못 찾으면 무한 폴링).
        "} else if (n > 100) {"
        "clearInterval(t);"
        "}"
        "}, 100);"
        "})();"
        "</script></body>",
    )
    return html


# ---------------------------------------------------------------------------
# 스타일
# ---------------------------------------------------------------------------

def inject_css():
    st.markdown(
        """
        <style>
        /* 디자인 토큰 -- amber 강조색 하나가 12곳 넘게 하드코딩돼 있던 것을
        비롯해 색이 전부 리터럴로 흩어져 있던 문제를 정리(2026-07-16, 감사에서
        발견). 값 자체는 전부 기존 그대로이고(팔레트 변경 아님), 같은 값을 쓰던
        곳을 변수 하나로 묶었을 뿐. TYPE_COLORS(엔티티 타입 색)는 pyvis 그래프에
        Python에서 인라인 스타일로 직접 넣어야 해서 이 CSS 토큰과는 별개로 유지. */
        :root {
            --bg-grad-start: #12190f;
            --bg-grad-end: #161f13;
            --surface-panel: #1a2417;
            --surface-sunken: #171f14;
            --surface-tag: #2a3524;
            --surface-card: #efe8d5;
            --surface-input: #f8f5ec;
            --border: #33402c;
            --border-input: #cfc6a8;

            --text: #eee8d8;
            --text-bright: #f3efe1;
            --text-label: #b7c2a8;
            --text-muted: #9aa38f;
            --text-soft: #a9b39c;
            --text-dim: #6b7362;
            --text-body: #cdd5c1;
            --text-on-accent: #1c2418;

            --accent: #d9a441;
            --accent-hover: #e8b658;
            --accent-chosen-bg: #24230f;
            --rag-blue: #7a9cc6;
            --danger: #e07a5f;
            --online: #7fbf7f;
        }

        .stApp {
            background: linear-gradient(180deg, var(--bg-grad-start) 0%, var(--bg-grad-end) 100%);
            color: var(--text);
        }
        [data-testid="stHeader"] { background: transparent; }
        /* !important 제거 -- .section-label/.compare-col-title처럼 이 태그를
        실제로 쓰는 클래스가 자기 색을 지정할 수 있어야 함(2026-07-16, 감사에서
        발견: !important가 있으면 heading 태그로 바꾸는 순간 클래스의 muted 색이
        전부 밝은 색으로 덮여버림). */
        h1, h2, h3, h4 { color: var(--text-bright); }
        p, span, label, div { color: var(--text); }

        .sr-only {
            position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px;
            overflow: hidden; clip: rect(0, 0, 0, 0); white-space: nowrap; border: 0;
        }

        .trace-topbar {
            display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
            padding: 8px 4px 20px 4px; border-bottom: 1px solid var(--border); margin-bottom: 8px;
        }
        .trace-logo {
            margin: 0 !important; font-weight: 800 !important; letter-spacing: 2px !important;
            font-size: 1.1rem !important; color: var(--text-bright) !important; line-height: 1.4 !important;
        }
        .trace-logo .sub { font-weight: 400; color: var(--text-muted); font-size: 0.8rem; margin-left: 8px; }
        .trace-status { font-size: 0.75rem; color: var(--text-muted); }
        .trace-status .dot { color: var(--online); }

        /* 값마다 !important -- Streamlit이 markdown 안의 진짜 h1~h4 태그에
        자체 기본 타이포그래피(큰 폰트 크기 등)를 얹어서, 이 클래스만으로는
        안 이겨 시각적으로 완전히 달라짐(2026-07-16, div->heading 전환 후
        실측 발견). 이 파일이 이미 다른 곳(.entity-chip 등)에서 쓰는 것과
        같은 패턴. */
        .section-label {
            display: flex !important; align-items: center; gap: 8px;
            font-size: 0.78rem !important; font-weight: 400 !important;
            letter-spacing: 1.5px !important; color: var(--text-label) !important;
            text-transform: uppercase !important; margin: 0 0 6px 0 !important;
            line-height: 1.4 !important;
        }
        .badge {
            background: var(--accent); color: var(--text-on-accent); font-weight: 700;
            border-radius: 4px; padding: 1px 7px; font-size: 0.72rem;
        }
        .badge.rag { background: var(--rag-blue); }
        /* route_label 배지 전용 -- 예전엔 .evidence-tag를 재사용하면서 배경만
        인라인으로 amber로 덮어썼는데, .evidence-tag의 color:var(--text-body) !important가
        그대로 남아 옅은 크림색 글자가 amber 배경 위에 얹혀 대비 ~1.49:1로 사실상
        안 읽히는 버그였음(2026-07-16, 감사에서 발견). .vote-tag.chosen처럼 amber
        배경엔 어두운 글자를 쓰는 전용 클래스로 분리. */
        .route-badge {
            display: inline-block; background: var(--accent); color: var(--text-on-accent);
            font-weight: 700; border-radius: 4px; padding: 1px 7px; font-size: 0.72rem;
        }

        .hero-title { font-size: 2.4rem; line-height: 1.25; font-weight: 700; color: var(--text-bright); margin: 4px 0 10px 0; }
        .hero-title .accent { color: inherit; }
        .hero-sub { color: var(--text-soft); font-size: 0.95rem; max-width: 620px; }

        .pipeline-box { font-size: 0.85rem; }
        .pipeline-item { display: flex; gap: 8px; padding: 4px 0; color: var(--text-body); }
        .pipeline-item .n { color: var(--accent); font-weight: 700; }

        .card-cream {
            background: var(--surface-card); color: var(--text-on-accent); border-radius: 12px;
            padding: 22px 26px; margin: 10px 0 18px 0;
        }
        .card-cream * { color: var(--text-on-accent); }
        .muted { color: var(--text-dim) !important; font-size: 0.8rem; }

        .stat-row {
            display: flex; flex-wrap: wrap; gap: 0; border: 1px solid var(--border); border-radius: 10px;
            overflow: hidden; margin-bottom: 18px;
        }
        .stat-cell {
            flex: 1 1 140px; padding: 14px 18px; border-right: 1px solid var(--border);
        }
        .stat-cell:last-child { border-right: none; }
        /* 좁은 화면(발표용 데스크톱이 주 대상이지만, 창을 작게 띄우거나 노트북
        화면에서 보는 경우까지 감안)에서는 3칸이 한 줄에 다 안 들어가면 라벨이
        구겨지는 대신 세로로 쌓이게 함(2026-07-16, 감사에서 발견). */
        @media (max-width: 640px) {
            .stat-cell { flex: 1 1 100%; border-right: none; border-bottom: 1px solid var(--border); }
            .stat-cell:last-child { border-bottom: none; }
        }
        .stat-cell .label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; }
        .stat-cell .value { font-size: 1.6rem; font-weight: 700; color: var(--text-bright); }
        .stat-cell .value .unit { font-size: 0.85rem; color: var(--text-muted); font-weight: 400; margin-left: 4px; }

        .panel {
            background: var(--surface-panel); border: 1px solid var(--border); border-radius: 12px;
            padding: 18px 20px; margin-bottom: 16px; height: 100%;
        }
        .panel * { color: inherit; }

        .entity-chip {
            display: inline-block; background: var(--accent); color: var(--text-on-accent) !important;
            font-weight: 700; font-size: 0.7rem; border-radius: 4px; padding: 2px 7px; margin-right: 8px;
        }
        .entity-item { padding: 10px 0; border-bottom: 1px solid var(--surface-tag); }
        .entity-item:last-child { border-bottom: none; }
        .entity-name { font-weight: 700; color: var(--text-bright); }
        .entity-quote { font-size: 0.78rem; color: var(--text-muted); }

        .step-item { display: flex; gap: 10px; padding: 8px 0; font-size: 0.85rem; color: var(--text-body); }
        .step-num { color: var(--accent); font-weight: 700; min-width: 18px; }

        .evidence-item {
            border: 1px solid var(--border); border-radius: 10px; padding: 14px 18px; margin-bottom: 10px;
            background: var(--surface-sunken);
        }
        .evidence-id { color: var(--accent); font-weight: 700; font-size: 0.75rem; }
        .evidence-triple { font-weight: 700; color: var(--text-bright); margin: 4px 0; }
        .evidence-quote { color: var(--text-soft); font-size: 0.82rem; font-style: italic; }
        .evidence-tag {
            display: inline-block; background: var(--surface-tag); color: var(--text-body) !important;
            font-size: 0.68rem; border-radius: 4px; padding: 2px 7px; margin-top: 6px; margin-right: 6px;
        }

        .vote-item {
            border: 1px solid var(--border); border-radius: 8px; padding: 10px 14px; margin-bottom: 6px;
            font-size: 0.82rem; background: var(--surface-sunken);
        }
        .vote-item.chosen { border-color: var(--accent); background: var(--accent-chosen-bg); }
        .vote-tag {
            display: inline-block; font-size: 0.65rem; font-weight: 700; border-radius: 4px;
            padding: 1px 6px; margin-bottom: 4px;
        }
        .vote-tag.chosen { background: var(--accent); color: var(--text-on-accent); }
        .vote-tag.other { background: var(--border); color: var(--text-body); }

        /* st.error()의 기본 빨간 배너는 이 페이지 어디에도 없는 스타일이라, 뭔가
        잘못됐을 때(데모 중 가장 안 좋은 타이밍)만 테마와 안 맞는 화면이 튀어나옴
        (2026-07-16, 감사에서 발견). 나머지 카드와 같은 어휘(전체 테두리 색으로
        상태 표시, .vote-item.chosen과 동일한 패턴)로 대체. */
        .error-card {
            background: var(--surface-sunken); border: 1px solid var(--danger); border-radius: 10px;
            padding: 16px 20px; margin-bottom: 16px;
        }
        .error-card .title { color: var(--danger); font-weight: 700; font-size: 0.85rem; margin-bottom: 4px; }
        .error-card .body { color: var(--text); font-size: 0.9rem; }

        .compare-col-title {
            font-weight: 800 !important; font-size: 1rem !important; letter-spacing: 1px !important;
            color: var(--text) !important; margin: 0 0 10px 0 !important; line-height: 1.4 !important;
            display: flex !important; align-items: center; gap: 8px;
        }

        /* st.container(key=...)가 생성하는 wrapper(stVerticalBlock 자체)에 카드 스타일 적용 */
        div.st-key-query_console {
            background: var(--surface-card); border-radius: 12px; padding: 22px 26px 14px 26px; margin: 10px 0 18px 0;
            gap: 0.6rem !important;
        }
        div.st-key-query_console textarea {
            background: var(--surface-input) !important; color: var(--text-on-accent) !important;
            border: 1px solid var(--border-input) !important;
        }
        div.st-key-query_console p { color: var(--text-dim) !important; margin: 0 !important; }
        div.st-key-chip_row button {
            background: transparent !important; color: var(--text-soft) !important; border: none !important;
            font-weight: 500; font-size: 0.78rem; white-space: normal; text-align: left !important;
            justify-content: flex-start !important;
            /* 4px -> 10px: 실제 탭 가능 높이가 ~20px로 WCAG 2.2 SC 2.5.8(24x24px, AA)
            미만이었음(2026-07-16, 감사에서 발견). 왼쪽 정렬/투명 배경 등 나머지
            디자인은 그대로 두고 세로 패딩만 키움. */
            padding: 10px 0 !important;
        }
        div.st-key-chip_row button p { text-align: left !important; }
        div.st-key-chip_row button:hover { color: var(--accent) !important; }

        div.st-key-history_row button {
            background: var(--surface-sunken) !important; color: var(--text-body) !important;
            border: 1px solid var(--border) !important;
            text-align: left !important; justify-content: flex-start !important; white-space: pre-line !important;
            font-weight: 500; line-height: 1.5; padding: 10px 14px;
        }
        div.st-key-history_row button:hover { border-color: var(--accent) !important; }
        div.st-key-history_row button p { text-align: left !important; }

        div.stButton > button {
            background: var(--accent); color: var(--text-on-accent); border: none; font-weight: 700; border-radius: 8px;
        }
        div.stButton > button:hover { background: var(--accent-hover); color: var(--text-on-accent); }

        button[data-baseweb="tab"] {
            background: transparent !important; color: var(--text-muted) !important; font-weight: 700 !important;
            flex: 1 1 0 !important; justify-content: center !important;
        }
        button[data-baseweb="tab"] p { color: inherit !important; font-size: 0.95rem !important; }
        button[data-baseweb="tab"][aria-selected="true"] { color: var(--accent) !important; }
        button[data-baseweb="tab"][aria-selected="true"] p { color: var(--accent) !important; }
        div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
        div[data-baseweb="tab-border"] { background-color: var(--border) !important; }
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


def render_error(message):
    """st.error()의 기본 빨간 배너 대신 테마에 맞는 에러 카드 (2026-07-16, 감사에서
    발견 -- 나머지 화면과 완전히 다른 스타일이 연결 실패처럼 상태가 안 좋을 때만
    튀어나오는 게 특히 나빴음). role="alert"를 붙여서(재감사에서 발견) 스크린
    리더가 카드가 나타나는 순간 자동으로 읽어주게 함 -- 일반 div는 그냥 지나칠 수 있음."""
    st.markdown(
        f"""
        <div class="error-card" role="alert">
            <div class="title">연결 오류</div>
            <div class="body">{message}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header(neo4j_online):
    # 정상 연결(기본 상태)일 땐 상태 표시를 아예 안 보여준다 -- 항상 켜져 있는
    # "ONLINE" 배지는 매번 같은 값이라 실질 정보가 없다는 사용자 피드백(2026-07-16).
    # 실제로 끊겼을 때만(드문, 행동이 필요한 상태) 눈에 띄게 알린다. 상태 표시
    # 자체는 이모지 대신 기존 .dot 패턴(원래 CSS에 정의만 돼 있고 안 쓰이던 클래스)을
    # 재사용 -- 감사에서 이모지가 기능 아이콘으로 곳곳에 쓰인 것을 정리하며 통일.
    status_html = (
        '<div class="trace-status" style="color:var(--danger);">'
        '<span class="dot" style="color:var(--danger);">●</span> KNOWLEDGE GRAPH OFFLINE</div>'
        if not neo4j_online else ""
    )
    st.markdown(
        f"""
        <div class="trace-topbar">
            <h1 class="trace-logo">TRACE <span class="sub">| KG WORKBENCH</span></h1>
            {status_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    st.markdown('<h2 class="section-label"><span class="badge">01</span> QUERY INTELLIGENCE CONSOLE</h2>', unsafe_allow_html=True)
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
        '<h3 class="compare-col-title">GraphRAG <span class="muted">관계 그래프 탐색</span></h3>',
        unsafe_allow_html=True,
    )

    route_label = ROUTE_LABELS_KO.get(result["route"], result["route"])
    st.markdown(
        f"""
        <div class="card-cream">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="font-size:0.75rem; font-weight:700;">다수결 최종 답변</div>
                <div class="route-badge">{route_label}</div>
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

    st.markdown('<h4 class="section-label" style="margin-top:6px;">KNOWLEDGE GRAPH MAPPING</h4>', unsafe_allow_html=True)
    if result["nodes"]:
        seed_names = {m["name"] for e in result["entity_results"] for m in e["matches"]}
        # 그래프 시각화(캔버스 기반 iframe)는 스크린리더에 아무 정보도 안 남는다 --
        # 노드/엣지 개수와 타입별 개체 목록을 시각적으로 숨긴 텍스트로 옆에 둬서
        # 최소한의 텍스트 대안을 제공(2026-07-16, 감사에서 발견).
        type_summary = {}
        for n in result["nodes"].values():
            type_summary.setdefault(TYPE_LABELS_KO.get(n["type"], n["type"]), []).append(n["name"])
        graph_summary = (
            f'그래프 시각화: 노드 {len(result["nodes"])}개, 관계 {len(result["edges"])}개. ' +
            " / ".join(f'{label} {len(names)}개({", ".join(names)})' for label, names in type_summary.items())
        )
        st.markdown(f'<div class="sr-only">{graph_summary}</div>', unsafe_allow_html=True)
        html = build_graph_html(result["nodes"], result["edges"], seed_names)
        components.html(html, height=380, scrolling=False)
    else:
        st.markdown('<div class="panel muted">탐색된 그래프가 없습니다.</div>', unsafe_allow_html=True)

    st.markdown('<h4 class="section-label">이렇게 답을 찾았어요</h4>', unsafe_allow_html=True)

    route_plain = ROUTE_PLAIN_KO.get(result["route"], "그래프를 탐색했어요")
    entity_word = f'"{", ".join(result["entities"])}"' if result["entities"] else "질문 속 대상"
    if result["route"] == "no_seed":
        summary_line = f'{entity_word}을(를) 그래프에서 찾지 못했어요.'
    else:
        summary_line = (
            f'{entity_word}에 대해 {route_plain} — 관련 사실 {n_edges}개를 근거로 답변을 만들었어요.'
        )
    st.markdown(f'<p style="font-weight:600; margin:4px 0 14px;">{summary_line}</p>', unsafe_allow_html=True)

    plain_steps = [
        f'{entity_word}에 대한 질문으로 이해했어요.',
        (
            f'그래프에서 {entity_word}과(와) 정확히 일치하는 대상을 찾았어요.'
            if n_entities else '그래프에서 일치하는 대상을 찾지 못했어요.'
        ),
        f'{route_plain} (관련 사실 {result["n_before_rerank"]}개 발견).',
        (
            f'찾은 사실이 많아서, 질문과 가장 관련 높은 {n_edges}개만 추려서 썼어요.'
            if result["n_before_rerank"] > n_edges else '찾은 사실이 많지 않아서, 전부 답변에 사용했어요.'
        ),
        (
            '같은 질문에 3번 답해보고, 가장 일관되게 나온 답을 최종 답변으로 골랐어요.'
            if result["votes"] is not None else
            '찾는 대상 자체가 없어서, 답을 만들지 않고 안내 문구만 보여드렸어요.'
            if result["route"] == "no_seed" else
            '이런 유형의 질문은 답이 안정적으로 나오는 편이라, 한 번만 답변을 만들었어요.'
        ),
    ]
    plain_html = '<div class="panel">'
    for i, s in enumerate(plain_steps, 1):
        plain_html += f'<div class="step-item"><span class="step-num">{i:02d}</span><span>{s}</span></div>'
    plain_html += '</div>'
    st.markdown(plain_html, unsafe_allow_html=True)

    with st.expander("🔧 기술적으로 어떻게 처리됐는지 보기"):
        value_suffix = f", 값={result['value']}" if result["value"] else ""
        tech_steps = [
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
        for i, s in enumerate(tech_steps, 1):
            st.markdown(f'<div class="step-item"><span class="step-num">{i:02d}</span><span>{s}</span></div>', unsafe_allow_html=True)

    if result["votes"] is not None:
        st.markdown('<h4 class="section-label">3회 샘플링 결과</h4>', unsafe_allow_html=True)
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
        f'<h4 class="section-label"><span class="badge">04</span> EVIDENCE LEDGER '
        f'<span class="muted" style="margin-left:auto;">{n_edges} sources</span></h4>',
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
        '<h3 class="compare-col-title">Naive RAG <span class="muted">문장 벡터 검색</span></h3>',
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
        f'<h4 class="section-label" style="margin-top:6px;">RETRIEVED CHUNKS '
        f'<span class="muted" style="margin-left:auto;">top-{RAG_TOP_K}</span></h4>',
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
    tab_graph, tab_rag = st.tabs(["GraphRAG", "Naive RAG"])
    with tab_graph:
        render_graphrag_panel(item["graphrag"])
    with tab_rag:
        render_rag_panel(item["rag"])


def _route_icon(route):
    """히스토리 미리보기용 짧은 마커. 전용 이모지 사전 대신 ROUTE_LABELS_KO의
    앞머리 기호(①/②'/③)를 그대로 재사용해 라우팅 어휘를 한 곳에서만 관리한다
    (2026-07-16, 감사에서 이모지가 기능 라벨로 곳곳에 흩어져 있던 것을 정리하며
    발견 -- 별도 이모지 사전을 만들 이유가 없었음)."""
    label = ROUTE_LABELS_KO.get(route, "")
    return label.split(" ", 1)[0] if label else "?"


def render_history():
    if not st.session_state.get("history"):
        return
    st.markdown('<h2 class="section-label" style="margin-top:24px;">최근 분석</h2>', unsafe_allow_html=True)
    with st.container(key="history_row"):
        cols = st.columns(2)
        for i, item in enumerate(reversed(st.session_state.history[-6:])):
            with cols[i % 2]:
                graphrag = item["graphrag"]
                icon = _route_icon(graphrag["route"])
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
            render_error(
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
