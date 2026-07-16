"""Trace — Knowledge Graph Workbench

자연어 질문 -> LLM으로 의도/엔티티 추출 -> Neo4j 지식그래프(DocRED)에서 엔티티
매핑 및 N-hop 서브그래프 탐색 -> 서브그래프 사실만 근거로 LLM이 답변하는
GraphRAG 데모 앱. `RAG/graphrag_query.py`와 같은 파이프라인(엔티티 추출 ->
그래프 매핑 -> N-hop 탐색 -> 근거 기반 답변)을 쓰되, 이 앱은 UI에 필요한 추가
정보(엔티티 매칭 점수, 엣지 confidence/evidence_source, 그래프 시각화용 노드
타입 등)를 함께 조회하고 화면에 표시하기 위해 별도로 파이프라인을 구현한다.

'생각 과정'은 LLM의 원문 추론을 그대로 노출하지 않고, 실제 실행된 처리
단계(의도 분류 -> 엔티티 정규화 -> N-hop 조회 -> 정확도 산출)를 실제 수치와
함께 감사 가능한(auditable) 형태로 보여준다. '정확도'는 정답 라벨과 비교한
정확도가 아니라, 엔티티 매칭 점수와 그래프 엣지 confidence를 결합한 자체
신뢰도 점수다.

실행:
    streamlit run data/streamlit/main.py

필요 환경변수 (레포 루트 `.env`): OPENAI_API_KEY, NEO4J_URI, NEO4J_USERNAME,
NEO4J_PASSWORD, NEO4J_DATABASE(선택).
"""

import difflib
import json
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from pyvis.network import Network

ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

CHAT_MODEL = "gpt-5.4-mini"
SEEDS_PER_MENTION = 3
MAX_HOPS = 2
FACT_LIMIT_PER_HOP = 80
HUB_DEGREE_THRESHOLD = 40
FALLBACK_CONFIDENCE = 0.40
MAX_QUERY_CHARS = 600

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
    "annotated": "원문 근거",
    "inferred_cooccurrence": "문장 동시 등장 추론",
    "unresolved_multihop": "멀티홉 (근거 미확정)",
}

SUGGESTED_QUESTIONS = [
    "AirAsia Zest의 본사는 어디이며, 그 도시는 어느 행정구역에 속하나요?",
    "Roketsan은 몇 년에 설립됐고, 어느 위원회가 세웠어?",
    "Health Sciences Centre가 있는 도시에 있는 대학교는 몇 년에 설립됐어?",
]


# ---------------------------------------------------------------------------
# 리소스 (캐시)
# ---------------------------------------------------------------------------

@st.cache_resource
def get_openai_client():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


@st.cache_resource
def get_neo4j_driver():
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    driver.verify_connectivity()
    return driver


NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def classify_query(openai_client, question):
    """질문 의도를 짧은 한국어 명사구로 분류하고, 그래프 탐색 시작점이 될
    고유명사 엔티티 멘션을 함께 추출한다 (LLM 호출 1회로 묶어 비용 절감)."""
    prompt = (
        "다음 질문을 분석해 JSON으로만 답하세요. 다른 설명은 절대 출력하지 마세요.\n"
        '형식: {"intent": "질문 의도를 15자 이내 한국어 명사구로 요약 (예: '
        '\'위치 및 행정구역 관계 탐색\', \'설립연도 조회\')", '
        '"entities": ["지식그래프 탐색의 시작점이 될 고유명사(회사/기관/인물/지명 등)", ...]}\n'
        f"질문: {question}"
    )
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(text)
        return parsed.get("intent", "일반 질의"), parsed.get("entities", [])
    except (json.JSONDecodeError, AttributeError):
        return "일반 질의", []


def find_seed_entities(session, mention):
    """엔티티 이름/별칭에 (대소문자 무시) 부분일치하는 노드를 찾고, 멘션과의
    문자열 유사도를 매칭 점수로 함께 반환한다."""
    query = """
    MATCH (e:ZEntity)
    WHERE toLower(e.name) CONTAINS toLower($mention)
       OR any(alias IN e.aliases WHERE toLower(alias) CONTAINS toLower($mention))
    RETURN e.id AS id, e.name AS name, e.type AS type
    LIMIT $limit
    """
    rows = list(session.run(query, mention=mention, limit=SEEDS_PER_MENTION))
    results = []
    for r in rows:
        score = difflib.SequenceMatcher(None, mention.lower(), r["name"].lower()).ratio()
        results.append({"id": r["id"], "name": r["name"], "type": r["type"], "score": score})
    return results


def expand_subgraph(session, seed_ids):
    """seed 노드에서 최대 MAX_HOPS까지 관계를 BFS로 펼쳐 사실(엣지) 목록과
    등장한 노드 목록을 함께 수집한다.

    매 hop마다 그 hop에서 새로 만난 노드 중 허브 노드(연결이
    HUB_DEGREE_THRESHOLD 초과)는 다음 hop의 확장 출발점에서 제외한다 —
    안 그러면 국가/대륙처럼 연결이 수백 개인 노드를 지나가는 순간 무관한
    사실이 쏟아져 정작 필요한 경로가 LIMIT 안에 못 들어간다.
    """
    edges = []
    nodes = {}
    if not seed_ids:
        return edges, nodes, 0

    visited = set(seed_ids)
    frontier = seed_ids
    depth_reached = 0
    seen_rel_ids = set()

    for hop in range(MAX_HOPS):
        rows = list(session.run(
            """
            MATCH (h:ZEntity)-[r]-(n:ZEntity)
            WHERE h.id IN $frontier
            RETURN DISTINCT
                elementId(r) AS rel_id,
                startNode(r).id AS src_id, startNode(r).name AS src_name, startNode(r).type AS src_type,
                type(r) AS relation, r.confidence AS confidence, r.evidence AS evidence,
                r.evidence_source AS evidence_source,
                endNode(r).id AS dst_id, endNode(r).name AS dst_name, endNode(r).type AS dst_type,
                n.id AS next_id, COUNT { (n)--() } AS next_degree
            LIMIT $limit
            """,
            frontier=frontier, limit=FACT_LIMIT_PER_HOP,
        ))
        if not rows:
            break
        depth_reached = hop + 1

        for r in rows:
            nodes[r["src_id"]] = {"name": r["src_name"], "type": r["src_type"]}
            nodes[r["dst_id"]] = {"name": r["dst_name"], "type": r["dst_type"]}
            # 양방향 매칭이라 hop이 진행되며 같은 관계 인스턴스를 반대편에서 다시
            # 만날 수 있다 (예: A의 이웃으로 B를 찾고, 다음 hop에서 B의 이웃으로
            # 같은 A-B 관계를 다시 찾음). Neo4j 관계 고유 id로 중복을 막는다.
            if r["rel_id"] in seen_rel_ids:
                continue
            seen_rel_ids.add(r["rel_id"])
            edges.append({
                "src_id": r["src_id"], "src_name": r["src_name"],
                "relation": r["relation"],
                "dst_id": r["dst_id"], "dst_name": r["dst_name"],
                "confidence": r["confidence"] if r["confidence"] is not None else 1.0,
                "evidence": " ".join(r["evidence"]) if r["evidence"] else None,
                "evidence_source": r["evidence_source"],
            })

        frontier = [
            r["next_id"] for r in rows
            if r["next_id"] not in visited and r["next_degree"] <= HUB_DEGREE_THRESHOLD
        ]
        visited.update(frontier)
        if not frontier:
            break

    return edges, nodes, depth_reached


def synthesize_answer(openai_client, question, edges):
    if not edges:
        context_block = "(그래프에서 관련 사실을 찾지 못함)"
    else:
        lines = []
        for e in edges:
            line = f"- {e['src_name']} -[{e['relation']}]-> {e['dst_name']}"
            if e["evidence"]:
                line += f" (근거: {e['evidence']})"
            lines.append(line)
        context_block = "\n".join(lines)

    prompt = (
        "아래는 지식그래프에서 가져온 사실(head -[relation]-> tail) 목록이고, 일부는 "
        "근거 문장이 같이 달려 있습니다. 이 사실들만 근거로 질문에 한국어로 답하세요. "
        "여러 사실을 연결해 추론해도 됩니다. 근거가 불충분하면 반드시 '모름'이라고만 "
        "답하세요. 2~3문장 이내로 간결하게 답하세요.\n\n"
        f"사실 목록:\n{context_block}\n\n질문: {question}"
    )
    resp = openai_client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()


def compute_confidence(entity_matches, edges):
    """엔티티 매칭 점수와 그래프 엣지 confidence를 결합한 자체 신뢰도 점수
    (0~1). 정답 라벨과 비교하는 정확도가 아니라, '이 답변이 그래프 근거에
    얼마나 단단히 뒷받침되는가'를 나타내는 지표."""
    if not edges:
        return FALLBACK_CONFIDENCE

    entity_score = (
        sum(m["score"] for m in entity_matches) / len(entity_matches)
        if entity_matches else 0.5
    )
    edge_score = sum(e["confidence"] for e in edges) / len(edges)
    return 0.3 * entity_score + 0.7 * edge_score


def run_analysis(question):
    openai_client = get_openai_client()
    driver = get_neo4j_driver()

    intent, mentions = classify_query(openai_client, question)

    entity_results = []
    seed_ids = []
    with driver.session(database=NEO4J_DATABASE) as session:
        for mention in mentions:
            matches = find_seed_entities(session, mention)
            entity_results.append({"mention": mention, "matches": matches})
            seed_ids.extend(m["id"] for m in matches)

        seed_ids = list(dict.fromkeys(seed_ids))  # dedupe, preserve order
        edges, nodes, depth_reached = expand_subgraph(session, seed_ids)

    answer = synthesize_answer(openai_client, question, edges)
    all_matches = [m for e in entity_results for m in e["matches"]]
    confidence = compute_confidence(all_matches, edges)

    return {
        "question": question,
        "intent": intent,
        "entity_results": entity_results,
        "seed_ids": seed_ids,
        "nodes": nodes,
        "edges": edges,
        "depth_reached": depth_reached,
        "answer": answer,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# 그래프 시각화
# ---------------------------------------------------------------------------

def build_graph_html(nodes, edges, seed_ids):
    net = Network(
        height="440px", width="100%", bgcolor="#131c11", font_color="#eee8d8",
        directed=True, cdn_resources="remote",
    )
    net.barnes_hut(gravity=-3500, central_gravity=0.25, spring_length=130, spring_strength=0.02)

    for node_id, info in nodes.items():
        is_seed = node_id in seed_ids
        color = TYPE_COLORS.get(info["type"], "#a8a397")
        net.add_node(
            node_id,
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
        pair = (e["src_id"], e["dst_id"], e["relation"])
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        net.add_edge(
            e["src_id"], e["dst_id"],
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
        .entity-score { float: right; color: #d9a441; font-weight: 700; }

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
        .evidence-conf { float: right; text-align: right; }
        .evidence-conf .n { font-size: 1.1rem; font-weight: 700; color: #d9a441; }
        .evidence-conf .l { font-size: 0.65rem; color: #9aa38f; display: block; }

        .history-item {
            display: flex; justify-content: space-between; align-items: center;
            padding: 10px 16px; border: 1px solid #33402c; border-radius: 8px; margin-bottom: 6px;
            font-size: 0.85rem; color: #cdd5c1;
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
            font-weight: 500; font-size: 0.78rem; white-space: normal; height: 100%;
        }
        div.st-key-chip_row button:hover { border-color: #d9a441 !important; color: #d9a441 !important; }
        .history-conf { color: #d9a441; font-weight: 700; }

        div.stButton > button {
            background: #d9a441; color: #1c2418; border: none; font-weight: 700; border-radius: 8px;
        }
        div.stButton > button:hover { background: #e8b658; color: #1c2418; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 렌더링
# ---------------------------------------------------------------------------

def render_header(neo4j_online):
    status = "KNOWLEDGE GRAPH ONLINE" if neo4j_online else "KNOWLEDGE GRAPH OFFLINE"
    dot = "🟢" if neo4j_online else "🔴"
    st.markdown(
        f"""
        <div class="trace-topbar">
            <div class="trace-logo">TRACE <span class="sub">| KG WORKBENCH</span></div>
            <div class="trace-status">{dot} {status} &nbsp;|&nbsp; V0.1</div>
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
            '<div class="hero-sub">자연어 질문에서 엔티티를 추출하고, DocRED 지식그래프를 '
            '탐색해 검증 가능한 관계와 근거를 반환합니다.</div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="pipeline-box">
            <div class="muted" style="letter-spacing:1px;">PIPELINE</div>
            <div class="pipeline-item"><span class="n">01</span> 구조 분석</div>
            <div class="pipeline-item"><span class="n">02</span> 엔티티 정규화</div>
            <div class="pipeline-item"><span class="n">03</span> 2-hop 그래프 탐색</div>
            <div class="pipeline-item"><span class="n">04</span> 근거 기반 결과 합성</div>
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
            chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
            for i, sq in enumerate(SUGGESTED_QUESTIONS):
                with chip_cols[i]:
                    st.button(f"{i+1} · {sq}", key=f"chip_{i}", on_click=_set_query, args=(sq,))
    return query, run_clicked


def render_result(result):
    st.markdown('<div class="section-label" style="margin-top:8px;">ANALYSIS COMPLETE</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown('<h2 style="margin:0;">그래프 분석 결과</h2>', unsafe_allow_html=True)
    with col2:
        conf_pct = round(result["confidence"] * 100)
        st.markdown(
            f'<div style="text-align:right;"><span class="muted">종합 정확도</span><br>'
            f'<span style="font-size:2rem;font-weight:800;color:#d9a441;">{conf_pct}%</span></div>',
            unsafe_allow_html=True,
        )

    matched = len(result["entity_results"]) and any(e["matches"] for e in result["entity_results"])
    status_label = "근거 기반 응답" if result["edges"] else "그래프 근거 없음 (일반 응답)"
    st.markdown(
        f"""
        <div class="card-cream">
            <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                <div style="font-size:0.75rem; font-weight:700;">✅ {status_label}</div>
                <div class="evidence-tag" style="background:#d9a441;">
                    {'사전 매칭 완료' if matched else '매칭된 엔티티 없음'}
                </div>
            </div>
            <div style="font-size:1.2rem; font-weight:700; margin:10px 0 6px 0;">{result['answer']}</div>
            <div class="muted">질문 의도: {result['intent']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_entities = len(result["seed_ids"])
    n_nodes = len(result["nodes"])
    n_edges = len(result["edges"])
    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-cell"><div class="label">매칭 엔티티</div><div class="value">{n_entities}<span class="unit">개</span></div></div>
            <div class="stat-cell"><div class="label">그래프 노드</div><div class="value">{n_nodes}<span class="unit">개</span></div></div>
            <div class="stat-cell"><div class="label">탐색 관계</div><div class="value">{n_edges}<span class="unit">개</span></div></div>
            <div class="stat-cell"><div class="label">최대 깊이</div><div class="value">{result['depth_reached']}<span class="unit">hop</span></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_graph, col_side = st.columns([2, 1])
    with col_graph:
        st.markdown('<div class="section-label"><span class="badge">02</span> KNOWLEDGE GRAPH MAPPING</div>', unsafe_allow_html=True)
        if result["nodes"]:
            html = build_graph_html(result["nodes"], result["edges"], set(result["seed_ids"]))
            components.html(html, height=460, scrolling=False)
            st.markdown(
                '<div class="muted">● 굵은 테두리 = 질문에서 매칭된 시드 엔티티 · 엣지 라벨 = 관계 타입</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown('<div class="panel muted">탐색된 그래프가 없습니다.</div>', unsafe_allow_html=True)

    with col_side:
        st.markdown('<div class="section-label"><span class="badge">01</span> EXTRACTED ENTITIES</div>', unsafe_allow_html=True)
        entity_html = '<div class="panel">'
        if not result["entity_results"]:
            entity_html += '<span class="muted">추출된 엔티티가 없습니다.</span>'
        for e in result["entity_results"]:
            if not e["matches"]:
                entity_html += (
                    f'<div class="entity-item"><span class="entity-chip" style="background:#6b7362;">?</span>'
                    f'<span class="entity-name">{e["mention"]}</span>'
                    f'<div class="entity-quote">그래프에서 매칭되는 엔티티를 찾지 못함</div></div>'
                )
                continue
            best = max(e["matches"], key=lambda m: m["score"])
            entity_html += (
                f'<div class="entity-item"><span class="entity-chip">{best["type"]}</span>'
                f'<span class="entity-name">{best["name"]}</span>'
                f'<span class="entity-score">{round(best["score"]*100)}%</span>'
                f'<div class="entity-quote">"{e["mention"]}"로 검지</div></div>'
            )
        entity_html += '</div>'
        st.markdown(entity_html, unsafe_allow_html=True)

        st.markdown('<div class="section-label"><span class="badge">03</span> AUDITABLE ANALYSIS PATH</div>', unsafe_allow_html=True)
        steps = [
            f'질문 의도를 "{result["intent"]}"으로 분류했습니다.',
            (
                f'{n_entities}개 엔티티를 사전 매칭으로 정규화했습니다.'
                if n_entities else '그래프에서 매칭되는 엔티티를 찾지 못했습니다.'
            ),
            f'{n_edges}개 관계를 최대 {result["depth_reached"]}-hop 범위에서 조회했습니다.',
            '관계 신뢰도와 엔티티 매칭 점수를 결합해 최종 정확도를 산출했습니다.',
        ]
        steps_html = '<div class="panel">'
        for i, s in enumerate(steps, 1):
            steps_html += f'<div class="step-item"><span class="step-num">{i:02d}</span><span>{s}</span></div>'
        steps_html += '<div class="muted" style="margin-top:8px;">내부 추론 단계 대신 감사 가능한 처리 단계만 표시합니다.</div></div>'
        st.markdown(steps_html, unsafe_allow_html=True)

    st.markdown(
        f'<div class="section-label"><span class="badge">04</span> EVIDENCE LEDGER '
        f'<span class="muted" style="margin-left:auto;">{n_edges} sources</span></div>',
        unsafe_allow_html=True,
    )
    if not result["edges"]:
        st.markdown('<div class="panel muted">수집된 근거가 없습니다.</div>', unsafe_allow_html=True)
    else:
        for i, e in enumerate(result["edges"], 1):
            conf_pct = round(e["confidence"] * 100)
            quote = e["evidence"] or "(근거 문장 없음)"
            tag = EVIDENCE_SOURCE_KO.get(e["evidence_source"], e["evidence_source"] or "")
            st.markdown(
                f"""
                <div class="evidence-item">
                    <div style="display:flex; justify-content:space-between;">
                        <div>
                            <div class="evidence-id">E{i:02d}</div>
                            <div class="evidence-triple">{e['src_name']} → {e['relation']} → {e['dst_name']}</div>
                            <div class="evidence-quote">"{quote}"</div>
                            <span class="evidence-tag">{e['src_name']}</span>
                            {f'<span class="evidence-tag">{tag}</span>' if tag else ''}
                        </div>
                        <div class="evidence-conf"><span class="n">{conf_pct}%</span><span class="l">신뢰도</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_history():
    if not st.session_state.get("history"):
        return
    st.markdown('<div class="section-label" style="margin-top:24px;">🕒 최근 분석</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, item in enumerate(reversed(st.session_state.history[-6:])):
        with cols[i % 2]:
            conf_pct = round(item["confidence"] * 100)
            clicked = st.button(
                f'{item["question"]}   ·   {conf_pct}%',
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
        get_neo4j_driver()
        neo4j_online = True
    except Exception:
        neo4j_online = False

    render_header(neo4j_online)
    render_hero()
    query, run_clicked = render_query_console()

    if run_clicked and query.strip():
        if not neo4j_online:
            st.error("Neo4j에 연결할 수 없습니다. .env의 NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD를 확인하세요.")
        else:
            with st.spinner("그래프를 탐색하고 근거를 종합하는 중..."):
                result = run_analysis(query.strip())
            result["ts"] = len(st.session_state.history)
            st.session_state.history.append(result)
            st.session_state.active_result = result

    if st.session_state.active_result:
        render_result(st.session_state.active_result)

    render_history()


if __name__ == "__main__":
    main()
