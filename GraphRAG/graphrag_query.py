"""Neo4j 지식그래프에서 실제 multi-hop traversal로 질문에 답하는 GraphRAG 조회 스크립트.

RAG/graphrag_query.py를 그대로 복사해온 실험용 사본. 고도화 실험(hop 수 조정,
엔티티 disambiguation, 엔티티 링킹 개선 등)은 여기서 하고, RAG/graphrag_query.py는
데모용으로 그대로 둔다.

파이프라인 (README.md "문제 4" 해결 방안 — Relation-aware Retrieval 반영):
    1. LLM 1회 호출로 엔티티 멘션 + relation_type + value + entity_type을 동시 추출
       (extract_entities(), 예: "Apple는 언제 설립됐어?" -> entities=["Apple"],
       relation_type="INCEPTION", value=null, entity_type=null; "1988년에 설립된
       조직이 뭐가 있어?" -> entities=[], relation_type="INCEPTION", value="1988",
       entity_type="ORG")
    2. 멘션마다 Exact -> Alias -> Word Boundary -> Fuzzy Match cascade로 그래프의
       seed 노드 탐색 (find_seed_entities(), README "문제 3" 해결 방안 — 앞 단계에서
       찾으면 그 즉시 채택하고 뒤 단계는 시도 안 함)
    3. 라우팅:
       - entity 없음 + relation_type/value 있음 -> property_scan() (seed 없이
         전역 속성 스캔, entity_type이 있으면 그 타입으로 결과 제한 -- 예:
         "1988년에 설립된 조직은?" -> INCEPTION=1988인 ORG 타입만)
       - entity 있음 + relation_type 있음 -> relation_lookup() 1-hop 우선 시도,
         사실을 못 찾으면(Not Found) expand_subgraph() BFS로 폴백
       - entity 있음 + relation_type 없음 (또는 entity/relation_type 둘 다 없음)
         -> 기존 expand_subgraph() BFS 그대로
    4. 서브그래프 사실만 근거로 LLM이 한국어로 답변 (근거 부족하면 '모름')

API 키/DB 자격증명은 .env(레포 루트)에서 읽는다. .env는 .gitignore에 등록되어 있음.

사용법:
    python GraphRAG/graphrag_query.py
"""

import json
import os
import re
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI
from rapidfuzz import fuzz, process as rf_process

from cache import cache_key, cached

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CHAT_MODEL = "gpt-5.4-mini"
SEEDS_PER_MENTION = 3
# README "문제 7" 실측 실험(3-hop -> 5-hop, 10개 질의 회귀 0건 + 1건 개선)에 따라
# 5로 설정 -- 2026-07-16, 문제 1/2/10/11 그래프 수정 이후 12개 baseline으로 재검증함.
MAX_HOPS = 5
# 리랭킹(아래 rerank_facts()) 도입 전 노이즈 절감 1차 조치, 2026-07-16 -- 200->60으로
# 낮춤. 실측 검증: Philippines 900->275개(-69%), Health Sciences Centre 495->240개
# (-52%), 12개 baseline 전부 정답 유지(회귀 없음). Roketsan/Cirit처럼 닫힌 작은
# 클러스터는 애초에 limit 밑이라 영향 없음(둘 다 66개 그대로).
SUBGRAPH_FACT_LIMIT_PER_HOP = 60
# hop 확장 시 이 값보다 연결이 많은 노드(국가/대륙 등 범용 개체)는 다음 hop의 확장
# 출발점에서 제외한다. 예: Cirit -> Turkey(연결 294개) -> ... 로 뻗으면 Cirit ->
# Roketsan(연결 13개) -> Ankara 같이 실제 필요한 경로가 무관한 사실 수백 개에 묻혀
# LIMIT 안에 못 들어간다.
HUB_DEGREE_THRESHOLD = 40
# 리랭킹(rerank_facts(), 2026-07-16 추가) 설정. 사실 수가 이 값 이하면 리랭킹 자체를
# 생략(작은 클러스터는 이미 충분히 작아서 임베딩 호출이 오히려 낭비).
# 실측 발견: top_k=40은 너무 공격적 -- "Outotec의 필터가 만들어지는 도시는 어느
# 지역에 속해있어?"(전체 64개)에서 정답 사실 "Lappeenranta -[LOCATED_IN_THE_
# ADMINISTRATIVE_TERRITORIAL_ENTITY]-> South Karelia"가 head+relation이 똑같고
# tail만 다른 사실들("...-> South-East Finland" 등, 같은 도시의 다른 상위 행정구역
# 표기)과 임베딩 유사도가 거의 붙어서 top 40 밖으로 밀려나 회귀 발생(South Karelia
# -> 모름). 80으로 올려서 이 케이스를 포함한 12개 baseline 전부 재검증 통과.
RERANK_TRIGGER_THRESHOLD = 80
RERANK_TOP_K = 80
EMBED_MODEL = "text-embedding-3-small"
# 반복 샘플링+다수결(README "A/B 테스트 방법론"에 이미 3회로 설계돼 있었음, 2026-07-16
# 자동화). bfs/1hop_fallback_bfs 라우팅에서만 적용 -- 1hop/property_scan은 사실 1~수개
# 짜리 짧은 조회라 원래도 안정적이었고 흔들림이 실측된 적이 없어서, 굳이 3배로 돌리면
# 비용만 늘고 얻는 게 없음(실측: bfs 라우팅 평균 컨텍스트가 1hop보다 10배 이상 커서
# 다수결 비용도 그만큼 커짐 -- 흔들리는 경로에만 선택 적용해 비용을 최소화).
MAJORITY_VOTE_N = 3
MAJORITY_VOTE_SIMILARITY_THRESHOLD = 0.92

# no_seed 라우팅(엔티티 링킹이 아무것도 못 찾고, relation_type+value 폴백도 없는 경우)
# 전용 고정 응답. facts=[]인 채로 answer_with_subgraph()를 호출하면 LLM이 매번 "모름"을
# 생성하는데(2026-07-16 확인 -- Neo4j에 Samsung이 실제로 존재하는데도 "삼성"이라는
# 한국어 멘션이 안 풀려 이 경로를 탄 사례), 결과가 항상 정해져 있는데 매 질문마다 LLM
# 호출 비용/지연을 쓰는 게 낭비이고, "모름"은 "관련 사실이 없음"과 "개체 자체를 못
# 찾음"을 구분 못 해 사용자에게 원인이 안 보임 -- 사용자 피드백으로 분리.
ENTITY_NOT_FOUND_ANSWER = "엔티티를 찾을 수 없습니다."
# Fuzzy Match(4단계, RapidFuzz) 컷오프. 구현 중 실측 발견: README 설계 초안은
# "threshold 예: 0.9"(RapidFuzz 0~100 스케일로 90)를 제시했지만, 실제로 rapidfuzz.
# fuzz.WRatio로 계산해보면 README가 예시로 든 "Googel" -> "Google" 오타조차 83.3점
# 밖에 안 나와 90 컷오프면 그 예시 자체를 못 잡는다. 게다가 WRatio는 문자열 길이
# 차이가 크면 내부적으로 partial-ratio(부분 문자열 매치)를 섞어 쓰기 때문에, 오히려
# "Ankara" vs "Vinod Mankara"에 81.8점을 줘버려 Word Boundary 단계에서 없앤 바로 그
# 부분일치 오탐이 fuzzy 단계를 거쳐 되살아난다(실측 확인). 그래서 스코어러를 전체
# 문자열 기준 순수 Levenshtein 비율인 fuzz.ratio로 바꾸고 컷오프를 80으로 내렸다 --
# 이 조합으로 "Googel"/"Google"(83.3)과 "Micorsoft"/"Microsoft"(88.9)는 통과시키면서
# "Ankara"/"Sankara"(76.9), "Ankara"/"Vinod Mankara"(52.6)는 여전히 걸러내는 것을
# 실측 검증함.
FUZZY_MATCH_THRESHOLD = 80

openai_client = OpenAI()
driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
)
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE")


def _slugify_relation(relation_name):
    """rel_info.json의 relation 이름을 Neo4j에 실제 로드된 관계 타입과 동일한
    UPPER_SNAKE_CASE로 변환 (Scripts/kg/load_ground_truth.py의 relation_type_name()과
    같은 로직 -- 이 폴더는 RAG/graphrag_query.py의 독립 실험 사본이라 다른 디렉토리에
    새 의존성을 만들지 않기 위해 여기서 자체 보유)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", relation_name).strip("_").upper()
    if not slug or slug[0].isdigit():
        slug = f"REL_{slug}"
    return slug


def _load_relation_types():
    with open(ROOT / "docred_data" / "data" / "rel_info.json", encoding="utf-8") as f:
        rel_info = json.load(f)
    return {_slugify_relation(name) for name in rel_info.values()}


# 실제 그래프에 로드된 96개 관계 타입 화이트리스트 -- relation_lookup()/property_scan()이
# Cypher 관계 타입을 문자열로 직접 삽입(파라미터 불가)하기 전에 반드시 이 안에 있는지
# 확인한다. LLM이 이 목록 밖의 값을 환각으로 내놓으면(오타/스키마 밖 이름) 조용히
# 무시되고 호출부가 BFS로 폴백하므로, 화이트리스트가 Cypher 인젝션 방지와 스키마
# 검증을 동시에 담당한다.
RELATION_TYPES = _load_relation_types()

# DocRED 고정 엔티티 타입 6종 (GraphRAG/fix_over_merging.py의 VALID_ENTITY_TYPES와 동일
# 값 -- 그래프 로더가 실제로 쓰는 값이라 여기서도 고정 상수로 둔다). property_scan()이
# "1988년에 설립된 조직" 같은 질문에서 h.type을 걸러낼 때 씀 -- entity_type 없이 값만
# 매치하면 국가/앨범/기차 등 "조직"이 아닌 개체도 같은 값(예: INCEPTION=1988)을 가지면
# 섞여 나옴(실측: Republic of China on Taiwan(MISC), Olympic Wilderness(LOC) 등).
ENTITY_TYPES = {"PER", "ORG", "LOC", "TIME", "NUM", "MISC"}


def extract_entities(question, repeat=0):
    """질문에서 그래프 탐색의 출발점이 될 엔티티 멘션과, 있다면 relation_type/value까지
    LLM 1회 호출로 함께 추출 (README "문제 4" 핵심 설계 결정 1 -- 엔티티 추출과
    관계/값 추출을 하나의 LLM 호출로 합쳐 API 호출 수는 기존과 동일하게 유지).

    반환: {"entities": [...], "relation_type": RELATION_TYPES 중 하나 또는 null,
    "value": relation_type이 특정 값을 묻는 질문일 때 그 값(문자열) 또는 null,
    "entity_type": ENTITY_TYPES 중 하나 또는 null}.
    relation_type이 RELATION_TYPES 화이트리스트 밖이면(LLM 환각) None으로 정규화 --
    호출부(answer_question)가 이를 "relation_type 없음"과 동일하게 취급해 안전하게
    BFS로 폴백한다. entity_type도 같은 이유로 ENTITY_TYPES 밖이면 None으로 정규화.

    캐시 키 이름표(cache_key 첫 인자)를 예전 "extract_entities"(엔티티 리스트만
    반환하던 구버전, GraphRAG/.cache/에 이미 남아있는 캐시 파일들)와 다르게
    "parse_query_v7"로 둔다 -- 스키마가 리스트에서 dict로 바뀌었으므로 같은 키를
    재사용하면 구버전 캐시가 새 코드에 잘못된 타입으로 반환될 수 있고, 프롬프트
    자체도 여섯 번 더 수정했으므로(Outotec 2-hop 오판정 -> 1988 property scan 회귀
    -> entity_type 추가 -> COUNTRY/COUNTRY_OF_ORIGIN 혼동 수정 -> temperature=0
    적용 -> Outotec 패턴 재발 수정(예5), 아래 v2~v7 버전 노트 참고) 그때마다 버전을
    올렸다.

    repeat: 같은 질문을 반복 샘플링(다수결 판정)할 때 캐시가 첫 결과만 반환하지
    않도록 구분하는 인덱스. 실험을 나중에 재실행할 때는 (question, repeat)이
    같으면 캐시를 재사용해 API를 다시 호출하지 않는다.
    """
    def compute():
        prompt = (
            "다음 질문을 분석해 아래 형식의 JSON 객체 하나만 출력하세요 (다른 설명 없이):\n"
            '{"entities": ["..."], "relation_type": "..." 또는 null, "value": "..." 또는 null, '
            '"entity_type": "..." 또는 null}\n\n'
            "- entities: 지식그래프 탐색의 시작점이 될 고유명사(회사/기관/인물/지명 등). "
            "없으면 빈 배열. 그래프의 개체명은 원어(주로 영어)로 저장돼 있으므로, 질문에 "
            "한국어 표기로 나온 잘 알려진 고유명사는 널리 쓰이는 영어 정식 명칭으로 "
            "정규화해서 담으세요(예: '삼성'->'Samsung', '애플'->'Apple', '구글'->'Google'). "
            "이미 영어로 쓰여 있으면 그대로 두고, 정확한 영어 표기를 확신할 수 없는 "
            "생소한 이름은 질문에 쓰인 표기 그대로 둡니다(추측 번역보다 원문 유지가 "
            "안전 -- 뒤 단계 Fuzzy Match가 오타 수준 차이는 흡수함).\n"
            "- relation_type: 다음 목록 중 하나를 채우거나(entities가 있을 때만) null:\n"
            f"  {', '.join(sorted(RELATION_TYPES))}\n"
            "  예1) \"Apple는 언제 설립됐어?\" -> entities=[\"Apple\"], "
            "relation_type=INCEPTION (Apple 자신의 속성을 1-hop으로 바로 물음)\n"
            "  예2) \"Cirit을 만드는 회사의 본사는 어느 도시에 있어?\" -> entities=[\"Cirit\"], "
            "relation_type=null (Cirit 자신이 아니라 'Cirit을 만드는 회사'라는 다른 개체의 "
            "속성을 물어서 실제로는 2-hop 이상)\n"
            "  예3) \"1988년에 설립된 조직이 뭐가 있어?\" -> entities=[], "
            "relation_type=INCEPTION, value=\"1988\", entity_type=ORG (특정 개체 이름이 "
            "아니라 값으로 역방향 검색하는 질문이라 entities는 비어도 relation_type/value는 "
            "채움 -- '조직'처럼 결과가 속해야 할 종류가 질문에 명시돼 있으면 entity_type도 "
            "같이 채워서 그 종류가 아닌 결과를 걸러냄)\n"
            "  예4) \"Roketsan은 어느 나라 회사야?\" -> entities=[\"Roketsan\"], "
            "relation_type=COUNTRY (COUNTRY/COUNTRY_OF_CITIZENSHIP/COUNTRY_OF_ORIGIN을 "
            "혼동하지 말 것 -- COUNTRY는 조직/사물이 속한 국가, COUNTRY_OF_CITIZENSHIP은 "
            "사람의 국적, COUNTRY_OF_ORIGIN은 제품·작품 등의 원산지 국가를 물을 때만 씀. "
            "질문의 주어가 회사/기관이면 COUNTRY_OF_ORIGIN이 아니라 COUNTRY를 쓸 것)\n"
            "  예5) \"Outotec의 필터가 만들어지는 도시는 어느 지역에 속해있어?\" -> "
            "entities=[\"Outotec\"], relation_type=null (Outotec 자신의 국가/위치가 아니라 "
            "'Outotec의 필터가 만들어지는 도시'라는 다른 개체를 먼저 찾고 그 도시가 속한 "
            "지역을 물어서 실제로는 2-hop 이상. 예2의 'X가 만드는 Y의 Z' 패턴과 같음 -- "
            "'~의 필터/제품/본사가 만들어지는/있는 장소의 상위 지역' 같은 질문은 대상 개체 "
            "자신의 직접 속성이 아니므로 null)\n"
            "  예6) \"삼성의 회장은 누구야?\" -> entities=[\"Samsung\"] (한국어 표기 '삼성'을 "
            "영어 정식 명칭으로 정규화), relation_type=CHAIRPERSON (Samsung 자신의 속성을 "
            "1-hop으로 바로 물음 -- '회장'이 화이트리스트의 CHAIRPERSON에 대응됨)\n"
            "  entities가 있을 때는 그 개체 '자신'의 속성을 1-hop으로 직접 묻는 경우만 "
            "채우고(예1), 중간에 다른 개체를 하나 더 거쳐야 답이 나오면(예2) null. "
            "entities가 없을 때는 값으로 역검색하는 질문이면 채우고(예3), 그마저도 "
            "아니면(예: 여러 문서를 종합 요약) null.\n"
            "- value: relation_type이 있고 질문이 그 관계의 특정 대상값(연도/지명 등)을 "
            "묻는 경우 그 값. 없으면 null.\n"
            "- entity_type: 다음 목록 중 하나 또는 null -- "
            f"{', '.join(sorted(ENTITY_TYPES))} (ORG=조직/기관/회사, PER=사람, LOC=장소/지역/"
            "국가, TIME=시간표현, NUM=수량, MISC=그 외). value로 역검색하는 질문(예3)에서 "
            "결과가 어떤 종류여야 하는지 질문에 나와 있으면(예: '조직', '회사', '사람') "
            "그 타입으로 채우고, 명시돼 있지 않거나 relation_type/value 자체가 null이면 "
            "null.\n\n"
            f"질문: {question}"
        )
        # README "문제 8" -- temperature=0으로 답변 생성 비결정성을 줄임(완전한 결정성
        # 보장은 아님, OpenAI seed도 system_fingerprint가 안정적으로 안 나와 참고용).
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(text)
        entities = parsed.get("entities") or []
        relation_type = parsed.get("relation_type") or None
        if relation_type not in RELATION_TYPES:
            relation_type = None
        value = parsed.get("value") or None
        entity_type = parsed.get("entity_type") or None
        if entity_type not in ENTITY_TYPES:
            entity_type = None
        return {
            "entities": entities, "relation_type": relation_type,
            "value": value, "entity_type": entity_type,
        }

    # 캐시 키 뒤 버전 번호는 프롬프트 내용 자체가 바뀔 때마다 올려야 함 -- cache_key가
    # question/repeat만 해시하고 프롬프트 텍스트는 안 넣으므로, 같은 이름으로 프롬프트만
    # 바꾸면 예전 프롬프트로 만든 결과가 캐시에서 그대로 반환돼 새 로직이 조용히
    # 무시된다. 실측으로 두 번 걸림: v1("parse_query")은 Outotec 같은 "이름은 1-hop처럼
    # 보이지만 실제로는 다른 개체를 한 번 더 거쳐야 하는" 질문에서 relation_type을
    # 성급히 채워 2-hop 답을 놓침 -> v2에서 그 케이스를 null 처리하도록 프롬프트를
    # 조였더니, 이번엔 "1988년에 설립된 조직은?" 같은 정상적인 property-scan 질문까지
    # relation_type을 null로 만들어버리는 회귀 발생 -> v3에서 세 가지 케이스(1-hop 직접
    # 질문/개체 경유 2-hop/entities 없이 값으로 역검색)를 예시로 명시해 둘 다 통과하도록
    # 수정, 12개 기존 데모 질의 전체 재실행으로 회귀 없음 확인. v4는 entity_type 필드
    # 추가 -- "1988년에 설립된 조직" 질문이 property_scan에서 INCEPTION=1988인 모든
    # 타입(회사뿐 아니라 국가/앨범/기차 등 MISC/LOC도 포함)을 다 가져오는 실측 버그를
    # 발견(사용자가 "Republic of China on Taiwan" 같은 비-조직 결과를 스크린샷으로 직접
    # 지적) -- property_scan()이 h.type으로 걸러낼 수 있도록 entity_type도 같이 추출.
    # v5는 relation_type 분류가 의미가 가까운 라벨끼리 흔들리는 것을 실측 발견하고 수정
    # -- "Roketsan은 어느 나라 회사야?"를 5회 반복 샘플링(repeat=0..4)했더니 COUNTRY(정답)
    # 2/5, COUNTRY_OF_ORIGIN(제품/작품용, 회사에는 안 맞음) 3/5로 갈림(relation_lookup이
    # 못 찾고 BFS로 폴백해 최종 답은 우연히 맞았지만, 다른 개체에서 COUNTRY_OF_ORIGIN에
    # 우연히 매칭되는 엣지가 있었다면 폴백 없이 조용히 틀렸을 것). 화이트리스트에 96개
    # 라벨을 아무 설명 없이 나열만 해서 COUNTRY/COUNTRY_OF_CITIZENSHIP/COUNTRY_OF_ORIGIN
    # 처럼 의미가 겹치는 라벨을 구분 못 하는 게 원인 -- v3와 같은 패턴(대조 예시 추가)으로
    # 예4를 추가해 이 그룹만 명시적으로 구분. 다른 91개 라벨까지 전부 예시를 달면 프롬프트가
    # 과도하게 커지므로, 실측으로 확인된 혼동 그룹만 최소 추가하는 원칙 유지. v6는 문제 8
    # 대응으로 temperature=0 추가 -- MAX_HOPS 3->5 변경 검증 중 "Outotec의 필터가...
    # 어느 지역에 속해있어?"가 repeat=1에서 relation_type을 LOCATED_IN_THE_ADMINISTRATIVE_
    # TERRITORIAL_ENTITY로 잘못 채워 1-hop으로 조기 종료(정답은 2-hop, null이어야 함)하는
    # 걸 실측 발견 -- 이 케이스도 v5의 COUNTRY/COUNTRY_OF_ORIGIN처럼 라벨링 비결정성이라
    # 근본 원인은 같음. temperature=0으로 완전히 없어지진 않지만(OpenAI seed도
    # system_fingerprint가 불안정해 결정성 보장 안 됨) 흔들림 폭을 줄이는 최소 조치.
    # v7 -- temperature=0을 걸고 나니 바로 위에서 고쳤다고 적은 "Outotec의 필터가...
    # 어느 지역에 속해있어?" 케이스가 repeat=0..3 전부(4/4) LOCATED_IN_THE_ADMINISTRATIVE_
    # TERRITORIAL_ENTITY로 100% 고정 오답되는 걸 재실측 -- temperature=0은 "가끔 틀림"을
    # "항상 틀림"으로 바꿔버릴 수 있다는 걸 보여주는 실제 사례(분산은 줄지만 정확도는
    # 보장 안 함). v2의 예2(Cirit)가 이미 같은 패턴(X가 만드는 Y의 Z)을 다루는데도 Outotec
    # 특유의 "도시는 어느 지역에 속해있어" 표현으로는 일반화가 안 됐음 -- 예5로 이 정확한
    # 문구를 명시. v8(2026-07-16) -- "삼성의 회장은 누구야?"가 entities=["삼성"]로 추출돼
    # find_seed_entities()가 그래프의 영어 이름("Samsung")과 매칭 못 해 "엔티티를 찾을 수
    # 없음"으로 실패하는 걸 실측 발견(Neo4j에 Samsung 노드 자체는 존재함 -- 그래프 커버리지
    # 문제가 아니라 순수 언어 불일치 버그). 별도 번역 API/캐싱 레이어를 추가하는 대신(질문당
    # 어차피 1번 호출되는 이 프롬프트에 그대로 얹음 -- 추가 호출/의존성 0개), entities 추출
    # 규칙에 "잘 알려진 고유명사는 영어 정식 명칭으로 정규화" 지시 + 예6(Samsung/CHAIRPERSON)
    # 추가. 확신 없는 이름은 원문 유지하도록 명시해 오번역으로 인한 오탐 방지(Fuzzy Match
    # 단계가 최후 안전망으로 남음).
    return cached(cache_key("parse_query_v8", CHAT_MODEL, question, repeat), compute)


# DocRED 주석이 국가 '수반'을 HEAD_OF_STATE와 HEAD_OF_GOVERNMENT에 비일관적으로 나눠
# 넣음 -- 실측(2026-07-16): 미국 대통령은 전부 HEAD_OF_GOVERNMENT, 필리핀 대통령은
# HEAD_OF_STATE. LLM이 "대통령"을 의미상 맞는 HEAD_OF_STATE로 뽑아도 미국은 1-hop이
# 사실을 못 찾고 BFS 폴백->"모름"으로 무너졌음(사용자 지적). 프롬프트로 한쪽 라벨을
# 고르면 다른 나라가 깨지므로(데이터가 양쪽에 흩어짐), 둘 중 하나가 나오면 둘 다 조회.
LEADER_RELATIONS = {"HEAD_OF_STATE", "HEAD_OF_GOVERNMENT"}


def relation_lookup(session, seed_ids, relation_type):
    """seed 노드들에서 relation_type 하나만 1-hop으로 직접 조회 (README "문제 4"
    핵심 설계 결정 3). expand_subgraph()의 BFS와 달리 딱 그 관계 하나만 보므로
    사실이 하나만 나와도 곧장 답변에 쓸 수 있다 -- relation_type은 반드시 호출 전에
    RELATION_TYPES 화이트리스트 검증을 거쳐야 함(Cypher 관계 타입은 파라미터로
    못 넘기고 문자열 삽입해야 하므로). LEADER_RELATIONS 계열이면 한 쌍을 함께 조회한다."""
    if relation_type not in RELATION_TYPES or not seed_ids:
        return []
    rels = LEADER_RELATIONS if relation_type in LEADER_RELATIONS else {relation_type}
    rel_pattern = "|".join(sorted(rels))  # 전부 RELATION_TYPES 검증 통과분이라 인젝션 안전
    query = f"""
    MATCH (h:ZEntity)-[r:{rel_pattern}]-(n:ZEntity)
    WHERE h.id IN $seed_ids
    RETURN DISTINCT startNode(r).name AS head, type(r) AS relation, endNode(r).name AS tail,
           r.evidence AS evidence
    LIMIT $limit
    """
    rows = list(session.run(query, seed_ids=seed_ids, limit=SUBGRAPH_FACT_LIMIT_PER_HOP))
    return [
        {"head": r["head"], "relation": r["relation"], "tail": r["tail"],
         "evidence": " ".join(r["evidence"]) if r["evidence"] else None}
        for r in rows
    ]


def property_scan(session, relation_type, value, entity_type=None):
    """seed 엔티티 없이 relation_type + value만으로 전역 속성 스캔 (README "문제 4"
    핵심 설계 결정 2 -- 예: "1988년에 설립된 조직이 뭐가 있어?"). value는 tail 노드의
    name과 대소문자 무시 완전일치만 매치 -- 연도/지명 등 Neo4j에 문자열 속성으로
    저장된 값 그대로 비교.

    entity_type이 주어지면 h.type을 그 값으로 제한한다 -- relation_type + value만으로는
    "조직"이 아닌 다른 타입의 개체도 같은 값을 가지면 섞여 나옴을 실측으로 확인함(예:
    "1988년에 설립된 조직" 질문에서 INCEPTION=1988인 결과에 실제 회사(ORG)뿐 아니라
    "Republic of China on Taiwan"(MISC, 이승만 정권기 구분 명칭), "Olympic Wilderness"
    (LOC, 자연보호구역 지정연도)까지 섞여 나옴). entity_type이 없으면(질문이 결과 종류를
    명시하지 않은 경우) 기존처럼 타입 제한 없이 전체 스캔."""
    if relation_type not in RELATION_TYPES or not value:
        return []
    if entity_type is not None and entity_type not in ENTITY_TYPES:
        entity_type = None
    query = f"""
    MATCH (h:ZEntity)-[r:{relation_type}]->(n:ZEntity)
    WHERE toLower(n.name) = toLower($value)
      AND ($entity_type IS NULL OR h.type = $entity_type)
    RETURN DISTINCT h.name AS head, type(r) AS relation, n.name AS tail,
           r.evidence AS evidence
    LIMIT $limit
    """
    rows = list(session.run(
        query, value=value, entity_type=entity_type, limit=SUBGRAPH_FACT_LIMIT_PER_HOP,
    ))
    return [
        {"head": r["head"], "relation": r["relation"], "tail": r["tail"],
         "evidence": " ".join(r["evidence"]) if r["evidence"] else None}
        for r in rows
    ]


_fuzzy_name_pool = None


def _load_fuzzy_name_pool(session):
    """Fuzzy Match(4단계) 후보 풀을 프로세스당 1회만 로드해 캐싱 -- 42,456개 노드
    전체를 매 fuzzy 폴백 호출마다 다시 스캔하면 비용이 크므로, 그래프가 이 프로세스
    실행 중에는 안 바뀐다는 전제(우리 자신의 병합/분리 스크립트를 동시에 돌리고
    있지 않은 한 사실) 하에 첫 호출 때 한 번만 가져와 메모리에 둔다."""
    global _fuzzy_name_pool
    if _fuzzy_name_pool is None:
        rows = session.run("MATCH (e:ZEntity) RETURN e.id AS id, e.name AS name, e.type AS type")
        _fuzzy_name_pool = [(r["name"], r["id"], r["type"]) for r in rows]
    return _fuzzy_name_pool


def _fuzzy_match_entities(session, mention):
    """4단계(최후 수단): RapidFuzz로 오타 대응 매치(예: "Googel" -> "Google").
    Exact/Alias/Word Boundary가 전부 실패했을 때만 호출된다."""
    pool = _load_fuzzy_name_pool(session)
    matches = rf_process.extract(
        mention, [name for name, _, _ in pool], scorer=fuzz.ratio,
        limit=SEEDS_PER_MENTION, score_cutoff=FUZZY_MATCH_THRESHOLD,
    )
    return [
        {"id": pool[idx][1], "name": pool[idx][0], "type": pool[idx][2]}
        for _, _score, idx in matches
    ]


def find_seed_entities(session, mention):
    """README "문제 3" 해결 방안 -- Exact -> Alias -> Word Boundary -> Fuzzy Match
    cascade(Priority-based Entity Linking). 앞 단계에서 하나라도 찾으면 그 즉시
    채택하고 뒤 단계는 시도하지 않는다(정확도가 높은 매치를 우선). 기존 CONTAINS
    부분일치는 "Ankara" 질의가 "Vinod Mankara"/"Sankara"까지 오탐으로 잡던 문제가
    있어 폐기 -- Word Boundary 단계(정규식 \\b)로 대체하면 이 오탐이 사라지는 것을
    실측 검증함. LLM은 안 씀(질의당 최소 1회 호출돼 트래픽이 높을 수 있어 비용/속도상
    결정론적 방법만 사용, README 설계 결정)."""
    # 1단계: Exact Match -- 정규화된 질의 멘션과 name이 완전히 일치.
    rows = list(session.run(
        """
        MATCH (e:ZEntity) WHERE toLower(e.name) = toLower($mention)
        RETURN e.id AS id, e.name AS name, e.type AS type LIMIT $limit
        """,
        mention=mention, limit=SEEDS_PER_MENTION,
    ))
    if rows:
        return rows

    # 2단계: Alias Match -- aliases 리스트에 완전히 일치하는 표현이 있으면 채택.
    rows = list(session.run(
        """
        MATCH (e:ZEntity) WHERE any(alias IN e.aliases WHERE toLower(alias) = toLower($mention))
        RETURN e.id AS id, e.name AS name, e.type AS type LIMIT $limit
        """,
        mention=mention, limit=SEEDS_PER_MENTION,
    ))
    if rows:
        return rows

    # 3단계: Word Boundary Match -- \b 단어 경계 정규식(대소문자 무시). Cypher =~는
    # 전체 문자열 매치(Java Pattern.matches와 동일)라 앞뒤에 .*를 둬야 "포함" 의미가
    # 된다.
    pattern = r"(?i).*\b" + re.escape(mention) + r"\b.*"
    rows = list(session.run(
        """
        MATCH (e:ZEntity)
        WHERE e.name =~ $pattern OR any(alias IN e.aliases WHERE alias =~ $pattern)
        RETURN e.id AS id, e.name AS name, e.type AS type LIMIT $limit
        """,
        pattern=pattern, limit=SEEDS_PER_MENTION,
    ))
    if rows:
        return rows

    # 4단계: Fuzzy Match -- 오타 대응 최후 수단.
    return _fuzzy_match_entities(session, mention)


def expand_subgraph(session, seed_ids):
    """seed 노드에서 MAX_HOPS까지 관계를 BFS로 가져온다.

    매 hop마다 그 hop에서 새로 만난 노드 중 허브 노드(연결이 HUB_DEGREE_THRESHOLD
    초과)는 다음 hop의 확장 출발점에서 제외한다 — 안 그러면 국가/대륙처럼 연결이
    수백 개인 노드를 지나가는 순간 무관한 사실이 쏟아져 정작 필요한 경로가
    LIMIT 안에 못 들어간다.

    중복 제거(2026-07-16 추가): 같은 엣지가 서로 다른 hop/방향에서 재방문돼 완전히
    동일한 (head, relation, tail, evidence)가 중복 수집되는 경우가 실측으로 27%
    (AirAsia Zest 66개 중 18개)나 됨 — LLM 프롬프트에 같은 문장을 두 번 보여줄 이유가
    없으므로 무료로 제거.
    """
    if not seed_ids:
        return []

    facts = []
    seen = set()
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
            ORDER BY head, relation, tail
            LIMIT $limit
            """,
            frontier=frontier, limit=SUBGRAPH_FACT_LIMIT_PER_HOP,
        ))
        if not rows:
            break

        # relation triple만으로는 DocRED의 고정된 96개 relation 스키마에 없는 세부
        # 정보(예: "어느 위원회가 세웠는지")를 놓친다. Neo4j 관계에 저장된 evidence
        # 문장을 같이 실어서, 구조화된 grpah 순회 + 원문 근거를 모두 활용한다.
        for r in rows:
            evidence = " ".join(r["evidence"]) if r["evidence"] else None
            key = (r["head"], r["relation"], r["tail"], evidence)
            if key in seen:
                continue
            seen.add(key)
            facts.append({
                "head": r["head"], "relation": r["relation"], "tail": r["tail"],
                "evidence": evidence,
            })

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


def rerank_facts(question, facts, top_k=RERANK_TOP_K):
    """2026-07-16 추가: expand_subgraph()가 가져온 사실이 많을 때(허브 인접
    질문, 예: Philippines 275개) 질문과의 임베딩 코사인 유사도로 정렬해 상위
    top_k개만 남긴다. facts가 top_k 이하면 그대로 반환(임베딩 호출 자체를 생략).

    text-embedding-3-small은 매우 저렴하고(gpt 채팅 모델 대비 토큰당 단가가
    훨씬 낮음) 배치 입력을 네이티브 지원 -- 질문 1개 + 사실 N개를 한 번의
    API 호출로 임베딩(요청 수 자체가 늘지 않음, chat completion처럼 요청당
    비용이 크지 않음).

    실측 검증(2026-07-16, 12개 baseline 중 노이즈가 컸던 4개 질의): Manila
    항공사 문서/Health Sciences Centre 대학 설립년도/Philippines 요약은
    top_k=25로 줄여도 정답 유지. 다만 **AirAsia Zest 합병 질문(어느 항공사가
    합쳐졌는지)은 리랭킹으로 고쳐지지 않음** -- 합병 근거 사실이 실제로는
    상위 1~9위에 이미 들어있는데도(직접 확인함) 답변이 흔들리는 걸 반복
    샘플링으로 재현. 이 질문은 relation 스키마에 없는 사실(문제 5, "합병"은
    96개 관계에 없어 evidence 원문에서만 추론 가능)이라 원인이 노이즈량이
    아니라 답변 생성 자체의 취약성(문제 8)으로 보임 -- 리랭킹은 "사실 유무"
    문제가 아니라 "사실 존재량/비용" 문제에 효과가 있고, 이 케이스처럼
    "구조화되지 않은 근거를 올바르게 추론하는지" 문제는 못 고침."""
    if len(facts) <= top_k:
        return facts

    texts = [question] + [_format_fact(f) for f in facts]
    resp = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    vectors = np.array([d.embedding for d in resp.data])
    q_vec, fact_vecs = vectors[0], vectors[1:]

    norms = np.linalg.norm(fact_vecs, axis=1) * np.linalg.norm(q_vec)
    norms[norms == 0] = 1e-9  # 0 벡터(빈 텍스트 등) 0-나눗셈 방지
    sims = fact_vecs @ q_vec / norms

    order = np.argsort(-sims)[:top_k]
    return [facts[i] for i in order]


def _answer_with_evidence_only(question, evidence_texts, repeat=0):
    """Graph-grounded Evidence Retrieval 폴백(2026-07-16, 사용자 제안): Graph Search
    -> 답 없음 -> 방문 노드 evidence만 재검색 -> LLM -> Answer. 구조화된 triple
    (head-relation-tail) 없이, 방문한 노드들의 evidence 원문 문장만 모아 다시 답을
    시도한다 -- answer_with_subgraph()가 "모름"을 낼 때만 호출됨(아래 참고).

    실측 검증(16회 반복 샘플링, AirAsia Zest 합병 질문): 1단계(triple 기반)만 쓰면
    정답 6 / 모름 8 / 오답 2였는데, "모름"이 나온 8건 중 7건이 이 폴백으로 정답
    복구됨(정답 6->14). **오답 2건은 이 폴백으로도 안 고쳐짐** -- 애초에 "모름"이
    아니라 확신에 찬 오답이라 폴백이 아예 안 걸리기 때문(설계상 의도된 범위 밖)."""
    context = "\n".join(f"- {e}" for e in evidence_texts)

    def compute():
        prompt = (
            "아래는 지식그래프 탐색 중 방문한 노드들의 원문 근거 문장 모음입니다. "
            "구조화된 사실(triple)이 아니라 원문 그대로입니다. 이 문장들만 근거로 "
            "질문에 한국어로 답하세요. 여러 문장을 종합해 추론해도 됩니다. 근거가 "
            "불충분하면 반드시 '모름'이라고만 답하세요.\n\n"
            f"근거 문장 모음:\n{context}\n\n질문: {question}"
        )
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

    return cached(
        cache_key("evidence_only_fallback_v1", CHAT_MODEL, question, context, repeat), compute
    )


def answer_with_subgraph(question, facts, repeat=0):
    """README "문제 8" 대응(2026-07-16): temperature=0으로 답변 생성 비결정성을
    줄임 -- 실측으로 재현됨: MAX_HOPS 3->5 검증 중 "AirAsia Zest는 어느 항공사들이
    합쳐져 만들어졌고, 언제 그렇게 됐어?" 질문이 그래프에 합병 정보가(evidence 텍스트
    안에) 그대로 있는데도 재실행 한 번에 정답 -> 다른 실행에 '모름'으로 뒤집히는 걸
    직접 확인함 -- relation 스키마 밖 사실(문제 5, 예: "합병")이라 구조화된 fact가 아니라
    evidence 원문에서 LLM이 직접 읽어내야 하는 케이스라 특히 취약. temperature=0으로
    완전히 결정적이 되진 않지만(OpenAI 쪽도 seed/system_fingerprint가 불안정해 보장 안
    됨) 흔들림 폭을 줄이는 최소 조치 -- 반복 샘플링+다수결(아래 majority_vote_answer())과
    Graph-grounded Evidence Retrieval 폴백(아래 _answer_with_evidence_only(), 2026-07-16
    추가)을 병행."""
    if not facts:
        context_block = "(그래프에서 관련 사실을 찾지 못함)"
    else:
        context_block = "\n".join(_format_fact(f) for f in facts)

    def compute():
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
            temperature=0,
        )
        return resp.choices[0].message.content.strip()

    answer = cached(
        cache_key("answer_with_subgraph_v2", CHAT_MODEL, question, context_block, repeat), compute
    )

    # Graph-grounded Evidence Retrieval 폴백: triple 기반 1차 답변이 "모름"일 때만
    # evidence 원문 전용 재시도(비용 절약 -- 이미 답을 찾은 경우는 재호출 안 함).
    if answer == "모름" and facts:
        evidence_texts = sorted({f["evidence"] for f in facts if f.get("evidence")})
        if evidence_texts:
            fallback = _answer_with_evidence_only(question, evidence_texts, repeat=repeat)
            if fallback != "모름":
                return fallback

    return answer


def majority_vote_answer(question, facts, n=MAJORITY_VOTE_N):
    """README "A/B 테스트 방법론"에 설계돼 있던 반복 샘플링+다수결을 2026-07-16
    자동화. answer_with_subgraph()를 n번(repeat=0..n-1) 불러 답을 모으고 다수결로
    최종 답을 정함.

    **그룹화를 임베딩이 아니라 LLM 클러스터링으로 함 -- 실측으로 임베딩 접근을
    폐기한 이유**: 처음엔 rerank_facts()처럼 코사인 유사도로 답끼리 묶으려 했는데,
    "AirAsia Zest는 Asian Spirit과 Zest Air가 합쳐져 만들어졌고, 2013년에 그렇게
    됐습니다" vs "...2016년 1월에 그렇게 됐습니다"(다른 사실!)의 유사도가 0.9727인
    반면, 진짜 같은 사실을 다른 문구로 쓴 두 답의 유사도는 0.9894 -- 격차가 0.017
    밖에 안 나서 안전한 임계값을 잡을 수가 없었음(문장 틀이 90% 겹치고 핵심 사실인
    날짜 한 군데만 다르면 임베딩이 둔감함, rerank_facts()에서 겪은 것과 같은
    유형의 함정). fix_over_merging.py의 cluster_conflict()와 같은 패턴(LLM에게
    N개 항목을 보여주고 그룹 번호 배열로 클러스터링 받기)으로 교체 -- 표현 차이는
    무시하고 핵심 사실이 같은지만 판정하도록 프롬프트에 명시. 만장일치(답이 전부
    똑같은 문자열)면 이 호출조차 생략.

    반환: (최종 답, 원본 n개 샘플, 최종 답이 속한 그룹의 표 수). 세 번째 값은
    호출부가 "몇 표 중 몇 표가 이 답에 동의했는지"를 보여줄 때 쓴다 -- 문자열
    단순 비교(`v == answer`)로 세면 안 됨: "터키의 SSİK가 세웠습니다"와 "Turkey의
    SSİK가 세웠습니다"는 클러스터링에서 같은 그룹(핵심 사실 동일)으로 묶여도
    문자열은 다르므로 과소 카운트된다(2026-07-16, UI에서 다수결 일치도가 실제
    3/3인데 1/3로 표시되는 걸 사용자가 발견해 수정)."""
    answers = [answer_with_subgraph(question, facts, repeat=i) for i in range(n)]

    if len(set(answers)) == 1:
        return answers[0], answers, n  # 만장일치 -- 클러스터링 호출 자체 생략

    group_ids = _cluster_answers(question, answers)
    if group_ids is None:
        return answers[0], answers, 1  # 클러스터링 실패 -- 안전하게 첫 샘플로 폴백

    counts = {}
    for gid in group_ids:
        counts[gid] = counts.get(gid, 0) + 1
    best_gid = max(counts, key=counts.get)
    best_idx = group_ids.index(best_gid)
    return answers[best_idx], answers, counts[best_gid]


def _cluster_answers(question, answers):
    """다수결 그룹화 -- 표현/어순 차이는 무시하고 핵심 사실(숫자/연도/이름 등)이
    같은 답끼리만 묶는다. 반환: 각 답(인덱스 순서대로)이 속한 그룹 번호 배열,
    파싱 실패 시 None."""
    lines = "\n".join(f"[{i}] {a}" for i, a in enumerate(answers))

    def compute():
        prompt = (
            f"질문: {question}\n\n다음은 이 질문에 대한 {len(answers)}개의 답변 "
            f"후보입니다:\n{lines}\n\n"
            "각 답변이 같은 구체적 사실을 말하고 있는지 판정하세요 -- 표현/어순/"
            "마크다운 차이는 무시하고, 핵심 사실(숫자, 연도, 이름 등)이 같은지만 "
            "보세요. 예를 들어 '2013년'과 '2016년 1월'은 핵심 사실(연도)이 다르므로 "
            "반드시 다른 그룹입니다. '모름'은 항상 다른 답들과 별개 그룹입니다.\n\n"
            "반드시 JSON 배열만 출력하세요(다른 설명 없이): 각 답변(위 인덱스 순서 "
            "그대로)이 속한 그룹 번호(0부터 시작하는 정수) 배열. 예: [0, 1, 0]"
            "(0번과 2번은 같은 사실, 1번은 다른 사실)."
        )
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}], temperature=0,
        )
        text = resp.choices[0].message.content.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)

    key = cache_key("cluster_answers_v1", CHAT_MODEL, question, answers)
    try:
        group_ids = cached(key, compute)
    except json.JSONDecodeError:
        return None
    if not isinstance(group_ids, list) or len(group_ids) != len(answers):
        return None
    return group_ids


def answer_question(question, repeat=0, verbose=True):
    """README "문제 4" 라우팅 (Relation-aware Retrieval):
        entity 없음  + relation_type/value 있음 -> property_scan
        entity 있음  + relation_type 있음        -> relation_lookup 1-hop
                                                     (Not Found면 BFS 폴백)
        그 외(entity 있음 + relation_type 없음, 또는 둘 다 없음) -> 기존 BFS
    route: 어느 경로를 탔는지 ("property_scan"/"1hop"/"1hop_fallback_bfs"/
    "bfs"/"no_seed") -- A/B 테스트에서 라우팅 자체가 의도대로 갈리는지 확인하는 용도."""
    parsed = extract_entities(question, repeat=repeat)
    mentions = parsed["entities"]
    relation_type = parsed["relation_type"]
    value = parsed["value"]
    entity_type = parsed["entity_type"]

    with driver.session(database=NEO4J_DATABASE) as session:
        seed_ids = []
        seed_names = []
        for mention in mentions:
            for row in find_seed_entities(session, mention):
                seed_ids.append(row["id"])
                seed_names.append(f"{row['name']} ({row['id']})")

        if not seed_ids:
            if relation_type and value:
                facts = property_scan(session, relation_type, value, entity_type)
                route = "property_scan"
            else:
                facts = []
                route = "no_seed"
        elif relation_type:
            facts = relation_lookup(session, seed_ids, relation_type)
            if facts:
                route = "1hop"
            else:
                facts = expand_subgraph(session, seed_ids)
                route = "1hop_fallback_bfs"
        else:
            facts = expand_subgraph(session, seed_ids)
            route = "bfs"

    # 리랭킹(rerank_facts()) -- 사실 수가 RERANK_TRIGGER_THRESHOLD 이하면 그 안에서
    # 이미 조기 반환되므로 property_scan/1hop처럼 원래 사실이 적은 라우팅은 임베딩
    # 호출 없이 그대로 통과.
    n_before_rerank = len(facts)
    facts = rerank_facts(question, facts)

    # 다수결(majority_vote_answer())은 bfs/1hop_fallback_bfs에서만 적용 -- 흔들림이
    # 실측된 라우팅이 이 둘뿐이고(AirAsia Zest 등), 1hop/property_scan은 사실이
    # 1~수개짜리 짧은 직접 조회라 원래도 안정적이라 3배로 돌릴 필요가 없음(비용 절약).
    votes = None
    agree_count = None
    if route == "no_seed":
        # facts가 항상 []이라 LLM을 불러도 결과가 뻔히 "모름"으로 고정됨 -- 호출 생략.
        answer = ENTITY_NOT_FOUND_ANSWER
    elif route in ("bfs", "1hop_fallback_bfs"):
        answer, votes, agree_count = majority_vote_answer(question, facts)
    else:
        answer = answer_with_subgraph(question, facts, repeat=repeat)

    if verbose:
        print(f"질문: {question}")
        print(f"  파싱 결과: entities={mentions} relation_type={relation_type} value={value} "
              f"entity_type={entity_type}")
        print(f"  seed 노드: {seed_names}")
        print(f"  라우팅: {route}")
        print(f"  서브그래프 사실 수: {n_before_rerank} -> 리랭킹 후 {len(facts)}")
        if votes is not None:
            print(f"  다수결 표본({len(votes)}개, {agree_count}표 일치): {votes}")
        print(f"  답변: {answer}")
        print()

    return answer, seed_names, facts, route


if __name__ == "__main__":
    demo_questions = [
        # 기존 BFS 데모 (회귀 확인용, 그대로 유지)
        "Cirit을 만드는 회사의 본사는 어느 도시에 있어?",
        "Roketsan은 어느 나라 회사야?",
        "Health Sciences Centre가 있는 도시에 있는 대학교는 몇 년에 설립됐어?",
        "Outotec의 필터가 만들어지는 도시는 어느 지역에 속해있어?",
        "AirAsia Zest의 본사는 어디였어?",
        # Relation-aware Retrieval 라우팅 확인용 (README "문제 4")
        "Roketsan은 몇 년에 설립됐어?",  # entity + relation_type -> 1-hop 기대
        "1988년에 설립된 조직이 뭐가 있어?",  # entity 없음 -> property_scan 기대
        "미국 대통령의 이름이 뭐야?",  # HEAD_OF_STATE/HEAD_OF_GOVERNMENT 계열 동시 조회
    ]
    for q in demo_questions:
        answer_question(q)

    driver.close()
