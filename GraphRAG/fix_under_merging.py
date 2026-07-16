"""언더-머징(회사 법인 접미사 표기 차이로 같은 개체가 별개 노드로 쪼개진 경우) 탐지 + 수정.

README.md "문제 2" 5단계 설계 구현 (Preventive Normalization은 재적재 스크립트
쪽 영역이라 여기서 다루지 않음 -- 이미 로드된 그래프를 고치는 것과 향후 재적재 시
재발을 막는 것은 별개 작업):

    1. Rule-based Candidate Detection -- 법인 접미사 사전으로 "A"/"A + Suffix"가
       둘 다 노드로 존재하는 쌍을 전수 탐지.
    2. Graph Topology Check -- 후보 쌍 사이에 이미 계층 관계(PARENT_ORGANIZATION/
       SUBSIDIARY/OWNED_BY/PART_OF)가 직접 있으면 즉시 기각 (Harvard/Harvard
       Corporation처럼 실제로 별개 법인인 케이스를 자동 병합하지 않기 위함).
    3. Candidate Verification -- 공유하는 관계 타입의 값이 일치하면 병합, 충돌하면
       기각(둘 다 규칙 기반, 확신 있음). 공유하는 관계 타입 자체가 없으면(정보 희소)
       LLM에 위임.
    4. Graph Merge -- 연결 수가 더 많은 쪽을 canonical로 두고 나머지 노드의 관계를
       전부 재배정 + alias 통합 + 중복 노드 삭제.

기존 graphrag_query.py의 RELATION_TYPES 화이트리스트/openai_client/driver를 그대로
재사용 (같은 폴더, 같은 Neo4j 인스턴스).

사용법:
    python GraphRAG/fix_under_merging.py --dry-run   # 후보/판정만 출력 (기본값, 그래프 변경 없음)
    python GraphRAG/fix_under_merging.py --apply      # 실제로 병합 수행 (되돌리기 어려우니 신중히)
"""

import argparse
import json
import re

from cache import cache_key, cached
from graphrag_query import CHAT_MODEL, NEO4J_DATABASE, RELATION_TYPES, driver, openai_client

# README에서 실제로 검증한 안전한 접미사만 사용 (Inc./Corp./Corporation/Ltd./LLC/GmbH/
# Oyj/plc/AG). 짧고 흔한 단어(AB/Co/Group/Holdings)는 오탐 위험이 커서 제외 -- 필요하면
# Candidate Verification/LLM 단계가 어차피 다시 걸러주므로, 굳이 넓힐 필요가 생기면
# 그때 추가.
_SUFFIX_ALTS = [
    r"Incorporated", r"Inc\.?",
    r"Corporation", r"Corp\.?",
    r"Limited", r"Ltd\.?",
    r"L\.L\.C\.", r"LLC",
    r"GmbH", r"Oyj", r"PLC", r"Plc", r"plc", r"AG",
]
_SUFFIX_RE = re.compile(r"\s+(?:" + "|".join(_SUFFIX_ALTS) + r")$")

HIERARCHY_RELATION_TYPES = {"PARENT_ORGANIZATION", "SUBSIDIARY", "OWNED_BY", "PART_OF"}
assert HIERARCHY_RELATION_TYPES <= RELATION_TYPES

TYPE_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")


def strip_suffix(name):
    return _SUFFIX_RE.sub("", name).strip() if _SUFFIX_RE.search(name) else None


def find_suffix_candidates(session):
    """1단계: "A"/"A + 접미사"가 둘 다 ORG 노드로 존재하는 쌍을 전수 탐지."""
    rows = list(session.run("MATCH (e:ZEntity:ORG) RETURN e.id AS id, e.name AS name"))
    by_name = {r["name"]: r["id"] for r in rows}
    candidates, seen = [], set()
    for r in rows:
        base = strip_suffix(r["name"])
        if not base or base not in by_name or base == r["name"]:
            continue
        pair = tuple(sorted([r["id"], by_name[base]]))
        if pair in seen:
            continue
        seen.add(pair)
        candidates.append({
            "short_id": by_name[base], "short_name": base,
            "long_id": r["id"], "long_name": r["name"],
        })
    return candidates


def topology_check(session, id1, id2):
    """2단계: 둘 사이에 이미 계층 관계가 직접 있으면 True (= 서로 다른 법인 확정, 기각)."""
    rows = list(session.run(
        "MATCH (a:ZEntity {id: $id1})-[r]-(b:ZEntity {id: $id2}) RETURN type(r) AS rel_type",
        id1=id1, id2=id2,
    ))
    hits = [r["rel_type"] for r in rows if r["rel_type"] in HIERARCHY_RELATION_TYPES]
    return hits


def get_relations(session, entity_id):
    """entity_id가 관여하는 (관계타입, 상대 노드 이름) 쌍 전부 (방향 무관)."""
    rows = list(session.run(
        """
        MATCH (e:ZEntity {id: $id})-[r]->(n:ZEntity) RETURN type(r) AS rel, n.name AS tail
        UNION
        MATCH (n:ZEntity)-[r]->(e:ZEntity {id: $id}) RETURN type(r) AS rel, n.name AS tail
        """,
        id=entity_id,
    ))
    return [(r["rel"], r["tail"]) for r in rows]


def llm_verify_candidate(cand, short_rels, long_rels):
    """규칙으로 판정 불가(일치하는 관계값도, 1:1 관계 충돌도 없음)한 후보만 LLM에 위임.

    실측 발견: Google vs Google Inc.(README에 이미 확인된 실제 언더-머징 사례)가 EMPLOYER
    값 차이(한쪽은 직원 A, 한쪽은 직원 B -- 한 회사가 여러 직원을 갖는 게 정상)만으로
    LLM이 잘못 reject 판정함 -- 규칙 기반 쪽은 FUNCTIONAL_FOR_VERIFICATION으로 이미
    막았지만 LLM 프롬프트엔 같은 가드가 없어서 발생. 프롬프트에 명시적으로 경고 추가."""
    def compute():
        facts_a = "\n".join(f"- {cand['short_name']} -[{r}]-> {t}" for r, t in short_rels) or "(관계 정보 없음)"
        facts_b = "\n".join(f"- {cand['long_name']} -[{r}]-> {t}" for r, t in long_rels) or "(관계 정보 없음)"
        prompt = (
            "다음 두 개체가 표기만 다른 같은 회사(법인 접미사 유무 차이)인지, 아니면 "
            "실제로 별개인 법인인지 판정하세요.\n\n"
            "주의: DEVELOPER/EMPLOYER/PUBLISHER/PRODUCT_OR_MATERIAL_PRODUCED처럼 한 "
            "회사가 원래 여러 값을 가질 수 있는 관계는, 두 개체의 값이 서로 다르다는 "
            "것 자체가 별개 법인이라는 증거가 아닙니다(예: 직원이 다르다고 다른 회사가 "
            "아님 -- 같은 회사도 문서마다 다른 직원이 언급될 수 있음). 이런 관계는 "
            "\"공통점 없음\"으로 취급하고, HEADQUARTERS_LOCATION/COUNTRY/INCEPTION처럼 "
            "정말 하나의 값만 가져야 하는 관계에서 값이 다를 때만 별개 법인의 증거로 "
            "삼으세요.\n\n"
            "반드시 \"merge\" 또는 \"reject\" 한 단어만 출력하세요.\n\n"
            f"개체 A: {cand['short_name']}\n{facts_a}\n\n"
            f"개체 B: {cand['long_name']}\n{facts_b}\n"
        )
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip().lower()
        return "merge" if "merge" in text else "reject"

    verdict = cached(
        cache_key("under_merge_llm_verify_v3", CHAT_MODEL, cand["short_name"], cand["long_name"]),
        compute,
    )
    return {"verdict": verdict, "reason": "규칙만으로 판정 불가 -- LLM에 위임", "method": "llm"}


# "충돌"을 신뢰할 수 있는 건 대체로 값이 하나뿐인(1:1) 관계뿐 -- DEVELOPER/PUBLISHER/
# PRODUCT_OR_MATERIAL_PRODUCED/EMPLOYER처럼 한 개체가 여러 값을 갖는 게 정상인 관계는
# 값이 다르다고 별개 개체라는 신호가 아님. 실측으로 확인: Apple/Apple Inc.(README에서
# 이미 확인된 실제 언더-머징 사례)가 DEVELOPER 값 차이만으로 오탐 기각됐던 걸 이 구분
# 없이 첫 버전에서 직접 봄 -- Google/Google Inc.(EMPLOYER), Sony Computer
# Entertainment/…Inc.(PUBLISHER)도 같은 패턴으로 오탐.
FUNCTIONAL_FOR_VERIFICATION = {
    "DATE_OF_BIRTH", "DATE_OF_DEATH", "PLACE_OF_BIRTH", "PLACE_OF_DEATH",
    "INCEPTION", "DISSOLVED_ABOLISHED_OR_DEMOLISHED",
    "HEADQUARTERS_LOCATION", "COUNTRY", "CAPITAL", "CAPITAL_OF",
}
assert FUNCTIONAL_FOR_VERIFICATION <= RELATION_TYPES


def verify_candidate(session, cand):
    """3단계: 공유하는 관계 타입의 값 일치/충돌로 규칙 기반 판정, 애매하면 LLM.
    일치(agree)는 관계 타입을 안 가리고 신뢰(같은 사실을 공유하면 강한 근거) -- 충돌은
    FUNCTIONAL_FOR_VERIFICATION(1:1 관계)에서만 신뢰."""
    short_rels = get_relations(session, cand["short_id"])
    long_rels = get_relations(session, cand["long_id"])
    short_by_type, long_by_type = {}, {}
    for rel, tail in short_rels:
        short_by_type.setdefault(rel, set()).add(tail)
    for rel, tail in long_rels:
        long_by_type.setdefault(rel, set()).add(tail)

    shared = set(short_by_type) & set(long_by_type)
    agree = [t for t in shared if short_by_type[t] & long_by_type[t]]
    functional_conflict = [
        t for t in shared
        if t in FUNCTIONAL_FOR_VERIFICATION and not (short_by_type[t] & long_by_type[t])
    ]

    # 관계 타입은 달라도 같은 대상을 가리키면(예: 한쪽은 LOCATED_IN_THE_ADMINISTRATIVE_
    # TERRITORIAL_ENTITY, 다른 쪽은 HEADQUARTERS_LOCATION인데 둘 다 값이 'Michigan') 약한
    # 긍정 신호로 취급. 실측 발견: General Motors/General Motors Corporation(README 확인
    # 실제 사례)이 이 패턴이라 같은 타입 매치만 보면 놓치고 LLM에 넘어갔는데, 그때그때
    # LLM 판정이 프롬프트 변경에 따라 흔들리는 것도 확인함(Google Inc. 오탐 고치려고
    # 프롬프트에 경고를 추가했더니 General Motors가 merge->reject로 뒤집힘) -- 이런
    # 식으로 매번 프롬프트를 조정하는 대신, 판정 가능한 신호는 규칙으로 옮겨 LLM 의존도를
    # 낮춤. 이미 접미사 매칭을 통과한 후보(같은 기본 이름)에 한해서만 적용되므로, 흔한
    # 도시 이름 하나 겹친다고 무관한 회사가 잘못 묶일 위험은 낮음.
    short_all_tails = {tail for tails in short_by_type.values() for tail in tails}
    long_all_tails = {tail for tails in long_by_type.values() for tail in tails}
    cross_type_overlap = short_all_tails & long_all_tails

    if functional_conflict:
        return {"verdict": "reject", "reason": f"1:1 관계값 충돌: {functional_conflict}", "method": "rule"}
    if agree:
        return {"verdict": "merge", "reason": f"일치하는 관계값: {agree}", "method": "rule"}
    if cross_type_overlap:
        return {"verdict": "merge", "reason": f"관계 타입은 다르지만 겹치는 대상: {cross_type_overlap}", "method": "rule"}
    return llm_verify_candidate(cand, short_rels, long_rels)


def get_degree(session, entity_id):
    row = session.run(
        "MATCH (e:ZEntity {id: $id}) RETURN COUNT { (e)--() } AS degree", id=entity_id
    ).single()
    return row["degree"]


def merge_entities(session, keep_id, drop_id, keep_name, drop_name):
    """4단계: drop_id의 모든 관계(양방향, 어떤 타입이든)를 keep_id로 재배정하고,
    alias를 합친 뒤 drop_id 노드를 삭제. Aura에서 APOC이 없을 수 있으므로 순수
    Cypher로 타입별 재연결(관계 타입은 파라미터화 불가 -- load_ground_truth.py와
    동일하게 화이트리스트 검증 후 문자열 삽입)."""
    rows = list(session.run(
        """
        MATCH (drop:ZEntity {id: $drop_id})-[r]->(other:ZEntity)
        RETURN 'out' AS dir, type(r) AS rel_type, other.id AS other_id, properties(r) AS props
        UNION
        MATCH (other:ZEntity)-[r]->(drop:ZEntity {id: $drop_id})
        RETURN 'in' AS dir, type(r) AS rel_type, other.id AS other_id, properties(r) AS props
        """,
        drop_id=drop_id,
    ))
    for row in rows:
        rel_type = row["rel_type"]
        if not TYPE_NAME_RE.match(rel_type):
            continue
        if row["dir"] == "out":
            query = f"""
            MATCH (keep:ZEntity {{id: $keep_id}}), (other:ZEntity {{id: $other_id}})
            MERGE (keep)-[r:{rel_type}]->(other)
            SET r += $props
            """
        else:
            query = f"""
            MATCH (keep:ZEntity {{id: $keep_id}}), (other:ZEntity {{id: $other_id}})
            MERGE (other)-[r:{rel_type}]->(keep)
            SET r += $props
            """
        session.run(query, keep_id=keep_id, other_id=row["other_id"], props=row["props"])

    keep_row = session.run("MATCH (e:ZEntity {id: $id}) RETURN e.aliases AS aliases", id=keep_id).single()
    drop_row = session.run("MATCH (e:ZEntity {id: $id}) RETURN e.aliases AS aliases", id=drop_id).single()
    merged_aliases = sorted(set(keep_row["aliases"]) | set(drop_row["aliases"]) | {drop_name})
    session.run("MATCH (e:ZEntity {id: $id}) SET e.aliases = $aliases", id=keep_id, aliases=merged_aliases)
    session.run("MATCH (e:ZEntity {id: $id}) DETACH DELETE e", id=drop_id)


# 수동 오버라이드: Topology Check가 CBS/CBS Corporation을 계층 관계 존재로 기각했지만,
# 그 엣지들을 직접 까보니 전부 "CBS Sports Network" 문서 하나에서 나왔고 근거 문장
# ("CBS Sports Network ... owned by the CBS Corporation")이 실제로는 CBS Sports
# Network(별도 개체)를 가리키는데 CBS 자신에게 잘못 귀속된 것으로 보임(엔티티 링킹
# 오류). Harvard/BAE Systems처럼 두 개체 자체의 관계를 직접 서술하는 진짜 계층 관계와
# 다름 -- README도 CBS/CBS Corporation을 실제 언더-머징 사례로 명시했으므로, 이 한
# 쌍만 Topology Check를 우회해 병합 대상에 포함.
MANUAL_TOPOLOGY_OVERRIDE = {("CBS", "CBS Corporation")}


def run(apply=False):
    with driver.session(database=NEO4J_DATABASE) as session:
        candidates = find_suffix_candidates(session)
        print(f"접미사 후보 쌍: {len(candidates)}개\n")

        accepted = []
        for cand in candidates:
            pair_key = (cand["short_name"], cand["long_name"])
            hits = topology_check(session, cand["short_id"], cand["long_id"])
            if hits and pair_key in MANUAL_TOPOLOGY_OVERRIDE:
                print(f"[MERGE-OVERRIDE] {cand['short_name']!r} vs {cand['long_name']!r} "
                      f"-- 계층 관계 존재하지만 수동 오버라이드(엔티티 링킹 오류로 판단, "
                      f"코드 주석 참고)")
                accepted.append(cand)
                continue
            if hits:
                print(f"[REJECT-TOPOLOGY] {cand['short_name']!r} vs {cand['long_name']!r} "
                      f"-- 직접 계층 관계 존재({hits}), 별개 법인으로 판단")
                continue
            verdict = verify_candidate(session, cand)
            tag = "MERGE" if verdict["verdict"] == "merge" else "REJECT"
            print(f"[{tag}] {cand['short_name']!r} vs {cand['long_name']!r} "
                  f"({verdict['method']}: {verdict['reason']})")
            if verdict["verdict"] == "merge":
                accepted.append(cand)

        print(f"\n최종 병합 대상: {len(accepted)}쌍")
        if not apply:
            print("--dry-run: 실제 병합은 수행하지 않음 (--apply로 재실행하면 병합됨)")
            return accepted

        # 같은 노드(예: "Google")가 여러 후보 쌍에 등장할 수 있음(Google Inc./Google LLC
        # 둘 다 Google과 병합) -- 이전 병합에서 삭제된 id가 나중 쌍에 다시 나오면 최종
        # canonical id로 치환해서 참조하도록 매핑 유지.
        merged_into = {}

        def resolve(entity_id):
            while entity_id in merged_into:
                entity_id = merged_into[entity_id]
            return entity_id

        for cand in accepted:
            short_id = resolve(cand["short_id"])
            long_id = resolve(cand["long_id"])
            if short_id == long_id:
                print(f"    [SKIP] {cand['short_name']!r}/{cand['long_name']!r} -- 이전 병합으로 이미 같은 노드")
                continue
            deg_short = get_degree(session, short_id)
            deg_long = get_degree(session, long_id)
            if deg_long >= deg_short:
                keep_id, keep_name = long_id, cand["long_name"]
                drop_id, drop_name = short_id, cand["short_name"]
            else:
                keep_id, keep_name = short_id, cand["short_name"]
                drop_id, drop_name = long_id, cand["long_name"]
            merge_entities(session, keep_id, drop_id, keep_name, drop_name)
            merged_into[drop_id] = keep_id
            print(f"[MERGED] {drop_name!r} -> {keep_name!r} (연결 수 {max(deg_short, deg_long)} 쪽을 canonical로 채택)")

        return accepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="실제로 병합 수행 (기본은 dry-run)")
    args = parser.parse_args()
    run(apply=args.apply)
    driver.close()
