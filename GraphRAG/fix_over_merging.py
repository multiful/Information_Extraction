"""오버-머징(동명이인/동명 조직이 한 노드로 합쳐진 경우) 탐지 + 수정.

README.md "문제 1" 3단계 설계 구현:
    1. Attribute Conflict Detection -- functional relation(1:1 관계, 화이트리스트로
       엄선)에서 한 엔티티가 서로 다른 값을 2개 이상 갖는 경우를 후보로 탐지.
    2. LLM-based Entity Verification -- 후보의 각 값이 나온 원본 evidence 문장을
       함께 LLM에 주고, 이진 판정이 아니라 N개 그룹으로 클러스터링.
    3. Mention-level Graph Repair -- 각 관계 엣지가 이미 document 속성을 갖고
       있으므로, Stage 2의 "문서 -> 그룹" 매핑을 그대로 따라 그 문서에서 나온
       엣지(어떤 관계 타입이든)를 그룹별 새 엔티티로 재배정.

**알려진 한계** (README에 이미 명시): Stage 1이 감시 중인 functional relation에
충돌이 있는 문서만 포착하므로, 그 관계 자체가 없는 문서(예: 이 엔티티를 다른 관계로만
언급하는 문서)는 이 파이프라인 밖이라 원래 병합 노드에 그대로 남는다 -- "증거가 남아있는
케이스"를 잡는 실용적 1차 필터.

사용법:
    python GraphRAG/fix_over_merging.py --dry-run   # 후보/클러스터링 결과만 출력 (기본값)
    python GraphRAG/fix_over_merging.py --apply      # 실제로 노드 분리 수행
"""

import argparse
import json
import re

from cache import cache_key, cached
from graphrag_query import CHAT_MODEL, NEO4J_DATABASE, RELATION_TYPES, driver, openai_client

# README가 명시한, 진짜 1:1이라 안전한 functional relation만 사용. COUNTRY_OF_CITIZENSHIP
# (이중국적 가능)/SPOUSE(재혼 가능)처럼 원래 여러 값이 정상인 관계는 제외.
FUNCTIONAL_RELATIONS = [
    "DATE_OF_BIRTH", "INCEPTION", "PLACE_OF_BIRTH", "DISSOLVED_ABOLISHED_OR_DEMOLISHED",
]
assert set(FUNCTIONAL_RELATIONS) <= RELATION_TYPES

TYPE_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
VALID_ENTITY_TYPES = {"PER", "ORG", "LOC", "TIME", "NUM", "MISC"}

_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def _extract_year(value):
    m = _YEAR_RE.search(value)
    return m.group(1) if m else None


def _distinct_value_groups(values):
    """날짜값의 표기 정밀도 차이("1991" vs "December 1991" vs "December 26, 1991")를
    실제 충돌로 오인하지 않도록, 같은 연도로 추출되는 값들은 하나의 그룹으로 묶는다.
    실측 발견: Soviet Union의 DISSOLVED_ABOLISHED_OR_DEMOLISHED가 '1991'/'December
    1991'/'December 26, 1991' 3가지 표기로 갈렸는데 전부 같은 사건(1991년 해체)이라
    문자열 단순 비교로는 가짜 충돌 3건이 잡혔던 것을 반영한 보정. 연도를 못 뽑는 값
    (지명 등)은 문자열 그대로 각자 그룹."""
    groups = {}
    for v in values:
        year = _extract_year(v)
        key = ("year", year) if year else ("raw", v)
        groups.setdefault(key, []).append(v)
    return list(groups.values())


def find_conflict_candidates(session, relation_type):
    """1단계: relation_type에서 서로 다른 값을 2개 이상 갖는 엔티티를 탐지.
    occurrences: 각 (value, document, evidence) -- 같은 문서에서 같은 값이 여러 번
    잡히는 경우는 문서 단위로 대표 1개만 남긴다(이후 문서->그룹 매핑에 문서당 값 1개면 충분)."""
    if relation_type not in RELATION_TYPES:
        return []
    rows = list(session.run(
        f"""
        MATCH (e:ZEntity)-[r:{relation_type}]->(v:ZEntity)
        RETURN e.id AS id, e.name AS name, e.type AS type, e.aliases AS aliases,
               collect({{value: v.name, document: r.document, evidence: r.evidence}}) AS occurrences
        """
    ))
    candidates = []
    for row in rows:
        by_doc = {}
        for occ in row["occurrences"]:
            by_doc.setdefault(occ["document"], occ)  # 문서당 대표 occurrence 1개
        occurrences = list(by_doc.values())
        distinct_values = {occ["value"] for occ in occurrences}
        if len(_distinct_value_groups(distinct_values)) >= 2:
            candidates.append({
                "id": row["id"], "name": row["name"], "type": row["type"], "aliases": row["aliases"],
                "relation_type": relation_type, "occurrences": occurrences,
            })
    return candidates


def cluster_conflict(candidate):
    """2단계: 충돌하는 (값, 문서, 근거) 목록을 LLM에 주고 실제 몇 명/몇 개 조직인지
    클러스터링. 반환: {document: group_id} 매핑, 실패 시 None."""
    occs = candidate["occurrences"]

    def compute():
        lines = []
        for i, occ in enumerate(occs):
            evidence = " ".join(occ["evidence"]) if occ["evidence"] else "(근거 문장 없음)"
            lines.append(f"[{i}] 값={occ['value']!r} 문서={occ['document']!r}\n    근거: {evidence}")
        prompt = (
            f"\"{candidate['name']}\"라는 이름의 개체가 지식그래프에서 하나의 노드로 병합돼 "
            f"있는데, {candidate['relation_type']} 값이 서로 다른 항목이 여러 개 있습니다 "
            "(동명이인/동명 조직이 여러 개 섞여있을 가능성). 각 항목의 근거 문장을 보고, "
            "같은 실존 개체를 가리키는 항목끼리 그룹으로 묶으세요.\n\n"
            + "\n".join(lines)
            + "\n\n반드시 JSON 배열만 출력하세요 (다른 설명 없이): 각 항목(위 인덱스 순서 "
            "그대로)이 속한 그룹 번호(0부터 시작하는 정수)로 이뤄진 배열. 예:"
            ' [0, 1, 0, 2] (항목 0과 2는 같은 개체, 1과 3은 각각 다른 개체). 근거가 '
            "부족해 도저히 구분 안 되는 항목들은 같은 그룹(보수적으로 병합 유지)으로 두세요."
        )
        resp = openai_client.chat.completions.create(
            model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)

    key = cache_key(
        "over_merge_cluster_v1", CHAT_MODEL, candidate["name"], candidate["relation_type"],
        [(occ["value"], occ["document"]) for occ in occs],
    )
    group_ids = cached(key, compute)
    if not isinstance(group_ids, list) or len(group_ids) != len(occs):
        return None  # LLM이 개수를 안 맞추면 안전하게 이 후보는 보류(그래프 변경 안 함)

    doc_to_group = {}
    for occ, gid in zip(occs, group_ids):
        doc_to_group[occ["document"]] = gid
    return doc_to_group


def get_degree(session, entity_id):
    row = session.run(
        "MATCH (e:ZEntity {id: $id}) RETURN COUNT { (e)--() } AS degree", id=entity_id
    ).single()
    return row["degree"] if row else 0


def split_entity(session, candidate, doc_to_group, apply=False):
    """3단계: 문서 -> 그룹 매핑을 따라, 각 문서에서 나온 엣지(관계 타입 무관)를
    그룹별 새 엔티티로 재배정. 그룹 0(가장 앞 인덱스가 속한 그룹)은 원래 노드를 그대로
    쓰고, 나머지 그룹은 새 노드를 만든다. alias는 원본 것을 그대로 복사(문서별로
    어떤 alias가 쓰였는지는 Neo4j 스키마상 추적 불가 -- 안전하지만 다소 부정확한
    단순화, 알려진 한계로 로그에 남김)."""
    n_groups = len(set(doc_to_group.values()))
    if n_groups <= 1:
        print(f"    [SKIP] 클러스터링 결과 그룹이 1개뿐 -- 실제로는 충돌 아닐 수 있음, 보류")
        return

    for gid in sorted(set(doc_to_group.values())):
        docs = sorted(d for d, g in doc_to_group.items() if g == gid)
        print(f"    그룹 {gid}: {docs}")

    if not apply:
        return

    if candidate["type"] not in VALID_ENTITY_TYPES:
        print(f"    [SKIP] 알 수 없는 엔티티 타입 {candidate['type']!r} -- 안전하게 건너뜀")
        return

    keep_id = candidate["id"]
    rows = list(session.run(
        """
        MATCH (e:ZEntity {id: $id})-[r]->(other:ZEntity)
        WHERE r.document IN $docs
        RETURN 'out' AS dir, elementId(r) AS rel_elem_id, type(r) AS rel_type,
               other.id AS other_id, r.document AS document, properties(r) AS props
        UNION
        MATCH (other:ZEntity)-[r]->(e:ZEntity {id: $id})
        WHERE r.document IN $docs
        RETURN 'in' AS dir, elementId(r) AS rel_elem_id, type(r) AS rel_type,
               other.id AS other_id, r.document AS document, properties(r) AS props
        """,
        id=keep_id, docs=list(doc_to_group.keys()),
    ))

    group_node_id = {}
    gids_sorted = sorted(set(doc_to_group.values()))
    group_node_id[gids_sorted[0]] = keep_id  # 첫 그룹은 원래 노드 재사용
    for gid in gids_sorted[1:]:
        # id는 그룹 번호(gid) 대신 그 그룹의 대표 문서명 기반으로 만든다 -- 같은
        # 이름("Jones" 등)이 여러 functional relation에서 각자 독립적으로 클러스터링
        # 되면 그룹 번호가 우연히 겹쳐서(둘 다 group 1 등) 실측으로 실제 충돌 발생
        # (Jones가 DATE_OF_BIRTH로 먼저 5분리된 뒤 PLACE_OF_BIRTH가 다시 처리하려다
        # 'Jones::PER#split1' 중복 생성 시도 -> ConstraintError). 문서명 기반이면
        # 실질적으로 충돌 안 하고, MERGE(CREATE 아님)라 혹시 이미 존재해도(이번처럼
        # 이전 관계에서 이미 분리해 사실상 처리할 게 없는 경우) 에러 없이 통과한다.
        docs_in_group = sorted(d for d, g in doc_to_group.items() if g == gid)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", docs_in_group[0]).strip("_")
        new_id = f"{keep_id}#{slug}"
        session.run(
            f"""
            MERGE (e:ZEntity {{id: $id}})
            ON CREATE SET e.name = $name, e.type = $type, e.aliases = $aliases
            SET e:{candidate['type']}
            """,
            id=new_id, name=candidate["name"], type=candidate["type"], aliases=candidate["aliases"],
        )
        group_node_id[gid] = new_id

    moved = 0
    for row in rows:
        gid = doc_to_group.get(row["document"])
        if gid is None:
            continue
        target_id = group_node_id[gid]
        if target_id == keep_id:
            continue  # 이미 keep_id에 붙어있으니 재배정 불필요
        rel_type = row["rel_type"]
        if not TYPE_NAME_RE.match(rel_type):
            continue
        session.run("MATCH ()-[r]-() WHERE elementId(r) = $eid DELETE r", eid=row["rel_elem_id"])
        if row["dir"] == "out":
            query = f"""
            MATCH (target:ZEntity {{id: $target_id}}), (other:ZEntity {{id: $other_id}})
            CREATE (target)-[r:{rel_type}]->(other)
            SET r = $props
            """
        else:
            query = f"""
            MATCH (other:ZEntity {{id: $other_id}}), (target:ZEntity {{id: $target_id}})
            CREATE (other)-[r:{rel_type}]->(target)
            SET r = $props
            """
        session.run(query, target_id=target_id, other_id=row["other_id"], props=row["props"])
        moved += 1

    print(f"    [SPLIT 완료] {candidate['name']!r} -> {n_groups}개 노드, 엣지 {moved}개 재배정")


def run(apply=False):
    with driver.session(database=NEO4J_DATABASE) as session:
        all_candidates = []
        for relation_type in FUNCTIONAL_RELATIONS:
            cands = find_conflict_candidates(session, relation_type)
            print(f"{relation_type}: 충돌 후보 {len(cands)}개")
            all_candidates.extend(cands)
        print()

        results = []
        for cand in all_candidates:
            n_values = len({occ["value"] for occ in cand["occurrences"]})
            print(f"[{cand['relation_type']}] {cand['name']!r} -- 충돌값 {n_values}개, "
                  f"문서 {len(cand['occurrences'])}개")
            doc_to_group = cluster_conflict(cand)
            if doc_to_group is None:
                print("    [SKIP] LLM 클러스터링 실패(형식 불일치) -- 보류")
                continue
            split_entity(session, cand, doc_to_group, apply=apply)
            results.append((cand, doc_to_group))

        if not apply:
            print("\n--dry-run: 실제 노드 분리는 수행하지 않음 (--apply로 재실행하면 분리됨)")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="실제로 노드 분리 수행 (기본은 dry-run)")
    args = parser.parse_args()
    run(apply=args.apply)
    driver.close()
