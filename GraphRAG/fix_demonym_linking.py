"""지명형용사(demonym) 오귀속 탐지 + 수정 -- 2026-07-16 확장판.

README.md "문제 11" 4단계 설계 구현 + "알려진 한계 확장" 대응:
    1. Candidate Detection -- relation_type이 지리/국적 화이트리스트(GEO_RELATION_TYPES)에
       속하는 엣지 중, head/tail 노드 이름이 DEMONYM_MAP에 있고 그 노드 타입이 LOC인 것만
       후보로 탐지. **v1과 달리 evidence_source 필터 없음** -- "Finnish" 하나만 스캔해도
       evidence_source=annotated(사람이 직접 라벨링한 gold 데이터)에도 같은 패턴이 있는 걸
       실측 확인해서, 최초 설계처럼 inferred_bridge만 보면 대부분을 놓친다.
    2. Canonical Target Resolution -- DEMONYM_MAP이 가리키는 정규형(고유명사) 노드가
       이미 :LOC 타입으로 그래프에 있는지 확인, 없으면 스킵(새 노드 생성 안 함).
    3. Evidence Re-check -- LLM으로 evidence 문장이 그 관계 자체를 실제로 뒷받침하는지
       재확인(형태 문제와 무근거 문제를 분리). gold 데이터라고 그냥 믿지 않음 -- 문제 10에서
       이미 confidence=1.0 gold 라벨도 틀린 사례(Republic of China on Taiwan)를 실측
       확인했으므로, evidence_source와 무관하게 전부 재확인.
    4. Edge Repoint -- 노드 병합(DETACH DELETE) 아님. 엣지 하나만 삭제 후 정규형
       노드로 재생성. 지명형용사 노드 자체는 그대로 둔다(다른 문맥, 예: 언어 이름
       관계에서 정당하게 계속 쓰일 수 있음).

주의: OFFICIAL_LANGUAGE/LANGUAGES_SPOKEN_WRITTEN_OR_SIGNED/ORIGINAL_LANGUAGE_OF_WORK/
ETHNIC_GROUP은 지명형용사 형태가 오히려 정답이라(예: "Russia -[OFFICIAL_LANGUAGE]->
Russian") GEO_RELATION_TYPES에 절대 포함하지 않는다.

기존 graphrag_query.py의 RELATION_TYPES 화이트리스트/openai_client/driver를 그대로
재사용 (같은 폴더, 같은 Neo4j 인스턴스).

사용법:
    python GraphRAG/fix_demonym_linking.py --dry-run   # 후보/판정만 출력 (기본값, 그래프 변경 없음)
    python GraphRAG/fix_demonym_linking.py --apply      # 실제로 엣지 재배정 수행
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai

from cache import CACHE_DIR, cache_key
from graphrag_query import CHAT_MODEL, NEO4J_DATABASE, RELATION_TYPES, driver, openai_client

# 3단계(evidence 재확인) 배치 크기 -- 2026-07-16 실측 발견: 개별 호출은 후보마다
# 같은 지시문(~150 토큰)을 매번 반복 전송해서, 후보가 1만 개 단위로 늘어나니 지시문
# 반복분만 백만 토큰 이상 낭비됨(사용자가 OpenAI 대시보드에서 직접 확인하고 지적).
# fix_over_merging.py의 cluster_conflict()처럼 N개를 한 프롬프트에 묶어 요청 수 자체를
# 줄인다 -- 지시문은 배치당 1번만 전송되므로 총 입력 토큰이 크게 준다.
GROUNDING_BATCH_SIZE = 25

# README "문제 11" 확정 화이트리스트 -- 지리/국적 정체성을 묻는 관계만. 언어/민족
# 관계는 지명형용사 형태가 정답이라 절대 포함하지 않는다(위 docstring 참고).
GEO_RELATION_TYPES = {
    "LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY", "COUNTRY", "COUNTRY_OF_CITIZENSHIP",
    "CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY", "COUNTRY_OF_ORIGIN", "APPLIES_TO_JURISDICTION",
    "CONTINENT", "HAS_PART", "PART_OF", "TERRITORY_CLAIMED_BY", "BASIN_COUNTRY",
}
assert GEO_RELATION_TYPES <= RELATION_TYPES

# 정규형(target) 노드가 실제로 :LOC로 존재하는지는 Canonical Target Resolution
# (find_candidates) 단계에서 매 후보마다 확인하므로, 여기 목록이 그래프에 없는
# 나라를 담고 있어도 안전하게 스킵될 뿐(오류 아님) -- 넉넉하게 등재.
# 2026-07-15(문제 11 최초 적용) 28쌍에서, "Finnish" 하나만 놓쳐도 지리 관계에서
# 185건이 빠지는 걸 실측 확인하고(2026-07-16) 대폭 확장.
# 의도적 제외: "Korean"(North/South Korea 중 어느 쪽인지 문맥별로 갈림),
# "Georgian"(국가 Georgia/미국 조지아주 이름이 겹쳐 같은 형용사를 씀),
# "Guinean"/"Congolese"(비슷한 이름의 나라가 여럿-- Guinea/Guinea-Bissau/
# Equatorial Guinea, Republic of Congo/DR Congo)처럼 정답 국가가 하나로 안 정해지는
# 경우 -- evidence 확인 후 수동 판단 필요.
DEMONYM_MAP = {
    "American": "United States", "Chinese": "China", "German": "Germany",
    "Austrian": "Austria", "Norwegian": "Norway", "Bavarian": "Bavaria",
    "Asian": "Asia", "Canadian": "Canada", "Icelandic": "Iceland",
    "Taiwanese": "Taiwan", "British": "United Kingdom", "French": "France",
    "Russian": "Russia", "Japanese": "Japan", "Indian": "India",
    "Australian": "Australia", "Italian": "Italy", "Spanish": "Spain",
    "Mexican": "Mexico", "Brazilian": "Brazil", "Egyptian": "Egypt",
    "Turkish": "Turkey", "Israeli": "Israel", "Swedish": "Sweden",
    "Dutch": "Netherlands", "Polish": "Poland", "Greek": "Greece",
    "Irish": "Ireland",
    # 2026-07-16 확장분
    "Finnish": "Finland", "Danish": "Denmark", "Belgian": "Belgium",
    "Portuguese": "Portugal", "Swiss": "Switzerland", "Scottish": "Scotland",
    "Welsh": "Wales", "Vietnamese": "Vietnam", "Thai": "Thailand",
    "Filipino": "Philippines", "Philippine": "Philippines",
    "Malaysian": "Malaysia", "Singaporean": "Singapore",
    "Indonesian": "Indonesia", "Pakistani": "Pakistan",
    "Bangladeshi": "Bangladesh", "Nepali": "Nepal",
    "Nigerian": "Nigeria", "Kenyan": "Kenya", "Ethiopian": "Ethiopia",
    "Ghanaian": "Ghana", "Moroccan": "Morocco", "Algerian": "Algeria",
    "Tunisian": "Tunisia", "Libyan": "Libya", "Sudanese": "Sudan",
    "Iraqi": "Iraq", "Iranian": "Iran", "Saudi": "Saudi Arabia",
    "Emirati": "United Arab Emirates", "Qatari": "Qatar",
    "Kuwaiti": "Kuwait", "Jordanian": "Jordan", "Lebanese": "Lebanon",
    "Syrian": "Syria", "Yemeni": "Yemen", "Afghan": "Afghanistan",
    "Chilean": "Chile", "Argentine": "Argentina",
    "Argentinian": "Argentina", "Peruvian": "Peru",
    "Colombian": "Colombia", "Venezuelan": "Venezuela",
    "Ecuadorian": "Ecuador", "Bolivian": "Bolivia",
    "Paraguayan": "Paraguay", "Uruguayan": "Uruguay", "Cuban": "Cuba",
    "Jamaican": "Jamaica", "Haitian": "Haiti",
    "Costa Rican": "Costa Rica", "Panamanian": "Panama",
    "Guatemalan": "Guatemala", "Honduran": "Honduras",
    "Ukrainian": "Ukraine", "Armenian": "Armenia",
    "Azerbaijani": "Azerbaijan", "Kazakh": "Kazakhstan",
    "Uzbek": "Uzbekistan", "Mongolian": "Mongolia",
    "Hungarian": "Hungary", "Czech": "Czech Republic",
    "Slovak": "Slovakia", "Romanian": "Romania", "Bulgarian": "Bulgaria",
    "Croatian": "Croatia", "Serbian": "Serbia", "Slovenian": "Slovenia",
    "Bosnian": "Bosnia and Herzegovina", "Macedonian": "Macedonia",
    "Albanian": "Albania", "Montenegrin": "Montenegro",
    "Moldovan": "Moldova", "Belarusian": "Belarus",
    "Estonian": "Estonia", "Latvian": "Latvia", "Lithuanian": "Lithuania",
    "Cypriot": "Cyprus", "Maltese": "Malta",
    "Luxembourgish": "Luxembourg", "Fijian": "Fiji",
    "Burmese": "Myanmar", "Cambodian": "Cambodia", "Laotian": "Laos",
}

TYPE_NAME_RE_ALLOWED = set(GEO_RELATION_TYPES)


def find_candidates(session):
    """1+2단계: 후보 엣지 탐지 + 정규형 노드 존재 확인. v1과 달리 evidence_source
    필터 없음 -- annotated(gold)/inferred_cooccurrence/unresolved_multihop/
    model_provided 전부 포함(docstring 참고)."""
    candidates = []
    canonical_cache = {}  # 같은 canonical_name을 매 후보마다 다시 조회하지 않도록 캐싱
    for rel in sorted(GEO_RELATION_TYPES):
        rows = list(session.run(
            f"""
            MATCH (h:ZEntity)-[r:{rel}]->(t:ZEntity)
            WHERE (t.name IN $demonyms AND t.type = 'LOC')
               OR (h.name IN $demonyms AND h.type = 'LOC')
            RETURN elementId(r) AS rel_id, h.id AS h_id, h.name AS h_name,
                   t.id AS t_id, t.name AS t_name,
                   r.evidence AS evidence, r.document AS document,
                   r.evidence_source AS evidence_source, properties(r) AS props
            """,
            demonyms=list(DEMONYM_MAP.keys()),
        ))
        for row in rows:
            if row["t_name"] in DEMONYM_MAP:
                side, wrong_name = "tail", row["t_name"]
            else:
                side, wrong_name = "head", row["h_name"]

            canonical_name = DEMONYM_MAP[wrong_name]
            if canonical_name not in canonical_cache:
                canonical = session.run(
                    "MATCH (e:ZEntity {name: $name, type: 'LOC'}) RETURN e.id AS id LIMIT 1",
                    name=canonical_name,
                ).single()
                canonical_cache[canonical_name] = canonical["id"] if canonical else None
            canonical_id = canonical_cache[canonical_name]
            if canonical_id is None:
                continue  # 정규형 노드가 없으면 새로 만들지 않고 스킵(안전 우선)

            candidates.append({
                "rel_id": row["rel_id"], "relation_type": rel, "side": side,
                "h_id": row["h_id"], "h_name": row["h_name"],
                "t_id": row["t_id"], "t_name": row["t_name"],
                "wrong_name": wrong_name,
                "canonical_id": canonical_id, "canonical_name": canonical_name,
                "evidence": row["evidence"], "document": row["document"], "props": row["props"],
                "evidence_source": row["evidence_source"],
            })
    return candidates


def _grounding_cache_key(cand):
    return cache_key(
        "demonym_grounding_check_v1", CHAT_MODEL, cand["h_name"], cand["relation_type"],
        cand["t_name"], cand["document"],
    )


def _cache_peek(key):
    """cache.cached()와 같은 파일 규칙을 쓰되, 없으면 계산하지 않고 None만 반환
    (배치 실행 전 이미 캐시된 후보를 걸러내 재호출 안 하기 위함)."""
    path = CACHE_DIR / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _cache_write(key, value):
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _call_with_retry(prompt):
    # 500 RPM 한도를 스레드풀 동시 호출이 넘겨서 실측으로 RateLimitError 발생
    # (600/1466 지점에서 크래시, 배치 전환 이후에도 유효한 안전장치) -- 지수 백오프 재시도.
    for attempt in range(6):
        try:
            return openai_client.chat.completions.create(
                model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}],
            )
        except openai.RateLimitError:
            time.sleep(2 ** attempt)
    raise RuntimeError("RateLimitError 재시도 초과")


def verify_grounding_one(cand):
    """3단계 개별판(배치 파싱 실패 시 폴백 전용). evidence가 관계 자체를 실제로
    뒷받침하는지 재확인 -- 형태(형용사형 vs 고유명사형) 문제와 애초에 무근거인 문제를
    분리."""
    evidence = " ".join(cand["evidence"]) if cand["evidence"] else "(근거 문장 없음)"
    prompt = (
        f"다음은 지식그래프의 관계 하나입니다: \"{cand['h_name']}\" -[{cand['relation_type']}]-> "
        f"\"{cand['t_name']}\"\n"
        f"근거 문장: {evidence}\n\n"
        f"이 근거 문장이 이 관계(\"{cand['h_name']}\"이(가) {cand['relation_type']} 관계로 "
        f"\"{cand['t_name']}\"와(과) 실제로 연결됨)를 뒷받침합니까? "
        "표기가 형용사형인지 고유명사형인지는 무시하고(예: China/Chinese는 같은 것으로 취급), "
        "관계 자체가 근거 문장에서 실제로 성립하는지만 판단하세요. "
        "\"yes\" 또는 \"no\" 한 단어만 출력하세요."
    )
    resp = _call_with_retry(prompt)
    text = resp.choices[0].message.content.strip().lower()
    return "yes" if "yes" in text else "no"


def verify_grounding_batch(batch):
    """3단계 배치판: 후보 여러 개를 LLM 호출 1번으로 판정 -- fix_over_merging.py의
    cluster_conflict()와 같은 배치 패턴. 지시문을 배치당 1번만 보내므로 개별 호출보다
    총 입력 토큰이 크게 줄어듦(실측: 1만 개 단위에서 지시문 반복분만 백만 토큰 이상
    낭비되는 걸 사용자가 OpenAI 대시보드에서 직접 확인). 파싱 실패(개수 불일치 등) 시
    None을 반환해 호출부가 개별 호출로 안전하게 폴백하게 한다."""
    lines = []
    for i, cand in enumerate(batch):
        evidence = " ".join(cand["evidence"]) if cand["evidence"] else "(근거 문장 없음)"
        lines.append(
            f"[{i}] \"{cand['h_name']}\" -[{cand['relation_type']}]-> \"{cand['t_name']}\"\n"
            f"    근거: {evidence}"
        )
    prompt = (
        "다음은 지식그래프의 관계 후보 목록입니다. 각 항목에 대해 근거 문장이 그 관계 "
        "자체를 실제로 뒷받침하는지 판단하세요. 표기가 형용사형인지 고유명사형인지는 "
        "무시하고(예: China/Chinese는 같은 것으로 취급), 관계 자체가 근거 문장에서 실제로 "
        "성립하는지만 보세요.\n\n"
        + "\n".join(lines)
        + f"\n\n반드시 JSON 배열만 출력하세요(다른 설명 없이): 각 항목(위 인덱스 순서 그대로)에 "
        f"대해 \"yes\" 또는 \"no\" 문자열로 이뤄진 배열, 길이는 반드시 {len(batch)}."
    )
    resp = _call_with_retry(prompt)
    text = resp.choices[0].message.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        verdicts = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(verdicts, list) or len(verdicts) != len(batch):
        return None
    return ["yes" if str(v).strip().lower() == "yes" else "no" for v in verdicts]


def verify_grounding_all(candidates, batch_size=GROUNDING_BATCH_SIZE, max_workers=8, progress=True):
    """전체 후보에 대해 캐시를 먼저 확인하고, 없는 것만 배치로 나눠 병렬 호출.
    캐시는 후보 1개 단위로 저장하므로(_grounding_cache_key) 배치 경계가 실행마다
    달라져도 이미 계산된 건 재호출 안 함."""
    n = len(candidates)
    verdicts = [None] * n
    keys = [_grounding_cache_key(c) for c in candidates]
    to_compute = []
    for i, key in enumerate(keys):
        cached_verdict = _cache_peek(key)
        if cached_verdict is None:
            to_compute.append(i)
        else:
            verdicts[i] = cached_verdict
    if progress:
        print(f"  캐시 재사용: {n - len(to_compute)}개, 새로 호출 필요: {len(to_compute)}개")

    batches = [to_compute[i:i + batch_size] for i in range(0, len(to_compute), batch_size)]

    def process_batch(idx_batch):
        batch_cands = [candidates[i] for i in idx_batch]
        results = verify_grounding_batch(batch_cands)
        if results is None:
            # 배치 파싱 실패 -- 이 배치만 개별 호출로 안전하게 폴백
            results = [verify_grounding_one(c) for c in batch_cands]
        for i, v in zip(idx_batch, results):
            _cache_write(keys[i], v)
        return list(zip(idx_batch, results))

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_batch, b) for b in batches]
        for future in as_completed(futures):
            for i, v in future.result():
                verdicts[i] = v
            done += 1
            if progress and done % 20 == 0:
                print(f"  ...배치 진행 {done}/{len(batches)}")

    return verdicts


def repoint_edge(session, cand):
    """4단계: 엣지 하나만 재배정(노드 병합 아님). 지명형용사 노드는 그대로 두고
    다른 엣지에서 계속 쓰일 수 있게 둔다."""
    session.run("MATCH ()-[r]-() WHERE elementId(r) = $eid DELETE r", eid=cand["rel_id"])
    if cand["side"] == "tail":
        query = f"""
        MATCH (h:ZEntity {{id: $h_id}}), (c:ZEntity {{id: $canonical_id}})
        CREATE (h)-[r:{cand['relation_type']}]->(c)
        SET r = $props
        """
        session.run(query, h_id=cand["h_id"], canonical_id=cand["canonical_id"], props=cand["props"])
    else:
        query = f"""
        MATCH (c:ZEntity {{id: $canonical_id}}), (t:ZEntity {{id: $t_id}})
        CREATE (c)-[r:{cand['relation_type']}]->(t)
        SET r = $props
        """
        session.run(query, canonical_id=cand["canonical_id"], t_id=cand["t_id"], props=cand["props"])


def run(apply=False, max_workers=8, verbose=True):
    with driver.session(database=NEO4J_DATABASE) as session:
        candidates = find_candidates(session)
        print(f"후보 엣지: {len(candidates)}개")

        src_counts = {}
        for c in candidates:
            src_counts[c["evidence_source"]] = src_counts.get(c["evidence_source"], 0) + 1
        print("evidence_source별 분포:", dict(sorted(src_counts.items(), key=lambda kv: -kv[1])))
        print()

        # 3단계(evidence 재확인)는 GROUNDING_BATCH_SIZE개씩 묶어 배치 호출(위 함수 참고).
        verdicts = verify_grounding_all(candidates, max_workers=max_workers)

        repoint_list, ungrounded_list = [], []
        for i, cand in enumerate(candidates):
            label = f"[{cand['document']}] {cand['h_name']} -[{cand['relation_type']}]-> {cand['t_name']} ({cand['evidence_source']})"
            if verdicts[i] == "yes":
                repoint_list.append(cand)
            elif verbose:
                print(f"[UNGROUNDED-SKIP] {label} -- 근거가 관계를 뒷받침 안 함, 재배정 안 함")
                ungrounded_list.append(cand)
            else:
                ungrounded_list.append(cand)

        repoint_src_counts = {}
        for c in repoint_list:
            repoint_src_counts[c["evidence_source"]] = repoint_src_counts.get(c["evidence_source"], 0) + 1
        print(f"\n재배정 대상: {len(repoint_list)}개(evidence_source별: {dict(sorted(repoint_src_counts.items(), key=lambda kv: -kv[1]))}), "
              f"무근거로 제외: {len(ungrounded_list)}개")
        if not apply:
            print("--dry-run: 실제 반영은 안 함 (--apply로 재실행하면 반영됨)")
            return repoint_list, ungrounded_list

        for cand in repoint_list:
            repoint_edge(session, cand)
        print(f"[DONE] {len(repoint_list)}개 엣지 재배정 완료")
        return repoint_list, ungrounded_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="실제로 엣지 재배정 수행 (기본은 dry-run)")
    args = parser.parse_args()
    run(apply=args.apply)
    driver.close()
