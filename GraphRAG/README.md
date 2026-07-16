# GraphRAG 고도화

> **최종 업데이트**: 2026-07-16 (**문제 8 반복 샘플링+다수결 자동화 완료**):
> `majority_vote_answer()` 추가 — `bfs`/`1hop_fallback_bfs` 라우팅에서만 3회
> 샘플링(1hop/property_scan은 원래 안정적이라 비용 아끼려고 제외), 3표 만장일치면
> 그룹화 호출 자체 생략. **그룹화를 처음엔 임베딩으로 하려다 실측으로 폐기** —
> "2013년" vs "2016년"(다른 사실) 유사도가 0.9727, 진짜 같은 사실의 다른 문구
> 유사도는 0.9894로 격차 0.017밖에 안 나 안전한 임계값을 못 잡음(문장 틀이 90%
> 겹치고 핵심 사실만 다르면 임베딩이 둔감 — `rerank_facts()`에서 겪은 것과 같은
> 함정). `fix_over_merging.py`의 `cluster_conflict()`와 같은 LLM 클러스터링
> 패턴으로 교체해 해결. **정직한 한계**: AirAsia Zest 합병 질문은 6회 샘플링에서
> 정답/모름/오답이 진짜 3분의 1씩 고르게 갈리는 케이스라 n=3 다수결로도 종종
> 동점이 남(다른 BFS 질문들은 전부 3표 만장일치로 안정적). 상세는 아래 "문제 8"
> 섹션 참고.
>
> 이전 (2026-07-16, **서브그래프 노이즈 감소 + 임베딩 리랭킹 구현,
> Cypher 비결정성 근본 원인 발견**): "리랭킹 도입 전에 무관한 사실부터 줄여달라"는
> 요청으로 3단계 작업 진행.
>
> **1) 무료 노이즈 감소**: `SUBGRAPH_FACT_LIMIT_PER_HOP` 200→60(실측: Philippines
> 900→275개, Health Sciences Centre 495→240개, 12개 baseline 회귀 없음, 작은
> 클러스터는 영향 없음) + `expand_subgraph()`에 완전 중복 사실 제거 추가(AirAsia
> Zest 66개 중 18개(27%)가 순수 중복이었음, 재방문 traversal 때문).
>
> **2) 임베딩 리랭킹 (`rerank_facts()`, `text-embedding-3-small`)**: 사실 80개
> 초과 시에만 발동(작은 질의는 임베딩 호출 자체 생략), 질문과의 코사인 유사도 상위
> 80개만 사용. **실측으로 top_k=40은 너무 공격적임을 발견** — "Outotec의 필터가
> 만들어지는 도시는 어느 지역에 속해있어?"의 정답 사실이 같은 head+relation의
> 다른 tail("South-East Finland" 등 지리적으로 인접한 오답 후보)과 임베딩 유사도가
> 거의 붙어서 top 40 밖으로 밀려나는 회귀를 실측 확인, 80으로 올려 해결.
>
> **3) 진짜 근본 원인 발견 — Cypher `LIMIT` without `ORDER BY`의 비결정성**:
> 위 작업 검증 중 "AirAsia Zest 합병 질문"이 완전히 동일한 조건에서도 답이 계속
> 바뀌는 걸 추적하다가, `expand_subgraph()`의 Cypher 쿼리가 `LIMIT`만 있고
> `ORDER BY`가 없어서 **결과 집합이 LIMIT을 넘을 때 어떤 행이 잘리는지 자체가
> 매 실행마다 달라질 수 있다는 것**을 발견 — 이번 세션 내내 관찰된 "같은 질문인데
> 가끔 다른 사실이 나온다"류 비결정성의 상당 부분이 여기서 왔을 가능성. `ORDER BY
> head, relation, tail`을 추가해 같은 질문·같은 그래프 상태에서는 항상 같은 사실
> 집합이 나오도록 고정. **다만 이렇게 고정한 뒤에도 AirAsia Zest 합병 질문은
> 여전히 흔들림**(6회 샘플링: 정답 2/모름 2/오답 2) — 즉 이 질문의 불안정성은
> 검색(노이즈/순서) 문제가 아니라 순수하게 **답변 생성 자체의 비결정성(문제 8)**
> 이었다는 게 실측으로 확정됨(temperature=0으로도 완전한 결정성이 보장 안 된다는
> 기존 관찰과 일치). 노이즈 감소·리랭킹 자체가 새 회귀를 만든 건 아니었음 — 12개
> baseline 재검증 결과 11/12 정답, 유일한 실패가 바로 이 pre-existing 케이스.
>
> 이전 (2026-07-16, **문제 11 확장판 적용 완료 — demonym 오귀속, evidence_source
> 전체/gold 데이터 포함**): "알려진 한계 확장"에서 발견한 사각지대(demonym 목록 협소 +
> `inferred_bridge`만 스캔)를 실제로 고침. `DEMONYM_MAP`을 28개→약 100개로 확장하고
> `evidence_source` 필터를 제거해 전체 111,925개 엣지를 대상으로 재스캔 — 후보
> **10,489개**(그중 `annotated`/gold **4,098개** 포함, 원래 예상보다 훨씬 큼).
>
> **비용 사고 + 수습**: 처음엔 개별 LLM 호출(요청당 지시문 반복)로 돌리다 사용자가
> OpenAI 대시보드에서 직접 토큰 사용량을 확인하고 지적 — 예산이 빠듯한 개인 API
> 키(월 $5, 크레딧 잔액 $2.08)라 무계획 대량 호출은 실제 비용 문제. 프로세스를
> 즉시 중단하고(이미 4,188개는 개별 호출로 처리돼 비용 발생, 되돌릴 수 없음 — 다만
> 캐시에 남아 재사용됨) `fix_over_merging.py`의 `cluster_conflict()`와 같은 배치
> 패턴(25개씩 묶어 JSON 배열 응답)으로 재작성. 남은 6,301개를 253번의 배치 호출로
> 처리(예상 입력 ~542K/출력 ~19K 토큰, 개별 호출 대비 훨씬 적음) — 사용자 확인 후
> 실행. 이 교훈은 `CLAUDE.md`에 "LLM 호출 비용 최적화" 섹션으로 별도 기록.
>
> **최종 결과**: dry-run에서 재배정 대상 **4,911개**(evidence_source별: annotated
> 2,774 / inferred_cooccurrence 1,567 / model_provided 223 / inferred_bridge 219 /
> inferred_mention_union 83 / unresolved_multihop 45), 무근거로 제외 5,578개. gold
> 데이터 표본 15개 + 그 외 소스 10개 직접 검증(전부 정확) 후 `--apply` 실행 —
> **4,910개 엣지 실제 반영**(순차 Cypher 쓰기라 완료까지 약 17분 소요, 이 단계는
> LLM 호출 없어 추가 비용 없음). 재검증 스캔 결과 재배정 대상 0개로 안정화 확인.
> **문제 11 누적 총계: 495개(1차, inferred_bridge만) + 4,910개(2차, 전체 evidence_
> source) = 5,405개 엣지 정정.**
>
> 이전 (2026-07-16, **문제 7 MAX_HOPS 5로 적용 + 문제 8 temperature=0
> 적용 중 새 발견 2건**): 사용자가 "문제 7을 5로 했는데 괜찮은지" 질문 — 코드 확인
> 결과 `MAX_HOPS`는 여전히 3이었음(README의 실험 기록과 실제 코드가 어긋나 있었음).
> README의 기존 실험(3→5, 10개 질의 회귀 0건+1건 개선)을 근거로 실제로 5로 적용.
>
> 검증 중 `answer_with_subgraph()`/`extract_entities()` 둘 다 temperature 미설정
> 상태였던 것도 확인(문제 8 미구현 상태 그대로) — MAX_HOPS 변경으로 캐시가 새로
> 생성되는 김에 문제 8도 같이 최소 적용(`temperature=0`, 완전한 결정성 보장은 아님).
> 적용 후 실측으로 **두 가지를 새로 발견**:
> 1. **temperature=0의 부작용**: "Outotec의 필터가 만들어지는 도시는 어느 지역에
>    속해있어?"가 temperature=0 적용 직후 4/4 샘플 전부 `LOCATED_IN_THE_
>    ADMINISTRATIVE_TERRITORIAL_ENTITY`로 고정 오분류(1-hop 조기 종료 → "모름") —
>    즉 temperature=0은 "가끔 틀림"을 "항상 틀림"으로 고정시킬 수도 있음(분산 감소가
>    정확도 보장은 아님). `extract_entities()` 프롬프트에 이 정확한 문구를 다루는
>    대조 예시(예5) 추가로 수정, 재검증 5/5 정답.
> 2. **문제 11의 demonym 목록이 크게 불완전했음**: "Finnish"가 목록에 없어서 위
>    Outotec 문제의 근본 원인 중 하나(`Outotec -[LOCATED_IN_THE_ADMINISTRATIVE_
>    TERRITORIAL_ENTITY]-> Finnish`, `Finland`여야 함)를 놓쳤던 것도 같이 발견.
>    더 심각한 건 범위 — "Finnish"만 전수 스캔해도 지리 관계에 **185건**이 걸리고,
>    그중 다수가 `evidence_source=annotated`(**사람이 직접 라벨링한 gold 데이터**,
>    예: `Nirvi -[COUNTRY_OF_CITIZENSHIP]-> Finnish`)까지 포함됨 — 문제 11은
>    `evidence_source=inferred_bridge`만 봤는데, 이 패턴은 gold 데이터에도 있고
>    다른 demonym(Danish 등)도 더 있을 수 있어 실제 범위가 훨씬 큼. **아직 미수정**
>    — 아래 "문제 11" 섹션에 "알려진 한계 확장"으로 기록, 별도 확장 작업 필요.
>
> 12개 baseline 최종 재검증 12/12 정답, 최근 회귀 있었던 2문항(AirAsia Zest 합병,
> Outotec 지역) 모두 안정화(각각 5회 샘플링 4/5, 5/5).
>
> 이전 (2026-07-16, **문제 11 실 그래프에 적용 완료**):
> `GraphRAG/fix_demonym_linking.py` 작성 후 `--dry-run` → `--apply`로 실제 반영까지
> 완료(계획을 바꿔 사용자 직접 구현 대신 내가 구현). Candidate Detection 결과 1,466개
> 후보 중, Stage 3 evidence 재확인(LLM, 병렬 8-way + 429 재시도 백오프)에서
> **493개만 실제로 근거가 뒷받침**되고 **973개(66%)는 형태 문제가 아니라 애초에
> 근거 문장이 그 관계 자체를 뒷받침하지 않는 것**으로 판정 — 사전 추정(1,627개
> 전부 수정 가능)보다 훨씬 적은 범위만 안전하게 고칠 수 있었음. 493개 표본 15개
> 직접 검증(전부 정확, 예: `Dennis Wilson -[COUNTRY_OF_CITIZENSHIP]-> American`
> → `United States`) 후 `--apply`, 재실행에서 2개 추가 케이스 발견해 마저 반영,
> 최종 재검증 재배정 대상 0개로 안정화(총 495개 엣지 재배정). 노드 자체는 병합 안
> 하고 엣지만 재배정(설계대로) — "Chinese" 등 지명형용사 노드는 그대로 남아있고
> 다른 문맥(언어 관계 등)에서 계속 정상 사용됨. 남은 974개(원래 966개 "Peking
> University -[COUNTRY]-> Chinese"류를 포함해 evidence_source=inferred_bridge에
> 남아있는 지명형용사 tail/head)는 bridging 추론 자체의 grounding 품질 문제로,
> 하영님께 별도 공유 권장(아래 "문제 11" 섹션에 기록).
>
> 이전 (2026-07-15, **문제 9 폐기**): confidence 기반 검색 필터링 방향은
> 사용자 판단으로 개선 백로그에서 제외 — 아래 "문제점 카탈로그"의 9번 항목 참고
> (원 내용은 취소선으로 보존).
>
> 이전 (2026-07-15, **문제 4 후속 버그 수정 — relation_type 분류가
> 의미 인접 라벨끼리 혼동**): 문제 3 재검토 중 12개 baseline 재실행으로 발견 —
> "Roketsan은 어느 나라 회사야?"를 5회 반복 샘플링(`repeat=0..4`)했더니
> `COUNTRY`(정답) 2/5, `COUNTRY_OF_ORIGIN`(제품/작품 원산지용, 회사에는 안 맞음)
> 3/5로 갈림. 이번엔 `relation_lookup()`이 못 찾고 BFS로 폴백해 최종 답은
> 우연히 맞았지만, 다른 개체에서 `COUNTRY_OF_ORIGIN`에 우연히 매칭되는 엣지가
> 있었다면 폴백 없이 조용히 틀렸을 위험한 케이스. 원인은 96개 relation 라벨을
> 아무 설명 없이 나열만 해서 `COUNTRY`/`COUNTRY_OF_CITIZENSHIP`/`COUNTRY_OF_ORIGIN`
> 처럼 의미가 겹치는 라벨을 LLM이 구분 못 하는 것 — 문제 4의 v3 수정(1-hop/2-hop
> 대조 예시)과 같은 패턴으로 이 그룹만 명시적 대조 예시(예4)를 프롬프트에 추가해
> 수정(캐시 키 `v4`→`v5`). 수정 후 같은 질문 5회 재샘플링 전부 `COUNTRY`로
> 정확히 분류되는 것 확인, 12개 baseline 전체 재실행 회귀 0건. 다른 91개 라벨은
> 실측으로 확인된 혼동이 없어 예시 추가 안 함(프롬프트 비대화 방지, 문제 생기면
> 그때 추가하는 원칙 유지).
>
> 이전 (2026-07-15, **문제 4 후속 버그 수정 — property_scan() 결과에
> 비-조직 타입 섞임**): 데모 캡처 스크린샷을 사용자가 직접 눈으로 검토하다 "1988년에
> 설립된 조직이 뭐가 있어?" 답변에 `Republic of China on Taiwan`(대만 정치 체제 명칭,
> MISC)처럼 조직이 아닌 결과가 섞여 있는 것을 발견 — `property_scan()`이 `relation_type`
> +`value`만으로 매치해 `h.type`을 전혀 안 봐서, INCEPTION=1988인 개체 16개 중 실제
> 조직(ORG) 12개 외에 국가 체제 구분 명칭(`Republic of China on Taiwan`, MISC), 자연보호구역
> 지정연도(`Olympic Wilderness`, LOC), 앨범/열차 발매·도입연도(`Wilburys`/`Vauban`, MISC)
> 4개까지 값이 같다는 이유만으로 결과에 포함됐던 것으로 확인. `extract_entities()`가
> `entity_type`(ENTITY_TYPES 6종 중 하나, LLM이 질문에 명시된 결과 종류로 채움 — 예:
> "조직"→ORG)까지 함께 추출하도록 확장하고, `property_scan()`이 주어지면 `h.type`으로
> 걸러내도록 수정(캐시 키 `parse_query_v3`→`v4`, 프롬프트가 바뀌었으므로). 기존 12개
> 데모 질의 전체 재실행으로 회귀 0건 확인, "1988년 조직" 질문은 12개(ORG만)로 정확히
> 줄어듦.
>
> 이전 (2026-07-15, **문제 3(엔티티 링킹 단순 부분일치) 구현 완료**):
> `graphrag_query.py`의 `find_seed_entities()`에 Exact → Alias → Word Boundary →
> Fuzzy Match cascade를 실제로 구현(이전까지는 README 설계/docstring만 갱신되고
> 함수 본문은 옛 substring(`CONTAINS`) 방식 그대로였음). Fuzzy 단계 스코어러를
> 설계 초안의 `WRatio`+threshold 90에서 `fuzz.ratio`+threshold 80으로 실측 교정
> (WRatio가 부분 문자열 매치를 섞어 써서 Word Boundary로 없앤 오탐이 fuzzy 폴백을
> 거쳐 되살아나는 것 발견). 상세는 아래 "문제 3" 섹션 "구현 완료" 항목 참고. 문제
> 1/2/3 전부 실 그래프/코드에 적용 완료, 남은 설계-only 항목은 문제 9/10(confidence
> 필터링, gold 라벨 오귀속 검증)뿐.
>
> 이전 (2026-07-15, **문제 1(오버-머징)/문제 2(언더-머징) 실 그래프에 적용
> 완료 — 버그 2건 추가 발견+수정, 최종 재검증까지 통과**): `fix_under_merging.py`/
> `fix_over_merging.py` `--apply` 실행 완료. 상세 교훈은 위 문제 1/2 섹션의 "구현 중
> 실측 발견" 항목에 직접 반영해뒀음 — 여기는 무슨 일이 있었는지 순서대로 요약.
>
> **언더-머징 최종 10쌍 병합**: CBS(Topology Check 수동 오버라이드 — 계층 엣지 자체가
> "CBS Sports Network" 문서의 엔티티 링킹 오류로 판단), Outotec, Mozilla, Google Inc.,
> FBOP, General Motors, Sony Computer Entertainment, Apple, American Airlines Group,
> Google LLC. **적용 중 버그 2건 추가 발견**: (1) Google Inc. 오탐(EMPLOYER 값 차이) 고치려
> LLM 프롬프트에 경고를 추가했더니 General Motors가 merge→reject로 뒤집힘(LLM 판정이
> 프롬프트 변화에 취약함을 실측 재확인) — "관계 타입은 달라도 같은 대상(Michigan)을
> 공유하면 병합" 규칙을 추가해 LLM 대신 규칙으로 확정 판정하도록 전환. (2) 같은 이름이
> 여러 관계에 걸쳐 있어도 문제없음을 확인(아래 오버-머징 항목 참고). 최종 `--apply` 후
> Apple/Google/CBS Corporation 등 삭제된 노드와 alias 병합을 직접 조회로 재확인 완료.
>
> **오버-머징 35개 후보 처리 완료**: DATE_OF_BIRTH 25 + INCEPTION 4(Yale University는
> LLM이 "같은 개체"로 정상 판정해 제외) + PLACE_OF_BIRTH 6, 전부 처리. **적용 중 실제
> 크래시 발생 + 수정**: "Jones"가 `DATE_OF_BIRTH`(5명 분리)로 먼저 처리된 뒤, 독립적으로
> 처리되던 `PLACE_OF_BIRTH`의 "Jones" 후보가 새 노드를 만들려다 그룹 번호가 우연히 겹쳐
> (`Jones::PER#split1`) `ConstraintError`로 실제로 죽음. 확인해보니 `PLACE_OF_BIRTH`가
> 잡은 문서(Gaynelle Griffin Jones/Marcia Jones-Smoke)는 `DATE_OF_BIRTH` 단계에서 이미
> 서로 다른 노드로 정확히 분리돼 있었음(중복 작업이었을 뿐, Cole/Lee/Young/Brown/Hackett
> 도 전부 같은 패턴). 노드 id를 그룹 번호 대신 대표 문서명 기반으로 바꾸고 `CREATE` 대신
> `MERGE ... ON CREATE SET`으로 변경해 멱등하게 수정, 처음부터 재실행해서 크래시 없이
> 완주 확인(재실행 시 DATE_OF_BIRTH/PLACE_OF_BIRTH 후보가 이미 0개로 잡혀 중복 처리도
> 안 됨을 확인).
>
> **최종 검증**: 두 스크립트 모두 dry-run 재실행 결과 남은 후보 없음(언더-머징은 정당한
> reject 3건만, 오버-머징은 Yale University 1건만 정상 skip) — 안정된 최종 상태 확인.
> 총 엔티티/관계 수는 하영님이 병행 작업 중인 stage-2 적재와 겹쳐 있어 이 작업만의
> 정확한 델타 산출은 어려움(개별 병합/분리는 전부 직접 조회로 확인했으므로 문제 없음).
>
> **재적재 시 주의**: 이번 수정은 이미 적재된 그래프에 대한 직접 패치라 재적재 불필요.
> 단, `Scripts/kg/load_ground_truth.py`가 나중에 다시 돌아가면(로더 자체는 이 수정을
> 모르므로) 오버/언더-머징이 원상복구됨 — 언더-머징 재발 방지용 Stage 5(Preventive
> Normalization, `normalize_name()`에 접미사 제거 규칙 추가)는 아직 미구현.
>
> 이전 (**문제 4 Relation-aware Retrieval 구현 + 회귀 발견/수정
> — 하영님 DB 적재 완료로 실 그래프 대상 실측 시작**): 하영님의 ground-truth 데이터 적재가
> 끝나 이제부터 실 Neo4j 그래프로 검증 가능. `graphrag_query.py`의 `extract_entities()`를
> 확장해 entities/relation_type/value를 LLM 1회 호출로 동시 추출하고, `relation_lookup()`
> (1-hop 우선 조회) + `property_scan()`(엔티티 없이 값으로 역검색) + 기존 `expand_subgraph()`
> BFS 폴백으로 라우팅하는 문제 4 해결 방안을 실제로 구현. 기존 12개 데모 질의
> (`RAG/demo_compare.py`)로 전수 회귀 테스트하다 회귀 1건 발견(Outotec 필터 도시 지역
> 질문 — 표면상 1-hop처럼 보이지만 실제론 2-hop이라 1-hop 성공에서 멈춰 지역 정보를
> 놓침) → 프롬프트에 판정 예시 3개 추가해 수정 → 재검증 통과, 회귀 0건 + "1988년에 설립된
> 조직" 같은 순수 속성 스캔 질문이 새로 해결됨. 상세 과정은 아래 "실험 2" 참고. 문제 1/2/3
> (엔티티 오버/언더-머징, 엔티티 링킹)은 아직 설계만 있고 미구현 — 다음 작업 대상.
>
> 이전 (2026-07-15):
> GraphRAG 고도화 실험용 폴더 신설. `RAG/graphrag_query.py`를 그대로 복사해와서 실험 사본으로 삼고(`graphrag_query.py`), API 호출 캐싱 유틸(`cache.py`)을 추가함. 지식그래프 문제점 9가지를 진단하고 해결 방안을 정리, A/B 테스트 방법론과 baseline(3-hop) 실험 결과를 기록. 문제 1(오버-머징)의 해결 방안을 Attribute Conflict Detection → LLM Verification → Mention-level Graph Repair 3단계로, 문제 2(언더-머징)의 해결 방안을 Rule-based Detection → Graph Topology Check(독립 게이트) → Candidate Verification → Graph Merge → Preventive Normalization 5단계로 확정. Graph Topology Check는 Harvard/Harvard Corporation이 실제로는 PARENT_ORGANIZATION/SUBSIDIARY로 연결된 별개 법인임을 실측으로 확인한 데서 나온 설계로, Candidate Verification보다 먼저 실행해 비용 0으로 명백한 오탐을 걸러낸다. 문제 3(엔티티 링킹)의 해결 방안을 Exact → Alias → Word Boundary → Fuzzy Match Cascade(Priority-based Entity Linking, LLM 미사용)로 확정 — Word Boundary 단계가 Ankara/Mankara/Sankara 오탐을 실제로 없애는 것 실측 검증함. Embedding/BM25는 필요 사례 발견 시 추가하기로 보류. 부수 발견: `IBM::ORG`/`IBM::MISC`가 타입 분류 불일치로 별개 노드로 쪼개져 있음. 문제 4(속성 스캔 실패)의 해결 방안을 Relation-aware Retrieval(1-hop 우선 조회 + 실패 시 기존 BFS 폴백, entities/relation_type/value를 한 LLM 호출로 동시 추출)로 확정 — 1-hop 성공/BFS 결과 모두 LLM Answer로 합류(템플릿화는 96개 relation_name 한국어 큐레이션 비용 대비 이득이 적어 기각, 트래픽 늘면 재검토). Cirit처럼 실제 2-hop인 케이스는 1-hop 실패 후 BFS로 자연스럽게 폴백되는 것 실측 확인.

## 이 폴더의 목적

`RAG/graphrag_query.py`(데모용, 건드리지 않음)와 별도로, GraphRAG 고도화 실험을 여기서 진행한다. 문제점을 진단하고 해결책을 적용해본 뒤, 기존 12개 데모 질의로 A/B 테스트해서 실제로 개선되는지 확인하고 하나씩 반영한다.

## 역할 분담

- **하영님**: DB 쪽 담당, 진행 중.
  - `test_revised`에 대한 모델 예측 triple을 낮은 confidence로 Neo4j에 적재 (KG README의 "2단계").
  - `evidence_source = "unresolved_multihop"`인, 근거 문장이 비어있는 관계(29,661개, 전체의 28.8%)를 LLM으로 채우는 작업.
- **나(이 폴더 담당)**: 연구 설계 + 문제점 진단 + 해결 방안 설계. 하영님 데이터 작업이 끝나면 실제 데이터로 구현/실험.

## 현재 파이프라인 (베이스라인, `RAG/graphrag_query.py`와 동일)

```
질문 → ① extract_entities(): LLM이 seed가 될 고유명사 추출
     → ② find_seed_entities(): 이름/aliases 부분일치로 Neo4j 노드 탐색
     → ③ expand_subgraph(): 최대 3-hop BFS 확장
          (매 hop마다 연결 40개 초과하는 허브 노드는 다음 hop 확장에서 제외)
     → ④ 각 fact에 relation triple + Neo4j에 저장된 evidence 원문 첨부
     → ⑤ LLM이 fact 목록만 근거로 한국어 답변 (근거 부족하면 "모름")
```

데이터: `train_revised.json` + `dev_revised.json` (3,553 문서) → Neo4j (`ZEntity` 노드 42,456개, 관계 103,161개, 96종 relation type).

---

## 문제점 카탈로그

### 1. 엔티티 오버-머징 (동명이인이 한 노드로 합쳐짐)

**증상**: 엔티티 병합이 `(정규화된 이름, 타입)` 문자열 일치만으로 이뤄져서, 흔한 성/이름(예: "Jones", "Lee", "Joseph Meyer")을 가진 서로 다른 실존 인물이 한 노드로 뭉침. `DATE_OF_BIRTH`에 값이 4개 충돌하는 "Jones" 노드가 실제 사례 (`Davy Jones`, `Gaynelle Griffin Jones`, `Marcia Ingram Jones Smoke` 등 서로 무관한 인물의 별칭이 섞여있음).

**규모**: `DATE_OF_BIRTH` 보유 엔티티 1,294개 중 55개(4.3%), `INCEPTION` 보유 엔티티 1,016개 중 32개(3.1%)가 값 충돌.

**해결 방안 — 3단계 설계 (2026-07-15 확정)**:

```
Stage 1. Attribute Conflict Detection
  Functional relation(예: DATE_OF_BIRTH, PLACE_OF_BIRTH, INCEPTION)에서
  하나의 엔티티가 둘 이상의 서로 다른 값을 갖는 경우를 잠재적 오버-머지 후보로 탐지.
        ↓
Stage 2. LLM-based Entity Verification
  후보 엔티티의 alias, 이웃 관계, 속성, 문맥 정보를 LLM에 제공하여
  동일 개체 여부를 판정.
        ↓
Stage 3. Mention-level Graph Repair
  오버-머지로 판정된 경우 DocRED의 vertexSet을 이용해 mention을
  원래 엔티티로 재배정하고 그래프를 재구성.
```

**단계별 구현 시 주의점**:

- **Stage 1**: "Functional relation" 목록을 신중히 골라야 함. `DATE_OF_BIRTH`/`INCEPTION`/`PLACE_OF_BIRTH`/`DISSOLVED_ABOLISHED_OR_DEMOLISHED`는 진짜 1:1이라 안전. `COUNTRY_OF_CITIZENSHIP`(이중국적 가능), `SPOUSE`(재혼 가능)처럼 원래 여러 값이 정상인 관계를 섞으면 오탐이 늘어남 — 안전한 관계만 화이트리스트로 선별.
- **Stage 2**: LLM에 "충돌하는 값"뿐 아니라 **각 값이 나온 원본 evidence 문장**까지 반드시 같이 줘야 함 (엣지에 이미 `evidence` 속성으로 저장돼 있어 조회만 하면 됨). 문장 없이 "1933년생" vs "1948년생" 같은 값만 보면 판정이 어렵고, "Davy Jones는 몽키스의 멤버였다" vs "Gaynelle Griffin Jones는 텍사스 판사였다" 같은 원문을 봐야 확신 있게 판정 가능. 출력도 이진(같다/다르다) 판정이 아니라 **N개 그룹으로 클러스터링**하는 형태로 설계해야 함 — "Jones"처럼 충돌값이 4개면 실제로는 최대 4명(그 이상)일 수 있음.
- **Stage 3**: 각 관계 엣지가 이미 `document` 속성을 갖고 있어서(어느 문서에서 나온 관계인지), Stage 2의 판정 결과를 "문서 → 그룹" 매핑으로 정리하면 엣지 재배정은 그 매핑만 따라가면 됨. 관계를 삭제하는 게 아니라 올바른 주인(재분리된 새 엔티티)에게 재배정하는 방식.
- **알려진 한계**: Stage 1이 "감시 중인 functional relation에 충돌값이 있는 경우"만 탐지하므로, 그런 관계 자체가 없는 오버-머징(예: 생년월일 정보가 아예 없는 동명이인)은 이 파이프라인으로 못 잡음 — 완전한 해결이 아니라 "증거가 남아있는 케이스"를 잡는 실용적 1차 필터.
- **날짜 값의 표기 정밀도 차이를 충돌로 오인하지 말 것** (구현 중 실측 발견, 2026-07-15): "1991"/"December 1991"/"December 26, 1991"처럼 같은 사건을 다른 정밀도로 적은 값은 문자열이 다르다고 진짜 충돌이 아님 — Soviet Union의 `DISSOLVED_ABOLISHED_OR_DEMOLISHED`가 이 패턴으로 가짜 충돌 3건이 잡혔던 사례(Prussia/Nazis/Derg/South Vietnam도 동일 패턴). Stage 1에서 연도만 추출해 같은 연도면 충돌 아님으로 정규화해야 함.
- **Stage 3 구현 시 노드 id는 그룹 번호가 아니라 그룹의 대표 문서명 기반으로 지어야 함** (구현 중 실측 발견, 2026-07-15): 같은 이름("Jones" 등)이 `DATE_OF_BIRTH`/`PLACE_OF_BIRTH`처럼 서로 다른 functional relation 여러 개에서 각각 독립적으로 충돌 후보로 잡히는 경우가 흔함(사람 이름은 생년월일과 출생지가 같이 기록되는 경우가 많으므로). 각 relation을 독립적으로 처리하면서 새 노드 id를 "그룹 번호"(`#split1`, `#split2`...)로만 지으면, 두 relation의 클러스터링이 각자 "그룹 1"을 다른 사람에게 배정했을 때 id가 우연히 겹쳐 `ConstraintError`(id 유일성 제약 위반)로 실제로 죽음. 대표 문서명 기반 id(예: `Jones::PER#Burwell_Jones`)로 바꾸고, 노드 생성도 `CREATE` 대신 `MERGE ... ON CREATE SET`으로 해야 함 — 어차피 한 relation의 분리가 먼저 처리되면 나머지 relation은 이미 다 분리된 문서만 남아 사실상 재확인만 하고 끝나는 경우가 많으므로(실측: Jones/Cole/Lee/Young/Brown/Hackett 전부 `DATE_OF_BIRTH` 분리가 `PLACE_OF_BIRTH`의 충돌 문서를 이미 다 커버해서 두 번째 패스는 완전히 중복 작업이었음), MERGE로 멱등하게 만들면 처리 순서와 무관하게 안전.

### 2. 엔티티 언더-머징 (같은 개체가 표기 차이로 분리됨)

**증상**: 회사 법인 접미사(Inc./Corp./Oyj 등) 차이로 정규화가 안 돼서 같은 회사가 별개 노드로 쪼개짐.

**확인된 진짜 언더-머징 사례**: `Apple` ↔ `Apple Inc.`, `Google` ↔ `Google Inc.`, `Outotec` ↔ `Outotec Oyj`, `CBS` ↔ `CBS Corporation`, `General Motors` ↔ `General Motors Corporation` 등 — 접미사 패턴 하나만 스캔해도 10건 이상 발견됨. 다른 접미사(Ltd., LLC, GmbH, plc, AG 등)까지 스캔하면 더 있을 것.

**주의(오탐 사례)**: `Harvard` ↔ `Harvard Corporation`은 접미사 패턴상 후보로 잡히지만 **실제로는 서로 다른 법인**이다 — 그래프에 이미 `Harvard -[PARENT_ORGANIZATION]-> Harvard Corporation`, `Harvard Corporation -[SUBSIDIARY]-> Harvard`, `Harvard -[OWNED_BY]-> Harvard Corporation` 관계가 직접 존재한다 (Harvard University와 그 이사회 격인 Harvard Corporation은 실제로 별개 법인). 순수 접미사 매칭만으로 자동 병합하면 이런 진짜 사실을 지워버리게 되므로, 아래 검증 단계가 반드시 필요하다.

**해결 방안 — 4단계 설계 (2026-07-15 확정, Graph Topology Check를 검증 앞단의 독립 게이트로 분리)**:

```
                Entity Nodes
                     │
                     ▼
      Rule-based Candidate Detection
        (Suffix Dictionary Scan)
                     │
                     ▼
        Candidate Entity Pairs
                     │
                     ▼
        Graph Topology Check
   (Direct structural relation exists?)
                     │
         ┌───────────┴───────────┐
         │                       │
       Yes                     No
         │                       │
         ▼                       ▼
 Reject Candidate     Candidate Verification
                       ┌─────────────────────┐
                       │ Entity Type         │
                       │ Alias Similarity    │
                       │ Relation            │
                       │ Attribute           │
                       │ Context             │
                       │ LLM (optional)      │
                       └─────────────────────┘
                                │
                                ▼
                           Graph Merge
                                │
                                ▼
                    Canonical Node + Alias
                                │
                                ▼
                 normalize_name() Rule Update
```

1. **Rule-based Candidate Detection** (비용 0, Cypher): 법인 접미사 사전(Inc., Corp., Corporation, Ltd., LLC, GmbH, Oyj, plc, AG 등)으로 "A"와 "A + Suffix"가 둘 다 노드로 존재하는 쌍을 전수 탐지.
2. **Graph Topology Check** (비용 0, Cypher, Candidate Verification보다 먼저 실행): 후보 쌍 사이에 이미 직접 관계(엣지)가 존재하는지 체크. 특히 `PARENT_ORGANIZATION`/`SUBSIDIARY`/`OWNED_BY`/`PART_OF` 같은 계층 관계가 있으면 LLM/속성 비교까지 갈 것도 없이 즉시 기각(Reject Candidate). 실측 확인: `Harvard`/`Harvard Corporation`은 이 관계가 있어 자동 기각되고, `Apple`/`Apple Inc.`는 둘 사이에 직접 연결이 전혀 없어(확인함) 다음 단계로 정상 통과.
3. **Candidate Verification** (경량 검증, Topology Check를 통과한 후보만 대상): Entity Type / Alias Similarity / Relation / Attribute / Context를 비교. 대부분은 규칙 기반만으로 판정 가능하고, 애매한 소수만 LLM 사용.
4. **Graph Merge**: 동일 개체로 판정된 경우만 관계를 대표 노드(Canonical Node)로 재배정, alias 통합, 중복 노드 제거.
5. **Preventive Normalization**: 재발 방지로 `normalize_name()`에 법인 접미사 제거 규칙 추가.

**세부 주의점**:
- **Alias Similarity는 이 케이스에서 큰 힘이 안 됨**: `Apple` 노드의 alias는 보통 `["Apple"]`뿐이고 `Apple Inc.`도 `["Apple Inc."]`뿐이라 겹치는 게 없다. 실제 판단 근거는 alias 겹침이 아니라 관계/속성 일치(같은 CEO, 같은 본사 등)다.
- **관계/속성 커버리지가 희소할 수 있음**: 두 노드가 겹치는 relation 타입 자체가 거의 없는 경우(예: 한쪽은 HEADQUARTERS만, 다른 쪽은 CEO만 기록됨) 규칙 기반 비교로는 판정 불가 → 예상보다 많은 후보가 LLM으로 넘어갈 수 있음.
- **"충돌"은 1:1 관계에서만 신뢰할 것** (구현 중 실측 발견, 2026-07-15): 공유하는 관계 타입의 값이 다르다고 무조건 별개 법인으로 보면 안 됨 — `DEVELOPER`/`EMPLOYER`/`PUBLISHER`/`PRODUCT_OR_MATERIAL_PRODUCED`처럼 한 회사가 원래 여러 값을 갖는 게 정상인 관계는 값이 다른 게 당연함. 첫 구현이 이 구분 없이 Apple/Apple Inc.(DEVELOPER 값 차이)·Google/Google Inc.(EMPLOYER 값 차이)·Sony Computer Entertainment/…Inc.(PUBLISHER 값 차이)를 전부 오탐 기각한 걸 실측으로 확인 — `DATE_OF_BIRTH`/`INCEPTION`/`HEADQUARTERS_LOCATION`/`COUNTRY`/`CAPITAL` 등 진짜 1:1 관계에서만 충돌을 신뢰하도록 화이트리스트로 제한.
- **관계 타입이 달라도 같은 대상을 공유하면 긍정 신호로 볼 것** (구현 중 실측 발견, 2026-07-15): 같은 관계 타입끼리만 값을 비교하면, General Motors(`LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY`=Michigan)와 General Motors Corporation(`HEADQUARTERS_LOCATION`=Michigan)처럼 관계 타입은 다르지만 같은 대상(Michigan)을 가리키는 강한 신호를 놓치고 애매한 LLM 판정으로 넘어가게 됨. 이 케이스는 실제로 LLM 프롬프트를 한 번 수정(EMPLOYER 오탐 방지 경고 추가)했더니 merge→reject로 뒤집히는 걸 확인 — **LLM 판정은 프롬프트 변경에 따라 흔들릴 수 있으므로, 판정 가능한 신호는 최대한 규칙으로 옮겨 LLM 의존도를 낮출 것.** 이미 접미사 매칭을 통과한 후보(같은 기본 이름)에 한해서만 적용되므로 무관한 회사가 흔한 지명 하나 겹친다고 잘못 병합될 위험은 낮음.
- **Topology Check가 기각한 케이스도 그 엣지 자체가 노이즈일 수 있음** (구현 중 실측 발견, 2026-07-15): CBS/CBS Corporation이 계층 관계 존재로 자동 기각됐지만, 그 엣지들을 직접 까보니 전부 "CBS Sports Network" 문서 하나에서 나왔고 근거 문장("CBS Sports Network ... owned by the CBS Corporation")이 실제로는 CBS Sports Network(제3의 개체)를 가리키는데 CBS 자신에게 잘못 귀속된 것으로 보임(엔티티 링킹 오류 의심). Harvard/BAE Systems처럼 두 개체 자체의 관계를 직접 서술하는 진짜 계층 관계와 겉보기엔 구분이 안 되므로, Topology Check가 기각한 케이스 중 이미 다른 경로로 실제 동일 개체임이 확인된 경우(이번엔 팀 사전 조사)는 근거 문서를 직접 확인해 수동 오버라이드를 검토할 것 — 자동화하기엔 "엔티티 링킹 오류 vs 진짜 계층 관계"를 일반적으로 구분할 신뢰할 만한 규칙이 아직 없음.
- **5단계(Preventive Normalization)도 검증 없이 적용하면 안 됨**: 접미사만 지웠는데 이름이 같아진 서로 다른 두 회사가 있다면(우연의 일치), 이건 다시 1번(오버-머징) 문제를 재발시킨다. 재적재 시점에도 최소한 Graph Topology Check 정도는 통과시켜야 함.

### 3. 엔티티 링킹이 단순 부분일치 (substring match)

**증상**: `find_seed_entities()`가 대소문자 무시 부분일치라, "Ankara" 질의가 "Vinod Mankara", "Sankara"까지 오탐으로 잡음.

**해결 방안 — Cascade Matching, Priority-based Entity Linking (2026-07-15 확정)**:

```
                    User Query
                         │
                         ▼
               Normalize Query
                         │
                         ▼
                 Exact Match ?
                 ┌──────┴──────┐
                 │             │
               Yes             No
                 │             ▼
                 │       Alias Match ?
                 │      ┌─────┴─────┐
                 │      │           │
                 │     Yes          No
                 │      │           ▼
                 │      │   Word Boundary Match ?
                 │      │   ┌──────┴──────┐
                 │      │   │             │
                 │      │  Yes            No
                 │      │   │             ▼
                 │      │   │      Fuzzy Match
                 │      │   │             │
                 └──────┴───┴─────────────┘
                            │
                            ▼
                     Seed Entity Set
```

1. **Exact Match**: 정규화된 질의 멘션과 노드 `name`이 완전히 일치하면 즉시 채택하고 종료 (부분일치 시도 안 함, 정확도 최고).
2. **Alias Match**: `aliases` 리스트에 정확히 일치하는 표현이 있으면 채택 (예: "Zest Airways"로 물으면 `AirAsia Zest` 노드의 alias `Zest Airways, Inc.`로 연결).
3. **Word Boundary Match**: `\bankara\b` 류 단어 경계 정규식으로 교체. **실측 검증 완료** — 이 정규식으로 실제 Cypher 쿼리를 돌려보면 "Ankara" 노드 하나만 잡히고, 지금 쓰는 substring 방식은 "Vinod Mankara"/"Sankara"/"Kamran Bagheri Lankarani"까지 오탐으로 잡는 걸 재현 확인함.
4. **Fuzzy Match** (RapidFuzz): 오타 대응 (예: "Googel" → "Google"). 여기서만 사용.

**LLM은 이 단계에 안 씀**: entity linking은 호출 빈도가 매우 높을 수 있어(하루 수천 번), LLM을 쓰면 비용/속도 모두 불리함. 결정론적/통계적 방법(String + Alias + 필요시 Embedding/BM25)으로 처리하고, LLM은 문제 1/2처럼 진짜 판단이 필요한 소수 케이스에만 예약.

**Embedding/BM25는 보류 (2026-07-15 결정)**: `extract_entities()`가 이미 LLM으로 질문에서 고유명사를 뽑아주는 구조라 "그 튀르키예 미사일 회사" 같은 설명형 멘션이 들어올 일이 적어서, 당장은 Exact→Alias→WordBoundary→Fuzzy 4단계로 충분하다고 판단. 필요한 실제 사례가 발견되면 그때 5번째 단계로 추가.

**검증 중 발견한 부수 이슈**: `IBM`이 `IBM::ORG`와 `IBM::MISC`로 별개 노드로 존재함 — 같은 이름인데 문서마다 DocRED 타입 분류가 갈려서 전역 병합 키 `(이름, 타입)` 기준으로 쪼개진 것. 언더-머징(2번)과 결이 비슷하지만 원인이 다름(표기 차이가 아니라 타입 분류 불일치) — 별도 이슈로 추후 검토.

**구현 완료 (2026-07-15)**: `GraphRAG/graphrag_query.py`의 `find_seed_entities()`에 위 4단계 cascade를 실제로 구현. Word Boundary 단계는 Cypher `=~`가 (Java `Pattern.matches`처럼) 전체 문자열 매치라 `(?i).*\b<이스케이프된 멘션>\b.*` 형태로 앞뒤에 `.*`를 둬야 "포함" 의미가 됨 — 실제 그래프에 쿼리해서 "Ankara"는 잡히고 "Vinod Mankara"/"Sankara"는 안 잡히는 것 재확인.

- **Fuzzy Match 단계 스코어러/threshold, 구현 중 실측 발견**: 설계 초안의 "threshold 예: 0.9"(RapidFuzz 0~100 스케일로 90)를 그대로 쓰면, 이 설계가 직접 든 예시인 "Googel" → "Google"조차 `fuzz.WRatio`로 83.3점밖에 안 나와 통과를 못 한다. 게다가 `WRatio`는 두 문자열 길이 차이가 크면 내부적으로 partial-ratio(부분 문자열 매치)를 섞어 쓰는데, 이 때문에 "Ankara" vs "Vinod Mankara"에 81.8점을 줘버려 — Word Boundary 단계에서 방금 없앤 바로 그 부분일치 오탐이 fuzzy 폴백을 거쳐 되살아나는 것을 실측으로 확인했다(Word Boundary가 실패해서 fuzzy까지 내려간 멘션에 대해 `WRatio`+threshold 90으로 직접 재현). 스코어러를 전체 문자열 기준 순수 Levenshtein 비율인 `fuzz.ratio`로 바꾸고 threshold를 80으로 내려서, "Googel"/"Google"(83.3)·"Micorsoft"/"Microsoft"(88.9)는 통과시키면서 "Ankara"/"Sankara"(76.9)·"Ankara"/"Vinod Mankara"(52.6)는 여전히 걸러내는 조합을 실측 검증 후 확정.
- **Fuzzy 후보 풀 캐싱**: 매 fuzzy 폴백 호출마다 전체 노드(42,456개)를 다시 스캔하면 비용이 크므로, 프로세스당 1회만 `MATCH (e:ZEntity) RETURN id, name, type`으로 가져와 메모리에 캐싱(`_fuzzy_name_pool`). fuzzy는 이름만 대상으로 하고 alias는 포함하지 않음(README의 "Googel"→"Google" 예시처럼 정식 명칭 오타 대응이 주 목적이고, alias까지 넣으면 후보 풀이 커지고 동일 엔티티가 여러 후보로 중복 등장해 로직이 복잡해짐 — 필요한 실제 사례가 나오면 그때 확장).
- **회귀 검증**: `RAG/demo_compare.py`의 12개 baseline 질의를 `GraphRAG/graphrag_query.py`의 `answer_question()`으로 전수 재실행 — 기존 기대 답변과 전부 일치(AirAsia Zest 본사 질문만 "Pasay City"로 나와 기대값 "Ninoy Aquino International Airport, Manila"과 표현이 다르지만, seed 엔티티가 Exact Match로 이전과 동일하게 잡혀 이 변경과 무관한 기존 데이터/응답 특이사항으로 확인 — 이번 회귀 대상 아님).

### 4. 순수 속성 스캔형 질문이 아예 실패

**증상**: "1988년에 설립된 조직이 뭐가 있어?" 같은 질문은 탐색을 시작할 고유명사 엔티티가 없어서, `extract_entities()`가 빈 리스트를 반환 → seed 0개 → 무조건 "모름".

**애초 설계의 한계**: "엔티티가 없으면 속성 스캔"이라는 조건만으로는 부족하다 — "Apple는 언제 설립됐어?"처럼 **엔티티도 있고 속성(관계) 타입도 동시에 있는 Hybrid Query**를 놓친다. 이런 질문은 지금도 blind BFS로 어떻게든 풀리긴 하지만(48~600여 개 사실을 다 긁어서 LLM이 그중 INCEPTION 값을 골라내는 식), 훨씬 비싸고 노이즈에 취약하다.

**해결 방안 — Relation-aware Retrieval, 1-hop 우선 조회 + BFS 폴백 (2026-07-15 확정)**:

```
                User Query
                     │
                     ▼
        LLM Query Parsing (1회)
 ┌────────────────────────────────────┐
 │ entities / relation_type / value   │
 └────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
  Entity 없음                Entity 존재
        │                         │
        ▼                         ▼
  Property Scan          relation_type 존재?
        │                 ┌────────┴────────┐
        │                Yes                No
        │                 │                 │
        │                 ▼                 │
        │        1-hop Relation Lookup      │
        │           ┌───────┴───────┐       │
        │        Success        Not Found   │
        │           │                 │     │
        │           │                 └─────┤
        │           │                       ▼
        │           │              Existing Blind BFS
        │           │                       │
        └───────────┴───────────┬───────────┘
                                 ▼
                            LLM Answer
```

**핵심 설계 결정**:

1. **엔티티 추출과 관계/값 추출을 하나의 LLM 호출로 합침** — `extract_entities()`를 확장해 `{"entities": [...], "relation_type": "INCEPTION" 또는 null, "value": "1988" 또는 null}`을 한 번에 반환. API 호출 수는 지금과 동일하게 유지.
2. **Entity 없음 → Property Scan**: seed 없이 `MATCH (o)-[:INCEPTION]->(d {name:"1988"}) RETURN o` 같은 전역 속성 스캔.
3. **Entity 있음 + relation_type 있음 → 1-hop Relation Lookup 우선 시도**: seed에서 해당 relation 타입만 딱 1-hop으로 직접 조회 (예: `Apple -[:INCEPTION]-> ?`). 성공하면 그 사실 하나만으로 충분.
4. **1-hop이 실패하거나(Not Found) relation_type이 애초에 null이면 → 기존 Blind BFS로 폴백**: "Cirit 제조사의 본사 도시"처럼 실제로 2-hop 이상 필요한 질문(Cirit은 `HEADQUARTERS_LOCATION`을 직접 안 가짐, Roketsan을 거쳐야 함 — 실측 확인)이나 "Roketsan에 대해 알려줘"처럼 relation_type을 특정할 수 없는 열린 질문은 여기로 자연스럽게 빠짐.
5. **1-hop 성공과 BFS 결과 모두 같은 `LLM Answer` 단계로 합류** — 처음엔 1-hop 성공 시 템플릿으로 LLM 없이 바로 답을 만드는 안을 검토했으나 기각함. 이유: (a) 96개 relation_name을 전부 자연스러운 한국어로 사전 큐레이션해야 하고 은/는·이/가 같은 조사 처리(받침 유무)까지 신경 써야 해서 생각보다 손이 많이 감, (b) 1-hop 성공 시엔 사실이 1개뿐이라 LLM 답변 생성 호출 자체가 원래 아주 싸고 빠름 — 지금 프로젝트 규모에서 템플릿화로 아끼는 비용이 크지 않음. 트래픽이 실제로 대량(하루 수천 콜)으로 늘어나면 그때 템플릿화 재검토.

**기대 효과**: "Apple는 언제 설립됐어?" 같은 단순 1-hop 질문은 노이즈 없이 정확하고 싸게 답하고(현재는 48~600여 개 사실을 다 긁어야 함), 진짜 멀티홉/열린 질문은 기존 BFS 그대로 유지되어 회귀 없음.

### 5. relation 스키마 밖의 사실 누락

**증상**: DocRED 고정 96종 relation에 없는 사실(예: "합병/제휴로 형성됨")은 evidence 문장에 텍스트로는 있어도 그래프에 구조화된 엣지가 아예 없음.

**해결 방안**: (a) 받아들이고 넘어가거나, (b) 그래프 검색이 실패하면 evidence 텍스트에 대한 보조 텍스트 검색(naive RAG와 유사)으로 폴백하는 하이브리드 구조, (c) 장기적으로 스키마 확장(재작업 규모 큼).

### 6. Evidence 없는 엣지 28.8% (unresolved_multihop)

**증상**: 103,161개 엣지 중 29,661개(28.8%)는 근거 문장이 없어서, 구조화된 사실은 맞아도 "왜?"를 설명 못 함.

**진행 상황**: **하영님이 LLM으로 채우는 작업 진행 중.** 데이터 도착하면 품질 검증(할루시네이션 여부) 필요 — 내가 할 일.

### 7. Hop 수 & 노이즈 증가 트레이드오프

**증상**: MAX_HOPS를 3→5로 늘리면 작고 닫힌 클러스터(Roketsan, Cirit)는 사실 개수 그대로인데, 국가/대륙급 노드에 인접한 엔티티(University of Manitoba, Health Sciences Centre, Philippines)는 거의 선형으로 계속 불어남(예: 224→624, 136→474, 575→975).

**실험 결과**: 3-hop → 5-hop 비교, 10개 질의 중 회귀 0건. 근거는 아래 실험 결과 표 참고 — 단, "AirAsia Zest 합병 이력 1건 개선" 결론은 **2026-07-16 정정됨(1회성 샘플의 우연, 실제로는 여전히 불안정 — 아래 표의 정정 메모 및 "문제 8" 섹션 참고)**.

**해결 방안**: 단기적으로 4~5-hop 정도가 안전한 절충점. 장기적으로는 방향 없는 BFS 대신, 질문 임베딩과의 유사도로 다음 hop 후보를 순위 매겨 관련 없는 방향은 아예 안 펼치는 가이드된 순회로 교체.

**적용 완료 (2026-07-16)**: `MAX_HOPS = 5`로 코드에 실제 반영(그동안 이 실험 결과만 기록돼 있고 `graphrag_query.py`는 3으로 남아있던 불일치를 사용자가 지적해 발견). 문제 1/2/10/11 그래프 수정 이후 상태로 12개 baseline 재검증 — 회귀 없음(과정에서 별도로 발견한 문제 8 관련 회귀 2건은 아래 "문제 8" 섹션 참고, MAX_HOPS 자체와는 무관).

**노이즈 감소 2차 적용 (2026-07-16)**: "장기적 해결"로 적어둔 "질문 임베딩 유사도로 순위 매기기"를 실제 구현 — 다만 BFS 확장 방향 자체를 유도하는 형태(가이드된 순회)가 아니라, `expand_subgraph()`가 다 모은 뒤 `rerank_facts()`로 사후 필터링하는 더 단순한 형태로 구현(구현 난이도 대비 효과는 비슷할 것으로 판단, 트래버설 자체를 유도하는 버전은 여전히 설계만 있고 미구현). `SUBGRAPH_FACT_LIMIT_PER_HOP` 200→60 축소 + 중복 사실 제거까지 포함해 자세한 내용은 위 "최종 업데이트" 항목과 "문제 8" 섹션 참고.

### 8. 답변 생성 비결정성

**증상**: 같은 subgraph를 줘도 답이 흔들리는 경우 확인 ("Manila 항공사 문서" 질문이 재실행마다 정답/오답 왔다갔다).

**해결 방안**: `temperature`를 낮게 고정(또는 0), 가능하면 `seed` 파라미터 사용. 그래도 완전히 결정적이진 않을 수 있어 A/B 테스트 방법론에서 반복 샘플링으로 보완 (아래 참고).

**부분 적용 완료 (2026-07-16)**: `extract_entities()`/`answer_with_subgraph()` 둘 다 `temperature=0` 추가(`seed`도 테스트했지만 `system_fingerprint`가 안정적으로 안 나와 결정성 보장 안 됨을 확인, 참고용으로만 남김). **실측으로 확인된 중요한 부작용**: temperature=0은 분산을 줄이지만 정확도를 보장하지 않음 — "Outotec의 필터가 만들어지는 도시는 어느 지역에 속해있어?" 질문이 temperature=0 적용 직후 오히려 4/4 샘플 전부 잘못된 relation_type으로 고정 오분류되는 것을 실측(이전엔 비결정적이라 가끔은 맞았음). "가끔 틀림"이 "항상 틀림"으로 바뀔 수 있다는 뜻이라, temperature 고정은 단독 해결책이 아니라 프롬프트 정확도 자체를 계속 개선하는 것과 병행해야 함(위 문제 4/7 changelog에 있는 예5 추가로 이 특정 케이스는 수정, 5/5 재검증).

**반복 샘플링+다수결 자동화 완료 (2026-07-16)**: `majority_vote_answer()` 추가 — `answer_with_subgraph()`를 3회(README "A/B 테스트 방법론"에 이미 3회로 설계돼 있던 값) 호출해 다수결로 최종 답을 정함. **비용 절감**: `bfs`/`1hop_fallback_bfs` 라우팅에서만 적용(1hop/property_scan은 원래도 사실 1~수개짜리라 흔들림이 실측된 적 없어서 3배로 돌릴 이유가 없음), 3개 답이 전부 똑같으면(실제로 대부분 이 경우) 그룹화 호출조차 생략.

**그룹화 방식 관련 실측 버그+수정**: 처음엔 `rerank_facts()`처럼 임베딩 코사인 유사도로 답끼리 묶으려 했는데, "...2013년에 그렇게 됐습니다"와 "...2016년 1월에 그렇게 됐습니다"(다른 사실!)의 유사도가 **0.9727**인 반면 진짜 같은 사실을 다른 문구로 쓴 두 답의 유사도는 **0.9894**로 격차가 0.017밖에 안 나 안전한 임계값을 잡을 수 없었음(문장 틀이 90% 겹치고 핵심 사실인 날짜 한 군데만 다르면 임베딩이 둔감함 — `rerank_facts()`에서 겪은 것과 같은 함정). `fix_over_merging.py`의 `cluster_conflict()`와 같은 패턴(LLM에게 N개 항목을 보여주고 그룹 번호 배열로 클러스터링 요청)으로 교체해 해결.

**AirAsia Zest 케이스에 대한 정직한 한계**: 다수결 로직 자체는 정상 작동(2013년/2016년/모름을 정확히 별개 그룹으로 분리 확인)하지만, 이 질문은 6회 샘플링에서 정답 2/모름 2/오답 2로 **진짜 3분의 1씩 고르게 갈리는 케이스**라 n=3 다수결로는 종종 동점이 나 확실히 못 고침(3표 중 우연히 같은 그룹이 2표 안 나오면 동점 처리 로직이 임의로 하나를 고름). 다른 BFS 질문(Cirit/Roketsan/Outotec/Manila/Philippines 등)은 전부 3표 만장일치로 안정적 — 이 케이스처럼 원래도 거의 균등하게 흔들리는 예외적인 질문에만 한계가 남음(근본 원인은 여전히 문제 5의 스키마 갭). n을 더 늘리면(5, 7...) 개선될 수 있으나 비용이 비례해서 커짐 — 필요시 고려.

**진짜 원인의 일부가 검색 쪽(Cypher 비결정성)이었다는 것도 실측 발견 (2026-07-16)**: `expand_subgraph()`의 Cypher 쿼리가 `LIMIT`만 있고 `ORDER BY`가 없어서, 결과 집합이 LIMIT을 넘을 때 **어떤 행이 잘리는지 자체가 매 실행마다 달라질 수 있었음** — 즉 "같은 질문을 다시 물었더니 사실이 조금 다르게 나왔다"는 것 자체가 답변 생성 이전에 이미 발생하고 있었을 가능성. `ORDER BY head, relation, tail`을 추가해 같은 조건이면 항상 같은 사실 집합이 나오도록 고정. **그런데 이렇게 완전히 고정한 뒤에도 "AirAsia Zest 어느 항공사가 합쳐졌는지" 질문은 여전히 흔들림**(동일 사실 집합으로 6회 샘플링: 정답 2 / 모름 2 / 오답 2("2016년 1월" — 이건 합병 시점이 아니라 나중에 AirAsia Philippines로 흡수된 시점) — 이걸로 이 질문의 불안정성이 순수하게 **LLM 답변 생성 자체의 비결정성**(검색/노이즈/순서 문제가 아님)이라는 게 확정됨. `relation` 스키마에 "합병" 자체가 없어(문제 5) 근거 문장에서만 추론 가능한 사실이라는 게 근본 원인으로 보이고, 반복 샘플링+다수결 자동화가 유일한 실질적 완화책으로 남음.

### 9. ~~confidence를 검색에 반영 안 함~~ — 폐기 (2026-07-15)

> **폐기**: 사용자 판단으로 개선 백로그에서 제외. 아래는 원래 내용(참고용).

~~**증상**: 2단계(모델 예측 triple, confidence < 1.0)가 들어오면, 지금 파이프라인은 confidence를 전혀 안 보고 모든 엣지를 동등하게 취급함.~~

~~**해결 방안 (설계만, 하영님 데이터 도착 후 결정)**: (a) 임계값 미만 엣지는 검색에서 제외, (b) LLM 프롬프트에 "이 사실은 confidence 0.6으로 예측된 것"이라고 명시해서 답변에 반영하게 하기, (c) 둘 다 병행.~~

### 10. Gold 라벨 자체의 관계-값 오귀속 (annotation noise, 2026-07-15 발견)

**증상**: `evidence_source=annotated`, `confidence=1.0`인 사람이 직접 라벨링한 gold 데이터에도 relation-value가 잘못 귀속된 케이스가 있음 — 문제 9(2단계 모델 예측 confidence)와 달리 confidence 필터로는 못 거름. "1988년에 설립된 조직이 뭐가 있어?" `property_scan` 결과 16개 중 실측 확인:

- **GPT** — 오탐 아님. 문서 `GEC Plessey Telecommunications`의 약칭이 "GPT"이고 evidence("GEC Plessey Telecommunications (GPT) was founded in 1988 as a joint venture...")가 실제로 1988년 설립을 말함. 다들 아는 그 GPT(OpenAI)가 아니라서 헷갈렸을 뿐, 관계는 정확함.
- **Republic of China on Taiwan** — 실제 오류. 문서 `Four-Stage Theory of the Republic of China`, evidence가 `"(Chinese: 中華民國在臺灣) (during Lee Teng-hui's presidency) (1988–2000)"` — 조직 설립년도가 아니라 **총통 재임 기간**을 설명하는 문장인데 `INCEPTION=1988`로 태깅되어 있음. `is_revised=True`인 revised 데이터셋에도 안 걸러지고 남아있는 gold annotation 오류.

**검토한 필터링 방식**:
- 임베딩으로 근접 값 배정/유사도 스코어링: 기각 후보. 이건 원래 "관계가 사실인지"를 판단하는 entailment 문제인데, "총통 재임 기간(1988–2000)" 같은 문장은 "설립년도" 문장과 주제(조직+연도)가 가까워 코사인 거리로는 잘 안 갈릴 가능성이 높음 — 임베딩은 topical 유사도지 논리적 함의 판단이 아님.
- 소규모 LLM 판별 호출: 권장 후보. head/relation/tail/evidence를 주고 "이 근거가 이 관계·값을 실제로 뒷받침하는가"를 yes/no로 판단 — entailment는 임베딩보다 LLM이 안정적.

**해결 방안 (설계만, 미구현)**: `property_scan()` 결과를 `answer_with_subgraph()`에 넘기기 전에 배치 1회 LLM 호출로 검증하는 단계를 라우팅에 추가 — `property_scan` → **Fact Verification(신규)** → `LLM Answer`. `property_scan`에 우선 적용을 권장하는 이유는, `relation_lookup`/BFS는 최소한 seed 엔티티 이름 매칭으로 head가 맞다는 보증이 있는데 `property_scan`은 seed 없이 그래프 전체 103,161개 엣지를 값만 보고 역매칭하므로 무관한 문서의 오귀속 사실이 섞여 들어올 구조적 위험이 가장 큼. 라우팅된 결과 집합(보통 수십 개 이하)만 배치 검증하므로 비용은 낮지만, DB 자체의 오류는 안 고쳐지고 서빙 단계에서만 걸러짐(같은 오류 엣지가 다른 질문에서 또 나올 수 있음) — 더 근본적으로 고치려면 103k 엣지 전체를 오프라인으로 한 번 감사해 `verified` 플래그를 영구 기록하는 별도 작업이 필요(비용/시간 크지만 모든 라우팅 경로에 재사용 가능).

**상태 재확인 (2026-07-16)**: 아직 미구현 — `Republic of China on Taiwan -[INCEPTION]-> 1988` 엣지를 직접 DB에서 재조회해 여전히 원래대로(`confidence=1.0`, evidence도 "총통 재임 기간" 원문 그대로) 있음을 확인. 문제 10의 **type-filtering 절반**(entity_type으로 ORG만 걸러내는 것, `property_scan()`에 이미 적용됨)과 **Fact-Verification 절반**(이 섹션, evidence가 관계 자체를 뒷받침하는지 재확인하는 것)은 서로 다른 절반이라 혼동 주의 — type-filtering은 이 엣지를 "조직" 질문 결과에서만 안 보이게 할 뿐, 엣지 자체의 오류는 그대로 남아있어 다른 질문 경로로는 여전히 잘못된 값이 나올 수 있음. 또한 `graphrag_query.py` 전체에 confidence를 조회/노출하는 코드가 전혀 없음(grep으로 확인) — 문제 9가 confidence 기반 검색을 폐기하기로 결정한 것과 일관됨, 사용자 쿼리에 대해 문서/사실별 confidence를 보여주는 기능은 없음.

### 11. `inferred_bridge` 엣지의 지명형용사(demonym) 오귀속 (2026-07-15 발견, 2026-07-16 적용 완료)

**증상**: 하영님이 bridging sentence 추론으로 채운 `evidence_source=inferred_bridge` 엣지(10,692개, 원문에 근거가 없어 값 자체는 정확하지만 `unresolved_multihop`이었던 것들) 중 상당수가, 원문의 "American engineer", "Chinese historian", "German conglomerate"처럼 **사람/조직을 수식하는 지명형용사**를 그 나라/지역 엔티티 자체로 잘못 링크했음. 예:

```
Peking University -[COUNTRY]-> Chinese                    (China여야 함)
Germany -[CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY]-> Bavarian   (Bavaria여야 함)
Hong Kong -[CONTINENT]-> Asian                             (Asia여야 함)
The Bronx -[LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY]-> American  (United States여야 함)
```

**규모**: 25개 무작위 표본 중 11개(44%)에서 이 패턴 확인. 지명형용사 목록(최소 29개 기준)으로 전수 스캔하면 `inferred_bridge` 10,692개 중 **1,627개(15.2%)** 가 head 또는 tail에 이 패턴을 가짐 — 목록을 늘리면 더 나올 가능성 높음. 그래프에는 형용사형("Chinese", "German" 등)과 정규형("China", "Germany" 등)이 **이미 둘 다 별개 노드로 존재**하는 것을 확인함(둘 다 `:LOC` 타입인 경우가 많음 — 형용사형이 `:LOC`로 잘못 타입 태깅된 것도 이 문제의 일부).

**관계 타입별 분포** (지리/국적 정체성을 묻는 관계에 집중됨):

| relation_type | 건수 |
|---|---|
| `LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY` | 572 |
| `COUNTRY` | 563 |
| `COUNTRY_OF_CITIZENSHIP` | 142 |
| `CONTAINS_ADMINISTRATIVE_TERRITORIAL_ENTITY` | 92 |
| `COUNTRY_OF_ORIGIN` | 39 |
| `APPLIES_TO_JURISDICTION` | 31 |
| `CONTINENT` | 17 |
| `HAS_PART` / `PART_OF` (h 또는 t가 `LOC` 타입일 때만) | 30 |
| `TERRITORY_CLAIMED_BY` | 4 |
| `BASIN_COUNTRY` | 3 |

**중요 — 언어/민족 관계는 제외할 것**: `OFFICIAL_LANGUAGE`(39)/`LANGUAGES_SPOKEN_WRITTEN_OR_SIGNED`(32)/`ORIGINAL_LANGUAGE_OF_WORK`(1)/`ETHNIC_GROUP`(13)도 지명형용사 목록에 걸리지만, 이건 **오탐**임 — "Russia -[OFFICIAL_LANGUAGE]-> Russian"은 정확한 사실(언어 이름은 영어에서 원래 지명형용사와 같은 형태). 위 표에 있는 지리/국적 정체성 관계만 화이트리스트로 삼고, 언어/민족 관계는 손대지 말 것. `HEAD_OF_GOVERNMENT`/`HEAD_OF_STATE`/`MEMBER_OF`/`PARTICIPANT`/`PARTICIPANT_OF`/`CONFLICT`/`LOCATION`/`RELIGION` 등 나머지(각 1~8건)는 우연히 이름이 지명형용사와 겹친 것일 가능성이 높아 블랭킷 처리 대상에서 제외, 수동 검토 대상.

**해결 방안 — 4단계 설계 (2026-07-15 확정, `GraphRAG/fix_demonym_linking.py`로 구현 + 2026-07-16 실 그래프에 `--apply` 완료)**:

```
Stage 1. Candidate Detection
  evidence_source='inferred_bridge' AND relation_type IN 화이트리스트(위 표) 인
  (h)-[r]->(t) 중 t.name(HAS_PART/PART_OF는 h.name도) 이 demonym_map에 있는 엣지 탐지.
        ↓
Stage 2. Canonical Target Resolution
  demonym_map[잘못된 이름]으로 같은 type(LOC)을 가진 노드가 이미 그래프에 있는지 확인.
  없으면 스킵(새 노드를 함부로 만들지 않음 -- 화이트리스트에 검증된 것만 안전).
        ↓
Stage 3. Evidence Re-check (기계적 치환과 분리)
  단순 이름 치환 전에, evidence 문장이 애초에 이 관계 자체를 뒷받침하는지 재확인.
  형태(tail 표기)만 문제인 게 아니라 애초에 근거 없는 케이스도 섞여있음(예: "Kern
  -[COUNTRY_OF_CITIZENSHIP]-> Austrian" -- evidence는 Kern이 아니라 Faymann 얘기).
  이런 건 이름만 바꾸면 "형태는 맞지만 내용은 틀린" 상태가 되므로 반드시 분리:
    - 근거가 실제로 이 관계를 뒷받침 -> Stage 4로 (기계적 치환, 안전)
    - 근거가 애초에 관계를 뒷받침 안 함 -> repoint 대상에서 제외, 문제 10의
      Fact Verification 쪽으로 넘기거나 confidence를 낮춰서 별도 처리
        ↓
Stage 4. Edge Repoint (노드 병합 아님, 엣지 하나만 재배정)
  (h)-[r]->(잘못된 노드) 삭제하고 (h)-[r {동일 속성}]->(정규형 노드) 재생성.
```

**핵심 주의점**:
- **노드 전체를 병합(`DETACH DELETE`)하면 절대 안 됨** — `fix_under_merging.py`의 `merge_entities()`와 다른 패턴. "Chinese" 노드는 다른 문맥(예: 언어 이름)에서 정당하게 쓰이고 있을 수 있어서, 이 엣지 하나만 재배정하고 노드 자체와 다른 엣지는 그대로 둔다.
- **relation_type 화이트리스트를 반드시 지킬 것** — 언어/민족 관계(위에서 제외한 것들)까지 같은 로직으로 훑으면 "Russia -[OFFICIAL_LANGUAGE]-> Russian" 같은 정상 사실을 "Russia" 쪽으로 잘못 바꿔버리는 새 오류를 만든다.
- **Stage 3을 생략하지 말 것** — 폐기된 문제 9(confidence 필터링)와 같은 함정: 이 엣지들도 전부 `confidence=1.0`이라 confidence로는 형태 문제와 근거 없음 문제를 구분 못 한다. 반드시 evidence 텍스트를 직접 봐야 함.
- **demonym_map 예시** (아래 30쌍은 그래프에 정규형 노드가 이미 존재함을 실측 확인, 필요시 확장):

  ```
  American→United States, Chinese→China, German→Germany, Austrian→Austria,
  Norwegian→Norway, Bavarian→Bavaria, Asian→Asia, Canadian→Canada,
  Icelandic→Iceland, Taiwanese→Taiwan, British→United Kingdom, French→France,
  Russian→Russia, Japanese→Japan, Indian→India, Australian→Australia,
  Italian→Italy, Spanish→Spain, Mexican→Mexico, Brazilian→Brazil,
  Egyptian→Egypt, Turkish→Turkey, Israeli→Israel, Swedish→Sweden,
  Dutch→Netherlands, Polish→Poland, Greek→Greece, Irish→Ireland
  ```
  `Korean`은 North Korea/South Korea 중 어느 쪽인지 문맥별로 갈릴 수 있어 자동 매핑에서 제외(evidence 확인 후 수동 판단), `Danish`(→Denmark)처럼 위 표에 없지만 실측 샘플에서 발견된 것도 있으니 실제 적용 전 스캔으로 목록을 늘려서 확인할 것.

**적용 결과 (2026-07-16)**: `GraphRAG/fix_demonym_linking.py`를 `fix_over_merging.py`/`fix_under_merging.py`와 같은 구조(`--dry-run` 기본, `--apply`로 실제 반영, `graphrag_query.py`의 `driver`/`RELATION_TYPES` 재사용)로 구현. Stage 3(evidence 재확인)은 후보마다 독립 LLM 호출이라 스레드풀(8-way)로 병렬화 — 1466개를 한 번에 검증 요청하다 500 RPM 한도로 `RateLimitError` 실제 발생(600/1466 지점에서 크래시), 지수 백오프 재시도 추가로 해결.

**실측 결과 — 사전 추정보다 훨씬 적은 범위만 안전**: Stage 1+2를 통과한 후보 **1,466개**(위 표의 1,627개 추정치보다 적은 건 LOC 타입 게이트를 추가로 걸었기 때문) 중, Stage 3에서 **493개(33.6%)만 evidence가 실제로 뒷받침**하고 **973개(66.4%)는 형태 문제가 아니라 애초에 근거 문장이 그 관계 자체를 뒷받침 안 함**으로 판정됨 — 예상보다 훨씬 큰 비율. 무근거로 skip된 예: `British -[PART_OF]-> Allied`(근거는 Ministry of Home Security 얘기, British-Allied 관계 자체는 안 나옴), `Taiwan -[TERRITORY_CLAIMED_BY]-> Chinese`(근거가 그 영유권 주장을 직접 뒷받침 안 함). 493개 중 15개 무작위 표본을 직접 검증(전부 정확 — 예: `Dennis Wilson -[COUNTRY_OF_CITIZENSHIP]-> American` → `United States`, `Arnold Schwarzenegger -[COUNTRY_OF_CITIZENSHIP]-> American` → `United States`, `Victoria -[LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY]-> Australian` → `Australia`) 후 `--apply` 실행. 재검증 스캔에서 2개 추가 케이스(`Asian -[HAS_PART]-> China`→`Asia`, `Chinese -[PART_OF]-> Asia`→`China`) 발견해 마저 반영, 최종 재실행에서 재배정 대상 0개로 안정화. **총 495개 엣지 재배정 완료**, 노드 자체(예: `Chinese` 노드)는 병합하지 않고 그대로 남겨 다른 문맥(언어 관계 등)에서 계속 정상 사용됨.

**남은 974개(무근거 판정) — 별도 이슈로 기록, 미해결**: 이건 "형태가 틀렸다"가 아니라 "bridging 추론이 애초에 grounding이 약한 엣지를 너무 많이 만든다"는, 문제 11보다 범위가 넓은 별개 데이터 품질 이슈로 보임. 지명형용사가 섞이지 않은 `inferred_bridge` 엣지에도 같은 grounding 문제가 있을 가능성이 있어(이번 스캔은 지명형용사가 낀 것만 봄), 하영님의 bridging 추론 파이프라인 쪽에 공유해서 검토를 권장. 문제 10의 Fact Verification 설계(아직 미구현)가 결국 이 974개를 포함해 `inferred_bridge` 전체를 커버해야 할 수도 있음.

(2026-07-16 갱신: 아래 "알려진 한계 확장" 섹션에서 `evidence_source` 필터를 없애고 전체 그래프로 재스캔한 결과, 무근거 총계는 974개가 아니라 **5,578개**로 늘어남 — 지명형용사가 낀 것만 봐도 이 정도이므로, 지명형용사 없는 `inferred_bridge`/기타 엣지까지 포함하면 문제 10의 Fact Verification이 커버해야 할 실제 범위는 이보다 더 클 것으로 추정.)

**알려진 한계 확장 — 발견(2026-07-16) 및 같은 날 적용 완료**: 문제 7/8 작업 중 우연히 "Outotec -[LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY]-> Finnish"(Finland여야 함) 엣지를 발견 — "Finnish"가 `DEMONYM_MAP`(28쌍)에 없어서 1차 적용에서 완전히 놓친 케이스. 확인해보니 규모가 훨씬 컸음("Finnish" 하나만 지리 관계 화이트리스트로 전수 스캔해도 185건, 그중 다수가 `evidence_source=annotated` 즉 사람이 직접 라벨링한 gold 데이터에도 있었음 — 1차 적용은 `evidence_source='inferred_bridge'`만 스캔했던 게 원인).

**2차 적용 완료**: `DEMONYM_MAP`을 28개→약 100개로 확장하고 `evidence_source` 필터를 제거해 전체 그래프(111,925개 엣지)로 스캔 범위 확장. 후보 **10,489개**(annotated 4,098 포함) → Stage 3(evidence 재확인, gold 데이터도 예외 없이 검증 — 문제 10에서 이미 확인한 "gold 라벨도 무조건 신뢰 금지" 교훈 재적용) → **재배정 대상 4,911개**(annotated 2,774 / inferred_cooccurrence 1,567 / model_provided 223 / inferred_bridge 219 / inferred_mention_union 83 / unresolved_multihop 45), 무근거 제외 5,578개. gold 25개 표본 직접 검증 후 `--apply` → **4,910개 엣지 실제 반영**, 재검증 재배정 대상 0개로 안정화. 1차(495개)+2차(4,910개) 합산 **총 5,405개 엣지 정정**.

**비용 관리 교훈(실측)**: 처음엔 개별 LLM 호출로 이 확장판을 돌리다 사용자가 OpenAI 대시보드에서 토큰 사용량 급증을 직접 확인하고 지적 — 예산이 빠듯한 개인 API 키(월 $5, 당시 크레딧 잔액 $2.08)에서 만 개 단위 개별 호출은 실제 비용 문제. 프로세스 중단 후 `fix_over_merging.py`의 `cluster_conflict()`와 같은 배치 패턴(25개씩 묶어 JSON 배열 응답, `GraphRAG/fix_demonym_linking.py`의 `verify_grounding_batch()`/`verify_grounding_all()`)으로 재작성 — 개별 호출 대비 요청 수 25배 감소, 사용자에게 예상 토큰(~542K 입력/~19K 출력) 공유 후 확인받고 실행. 이 교훈은 `CLAUDE.md`에 "LLM 호출 비용 최적화" 섹션으로 프로젝트 전역 가이드라인으로 등록.

---

## A/B 테스트 방법론

1. **Baseline 고정**: 지금 3-hop 버전의 12개 질의 결과(정답/오답)를 "before"로 고정 — 데모 리포트에 이미 기록됨.
2. **반복 샘플링**: 애매한 케이스(정답이 명확히 갈리지 않는 질문)는 3회 반복 실행 후 다수결로 판정 — `answer_question(question, repeat=0/1/2)`로 호출.
3. **캐싱**: `cache.py` — `(질문, repeat 인덱스)` 조합으로 캐싱하므로, 같은 실험을 나중에 다시 돌릴 때는 API 재호출 없이 즉시 결과 반환. 단, repeat 인덱스가 다르면 무조건 새로 호출(독립 샘플 보장).
4. **비용 규모**: 질문 1개당 LLM 호출 2회(엔티티 추출 + 답변 생성). 12개 질문 × 3회 반복 × 2버전(before/after) ≈ 144회 호출 — 무시할 수준.

## 실험 결과

### 실험 1: MAX_HOPS 3 → 5 (2026-07-15)

| 질문 | 3-hop | 5-hop | 비고 |
|---|---|---|---|
| Roketsan 국가 | Turkey ✅ | Turkey ✅ | 사실 48개, 변화 없음 |
| University of Manitoba 설립년도 | 1877 ✅ | 1877 ✅ | 사실 224→624개, 답 그대로 |
| Lappeenranta 국가 | Finland ✅ | Finland ✅ | 변화 없음 |
| AirAsia Zest 본사 | Pasay City ✅ | Pasay City ✅ | 변화 없음 |
| Cirit 제조사 본사 도시 | Ankara ✅ | Ankara ✅ | 변화 없음 |
| Roketsan 설립년도+위원회 | 정답 ✅ | 정답 ✅ | 변화 없음 |
| ~~**AirAsia Zest 합병 이력**~~ | ~~**모름 ❌**~~ | ~~**부분 정답 ⚠️→✅**~~ | ~~Zest Air+AirAsia 제휴, 2016년 1월 AirAsia Philippines로 합병까지 언급~~ **(2026-07-16 정정: 아래 참고, 이 row는 신뢰 불가)** |
| Health Sciences Centre→대학 설립년도 | 1877 ✅ | 1877 ✅ | 사실 136→474개, 답 그대로 |
| Outotec 필터 도시 지역 | South Karelia ✅ | South Karelia ✅ | 변화 없음 |
| Ankara 소재 회사 | Roketsan ✅ | Roketsan ✅ | 사실 159→559개, 답 그대로 |

**결론**: 회귀 0건, 개선 1건(당시 판단, 아래 정정 참고). 4~5-hop은 안전한 선택지로 보임. (Manila 문서 질문, 1988년 조직 질문은 hop 수와 무관한 실패 원인이라 이 실험에서 제외.)

**정정 (2026-07-16)**: 위 "AirAsia Zest 합병 이력" row와 아래 "실험 2" 표의 같은 항목 모두 **1회성 단일 샘플 비교**였음 — "모름❌ → 정답✅"이 hop 수 증가로 실제로 고쳐진 게 아니라, 우연히 그 1번의 5-hop 샘플이 정답을 맞힌 것뿐이었음. 같은 질문을 사실 집합이 완전히 동일하도록 고정한 뒤 6회 반복 샘플링했더니 **정답 2 / 모름 2 / 오답 2**로 확인됨(순수 LLM 답변 생성 비결정성, 문제 8 — 상세는 위 "문제 8" 섹션 참고). 이 표의 해당 row는 **역사적 기록으로만 남기고 결론으로 인용하지 말 것** — hop 수/노이즈와 무관하게 원래부터 불안정했던 질문이었음.

### 실험 2: Relation-aware Retrieval 구현 (문제 4, 2026-07-15)

`extract_entities()`를 확장(1-hop 우선 조회 + property scan + 기존 BFS 폴백)한 뒤 기존
12개 데모 질의(`RAG/demo_compare.py`의 `QUERIES`, 실험 1의 baseline과 동일 세트) 전체로
회귀 테스트. 구현 중 실측으로 회귀 1건을 발견 → 프롬프트 수정 → 재검증까지 진행:

| 질문 | 이전(BFS만) | v1 구현 직후 | 최종(v3) | 라우팅(최종) |
|---|---|---|---|---|
| Roketsan 국가 | Turkey ✅ | Turkey ✅ | Turkey ✅ | 1hop |
| University of Manitoba 설립년도 | 1877 ✅ | 1877 ✅ | 1877 ✅ | 1hop |
| Lappeenranta 국가 | Finland ✅ | Finland ✅ | Finland ✅ | 1hop |
| AirAsia Zest 본사 | Pasay City ✅ | Pasay City ✅ | Pasay City ✅ | 1hop |
| Cirit 제조사 본사 도시 | Ankara ✅ | Ankara ✅(폴백) | Ankara ✅ | bfs |
| Roketsan 설립년도+위원회 | 정답 ✅ | 정답 ✅(폴백) | 정답 ✅ | bfs / 1hop_fallback_bfs |
| AirAsia Zest 합병 이력 | 정답(5-hop 기준) | 정답 유지 | 정답 유지 *(2026-07-16 정정: 이것도 1회성 샘플 — 실제로는 불안정, 위 "실험 1" 정정 메모 참고)* | bfs |
| Health Sciences Centre→대학 설립년도 | 1877 ✅ | 1877 ✅(폴백) | 1877 ✅ | bfs |
| **Outotec 필터 도시 지역** | **South Karelia ✅** | **Lappeenranta ❌ (회귀!)** | **South Karelia ✅** | bfs |
| Manila 항공사 문서 | (참고 답변) | 동일 | 동일 | bfs |
| Philippines 요약 | (참고 답변) | 동일 | 동일 | bfs |
| **1988년 설립 조직** | **모름(엔티티 없어 실패)** | **16개 조직 나열 ✅(신규 성공)** | **16개 조직 나열 ⚠️(실제론 4개가 비-조직, v4에서 12개로 교정 — 아래 "후속 수정" 참고)** | property_scan |
| Ankara 소재 회사 | Roketsan ✅ | Roketsan ✅ | Roketsan ✅ | property_scan(최종)/1hop(v1) |

**회귀 원인과 수정 과정**: "Outotec의 필터가 만들어지는 도시는 어느 지역에 속해있어?"는
표면적으로 "Outotec의 어떤 속성" 질문처럼 보이지만 실제로는 (Outotec 헤드쿼터 도시) →
(그 도시가 속한 지역) 2-hop 질문 — v1 프롬프트가 이를 1-hop(HEADQUARTERS_LOCATION)으로
오판정해 1-hop 조회 성공(사실 1개, "Lappeenranta")에서 멈춰버리고 그 다음 hop(지역)을
못 감. **pure BFS로 강제 재실행해서 비교한 결과 BFS는 처음부터 정답(South Karelia)을
맞혔던 것으로 확인** — 즉 이건 1-hop 우선 조회 도입으로 인한 신규 회귀였지 원래 있던
문제가 아님. `extract_entities()` 프롬프트에 "entities 자신의 속성인지 vs entities와
연결된 다른 개체의 속성인지"를 구분하는 예시 3개(1-hop 직접 질문/개체 경유 2-hop/
entities 없이 값으로 역검색)를 추가해 재검증 → 12개 질의 전부 회귀 없이 통과.
**교훈**: relation_type을 채울지 null로 둘지는 결국 LLM의 프롬프트 해석에 의존하므로,
문구를 바꿀 때마다 이 12개 세트로 재검증이 필요함 — 프롬프트를 건드릴 때는 캐시 키
버전(`parse_query_v1`→`v2`→`v3`)도 같이 올려야 예전 프롬프트로 만든 캐시가 새 코드에
잘못 반환되지 않음(v2에서 한 번 이 문제로 실제로 결과가 안 바뀌는 걸 겪음).

**부가 성과**: "1988년에 설립된 조직이 뭐가 있어?"처럼 엔티티가 아예 없어 기존엔 무조건
"모름"이었던 질문이 이제 `property_scan`으로 16개 조직(Roketsan 포함, inception=1988
확인됨)을 정확히 찾아냄 — README "문제 4" 증상 항목이 실제로 해결됨.

**후속 수정(2026-07-15, 같은 날 추가 발견)**: 위 "16개 조직"이 사실 전부 조직은 아니었음
— `h.type`을 안 봐서 `Republic of China on Taiwan`(MISC)/`Olympic Wilderness`(LOC)/
`Wilburys`/`Vauban`(둘 다 MISC) 4개가 INCEPTION=1988 값만 같다는 이유로 섞여 있었음.
사용자가 데모 스크린샷을 직접 검토하다 발견. `extract_entities()`에 `entity_type` 필드를
추가해 `property_scan()`이 `h.type`으로 걸러내도록 수정, 실제 조직 12개만 남도록 교정
(상세는 위 맨 위 changelog 항목 참고).

<!-- 다음 실험 결과를 여기에 표로 추가 -->
