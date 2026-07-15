# GraphRAG 고도화

> **최종 업데이트**: 2026-07-15:
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
4. **Fuzzy Match** (Levenshtein/Jaro-Winkler/RapidFuzz, threshold 예: 0.9): 오타 대응 (예: "Googel" → "Google"). 여기서만 사용.

**LLM은 이 단계에 안 씀**: entity linking은 호출 빈도가 매우 높을 수 있어(하루 수천 번), LLM을 쓰면 비용/속도 모두 불리함. 결정론적/통계적 방법(String + Alias + 필요시 Embedding/BM25)으로 처리하고, LLM은 문제 1/2처럼 진짜 판단이 필요한 소수 케이스에만 예약.

**Embedding/BM25는 보류 (2026-07-15 결정)**: `extract_entities()`가 이미 LLM으로 질문에서 고유명사를 뽑아주는 구조라 "그 튀르키예 미사일 회사" 같은 설명형 멘션이 들어올 일이 적어서, 당장은 Exact→Alias→WordBoundary→Fuzzy 4단계로 충분하다고 판단. 필요한 실제 사례가 발견되면 그때 5번째 단계로 추가.

**검증 중 발견한 부수 이슈**: `IBM`이 `IBM::ORG`와 `IBM::MISC`로 별개 노드로 존재함 — 같은 이름인데 문서마다 DocRED 타입 분류가 갈려서 전역 병합 키 `(이름, 타입)` 기준으로 쪼개진 것. 언더-머징(2번)과 결이 비슷하지만 원인이 다름(표기 차이가 아니라 타입 분류 불일치) — 별도 이슈로 추후 검토.

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

**실험 결과**: 3-hop → 5-hop 비교, 10개 질의 중 회귀 0건, "AirAsia Zest 합병 이력" 1건 개선(모름 → 부분 정답). 근거는 아래 실험 결과 표 참고.

**해결 방안**: 단기적으로 4~5-hop 정도가 안전한 절충점. 장기적으로는 방향 없는 BFS 대신, 질문 임베딩과의 유사도로 다음 hop 후보를 순위 매겨 관련 없는 방향은 아예 안 펼치는 가이드된 순회로 교체.

### 8. 답변 생성 비결정성

**증상**: 같은 subgraph를 줘도 답이 흔들리는 경우 확인 ("Manila 항공사 문서" 질문이 재실행마다 정답/오답 왔다갔다).

**해결 방안**: `temperature`를 낮게 고정(또는 0), 가능하면 `seed` 파라미터 사용. 그래도 완전히 결정적이진 않을 수 있어 A/B 테스트 방법론에서 반복 샘플링으로 보완 (아래 참고).

### 9. (하영님 작업 완료 후 다룰 질문) confidence를 검색에 반영 안 함

**증상**: 2단계(모델 예측 triple, confidence < 1.0)가 들어오면, 지금 파이프라인은 confidence를 전혀 안 보고 모든 엣지를 동등하게 취급함.

**해결 방안 (설계만, 하영님 데이터 도착 후 결정)**: (a) 임계값 미만 엣지는 검색에서 제외, (b) LLM 프롬프트에 "이 사실은 confidence 0.6으로 예측된 것"이라고 명시해서 답변에 반영하게 하기, (c) 둘 다 병행.

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
| **AirAsia Zest 합병 이력** | **모름 ❌** | **부분 정답 ⚠️→✅** | Zest Air+AirAsia 제휴, 2016년 1월 AirAsia Philippines로 합병까지 언급 |
| Health Sciences Centre→대학 설립년도 | 1877 ✅ | 1877 ✅ | 사실 136→474개, 답 그대로 |
| Outotec 필터 도시 지역 | South Karelia ✅ | South Karelia ✅ | 변화 없음 |
| Ankara 소재 회사 | Roketsan ✅ | Roketsan ✅ | 사실 159→559개, 답 그대로 |

**결론**: 회귀 0건, 개선 1건. 4~5-hop은 안전한 선택지로 보임. (Manila 문서 질문, 1988년 조직 질문은 hop 수와 무관한 실패 원인이라 이 실험에서 제외.)

### 실험 2: (TBD)

<!-- 다음 실험 결과를 여기에 표로 추가 -->
