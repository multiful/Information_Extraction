# Knowledge Graph 적재 (`Scripts/kg/`)

> **최종 업데이트**: 2026-07-15: `upsert_pinecone.py`가 대상 Pinecone 인덱스(`informationrag`)의 기존 차원(현재 512)을 `describe_index`로 조회해서 그 차원으로 임베딩하도록 수정 — 인덱스가 이미 다른 차원으로 만들어져 있으면 기본값(1536)으로 임베딩할 때 `Vector dimension ... does not match` 에러가 났던 문제 수정.
>
> 2026-07-15: 1단계 대상을 원본 DocRED(`train_annotated`+`dev`)에서 Re-DocRED 재정제본(`train_revised`+`dev_revised`)으로 완전 교체 — 같은 3,053개 문서에 대한 상위호환 라벨(오라벨 정리 + 누락 관계 보강)이라, 원본과 병행 적재하면 revised가 지운 오라벨이 그래프에 남는 문제가 있어 교체를 택함. 모든 엣지에 `is_revised` 속성 추가(`split`이 `_revised`로 끝나면 True — 향후 2단계 모델 예측 triple은 자연히 False). `export_csv.py`/`export_pinecone.py`/`export_postgres.py`도 `is_revised` 컬럼/필드 반영.
>
> 2026-07-14: 관계 엣지를 문서 간 병합하지 않도록 재설계 — 같은 (head, tail, relation) triple이 여러 문서에서 나오면 엣지를 여러 개 만들고, 각 엣지가 `confidence`/`document`/`sentence_id`/`evidence`/`evidence_source`를 온전한 속성으로 가짐 (LangGraph 등에서 관계별 근거를 바로 꺼내 쓸 수 있도록). 공용 로직을 `docred_common.py`로 분리하고, Pinecone(`export_pinecone.py`)·PostgreSQL/Supabase(`export_postgres.py`) export 스크립트 추가.

## 1단계: 확실한 정보 적재 (Ground Truth)

- **대상**: `train_revised` (3,053개 문서), `dev_revised` (500개 문서) — Re-DocRED가 사람이 재검증한 재정제본. 원본 `train_annotated`/`dev`는 이걸로 완전히 대체됨(원본 dev 998개 문서 중 500개만 `dev_revised`, 나머지 498개는 `test_revised`로 분리되어 이번 적재에는 포함하지 않음). `train_distant`도 제외.
- **방식**: 문서의 `labels`(head/tail/relation) triple을 그대로 confidence `1.0`으로 Neo4j에 적재.
- **개체 병합 정책**: DocRED는 문서 간 entity linking 정보를 제공하지 않으므로, `(정규화된 이름, type)`이 같은 개체는 문서 경계를 넘어 하나의 노드로 전역 병합함 (예: 여러 문서에 등장하는 "Greece" → 노드 1개). 동명이인처럼 이름은 같지만 실제로는 다른 개체인 경우 잘못 병합될 수 있는 것이 알려진 한계.
- **관계는 반대로 문서 간 병합하지 않음**: 같은 (head, tail, relation) triple이 여러 문서에서 나오면 문서 수만큼 별도 엣지가 생김 (같은 노드 쌍 사이에 병렬 엣지가 여러 개 있을 수 있음). 각 엣지는 자신이 나온 문서의 confidence/evidence를 그대로 속성으로 가짐 — LangGraph 등에서 "이 관계의 근거는?"을 물었을 때 문서별로 바로 꺼내 쓸 수 있게 하기 위함 (엣지 하나로 합쳐서 문서 목록만 누적하던 이전 방식은 폐기).

### 공용 모듈 (`docred_common.py`)

`export_triples.py`, `load_ground_truth.py`, `export_pinecone.py`, `export_postgres.py`가 공유하는 DocRED 로딩/정규화/evidence 보완 로직:

- `cluster_canonical(cluster)`: 멘션 클러스터의 대표 이름/타입 (최빈값).
- `resolve_evidence(label, mention_sents_h, mention_sents_t, sents)`: evidence 보완. DocRED 원본 라벨 중 evidence가 비어있는 경우(train_annotated 1,421개/38,180개, dev 487개/12,275개), head/tail이 같은 문장에 함께 언급되면 그 문장을 추론해서 채우고(`inferred_cooccurrence`, 총 515개), 함께 언급되는 문장이 아예 없으면(multi-hop 추론 필요, 총 1,393개) 억지로 채우지 않고 `unresolved_multihop`로 표시.

### 그래프 스키마

- 노드: `(:ZEntity:<TYPE> {id, name, type, aliases})` — `id = "{정규화된 이름}::{type}"`, `aliases`는 클러스터 내 모든 mention 표기. `<TYPE>`은 DocRED type을 그대로 보조 라벨로 쓴 것(`PER`/`ORG`/`LOC`/`TIME`/`NUM`/`MISC`, 6종) — `type` 속성과 중복되지만, Bloom/Browser가 라벨 기준으로 노드를 분류·색칠하기 때문에 필요. 공통 라벨 이름이 `Entity`가 아니라 `ZEntity`인 이유: Neo4j Browser/Bloom이 다중 라벨 노드의 스타일을 알파벳순으로 정렬된 첫 라벨 기준으로 고르는 것으로 보여, 타입 라벨(`PER`~`TIME`)이 항상 먼저 오도록 의도적으로 `Z`로 시작하는 이름을 씀.
- 엣지: `(:ZEntity)-[:<RELATION_TYPE> {relation_id, relation_name, confidence, split, document, sentence_id, evidence, evidence_source, is_revised}]->(:ZEntity)` — `<RELATION_TYPE>`은 `relation_name`을 슬러그화(UPPER_SNAKE_CASE)한 동적 관계 타입(예: `country` → `:COUNTRY`, 96개 존재). `document`가 MERGE 매칭 키라 같은 triple이 여러 문서에서 나오면 문서별로 별도 엣지가 생김. `confidence`는 이 단계에서 항상 `1.0`. `is_revised`는 `split`이 `_revised`로 끝나면 True(`docred_common.is_revised_split`) — 지금은 SPLITS 전부가 revised라 항상 True.
- Neo4j Browser에서 노드 이름을 보려면 결과 화면 하단 범례의 `ZEntity`/타입 항목 → Caption을 `name`으로 지정 (엣지는 타입 자체가 관계 이름이라 별도 설정 불필요). Bloom에서는 Perspective 편집 화면에서 각 타입 카테고리의 Caption을 `name`으로 지정.

### 실행

```
# 환경변수는 저장소 루트 .env에서 로드 (git에는 커밋되지 않음, .gitignore 처리됨)
python Scripts/kg/load_ground_truth.py --dry-run   # DB 연결 없이 집계 수치만 확인
python Scripts/kg/load_ground_truth.py              # 실제 Neo4j Aura 적재 (재실행해도 idempotent)
```

필요 환경변수 (`.env`): `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.

### 적재 결과 (2026-07-15 실제 적재 기준, train_revised+dev_revised)

| 항목 | 개수 |
|---|---|
| 고유 개체 노드 | 42,456 (PER 9,846 / LOC 9,566 / MISC 9,151 / ORG 7,357 / TIME 4,770 / NUM 1,766) |
| 관계 엣지 (문서별 비병합, Neo4j 실제 저장 기준) | 103,161 (96개 관계 타입) — 원본 라벨 수는 103,216인데, 같은 문서 내에서 (head_id, tail_id, relation_type)이 겹치는 MERGE 충돌로 55개가 합쳐짐 |

원본(`train_annotated`+`dev`, 2026-07-14 실행) 대비 엣지가 50,286 → 103,161로 약 2배 늘어남 — Re-DocRED가 원본 DocRED의 누락된 관계(false negative)를 대거 보강한 결과.

**주의**: `load_ground_truth.py`는 관계 엣지만 지우고(`MATCH ()-[r]->() DELETE r`) 개체 노드는 지우지 않으므로, split을 바꿔 재적재하면 이전 split에만 있던 개체가 관계 없는 고아 노드로 남는다. 이번엔 원본 `dev`(998개 문서)에만 있고 `dev_revised`(500개 문서)엔 없는 문서에서 온 고아 노드 5,428개를 수동으로 확인 후 삭제함 (관계가 하나도 없는 것만 골라 안전하게 삭제 — 관계가 있는 노드는 절대 건드리지 않음).

## 다른 스토어로 내보내기

Neo4j 그래프를 그대로 다른 시스템에 옮겨 쓸 수 있도록 내보내는 스크립트들. 전부 Neo4j를 소스로 읽어서 생성하므로, `load_ground_truth.py`를 다시 돌린 뒤 재실행하면 최신 상태로 갱신됨.

| 스크립트 | 출력 | 용도 |
|---|---|---|
| `export_csv.py` | `triples.csv` | Excel/pandas로 바로 열어보는 평면 CSV (head/relation/tail/confidence/document/sentence_id/evidence/evidence_source) |
| `export_triples.py` | `triples.jsonl`, `unresolved_multihop.jsonl` | 원본 DocRED에서 직접 뽑은 문서 단위 raw triple (엔티티 id가 문서 내 vertexSet 인덱스, 전역 병합 없음) — 요청받은 head/relation/tail/source/evidence JSON 스키마 |
| `export_pinecone.py` | `pinecone_upsert.jsonl` | Pinecone upsert용 `{id, text, metadata}`. `text`는 evidence 문장을 이어붙인 것(없으면 `"{head} {relation} {tail}"`로 대체). `metadata`는 Pinecone 제약(문자열/숫자/불리언/문자열리스트, 중첩 객체 불가)에 맞춰 평평하게 구성. |
| `upsert_pinecone.py` | Pinecone 인덱스(`informationrag`) | `pinecone_upsert.jsonl`을 읽어 OpenAI `text-embedding-3-small`로 `text`를 임베딩하고 Pinecone에 업서트. 인덱스가 없으면 서버리스로 자동 생성, 있으면 기존 차원에 맞춰 임베딩. |
| `export_postgres.py` | `schema.sql`, `entities.csv`, `relations.csv` | PostgreSQL(Supabase)용. `entities`/`relations` 2개 테이블, `relations.head_id`/`tail_id`가 `entities.id`를 참조하는 FK. `aliases`/`sentence_id`/`evidence`는 JSONB. Supabase SQL Editor에서 `schema.sql` 실행 후 Table Editor로 CSV 임포트(entities 먼저), 또는 `psql`의 `\copy`. |

## 다음 단계 (미구현)

- 2단계: 모델이 예측한 (미검증) triple을 낮은 confidence로 적재 — 향후 작업.
- Neo4j Bloom 시각화 씬 구성 (`README.md`의 5.2절 참고).
- LangGraph 쪽에서 confidence를 가드레일로 쓰는 로직 (예: 임계값 미만이면 답변에서 제외/재확인 요청) — 이 저장소 범위 밖, LangGraph 프로젝트에서 구현.
