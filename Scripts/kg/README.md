# Knowledge Graph 적재 (`Scripts/kg/`)

> **최종 업데이트**: 2026-07-14: `export_triples.py`에 `evidence_source` 필드 추가 — DocRED 원본 라벨 중 evidence가 비어있던 것들을 head/tail 공동 언급 문장으로 추론해서 채우고(`inferred_cooccurrence`), 근거 문장이 아예 없는 multi-hop 케이스는 억지로 채우지 않고 표시만 함(`unresolved_multihop`, 별도 파일 `unresolved_multihop.jsonl`로도 분리). Neo4j에는 반영되지 않음(아래 "evidence와 Neo4j" 참고). (이전 업데이트: 공통 개체 라벨을 `:Entity`→`:ZEntity`로 개명, `:Triple` 오적재 정리, `export_csv.py` 추가, 개체 타입 보조 라벨, 관계 동적 타입 마이그레이션.)

## 1단계: 확실한 정보 적재 (Ground Truth)

- **대상**: `train_annotated` (3,053개 문서), `dev` (998개 문서) — 사람이 직접 annotate한 라벨만 사용, `train_distant`는 제외.
- **방식**: 문서의 `labels`(head/tail/relation) triple을 그대로 confidence `1.0`으로 Neo4j에 적재.
- **개체 병합 정책**: DocRED는 문서 간 entity linking 정보를 제공하지 않으므로, `(정규화된 이름, type)`이 같은 개체는 문서 경계를 넘어 하나의 노드로 전역 병합함 (예: 여러 문서에 등장하는 "Greece" → 노드 1개). 동명이인처럼 이름은 같지만 실제로는 다른 개체인 경우 잘못 병합될 수 있는 것이 알려진 한계.
- **관계 병합 정책**: 동일한 `(head, tail, relation_id)` triple이 여러 문서에서 반복되면 엣지 하나로 병합하고, 출처 문서 목록을 `sources` 속성(`"{split}::{title}"` 문자열 배열)에 누적.

### 그래프 스키마

- 노드: `(:ZEntity:<TYPE> {id, name, type, aliases})` — `id = "{정규화된 이름}::{type}"`, `aliases`는 클러스터 내 모든 mention 표기. `<TYPE>`은 DocRED type을 그대로 보조 라벨로 쓴 것(`PER`/`ORG`/`LOC`/`TIME`/`NUM`/`MISC`, 6종) — `type` 속성과 중복되지만, Bloom/Browser가 라벨 기준으로 노드를 분류·색칠하기 때문에 필요. 공통 라벨 이름이 `Entity`가 아니라 `ZEntity`인 이유: Neo4j Browser/Bloom이 다중 라벨 노드의 스타일을 알파벳순으로 정렬된 첫 라벨 기준으로 고르는 것으로 보여, 타입 라벨(`PER`~`TIME`)이 항상 먼저 오도록 의도적으로 `Z`로 시작하는 이름을 씀.
- 엣지: `(:ZEntity)-[:<RELATION_TYPE> {relation_id, relation_name, confidence, sources}]->(:ZEntity)` — `<RELATION_TYPE>`은 `relation_name`을 슬러그화(UPPER_SNAKE_CASE)한 동적 관계 타입(예: `country` → `:COUNTRY`, 96개 존재). `confidence`는 이 단계에서 항상 `1.0`.
- Neo4j Browser에서 노드 이름을 보려면 결과 화면 하단 범례의 `ZEntity`/타입 항목 → Caption을 `name`으로 지정 (엣지는 타입 자체가 관계 이름이라 별도 설정 불필요). Bloom에서는 Perspective 편집 화면에서 각 타입 카테고리의 Caption을 `name`으로 지정.
- `Scripts/kg/export_csv.py`: Neo4j에 적재된 그래프를 그대로 저장소 루트 `triples.csv` 한 파일로 내보냄 (head/relation/tail/confidence/sources 평면 구조, Excel/pandas용).

### 실행

```
# 환경변수는 저장소 루트 .env에서 로드 (git에는 커밋되지 않음, .gitignore 처리됨)
python Scripts/kg/load_ground_truth.py --dry-run   # DB 연결 없이 집계 수치만 확인
python Scripts/kg/load_ground_truth.py              # 실제 Neo4j Aura 적재
```

필요 환경변수 (`.env`): `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`.

### 적재 결과 (2026-07-14 실행 기준)

| 항목 | 개수 |
|---|---|
| 고유 개체 노드 | 47,869 (PER 11,182 / LOC 10,755 / MISC 10,344 / ORG 8,281 / TIME 5,338 / NUM 1,969) |
| 고유 관계 엣지 | 45,785 (96개 관계 타입) |

### evidence (`export_triples.py` / `triples.jsonl`, Neo4j에는 없음)

- `export_triples.py`는 원본 DocRED JSON에서 직접 문서 단위 raw triple을 뽑으며, 각 레코드에 `evidence_source` 필드로 evidence의 출처를 표시함:
  - `annotated` (48,547개): 사람이 직접 단 evidence 그대로.
  - `inferred_cooccurrence` (515개): 원본 evidence가 비어있었지만, head/tail이 같은 문장에 함께 언급되는 문장을 찾아 evidence로 채움.
  - `unresolved_multihop` (1,393개): 함께 언급되는 문장이 아예 없어(여러 문장에 걸친 multi-hop 추론 필요) evidence를 비워둔 채 표시만 함 — 근거 없는 추론으로 데이터 정합성을 해치지 않기 위함. `unresolved_multihop.jsonl`로 별도 추출됨.
- **Neo4j 그래프에는 이 evidence 정보가 없음**: `load_ground_truth.py`가 만드는 그래프는 같은 (head, tail, relation) triple이 여러 문서에서 반복되면 엣지 하나로 병합해 `sources`(문서 목록)만 남기기 때문에, 문서별 evidence 문장을 그대로 붙이기 애매함. 필요하면 `sources`를 `[{document, sentence_id, evidence_text}, ...]` 구조로 바꿔서 반영 가능 (아직 미구현).

## 다음 단계 (미구현)

- 2단계: 모델이 예측한 (미검증) triple을 낮은 confidence로 적재 — 향후 작업.
- Neo4j Bloom 시각화 씬 구성 (`README.md`의 5.2절 참고).
- Neo4j 그래프 엣지에도 evidence 문장 반영 (원할 경우).
