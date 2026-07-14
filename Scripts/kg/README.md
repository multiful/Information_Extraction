# Knowledge Graph 적재 (`Scripts/kg/`)

> **최종 업데이트**: 2026-07-14: 공통 개체 라벨을 `:Entity`에서 `:ZEntity`로 개명 — Neo4j Browser/Bloom이 다중 라벨 노드의 색을 알파벳순 첫 라벨 기준으로 정하는 것으로 보여, `PER`/`LOC`/... 보다 항상 뒤에 오도록 이름을 바꿈. Neo4j에 잘못 올라간 `:Triple`(CSV를 그래프 매핑 없이 통째로 노드화한 것) 20,528개도 제거. `Scripts/kg/export_csv.py`(Neo4j → `triples.csv` 평면 내보내기) 추가. (이전 업데이트: 개체 노드에 DocRED type을 보조 라벨로 추가, 관계 엣지를 `relation_name` 기반 동적 타입으로 마이그레이션.)

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

## 다음 단계 (미구현)

- 2단계: 모델이 예측한 (미검증) triple을 낮은 confidence로 적재 — 향후 작업.
- Neo4j Bloom 시각화 씬 구성 (`README.md`의 5.2절 참고).
