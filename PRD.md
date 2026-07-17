# PRD: DocRED 관계 추출 — 우리 모델 vs ATLOP 비교

> 가이드라인/설계 문서입니다. 코드는 포함하지 않습니다.
>
> **최종 업데이트**: 2026-07-17: 이 문서는 프로젝트 초기(2026-07-10)에 확정한 범위 — "팀원 각자의 경량 모델 vs ATLOP 재현을 비교해 택일"·"KG/Neo4j/GraphRAG/LangGraph는 범위 밖" — 를 기록한 것으로, 이후 실제 진행은 이 결정 절차 대신 **ATLOP 뼈대를 직접 개선(GCN/GAT, DREEAM/GREP, PU AT-Loss)하고 Neo4j 지식그래프·GraphRAG 파이프라인까지 구축**하는 방향으로 확장되었다(최종 결과는 `README.md`·`1조_발표자료.pdf` 참고). 아래 내용은 초기 계획 기록으로 남겨둔다.

## 0. 이번 프로젝트의 최종 목표 (범위 확정)

1. 팀원 각자 자신만의 **경량 RE 모델**을 만든다 (RoBERTa 계열 인코더 + pooling + MLP classifier 기반, 세부 구성은 자유)
2. 별도로 **ATLOP을 재현**해서 비교 기준(baseline)으로 둔다
3. 두 트랙을 **동일한 dev 셋 + 동일한 평가 스크립트**로 F1/Ign F1 비교한다
4. **판단 기준(정량)**: 우리 모델 중 최고 성능 모델의 dev Micro F1이 **ATLOP dev Micro F1의 90% 이상**이면 → 그 모델을 최종 채택. 미만이면 → ATLOP을 최종 채택. (예: ATLOP이 60.0이면 54.0 이상이면 우리 모델 채택. 이 90% 기준은 팀 논의로 조정 가능하지만, 반드시 숫자로 미리 정해두고 시작할 것 — "비슷하다"를 사후에 판단하면 팀 내 이견이 생김)
5. **이 판단까지가 프로젝트 범위**. PRD 하단의 "확장성"(Knowledge Graph/Neo4j/GraphRAG/LangGraph)은 이번 범위에 포함하지 않음 — 아이디어로만 남겨둠

## 1. 현재 구조 (이미 구현/검증됨)

```
data/
  docred_io.py        # 원본 JSON 로드(4개 split) + rel2id 매핑 (Na=0 + 96종 P-code)
  docred_dataset.py    # DocREDataset — 팀 공통 분기점. title/sents/vertexSet/labels를
                        #   토크나이저 없이 원본 그대로 반환 (torch Dataset)
  tokenization.py       # tokenize_document(doc, tokenizer, max_length) — 토크나이저를
                        #   인자로 받는 범용 함수. 각자 브랜치에서 자기 모델의
                        #   tokenizer를 넣어 호출. 문서 전체를 하나의 시퀀스로 토큰화하고
                        #   각 멘션의 subword 위치(entity_pos)를 계산해줌
  collate.py             # 토큰화 이후 배치 패딩용 선택적 유틸

Scripts/
  eda_docred.py          # 완료된 EDA 스크립트 (EDA/summary.md, EDA/figures/*.png 생성)

EDA/                      # 데이터 통계 (완료)
docred_data/               # 원본 DocRED 데이터
```

`DocREDataset`은 RoBERTa든 BERT든 어떤 토크나이저에도 종속되지 않음을 실제로 검증함 (동일 멘션이 두 토크나이저 모두에서 정확히 복원됨). **팀원 분기는 정확히 이 지점 이후부터** — 토크나이저 선택, 엔티티 임베딩 추출, pair 구성, classifier는 각자 브랜치.

## 2. 트랙 1 — 우리 모델(들)

각 팀원이 `data.docred_dataset.DocREDataset`을 공통 입력으로 삼아 자기 브랜치에서 아래 기본 파이프라인을 구현:

```
DocREDataset (공통) → 토큰화(각자 tokenizer) → Entity Embedding(Mention Average Pooling)
  → Entity Pair 생성(Head⊕Tail concat) → 2-layer MLP → BCEWithLogitsLoss → Micro F1/Ign F1
```

| 구성 요소 | 기본값 |
|---|---|
| Encoder | RoBERTa-base (각자 원하는 모델로 대체 가능) |
| Entity Pooling | Mention Average Pooling |
| Pair Representation | Head ⊕ Tail (concat) |
| Classifier | 2-layer MLP |
| Loss | BCEWithLogitsLoss |
| Optimizer | AdamW |
| Scheduler | Linear Warmup + Decay |
| 학습 데이터 | train_annotated만 사용 (이번 범위에서는 train_distant 사전학습 생략) |

**⚠️ 불균형 처리 필수 (빠지면 안 됨)**: DocRED는 엔티티 쌍의 **97%가 NA**(EDA에서 확인)라서, `BCEWithLogitsLoss`를 아무 조치 없이 0.5 threshold로 쓰면 모델이 거의 다 "관계 없음"으로 찍어버려 recall이 붕괴할 가능성이 높음. ATLOP은 Adaptive Thresholding으로 이 문제를 구조적으로 푸는데, Track 1이 이 처리를 안 하면 "아키텍처가 나빠서"가 아니라 "불균형 처리를 안 해서" 지는 것이라 ATLOP과의 비교 자체가 무의미해짐. 최소 아래 중 하나는 반드시 적용:
- `BCEWithLogitsLoss(pos_weight=...)`로 양성 클래스에 가중치 부여, 그리고/또는
- 고정 0.5 대신 **dev 셋에서 F1을 최대화하는 threshold를 탐색**(예: 0.1~0.9를 sweep)해서 그 값으로 최종 예측 결정

**고도화 옵션** (여유 있는 팀원이 실험해볼 것, 강제 아님):
- Entity Marker (`[E1]`/`[/E1]` 등 멘션 앞뒤 마커)
- Mention Attention Pooling (평균 대신 attention으로 멘션 통합)
- Distance Embedding (두 엔티티 간 문장 거리)
- Sentence Position Embedding
- Pair Filtering (후보 쌍 축소)
- Classifier를 Bilinear로 교체

각자 스크립트는 `Scripts/models/<이름>_model.py` 형태로 분리 저장 (예: `Scripts/models/dk_model.py`) — 서로 다른 브랜치/실험이 파일 하나로 섞이지 않도록.

## 3. 트랙 2 — ATLOP (비교 baseline)

기존에 조사해둔 내용(공식 저장소 확인 완료, 추측 아님) 재사용:

- **엔티티 표현**: mean이 아니라 **logsumexp pooling**. 멘션 앞뒤에 `"*"` 마커 삽입 후 마커 위치 hidden state를 logsumexp로 결합
- **Localized Context Pooling**: head/tail attention row를 곱해서 관련 문맥에 집중
- **Adaptive Thresholding Loss (ATLoss)**: NA/TH 클래스를 학습 가능한 임계값으로 처리, 불균형 대응
- **구현 방식**: `wzhouad/ATLOP`은 라이선스 불명확(`license: null`)이라 그대로 vendoring하지 않고, 핵심 로직만 직접 재구현(출처 주석 명시). `thunlp/DocRED`(MIT 라이선스)의 공식 스코어러(`evaluation.py`)는 포팅 가능
- **512 토큰 초과 문제**: `process_long_input` 방식(512~1024 구간을 겹치는 두 청크로 평균) 적용 시 실측상 전량 문서가 1024 토큰 이내라 문제 없음
- **학습 데이터**: 이번 범위에서는 트랙 1과 동일하게 **train_annotated만** 사용 (distant 2단계 학습은 생략 — 원래 계획에 있었지만 "여기까지만" 범위에 맞춰 축소)
- 스크립트는 `Scripts/atlop/` 아래 별도 위치 (예: `re_model.py`, `train_re.py`)

## 4. 공통 평가 (반드시 통일해야 하는 두 번째 지점)

모델 구조는 팀원마다 다르지만, **평가는 반드시 하나의 기준으로 통일**해야 "우리 모델이 ATLOP만큼 잘하는지" 비교가 의미 있어짐. 데이터 로딩만 공통화하고 평가를 각자 짜면 비교 자체가 성립하지 않음.

- **예측 결과 포맷 통일** (모든 트랙이 이 형식으로 결과를 내보냄):
  ```json
  [{"title": "...", "h_idx": 0, "t_idx": 4, "r": "P17"}, ...]
  ```
- **공통 스코어러** 하나를 `data/` 또는 별도 `eval/` 폴더에 두고, 모든 트랙(우리 모델들 + ATLOP)이 동일 스코어러로 dev F1 / Ign F1 계산
- Ign F1: train_annotated에 등장한 triple은 제외하고 계산 (누수 방지, "새로운 사실을 얼마나 잘 뽑는지" 평가)
- 필요하면 EDA에서 이미 확인한 intra-/inter-sentence 구분(`Scripts/eda_docred.py`의 `head_sents & tail_sents` 로직)을 재사용해 관계 유형별로도 비교 가능하게 확장 가능 (필수는 아님)

## 5. 최종 결정 절차

**시드(seed) 관련 주의**: DocRED는 학습 run마다 dev F1이 ±1점 정도 흔들리는 것으로 알려져 있음. 시간 여유가 있으면 최종 후보(우리 최고 모델, ATLOP) 각각 **최소 2 seed**로 돌려 평균을 비교값으로 쓸 것. 시간이 없어 1 seed만 돌린다면, **0번 판단 기준(90%)에 근접한 차이(±2~3점 이내)는 seed 노이즈일 수 있다는 점을 감안**하고 팀 논의로 최종 결정할 것 — 근소한 차이만으로 기계적으로 결론 내리지 않기.

1. 각 팀원이 자기 모델을 train_annotated로 학습, 공통 스코어러로 dev F1/Ign F1 산출
2. ATLOP도 동일하게 dev F1/Ign F1 산출
3. 팀원 모델들 중 최고 성능 모델을 뽑아 ATLOP과 비교
4. **0번의 정량 기준(ATLOP dev F1의 90% 이상)** 충족 → 그 모델을 최종 채택
5. **미충족** → ATLOP을 최종 모델로 채택
6. 여기서 프로젝트 종료 — 이후 KG 구축/Neo4j/GraphRAG/LangGraph는 이번 범위 밖 (아이디어 노트로만 남김)

## 6. 제안 폴더 구조

```
data/                      # (기존) 공통 로더 — 수정 없음
Scripts/
  eda_docred.py             # (기존)
  models/
    <각자이름>_model.py       # 트랙 1: 팀원별 모델 스크립트
  atlop/
    re_model.py               # 트랙 2: ATLOP 재구현
    train_re.py
  eval/
    scorer.py                  # 공통 평가 스코어러 (F1/Ign F1), 모든 트랙이 공유

results/
  <모델명>_dev_predictions.json   # 트랙별 예측 결과 (gitignore, 재생성 가능)
  comparison.md                     # 최종 비교표 (git 추적) — 5번 결정 근거를 여기 기록
```

## 7. 체크리스트 (구현 전 확인용)

- [ ] 모든 트랙이 `DocREDataset`을 그대로 입력으로 쓰는지 (데이터 로딩 재구현 금지)
- [ ] 모든 트랙이 동일한 `rel2id`(Na=0 + 정렬된 96 P-code)를 쓰는지
- [ ] 모든 트랙이 4번의 공통 예측 포맷으로 결과를 내보내는지
- [ ] 모든 트랙이 같은 공통 스코어러로 dev F1/Ign F1을 계산하는지
- [ ] Track 1 각 모델이 `pos_weight` 또는 dev-threshold sweep 중 하나로 NA 불균형을 처리했는지 (안 했으면 ATLOP과 비교 무의미)
- [ ] `results/comparison.md`에 각 모델의 dev F1/Ign F1과 최종 선택 근거가 기록되는지 (몇 seed로 측정했는지도 함께 기록)
