# 연구 계획서
## 문서 수준 관계 추출(Document-level Relation Extraction)을 통한 비정형 텍스트의 구조화 지식 자동 생성

> 본 문서는 README의 **Information Extraction** 프로젝트 정의(DocRED 데이터셋)에 맞춰 작성한 연구 계획서입니다.
> "비정형 텍스트에서 구조화된 지식(Entity–Relation)을 자동으로 추출한다"는 과제를 AI 문제로 재해석하는 과정을 담았습니다.

---

## 1. Problem Definition

### 1.1 문제점 : 비즈니스 니즈
기업이 보유한 지식의 대부분은 뉴스·보고서·위키·계약서 등 **비정형 텍스트**에 흩어져 있다. 검색·분석·추론(예: Knowledge Graph 구축, RAG의 검색 품질 개선, LLM 출력의 사실성/안정화)을 하려면 이 텍스트를 `(객체, 관계, 객체)` 형태의 **정형 데이터**로 바꿔야 한다. 그러나 실제 관계 정보는 한 문장 안에 있지 않고 **문서 전반에 분산**되어 있으며, 동일 객체가 여러 표현으로 등장하고(coreference), 관계가 **명시적으로 드러나지 않아 추론이 필요**한 경우가 많다. 요구사항은 **"문서를 읽고, 흩어진 객체와 관계를 통합해 구조화된 지식으로 만들어라"**이다.

### 1.2 Task Reformulation
- **초기 접근 방법:** "관계 추출 = 한 문장 안 두 엔티티 쌍의 관계 분류"(sentence-level RE).
  → 그러나 문서에서 관계의 상당수(DocRED 기준 약 40%)는 **여러 문장을 종합해야** 성립하고, 하나의 엔티티가 여러 멘션(대명사·약칭 포함)으로 흩어져 있으며, "A는 B의 수도" + "B는 C에 위치" → "A는 C에 위치"처럼 **다단계 추론(multi-hop)**이 필요하다. 문장 단위 RE는 이 관계들을 구조적으로 놓친다. **"문서 전체를 읽고 통합하라"에 답하지 못한다.**
- **현재 접근 방법:** 비즈니스 질문의 핵심은 "문서 수준(document-level)"이다. 이를 AI 문제로 옮기면 세 하위 문제로 분해된다.
  1. 문서 내 **객체(멘션) 식별 및 동일 객체 통합** — NER + *coreference resolution*
  2. 임의의 두 엔티티 쌍에 대한 **관계 분류** (96종 관계 + 관계없음) → *document-level relation classification*
  3. 관계의 근거가 되는 **evidence 문장 식별** 및 **다중 홉 추론** → *evidence / multi-hop reasoning*

### 1.3 Task 설계
**Document-level Relation Extraction (DocRE)** — 문서가 주어지면 모든 유효한 `(head entity, relation, tail entity)` triple과 그 근거(evidence) 문장을 추출하는 문제.
- 하위 태스크: Entity/Mention Detection + Coreference → Relation Classification(RC) + Evidence Prediction
- 최종 산출물: 문서별 관계 triple 집합 → **Knowledge Graph / Relation DB / Graph-based reasoning system**의 입력으로 직결

### 1.4 이 문제가 중요한 이유
- 문장 단위 IE로는 문서에 흩어진 사실을 통합하지 못한다. DocRE는 이를 **재사용 가능한 구조화 지식**으로 바꾼다.
- **응용 맥락:** KG 자동 구축, RAG 검색 컨텍스트의 정밀화, LLM 할루시네이션 억제(온톨로지 기반 검증 레이어), 관계형 DB·그래프 추론 시스템의 자동 적재.
- 확장성: 도메인(뉴스·바이오·법률)만 바꿔 재사용 가능하고, 시계열·다국어로 확장된다.

---

## 2. Background & Baseline

### 2.1 Related Works
- **그래프 기반 신경망:** 문서를 멘션·엔티티·문장 노드의 그래프로 모델링하고 GCN/경로 추론으로 관계를 잡음(예: EoG, LSR, GAIN 계열). 문서 수준 추론에 강하지만 구조가 복잡하다.
- **Pretrained LM 기반 pooling:** BERT/RoBERTa로 문서를 인코딩하고 엔티티 표현을 pooling해 쌍을 분류. 재현이 쉽고 강력한 표준 계열.
- **ATLOP:** *Adaptive Thresholding*(관계마다 임계값을 학습해 다중 라벨·불균형 대응) + *Localized Context Pooling*(엔티티 쌍에 관련된 문맥만 집중). 현재 재현이 쉽고 강한 표준 baseline.
- **엔티티-쌍 행렬/추론 강화:** DocuNet(엔티티 쌍 행렬에 U-Net 적용), SSAN(구조적 self-attention) 등.
- **Evidence/Generative/LLM 방식:** evidence를 명시적으로 예측해 추론을 유도하거나, seq2seq로 triple을 직접 생성, 또는 LLM few-shot/in-context로 추출. 오류 전파가 적고 저자원에 유리하나 관계 스키마 정합·환각 이슈가 있다.

### 2.2 이번 프로젝트의 Baseline
- **Primary baseline: ATLOP (BERT / RoBERTa 기반)**
  1. 문서 인코딩 후 엔티티(멘션 통합) 표현 생성
  2. 엔티티 쌍마다 localized context pooling → 관계 분류 + adaptive threshold로 최종 triple 결정
- **선정 이유:** (a) 성능이 충분히 강하고 재현이 쉬워 **공정한 비교 기준**이 되며, (b) 우리가 개선하려는 지점(coreference로 흩어진 엔티티, inter-sentence·multi-hop 관계, 극심한 NA 불균형)이 구조적으로 명확히 드러난다.
- **(참고용 하한 baseline)** sentence-level RE, LLM few-shot 프롬프팅 — 문서 수준 접근의 이득과 절대 성능 감을 잡는 용도.

---

## 3. Proposed Method

### 3.1 핵심 아이디어
**Coreference-aware & Evidence-guided DocRE** — ATLOP 구조를 유지하되 두 가지를 결합한다.
1. **Coreference-aware entity representation:** 동일 엔티티의 여러 멘션을 명시적으로 묶어 표현을 강화(멘션 간 attention/그룹 pooling). 멘션이 문서 전반에 흩어진 엔티티의 관계를 더 잘 포착.
2. **Evidence-guided multi-hop 학습:** evidence 문장 예측을 **보조 태스크**로 함께 학습해, 여러 문장을 잇는 추론 경로에 모델이 집중하도록 유도.
추가로 **distant supervision 데이터로 사전학습 → annotated 데이터로 fine-tune**하여 long-tail 관계 일반화를 강화한다.

### 3.2 Baseline 대비 무엇이 다른가
| 구분 | Baseline (ATLOP) | Proposed (Coref + Evidence-guided) |
|---|---|---|
| 엔티티 표현 | 멘션 pooling(암묵적 통합) | 명시적 coreference 그룹 표현 |
| Multi-hop 추론 | 문맥에 암묵적으로 의존 | evidence 예측 보조 태스크로 경로에 집중 |
| Inter-sentence 관계 | 상대적으로 취약 | evidence·coref 신호로 강화 |
| Long-tail 관계 | annotated만 사용 시 부족 | distant 사전학습으로 일반화 |
| 불균형(NA 다수) | adaptive threshold | adaptive threshold 유지(계승) |

### 3.3 Dataset & Preprocessing
- **사용 데이터셋: DocRED** (Wikipedia + Wikidata, 영어)
  - train_annotated 3,053 / train_distant 101,873 / validation 998 / test 1,000 문서
  - 관계 96종, 엔티티 타입 6종(PER, ORG, LOC, TIME, NUM, MISC), evidence 문장 라벨 포함
- **데이터 특성**
  - 문서당 평균 약 26문장·19.5엔티티·다수의 관계 → **문서 수준·다중 멘션**
  - 관계의 상당수가 **inter-sentence(≈40%)**, 추론 필요 관계 비중이 큼(coreference·다중 홉·상식)
  - 엔티티 쌍의 대다수가 **관계 없음(NA)** → 극심한 클래스 불균형, 관계 분포는 **long-tail**
  - train_distant는 규모가 크지만 **원거리 감독(noisy label)**
- **전처리**
  - 토크나이저(subword) 정렬 및 멘션 span 매핑, 엔티티 쌍 후보 생성
  - coreference 그룹(vertexSet) 구조화, evidence 라벨 정리
  - NA 쌍 **negative sampling** 및 클래스 가중치로 불균형 대응
  - distant→annotated **2단계 학습** 파이프라인 구성, 공식 평가 포맷에 맞춘 예측 직렬화

### 3.4 왜 효과가 있을 것으로 예상하는가 (검증 대상 가설)
- **H1.** 명시적 coreference 표현은 멘션이 흩어진 엔티티의 관계 **recall**을 개선할 것이다.
- **H2.** evidence-guided 학습은 여러 문장을 잇는 **inter-sentence / multi-hop 관계 F1**을 높일 것이다.
- **H3.** distant 사전학습은 **long-tail 관계** 일반화에 이점을 주어 소수 관계 성능을 개선할 것이다.

---

## 4. Experiment Design

### 4.1 수행할 실험
- **E1.** Baseline(ATLOP) 재현 및 성능 측정
- **E2.** Proposed(Coref + Evidence-guided) 학습 및 측정
- **E3.** Ablation — coreference 표현 유무 / evidence 보조 태스크 유무 / distant 사전학습 유무의 효과 분리
- **E4.** (선택) LLM few-shot vs fine-tuned 소형 모델 비교
- **E5.** 정성 분석 — 오류 유형 분류, 특히 inter-sentence·multi-hop·희소 관계 사례 집중

### 4.2 비교 축
- Sentence-level vs Document-level (문제 정의 차원의 이득)
- **Intra-sentence vs Inter-sentence** 관계별 성능(주 비교)
- Frequent vs Long-tail 관계별 성능
- Distant 사전학습 유무에 따른 데이터 효율 곡선

### 4.3 Evaluation Metrics 및 검증
- **지표 설계**
  - **F1 / Ign F1** (train에 등장한 triple을 제외한 F1, 누수 방지 — 주 지표)
  - **Evidence F1** (근거 문장 예측 품질)
  - Intra-/Inter-sentence F1을 **분리 보고**해 multi-hop 개선을 직접 확인
  - Precision / Recall 병기(불균형·recall 개선 가설 확인)
- **지표 검증 방법**
  - *타당성:* Ign F1이 "새로운 사실 추출 능력"(KG 신규 적재 가치)과 정렬되는지 확인
  - *신뢰성:* 3개 seed 평균 ± 표준편차 보고, DocRED **공식 스코어러**로 평가(dev 기준, test는 리더보드 포맷)
  - *사람 평가:* 무작위 표본에 대해 예측 triple의 정확성/근거 타당성을 사람이 대조
  - *sanity check:* 추출 triple로 소규모 KG를 만들어 관계 방향·타입 제약(엔티티 타입 정합성) 위반 여부 점검

---

## 5. Plan (일자별)

| 일자 | 수행 내용 | 산출물 |
|---|---|---|
| 7/09 | 팀 구성 및 주제 선정(Information Extraction / DocRED) | 주제 확정 |
| 7/10 | 데이터 분석(DocRED 구조·통계), baseline 후보 탐색 | 데이터 분석 노트북 |
| 7/13 | 연구 계획서 확정, 전처리·평가 파이프라인 설계 | 연구 계획서, 스켈레톤 코드 |
| 7/14 | Baseline(ATLOP) 재현 및 1차 성능 측정 | 1차 모델·성능(F1/Ign F1) |
| 7/14 – 7/16 | Proposed 학습, Ablation·하이퍼파라미터 탐색, coref/evidence 실험 | 실험 표, 학습 곡선 |
| 7/17 – 7/19 | 결과 정리, 정성 분석, 발표자료 작성 | 시연/데모, PPT |
| 7/20 | 발표 진행 | 최종 발표 |
