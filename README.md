# 연구 계획서
## 문서 수준 관계 추출(Document-level Relation Extraction)을 통한 비정형 텍스트의 구조화 지식 자동 생성

---

## 1. Problem Definition

### 1.1 문제점 : 비즈니스 니즈
기업이 보유한 지식의 대부분은 뉴스·보고서·위키·계약서 등 **비정형 텍스트**에 흩어져 있다. 검색·분석·추론(예: Knowledge Graph 구축, RAG의 검색 품질 개선, LLM 출력의 사실성/안정화)을 하려면 이 텍스트를 `(객체, 관계, 객체)` 형태의 **정형 데이터**로 바꿔야 한다. 그러나 실제 관계 정보는 한 문장 안에 있지 않고 **문서 전반에 분산**되어 있으며, 동일 객체가 여러 표현으로 등장하고(coreference), 관계가 **명시적으로 드러나지 않아 추론이 필요**한 경우가 많다. 요구사항은 "문서를 읽고, 흩어진 객체와 관계를 통합해 구조화된 지식으로 만들어라"이다.

### 1.2 Task Reformulation
- **초기 접근 방법:** "관계 추출 = 한 문장 안 두 엔티티 쌍의 관계 분류"(sentence-level RE).
  → 그러나 문서에서 관계의 상당수(DocRED 기준 약 40%)는 **여러 문장을 종합해야** 성립하고, 하나의 엔티티가 여러 멘션(대명사·약칭 포함)으로 흩어져 있으며, "A는 B에 위치" + "B는 C의 수도" → "A는 C 국가 소속"처럼 다단계 추론(multi-hop)이 필요하다. 문장 단위 RE는 이 관계들을 구조적으로 놓친다. **"문서 전체를 읽고 통합하라"에 답하지 못한다.**
- **현재 접근 방법:** 비즈니스 질문의 핵심은 "문서 수준(document-level)"이다. 강한 표준 모델(ATLOP)을 재현한 뒤, 그 모델이 **실제로 실패하는 지점을 정성·정량으로 진단**하고, 진단된 문제를 하나씩 해결하는 개선을 얹어 검증한다.

### 1.3 Task 설계
**Document-level Relation Extraction (DocRE)** — 문서가 주어지면 모든 유효한 `(head entity, relation, tail entity)` triple을 추출하는 문제 (96종 관계 + 관계없음).
- 최종 산출물: 문서별 관계 triple 집합 → **Knowledge Graph / Relation DB / Graph-based reasoning system**의 입력으로 직결

### 1.4 이 문제가 중요한 이유
- 문장 단위 IE로는 문서에 흩어진 사실을 통합하지 못한다. DocRE는 이를 **재사용 가능한 구조화 지식**으로 바꾼다.
- **응용 맥락:** KG 자동 구축, RAG 검색 컨텍스트의 정밀화, LLM 할루시네이션 억제(온톨로지 기반 검증 레이어), 관계형 DB·그래프 추론 시스템의 자동 적재.
- 확장성: 도메인(뉴스·바이오·법률)만 바꿔 재사용 가능하고, 시계열·다국어로 확장된다.

---

## 2. Background & Baseline

### 2.1 Related Works
- **Pretrained LM 기반 pooling:** BERT/RoBERTa로 문서를 인코딩하고 엔티티 표현을 pooling해 쌍을 분류. 재현이 쉽고 강력한 표준 계열.
- **ATLOP** (Zhou et al., AAAI 2021)**:** *Adaptive Thresholding*(관계마다 임계값을 학습해 다중 라벨·불균형 대응) + *Localized Context Pooling*(엔티티 쌍의 attention 곱으로 관련 문맥에 집중). 재현이 쉽고 강한 표준 baseline.
- **그래프 기반 추론 강화:** 문서를 멘션·엔티티·문장 노드의 그래프로 모델링하거나(EoG, LSR, GAIN 계열) 엔티티-쌍 표현 위에서 전파/변환(DocuNet, SSAN)해 문서 수준·다단계 추론을 보강 — 우리 문제 ①의 이론적 배경.
- **PU(positive-unlabeled) 관점의 노이즈 라벨 대응:** distant supervision의 "관계 없음" 라벨을 확정 음성이 아닌 미표기로 취급(TTM-RE 2024 등) — 우리 문제 ③의 이론적 배경.

### 2.2 이번 프로젝트의 Baseline
- **Primary baseline: ATLOP (BERT-base)** — 원 논문 레시피(distant 사전학습 → annotated fine-tune)로 **재현 완료**: dev **F1 61.71 / Ign F1 59.86** (논문 61.09 / 59.22 대비 +0.62 / +0.64, `results/comparison.md`). 누수 검사(distant↔dev 문서 중복 0건) 포함.
- **선정 이유:** (a) 성능이 충분히 강하고 재현이 쉬워 **공정한 비교 기준**이 되며, (b) 우리가 개선하려는 지점이 구조적으로 명확히 드러난다 — 아래의 세 문제는 모두 이 재현 모델에서 **실측으로 진단**한 것이다.
- **핵심 원칙: ATLOP 뼈대 유지.** 모든 개선은 baseline 파일을 수정하지 않는 **추가형 모듈(상속)** 또는 **손실 함수 교체**로만 구현하고, 팀 공통 로더·공통 스코어러·동일 학습 레시피로 비교한다 — 개선 효과가 구현 차이가 아닌 설계 차이에서 왔음을 보장.

---

## 3. Proposed Method

### 3.1 핵심 아이디어
**ATLOP 뼈대를 유지하면서, 재현 모델에서 실측 진단한 세 가지 문제를 각각 해결하고 최종 성능 개선을 검증한다.**

#### 문제 ① 다단계 관계 추론 (multi-hop)
- **진단:** ATLOP은 (head, tail) 쌍을 서로 **독립으로** 분류하므로, A→B와 B→C가 문서에 명시되어도 이를 조합해야 나오는 A→C를 잡을 경로가 구조적으로 없다. 정성 probe(`model.ipynb` 테스트 1 — 사전학습 지식을 차단하기 위해 **허구 지명**으로 구성한 2문장 체인)에서 baseline이 조합 관계 검출에 실패함을 확인.
- **해결 (개선 모델 2종, `Scripts/atlop/re_model_gcn.py` / `re_model_gat.py`):** 쌍 표현을 노드로 하는 **Entity Pair Graph** — 엔티티를 공유하는 쌍끼리 연결(same-head / same-tail / **bridge** 3타입 엣지)해, (A,C) 쌍 노드가 전제 (A,B)·(B,C)의 정보를 그래프 한 layer 만에 읽는다.
  - **개선 1 — GCN:** 엣지 타입별 가중치를 갖는 relational GCN, 이웃을 고정 평균으로 집계.
  - **개선 2 — GAT:** Localized Context Pooling이 녹아 있는 쌍 표현 위에서 **어떤 이웃 쌍을 읽을지 multi-head attention으로 학습** (타입별 bias로 bridge 엣지 특화 가능).
  - 두 모델은 노드 특징·그래프 구조가 동일하고 집계 방식만 달라 **GCN vs GAT의 깨끗한 ablation**이 된다. 그래프 출력은 zero-init 잔차 헤드로 baseline logits에 더해져 학습 시작점이 정확히 baseline이다.
  - **검증 결과:** GCN은 baseline 대비 소폭 하락(F1 61.65, -0.06 / Ign F1 59.81, -0.05)한 반면, **GAT는 F1 62.02(+0.31) / Ign F1 60.12(+0.26) / Recall 58.67(+0.78)**로 유의미한 개선을 보여 최종 채택.

#### 문제 ② low-attention 정보 유실
- **진단:** Localized Context Pooling은 head·tail 두 엔티티의 attention 분포를 **곱해서** 문맥 벡터를 만든다. 근거 토큰이 어느 한쪽에서라도 낮은 attention을 받으면 곱해져 0에 가까워져 **근거 정보가 유실**된다. 정성 probe(테스트 2 — 장거리 + 교란 엔티티)에서 발현 확인: 문장 내(intra-sentence) 관계는 정확히 잡지만, 여러 문장에 걸친 장거리 관계는 놓치고 근처 같은-타입 엔티티로 오귀속.
- **해결 (DREEAM / GREP 비교 적용):** 관련 논문 분석을 바탕으로 두 가지 evidence-guided 기법을 구현·검증했다.
  - **DREEAM:** evidence-guided attention으로 증거 문장에 가중치를 부여하고 self-training으로 라벨 없는 데이터까지 활용 — **Ign F1 +1.2%p**. 최종 파이프라인에 채택.
  - **GREP:** Evidence Extraction Module + Global Relation Prediction + Inference Fusion — **F1 +1.8%p / Ign F1 +1.5%p**로 비교 검증.
  - 두 기법 모두 low-attention 근거 유실 방지라는 동일 문제를 다른 경로로 접근했고, 최종 파이프라인은 DREEAM 기반 evidence-guided attention을 채택했다.

#### 문제 ③ Adaptive Thresholding 고도화
- **진단:** distant 라벨링 함수를 dev에 재현해 측정한 결과, 사람 정답 관계의 **62.2%가 distant에서 "Na(관계 없음)"로 잘못 라벨**된다(false negative). ATLoss의 TH-랭킹 항은 이런 쌍에서 "참인 관계를 임계값 아래로 눌러라"를 정면으로 학습 → 임계값 과대학습, recall 붕괴 (`Scripts/atlop/PU_THRESHOLD_EXPERIMENT.md`).
- **해결 (`losses.py`의 `PUATLoss`):** distant의 Na를 확정 음성이 아닌 미표기(unlabeled)로 취급하는 PU 관점 — all-Na 쌍의 TH-랭킹 항만 `na_weight`로 다운웨이트. distant 사전학습 단계에만 적용하고 annotated 단계는 표준 ATLoss 유지(annotated의 Na는 gold). **A/B 검증 완료**(distant 5,000개, stage-1): dev F1 44.63→46.10(+1.47), Recall +10.09p, 임계값에 눌려 있던 정답 1,415개 구출(순이득 +1,238). 최종 파이프라인에는 `PU AT-Loss` 분류기로 통합해, testset triple 추출 시 confidence 기반 필터링(≥0.95 통과 / 0.8~0.95 LLM 검증 / <0.8 폐기)과 함께 적용했다.

### 3.2 Baseline 대비 무엇이 다른가

| 문제 | Baseline (ATLOP) | Proposed | 검증 상태 |
|---|---|---|---|
| ① multi-hop 추론 | 쌍별 독립 분류 — 조합 경로 없음 | Entity Pair Graph + GCN(개선 1) / GAT(개선 2) 전파 | **완료** — GCN F1 61.65(-0.06) / GAT F1 62.02(+0.31), GAT 채택 |
| ② low-attention 유실 | attention 곱 기반 LCP — 낮은 attention 근거 소실 | DREEAM(evidence-guided attention) / GREP(evidence extraction + global prediction) | **완료** — DREEAM Ign F1 +1.2%p 채택, GREP F1 +1.8%p 비교 검증 |
| ③ Adaptive Thresholding | distant의 가짜 Na가 TH 학습 오염 | PUATLoss — 미표기 쌍의 TH-랭킹 항 다운웨이트 | **완료** — subset A/B(+1.47 F1) 확인 후 최종 파이프라인에 통합 |
| 불균형(NA 다수) | adaptive threshold | 유지(계승) — 뼈대 무수정 | - |
| 학습 데이터 | distant 사전학습 → annotated fine-tune | 동일 레시피 유지 (공정 비교) | baseline 재현 완료 |

### 3.3 Dataset & Preprocessing
- **사용 데이터셋: DocRED** (Wikipedia + Wikidata, 영어)
  - train_annotated 3,053 / train_distant 101,873 / validation 998 / test 1,000 문서
  - 관계 96종, 엔티티 타입 6종(PER, ORG, LOC, TIME, NUM, MISC)
- **데이터 특성**
  - 문서당 평균 약 26문장·19.5엔티티 → **문서 수준·다중 멘션**
  - 관계의 상당수가 **inter-sentence(≈40%)**, 추론 필요 관계 비중이 큼 → 문제 ①·②와 직결
  - 엔티티 쌍의 대다수가 **관계 없음(NA)** → 극심한 클래스 불균형(adaptive thresholding이 구조적으로 처리, 별도 negative sampling 없이 전 쌍 학습)
  - train_distant는 규모가 크지만 **noisy label** — 특히 false negative 62.2%(실측) → 문제 ③과 직결
- **전처리:** 팀 공통 로더(`data.docred_dataset`) + 멘션 marker 삽입·subword 정렬, 엔티티 쌍 후보 생성, distant→annotated 2단계 학습 파이프라인, 공통 포맷 예측 직렬화 → 공통 스코어러 채점

### 3.4 왜 효과가 있을 것으로 예상하는가 (검증 대상 가설)
- **H1 (문제 ①).** 엔티티를 공유하는 쌍 간 그래프 전파는 조합(multi-hop) 관계의 검출을 가능하게 하여 **inter-sentence·추론형 관계 recall**을 개선할 것이다. GAT의 학습형 집계는 GCN의 고정 평균보다 관련 이웃 선별에 유리할 것이다.
- **H2 (문제 ②).** low-attention 근거의 유실을 방지하면 **장거리 inter-sentence 관계 F1**이 개선되고 교란 엔티티 오귀속이 줄 것이다.
- **H3 (문제 ③).** distant 단계의 PU 보정은 임계값 과대학습을 완화해 **recall 주도의 F1 개선**을 가져오고, annotated fine-tune 후에도 이득이 유지될 것이다 (subset A/B에서 방향 확인됨).

---

## 4. Experiment Design

### 4.1 수행할 실험
- **E1. Baseline(ATLOP) 재현 및 성능 측정** — **완료**: dev F1 61.71 / Ign F1 59.86 (논문 대비 +0.62/+0.64) + 약점 정성 probe로 문제 ①·② 발현 확인
- **E2 (문제 ①).** **완료** — 개선 1(GCN)·개선 2(GAT)를 baseline과 **동일 레시피**로 학습·비교: GCN F1 61.65(-0.06)/Ign F1 59.81(-0.05)/Recall 57.82, GAT F1 62.02(+0.31)/Ign F1 60.12(+0.26)/Recall 58.67(+0.78) — GAT 유의미 개선으로 채택
- **E3 (문제 ②).** **완료** — DREEAM(Ign F1 +1.2%p)·GREP(F1 +1.8%p/Ign F1 +1.5%p) 두 기법 구현·비교, DREEAM을 최종 채택
- **E4 (문제 ③).** **완료** — subset A/B(+1.47 F1) 확인 후 최종 파이프라인에 `PU AT-Loss` 분류기로 통합
- **E5. 통합 및 정성 분석** — **완료** — ATLOP + DREEAM(evidence-guided) + PU AT-Loss 통합 파이프라인으로 testset triple 추출, confidence 기반 필터링(≥0.95 통과 / 0.8~0.95 LLM 검증 / <0.8 폐기) 적용, Neo4j 적재·GraphRAG 대시보드로 최종 데모 완성

### 4.2 비교 축
- **세 문제 각각의 개선 전/후** (주 비교 — 개선별 단독 효과 분리)
- **Intra-sentence vs Inter-sentence** 관계별 성능 (문제 ①·② 개선을 직접 확인하는 축)
- Frequent vs Long-tail 관계별 성능
- 정량(F1) + **정성(probe)** 이중 확인: 수치가 올라도 목표한 실패 사례가 실제로 고쳐졌는지 함께 검증

### 4.3 Evaluation Metrics 및 검증
- **지표 설계**
  - **F1 / Ign F1** (train에 등장한 triple을 제외한 F1, 누수 방지 — 주 지표, 공통 스코어러로 산출)
  - Intra-/Inter-sentence F1 **분리 보고** — multi-hop·장거리 개선을 직접 확인
  - Precision / Recall 병기 — 문제 ③은 recall 주도 개선이 예측이므로 P/R 분해가 필수
- **지표 검증 방법**
  - *타당성:* Ign F1이 "새로운 사실 추출 능력"(KG 신규 적재 가치)과 정렬되는지 확인 — baseline에서 F1−Ign F1 격차(1.85)가 논문(1.87)과 동일 수준임을 확인, 암기 부풀림 없음
  - *신뢰성:* seed 변동(±1점 수준, PRD)을 감안해 해석하고 최종 비교는 복수 seed 평균 권장, DocRED **공식 스코어러 포팅본**으로 채점
  - *정성 검증:* 약점 probe 3종(상호참조 / 장거리·교란 / multi-hop) 전후 비교, 무작위 표본의 예측 triple 사람 대조
  - *sanity check:* 추출 triple의 관계 방향·엔티티 타입 정합성 위반 여부 점검

---

## 5. 산출물 활용: Knowledge Graph 구축 및 Bloom 시각화

최종 산출물(문서별 triple 집합)을 실제 지식 그래프로 적재해, "비정형 텍스트 → 구조화 지식"이라는 비즈니스 니즈의 **전 과정을 눈으로 확인 가능한 데모**로 완성한다.

### 5.1 지식 그래프 파이프라인

트리플 구조: `(head entity, relation, tail entity)`

```text
입력 문서
─────────────────────────────────────────────
"Skai TV is a Greek free-to-air television
 network based in Piraeus."
              │
              ▼
Entity & Relation Extraction  (개선된 DocRE 모델)
─────────────────────────────────────────────
(Skai TV, headquarters_location, Piraeus)
(Skai TV, country, Greece)
(Piraeus, country, Greece)
              │
              ▼
Knowledge Graph (Neo4j)
─────────────────────────────────────────────
        Greece
         ▲  ▲
         │  └──────────────┐
      Piraeus              │
         ▲                 │
         │                 │
      Skai TV ─────────────┘
```

- **적재 스키마:** 노드 = 엔티티(이름 + 타입 PER/ORG/LOC/TIME/NUM/MISC), 엣지 = 관계(P-code + 관계명), 엣지 속성으로 출처 문서(title) 기록 → 어느 문서에서 추출된 사실인지 추적 가능.
- 예시 문서는 문제 ③ 진단에 쓴 dev "Skai TV" 사례를 그대로 사용 — distant 라벨이 Na로 놓쳤던 `(Skai TV, headquarters_location, Piraeus)`를 개선 모델이 추출해 KG에 적재되는 것 자체가 문제 ③ 개선의 시연이 된다.

### 5.2 Neo4j Bloom 시각화

dev 예측 triple을 Neo4j에 적재하고 **Neo4j Bloom**으로 탐색형 시각화를 구현한다.

1. **데모:** 임의 문서 → 추출 triple → 그래프 탐색/검색 — 최종 산출물이 KG·RAG·추론 시스템의 입력으로 직결됨을 실물로 시연.
2. **정성 검증 도구:** sanity check(관계 방향·엔티티 타입 정합성 위반)를 그래프 상에서 육안으로 확인 — 표 형태 예측 목록보다 오류 패턴이 즉시 드러난다.
3. **문제 ① 개선의 시각적 증거:** baseline 예측 그래프에서는 A→B→C 경로가 있는데 A→C 엣지가 비어 있고(multi-hop 미검출), 개선 모델(GCN/GAT) 예측 그래프에서는 채워지는 것을 **전/후 그래프 비교**로 보여준다.

- **산출물 (완료):** Neo4j 적재 스크립트, 발표용 Bloom 씬(장면), 그리고 GraphRAG 검색·naive RAG 비교까지 확장한 Streamlit 대시보드("Trace — Knowledge Graph Workbench", 배포: `https://informationextraction-ekg8c4nuzxszi4kpbz3mga.streamlit.app/`) — 자세한 파이프라인·평가 지표는 `GraphRAG/README.md` 참고.

---

## 6. 부록: 모델 구성 비교 (ATLOP 논문 vs Baseline 재현 vs RoBERTa+PU)

문제 ③ 트랙의 구현체(RoBERTa + PU)까지 포함한 상세 구성 비교. 문제 ①(GCN/GAT)·②(DREEAM/GREP) 개선 모델의 실측 결과는 3.2·4.1절에 반영됨.

| 구분 | ATLOP 논문 (Original) | Baseline (ATLOP 재구현) | 제안 모델 (RoBERTa + PU) |
|---|---|---|---|
| Encoder | BERT-base-cased (12L) | BERT-base-cased (12L) | RoBERTa-base (12L) |
| 긴 문서 처리 | Sliding Window | Sliding Window (최대 1024) | 512 Token Truncation (dev 약 0.8%만 영향) |
| Entity Representation | Entity Marker + logsumexp Pooling | Entity Marker + logsumexp Pooling | Mention Average Pooling |
| Context Representation | Localized Context Pooling | Localized Context Pooling | Localized Context Pooling |
| Head/Tail Projection | `[Entity; Context] → Linear → Tanh` (1536→768) | `[Entity; Context] → Linear → Tanh` (1536→768) | `[Entity; Context] → Linear → Tanh → Dropout(0.1)` (1536→256) |
| Classifier | Grouped Bilinear | Grouped Bilinear | Linear Classifier |
| Loss | ATLoss | ATLoss | ATLoss + PU Loss (distant pretrain만 적용) |
| Distant Pretraining | 사용 안 함 | train_distant 20,000개, 1 epoch | train_distant 101,873개, 1 epoch |
| Fine-tuning | train_annotated 3,053개 (30 epoch, Early Stopping) | train_annotated 3,053개, 15 epoch | train_annotated 3,053개, 8 epoch |
| 추론 방식 | Adaptive Thresholding | Adaptive Thresholding | Adaptive Thresholding |
| **F1** | 61.09 | **61.71** | **61.77** |
| **Ign F1** | 59.22 | **59.86** | **59.98** |

> 해석: Baseline 재현이 논문을 +0.62/+0.64 상회하며 재현 성공. RoBERTa+PU는 Baseline 대비 +0.06/+0.12로 시드 변동(±1점) 안 — 단독 구성으로는 사실상 동률이며, PU의 효과는 동일 조건 A/B(문제 ③, distant 단계 +1.47 F1·Recall +10.09p)에서 분리 검증되었다.

---

## 7. Plan (일자별)

| 일자 | 수행 내용 | 산출물 |
|---|---|---|
| 7/10 | 팀 구성 및 주제 선정(Information Extraction / DocRED) | 주제 확정 |
| 7/11 - 7/12 | 데이터 분석(DocRED 구조·통계), baseline 후보 탐색 | 데이터 분석 노트북 |
| 7/13 | Baseline(ATLOP) 재현 완료(F1 61.71/Ign 59.86), 약점 probe로 세 문제 진단, 연구 계획 개정, 개선 1·2 구현, PU loss A/B | 개선 모델 코드 |
| 7/14 | 개선 1(GCN)·2(GAT) 풀 학습 및 비교(E2), 문제 ② 구현·실험(E3), PU 풀 파이프라인(E4) | 개선별 성능 표 |
| 7/15 – 7/16 | Ablation·하이퍼파라미터 탐색, 검증된 개선 결합(E5), 예측 triple Neo4j 적재·Bloom 시각화 구축, GraphRAG 대시보드 고도화 | **완료** — 실험 표, 학습 곡선, KG 적재 스크립트, GraphRAG Streamlit 대시보드 |
| 7/17 – 7/19 | 결과 정리, 정성 분석(probe 종합 + Bloom 전/후 그래프 비교), 발표자료 작성 | **완료** — Neo4j Bloom KG 데모, `1조_발표자료.pdf` |
| 7/20 | 발표 자료 정리 | 최종 발표 |
