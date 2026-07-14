# 트랙 3: Adaptive Thresholding 고도화 — 문제정의 및 PU Loss 실험

> **최종 업데이트**: 2026-07-13: distant 라벨의 62.2% false-negative가 ATLoss의 TH(임계값) 학습을
> 오염시키는 문제를 데이터→수식→모델 3단계로 규명. ATLOP baseline에 PUATLoss를 추가해 A/B 검증 —
> na_weight sweep(1.0/0.7/0.5/0.3) 결과 **0.7이 최적** (stage-1 dev F1 44.63→47.56). **Colab A100
> 풀 파이프라인(distant 2만 + annotated 15ep)에서 baseline 61.71/59.86 → 62.06/60.16으로 이득이
> 파인튜닝 후에도 유지됨을 확인** (단, +0.35는 seed 변동폭 이내 — §4 참고). dev 정답 1,415개가
> 임계값 아래에서 구출됨(na_weight=0.5, 5천개 기준, 순이득 +1,238).

## 1. 문제정의: 적응형 임계값은 distant 라벨의 거짓말을 그대로 배운다

### 1-1. 데이터 근거 — distant 라벨링의 false negative 비율 측정

distant 라벨링 함수(Wikidata fact 매칭)를 dev 998문서에 재현해 사람 정답과 비교
(KB 근사 = `train_distant` 라벨의 (이름, 이름, 관계) fact 전체 1,112,120개):

| | 개수 | 비율 |
|---|---|---|
| 사람이 단 정답 관계 | 12,275 | 100% |
| distant 방식도 잡는 관계 | 4,640 | 37.8% |
| **distant 방식이 "Na"로 잘못 라벨 (FN)** | **7,635** | **62.2%** |
| distant 방식이 잘못 붙이는 라벨 (FP) | 9,534 | - |

**정답 관계 3개 중 2개가 "관계 없음"으로 라벨된 데이터**로 distant 프리트레인을 하는 셈.

### 1-2. 직관적 예시

> **문장** (dev "Skai TV", 문장 0): "Skai TV is a Greek free-to-air television network **based in Piraeus**."
> **사람 정답**: (Skai TV) —P159 headquarters location→ (Piraeus) — 문장에 명시적으로 서술됨
> **distant 라벨**: **Na** — Wikidata에 이 fact가 미등록이라는 이유만으로

### 1-3. 오염 메커니즘 (`losses.py` ATLoss Part 2)

ATLoss Part 2는 "TH를 모든 non-positive 클래스 위로" 올리도록 학습한다. 위 같은 pair는 라벨이
all-zero(Na)이므로 **참인 관계 P159도 non-positive로 취급되어, "logit_TH > logit_P159가 되도록"
그라디언트가 정면으로 흐른다.** 즉 62.2% 규모의 pair가 "참인 관계를 임계값 아래로 눌러라"를
모델에 직접 가르친다. 결과는 임계값 과대학습 → recall 붕괴 (아래 A/B의 R=38.02가 그 증상).

## 2. 해결: PUATLoss (TTM-RE 2024의 핵심 아이디어 간소화)

distant의 Na는 "확정 음성"이 아니라 **미표기(unlabeled)**라는 PU(positive-unlabeled) 관점 —
all-Na pair의 Part 2(TH-랭킹 항)만 `na_weight`(0.5)로 다운웨이트한다. positive가 하나라도 있는
pair와 Part 1은 건드리지 않고, `na_weight=1.0`이면 표준 ATLoss와 동일. (정식 TTM-RE의 nnPU risk
estimator/클래스 prior 추정/메모리 모듈은 제외한 최소 구현 — 트랙 1 dk 모델에서 동일 방식이
A/B 검증된 바 있음, `Scripts/models/EXPERIMENTS.md` 실험 2.)

**코드 변경 (모두 추가형, 기존 동작 불변)**
- `losses.py`: `PUATLoss(ATLoss)` 클래스 추가
- `re_model.py`: `DocREModel(loss_fnt=...)` 주입 파라미터 (기본값 = 기존 ATLoss)
- `train_re.py`: `--use_pu_loss`, `--na_weight` 플래그. PU는 distant 프리트레인 stage에만 적용되고
  annotated fine-tune은 항상 표준 ATLoss(annotated의 Na는 gold이므로). `--epochs 0`으로 stage-1만
  실행 가능, `--save_model` 시 `{run_name}_stage1.pt`도 저장.

## 3. 증거: A/B 실험 (ATLOP baseline, distant 5,000개, seed 66, stage-1만)

```bash
python -m Scripts.atlop.train_re --distant_mode pretrain --distant_limit 5000 \
  --distant_epochs 1 --epochs 0 --run_name atlop_pu_off --save_model --seed 66
python -m Scripts.atlop.train_re --distant_mode pretrain --distant_limit 5000 \
  --distant_epochs 1 --epochs 0 --use_pu_loss --na_weight 0.5 \
  --run_name atlop_pu_on --save_model --seed 66
```

### 3-1. dev 성능 (공통 스코어러)

| | 일반 ATLoss | PUATLoss | 차이 |
|---|---|---|---|
| dev F1 | 44.63 | **46.10** | +1.47 |
| Ign F1 | 42.59 | **43.36** | +0.77 |
| Precision | **54.03** | 44.25 | -9.78 |
| Recall | 38.02 | **48.11** | **+10.09** |

방향이 정확히 메커니즘과 일치: 임계값이 내려가며 눌려있던 정답들이 살아나 recall이 크게 뛰고
(precision은 일부 희생), 종합 F1/Ign F1 모두 개선.

### 3-2. 임계값 아래 눌린 정답 구출 통계 (dev 정답 12,275개 전수)

각 정답 (pair, r)에 대해 logit_r과 TH logit의 대소를 두 모델에서 비교:

| 변화 | 개수 | 비율 |
|---|---|---|
| **구출** (표준: TH에 눌림 → PU: 통과) | **1,415** | **11.5%** |
| 새로 잃음 (표준: 통과 → PU: 눌림) | 177 | 1.4% |
| 둘 다 통과 | 4,490 | 36.6% |
| 둘 다 눌림 | 6,193 | 50.5% |
| **순이득** | **+1,238** | - |

### 3-3. 예시 pair의 logit 변화 (Skai TV → Piraeus, 정답 P159)

| | TH logit | P159 logit | P159 − TH | 예측 |
|---|---|---|---|---|
| 일반 ATLoss | 7.109 | 5.670 | **−1.439** | 실패 (크게 눌림) |
| PUATLoss | 6.877 | 6.778 | **−0.098** | 실패 (경계 직전) |

이 pair는 격차의 93%가 회복됐지만 아직 경계를 못 넘었다 — 5,000개·1 epoch의 짧은 학습 한계로
해석(트랙 1 dk 모델의 2만개 실험에서는 동일 메커니즘으로 완전히 뒤집힌 pair들이 다수 확인됨).
단건 예시보다 3-2의 전수 통계(1,415개 구출)가 본질적 증거.

### 3-4. na_weight sweep (동일 조건: distant 5,000개, 1 epoch, seed 66, stage-1만)

| na_weight | dev F1 | Ign F1 | Precision | Recall |
|---|---|---|---|---|
| 1.0 (= 일반 ATLoss) | 44.63 | 42.59 | 54.03 | 38.02 |
| **0.7** | **47.56** | **45.07** | 49.43 | 45.82 |
| 0.5 | 46.10 | 43.36 | 44.25 | 48.11 |
| 0.3 | 46.36 | 43.37 | 38.68 | 57.84 |

- 예상과 달리 단조 관계가 아님: 낮출수록 recall은 계속 오르지만(38→46→48→58) precision 하락이
  더 가팔라져서, F1 최적점은 **0.7** (F1 +2.93, Ign +2.48 vs 일반 ATLoss — 0.5/0.3보다도 우위).
- 해석: FN이 62%라도 Na 라벨의 38%는 진짜 음성이므로, TH-랭킹 신호를 절반 이하로 꺾으면(0.5↓)
  진짜 음성에서 오는 정상 학습 신호까지 손상 → 과하게 낮아진 임계값이 precision을 무너뜨림.
  0.7은 "의심은 하되 신호는 유지"하는 지점.
- **Colab 풀 스케일 A/B는 na_weight=0.7로 진행 권장** (stage-1 F1/Ign F1 모두 최고 + P/R 균형이
  가장 좋아 annotated fine-tune에서 precision 회복 부담도 가장 적음).

## 4. 풀 파이프라인 검증 (Colab A100, `colab_pu07_reproduce.ipynb`)

distant 20,000개 PU(0.7) 프리트레인 → annotated 15 epoch 파인튜닝. 비교군은 팀원 baseline
`atlop`(완전 동일 레시피·seed 66, PU만 꺼짐 — `results/comparison.md`):

| | dev F1 | Ign F1 | P | R |
|---|---|---|---|---|
| baseline (PU off) | 61.71 | 59.86 | 66.08 | 57.89 |
| **PU na_weight=0.7** | **62.06** | **60.16** | 65.23 | 59.18 |
| 차이 | +0.35 | +0.30 | −0.85 | +1.29 |

- stage-1(distant 직후) 시점: dev F1 52.92 / Ign 50.23 — 로컬 5천개(47.56)보다 크게 높아
  데이터 규모 효과 확인.
- **결론**: stage-1의 큰 이득(+2.93, 5천개 기준)이 annotated 파인튜닝을 거치며 +0.35로 줄지만
  방향은 유지 — dk 모델 실험 2(+4.6→+0.24)와 동일한 패턴 재현. recall 우위(+1.29)가 최종까지
  남는 것이 PU 메커니즘("눌린 정답 구출")의 시그니처.
- **주의**: +0.35는 PRD §5의 seed 변동폭(±1점) 이내 — single seed로 "확정 우위" 주장은 불가.
  확정하려면 2 seed 평균 필요. 참고로 62.06/60.16은 현재 팀 비교표 1위(트랙1 dk 61.77/59.98,
  ATLOP 논문 61.09/59.22).

## 5. 한계 및 다음 단계

- **한계**: single seed(66); 스크리닝(na_weight sweep)은 distant 5,000개·stage-1 기준이라 풀
  파이프라인에서의 na_weight 최적값은 미검증(0.7만 풀 런 완료).
- **다음 단계**: ① seed 1개 추가로 2-seed 평균 확정, ② 여력이 되면 TTM-RE의 클래스 prior 기반
  risk estimator(정식 nnPU)로 확장 검토.
