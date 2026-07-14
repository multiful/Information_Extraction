# GREP (ATLOP 확장 실험)

> **최종 업데이트**: 2026-07-13: GREP(Zhang, Yan & Cheng, "Document-Level Relation Extraction with
> Global Relations and Entity Pair Reasoning," ACL Findings 2025) 재구현 초안 작성. `Scripts/grep/`
> 신규 패키지(`re_model.py`/`losses.py`/`train_grep.py`/`smoke_test.py`) 추가, `Scripts/atlop/preprocess.py`에
> `sent_pos`/`evidence`/`doc_rel_labels` 필드 추가(additive, 기존 ATLOP 파이프라인 영향 없음 — 확장 후
> `Scripts.atlop.smoke_test` 재통과 확인). CPU 스모크 테스트(`Scripts.grep.smoke_test`) 통과, 실제 dev
> F1/Ign F1 학습은 미실시(GPU/Colab 필요).

`Scripts/atlop/`(ATLOP baseline, `results/comparison.md` 비교 기준: **dev F1 61.71 / Ign F1 59.86**)
위에 GREP 논문의 세 모듈을 얹은 실험. ATLOP의 문서 인코딩(마커 삽입 → logsumexp 엔티티 풀링 →
Localized Context Pooling)은 그대로 재사용하고, 그 위에 아래를 추가한다:

1. **Entity Pair Reasoning Graph** — 문서 내 모든 (head, tail) pair를 노드로 하는 그래프를 만들고
   (pair `(s,o)`→`(o,t)` edge, tail==head 매칭), 2-layer GAT+GCN으로 pair 간 정보를 교환한 뒤
   최종 분류에 사용
2. **Evidence Extraction Module** — Localized Context Pooling의 pair-token attention을 문장
   단위로 합산해 "이 pair의 관계가 어느 문장에 근거하는지" 분포를 학습(KL loss, gold evidence 문장
   지도)
3. **Global Relation Prediction Module** — 문서의 [CLS] 표현으로 "이 문서에 어떤 관계들이 존재하는가"를
   먼저 예측(BCE loss)해서 pair별 로짓에 더함(fusion)
4. **Inference Fusion Phase** — evidence loss로 학습한 모델과 evidence loss 없이 학습한 모델을
   각각 하나씩 준비해, 첫 모델이 예측한 핵심 문장만으로 구성한 pseudo-document를 두 번째 모델에 다시
   태워 두 추론 결과를 융합(`γ` 하이퍼파라미터, dev에서 튜닝)

## 논문 대비 구현 결정 (미명시 부분)

논문 수식이 명시하지 않은 부분은 아래처럼 구현했다(모두 `Scripts/grep/re_model.py` 상단 docstring에도
동일하게 기록):

- **그래프 노드 차원** (`f^(s,o)`, Eq 7): ATLOP의 grouped-bilinear 결과(`emb_size*block_size`)를
  `node_dim`(기본값 = 인코더 hidden size)으로 투영해서 사용
- **Eq 9 그래프 업데이트**: 논문 표기 그대로면 attention 대상이 이웃 `f_k`가 아니라 자기 자신 `f_j`라
  attention 가중치 합(=1)과 상쇄되어 의미가 없어짐 — 이웃 `f_k`를 집계하는 표준 GAT 방식으로 수정
- **그래프 attention head 수**: 논문 미기재, 4로 설정(레이어 간 Q/K 공유, `W^l`은 레이어별)
- **Evidence Extraction KL 부호** (Eq 16): 논문 표기 그대로면 KL의 부호가 반대라 "정답 문장에서
  멀어지도록" 학습됨 — 본문 설명("KL divergence를 최소화")에 맞춰 표준 `KL(v‖u)` 최소화로 정정
  (`Scripts/grep/losses.py`)
- **Inference Fusion pseudo-document 단위** (Sec 4.6): pair별이 아니라 **문서별**로 하나의
  pseudo-document를 구성(그 문서에서 non-NA로 예측된 모든 pair의 evidence 문장 합집합) — Eider/AA류
  선행 연구의 실무적 관행을 따름(pair마다 인코더를 다시 돌리면 비용이 지나치게 커짐)
- **Evidence 문장 선택 임계값**: pair의 문장별 attention 점수(`u`)가 그 pair의 평균 이상인 문장을 채택
  (논문 미기재)

## 실행법

프로젝트 루트에서 `-m`(모듈) 형태로 실행.

```bash
# 0) CPU 정합성 검증 (다운로드 X, 랜덤 가중치 — 정확도 아님)
python -m Scripts.grep.smoke_test

# 1) CPU 미니 학습 (배선 검증용, 실제 성능 아님)
python -m Scripts.grep.train_grep --limit_docs 8 --epochs 1

# 2) 풀 학습 (GPU/Colab 권장 — model_full/model_no_evi 두 개를 각각 30 epoch 학습하므로
#    ATLOP 풀 학습(1개 모델)보다 학습 시간이 대략 2배)
python -m Scripts.grep.train_grep --epochs 30 --run_name grep --save_model

# 2') Inference Fusion의 gamma를 dev F1 기준으로 작은 grid에서 탐색하고 싶으면
python -m Scripts.grep.train_grep --epochs 30 --run_name grep --sweep_gamma --save_model
```

### Colab에서 풀 학습

`Scripts/atlop/README.md`와 동일한 방식(레포 clone → 데이터 압축 해제 → 실행). GPU가 자동 감지되어
`--device cuda`로 잡힌다.

## 하이퍼파라미터 (GREP 논문 Table 4, BERT_base/DocRED 기준)

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--model_name_or_path` | `bert-base-cased` | 인코더 |
| `--epochs` | 30 | `model_full`/`model_no_evi` 각각의 학습 epoch |
| `--train_batch_size` | 4 | |
| `--encoder_lr` / `--classifier_lr` | 5e-5 / 1e-4 | |
| `--warmup_ratio` | 0.06 | |
| `--emb_size` / `--block_size` | 768 / 64 | grouped bilinear 차원(ATLOP과 동일) |
| `--node_dim` | 0(=인코더 hidden size) | 그래프 노드 피처 차원 |
| `--graph_layers` / `--graph_heads` | 2 / 4 | Entity Pair Reasoning Graph |
| `--alpha` | 0.1 | Global Relation Prediction loss 가중치 |
| `--beta` | 0.1 | Evidence Extraction loss 가중치 (`model_full`에서만 적용) |
| `--gamma` | 0.0 | Inference Fusion 오프셋(Eq 22); `--sweep_gamma`로 dev에서 자동 탐색 가능 |

## 출력

- `results/grep_dev_predictions.json` — 공통 포맷(gitignore), Inference Fusion까지 적용된 최종 예측
- 콘솔에 `model_full` 단독 dev F1/Ign F1과 fusion 이후 dev F1/Ign F1을 모두 출력(fusion의 실제 기여도
  확인용)
- `--save_model` 지정 시 `results/grep_full.pt` / `results/grep_no_evi.pt` 저장

## 비교

풀 학습 후 `results/comparison.md`의 ATLOP baseline(dev F1 61.71 / Ign F1 59.86)과 같은 스코어러
(`Scripts/eval/scorer.py`) 기준으로 비교해 같은 표에 행을 추가한다. 이 저장소는 CPU 전용이라(ATLOP도
동일) 실제 수치는 Colab/GPU 풀 학습 후에만 나온다 — 로컬 스모크 테스트는 배선 검증용일 뿐 성능 지표가
아니다.
