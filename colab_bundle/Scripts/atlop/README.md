# ATLOP baseline (PRD 트랙 2)

> **최종 업데이트**: 2026-07-12: 학습 순서를 원 논문 방식으로 전환 — `--distant_mode` 옵션 추가. 기본값 `pretrain`(① `train_distant` 사전학습 ② `train_annotated` fine-tune ③ `dev` 평가, ATLOP 원 논문 순서). 기존 팀 레시피(annotated 학습 → teacher denoise → distant 학습)는 `--distant_mode denoise`로 유지, `--distant_mode none`은 annotated 단독.
>
> 2026-07-12: 학습 순서 교정 — ① `train_annotated`로 먼저 학습(teacher) ② 그 모델로 `train_distant`의 노이즈 라벨을 걸러낸 뒤 이어서 학습 ③ `dev`로 최종 평가. 대용량 distant를 위해 라벨 sparse(positive-id) 저장. CPU에서 2단계 미니 학습 경로 검증 완료.
>
> 2026-07-10: `Scripts/atlop/`에 ATLOP 재구현(logsumexp 엔티티 풀링 + Localized Context Pooling + Adaptive Thresholding Loss) 및 `Scripts/eval/scorer.py` 공통 스코어러(F1/Ign F1) 추가. 풀 학습은 GPU/Colab.

DocRED 문서 단위 관계 추출의 비교 기준선인 **ATLOP**(Zhou et al., AAAI 2021)을 재현한다. 팀 공통 로더(`data.docred_dataset.DocREDataset`)와 공통 `rel2id`(Na=0 + 96 P-code)를 그대로 입력으로 쓰고, 예측을 팀 공통 포맷으로 내보내 공통 스코어러로 채점한다 — 트랙 1(우리 모델들)과 dev F1/Ign F1을 직접 비교하기 위함.

## 학습 방식: `--distant_mode`로 순서 선택 (기본 = 원 논문 순서)

**`pretrain` (기본, ATLOP 원 논문 순서)**

1. **Stage 1 — distant 사전학습**: `train_distant`(101,873문서, distant supervision이라 노이즈 많음)로 먼저 사전학습한다.
2. **Stage 2 — annotated fine-tune**: 사람이 라벨링한 깨끗한 `train_annotated`(3,053문서)로 이어서 fine-tune한다. 최종 예측/체크포인트는 이 단계 산출물.
3. **평가**: 매 epoch 및 최종적으로 `dev`에서 F1/Ign F1 산출.

**`denoise` (팀 내부 비교용, 이전 기본값)**

1. `train_annotated`로 먼저 학습 — 이 모델이 **teacher** 역할을 겸한다.
2. `train_distant`의 각 양성 라벨을 teacher로 검사해서, **모델이 adaptive threshold 위로 동의하는 라벨만 남기고 나머지는 노이즈로 제거**한 뒤(해당 쌍은 Na로 강등) 걸러진 distant로 이어서 학습한다. 콘솔에 `kept/dropped` 라벨 수가 출력된다.

**`none`**: annotated 단독 학습. 어느 모드든 dev 평가와 Ign-F1 fact 필터는 항상 `dev` / `train_annotated` 기준.

> `pretrain`이 원 논문의 distant 활용 순서(distant 사전학습 → annotated fine-tune)와 같아 논문 수치와의 비교 기준으로 적합하다. `denoise`/`none` 수치는 논문과 직접 비교 대상이 아닌 팀 내부 비교용.

> 출처/라이선스: `wzhouad/ATLOP`은 라이선스 미표기(`license: null`)라 코드를 그대로 가져오지 않고 핵심 로직만 출처 주석과 함께 **재구현**했다. 스코어러만 MIT 라이선스인 `thunlp/DocRED`의 공식 `evaluation.py`를 포팅.

## 구성 요소 (ATLOP 3대 핵심)

| 파일 | 역할 |
|---|---|
| `preprocess.py` | 멘션 앞뒤에 `*` 마커 삽입 → 각 마커의 subword 위치 기록, 엔티티 쌍(hts)·97차원 멀티핫 라벨 생성 |
| `re_model.py` | **① logsumexp 엔티티 풀링**(멘션 `*` 마커 hidden state 결합) **② Localized Context Pooling**(head·tail attention 곱으로 관련 문맥 집중) **③ grouped bilinear** 분류기 |
| `losses.py` | **Adaptive Thresholding Loss (ATLoss)** — 클래스 0(=Na/TH)을 학습형 임계값으로 사용, 97% NA 불균형을 구조적으로 처리 |
| `long_input.py` | 512 토큰 초과 문서를 겹치는 두 윈도로 분할·평균 (`process_long_input`). 모든 DocRED 문서는 1024 이내 |
| `train_re.py` | 학습/평가 진입점. 인코더/분류기 차등 LR, 공통 포맷 예측 저장, 공통 스코어러 호출 |
| `smoke_test.py` | 다운로드 없이 랜덤 소형 BERT로 파이프라인 전체 정합성 검증(CPU) |
| `../eval/scorer.py` | 공통 스코어러: F1, **Ign F1**(train_annotated에 이미 등장한 fact 제외) |

## 실행법

프로젝트 루트에서 실행. `-m`(모듈) 형태로 실행해야 `data`/`Scripts` import가 맞는다.

```bash
# 0) CPU 정합성 검증 (다운로드 X, 랜덤 가중치 — 정확도 아님)
python -m Scripts.atlop.smoke_test

# 1) CPU 미니 학습 (2단계 경로 검증; --limit_docs 는 distant 도 함께 제한)
python -m Scripts.atlop.train_re --limit_docs 8 --epochs 1 --distant_epochs 1 --train_batch_size 2

# 2) 풀 학습, 원 논문 순서(기본): distant 사전학습 1에폭 → annotated fine-tune 30에폭 (GPU/Colab 권장)
#    distant 는 우선 subset(예: 20000)으로 시작 권장(전체 10만은 RAM/시간 큼)
python -m Scripts.atlop.train_re --epochs 30 --distant_limit 20000 --distant_epochs 1 --run_name atlop --save_model

# 2') 팀 레시피(annotated 학습 → teacher denoise → distant 학습)로 돌리려면
python -m Scripts.atlop.train_re --distant_mode denoise --epochs 30 --distant_limit 20000 --run_name atlop_denoise

# 2'') annotated 단독 학습만 하고 싶으면
python -m Scripts.atlop.train_re --distant_mode none --epochs 30 --run_name atlop_annot_only
```

### Colab에서 풀 학습

```python
# Colab 셀
!git clone https://github.com/multiful/Information_Extraction.git
%cd Information_Extraction
!git checkout dh
!pip install -q transformers==4.57.6 accelerate
# docred_data/data/*.json 이 없으면 docred_data/ 압축 해제(README 참고) 먼저 수행
!python -m Scripts.atlop.train_re --epochs 30 --distant_limit 20000 --distant_epochs 1 --run_name atlop --save_model
```

Colab은 GPU가 자동 감지되어 `--device cuda`로 잡힌다(런타임 > GPU 선택). 기본(`pretrain`) 기준 Stage 1(distant subset 2만 × 1 epoch)이 T4 대략 1~2시간, Stage 2(annotated 30 epoch)가 추가로 수 시간 붙는다.

> **메모리 주의**: distant 전처리 feature는 전부 RAM에 올린다. `--distant_limit 0`(전체 10만)은 Colab 기본 RAM(≈12GB)에서 위험할 수 있으니 `20000`~`30000`부터 시작하고, 여유를 보며 늘릴 것. (라벨은 sparse 저장이라 예전보다 훨씬 가볍지만 문서 수가 많으면 여전히 큼.)

## 주요 하이퍼파라미터 (ATLOP DocRED 기본값)

| 인자 | 기본 | 설명 |
|---|---|---|
| `--model_name_or_path` | `bert-base-cased` | 인코더 (roberta-base 등으로 교체 가능) |
| `--epochs` | 30 | annotated 학습/fine-tune epoch |
| `--train_batch_size` | 4 | 문서 단위 배치 |
| `--distant_mode` | `pretrain` | distant 활용 방식: `pretrain`(원 논문 순서) / `denoise`(팀 레시피) / `none`(annotated 단독) |
| `--distant_limit` | 0(전체) | distant 문서 수 제한 (Colab은 20000~30000 권장) |
| `--distant_epochs` | 1 | distant 학습 epoch (보통 1 pass) |
| `--distant_batch_size` | 4 | distant 학습 배치 |
| `--encoder_lr` / `--classifier_lr` | 5e-5 / 1e-4 | 인코더/분류기 차등 LR (두 스테이지 공통) |
| `--warmup_ratio` | 0.06 | linear warmup 비율 (스테이지별 스케줄러 각각 적용) |
| `--emb_size` / `--block_size` | 768 / 64 | grouped bilinear 차원 |
| `--seed` | 66 | dev F1은 seed마다 ±1점 흔들림(PRD §5) — 최종 비교는 2 seed 평균 권장 |
| `--limit_docs` | 0(전체) | 빠른 실행용 문서 수 제한 (train/dev/**distant 모두** 제한) |
| `--train_split` | `train_annotated` | Stage 1 학습 스플릿 |

## 출력

- `results/atlop_dev_predictions.json` — 공통 포맷 `[{"title","h_idx","t_idx","r"}]` (r=P-code, gitignore)
- 콘솔 로그에 epoch별 `dev_F1 / Ign_F1 / P / R`
- `--save_model` 지정 시 `results/atlop.pt` 체크포인트 저장

최종 비교표는 PRD §6대로 `results/comparison.md`(git 추적)에 기록.
