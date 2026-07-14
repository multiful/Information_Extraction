# dk EGAT 모델 — 제안 아키텍처 파이프라인 문서

> **최종 업데이트**: 2026-07-14 (Gated Fusion / Bilinear Classifier / abs-diff 구현 + 학습 전략
> 플래그 추가, 전부 토글식 — A/B 검증 전까지 기본값은 변경 없음):
>
> **✅ Gated Fusion A/B 결론 (distant 3,000문서 스크리닝, 2026-07-14 15:29 완료) — 채택 확정**:
>
> | 설정 | dev F1 | Ign F1 | Precision | Recall |
> |---|---|---|---|---|
> | baseline (JK on, gate 없음) | 25.16 | 24.45 | 59.50 | 15.95 |
> | **Gated Fusion** | **27.38** | **26.47** | 56.50 | 18.07 |
>
> F1 +2.2 / Ign F1 +2.0 (precision 소폭 하락하지만 recall 개선폭이 커 순이익). 앞서 나온 JK
> on/off 스크리닝 결과(F1 26.77 vs 25.32)보다도 둘 다 앞섬 — Gated Fusion이 JK의 결합 단계를
> 코드상 완전히 대체하므로(`use_gated_fusion`이 `use_jk`보다 우선 적용, 둘을 스택하지 않음)
> **JK on/off 논쟁 자체가 무의미해짐**. **다음 전체 Colab 실행부터 `--use_gated_fusion` 필수
> 추가로 결론.**
>
> **⭐ Gated Fusion** (`--use_gated_fusion`, argparse 기본값은 여전히 off — `--use_pu_loss`와
> 같은 패턴으로, 학습 커맨드에 명시적으로 플래그를 추가하는 방식 유지): GAT 출력과 원본(pre-GAT)
> 엔티티 임베딩을 학습 가능한 per-dimension sigmoid 게이트로 blend —
> `gate=sigmoid(W[e_orig;e_gat])`, `g=gate*e_gat+(1-gate)*e_orig`. GAT 메시지패싱이 강한 엔티티
> 표현을 이웃과 평균내며 희석시킬 수 있다는 우려에 대응, JK의 max 결합 대신 학습된 결합을 씀.
>
> **✅ Bilinear Classifier A/B 결론 (distant 3,000문서 스크리닝, 2026-07-14 15:59 완료) — 채택
> 확정, 지금까지 중 가장 큰 개선폭**:
>
> | 설정 | dev F1 | Ign F1 | Precision | Recall |
> |---|---|---|---|---|
> | baseline (기존 concat+`g_h*g_t`+MLP) | 25.12 | 24.42 | 59.44 | 15.93 |
> | **Bilinear Classifier** | **28.79** | **27.79** | 57.25 | 19.23 |
>
> F1 +3.67 / Ign F1 +3.37 (Gated Fusion의 +2.2보다 큼). Gated Fusion(엔티티 임베딩 결합 단계)과
> Bilinear Classifier(분류기 헤드)는 코드상 서로 다른 지점을 건드리는 독립적 변경이라 함께 스택
> 가능. **다음 전체 Colab 실행부터 `--use_bilinear_classifier` 필수 추가로 결론.**
>
> **⭐ ATLOP식 Grouped Bilinear Classifier** (`--use_bilinear_classifier`, argparse 기본값은
> 여전히 off — `--use_pu_loss`와 같은 패턴): 기존 concat+`g_h*g_t`+MLP 분류 경로를 대체 —
> head/tail extractor(Linear+tanh) 후 block-wise outer product(12블록×64) →
> Linear(768*64→97). 학습 곡선이 정상(loss 꾸준히 감소, 불안정 없음)이라 GAT 자체보다 분류기
> 용량이 병목일 가능성에 대응. interaction term과 중복이라 대체(스택 안 함).
>
> **✅ abs-diff A/B 결론 (distant 3,000문서 스크리닝, 2026-07-14 16:27 완료) — 단독으로는 이기지만
> 최종 조합에서는 무의미**:
>
> | 설정 | dev F1 | Ign F1 | Precision | Recall |
> |---|---|---|---|---|
> | baseline (abs-diff 없음) | 25.20 | 24.49 | 59.51 | 15.98 |
> | **abs-diff 추가** | **27.92** | **26.95** | 57.77 | 18.41 |
>
> F1 +2.72 / Ign F1 +2.46 — MLP+interaction 경로 단독으로 보면 확실한 개선. **하지만**
> `use_bilinear_classifier`가 켜지면 `pair_proj` MLP 경로 자체를 안 타므로 `use_abs_diff`는
> 완전히 무시됨(코드: `forward()`의 `if self.use_bilinear_classifier: ... else: (abs-diff는 여기
> 안에서만 적용)`). 이미 Bilinear Classifier(F1 28.79)가 abs-diff(27.92)보다도 앞서 채택
> 확정됐으므로, **JK가 Gated Fusion으로 무의미해진 것과 같은 패턴 — 최종 추천 조합에서 abs-diff는
> 넣으나 안 넣으나 결과가 같음(bilinear 경로에서 그냥 무시되는 죽은 플래그)**. `--use_abs_diff`는
> 넣지 않음.
>
> **⭐ Pair Representation에 `abs(g_h-g_t)` 추가** (`--use_abs_diff`, 기본 off): 기존
> `[g_h;g_t;g_h*g_t;c_ht]`에 InferSent 스타일 절대차 항을 추가 — 곱 항이 못 잡는 "head/tail 특징
> 크기 차이" 신호. bilinear classifier 사용 시엔 무시(그 경로는 pair_proj를 안 씀).
>
> ---
>
> **🏁 A/B 큐 최종 결론 — 다음 전체 Colab 실행(distant_limit=20000, epochs=15)에 반영할 조합**:
> `--use_gated_fusion --use_bilinear_classifier` (abs-diff/JK는 위 이유로 미포함).
> 학습 전략 플래그(`--lr2`/`--layerwise_lr_decay`/`--freeze_encoder_epochs`/
> `--evidence_start_epoch`/`--early_stop_patience`)는 CPU 스크리닝으로 검증 불가능한 항목이라
> 이번 실행엔 미포함, 사용자 확인 후 결정.
>
> **학습 전략 플래그 (신규, 기본값은 전부 현재 동작 그대로 — off/미변경)**: `--lr2`(stage2 전용
> learning rate, 미지정 시 `--lr`과 동일), `--layerwise_lr_decay`(BERT 층별 LR 감쇠, 1.0=비활성),
> `--freeze_encoder_epochs`(스테이지 시작 N epoch 동안 인코더 동결), `--evidence_start_epoch`
> (evidence loss를 N epoch부터 curriculum 방식으로 활성화), `--early_stop_patience`(N epoch
> 연속 dev F1 개선 없으면 조기 종료). CPU smoke test(4문서, 전 플래그 동시 적용)로 크래시 없음만
> 확인 — 이 항목들은 여러 epoch에 걸친 수렴 양상이 핵심이라 distant-only 1epoch CPU 스크리닝으로는
> 의미있는 A/B가 어려워, 다음 전체 Colab 실행(15 epoch)에서 실측 예정.
>
> 이전 (Jump Knowledge 추가 + `--epochs 0` 버그 수정):
> ④ **Jump Knowledge**: GAT 마지막 층 출력만 쓰던 것을 `max(입력 임베딩, layer1 출력, layer2
> 출력)`(element-wise)로 바꿈 — 분류기가 0/1/2-hop 정보 중 필요한 걸 노드마다 가져다 쓸 수 있게
> 함. **새 파라미터 없음**(concat+Linear가 아니라 max 선택) — 같은 날 이미 이종 그래프/interaction
> term으로 파라미터가 늘어난 상태라, "도움이 되는지 안 되는지"를 추가 용량 효과와 분리해서 보기
> 위함. `model.py`의 `forward()`만 수정.
>
> **버그 수정**: `--epochs 0`(distant만 빠르게 스크리닝하는 용도, na_weight/gat_heads sweep에서
> 사용)일 때 stage 2가 `(None, None)`을 반환해 마지막에 `len(preds)`에서 `TypeError`로 죽던 버그
> 발견(실측: gat_heads A/B 테스트 중 발견, exit code 1). stage 2를 `if args.epochs > 0`으로
> 감싸고 스킵 시 stage-1 결과로 폴백하도록 수정 — `Scripts/atlop/train_re.py`엔 이미 있던 가드를
> `train_gat.py`엔 빠뜨렸던 것.
>
> **GAT heads A/B (참고, 반영 안 함)**: distant 3,000개 스크리닝 결과 heads=4가 heads=8보다 우세
> (F1 26.77 vs 25.17, Ign 25.81 vs 24.45) — heads=8은 recall만 더 떨어짐. **기본값 4 유지**로 결론.
>
> 이전 (그래프/pair representation 고도화): ATLoss 교체가 실측으로
> 검증됨(distant 프리트레인만 dev F1 46.58/Ign 44.21 — 20k 기준 참고치 43.15~47.79 범위 안,
> BCE 때 24.77에서 회복 확인) → **보류해뒀던 고도화 2건을 반영**.
> ① **Entity-Sentence Heterogeneous Graph**: 노드에 문장(Sentence)을 추가, entity-entity
> edge에 더해 entity-sentence("appears in") edge를 신설. 직접/같은 문장/멘션겹침으로 안 이어진
> 두 엔티티도 공통으로 등장하는 문장 노드를 거쳐 2-hop 만에 정보 교환 가능해짐 (예: Steve Jobs
> -[S1]- Apple -[S2]- California). GAT 통과 후 엔티티 행만 잘라내 pair 구성에 사용, 문장 노드
> 자체는 메시지 전달 매개체로만 씀. `preprocess_gat.py`(그래프 생성)/`model.py`(GAT 입력 확장) 수정.
> ② **Pair Representation에 element-wise interaction 추가**: `[g_h ‖ g_t ‖ c_ht]`(2304-d)에
> `g_h*g_t`(768-d) 곱 항을 더해 `[g_h ‖ g_t ‖ g_h*g_t ‖ c_ht]`(3072-d)로 확장, `pair_proj`도
> Linear(3072→768)로 변경. 관계 분류에서 곱 interaction이 concat만으로는 못 잡는 신호를 준다는
> 통상적 근거(ATLOP의 grouped bilinear와 같은 취지, 여기선 명시적 단일 곱 항으로 간소화).
> ③ **train_gat.py에 best-epoch 체크포인트 저장 추가**: 매 epoch dev F1을 비교해 갱신될 때마다
> `{run_name}_best.pt`(및 stage1은 `_stage1_best.pt`)로 저장 — 마지막 epoch이 우연히 저점일 때
> (실측: epoch 6 F1 59.85 → epoch 7 59.57 하락 관측) 최고 기록을 놓치지 않기 위함. 최종 epoch
> 체크포인트(`{run_name}.pt`, baseline과 동일 비교 기준 유지용)는 그대로 별도 저장.
> CPU smoke test(8/20문서) 재검증 통과. Colab 세션은 이 변경들과 무관하게 계속 진행 중이었음
> (로컬 편집이 이미 클론된 세션에 영향 없음) — 다음 Colab 실행부터 반영됨.
>
> 2026-07-14 (손실함수 교체): Colab 실측에서 BCE+threshold sweep이 dev F1
> 24.77(distant 프리트레인만)로, 같은 20k distant subset 기준 RoBERTa+LCP+ATLoss의
> 43.15(`Scripts/models/EXPERIMENTS.md` 실험 2)보다 낮게 나옴 — train_loss가 0.6→0.02로
> 비정상적으로 빨리 떨어진 것과 함께, BCE가 DocRED 97% NA 불균형을 학습 중엔 구조적으로 다루지
> 않고 사후 threshold로만 보정하는 데서 온 문제로 진단. **Adaptive Thresholding(ATLoss/PUATLoss,
> `Scripts/atlop/losses.py` 재사용)으로 교체** — distant 단계는 PUATLoss(na_weight=0.7, 기존
> sweep 검증값), annotated는 자동으로 일반 ATLoss (`Scripts/atlop/train_re.py`와 동일 패턴).
> threshold sweep 로직은 전부 제거, 예측은 `ATLoss.get_label`(페어별 학습된 TH 클래스 비교)로
> 결정. 검토했던 Heterogeneous(Entity+Sentence) 그래프/Meta-path attention/Curriculum PU-weight는
> 의도적으로 보류 — 손실함수 하나만 바꿔 회복되는지 먼저 확인 후 별도 실험으로 진행 예정.
>
> 2026-07-14: 제안 아키텍처(BERT + ATLOP LCP + 2-Layer Edge-featured GAT) 확정 및 구현
> (`preprocess_gat.py`/`model.py`/`train_gat.py`), CPU smoke test 통과, Colab A100용 노트북
> (`colab_gat_a100.ipynb`) 추가. 배경: DREEAM+GAT+GREP+PUATLoss 전부 결합 시 GAT 단독보다
> 성능이 낮아 GAT 고도화에 집중하기로 결정. 학습 epoch 기본값을 `Scripts/atlop` baseline
> (distant_epochs=1, epochs=15)과 정확히 일치하도록 수정 — 통제 비교를 위해 아키텍처만 변수로 남김.

## 확정 파이프라인

```text
                Input Document (DocRED + Entity Mention)
                              │
                              ▼
                      BERT-base Encoder
        WordPiece / [CLS]+Tokens+[SEP] / 768-d / 12 layers
        (512 초과 문서는 atlop/long_input.process_long_input이
         겹치는 두 윈도로 분할·평균 — 정보 손실 없음)
                              │
                              ▼
              ATLOP Localized Context Pooling
        ① Mention: span 토큰 임베딩 평균
        ② Entity: 멘션 임베딩 평균 (768-d)
        ③ Sentence: 문장 토큰 span 임베딩 평균 (768-d, 엔티티와 동일 풀링 형태)
        ④ Pair context: head·tail attention 곱 → 문맥 벡터 c_ht
                              │
                              ▼
    2-Layer Edge-featured GAT (노드 = Entity + Sentence, 이종 그래프)
        Edge Embedding 32-d = [엣지카테고리 8 ; 문장거리 8 ;
                                head타입 8 ; tail타입 8]
        엣지: entity-entity(같은문장/멘션겹침) + entity-sentence("등장함")
              — sentence-sentence 엣지는 없음, 두 entity는 공통 문장 노드를
              거쳐 2-hop으로 연결 (예: Steve Jobs -[S1]- Apple -[S2]- California)
        α_ij = softmax( LeakyReLU( aᵀ[Wh_i ‖ Wh_j ‖ e_ij] ) )
        Layer1: 투영→edge-aware 멀티헤드 attention→집계
                →residual→LayerNorm→GELU→Dropout
        Layer2: 동일 (GELU 없이 LayerNorm까지)
                              │
                              ▼
           GAT 출력에서 Entity 행만 추출 (Sentence 노드는 폐기)
                              │
                              ▼
     Pair Representation = Linear([g_h ; g_t ; g_h*g_t ; c_ht])
                        (3072 → 768)
                              │
                              ▼
                    Relation Classifier
        LayerNorm → Linear(768→768) → GELU → Dropout(0.1)
                 → Linear(768→97)
                              │
                              ▼
           Adaptive Thresholding (TH 클래스, 페어별 학습)
                              │
                              ▼
        Loss = ATLoss/PUATLoss(na_weight=0.7, distant만) + 0.2 × Evidence Contrastive
```

## 구현 파일

| 파일 | 역할 |
|---|---|
| `preprocess_gat.py` | atlop의 `*` 마커 방식 확장: 문장 토큰 span, 엔티티+문장 이종 그래프(edge 카테고리/거리/인접행렬), pair별 evidence 집합 |
| `model.py` | `DocREGATModel` — 인코더/엔티티+문장 풀링/LCP/이종 EGAT 2층/interaction pair repr/분류기/ATLoss(주입형)+InfoNCE |
| `train_gat.py` | 2단계 학습(distant PUATLoss→annotated ATLoss), best-epoch 체크포인트, `ATLoss.get_label` 디코드, 공통 스코어러 평가 |
| `colab_gat_a100.ipynb` | Colab A100 원클릭 실행 노트북 |

## 그래프 구성 (구현 확정값)

- **노드** = 엔티티(0..n_ent-1) + 문장(n_ent..n_ent+n_sent-1), 노드 피처 = 엔티티/문장 각각 토큰 평균 풀링(768, 같은 형태)
- **Edge 카테고리** (Embedding(5, 8)): `4`=self-loop, `3`=entity-sentence("등장함"), `2`=entity-entity 멘션 span 겹침, `1`=entity-entity 같은 문장 공출현, `0`=그 외
- **문장 거리** (Embedding(6, 8)): entity-entity 쌍의 최소 문장거리 버킷 (0,1,2,3,4,5+); entity-sentence/self 엣지는 0
- **노드 타입** (Embedding(8, 8) × head/tail): PER/ORG/LOC/TIME/NUM/MISC/unk(엔티티) + 문장 전용 공유 타입 1개
- **Sparse 인접행렬**: entity-entity는 기존 기준(같은 문장/멘션겹침/거리≤2) 그대로, entity-sentence는 해당 문장에 멘션이 있으면 연결, sentence-sentence 엣지는 없음(엔티티가 공유 문장 노드를 거쳐 2-hop 연결) + self-loop 상시

## 스펙 대비 구현 해석

1. **GAT 위치**: **엔티티+문장 노드 임베딩 → 이종 GAT → graph-enhanced E′로 pair 구성** — 최종 확정
   스펙(2026-07-14 "Final Proposed Architecture")과 일치. pair는 GAT 통과 후 엔티티 행만 뽑은
   g_h, g_t에 인코더 attention 기반 LCP 문맥 c_ht를 concat (LCP의 attention은 GAT가 아니라
   인코더에서 나오므로 원 ATLOP 방식 그대로 인코더 마지막 층 attention 사용). 문장 노드는
   메시지 전달에만 관여하고 GAT 출력 후 버려짐 — pair representation엔 등장하지 않음.
2. **분류기 입력 768-d**: [g_h; g_t; g_h*g_t; c_ht]는 3072-d이므로 `pair_proj`(Linear 3072→768)로
   투영 후 스펙의 2-layer MLP(LayerNorm→768→768→97)에 투입. `g_h*g_t`(element-wise interaction)는
   나중에 추가된 항 — concat만으로는 못 잡는 head/tail 특징 간 곱셈적 상호작용을 명시적으로 준다
   (ATLOP의 grouped bilinear 분류기가 하는 일과 같은 취지를 단일 곱 항으로 간소화).
3. **Evidence Contrastive Loss**: InfoNCE로 구현 — pair의 LCP 문맥 c_ht를 anchor로, 정답 evidence 문장 임베딩
   (토큰 평균)을 positive, 문서 내 나머지 문장을 negative로 (τ=0.1). 스펙의 "random evidence/masking"
   negative보다 강한 표준형. **train_distant는 evidence가 없어 이 항이 자동으로 비활성** (가중치 0.2는
   annotated 단계에서만 실질 작동).
4. **NA 불균형 처리** (PRD §2 필수): 최초 BCE+sigmoid+threshold sweep 버전은 실측 dev F1
   24.77(distant 프리트레인만, 같은 20k 기준 RoBERTa+ATLoss 43.15보다 낮음)로 실패 —
   BCE는 97% NA 불균형을 학습 중엔 안 다루고 사후 threshold로만 보정해서, positive 클래스
   logit이 충분히 학습되지 않은 것으로 진단. **Adaptive Thresholding으로 교체**해 이 처리를
   손실함수 자체에 내장 (`Scripts/atlop/losses.py`의 `ATLoss`/`PUATLoss` 재사용, distant만
   PUATLoss na_weight=0.7). 전역 threshold 없이 페어마다 학습된 TH 클래스와 비교해 결정.

## 학습 설정

AdamW / lr 2e-5 / weight decay 0.01 / dropout 0.1 / warmup 6% / grad clip 1.0 — 스펙 그대로.

**epoch 수는 `Scripts/atlop` baseline과 정확히 동일하게 맞춤**: distant 20,000개 × **1 epoch**
→ annotated × **15 epoch**, seed 66. 원 스펙은 "distant 2~3ep / annotated 12~15ep"라는 범위를
제안했지만, baseline(`atlop`, `atlop_full_pu07`)과 epoch 수가 다르면 성능 차이가 "GAT 때문"인지
"학습을 더/덜 해서"인지 구분이 안 되므로, 통제 비교를 위해 baseline과 완전히 동일한 스케줄로
확정했다 (distant_limit/distant_epochs/epochs/seed 전부 일치, 차이는 아키텍처뿐).

## 실행

```bash
# CPU 정합성 검증 (통과 확인됨, 2026-07-14 -- 이종 그래프+interaction term 반영 후 재검증 완료)
python -m Scripts.dk_gat.train_gat --limit_docs 8 --epochs 2 --distant_epochs 1 --use_pu_loss --na_weight 0.7

# 풀 학습 (Colab A100 권장 — colab_gat_a100.ipynb 사용)
python -m Scripts.dk_gat.train_gat --distant_limit 20000 --distant_epochs 1 \
    --epochs 15 --use_pu_loss --na_weight 0.7 --run_name dk_gat --save_model --seed 66
```

`--save_model` 사용 시 저장물: `{run_name}_stage1.pt`(distant 마지막 epoch), `{run_name}_stage1_best.pt`
(distant 중 dev F1 최고), `{run_name}.pt`(annotated 마지막 epoch, baseline과 비교 기준),
`{run_name}_best.pt`(annotated 중 dev F1 최고) + 그 예측(`_best_dev_predictions.json`). 매 epoch
dev F1이 단조 증가하지 않으므로(실측: epoch 6 F1 59.85 → epoch 7 59.57) best 체크포인트가 최종
epoch보다 나을 수 있음 — 최종 보고 시 둘 다 확인 권장.

## 비교 기준

`results/comparison.md`: ATLOP baseline 61.71/59.86, ATLOP+PU(0.7) 62.06/60.16, 트랙1 61.77/59.98.
동일 스코어러(F1/Ign F1)·동일 seed(66)로 비교. 예측 포맷은 팀 공통 `[{"title","h_idx","t_idx","r"}]`.
