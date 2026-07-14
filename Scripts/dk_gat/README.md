# dk EGAT 모델 — 제안 아키텍처 파이프라인 문서

> **최종 업데이트**: 2026-07-14: 제안 아키텍처(BERT + ATLOP LCP + 2-Layer Edge-featured GAT +
> BCE + Evidence Contrastive Loss) 확정 및 구현(`preprocess_gat.py`/`model.py`/`train_gat.py`),
> CPU smoke test 통과, Colab A100용 노트북(`colab_gat_a100.ipynb`) 추가. 배경: DREEAM+GAT+GREP+
> PUATLoss 전부 결합 시 GAT 단독보다 성능이 낮아 GAT 고도화에 집중하기로 결정.
>
> 2026-07-14 (수정): 학습 epoch 기본값이 `Scripts/atlop` baseline(distant_epochs=1, epochs=15)과
> 달랐던 것(2/12)을 발견해 baseline과 정확히 일치하도록 수정 — 통제 비교를 위해 아키텍처만 변수로
> 남김. `train_gat.py`, `colab_gat_a100.ipynb` 기본값/문서 모두 반영 완료.

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
        ③ Pair context: head·tail attention 곱 → 문맥 벡터 c_ht
                              │
                              ▼
         2-Layer Edge-featured GAT (노드 = Entity)
        Edge Embedding 32-d = [관계카테고리 8 ; 문장거리 8 ;
                                head타입 8 ; tail타입 8]
        α_ij = softmax( LeakyReLU( aᵀ[Wh_i ‖ Wh_j ‖ e_ij] ) )
        Layer1: 투영→edge-aware 멀티헤드 attention→집계
                →residual→LayerNorm→GELU→Dropout
        Layer2: 동일 (GELU 없이 LayerNorm까지)
                              │
                              ▼
             Pair Representation = Linear([g_h ; g_t ; c_ht])
                        (2304 → 768)
                              │
                              ▼
                    Relation Classifier
        LayerNorm → Linear(768→768) → GELU → Dropout(0.1)
                 → Linear(768→97)
                              │
                              ▼
                  Sigmoid (관계별 독립 확률)
                              │
                              ▼
        Loss = BCEWithLogitsLoss + 0.2 × Evidence Contrastive
```

## 구현 파일

| 파일 | 역할 |
|---|---|
| `preprocess_gat.py` | atlop의 `*` 마커 방식 확장: 문장 토큰 span, 엔티티 타입, edge 카테고리/거리/인접행렬, pair별 evidence 집합 |
| `model.py` | `DocREGATModel` — 인코더/엔티티 풀링/LCP/EGAT 2층/분류기/BCE+InfoNCE |
| `train_gat.py` | 2단계 학습(distant→annotated), dev threshold sweep, 공통 스코어러 평가 |
| `colab_gat_a100.ipynb` | Colab A100 원클릭 실행 노트북 |

## 그래프 구성 (구현 확정값)

- **노드** = 엔티티, 노드 피처 = ATLOP 엔티티 임베딩(768)
- **Edge 카테고리** (Embedding(4, 8)): `3`=self-loop, `2`=멘션 span 겹침, `1`=같은 문장 공출현, `0`=그 외
- **문장 거리** (Embedding(6, 8)): 두 엔티티 멘션 간 최소 문장거리 버킷 (0,1,2,3,4,5+)
- **엔티티 타입** (Embedding(7, 8) × head/tail): PER/ORG/LOC/TIME/NUM/MISC + unk
- **Sparse 인접행렬**: 같은 문장 공출현 OR 멘션 겹침 OR 문장거리 ≤ 2일 때만 edge (+ self-loop 상시) — 노이즈 message passing 억제

## 스펙 대비 구현 해석

1. **GAT 위치**: **엔티티 임베딩 → GAT → graph-enhanced E′로 pair 구성** — 최종 확정 스펙
   (2026-07-14 "Final Proposed Architecture")과 일치. pair는 GAT 통과 후의 g_h, g_t에 인코더
   attention 기반 LCP 문맥 c_ht를 concat (LCP의 attention은 GAT가 아니라 인코더에서 나오므로
   원 ATLOP 방식 그대로 인코더 마지막 층 attention 사용).
2. **분류기 입력 768-d**: [g_h; g_t; c_ht]는 2304-d이므로 `pair_proj`(Linear 2304→768)로 투영 후 스펙의
   2-layer MLP(LayerNorm→768→768→97)에 투입.
3. **Evidence Contrastive Loss**: InfoNCE로 구현 — pair의 LCP 문맥 c_ht를 anchor로, 정답 evidence 문장 임베딩
   (토큰 평균)을 positive, 문서 내 나머지 문장을 negative로 (τ=0.1). 스펙의 "random evidence/masking"
   negative보다 강한 표준형. **train_distant는 evidence가 없어 이 항이 자동으로 비활성** (가중치 0.2는
   annotated 단계에서만 실질 작동).
4. **NA 불균형 처리** (PRD §2 필수): BCE+sigmoid는 고정 0.5 threshold에서 recall 붕괴 위험 →
   **매 평가마다 dev threshold sweep(0.10~0.90, 0.05 간격)** 으로 F1 최대 지점을 선택, 로그에 함께 출력.

## 학습 설정

AdamW / lr 2e-5 / weight decay 0.01 / dropout 0.1 / warmup 6% / grad clip 1.0 — 스펙 그대로.

**epoch 수는 `Scripts/atlop` baseline과 정확히 동일하게 맞춤**: distant 20,000개 × **1 epoch**
→ annotated × **15 epoch**, seed 66. 원 스펙은 "distant 2~3ep / annotated 12~15ep"라는 범위를
제안했지만, baseline(`atlop`, `atlop_full_pu07`)과 epoch 수가 다르면 성능 차이가 "GAT 때문"인지
"학습을 더/덜 해서"인지 구분이 안 되므로, 통제 비교를 위해 baseline과 완전히 동일한 스케줄로
확정했다 (distant_limit/distant_epochs/epochs/seed 전부 일치, 차이는 아키텍처뿐).

## 실행

```bash
# CPU 정합성 검증 (통과 확인됨, 2026-07-14)
python -m Scripts.dk_gat.train_gat --limit_docs 6 --epochs 1 --distant_epochs 1

# 풀 학습 (Colab A100 권장 — colab_gat_a100.ipynb 사용)
python -m Scripts.dk_gat.train_gat --distant_limit 20000 --distant_epochs 2 \
    --epochs 12 --run_name dk_gat --save_model --seed 66
```

## 비교 기준

`results/comparison.md`: ATLOP baseline 61.71/59.86, ATLOP+PU(0.7) 62.06/60.16, 트랙1 61.77/59.98.
동일 스코어러(F1/Ign F1)·동일 seed(66)로 비교. 예측 포맷은 팀 공통 `[{"title","h_idx","t_idx","r"}]`.
