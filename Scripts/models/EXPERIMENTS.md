# dk 브랜치 실험 기록

> **최종 업데이트**: 2026-07-13: 실험 3(최종 파이프라인) distant 전체(101,873개, 34 chunk) +
> annotated 8epoch 파인튜닝 결과 추가 (dev F1 0.5710 → 0.6177). 공식 Ign-F1 스코어러
> (`Scripts/docred_scorer.py`, thunlp/DocRED 포팅) 신규 구현 및 재평가 (Ign F1 0.5998, ATLOP
> 논문 59.22 대비 근소 우위). MPS OOM 3가지 원인 진단/수정 내역도 `dk_model.py`/`dk_train.py`
> 모듈 docstring에 기록.

베이스라인: RoBERTa-base + Mention Average Pooling + Localized Context Pooling + Adaptive
Thresholding (ATLOP의 핵심 두 기법만 차용, entity marker/logsumexp pooling은 제외 —
`dk_model.py` 상단 docstring 참고). 512 토큰 초과 문서는 train_annotated 0.49%, dev 0.80%뿐이라
허용 범위로 판단, 별도 sliding-window 처리 없이 truncation.

평가: 정확 일치 기반 간이 F1 (공식 Ign-F1 스코어러 아님 — 팀 공용 스코어러 완성 전까지 이걸로 비교).
검증셋: dev (998 docs).

---

## 실험 1: Distance Embedding (bucketed sentence-distance) A/B

- **가설**: head-tail 엔티티 간 문장 거리를 명시적으로 임베딩해서 주면 inter-sentence 관계 판단에 도움될 것
- **설정**: train_annotated 3,053개 전체, 4 epoch, patience=2, seed=42 (두 run 동일), 나머지 하이퍼파라미터 동일 — `use_dist_embedding` 플래그만 다름
- **결과**:

  | | Without | With |
  |---|---|---|
  | 최종 dev F1 | **0.5387** | 0.5349 |
  | Precision | 0.6800 | 0.6725 |
  | Recall | 0.4460 | 0.4441 |

- **결론**: **효과 없음 (채택 안 함)**. 차이(0.0038)가 CPU 멀티스레드 연산의 eval 비결정성으로 관측된 변동폭(±0.1 수준) 안에 있어 유의미하지 않음. epoch별 추이도 일관성 없음(epoch1은 with가 우세, 나머지는 without 우세).
- **추정 원인**: Localized Context Pooling이 attention을 통해 이미 "가까운/관련 문맥"에 암묵적으로 가중치를 주고 있어 distance bucket 정보가 상당 부분 중복 신호였을 가능성. 3,053개·4epoch로는 추가 파라미터(임베딩 테이블) 학습에 데이터가 부족했을 가능성도 있음.
- 로그: `logs/ab_nodist.log`, `logs/ab_dist.log` / 체크포인트: `dk_checkpoints/ab_nodist.pt`, `dk_checkpoints/ab_dist.pt`

---

## 실험 2: PU Loss (TTM-RE 2024 핵심 아이디어 간소화) A/B

- **가설**: train_distant의 "Na" 라벨은 확정된 음성(negative)이 아니라 미표기(unlabeled)일 수 있음(distant supervision이 놓친 실제 관계). Na 페어의 TH-랭킹 loss 항을 다운웨이트하면 distant 프리트레인의 신호 품질이 개선될 것
- **구현**: `PUATLoss` (`dk_model.py`) — 정식 TTM-RE의 nnPU risk estimator나 Token Turing Machine 메모리 모듈은 포함하지 않은 근사 버전, "distant-Na 페어의 TH-랭킹 항에 `na_weight`(기본 0.5) 가중치"만 반영. `na_weight=1.0`이면 표준 ATLoss와 완전히 동일함을 단위 테스트로 확인함
- **설정**: train_distant 20,000개 랜덤 샘플(sample_seed=123, 두 조건 동일) 1 epoch 프리트레인 → train_annotated 2 epoch 파인튜닝(seed=42, 두 조건 동일), `na_weight=0.5`
- **결과**:

  | | 일반 ATLoss | PUATLoss |
  |---|---|---|
  | distant 프리트레인만 (dev F1) | 0.4315 | **0.4779** |
  | → annotated 파인튜닝 후 (최종 dev F1) | 0.5686 | **0.5710** |
  | (참고) annotated만 사용, distant 없음 | 0.5387 | 0.5387 |

- **결론**: **PU Loss 채택**. distant 단계에서 명확한 개선(+10.8%), 파인튜닝 후에도 근소하게 우위 유지(파인튜닝이 노이즈를 어느 정도 보정하면서 격차는 줄어듦, 방향은 일관됨). 추가로 distant 프리트레인 자체가 annotated-only 대비 확실히 도움됨을 확인(+0.03 전후) — distant→annotated 표준 순서 전환이 유효했음을 뒷받침.
- 로그: `logs/pu_{off,on}_{distant,final}.log` / 체크포인트: `dk_checkpoints/pu_{off,on}_{distant,final}.pt`

---

## 실험 3: 최종 파이프라인 (distant 프리트레인 → annotated 파인튜닝)

- 채택된 설정: Distance Embedding 미사용, PU Loss 사용 (na_weight=0.5)
- 실험 2 결과(2만 샘플 기준): dev F1 = 0.5710 (`dk_checkpoints/pu_on_final.pt`) — 이 결과를 보고
  전체 101,873개로 확장 결정

### 전체 규모 실행 (distant 101,873개 1epoch-eq. → annotated 8epoch)

- **distant**: `chunked_distant_full.sh` — 3,000개씩 34 chunk(총 102,000 doc-exposure, ~1epoch),
  각 chunk가 이전 체크포인트에서 이어받아 학습(중간에 죽어도 chunk 단위로만 손실). chunk 1-14는
  `--device cpu`, chunk 15-34는 `--device mps`(MPS의 attention/그래프 캐시 OOM 3가지 원인을 모두
  진단·수정한 뒤 전환 — 상세 내역은 `dk_model.py`/`dk_train.py` 모듈 docstring 참고). 체크포인트:
  `dk_checkpoints/distant_full_pu.pt`, 로그: `logs/distant_full_pu_chunks.log`
- **annotated**: distant 체크포인트에서 이어받아 8 epoch, patience=3, `--device mps`. 체크포인트:
  `dk_checkpoints/final_full_pu.pt`, 로그: `logs/final_full_pu.log`

| 단계 | dev F1 |
|---|---|
| distant 프리트레인 직후 (resume) | 0.5214 |
| annotated epoch 0 | 0.5177 |
| annotated epoch 1 | 0.5684 |
| annotated epoch 2 | 0.5825 |
| annotated epoch 3 | 0.6018 |
| annotated epoch 4 | 0.6068 |
| annotated epoch 5 | 0.6148 |
| annotated epoch 6 | 0.6148 |
| **annotated epoch 7 (final)** | **P=0.6506 R=0.5879 F1=0.6177** |

- **결론**: 2만 샘플 결과(0.5710) 대비 전체 규모(0.6177)가 +0.0467 개선. 8epoch 내내 patience
  조기종료 없이 꾸준히 상승(epoch 0만 resume 대비 소폭 하락 후 반등, 최근 3epoch은 증가폭이
  +0.008/+0.000/+0.003로 수렴 조짐) — 상한 근처로 보이나 여지는 있음.
- MPS 속도: chunk당 CPU 평균 13분53초 → MPS 평균 10분6초 (약 1.37배, chunk 15-34 20개 평균 기준)

### 공식 Ign-F1 스코어러로 재평가 (`Scripts/docred_scorer.py`)

`thunlp/DocRED`의 `code/evaluation.py`(MIT 라이선스)를 그대로 포팅해서 팀 공용 스코어러로 추가함.
`final_full_pu.json` 예측 결과를 실제로 채점:

| 지표 | 우리 (final_full_pu) | ATLOP 논문 (BERT-base, Dev) |
|---|---|---|
| F1 (일반) | 0.6177 | 61.09 |
| **Ign F1** (논문이 실제 보고하는 지표) | **0.5998** | **59.22** |
| Ign F1 (distant 기준, 참고용) | 0.5545 | - |
| Evidence F1 | 0.0000 (evidence 예측 자체를 안 함, 버그 아님) | - |

일반 F1(0.6177)이 간이 스코어러 결과(0.6177)와 정확히 일치함을 확인 — 간이 스코어러가 애초에
"일반 F1"은 정확히 계산하고 있었고 Ign 필터링만 없었던 것으로 검증됨. **Ign F1 기준으로도 근소
우위(59.98 vs 59.22, +0.76)가 유지됨** — 지표를 올바르게 바꿔도 결론이 안 뒤집힘.
여전히 남는 단서: single seed 결과이고, 비교 대상이 논문 수치이지 팀원의 실제 ATLOP 구현체가
아님 — PRD.md 결정 규칙(90% 임계값) 적용은 팀원 결과가 나온 뒤 가능.
