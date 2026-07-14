# 모델 비교 기록 (PRD §6)

> **최종 업데이트**: 2026-07-13: 개선 모델 2종 비교 섹션 추가 — 개선 1(GCN)·개선 2(GAT), Entity Pair Graph로 multi-hop 약점(테스트 1) 공략. 학습 명령·판정 기준 기록, 결과 수치는 Colab 학습 후 기입 예정.
>
> 2026-07-13: 트랙3 `atlop_full_pu07`(ATLOP + PUATLoss na_weight=0.7, Colab A100) **dev F1 62.06 / Ign F1 60.16** 추가 — 현재 비교표 1위 (단 baseline 대비 +0.35는 seed 변동폭 이내). ATLOP baseline 61.71/59.86 기록(논문 대비 +0.62/+0.64, 재현 성공), 누수 검사(문서 중복 0건), 트랙1 `final_full_pu`(61.77/59.98) 비교 행, 약점 probe(상호참조 통과, 장거리·교란 실패).

## ATLOP baseline (트랙 2)

- 학습: `python -m Scripts.atlop.train_re --epochs 15 --distant_limit 20000 --distant_epochs 1 --eval_batch_size 32 --run_name atlop --save_model` (Colab Pro+ A100, 2026-07-13)
- 체크포인트: `results/atlop.pt` / 예측: `results/atlop_dev_predictions.json` (둘 다 gitignore, Drive `MS_AI_NLP(2026)_실습자료/21_실전프로젝트1/`에 백업)
- 최종 성적 (epoch 14): **dev F1 61.71 / Ign F1 59.86** (P 66.08 / R 57.89, train_loss 0.0078). 논문 BERT-base 대비 F1 +0.62, Ign F1 +0.64 — 재현 성공. F1−Ign F1 격차 1.85점은 논문의 1.87점과 동일 수준으로, distant 사실 노출로 인한 점수 부풀림 없음.
- 누수 검사(2026-07-13, 로컬 검증 완료): 사전학습에 쓴 distant 앞 2만 문서 ↔ dev **문서(제목) 중복 0건**, train_annotated ↔ dev도 0건 → 문서 수준 누수 없음. 단, dev 정답 사실의 **25.6%**(3,141/12,275)는 다른 문서를 통해 distant 라벨로도 등장(엔티티 이름+관계 기준) — distant supervision 고유의 사실 반복으로, 스코어러의 Ign F1은 train_annotated 사실만 제외하므로 이 부분은 반영 안 됨. 공정 비교용 annotated 단독 실행(`--distant_mode none`)은 별도 권장.

## 약점 정성 probe (`model.ipynb`, 각 1건짜리 시연 — 통계적 결론 아님)

| Probe | 기대 | 결과 | 판정 |
|---|---|---|---|
| 1. 상호참조 (근거 문장 주어가 대명사 "She"뿐) | (Marie Curie) —P19→ (Warsaw) | **검출 성공** | 약점 미발현 — 2문장 거리 대명사는 attention으로 해소 |
| 2. 장거리 + 교란 엔티티 | (War and Peace) —P50→ (Leo Tolstoy) | **미검출** + 교란 오예측: (War and Peace→Dostoevsky), (Crime and Punishment→Chekhov) | **약점 확인** — 장거리 관계를 놓치고 근처 같은-타입(PER) 엔티티로 오귀속 |

참고: probe 2에서 문장 내 관계 (Crime and Punishment→Dostoevsky)는 정확히 검출 — intra-sentence는 강하고 inter-sentence에서 무너지는 패턴. 트랙 1 모델의 공략 지점: 거리 인식(distance embedding 등), 엔티티 구분 신호(entity marker/type 강화). 정량 확인은 dev를 intra-/inter-sentence로 나눈 F1 비교로 후속 예정.

## 트랙 1 모델 비교

| 모델 | dev F1 | Ign F1 | probe 1 | probe 2 |
|---|---|---|---|---|
| ATLOP (baseline, distant 2만 pretrain + annotated 15ep) | 61.71 | 59.86 | 통과 | 실패 |
| ATLOP 논문 (BERT-base, annotated 단독 30ep, 5시드 평균) | 61.09 | 59.22 | - | - |
| RoBERTa + LCP + AT (+PU distant, 트랙1 `final_full_pu`) | 61.77 | 59.98 | 예정 | 예정 |
| **ATLOP + PUATLoss na_weight=0.7 (트랙3, `atlop_full_pu07`)** | **62.06** | **60.16** | 예정 | 예정 |

트랙1 `final_full_pu` vs baseline: F1 +0.06 / Ign +0.12 — 시드 변동(±1점) 안이라 사실상 동률. 구조 차이: mention 평균 풀링(단순화, ablation −1.3점 감수) + RoBERTa + PU Loss + distant 전체 10만으로 만회한 구성. probe 1·2 정성 비교 예정.

트랙3 `atlop_full_pu07` (2026-07-13, Colab A100, seed 66): baseline과 완전 동일 레시피에서 distant
프리트레인 손실만 PUATLoss(na_weight=0.7)로 교체 — F1 +0.35 / Ign +0.30, recall +1.29 (P −0.85).
stage-1(distant 직후) 시점 이득은 훨씬 크고(+2.93 F1, 5천개 A/B 기준) 파인튜닝을 거치며 줄지만 방향
유지. 문제정의·메커니즘·구출 통계(정답 1,415개 임계값 위로 복귀)는 `Scripts/atlop/PU_THRESHOLD_EXPERIMENT.md`
참고. 역시 single seed라 ±1점 유의성 주의 — 확정 비교는 2 seed 평균 필요.

## 개선 모델 (트랙 2 후속) — Entity Pair Graph로 multi-hop 약점 공략

`model.ipynb` 테스트 1(허구 지명 multi-hop: A→B, B→C 명시 시 A→C 조합 추론)에서 baseline이 실패 — (0,1)·(1,2) 명시 관계와 (0,2) 조합 관계 모두 미검출. 이를 겨냥해 baseline 파일 무수정(상속)으로 개선 모델 2종 추가 (`Scripts/atlop/README.md`의 "개선 모델 2종" 참고):

- **개선 1** `re_model_gcn.py` — GREP: Entity Pair Graph + relational GCN (이웃 고정 평균 집계)
- **개선 2** `re_model_gat.py` — Localized Context Pooling + Entity Pair Graph + GAT (attention 집계, bridge 엣지 특화 가능)
- 노드 특징·그래프 동일, 집계 방식만 달라 GCN vs GAT ablation이 됨. zero-init 잔차 헤드라 warm-start 시 시작점 = baseline (smoke_test_graph.py로 parity 검증 완료).

학습 (Colab A100, baseline과 동일 레시피):
```bash
python -m Scripts.atlop.train_graph --model gcn --epochs 15 --distant_limit 20000 --distant_epochs 1 --eval_batch_size 32 --save_model
python -m Scripts.atlop.train_graph --model gat --epochs 15 --distant_limit 20000 --distant_epochs 1 --eval_batch_size 32 --save_model
```

| 모델 | dev F1 | Ign F1 | 테스트 1 (0,1)·(1,2) 명시 | 테스트 1 (0,2) 조합 |
|---|---|---|---|---|
| ATLOP baseline | 61.71 | 59.86 | 실패 | 실패 |
| 개선 1 — GCN (`atlop_gcn.pt`) | (학습 후 기입) | | | |
| 개선 2 — GAT (`atlop_gat.pt`) | (학습 후 기입) | | | |

판정 기준: dev F1/Ign F1은 시드 변동 ±1점을 감안해 해석. 정성 판정은 `model.ipynb` "개선 모델 검증" 셀 실행 결과로 기입.
