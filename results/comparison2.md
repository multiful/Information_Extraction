# 개선 모델 검증 (개선 1 GCN / 개선 2 GAT / 개선 3 통합)

> 2026-07-13: 세 개선 모두 baseline 대비 **유의미한 개선 없음**(전부 seed 변동 ±1점 안). multi-hop probe도 판정 불가. 상세 재현·PU 실험은 `comparison.md` 참고.

## 정량 (dev, 공통 스코어러 / 모두 seed 66, distant 2만 pretrain + annotated 15ep)

| 모델 | 파일 | 인코더 | dev F1 | Ign F1 |
|---|---|---|---|---|
| baseline | `re_model.py` | BERT | 61.71 | 59.86 |
| 개선 1 — GCN | `re_model_gcn.py` | BERT | 61.65 | 59.81 |
| 개선 2 — GAT | `re_model_gat.py` | BERT | 62.02 | 60.12 |
| 개선 3 — 통합(DREEAM+GAT+PU 0.7) | `re_model_full.py` | RoBERTa | 61.82 | 60.04 |

- 네 모델 격차(최대 0.37)가 전부 seed 변동 범위 → **우열 없음.** GAT가 근소 최고, GCN 동률, 통합은 세 요소를 더했는데도 GAT 단독보다 낮음(RoBERTa 사용에도).

## 정성 — 테스트 1 multi-hop probe

네 모델 **출력 완전 동일**: `(Tressonia)--P36-->(Marniva)` 1건, 관심 쌍 3개(0,1)/(1,2)/(0,2) 전부 미검출.
→ **판정 불가**: baseline이 허구 지명(OOV) 때문에 전제 관계부터 못 읽음 → 그래프가 결합할 입력이 없음. 개선의 실패가 아니라 probe가 이 능력을 테스트하지 못하는 것.

## 다음 단계 (택1)

- 요소별 ablation(`--evi_lambda 0` / `--na_weight 1.0` / `--model_name_or_path bert-base-cased`) + 복수 seed로 각 기여 분리.
- 또는 probe를 전제가 읽히는 형태(실존 엔티티)로 교체하거나 inter-sentence F1 분해로 multi-hop 효과 직접 확인.
