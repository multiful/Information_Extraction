# dk EGAT 모델 — 제안 아키텍처 파이프라인 문서

> **최종 업데이트**: 2026-07-14 (**Evidence Fusion 추가 — EIDER 스타일 추론 시점 기법,
> 학습 불필요**): 외부 논문 조사 결과, 지금까지 실패해온 "인코딩 후 그래프를 얹는" 계열
> 말고 완전히 다른 지점(추론 시점 예측 융합)을 건드리는 EIDER(Xie et al., Findings ACL
> 2022)를 채택. **새 파라미터 없음** — 이미 학습된 아무 체크포인트에나 바로 켜볼 수 있음.
>
> **동작**: 전체 문서로 만든 원래 예측과, "이 쌍에 중요할 것 같은 문장만으로 만든" 예측을
> 평균(fusion). 후자를 위해 각 쌍마다 자신의 LCP 문맥과 문장 임베딩 간 코사인 유사도(기존
> InfoNCE evidence loss가 쓰던 것과 같은 유사도 함수, gold evidence 불필요 — test에도 적용
> 가능)로 top-k 문장을 뽑고, joint attention을 그 문장 토큰들로만 재정규화해 "증거 전용"
> LCP 문맥을 다시 만든 뒤, **같은 분류기 헤드**로 재분류. 원 논문은 이걸 위해 BERT를 통째로
> 한 번 더 인코딩하지만, 여긴 **이미 계산된 BERT 출력을 재사용**해서 두 번째 인코더 forward
> 없이 근사(더 저렴, 완전히 동일하진 않음).
>
> `--evidence_fusion --evidence_fusion_top_k 3`(기본 off): `model.py`의 `_classify`/
> `_evidence_fusion_context` 참고. `train_gat.py`의 `predict()`에 인자로 흘러들어가서
> annotated 단계 매 epoch dev 평가·최종 best-checkpoint 예측에 반영됨(학습 loop 자체엔
> 영향 없음 — 학습 시 forward는 이 인자를 안 씀). CPU에서 로짓이 실제로 바뀌는 것까지
> 검증(같은 입력, `evidence_fusion` on/off 시 logits 차이 0 아님) — distant/실측 검증은
> 안 된 상태, 하지만 새 파라미터가 없어서 **`dk_gat`/`dk_gat_v2`의 기존 체크포인트에 재학습
> 없이 바로 켜서 dev F1 비교 가능** (다음 실행부터 플래그만 추가하면 됨).
>
> 이전 (**`--freeze_encoder_epochs` 버그성 설계 수정 — distant
> 단계엔 더 이상 적용 안 함**): `dk_gat_v2` 실제 GPU 실행 중 stage-1 dev F1이 37.44로, 이
> 플래그가 없던 이전 실행(50.08)보다 크게 낮게 나온 걸 발견. 원인: `--distant_epochs 1`인데
> `freeze_encoder_epochs 1`을 "각 스테이지 첫 1epoch 동결"로 그대로 적용해서 **distant
> 스테이지 전체(20,000문서)가 통째로 동결**돼버림 — distant 사전학습의 목적 자체(노이즈는
> 있지만 대용량인 도메인 데이터로 인코더를 적응시키는 것)를 무력화. annotated 단계 첫 epoch
> 동결(15개 중 1개, 훨씬 작은 비용)은 원래 의도(무작위 초기화 헤드 워밍업)대로 의미가 있어
> 그대로 두고, **distant 단계(`train_gat.py`의 stage-1 `run_stage` 호출)에서만
> `freeze_encoder_epochs=0`으로 하드코딩** — CPU smoke test로 "encoder frozen" 로그가 stage
> 1엔 안 찍히고 stage 2 epoch 0에만 찍히는 것 확인. **이미 돌아가고 있던 `dk_gat_v2` 실행에는
> 반영 안 됨**(로컬 코드 수정이 실행 중인 Colab 세션엔 영향 없음) — 다음 실행부터 적용.
>
> 이전 (**Entity-Pair Graph 신규 추가** — A→B,B→C⇒A→C 합성 추론을
> 직접 겨냥, `dk_gat_v2`에 반영): `Scripts/atlop`의 별도 트랙(`re_model_gat.py` 등, 팀원 관할이라
> 미수정)과 상의하며 나온 통찰을 dk_gat 자체에 반영 — 지금까지 dk_gat의 Entity+Sentence GAT는
> **엔티티 임베딩만 보강**할 뿐, "쌍 (a,b)가 지금 어떤 관계로 예측되는지"를 다른 쌍이 직접 읽는
> 경로가 없었음. 이를 겨냥해 **두 번째 그래프 단계**를 추가:
>
> - **노드 = (h,t) 쌍 자체** (엔티티 아님). `same-head`/`same-tail`/`bridge-succ`/`bridge-pred`
>   4타입 엣지로 연결 — bridge를 방향성 있게 분리해 "내가 체인의 선행/후행 쌍인지"를 구분.
> - **관계-조건부 메시지**: 이 쌍의 provisional 관계 예측(logits)을 노드 특징에 얹어서, (a,c)가
>   (a,b)/(b,c)의 "현재 어떤 관계로 보이는지"를 직접 읽을 수 있게 함 — father_of+father_of면
>   grandfather_of라는 합성에 필요한 신호.
> - `pair_graph_out`이 **zero-init**이라 껐을 때(`--use_pair_graph` 미지정)와 완전히 동일한
>   동작 보장 — CPU에서 zero-init parity 직접 검증 완료(같은 seed로 pair-graph 켜고 끈 두
>   모델의 logits/loss가 소수점까지 동일).
>
> `--use_pair_graph --pair_graph_dim 256`(기본값), CPU smoke test(전 플래그 동시 적용)만
> 통과했고 distant 스크리닝은 생략 — `dk_gat_v2` 실행이 첫 실측.
>
> 이전 (`dk_gat`(Gated Fusion + Bilinear Classifier) 실행 완료 확인 —
> **여전히 baseline보다 낮고, 직전 버전 대비 개선폭도 스크리닝 기대치에 훨씬 못 미침**):
> 노트북 셀 5가 이미 GPU에서 완료돼 있었음(RTX PRO 6000, A100 아님 — GPU 종류는 무관, 완주는
> 됨). **결과: dev F1 60.62 / Ign F1 58.69 (best epoch 6, patience=3으로 epoch 9에서 조기
> 종료)** — baseline(61.71/59.86)보다 여전히 −1.09/−1.17 낮고, 직전 heterogeneous-graph-only
> 버전(60.29/58.39, best epoch 13)보다는 **겨우 +0.33/+0.30**만 오름. distant 스크리닝에서
> 따로 측정된 Gated Fusion(+2.2 F1)·Bilinear Classifier(+3.67 F1)를 단순히 더하면 +5.9를
> 기대했겠지만 실제론 1/10도 안 됨 — PU loss 선례(스크리닝 +2.93 → 실측 +0.35)보다도 더 크게
> 줄어든 사례. **이 프로젝트에서 두 번째로 확인된 패턴**: distant-only 1-epoch CPU 스크리닝
> 점수는 이 아키텍처 계열에서 annotated 파인튜닝 후 최종 성능의 신뢰할 수 있는 예측 지표가 아님
> — 절대적 개선폭이 아니라 "방향(+/-)" 정도만 참고하는 게 안전. epoch별 추이: distant epoch0
> 50.08 → annotated epoch2 58.75 → epoch6 60.62(peak) → epoch7~9 소폭 하락, early stop 정상
> 작동(더 돌려도 15 epoch 끝까지 이득 없었을 가능성 높음).
>
> 이전 (이번 사이클 "최종 고도화" — Meta-path Attention + Curriculum
> PU-weight 신규 구현, 학습전략 플래그 실전 반영, `colab_gat_a100.ipynb` 커맨드를
> `dk_gat_v2`로 갱신): 사용자가 "더 좋게 못 해?"라고 물어 다음 두 신규 아키텍처 요소를
> 구현하고 CPU smoke test(`--limit_docs 8 --epochs 2 --distant_epochs 1`, 전 플래그 동시
> 적용)로 크래시 없음만 확인함 — **아직 distant 스크리닝/실측 검증은 안 된 상태**, 이번 전체
> Colab 실행이 첫 실측 신호.
>
> **① Meta-path Attention** (`--use_metapath_attention`, 기본 off): `EdgeFeaturedGATLayer`의
> attention 파라미터 `a`를 엣지 전체가 공유하던 것에서 **엣지 카테고리별(self/entity-entity
> 같은문장/entity-entity 멘션겹침/entity-sentence) 독립 벡터**로 분리 — 기존엔 엣지 타입 정보가
> `e_ij` 임베딩을 통해서만 간접적으로 attention score에 반영됐는데, 카테고리별 `a`를 두면 모델이
> 엣지 타입마다 완전히 다른 스코어링 함수를 학습할 수 있음(예: "같은 문장 공출현"과 "멘션
> 겹침"을 다르게 취급). 실제 다중 홉 meta-path는 없는 그래프(entity-sentence-entity가 이미
> 2-layer GAT의 2-hop 경로 그 자체)라, 여기서 "meta-path"는 이 그래프가 가진 가장 세밀한
> 관계-타입 단위인 **엣지 카테고리 단위**로 해석해 구현함. 새 파라미터는 `att` 텐서가
> `(EDGE_CATS, num_heads, ...)`로 5배 커지는 것뿐(수천 개 float, encoder 대비 무시 가능한 규모).
> `model.py`의 `EdgeFeaturedGATLayer`(+ `DocREGATModel.forward`가 raw edge_cat 인덱스를
> `edge_cat_idx`로 전달) 수정.
>
> **② Curriculum PU-weight** (`--curriculum_na_weight` + `--na_weight_start`, 기본 off):
> distant 단계의 PUATLoss `na_weight`를 **step 단위로** `na_weight_start`(기본 1.0, = 사실상
> 일반 ATLoss)에서 `--na_weight`(0.7, 기존 검증값)까지 선형 anneal. epoch 단위가 아니라 step
> 단위인 이유: 현재 스케줄의 distant_epochs=1이라 epoch 단위 스케줄은 적용될 여지가 없음(1
> epoch 안에서 시작값 그대로 끝남). 동기: step 0의 GAT/분류기는 무작위 초기화 상태라 distant
> Na 레이블이 실제로 잘못됐는지(=숨은 양성) 판단할 능력이 아직 없음 — 학습 초반엔 distant
> 레이블을 전부 신뢰(na_weight→1.0)해 깨끗한 부트스트래핑 신호를 주고, 학습이 진행되며
> 검증된 na_weight=0.7로 서서히 down-weight를 도입. `train_gat.py`의 `run_stage()`에
> `na_weight_schedule` 인자 추가(`model.loss_fnt.na_weight`를 step마다 갱신, `PUATLoss`가
> 아니면 무시).
>
> **학습전략 플래그 실전 반영** (이미 구현은 돼 있었으나 실행 커맨드엔 미반영 상태였던 3개,
> 이번 `dk_gat_v2` 실행부터 추가): `--layerwise_lr_decay 0.9`(BERT 상위층일수록 더 낮은
> LR), `--freeze_encoder_epochs 1`(각 스테이지 첫 epoch은 인코더 동결 — 무작위 초기화된
> GAT/Gated-Fusion/Bilinear 헤드가 인코더를 흔들기 전에 먼저 적응할 시간을 줌),
> `--evidence_start_epoch 2`(evidence contrastive loss를 처음 2 epoch은 끄고 이후 활성화 —
> 초반 LCP context가 아직 노이즈가 많은 상태에서 evidence 쪽으로 당기면 주 ATLoss 신호와
> 충돌할 수 있음). `--lr2`/`--early_stop_patience`는 이전 사이클에 이미 반영됨.
>
> 이전 (예전 실행 결과 뒤늦게 발견 + 기록): `colab_gat_a100.ipynb` 셀
> 4의 예전 출력(Heterogeneous graph + interaction term, JK/Gated Fusion/Bilinear 없음)이 실은
> 이미 15 epoch 전부 완료돼 있었음 — **dev F1 60.27(final epoch)/60.29(best epoch 13), Ign F1
> 58.41/58.39로 baseline(61.71/59.86)보다 낮음**. Gated Fusion(+2.2 F1)·Bilinear
> Classifier(+3.67 F1) 스크리닝 결과가 이 60.29를 baseline 이상으로 끌어올릴 걸로 기대되지만,
> **`results/comparison.md`에 이미 기록된 선례**(PU loss가 distant-only 스크리닝에서 +2.93 F1이었다가
> 실제 파인튜닝 후엔 +0.35로 줄어든 사례)를 감안하면 스크리닝 이득이 그대로 다 전이된다는 보장은
> 없음 — 다음 전체 실행(Gated Fusion+Bilinear 반영)이 실제 확인 수단.
>
> 이전 (Gated Fusion / Bilinear Classifier / abs-diff 구현 + 학습 전략 플래그 추가, 전부 토글식
> — A/B 검증 전까지 기본값은 변경 없음):
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
| `model.py` | `DocREGATModel` — 인코더/엔티티+문장 풀링/LCP/이종 EGAT 2층(옵션: meta-path attention)/interaction pair repr/분류기/**옵션: Entity-Pair Graph 2차 정제**(`use_pair_graph`)/ATLoss(주입형)+InfoNCE |
| `train_gat.py` | 2단계 학습(distant PUATLoss→annotated ATLoss), best-epoch 체크포인트, `ATLoss.get_label` 디코드, 공통 스코어러 평가 |
| `colab_gat_a100.ipynb` | Colab A100 원클릭 실행 노트북 |

## 그래프 구성 (구현 확정값)

- **노드** = 엔티티(0..n_ent-1) + 문장(n_ent..n_ent+n_sent-1), 노드 피처 = 엔티티/문장 각각 토큰 평균 풀링(768, 같은 형태)
- **Edge 카테고리** (Embedding(5, 8)): `4`=self-loop, `3`=entity-sentence("등장함"), `2`=entity-entity 멘션 span 겹침, `1`=entity-entity 같은 문장 공출현, `0`=그 외
- **문장 거리** (Embedding(6, 8)): entity-entity 쌍의 최소 문장거리 버킷 (0,1,2,3,4,5+); entity-sentence/self 엣지는 0
- **노드 타입** (Embedding(8, 8) × head/tail): PER/ORG/LOC/TIME/NUM/MISC/unk(엔티티) + 문장 전용 공유 타입 1개
- **Sparse 인접행렬**: entity-entity는 기존 기준(같은 문장/멘션겹침/거리≤2) 그대로, entity-sentence는 해당 문장에 멘션이 있으면 연결, sentence-sentence 엣지는 없음(엔티티가 공유 문장 노드를 거쳐 2-hop 연결) + self-loop 상시
- **Meta-path Attention** (`--use_metapath_attention`, 기본 off): 위 4개 엣지 카테고리(self/같은문장/멘션겹침/entity-sentence)마다 GAT attention 벡터 `a`를 독립으로 둘지 여부 — 기본은 전체 공유(기존 동작)
- **Entity-Pair Graph** (`--use_pair_graph`, 기본 off, `model.py`의 `_build_pair_adjacency`/`EntityPairGATLayer`): 위 엔티티+문장 그래프와는 별개인 **2차 그래프** — 노드가 엔티티가 아니라 이 문서의 (h,t) 쌍 자체. `same-head`/`same-tail`/`bridge-succ`/`bridge-pred` 4타입 엣지로 연결하고, 노드 특징에 엔티티 표현뿐 아니라 그 쌍의 provisional 관계 예측(logits)까지 얹어(관계-조건부 메시지) A→B,B→C⇒A→C 합성을 직접 겨냥. `pair_graph_out`이 zero-init이라 끄면(`--use_pair_graph` 미지정) 완전히 이전과 동일 동작

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

**학습전략 플래그** (전부 토글, 기본값은 현재 동작 그대로): `--lr2`(stage2 전용 LR, 미지정 시
`--lr`과 동일 — 이번 `dk_gat_v2`는 1e-5), `--layerwise_lr_decay`(BERT 층별 LR 감쇠, 1.0=비활성
— 이번엔 0.9), `--freeze_encoder_epochs`(**annotated 단계에만 적용**, 시작 N epoch 인코더
동결 — 이번엔 1; distant 단계는 항상 미동결로 하드코딩, 위 최상단 업데이트 참고),
`--evidence_start_epoch`(evidence loss를 N epoch부터 curriculum 활성화 — 이번엔 2),
`--early_stop_patience`(N epoch 연속 dev F1 개선 없으면 조기 종료 — 이번엔 3),
`--curriculum_na_weight`/`--na_weight_start`(distant 단계 PUATLoss na_weight를 step 단위로
`na_weight_start`→`--na_weight` 선형 anneal — 이번엔 1.0→0.7). 뒤 3개(layerwise_lr_decay/
freeze_encoder_epochs/curriculum_na_weight)는 CPU smoke test만 통과했고 distant 스크리닝은
안 된 상태 — `dk_gat_v2` 실행이 첫 실측.

## 실행

```bash
# CPU 정합성 검증 (통과 확인됨, 2026-07-14 -- meta-path attention/curriculum na_weight/
# layerwise_lr_decay/freeze_encoder_epochs/evidence_start_epoch/entity-pair graph/
# evidence fusion 전부 동시 적용 후 재검증 완료, zero-init parity도 별도 검증)
python -m Scripts.dk_gat.train_gat --limit_docs 8 --epochs 2 --distant_epochs 1 \
    --use_pu_loss --na_weight 0.7 --use_gated_fusion --use_bilinear_classifier \
    --use_metapath_attention --curriculum_na_weight --na_weight_start 1.0 \
    --use_pair_graph --pair_graph_dim 64 --evidence_fusion --evidence_fusion_top_k 3 \
    --lr2 1e-5 --layerwise_lr_decay 0.9 --freeze_encoder_epochs 1 --evidence_start_epoch 1 \
    --early_stop_patience 3

# 풀 학습 (Colab A100 권장 — colab_gat_a100.ipynb 사용, run_name=dk_gat_v2)
python -m Scripts.dk_gat.train_gat --distant_limit 20000 --distant_epochs 1 --epochs 15 \
    --use_pu_loss --na_weight 0.7 --curriculum_na_weight --na_weight_start 1.0 \
    --use_gated_fusion --use_bilinear_classifier --use_metapath_attention \
    --use_pair_graph --pair_graph_dim 256 --evidence_fusion --evidence_fusion_top_k 3 \
    --lr 2e-5 --lr2 1e-5 --layerwise_lr_decay 0.9 --freeze_encoder_epochs 1 \
    --evidence_start_epoch 2 --early_stop_patience 3 \
    --run_name dk_gat_v2 --save_model --seed 66
```

`--save_model` 사용 시 저장물: `{run_name}_stage1.pt`(distant 마지막 epoch), `{run_name}_stage1_best.pt`
(distant 중 dev F1 최고), `{run_name}.pt`(annotated 마지막 epoch, baseline과 비교 기준),
`{run_name}_best.pt`(annotated 중 dev F1 최고) + 그 예측(`_best_dev_predictions.json`). 매 epoch
dev F1이 단조 증가하지 않으므로(실측: epoch 6 F1 59.85 → epoch 7 59.57) best 체크포인트가 최종
epoch보다 나을 수 있음 — 최종 보고 시 둘 다 확인 권장.

## 비교 기준

`results/comparison.md`: ATLOP baseline 61.71/59.86, ATLOP+PU(0.7) 62.06/60.16, 트랙1 61.77/59.98.
동일 스코어러(F1/Ign F1)·동일 seed(66)로 비교. 예측 포맷은 팀 공통 `[{"title","h_idx","t_idx","r"}]`.
