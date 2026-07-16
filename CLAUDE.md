# CLAUDE.md

> **최종 업데이트**: 2026-07-16:
> LLM 호출 비용 최적화 가이드라인 추가 — 대량 호출은 배치 처리를 기본으로, 실행 전 비용 추정 후 확인.
>
> 이전 (2026-07-10):
> 작성하는 md 파일에는 최종 업데이트 날짜와 요약을 작성해야함. 
> 요약 예시 : `docred_data`의 `.json.gz` 원본 압축 해제 및 정리, 압축 해제 산출물 `.gitignore` 처리, `requirements.txt` 작성.

## LLM 호출 비용 최적화

- 후보가 많을 때(수백 개 이상) LLM 검증/분류는 개별 호출이 아니라 N개씩 묶어 배치 호출(JSON 배열 응답)로 처리하는 것을 기본으로 한다 — 개별 호출은 매번 같은 지시문을 반복 전송해서 후보가 늘수록 토큰 낭비가 커짐.
- 수천 건 이상 규모로 LLM을 호출하기 전에는 반드시 예상 요청 수/토큰을 먼저 추정해서 공유하고, 확인받은 뒤 실행한다 — 예산이 빠듯한 개인 API 키라 무계획 대량 호출이 바로 비용 문제가 됨(2026-07-16, 배치 없이 만 개 단위 개별 호출을 돌리다 실측으로 발견).

## 프로젝트 개요

문서 단위 관계 추출(Document-level Relation Extraction) 프로젝트. 자연어 문서에서 개체(Entity)를 식별하고 개체 간 관계(Relation)를 추출하여 Knowledge Graph 등으로 구조화하는 것이 목표. 자세한 과제 설명은 `README.md` 참고.

- 팀 진행 상황은 `Daily_Standup/YY-MM-DD.md`에 기록됨.
- 연구 계획서 마감: 2026-07-13.

## 데이터: `docred_data/`

[DocRED](https://huggingface.co/datasets/thunlp/docred) 데이터셋 (HuggingFace `thunlp/docred`).

- `docred.py`: HF `datasets` 로더 스크립트 (원본 필드 `r/h/t` → `relation_id/head/tail`로 변환).
- `data/*.json`: 압축 해제된 산출물 (용량이 커서 `.gitignore`에 등록, git에는 커밋되지 않음).

| 파일 | 문서 수 | 레이블 포함 |
|---|---|---|
| `train_annotated.json` | 3,053 | O (사람이 직접 annotate) |
| `train_distant.json` | 101,873 | O (distant supervision) |
| `dev.json` | 998 | O |
| `test.json` | 1,000 | X (Codalab 제출용) |
| `rel_info.json` | 96개 relation id → 이름 매핑 | - |

각 문서는 `title`, `sents`(문장 토큰 리스트), `vertexSet`(개체 mention 클러스터), `labels`(head/tail 개체 간 relation, evidence 문장 포함) 구조.

## 개발 환경

`requirements.txt` 참고 (데이터 로딩용 `datasets`/`huggingface_hub`, 베이스라인 모델용 `torch`/`transformers`, 분석용 `pandas`/`scikit-learn`/`matplotlib` 포함). 설치: `pip install -r requirements.txt`.
