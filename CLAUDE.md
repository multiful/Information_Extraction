# CLAUDE.md

> **최종 업데이트**: 2026-07-10
> **요약**: `docred_data`의 `.json.gz` 원본 압축 해제 및 정리, 압축 해제 산출물 `.gitignore` 처리, `requirements.txt` 작성.

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
