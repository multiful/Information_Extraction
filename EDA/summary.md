# DocRED EDA Summary

생성 스크립트: `Scripts/eda_docred.py`

## 1. 스플릿별 문서 수

| split | documents |
|---|---|
| train_annotated | 3,053 |
| train_distant | 101,873 |
| dev | 998 |
| test | 1,000 |

## 2. 문서 구조 통계 (문장/토큰/엔티티/관계 수, 문서당 평균±표준편차)

| split | sents/doc | tokens/doc | entities/doc | mentions/doc | relations/doc |
|---|---|---|---|---|---|
| train_annotated | 7.9±2.8 | 197.7±62.7 | 19.5±5.7 | 26.0±8.0 | 12.5±7.6 |
| train_distant | 8.1±2.9 | 199.6±64.3 | 19.3±5.6 | 25.1±7.8 | 14.8±7.6 |
| dev | 8.1±2.8 | 200.5±67.1 | 19.6±5.8 | 26.2±8.1 | 12.3±7.7 |
| test | 7.9±2.7 | 197.8±63.4 | 19.5±5.5 | 26.7±8.0 | - |

## 3. 엔티티 타입 분포 (train_annotated)

| type | count |
|---|---|
| LOC | 24,400 |
| PER | 14,676 |
| TIME | 12,748 |
| MISC | 12,267 |
| ORG | 11,241 |
| NUM | 4,149 |

## 4. 관계(Relation) 통계

- **train_annotated**: 총 relation label 38,180개, 고유 relation type 96개 (전체 96개 중)
- **train_distant**: 총 relation label 1,505,638개, 고유 relation type 96개 (전체 96개 중)
- **dev**: 총 relation label 12,275개, 고유 relation type 96개 (전체 96개 중)

### Top-10 relation types (train_annotated)

| relation | id | count |
|---|---|---|
| country | P17 | 8,921 |
| located in the administrative territorial entity | P131 | 4,193 |
| country of citizenship | P27 | 2,689 |
| contains administrative territorial entity | P150 | 2,004 |
| publication date | P577 | 1,142 |
| performer | P175 | 1,052 |
| date of birth | P569 | 1,044 |
| date of death | P570 | 805 |
| has part | P527 | 632 |
| cast member | P161 | 621 |

## 5. Intra- vs Inter-sentence 관계 비율

- **train_annotated**: intra-sentence 20,818 (54.5%), inter-sentence 17,362 (45.5%) — 총 38,180개
- **dev**: intra-sentence 6,693 (54.5%), inter-sentence 5,582 (45.5%) — 총 12,275개

> inter-sentence 비율이 높을수록 문서 전체를 읽어야 관계를 추론할 수 있는 multi-hop 성격이 강하다는 뜻입니다.

## 6. Evidence 문장 수 분포 (train_annotated)

- 평균 1.61개, 중앙값 1개, 최대 11개

## 7. 생성된 그래프

`EDA/figures/` 폴더 참고:
- dev_entities_hist.png
- dev_relations_hist.png
- dev_sents_hist.png
- doc_counts.png
- entity_types_train_annotated.png
- intra_vs_inter_sentence.png
- relation_types_train_annotated.png
- test_entities_hist.png
- test_sents_hist.png
- train_annotated_entities_hist.png
- train_annotated_evidence_len_hist.png
- train_annotated_relations_hist.png
- train_annotated_sents_hist.png
- train_distant_entities_hist.png
- train_distant_relations_hist.png
- train_distant_sents_hist.png