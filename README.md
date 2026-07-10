## Information Extraction

### task 소개

자연어 텍스트에는 다양한 객체(Entity)와 그 관계(Relation)이 포함되어있지만 이러한 정보는 문장 단위로, 혹은 문서 단위로 분산되어있으며 명시적으로 드러나지 않는 경우가 많습니다.

Information Extraction은 텍스트로부터 의미있는 객체와 그 관계를 추출하여 구조화된 데이터로 변환하는 작업이며 기업에서 활용하는 데이터는 많은 경우 검색, 분석, 추론 등의 작업을 위해 대부분 이러한 정형화된 형태를 요구하고 있습니다.최근에는 단순 정보 추출을 넘어,(Knowledge Graph) 구축etrieval-Augmented Generation)에서의 검색 품질 개선

- LLM 출력의 안정화

등의 목적으로 활용되며 특히 온톨로지 기반 구조화 레이어로서 그 중요성이 다시 강조되고 있습니다.

### 수행 내용

데이터를 통해 다음 작업이 수행되어야합니다.

- 문서 내 등장하는 객체(Entity)를 식별하고 동일 객체(Entity)의 다양한 표현을 통합(coreference resolution)
- 객체(Entity) 간 관계(Relation)를 식별
- 문서 전반에 걸친 정보를 활용하여 관계를 추론(Multi-hop reasoning 포함)

최종적으로는 각 객체와 관계가 담긴 구조화된 표현을 생성해야합니다.

이 결과는 다음과 같은 구조로 확장될 수 있어야 합니다.

- Knowledge Graph
- Relation Database
- Graph-based reasoning system(의 일부)

### 난이도

★★★★☆(4/5, NER까지 진행은 어렵지 않지만 relation 생성과 multi-hop 기반 schema 정립이 난해함)

### 학습 데이터

DocRED[[링크]](https://huggingface.co/datasets/thunlp/docred)

---

#### 연구 계획서

주제가 선정되면 연구 계획서를 작성합니다. 연구 계획서는 **7월 13일(월)**까지 작성되어야하며 다음 내용이 포함되어야합니다. 예시 링크

- Problem Definition
    - 어떤 task를 선택했는가?(예: Information Extraction / Keyphrase Extraction),
    - 구체적으로 어떤 문제를 해결하려고 하는가?
    - 왜 이 문제가 중요한가? (응용 맥락 포함)
- Background & Baseline
    - 기존에 연구들은 이 문제, 혹은 유사 문제에 대하여 어떻게 접근하였는가?(논문 리서치. 단, 논문 리뷰는 생략합니다)
    - 이번 프로젝트에서 사용할 baseline 모델
- Proposed Method
    - 어떤 아이디어를 적용할 것인가?
    - 채용한 아이디어가 baseline 대비 무엇이 다른가?
        - Dataset & Preprocessing
            - 사용할 데이터셋
            - 데이터 특성
            - 전처리 방법
    - 왜 효과가 있을 것으로 예상하는가?
- Experiment Design
    - 어떤 실험을 할 것인가?
    - 어떤 비교를 할 것인가?
    - Evaluation Metrics은 어떻게 설계하였고 어떻게 검증할 것인가?
- Plan
    - 일자별 어떤 것을 수행할 것인가?