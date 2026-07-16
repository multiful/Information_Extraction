# Product

## Register

product

## Users

연구팀 본인(로컬에서 GraphRAG 파이프라인 디버깅/결과 확인)과, 교수·외부 평가자에게 보여주는 발표·데모 상황의 두 종류 사용자가 있다. 두 경우 모두 데스크톱 화면에서 짧은 시간 안에 "이 답변이 어떤 경로로, 어떤 근거로 나왔는지"를 확인하는 것이 목적이라, 반응형/터치 대응보다 데스크톱 가독성·정보 밀도가 우선이다.

## Product Purpose

DocRED 기반 지식그래프(Neo4j)에 자연어로 질의하면 GraphRAG(관계 그래프 탐색)와 naive RAG(문장 벡터 검색)를 나란히 비교해 보여주는 연구용 워크벤치("Trace — Knowledge Graph Workbench"). 핵심 가치는 "질문을 경로로, 답변을 증거로" — 최종 답변뿐 아니라 질의 분석→엔티티 링킹→라우팅(속성 스캔/1-hop/멀티홉 BFS)→리랭킹→반복 샘플링+다수결까지 전 과정을 실제 수치와 근거 문장(evidence)으로 감사 가능하게(auditable) 드러내는 것. 성공은 "이 파이프라인이 왜 이렇게 답했는지"를 사용자가 화면만 보고 납득할 수 있는가로 판단한다.

## Brand Personality

조사관의 증거 대장(evidence ledger) 같은 톤 — forensic lab notebook / investigative. 딱딱한 기업 대시보드도, 캐주얼한 챗봇 UI도 아니고, "근거를 추적하는 연구소"에 가깝다. 다크 포레스트그린 배경 + 크림 카드 + 앰버 강조라는 팔레트, "TRACE", "QUERY INTELLIGENCE CONSOLE", "AUDITABLE ANALYSIS PATH", "EVIDENCE LEDGER" 같은 조사/연구소 어휘가 이미 코드에 구현돼 있고, 이 톤을 유지하기로 확정함(2026-07-16).

## Anti-references

화려한 SaaS 마케팅 랜딩페이지 톤(그라디언트 히어로, 화려한 CTA)은 아님. 지나치게 캐주얼하거나 장난스러운 챗봇 UI도 아님. 이 도구는 "결과를 파는" 게 아니라 "과정을 증명하는" 도구.

## Design Principles

- 모든 답변에는 경로(route)와 근거(evidence)가 따라붙는다 — 답변만 덜렁 보여주지 않는다.
- 실질 정보가 없는 상태 표시는 만들지 않는다(예: 항상 켜져 있는 "ONLINE" 배지, 항상 1.0인 confidence 숫자 — 실측으로 제거된 사례).
- 라우팅/투표/리랭킹처럼 파이프라인 내부에서 실제로 갈리는 선택만 UI에 노출한다.
- 데스크톱 발표 화면 기준 정보 밀도를 우선하되, 접근성 기본기(WCAG AA 권장 관행)는 지킨다.

## Accessibility & Inclusion

표준 WCAG AA 권장 관행 준수를 목표로 한다. 별도의 스크린리더/색약 대응 등 특수 요구사항은 없음.
