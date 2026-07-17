---
target: data/streamlit/main.py
total_score: 25
p0_count: 0
p1_count: 3
timestamp: 2026-07-16T14-23-39Z
slug: data-streamlit-main-py
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3/4 | No lock/loading state on "분석 실행" during the 8-10s pipeline call — reproduced duplicate runs by double-clicking |
| 2 | Match System / Real World | 3/4 | Graph edge labels are raw DocRED codes (`LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY`) inside an otherwise all-Korean, plain-language UI |
| 3 | User Control and Freedom | 2/4 | No cancel for an in-flight analysis; no undo for accidental duplicate submission |
| 4 | Consistency and Standards | 3/4 | One remaining emoji (🔧) breaks the route-glyph convention the rest of the app follows; two adjacent evidence cards share an identical quote but carry different evidence-source tags |
| 5 | Error Prevention | 2/4 | No debounce on the primary button — double-click produced 3 duplicate, costly LLM+Neo4j+embedding runs, reproduced live |
| 6 | Recognition Rather Than Recall | 3/4 | Strong chips/history/sidebar, undercut by a history-card bug (see Minor Observations) |
| 7 | Flexibility and Efficiency of Use | 2/4 | No fast-iteration mode (every run pays full 3x-sampling cost); no export; no adjustable top-k |
| 8 | Aesthetic and Minimalist Design | 3/4 | On-brand and restrained; undercut by graph noise on multihop queries and orphaned dead CSS |
| 9 | Error Recovery | 3/4 | Neo4j-offline and no-entity-found states are calm, specific, and non-alarming; no-result state offers no forward path |
| 10 | Help and Documentation | 1/4 | No tooltips/explanations for terms like "다수결 일치도" or route names for a first-time viewer |
| **Total** | | **25/40** | **Acceptable — solid foundation, specific improvements needed** |

## Anti-Patterns Verdict

**No — this does not read as AI-generated.** Both assessments agree independently: the dark forest-green/cream/amber palette is distinctive and matches the stated "forensic lab notebook" brief; no gradient text, no glassmorphism, no hero-metric template, no side-stripe alert borders, no generic AI palette. Route indicators use text glyphs instead of icon soup — a deliberate choice visible in the code's own comments documenting prior emoji removal.

**Deterministic scan** (the bundled `detect.mjs` was unavailable — `Error: bundled detector not found`, missing `detector/detect-antipatterns.mjs` — Assessment B built an ad-hoc static scanner as a substitute and cross-validated it against the live DOM):
- 36 hard-coded hex colors total; 23 live inside the `:root` token block, 13 are `TYPE_COLORS`/pyvis literals that can't consume CSS variables (Python API constraint) — **false positive** if flagged as a token-system gap.
- 1 `@media` query, covering only `.stat-cell`.
- 2 emoji in rendered content: 🔧 (expander toggle) and 🕸️ (confirmed via live DOM to be `page_icon` only, not page content — **false positive** if flagged as an in-content anti-pattern).
- 63 `!important` occurrences — mostly justified: they override Streamlit's own default component styles, a documented, necessary pattern in this codebase, not indiscriminate use.
- Static ARIA scan found only 2 matches (one of which is a CSS selector reading Streamlit's own attribute, not authored ARIA) — but the **live DOM has 18** ARIA-bearing elements, because Streamlit's BaseWeb components (tabs, expander, buttons) inject `aria-expanded`/`aria-haspopup`/`aria-label` automatically. Static source scanning significantly undercounts real accessibility surface for a Streamlit app — a useful methodological finding in itself.
- Zero heading-hierarchy skip violations, confirmed identically by both static scan and live DOM query.
- Zero browser console errors/warnings across full page load, query execution, iframe render, tab switching, and expander interaction.

**Live visual evidence**: both assessments independently ran the same suggested question ("AirAsia Zest...") and both independently found the same phenomenon — the multihop BFS graph pulls in clearly unrelated entities (Assessment A saw a "Punch Club" video-game cluster; Assessment B saw Nintendo/Xbox/PlayStation/Zelda) alongside the real answer, with no visual signal distinguishing used-in-answer nodes from merely-collected ones.

## Overall Impression

The visual identity is genuinely earned, not templated, and the plain-language/technical-detail two-tier disclosure is a well-executed, non-obvious pattern that serves both stated user types without duplicating UI. But the app's single biggest gap is between what it claims to do and what it does: PRODUCT.md's core pitch is "나란히 비교" (side-by-side comparison), yet GraphRAG and Naive RAG are tabs — you can never see both at once, and must hold one answer in working memory while switching to read the other. A close second: a 1-in-3 vote consensus and a 3-in-3 consensus render in the exact same confident card style, which is the one place "auditable" quietly isn't delivered. The biggest opportunity is closing that gap between the tool's premise and its layout, not general polish.

## What's Working

1. **The plain/technical two-tier disclosure** — the default-visible "이렇게 답을 찾았어요" plain-language summary and the collapsed "기술적으로 어떻게 처리됐는지" expander draw from the exact same underlying numbers (same route, same fact counts, same sampling rationale). Nothing is dumbed down or hidden; it's a faithful compression, one click from full technical detail.
2. **Per-fact evidence provenance tags** (원문 근거(gold) / 브리징 추론 / 문장 동시 등장 추론 / 멀티홉(근거 미확정)) are unusually honest for a RAG demo — most tools just cite "a source"; this one admits when it's inferring versus quoting verbatim.
3. **The Naive RAG panel telling the truth when it doesn't know** — on an unanswerable test question it returned "모름" with visibly low similarity scores rather than confabulating, making the GraphRAG-vs-naive-RAG contrast genuinely informative instead of a rigged demo.
4. **Zero console errors across a full real end-to-end run** (confirmed independently) — the app is technically solid under the hood, not just visually polished.

## Priority Issues

**[P1] Tabs break the product's central "side-by-side" promise.**
*Why it matters*: PRODUCT.md states the core value as GraphRAG vs. naive RAG shown "나란히" (side-by-side). In the live app they're `st.tabs()`, mutually exclusive — comparing them requires reading one, switching, and holding it in memory against the other. This is the single biggest gap between stated purpose and actual UX, and it directly fails the Cognitive Load "working memory" check.
*Fix*: default to two columns for the top-line answer + key stat; keep tabs (if needed at all) for the deep evidence/graph drill-down only.
*Suggested command*: `/impeccable layout`

**[P1] Weak consensus is visually identical to strong consensus.**
*Why it matters*: reproduced live — a 1/3-vote answer (AirAsia Zest) and a 3/3-vote answer (Roketsan) render in the same cream card, same weight, same badge style. A tool whose entire pitch is auditability shouldn't let a 1-in-3 agreement hide in plain sight, especially in front of the "발표 참관 평가자" (demo-watching evaluator) persona who judges credibility purely from what's on screen.
*Fix*: distinct visual register (badge tone, inline caveat copy) when `agree_count/len(votes)` is low.
*Suggested command*: `/impeccable clarify`

**[P1] WCAG AA contrast failure on `.muted`/`--text-dim`, cross-validated by both assessments.**
*Why it matters*: measured independently twice (~4.0-4.04:1 against the cream card, ~3.25-3.4:1 against the dark panel background), both below the 4.5:1 AA minimum PRODUCT.md explicitly commits to. Affects real, frequently-seen text: the character counter, "N sources"/"top-5" counts, compare-column subtitles, and the RAG query-translation caption.
*Fix*: darken/desaturate `--text-dim` (#6b7362) until it clears 4.5:1 on both surfaces it's used against, or split it into two tokens (one per background).
*Suggested command*: `/impeccable polish`

**[P2] No protection against duplicate submission.**
*Why it matters*: reproduced live — double-clicking "분석 실행" produced three identical entries in "최근 분석" for the same question, each triggering a real, costly 3x-sampling LLM + Neo4j + embedding round-trip. This is a correctness and cost bug, not a style nitpick.
*Fix*: disable the button immediately on click, and/or dedupe consecutive identical submissions.
*Suggested command*: `/impeccable harden`

**[P2] Knowledge graph panel shows irrelevant nodes with no relevance signal.**
*Why it matters*: cross-validated by both assessments on independent runs of the same query — BFS expansion pulls in clearly unrelated entities (a video game and its platforms) into the same visualization as the real answer, with no dimming/filtering to distinguish used-in-answer from merely-collected. This is precisely the visual moment meant to build trust in a demo, and currently can undercut it instead.
*Fix*: visually de-emphasize (lower opacity, thinner edges) nodes that were collected by BFS but not part of the final reranked fact set actually used for the answer.
*Suggested command*: `/impeccable clarify`

## Persona Red Flags

**Alex (Impatient Power User — the 연구팀 본인 debugging the pipeline)**
- Wants to see which mention matched which entity via which cascade tier (Exact/Alias/Word Boundary/Fuzzy) to diagnose bad matches, but the UI only shows an aggregate count ("매칭 엔티티: 1개"). The data (`entity_results`) is already computed, and matching CSS classes (`.entity-chip`, `.entity-item`, `.entity-name`, `.entity-quote`) already exist in the stylesheet — they're just never rendered anywhere.
- No fast-iteration mode: every run pays the full 3x-sampling cost even during quick dev-loop testing.
- Double-clicking Run while impatient silently queues a duplicate expensive call, with no guard or warning.

**Sam (Accessibility-Dependent User)**
- Hits the measured contrast failures above (~3.25-4.04:1 vs. 4.5:1 required) on exactly the kind of small-caption text Sam relies on labels to read.
- The knowledge graph is canvas/iframe-based; the `sr-only` text alternative gives node/edge counts and type breakdowns but not the actual relation triples a sighted user sees in the Evidence Ledger — a screen-reader user gets meaningfully less audit detail than a sighted one, in a tool whose entire point is auditability.
- The physics simulation restarts and re-animates on rerun with no "reduce motion" affordance.

**Project-specific — "발표 참관 평가자" (professor/evaluator watching a live demo, hands off keyboard)**
- The video-game noise cluster in the graph (confirmed by both assessments) is exactly the kind of thing this persona notices and asks about mid-demo, with the presenter having no in-UI way to explain it away.
- Seeing "1/3표 일치" rendered with the same confident styling as a unanimous answer elsewhere would reasonably prompt "so how do I know when to trust this number?" — undercutting the "auditable" pitch at the worst possible moment.
- If Neo4j drops mid-demo, the calm, well-worded, `role="alert"` error card is a genuine strength — this persona's worst-case path is handled well (contrast on it measured at 5.73:1, passing AA).

## Minor Observations

- `_route_icon()` (history-card preview marker) takes the first word of `ROUTE_LABELS_KO[route]`, which works for routes like "① 1-hop 직행 조회" but breaks for `no_seed` → "매칭된 엔티티 없음", producing a history card whose icon reads as an unrelated fragment for a question that found nothing. Reproduced live.
- One remaining emoji-as-functional-icon (🔧 in the expander label) in an app whose own code comments document removing this exact pattern elsewhere (status dot, route badges).
- Two adjacent evidence cards for the same underlying quote sentence carry different `evidence_source` tags ("문장 동시 등장 추론" vs. "원문 근거(gold)") in a live run — worth a second look at the tagging logic.
- Graph edge labels are raw untranslated DocRED relation codes that visually overlap when multiple relations converge on one node — inconsistent with the plain-language philosophy applied everywhere else in the app.
- `.hero-title .accent { color: inherit; }` — the "경로"/"증거" accent spans in the hero headline have zero visual distinction from surrounding text; reads as an unfinished affordance rather than intentional restraint.
- Evidence quotes are in English (DocRED source text) inside an all-Korean UI with no note explaining the language switch — inherent to the dataset, minor friction for a first-time Korean-speaking viewer.
- Dead CSS (`.entity-chip`, `.entity-item`, `.entity-name`, `.entity-quote`) styled and ready but never rendered anywhere in current templates — likely a half-finished feature thread (see Alex's red flags above).

## Questions to Consider

1. If the core pitch is "나란히 비교" (side-by-side), what would this look like as two live columns instead of tabs — and is there a reason tabs were chosen that the layout should account for rather than override?
2. A 1-in-3 consensus and a 3-in-3 consensus currently look pixel-identical — for a tool built entirely around auditability, shouldn't the one number most worth foregrounding as a trust signal be the one that visually stands out the least right now?
3. Should nodes/edges that BFS collected but the reranker discarded ever appear at full visual weight in the graph, or would a faded "explored but unused" state better serve a skeptical viewer?
4. The exact debugging data a researcher needs (which mention matched which entity, via which cascade tier) is already computed and has orphaned CSS waiting for it — what was the original plan for `.entity-chip`/`.entity-item`, and is it worth finishing that thread now?
