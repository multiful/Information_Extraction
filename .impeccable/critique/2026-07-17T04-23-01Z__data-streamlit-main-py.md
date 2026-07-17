---
target: data/streamlit/main.py
total_score: 26
p0_count: 1
p1_count: 2
timestamp: 2026-07-17T04-23-01Z
slug: data-streamlit-main-py
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3/4 | Spinner copy is specific and dual-pipeline-aware; one static spinner covers 5 distinct pipeline stages for 8-15s with no stage-level progress, in a tool whose entire premise is showing its work |
| 2 | Match System / Real World | 3/4 | `ROUTE_PLAIN_KO`/`EVIDENCE_SOURCE_KO` translation layers are excellent, but the plain-language summary unconditionally claims `"정확히 일치하는 대상을 찾았어요"` ("found an **exact** match") whenever any entity matched, even when the cascade shown inches above actually resolved via Alias/Word-Boundary/Fuzzy, not Exact |
| 3 | User Control and Freedom | 2/4 | No cancel for the in-flight 8-15s run; no explicit reset to empty state once a result is showing |
| 4 | Consistency and Standards | 2/4 | The single most important object the app produces, the answer, is handled three different, inconsistent ways: shown only inside vote cards when `votes is not None`, never shown for 1hop/property_scan routes, never shown at all in the Naive RAG panel — verified directly in source, see Priority Issues |
| 5 | Error Prevention | 3/4 | Character counter, Neo4j pre-check, and the duplicate-submit guard are all present and sound, re-verified working this round |
| 6 | Recognition Rather Than Recall | 2/4 | The answer must be hunted from vote-card text or a 50-char-truncated history card at the page bottom; route glyphs (①/②'/③) appear with no visible legend anywhere in the UI |
| 7 | Flexibility and Efficiency | 3/4 | Suggested-question chips, full keyboard navigation, cached translation, and click-to-revisit history are solid accelerators |
| 8 | Aesthetic and Minimalist Design | 3/4 | Strong token discipline; undercut by Evidence Ledger redundancy — the same boilerplate source sentence is quoted verbatim as "evidence" on 8+ separate cards in a live-tested run |
| 9 | Error Recovery | 3/4 | Not exercised live this round (Neo4j stayed healthy); `role="alert"` error card is specific and actionable by code inspection, consistent with prior verified rounds |
| 10 | Help and Documentation | 2/4 | Pipeline-step labels exist but no contextual help for jargon ("BFS", "리랭킹", "다수결") aimed at a hands-off evaluator persona |
| **Total** | | **26/40** | **Acceptable — a genuinely well-crafted shell around one serious, newly-introduced structural gap** |

## Anti-Patterns Verdict

**No — still does not read as AI-generated**, reconfirmed a third time by both independent assessments. Component craft remains Linear/Notion-caliber: centralized CSS tokens, Restrained color use, no gradient-hero tone, and evidentiary honesty (edges with no supporting sentence are openly tagged `"(근거 문장 없음)"` / `멀티홉 (근거 미확정)` rather than hidden).

**Deterministic scan**: `detect.mjs` remains genuinely broken in this install (`Error: bundled detector not found`), same as every prior round — not a skipped step. The ad-hoc static scan is clean: 34 hex literals, 100% accounted for as `:root` tokens or pyvis Python-API params; ~65 real `!important` declarations, all documented overrides of Streamlit/BaseWeb defaults; exactly 1 functional emoji (🔧, unchanged); zero `border-left`, and the only `border-right` usage is a neutral `.stat-cell` divider, not a colored accent (false-positive candidate, correctly not flagged); zero gradient text, glassmorphism, or hero-metric template; zero heading-hierarchy skips; 1 `@media` query (unchanged, desktop-first by design).

**Live evidence, re-verified with exact numbers this round**:
- All 5 touch-target selectors independently re-measured: primary button 96×44px, tabs 659×44px, expander toggle ~1330×44px, chip buttons 45.6×1289px, history buttons 94×662.5px. All ≥44px, confirmed with `getBoundingClientRect()`, not just visual inspection.
- Focus ring color confirmed via `getComputedStyle`: `outlineColor: rgb(217, 164, 65)` = exactly `#d9a441` (`--accent`) on three different tabbed-to elements, not Streamlit's default red.
- Evidence Ledger near/far split confirmed present and functioning: one live run split 37 near / 2 far; a second live run (different query) split differently, both correctly gated behind the same `opacity >= 1.0` cutoff.
- Zero console errors/warnings across both assessments' full sessions, including keyboard navigation and expander interaction.

**One real, newly-surfaced structural finding neither prior round caught** (both this round's assessments converged on adjacent parts of it independently): the recent removal of the side-by-side compare row — done, per the code's own comment, because showing the same answer twice (summary row + tabs) seemed redundant — assumed the tabs already displayed each answer reliably on their own. Direct source verification confirms they don't:
- `render_rag_panel()` (main.py:1016-1043) never references `result["answer"]` anywhere. The Naive RAG answer is computed via a real, paid LLM call (main.py:306-315, returned at line 325) and then never rendered to the user under any circumstance.
- `render_graphrag_panel()` only surfaces `result["answer"]` inside vote cards, gated by `if result["votes"] is not None:` (main.py:954). Per the routing logic, `votes` is `None` for `1hop`/`property_scan` routes — the app's own comment (main.py:209-214) describes these as the *more common, cheaper, more stable* paths. For those routes, the answer appears nowhere in the main content area; the only place it exists in the DOM is truncated to 50 characters inside a small, equal-weight history button at the very bottom of the page (main.py:1079).

This was not a hypothetical: it reproduces on the currently-running app for any non-BFS route, and Naive RAG unconditionally, and was independently confirmed by direct code inspection (not just live browsing) after synthesis.

## Overall Impression

The floor keeps getting more solid: every fix from the last two rounds (touch targets, focus states, contrast, evidence relevance signal, duplicate-submission guard) is re-verified working, with harder numeric evidence this time than before. But this round surfaces the most consequential finding of the whole series so far, and it's a genuine regression, not a pre-existing gap: removing the compare row quietly broke the one guarantee that made removing it defensible. The core deliverable of an "answer engine" — the answer — currently has no reliable, dedicated place to render in either panel. Everything else this round (evidence redundancy, an overclaiming plain-language sentence, a relevance heuristic that can bury the exact fact that proves a BFS answer) is real but secondary to this.

## What's Working

1. **Evidentiary honesty, re-confirmed under live testing.** Facts with no supporting sentence are tagged openly rather than hidden or fabricated — a genuine trust-building choice most RAG demos skip.
2. **Shared relevance plumbing is disciplined engineering.** `_hop_distances()`/`_relevance_opacity()` are computed once and fed to both the graph and the Evidence Ledger, so the two panels never contradict each other on "how far is this from the seed" — even though this round found a real gap in what that shared signal actually measures (see Priority Issues).
3. **The accessibility floor is now measured, not just claimed.** Focus-ring color and all five touch-target sizes were independently re-verified with exact pixel/RGB values this round, by two separate assessments using two different live queries.

## Priority Issues

**[P0] Neither panel reliably displays the answer.**
*Why it matters*: verified directly in source. `render_rag_panel()` never renders `result["answer"]` under any circumstance — the Naive RAG conclusion is computed with a real paid API call and then silently discarded from the UI. `render_graphrag_panel()` only shows the answer inside vote cards, which only exist when `votes is not None` — meaning for `1hop`/`property_scan` routes (described in the code's own comment as the more common, stable paths), the GraphRAG answer also renders nowhere in the main view. This directly undercuts the rationale for removing the compare row: that removal assumed the tabs already showed each answer once, reliably. They don't. For a live demo audience this means the presenter reaches "분석 결과" and has to verbally supply the conclusion the screen should be delivering.
*Fix*: add an always-rendered, visually prominent "정답" element to the top of both `render_graphrag_panel()` and `render_rag_panel()`, independent of route or vote state — this is effectively resurrecting the compare row's one indispensable job (guaranteed answer visibility) without necessarily resurrecting the side-by-side layout itself.
*Suggested command*: `/impeccable harden`

**[P1] The relevance heuristic buries the evidence that actually proves multihop answers.**
*Why it matters*: `near`/`far` in the Evidence Ledger is computed purely from hop-distance-to-seed-mention (`main.py:990-1003`), not from which entities the answer text actually cites. On a live BFS-route test, the one gold-tagged fact directly supporting the answer sat at hop-distance 2 and was collapsed behind "관련도 낮은 근거 더 보기" by default, while all always-visible "near" facts were closer-but-answer-irrelevant corporate-structure trivia. BFS routes exist specifically to reach hop-distant nodes — this heuristic systematically de-emphasizes exactly the evidence that matters most for the one route type it's meant to help audit.
*Fix*: derive "near" from proximity to the entities named in the final answer text, not proximity to the original seed mention.
*Suggested command*: `/impeccable clarify`

**[P1] Plain-language summary overclaims match precision.**
*Why it matters*: `main.py:913` unconditionally states `"정확히 일치하는 대상을 찾았어요"` ("found an exact match") whenever any entity matched, regardless of whether the cascade shown in the same expander actually resolved via Alias/Word-Boundary/Fuzzy. In a tool whose entire premise is auditability, a small factual inaccuracy sitting next to the data that contradicts it is a real, if minor, credibility gap.
*Fix*: branch the copy on the actual matched tier (already available in `entity_results`), e.g. "그래프에서 [대상]과(와) 관련된 항목을 찾았어요" when the match wasn't Exact.
*Suggested command*: `/impeccable clarify`

**[P2] Evidence Ledger redundancy.**
*Why it matters*: the same boilerplate source sentence is quoted verbatim as "evidence" for 8+ distinct edges in a live-tested run — reads as padding rather than corroboration in a ledger whose credibility depends on each entry being a distinct, checkable fact.
*Fix*: group edges that share one source sentence into a single card listing multiple relations, rather than repeating the full quote per edge.
*Suggested command*: `/impeccable clarify`

**[P2] Revisiting history resets expander state.**
*Why it matters*: clicking a "최근 분석" card calls `st.rerun()` (`main.py:1089`), which collapses both the technical-detail and far-evidence expanders every time — minor friction for a researcher comparing two history entries side by side.
*Suggested command*: `/impeccable polish`

## Persona Red Flags

**연구팀 본인 (researcher debugging locally)**: auditing a BFS answer now requires remembering to always expand "관련도 낮은 근거 더 보기," since the heuristic that decides what's hidden isn't answer-aware. Cannot compare GraphRAG vs. naive RAG through the UI at all right now, since the naive RAG conclusion is never shown anywhere — this defeats the tool's stated purpose for the researcher's own use.

**발표 참관 평가자 (professor/evaluator, hands off keyboard)**: reaches "분석 결과" on the projector and sees a graph and a route description but, on the common non-BFS routes, no headline answer anywhere in the main view — has to be verbally supplied by the presenter. Switching to Naive RAG on stage to make the comparison point, the tab shows only retrieved chunks with no conclusion, which reads as broken mid-demo even though it's a display gap, not a pipeline failure.

**Sam (accessibility-dependent user)**: the `.sr-only` graph summary is a genuinely good, already-verified mitigation. But since the answer itself has no dedicated markup anywhere (no heading, no distinguishing role), a screen-reader user has no shortcut to it either. Evidence and vote cards are also plain `<div>`s with no list semantics, so list-navigation shortcuts don't work across entries — a gap none of the three prior rounds addressed, since they focused on contrast/focus/alt-text (all solid) rather than list structure.

## Minor Observations

- `st.text_area`'s native resize handle is the one un-styled affordance in an otherwise fully-controlled component vocabulary — Streamlit-default, low priority.
- Route glyphs (①/②'/③) shown on history cards have no visible legend anywhere in the UI decoding what they mean.
- `div.st-key-chip_row button`/`div.st-key-history_row button` still rely on padding rather than explicit `min-height` to clear 44px (unlike the three buttons fixed this round) — re-verified this still measures correctly (45.6px/94px), so not a defect, just a different technique achieving the same result; worth knowing if that padding is ever adjusted in isolation later.
- Static ARIA scanning continues to badly undercount real accessibility surface for this Streamlit app — reconfirmed methodological note carried from prior rounds.

## Questions to Consider

1. If "every answer carries route + evidence" is the app's own stated design principle, why does the answer itself have no dedicated CSS class or guaranteed render path, while "route" and "evidence" both do?
2. Was the compare-row removal checked against the non-BFS routes and the Naive RAG panel specifically, or only against the one path (BFS, majority-vote) where the answer happens to be findable inside a vote card?
3. Should "relevance" in the Evidence Ledger be redefined around what the answer actually cites, now that a concrete case exists where the current definition hides the one fact that proves a multihop answer?
