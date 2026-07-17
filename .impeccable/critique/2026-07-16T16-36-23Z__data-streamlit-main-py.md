---
target: data/streamlit/main.py
total_score: 26
p0_count: 0
p1_count: 2
timestamp: 2026-07-16T16-36-23Z
slug: data-streamlit-main-py
---
## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3/4 | Custom char counter goes stale on an emptied textarea while Streamlit's own tooltip already shows 0/600; "RETRIEVED CHUNKS · TOP-5" label showed only 3 chunks live with no note on why |
| 2 | Match System / Real World | 3/4 | Evidence Ledger shows raw DocRED enum strings verbatim (`LOCATED_IN_THE_ADMINISTRATIVE_TERRITORIAL_ENTITY`) — a register break against every other translated-jargon surface in the app |
| 3 | User Control and Freedom | 3/4 | No modals to escape, history/tabs work well; nested Evidence Ledger scroll region traps the mouse wheel |
| 4 | Consistency and Standards | 2/4 | Token system is disciplined everywhere except focus states — `:hover` is defined for every button class, `:focus`/`:focus-visible` never is, so unstyled Streamlit red bleeds through app-wide |
| 5 | Error Prevention | 3/4 | Duplicate-submission guard verified working live by both independent assessments; primary button stays clickable (not visually disabled) on empty input |
| 6 | Recognition Rather Than Recall | 2/4 | No visible legend for the 6 entity-type graph colors — only discoverable via node-hover, invisible to a passive demo viewer |
| 7 | Flexibility and Efficiency | 2/4 | Caching avoids duplicate spend, suggested chips help; still no fast-iteration/low-cost mode or bulk export |
| 8 | Aesthetic and Minimalist Design | 3/4 | Restrained and on-brand; undercut by the un-hidden Streamlit toolbar and 54-item uniform-weight Evidence Ledger noise |
| 9 | Error Recovery | 3/4 | `role="alert"` error card verified well-designed in code; no visible try/except around the live OpenAI calls inside the spinner, so an API hiccup mid-demo risks a raw traceback (untested) |
| 10 | Help and Documentation | 2/4 | The technical-detail expander is a strong contextual-help pattern, but only appears after a query runs; pipeline-box jargon ("라우팅(스캔/1-hop/멀티홉)") is unexplained before that |
| **Total** | | **26/40** | **Acceptable — real fixes landed, offset by newly-surfaced issues** |

## Anti-Patterns Verdict

**No — this does not read as AI-generated**, confirmed independently by both assessments again this round. The forest-green/cream/amber system and bespoke vocabulary (TRACE, QUERY INTELLIGENCE CONSOLE, EVIDENCE LEDGER) read as authored, not templated. Zero gradient text, zero glassmorphism, zero hero-metric template, zero side-stripe accent borders (the only `border-right` usage is a neutral 1px stat-cell divider, not a colored accent), zero identical decorative card grids.

**Product-register slop check** (would a Linear/Notion/Stripe-fluent user trust this or pause at something subtly off?) surfaced two real stumbles neither assessment found last round:
- Tabbing to the primary CTA or a chip turns it **unstyled Streamlit red** — the same button reads as two different components depending on mouse vs. keyboard input, and collides with the app's own `--danger` semantic color.
- Streamlit's default "Deploy"/"⋮" toolbar is never hidden, sitting in the exact corner the custom "TRACE" header works to own — outs the framework before a query even runs.

**Deterministic scan**: the bundled `detect.mjs` is still not installed in this environment (`Error: bundled detector not found`, same as last run) — not a skipped step, a genuinely broken entrypoint, so Assessment B built an ad-hoc static scan and cross-validated it against the live DOM, same substitute method as before:
- 34 hex-color literals total, **100% accounted for**: 21 live inside `:root` tokens, the rest are pyvis Python-API params (`TYPE_COLORS`, `Network()` ctor, node/edge colors) that structurally can't consume CSS variables — zero real token-system violations.
- 59 `!important` declarations across 36 lines — almost all justified (overriding Streamlit/BaseWeb's own injected styles, a documented necessary pattern here), except three sites (`.muted`, `.entity-chip`, `.evidence-tag`) that lack the explanatory comment the rest of the file consistently uses.
- Exactly 1 `@media` query, unchanged from last round — still only reflows `.stat-cell`, nothing else in the layout has custom responsive handling.
- 1 functional emoji in rendered content (🔧 on the technical-detail expander), unchanged from last round; `page_icon="🕸️"` correctly excluded as tab-icon metadata, not content.
- Zero heading-hierarchy skips, confirmed by both static trace and live DOM order.
- Zero browser console errors or warnings, confirmed independently by both assessments across two different full pipeline runs.
- Static ARIA scan again badly undercounts reality: 1 authored `role` in source vs. **~265 distinct labeled nodes** in the live accessibility tree (BaseWeb injects `tablist`/`tab`/`tabpanel`/`textbox`/proper heading roles at runtime). Reconfirmed methodological point from last round: don't judge this app's a11y surface from source grep alone.

**Live visual evidence, two concrete fix verifications**:
- The hop-distance graph-dimming fix from last round is **confirmed actually working**, not just present in source — Assessment B read the live vis-network DataSet directly and found the one 2-hop node ("Turkish") rendered at `opacity:0.7` against `opacity:1.0` seed/1-hop nodes, visually perceptible in a zoomed screenshot.
- The weak-consensus visual-distinction fix from last round is **confirmed actually working under real conditions** — Assessment A happened to land on a genuine 1/3 vote split live (AirAsia Zest question) and saw the stat cell turn red, the caveat copy appear, and the "3회 샘플링 결과" detail correctly explain the disagreement. Assessment B's run landed 3/3 unanimous and explicitly said it could not observe this path rather than guessing — a good example of the two assessments' evidence combining into full coverage neither had alone.
- The duplicate-submission guard is **confirmed working by both assessments independently**, on different queries, one via double-click and one via triple-click — strong cross-validation.
- WCAG contrast fix on the split `--text-dim`/`--text-muted` tokens is **confirmed passing**, with actual resolved colors extracted live: cream-card caption text measures 5.98:1, dark-surface `.muted` text measures 6.12–6.46:1, and a sanity check confirmed the *old* single-token value would have failed at 2.32:1 if misapplied to a dark background — the split was the correct fix, not just a color tweak.

## Overall Impression

This is a real second-cycle result, not a status-quo re-critique: every P1 and P2 from the 2026-07-16 critique (tabs-vs-columns, invisible weak consensus, WCAG contrast, duplicate submission, unfiltered graph noise, buried entity-match detail) is now live-verified fixed, several under adversarial re-testing. The score barely moved (25 → 26/40) not because the work didn't land, but because a rigorous independent pass surfaces its own new floor every time — here, an unstyled keyboard-focus state that collides with the app's own danger color, and an Evidence Ledger that never got the same relevance-signal treatment its sibling graph view already has. The single biggest opportunity now is applying the *exact pattern already proven correct in the graph* (fade by relevance, computed from data already in hand) to the Evidence Ledger, the one screen a skeptical viewer trusts most and currently has the least signal on.

## What's Working

1. **The side-by-side compare row does the tool's entire persuasive job in one glance** — verified live with a real asymmetric result: GraphRAG returned a full sourced answer while Naive RAG said "모름" despite having a directly relevant top-ranked chunk. No narration needed; this screen alone demonstrates why GraphRAG exists.
2. **Every claimed fix from the last critique round was re-tested live, not just re-read in source, and held up** — hop-distance graph dimming, weak-consensus red styling, and the duplicate-submission guard all fired correctly under real, adversarial (double/triple-click) conditions across two independent sessions.
3. **The plain-language/technical-detail two-tier disclosure remains a genuinely non-obvious, well-executed pattern** — same underlying numbers surface at both the glance level and the full-audit level, with the entity-match cascade detail (Exact/Alias/Word Boundary/Fuzzy) now actually rendered instead of sitting as dead CSS.

## Priority Issues

**[P1] Focus states use unstyled Streamlit red, not the app's palette.**
*Why it matters*: verified live on both the primary CTA (turns solid Streamlit red on Tab-focus) and secondary chip buttons (red outline). The stylesheet defines `:hover` for every button selector but never `:focus`/`:focus-visible`, so keyboard navigation collides visually with the app's own `--danger`/weak-consensus semantic color — a focused primary button reads as an error state, at the exact moment a keyboard user is about to trigger the app's most important action.
*Fix*: add `:focus`/`:focus-visible` rules mirroring the existing `:hover` amber treatment to every button selector in `inject_css()` (`div.stButton > button`, `div.st-key-chip_row button`, `div.st-key-history_row button`, `button[data-baseweb="tab"]`).
*Suggested command*: `/impeccable polish`

**[P1] Evidence Ledger signal-to-noise undermines the core "audit" promise.**
*Why it matters*: live-verified on the AirAsia Zest question — of 54 evidence cards, most are repetitive administrative-hierarchy trivia and entries E51–E54 are about an unrelated video game, all rendered with identical visual weight. The graph view already solved this exact problem last round (hop-distance opacity fading, confirmed working), but the Evidence Ledger — the screen a user actually goes to *verify* an answer, not just glance at — never got the same treatment.
*Fix*: reuse the already-computed hop-distance values from `_hop_distances()` to visually de-emphasize (or group behind a "+N more, lower relevance" disclosure) evidence cards far from the seed entity, mirroring the graph's own pattern.
*Suggested command*: `/impeccable clarify`

**[P2] Streamlit's default toolbar is not hidden.**
*Why it matters*: "Deploy" + "⋮" sit in the exact top-right corner the custom "TRACE" header is designed to own, on every screen — for the "발표 참관 평가자" persona this app is explicitly built to present to, it's the very first thing on screen and immediately signals "generic Streamlit app" before the custom identity gets a chance to land.
*Fix*: set `toolbarMode = "minimal"` in `.streamlit/config.toml`, or hide `[data-testid="stToolbar"]` via CSS.
*Suggested command*: `/impeccable polish`

**[P2] pyvis graph renders a persistent unstyled white bar at the top of the canvas.**
*Why it matters*: confirmed via zoomed screenshot — a thin cream/white strip sits above the vis-network canvas on every graph render, inside what is otherwise a fully controlled dark surface (the same file already patched CDN-inlining and physics-auto-stop into this exact generated HTML). Reads as a rendering glitch at the precise moment the app is trying to look most polished.
*Fix*: extend the existing `html.replace(...)` patch in `build_graph_html()` to also set the generated document's body background/margin to match `bgcolor`.
*Suggested command*: `/impeccable polish`

**[P2] No visible legend for the graph's 6 entity-type colors.**
*Why it matters*: `TYPE_COLORS`/`TYPE_LABELS_KO` meaning is only discoverable via per-node hover — invisible to the "발표 참관 평가자" persona, who by definition never touches the mouse to hover a node. The sr-only accessibility text ironically carries more type information than the default sighted view.
*Fix*: render a small always-visible legend chip row above or beside the graph using the already-defined `TYPE_COLORS`/`TYPE_LABELS_KO` maps.
*Suggested command*: `/impeccable clarify`

## Persona Red Flags

**연구팀 본인 (researcher debugging the pipeline locally)**
- The Evidence Ledger noise problem hurts this persona most directly: diagnosing *why* BFS pulled in an unrelated video game means manually scrolling a 54-item, mouse-wheel-trapping container with zero relevance signal.
- Genuine win this round: the entity-match cascade detail (mention → matched entity via Exact/Alias/Word Boundary/Fuzzy) is now actually rendered in the expander instead of sitting as dead, unused CSS — exactly the debugging data this persona needed, confirmed live.

**발표 참관 평가자 (professor/evaluator watching a live demo, hands off keyboard)**
- Never hovers a graph node, so the 6-color entity-type legend is effectively invisible to them — six colors that read as decorative rather than informative from where they're sitting.
- Sees "Deploy" in the header before any query even runs, before the custom "TRACE" identity gets a chance to establish itself.
- If anyone in the room tabs to a button instead of clicking, the unstyled red focus flash could plausibly read as "something just broke" at the highest-stakes moment of a live demo.

**Sam (accessibility-dependent user)**
- The systemic red focus state is the most consequential finding for this persona specifically: focus visibility technically exists (not a hard WCAG failure), but its color actively fights the app's own semantic "red = danger/weak consensus" convention, which is a confusing signal, not just an aesthetic mismatch.
- Genuine, verified wins this round: `role="alert"` on the error card, the split contrast tokens (5.98–6.46:1, both passing AA with margin), and sr-only graph summary text are all real, working accessibility investments.

## Minor Observations

- Nested Evidence Ledger scroll container traps the mouse wheel — scrolling inside it never moves the outer page; the cursor has to leave the container entirely to escape.
- Custom "N/600" character counter can go stale relative to Streamlit's own native live tooltip after clearing the textarea.
- Evidence Ledger's displayed count (54, includes duplicate triples) and the graph's deduped edge count can disagree — the two panels don't always tell the same "how many facts" story.
- "RETRIEVED CHUNKS · TOP-5" label showed only 3 chunks live with no explanation for the shortfall (filtered metadata vs. genuinely fewer matches).
- No "run another question" affordance near the bottom of a long result — a presenter mid-demo has to scroll all the way back to the top for the next query.
- Primary button stays fully clickable on an empty textarea rather than visually disabling — relies on a silent no-op instead of preventing the click.
- Three `!important` sites (`.muted`, `.entity-chip`, `.evidence-tag`) lack the explanatory comment this file otherwise consistently attaches to every other `!important` use.
- Still open from the prior round, untouched by this diff: `_route_icon()` on the `no_seed` route still returns a fragment ("매칭된") rather than a real marker; `.hero-title .accent { color: inherit; }` still gives the "경로"/"증거" hero accent spans zero visual distinction from surrounding text; the 🔧 emoji in the expander label is still the one remaining functional-emoji instance in an app whose comments document removing this exact pattern elsewhere.

## Questions to Consider

1. The Evidence Ledger already has the hop-distance data it needs to solve its own noise problem, because the graph view next to it solved the identical problem with the identical data last round — why did one panel get the fix and not its sibling, and is there a reason to treat them differently?
2. The file's own comment history shows a rigorous, measured first critique pass (contrast ratios, hover states, physics auto-stop). Was that entire pass done exclusively with a mouse? If keyboard navigation had been part of that test, would the focus-state gap have surfaced then instead of now?
3. Is "54 sources" actually more honest as an unfiltered researcher debug view, or does presenting it as if all 54 are equally load-bearing evidence quietly break the "auditable" promise for anyone who doesn't scroll to the bottom?
