# Handing a Match Analysis Pack to an AI assistant

This is the companion doc for the `*-match-analysis-pack.zip` produced by
`src/reporting/pack_builder.py` (schema per `docs/MATCH_ANALYSIS_PACK.md`).
Paste this doc's "Prompt templates" section (or link it) at the start of a
conversation with an AI assistant, alongside the pack, to get a good match
analysis with minimal back-and-forth.

## What's in the zip

| File | What it is | When to open it |
|---|---|---|
| `analysis-summary.json` | Compact, pre-computed tactical digest — the primary source. Team comparison, formations, passing, progression, crosses, buildup, defensive shape, PPDA, transitions, restarts, top-player rankings, a chronological shot-by-shot list with location and game state (leading/drawing/trailing), a chronological cards/discipline list, and a handful of full-detail "representative sequences" (actual event-by-event x/y data for the best buildup, defensive-transition, offensive-transition and restart sequence per team). | **Read this first, always.** It's designed to be consumed whole — it's capped at 1MB and never contains full event dumps. |
| `report-manifest.json` | Generation/integrity ledger: which sections/tables/figures were produced, empty, skipped, or failed, and the overall `pack_kind`. | Check `generation_status` (mirrored inside `analysis-summary.json` too) **before** trusting the rest of the pack — see below. |
| `<match>-report.pdf` | The branded editorial PDF with all figures (pitch maps, networks, heatmaps, Sankey-style cross routing, etc.) | Use for visual description or to sanity-check a claim against the actual chart. An AI without vision can't read this — flag it as unreadable if no image capability is available, don't guess at chart contents. |
| `tables/events-core.csv` | The full annotated event log (every event, ~1,600+ rows/match) with `shot_outcome`, `is_progressive`, `is_cross`, `is_key_pass`, sequence IDs, etc. already computed. | Only pull this in for a specific drill-down (e.g. "list every shot Napoli had after minute 70") — never summarize it wholesale, that's what `analysis-summary.json` is for. |
| `tables/player-rankings.csv` | Every metric family (passing/shooting/defending) for **every** player, not just the top 10 in the summary. | Use when you need a player outside the top 10, or a metric column not exposed in the summary ranking. |
| `tables/*.csv` (the rest) | Team-grain detail tables mirroring each summary section (match-comparison, pass-network-connections, progressive-passers, final-third-entries, cross-routes, buildup-summary, defensive/offensive-transitions, restarts, data-coverage). | Use only if `analysis-summary.json`'s equivalent section is truncated (e.g. cross routes cap at 8) and you need the full list. |

## Reading order for an AI assistant

1. Read `analysis-summary.json` top to bottom once — `match`, `score_and_goals` and `team_comparison` give the spine of the story; the rest are supporting detail.
2. Check `generation_status.pack_kind`. If it isn't `"complete"`, read `generation_status.section_data_statuses` / `section_composition_statuses` and **say explicitly which sections are missing or degraded** rather than writing around the gap silently.
3. Only open a CSV table or the PDF when the summary doesn't have enough resolution for the specific point you're making.
4. Use `shots` for concrete shot-by-shot narrative (who, when, where, what happened) rather than only citing aggregate shot counts.
5. Use `cards` for discipline narrative (who was booked/sent off, when, and whether it changed the game via a dismissal) — an empty list means no cards, not missing data.
6. Use each shot's `game_state` to say *when in the scoreline* a team created its chances — e.g. "all three of their clear-cut chances came while already behind" — rather than treating shot volume as state-independent.
7. Use `representative_sequences` (event-level x/y/outcome/receiver) to describe one or two concrete passages of play in prose — this is what turns a stats table into an actual tactical narrative ("in the 23rd move that built the goal, De Bruyne's through ball...").

## Glossary (pack-specific terms)

- **PPDA** (passes per defensive action) — lower means more aggressive pressing; computed per half in `ppda`.
- **Progressive pass** — a pass classified as moving the ball meaningfully upfield toward goal (see `progression`), not raw forward-pass count.
- **Zone 14 / half-space (`hs_left`/`hs_right`)** — classic final-third sub-zones in `final_third_entries.summary`; Zone 14 is the central area just outside the box, half-spaces are the vertical channels either side of the central lane.
- **Compactness / block height / width (`defensive_shape`)** — convex-hull-derived shape metrics of the team's defensive actions, split by half.
- **Terminal outcome vs. outcome** (buildup/transitions) — `outcome` is the human-readable label, `terminal_outcome` is its canonical machine category (`consolidated`, `turnover`, `shot`, `foul`, `out`, `offside`).
- **`data_quality`** — coverage stats for the underlying Opta feed itself (receiver resolution %, coordinate coverage, unmapped qualifier IDs). A low `receiver.coverage_pct` or a long `unmapped_qualifiers.ids` list means some claims in this pack rest on incomplete raw data — say so if it's materially low, don't present every number with equal confidence.
- **`shots`** — every classified shot (goal, miss, attempt saved, post) in chronological order, with minute, player, team, pitch location (`x`/`y`), `outcome` (`goal`/`saved`/`blocked`/`off_target`/`post`/`own_goal`/`unknown`), and `on_target`/`blocked`/`own_goal` flags. Use this for concrete shot narrative ("Napoli's best chance came in the 23rd minute...") instead of just citing the aggregate shot counts in `team_comparison`.
- **`cards`** — every yellow/red card in chronological order, with `card_type` (`yellow`/`second_yellow`/`red`/`unknown`), `resulted_in_dismissal` (true for a straight red or second yellow), and `rescinded` (true if the referee later cancelled the card — don't drop these rows, but don't treat a rescinded card as an active dismissal either).
- **`game_state`** (on each `shots` entry) — whether the shooting team was `leading`, `drawing`, or `trailing` the instant *before* that shot. A scoring shot reflects the state entering it, not the state its own goal produced. Scoped to shots only — no other metric in this pack is split by game state yet.

## What this pack does **not** contain (don't invent it)

- No expected threat (xT) or proprietary shot-level xG — Sportmonks-sourced xG (a separate, non-event data source) is not merged into this pack.
- No raw possession-time percentage (only pass counts/completion %).
- No general game-state split — only `shots` are tagged with "while leading/drawing/trailing"; passing, PPDA, defensive shape, etc. are still match-wide aggregates only.
- No pre-match mode yet — every pack assumes a completed match with a score.
- No external context: competition importance, league table position, recent form, head-to-head history, injuries/team news. If your analysis needs any of this, it must come from the person handing you the pack, not be inferred from the data.
- No xG per shot in `shots` — location and outcome are Opta-derived, but expected-goals values (when present at all) only exist as match-level Sportmonks totals elsewhere, never merged in per shot.

## Prompt templates

**Post-match article draft:**
> Here's the match analysis pack for [Team A] vs [Team B] ([competition], [date]). Read `analysis-summary.json` first per the reading order in `docs/AI_ANALYSIS_HANDOFF.md`. Write a [N]-word post-match tactical analysis in [tone/voice] focused on [angle, e.g. "why the press broke down after the 60th minute"]. Reference at least one concrete sequence from `representative_sequences`. Flag anything you're inferring beyond what the pack supports.

**Instagram carousel bullets:**
> Same pack. Instead of prose, give me [N] carousel slides: one punchy headline stat or claim per slide, ranked by how surprising/shareable it is, each under 15 words, in Italian. Use the actual computed numbers, don't round to the point of being misleading.

**Pre-match preview (until pre-match packs exist — see backlog):**
> This pack is post-match, but I want a pre-match preview for the *next* fixture. Use only the tactical *tendencies* here (pressing shape, buildup patterns, set-piece routines) as historical baseline — do not treat the score or goals section as relevant to the upcoming match.

## Known pack limitations tracked as backlog items

See the project's task backlog (analyst-hat: xT model, shot-level xG, game-state splits, possession %; dev-hat: pre-match pack mode) for planned fixes to the gaps listed above.
