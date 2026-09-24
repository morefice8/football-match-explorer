# Pre-match scouting context checklist

Two different kinds of "context the event data can't give you" exist here,
and they need different handling:

1. **Public information** (results, table position, fixture congestion,
   manager tenure, well-reported injuries, head-to-head) — the AI assistant
   should research this itself via web search when a real scouting job
   starts, not ask Michele to compile it. See "What the AI assistant
   researches" below.
2. **Genuinely non-public or creative information** — only Michele can
   supply this. See "What Michele needs to provide" below.

## What the AI assistant researches (don't ask Michele for these)

- Recent results and current form (last 5–10 matches) for both sides
- League table position and what's at stake (title race, relegation fight,
  dead rubber, derby)
- Fixture congestion — European competition or deep cup run causing a
  3-day turnaround, which affects rotation and fatigue
- Whether the manager is new this season, or there's been a recent
  sacking/appointment
- Well-reported injuries/suspensions to key, first-team players
- The reverse fixture result and general head-to-head pattern, if played

**Caveats that always apply to this research, stated explicitly rather than
buried:**
- Coverage quality varies by league. Serie A-level clubs are well covered;
  a lower-profile opponent or smaller league may have thin or contradictory
  search results — say so rather than presenting a guess as settled.
- Research done when the fixture is first named goes stale fast, especially
  injury news. It must be re-checked close to matchday (ideally within 2–3
  days), not trusted at the two-week mark.
- The exact starting XI is never truly knowable this far out — official
  team news only firms up 24–48h before kickoff, and lineups are only
  confirmed at the team-sheet stage ~1 hour before. Any "expected lineup"
  offered ahead of that is a best-effort estimate and must be labeled as
  one, not presented as fact.

## What Michele needs to provide

Fill in what applies and say "unknown" rather than guessing — that's more
useful than a wrong answer.

```
## Corrections
- Anything the AI assistant's research got wrong or missed:
- Anything non-public you know (training-ground news, insider info) that
  changes the picture:

## Article angle
- Working angle for the piece, if you already have one (leave blank if not):
- Format: article (micheleorefice.com) / Instagram carousel / both:
```

## Why this exists

The reporting pack (`analysis-summary.json`) is deliberately scoped to what
the raw event data can support — see "What this pack does not contain" in
`docs/AI_ANALYSIS_HANDOFF.md`. Some of that gap (table position, form,
well-reported injuries) is actually public information the AI assistant can
and should go find; only the genuinely non-public or creative pieces need
to come from Michele. See the project's scouting workflow notes for how
this fits into a full pre-match analysis.
