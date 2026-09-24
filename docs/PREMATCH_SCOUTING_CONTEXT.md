# Pre-match scouting context checklist

No Opta event file — for either team, any competition — contains this
information. It has to come from Michele. Fill this in once per scouting job
and hand it over alongside the match analysis packs; an AI assistant should
never guess or infer these fields from the event data.

Copy the template below, fill in what you know, and say explicitly which
fields you don't have an answer for — "unknown" is a valid answer and is
more useful than a guess.

```
## Fixture
- Teams / competition / date:
- Venue, and which side is at home:

## Stakes
- What's on the line for each side (league position fight, cup knockout,
  dead rubber, derby, etc.):

## Form and table position
- [Opponent]'s current league position and result trend (last 5):
- [Own team]'s current league position and result trend (last 5):

## Expected lineup and system
- [Opponent]'s manager and expected formation/system for this match:
- Confirmed injuries / suspensions / rotation risk (opponent):
- Confirmed injuries / suspensions / rotation risk (own team):
- Any recent tactical change (new manager, formation switch, new signing
  forced into the XI) that makes older matches less representative:

## History
- Reverse fixture result/date this season, if already played:
- Any head-to-head pattern worth knowing beyond the reverse fixture:

## Article angle
- Working angle for the piece, if you already have one (leave blank if not):
- Format: article (micheleorefice.com) / Instagram carousel / both:
```

## Why this exists

The reporting pack (`analysis-summary.json`) is deliberately scoped to what
the raw event data can support — see "What this pack does not contain" in
`docs/AI_ANALYSIS_HANDOFF.md`. Table position, injuries, and tactical news
are exactly the kind of thing that data can't answer, and forgetting to
mention one (e.g. an injury that changes the expected shape) is the failure
mode this checklist exists to prevent. See the project's scouting workflow
notes for how this fits into a full pre-match analysis: Michele names the
upcoming fixture, the AI assistant researches and requests the specific past
matches worth analysing, Michele supplies the raw event files plus this
checklist, and only then does synthesis/article drafting start.
