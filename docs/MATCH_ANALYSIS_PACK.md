# Match Analysis Pack schema

## Pack schema 1.12

Additive update: the pack gains a **window analysis** that reads the match in
phases and answers post-match questions with numbers an article can quote:
*how did each team access the final third and the box, where did the opponent
get hands on the play, where did each player act, how did the defensive block
move after the goals and the changes, and how long did a team go without a shot
or a touch in the box?*

Seven new tables (all built by `src/reporting/window_analysis.py` from the same
event frame as `events-core.csv`; scope `phase_id = 0` is the whole match,
`1..n` are the match phases):

- `tables/match-phases.csv`: the phases. The match is cut at goals,
  dismissals and substitution clusters (merged when closer than 3 / 10 minutes,
  a goal or dismissal always wins over a nearby substitution), each with its
  score state, length in **minutes of play**, the trigger and who came on.
- `tables/window-profile.csv`: one row per team and scope: touches and share,
  final-third and box touches, completed final-third entries, crosses, shots,
  loss share in the final third, block height, team length, pressed-touch share.
- `tables/pressure-profile.csv`: share of a team's touches followed by an
  opponent intervention, by all / lane / third / player, with Wilson 95%
  intervals, a `low_sample` flag (< 20 touches) and a Fisher test against the
  rest of the group.
- `tables/pressure-touches.csv`: one row per touch with who intervened and
  where (the plotting base for pressure maps; goalkeeper touches included).
- `tables/window-positions.csv`: median location of each outfield player's
  touches and interventions per scope, in the team's own frame and in a shared
  frame (home team unchanged, away team rotated 180 degrees, so both teams can
  be drawn on one pitch facing each other).
- `tables/shot-droughts.csv`: stretches of at least 15 minutes of play without
  a shot or without a touch in the opponent's box, with what the team did
  meanwhile.
- `tables/insight-candidates.csv`: ranked findings that pass an explicit size
  and evidence bar, each with its numbers and its caveat. They are article
  material, not conclusions.

`analysis-summary.json` moves from schema 1.7 to 1.8 with top-level
`match_phases`, `window_profile`, `pressure_profile`, `shot_droughts` and
`insight_candidates` sections.

### Definitions (all in `src/metrics`)

- **Play clock** (`match_clock.py`). Opta restarts the clock at 45:00 in the
  second half even when the first half ran into stoppage time, so
  `minute * 60 + second` overlaps across the interval. Windows, durations and
  every "within N seconds" pairing use `play_seconds` (periods laid end to end)
  and pair events only inside one period. Durations are minutes of play: the
  26'-59' stretch of a match with 2 minutes of first-half stoppage is 36
  minutes, not 34.
- **Zones** (`pitch_zones.py`). Five lanes by y (right = low y, penalty-area
  width for the half-spaces), thirds by x, the penalty area in true metres. Lane
  statistics are in the touching team's own attacking frame.
- **Touch** = pass, take-on, ball touch, shot, dispossession. **Intervention** =
  an opponent tackle, challenge, interception, blocked pass, ball recovery,
  clearance or committed foul.
- **Pressed touch** = a touch followed by an opponent intervention within 6 s
  and 10 m in the same period. It measures where the opponent got hands on the
  play, and it also counts balls lost and recovered next to the touch. Pressure
  grows towards the opponent's area (in Villarreal-Napoli: 1% of touches in the
  own third, 26-30% in the final third), so compare like with like: the pressure
  table carries lanes and thirds for that.
- **Block height** = median distance (m) from the own goal of the team's own
  interventions, goalkeepers excluded. It uses the intervention set above, not
  `defensive_metrics.get_defensive_actions` (which also counts committed fouls,
  errors and defensive-third aerials), so it can differ from the Defensive Shape
  figures by a metre or two.
- **Position** = where a player acts (median of touches and interventions), not
  where he stands; players need 12 minutes in the scope (or the whole of a
  shorter scope), and `low_sample` marks fewer than 8 actions. **Team length** =
  deepest to highest median position, only from at least 7 players.
- **Drought** gaps are measured between the team's consecutive shots (or box
  touches); the first gap starts at kick-off and the last ends at the final whistle
  (`ends_with_event` is false for it).

### Reading rules

- Quote every share with its counts and interval; a `low_sample` row is an
  anecdote.
- State the basis of every metre figure (touches only, or touches plus
  interventions): they differ by several metres for the same player.
- The insight rules and their thresholds are in
  `src/metrics/insight_candidates.py` (hotspot player: at least 25 touches, twice
  the team average and at least 12 points more, Fisher p < 0.05, "strong" below
  0.01; lane gap: lanes with at least 40 touches, 15 points apart, p < 0.05;
  phase shifts of at least 10 m with at least 10 interventions; droughts of at
  least 20 minutes).
- Everything is one match: it says what happened, not what the team usually does.

### Insight sheet for articles

`py -3.11 scripts/pack_insights.py pack.zip --out sheet.md` (optionally
`--team "Napoli"`) turns a pack into a markdown working sheet: the phases, the
ranked insight candidates with their numbers, a numbered caveat and the
questions to settle before writing each one, the phase-by-phase and pressure
tables (players as name and shirt number) and the droughts. It reads only the
pack tables (`src/reporting/insight_sheet.py`) and adds no analysis of its own.
Packs older than schema 1.12 are reported and need rebuilding.

### Defensive action filter (corrected)

`defensive_metrics.get_defensive_actions` used to list `Blocked pass` while the
data type is `Blocked Pass`, so blocked passes never counted, and it counted
every `Foul` row (committed and suffered). It now matches type names ignoring
case and keeps only committed fouls (outcome `Unsuccessful`), as the PPDA count
already did. Reports and Defensive Shape / density figures built before this
change are not comparable with new ones: on Villarreal-Napoli the Villarreal
action count goes 137 to 130 (block height 28.5 to 26.9 m) and the Napoli one
121 to 120 (34.1 to 35.1 m).

Still open: the player-level defensive lists in `app.py`, `player_plots.py`,
`player_metrics.py` and `config.py` spell `Blocked pass` too, and
`game_state.py` / `player_metrics` still compare printed clocks and could be
moved to `match_clock` in a follow-up.

## Pack schema 1.11

Additive update: the pack gains a **phase profile**, answering two scouting
questions per pitch third (defensive / middle / attacking):

1. *Where the team wins the ball, how does it attack?* (first pass forward,
   whether and how fast it reaches the final third, metres gained, shot rate)
2. *Where the team loses the ball, how does it win it back?* (regained, regained
   within 5 and 10 seconds, median seconds to regain, how often the opponent
   reaches the final third or shoots before the ball is won back)

Two new tables, both built from the transition sequences the pack already
contains (no new detection logic):

- `tables/phase-sequences.csv`, one row per recovery or loss sequence, the
  auditable base of the profile. Concatenating these tables from several
  matches and re-running `src.metrics.phase_profile.summarize_recoveries` /
  `summarize_losses` gives a correctly weighted multi-match profile.
- `tables/phase-profile.csv`, one row per team, phase (`recovery`/`loss`) and
  pitch third (plus an `All` row). `low_sample` is true when a cell has fewer
  than 10 sequences; read those cells as anecdotes, not tendencies.

`analysis-summary.json` moves from schema 1.6 to 1.7 with a new top-level
`phase_profile` section (`teams[].recoveries` / `teams[].losses`).

Definitions and thresholds live in `src/metrics/phase_profile.py`
(final-third line 66.67, "forward" first pass = at least 10 pitch units, quick
regain = 5 s). Reaching the final third and shot rates use the same
"controlled location" milestones as the existing transition KPIs, so the new
table cannot disagree with them. Event data has no off-ball positions: this
profile says *how fast and where*, not *how the rest defence is organised*.

## Pack schema 1.10

This is an additive schema update extending game-state splits (introduced in
schema 1.9) from shots-only to also cover passing volume and accuracy. It is
still deliberately scoped to two metrics, not every metric in the pack.

`analysis-summary.json` moves from schema 1.5 to 1.6: each entry in
`game_state_splits` gains a `passing` object keyed by `leading`/`drawing`/
`trailing`, each with `attempted`, `completed` and `completion_pct` (null
when a team attempted zero passes in that state). Passes are re-tagged with
the same per-instant game-state logic already used for shots, reusing the
same `pass-locations` frame the pack's own `passing` section and the live
app already read — no new pass-detection logic, no new football semantics.

## Pack schema 1.9

This is an additive schema update that aggregates the per-shot `game_state`
tag introduced in schema 1.8. It is deliberately scoped to shots only — it
does not split every metric in the pack by game state.

`analysis-summary.json` moves from schema 1.4 to 1.5, adding a top-level
`game_state_splits` array (one entry per team) that aggregates `shots` and
`goals` counts by `leading`/`drawing`/`trailing` state. It is derived
directly from the already-tagged `shots` list, so the two can never
disagree — no independent computation, no new football semantics.

## Pack schema 1.8

This is an additive schema update for game-state (leading/drawing/trailing)
narrative in the AI-oriented summary. It is deliberately scoped to shots
only — it does not split every metric in the pack by game state.

`analysis-summary.json` moves from schema 1.3 to 1.4: each entry in `shots`
gains a `game_state` field (`leading`, `drawing` or `trailing`, from the
shooting team's own perspective, evaluated the instant *before* that shot —
a scoring shot itself reflects the state it was taken in, not the state its
own goal just created).

The running scoreline is derived entirely from `score_and_goals.goals`
(which already attributes own goals to the correct beneficiary team) — no
new football semantics, no external data source.

## Pack schema 1.7

This is an additive schema update for discipline (cards) narrative in the
AI-oriented summary.

`analysis-summary.json` moves from schema 1.2 to 1.3, adding a top-level
`cards` array: one compact, chronologically-ordered entry per Opta `Card`
event (typeId 17), each with `event_id`, `period`, `minute`, `second`,
`team`, `player`, `card_type` (`yellow`/`second_yellow`/`red`/`unknown`),
`resulted_in_dismissal`, and `rescinded`. Classification is handled by a new
shared module, `src/metrics/card_events.py`, mirroring the existing
`shot_classification.py` pattern: qualifiers 31/32/33 (yellow / second
yellow / red) are the canonical discriminators, and qualifier 171
(`Rescinded card`) is surfaced as a flag rather than silently dropping the
card. The array is capped at 20 entries. A match with no cards produces an
empty array — that is a legitimate outcome, not a failure.

## Pack schema 1.6

This is an additive schema update for shot-level narrative in the AI-oriented
summary.

`analysis-summary.json` moves from schema 1.1 to 1.2, adding a top-level
`shots` array: one compact, chronologically-ordered entry per classified shot
event (goal, miss, attempt saved, post), each with `event_id`, `period`,
`minute`, `second`, `team`, `player`, `x`, `y`, `outcome`
(the canonical `shot_outcome` value), `on_target`, `blocked`, `own_goal`, and
`distance_m`. The list is a direct projection of the existing
`shot_classification.classify_shots` output already used for
`tables/events-core.csv` and `tables/match-comparison.csv` — no new football
semantics are introduced, only surfaced. The array is capped at 60 entries.

`distance_m` (and the `avg_shot_distance` KPI, and the shot-map hover) is the
straight-line distance to the centre of the goal line, computed by
`src/metrics/shot_geometry.py` with each axis scaled by its own pitch dimension
(1 x-unit = 1.05 m, 1 y-unit = 0.68 m on a 105 x 68 m pitch). Earlier builds
scaled both axes by the pitch length, which overstated every off-centre shot
(a shot from the edge of the box on the wing read ~25 m instead of ~16 m), so
packs generated before this fix are not comparable on distance.

## Pack schema 1.5

This release adds final-composition accounting. Data availability and final PDF
composition are now separate states for declared report content.

`report-manifest.json` keeps the existing section `generation.status` field and
adds explicit status dimensions:

- `data_status`: availability/result of neutral data generation;
- `composition_status`: result of final PDF composition;
- `status`: final item status, which becomes `error` when either required data
  generation or final composition fails.

The manifest now records section and table generation metadata, plus a
`generation.content` audit list covering sections, tables and figures. The
existing `generation.artifacts` figure list remains for backward compatibility.

Top-level generation metadata adds:

- `complete`: `false` when any required content failed;
- `pack_kind`: `complete`, `complete-with-warnings`, or
  `incomplete-diagnostic`;
- section/table/figure expected, generated, empty, skipped and failed counts;
- `required_content_failed` and `optional_content_failed`;
- `section_data_statuses` and `section_composition_statuses`.

A legitimately empty item is not a failure. A failed optional item produces a
warning. A failed required item produces an incomplete diagnostic pack and
must never be surfaced as a successful generation.

`analysis-summary.json` moves from schema 1.0 to 1.1 and mirrors the additive
final-generation status fields above. Analytical football content is unchanged.

Technical exception detail remains in application logs. PDF placeholders,
manifest error messages and UI messages use concise non-technical text.

## Pack schema 1.4

This is an additive schema update for shot auditability.

`tables/events-core.csv` adds:

- `shot_outcome`: canonical classification (`goal`, `saved`, `blocked`, `off_target`, `post`, `own_goal`, `unknown`);
- `shot_on_target`: nullable boolean. `true` and `false` are explicit values; blank means not applicable or unknown;
- `shot_blocked`: nullable boolean with the same blank-vs-false semantics.

Existing columns are retained. `tables/match-comparison.csv` also gains additive
shot-breakdown fields through the canonical overview game profile
(`saved_shots`, `blocked_shots`, `off_target_shots`, `woodwork_shots`,
`unknown_shots`, `own_goals`).

Own goals are classified explicitly as `own_goal` and are not counted as a
shooting-team attempt or shot on target. Contradictory qualifier evidence is
classified as `unknown` rather than silently coerced.
