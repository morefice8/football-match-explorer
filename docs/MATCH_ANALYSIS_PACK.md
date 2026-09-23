# Match Analysis Pack schema

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
