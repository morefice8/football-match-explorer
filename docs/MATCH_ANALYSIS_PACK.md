# Match Analysis Pack schema

## Pack schema 1.4

This is an additive schema update for shot auditability.

`tables/events-core.csv` adds:

- `shot_outcome`: canonical classification (`goal`, `saved`, `blocked`, `off_target`, `post`, `own_goal`, `unknown`);
- `shot_on_target`: nullable boolean. `true` and `false` are explicit values; blank means not applicable or unknown;
- `shot_blocked`: nullable derived boolean with the same blank-vs-false semantics;
- `shot_blocked_qualifier`: nullable boolean preserving Opta qualifier 82 (`Blocked`);
- `shot_keeper_saved_off_target`: nullable boolean preserving off-target-save evidence (qualifier 137/190);
- `shot_hit_woodwork`: nullable boolean preserving woodwork evidence;
- `shot_own_goal_qualifier`: nullable boolean preserving own-goal evidence;
- `shot_classification_issue`: optional ambiguity/consistency diagnostic.

Existing columns are retained. `tables/match-comparison.csv` also gains additive shot-breakdown fields through the canonical overview game profile (`saved_shots`, `blocked_shots`, `off_target_shots`, `woodwork_shots`, `unknown_shots`, `own_goals`).

`analysis-summary.json` keeps schema version 1.0 because `team_comparison` already permits additive metric keys; the same shot-breakdown fields are now emitted there.

Own goals are classified explicitly as `own_goal` and are not counted as a shooting-team attempt or shot on target. Contradictory semantic evidence is classified as `unknown` rather than silently coerced. For `Attempt Saved`, only Opta qualifier 82 (`Blocked`) is authoritative for the blocked-vs-save split; blocked x/y coordinates and defender/wall/hand context are non-authoritative and do not independently mark the shot blocked.
