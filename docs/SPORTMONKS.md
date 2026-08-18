# Sportmonks integration

## Quick start (Windows PowerShell)

```powershell
python -m pip install -r .\requirements-app.txt
Copy-Item .\.env.example .\.env
notepad .\.env
python .\import_sportmonks.py --season 2025-2026 --league serie-a --max-fixtures 2
python .\import_sportmonks.py --season 2025-2026
python .\app.py
```

The `.env` file must contain `SPORTMONKS_API_TOKEN=...`. Never place the token
in source code or commit it.

## Importer behaviour

- API authentication is sent as the raw token in the `Authorization` header.
- JSON responses are cached atomically under
  `data/sportmonks/raw/<season>/<league>/`.
- A repeated command reuses successful responses. `--refresh` intentionally
  bypasses the cache for the selected scope.
- League-specific imports replace that league and preserve other already
  processed Sportmonks leagues for the same season.
- A failed import of every selected league stops before changing existing
  processed datasets.
- HTTP 429 and transient server errors are retried with backoff.
- The token is never stored in cached responses, reports, URLs or errors.

Run `python .\import_sportmonks.py --help` for all CLI options.

## Output datasets

Canonical CSV and Parquet files are written under
`data/sportmonks/processed/<season>/`:

| Dataset | Grain |
| --- | --- |
| `leagues` | One row per league and season |
| `teams` | One row per team and season |
| `players` | One row per Sportmonks player ID |
| `squads` | One row per team/player registration |
| `fixtures` | One row per fixture, including scheduled fixtures |
| `standings` | One row per team in the current standings |
| `events` | One row per goal/card/substitution/VAR event |
| `team_matchlogs` | One row per fixture and team |
| `player_matchlogs` | One row per fixture, team and lineup player |
| `team_stats` | Season aggregation per team |
| `player_stats` | Season aggregation per player and league |
| `metric_catalog` | Sportmonks type ID to stable column-name mapping |

The app-compatible files are:

- `data/processed/team_stats_<season>.parquet`
- `data/processed/player_stats_<season>.parquet`

CSV copies are written alongside every Parquet file for inspection.

## Metrics used by Team Statistics

All comparisons use only detailed fixtures already imported. Index-only
fixtures never contribute to matches played or season aggregates.

| Area | Metrics shown | Calculation |
| --- | --- | --- |
| Attacking | Goals p90, xG p90, xG/shot, conversion % | Goals and Sportmonks xG divided by team 90s; xG/shots; goals/shots |
| Possession | Possession %, pass completion %, final-third passes p90, dribble success % | Mean fixture possession; successful/attempted passes; lineup final-third passes/90; successful/attempted dribbles |
| Defending | GA p90, xGA p90, shots against p90, tackles won + interceptions p90 | Scores and opponent fixture statistics, normalized by team 90s |

The quadrant plots and team radar reuse these exact columns. GA, xGA and shots
against are ranked with lower values treated as better.

## Metrics used by Player Statistics

| Area | Metrics shown | Calculation |
| --- | --- | --- |
| Attacking | Goals p90, xG p90, G-xG p90, shots on target p90 | Additive lineup totals divided by player 90s |
| Creation | Assists p90, chances created p90, final-third passes p90, pass completion % | Sportmonks lineup fields; accurate/attempted passes for completion |
| Defending | Tackles won p90, interceptions p90, clearances p90, aerial-duel win % | Sportmonks lineup fields normalized by player 90s or attempts |
| Goalkeeping | Save %, xGoT faced-GA p90, saves p90, GA p90 | Saves/shots faced; opponent team xGoT allocated by goalkeeper minutes; goalkeeper totals per 90 |

`xGoT faced-GA p90` is deliberately not labelled PSxG. Sportmonks supplies xG
on target at team level; when more than one goalkeeper appears, the importer
allocates it in proportion to minutes played. The compatibility aliases
`PSxG` and `PSxG+/-` remain only so older saved views do not break.

Metric labels, tooltips, card formats, ranking direction, quadrants and radar
dimensions are centralised in `src/metrics/sportmonks.py`.

After updating the importer or metric definitions, rerun the import command to
regenerate the processed Parquet files. Cached API responses are reused, so
`--refresh` is not required:

```powershell
python .\import_sportmonks.py --season 2025-2026
```

## Source boundaries

Sportmonks is used for competition tables, squads, fixtures, results, team and
player statistics, events, and expected-goal relationships. The existing Opta
workflow remains responsible for coordinate-level match analysis such as pass
networks, pitch maps, transitions and detailed action sequences.

The audited Sportmonks plan does not expose exact FBref definitions for shot
creating actions, progressive passes, carries into the final third, goalkeeper
crosses stopped, or sweeper actions. Those fields remain null in compatibility
data. Sportmonks-native charts use chances created, final-third passes, key
passes, successful dribbles, saves and goals conceded instead; they are not
silently relabelled as FBref metrics.

## Troubleshooting

- **401**: check the token in `.env` and ensure there are no quotes or spaces.
- **403**: the token is valid but the endpoint or league is outside the plan.
- **429**: leave the importer running; it respects the provider retry delay.
- **Interrupted run**: rerun the same command. Cached fixtures are skipped.
- **Missing Parquet support**: run
  `python -m pip install -r .\requirements-sportmonks.txt`.
- **Few teams after `--max-fixtures 2`**: this option imports two fixtures, not
  two fixtures per team. Use the unrestricted command before evaluating
  cross-league rankings.
