import os
from collections import Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

from src.components.layout_components import app_signature
from src.metrics.sportmonks import TEAM_METRIC_GROUPS
from src.visualization import team_plots


DATA_PATH = os.path.join("data", "fbref")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
SPORTMONKS_DATA_PATH = os.path.join("data", "sportmonks", "processed")
LEAGUES = {
    "bundesliga": {"name": "Bundesliga"},
    "la-liga": {"name": "La Liga"},
    "ligue-1": {"name": "Ligue 1"},
    "premier-league": {"name": "Premier League"},
    "serie-a": {"name": "Serie A"},
}
LEAGUE_NAME_TO_FOLDER = {value["name"]: key for key, value in LEAGUES.items()}

PROFILE_TAB_LABELS = {
    "attacking": "⚔️ Attacking",
    "possession": "⚽ Possession & Territory",
    "defending": "🛡️ Defending & Pressing",
    "set_pieces": "🎯 Set Pieces",
}

PROFILE_TAB_DESCRIPTIONS = {
    "attacking": "Scoring output, chance quality and shot execution.",
    "possession": "Ball control, passing efficiency and territorial presence.",
    "defending": "Shot suppression, chance prevention and pressing activity.",
    "set_pieces": "Attacking and defensive impact from dead-ball situations.",
}


def get_available_seasons_for_team(team_folder_name, league_folder):
    """Return all locally available seasons, including canonical Sportmonks data."""
    seasons = set()
    if os.path.isdir(PROCESSED_DATA_PATH):
        seasons.update(
            filename[len("team_stats_"):-len(".parquet")]
            for filename in os.listdir(PROCESSED_DATA_PATH)
            if filename.startswith("team_stats_") and filename.endswith(".parquet")
        )
    if os.path.isdir(SPORTMONKS_DATA_PATH):
        seasons.update(
            name
            for name in os.listdir(SPORTMONKS_DATA_PATH)
            if "-" in name and os.path.isdir(os.path.join(SPORTMONKS_DATA_PATH, name))
        )
    base_path = os.path.join(DATA_PATH, league_folder, f"{league_folder}_stats")
    if os.path.isdir(base_path):
        seasons.update(
            name
            for name in os.listdir(base_path)
            if "-" in name and os.path.isdir(os.path.join(base_path, name))
        )
    return sorted(seasons, reverse=True) if seasons else ["2024-2025"]


def get_team_options(season, league_name=None, exclude_team=None):
    """Build comparison options from the selected season, never from stale data."""
    from pages.team_stats import load_all_team_stats

    teams = load_all_team_stats(season)
    if teams.empty or not {"Squad", "League"}.issubset(teams.columns):
        return []
    if league_name:
        teams = teams[teams["League"] == league_name]
    if exclude_team:
        teams = teams[teams["Squad"] != exclude_team]
    teams = teams[["Squad", "League"]].dropna().drop_duplicates()
    return [
        {"label": f"{row['Squad']} ({row['League']})", "value": row["Squad"]}
        for _, row in teams.sort_values(["League", "Squad"]).iterrows()
    ]


def _safe_ratio(numerator, denominator, multiplier=1):
    result = numerator / denominator.replace(0, np.nan) * multiplier
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def _formation_from_fields(fields):
    """Convert Sportmonks formation fields (row:column) into 4-2-3-1."""
    rows = []
    for field in fields:
        try:
            row_number = int(str(field).split(":", 1)[0])
        except (TypeError, ValueError):
            continue
        if row_number > 1:
            rows.append(row_number)
    counts = Counter(rows)
    if sum(counts.values()) != 10:
        return None
    return "-".join(str(counts[row]) for row in sorted(counts))


def _derive_formation_profile(player_matchlogs):
    """Derive the common shape and a representative XI from existing raw lineups."""
    if player_matchlogs.empty:
        return [], []
    required = {"fixture_id", "player_name", "starter", "formation_field"}
    if not required.issubset(player_matchlogs.columns):
        return [], []

    starter_values = player_matchlogs["starter"]
    if pd.api.types.is_bool_dtype(starter_values):
        starter_mask = starter_values.fillna(False)
    else:
        starter_mask = (
            starter_values.fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "1", "yes"})
        )
    starters = player_matchlogs[
        starter_mask & player_matchlogs["formation_field"].notna()
    ].copy()
    if starters.empty:
        return [], []

    fixture_shapes = (
        starters.groupby("fixture_id")["formation_field"]
        .apply(_formation_from_fields)
        .dropna()
    )
    if fixture_shapes.empty:
        return [], []

    counts = fixture_shapes.value_counts()
    total = int(counts.sum())
    analysis = [
        {
            "name": shape,
            "count": int(count),
            "percentage": float(count / total * 100),
        }
        for shape, count in counts.head(3).items()
    ]

    primary_shape = analysis[0]["name"]
    primary_fixtures = fixture_shapes[fixture_shapes == primary_shape].index
    primary_starters = starters[starters["fixture_id"].isin(primary_fixtures)].copy()
    primary_starters["formation_field"] = primary_starters["formation_field"].astype(str)
    image_column = "image_path" if "image_path" in primary_starters else None
    aggregation = {"starts": ("fixture_id", "nunique")}
    if image_column:
        aggregation["image_path"] = (image_column, "last")
    slot_counts = (
        primary_starters.groupby(["formation_field", "player_name"], as_index=False)
        .agg(**aggregation)
        .sort_values(["formation_field", "starts", "player_name"], ascending=[True, False, True])
    )

    def slot_key(value):
        try:
            row, column = str(value).split(":", 1)
            return int(row), int(column)
        except (TypeError, ValueError):
            return 99, 99

    typical_lineup = []
    selected_players = set()
    for slot in sorted(slot_counts["formation_field"].unique(), key=slot_key):
        candidates = slot_counts[slot_counts["formation_field"] == slot]
        candidate = next(
            (
                row
                for _, row in candidates.iterrows()
                if row["player_name"] not in selected_players
            ),
            candidates.iloc[0],
        )
        selected_players.add(candidate["player_name"])
        row_number, column_number = slot_key(slot)
        typical_lineup.append(
            {
                "player_name": candidate["player_name"],
                "image_path": candidate.get("image_path") if image_column else None,
                "starts": int(candidate["starts"]),
                "row": row_number,
                "column": column_number,
            }
        )
    return analysis, typical_lineup


def _get_sportmonks_team_data(team_name, season):
    team_stats_path = os.path.join(PROCESSED_DATA_PATH, f"team_stats_{season}.parquet")
    player_stats_path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
    matchlogs_path = os.path.join(SPORTMONKS_DATA_PATH, season, "team_matchlogs.parquet")
    player_matchlogs_path = os.path.join(
        SPORTMONKS_DATA_PATH, season, "player_matchlogs.parquet"
    )
    if not all(os.path.exists(path) for path in (team_stats_path, player_stats_path, matchlogs_path)):
        return None

    all_teams = pd.read_parquet(team_stats_path)
    team_profile = all_teams[all_teams["Squad"] == team_name]
    if team_profile.empty:
        return None
    league_name = team_profile.iloc[0]["League"]
    league_folder = LEAGUE_NAME_TO_FOLDER.get(league_name)
    if not league_folder:
        return None

    roster = pd.read_parquet(player_stats_path)
    roster = roster[roster["Club"] == team_name].copy()
    aliases = {
        "Interceptions": "Int",
        "Att": "Passes",
        "Cmp": "Accurate_Passes",
        "AerialsWon": "Aerial_Won",
        "AerialsLost": "Aerial_Lost",
        "TakeOns": "Dribble_Attempts",
        "TakeOnSucc": "Successful_Dribbles",
    }
    for target, source in aliases.items():
        roster[target] = (
            pd.to_numeric(roster[source], errors="coerce").fillna(0)
            if source in roster
            else 0
        )
    numeric_columns = [
        "Gls", "Ast", "MP", "Min", "90s", "TklW", "Interceptions",
        "Att", "Cmp", "Sh", "Chances_Created", "AerialsWon",
        "AerialsLost", "TakeOns", "TakeOnSucc", "Age",
    ]
    for column in numeric_columns:
        roster[column] = (
            pd.to_numeric(roster[column], errors="coerce").fillna(0)
            if column in roster
            else 0
        )
    roster["Pass_Completion_Perc"] = _safe_ratio(roster["Cmp"], roster["Att"], 100)
    roster["Tkl_Int_per_90"] = _safe_ratio(
        roster["TklW"] + roster["Interceptions"], roster["90s"]
    )
    roster["Chances_Created_per_90"] = _safe_ratio(
        roster["Chances_Created"], roster["90s"]
    )
    roster["Min_per_Match"] = _safe_ratio(roster["Min"], roster["MP"])
    roster["dominant_position"] = roster["Pos"].fillna("N/A")

    matchlogs = pd.read_parquet(matchlogs_path)
    matchlogs = matchlogs[matchlogs["team_name"] == team_name].copy()
    matchlogs.rename(
        columns={
            "starting_at": "Date",
            "result": "Result",
            "opponent": "Opponent",
            "goals_for": "GF",
            "goals_against": "GA",
            "formation": "Formation",
            "location": "Location",
        },
        inplace=True,
    )

    formation_analysis = []
    typical_lineup = []
    if os.path.exists(player_matchlogs_path):
        player_matchlogs = pd.read_parquet(player_matchlogs_path)
        player_matchlogs = player_matchlogs[
            player_matchlogs["team_name"] == team_name
        ].copy()
        formation_analysis, typical_lineup = _derive_formation_profile(player_matchlogs)
        if typical_lineup and {"Player", "image_path"}.issubset(roster.columns):
            player_images = (
                roster.dropna(subset=["Player"])
                .drop_duplicates("Player")
                .set_index("Player")["image_path"]
                .to_dict()
            )
            for player in typical_lineup:
                player["image_path"] = player_images.get(player["player_name"])

    logo_path = ""
    coach = {}
    teams_path = os.path.join(SPORTMONKS_DATA_PATH, season, "teams.parquet")
    if os.path.exists(teams_path):
        teams = pd.read_parquet(teams_path)
        team_row = teams[teams["team_name"] == team_name]
        if not team_row.empty:
            team_record = team_row.iloc[0]
            logo_value = team_record.get("image_path")
            logo_path = str(logo_value) if pd.notna(logo_value) else ""
            coach_name = team_record.get("coach_name")
            if pd.notna(coach_name) and str(coach_name).strip():
                coach_image = team_record.get("coach_image_path")
                coach = {
                    "name": str(coach_name).strip(),
                    "image_path": (
                        str(coach_image)
                        if pd.notna(coach_image) and str(coach_image).strip()
                        else ""
                    ),
                }

    standing = {}
    standings_path = os.path.join(SPORTMONKS_DATA_PATH, season, "standings.parquet")
    if os.path.exists(standings_path):
        standings = pd.read_parquet(standings_path)
        standing_row = standings[standings["team_name"] == team_name]
        if not standing_row.empty:
            standing = standing_row.iloc[0].to_dict()

    return {
        "profile_df": team_profile,
        "all_teams_df": all_teams,
        "roster_df": roster,
        "matchlogs_df": matchlogs,
        "league_name": league_name,
        "league_folder": league_folder,
        "folder_name": str(team_name).replace(" ", "_"),
        "logo_path": logo_path,
        "standing": standing,
        "coach": coach,
        "formation_analysis": formation_analysis,
        "typical_lineup": typical_lineup,
        "data_source": "Sportmonks",
    }


def get_team_data(team_name, season):
    """Load Sportmonks first and retain the legacy FBref fallback."""
    from pages.team_stats import load_all_team_stats

    try:
        sportmonks_data = _get_sportmonks_team_data(team_name, season)
        if sportmonks_data is not None:
            return sportmonks_data
    except Exception as exc:
        print(f"Error loading Sportmonks team profile for {team_name}: {exc}")

    team_folder_name = str(team_name).replace(" ", "_")
    league_folder = None
    league_name = None
    for folder, info in LEAGUES.items():
        team_path = os.path.join(DATA_PATH, folder, f"{folder}_teams", team_folder_name)
        if os.path.isdir(team_path):
            league_folder = folder
            league_name = info["name"]
            break
    if not league_folder:
        return {"error": "Team not found."}

    all_teams = load_all_team_stats(season)
    if all_teams.empty:
        return {"error": f"Could not load stats for season {season}."}
    team_profile = all_teams[all_teams["Squad"] == team_name]
    if team_profile.empty:
        return {"error": f"Stats for {team_name} in {season} not found."}

    team_path = os.path.join(
        DATA_PATH,
        league_folder,
        f"{league_folder}_teams",
        team_folder_name,
        "team_stats",
    )
    try:
        standard = pd.read_csv(os.path.join(team_path, "standard.csv"))
        defending = pd.read_csv(os.path.join(team_path, "defensive.csv"))
        passing = pd.read_csv(os.path.join(team_path, "passing.csv"))
        shooting = pd.read_csv(os.path.join(team_path, "shooting.csv"))
        gca = pd.read_csv(os.path.join(team_path, "gca.csv"))
        miscellaneous = pd.read_csv(os.path.join(team_path, "misc.csv"))
        possession = pd.read_csv(os.path.join(team_path, "possession.csv"))
        roster = pd.merge(standard, defending[["Player", "TklW", "Interceptions"]], on="Player", how="left")
        roster = pd.merge(roster, passing[["Player", "Att", "Cmp", "PrgP"]], on="Player", how="left")
        roster = pd.merge(roster, shooting[["Player", "Sh"]], on="Player", how="left")
        roster = pd.merge(roster, gca[["Player", "GCA"]], on="Player", how="left")
        roster = pd.merge(roster, miscellaneous[["Player", "AerialsWon", "AerialsLost"]], on="Player", how="left")
        roster = pd.merge(roster, possession[["Player", "TakeOns", "TakeOnSucc"]], on="Player", how="left")
        roster = roster[~roster["Player"].isin(["Squad Total", "Opponent Total"])]
        numeric_columns = [
            "Gls", "Ast", "MP", "Min", "90s", "TklW", "Interceptions",
            "Att", "Cmp", "Sh", "GCA", "AerialsWon", "AerialsLost",
            "TakeOns", "TakeOnSucc", "Age",
        ]
        for column in numeric_columns:
            if column in roster:
                roster[column] = pd.to_numeric(
                    roster[column].astype(str).str.replace(",", ""), errors="coerce"
                ).fillna(0)
        roster["Pass_Completion_Perc"] = _safe_ratio(roster["Cmp"], roster["Att"], 100)
        roster["Tkl_Int_per_90"] = _safe_ratio(
            roster["TklW"] + roster["Interceptions"], roster["90s"]
        )
        roster["Chances_Created_per_90"] = _safe_ratio(roster.get("GCA", 0), roster["90s"])
        roster["Min_per_Match"] = _safe_ratio(roster["Min"], roster["MP"])
        roster["dominant_position"] = roster.get("Pos", "N/A")
        matchlogs = pd.read_csv(os.path.join(team_path, "matchlogs.csv"))
    except Exception as exc:
        return {"error": f"Error loading team-specific files for {team_name}: {exc}"}

    return {
        "profile_df": team_profile,
        "all_teams_df": all_teams,
        "roster_df": roster,
        "matchlogs_df": matchlogs,
        "league_name": league_name,
        "league_folder": league_folder,
        "folder_name": team_folder_name,
        "logo_path": get_team_logo_path(league_folder, team_folder_name),
        "standing": {},
        "coach": {},
        "formation_analysis": [],
        "typical_lineup": [],
        "data_source": "FBref",
    }


def get_team_logo_path(league_folder, team_folder_name):
    if not league_folder:
        return ""
    return f"/data/fbref/{league_folder}/{league_folder}_teams/{team_folder_name}/logo.png"


def get_player_photo_path(player_name):
    player_folder = str(player_name).replace(" ", "_")
    local_path = os.path.join(
        "data", "fbref", "player_top5_europe", player_folder, f"{player_folder}.png"
    )
    if os.path.exists(local_path):
        return f"/{local_path.replace(os.sep, '/')}"
    return "/assets/avatar_placeholder.png"


def _numeric(value):
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def _format_metric(value, spec):
    value = _numeric(value)
    if pd.isna(value):
        return "N/A"
    formatted = spec.format_spec.format(value)
    return f"{formatted}{spec.unit}" if spec.unit else formatted


def _ordinal(number):
    number = int(number)
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _metric_context(profile, cohort, spec):
    if spec.column not in cohort:
        return None, None, 0
    values = pd.to_numeric(cohort[spec.column], errors="coerce").dropna()
    target = _numeric(profile.get(spec.column))
    if values.empty or pd.isna(target):
        return None, len(values), 0
    if spec.ascending:
        rank = int((values < target).sum()) + 1
        percentile = float((values >= target).mean() * 100)
    else:
        rank = int((values > target).sum()) + 1
        percentile = float((values <= target).mean() * 100)
    return rank, len(values), percentile


def create_team_metric_card(profile, cohort, spec, group_name):
    rank, cohort_size, percentile = _metric_context(profile, cohort, spec)
    metric_id = f"profile-metric-{group_name}-{spec.column}".replace("_", "-")
    direction = "Lower is better" if spec.ascending else "Higher is better"
    context = (
        f"{_ordinal(rank)} of {cohort_size} in league"
        if rank is not None and cohort_size
        else "League rank unavailable"
    )
    return dbc.Col(
        [
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.I(className=f"{spec.icon} team-profile-metric-icon"),
                            html.Span(spec.title),
                        ],
                        id=metric_id,
                        className="team-profile-metric-header",
                    ),
                    dbc.CardBody(
                        [
                            html.Div(
                                _format_metric(profile.get(spec.column), spec),
                                className="team-profile-metric-value",
                            ),
                            html.Div(context, className="team-profile-metric-rank"),
                            html.Div(
                                html.Div(
                                    style={"width": f"{max(0, min(percentile, 100)):.1f}%"},
                                    className="team-profile-percentile-fill",
                                ),
                                className="team-profile-percentile-track",
                            ),
                            html.Div(
                                [
                                    html.Span(direction),
                                    html.Span(f"{percentile:.0f}th percentile" if rank else "N/A"),
                                ],
                                className="team-profile-metric-footnote",
                            ),
                        ]
                    ),
                ],
                className=f"metric-card team-profile-metric-card metric-group-{group_name} h-100",
            ),
            dbc.Tooltip(spec.tooltip, target=metric_id, placement="top"),
        ],
        lg=3,
        md=6,
        sm=12,
        className="mb-3",
    )


def create_team_metric_tabs(profile, all_teams, league_name):
    cohort = all_teams[all_teams["League"] == league_name].copy()
    tabs = []
    for group_name, specs in TEAM_METRIC_GROUPS.items():
        cards = [create_team_metric_card(profile, cohort, spec, group_name) for spec in specs]
        tabs.append(
            dbc.Tab(
                label=PROFILE_TAB_LABELS[group_name],
                tab_id=f"profile-tab-{group_name}",
                children=html.Div(
                    [
                        html.P(
                            PROFILE_TAB_DESCRIPTIONS[group_name],
                            className="team-profile-tab-description",
                        ),
                        dbc.Row(cards),
                    ],
                    className="pt-3",
                ),
            )
        )
    return dbc.Tabs(
        tabs,
        active_tab="profile-tab-attacking",
        className="team-metric-tabs team-profile-tabs",
    )


def _snapshot_value(label, value, icon):
    return html.Div(
        [
            html.Div(html.I(className=icon), className="team-profile-snapshot-icon"),
            html.Div(
                [
                    html.Span(label, className="team-profile-snapshot-label"),
                    html.Strong(value, className="team-profile-snapshot-value"),
                ]
            ),
        ],
        className="team-profile-snapshot-item",
    )


def create_recent_results(matchlogs):
    required = {"Result", "Opponent", "GF", "GA"}
    if matchlogs.empty or not required.issubset(matchlogs.columns):
        return html.P("No recent match data available.", className="team-profile-empty-note")
    recent = matchlogs.dropna(subset=["Result"]).copy()
    if "Date" in recent:
        recent["Date"] = pd.to_datetime(recent["Date"], errors="coerce")
        recent.sort_values("Date", inplace=True)
    recent = recent.tail(5).iloc[::-1]
    colors = {"W": "success", "D": "warning", "L": "danger"}
    rows = []
    for _, row in recent.iterrows():
        result = str(row["Result"])[0]
        location = str(row.get("Location", ""))[:1].upper()
        rows.append(
            html.Div(
                [
                    dbc.Badge(result, color=colors.get(result, "secondary"), className="result-badge"),
                    html.Div(
                        [
                            html.Strong(row["Opponent"], className="recent-opponent"),
                            html.Span(
                                "Home" if location == "H" else "Away" if location == "A" else "",
                                className="recent-location",
                            ),
                        ],
                        className="recent-opponent-wrap",
                    ),
                    html.Strong(f"{int(row['GF'])}–{int(row['GA'])}", className="recent-score"),
                ],
                className="team-profile-result-row",
            )
        )
    return html.Div(rows)


def _season_summary_values(profile, standing, cohort):
    position = _numeric(standing.get("position")) if standing else np.nan
    if pd.isna(position) and "Pts_per_MP" in cohort:
        values = pd.to_numeric(cohort["Pts_per_MP"], errors="coerce")
        target = _numeric(profile.get("Pts_per_MP"))
        position = int((values > target).sum()) + 1 if not pd.isna(target) else np.nan
    points = _numeric(standing.get("points")) if standing else np.nan
    if pd.isna(points):
        points = _numeric(profile.get("Pts"))
    values = {
        "position": _ordinal(position) if not pd.isna(position) else "N/A",
        "points": f"{int(points)}" if not pd.isna(points) else "N/A",
        "wins": _numeric(profile.get("W")),
        "draws": _numeric(profile.get("D")),
        "losses": _numeric(profile.get("L")),
        "goal_difference": _numeric(profile.get("GD")),
    }
    values["record"] = " · ".join(
        f"{int(0 if pd.isna(values[key]) else values[key])}{suffix}"
        for key, suffix in (("wins", "W"), ("draws", "D"), ("losses", "L"))
    )
    values["goal_difference"] = (
        f"{values['goal_difference']:+.0f}"
        if not pd.isna(values["goal_difference"])
        else "N/A"
    )
    return values


def create_recent_form_strip(matchlogs):
    required = {"Result", "Opponent", "GF", "GA"}
    if matchlogs.empty or not required.issubset(matchlogs.columns):
        return html.Span("No recent results", className="team-profile-form-empty")
    recent = matchlogs.dropna(subset=["Result"]).copy()
    if "Date" in recent:
        recent["Date"] = pd.to_datetime(recent["Date"], errors="coerce")
        recent.sort_values("Date", inplace=True)
    recent = recent.tail(5)
    chips = []
    for _, row in recent.iterrows():
        result = str(row["Result"])[0].upper()
        goals_for = _numeric(row.get("GF"))
        goals_against = _numeric(row.get("GA"))
        score = (
            f"{int(goals_for)}–{int(goals_against)}"
            if not pd.isna(goals_for) and not pd.isna(goals_against)
            else ""
        )
        location = str(row.get("Location", ""))[:1].upper()
        venue = "Home" if location == "H" else "Away" if location == "A" else ""
        chips.append(
            html.Div(
                [
                    html.Strong(result),
                    html.Span(score),
                ],
                className=f"team-profile-form-chip form-{result.lower()}",
                title=f"{venue} vs {row.get('Opponent', '')} · {score}",
            )
        )
    return html.Div(chips, className="team-profile-form-chips")


def create_hero_summary(profile, standing, matchlogs, cohort):
    summary = _season_summary_values(profile, standing, cohort)
    items = [
        ("League position", summary["position"], "fa-solid fa-ranking-star"),
        ("Points", summary["points"], "fa-solid fa-star"),
        ("Season record", summary["record"], "fa-solid fa-chart-simple"),
        ("Goal difference", summary["goal_difference"], "fa-solid fa-scale-balanced"),
    ]
    return html.Div(
        [
            html.Div(
                [
                    html.I(className=icon),
                    html.Div([html.Span(label), html.Strong(value)]),
                ],
                className="team-profile-hero-stat",
            )
            for label, value, icon in items
        ]
        + [
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="fa-solid fa-clock-rotate-left"),
                            html.Span("Recent form"),
                        ],
                        className="team-profile-form-label",
                    ),
                    create_recent_form_strip(matchlogs),
                ],
                className="team-profile-hero-form",
            )
        ],
        className="team-profile-hero-summary",
    )


def create_formation_card(formation_analysis, typical_lineup):
    if not formation_analysis or not typical_lineup:
        body = html.Div(
            [
                html.I(className="fa-solid fa-people-group"),
                html.Strong("Formation data unavailable"),
                html.Span("Starting-position data was not found for this season."),
            ],
            className="team-profile-formation-empty",
        )
    else:
        primary = formation_analysis[0]
        figure = team_plots.plot_typical_formation_plotly(
            primary["name"],
            typical_lineup,
            height=550,
        )
        alternatives = html.Div(
            [
                html.Div(
                    [
                        html.Strong(item["name"]),
                        html.Span(f"{item['count']} matches · {item['percentage']:.0f}%"),
                    ],
                    className=(
                        "team-profile-shape-chip active"
                        if index == 0
                        else "team-profile-shape-chip"
                    ),
                )
                for index, item in enumerate(formation_analysis)
            ],
            className="team-profile-shape-list",
        )
        body = [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Primary shape", className="team-profile-eyebrow"),
                            html.Strong(primary["name"], className="team-profile-shape-title"),
                        ]
                    ),
                    html.Div(
                        [
                            html.Strong(f"{primary['percentage']:.0f}%"),
                            html.Span(f"{primary['count']} starts"),
                        ],
                        className="team-profile-shape-usage",
                    ),
                ],
                className="team-profile-formation-summary",
            ),
            dcc.Graph(
                figure=figure,
                config={"displayModeBar": False, "responsive": True},
                className="team-profile-formation-graph",
            ),
            alternatives,
            html.P(
                "Representative XI: the most frequent starter in each formation slot.",
                className="team-profile-method-note mb-0",
            ),
        ]
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div(
                        [
                            html.H3("Typical formation", className="mb-1"),
                            html.P("Most-used structure and representative XI", className="mb-0"),
                        ]
                    )
                ],
                className="team-profile-section-header",
            ),
            dbc.CardBody(body, className="team-profile-formation-body"),
        ],
        className="team-profile-panel team-profile-formation-card h-100",
    )


def create_season_snapshot(profile, standing, matchlogs, cohort):
    position = _numeric(standing.get("position")) if standing else np.nan
    if pd.isna(position) and "Pts_per_MP" in cohort:
        values = pd.to_numeric(cohort["Pts_per_MP"], errors="coerce")
        target = _numeric(profile.get("Pts_per_MP"))
        position = int((values > target).sum()) + 1 if not pd.isna(target) else np.nan
    points = _numeric(standing.get("points")) if standing else np.nan
    if pd.isna(points):
        points = _numeric(profile.get("Pts"))
    wins_value = _numeric(profile.get("W"))
    draws_value = _numeric(profile.get("D"))
    losses_value = _numeric(profile.get("L"))
    wins = int(wins_value) if not pd.isna(wins_value) else 0
    draws = int(draws_value) if not pd.isna(draws_value) else 0
    losses = int(losses_value) if not pd.isna(losses_value) else 0
    goal_difference = _numeric(profile.get("GD"))
    return dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div(
                        [
                            html.H3("Season snapshot", className="mb-1"),
                            html.P("League performance and latest results", className="mb-0"),
                        ]
                    )
                ],
                className="team-profile-section-header",
            ),
            dbc.CardBody(
                [
                    html.Div(
                        [
                            _snapshot_value(
                                "League position",
                                _ordinal(position) if not pd.isna(position) else "N/A",
                                "fa-solid fa-ranking-star",
                            ),
                            _snapshot_value(
                                "Points",
                                f"{int(points)}" if not pd.isna(points) else "N/A",
                                "fa-solid fa-star",
                            ),
                            _snapshot_value(
                                "Record",
                                f"{wins}W · {draws}D · {losses}L",
                                "fa-solid fa-chart-simple",
                            ),
                            _snapshot_value(
                                "Goal difference",
                                f"{goal_difference:+.0f}" if not pd.isna(goal_difference) else "N/A",
                                "fa-solid fa-scale-balanced",
                            ),
                        ],
                        className="team-profile-snapshot-grid",
                    ),
                    html.H4("Last five matches", className="team-profile-card-subtitle"),
                    create_recent_results(matchlogs),
                ]
            ),
        ],
        className="team-profile-panel h-100",
    )


def _player_info(player):
    position = str(player.get("Pos", "N/A"))
    age = _numeric(player.get("Age"))
    return f"{position} · {int(age)} years" if not pd.isna(age) and age > 0 else position


def create_player_ranking_row(player, metric_column, unit="", format_spec="{:,.1f}", winner=False):
    player_name = str(player["Player"])
    value = _numeric(player.get(metric_column))
    return html.Div(
        [
            html.Img(
                src=player.get("image_path") or get_player_photo_path(player_name),
                className="leader-photo winner" if winner else "leader-photo",
            ),
            html.Div(
                [
                    dcc.Link(
                        player_name,
                        href=f"/player-stats/{player_name.replace(' ', '_')}",
                        className="leader-name",
                    ),
                    html.Span(_player_info(player), className="leader-meta"),
                ],
                className="leader-identity",
            ),
            html.Strong(
                f"{format_spec.format(value)}{unit}" if not pd.isna(value) else "N/A",
                className="leader-value",
            ),
        ],
        className="team-profile-leader-row winner" if winner else "team-profile-leader-row",
    )


def create_top_performer_card(
    roster,
    title,
    metric_column,
    icon,
    unit="",
    format_spec="{:,.1f}",
    min_minutes=0,
):
    eligible = roster.copy()
    if metric_column not in eligible:
        eligible = eligible.iloc[0:0]
    else:
        eligible[metric_column] = pd.to_numeric(eligible[metric_column], errors="coerce")
        eligible["Min"] = pd.to_numeric(eligible.get("Min", 0), errors="coerce").fillna(0)
        eligible = eligible[
            eligible[metric_column].notna()
            & (eligible["Min"] >= min_minutes)
            & (eligible["Min"] > 0)
        ].sort_values(metric_column, ascending=False).head(3)

    body = (
        [
            create_player_ranking_row(
                player,
                metric_column,
                unit=unit,
                format_spec=format_spec,
                winner=index == 0,
            )
            for index, (_, player) in enumerate(eligible.iterrows())
        ]
        if not eligible.empty
        else [html.P("No eligible player data.", className="team-profile-empty-note mb-0")]
    )
    return dbc.Col(
        dbc.Card(
            [
                dbc.CardHeader(
                    [html.I(className=f"{icon} me-2"), title],
                    className="team-profile-leader-header",
                ),
                dbc.CardBody(body, className="p-0"),
            ],
            className="team-profile-leader-card h-100",
        ),
        lg=3,
        md=6,
        sm=12,
        className="mb-3",
    )


def create_player_card(player):
    player_name = str(player["Player"])
    matches_value = _numeric(player.get("MP"))
    minutes_value = _numeric(player.get("Min"))
    matches = int(matches_value) if not pd.isna(matches_value) else 0
    minutes = int(minutes_value) if not pd.isna(minutes_value) else 0
    return dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.Img(
                        src=player.get("image_path") or get_player_photo_path(player_name),
                        className="team-profile-player-photo",
                    ),
                    html.Div(
                        [
                            dcc.Link(
                                player_name,
                                href=f"/player-stats/{player_name.replace(' ', '_')}",
                                className="team-profile-player-name",
                            ),
                            html.Span(_player_info(player), className="team-profile-player-meta"),
                            html.Div(
                                [html.Span(f"{matches} apps"), html.Span(f"{minutes:,} min")],
                                className="team-profile-player-usage",
                            ),
                        ],
                        className="team-profile-player-copy",
                    ),
                ],
                className="team-profile-player-body",
            ),
            className="team-profile-player-card h-100",
        ),
        xl=3,
        lg=4,
        md=6,
        sm=12,
        className="mb-3",
    )


def _assign_role(position):
    position = str(position).upper()
    if "GK" in position:
        return "Goalkeepers 🧤"
    if "FW" in position or position in {"RW", "LW", "ST", "CF"}:
        return "Forwards ⚔️"
    if "MF" in position or position in {"DM", "CM", "AM", "LM", "RM"}:
        return "Midfielders 🧠"
    if "DF" in position or position in {"CB", "LB", "RB", "WB"}:
        return "Defenders 🛡️"
    return "Other"


def create_roster_section(roster):
    active = roster.copy()
    active["Min"] = pd.to_numeric(active.get("Min", 0), errors="coerce").fillna(0)
    active = active[active["Min"] > 0].sort_values("Min", ascending=False)
    positions = active["dominant_position"] if "dominant_position" in active else active.get("Pos", "N/A")
    active["Role"] = positions.apply(_assign_role)
    role_order = [
        "Goalkeepers 🧤",
        "Defenders 🛡️",
        "Midfielders 🧠",
        "Forwards ⚔️",
        "Other",
    ]
    items = []
    for role in role_order:
        players = active[active["Role"] == role]
        if players.empty:
            continue
        items.append(
            dbc.AccordionItem(
                dbc.Row([create_player_card(player) for _, player in players.iterrows()]),
                title=f"{role} ({len(players)})",
            )
        )
    return dbc.Accordion(
        items,
        start_collapsed=True,
        always_open=True,
        className="team-profile-roster-accordion",
    )


def layout(team_name_url, season="2024-2025"):
    team_name = team_name_url.replace("_", " ")
    team_data = get_team_data(team_name, season)
    if "error" in team_data:
        return dbc.Container(
            dbc.Alert(team_data["error"], color="danger"),
            fluid=True,
            className="team-stats-page team-profile-page p-4",
        )

    profile = team_data["profile_df"].iloc[0]
    all_teams = team_data["all_teams_df"]
    roster = team_data["roster_df"].copy()
    matchlogs = team_data["matchlogs_df"].copy()
    league_name = team_data["league_name"]
    league_folder = team_data["league_folder"]
    folder_name = team_data["folder_name"]
    league_cohort = all_teams[all_teams["League"] == league_name].copy()
    available_seasons = get_available_seasons_for_team(folder_name, league_folder)
    logo = team_data.get("logo_path") or get_team_logo_path(league_folder, folder_name)
    coach = team_data.get("coach") or {}
    identity_copy = [
        html.Div(
            [
                dbc.Badge(league_name, className="team-profile-league-badge"),
                dbc.Badge(
                    team_data.get("data_source", ""),
                    className="team-profile-source-badge",
                ),
            ],
            className="team-profile-badges",
        ),
        html.H1(team_name, className="team-profile-title mb-1"),
        html.P(
            "Season performance, tactical identity and squad leaders",
            className="team-profile-subtitle mb-0",
        ),
    ]
    if coach.get("name"):
        identity_copy.append(
            html.Div(
                [
                    html.Img(
                        src=coach.get("image_path") or "/assets/avatar_placeholder.png",
                        className="team-profile-coach-photo",
                    ),
                    html.Div(
                        [
                            html.Span("Head coach"),
                            html.Strong(coach["name"]),
                        ]
                    ),
                ],
                className="team-profile-coach",
            )
        )

    header = html.Div(
        [
            html.Div(
                dbc.Button(
                    [html.I(className="fas fa-arrow-left me-2"), "Teams Overview"],
                    href="/team-stats",
                    color="secondary",
                    className="back-home-btn team-profile-back-btn",
                ),
                className="team-profile-hero-toolbar",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                html.Img(src=logo, className="team-profile-logo"),
                                className="team-profile-logo-wrap",
                            ),
                            html.Div(identity_copy, className="team-profile-title-wrap"),
                        ],
                        className="team-profile-identity",
                    ),
                    html.Div(
                        [
                            html.Label("Season", className="team-profile-filter-label"),
                            dcc.Dropdown(
                                id="team-profile-season-dropdown",
                                options=[
                                    {"label": value, "value": value}
                                    for value in available_seasons
                                ],
                                value=season,
                                clearable=False,
                                className=(
                                    "season-filter-dropdown "
                                    "team-profile-season-dropdown"
                                ),
                            ),
                        ],
                        className="team-profile-season-control",
                    ),
                ],
                className="team-profile-hero-main",
            ),
            create_hero_summary(
                profile,
                team_data.get("standing", {}),
                matchlogs,
                league_cohort,
            ),
        ],
        className="team-profile-hero",
    )

    metrics_section = html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Team metrics", className="team-profile-section-title mb-1"),
                            html.P(
                                f"Values are ranked against all {len(league_cohort)} teams in {league_name}.",
                                className="team-profile-section-copy mb-0",
                            ),
                        ]
                    ),
                    html.Div(
                        [html.I(className="fa-solid fa-circle-info me-2"), "Hover a metric title to see its definition."],
                        className="team-profile-info-pill",
                    ),
                ],
                className="team-profile-section-heading",
            ),
            create_team_metric_tabs(profile, all_teams, league_name),
        ],
        className="team-profile-metrics-section",
    )

    formation_card = create_formation_card(
        team_data.get("formation_analysis", []),
        team_data.get("typical_lineup", []),
    )
    radar = dbc.Card(
        [
            dbc.CardHeader(
                [
                    html.Div(
                        [
                            html.H3("Statistical profile", className="mb-1"),
                            html.P(
                                "A higher percentile always means stronger relative performance.",
                                className="mb-0",
                            ),
                        ]
                    )
                ],
                className="team-profile-section-header",
            ),
            dbc.CardBody(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dcc.Dropdown(
                                    id="team-radar-compare-dropdown",
                                    options=get_team_options(season, league_name, team_name),
                                    placeholder="Compare with another team...",
                                    clearable=True,
                                    className="team-profile-compare-dropdown",
                                ),
                                lg=7,
                                md=12,
                            ),
                            dbc.Col(
                                dbc.RadioItems(
                                    id="radar-normalization-filter",
                                    options=[
                                        {"label": "vs League", "value": "league"},
                                        {"label": "vs Top 5", "value": "top5"},
                                    ],
                                    value="league",
                                    inline=True,
                                    className="btn-group team-profile-radar-scope",
                                    inputClassName="btn-check",
                                    labelClassName="btn btn-outline-primary",
                                ),
                                lg=5,
                                md=12,
                                className="d-flex justify-content-lg-end mt-3 mt-lg-0",
                            ),
                        ],
                        align="center",
                        className="mb-2",
                    ),
                    dcc.Loading(
                        dcc.Graph(
                            id="team-profile-radar",
                            config={"displayModeBar": False, "responsive": True},
                            className="team-profile-radar-graph",
                        )
                    ),
                    html.P(
                        "The radar uses percentile ranks, not min-max scaling. Lower-is-better metrics are inverted automatically.",
                        className="team-profile-method-note mb-0",
                    ),
                ]
            ),
        ],
        className="team-profile-panel h-100",
    )

    team_matches = _numeric(profile.get("MP"))
    qualifying_minutes = max(
        180,
        min(900, int((0 if pd.isna(team_matches) else team_matches) * 90 * 0.25)),
    )
    leaders = dbc.Row(
        [
            create_top_performer_card(roster, "Goals", "Gls", "fa-solid fa-futbol", format_spec="{:,.0f}"),
            create_top_performer_card(roster, "Assists", "Ast", "fa-solid fa-hands-helping", format_spec="{:,.0f}"),
            create_top_performer_card(
                roster,
                "Chances Created p90",
                "Chances_Created_per_90",
                "fa-solid fa-wand-magic-sparkles",
                format_spec="{:,.2f}",
                min_minutes=qualifying_minutes,
            ),
            create_top_performer_card(
                roster,
                "Tackles + Interceptions p90",
                "Tkl_Int_per_90",
                "fa-solid fa-shield-halved",
                format_spec="{:,.2f}",
                min_minutes=qualifying_minutes,
            ),
        ]
    )

    return dbc.Container(
        [
            dcc.Store(id="team-profile-league-name-store", data=league_name),
            dcc.Store(id="team-profile-name-store", data=team_name),
            html.Div(
                [
                    header,
                    metrics_section,
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2(
                                                "Tactical & statistical identity",
                                                className="team-profile-section-title mb-1",
                                            ),
                                            html.P(
                                                "How the team sets up, and how its performance compares with the league.",
                                                className="team-profile-section-copy mb-0",
                                            ),
                                        ]
                                    )
                                ],
                                className="team-profile-section-heading",
                            ),
                            dbc.Row(
                                [
                                    dbc.Col(
                                        formation_card,
                                        xl=5,
                                        lg=5,
                                        md=12,
                                        className="mb-4",
                                    ),
                                    dbc.Col(
                                        radar,
                                        xl=7,
                                        lg=7,
                                        md=12,
                                        className="mb-4",
                                    ),
                                ],
                                align="stretch",
                            ),
                        ],
                        className="team-profile-identity-section",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2(
                                                "Squad leaders",
                                                className="team-profile-section-title mb-1",
                                            ),
                                            html.P(
                                                "Totals for goals and assists; per-90 rankings require "
                                                f"at least {qualifying_minutes:,} minutes.",
                                                className="team-profile-section-copy mb-0",
                                            ),
                                        ]
                                    )
                                ],
                                className="team-profile-section-heading",
                            ),
                            leaders,
                        ],
                        className="team-profile-leaders-section",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.H2(
                                                "Season squad",
                                                className="team-profile-section-title mb-1",
                                            ),
                                            html.P(
                                                "Players are grouped by role and ordered by minutes played.",
                                                className="team-profile-section-copy mb-0",
                                            ),
                                        ]
                                    )
                                ],
                                className="team-profile-section-heading",
                            ),
                            create_roster_section(roster),
                        ],
                        className="team-profile-roster-section",
                    ),
                    app_signature(),
                ],
                className="team-profile-shell",
            ),
        ],
        fluid=True,
        className="team-stats-page team-profile-page",
    )


@callback(
    Output("url", "search"),
    Input("team-profile-season-dropdown", "value"),
    State("team-profile-name-store", "data"),
    prevent_initial_call=True,
)
def change_season_url(selected_season, team_name):
    if not selected_season or not team_name:
        return no_update
    return f"?season={selected_season}"


@callback(
    Output("team-radar-compare-dropdown", "options"),
    Output("team-radar-compare-dropdown", "value"),
    Input("radar-normalization-filter", "value"),
    State("team-profile-season-dropdown", "value"),
    State("team-profile-league-name-store", "data"),
    State("team-profile-name-store", "data"),
)
def update_radar_comparison_options(scope, season, league_name, primary_team):
    comparison_league = league_name if scope == "league" else None
    return get_team_options(season, comparison_league, primary_team), None


@callback(
    Output("team-profile-radar", "figure"),
    Input("team-radar-compare-dropdown", "value"),
    Input("radar-normalization-filter", "value"),
    State("team-profile-name-store", "data"),
    State("team-profile-league-name-store", "data"),
    State("team-profile-season-dropdown", "value"),
)
def update_team_radar(comparison_team_name, scope, primary_team_name, league_name, season):
    from pages.team_stats import load_all_team_stats

    if not primary_team_name or not season:
        return go.Figure()
    all_teams = load_all_team_stats(season)
    cohort = all_teams[all_teams["League"] == league_name] if scope == "league" else all_teams
    if cohort.empty:
        return go.Figure()
    figure = team_plots.create_team_profile_radar(
        df_for_normalization=cohort,
        primary_team_name=primary_team_name,
        comparison_team_name=comparison_team_name,
    )
    comparison_label = league_name if scope == "league" else "Top 5 leagues"
    figure.update_layout(title_text=f"Percentile profile vs {comparison_label} · {season}")
    return figure
