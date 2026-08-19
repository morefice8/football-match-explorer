import json
import os
from urllib.parse import unquote

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

from src.components.brand_components import app_footer
from src.metrics.sportmonks import PLAYER_METRIC_GROUPS, uses_sportmonks
from src.utils.path_helpers import get_player_photo_path, get_team_logo_path
from src.utils.player_helpers import METRIC_TOOLTIPS, get_nationality_flag_code
from src.visualization import player_plots


DATA_PATH = os.path.join("data", "fbref")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
SPORTMONKS_DATA_PATH = os.path.join("data", "sportmonks", "processed")
MIN_AVG_MINUTES_PER_GAME = 30

ROLE_LABELS = {
    "GK": "Goalkeepers",
    "DF": "Defenders",
    "MF": "Midfielders",
    "FW": "Forwards",
}


def _clean_player_name(value):
    """Collapse regular and non-breaking whitespace in provider names."""
    return " ".join(str(value or "").split())


def _player_name_mask(series, player_name):
    cleaned = _clean_player_name(player_name).casefold()
    return series.astype(str).map(_clean_player_name).str.casefold().eq(cleaned)

ATTACKING_METRICS = [
    {"title": "Goals", "metric_col": "Gls", "icon": "fa-solid fa-futbol", "format_spec": "{:,.0f}"},
    {"title": "G-xG p90", "metric_col": "G_minus_xG_per_90", "icon": "fa-solid fa-chart-line"},
    {"title": "SCA p90", "metric_col": "SCA90", "icon": "fa-solid fa-wand-magic-sparkles"},
    {"title": "Shots on Target p90", "metric_col": "SoT/90", "icon": "fa-solid fa-bullseye"},
]
POSSESSION_METRICS = [
    {"title": "Assists", "metric_col": "Ast", "icon": "fa-solid fa-hands-helping", "format_spec": "{:,.0f}"},
    {"title": "Passes Final Third p90", "metric_col": "Passes_F3_per_90", "icon": "fa-solid fa-arrow-right-to-bracket"},
    {"title": "Progressive Passes p90", "metric_col": "PrgP_per_90", "icon": "fa-solid fa-angles-up"},
    {"title": "Carries Final Third p90", "metric_col": "Carries_F3_per_90", "icon": "fa-solid fa-arrow-trend-up"},
]
DEFENSIVE_METRICS = [
    {"title": "Tackles + Int p90", "metric_col": "Tkl+Int_per_90", "icon": "fa-solid fa-shield-halved"},
    {"title": "Aerial Duels Won %", "metric_col": "Aerial_Duels_perc", "icon": "fa-solid fa-plane-up", "format_spec": "{:,.1f}%"},
    {"title": "Clearances p90", "metric_col": "Clr_per_90", "icon": "fa-solid fa-broom"},
    {"title": "Blocks p90", "metric_col": "Blocks_per_90", "icon": "fa-solid fa-person-falling-burst"},
]
GOALKEEPING_METRICS = [
    {"title": "Save %", "metric_col": "Save%", "icon": "fa-solid fa-mitten", "format_spec": "{:,.1f}%"},
    {"title": "PSxG-GA", "metric_col": "PSxG+/-", "icon": "fa-solid fa-chart-line"},
    {"title": "Crosses Stopped %", "metric_col": "Stp%", "icon": "fa-solid fa-plane-slash", "format_spec": "{:,.1f}%"},
    {"title": "Sweeper Actions p90", "metric_col": "#OPA/90", "icon": "fa-solid fa-shoe-prints"},
]


def _profile_metric_definitions(group):
    definitions = []
    for spec in PLAYER_METRIC_GROUPS[group]:
        item = spec.card_kwargs("metric_col")
        if spec.unit:
            item["format_spec"] = f"{spec.format_spec}{spec.unit}"
        definitions.append(item)
    return definitions


SPORTMONKS_ATTACKING_METRICS = _profile_metric_definitions("attacking")
SPORTMONKS_POSSESSION_METRICS = _profile_metric_definitions("creation")
SPORTMONKS_DEFENSIVE_METRICS = _profile_metric_definitions("defending")
SPORTMONKS_GOALKEEPING_METRICS = _profile_metric_definitions("goalkeeping")


def get_metric_definitions(data):
    if uses_sportmonks(data):
        return (
            SPORTMONKS_ATTACKING_METRICS,
            SPORTMONKS_POSSESSION_METRICS,
            SPORTMONKS_DEFENSIVE_METRICS,
            SPORTMONKS_GOALKEEPING_METRICS,
        )
    return ATTACKING_METRICS, POSSESSION_METRICS, DEFENSIVE_METRICS, GOALKEEPING_METRICS


def get_available_player_seasons(player_name=None):
    if not os.path.isdir(PROCESSED_DATA_PATH):
        return []
    seasons = sorted({
        filename[len("player_stats_"):-len(".parquet")]
        for filename in os.listdir(PROCESSED_DATA_PATH)
        if filename.startswith("player_stats_") and filename.endswith(".parquet")
    }, reverse=True)
    if not player_name:
        return seasons

    available = []
    for season in seasons:
        path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
        try:
            players = pd.read_parquet(path, columns=["Player"])
            if _player_name_mask(players["Player"], player_name).any():
                available.append(season)
        except Exception:
            continue
    return available


def get_all_players_for_dropdown(season, position=None, exclude_player=None):
    path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
    if not os.path.exists(path):
        return []
    df = pd.read_parquet(path)
    if position and "Pos" in df.columns:
        df = df[df["Pos"].astype(str).eq(position)]
    if exclude_player:
        df = df[~_player_name_mask(df["Player"], exclude_player)]
    names = sorted({_clean_player_name(name) for name in df["Player"].dropna()})
    return [
        {"label": name, "value": name}
        for name in names if name
    ]


def _select_player_stats_row(rows, season):
    """For transfers, select the club from the player's latest season appearance."""
    if len(rows) <= 1:
        return rows.iloc[0]

    player_id = pd.to_numeric(rows.get("sportmonks_player_id"), errors="coerce").dropna()
    logs_path = os.path.join(SPORTMONKS_DATA_PATH, season, "player_matchlogs.parquet")
    if not player_id.empty and os.path.exists(logs_path):
        try:
            logs = pd.read_parquet(logs_path)
            log_ids = pd.to_numeric(logs.get("player_id"), errors="coerce")
            player_logs = logs[log_ids.eq(float(player_id.iloc[0]))].copy()
            if not player_logs.empty:
                played_minutes = pd.to_numeric(player_logs.get("minutes_played"), errors="coerce")
                appearances = player_logs[played_minutes.gt(0)]
                if not appearances.empty:
                    player_logs = appearances.copy()
                player_logs["_date"] = pd.to_datetime(player_logs.get("starting_at"), errors="coerce")
                latest = player_logs.sort_values("_date", na_position="first").iloc[-1]
                latest_team_id = pd.to_numeric(
                    pd.Series([latest.get("team_id")]), errors="coerce"
                ).iloc[0]
                row_team_ids = pd.to_numeric(rows.get("sportmonks_team_id"), errors="coerce")
                current_rows = rows[row_team_ids.eq(latest_team_id)]
                if not current_rows.empty:
                    return current_rows.iloc[0]
        except Exception:
            pass

    minutes = pd.to_numeric(rows.get("Min"), errors="coerce").fillna(0)
    return rows.loc[minutes.idxmax()]


def get_player_data(player_name, season):
    profile_path = os.path.join(
        DATA_PATH,
        "player_top5_europe",
        player_name.replace(" ", "_"),
        "profile.json",
    )
    profile_info = {}
    if os.path.exists(profile_path):
        with open(profile_path, "r", encoding="utf-8") as file:
            profile_info = json.load(file)

    stats_path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
    if not os.path.exists(stats_path):
        return {"error": f"Processed stats file for season {season} not found."}
    all_players = pd.read_parquet(stats_path)
    if "Player" in all_players:
        all_players["Player"] = all_players["Player"].map(_clean_player_name)
    rows = all_players[_player_name_mask(all_players["Player"], player_name)]
    if rows.empty:
        return {"error": f"Stats for {player_name} in season {season} not found."}
    stats = _select_player_stats_row(rows, season)

    if stats.get("Data_Source") == "Sportmonks":
        players_path = os.path.join(SPORTMONKS_DATA_PATH, season, "players.parquet")
        if os.path.exists(players_path):
            players = pd.read_parquet(players_path)
            profile = players[players["player_id"].eq(stats.get("sportmonks_player_id"))]
            if not profile.empty:
                item = profile.iloc[0]
                sportmonks_profile = {
                    "height_cm": item.get("height"),
                    "weight_kg": item.get("weight"),
                    "image_path": item.get("image_path"),
                    "birth_date": item.get("date_of_birth"),
                }
                for key, value in sportmonks_profile.items():
                    if pd.notna(value) and str(value).strip():
                        profile_info[key] = value

        teams_path = os.path.join(SPORTMONKS_DATA_PATH, season, "teams.parquet")
        if os.path.exists(teams_path):
            teams = pd.read_parquet(teams_path)
            team = teams[teams["team_id"].eq(stats.get("sportmonks_team_id"))]
            if not team.empty:
                profile_info["team_logo_path"] = team.iloc[0].get("image_path")

    return {"profile_info": profile_info, "stats_df": stats, "all_players": all_players}


def _qualified_role_cohort(all_players, player_stats, normalization_scope):
    cohort = all_players.copy()
    if not {"League_MP", "Min", "Pos"}.issubset(cohort.columns):
        return cohort.iloc[0:0].copy()
    cohort["League_MP"] = pd.to_numeric(cohort["League_MP"], errors="coerce")
    cohort["Min"] = pd.to_numeric(cohort["Min"], errors="coerce")
    cohort = cohort[cohort["League_MP"] > 0]
    cohort = cohort[cohort["Min"] >= cohort["League_MP"] * MIN_AVG_MINUTES_PER_GAME]
    role = str(player_stats.get("Pos", ""))
    cohort = cohort[cohort["Pos"].astype(str).eq(role)]
    if normalization_scope == "league":
        cohort = cohort[cohort["League"].eq(player_stats.get("League"))]
    return cohort.copy()


def get_dominant_role_and_usage(player_name, season):
    path = os.path.join(SPORTMONKS_DATA_PATH, season, "player_matchlogs.parquet")
    if os.path.exists(path):
        try:
            logs = pd.read_parquet(path)
            minutes = pd.to_numeric(logs.get("minutes_played"), errors="coerce").fillna(0)
            played = logs[logs["player_name"].eq(player_name) & minutes.gt(0)]
            if not played.empty:
                role = played["position"].dropna().mode()
                return (
                    role.iloc[0] if not role.empty else "N/A",
                    len(played),
                    int(pd.to_numeric(played["minutes_played"], errors="coerce").fillna(0).sum()),
                )
        except Exception:
            pass
    return "N/A", 0, 0


def _starter_mask(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.casefold().isin({"true", "1", "yes"})


def _tactical_role(position):
    """Translate a formation slot into a conventional 1–11 football role."""
    row = int(position["formation_row"])
    column = int(position.get("tactical_column", position["formation_column"]))
    width = max(int(position["row_width"]), 1)
    last_row = int(position["last_row"])
    last_width = max(int(position["last_width"]), 1)

    if width == 1:
        side = "Central"
    else:
        relative = (column - 1) / (width - 1)
        if relative <= 0.2:
            side = "Right"
        elif relative >= 0.8:
            side = "Left"
        elif relative < 0.5:
            side = "Right-centre"
        elif relative > 0.5:
            side = "Left-centre"
        else:
            side = "Central"

    if row == 1:
        return 1, "Goalkeeper"

    if row == 2:
        if width >= 4 and column == 1:
            return 2, "Right-back"
        if width >= 4 and column == width:
            return 3, "Left-back"
        if width == 3 and column == 1:
            return 4, "Right centre-back"
        if width == 3 and column == width:
            return 5, "Left centre-back"
        return (5, "Right centre-back") if side.startswith("Right") else (4, "Left centre-back")

    if row == last_row:
        if width == 1:
            return 9, "Centre-forward"
        if width == 2:
            return (9, "Right striker") if column == 1 else (10, "Left striker")
        if column == 1:
            return 7, "Right winger"
        if column == width:
            return 11, "Left winger"
        return 9, "Centre-forward"

    attacking_band = row == last_row - 1 and last_width == 1
    if attacking_band:
        if width == 1:
            return 10, "Attacking midfielder"
        if column == 1:
            return 7, "Right attacking midfielder"
        if column == width:
            return 11, "Left attacking midfielder"
        return 10, "Attacking midfielder"

    if width == 1:
        return 6, "Holding midfielder"
    if width == 2:
        return (8, "Right central midfielder") if column == 1 else (6, "Left central midfielder")
    if width == 3:
        if column == 1:
            return 8, "Right central midfielder"
        if column == width:
            return 8, "Left central midfielder"
        return 6, "Holding midfielder"
    if column == 1:
        return 7, "Right midfielder"
    if column == width:
        return 11, "Left midfielder"
    return (8, "Right central midfielder") if side.startswith("Right") else (6, "Left central midfielder")


def get_player_position_usage(player_name, season, player_id=None, team_id=None):
    path = os.path.join(SPORTMONKS_DATA_PATH, season, "player_matchlogs.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(), {"starts": 0, "mapped_starts": 0}
    try:
        logs = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame(), {"starts": 0, "mapped_starts": 0}
    required = {"player_name", "player_id", "starter", "formation_field", "fixture_id", "team_id"}
    if not required.issubset(logs.columns):
        return pd.DataFrame(), {"starts": 0, "mapped_starts": 0}

    minutes = pd.to_numeric(logs.get("minutes_played"), errors="coerce").fillna(0)
    logs = logs[minutes.gt(0)].copy()
    logs["is_starter"] = _starter_mask(logs["starter"])
    if pd.notna(player_id):
        selected_player = pd.to_numeric(logs["player_id"], errors="coerce").eq(float(player_id))
    else:
        selected_player = _player_name_mask(logs["player_name"], player_name)
    if pd.notna(team_id):
        selected_player &= pd.to_numeric(logs["team_id"], errors="coerce").eq(float(team_id))
    player_starts = logs[selected_player & logs["is_starter"]]
    total_starts = len(player_starts)

    formation = logs["formation_field"].astype(str).str.extract(r"^\s*(\d+):(\d+)\s*$")
    logs["formation_row"] = pd.to_numeric(formation[0], errors="coerce")
    logs["formation_column"] = pd.to_numeric(formation[1], errors="coerce")
    lineup = logs[
        logs["is_starter"]
        & logs["formation_row"].notna()
        & logs["formation_column"].notna()
    ].copy()
    if lineup.empty:
        return pd.DataFrame(), {"starts": total_starts, "mapped_starts": 0}

    lineup["row_width"] = lineup.groupby(
        ["fixture_id", "team_id", "formation_row"]
    )["formation_column"].transform("max")
    lineup["last_row"] = lineup.groupby(["fixture_id", "team_id"])["formation_row"].transform("max")
    last_widths = (
        lineup[lineup["formation_row"].eq(lineup["last_row"])]
        .drop_duplicates(["fixture_id", "team_id"])
        .set_index(["fixture_id", "team_id"])["row_width"]
    )
    lineup["last_width"] = pd.MultiIndex.from_frame(
        lineup[["fixture_id", "team_id"]]
    ).map(last_widths)

    formation_rows = (
        lineup[lineup["formation_row"].gt(1)]
        .groupby(["fixture_id", "team_id", "formation_row"], as_index=False)["row_width"]
        .max()
        .sort_values(["fixture_id", "team_id", "formation_row"])
    )
    shapes = formation_rows.groupby(["fixture_id", "team_id"])["row_width"].agg(
        lambda values: "-".join(str(int(value)) for value in values)
    )
    lineup["formation"] = pd.MultiIndex.from_frame(
        lineup[["fixture_id", "team_id"]]
    ).map(shapes)

    if pd.notna(player_id):
        selected_mask = pd.to_numeric(lineup["player_id"], errors="coerce").eq(float(player_id))
    else:
        selected_mask = _player_name_mask(lineup["player_name"], player_name)
    if pd.notna(team_id):
        selected_mask &= pd.to_numeric(lineup["team_id"], errors="coerce").eq(float(team_id))
    selected = lineup[selected_mask].copy()
    if selected.empty:
        return pd.DataFrame(), {"starts": total_starts, "mapped_starts": 0}

    selected["tactical_column"] = selected["formation_column"]
    if "location" in selected:
        away = selected["location"].astype(str).str.casefold().eq("away")
        selected.loc[away, "tactical_column"] = (
            selected.loc[away, "row_width"] + 1 - selected.loc[away, "formation_column"]
        )
    selected[["role_number", "role_name"]] = selected.apply(
        lambda row: pd.Series(_tactical_role(row)), axis=1
    )
    selected["x"] = 68 * (selected["row_width"] + 1 - selected["tactical_column"]) / (
        selected["row_width"] + 1
    )
    selected["y"] = 29 + (selected["formation_row"] - 2) * 65 / (
        selected["last_row"] - 2
    ).clip(lower=1)
    selected.loc[selected["formation_row"].eq(1), "y"] = 8

    role_keys = ["role_number", "role_name"]
    grouped = selected.groupby(role_keys, as_index=False).agg(
        starts=("fixture_id", "size"),
        x=("x", "mean"),
        y=("y", "mean"),
    )
    formation_counts = (
        selected.groupby(role_keys + ["formation"], dropna=False)
        .size()
        .reset_index(name="formation_starts")
        .sort_values(role_keys + ["formation_starts"], ascending=[True, True, False])
    )
    formation_counts["formation_item"] = formation_counts.apply(
        lambda row: (
            f"{row['formation'] if pd.notna(row['formation']) else 'Unknown shape'}"
            f" × {int(row['formation_starts'])}"
        ),
        axis=1,
    )
    formation_summary = (
        formation_counts.groupby(role_keys, as_index=False)["formation_item"]
        .agg(" · ".join)
        .rename(columns={"formation_item": "formation_summary"})
    )
    grouped = grouped.merge(formation_summary, on=role_keys, how="left")
    mapped_starts = int(grouped["starts"].sum())
    grouped["share"] = grouped["starts"] / max(mapped_starts, 1)
    grouped["label"] = grouped.apply(
        lambda row: f"No. {int(row['role_number'])} · {row['role_name']}", axis=1
    )
    return grouped.sort_values("starts", ascending=False), {
        "starts": total_starts,
        "mapped_starts": mapped_starts,
        "team": player_starts["team_name"].dropna().iloc[-1] if "team_name" in player_starts and not player_starts["team_name"].dropna().empty else None,
    }


def get_player_archetype(player_stats, comparison_cohort, dominant_pos):
    attacking, creation, defending, goalkeeping = get_metric_definitions(player_stats)
    metrics = attacking + creation + defending + goalkeeping
    percentiles = {}
    for metric in metrics:
        column = metric["metric_col"]
        if column not in comparison_cohort.columns:
            continue
        values = pd.to_numeric(comparison_cohort[column], errors="coerce")
        if values.notna().sum() == 0:
            continue
        lower_is_better = bool(metric.get("ascending", False))
        percentile = values.rank(pct=True, ascending=not lower_is_better)
        if "sportmonks_player_id" in comparison_cohort and pd.notna(player_stats.get("sportmonks_player_id")):
            cohort_ids = pd.to_numeric(comparison_cohort["sportmonks_player_id"], errors="coerce")
            player_match = cohort_ids.eq(float(player_stats.get("sportmonks_player_id")))
            if "sportmonks_team_id" in comparison_cohort and pd.notna(player_stats.get("sportmonks_team_id")):
                cohort_team_ids = pd.to_numeric(comparison_cohort["sportmonks_team_id"], errors="coerce")
                player_match &= cohort_team_ids.eq(float(player_stats.get("sportmonks_team_id")))
            player_index = comparison_cohort[player_match].index
        else:
            player_index = comparison_cohort[
                _player_name_mask(comparison_cohort["Player"], player_stats["Player"])
            ].index
        if len(player_index):
            percentiles[column] = percentile.loc[player_index[0]]

    p = lambda column, default=0: percentiles.get(column, default)
    is_sportmonks = player_stats.get("Data_Source") == "Sportmonks"
    goals = "Gls_per_90" if is_sportmonks else "Gls"
    creation_metric = "Chances_Created_per_90" if is_sportmonks else "SCA90"
    progression = "Passes_F3_per_90" if is_sportmonks else "PrgP_per_90"
    carrying = "Successful_Dribbles_per_90" if is_sportmonks else "Carries_F3_per_90"

    if dominant_pos == "GK":
        shot_stopping = "xGoT_minus_GA_per_90" if is_sportmonks else "PSxG+/-"
        if p(shot_stopping) > 0.9 and p("Save%") > 0.85:
            return "Elite Shot-Stopper", "fa-solid fa-star"
        return "Goalkeeper", "fa-solid fa-mitten"
    if dominant_pos == "FW":
        if p(goals) > 0.9 and p("G_minus_xG_per_90") > 0.8:
            return "Lethal Finisher", "fa-solid fa-bullseye"
        if p(goals) > 0.85 and p(creation_metric) > 0.85:
            return "Complete Forward", "fa-solid fa-star-of-life"
        if p(carrying) > 0.9:
            return "Dynamic Dribbler", "fa-solid fa-bolt"
        return "Forward", "fa-solid fa-person-running"
    if dominant_pos == "MF":
        creative = p(creation_metric) > 0.85 and p("Ast_per_90") > 0.7
        engine = p("Tkl+Int_per_90") > 0.75 and p(progression) > 0.75
        if creative and engine:
            return "Box-to-Box Maestro", "fa-solid fa-arrows-up-down"
        if creative:
            return "Creative Playmaker", "fa-solid fa-wand-magic-sparkles"
        if p(progression) > 0.9:
            return "Deep-Lying Playmaker", "fa-solid fa-compass-drafting"
        if p("Tkl+Int_per_90") > 0.85:
            return "Ball-Winning Midfielder", "fa-solid fa-anchor"
        return "Midfielder", "fa-solid fa-arrows-left-right"
    if dominant_pos == "DF":
        ball_player = p(progression) > 0.8
        dominant = p("Tkl+Int_per_90") > 0.8 and p("Aerial_Duels_perc") > 0.75
        if ball_player and dominant:
            return "Complete Defender", "fa-solid fa-chess-king"
        if ball_player:
            return "Ball-Playing Defender", "fa-solid fa-feather-pointed"
        if p("Aerial_Duels_perc") > 0.9:
            return "Aerial Dominator", "fa-solid fa-jet-fighter-up"
        return "Defender", "fa-solid fa-shield"
    return "Player", "fa-solid fa-user"


def _present(value):
    return pd.notna(value) and str(value).strip() and str(value).casefold() != "nan"


def _physical_value(value, unit):
    if not _present(value):
        return None
    try:
        return f"{float(value):,.0f} {unit}"
    except (TypeError, ValueError):
        return f"{value} {unit}"


def build_player_identity(player_name, season, archetype_text, archetype_icon):
    data = get_player_data(player_name, season)
    if "error" in data:
        return dbc.Alert(data["error"], color="danger", className="m-3")
    profile = data["profile_info"]
    stats = data["stats_df"]
    flag_code = get_nationality_flag_code(stats.get("Nationality_Code"), stats.get("Nationality"))
    flag = html.Img(
        src=f"https://cdnjs.cloudflare.com/ajax/libs/flag-icon-css/7.2.1/flags/4x3/{flag_code}.svg",
        className="player-profile-flag",
    ) if flag_code else None
    club_logo = profile.get("team_logo_path") or get_team_logo_path(stats.get("League"), stats.get("Club"))
    details = [
        ("Position", stats.get("Pos"), "fa-solid fa-location-dot"),
        ("Age", f"{int(stats.get('Age'))}" if pd.notna(stats.get("Age")) else None, "fa-solid fa-cake-candles"),
        ("Height", _physical_value(profile.get("height_cm"), "cm"), "fa-solid fa-ruler-vertical"),
        ("Weight", _physical_value(profile.get("weight_kg"), "kg"), "fa-solid fa-weight-scale"),
    ]
    detail_chips = [
        html.Div([
            html.I(className=f"{icon} player-profile-detail-icon"),
            html.Div([
                html.Span(label, className="player-profile-detail-label"),
                html.Strong(value, className="player-profile-detail-value"),
            ]),
        ], className="player-profile-detail-chip")
        for label, value, icon in details if _present(value)
    ]
    return html.Div([
        html.Img(
            src=profile.get("image_path") or get_player_photo_path(player_name),
            className="player-profile-photo",
        ),
        html.Div([
            html.Div([
                html.Span(str(stats.get("League", "League")).upper(), className="player-profile-badge"),
                html.Span("SPORTMONKS", className="player-profile-badge") if uses_sportmonks(stats) else None,
            ], className="player-profile-badges"),
            html.H1(player_name, className="player-profile-name"),
            html.Div([
                html.Img(src=club_logo, className="player-profile-club-logo"),
                html.Strong(stats.get("Club", "N/A"), className="player-profile-club-name"),
                html.Span("·", className="player-profile-separator"),
                flag,
                html.Span(stats.get("Nationality", "N/A"), className="player-profile-nationality"),
            ], className="player-profile-club-row"),
            html.Div([
                html.I(className=f"{archetype_icon} me-2"),
                archetype_text,
            ], className="player-profile-archetype"),
            html.Div(detail_chips, className="player-profile-details"),
        ], className="player-profile-copy"),
    ], className="player-profile-identity-grid")


def _format_kpi(value, format_spec):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return format_spec.format(numeric) if pd.notna(numeric) else "—"


def build_usage_snapshot(stats):
    appearances = pd.to_numeric(pd.Series([stats.get("MP")]), errors="coerce").fillna(0).iloc[0]
    minutes = pd.to_numeric(pd.Series([stats.get("Min")]), errors="coerce").fillna(0).iloc[0]
    is_gk = str(stats.get("Pos", "")) == "GK"
    items = [
        ("Appearances", stats.get("MP"), "{:,.0f}", "fa-solid fa-shirt"),
        ("Starts", stats.get("Starts"), "{:,.0f}", "fa-solid fa-flag-checkered"),
        ("Minutes", stats.get("Min"), "{:,.0f}", "fa-solid fa-clock"),
        ("Minutes / App", minutes / appearances if appearances else np.nan, "{:,.1f}", "fa-solid fa-gauge"),
    ]
    if is_gk:
        items.extend([
            ("Saves", stats.get("Saves"), "{:,.0f}", "fa-solid fa-hands"),
            ("Save %", stats.get("Save%"), "{:,.1f}%", "fa-solid fa-mitten"),
        ])
    else:
        items.extend([
            ("Goals", stats.get("Gls"), "{:,.0f}", "fa-solid fa-futbol"),
            ("Assists", stats.get("Ast"), "{:,.0f}", "fa-solid fa-handshake-angle"),
        ])
    return html.Div([
        html.Div([
            html.Div(html.I(className=icon), className="player-profile-kpi-icon"),
            html.Div([
                html.Strong(_format_kpi(value, fmt), className="player-profile-kpi-value"),
                html.Span(label, className="player-profile-kpi-label"),
            ]),
        ], className="player-profile-kpi")
        for label, value, fmt, icon in items
    ], className="player-profile-kpi-grid")


def build_position_summary(usage, metadata):
    if usage.empty:
        return html.Div(
            html.P("No spatial starting-position data is available for this season.", className="mb-0"),
            className="player-position-empty",
        )
    mapped = metadata.get("mapped_starts", 0)
    total = metadata.get("starts", 0)
    rows = []
    for index, (_, position) in enumerate(usage.head(4).iterrows()):
        rows.append(html.Div([
            html.Span(className=f"player-position-dot {'primary' if index == 0 else ''}"),
            html.Div([
                html.Span(str(position["label"]), className="player-position-name"),
                html.Span(
                    str(position.get("formation_summary") or "Formation unavailable"),
                    className="player-position-formation",
                ),
            ], className="player-position-role-copy"),
            html.Strong(f"{int(position['starts'])} · {position['share']:.0%}", className="player-position-frequency"),
        ], className="player-position-row"))
    coverage = f"{mapped}/{total} starts mapped" if total else f"{mapped} starts mapped"
    return html.Div([
        html.Div(rows, className="player-position-list"),
        html.P([
            html.I(className="fa-solid fa-circle-info me-2"),
            coverage,
            f" for {metadata.get('team')}" if metadata.get("team") else "",
            ". Circle size represents frequency; the number is the conventional tactical role, not the shirt number. In-match changes and substitute locations are not available.",
        ], className="player-position-note mb-0"),
    ])


def _metric_value(value, format_spec):
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "—"
    try:
        return format_spec.format(numeric)
    except (ValueError, KeyError):
        return f"{numeric:,.2f}"


def create_metric_stat_card(metric, player_stats, rank_info, group_key):
    column = metric["metric_col"]
    info = rank_info.get(column, {}) if isinstance(rank_info, dict) else {}
    percentile = info.get("percentile")
    eligible = info.get("eligible", True)
    if eligible and info.get("rank") and info.get("total"):
        comparison = f"#{info['rank']} of {info['total']}"
    elif not eligible:
        comparison = "Small sample · no rank"
    else:
        comparison = "Rank unavailable"
    tooltip = METRIC_TOOLTIPS.get(column, "No description available.")
    card_id = "player-profile-metric-" + "".join(char if char.isalnum() else "-" for char in column)
    return html.Div([
        html.Div([
            html.I(className=f"{metric['icon']} player-profile-metric-icon"),
            html.Span(metric["title"], className="player-profile-metric-label"),
        ], id=card_id, className="player-profile-metric-label-row"),
        html.Strong(
            _metric_value(player_stats.get(column), metric.get("format_spec", "{:,.2f}")),
            className="player-profile-metric-value",
        ),
        html.Div([
            html.Span(comparison),
            html.Span(f"{percentile:.0%}ile" if percentile is not None else "—"),
        ], className="player-profile-metric-comparison"),
        html.Div(
            html.Div(
                className="player-profile-percentile-fill",
                style={"width": f"{max(0, min(float(percentile or 0), 1)) * 100:.1f}%"},
            ),
            className="player-profile-percentile-track",
        ),
        dbc.Tooltip(tooltip, target=card_id, placement="top"),
    ], className=f"player-profile-metric-tile metric-group-{group_key}")


def create_metric_category_card(title, metrics, player_stats, ranks, group_key):
    return html.Div([
        html.Div([
            html.H3(title, className="mb-1"),
            html.P("Role-adjusted ranking within the selected comparison scope.", className="mb-0"),
        ], className=f"player-profile-metric-group-header metric-group-{group_key}"),
        html.Div([
            create_metric_stat_card(metric, player_stats, ranks, group_key)
            for metric in metrics
        ], className="player-profile-metric-tile-grid"),
    ], className="player-profile-metric-group")


def layout(player_name_url, initial_season=None):
    player_name = _clean_player_name(unquote(player_name_url).replace("_", " "))
    available_seasons = get_available_player_seasons(player_name)
    if not available_seasons:
        return dbc.Container(
            dbc.Alert(f"No processed seasons found for {player_name}.", color="warning"),
            className="mt-4",
        )
    default_season = initial_season if initial_season in available_seasons else available_seasons[0]

    return html.Main([
        dbc.Container([
            html.Div([
        dcc.Store(id="player-profile-name-store", data=player_name),
        dcc.Store(id="player-profile-kpi-ranks-store"),
        html.Section([
            html.Div([
                dbc.Button(
                    [html.I(className="fas fa-arrow-left me-2"), "Player Statistics"],
                    href=f"/player-stats?season={default_season}",
                    color="light",
                    outline=True,
                    className="player-profile-back-btn",
                ),
                html.Div([
                    html.Span("SEASON", className="player-profile-season-label"),
                    dcc.Dropdown(
                        id="player-profile-season-dropdown",
                        options=[{"label": season, "value": season} for season in available_seasons],
                        value=default_season,
                        clearable=False,
                        className="player-profile-season-dropdown",
                    ),
                ], className="player-profile-season-control"),
            ], className="player-profile-hero-toolbar"),
            dcc.Loading(html.Div(id="player-profile-identity"), type="circle"),
        ], className="player-profile-hero"),

        dcc.Loading(html.Div(id="player-usage-stats-row"), type="circle"),

        dbc.Row([
            dbc.Col(
                html.Section([
                    html.Div([
                        html.Div([
                            html.H2("Starting-position profile", className="mb-1"),
                            html.P("Where the player starts within the team formation.", className="mb-0"),
                        ]),
                        html.I(className="fa-solid fa-location-crosshairs player-profile-panel-heading-icon"),
                    ], className="player-profile-panel-header"),
                    dcc.Loading(dcc.Graph(
                        id="player-position-map",
                        config={"displayModeBar": False, "responsive": True},
                        className="player-position-map",
                    )),
                    html.Div(id="player-position-summary", className="player-position-summary"),
                ], className="player-profile-panel h-100"),
                lg=4,
                md=12,
                className="mb-4",
            ),
            dbc.Col(
                html.Section([
                    html.Div([
                        html.Div([
                            html.H2("Statistical profile", className="mb-1"),
                            html.P("Percentile performance against comparable players.", className="mb-0"),
                        ]),
                        html.I(className="fa-solid fa-chart-simple player-profile-panel-heading-icon"),
                    ], className="player-profile-panel-header"),
                    html.Div([
                        dbc.Row([
                            dbc.Col(
                                dcc.Dropdown(
                                    id="player-radar-compare-dropdown",
                                    placeholder="Compare with another player…",
                                    className="player-profile-compare-dropdown",
                                ),
                                lg=7,
                                md=12,
                            ),
                            dbc.Col(
                                dbc.RadioItems(
                                    id="player-radar-norm-filter",
                                    options=[
                                        {"label": "vs League", "value": "league"},
                                        {"label": "vs Top 5", "value": "top5"},
                                    ],
                                    value="league",
                                    inline=True,
                                    className="player-profile-radar-scope",
                                    inputClassName="btn-check",
                                    labelClassName="btn",
                                    labelCheckedClassName="active",
                                ),
                                lg=5,
                                md=12,
                                className="d-flex justify-content-lg-end",
                            ),
                        ], className="g-3 align-items-center"),
                        html.Div(id="player-radar-context", className="player-profile-radar-context"),
                        html.Div(id="player-radar-legend", className="player-profile-radar-html-legend"),
                        dcc.Loading(dcc.Graph(
                            id="player-profile-radar",
                            config={"displayModeBar": False, "responsive": True},
                            className="player-profile-radar-graph",
                        )),
                        html.P([
                            html.I(className="fa-solid fa-circle-info me-2"),
                            "Percentiles compare players in the same position. Higher always represents stronger relative performance; lower-is-better metrics are inverted.",
                        ], className="player-profile-method-note mb-0"),
                    ], className="player-profile-radar-body"),
                ], className="player-profile-panel h-100"),
                lg=8,
                md=12,
                className="mb-4",
            ),
        ], className="g-4"),

        html.Section([
            html.Div([
                html.Div([
                    html.H2("Performance metrics", className="mb-1"),
                    html.P("Output, creation and defensive contribution with rank and percentile context.", className="mb-0"),
                ]),
                html.Div([
                    html.I(className="fa-solid fa-users-viewfinder me-2"),
                    "Same-position cohort",
                ], className="player-profile-cohort-badge"),
            ], className="player-profile-section-heading"),
            dcc.Loading(html.Div(id="player-profile-metric-cards"), type="circle"),
        ], className="player-profile-metrics-section"),
            ], className="player-profile-content-shell"),
        ], fluid=True, className="player-profile-container px-3 px-lg-4 py-4"),
        app_footer("Individual performance, positional usage and role-adjusted comparison."),
    ], className="player-profile-page branded-profile-page")


@callback(
    Output("url", "search", allow_duplicate=True),
    Input("player-profile-season-dropdown", "value"),
    prevent_initial_call=True,
)
def sync_player_profile_season_url(season):
    return f"?season={season}" if season else no_update


@callback(
    Output("player-radar-compare-dropdown", "options"),
    Output("player-radar-compare-dropdown", "value"),
    Input("player-profile-season-dropdown", "value"),
    State("player-profile-name-store", "data"),
    State("player-radar-compare-dropdown", "value"),
)
def update_player_compare_options(season, player_name, current_value):
    if not season or not player_name:
        return [], None
    data = get_player_data(player_name, season)
    if "error" in data:
        return [], None
    position = str(data["stats_df"].get("Pos", ""))
    options = get_all_players_for_dropdown(season, position, player_name)
    values = {option["value"] for option in options}
    return options, current_value if current_value in values else None


@callback(
    Output("player-profile-identity", "children"),
    Output("player-usage-stats-row", "children"),
    Output("player-position-map", "figure"),
    Output("player-position-summary", "children"),
    Input("player-profile-season-dropdown", "value"),
    Input("player-radar-norm-filter", "value"),
    State("player-profile-name-store", "data"),
)
def update_player_profile_summary(season, normalization_scope, player_name):
    if not season or not player_name:
        return no_update, no_update, go.Figure(), no_update
    data = get_player_data(player_name, season)
    if "error" in data:
        alert = dbc.Alert(data["error"], color="danger")
        return alert, alert, go.Figure(), alert
    stats = data["stats_df"]
    role = str(stats.get("Pos", "N/A"))
    cohort = _qualified_role_cohort(data["all_players"], stats, normalization_scope)
    archetype, icon = get_player_archetype(stats, cohort, role)
    identity = build_player_identity(player_name, season, archetype, icon)
    usage_snapshot = build_usage_snapshot(stats)
    position_usage, position_meta = get_player_position_usage(
        player_name,
        season,
        player_id=stats.get("sportmonks_player_id"),
        team_id=stats.get("sportmonks_team_id"),
    )
    position_figure = player_plots.create_player_position_map(position_usage)
    position_summary = build_position_summary(position_usage, position_meta)
    return identity, usage_snapshot, position_figure, position_summary


@callback(
    Output("player-profile-kpi-ranks-store", "data"),
    Input("player-radar-norm-filter", "value"),
    Input("player-profile-season-dropdown", "value"),
    State("player-profile-name-store", "data"),
)
def update_kpi_rankings(normalization_scope, season, player_name):
    if not player_name or not season:
        return {}
    data = get_player_data(player_name, season)
    if "error" in data:
        return {}
    stats = data["stats_df"]
    cohort = _qualified_role_cohort(data["all_players"], stats, normalization_scope)
    attacking, creation, defending, goalkeeping = get_metric_definitions(stats)
    metrics = goalkeeping if str(stats.get("Pos")) == "GK" else attacking + creation + defending
    rank_info = {}
    player_minutes = pd.to_numeric(pd.Series([stats.get("Min")]), errors="coerce").fillna(0).iloc[0]
    league_mp = pd.to_numeric(pd.Series([stats.get("League_MP")]), errors="coerce").fillna(0).iloc[0]
    minutes_eligible = league_mp > 0 and player_minutes >= league_mp * MIN_AVG_MINUTES_PER_GAME

    for metric in metrics:
        column = metric["metric_col"]
        if column not in cohort.columns:
            rank_info[column] = {"eligible": False}
            continue
        metric_cohort = cohort.copy()
        minimum_column = metric.get("minimum_column")
        minimum_value = metric.get("minimum_value")
        opportunity_eligible = True
        if minimum_column and minimum_value is not None:
            if minimum_column not in metric_cohort.columns:
                rank_info[column] = {"eligible": False}
                continue
            opportunities = pd.to_numeric(metric_cohort[minimum_column], errors="coerce").fillna(0)
            metric_cohort = metric_cohort[opportunities >= minimum_value]
            player_opportunities = pd.to_numeric(
                pd.Series([stats.get(minimum_column)]), errors="coerce"
            ).fillna(0).iloc[0]
            opportunity_eligible = player_opportunities >= minimum_value

        values = pd.to_numeric(metric_cohort[column], errors="coerce")
        metric_cohort = metric_cohort[values.notna()].copy()
        metric_cohort[column] = pd.to_numeric(metric_cohort[column], errors="coerce")
        eligible = bool(minutes_eligible and opportunity_eligible)
        if "sportmonks_player_id" in metric_cohort and pd.notna(stats.get("sportmonks_player_id")):
            cohort_ids = pd.to_numeric(metric_cohort["sportmonks_player_id"], errors="coerce")
            player_rows = metric_cohort[
                cohort_ids.eq(float(stats.get("sportmonks_player_id")))
                & pd.to_numeric(metric_cohort.get("sportmonks_team_id"), errors="coerce").eq(
                    float(stats.get("sportmonks_team_id"))
                )
            ]
        else:
            player_rows = metric_cohort[_player_name_mask(metric_cohort["Player"], player_name)]
        if not eligible or player_rows.empty:
            rank_info[column] = {"eligible": eligible, "total": len(metric_cohort)}
            continue
        lower_is_better = bool(metric.get("ascending", False))
        metric_cohort["_profile_player"] = metric_cohort.index.isin(player_rows.index)
        ordered = metric_cohort.sort_values(column, ascending=lower_is_better).reset_index(drop=True)
        rank = int(ordered.index[ordered["_profile_player"]][0]) + 1
        percentiles = metric_cohort[column].rank(pct=True, ascending=not lower_is_better)
        percentile = float(percentiles.loc[player_rows.index[0]])
        rank_info[column] = {
            "eligible": True,
            "rank": rank,
            "total": len(metric_cohort),
            "percentile": percentile,
        }
    return rank_info


@callback(
    Output("player-profile-metric-cards", "children"),
    Input("player-profile-kpi-ranks-store", "data"),
    Input("player-profile-season-dropdown", "value"),
    State("player-profile-name-store", "data"),
)
def render_metric_macro_cards(ranks, season, player_name):
    if not player_name or not season:
        return dbc.Alert("Select a player and season to see stats.", color="info")
    data = get_player_data(player_name, season)
    if "error" in data:
        return dbc.Alert(data["error"], color="danger")
    stats = data["stats_df"]
    attacking, creation, defending, goalkeeping = get_metric_definitions(stats)
    if str(stats.get("Pos")) == "GK":
        cards = [create_metric_category_card("🧤 Goalkeeping", goalkeeping, stats, ranks or {}, "goalkeeping")]
    else:
        cards = [
            create_metric_category_card("⚔️ Attacking", attacking, stats, ranks or {}, "attacking"),
            create_metric_category_card("⚽ Creation & Ball Use", creation, stats, ranks or {}, "creation"),
            create_metric_category_card("🛡️ Defending", defending, stats, ranks or {}, "defending"),
        ]
    return html.Div(cards, className=f"player-profile-metric-groups {'goalkeeper' if len(cards) == 1 else ''}")


@callback(
    Output("player-profile-radar", "figure"),
    Output("player-radar-context", "children"),
    Output("player-radar-legend", "children"),
    Input("player-radar-compare-dropdown", "value"),
    Input("player-radar-norm-filter", "value"),
    Input("player-profile-season-dropdown", "value"),
    State("player-profile-name-store", "data"),
)
def update_player_radar(comparison_player_name, normalization_scope, season, primary_player_name):
    if not primary_player_name or not season:
        return go.Figure(), "", None
    data = get_player_data(primary_player_name, season)
    if "error" in data:
        return go.Figure().update_layout(title_text=data["error"]), "", None
    all_players = data["all_players"]
    primary = data["stats_df"].copy()
    cohort = _qualified_role_cohort(all_players, primary, normalization_scope)
    comparison = None
    if comparison_player_name:
        comparison_data = get_player_data(comparison_player_name, season)
        if "error" not in comparison_data:
            candidate = comparison_data["stats_df"]
            if str(candidate.get("Pos")) == str(primary.get("Pos")):
                comparison = candidate
    figure = player_plots.create_player_profile_radar(
        df_for_normalization=cohort,
        primary_player_series=primary,
        comparison_player_series=comparison,
    )
    role_label = ROLE_LABELS.get(str(primary.get("Pos")), "Players")
    if normalization_scope == "league":
        context = f"vs {primary.get('League')} {role_label}"
    else:
        context = f"vs Top 5 {role_label}"
    figure.update_layout(title_text=None)
    legend = None
    if comparison is not None:
        legend = [
            html.Span([
                html.Span(className="player-radar-legend-swatch primary"),
                str(primary.get("Player")),
            ], className="player-radar-legend-item"),
            html.Span([
                html.Span(className="player-radar-legend-swatch comparison"),
                str(comparison.get("Player")),
            ], className="player-radar-legend-item"),
        ]
    return figure, f"Percentile ranks · {context}", legend
