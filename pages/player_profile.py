# --- START OF FILE pages/player_profile.py ---
import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from io import StringIO
from urllib.parse import unquote
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

# Importa le funzioni di utilità e visualizzazione
from src.utils.path_helpers import get_player_photo_path, get_team_logo_path
from src.utils.player_helpers import LEAGUES, METRIC_TOOLTIPS
from src.visualization import player_plots
from src.components.layout_components import app_signature

# --- CONFIGURATION ---
DATA_PATH = os.path.join("data", "fbref")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
MIN_AVG_MINUTES_PER_GAME = 60

# --- METRIC DEFINITIONS (invariate) ---
ATTACKING_METRICS = [{'title': 'Goals', 'metric_col': 'Gls', 'icon': 'fa-solid fa-futbol', 'format_spec': '{:,.0f}'}, {'title': 'G-xG p90', 'metric_col': 'G_minus_xG_per_90', 'icon': 'fa-solid fa-chart-line'}, {'title': 'SCA p90', 'metric_col': 'SCA90', 'icon': 'fa-solid fa-wand-magic-sparkles'}, {'title': 'Shots on Target p90', 'metric_col': 'SoT/90', 'icon': 'fa-solid fa-bullseye'}]
POSSESSION_METRICS = [{'title': 'Assists', 'metric_col': 'Ast', 'icon': 'fa-solid fa-hands-helping', 'format_spec': '{:,.0f}'}, {'title': 'Passes Final Third p90', 'metric_col': 'Passes_F3_per_90', 'icon': 'fa-solid fa-arrow-right-to-bracket'}, {'title': 'Progressive Passes p90', 'metric_col': 'PrgP_per_90', 'icon': 'fa-solid fa-angles-up'}, {'title': 'Carries Final Third p90', 'metric_col': 'Carries_F3_per_90', 'icon': 'fa-solid fa-arrow-trend-up'}]
DEFENSIVE_METRICS = [{'title': 'Tackles + Int p90', 'metric_col': 'Tkl+Int_per_90', 'icon': 'fa-solid fa-shield-halved'}, {'title': 'Aerial Duels Won %', 'metric_col': 'Aerial_Duels_perc', 'icon': 'fa-solid fa-plane-up', 'format_spec': '{:,.1f}%'}, {'title': 'Clearances p90', 'metric_col': 'Clr_per_90', 'icon': 'fa-solid fa-broom'}, {'title': 'Blocks p90', 'metric_col': 'Blocks_per_90', 'icon': 'fa-solid fa-person-falling-burst'}]
GOALKEEPING_METRICS = [{'title': 'Save %', 'metric_col': 'Save%', 'icon': 'fa-solid fa-mitten', 'format_spec': '{:,.1f}%'}, {'title': 'PSxG-GA', 'metric_col': 'PSxG+/-', 'icon': 'fa-solid fa-chart-line'}, {'title': 'Crosses Stopped %', 'metric_col': 'Stp%', 'icon': 'fa-solid fa-plane-slash', 'format_spec': '{:,.1f}%'}, {'title': 'Sweeper Actions p90', 'metric_col': '#OPA/90', 'icon': 'fa-solid fa-shoe-prints'}]
ALL_METRICS = ATTACKING_METRICS + POSSESSION_METRICS + DEFENSIVE_METRICS + GOALKEEPING_METRICS

# --- DATA LOADING & HELPER FUNCTIONS (invariate) ---
# ... (il codice di get_all_players_for_dropdown, get_player_data, get_dominant_role_and_usage, get_player_archetype, create_metric_stat_card, create_metric_category_card è identico) ...
def get_all_players_for_dropdown(season):
    file_path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
    if not os.path.exists(file_path): return []
    df = pd.read_parquet(file_path)
    return [{'label': name, 'value': name} for name in sorted(df['Player'].unique())]

def get_player_data(player_name, season):
    player_folder_name = player_name.replace(' ', '_')
    player_folder_path = os.path.join(DATA_PATH, "player_top5_europe", player_folder_name)
    profile_path = os.path.join(player_folder_path, "profile.json")
    profile_info = {}
    if os.path.exists(profile_path):
        with open(profile_path, 'r', encoding='utf-8') as f: profile_info = json.load(f)
    stats_file_path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
    if not os.path.exists(stats_file_path): return {'error': f'Processed stats file for season {season} not found.'}
    df_all_players = pd.read_parquet(stats_file_path)
    player_stats = df_all_players[df_all_players['Player'] == player_name]
    if player_stats.empty: return {'error': f'Stats for {player_name} in season {season} not found.'}
    return {'profile_info': profile_info, 'stats_df': player_stats.iloc[0]}

def get_dominant_role_and_usage(player_name, season):
    player_folder_name = str(player_name).replace(' ', '_')
    matchlog_path = os.path.join(DATA_PATH, "player_top5_europe", player_folder_name, season, "player_matchlogs.csv")
    if not os.path.exists(matchlog_path): return 'N/A', 0, 0
    try:
        df_logs = pd.read_csv(matchlog_path)
        df_played = df_logs[pd.to_numeric(df_logs['Min'], errors='coerce').fillna(0) > 0]
        if df_played.empty: return 'N/A', 0, 0
        dominant_pos = df_played['Pos'].mode()[0] if 'Pos' in df_played.columns and not df_played['Pos'].dropna().empty else 'N/A'
        matches_played = len(df_played)
        minutes_played = int(df_played['Min'].sum())
        return dominant_pos, matches_played, minutes_played
    except Exception: return 'N/A', 0, 0

def get_player_archetype(player_stats, df_comp, dominant_pos):
    player_name = player_stats['Player']
    percentiles = {}
    for metric in ALL_METRICS:
        col = metric['metric_col']
        if col in df_comp.columns:
            is_gk_metric = col in [m['metric_col'] for m in GOALKEEPING_METRICS]
            df_role_comp = df_comp[df_comp['Pos'].str.contains('GK', na=False)] if is_gk_metric else df_comp[~df_comp['Pos'].str.contains('GK', na=False)]
            if not df_role_comp.empty and col in df_role_comp.columns:
                ranks = df_role_comp[col].rank(pct=True)
                player_rank_index = df_role_comp[df_role_comp['Player'] == player_name].index
                if not player_rank_index.empty: percentiles[col] = ranks.loc[player_rank_index[0]]
    p = lambda col, default=0: percentiles.get(col, default)
    if 'GK' in dominant_pos:
        if p('PSxG+/-') > 0.9 and p('Save%') > 0.85: return "Elite Shot-Stopper", "fa-solid fa-star"
        if p('#OPA/90') > 0.9: return "Sweeper Keeper", "fa-solid fa-shoe-prints"
        if p('Stp%') > 0.85: return "Cross Dominator", "fa-solid fa-plane-slash"
        return "Goalkeeper", "fa-solid fa-mitten"
    if 'FW' in dominant_pos or 'W' in dominant_pos:
        if p('Gls') > 0.9 and p('G_minus_xG_per_90') > 0.8: return "Lethal Finisher", "fa-solid fa-bullseye"
        if p('SCA90') > 0.9 and p('Ast') > 0.8: return "Elite Creator", "fa-solid fa-gears"
        if p('Gls') > 0.85 and p('SCA90') > 0.85: return "Complete Forward", "fa-solid fa-star-of-life"
        if p('Carries_F3_per_90') > 0.9: return "Dynamic Dribbler", "fa-solid fa-bolt"
        return "Forward", "fa-solid fa-person-running"
    if 'MF' in dominant_pos or 'M' in dominant_pos:
        is_creative = p('SCA90') > 0.85 and p('Ast') > 0.7
        is_engine = p('Tkl+Int_per_90') > 0.75 and p('PrgP_per_90') > 0.75
        if is_creative and is_engine: return "Box-to-Box Maestro", "fa-solid fa-arrows-up-down"
        if is_creative: return "Creative Playmaker", "fa-solid fa-wand-magic-sparkles"
        if p('PrgP_per_90') > 0.9: return "Deep-Lying Playmaker", "fa-solid fa-compass-drafting"
        if p('Tkl+Int_per_90') > 0.85: return "Defensive Midfielder", "fa-solid fa-anchor"
        return "Midfielder", "fa-solid fa-arrows-left-right"
    if 'DF' in dominant_pos or 'B' in dominant_pos:
        is_ball_player = p('PrgP_per_90') > 0.8 and p('Passes_F3_per_90') > 0.75
        is_dominant = p('Tkl+Int_per_90') > 0.8 and p('Aerial_Duels_perc') > 0.75
        if is_ball_player and is_dominant: return "Complete Defender", "fa-solid fa-chess-king"
        if is_ball_player: return "Ball-Playing Defender", "fa-solid fa-feather-pointed"
        if p('Aerial_Duels_perc') > 0.9: return "Aerial Dominator", "fa-solid fa-jet-fighter-up"
        if p('Tkl+Int_per_90') > 0.9: return "Defensive Rock", "fa-solid fa-shield-halved"
        return "Defender", "fa-solid fa-shield"
    return "Player", "fa-solid fa-user"

def create_metric_stat_card(title, value, icon, rank=None, format_spec="{:,.2f}", tooltip_text=""):
    display_value = format_spec.format(value) if isinstance(value, (int, float)) and pd.notna(value) else value
    rank_str = f"#{rank}" if rank is not None else ""
    card_id = f"stat-card-{title.replace(' ', '-')}"
    return dbc.Col(dbc.Card([dbc.CardBody([html.H6([html.I(className=f"{icon} me-2"), title], className="card-title text-muted", style={'font-size': '0.9rem'}), html.Div(display_value, className="fw-bold fs-3 text-white text-center"), html.Div(rank_str, className="text-secondary", style={"font-size": "1rem", "position": "absolute", "bottom": "5px", "right": "10px", "font-weight": "bold"})], className="p-2"), dbc.Tooltip(tooltip_text, target=card_id, placement="top")], id=card_id, className="position-relative bg-dark text-center h-100 shadow-sm"), md=6, className="mb-2")
def create_metric_category_card(title, metrics_list, player_stats, ranks, card_color_rgba):
    card_content = []
    for metric in metrics_list:
        metric_col = metric['metric_col']
        value = player_stats.get(metric_col, 'N/A')
        rank = ranks.get(metric_col)
        tooltip = METRIC_TOOLTIPS.get(metric_col, "No description available.")
        card_content.append(create_metric_stat_card(title=metric['title'], value=value, icon=metric['icon'], rank=rank, format_spec=metric.get('format_spec', "{:,.2f}"), tooltip_text=tooltip))
    return dbc.Card([dbc.CardHeader(html.H5(title, className="m-0 text-white"), className="bg-dark"), dbc.CardBody(dbc.Row(card_content, className="g-2"))], className="mb-4 shadow", style={"background-color": card_color_rgba})

# --- LAYOUT (invariato) ---
def layout(player_name_url):
    player_name = unquote(player_name_url).replace('_', ' ')
    available_seasons = ["2024-2025", "2023-2024"]
    default_season = available_seasons[0]
    player_data = get_player_data(player_name, default_season)
    if 'error' in player_data: return dbc.Container(dbc.Alert(player_data['error'], color="danger"), className="mt-4")
    profile_info, stats = player_data['profile_info'], player_data['stats_df']
    header = dbc.Row([
        dbc.Col(dbc.Button([html.I(className="fas fa-arrow-left me-2"), "Hub"], href="/player-stats", color="secondary", outline=True, size="sm"), width="auto", className="align-self-start"),
        dbc.Col(html.Img(src=get_player_photo_path(player_name), style={'height': '120px', 'width': '120px', 'object-fit': 'cover', 'border-radius': '50%'}), width="auto"),
        dbc.Col([html.H1(player_name, className="text-white mb-0"), html.H4([html.Img(src=get_team_logo_path(stats.get('League'), stats.get('Club')), style={'height': '24px', 'margin-right': '8px'}), stats.get('Club', 'N/A')], className="text-muted"), html.H6([html.Img(src=f"https://cdnjs.cloudflare.com/ajax/libs/flag-icon-css/7.2.1/flags/4x3/{stats.get('Nationality_Code', '')}.svg", style={'width': '20px', 'margin-right': '8px'}), stats.get('Nationality', 'N/A')], className="text-white d-flex align-items-center"), html.Div(id='player-archetype-badge', className="mt-2")], width=True, className="align-self-center"),
        dbc.Col([dbc.Row([dbc.Col(html.Div([html.Strong("Pos:"), f" {stats.get('Pos', 'N/A')}"]), width=12), dbc.Col(html.Div([html.Strong("Age:"), f" {int(stats.get('Age', 0)) if pd.notna(stats.get('Age')) else 'N/A'}"]), width=6), dbc.Col(html.Div([html.Strong("Foot:"), f" {profile_info.get('footed', 'N/A')}"]), width=6), dbc.Col(html.Div([html.Strong("Height:"), f" {profile_info.get('height_cm', 'N/A')}"]), width=6), dbc.Col(html.Div([html.Strong("Weight:"), f" {profile_info.get('weight_kg', 'N/A')}"]), width=6), dbc.Col(html.Div([html.Strong("Wages:"), f"{profile_info.get('wages_weekly_euro', 'N/A')}"]), width=12), dbc.Col(html.Div([html.Strong("Expires:"), f" {profile_info.get('contract_expires', 'N/A')}"]), width=12)], className="text-white small g-1")], lg=3, md=12, className="align-self-center border-start border-secondary ps-3"),
    ], align="center", className="my-4 p-3 bg-dark rounded shadow-lg")
    return dbc.Container([
        dcc.Store(id='player-profile-name-store', data=player_name), dcc.Store(id='player-profile-kpi-ranks-store'), header,
        html.Div(id='player-usage-stats-row', className="mb-4"),
        dbc.Row([dbc.Col(dcc.Loading(html.Div(id='player-profile-metric-cards')), md=5), dbc.Col([dbc.Card([dbc.CardHeader("Player Skill Radar"), dbc.CardBody([dbc.Row([dbc.Col(dcc.Dropdown(id='player-radar-compare-dropdown', options=get_all_players_for_dropdown(default_season), placeholder="Select a player to compare...", style={'color': 'black'}), md=6), dbc.Col(dcc.Dropdown(id='player-profile-season-dropdown', options=available_seasons, value=default_season, clearable=False, style={'color': 'black'}), md=3), dbc.Col(dbc.RadioItems(id='player-radar-norm-filter', options=[{'label': 'vs League', 'value': 'league'}, {'label': 'vs Top 5', 'value': 'top5'}], value='league', inline=True, className="btn-group", inputClassName="btn-check", labelClassName="btn btn-outline-primary"), md=3, className="d-flex justify-content-end")], className="mb-3 g-2"), dcc.Loading(dcc.Graph(id='player-profile-radar'))])], className="h-100 shadow")], md=7)]), app_signature()
    ], fluid=True, className="p-4")

# --- CALLBACKS ---

@callback(
    Output('player-archetype-badge', 'children'),
    Output('player-usage-stats-row', 'children'),
    [Input('player-radar-norm-filter', 'value')],
    [State('player-profile-name-store', 'data'),
     State('player-profile-season-dropdown', 'value')]
)
def update_archetype_and_usage_stats(normalization_scope, player_name, season):
    if not player_name or not season: return no_update, no_update
    dominant_pos, matches_played, minutes_played = get_dominant_role_and_usage(player_name, season)
    stats_file_path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
    if not os.path.exists(stats_file_path): return dbc.Badge("Data Error", color="danger"), html.Div("Stats file not found.")
    df_all = pd.read_parquet(stats_file_path)
    player_stats_row = df_all[df_all['Player'] == player_name]
    if player_stats_row.empty: return dbc.Badge("No Stats", color="warning"), html.Div("Player stats not found.")
    player_stats = player_stats_row.iloc[0]

    # --- FIX: Aggiunto .copy() per evitare SettingWithCopyWarning ---
    df_filtered = df_all[df_all['League_MP'] > 0].copy()
    df_filtered['min_threshold'] = df_filtered['League_MP'] * MIN_AVG_MINUTES_PER_GAME
    df_filtered = df_filtered[df_filtered['Min'] >= df_filtered['min_threshold']]

    if normalization_scope == 'league':
        df_comp = df_filtered[df_filtered['League'] == player_stats['League']]
    else:
        df_comp = df_filtered
    archetype_text, archetype_icon = get_player_archetype(player_stats, df_comp, dominant_pos)
    archetype_badge = dbc.Badge([html.I(className=f"{archetype_icon} me-2"), archetype_text], color="info", text_color="dark", className="p-2", style={'font-size': '1.1rem'})
    avg_mins = (minutes_played / matches_played) if matches_played > 0 else 0
    usage_stats_row = dbc.Row([dbc.Col(html.Div([html.Div(f"{matches_played:,.0f}", className="fs-4 fw-bold"), html.Small("Matches Played")], className="text-center text-white bg-dark p-2 rounded")), dbc.Col(html.Div([html.Div(f"{minutes_played:,.0f}", className="fs-4 fw-bold"), html.Small("Minutes Played")], className="text-center text-white bg-dark p-2 rounded")), dbc.Col(html.Div([html.Div(f"{avg_mins:,.1f}", className="fs-4 fw-bold"), html.Small("Avg. Mins / Match")], className="text-center text-white bg-dark p-2 rounded"))], className="justify-content-center g-3")
    return archetype_badge, usage_stats_row

@callback(
    Output('player-profile-kpi-ranks-store', 'data'),
    [Input('player-radar-norm-filter', 'value')],
    [State('player-profile-name-store', 'data'),
     State('player-profile-season-dropdown', 'value')]
)
def update_kpi_rankings(normalization_scope, player_name, season):
    stats_file_path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
    if not os.path.exists(stats_file_path) or player_name is None: return {}
    df_all = pd.read_parquet(stats_file_path)
    
    # --- FIX: Aggiunto .copy() per evitare SettingWithCopyWarning ---
    df_all = df_all[df_all['League_MP'] > 0].copy()
    # df_all['min_threshold'] = df_all['League_MP'] * MIN_AVG_MINUTES_PER_GAME
    # df_all = df_all[df_all['Min'] >= df_all['min_threshold']]

    player_stats_row = df_all[df_all['Player'] == player_name]
    if player_stats_row.empty: return {}
    player_stats = player_stats_row.iloc[0]
    if normalization_scope == 'league':
        df_comp = df_all[df_all['League'] == player_stats['League']]
    else: df_comp = df_all
    def get_rank(df, column, p_name):
        if column not in df.columns or df[column].isnull().all(): return None
        is_gk_metric = column in [m['metric_col'] for m in GOALKEEPING_METRICS]
        df_filtered = df[df['Pos'].str.contains('GK', na=False)] if is_gk_metric else df[~df['Pos'].str.contains('GK', na=False)]
        if df_filtered.empty: return None
        df_sorted = df_filtered.sort_values(by=column, ascending=False).reset_index()
        rank_series = df_sorted[df_sorted['Player'] == p_name].index
        return rank_series[0] + 1 if len(rank_series) > 0 else None
    ranks = {metric['metric_col']: get_rank(df_comp, metric['metric_col'], player_name) for metric in ALL_METRICS}
    return ranks

@callback(
    Output('player-profile-metric-cards', 'children'),
    [Input('player-profile-kpi-ranks-store', 'data')],
    [State('player-profile-name-store', 'data'),
     State('player-profile-season-dropdown', 'value')]
)
def render_metric_macro_cards(ranks, player_name, season):
    if not ranks or not player_name or not season: return dbc.Alert("Select a player and season to see stats.", color="info")
    player_data = get_player_data(player_name, season)
    if 'error' in player_data: return dbc.Alert(player_data['error'], color="danger")
    stats = player_data['stats_df']
    dominant_pos, _, _ = get_dominant_role_and_usage(player_name, season)
    is_gk = 'GK' in dominant_pos
    cards_to_render = []
    colors = {"goalkeeping": "rgba(255, 193, 7, 0.15)", "attacking": "rgba(220, 53, 69, 0.15)", "possession": "rgba(13, 110, 253, 0.15)", "defending": "rgba(25, 135, 84, 0.15)"}
    if is_gk: cards_to_render.append(create_metric_category_card("🧤 Goalkeeping", GOALKEEPING_METRICS, stats, ranks, colors['goalkeeping']))
    cards_to_render.append(create_metric_category_card("⚔️ Attacking", ATTACKING_METRICS, stats, ranks, colors['attacking']))
    cards_to_render.append(create_metric_category_card("⚽ Possession", POSSESSION_METRICS, stats, ranks, colors['possession']))
    cards_to_render.append(create_metric_category_card("🛡️ Defending", DEFENSIVE_METRICS, stats, ranks, colors['defending']))
    return html.Div(cards_to_render)

@callback(
    Output('player-profile-radar', 'figure'),
    [Input('player-radar-compare-dropdown', 'value'),
     Input('player-radar-norm-filter', 'value')],
    [State('player-profile-name-store', 'data'),
     State('player-profile-season-dropdown', 'value')]
)
def update_player_radar(comparison_player_name, normalization_scope, primary_player_name, season):
    if not primary_player_name or not season: return go.Figure()
    stats_file_path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
    if not os.path.exists(stats_file_path): return go.Figure().update_layout(title_text="Stats file not found", template="plotly_dark")
    df_all_players = pd.read_parquet(stats_file_path)
    primary_player_series_rows = df_all_players[df_all_players['Player'] == primary_player_name]
    if primary_player_series_rows.empty: return go.Figure().update_layout(title_text=f"{primary_player_name} not found in stats.", template="plotly_dark")
    primary_player_series = primary_player_series_rows.iloc[0]

    # --- FIX: Aggiunto .copy() per evitare SettingWithCopyWarning ---
    df_all_players_filtered = df_all_players[df_all_players['League_MP'] > 0].copy()
    df_all_players_filtered['min_threshold'] = df_all_players_filtered['League_MP'] * MIN_AVG_MINUTES_PER_GAME
    df_all_players_filtered = df_all_players_filtered[df_all_players_filtered['Min'] >= df_all_players_filtered['min_threshold']]

    if normalization_scope == 'league':
        df_for_normalization = df_all_players_filtered[df_all_players_filtered['League'] == primary_player_series['League']]
    else: df_for_normalization = df_all_players_filtered
    comparison_player_series = None
    if comparison_player_name:
        comparison_series_rows = df_all_players[df_all_players['Player'] == comparison_player_name]
        if not comparison_series_rows.empty: comparison_player_series = comparison_series_rows.iloc[0]
    dominant_pos, _, _ = get_dominant_role_and_usage(primary_player_name, season)
    primary_player_series_copy = primary_player_series.copy()
    primary_player_series_copy['Pos'] = dominant_pos
    fig = player_plots.create_player_profile_radar(df_for_normalization=df_for_normalization, primary_player_series=primary_player_series_copy, comparison_player_series=comparison_player_series)
    title_suffix = f"vs {primary_player_series['League']}" if normalization_scope == 'league' else "vs Top 5 Leagues"
    fig.update_layout(title_text=f"Player Skill Radar ({season}) - Percentile Ranks {title_suffix}")
    return fig