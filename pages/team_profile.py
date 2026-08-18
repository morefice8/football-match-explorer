# --- START OF FILE pages/team_profile.py ---
import os
import json
import pandas as pd
import numpy as np
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update
import dash_bootstrap_components as dbc
from src.visualization import team_plots
import plotly.graph_objects as go
from src.components.layout_components import app_signature
from src.metrics.sportmonks import TEAM_METRIC_GROUPS

# --- CONFIGURATION & HELPER FUNCTIONS ---
# ... (nessuna modifica qui)
DATA_PATH = os.path.join("data", "fbref")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
SPORTMONKS_DATA_PATH = os.path.join("data", "sportmonks", "processed")
LEAGUES = {
    "bundesliga": {"name": "Bundesliga"}, "la-liga": {"name": "La Liga"},
    "ligue-1": {"name": "Ligue 1"}, "premier-league": {"name": "Premier League"},
    "serie-a": {"name": "Serie A"}
}
LEAGUE_NAME_TO_FOLDER = {v['name']: k for k, v in LEAGUES.items()}

def get_available_seasons_for_team(team_folder_name, league_folder):
    seasons = set()
    if os.path.isdir(PROCESSED_DATA_PATH):
        seasons.update(
            filename[len("team_stats_"):-len(".parquet")]
            for filename in os.listdir(PROCESSED_DATA_PATH)
            if filename.startswith("team_stats_") and filename.endswith(".parquet")
        )
    base_path = os.path.join(DATA_PATH, league_folder, f"{league_folder}_stats")
    if os.path.isdir(base_path):
        seasons.update(s for s in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, s)) and '-' in s)
    return sorted(seasons, reverse=True) if seasons else ["2024-2025"]

def get_all_teams_for_dropdown():
    options = []
    if os.path.isdir(PROCESSED_DATA_PATH):
        datasets = sorted(
            (filename for filename in os.listdir(PROCESSED_DATA_PATH)
             if filename.startswith("team_stats_") and filename.endswith(".parquet")),
            reverse=True,
        )
        if datasets:
            try:
                teams = pd.read_parquet(os.path.join(PROCESSED_DATA_PATH, datasets[0]))
                options.extend(
                    {'label': f"{row['Squad']} ({row['League']})", 'value': row['Squad']}
                    for _, row in teams[['Squad', 'League']].drop_duplicates().iterrows()
                )
            except Exception:
                pass
    for league_folder, league_info in LEAGUES.items():
        teams_dir_path = os.path.join(DATA_PATH, league_folder, f"{league_folder}_teams")
        if os.path.isdir(teams_dir_path):
            for team_folder in os.listdir(teams_dir_path):
                options.append({'label': f"{team_folder.replace('_', ' ')} ({league_info['name']})", 'value': team_folder.replace('_', ' ')})
    unique = {option['value']: option for option in options}
    return sorted(unique.values(), key=lambda x: x['label'])


def _get_sportmonks_team_data(team_name, season):
    team_stats_path = os.path.join(PROCESSED_DATA_PATH, f"team_stats_{season}.parquet")
    player_stats_path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{season}.parquet")
    matchlogs_path = os.path.join(SPORTMONKS_DATA_PATH, season, "team_matchlogs.parquet")
    if not all(os.path.exists(path) for path in (team_stats_path, player_stats_path, matchlogs_path)):
        return None

    df_all_teams = pd.read_parquet(team_stats_path)
    team_profile = df_all_teams[df_all_teams['Squad'] == team_name]
    if team_profile.empty:
        return None
    league_name = team_profile.iloc[0]['League']
    league_folder = LEAGUE_NAME_TO_FOLDER.get(league_name)
    if not league_folder:
        return None

    roster = pd.read_parquet(player_stats_path)
    roster = roster[roster['Club'] == team_name].copy()
    aliases = {
        'Interceptions': 'Int', 'Att': 'Passes', 'Cmp': 'Accurate_Passes',
        'AerialsWon': 'Aerial_Won', 'AerialsLost': 'Aerial_Lost',
        'TakeOns': 'Dribble_Attempts', 'TakeOnSucc': 'Successful_Dribbles',
    }
    for target, source in aliases.items():
        roster[target] = pd.to_numeric(roster[source], errors='coerce').fillna(0) if source in roster else 0
    numeric_cols = ['Gls', 'Ast', 'MP', 'Min', '90s', 'TklW', 'Interceptions', 'Att', 'Cmp', 'Sh', 'Chances_Created', 'AerialsWon', 'AerialsLost', 'TakeOns', 'TakeOnSucc', 'Age']
    for column in numeric_cols:
        roster[column] = pd.to_numeric(roster[column], errors='coerce').fillna(0) if column in roster else 0
    roster['Pass_Completion_Perc'] = (roster['Cmp'] / roster['Att'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    roster['Tkl_Int_per_90'] = ((roster['TklW'] + roster['Interceptions']) / roster['90s']).replace([np.inf, -np.inf], 0).fillna(0)
    roster['Shot_Conversion_Perc'] = (roster['Gls'] / roster['Sh'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    roster['Chances_Created_per_90'] = (roster['Chances_Created'] / roster['90s']).replace([np.inf, -np.inf], 0).fillna(0)
    roster['Dribble_Success_Perc'] = (roster['TakeOnSucc'] / roster['TakeOns'] * 100).replace([np.inf, -np.inf], 0).fillna(0)
    roster['Min_per_Match'] = (roster['Min'] / roster['MP']).replace([np.inf, -np.inf], 0).fillna(0)
    roster['dominant_position'] = roster['Pos'].fillna('N/A')

    matchlogs = pd.read_parquet(matchlogs_path)
    matchlogs = matchlogs[matchlogs['team_name'] == team_name].copy()
    matchlogs.rename(columns={
        'starting_at': 'Date', 'result': 'Result', 'opponent': 'Opponent',
        'goals_for': 'GF', 'goals_against': 'GA', 'formation': 'Formation',
    }, inplace=True)

    logo_path = ""
    teams_path = os.path.join(SPORTMONKS_DATA_PATH, season, "teams.parquet")
    if os.path.exists(teams_path):
        teams = pd.read_parquet(teams_path)
        team_row = teams[teams['team_name'] == team_name]
        if not team_row.empty:
            logo_path = team_row.iloc[0].get('image_path') or ""
    return {
        'profile_df': team_profile,
        'roster_df': roster,
        'matchlogs_df': matchlogs,
        'league_name': league_name,
        'league_folder': league_folder,
        'folder_name': str(team_name).replace(' ', '_'),
        'formation_analysis': [],
        'logo_path': logo_path,
        'data_source': 'Sportmonks',
    }

def get_team_data(team_name, season):
    from pages.team_stats import load_all_team_stats
    from src.utils import formation_layouts # Import per la mappa

    try:
        sportmonks_data = _get_sportmonks_team_data(team_name, season)
        if sportmonks_data is not None:
            return sportmonks_data
    except Exception as exc:
        print(f"Error loading Sportmonks team profile for {team_name}: {exc}")
    
    team_folder_name = str(team_name).replace(' ', '_')
    team_league_folder, team_league_name = None, None
    for lf, li in LEAGUES.items():
        if os.path.isdir(os.path.join(DATA_PATH, lf, f"{lf}_teams", team_folder_name)):
            team_league_folder, team_league_name = lf, li['name']
            break
    if not team_league_folder: return {'error': 'Team not found'}

    df_all_teams = load_all_team_stats(season)
    if df_all_teams.empty: return {'error': f'Could not load stats for season {season}.'}
    team_profile = df_all_teams[df_all_teams['Squad'] == team_name]
    if team_profile.empty: return {'error': f'Stats for {team_name} in {season} not found.'}

    team_specific_path = os.path.join(DATA_PATH, team_league_folder, f"{team_league_folder}_teams", team_folder_name, "team_stats")
    try:
        df_roster_std = pd.read_csv(os.path.join(team_specific_path, "standard.csv"))
        df_roster_def = pd.read_csv(os.path.join(team_specific_path, "defensive.csv"))
        df_roster_pass = pd.read_csv(os.path.join(team_specific_path, "passing.csv"))
        df_roster_shoot = pd.read_csv(os.path.join(team_specific_path, "shooting.csv"))
        df_roster_gca = pd.read_csv(os.path.join(team_specific_path, "gca.csv"))
        df_roster_misc = pd.read_csv(os.path.join(team_specific_path, "misc.csv"))
        df_roster_poss = pd.read_csv(os.path.join(team_specific_path, "possession.csv"))
        
        df_roster = pd.merge(df_roster_std, df_roster_def[['Player', 'TklW', 'Interceptions']], on="Player", how="left")
        df_roster = pd.merge(df_roster, df_roster_pass[['Player', 'Att', 'Cmp', 'PrgP']], on="Player", how="left")
        df_roster = pd.merge(df_roster, df_roster_shoot[['Player', 'Sh']], on="Player", how="left")
        df_roster = pd.merge(df_roster, df_roster_gca[['Player', 'GCA']], on="Player", how="left")
        df_roster = pd.merge(df_roster, df_roster_misc[['Player', 'AerialsWon', 'AerialsLost']], on="Player", how="left")
        df_roster = pd.merge(df_roster, df_roster_poss[['Player', 'TakeOns', 'TakeOnSucc']], on="Player", how="left")
        df_roster = df_roster[~df_roster['Player'].isin(['Squad Total', 'Opponent Total'])]

        numeric_cols = ['Gls', 'Ast', 'MP', 'Min', '90s', 'TklW', 'Interceptions', 'PrgP', 'Att', 'Cmp', 'Sh', 'GCA', 'AerialsWon', 'AerialsLost', 'TakeOns', 'TakeOnSucc', 'Age']
        for col in numeric_cols:
            if col in df_roster.columns:
                df_roster[col] = pd.to_numeric(df_roster[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)
        df_roster['Pass_Completion_Perc'] = (df_roster['Cmp'] / df_roster['Att'] * 100).fillna(0)
        df_roster['Tkl_Int_per_90'] = ((df_roster['TklW'] + df_roster['Interceptions']) / df_roster['90s']).fillna(0)
        df_roster['Shot_Conversion_Perc'] = (df_roster['Gls'] / df_roster['Sh'] * 100).fillna(0)
        df_roster['GCA_per_90'] = (df_roster['GCA'] / df_roster['90s']).fillna(0)
        df_roster['TakeOn_Success_Rate'] = (df_roster['TakeOnSucc'] / df_roster['TakeOns'] * 100).fillna(0)

        df_roster['Min_per_Match'] = 0.0
        mask = df_roster['MP'] > 0
        df_roster.loc[mask, 'Min_per_Match'] = (df_roster.loc[mask, 'Min'] / df_roster.loc[mask, 'MP'])

        df_roster.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

        # --- LOGICA PER LA POSIZIONE DOMINANTE ---
        player_data_path = os.path.join(DATA_PATH, "player_top5_europe")
        dominant_positions = {}
        for _, player_row in df_roster.iterrows():
            player_folder_name = str(player_row['Player']).replace(' ', '_')
            matchlog_path = os.path.join(player_data_path, player_folder_name, season, "player_matchlogs.csv")
            pos = player_row.get('Pos', 'N/A') # Fallback
            try:
                if os.path.exists(matchlog_path):
                    df_player_logs = pd.read_csv(matchlog_path)
                    if 'Pos' in df_player_logs.columns and not df_player_logs['Pos'].dropna().empty:
                        # Calcola la moda (posizione più frequente)
                        pos = df_player_logs['Pos'].mode()[0]
            except Exception as e:
                print(f"Could not process matchlog for {player_row['Player']}: {e}")
            dominant_positions[player_row['Player']] = pos
        df_roster['dominant_position'] = df_roster['Player'].map(dominant_positions)
        # --- FINE LOGICA POSIZIONE DOMINANTE ---
        
        df_matchlogs = pd.read_csv(os.path.join(team_specific_path, "matchlogs.csv"))
        
        # --- LOGICA DI ANALISI FORMAZIONE (REINSERITA QUI) ---
        formation_name_to_id = {v: k for k, v in formation_layouts.OPTA_FORMATION_ID_TO_NAME.items()}
        formation_counts = df_matchlogs.dropna(subset=['Formation'])['Formation'].value_counts()
        top_formations = []
        for formation_name, count in formation_counts.head(3).items():
            formation_id = formation_name_to_id.get(formation_name)
            if formation_id:
                top_formations.append({'id': formation_id, 'name': formation_name, 'count': count})
        # --------------------------------------------------------

    except Exception as e:
        return {'error': f'Error loading team-specific files for {team_name}: {e}'}
    
    return {
        'profile_df': team_profile, 'roster_df': df_roster, 'matchlogs_df': df_matchlogs,
        'league_name': team_league_name, 'league_folder': team_league_folder, 
        'folder_name': team_folder_name, 'formation_analysis': top_formations,
        'logo_path': get_team_logo_path(team_league_folder, team_folder_name),
        'data_source': 'FBref',
    }

def get_team_logo_path(league_folder, team_folder_name):
    if not league_folder: return ""
    teams_folder_name = f"{league_folder}_teams"
    return f"/data/fbref/{league_folder}/{teams_folder_name}/{team_folder_name}/logo.png"

def get_player_photo_path(player_name):
    player_folder = str(player_name).replace(' ', '_')
    local_path = os.path.join("data", "fbref", "player_top5_europe", player_folder, f"{player_folder}.png")

    if os.path.exists(local_path):
        return f"/{local_path.replace(os.sep, '/')}"  # Percorso URL valido
    else:
        # Fallback immagine default (es. avatar generico)
        return "/assets/avatar_placeholder.png" 

def create_summary_stat_card(title, value, icon, format_spec="{:,.1f}"):
    display_value = format_spec.format(value) if isinstance(value, (int, float)) else value
    return html.Div([
        html.P([html.I(className=f"{icon} me-2"), title], className="text-muted small mb-0"),
        html.H5(display_value, className="fw-bold")
    ], className="text-center mb-3")


def create_sportmonks_summary_column(title, specs, profile, *, border=True):
    metrics = []
    for spec in specs:
        raw_value = pd.to_numeric(pd.Series([profile.get(spec.column)]), errors='coerce').iloc[0]
        if pd.isna(raw_value):
            display_value = "N/A"
        else:
            display_value = spec.format_spec.format(raw_value)
            if spec.unit:
                display_value = f"{display_value}{spec.unit}"
        metrics.append(create_summary_stat_card(spec.title, display_value, spec.icon))
    class_name = "border-end" if border else ""
    return dbc.Col([
        html.H5(title, className="text-center mb-3 border-bottom pb-2"),
        *metrics,
    ], md=4, className=class_name)

def create_form_guide(matchlogs_df):
    df = matchlogs_df.dropna(subset=['Result']).tail(5)
    if df.empty: return html.P("No recent match data.", className="text-muted")
    result_map = {'W': 'success', 'D': 'warning', 'L': 'danger'}
    form_badges = [dbc.Badge(row['Result'][0], color=result_map.get(row['Result'][0], 'secondary'), className="fs-5 me-1") for _, row in df.iterrows()]
    return html.Div([html.H6("Recent Form (Last 5):", className="me-2 d-inline-block"), html.Div(form_badges, className="d-inline-block")])

# --- MODIFICA QUI: Aggiunto Age, Pos al posto del numero di maglia ---
def create_player_ranking_row(player_series, metric_col, unit="", is_first=False, format_spec="{:,.1f}"):
    player_name = player_series['Player']
    metric_value = player_series[metric_col]
    position = player_series.get('Pos', '')
    age = player_series.get('Age', '')
    
    photo_style = {'height': '60px', 'width': '60px', 'object-fit': 'cover', 'border-radius': '50%'} if is_first else {'height': '40px', 'width': '40px', 'object-fit': 'cover', 'border-radius': '50%'}
    name_class = "fs-5 fw-bold" if is_first else "fs-6"
    metric_class = "display-6 fw-bold" if is_first else "fs-5"
    
    player_info = f"{position} • {int(age)} years" if age else position

    return dbc.Row([
        dbc.Col(html.Img(src=player_series.get('image_path') or get_player_photo_path(player_name), style=photo_style), width="auto"),
        dbc.Col([
            dcc.Link(player_name, href=f"/player-stats/{str(player_name).replace(' ', '_')}", className=f"text-white text-decoration-none {name_class}"),
            html.Span(player_info, className="text-muted small d-block")
        ], width=True, className="align-self-center"),
        dbc.Col(html.Span(f"{format_spec.format(metric_value)} {unit}", className=metric_class), width="auto", className="align-self-center text-end")
    ], align="center", className="mb-2")

def create_top_performer_card(df_roster, title, metric_col, unit="", ascending=False, format_spec="{:,.1f}", icon=""):
    if df_roster.empty or metric_col not in df_roster.columns or df_roster[metric_col].isnull().all():
        return dbc.Col(dbc.Alert(f"Data for {title} not available.", color="secondary"))
    top_3 = df_roster.sort_values(by=metric_col, ascending=ascending).head(3)
    if len(top_3) < 1:
        return dbc.Col(dbc.Alert(f"Not enough data for {title}.", color="info"))
    return dbc.Col(
        dbc.Card([
            dbc.CardHeader(html.H5([html.I(className=f"{icon} me-2"), title], className="text-center m-0")),
            dbc.CardBody([
                create_player_ranking_row(top_3.iloc[0], metric_col, unit, is_first=True, format_spec=format_spec),
                html.Hr(className="my-2"),
                *[create_player_ranking_row(top_3.iloc[i], metric_col, unit, format_spec=format_spec) for i in range(1, len(top_3))]
            ])
        ]), lg=3, md=6, className="mb-4"
    )

def create_player_card(player_series):
    """Creates a small card for a single player in the roster, with side-by-side stats."""
    player_name = player_series['Player']
    
    # --- NUOVO BLOCCO STATISTICHE ---
    stats_block = dbc.Row([
        # Colonna Sinistra: Minuti Medi
        dbc.Col([
            html.P("Avg. Mins / Match", className="small text-muted mb-0"),
            html.H4(f"{player_series.get('Min_per_Match', 0):.1f}", className="fw-bold"),
            html.P(f"({int(player_series.get('MP', 0))} Matches)", className="small text-muted")
        ], className="text-center"),
        
        # Colonna Destra: Posizione Dominante
        dbc.Col([
            html.P("Main Position", className="small text-muted mb-0"),
            html.H4(player_series.get('dominant_position', 'N/A'), className="fw-bold"),
            # Placeholder per allineamento verticale
            html.P(html.Br(), className="small") 
        ], className="text-center")
    ])

    photo_style = {'height': '100px', 'width': '100px', 'object-fit': 'cover', 'border-radius': '50%'}

    return dbc.Col(
        dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Img(src=player_series.get('image_path') or get_player_photo_path(player_name), style=photo_style, className="img-fluid rounded-circle"), width="auto"),
                    dbc.Col([
                        dcc.Link(html.H5(player_name, className="card-title"), href=f"/player-stats/{str(player_name).replace(' ', '_')}"),
                        html.P(f"{player_series.get('Pos', 'N/A')} • {int(player_series.get('Age', 0))} years", className="card-text small text-muted"),
                        html.Hr(className="my-2"),
                        stats_block # Inserisce il nuovo blocco di statistiche
                    ], width=8, className="ps-3")
                ], align="center")
            ])
        ], className="h-100 shadow-sm"),
        lg=3, md=4, sm=6, className="mb-4"
    )


def layout(team_name_url, season="2024-2025"):
    team_name = team_name_url.replace('_', ' ')
    team_data = get_team_data(team_name, season)
    if 'error' in team_data: return dbc.Container(dbc.Alert(team_data['error'], color="danger"), className="mt-4")
    
    profile, roster, matchlogs, formation_analysis = (
        team_data['profile_df'].iloc[0], team_data['roster_df'], 
        team_data['matchlogs_df'], team_data['formation_analysis']
    )
    league_name, league_folder, folder_name = team_data['league_name'], team_data['league_folder'], team_data['folder_name']
    available_seasons = get_available_seasons_for_team(folder_name, league_folder)

    header = dbc.Row([
        dbc.Col(dbc.Button([html.I(className="fas fa-arrow-left me-2"), "Back to Teams Overview"], href="/team-stats", color="secondary", outline=True), width="auto"),
        dbc.Col(html.Img(src=team_data.get('logo_path') or get_team_logo_path(league_folder, folder_name), style={'height': '80px'}), width="auto"),
        dbc.Col([html.H1(team_name, className="text-white mb-0"), html.H4(league_name, className="text-muted")], className="align-self-center"),
        dbc.Col(dcc.Dropdown(id='team-profile-season-dropdown', options=[{'label': s, 'value': s} for s in available_seasons], value=season, clearable=False, style={'width': '200px', 'color': 'black'}), width="auto", className="align-self-center")
    ], align="center", className="my-4")

    if team_data.get('data_source') == 'Sportmonks':
        summary_columns = [
            create_sportmonks_summary_column("Attacking", TEAM_METRIC_GROUPS['attacking'], profile),
            create_sportmonks_summary_column("Possession & Territory", TEAM_METRIC_GROUPS['possession'], profile),
            create_sportmonks_summary_column("Defending", TEAM_METRIC_GROUPS['defending'], profile, border=False),
        ]
    else:
        summary_columns = [
            dbc.Col([
                html.H5("Attacking", className="text-center mb-3 border-bottom pb-2"),
                create_summary_stat_card("Goals", profile.get('Gls', 0), "fa-solid fa-futbol", format_spec="{:,.0f}"),
                create_summary_stat_card("xG per Shot", profile.get('xG_per_Shot', 0), "fa-solid fa-bullseye", format_spec="{:.3f}"),
                create_summary_stat_card("Goal Conversion", f"{profile.get('Goal_Conversion', 0):.1f}%", "fa-solid fa-percent"),
                create_summary_stat_card("Ast vs xAG Ratio", profile.get('Ast_xAG_ratio', 0), "fa-solid fa-wand-magic-sparkles", format_spec="{:,.2f}"),
            ], md=4, className="border-end"),
            dbc.Col([
                html.H5("Possession & Style", className="text-center mb-3 border-bottom pb-2"),
                create_summary_stat_card("Passing Tempo", profile.get('Passing_Tempo', 0), "fa-solid fa-gauge-high", format_spec="{:,.2f}"),
                create_summary_stat_card("Progressions / Touch", profile.get('Progressions_per_Touch', 0), "fa-solid fa-angles-up", format_spec="{:,.4f}"),
                create_summary_stat_card("Cross / Touch Ratio", profile.get('Cross_Touch_Ratio', 0), "fa-solid fa-arrows-left-right", format_spec="{:,.4f}"),
                create_summary_stat_card("Take-On Success", f"{profile.get('TakeOn_Success_Rate', 0):.1%}", "fa-solid fa-person-running"),
            ], md=4, className="border-end"),
            dbc.Col([
                html.H5("Defending", className="text-center mb-3 border-bottom pb-2"),
                create_summary_stat_card("Goals Against", profile.get('GA', 0), "fa-solid fa-shield-halved", format_spec="{:,.0f}"),
                create_summary_stat_card("Shots Conceded / DA", profile.get('Shots_Conceded_per_DA', 0), "fa-solid fa-calculator", format_spec="{:,.2f}"),
                create_summary_stat_card("Tkl+Int / 90", profile.get('Tkl_Int_per_90', 0), "fa-solid fa-person-falling-burst", format_spec="{:,.2f}"),
                create_summary_stat_card("Errors / GA", profile.get('Errors_per_GA', 0), "fa-solid fa-bug", format_spec="{:,.3f}"),
            ], md=4),
        ]

    summary_card = dbc.Card([
        dbc.CardHeader(create_form_guide(matchlogs)),
        dbc.CardBody([
            dbc.Row(summary_columns)
        ])
    ], className="mb-4")

    formation_content = dbc.Alert("No formation data available.", color="warning")
    if formation_analysis:
        total_matches_with_formation = len(matchlogs.dropna(subset=['Formation']))
        
        # --- Sezione Principale (formazione più usata) ---
        most_used = formation_analysis[0]
        most_used_perc = (most_used['count'] / total_matches_with_formation * 100) if total_matches_with_formation > 0 else 0
        formation_plot_large = team_plots.plot_most_used_formation_plotly(most_used['id'], height=350) # Grafico grande
        
        main_formation_section = html.Div([
            html.H1(f"{most_used['name']}", className="text-center text-white display-4 fw-bold"),
            html.P(f"Used in {most_used_perc:.1f}% of matches", className="text-center text-muted small mb-3"),
            dcc.Graph(figure=formation_plot_large, config={'displayModeBar': False}),
        ])

        # --- Sezione Secondaria (altre formazioni) ---
        secondary_formations_section = html.Div() # Vuoto se non ci sono altre formazioni
        if len(formation_analysis) > 1:
            # Colonna per la seconda formazione
            second_used = formation_analysis[1]
            second_used_perc = (second_used['count'] / total_matches_with_formation * 100) if total_matches_with_formation > 0 else 0
            plot_second = team_plots.plot_most_used_formation_plotly(second_used['id'], height=200) # Grafico piccolo
            
            second_col = dbc.Col([
                html.H6(f"2nd: {second_used['name']}", className="text-center"),
                html.P(f"{second_used_perc:.1f}% of matches", className="text-center text-muted small"),
                dcc.Graph(figure=plot_second, config={'displayModeBar': False})
            ], md=6)

            # Colonna per la terza formazione (se esiste)
            third_col = dbc.Col(md=6)
            if len(formation_analysis) > 2:
                third_used = formation_analysis[2]
                third_used_perc = (third_used['count'] / total_matches_with_formation * 100) if total_matches_with_formation > 0 else 0
                plot_third = team_plots.plot_most_used_formation_plotly(third_used['id'], height=200) # Grafico piccolo
                third_col = dbc.Col([
                    html.H6(f"3rd: {third_used['name']}", className="text-center"),
                    html.P(f"{third_used_perc:.1f}% of matches", className="text-center text-muted small"),
                    dcc.Graph(figure=plot_third, config={'displayModeBar': False})
                ], md=6)

            secondary_formations_section = html.Div([
                html.Hr(),
                dbc.Row([second_col, third_col])
            ])

        formation_content = html.Div([main_formation_section, secondary_formations_section])
    
    formation_card = dbc.Card([
        dbc.CardHeader("Most Used Formation"),
        dbc.CardBody(formation_content)
    ], className="h-100")

    radar_section = dbc.Card([
        dbc.CardHeader("Team Profile Radar"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='team-radar-compare-dropdown', options=get_all_teams_for_dropdown(), placeholder="Select a team to compare...", style={'color': 'black'}), md=8),
                dbc.Col(dbc.RadioItems(id='radar-normalization-filter', options=[{'label': 'vs League', 'value': 'league'}, {'label': 'vs Top 5', 'value': 'top5'}], value='league', inline=True, className="btn-group", inputClassName="btn-check", labelClassName="btn btn-outline-primary"), md=4, className="d-flex justify-content-end")
            ], className="mb-3"),
            dcc.Loading(dcc.Graph(id='team-profile-radar'))
        ])
    ], className="mb-4")

    # --- NUOVA SEZIONE ROSTER CON ACCORDION ---
    roster.dropna(subset=['dominant_position'], inplace=True)
    roster_sorted = roster.sort_values(by="Min", ascending=False)

    def assign_role_from_specific_pos(pos_str):
        """
        Maps a specific position string (e.g., 'LB', 'RW,FW', 'DM') to a general role.
        Handles multiple positions by prioritizing the most offensive one.
        """
        if not isinstance(pos_str, str): return 'Other'
        
        # Gerarchia: Attaccanti > Centrocampisti > Difensori > Portieri
        if 'FW' in pos_str or 'W' in pos_str: # RW, LW, WF
            return 'Forwards ⚔️'
        if 'MF' in pos_str or 'M' in pos_str: # DM, CM, AM, LM, RM
            return 'Midfielders 🧠'
        if 'DF' in pos_str or 'B' in pos_str: # CB, LB, RB, WB
            return 'Defenders 🛡️'
        if 'GK' in pos_str:
            return 'Goalkeepers 🧤'
        return 'Other' # Fallback
    
    roster_sorted['Role'] = roster_sorted['dominant_position'].apply(assign_role_from_specific_pos)

    # Raggruppa i giocatori per ruolo
    roles = {
        'Goalkeepers 🧤': roster_sorted[roster_sorted['Role'] == 'Goalkeepers 🧤'],
        'Defenders 🛡️': roster_sorted[roster_sorted['Role'] == 'Defenders 🛡️'],
        'Midfielders 🧠': roster_sorted[roster_sorted['Role'] == 'Midfielders 🧠'],
        'Forwards ⚔️': roster_sorted[roster_sorted['Role'] == 'Forwards ⚔️']
    }

    accordion_items = []
    for role, df_role in roles.items():
        if not df_role.empty:
            player_cards = dbc.Row([create_player_card(player) for _, player in df_role.iterrows()])
            accordion_items.append(
                dbc.AccordionItem(
                    player_cards,
                    title=f"{role} ({len(df_role)})",
                )
            )

    roster_section = dbc.Accordion(accordion_items, start_collapsed=False, always_open=True, className="mt-4")
    
    if 'TklW' in roster.columns and 'Interceptions' in roster.columns: roster['TklW_Int'] = roster['TklW'] + roster['Interceptions']
    top_performers_row1 = dbc.Row([
        create_top_performer_card(roster, "Top Scorers", "Gls", "Gls", format_spec="{:,.0f}", icon="fa-solid fa-futbol"),
        create_top_performer_card(roster, "Top Playmakers", "Ast", "Ast", format_spec="{:,.0f}", icon="fa-solid fa-hands-helping"),
        create_top_performer_card(roster, "Best Pass Completion", "Pass_Completion_Perc", "%", format_spec="{:,.1f}", icon="fa-solid fa-chart-line"),
        create_top_performer_card(roster, "Top Defenders (Tkl+Int)/90", "Tkl_Int_per_90", "", format_spec="{:,.2f}", icon="fa-solid fa-shield-halved"),
    ])
    if team_data.get('data_source') == 'Sportmonks':
        top_performers_row2 = dbc.Row([
            create_top_performer_card(roster, "Best Shot Conversion", "Shot_Conversion_Perc", "%", format_spec="{:,.1f}", icon="fa-solid fa-bullseye"),
            create_top_performer_card(roster, "Top Chance Creators p90", "Chances_Created_per_90", "", format_spec="{:,.2f}", icon="fa-solid fa-wand-magic-sparkles"),
            create_top_performer_card(roster, "Best Dribblers", "Dribble_Success_Perc", "%", format_spec="{:,.1f}", icon="fa-solid fa-person-running"),
            create_top_performer_card(roster, "Most Used Players", "Min", "mins", format_spec="{:,.0f}", icon="fa-solid fa-clock"),
        ])
    else:
        top_performers_row2 = dbc.Row([
            create_top_performer_card(roster, "Best Shot Conversion", "Shot_Conversion_Perc", "%", format_spec="{:,.1f}", icon="fa-solid fa-bullseye"),
            create_top_performer_card(roster, "Top Goal Creators (GCA/90)", "GCA_per_90", "", format_spec="{:,.2f}", icon="fa-solid fa-wand-magic-sparkles"),
            create_top_performer_card(roster, "Best Dribblers (Take-On %)", "TakeOn_Success_Rate", "%", format_spec="{:,.1f}", icon="fa-solid fa-person-running"),
            create_top_performer_card(roster, "Most Used Players", "Min", "mins", format_spec="{:,.0f}", icon="fa-solid fa-clock"),
        ])

    return dbc.Container([
        dcc.Store(id='team-profile-df-store', data=profile.to_json()),
        dcc.Store(id='team-profile-league-name-store', data=league_name),
        dcc.Store(id='team-profile-name-store', data=team_name),
        header,
        dbc.Row([dbc.Col(summary_card, md=12)]),
        dbc.Row([dbc.Col(formation_card, md=4), dbc.Col(radar_section, md=8)], className="mb-4", align="stretch"),
        html.H2("Top Performers", className="text-white text-center my-4"),
        top_performers_row1,
        top_performers_row2,
        html.H2("Team Roster", className="text-white text-center my-4"),
        roster_section,
        app_signature(),
    ], fluid=True, className="p-4")

# --- CALLBACKS (invariati) ---
@callback(
    Output('url', 'href'), # Usiamo 'href' per un reindirizzamento completo
    Input('team-profile-season-dropdown', 'value'),
    State('team-profile-name-store', 'data'),
    State('url', 'search'), # Leggiamo i parametri attuali per evitare di aggiungere duplicati
    prevent_initial_call=True # Fondamentale per non attivarsi al caricamento iniziale
)
def change_season_url(selected_season, team_name, current_search):
    """
    Redirects to a new URL ONLY when the user manually changes the season dropdown.
    Prevents the infinite callback loop.
    """
    from urllib.parse import parse_qs, urlencode
    
    # Se per qualche motivo il callback si attiva senza un'azione dell'utente, non fare nulla
    if not selected_season or not team_name:
        return no_update

    # Controlla se la stagione selezionata è già quella nell'URL
    current_params = parse_qs(current_search.lstrip('?'))
    current_season = current_params.get('season', [None])[0]

    if selected_season == current_season:
        return no_update # La stagione non è cambiata, non fare nulla

    # Costruisci il nuovo URL
    team_name_url = str(team_name).replace(' ', '_')
    new_pathname = f"/team-stats/team/{team_name_url}?season={selected_season}"
    return new_pathname

@callback(
    Output('team-profile-radar', 'figure'),
    Input('team-radar-compare-dropdown', 'value'),
    Input('radar-normalization-filter', 'value'),
    State('team-profile-df-store', 'data'),
    State('team-profile-league-name-store', 'data'),
    State('team-profile-season-dropdown', 'value')
)
def update_team_radar(comparison_team_name, normalization_scope, primary_team_json, league_name, season):
    from pages.team_stats import load_all_team_stats
    if not primary_team_json: return go.Figure()
    primary_team_name = pd.read_json(primary_team_json, typ='series')['Squad']
    df_all_teams = load_all_team_stats(season)
    df_for_normalization = df_all_teams[df_all_teams['League'] == league_name] if normalization_scope == 'league' else df_all_teams
    if df_for_normalization.empty: return go.Figure()
    fig = team_plots.create_team_profile_radar(
        df_for_normalization=df_for_normalization,
        primary_team_name=primary_team_name,
        comparison_team_name=comparison_team_name
    )
    title_suffix = f"vs {league_name}" if normalization_scope == 'league' else "vs Top 5 Leagues"
    fig.update_layout(title_text=f"Team Statistical Profile ({season}) - Percentile Ranks {title_suffix}")
    return fig
