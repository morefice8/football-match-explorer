# --- START OF FILE team_stats.py ---
import os
import pandas as pd
from io import StringIO # <-- FIX: AGGIUNTO IMPORT
from dash import html, dcc, Input, Output, State, callback
from src.visualization import league_plots
import dash_bootstrap_components as dbc
from src.components.layout_components import app_signature

# --- CONFIGURATION & HELPER FUNCTIONS ---
DATA_PATH = os.path.join("data", "fbref")
LEAGUES = {
    "bundesliga": {"name": "Bundesliga", "logo": "/data/fbref/bundesliga/bundesliga.png"},
    "la-liga": {"name": "La Liga", "logo": "/data/fbref/la-liga/la-liga.png"},
    "ligue-1": {"name": "Ligue 1", "logo": "/data/fbref/ligue-1/ligue-1.png"},
    "premier-league": {"name": "Premier League", "logo": "/data/fbref/premier-league/premier-league.png"},
    "serie-a": {"name": "Serie A", "logo": "/data/fbref/serie-a/serie-a.png"}
}
LEAGUE_NAME_TO_FOLDER = {v['name']: k for k, v in LEAGUES.items()}
METRIC_TOOLTIPS = {
    "Gls": "Total goals scored by the team.", "xG_per_Shot": "Expected Goals per shot. A measure of average shot quality. Higher is better.",
    "Goal_Conversion": "Percentage of shots that result in a goal. (Goals / Total Shots).", "Ast_xAG_ratio": "Ratio of actual assists to Expected Assisted Goals. A value > 1 suggests high-quality chance creation or finishing.",
    "Passing_Tempo": "Average passes made per touch. Higher values indicate a faster, more direct style of play.", "Progressions_per_Touch": "Combined progressive passes and receptions per touch. Indicates how vertically a team plays.",
    "Cross_Touch_Ratio": "Percentage of touches that are crosses. Indicates reliance on wide play.", "TakeOn_Success_Rate": "Percentage of successful take-ons (dribbles).",
    "GA": "Total goals conceded by the team.", "Shots_Conceded_per_DA": "Shots conceded for every defensive action made. Lower is better, indicating efficiency.",
    "Tkl_Int_per_90": "Combined successful tackles and interceptions, normalized per 90 minutes.", "Errors_per_GA": "Number of errors leading to a shot, normalized by goals conceded. Lower is better."
}

def get_team_logo_path(league_name, team_name):
    league_folder = LEAGUE_NAME_TO_FOLDER.get(league_name)
    if not league_folder: return ""
    teams_folder_name = f"{league_folder}_teams"
    team_folder_name = team_name.replace(' ', '_')
    return f"/data/fbref/{league_folder}/{teams_folder_name}/{team_folder_name}/logo.png"

def get_available_seasons():
    seasons = set()
    for league_folder in LEAGUES.keys():
        stats_path = os.path.join(DATA_PATH, league_folder, f"{league_folder}_stats")
        if os.path.isdir(stats_path):
            for item in os.listdir(stats_path):
                if os.path.isdir(os.path.join(stats_path, item)) and '-' in item:
                    seasons.add(item)
    return sorted(list(seasons), reverse=True)

def load_all_team_stats(season):
    all_league_dfs = []
    for league_folder, league_info in LEAGUES.items():
        stats_dir_path = os.path.join(DATA_PATH, league_folder, f"{league_folder}_stats", season)
        if not os.path.isdir(stats_dir_path): continue
        try:
            df_std = pd.read_csv(os.path.join(stats_dir_path, "standard.csv"))[["Squad", "Poss", "Gls.", "Ast", "xG", "xAG", "90s"]]
            df_pass = pd.read_csv(os.path.join(stats_dir_path, "passing.csv"))[["Squad", "Att", "Cmp", "KP", "PassesFinal3rd", "PrgP", "Att(Short)", "Att(Long)"]]
            df_poss = pd.read_csv(os.path.join(stats_dir_path, "possession.csv"))[["Squad", "Touches", "AttTakeOn", "SuccTakeOn", "PrgPassesReceived"]]
            df_misc = pd.read_csv(os.path.join(stats_dir_path, "miscellaneous.csv"))[["Squad", "TklW", "Int", "Crs", "AerialsWon", "AerialsLost"]]
            df_def = pd.read_csv(os.path.join(stats_dir_path, "defensive.csv"))[["Squad", "Errors"]]
            df_gk = pd.read_csv(os.path.join(stats_dir_path, "goalkeeping.csv"))[["Squad", "GA"]]
            df_shoot = pd.read_csv(os.path.join(stats_dir_path, "shooting.csv"))[["Squad", "Sh"]]
            dfs_to_merge = [df_std, df_pass, df_poss, df_misc, df_def, df_gk, df_shoot]
            df_merged = dfs_to_merge[0]
            for df_to_merge in dfs_to_merge[1:]:
                df_merged = pd.merge(df_merged, df_to_merge, on="Squad", how="left")
            df_merged['League'] = league_info['name']
            all_league_dfs.append(df_merged)
        except Exception as e:
            print(f"Error processing {season} data for {league_info['name']}: {e}")
    if not all_league_dfs: return pd.DataFrame()
    df_full = pd.concat(all_league_dfs, ignore_index=True)
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]
    df_full.rename(columns={'Gls.': 'Gls'}, inplace=True)
    shots_against_data = {}
    for league_folder in LEAGUES.keys():
        teams_dir_path = os.path.join(DATA_PATH, league_folder, f"{league_folder}_teams")
        if not os.path.isdir(teams_dir_path): continue
        for team_folder in os.listdir(teams_dir_path):
            stats_path = os.path.join(teams_dir_path, team_folder, "team_stats", "shooting.csv")
            try:
                df_team_shoot = pd.read_csv(stats_path)
                opponent_total = df_team_shoot[df_team_shoot['Player'] == 'Opponent Total'].iloc[0]
                team_name = team_folder.replace('_', ' ')
                shots_against_data[team_name] = opponent_total.get('Sh', 0)
            except (FileNotFoundError, IndexError, KeyError): continue
    df_full['Shots_Against'] = df_full['Squad'].map(shots_against_data).fillna(0)
    for col in df_full.columns.drop(['Squad', 'League']):
        df_full[col] = pd.to_numeric(df_full[col], errors='coerce')
    df_full['Goal_Conversion'] = (df_full['Gls'] / df_full['Sh'] * 100).fillna(0)
    df_full['Passing_Tempo'] = (df_full['Att'] / df_full['Touches']).fillna(0)
    df_full['Aerial_Duels_Won_Perc'] = (df_full['AerialsWon'] / (df_full['AerialsWon'] + df_full['AerialsLost']) * 100).fillna(0)
    df_full['Defensive_Actions'] = df_full['TklW'] + df_full['Int']
    df_full['Shots_Conceded_per_DA'] = (df_full['Shots_Against'] / df_full['Defensive_Actions']).fillna(0)
    df_full['xG_per_Shot'] = (df_full['xG'] / df_full['Sh']).fillna(0)
    df_full['Ast_xAG_ratio'] = (df_full['Ast'] / df_full['xAG']).fillna(0)
    df_full['KP_per_90'] = (df_full['KP'] / df_full['90s']).fillna(0)
    df_full['FinalThird_per_90'] = (df_full['PassesFinal3rd'] / df_full['90s']).fillna(0)
    df_full['Total_Progressions'] = df_full['PrgP'] + df_full['PrgPassesReceived']
    df_full['Progressions_per_Touch'] = (df_full['Total_Progressions'] / df_full['Touches']).fillna(0)
    df_full['Cross_Touch_Ratio'] = (df_full['Crs'] / df_full['Touches']).fillna(0)
    df_full['TakeOn_Success_Rate'] = (df_full['SuccTakeOn'] / df_full['AttTakeOn']).fillna(0)
    df_full['Tkl_Int_per_90'] = ((df_full['TklW'] + df_full['Int']) / df_full['90s']).fillna(0)
    df_full['Errors_per_GA'] = (df_full['Errors'] / df_full['GA']).fillna(0)
    df_full['Short_vs_Long_Ratio'] = (df_full['Att(Short)'] / df_full['Att(Long)']).fillna(0)
    df_full.replace([pd.NA, float('inf'), -float('inf')], 0, inplace=True)
    return df_full

def create_league_filter_buttons():
    button_style = {'width': '80px', 'height': '60px', 'padding': '0.25rem', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}
    logo_style = {'height': '40px', 'width': 'auto', 'object-fit': 'contain'}
    buttons = [dbc.Button(html.I(className="fas fa-globe fa-2x"), id="filter-all", n_clicks=0, color="primary", outline=True, style=button_style), dbc.Tooltip("Show All Leagues", target="filter-all")]
    for league_folder, league_info in LEAGUES.items():
        button_id = f"filter-{league_folder}"
        buttons.append(dbc.Button(html.Img(src=league_info['logo'], style=logo_style), id=button_id, n_clicks=0, color="secondary", outline=True, style=button_style))
        buttons.append(dbc.Tooltip(league_info['name'], target=button_id))
    return dbc.ButtonGroup(buttons, id="league-filter-buttons")

def create_team_ranking_rows(top_teams_df, metric_column, unit="", format_spec="{:,.1f}"):
    rows = []
    for i, row in top_teams_df.iterrows():
        team_name, league_name, metric_value = row['Squad'], row['League'], row[metric_column]
        is_first = (i == top_teams_df.index[0])
        logo_style = {'height': '60px', 'width': '60px', 'object-fit': 'contain'} if is_first else {'height': '30px', 'width': '30px', 'object-fit': 'contain'}
        team_font_size = "fs-5" if is_first else "fs-6"
        metric_font_size = "display-5" if is_first else "fs-4"
        metric_display = format_spec.format(metric_value)
        logo_path = get_team_logo_path(league_name, team_name)
        row_content = dbc.Row([
            dbc.Col(html.Img(src=logo_path, style=logo_style, title=team_name), width="auto", className="pe-3"),
            dbc.Col([
                dcc.Link(team_name, href=f"/team-stats/team/{team_name.replace(' ', '_')}", className=f"fw-bold text-white text-decoration-none {team_font_size}"),
                dcc.Link(league_name, href=f"/team-stats/league/{league_name.replace(' ', '_')}", className="small text-muted text-decoration-none d-block")
            ], width=True, className="align-self-center"),
            dbc.Col(html.Span(f"{metric_display} {unit}", className=f"fw-bold text-end {metric_font_size}"), width="auto", className="align-self-center")
        ], align="center", className="py-2")
        list_item_class = "border-top-0" if is_first else ""
        rows.append(dbc.ListGroupItem(row_content, color="dark", className=list_item_class))
    return rows

def create_metric_card(df, title, metric_column, ascending, unit="", format_spec="{:,.1f}", icon=""):
    if df.empty or metric_column not in df.columns:
        return dbc.Col(dbc.Alert(f"Data not available for metric: '{title}'", color="warning"))
    top_5_teams = df.sort_values(by=metric_column, ascending=ascending).head(5)
    header_id = f"card-header-{metric_column}"
    card_header = dbc.CardHeader(dbc.Row([dbc.Col(html.I(className=f"{icon} me-2"), width="auto"), dbc.Col(html.H5(title, className="m-0"), width=True)], align="center"), id=header_id, className="text-center")
    tooltip = dbc.Tooltip(METRIC_TOOLTIPS.get(metric_column, "No description available."), target=header_id, placement="top")
    return dbc.Col([
        dbc.Card([card_header, dbc.CardBody(dbc.ListGroup(create_team_ranking_rows(top_5_teams, metric_column, unit, format_spec), flush=True), className="p-0")], className="h-100 shadow"),
        tooltip
    ], lg=3, md=6, sm=12, className="mb-4")

def generate_tabs_content(df, selected_league='All'):
    df_filtered = df[df['League'] == selected_league] if selected_league and selected_league != 'All' else df
    if df_filtered.empty:
        return dbc.Alert(f"No data available for the current selection.", color="info", className="mt-4")
    attacking_metrics = [
        {"title": "Best Attack (Goals)", "metric_column": "Gls", "ascending": False, "unit": "Gls", "format_spec": "{:,.0f}", "icon": "fa-solid fa-futbol"},
        {"title": "xG per Shot", "metric_column": "xG_per_Shot", "ascending": False, "unit": "", "format_spec": "{:.3f}", "icon": "fa-solid fa-bullseye"},
        {"title": "Goal Conversion Rate", "metric_column": "Goal_Conversion", "ascending": False, "unit": "%", "format_spec": "{:,.1f}", "icon": "fa-solid fa-percent"},
        {"title": "Assist vs xAG Ratio", "metric_column": "Ast_xAG_ratio", "ascending": False, "unit": "", "format_spec": "{:,.2f}", "icon": "fa-solid fa-wand-magic-sparkles"}
    ]
    possession_metrics = [
        {"title": "Passing Tempo", "metric_column": "Passing_Tempo", "ascending": False, "unit": "p/touch", "format_spec": "{:,.2f}", "icon": "fa-solid fa-gauge-high"},
        {"title": "Progressions per Touch", "metric_column": "Progressions_per_Touch", "ascending": False, "unit": "", "format_spec": "{:,.4f}", "icon": "fa-solid fa-angles-up"},
        {"title": "Cross to Touch Ratio", "metric_column": "Cross_Touch_Ratio", "ascending": False, "unit": "", "format_spec": "{:,.4f}", "icon": "fa-solid fa-arrows-left-right"},
        {"title": "Take-On Success %", "metric_column": "TakeOn_Success_Rate", "ascending": False, "unit": "", "format_spec": "{:,.1%}", "icon": "fa-solid fa-person-running"}
    ]
    defensive_metrics = [
        {"title": "Best Defense (GA)", "metric_column": "GA", "ascending": True, "unit": "GA", "format_spec": "{:,.0f}", "icon": "fa-solid fa-shield-halved"},
        {"title": "Shots Conceded per DA", "metric_column": "Shots_Conceded_per_DA", "ascending": True, "unit": "Sh/DA", "format_spec": "{:,.2f}", "icon": "fa-solid fa-calculator"},
        {"title": "Tackles+Interceptions per 90", "metric_column": "Tkl_Int_per_90", "ascending": False, "unit": "", "format_spec": "{:,.2f}", "icon": "fa-solid fa-person-falling-burst"},
        {"title": "Errors per Goal Against", "metric_column": "Errors_per_GA", "ascending": True, "unit": "", "format_spec": "{:,.3f}", "icon": "fa-solid fa-bug"}
    ]
    return dbc.Tabs(
        id="team-stats-tabs", active_tab="tab-attacking",
        children=[
            dbc.Tab(label="⚔️ Attacking", tab_id="tab-attacking", children=html.Div(dbc.Row([create_metric_card(df_filtered, **m) for m in attacking_metrics]), className="p-2 mt-3")),
            dbc.Tab(label="⚽ Possession", tab_id="tab-possession", children=html.Div(dbc.Row([create_metric_card(df_filtered, **m) for m in possession_metrics]), className="p-2 mt-3")),
            dbc.Tab(label="🛡️ Defending", tab_id="tab-defending", children=html.Div(dbc.Row([create_metric_card(df_filtered, **m) for m in defensive_metrics]), className="p-2 mt-3")),
        ], className="mt-4"
    )

def layout():
    available_seasons = get_available_seasons()
    default_season = available_seasons[0] if available_seasons else None
    return dbc.Container([
        dbc.Row(dbc.Col(dbc.Button([html.I(className="fas fa-arrow-left me-2"), "Back to Home"], href="/", color="secondary")), className="my-3"),
        dcc.Store(id='team-stats-full-df-store'),
        dcc.Store(id='active-league-filter-store', data='All'),
        dbc.Row(dbc.Col(html.H1("Teams Overview - Top 5 European Leagues", className="text-center text-white")), className="my-4"),
        dbc.Row([
            dbc.Col(create_league_filter_buttons(), width="auto"),
            dbc.Col(dcc.Dropdown(id='season-filter-dropdown', options=[{'label': s, 'value': s} for s in available_seasons], value=default_season, clearable=False, style={'width': '200px', 'color': 'black'}), width="auto")
        ], justify="center", align="center", className="mb-4"),
        dcc.Loading(id="loading-main-content", type="circle", children=[
            html.Div(id="team-stats-tabs-content"),
            html.Div(id="league-quadrant-plots", className="mt-5")
        ]),
        app_signature()
    ], fluid=True, className="p-4")

@callback(
    Output('team-stats-full-df-store', 'data'),
    Input('season-filter-dropdown', 'value'),
    prevent_initial_call=False
)
def update_store_on_season_change(selected_season):
    if not selected_season:
        return None
    df = load_all_team_stats(selected_season)
    return df.to_json(orient='split')

@callback(
    Output('team-stats-tabs-content', 'children'),
    [Input('active-league-filter-store', 'data'),
     Input('team-stats-full-df-store', 'data')]
)
def update_cards_on_filter_change(selected_league, df_json):
    if not df_json:
        return dbc.Alert("Select a season to view data.", color="info", className="mt-4")
    df = pd.read_json(StringIO(df_json), orient='split')
    return generate_tabs_content(df, selected_league)

@callback(
    Output('active-league-filter-store', 'data'),
    [Input(f"filter-{league_folder}", 'n_clicks') for league_folder in LEAGUES.keys()] + [Input('filter-all', 'n_clicks')],
    prevent_initial_call=True
)
def update_active_league_filter(*clicks):
    from dash import ctx
    button_id = ctx.triggered_id
    if not button_id or button_id == 'filter-all':
        return 'All'
    league_folder = button_id.replace('filter-', '')
    return LEAGUES[league_folder]['name']

@callback(
    Output('league-quadrant-plots', 'children'),
    [Input('team-stats-tabs', 'active_tab')],
    [State('active-league-filter-store', 'data'),
     State('team-stats-full-df-store', 'data')]
)
def update_quadrant_plots(active_tab, selected_league, df_json):
    if not df_json:
        return None
    df = pd.read_json(StringIO(df_json), orient='split')

    title_prefix = selected_league if selected_league != 'All' else "European Top 20"
    
    if selected_league == 'All':
        source_df = df
    else:
        source_df = df[df['League'] == selected_league]

    if source_df.empty:
        return None
    
    plot_definitions = {
        "tab-attacking": {
            "title": f"{title_prefix} Attacking Style", "x_metric": "xG_per_Shot", "y_metric": "Goal_Conversion",
            "x_label": "Shot Quality (xG per Shot)", "y_label": "Finishing (% Conversion)", "quadrant_labels": ['Clinical & High Quality', 'Wasteful but High Quality', 'Low Quality & Poor Finishing', 'Clinical but Low Quality']
        },
        "tab-possession": {
            "title": f"{title_prefix} Possession Style", "x_metric": "Passing_Tempo", "y_metric": "Progressions_per_Touch",
            "x_label": "Passing Tempo", "y_label": "Verticality", "quadrant_labels": ['Fast & Vertical', 'Fast & Patient', 'Slow & Patient', 'Slow & Vertical']
        },
        "tab-defending": {
            "title": f"{title_prefix} Defensive Performance", "x_metric": "Defensive_Actions", "y_metric": "GA",
            "x_label": "Defensive Volume", "y_label": "Goals Conceded", "invert_y": True, "quadrant_labels": ['Proactive & Solid', 'Busy & Leaky', 'Passive & Vulnerable', 'Organized & Efficient']
        }
    }

    if active_tab not in plot_definitions:
        return None
    
    plot_params = plot_definitions[active_tab]
    plot_df = source_df

    if selected_league == 'All':
        if active_tab == 'tab-defending':
            # Seleziona le top 20 squadre per minor numero di gol subiti (GA)
            plot_df = plot_df.sort_values(by='GA', ascending=True).head(20)
        else:
            # Per le altre tab, usa la metrica Y per trovare i top 20
            plot_df = plot_df.sort_values(by=plot_params['y_metric'], ascending=False).head(20)

    if plot_df.empty:
        return None
        
    graph = dcc.Graph(figure=league_plots.create_quadrant_plot(plot_df, **plot_params))
    
    return dbc.Container([
        html.H2(f"{title_prefix} Quadrant Analysis", className="text-white text-center mb-4"),
        dbc.Row(dbc.Col(graph))
    ], fluid=True)