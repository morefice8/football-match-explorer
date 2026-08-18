# --- START OF FILE team_stats.py ---
import os
import pandas as pd
from io import StringIO # <-- FIX: AGGIUNTO IMPORT
from dash import html, dcc, Input, Output, callback
from src.visualization import league_plots
import dash_bootstrap_components as dbc
from src.components.layout_components import app_signature
from src.metrics.sportmonks import (
    SPORTMONKS_TOOLTIPS,
    TEAM_METRIC_GROUPS,
    TEAM_QUADRANTS,
    card_definitions,
    uses_sportmonks,
)
from src.utils.league_config import LEAGUES, LEAGUE_NAME_TO_FOLDER

# --- CONFIGURATION & HELPER FUNCTIONS ---
DATA_PATH = os.path.join("data", "fbref")
PROCESSED_DATA_PATH = os.path.join("data", "processed")
METRIC_TOOLTIPS = {
    "Gls": "Total goals scored by the team.", "xG_per_Shot": "Expected Goals per shot. A measure of average shot quality. Higher is better.",
    "Goal_Conversion": "Percentage of shots that result in a goal. (Goals / Total Shots).", "Ast_xAG_ratio": "Ratio of actual assists to Expected Assisted Goals. A value > 1 suggests high-quality chance creation or finishing.",
    "Passing_Tempo": "Average passes made per touch. Higher values indicate a faster, more direct style of play.", "Progressions_per_Touch": "Combined progressive passes and receptions per touch. Indicates how vertically a team plays.",
    "Cross_Touch_Ratio": "Percentage of touches that are crosses. Indicates reliance on wide play.", "TakeOn_Success_Rate": "Percentage of successful take-ons (dribbles).",
    "GA": "Total goals conceded by the team.", "Shots_Conceded_per_DA": "Shots conceded for every defensive action made. Lower is better, indicating efficiency.",
    "Tkl_Int_per_90": "Combined successful tackles and interceptions, normalized per 90 minutes.", "Errors_per_GA": "Number of errors leading to a shot, normalized by goals conceded. Lower is better."
}
METRIC_TOOLTIPS.update(SPORTMONKS_TOOLTIPS)

def get_team_logo_path(league_name, team_name):
    league_folder = LEAGUE_NAME_TO_FOLDER.get(league_name)
    if not league_folder: return ""
    teams_folder_name = f"{league_folder}_teams"
    team_folder_name = team_name.replace(' ', '_')
    return f"/data/fbref/{league_folder}/{teams_folder_name}/{team_folder_name}/logo.png"

def get_available_seasons():
    seasons = set()
    if os.path.isdir(PROCESSED_DATA_PATH):
        for filename in os.listdir(PROCESSED_DATA_PATH):
            if filename.startswith("team_stats_") and filename.endswith(".parquet"):
                seasons.add(filename[len("team_stats_"):-len(".parquet")])
    for league_folder in LEAGUES.keys():
        stats_path = os.path.join(DATA_PATH, league_folder, f"{league_folder}_stats")
        if os.path.isdir(stats_path):
            for item in os.listdir(stats_path):
                if os.path.isdir(os.path.join(stats_path, item)) and '-' in item:
                    seasons.add(item)
    return sorted(list(seasons), reverse=True)

def load_all_team_stats(season):
    # Sportmonks is the primary source when its normalized dataset exists.
    # The FBref loader below remains as a backwards-compatible fallback.
    sportmonks_path = os.path.join(PROCESSED_DATA_PATH, f"team_stats_{season}.parquet")
    if os.path.exists(sportmonks_path):
        try:
            return pd.read_parquet(sportmonks_path)
        except Exception as exc:
            print(f"Error reading Sportmonks team dataset for {season}: {exc}")

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
    buttons = [
        dbc.Button(
            html.I(className="fas fa-globe fa-2x"),
            id="filter-all", n_clicks=0, color="primary", outline=True,
            className="league-filter-btn", style=button_style,
        ),
        dbc.Tooltip("Show All Leagues", target="filter-all"),
    ]
    for league_folder, league_info in LEAGUES.items():
        button_id = f"filter-{league_folder}"
        logo = html.Div([
            html.Span(
                league_info['fallback'],
                className="fw-bold text-light",
                style={'position': 'absolute', 'inset': '0', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}
            ),
            html.Img(
                src=league_info['logo'], alt=league_info['name'], title=league_info['name'],
                style={**logo_style, 'position': 'relative', 'zIndex': 1}
            ),
        ], style={'position': 'relative', 'height': '40px', 'width': '64px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'})
        buttons.append(dbc.Button(
            logo, id=button_id, n_clicks=0, color="secondary", outline=True,
            className="league-filter-btn", style=button_style,
        ))
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
        logo_path = row.get('image_path') or get_team_logo_path(league_name, team_name)
        row_content = dbc.Row([
            dbc.Col(html.Img(src=logo_path, style=logo_style, title=team_name), width="auto", className="pe-3"),
            dbc.Col([
                dcc.Link(team_name, href=f"/team-stats/team/{team_name.replace(' ', '_')}", className=f"team-ranking-name fw-bold text-decoration-none {team_font_size}"),
                dcc.Link(league_name, href=f"/team-stats/league/{league_name.replace(' ', '_')}", className="team-ranking-league small text-decoration-none d-block")
            ], width=True, className="align-self-center"),
            dbc.Col(html.Span(f"{metric_display} {unit}", className=f"team-ranking-value fw-bold text-end {metric_font_size}"), width="auto", className="align-self-center")
        ], align="center", className="py-2")
        list_item_class = "team-ranking-item ranking-winner border-top-0" if is_first else "team-ranking-item"
        rows.append(dbc.ListGroupItem(row_content, color="dark", className=list_item_class))
    return rows

def create_metric_card(df, title, metric_column, ascending, unit="", format_spec="{:,.1f}", icon=""):
    if (df.empty or metric_column not in df.columns
            or pd.to_numeric(df[metric_column], errors='coerce').notna().sum() == 0):
        return dbc.Col(dbc.Alert(f"Data not available for metric: '{title}'", color="warning"))
    ranked = df.copy()
    ranked[metric_column] = pd.to_numeric(ranked[metric_column], errors='coerce')
    top_5_teams = ranked.dropna(subset=[metric_column]).sort_values(by=metric_column, ascending=ascending).head(5)
    header_id = f"card-header-{metric_column}"
    card_header = dbc.CardHeader(dbc.Row([dbc.Col(html.I(className=f"{icon} me-2"), width="auto"), dbc.Col(html.H5(title, className="m-0"), width=True)], align="center"), id=header_id, className="metric-card-header text-center")
    tooltip = dbc.Tooltip(METRIC_TOOLTIPS.get(metric_column, "No description available."), target=header_id, placement="top")
    return dbc.Col([
        dbc.Card([card_header, dbc.CardBody(dbc.ListGroup(create_team_ranking_rows(top_5_teams, metric_column, unit, format_spec), flush=True), className="p-0")], className="metric-card h-100"),
        tooltip
    ], lg=3, md=6, sm=12, className="mb-4")


def _select_quadrant_teams(df, selected_league):
    """Keep a stable comparison cohort across all four quadrant plots."""
    if df.empty or selected_league != "All" or "League" not in df.columns:
        return df.copy()

    ranked = df.copy()
    ranking_columns = [
        column for column in ("Pts_per_MP", "Pts", "GD", "Gls")
        if column in ranked.columns
    ]
    for column in ranking_columns:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")

    if ranking_columns:
        ranked = ranked.sort_values(
            ["League", *ranking_columns],
            ascending=[True, *([False] * len(ranking_columns))],
            na_position="last",
            kind="stable",
        )
    return (
        ranked.groupby("League", sort=True, group_keys=False)
        .head(4)
        .reset_index(drop=True)
    )


def create_quadrant_guide(labels, descriptions, comparison_note):
    """Render an always-visible explanation; no hover is required to understand the plot."""
    positions = (
        ("top_right", "Upper right"),
        ("bottom_right", "Lower right"),
        ("bottom_left", "Lower left"),
        ("top_left", "Upper left"),
    )
    if not isinstance(labels, dict):
        labels = dict(zip((position for position, _ in positions), labels or ()))
    if not isinstance(descriptions, dict):
        descriptions = {}
    cards = []
    for position, screen_label in positions:
        cards.append(
            dbc.Col(
                html.Div([
                    html.Div([
                        html.Span(className=f"quadrant-guide-swatch {position}"),
                        html.Span(screen_label, className="quadrant-guide-position"),
                    ], className="d-flex align-items-center gap-2 mb-2"),
                    html.H5(labels.get(position, ""), className="quadrant-guide-title"),
                    html.P(descriptions.get(position, ""), className="quadrant-guide-description mb-0"),
                ], className=f"quadrant-guide-card {position}"),
                lg=3,
                md=6,
                sm=12,
            )
        )

    return html.Div([
        html.P([
            html.I(className="fa-solid fa-circle-info me-2"),
            comparison_note,
            " Dashed axes are cohort medians, so the quadrants describe relative profiles rather than fixed quality thresholds.",
        ], className="quadrant-guide-intro"),
        dbc.Row(cards, className="g-3"),
    ], className="quadrant-guide mt-3")

def _filter_team_stats(df, selected_league):
    if selected_league and selected_league != "All":
        return df[df["League"] == selected_league].copy()
    return df.copy()


def generate_card_sections(df, selected_league='All'):
    """Build the four ranking sections from the currently selected league only."""
    df_filtered = _filter_team_stats(df, selected_league)
    if df_filtered.empty:
        return tuple(
            dbc.Alert(
                "No data available for the current selection.",
                color="info",
                className="m-3",
            )
            for _ in range(4)
        )
    if uses_sportmonks(df_filtered):
        attacking_metrics = card_definitions(TEAM_METRIC_GROUPS['attacking'], 'metric_column')
        possession_metrics = card_definitions(TEAM_METRIC_GROUPS['possession'], 'metric_column')
        defensive_metrics = card_definitions(TEAM_METRIC_GROUPS['defending'], 'metric_column')
        set_piece_metrics = card_definitions(TEAM_METRIC_GROUPS['set_pieces'], 'metric_column')
    else:
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
        set_piece_metrics = None

    if set_piece_metrics:
        set_piece_content = dbc.Row([
            create_metric_card(df_filtered, **metric) for metric in set_piece_metrics
        ])
    else:
        set_piece_content = dbc.Alert(
            "Set-piece xG is not available for legacy FBref seasons.",
            color="info",
            className="m-3",
        )

    return (
        dbc.Row([create_metric_card(df_filtered, **metric) for metric in attacking_metrics]),
        dbc.Row([create_metric_card(df_filtered, **metric) for metric in possession_metrics]),
        dbc.Row([create_metric_card(df_filtered, **metric) for metric in defensive_metrics]),
        set_piece_content,
    )

def layout():
    available_seasons = get_available_seasons()
    default_season = available_seasons[0] if available_seasons else None
    return dbc.Container([
        dbc.Row(dbc.Col(dbc.Button([html.I(className="fas fa-arrow-left me-2"), "Back to Home"], href="/", color="secondary", className="back-home-btn")), className="mb-3"),
        dcc.Store(id='team-stats-full-df-store'),
        dcc.Store(id='active-league-filter-store', data='All'),
        html.Div([
            html.H1("Teams Overview", className="team-stats-title text-center mb-2"),
            html.P(
                "Compare team performance across Europe's top five leagues",
                className="team-stats-subtitle text-center mb-0",
            ),
        ], className="team-stats-heading mb-4"),
        dbc.Row([
            dbc.Col(create_league_filter_buttons(), width="auto", className="league-filter-col"),
            dbc.Col(
                dcc.Dropdown(
                    id='season-filter-dropdown',
                    options=[{'label': s, 'value': s} for s in available_seasons],
                    value=default_season, clearable=False,
                    className="season-filter-dropdown",
                    style={'width': '200px'},
                ),
                width="auto",
            )
        ], justify="center", align="center", className="team-stats-filter-bar g-3 mb-4"),
        dcc.Loading(id="loading-main-content", type="circle", children=[
            dbc.Tabs(
                id="team-stats-tabs",
                active_tab="tab-attacking",
                children=[
                    dbc.Tab(
                        label="⚔️ Attacking",
                        tab_id="tab-attacking",
                        children=html.Div(id="team-attacking-cards", className="p-2 mt-3"),
                    ),
                    dbc.Tab(
                        label="⚽ Possession & Territory",
                        tab_id="tab-possession",
                        children=html.Div(id="team-possession-cards", className="p-2 mt-3"),
                    ),
                    dbc.Tab(
                        label="🛡️ Defending & Pressing",
                        tab_id="tab-defending",
                        children=html.Div(id="team-defending-cards", className="p-2 mt-3"),
                    ),
                    dbc.Tab(
                        label="🎯 Set Pieces",
                        tab_id="tab-set-pieces",
                        children=html.Div(id="team-set-pieces-cards", className="p-2 mt-3"),
                    ),
                ],
                className="team-metric-tabs mt-4",
            ),
            html.Div(id="league-quadrant-plots", className="mt-5")
        ]),
        app_signature()
    ], fluid=True, className="team-stats-page px-3 px-lg-4 py-4")

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
    [Output('team-attacking-cards', 'children'),
     Output('team-possession-cards', 'children'),
     Output('team-defending-cards', 'children'),
     Output('team-set-pieces-cards', 'children')],
    [Input('active-league-filter-store', 'data'),
     Input('team-stats-full-df-store', 'data')]
)
def update_cards_on_filter_change(selected_league, df_json):
    if not df_json:
        return [], [], [], []
    df = pd.read_json(StringIO(df_json), orient='split')
    return generate_card_sections(df, selected_league)

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
    [Output('filter-all', 'className')]
    + [Output(f"filter-{league_folder}", 'className') for league_folder in LEAGUES.keys()],
    Input('active-league-filter-store', 'data'),
)
def style_active_league_filter(selected_league):
    base_class = "league-filter-btn"
    classes = [f"{base_class} active" if selected_league == 'All' else base_class]
    classes.extend(
        f"{base_class} active" if selected_league == league_info['name'] else base_class
        for league_info in LEAGUES.values()
    )
    return classes

@callback(
    Output('league-quadrant-plots', 'children'),
    [Input('team-stats-tabs', 'active_tab'),
     Input('active-league-filter-store', 'data'),
     Input('team-stats-full-df-store', 'data')]
)
def update_quadrant_plots(active_tab, selected_league, df_json):
    if not df_json:
        return None
    df = pd.read_json(StringIO(df_json), orient='split')

    title_prefix = selected_league if selected_league != 'All' else "European Top 20"
    
    source_df = _filter_team_stats(df, selected_league)

    if source_df.empty:
        return None
    
    legacy_plot_definitions = {
        "tab-attacking": {
            "title": "Attacking Style", "x_metric": "xG_per_Shot", "y_metric": "Goal_Conversion",
            "x_label": "Shot Quality (xG per Shot)", "y_label": "Finishing (% Conversion)", "quadrant_labels": ['Clinical & High Quality', 'Wasteful but High Quality', 'Low Quality & Poor Finishing', 'Clinical but Low Quality']
        },
        "tab-possession": {
            "title": "Possession Style", "x_metric": "Passing_Tempo", "y_metric": "Progressions_per_Touch",
            "x_label": "Passing Tempo", "y_label": "Verticality", "quadrant_labels": ['Fast & Vertical', 'Fast & Patient', 'Slow & Patient', 'Slow & Vertical']
        },
        "tab-defending": {
            "title": "Defensive Performance", "x_metric": "Defensive_Actions", "y_metric": "GA",
            "x_label": "Defensive Volume", "y_label": "Goals Conceded", "invert_y": True, "quadrant_labels": ['Proactive & Solid', 'Busy & Leaky', 'Passive & Vulnerable', 'Organized & Efficient']
        }
    }

    plot_definitions = TEAM_QUADRANTS if uses_sportmonks(source_df) else legacy_plot_definitions
    if active_tab not in plot_definitions:
        return dbc.Alert(
            "This quadrant requires Sportmonks set-piece xG data and is not available for the selected legacy season.",
            color="info",
        )
    
    plot_params = dict(plot_definitions[active_tab])
    guide_descriptions = plot_params.pop('quadrant_guide', {})
    guide_labels = plot_params.get('quadrant_labels', {})
    plot_params['title'] = f"{title_prefix} {plot_params['title']}"
    plot_df = _select_quadrant_teams(source_df, selected_league)
    required = (plot_params['x_metric'], plot_params['y_metric'])
    if any(metric not in plot_df.columns or pd.to_numeric(plot_df[metric], errors='coerce').notna().sum() == 0 for metric in required):
        return None

    if plot_df.empty:
        return None
        
    graph = dcc.Graph(
        figure=league_plots.create_quadrant_plot(plot_df, **plot_params),
        config={'displayModeBar': False, 'responsive': True},
        className="quadrant-graph",
    )
    
    comparison_note = (
        "The same 20-team cohort is used in every tab: the top four by points per match from each league."
        if selected_league == 'All'
        else "All teams from the selected league are shown in every tab."
    )
    guide = create_quadrant_guide(guide_labels, guide_descriptions, comparison_note)

    return dbc.Container([
        html.H2(f"{title_prefix} Quadrant Analysis", className="quadrant-section-title text-center mb-4"),
        dbc.Row(dbc.Col(graph)),
        guide,
    ], fluid=True, className="quadrant-section")
