# --- START OF FILE app.py ---
import matplotlib
matplotlib.use('Agg')

import base64
import io
import os
import json
import re
import ast
from matplotlib import pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from dash import Dash, html as dash_html, dcc, Input, Output, dash_table, no_update
from dash.dependencies import State
import dash_bootstrap_components as dbc
import traceback
from datetime import datetime
import html
import dash
from dash.dependencies import ALL


from src.config import TEAM_NAME_TO_LOGO_CODE, LOGO_PREFIX, LOGO_EXTENSION, DEFAULT_LOGO_PATH
from src.visualization import pitch_plots, player_plots, buildup_phases, buildup_plotly, defensive_transitions_plotly, offensive_transitions_plotly, set_piece_plotly, cross_plots, league_plots, formation_plotly, formations, pass_plotly
from src.data_processing import preprocess, pass_processing
from src.utils import mapping_loader, formation_layouts
from src import config
from src.metrics import pass_metrics, player_metrics, buildup_metrics, transition_metrics, set_piece_metrics, cross_metrics, league_metrics, defensive_metrics

# Define colors (get from config if available, otherwise define fallbacks)
HCOL = getattr(config, 'DEFAULT_HCOL', 'tomato')
ACOL = getattr(config, 'DEFAULT_ACOL', 'skyblue')
VIOLET = getattr(config, 'VIOLET', '#a369ff') # From your main_analyze_match
GREEN = getattr(config, 'GREEN', '#69f900')    # From your main_analyze_match
BG_COLOR = getattr(config, 'BG_COLOR', 'white')
LINE_COLOR = getattr(config, 'LINE_COLOR', 'black')
PATH_EFFECTS_HEATMAP = [] # Define if plot_pass_heatmap needs it, or remove from plot function

# Imports for Buildup Tab - Commented out for now
# from src.data_preparation_for_plots import prepare_offensive_buildups_data
# from src.visualization.buildup_phases import plot_buildup_phases_with_summary

# App configuration
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
server = app.server

# Helper functions
def get_leagues():
    path = os.path.join("data", "matches")
    if not os.path.exists(path): return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_seasons(league):
    path = os.path.join("data", "matches", league)
    if not os.path.exists(path): return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_matches(league, season):
    path = os.path.join("data", "matches", league, season, "partidos")
    if not os.path.exists(path): return []
    return [f for f in os.listdir(path) if f.endswith(".json")]

def extract_rounds(matches):
    """Extracts and sorts round names. Attempts numerical sort for round numbers."""
    rounds_data = set() # Store tuples of (numeric_part, original_string) for sorting

    for file_name in matches:
        original_round_name = file_name.split("_")[0]
        
        # Try to extract a number from the beginning of the round name
        numeric_part_match = re.match(r"(\d+)", original_round_name) # Matches one or more digits at the start
        
        if numeric_part_match:
            numeric_value = int(numeric_part_match.group(1))
            rounds_data.add((numeric_value, original_round_name))
        else:
            # If no number found at the start, use a large number for sorting to place non-numeric/complex names last,
            # or handle as pure strings. For now, let's treat them as strings to be sorted alphabetically after numbers.
            # To sort them alphabetically *after* numbers, we can give them a sort key that's a tuple starting with a very large number or float('inf')
            rounds_data.add((float('inf'), original_round_name)) # Sorts non-numeric last

    # Sort first by the numeric part, then by the original string if numbers are the same (or for non-numeric)
    # For those with float('inf'), they will be sorted alphabetically amongst themselves at the end.
    sorted_rounds_data = sorted(list(rounds_data), key=lambda x: (x[0], x[1]))
    
    # Extract just the original round names for the dropdown
    sorted_round_names = [original_name for numeric_val, original_name in sorted_rounds_data]
    
    return sorted_round_names

def parse_match(filename):
    parts = filename.replace(".json", "").split("_")
    if len(parts) < 4: # Ensure enough parts for round, home, away, id
        print(f"Warning: Could not parse filename '{filename}'. Expected at least 4 parts.")
        return None
    round_name = parts[0]
    # Handle cases where team names might have underscores if the ID is not the last part
    match_id = parts[-1]
    home_team = parts[1]
    away_team = "_".join(parts[2:-1]) if len(parts) > 3 else "UnknownAway" # Simplistic handling

    return {
        "round": round_name,
        "home_team": home_team,
        "away_team": away_team,
        "id": match_id,
        "file": filename
    }

def get_team_logo_src(team_name, default_logo_path="/assets/logos/_default_badge.png"):
    if not team_name:
        return default_logo_path

    team_code = TEAM_NAME_TO_LOGO_CODE.get(str(team_name).strip()) # Use .strip() for safety

    if not team_code:
        # Fallback if team name not in mapping: try to generate a code from the name
        # This is less reliable but can be a fallback.
        # For example, take the first 3 letters if you don't have an explicit code.
        # Or, if you expect team names like "Wolverhampton Wanderers" and want "WOL"
        # you might need a more complex fallback logic.
        # For now, if not in map, use default.
        print(f"Warning: Team name '{team_name}' not found in TEAM_NAME_TO_LOGO_CODE mapping. Using default logo.")
        return default_logo_path

    logo_filename = f"{LOGO_PREFIX}{team_code}{LOGO_EXTENSION}" # e.g., ENG_BOU.png
    prospective_src = f"/assets/logos/{logo_filename}"
    
    # Optional server-side check (as before)
    # logo_path_on_server = os.path.join("assets", "logos", logo_filename)
    # if not os.path.exists(logo_path_on_server):
    #     print(f"DEV_NOTE: Logo file not found on server: {logo_path_on_server} for team '{team_name}' (code: {team_code})")
    #     return default_logo_path
        
    return prospective_src

def get_team_logo_src_by_code(team_short_code): # Removed default here, handle in show_cards
    if not team_short_code:
        # print(f"Warning: No team short code provided for logo. Using default.")
        return DEFAULT_LOGO_PATH

    # Assuming your logo files like ENG_BOU.png use uppercase codes
    logo_filename = f"{LOGO_PREFIX}{str(team_short_code).upper()}{LOGO_EXTENSION}"
    prospective_src = f"/assets/logos/{logo_filename}"
    
    # Optional server-side check for debugging (uncomment if needed)
    # logo_path_on_server = os.path.join("assets", "logos", logo_filename)
    # if not os.path.exists(logo_path_on_server):
    #     print(f"DEV_NOTE: Logo file not found on server: {logo_path_on_server} (code: {team_short_code})")
    #     return DEFAULT_LOGO_PATH # Fallback if server check fails
            
    return prospective_src

# Main layout (only one layout, using dynamic display)
app.layout = dash_html.Div([
    dcc.Location(id='url'),
    dash_html.Div(id='page-content', className='default-background')
])

@app.callback(Output('page-content', 'className'),
              Input('url', 'pathname'))
def update_background_class(pathname):
    if pathname == '/' or pathname == '/home':
        return 'home-background'
    return 'default-background'

# Home page layout
def layout_home():
    return dbc.Container([
        dash_html.H1("Match Explorer", className="text-center mt-3 mb-4"),
         dbc.Row(
            dbc.Col(
                dbc.Button(
                    [dash_html.I(className="fas fa-chart-line me-2"), "Go to League Analysis"],
                    href="/league-analysis",
                    color="success",
                    size="lg",
                    className="w-100"
                ),
                width={"size": 6, "offset": 3} # Centra il bottone
            ),
            className="my-4" # Aggiunge margine sopra e sotto
        ),
        dash_html.Hr(),
        dbc.Row([
            dbc.Col([
                dash_html.Label("League"),
                dcc.Dropdown(
                    id="dropdown-league",
                    options=[{"label": league, "value": league} for league in get_leagues()],
                    placeholder="Select a league"
                ),
            ], width=3),
            dbc.Col([
                dash_html.Label("Season"),
                dcc.Dropdown(id="dropdown-season", placeholder="Select a season"),
            ], width=3),
            dbc.Col([
                dash_html.Label("Team"),
                dcc.Dropdown(id="dropdown-team-filter", placeholder="Optional: Select a team..."),
            ], width=3),
            dbc.Col([
                dash_html.Label("Round"),
                dcc.Dropdown(id="dropdown-round", placeholder="Select a round"),
            ], width=3),
        ], className="mb-4"),
        dash_html.Hr(),
        dash_html.H4("Matches found:", className="mt-3 mb-3 text-white"),
        dbc.Row(id="match-list")
    ])

# --- Match page layout (MODIFIED FOR SIDEBAR TABS) ---
def layout_match(match_id):
    sidebar_style = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "18rem",
        "padding": "2rem 1rem",
        "backgroundColor": "#2E3439",
        "overflowY": "auto",
        "display": "flex", # Added for flex layout
        "flexDirection": "column" # Added for flex layout
    }
    content_style = {
        "marginLeft": "20rem",
        "marginRight": "1rem",
        "padding": "2rem 1rem",
        "maxWidth": "calc(100vw - 22rem)",
    }
    tabs_config = [
        {"label": "Match Overview", "value": "overview", "icon": "fas fa-clipboard-list"},
        {"label": "Formation", "value": "formation", "icon": "fas fa-users"},
        {"label": "Passes", "value": "passes", "icon": "fas fa-exchange-alt"},
        {"label": "Buildup", "value": "buildup", "icon": "fas fa-sitemap"},
        {"label": "Def. Transition", "value": "defensive-transition", "icon": "fas fa-shield-alt"},
        {"label": "Off. Transition", "value": "offensive-transition", "icon": "fas fa-bolt"},
        {"label": "Set Piece", "value": "set-piece", "icon": "fas fa-flag"},
        {"label": "Player Analysis", "value": "player_analysis", "icon": "fas fa-user-astronaut"},
    ]
    sidebar = dash_html.Div(
        [
            dash_html.Div(id="sidebar-match-header", className="text-white text-center mb-3"),
            dash_html.Hr(className="text-white"),
            dbc.Nav(
                [
                    dbc.NavLink(
                        [dash_html.I(className=f"{tab['icon']} me-2"), tab["label"]],
                        href=f"/match/{match_id}?tab={tab['value']}",
                        active="exact",
                        id=f"navlink-{tab['value']}",
                        className="py-2"
                    ) for tab in tabs_config
                ],
                vertical=True, pills=True, className="mb-3 flex-grow-1" # flex-grow-1 to push buttons down
            ),
            dash_html.Img(src="/assets/liverpool_logo.png", style={
                "width": "120px",
                "margin": "30px auto 20px auto",
                "display": "block"
                }
            ),
            dbc.Button(
                [dash_html.I(className="fas fa-file-alt me-2"), "Generate Report"], # Icon for report
                id="generate-report-button",
                color="success",
                className="mb-2 w-100" # Margin bottom and full width
            ),
            dbc.Button(
                [dash_html.I(className="fas fa-home me-2"), "Back to Home"], # Icon for home
                href="/",
                color="secondary",
                className="w-100" # Takes full width of its container
            )
        ],
        style=sidebar_style,
    )
    content_area = dash_html.Div(id="match-tab-content", style=content_style)

    return dash_html.Div([
    dcc.Store(id="store-df-match"),
    dcc.Store(id="store-comment-pass-network", storage_type="local"),
    dcc.Store(id="store-comment-progressive-passes", storage_type="local"),
    dcc.Store(id="store-comment-formation", storage_type="local"),
    dcc.Store(id="store-comment-final-third", storage_type="local"),
    dcc.Store(id="store-comment-pass-density", storage_type="local"),
    dcc.Store(id="store-comment-pass-heatmap", storage_type="local"),
    dcc.Store(id="store-comment-top-passers-bar", storage_type="local"),
    dcc.Store(id="store-comment-home-top-passer-map", storage_type="local"),
    dcc.Store(id="store-comment-away-top-passer-map", storage_type="local"),
    dcc.Store(id="store-comment-shot-sequence-bar", storage_type="local"),
    dcc.Store(id="store-comment-home-top-shot-contributor-map", storage_type="local"),
    dcc.Store(id="store-comment-away-top-shot-contributor-map", storage_type="local"),
    dcc.Store(id="store-comment-defender-stats-bar", storage_type="local"),
    dcc.Store(id="store-comment-home-top-defender-map", storage_type="local"),
    dcc.Store(id="store-comment-away-top-defender-map", storage_type="local"),
    dcc.Store(id="store-comment-buildup", storage_type="local"), 
    dcc.Store(id="store-player-stats-df"),
    dcc.Store(id="store-buildup-filter", storage_type="memory"),
    dcc.Store(id="store-def-transition-filter", data=None),
    dcc.Store(id="store-off-transition-filter", data=None),
    dcc.Store(id="store-set-piece-filter", data=None),
    dcc.Store(id="set-piece-analyzed-df-store", data=None),
    dcc.Store(id="cross-filter-store", data=None),
    dcc.Store(id="cross-selection-store", data=None),
    # ... other comment stores ...
    dcc.Store(id="report-html-content-store"),
    dcc.Location(id="url-match-page", refresh=False),
    dash_html.Div(id="clientside-report-trigger-div", style={"display": "none"}),
    # dash_html.Div(id="debug-def-filter", className="text-white small"),
    sidebar,
    content_area
])

# Callback to change the page based on the URL
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def show_page(pathname):
    if pathname and pathname.startswith("/match/"):
        match_id = pathname.split("/")[-1]
        if not match_id: return layout_home()
        return layout_match(match_id)
    elif pathname == "/league-analysis":
        return layout_league_analysis()
    return layout_home()

# Dynamic callbacks for home
@app.callback(
    Output("dropdown-season", "options"),
    Output("dropdown-season", "value"),
    Input("dropdown-league", "value")
)
def update_seasons(selected_league):
    if selected_league:
        seasons = get_seasons(selected_league)
        options = [{"label": s, "value": s} for s in seasons]
        return options, None
    return [], None

@app.callback(
    Output("dropdown-round", "options"),
    Output("dropdown-round", "value"),
    Input("dropdown-league", "value"),
    Input("dropdown-season", "value"),
    prevent_initial_call=True
)
def update_rounds(league, season):
    if league and season:
        matches = get_matches(league, season)
        rounds = extract_rounds(matches)
        options = [{"label": round_name, "value": round_name} for round_name in rounds]
        return options, None
    return [], None

@app.callback(
    Output("match-list", "children"),
    Input("dropdown-league", "value"),
    Input("dropdown-season", "value"),
    Input("dropdown-team-filter", "value"),
    Input("dropdown-round", "value"),
    prevent_initial_call=True,
)
def show_cards(league, season, team_filter, round_name_filter):
    if not (league and season):
        return ""

    all_matches_in_season_files = get_matches(league, season)
    matches_data_for_cards = []
    base_path_matches = os.path.join("data", "matches", league, season, "partidos")

    for m_filename in all_matches_in_season_files:
        try:
            # --- Blocco logico per ogni file ---
            filename_parsed_info = parse_match(m_filename)
            if not filename_parsed_info:
                continue

            # Filtro per Girone (Round)
            if round_name_filter and filename_parsed_info['round'] != round_name_filter:
                continue

            # Caricamento e parsing del JSON
            match_file_path = os.path.join(base_path_matches, m_filename)
            with open(match_file_path, 'r', encoding='utf-8') as f:
                json_data_for_card = json.load(f)

            temp_match_info = config.extract_match_info(json_data_for_card)
            
            # --- LOGICA DI FILTRAGGIO SPOSTATA QUI DENTRO ---
            home_team_display = temp_match_info.get('hteamDisplayName')
            away_team_display = temp_match_info.get('ateamDisplayName')

            if team_filter and (team_filter not in [home_team_display, away_team_display]):
                continue
            # -----------------------------------------------

            match_details_for_card = {
                'filename': m_filename,
                'parsed_base_info': filename_parsed_info,
                'home_team_display_name': home_team_display,
                'away_team_display_name': away_team_display,
                'home_team_code_for_logo': temp_match_info.get('hteamCode'),
                'away_team_code_for_logo': temp_match_info.get('ateamCode'),
                'home_score': temp_match_info.get('home_score'),
                'away_score': temp_match_info.get('away_score'),
                'date_iso_for_sort': None,
                'date_formatted_for_display': temp_match_info.get('date_formatted', "N/A"),
                'competitionName': temp_match_info.get('competitionName'),
                'numeric_round_sort_key': float('inf'),
                'original_round_name': filename_parsed_info['round']
            }
            
            numeric_parts_round = re.findall(r"(\d+)", filename_parsed_info['round'])
            if numeric_parts_round:
                try:
                    match_details_for_card['numeric_round_sort_key'] = int(numeric_parts_round[0])
                except ValueError:
                    pass

            iso_date_str = temp_match_info.get('date_iso')
            if iso_date_str:
                try:
                    cleaned_iso_date_str = iso_date_str.replace('Z', '')
                    dt_obj = datetime.strptime(cleaned_iso_date_str.split("T")[0], "%Y-%m-%d")
                    match_details_for_card['date_iso_for_sort'] = dt_obj
                except ValueError:
                    match_details_for_card['date_iso_for_sort'] = datetime.max
            
            matches_data_for_cards.append(match_details_for_card)

        except Exception as e:
            print(f"Warning: Could not process card data for {m_filename}: {e}")
            continue # Se c'è un errore, salta al prossimo file

    if not matches_data_for_cards:
        return dbc.Alert("No matches found for this selection.", color="warning")

    # --- Il resto della funzione per ordinare e creare le card rimane IDENTICO ---
    def sort_key_for_card(match_data):
        date_for_sort = match_data.get('date_iso_for_sort', datetime.max)
        if date_for_sort is None:
            date_for_sort = datetime.max
        return (
            match_data['numeric_round_sort_key'],
            match_data['original_round_name'],
            date_for_sort,
            match_data.get('home_team_display_name', '')
        )
    sorted_matches_data = sorted(matches_data_for_cards, key=sort_key_for_card)
    
    cards = []
    logo_style = {"height": "40px", "width": "40px", "objectFit": "contain", "marginRight": "8px", "marginLeft": "8px"}

    for match_data in sorted_matches_data:
        pbi_from_filename = match_data['parsed_base_info']
        
        home_display_name_for_card = match_data['home_team_display_name']
        away_display_name_for_card = match_data['away_team_display_name']
        home_code_for_logo_card = match_data['home_team_code_for_logo']
        away_code_for_logo_card = match_data['away_team_code_for_logo']

        home_logo_src = get_team_logo_src_by_code(home_code_for_logo_card)
        away_logo_src = get_team_logo_src_by_code(away_code_for_logo_card)
        
        score_display_elements = [dash_html.Span("vs", className="mx-2")]
        if match_data['home_score'] is not None and match_data['away_score'] is not None:
            score_display_elements = [
                dash_html.Span(f"{match_data['home_score']}", className="fw-bold fs-5"),
                dash_html.Span("-", className="mx-2"),
                dash_html.Span(f"{match_data['away_score']}", className="fw-bold fs-5")
            ]
        
        card_header_content = [
            dash_html.Span(f"Round: {match_data['original_round_name']}", className="me-3"),
        ]
        if match_data['date_formatted_for_display'] != "N/A":
            card_header_content.append(dash_html.Span(f"{match_data['date_formatted_for_display']}"))
        card_header = dbc.CardHeader(dash_html.Div(card_header_content, className="small text-muted text-center"))

        teams_and_score_row = dbc.Row([
            dbc.Col([
                dash_html.Img(src=home_logo_src, style=logo_style),
                dash_html.Span(home_display_name_for_card, className="fw-bold")
            ], width="auto", className="d-flex align-items-center justify-content-end"),
            
            dbc.Col(
                score_display_elements,
                width="auto", 
                className="d-flex align-items-center justify-content-center px-0"
            ),
            
            dbc.Col([
                dash_html.Img(src=away_logo_src, style=logo_style),
                dash_html.Span(away_display_name_for_card, className="fw-bold")
            ], width="auto", className="d-flex align-items-center justify-content-start")
        ], justify="center", align="center", className="my-3")

        card_body_content = [
            teams_and_score_row,
            dbc.Button(
                "View Match", color="primary", href=f"/match/{pbi_from_filename['id']}",
                className="w-100 mt-auto"
            )
        ]
        
        card = dbc.Col(
            dbc.Card([
                card_header,
                dbc.CardBody(card_body_content, className="d-flex flex-column")
            ], className="mb-4 shadow-sm h-100"),
            lg=4, md=6, sm=12
        )
        cards.append(card)
        
    return dbc.Row(cards)


@app.callback(
    Output("store-df-match", "data"), 
    Input("url", "pathname")          
)
def load_data_to_store(pathname):
    if not pathname or not pathname.startswith("/match/"):
        return None

    path_parts = pathname.split("/")
    match_id_from_url = path_parts[-1] if path_parts[-1] else path_parts[-2]

    if not match_id_from_url:
        print("load_data_to_store: Invalid match URL.")
        return None

    print(f"load_data_to_store: Attempting to load data for match_id: {match_id_from_url}")

    for league in get_leagues():
        for season in get_seasons(league):
            path_matches = os.path.join("data", "matches", league, season, "partidos")
            if not os.path.exists(path_matches):
                continue
            for file_name in os.listdir(path_matches): # file_name is the actual filename e.g., Round1_TeamA_TeamB_ID.json
                if file_name.endswith(".json"):
                    parsed_file_info = parse_match(file_name) # This gets {'round': 'Round1', ...}
                    if parsed_file_info and parsed_file_info["id"] == match_id_from_url:
                        input_file_path = os.path.join(path_matches, file_name)
                        try:
                            with open(input_file_path, 'r', encoding='utf-8') as f:
                                json_data = json.load(f)
                            
                            event_map = mapping_loader.load_opta_event_mapping(config.OPTA_EVENTS_XLSX)
                            qualifier_map = mapping_loader.load_opta_qualifier_mapping(config.OPTA_QUALIFIERS_JSON)
                            
                            # Get base match_info from the JSON content
                            match_info = config.extract_match_info(json_data)
                            if not isinstance(match_info, dict):
                                match_info = dict(match_info) if hasattr(match_info, '__dict__') else {}

                            # --- ADD ROUND FROM FILENAME PARSING ---
                            if parsed_file_info and 'round' in parsed_file_info:
                                match_info['roundNameFromFilename'] = parsed_file_info['round']
                                print(f"Added round from filename: {parsed_file_info['round']}")
                            # --------------------------------------

                            df, _, _, _ = preprocess.process_opta_events(
                                json_data, event_map, qualifier_map, match_info
                            )
                            if df is not None and not df.empty:
                                print("--- load_data_to_store: Post-Preprocessing Checks ---")
                                print(f"'is_key_pass' in df_processed: {'is_key_pass' in df.columns}")
                                if 'is_key_pass' in df.columns:
                                    print(f"  df_processed['is_key_pass'] dtype: {df['is_key_pass'].dtype}")
                                    print(f"  df_processed['is_key_pass'] sum: {df['is_key_pass'].sum()}")
                                    # Check for J. O'Brien specifically if you know an eventId
                                    # print(df[df['playerName'] == "J. O'Brien"][['is_key_pass', 'type_name', 'Assist']]) # 'Assist' or your raw qualifier col
                                stored_data = {
                                    'df': df.to_json(date_format='iso', orient='split'),
                                    'match_info': json.dumps(match_info) # match_info now includes roundNameFromFilename
                                }
                                print(f"Data for {match_id_from_url} loaded into store successfully.")
                                return stored_data
                            else:
                                print(f"Processing resulted in an empty DataFrame for {match_id_from_url}.")
                                return None
                        except Exception as e:
                            tb_str = traceback.format_exc()
                            print(f"Error processing JSON file '{input_file_path}' for store: {e}\n{tb_str}")
                            return None
    
    print(f"Match file for ID '{match_id_from_url}' not found for store loading.")
    return None

@app.callback(
    Output("sidebar-match-header", "children"),
    Input("store-df-match", "data"),
    Input("url-match-page", "pathname") # To get the raw match_id if store is not yet populated
)
def update_sidebar_header(stored_data_json, pathname):
    match_id_from_url = "Loading..."
    if pathname and pathname.startswith("/match/"):
        path_parts = pathname.split("/")
        match_id_from_url = path_parts[-1] if path_parts[-1] else path_parts[-2]

    default_header_content = [
        dash_html.H5(f"Match ID: {match_id_from_url}", className="mb-1"),
        dash_html.P("Loading details...", className="small text-muted opacity-75 mb-0")
    ]
    header_content = default_header_content

    if stored_data_json:
        try:
            match_info_json_str = stored_data_json.get('match_info')
            if match_info_json_str:
                match_info = json.loads(match_info_json_str)

                # --- Use specific names for display and codes for logos ---
                hteam_display_name = match_info.get('hteamDisplayName', 'Home') # Use contestant.shortName
                ateam_display_name = match_info.get('ateamDisplayName', 'Away')   # Use contestant.shortName

                home_code_for_logo = match_info.get('hteamCode') # Use contestant.code
                away_code_for_logo = match_info.get('ateamCode')   # Use contestant.code
                # ----------------------------------------------------------
                
                home_score = match_info.get('home_score')
                away_score = match_info.get('away_score')
                
                competition = match_info.get('competitionName', '')
                round_name_from_file = match_info.get('roundNameFromFilename', '')
                gw = match_info.get('gw', '')
                
                round_display = round_name_from_file
                if not round_display and gw:
                    round_display = f"GW {gw}"
                
                game_date = match_info.get('date_formatted', '')

                home_logo_src = get_team_logo_src_by_code(home_code_for_logo) # Pass the code
                away_logo_src = get_team_logo_src_by_code(away_code_for_logo) # Pass the code
                
                sidebar_logo_style = {"height": "28px", "width": "28px", "objectFit": "contain"}
                team_name_style = {"fontSize": "0.9rem"}

                line1_parts = []
                if competition: line1_parts.append(competition)
                if round_display:
                    if line1_parts: line1_parts.append(f"- {round_display}")
                    else: line1_parts.append(round_display)
                line1_display_text = " ".join(line1_parts)

                home_team_elements = [
                    dbc.Col(dash_html.Img(src=home_logo_src, style=sidebar_logo_style), width="auto", className="pe-2 align-self-center"),
                    dbc.Col(dash_html.Span(hteam_display_name, className="fw-bold", style=team_name_style), width=True, className="align-self-center text-start"), # text-start
                ]
                if home_score is not None:
                    home_team_elements.append(dbc.Col(dash_html.Span(str(home_score), className="fw-bold fs-5"), width="auto", className="ps-2 align-self-center"))
                home_team_display_row = dbc.Row(home_team_elements, align="center", className="mb-1 gx-2")

                away_team_elements = [
                    dbc.Col(dash_html.Img(src=away_logo_src, style=sidebar_logo_style), width="auto", className="pe-2 align-self-center"),
                    dbc.Col(dash_html.Span(ateam_display_name, className="fw-bold", style=team_name_style), width=True, className="align-self-center text-start"), # text-start
                ]
                if away_score is not None:
                    away_team_elements.append(dbc.Col(dash_html.Span(str(away_score), className="fw-bold fs-5"), width="auto", className="ps-2 align-self-center"))
                away_team_display_row = dbc.Row(away_team_elements, align="center", className="gx-2")

                separator = dash_html.Div()
                if home_score is None or away_score is None:
                    separator = dash_html.P("vs", className="text-center my-1 small text-muted")

                line4_display_text = game_date if game_date else ""

                header_content_list = []
                if line1_display_text:
                    header_content_list.append(dash_html.P(line1_display_text, className="mb-2 small text-muted opacity-75 text-center"))
                
                header_content_list.append(home_team_display_row)
                if separator.children:
                     header_content_list.append(separator)
                header_content_list.append(away_team_display_row)

                if line4_display_text:
                    header_content_list.append(dash_html.P(line4_display_text, className="mt-2 small text-muted opacity-75 text-center mb-0"))
                
                header_content = dash_html.Div(header_content_list)
            
            else:
                 header_content = dash_html.Div([dash_html.H5(f"Match: {match_id_from_url}", className="mb-1"), dash_html.P("Details loading...", className="small text-muted")])
        
        except Exception as e:
            tb_str = traceback.format_exc()
            print(f"Error updating sidebar header: {e}\n{tb_str}")
            header_content = dash_html.Div([dash_html.H5(f"Match ID: {match_id_from_url}", className="mb-1"), dash_html.P("Error loading details.", className="small text-danger")])
            
    return header_content

# --- NEW CALLBACK TO RENDER TAB CONTENT ---
@app.callback(
    Output("match-tab-content", "children"),
    Input("url-match-page", "search"),  # Listen to query parameters like ?tab=formation
    Input("store-df-match", "data")   # Depends on the match data being loaded
)
def render_match_tab_content(search_query, stored_data_json):
    print(f"--- render_match_tab_content ---")
    print(f"Search Query: {search_query}")

    active_tab = "overview" # Initialize with a default value HERE

    if search_query: # If search_query is not None and not empty
        try:
            # Robust parsing for query parameters
            query_params = {}
            stripped_query = search_query.lstrip("?")
            if stripped_query: # Ensure there's something to split
                for qc in stripped_query.split("&"):
                    if "=" in qc:
                        key, value = qc.split("=", 1) # Split only on the first '='
                        query_params[key] = value
            active_tab = query_params.get("tab", "overview") # Get 'tab', default to 'overview' if not found
        except ValueError:
            print(f"Warning: Could not parse query_params from '{search_query}'. Defaulting to overview.")
            active_tab = "overview" # Fallback in case of parsing error
            
    print(f"Active Tab Determined: {active_tab}")

    if not stored_data_json and active_tab not in ["overview", None]: # Allow overview to attempt render even if store is briefly None
        return dbc.Alert("Match data loading...", color="info")
    
   
    # print(f"Rendering tab: {active_tab}") # Moved this print after active_tab is definitely set

    if active_tab == "overview":
        # ... (rest of your overview logic)
        if not stored_data_json: # This check is a bit redundant if already done above
            return dash_html.P("⚠ No data in store for overview.", style={"color": "orange"})
        try:
            df_json_str = stored_data_json.get('df')
            if not df_json_str: return dash_html.P("⚠ DataFrame missing in stored data.", style={"color": "orange"})
            df = pd.read_json(io.StringIO(df_json_str), orient='split')
            if df.empty: return dash_html.P("⚠ The DataFrame is empty.", style={"color": "orange"})

            datatable_component = dash_table.DataTable(
                id='overview-datatable',  
                data=df.to_dict("records"), 
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=15,
                filter_action="native", 
                sort_action="native",   
                style_table={"overflowX": "scroll", "maxWidth":"100%"},
                style_cell={"backgroundColor": "#343A40", "color": "white", "textAlign": "left", 
                            "minWidth": "120px", "maxWidth":"250px", "whiteSpace":"normal", "border": "1px solid #454D55"},
                style_header={"backgroundColor": "#454D55", "color": "white", "fontWeight": "bold", "borderBottom": "2px solid #6C757D"}
            )

            return dash_html.Div([
                dbc.Row([
                    dbc.Col(dash_html.H4("Match Events Overview", className="text-white mb-3"), width='auto'),
                    dbc.Col(dbc.Button([dash_html.I(className="fas fa-download me-2"), "Download Full CSV"], id="btn-download-csv", color="info", size="sm"), width='auto', className="ms-auto")
                ], align="center"),
                
                dash_html.P(f"Displaying all {df.shape[0]} events:", className="text-muted small"),
                datatable_component,
                dcc.Download(id="download-dataframe-csv"), # Componente per gestire il download
            ], className="p-3")
            
            # return dash_html.Div([
            #     dash_html.H4("Match Events Overview", className="text-white mb-3"),
            #     dash_html.P(f"Displaying first 100 (of {df.shape[0]}) events:", className="text-muted small"),
            #     dash_table.DataTable(
            #         data=df.head(100).to_dict("records"),
            #         columns=[{"name": i, "id": i} for i in df.columns],
            #         page_size=15,
            #         style_table={"overflowX": "scroll", "maxWidth":"100%"},
            #         style_cell={"backgroundColor": "#343A40", "color": "white", "textAlign": "left", 
            #                     "minWidth": "120px", "maxWidth":"250px", "whiteSpace":"normal", "border": "1px solid #454D55"},
            #         style_header={"backgroundColor": "#454D55", "color": "white", "fontWeight": "bold", "borderBottom": "2px solid #6C757D"} # Using a theme color
            #     )
            # ], className="p-3")
        except Exception as e:
            return dbc.Alert(f"Error loading overview: {e}", color="danger")

    elif active_tab == "formation":
        # if not stored_data_json:
        #     return dbc.Alert("Match data loading for formation...", color="info")
        # try:
        #     df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        #     df_processed = df_processed.reset_index().rename(columns={'index': 'event_sequence_index'})
            
        #     match_info = json.loads(stored_data_json['match_info'])

        #     # --- 1. SETUP INIZIALE (ROBUSTO) ---
            
        #     # Mappa dati giocatori
        #     player_data_map = {}
        #     if not df_processed.empty:
        #         df_players_unique = df_processed.dropna(subset=['playerId', 'Mapped Jersey Number']).drop_duplicates(subset=['playerId'])
        #         for _, player in df_players_unique.iterrows():
        #             player_id = player['playerId']
        #             jersey_num_raw = player['Mapped Jersey Number']
        #             try:
        #                 jersey_num = int(jersey_num_raw)
        #             except (ValueError, TypeError):
        #                 jersey_num = '?'
        #             player_data_map[player_id] = {'name': player.get('playerName', 'N/A'), 'jersey': str(jersey_num)}

        #     # Recupero sicuro degli eventi di formazione iniziale
        #     start_events = df_processed[df_processed['typeId'] == 34].sort_values('eventId')
        #     if len(start_events) < 2:
        #         return dbc.Alert("Error: Could not find starting formation events for both teams.", color="danger")
            
        #     home_team_name_from_info = match_info.get('hteamName')
        #     event1, event2 = start_events.iloc[0], start_events.iloc[1]
            
        #     if home_team_name_from_info and event1['team_name'] == home_team_name_from_info:
        #         home_start_event, away_start_event = event1, event2
        #     elif home_team_name_from_info and event2['team_name'] == home_team_name_from_info:
        #         home_start_event, away_start_event = event2, event1
        #     else:
        #         home_start_event, away_start_event = event1, event2
            
        #     home_id, away_id = home_start_event['contestantId'], away_start_event['contestantId']
        #     home_name, away_name = home_start_event['team_name'], away_start_event['team_name']
            
        #     home_state = {'formation_id': int(home_start_event['Team formation']), 'players': formations._extract_player_positions(home_start_event)}
        #     away_state = {'formation_id': int(away_start_event['Team formation']), 'players': formations._extract_player_positions(away_start_event)}
            
        #     # --- 2. LOGICA DI COSTRUZIONE SINCRONA (AGGIORNATA) ---
        #     home_plots, timeline_items, away_plots = [], [], []

        #     # Stato iniziale (t=0)
        #     title = f"0' | Starting XI"
        #     home_plots.append(dash_html.Img(src=formations.plot_formation_snapshot(home_state, {}, player_data_map, HCOL, title), style={'width': '100%', 'height': 'auto', 'margin-bottom': '15px'}))
        #     away_plots.append(dash_html.Img(src=formations.plot_formation_snapshot(away_state, {}, player_data_map, ACOL, title, is_away=True), style={'width': '100%', 'height': 'auto', 'margin-bottom': '15px'}))
        #     timeline_items.append(dbc.ListGroupItem([dash_html.H5("Match Timeline", className="text-white"), dash_html.P("0' - Kick Off")], className="bg-dark text-white text-center"))
            
            
            
        #     # Prendi solo gli eventi di cambio formazione
        #     formation_change_events = df_processed[df_processed['typeId'] == 40].sort_values('event_sequence_index')

        #     for _, fc_event in formation_change_events.iterrows():
        #         time_str = f"{fc_event['timeMin']}'"
        #         previous_home_state, previous_away_state = home_state.copy(), away_state.copy()
                
        #         # Aggiorna lo stato della squadra che ha cambiato formazione
        #         if fc_event['contestantId'] == home_id:
        #             home_state = {'formation_id': int(fc_event['Team formation']), 'players': formations._extract_player_positions(fc_event)}
        #         else:
        #             away_state = {'formation_id': int(fc_event['Team formation']), 'players': formations._extract_player_positions(fc_event)}
                
        #         # Calcola lo score PRIMA di questo evento, per riflettere lo stato al momento del cambio
        #         goals_before = df_processed[(df_processed['typeId'] == 16) & (df_processed['event_sequence_index'] < fc_event['event_sequence_index'])]
        #         home_score = (goals_before['contestantId'] == home_id).sum()
        #         away_score = (goals_before['contestantId'] == away_id).sum()
        #         score_str = f"{home_score} - {away_score}"

        #         # Determina i colori per l'highlight
        #         home_player_colors = {pid: '#00FFFF' for pid, pos in home_state['players'].items() if previous_home_state['players'].get(pid) != pos}
        #         away_player_colors = {pid: '#00FFFF' for pid, pos in away_state['players'].items() if previous_away_state['players'].get(pid) != pos}

        #         # Costruisci i titoli per i plot
        #         event_team_name = home_name if fc_event['contestantId'] == home_id else away_name
        #         title = f"{time_str} | Formation Change: {event_team_name}"
                
        #         # Crea un titolo per lo score
        #         away_title = f"{time_str} | Formation Change: {event_team_name} | Score: {score_str}"
        #         home_title = f"{time_str} | Formation Change: {event_team_name} | Score: {score_str}"
                
        #         home_plots.append(dash_html.Img(src=formations.plot_formation_snapshot(home_state, home_player_colors, player_data_map, HCOL, home_title), style={'width': '100%', 'height': 'auto', 'margin-bottom': '15px'}))
        #         away_plots.append(dash_html.Img(src=formations.plot_formation_snapshot(away_state, away_player_colors, player_data_map, ACOL, away_title, is_away=True), style={'width': '100%', 'height': 'auto', 'margin-bottom': '15px'}))
            
        #     # Usa la timeline unificata solo per la colonna centrale
        #     central_timeline_events = formations.create_unified_timeline(df_processed, home_id, away_id, player_data_map)
        #     for event in central_timeline_events:
        #         timeline_items.append(dbc.ListGroupItem([dash_html.Strong(f"{event['time_str']} "), event['description_component']], className="bg-transparent text-white border-secondary"))

        #     # --- 3. COSTRUZIONE LAYOUT FINALE ---
        #     # ... (il layout flexbox rimane identico alla mia risposta precedente) ...
        #     final_layout = dash_html.Div([
        #         dash_html.Div([
        #             dash_html.H4(home_name, className="text-center text-white", style={'flex': '0 0 38%'}),
        #             dash_html.H4("Key Events", className="text-center text-white", style={'flex': '0 0 24%'}),
        #             dash_html.H4(away_name, className="text-center text-white", style={'flex': '0 0 38%'}),
        #         ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '1rem'}),
        #         dash_html.Div([
        #             dash_html.Div(home_plots, style={'flex': '0 0 38%', 'paddingRight': '10px'}),
        #             dash_html.Div(dbc.ListGroup(timeline_items, flush=True), style={'flex': '0 0 24%'}),
        #             dash_html.Div(away_plots, style={'flex': '0 0 38%', 'paddingLeft': '10px'}),
        #         ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'flex-start'}),
        #         dash_html.Hr(className="my-4"),
        #         dash_html.H6("Comments for Formation Analysis:", className="mt-3 text-white"),
        #         dcc.Textarea(id="comment-formation", placeholder="Enter your summary analysis here...", style={'width': '100%', 'height': 120, 'backgroundColor': '#495057', 'color': 'white'}),
        #         dbc.Button("Save Comment", id="save-comment-formation", color="info", size="sm", className="mt-2"),
        #         dash_html.Div(id="save-status-formation", className="small d-inline-block ms-2")
        #     ])
            
            return dash_html.Div([
                dash_html.H4("Formation & Shape Analysis", className="text-white mb-3"),
                dbc.Tabs(
                    id="formation-primary-tabs", # Un ID per questo gruppo di tab
                    active_tab="formation_timeline", # La tab predefinita
                    children=[
                        dbc.Tab(label="Formation Timeline", tab_id="formation_timeline"),
                        dbc.Tab(label="Mean Positions", tab_id="mean_positions")
                    ],
                    className="mt-3"
                ),
                # Un contenitore vuoto che verrà riempito dal callback sottostante
                dcc.Loading(
                    type="circle",
                    children=dash_html.Div(id="formation-tab-content")
                )
            ], className="p-3")

        # except Exception as e:
        #     tb_str = traceback.format_exc()
        #     return dbc.Alert(f"Error generating formation analysis: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})
        
        # return dash_html.Div([
        #     dash_html.H4("Formation Chart", style={"color": "white"}, className="mb-3"),
        #     dcc.Loading(type="circle", children=dash_html.Div(id="div-formation-match-content")),
        #     # --- ADDED COMMENT SECTION FOR FORMATION ---
        #     dash_html.Hr(),
        #     dash_html.H6("Comments for Formation:", className="mt-3 text-white"),
        #     dcc.Textarea(
        #         id="comment-formation", # Unique ID
        #         placeholder="Enter your analysis comments for formation...",
        #         style={'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'},
        #         className="mb-2"
        #     ),
        #     dbc.Button("Save Comment", id="save-comment-formation", color="info", size="sm", className="me-2"),
        #     dash_html.Div(id="save-status-formation", className="small d-inline-block")
        #     # -----------------------------------------
        # ], className="p-3")
    
    elif active_tab == "passes":
        passes_content = dash_html.Div([
            dash_html.H4("Passing Analysis", style={"color": "white"}, className="mb-3"),
            dbc.Tabs(
                id="passes-nested-tabs",
                active_tab="pass_network",
                children=[
                    dbc.Tab(label="Pass Network", tab_id="pass_network", children=[
                        dcc.Loading(type="circle", children=dash_html.Div(id="div-pass-network-content")),
                        dash_html.Hr(),
                        dash_html.H6("Comments for Pass Network:", className="mt-3 text-white"),
                        dcc.Textarea(
                            id="comment-pass-network",
                            placeholder="Enter your analysis comments here...",
                            style={'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'},
                            className="mb-2"
                        ),
                        dbc.Button("Save Comment", id="save-comment-pass-network", color="info", size="sm", className="me-2"),
                        dash_html.Div(id="save-status-pass-network", className="small d-inline-block") # For feedback
                    ]),
                    dbc.Tab(label="Progressive Passes", tab_id="progressive_passes", children=[
                        # REMOVE placeholder alert, content will be filled by callback
                        dcc.Loading(type="circle", children=dash_html.Div(id="div-progressive-passes-content")),
                        # ... (comment section for progressive passes) ...
                        dash_html.Hr(),
                        dash_html.H6("Comments for Progressive Passes:", className="mt-3 text-white"),
                        dcc.Textarea(
                            id="comment-progressive-passes",
                            placeholder="Enter comments for Progressive Passes...",
                            style={'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'},
                            className="mb-2"
                        ),
                        dbc.Button("Save Comment", id="save-comment-progressive-passes", color="info", size="sm", className="me-2"),
                        dash_html.Div(id="save-status-progressive-passes", className="small d-inline-block")
                    ]),
                    dbc.Tab(label="Final Third Entries", tab_id="final_third_entries", children=[
                        dash_html.Div([ # Flex Container for plot and comments
                            dash_html.Div( # Plot Area
                                dcc.Loading(type="circle", children=dash_html.Div(id="div-final-third-content")),
                                style={"flex": "1 1 75%", "minHeight": "400px"} # Adjust flex-basis as needed
                            ),
                            # dash_html.Div([ # Comment Area
                            #     dash_html.Hr(),
                            #     dash_html.H6("Comments for Final Third Entries:", className="mt-3 text-white"),
                            #     dcc.Textarea(
                            #         id="comment-final-third",
                            #         placeholder="Enter comments for Final Third Entries...",
                            #         style={'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'},
                            #         className="mb-2"
                            #     ),
                            #     dbc.Button("Save Comment", id="save-comment-final-third", color="info", size="sm", className="me-2"),
                            #     dash_html.Div(id="save-status-final-third", className="small d-inline-block")
                            # ], style={"flex": "0 0 20%", "paddingTop": "20px"}) # Adjust flex-basis
                        ], style={"display": "flex", "flexDirection": "column", "height": "calc(100vh - 250px)"}) # Adjust height
                    ]),
                    dbc.Tab(label="Pass Locations", tab_id="pass_locations", children=[ 
                        dash_html.Div([ # Main container for this tab's content
                            dcc.Loading(type="circle", children=dash_html.Div(id="div-pass-density-content")),
                            dcc.Loading(type="circle", children=dash_html.Div(id="div-pass-heatmap-content")),
                            dash_html.Hr(className="my-4"),
                            dash_html.H6("Comments for Pass Locations:", className="mt-3 text-white"),
                            dcc.Textarea(
                                id="comment-pass-locations", # Un solo ID per i commenti
                                placeholder="Enter your analysis on pass locations...",
                                style={'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'},
                                className="mb-2"
                            ),
                            dbc.Button("Save Comment", id="save-comment-pass-locations", color="info", size="sm"),
                            dash_html.Div(id="save-status-pass-locations", className="small d-inline-block ms-2")
                        ])
                    ]),
                    dbc.Tab(label="Crosses", tab_id="crosses", children=[
                        dcc.Loading(type="circle", children=dash_html.Div(id="crosses-content")),
                        dash_html.Div(id="crosses-content-wrapper")
                    ]),
                ],
                className="mt-3"
            )
        ], className="p-3")
        return passes_content

    elif active_tab == "player_analysis":
        print("--- render_match_tab_content: RENDERING NEW 'player_analysis' PRIMARY TAB STRUCTURE ---")
        return dash_html.Div([
            dash_html.H4("Player Analysis", style={"color": "white"}, className="mb-3"),
            
            # 1. The new PRIMARY tabs
            dbc.Tabs(
                id="player-analysis-primary-tabs",
                active_tab="pa_primary_passing", # Default to passing analysis
                children=[
                    dbc.Tab(label="Passing Analysis", tab_id="pa_primary_passing"),
                    dbc.Tab(label="Shooting Analysis", tab_id="pa_primary_shooting"),
                    dbc.Tab(label="Defending Analysis", tab_id="pa_primary_defending"),
                ],
                className="mt-3"
            ),
            
            # 2. A single content area that will be filled by our new "router" callback
            dcc.Loading(type="circle", children=dash_html.Div(id="player-analysis-primary-tab-content"))
        ], className="p-3")
    
    elif active_tab == "buildup":
        return dash_html.Div([
             dash_html.H4("Buildup Analysis", style={"color": "white"}, className="mb-3"),
             # Primary tabs for Home/Away
             dbc.Tabs(
                 id="buildup-primary-tabs",
                 active_tab="buildup_home",
                 children=[
                     dbc.Tab(label="Home Team Buildups", tab_id="buildup_home"),
                     dbc.Tab(label="Away Team Buildups", tab_id="buildup_away"),
                 ],
                 className="mt-3"
             ),
             # A single content area to be filled by the new callback
             dcc.Loading(
                 type="circle",
                 children=dash_html.Div(id="buildup-tab-content")
             )
        ], className="p-3")
    
    elif active_tab == "defensive-transition":
        return dash_html.Div([
            dash_html.H4("Defensive Transition Analysis", style={"color": "white"}, className="mb-3"),
            dbc.Tabs(
                id="def-transition-primary-tabs",
                active_tab="def_shape",
                children=[
                    dbc.Tab(label="Defensive Block", tab_id="def_shape"), 
                    dbc.Tab(label="Defensive Hull", tab_id="def_hull"), 
                    dbc.Tab(label="Pressing (PPDA)", tab_id="def_ppda"),
                    dbc.Tab(label="Home Defensive Transitions", tab_id="def_transitions_home"),
                    dbc.Tab(label="Away Defensive Transitions", tab_id="def_transitions_away"),
                 ],
                 className="mt-3"
            ),
            dcc.Loading(
                 type="circle",
                 children=dash_html.Div(id="def-transition-tab-content")
             )
        ], className="p-3")
    
    elif active_tab == "offensive-transition":
        return dash_html.Div([
            dash_html.H4("Offensive Transition Analysis", style={"color": "white"}, className="mb-3"),
            dbc.Tabs(
                id="off-transition-primary-tabs",
                active_tab="off_transitions_home",
                children=[
                     dbc.Tab(label="Home Offensive Transitions", tab_id="off_transitions_home"),
                     dbc.Tab(label="Away Offensive Transitions", tab_id="off_transitions_away"),
                 ],
                 className="mt-3"
            ),
            dcc.Loading(
                 type="circle",
                 children=dash_html.Div(id="off-transition-tab-content")
             )
        ], className="p-3")
    
    elif active_tab == "set-piece":
        return dash_html.Div([
            dash_html.H4("Set Piece Analysis", style={"color": "white"}, className="mb-3"),
            dbc.Tabs(
                id="set-piece-primary-tabs",
                active_tab="set_piece_home",
                children=[
                     dbc.Tab(label="Home Set Pieces", tab_id="set_piece_home"),
                     dbc.Tab(label="Away Set Pieces", tab_id="set_piece_away"),
                 ],
                 className="mt-3"
            ),
            dcc.Loading(
                 type="circle",
                 children=dash_html.Div(id="set-piece-tab-content")
             )
        ], className="p-3")


### Formaion Tab Content Callback
@app.callback(
    Output("formation-tab-content", "children"),
    Input("formation-primary-tabs", "active_tab"),
    State("store-df-match", "data")
)
def render_formation_content(active_tab, stored_data_json):
    if not stored_data_json:
        return dbc.Alert("Match data loading...", color="info")

    try:
        # Questi dati sono comuni a tutte le sotto-tab
        df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        match_info = json.loads(stored_data_json['match_info'])
        HTEAM_NAME = match_info.get('hteamName')
        ATEAM_NAME = match_info.get('ateamName')

        # --- CASO 1: TIMELINE DELLE FORMAZIONI (la tua logica esistente) ---
        if active_tab == 'formation_timeline':
            if not stored_data_json:
                return dbc.Alert("Match data loading for formation...", color="info")
            try:
                df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
                df_processed = df_processed.reset_index().rename(columns={'index': 'event_sequence_index'})
                
                match_info = json.loads(stored_data_json['match_info'])

                # --- 1. SETUP INIZIALE (ROBUSTO) ---
                
                # Mappa dati giocatori
                player_data_map = {}
                if not df_processed.empty:
                    df_players_unique = df_processed.dropna(subset=['playerId', 'Mapped Jersey Number']).drop_duplicates(subset=['playerId'])
                    for _, player in df_players_unique.iterrows():
                        player_id = player['playerId']
                        jersey_num_raw = player['Mapped Jersey Number']
                        try:
                            jersey_num = int(jersey_num_raw)
                        except (ValueError, TypeError):
                            jersey_num = '?'
                        player_data_map[player_id] = {'name': player.get('playerName', 'N/A'), 'jersey': str(jersey_num)}

                # Recupero sicuro degli eventi di formazione iniziale
                start_events = df_processed[df_processed['typeId'] == 34].sort_values('eventId')
                if len(start_events) < 2:
                    return dbc.Alert("Error: Could not find starting formation events for both teams.", color="danger")
                
                home_team_name_from_info = match_info.get('hteamName')
                event1, event2 = start_events.iloc[0], start_events.iloc[1]
                
                if home_team_name_from_info and event1['team_name'] == home_team_name_from_info:
                    home_start_event, away_start_event = event1, event2
                elif home_team_name_from_info and event2['team_name'] == home_team_name_from_info:
                    home_start_event, away_start_event = event2, event1
                else:
                    home_start_event, away_start_event = event1, event2
                
                home_id, away_id = home_start_event['contestantId'], away_start_event['contestantId']
                home_name, away_name = home_start_event['team_name'], away_start_event['team_name']
                
                home_state = {'formation_id': int(home_start_event['Team formation']), 'players': formations._extract_player_positions(home_start_event)}
                away_state = {'formation_id': int(away_start_event['Team formation']), 'players': formations._extract_player_positions(away_start_event)}
                
                # --- 2. LOGICA DI COSTRUZIONE SINCRONA (AGGIORNATA) ---
                home_plots, timeline_items, away_plots = [], [], []

                # Stato iniziale (t=0)
                title = f"0' | Starting XI"
                home_plots.append(dash_html.Img(src=formations.plot_formation_snapshot(home_state, {}, player_data_map, HCOL, title), style={'width': '100%', 'height': 'auto', 'margin-bottom': '15px'}))
                away_plots.append(dash_html.Img(src=formations.plot_formation_snapshot(away_state, {}, player_data_map, ACOL, title, is_away=True), style={'width': '100%', 'height': 'auto', 'margin-bottom': '15px'}))
                timeline_items.append(dbc.ListGroupItem([dash_html.H5("Match Timeline", className="text-white"), dash_html.P("0' - Kick Off")], className="bg-dark text-white text-center"))
                
                
                
                # Prendi solo gli eventi di cambio formazione
                formation_change_events = df_processed[df_processed['typeId'] == 40].sort_values('event_sequence_index')

                for _, fc_event in formation_change_events.iterrows():
                    time_str = f"{fc_event['timeMin']}'"
                    previous_home_state, previous_away_state = home_state.copy(), away_state.copy()
                    
                    # Aggiorna lo stato della squadra che ha cambiato formazione
                    if fc_event['contestantId'] == home_id:
                        home_state = {'formation_id': int(fc_event['Team formation']), 'players': formations._extract_player_positions(fc_event)}
                    else:
                        away_state = {'formation_id': int(fc_event['Team formation']), 'players': formations._extract_player_positions(fc_event)}
                    
                    # Calcola lo score PRIMA di questo evento, per riflettere lo stato al momento del cambio
                    goals_before = df_processed[(df_processed['typeId'] == 16) & (df_processed['event_sequence_index'] < fc_event['event_sequence_index'])]
                    home_score = (goals_before['contestantId'] == home_id).sum()
                    away_score = (goals_before['contestantId'] == away_id).sum()
                    score_str = f"{home_score} - {away_score}"

                    # Determina i colori per l'highlight
                    home_player_colors = {pid: '#00FFFF' for pid, pos in home_state['players'].items() if previous_home_state['players'].get(pid) != pos}
                    away_player_colors = {pid: '#00FFFF' for pid, pos in away_state['players'].items() if previous_away_state['players'].get(pid) != pos}

                    # Costruisci i titoli per i plot
                    event_team_name = home_name if fc_event['contestantId'] == home_id else away_name
                    title = f"{time_str} | Formation Change: {event_team_name}"
                    
                    # Crea un titolo per lo score
                    away_title = f"{time_str} | Formation Change: {event_team_name} | Score: {score_str}"
                    home_title = f"{time_str} | Formation Change: {event_team_name} | Score: {score_str}"
                    
                    home_plots.append(dash_html.Img(src=formations.plot_formation_snapshot(home_state, home_player_colors, player_data_map, HCOL, home_title), style={'width': '100%', 'height': 'auto', 'margin-bottom': '15px'}))
                    away_plots.append(dash_html.Img(src=formations.plot_formation_snapshot(away_state, away_player_colors, player_data_map, ACOL, away_title, is_away=True), style={'width': '100%', 'height': 'auto', 'margin-bottom': '15px'}))
                
                # Usa la timeline unificata solo per la colonna centrale
                central_timeline_events = formations.create_unified_timeline(df_processed, home_id, away_id, player_data_map)
                for event in central_timeline_events:
                    timeline_items.append(dbc.ListGroupItem([dash_html.Strong(f"{event['time_str']} "), event['description_component']], className="bg-transparent text-white border-secondary"))

                # --- 3. COSTRUZIONE LAYOUT FINALE ---
                # ... (il layout flexbox rimane identico alla mia risposta precedente) ...
                final_layout = dash_html.Div([
                    dash_html.Div([
                        dash_html.H4(home_name, className="text-center text-white", style={'flex': '0 0 38%'}),
                        dash_html.H4("Key Events", className="text-center text-white", style={'flex': '0 0 24%'}),
                        dash_html.H4(away_name, className="text-center text-white", style={'flex': '0 0 38%'}),
                    ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin-bottom': '1rem'}),
                    dash_html.Div([
                        dash_html.Div(home_plots, style={'flex': '0 0 38%', 'paddingRight': '10px'}),
                        dash_html.Div(dbc.ListGroup(timeline_items, flush=True), style={'flex': '0 0 24%'}),
                        dash_html.Div(away_plots, style={'flex': '0 0 38%', 'paddingLeft': '10px'}),
                    ], style={'display': 'flex', 'flex-direction': 'row', 'align-items': 'flex-start'}),
                    dash_html.Hr(className="my-4"),
                    dash_html.H6("Comments for Formation Analysis:", className="mt-3 text-white"),
                    dcc.Textarea(id="comment-formation", placeholder="Enter your summary analysis here...", style={'width': '100%', 'height': 120, 'backgroundColor': '#495057', 'color': 'white'}),
                    dbc.Button("Save Comment", id="save-comment-formation", color="info", size="sm", className="mt-2"),
                    dash_html.Div(id="save-status-formation", className="small d-inline-block ms-2")
                ])
                return final_layout
            except Exception as e:
                tb_str = traceback.format_exc()
                return dbc.Alert(f"Error generating formation analysis: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})

        # --- CASO 2: POSIZIONI MEDIE (la nuova logica) ---
        elif active_tab == 'mean_positions':
            # Prepara i dati usando la nuova funzione
            df_home_touches, df_home_agg = player_metrics.get_mean_positions_data(df_processed, HTEAM_NAME)
            df_away_touches, df_away_agg = player_metrics.get_mean_positions_data(df_processed, ATEAM_NAME)

            # Crea i grafici con la nuova funzione di plot
            fig_home = formation_plotly.plot_mean_positions_plotly(df_home_touches, df_home_agg, HCOL, is_away=False)
            fig_away = formation_plotly.plot_mean_positions_plotly(df_away_touches, df_away_agg, ACOL, is_away=True)
            
            return dbc.Row([
                dbc.Col([
                    dash_html.H5(f"{HTEAM_NAME} - Mean Positions", className="text-center text-white mt-3"),
                    dcc.Graph(figure=fig_home, config={'displayModeBar': False})
                ], md=6),
                dbc.Col([
                    dash_html.H5(f"{ATEAM_NAME} - Mean Positions", className="text-center text-white mt-3"),
                    dcc.Graph(figure=fig_away, config={'displayModeBar': False})
                ], md=6)
            ], className="mt-4")
        
        # # --- CASO 3: BLOCCO DIFENSIVO ---
        # elif active_tab == 'defensive_shape':
        #     df_home_def, df_home_agg = defensive_metrics.get_defensive_block_data(df_processed, HTEAM_NAME)
        #     df_away_def, df_away_agg = defensive_metrics.get_defensive_block_data(df_processed, ATEAM_NAME)
            
        #     fig_home = defensive_transitions_plotly.plot_defensive_block_plotly(df_home_def, df_home_agg, HCOL, is_away=False)
        #     fig_away = defensive_transitions_plotly.plot_defensive_block_plotly(df_away_def, df_away_agg, ACOL, is_away=True)
            
        #     return dbc.Row([
        #         dbc.Col([dash_html.H5(f"{HTEAM_NAME} - Defensive Block", className="text-center text-white mt-3"), dcc.Graph(figure=fig_home)], md=6),
        #         dbc.Col([dash_html.H5(f"{ATEAM_NAME} - Defensive Block", className="text-center text-white mt-3"), dcc.Graph(figure=fig_away)], md=6)
        #     ])

        # # --- CASO 4: HULL DIFENSIVO ---
        # elif active_tab == 'defensive_hull':
        #     _, df_home_agg = defensive_metrics.get_defensive_block_data(df_processed, HTEAM_NAME)
        #     _, df_away_agg = defensive_metrics.get_defensive_block_data(df_processed, ATEAM_NAME)

        #     fig_home_hull = defensive_transitions_plotly.plot_defensive_hull_plotly(df_home_agg, HCOL, is_away=False)
        #     fig_away_hull = defensive_transitions_plotly.plot_defensive_hull_plotly(df_away_agg, ACOL, is_away=True)
            
        #     return dbc.Row([
        #         dbc.Col([dash_html.H5(f"{HTEAM_NAME} - Defensive Shape (Hull)", className="text-center text-white mt-3"), dcc.Graph(figure=fig_home_hull)], md=6),
        #         dbc.Col([dash_html.H5(f"{ATEAM_NAME} - Defensive Shape (Hull)", className="text-center text-white mt-3"), dcc.Graph(figure=fig_away_hull)], md=6)
        #     ])

    except Exception as e:
        return dbc.Alert(f"Error rendering formation/shape content: {traceback.format_exc()}", color="danger", style={"whiteSpace": "pre-wrap"})

    return dash_html.P("Select a sub-tab.")



# --- CALLBACKS FOR FORMATION COMMENTS ---
@app.callback(
    Output("store-comment-formation", "data"),
    Output("save-status-formation", "children"),
    Input("save-comment-formation", "n_clicks"),
    State("comment-formation", "value"),
    State("url-match-page", "pathname"),
    State("store-comment-formation", "data"),
    prevent_initial_call=True
)
def save_formation_comment(n_clicks, comment_value, pathname, existing_data):
    if not n_clicks:
        return no_update, ""
    key = get_comment_key(pathname, "formation") # Use "formation" as plot_identifier
    if not key:
        store_output = existing_data if existing_data is not None else no_update
        return store_output, dbc.Alert("Error: Invalid context for saving comment.", color="danger", duration=3000)
    
    if existing_data is None:
        existing_data = {}
    existing_data[key] = comment_value
    return existing_data, dbc.Alert("Comment saved!", color="success", duration=2000, className="ms-2")

@app.callback(
    Output("comment-formation", "value"),
    Input("store-comment-formation", "data"),
    Input("url-match-page", "pathname")
)
def load_formation_comment(stored_data, pathname):
    key = get_comment_key(pathname, "formation") # Use "formation"
    if not key or stored_data is None:
        return ""
    return stored_data.get(key, "")
# ------------------------------------


# --- Callback for "Passes" Tab (Pass Network) ---
def show_pass_network_graph(stored_data_json): # Note: this is now a helper, not a direct callback outputting to a tab's main div
    if not stored_data_json:
        return dash_html.P("⚠ No data in store for pass network.", style={"color": "orange"})
    try:
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')
        if not df_json_str or not match_info_json_str:
            return dash_html.P("⚠ DataFrame or match_info missing in stored data.", style={"color": "orange"})
        df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)
        if df_processed.empty: return dash_html.P("⚠ DataFrame is empty.", style={"color": "orange"})

        HTEAM_NAME = match_info.get('hteamName', 'Home Team Fallback') 
        ATEAM_NAME = match_info.get('ateamName', 'Away Team Fallback')
        HTEAM_COLOR = getattr(config, 'DEFAULT_HCOL', "#FF0000")
        ATEAM_COLOR = getattr(config, 'DEFAULT_ACOL', "#0000FF")
        FIG_BG_COLOR = getattr(config, 'BG_COLOR', 'white')
        TEXT_COLOR = getattr(config, 'LINE_COLOR', 'black')

        passes_df = pass_processing.get_passes_df(df_processed.copy())
        sub_list = pass_processing.get_sub_list(df_processed.copy())
        if passes_df.empty: return dash_html.P("⚠ Could not process passes (passes_df empty).", style={"color": "orange"})
        
        if 'outcome' in passes_df.columns: successful_passes = passes_df[passes_df['outcome'] == 'Successful'].copy()
        elif 'successful' in passes_df.columns and passes_df['successful'].dtype == 'bool': successful_passes = passes_df[passes_df['successful'] == True].copy()
        else: successful_passes = passes_df.copy() # Fallback
        if successful_passes.empty: return dash_html.P("⚠ No successful passes.", style={"color": "orange"})

        home_passes_between, home_avg_locs = pass_metrics.calculate_pass_network_data(successful_passes, HTEAM_NAME)
        away_passes_between, away_avg_locs = pass_metrics.calculate_pass_network_data(successful_passes, ATEAM_NAME)

        fig_network, axs_network = plt.subplots(1, 2, figsize=(25, 10.5), facecolor=FIG_BG_COLOR)
        # fig_network.suptitle(f'{HTEAM_NAME} vs {ATEAM_NAME} - Passing Networks', fontsize=20, fontweight='bold', color=TEXT_COLOR) # Title handled by render_match_tab_content

        plot_home_network = not (home_passes_between.empty if isinstance(home_passes_between, pd.DataFrame) else not bool(home_passes_between)) and \
                            not (home_avg_locs.empty if isinstance(home_avg_locs, pd.DataFrame) else not bool(home_avg_locs))
        plot_away_network = not (away_passes_between.empty if isinstance(away_passes_between, pd.DataFrame) else not bool(away_passes_between)) and \
                            not (away_avg_locs.empty if isinstance(away_avg_locs, pd.DataFrame) else not bool(away_avg_locs))

        if plot_home_network: pitch_plots.plot_pass_network(axs_network[0], home_passes_between, home_avg_locs, HTEAM_COLOR, HTEAM_NAME, sub_list, False)
        else: axs_network[0].text(0.5, 0.5, f"{HTEAM_NAME}\nNetwork N/A", ha='center', va='center', color=TEXT_COLOR); axs_network[0].set_facecolor(FIG_BG_COLOR); axs_network[0].axis('off')
        if plot_away_network: pitch_plots.plot_pass_network(axs_network[1], away_passes_between, away_avg_locs, ATEAM_COLOR, ATEAM_NAME, sub_list, True)
        else: axs_network[1].text(0.5, 0.5, f"{ATEAM_NAME}\nNetwork N/A", ha='center', va='center', color=TEXT_COLOR); axs_network[1].set_facecolor(FIG_BG_COLOR); axs_network[1].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust if suptitle removed
        buf = io.BytesIO(); plt.savefig(buf, format="png", bbox_inches='tight', facecolor=fig_network.get_facecolor())
        buf.seek(0); encoded_img = base64.b64encode(buf.read()).decode('ascii'); img_src = "data:image/png;base64," + encoded_img
        plt.close(fig_network)
        return dash_html.Img(src=img_src, style={"width": "100%", "maxWidth": "1500px", "display":"block", "margin":"auto"})
    except KeyError as ke:
        return dash_html.P(f"❌ Error generating pass network (KeyError): {ke}", style={"color": "red"})
    except Exception as e:
        return dash_html.P(f"❌ Error generating pass network: {e}", style={"color": "red"})

def show_pass_network_graph_plotly(stored_data_json):
    if not stored_data_json:
        return dash_html.P("No data in store for pass network.", style={"color": "orange"})
    try:
        df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        match_info = json.loads(stored_data_json['match_info'])

        HTEAM_NAME = match_info.get('hteamName', 'Home')
        ATEAM_NAME = match_info.get('ateamName', 'Away')

        # Dati sui passaggi (la tua logica esistente va bene)
        passes_df = pass_processing.get_passes_df(df_processed)
        successful_passes = passes_df[passes_df['outcome'] == 'Successful']
        
        if successful_passes.empty:
            return dbc.Alert("No successful passes in the match.", color="warning")

        # **Ottieni la lista dei subentrati per ogni squadra**
        home_subs = pass_processing.get_sub_list(df_processed[df_processed['team_name'] == HTEAM_NAME])
        away_subs = pass_processing.get_sub_list(df_processed[df_processed['team_name'] == ATEAM_NAME])

        # --- Dati per Home Team ---
        home_passes_between, home_avg_locs = pass_metrics.calculate_pass_network_data(successful_passes, HTEAM_NAME)
        fig_home = pass_plotly.plot_pass_network_plotly(home_passes_between, home_avg_locs, HTEAM_NAME, HCOL, home_subs, is_away=False)
        
        # **MODIFICA TABELLA: Aggiungi numeri di maglia**
        home_jersey_map = home_avg_locs.set_index('playerName')['jersey_number'].to_dict()
        home_table_df = home_passes_between[['player1', 'player2', 'pass_count']].copy()
        home_table_df['player1'] = home_table_df['player1'].apply(lambda name: f"#{int(home_jersey_map.get(name, '?')) if pd.notna(home_jersey_map.get(name)) else '?'} - {name}")
        home_table_df['player2'] = home_table_df['player2'].apply(lambda name: f"#{int(home_jersey_map.get(name, '?')) if pd.notna(home_jersey_map.get(name)) else '?'} - {name}")
        home_table = dbc.Table.from_dataframe(home_table_df.sort_values('pass_count', ascending=False).head(10), striped=True, bordered=True, hover=True, color="dark")
        
        # --- Dati per Away Team ---
        away_passes_between, away_avg_locs = pass_metrics.calculate_pass_network_data(successful_passes, ATEAM_NAME)
        fig_away = pass_plotly.plot_pass_network_plotly(away_passes_between, away_avg_locs, ATEAM_NAME, ACOL, away_subs, is_away=True)
        
        # **MODIFICA TABELLA: Aggiungi numeri di maglia**
        away_jersey_map = away_avg_locs.set_index('playerName')['jersey_number'].to_dict()
        away_table_df = away_passes_between[['player1', 'player2', 'pass_count']].copy()
        away_table_df['player1'] = away_table_df['player1'].apply(lambda name: f"#{int(away_jersey_map.get(name, '?')) if pd.notna(away_jersey_map.get(name)) else '?'} - {name}")
        away_table_df['player2'] = away_table_df['player2'].apply(lambda name: f"#{int(away_jersey_map.get(name, '?')) if pd.notna(away_jersey_map.get(name)) else '?'} - {name}")
        away_table = dbc.Table.from_dataframe(away_table_df.sort_values('pass_count', ascending=False).head(10), striped=True, bordered=True, hover=True, color="dark")

        # Layout a due colonne per mostrare i grafici affiancati
        return dash_html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_home), md=6),
                dbc.Col(dcc.Graph(figure=fig_away), md=6),
            ]),
            dash_html.Hr(),
            dbc.Row([
                dbc.Col([
                    dbc.Button("Toggle Top Home Connections", id="btn-collapse-home", className="mb-3"),
                    dbc.Collapse(home_table, id="collapse-home")
                ], md=6),
                dbc.Col([
                    dbc.Button("Toggle Top Away Connections", id="btn-collapse-away", className="mb-3"),
                    dbc.Collapse(away_table, id="collapse-away")
                ], md=6)
            ])
        ])

    except Exception as e:
        tb_str = traceback.format_exc()
        return dbc.Alert(f"Error generating interactive pass network: {e}\n{tb_str}", color="danger", style={"whiteSpace":"pre-wrap"})

# Aggiorna il callback per chiamare la nuova funzione
@app.callback(
    Output("div-pass-network-content", "children"),
    Input("store-df-match", "data"),
)
def show_pass_network_graph_content_callback(stored_data_json):
    print("--- show_pass_network_graph_content_callback (Plotly) TRIGGERED ---")
    return show_pass_network_graph_plotly(stored_data_json)
    
# @app.callback(
#     Output("div-pass-network-content", "children"), # <--- UPDATED ID
#     Input("store-df-match", "data"),
#     # Optionally, trigger only if "pass_network" nested tab is active
#     # Input("passes-nested-tabs", "active_tab") # Add this if you want to optimize
# )
# # def show_pass_network_graph_content_callback(stored_data_json, active_nested_tab=None): # Add active_nested_tab
# def show_pass_network_graph_content_callback(stored_data_json):
#     print(f"--- show_pass_network_graph_content_callback TRIGGERED ---")
#     return show_pass_network_graph(stored_data_json)

@app.callback(
    Output("collapse-home", "is_open"),
    [Input("btn-collapse-home", "n_clicks")],
    [State("collapse-home", "is_open")],
    prevent_initial_call=True,
)
def toggle_home_connections_table(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-away", "is_open"),
    [Input("btn-collapse-away", "n_clicks")],
    [State("collapse-away", "is_open")],
    prevent_initial_call=True,
)
def toggle_away_connections_table(n, is_open):
    if n:
        return not is_open
    return is_open

# --- CALLBACK TO SAVE COMMENT FOR PASS NETWORK ---
def get_comment_key(pathname, plot_identifier):
    if pathname and pathname.startswith("/match/"):
        path_parts = pathname.split("/")
        # Handle potential trailing slash in pathname
        match_id = path_parts[-1] if path_parts[-1] else path_parts[-2]
        if match_id: # Ensure match_id is not empty
            return f"comments_{match_id}_{plot_identifier}"
    print(f"Warning: Could not generate comment key for pathname '{pathname}' and plot '{plot_identifier}'")
    return None # Return None if key cannot be formed

@app.callback(
    Output("store-comment-pass-network", "data"),
    Output("save-status-pass-network", "children"),
    Input("save-comment-pass-network", "n_clicks"),
    State("comment-pass-network", "value"),
    State("url-match-page", "pathname"), # To get match_id for unique storage key
    State("store-comment-pass-network", "data"), # Existing comments
    prevent_initial_call=True
)
def save_pass_network_comment(n_clicks, comment_value, pathname, existing_comments_data):
    if not n_clicks:
        return no_update, ""

    comment_storage_key = get_comment_key(pathname, "pass_network")
    if not comment_storage_key:
        # For the store output, return current data or no_update if no current data
        store_output_val = existing_comments_data if existing_comments_data is not None else no_update # Corrected this logic
        return store_output_val, dbc.Alert("Error: Could not determine match context for comment key.", color="danger", duration=3000)

    # Store data as a dictionary where keys are our unique comment_storage_key
    if existing_comments_data is None:
        existing_comments_data = {}
    
    existing_comments_data[comment_storage_key] = comment_value
    
    # print(f"Saving comment for {comment_storage_key}: {comment_value}")
    return existing_comments_data, dbc.Alert("Comment saved!", color="success", duration=2000, className="ms-2")

# --- CALLBACK TO LOAD COMMENT FOR PASS NETWORK ---
@app.callback(
    Output("comment-pass-network", "value"),
    Input("store-comment-pass-network", "data"), # Trigger when stored comments change (e.g., on load)
    Input("url-match-page", "pathname")      # Trigger when page/match changes
)
def load_pass_network_comment(stored_comments, pathname):
    comment_storage_key = get_comment_key(pathname, "pass_network")
    if not comment_storage_key or stored_comments is None:
        return "" # No comment to load or no key

    return stored_comments.get(comment_storage_key, "") # Get comment for current match_id/plot

def generate_progressive_passes_plot(stored_data_json):
    print("--- Helper generate_progressive_passes_plot EXECUTING ---")
    if not stored_data_json:
        print("Helper generate_progressive_passes_plot: No stored_data_json.")
        return dash_html.P("⚠ No data in store for Progressive Passes plot.", style={"color": "orange"})
    try:
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')
        if not df_json_str or not match_info_json_str:
            print("Helper generate_progressive_passes_plot: DataFrame or match_info missing.")
            return dash_html.P("⚠ DataFrame or match_info missing.", style={"color": "orange"})

        df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)
        if df_processed.empty:
            print("Helper generate_progressive_passes_plot: DataFrame is empty.")
            return dash_html.P("⚠ DataFrame is empty for Progressive Passes.", style={"color": "orange"})

        HTEAM_NAME = match_info.get('hteamName', 'Home')
        ATEAM_NAME = match_info.get('ateamName', 'Away')
        HTEAM_COLOR = getattr(config, 'DEFAULT_HCOL', "#FF0000")
        ATEAM_COLOR = getattr(config, 'DEFAULT_ACOL', "#0000FF")
        FIG_BG_COLOR = getattr(config, 'BG_COLOR', 'white')
        TEXT_COLOR = getattr(config, 'LINE_COLOR', 'black') # TEXT_COLOR not explicitly used in plot_progressive_passes, but good to have

        # --- Data Preparation for Progressive Passes ---
        # Use your defined exclusions from main_analyze_match.py or a config file
        prog_pass_exclusions = ['cross', 'Launch', 'ThrowIn'] # Example from your main script
        # If your config module has a better list, use that:
        # prog_pass_exclusions = getattr(config, 'PROGRESSIVE_PASS_EXCLUSIONS', ['cross', 'Launch', 'ThrowIn'])

        df_prog_passes, zone_counts = pass_metrics.analyze_progressive_passes(
            df_processed.copy(),
            exclude_qualifiers=prog_pass_exclusions
        )
        # analyze_progressive_passes now returns the full df_prog_passes and overall zone_counts.
        # We need to split them per team for plotting.

        home_prog_passes = pd.DataFrame()
        away_prog_passes = pd.DataFrame()
        home_prog_zone_stats = {'total': 0, 'left': 0, 'mid': 0, 'right': 0}
        away_prog_zone_stats = {'total': 0, 'left': 0, 'mid': 0, 'right': 0}

        if df_prog_passes is not None and not df_prog_passes.empty:
            home_prog_passes_all_zones = df_prog_passes[df_prog_passes['team_name'] == HTEAM_NAME].copy()
            away_prog_passes_all_zones = df_prog_passes[df_prog_passes['team_name'] == ATEAM_NAME].copy()

            # The zone_counts from analyze_progressive_passes is overall.
            # We need to recalculate per team if plot_progressive_passes expects per-team zone counts.
            # OR, adapt plot_progressive_passes to take the full df_prog_passes and filter internally.
            # Your plot_progressive_passes takes df_prog_passes_TEAM and zone_counts_TEAM.
            # So, we need to get per-team zone_counts. The easiest way is if analyze_progressive_passes
            # returned per-team stats, or we recalculate them here based on the filtered per-team prog passes.

            # Let's assume for now your plot_progressive_passes can work with the subset of passes
            # and we derive zone_counts for that subset here.
            # OR, if analyze_progressive_passes returns counts for EACH team in zone_counts, that's better.
            # The provided analyze_progressive_passes seems to return overall zone_counts of *all* prog passes.

            # For now, let's recalculate zone stats per team from the filtered df_prog_passes
            def calculate_team_prog_zone_stats_inline(df_team_prog_passes):
                if df_team_prog_passes.empty: return {'total': 0, 'left': 0, 'mid': 0, 'right': 0}
                y_start = df_team_prog_passes['y'].fillna(50)
                total = len(df_team_prog_passes)
                left_count = (y_start >= 66.67).sum() # Passes starting from team's attacking left
                mid_count = ((y_start >= 33.33) & (y_start < 66.67)).sum()
                right_count = (y_start < 33.33).sum() # Passes starting from team's attacking right
                return {'total': total, 'left': left_count, 'mid': mid_count, 'right': right_count}

            home_prog_passes = home_prog_passes_all_zones # Use the already filtered df
            home_prog_zone_stats = calculate_team_prog_zone_stats_inline(home_prog_passes)

            away_prog_passes = away_prog_passes_all_zones # Use the already filtered df
            away_prog_zone_stats = calculate_team_prog_zone_stats_inline(away_prog_passes)
        else:
            print("Helper generate_progressive_passes_plot: No progressive passes found after analysis.")


        # --- Plotting ---
        fig_prog, axs_prog = plt.subplots(1, 2, figsize=(20, 8), facecolor=FIG_BG_COLOR)

        # Call your existing plotting function from src.visualization.pitch_plots
        # Ensure the module is correctly imported as `pitch_plots`
        if not home_prog_passes.empty or home_prog_zone_stats.get('total',0) > 0 : # Check if there's anything to plot
            pitch_plots.plot_progressive_passes(axs_prog[0], home_prog_passes, home_prog_zone_stats, HTEAM_NAME, HTEAM_COLOR, False, prog_pass_exclusions)
        else:
            axs_prog[0].text(0.5,0.5, f"{HTEAM_NAME}\nNo Progressive Passes", ha='center', va='center', color=TEXT_COLOR if TEXT_COLOR else 'black')
            axs_prog[0].set_facecolor(FIG_BG_COLOR); axs_prog[0].axis('off')

        if not away_prog_passes.empty or away_prog_zone_stats.get('total',0) > 0:
            pitch_plots.plot_progressive_passes(axs_prog[1], away_prog_passes, away_prog_zone_stats, ATEAM_NAME, ATEAM_COLOR, True, prog_pass_exclusions)
        else:
            axs_prog[1].text(0.5,0.5, f"{ATEAM_NAME}\nNo Progressive Passes", ha='center', va='center', color=TEXT_COLOR if TEXT_COLOR else 'black')
            axs_prog[1].set_facecolor(FIG_BG_COLOR); axs_prog[1].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjusted rect based on your plot function
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight', facecolor=fig_prog.get_facecolor())
        buf.seek(0)
        encoded_img = base64.b64encode(buf.read()).decode('ascii')
        img_src = f"data:image/png;base64,{encoded_img}"
        plt.close(fig_prog)
        
        print("Helper generate_progressive_passes_plot: Successfully created Img.")
        return dash_html.Img(src=img_src, style={"width": "100%", "maxWidth": "1200px", "display":"block", "margin":"auto"})

    except KeyError as ke:
        print(f"Helper generate_progressive_passes_plot: KeyError: {ke}")
        return dash_html.P(f"❌ Error (KeyError) generating Progressive Passes plot: {ke}", style={"color": "red"})
    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Helper generate_progressive_passes_plot: Exception: {e}\n{tb_str}")
        return dash_html.P(f"❌ Error generating Progressive Passes plot: {e}", style={"color": "red"})


# --- CALLBACK FOR PROGRESSIVE PASSES CONTENT ---
# @app.callback(
#     Output("div-progressive-passes-content", "children"),
#     Input("store-df-match", "data"),
#     Input("passes-nested-tabs", "active_tab") # Listen to which nested tab is active
# )
# def show_progressive_passes_content_callback(stored_data_json, active_nested_tab):
#     print(f"--- show_progressive_passes_content_callback TRIGGERED (Active Nested Tab: {active_nested_tab}) ---")
#     if active_nested_tab == "progressive_passes":
#         if not stored_data_json:
#             return dash_html.P("Waiting for match data for progressive passes...", style={"color": "orange"})
#         # Call the helper function to generate the plot
#         return generate_progressive_passes_plot(stored_data_json)
#     return no_update # Or an empty div if you want to clear content when tab is not active

@app.callback(
    Output("div-progressive-passes-content", "children"),
    Input("store-df-match", "data"),
    Input("passes-nested-tabs", "active_tab")
)
def show_progressive_passes_content_callback(stored_data_json, active_nested_tab):
    if active_nested_tab != "progressive_passes" or not stored_data_json:
        return no_update

    try:
        df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        match_info = json.loads(stored_data_json['match_info'])

        HTEAM_NAME = match_info.get('hteamName', 'Home')
        ATEAM_NAME = match_info.get('ateamName', 'Away')
        
        # Usa la funzione di pass_processing per ottenere i passaggi con i flag corretti
        all_passes = pass_processing.get_passes_df(df_processed)
        prog_passes = all_passes[all_passes['is_progressive'] == True]

        if prog_passes.empty:
            return dbc.Alert("No progressive passes found in the match.", color="warning")

        # --- Funzione helper interna per non duplicare il codice ---
        def create_prog_pass_layout_for_team(team_name, team_color, is_away):
            team_passes = prog_passes[prog_passes['team_name'] == team_name]
            
            fig = pass_plotly.plot_progressive_passes_plotly(team_passes, team_name, team_color, is_away)
            graph_component = dcc.Graph(figure=fig, config={'displayModeBar': False})
            
            table_content = dbc.Alert("No data", color="secondary", className="mt-4")
            if not team_passes.empty:
                # 1. Conta i passaggi per giocatore
                top_passers_series = team_passes['playerName'].value_counts().nlargest(5)
                top_passers_df = top_passers_series.reset_index()
                top_passers_df.columns = ['Player', 'Progressive Passes']

                # **2. Crea la mappa Nome -> Numero Maglia**
                player_jersey_map = team_passes.drop_duplicates('playerName').set_index('playerName')['Mapped Jersey Number']

                # **3. Funzione per formattare il nome**
                def format_player_name_with_jersey(player_name):
                    jersey_raw = player_jersey_map.get(player_name)
                    try:
                        jersey = str(int(jersey_raw))
                    except (ValueError, TypeError):
                        jersey = '?'
                    return f"#{jersey} - {player_name}"

                # **4. Applica la formattazione**
                top_passers_df['Player'] = top_passers_df['Player'].apply(format_player_name_with_jersey)
                
                table_component = dbc.Table.from_dataframe(
                    top_passers_df, 
                    striped=True, bordered=True, hover=True, color="dark"
                    
                )
                table_content = dash_html.Div([
                    dash_html.H6("Top Progressive Passers", className="text-white text-center mt-4"),
                    table_component
                ])
            
            return dbc.Row([
                dbc.Col(graph_component, md=9),
                dbc.Col(table_content, md=3, className="align-self-center")
            ], className="mb-4", align="center")

        # --- Crea i layout per entrambe le squadre ---
        home_layout = create_prog_pass_layout_for_team(HTEAM_NAME, HCOL, is_away=False)
        away_layout = create_prog_pass_layout_for_team(ATEAM_NAME, ACOL, is_away=True)

        return dash_html.Div([
            home_layout,
            dash_html.Hr(),
            away_layout
        ])

    except Exception as e:
        tb_str = traceback.format_exc()
        return dbc.Alert(f"Error generating progressive passes plot: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})

# Example for progressive passes save:
@app.callback(
    Output("store-comment-progressive-passes", "data"),
    Output("save-status-progressive-passes", "children"),
    Input("save-comment-progressive-passes", "n_clicks"),
    State("comment-progressive-passes", "value"),
    State("url-match-page", "pathname"),
    State("store-comment-progressive-passes", "data"),
    prevent_initial_call=True
)
def save_progressive_passes_comment(n_clicks, comment_value, pathname, existing_data):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pathname, "progressive_passes")
    if not key: return no_update, dbc.Alert("Error: Context missing.", color="danger", duration=3000)
    if existing_data is None: existing_data = {}
    existing_data[key] = comment_value
    return existing_data, dbc.Alert("Comment saved!", color="success", duration=2000, className="ms-2")

@app.callback(
    Output("comment-progressive-passes", "value"),
    Input("store-comment-progressive-passes", "data"),
    Input("url-match-page", "pathname")
)
def load_progressive_passes_comment(stored_data, pathname):
    key = get_comment_key(pathname, "progressive_passes")
    if not key or stored_data is None: return ""
    return stored_data.get(key, "")

@app.callback(
    Output("collapse-prog-home", "is_open"),
    Input("btn-collapse-prog-home", "n_clicks"),
    State("collapse-prog-home", "is_open"),
    prevent_initial_call=True,
)
def toggle_prog_home_table(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("collapse-prog-away", "is_open"),
    Input("btn-collapse-prog-away", "n_clicks"),
    State("collapse-prog-away", "is_open"),
    prevent_initial_call=True,
)
def toggle_prog_away_table(n, is_open):
    if n:
        return not is_open
    return is_open


# --- HELPER FUNCTION TO GENERATE FINAL THIRD PLOT ---
def generate_final_third_plot(stored_data_json):
    print("--- Helper generate_final_third_plot EXECUTING ---")
    if not stored_data_json:
        return dash_html.P("⚠ No data for Final Third plot.", style={"color": "orange"})
    try:
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')
        if not df_json_str or not match_info_json_str:
            return dash_html.P("⚠ DataFrame or match_info missing.", style={"color": "orange"})

        df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)
        if df_processed.empty:
            return dash_html.P("⚠ DataFrame empty for Final Third plot.", style={"color": "orange"})

        HTEAM_NAME = match_info.get('hteamName', 'Home')
        ATEAM_NAME = match_info.get('ateamName', 'Away')
        HTEAM_COLOR = getattr(config, 'DEFAULT_HCOL', "#FF0000")
        ATEAM_COLOR = getattr(config, 'DEFAULT_ACOL', "#0000FF")
        FIG_BG_COLOR = getattr(config, 'BG_COLOR', 'white')
        TEXT_COLOR = getattr(config, 'LINE_COLOR', 'black')
        ZONE14_PLOT_COLOR = getattr(config, 'ZONE14_COLOR', 'orange') # Example, define in config

        # --- Data Preparation ---
        # 1. Get all successful passes
        passes_df = pass_processing.get_passes_df(df_processed.copy())
        if passes_df.empty or 'outcome' not in passes_df.columns:
             return dash_html.P("⚠ Error processing passes or 'outcome' column missing.", style={"color": "red"})
        successful_passes = passes_df[passes_df['outcome'] == 'Successful'].copy()
        if successful_passes.empty:
            return dash_html.P("⚠ No successful passes found for Final Third analysis.", style={"color": "orange"})

        # 2. Analyze for Home Team
        home_successful_passes = successful_passes[successful_passes['team_name'] == HTEAM_NAME]
        df_z14_home, df_lhs_home, df_rhs_home, stats_home = pass_metrics.analyze_final_third_passes(home_successful_passes)

        # 3. Analyze for Away Team
        away_successful_passes = successful_passes[successful_passes['team_name'] == ATEAM_NAME]
        df_z14_away, df_lhs_away, df_rhs_away, stats_away = pass_metrics.analyze_final_third_passes(away_successful_passes)

        # --- Plotting (Dual Plot) ---
        fig, axs = plt.subplots(1, 2, figsize=(25, 10.5), facecolor=FIG_BG_COLOR) # Adjusted figsize

        # Plot Home Team
        if stats_home.get('total_final_third', 0) > 0:
            pitch_plots.plot_zone14_halfspace_map(
                axs[0], df_z14_home, df_lhs_home, df_rhs_home, stats_home,
                HTEAM_NAME, HTEAM_COLOR, is_away_team=False,
                zone14_color=ZONE14_PLOT_COLOR, halfspace_color=HTEAM_COLOR, # Pass team color for halfspace
                bg_color=FIG_BG_COLOR, line_color=TEXT_COLOR
            )
        else:
            axs[0].text(0.5,0.5, f"{HTEAM_NAME}\nNo Final Third Entries", ha='center', va='center', color=TEXT_COLOR)
            axs[0].set_facecolor(FIG_BG_COLOR); axs[0].axis('off')
            pitch_plots.setup_pitch(axs[0], pitch_type='opta', line_color=TEXT_COLOR, background_color=FIG_BG_COLOR) # Draw empty pitch


        # Plot Away Team
        if stats_away.get('total_final_third', 0) > 0:
            pitch_plots.plot_zone14_halfspace_map(
                axs[1], df_z14_away, df_lhs_away, df_rhs_away, stats_away,
                ATEAM_NAME, ATEAM_COLOR, is_away_team=True,
                zone14_color=ZONE14_PLOT_COLOR, halfspace_color=ATEAM_COLOR,
                bg_color=FIG_BG_COLOR, line_color=TEXT_COLOR
            )
        else:
            axs[1].text(0.5,0.5, f"{ATEAM_NAME}\nNo Final Third Entries", ha='center', va='center', color=TEXT_COLOR)
            axs[1].set_facecolor(FIG_BG_COLOR); axs[1].axis('off')
            pitch_plots.setup_pitch(axs[1], pitch_type='opta', line_color=TEXT_COLOR, background_color=FIG_BG_COLOR) # Draw empty pitch
            axs[1].invert_xaxis(); axs[1].invert_yaxis() # Still invert for consistent away view


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0); encoded_img = base64.b64encode(buf.read()).decode('ascii')
        img_src = f"data:image/png;base64,{encoded_img}"
        plt.close(fig)
        
        print("Helper generate_final_third_plot: Successfully created Img.")
        return dash_html.Img(src=img_src, style={"width": "100%", "maxWidth": "100%", "display":"block", "objectFit": "contain"})

    except KeyError as ke:
        return dash_html.P(f"❌ Error (KeyError) in Final Third plot: {ke}", style={"color": "red"})
    except Exception as e:
        tb_str = traceback.format_exc()
        return dash_html.P(f"❌ Error in Final Third plot: {e}\n{tb_str}", style={"color": "red", "whiteSpace": "pre-wrap"})


# --- CALLBACK FOR FINAL THIRD ENTRIES CONTENT ---
# @app.callback(
#     Output("div-final-third-content", "children"),
#     Input("store-df-match", "data"),
#     Input("passes-nested-tabs", "active_tab")
# )
# def show_final_third_content_callback(stored_data_json, active_nested_tab):
#     print(f"--- show_final_third_content_callback TRIGGERED (Active Nested Tab: {active_nested_tab}) ---")
#     if active_nested_tab == "final_third_entries":
#         if not stored_data_json:
#             return dash_html.P("Waiting for match data...", style={"color": "orange"})
#         return generate_final_third_plot(stored_data_json)
#     return no_update

@app.callback(
    Output("div-final-third-content", "children"),
    Input("store-df-match", "data"),
    Input("passes-nested-tabs", "active_tab")
)
def show_final_third_content_callback(stored_data_json, active_nested_tab):
    if active_nested_tab != "final_third_entries" or not stored_data_json:
        return no_update

    try:
        df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        match_info = json.loads(stored_data_json['match_info'])

        HTEAM_NAME = match_info.get('hteamName', 'Home')
        ATEAM_NAME = match_info.get('ateamName', 'Away')
        
        passes_df = pass_processing.get_passes_df(df_processed.copy())
        successful_passes = passes_df[passes_df['outcome'] == 'Successful'].copy()
        
        if successful_passes.empty:
            return dbc.Alert("No successful passes for Final Third analysis.", color="warning")

        # --- Funzione helper interna per non duplicare il codice ---
        def create_final_third_layout_for_team(team_name, team_color, zone14_color, is_away):
            team_passes = successful_passes[successful_passes['team_name'] == team_name]
            
            df_z14, df_lhs, df_rhs, stats = pass_metrics.analyze_final_third_passes(team_passes)
            
            # Grafico
            fig = pass_plotly.plot_final_third_plotly(
                df_z14, df_lhs, df_rhs, stats, team_name, team_color, zone14_color, is_away
            )
            # fig.update_layout(height=500) 
            graph_component = dcc.Graph(figure=fig, config={'displayModeBar': False})
            
            # Tabella dei Top Receivers
            # Concatena tutti i passaggi che finiscono nelle zone di interesse
            df_all_final_third = pd.concat([df_z14, df_lhs, df_rhs])
            
            table_content = dbc.Alert("No receivers in the final third.", color="secondary", className="mt-4")
            if not df_all_final_third.empty:
                # 1. Conta i passaggi ricevuti per ogni giocatore
                top_receivers_series = df_all_final_third['receiver'].value_counts().nlargest(5)
                
                # 2. Crea un DataFrame da questa Serie
                top_receivers_df = top_receivers_series.reset_index()
                top_receivers_df.columns = ['Player', 'Passes Received']
                player_jersey_map = team_passes.drop_duplicates('playerName').set_index('playerName')['Mapped Jersey Number']
                def format_player_name_with_jersey(player_name):
                    jersey_raw = player_jersey_map.get(player_name)
                    try:
                        jersey = str(int(jersey_raw))
                    except (ValueError, TypeError):
                        jersey = '?'
                    return f"#{jersey} - {player_name}"

                top_receivers_df['Player'] = top_receivers_df['Player'].apply(format_player_name_with_jersey)
                
                # **Stile della tabella più compatto**
                table_component = dbc.Table.from_dataframe(
                    top_receivers_df, 
                    striped=True, 
                    bordered=True, 
                    hover=True, 
                    color="dark",
                    #style={'fontSize': '0.8rem'} # Riduci la dimensione del font
                )
                table_content = dash_html.Div([
                    dash_html.H6("Top Receivers in Final Third", className="text-white text-center mt-4"),
                    table_component
                ])
            
            # Layout a due colonne per questo team
            return dbc.Row([
                dbc.Col(graph_component, md=9),
                dbc.Col(table_content, md=3, className="align-self-center")
            ], className="mb-4", align="center")

        # --- Crea i layout per entrambe le squadre ---
        home_layout = create_final_third_layout_for_team(HTEAM_NAME, HCOL, 'orange', is_away=False)
        away_layout = create_final_third_layout_for_team(ATEAM_NAME, ACOL, 'cyan', is_away=True)

        return dash_html.Div([
            home_layout,
            dash_html.Hr(className="my-3"),
            away_layout,
            
            # **Sezione commenti spostata qui**
            dash_html.Hr(className="my-4"),
            dash_html.H6("Comments for Final Third Entries:", className="mt-3 text-white"),
            dcc.Textarea(
                id="comment-final-third",
                placeholder="Enter comments for Final Third Entries...",
                style={'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'},
                className="mb-2"
            ),
            dbc.Button("Save Comment", id="save-comment-final-third", color="info", size="sm"),
            dash_html.Div(id="save-status-final-third", className="small d-inline-block ms-2")
        ])

    except Exception as e:
        tb_str = traceback.format_exc()
        return dbc.Alert(f"Error generating final third plot: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})

# --- COMMENT CALLBACKS FOR FINAL THIRD ENTRIES ---
@app.callback(
    Output("store-comment-final-third", "data"),
    Output("save-status-final-third", "children"),
    Input("save-comment-final-third", "n_clicks"),
    State("comment-final-third", "value"),
    State("url-match-page", "pathname"),
    State("store-comment-final-third", "data"),
    prevent_initial_call=True
)
def save_final_third_comment(n_clicks, comment_value, pathname, existing_data):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pathname, "final_third_entries")
    if not key:
        store_output = existing_data if existing_data is not None else no_update
        return store_output, dbc.Alert("Error: Invalid context.", color="danger", duration=3000)
    if existing_data is None: existing_data = {}
    existing_data[key] = comment_value
    return existing_data, dbc.Alert("Comment saved!", color="success", duration=2000, className="ms-2")

@app.callback(
    Output("comment-final-third", "value"),
    Input("store-comment-final-third", "data"),
    Input("url-match-page", "pathname")
)
def load_final_third_comment(stored_data, pathname):
    key = get_comment_key(pathname, "final_third_entries")
    if not key or stored_data is None: return ""
    return stored_data.get(key, "")

# ------------------------------------

# --- HELPER FUNCTION TO GENERATE PASS DENSITY PLOTS ---
def generate_pass_density_plots(stored_data_json):
    print("--- Helper generate_pass_density_plots EXECUTING ---")
    # ... (Similar structure to other plot generators: get df, match_info, team names, colors)
    if not stored_data_json: return dash_html.P("⚠ No data for Pass Density.", style={"color": "orange"})
    try:
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')
        if not df_json_str or not match_info_json_str: return dash_html.P("⚠ Data missing.", style={"color": "orange"})
        df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)
        if df_processed.empty: return dash_html.P("⚠ DataFrame empty.", style={"color": "orange"})

        HTEAM_NAME = match_info.get('hteamName', 'Home')
        ATEAM_NAME = match_info.get('ateamName', 'Away')
        # Use specific cmap from config or default
        HOME_CMAP_DENSITY = getattr(config, 'HOME_HEATMAP_CMAP', 'Reds') # Example
        AWAY_CMAP_DENSITY = getattr(config, 'AWAY_HEATMAP_CMAP', 'Blues') # Example
        FIG_BG_COLOR = getattr(config, 'BG_COLOR', 'white')

        passes_df = pass_processing.get_passes_df(df_processed.copy()) # Get all passes
        if passes_df.empty: return dash_html.P("⚠ No passes found for density plots.", style={"color": "orange"})

        home_passes = passes_df[passes_df['team_name'] == HTEAM_NAME]
        away_passes = passes_df[passes_df['team_name'] == ATEAM_NAME]

        fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor=FIG_BG_COLOR) # VerticalPitch usually needs more height per plot

        pitch_plots.plot_pass_density(axs[0], home_passes, HTEAM_NAME, cmap=HOME_CMAP_DENSITY, is_away_team=False)
        pitch_plots.plot_pass_density(axs[1], away_passes, ATEAM_NAME, cmap=AWAY_CMAP_DENSITY, is_away_team=True)
        
        # Common title for the dual plot can be handled by the H5 in render_match_tab_content
        # fig.suptitle("Pass Density Comparison", fontsize=18, color=getattr(config, 'LINE_COLOR', 'black'))
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle if you add one in Matplotlib

        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=90, bbox_inches='tight', facecolor=fig.get_facecolor()); buf.seek(0)
        encoded_img = base64.b64encode(buf.read()).decode('ascii'); img_src = f"data:image/png;base64,{encoded_img}"
        plt.close(fig)
        return dash_html.Img(src=img_src, style={"width": "100%", "maxWidth": "100%", "display":"block", "objectFit": "contain"})
    except Exception as e:
        return dash_html.P(f"❌ Error generating Pass Density plots: {e}", style={"color": "red"})


# --- HELPER FUNCTION TO GENERATE PASS HEATMAP PLOTS ---
def generate_pass_heatmap_plots(stored_data_json):
    print("--- Helper generate_pass_heatmap_plots EXECUTING ---")
    # ... (Similar structure: get df, match_info, team names, cmaps) ...
    if not stored_data_json: return dash_html.P("⚠ No data for Pass Heatmap.", style={"color": "orange"})
    try:
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')
        if not df_json_str or not match_info_json_str: return dash_html.P("⚠ Data missing.", style={"color": "orange"})
        df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)
        if df_processed.empty: return dash_html.P("⚠ DataFrame empty.", style={"color": "orange"})

        HTEAM_NAME = match_info.get('hteamName', 'Home')
        ATEAM_NAME = match_info.get('ateamName', 'Away')
        HOME_CMAP_HEATMAP = getattr(config, 'HOME_HEATMAP_CMAP', 'Reds')
        AWAY_CMAP_HEATMAP = getattr(config, 'AWAY_HEATMAP_CMAP', 'Blues')
        FIG_BG_COLOR = getattr(config, 'BG_COLOR', 'white')

        passes_df = pass_processing.get_passes_df(df_processed.copy())
        if passes_df.empty: return dash_html.P("⚠ No passes found for heatmaps.", style={"color": "orange"})

        home_passes = passes_df[passes_df['team_name'] == HTEAM_NAME]
        away_passes = passes_df[passes_df['team_name'] == ATEAM_NAME]

        fig, axs = plt.subplots(1, 2, figsize=(20, 10), facecolor=FIG_BG_COLOR)

        pitch_plots.plot_pass_heatmap(axs[0], home_passes, HTEAM_NAME, cmap=HOME_CMAP_HEATMAP, is_away_team=False)
        pitch_plots.plot_pass_heatmap(axs[1], away_passes, ATEAM_NAME, cmap=AWAY_CMAP_HEATMAP, is_away_team=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=90, bbox_inches='tight', facecolor=fig.get_facecolor()); buf.seek(0)
        encoded_img = base64.b64encode(buf.read()).decode('ascii'); img_src = f"data:image/png;base64,{encoded_img}"
        plt.close(fig)
        return dash_html.Img(src=img_src, style={"width": "100%", "maxWidth": "100%", "display":"block", "objectFit": "contain"})
    except Exception as e:
        return dash_html.P(f"❌ Error generating Pass Heatmap plots: {e}", style={"color": "red"})


# --- CALLBACKS FOR PASS LOCATIONS (DENSITY & HEATMAP) ---
# @app.callback(
#     Output("div-pass-density-content", "children"),
#     Input("store-df-match", "data"),
#     Input("passes-nested-tabs", "active_tab")
# )
# def show_pass_density_content_callback(stored_data_json, active_nested_tab):
#     if active_nested_tab == "pass_locations" and stored_data_json: # Only generate if parent tab is active
#         return generate_pass_density_plots(stored_data_json)
#     return no_update # Or dash_html.Div() if you want to clear it

@app.callback(
    Output("div-pass-density-content", "children"), # Il nome dell'ID non è più perfetto, ma funziona
    Input("store-df-match", "data"),
    Input("passes-nested-tabs", "active_tab")
)
def show_pass_location_plots_callback(stored_data_json, active_nested_tab):
    if active_nested_tab != "pass_locations" or not stored_data_json:
        return no_update

    try:
        df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        match_info = json.loads(stored_data_json['match_info'])

        HTEAM_NAME = match_info.get('hteamName', 'Home')
        ATEAM_NAME = match_info.get('ateamName', 'Away')
        
        passes_df = pass_processing.get_passes_df(df_processed.copy())
        if passes_df.empty:
            return dbc.Alert("No passes found to generate location plots.", color="warning")

        home_passes = passes_df[passes_df['team_name'] == HTEAM_NAME]
        away_passes = passes_df[passes_df['team_name'] == ATEAM_NAME]
        
        # Crea i grafici interattivi separatamente
        fig_home_density = pass_plotly.plot_pass_density_plotly(home_passes, HTEAM_NAME, is_away=False)
        fig_home_heatmap = pass_plotly.plot_pass_heatmap_plotly(home_passes, HTEAM_NAME, is_away=False)
        
        fig_away_density = pass_plotly.plot_pass_density_plotly(away_passes, ATEAM_NAME, is_away=True)
        fig_away_heatmap = pass_plotly.plot_pass_heatmap_plotly(away_passes, ATEAM_NAME, is_away=True)
        
        # Costruisci il layout finale con i subplot gestiti da Dash Bootstrap
        return dash_html.Div([
            # Sezione Home Team
            dash_html.H4(f"{HTEAM_NAME} - Pass Locations", className="text-center text-white mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_home_density), md=6),
                dbc.Col(dcc.Graph(figure=fig_home_heatmap), md=6),
            ], className="g-2"), # g-2 riduce lo spazio tra le colonne

            dash_html.Hr(className="my-4"),

            # Sezione Away Team
            dash_html.H4(f"{ATEAM_NAME} - Pass Locations", className="text-center text-white mt-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_away_density), md=6),
                dbc.Col(dcc.Graph(figure=fig_away_heatmap), md=6),
            ], className="g-2")
        ])
    except Exception as e:
        tb_str = traceback.format_exc()
        return dbc.Alert(f"Error generating pass location plots: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})

# @app.callback(
#     Output("div-pass-heatmap-content", "children"),
#     Input("store-df-match", "data"),
#     Input("passes-nested-tabs", "active_tab")
# )
# def show_pass_heatmap_content_callback(stored_data_json, active_nested_tab):
#     if active_nested_tab == "pass_locations" and stored_data_json:
#         return generate_pass_heatmap_plots(stored_data_json)
#     return no_update


# --- COMMENT CALLBACKS FOR PASS DENSITY ---
@app.callback(Output("store-comment-pass-density", "data"), Output("save-status-pass-density", "children"),
              Input("save-comment-pass-density", "n_clicks"),
              State("comment-pass-density", "value"), State("url-match-page", "pathname"), State("store-comment-pass-density", "data"),
              prevent_initial_call=True)
def save_pass_density_comment(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pass_density"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-pass-density", "value"),
              Input("store-comment-pass-density", "data"), Input("url-match-page", "pathname"))
def load_pass_density_comment(data, pn):
    key = get_comment_key(pn, "pass_density")
    if not key or data is None: return ""
    return data.get(key, "")

# --- COMMENT CALLBACKS FOR PASS HEATMAP ---
@app.callback(Output("store-comment-pass-heatmap", "data"), Output("save-status-pass-heatmap", "children"),
              Input("save-comment-pass-heatmap", "n_clicks"),
              State("comment-pass-heatmap", "value"), State("url-match-page", "pathname"), State("store-comment-pass-heatmap", "data"),
              prevent_initial_call=True)
def save_pass_heatmap_comment(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pass_heatmap"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-pass-heatmap", "value"),
              Input("store-comment-pass-heatmap", "data"), Input("url-match-page", "pathname"))
def load_pass_heatmap_comment(data, pn):
    key = get_comment_key(pn, "pass_heatmap")
    if not key or data is None: return ""
    return data.get(key, "")

# -----------------------------------------

# # --- NEW SINGLE CALLBACK FOR CONTENT WITHIN "Player Analysis" NESTED TABS ---
# @app.callback(
#     Output("player-analysis-nested-tab-content", "children"),
#     Input("player-analysis-nested-tabs", "active_tab"),
#     Input("store-df-match", "data"),
#     Input("store-player-stats-df", "data"),
#     State("url-match-page", "pathname") # For comment keys
# )
# def render_player_analysis_nested_content(active_nested_tab, stored_match_data_json, player_stats_df_json, pathname):
#     print(f"--- render_player_analysis_nested_content TRIGGERED --- Active Nested Tab: {active_nested_tab}")

#     common_textarea_style = {'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'}
#     common_flex_column_style = {"display": "flex", "flexDirection": "column", "height": "calc(100vh - 220px)"}
#     common_plot_area_style = {"flex": "1 1 75%", "minHeight": "350px", "overflow": "hidden"}
#     common_comment_area_style = {"flex": "0 0 20%", "paddingTop": "15px", "overflowY": "auto"}

#     if not stored_match_data_json:
#         return dash_html.P("Waiting for base match data...", style={"color": "orange"})

#     if active_nested_tab == "pa_top_passers_stats":
#         print("  Rendering content for 'pa_top_passers_stats'")
#         if not player_stats_df_json:
#             return dash_html.P("Player stats data not yet available for bar chart.", style={"color": "orange"})
        
#         bar_chart_img_component = generate_top_passer_stats_plot(player_stats_df_json)
        
#         return dash_html.Div([ # Flex container for this nested tab
#             dash_html.Div(bar_chart_img_component, style=common_plot_area_style),
#             dash_html.Div([ # Comment Area
#                 dash_html.Hr(),
#                 dash_html.H6("Comments for Top Passers Chart:", className="mt-3 text-white"),
#                 dcc.Textarea(id="comment-top-passers-bar", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
#                 dbc.Button("Save Comment", id="save-comment-top-passers-bar", color="info", size="sm", className="me-2"),
#                 dash_html.Div(id="save-status-top-passers-bar", className="small d-inline-block")
#             ], style=common_comment_area_style)
#         ], style=common_flex_column_style)

#     elif active_nested_tab == "pa_home_passer_map":
#         print("  Rendering content for 'pa_home_passer_map'")
#         if not player_stats_df_json:
#             return dash_html.P("Player stats data not yet available for home map.", style={"color": "orange"})

#         match_info = json.loads(stored_match_data_json['match_info'])
#         df_processed = pd.read_json(stored_match_data_json['df'], orient='split')
#         player_stats_df = pd.read_json(player_stats_df_json, orient='split')
#         HTEAM_NAME = match_info.get('hteamName')
#         top_home_passer_name = None
#         if HTEAM_NAME and not player_stats_df.empty: # Ensure player_stats_df is not empty
#             home_team_player_names = df_processed[df_processed['team_name'] == HTEAM_NAME]['playerName'].unique()
#             home_player_stats_df = player_stats_df[player_stats_df.index.isin(home_team_player_names)]
#             if not home_player_stats_df.empty:
#                 top_home_series = home_player_stats_df.sort_values('Offensive Pass Total', ascending=False)
#                 if not top_home_series.empty: top_home_passer_name = top_home_series.index[0]
        
#         home_pass_map_img = generate_player_pass_map_plot(stored_match_data_json, top_home_passer_name, False) if top_home_passer_name else dash_html.P(f"Could not determine top passer for {HTEAM_NAME} map.", style={"color":"orange"})
        
#         return dash_html.Div([ # Flex container
#             dash_html.Div(home_pass_map_img, style=common_plot_area_style),
#             dash_html.Div([ # Comment Area
#                 dash_html.Hr(),
#                 dash_html.H6("Comments for Home Passer Map:", className="mt-3 text-white"),
#                 dcc.Textarea(id="comment-home-top-passer-map", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
#                 dbc.Button("Save Comment", id="save-comment-home-top-passer-map", color="info", size="sm", className="me-2"),
#                 dash_html.Div(id="save-status-home-top-passer-map", className="small d-inline-block")
#             ], style=common_comment_area_style)
#         ], style=common_flex_column_style)

#     elif active_nested_tab == "pa_away_passer_map":
#         print("  Rendering content for 'pa_away_passer_map'")
#         if not player_stats_df_json:
#             return dash_html.P("Player stats data not yet available for away map.", style={"color": "orange"})
            
#         match_info = json.loads(stored_match_data_json['match_info'])
#         df_processed = pd.read_json(stored_match_data_json['df'], orient='split')
#         player_stats_df = pd.read_json(player_stats_df_json, orient='split')
#         ATEAM_NAME = match_info.get('ateamName')
#         top_away_passer_name = None
#         if ATEAM_NAME and not player_stats_df.empty: # Ensure player_stats_df is not empty
#             away_team_player_names = df_processed[df_processed['team_name'] == ATEAM_NAME]['playerName'].unique()
#             away_player_stats_df = player_stats_df[player_stats_df.index.isin(away_team_player_names)]
#             if not away_player_stats_df.empty:
#                 top_away_series = away_player_stats_df.sort_values('Offensive Pass Total', ascending=False)
#                 if not top_away_series.empty: top_away_passer_name = top_away_series.index[0]

#         away_pass_map_img = generate_player_pass_map_plot(stored_match_data_json, top_away_passer_name, True) if top_away_passer_name else dash_html.P(f"Could not determine top passer for {ATEAM_NAME} map.", style={"color":"orange"})

#         return dash_html.Div([ # Flex container
#             dash_html.Div(away_pass_map_img, style=common_plot_area_style),
#             dash_html.Div([ # Comment Area
#                 dash_html.Hr(),
#                 dash_html.H6("Comments for Away Passer Map:", className="mt-3 text-white"),
#                 dcc.Textarea(id="comment-away-top-passer-map", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
#                 dbc.Button("Save Comment", id="save-comment-away-top-passer-map", color="info", size="sm", className="me-2"),
#                 dash_html.Div(id="save-status-away-top-passer-map", className="small d-inline-block")
#             ], style=common_comment_area_style)
#         ], style=common_flex_column_style)
    
#     elif active_nested_tab == "pa_shot_sequence_stats":
#         print("  Rendering content for 'pa_shot_sequence_stats'")
#         if not player_stats_df_json:
#             return dash_html.P("Player stats data not yet available for shot sequence chart.", style={"color": "orange"})
        
#         shot_seq_bar_chart = generate_shot_sequence_bar_plot(player_stats_df_json)
        
#         return dash_html.Div([ # Flex container for this nested tab
#             dash_html.Div(shot_seq_bar_chart, style=common_plot_area_style),
#             dash_html.Div([ # Comment Area
#                 dash_html.Hr(),
#                 dash_html.H6("Comments for Shot Sequence Chart:", className="mt-3 text-white"),
#                 dcc.Textarea(id="comment-shot-sequence-bar", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
#                 dbc.Button("Save Comment", id="save-comment-shot-sequence-bar", color="info", size="sm", className="me-2"),
#                 dash_html.Div(id="save-status-shot-sequence-bar", className="small d-inline-block")
#             ], style=common_comment_area_style)
#         ], style=common_flex_column_style)

#     elif active_nested_tab == "pa_home_shot_contributor_map":
#         print("  Rendering content for 'pa_home_shot_contributor_map'")
#         if not player_stats_df_json:
#             return dash_html.P("Player stats not yet available for home contributor map.", style={"color": "orange"})
            
#         home_contributor_map = generate_team_top_shot_contributor_map_plot(stored_match_data_json, player_stats_df_json, is_for_home_team=True)

#         return dash_html.Div([ # Flex container
#             dash_html.Div(home_contributor_map, style=common_plot_area_style),
#             dash_html.Div([ # Comment Area
#                 dash_html.Hr(),
#                 dash_html.H6("Comments for Home Contributor Map:", className="mt-3 text-white"),
#                 dcc.Textarea(id="comment-home-top-shot-contributor-map", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
#                 dbc.Button("Save Comment", id="save-comment-home-top-shot-contributor-map", color="info", size="sm", className="me-2"),
#                 dash_html.Div(id="save-status-home-top-shot-contributor-map", className="small d-inline-block")
#             ], style=common_comment_area_style)
#         ], style=common_flex_column_style)

#     elif active_nested_tab == "pa_away_shot_contributor_map":
#         print("  Rendering content for 'pa_away_shot_contributor_map'")
#         if not player_stats_df_json:
#             return dash_html.P("Player stats not yet available for away contributor map.", style={"color": "orange"})
            
#         away_contributor_map = generate_team_top_shot_contributor_map_plot(stored_match_data_json, player_stats_df_json, is_for_home_team=False)

#         return dash_html.Div([ # Flex container
#             dash_html.Div(away_contributor_map, style=common_plot_area_style),
#             dash_html.Div([ # Comment Area
#                 dash_html.Hr(),
#                 dash_html.H6("Comments for Away Contributor Map:", className="mt-3 text-white"),
#                 dcc.Textarea(id="comment-away-top-shot-contributor-map", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
#                 dbc.Button("Save Comment", id="save-comment-away-top-shot-contributor-map", color="info", size="sm", className="me-2"),
#                 dash_html.Div(id="save-status-away-top-shot-contributor-map", className="small d-inline-block")
#             ], style=common_comment_area_style)
#         ], style=common_flex_column_style)
    
#     elif active_nested_tab == "pa_defender_stats":
#         print("  Rendering content for 'pa_defender_stats' (Simple Static Plot)")
#         if not player_stats_df_json:
#             return dash_html.P("Player stats data not available.", style={"color": "orange"})

#         try:
#             player_stats_df = pd.read_json(player_stats_df_json, orient='split')
#             df_processed = pd.read_json(stored_match_data_json['df'], orient='split')
#             match_info = json.loads(stored_match_data_json['match_info'])
#             home_team_name = match_info.get('hteamName', '')

#             # --- Generate the Matplotlib Bar Chart Image ---
#             # Using a dark background consistent with the theme
#             fig_bar, ax_bar = plt.subplots(figsize=(12, 8), facecolor='#2E3439')
#             ax_bar.set_facecolor('#2E3439')

#             # Call our robust plotting function
#             player_plots.plot_defender_stats_bar_by_team(
#                 ax_bar, player_stats_df, df_processed, home_team_name, hcol=HCOL, acol=ACOL
#             )
            
#             # --- Convert plot to image for Dash ---
#             buf = io.BytesIO()
#             plt.savefig(buf, format="png", dpi=100, bbox_inches='tight', facecolor=fig_bar.get_facecolor())
#             plt.close(fig_bar)
#             buf.seek(0)
#             img_src = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"

#             # --- Return the Bar Chart and the Comment Section ---
#             # This layout is clean and focuses on the single plot
#             return dash_html.Div([
#                 dash_html.Div(dash_html.Img(src=img_src, style={'width': '100%', 'maxWidth': '900px', 'display': 'block', 'margin': 'auto'}), style=common_plot_area_style),
#                 dash_html.Div([
#                     dash_html.Hr(),
#                     dash_html.H6("Comments for Defender Stats Chart:", className="mt-3 text-white"),
#                     dcc.Textarea(id="comment-defender-stats-bar", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
#                     dbc.Button("Save Comment", id="save-comment-defender-stats-bar", color="info", size="sm", className="me-2"),
#                     dash_html.Div(id="save-status-defender-stats-bar", className="small d-inline-block")
#                 ], style=common_comment_area_style)
#             ], style=common_flex_column_style)

#         except Exception as e:
#             tb_str = traceback.format_exc()
#             return dbc.Alert(f"Error creating defender bar chart: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})

#     elif active_nested_tab == "pa_home_defender_map":
#         print("  Rendering content for 'pa_home_defender_map'")
#         if not player_stats_df_json:
#             return dash_html.P("Player stats not available for home defender map.", style={"color": "orange"})
            
#         home_defender_map, home_stats_table, home_players = generate_team_top_defender_map_plot(
#             stored_match_data_json, player_stats_df_json, is_for_home_team=True
#         )
        
#         if home_stats_table is None:
#             return home_defender_map

#         # Construct the initial layout with the modified dropdown
#         return dash_html.Div([
#             dbc.Row(dbc.Col(
#                 dcc.Dropdown(
#                     id='home-defender-dropdown',
#                     options=home_players,
#                     # --- START CHANGES ---
#                     value=None,  # Set default value to None (empty)
#                     placeholder="Select another player to analyze...", # Add placeholder text
#                     # --- END CHANGES ---
#                     clearable=True, # Allow user to clear selection to see default again (optional but good UX)
#                     style={'color': 'black'}
#                 ),
#                 md=6,
#             ), justify="center", className="mb-3"),
            
#             # This div is updated by the other callback
#             dash_html.Div(id='home-defender-output', children=[
#                 dbc.Row(dbc.Col(home_defender_map, width=12)),
#                 dbc.Row([
#                     dbc.Col([
#                         dash_html.H5("Action Summary", className="text-center text-white mt-4"),
#                         home_stats_table
#                     ], md=6),
#                     dbc.Col([
#                         dash_html.H5("Analyst Comments", className="text-center text-white mt-4"),
#                         dcc.Textarea(id="comment-home-top-defender-map", placeholder="Comments for the TOP defender...", style=common_textarea_style, className="mb-2"),
#                         dbc.Button("Save Comment", id="save-comment-home-top-defender-map", color="info", size="sm", className="me-2"),
#                         dash_html.Div(id="save-status-home-top-defender-map", className="small d-inline-block")
#                     ], md=6)
#                 ], className="mt-3")
#             ])
#         ])

#     elif active_nested_tab == "pa_away_defender_map":
#         print("  Rendering content for 'pa_away_defender_map'")
#         if not player_stats_df_json:
#             return dash_html.P("Player stats not available for away defender map.", style={"color": "orange"})
            
#         away_defender_map, away_stats_table, away_players = generate_team_top_defender_map_plot(
#             stored_match_data_json, player_stats_df_json, is_for_home_team=False
#         )
        
#         if away_stats_table is None:
#             return away_defender_map
            
#         return dash_html.Div([
#             dbc.Row(dbc.Col(
#                 dcc.Dropdown(
#                     id='away-defender-dropdown',
#                     options=away_players,
#                     # --- START CHANGES ---
#                     value=None, # Set default value to None (empty)
#                     placeholder="Select another player to analyze...", # Add placeholder text
#                     # --- END CHANGES ---
#                     clearable=True,
#                     style={'color': 'black'}
#                 ),
#                 md=6,
#             ), justify="center", className="mb-3"),
            
#             dash_html.Div(id='away-defender-output', children=[
#                  dbc.Row(dbc.Col(away_defender_map, width=12)),
#                  dbc.Row([
#                     dbc.Col([
#                         dash_html.H5("Action Summary", className="text-center text-white mt-4"),
#                         away_stats_table
#                     ], md=6),
#                     dbc.Col([
#                         dash_html.H5("Analyst Comments", className="text-center text-white mt-4"),
#                         dcc.Textarea(id="comment-away-top-defender-map", placeholder="Comments for the TOP defender...", style=common_textarea_style, className="mb-2"),
#                         dbc.Button("Save Comment", id="save-comment-away-top-defender-map", color="info", size="sm", className="me-2"),
#                         dash_html.Div(id="save-status-away-top-defender-map", className="small d-inline-block")
#                     ], md=6)
#                  ], className="mt-3")
#             ])
#         ])

#     return dash_html.P(f"Content for player analysis sub-tab '{active_nested_tab}' not yet implemented.", style={"color":"white"})

@app.callback(
    Output("player-analysis-primary-tab-content", "children"),
    Input("player-analysis-primary-tabs", "active_tab")
)
def render_player_analysis_secondary_layout(active_primary_tab):
    """
    This callback acts as a router. Based on the selected primary tab
    (Passing, Shooting, or Defending), it renders the appropriate
    secondary tab layout.
    """
    if active_primary_tab == "pa_primary_passing":
        return dash_html.Div([
            dbc.Tabs(
                id="passing-secondary-tabs",
                active_tab="pa_top_passers_stats",
                children=[
                    dbc.Tab(label="Top Passers Stats", tab_id="pa_top_passers_stats"),
                    dbc.Tab(label="Home Top Passer Map", tab_id="pa_home_passer_map"),
                    dbc.Tab(label="Away Top Passer Map", tab_id="pa_away_passer_map"),
                ], className="mt-2"
            ),
            dcc.Loading(type="circle", children=dash_html.Div(id="passing-secondary-tab-content"))
        ])
    elif active_primary_tab == "pa_primary_shooting":
        return dash_html.Div([
            dbc.Tabs(
                id="shooting-secondary-tabs",
                active_tab="pa_shot_sequence_stats",
                children=[
                    dbc.Tab(label="Shot Sequence Stats", tab_id="pa_shot_sequence_stats"),
                    dbc.Tab(label="Home Contributor Map", tab_id="pa_home_shot_contributor_map"),
                    dbc.Tab(label="Away Contributor Map", tab_id="pa_away_shot_contributor_map"),
                ], className="mt-2"
            ),
            dcc.Loading(type="circle", children=dash_html.Div(id="shooting-secondary-tab-content"))
        ])
    elif active_primary_tab == "pa_primary_defending":
        return dash_html.Div([
             dbc.Tabs(
                id="defending-secondary-tabs",
                active_tab="pa_defender_stats",
                children=[
                    dbc.Tab(label="Defender Stats", tab_id="pa_defender_stats"),
                    dbc.Tab(label="Home Defender Map", tab_id="pa_home_defender_map"),
                    dbc.Tab(label="Away Defender Map", tab_id="pa_away_defender_map"),
                ], className="mt-2"
            ),
            dcc.Loading(type="circle", children=dash_html.Div(id="defending-secondary-tab-content"))
        ])
    
    return dash_html.P("Select an analysis category.")

########################################################################
def create_player_pass_map_layout(team_type, stored_match_data_json, player_stats_df_json):
    """
    Versione 3: Corregge l'errore 'numpy.ndarray' object has no attribute 'empty'.
    """
    try:
        match_info = json.loads(stored_match_data_json['match_info'])
        df_processed = pd.read_json(io.StringIO(stored_match_data_json['df']), orient='split')
        player_stats_df = pd.read_json(io.StringIO(player_stats_df_json), orient='split')

        team_name = match_info.get('hteamName') if team_type == 'home' else match_info.get('ateamName')
        is_away = (team_type == 'away')

        passes_df = pass_processing.get_passes_df(df_processed)
        team_passers_df = passes_df[passes_df['team_name'] == team_name]
        
        # Se non ci sono passaggi per questa squadra, mostra un avviso e fermati.
        if team_passers_df.empty:
            return dbc.Alert(f"No passes recorded for {team_name}", color="warning", className="mt-3")

        # Ora che sappiamo che non è vuoto, possiamo procedere in sicurezza.
        player_jersey_map = team_passers_df.drop_duplicates('playerName').set_index('playerName')['Mapped Jersey Number']
        
        sorted_player_names = sorted(player_jersey_map.index.tolist())

        dropdown_options = []
        for name in sorted_player_names:
            jersey_raw = player_jersey_map.get(name)
            try:
                jersey = str(int(jersey_raw))
            except (ValueError, TypeError):
                jersey = '?'
            dropdown_options.append({'label': f"#{jersey} - {name}", 'value': name})

        # Trova il top passer per il valore di default
        # Filtra le statistiche solo per i giocatori che hanno effettivamente passato la palla
        team_player_stats = player_stats_df[player_stats_df.index.isin(player_jersey_map.index)]
        
        top_passer_name = None
        # Controlla se il DataFrame delle statistiche per questi giocatori non è vuoto
        if not team_player_stats.empty:
            top_passer_name = team_player_stats['Offensive Pass Total'].idxmax()

        return dash_html.Div([
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        id=f'{team_type}-passer-dropdown',
                        options=dropdown_options,
                        value=top_passer_name, # Sarà None se non ci sono stats, che è ok
                        placeholder="Select a player...",
                        style={'color': 'black'}
                    ),
                    md=6,
                ),
                justify="center",
                className="my-3"
            ),
            dcc.Loading(
                dash_html.Div(id=f'player-pass-map-graph-container-{team_type}')
            )
        ])

    except Exception as e:
        tb_str = traceback.format_exc()
        return dbc.Alert(f"Error creating layout for {team_type} passer map: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})


# --- CALLBACK 1: For the PASSING secondary tabs ---
@app.callback(
    Output("passing-secondary-tab-content", "children"),
    Input("passing-secondary-tabs", "active_tab"),
    Input("store-player-stats-df", "data"), 
    State("store-df-match", "data"),
)
def render_passing_analysis_content(active_tab, player_stats_df_json, stored_match_data_json):
    # This guard clause is now very important. It handles the initial moment
    # before the player stats data has been calculated.
    if not player_stats_df_json:
        return dash_html.P("Player stats are loading...")
    
    # We reuse the logic from the old callback here
    common_textarea_style = {'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'}
    common_flex_column_style = {"display": "flex", "flexDirection": "column", "height": "calc(100vh - 280px)"}
    common_plot_area_style = {"flex": "1 1 75%", "minHeight": "350px", "overflow": "hidden"}
    common_comment_area_style = {"flex": "0 0 20%", "paddingTop": "15px", "overflowY": "auto"}

    if active_tab == "pa_top_passers_stats":
        try:
            # Carica tutti i dati necessari
            player_stats_df = pd.read_json(io.StringIO(player_stats_df_json), orient='split')
            df_processed = pd.read_json(io.StringIO(stored_match_data_json['df']), orient='split')
            match_info = json.loads(stored_match_data_json['match_info'])
            home_team_name = match_info.get('hteamName', '')

            # **CHIAMATA ALLA NUOVA FUNZIONE PLOTLY CON I DATI AGGIUNTIVI**
            fig = player_plots.plot_passer_stats_bar_plotly(
                player_stats_df,
                df_processed, # Passiamo il df completo per la mappatura del team
                home_team_name, # Passiamo il nome della squadra di casa
                hcol=HCOL,
                acol=ACOL,
                violet_col=VIOLET
            )
            
            return dash_html.Div([
                # Grafico a larghezza piena
                dbc.Row(
                    dbc.Col(dcc.Graph(figure=fig), width=12)
                ),
                # Sezione commenti sotto
                dbc.Row(
                    dbc.Col([
                        dash_html.Hr(),
                        dash_html.H5("Analyst Comments", className="mt-3"),
                        dcc.Textarea(
                            id="comment-top-passers-bar",
                            placeholder="Enter your analysis on top passers...",
                            style=common_textarea_style,
                            className="mb-2"
                        ),
                        dbc.Button("Save Comment", id="save-comment-top-passers-bar", color="info", size="sm"),
                        dash_html.Div(id="save-status-top-passers-bar", className="small d-inline-block ms-2 mt-2")
                    ], width=12),
                    className="mt-4"
                )
            ])
        except Exception as e:
            tb_str = traceback.format_exc()
            return dbc.Alert(f"Error generating passer stats plot: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})


    elif active_tab == "pa_home_passer_map":
        return create_player_pass_map_layout('home', stored_match_data_json, player_stats_df_json)
        # print("  Rendering content for 'pa_home_passer_map'")
        # if not player_stats_df_json:
        #     return dash_html.P("Player stats data not yet available for home map.", style={"color": "orange"})

        # match_info = json.loads(stored_match_data_json['match_info'])
        # df_processed = pd.read_json(stored_match_data_json['df'], orient='split')
        # player_stats_df = pd.read_json(player_stats_df_json, orient='split')
        # HTEAM_NAME = match_info.get('hteamName')
        # top_home_passer_name = None
        # if HTEAM_NAME and not player_stats_df.empty: # Ensure player_stats_df is not empty
        #     home_team_player_names = df_processed[df_processed['team_name'] == HTEAM_NAME]['playerName'].unique()
        #     home_player_stats_df = player_stats_df[player_stats_df.index.isin(home_team_player_names)]
        #     if not home_player_stats_df.empty:
        #         top_home_series = home_player_stats_df.sort_values('Offensive Pass Total', ascending=False)
        #         if not top_home_series.empty: top_home_passer_name = top_home_series.index[0]
        
        # home_pass_map_img = generate_player_pass_map_plot(stored_match_data_json, top_home_passer_name, False) if top_home_passer_name else dash_html.P(f"Could not determine top passer for {HTEAM_NAME} map.", style={"color":"orange"})
        
        # return dash_html.Div([ # Flex container
        #     dash_html.Div(home_pass_map_img, style=common_plot_area_style),
        #     dash_html.Div([ # Comment Area
        #         dash_html.Hr(),
        #         dash_html.H6("Comments for Home Passer Map:", className="mt-3 text-white"),
        #         dcc.Textarea(id="comment-home-top-passer-map", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
        #         dbc.Button("Save Comment", id="save-comment-home-top-passer-map", color="info", size="sm", className="me-2"),
        #         dash_html.Div(id="save-status-home-top-passer-map", className="small d-inline-block")
        #     ], style=common_comment_area_style)
        # ], style=common_flex_column_style)

    elif active_tab == "pa_away_passer_map":
        return create_player_pass_map_layout('away', stored_match_data_json, player_stats_df_json)
    #     print("  Rendering content for 'pa_away_passer_map'")
    #     if not player_stats_df_json:
    #         return dash_html.P("Player stats data not yet available for away map.", style={"color": "orange"})
            
    #     match_info = json.loads(stored_match_data_json['match_info'])
    #     df_processed = pd.read_json(stored_match_data_json['df'], orient='split')
    #     player_stats_df = pd.read_json(player_stats_df_json, orient='split')
    #     ATEAM_NAME = match_info.get('ateamName')
    #     top_away_passer_name = None
    #     if ATEAM_NAME and not player_stats_df.empty: # Ensure player_stats_df is not empty
    #         away_team_player_names = df_processed[df_processed['team_name'] == ATEAM_NAME]['playerName'].unique()
    #         away_player_stats_df = player_stats_df[player_stats_df.index.isin(away_team_player_names)]
    #         if not away_player_stats_df.empty:
    #             top_away_series = away_player_stats_df.sort_values('Offensive Pass Total', ascending=False)
    #             if not top_away_series.empty: top_away_passer_name = top_away_series.index[0]

    #     away_pass_map_img = generate_player_pass_map_plot(stored_match_data_json, top_away_passer_name, True) if top_away_passer_name else dash_html.P(f"Could not determine top passer for {ATEAM_NAME} map.", style={"color":"orange"})

    #     return dash_html.Div([ # Flex container
    #         dash_html.Div(away_pass_map_img, style=common_plot_area_style),
    #         dash_html.Div([ # Comment Area
    #             dash_html.Hr(),
    #             dash_html.H6("Comments for Away Passer Map:", className="mt-3 text-white"),
    #             dcc.Textarea(id="comment-away-top-passer-map", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
    #             dbc.Button("Save Comment", id="save-comment-away-top-passer-map", color="info", size="sm", className="me-2"),
    #             dash_html.Div(id="save-status-away-top-passer-map", className="small d-inline-block")
    #         ], style=common_comment_area_style)
    #     ], style=common_flex_column_style)
    # return dash_html.P(f"Content for {active_tab} not found.")

@app.callback(
    Output('player-pass-map-graph-container-home', 'children'),
    Input('home-passer-dropdown', 'value'),
    State('store-df-match', 'data')
)
def update_home_passer_map(selected_player, stored_data_json):
    if not selected_player or not stored_data_json:
        return no_update

    df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
    team_color = HCOL
    
    all_passes = pass_processing.get_passes_df(df_processed.copy())
    player_passes = all_passes[all_passes['playerName'] == selected_player]
    
    # **Estrai il numero di maglia**
    jersey_num = '?'
    if not player_passes.empty:
        # Assumendo che 'Mapped Jersey Number' sia una colonna nel df
        jersey_num_raw = player_passes['Mapped Jersey Number'].iloc[0]
        try:
            jersey_num = str(int(jersey_num_raw))
        except (ValueError, TypeError):
            pass # Lascia '?' se non è un numero

    fig = player_plots.plot_player_pass_map_plotly(
        player_passes, selected_player, team_color, jersey_num, is_away_team=False
    )
    
    return dcc.Graph(figure=fig)


@app.callback(
    Output('player-pass-map-graph-container-away', 'children'),
    Input('away-passer-dropdown', 'value'),
    State('store-df-match', 'data')
)
def update_away_passer_map(selected_player, stored_data_json):
    if not selected_player or not stored_data_json:
        return no_update

    df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
    team_color = ACOL
    
    all_passes = pass_processing.get_passes_df(df_processed.copy())
    player_passes = all_passes[all_passes['playerName'] == selected_player]

    # **Estrai il numero di maglia anche qui**
    jersey_num = '?'
    if not player_passes.empty:
        jersey_num_raw = player_passes['Mapped Jersey Number'].iloc[0]
        try:
            jersey_num = str(int(jersey_num_raw))
        except (ValueError, TypeError):
            pass

    fig = player_plots.plot_player_pass_map_plotly(
        player_passes, selected_player, team_color, jersey_num, is_away_team=True
    )
    
    return dcc.Graph(figure=fig)


# --- CALLBACK 2: For the SHOOTING secondary tabs ---
@app.callback(
    Output("shooting-secondary-tab-content", "children"),
    Input("shooting-secondary-tabs", "active_tab"),
    Input("store-player-stats-df", "data"),
    State("store-df-match", "data"),
)
def render_shooting_analysis_content(active_tab, player_stats_df_json, stored_match_data_json):
    if not player_stats_df_json:
        return dash_html.P("Player stats are loading...")
    
    # Reuse common styles
    common_textarea_style = {'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'}
    common_flex_column_style = {"display": "flex", "flexDirection": "column", "height": "calc(100vh - 280px)"}
    common_plot_area_style = {"flex": "1 1 75%", "minHeight": "350px", "overflow": "hidden"}
    common_comment_area_style = {"flex": "0 0 20%", "paddingTop": "15px", "overflowY": "auto"}

    if active_tab == "pa_shot_sequence_stats":
        try:
            player_stats_df = pd.read_json(io.StringIO(player_stats_df_json), orient='split')
            df_processed = pd.read_json(io.StringIO(stored_match_data_json['df']), orient='split')
            match_info = json.loads(stored_match_data_json['match_info'])
            home_team_name = match_info.get('hteamName', '')

            # **CHIAMATA ALLA NUOVA FUNZIONE PLOTLY**
            fig = player_plots.plot_shot_sequence_bar_plotly(
                player_stats_df,
                df_processed,
                home_team_name,
                hcol=HCOL,
                acol=ACOL,
                violet_col=VIOLET
            )
            
            # Layout con il grafico interattivo e la sezione commenti
            return dash_html.Div([
                dbc.Row(
                    dbc.Col(dcc.Graph(figure=fig), width=12)
                ),
                dbc.Row(
                    dbc.Col([
                        dash_html.Hr(),
                        dash_html.H5("Analyst Comments", className="mt-3"),
                        dcc.Textarea(
                            id="comment-shot-sequence-bar",
                            placeholder="Enter your analysis on shot sequences...",
                            style=common_textarea_style,
                            className="mb-2"
                        ),
                        dbc.Button("Save Comment", id="save-comment-shot-sequence-bar", color="info", size="sm"),
                        dash_html.Div(id="save-status-shot-sequence-bar", className="small d-inline-block ms-2 mt-2")
                    ], width=12),
                    className="mt-4"
                )
            ])
        except Exception as e:
            tb_str = traceback.format_exc()
            return dbc.Alert(f"Error generating shot sequence plot: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})

    elif active_tab == "pa_home_shot_contributor_map":
        return create_shot_contributor_layout('home', stored_match_data_json, player_stats_df_json)

    elif active_tab == "pa_away_shot_contributor_map":
        return create_shot_contributor_layout('away', stored_match_data_json, player_stats_df_json)
    
    return html.P(f"Content for {active_tab} not found.")

# --- 3. Aggiungi i NUOVI callback di aggiornamento ---
@app.callback(
    Output('shot-contributor-map-container-home', 'children'),
    Input('home-shot-contributor-dropdown', 'value'),
    State('store-df-match', 'data')
)
def update_home_shot_contributor_map(selected_player, stored_data_json):
    if not selected_player or not stored_data_json:
        return no_update

    df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
    all_passes = pass_processing.get_passes_df(df_processed.copy())
    
    # Filtra i passaggi ricevuti dal giocatore selezionato
    player_team_name = df_processed[df_processed['playerName'] == selected_player]['team_name'].iloc[0]
    received_passes = all_passes[
        (all_passes['receiver'] == selected_player) &
        (all_passes['team_name'] == player_team_name)
    ].copy()
    
    jersey_num = '?'
    player_info = df_processed[df_processed['playerName'] == selected_player].iloc[0]
    if pd.notna(player_info['Mapped Jersey Number']):
        jersey_num = str(int(player_info['Mapped Jersey Number']))

    fig = player_plots.plot_player_received_passes_plotly(
        received_passes, selected_player, HCOL, jersey_num, is_away_team=False
    )
    return dcc.Graph(figure=fig)

@app.callback(
    Output('shot-contributor-map-container-away', 'children'),
    Input('away-shot-contributor-dropdown', 'value'),
    State('store-df-match', 'data')
)
def update_away_shot_contributor_map(selected_player, stored_data_json):
    if not selected_player or not stored_data_json:
        return no_update

    df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
    all_passes = pass_processing.get_passes_df(df_processed.copy())

    player_team_name = df_processed[df_processed['playerName'] == selected_player]['team_name'].iloc[0]
    received_passes = all_passes[
        (all_passes['receiver'] == selected_player) &
        (all_passes['team_name'] == player_team_name)
    ].copy()
    
    jersey_num = '?'
    player_info = df_processed[df_processed['playerName'] == selected_player].iloc[0]
    if pd.notna(player_info['Mapped Jersey Number']):
        jersey_num = str(int(player_info['Mapped Jersey Number']))

    fig = player_plots.plot_player_received_passes_plotly(
        received_passes, selected_player, ACOL, jersey_num, is_away_team=True
    )
    return dcc.Graph(figure=fig)
    
    # elif active_tab == "pa_home_shot_contributor_map":
    #     print("  Rendering content for 'pa_home_shot_contributor_map'")
    #     if not player_stats_df_json:
    #         return dash_html.P("Player stats not yet available for home contributor map.", style={"color": "orange"})
            
    #     home_contributor_map = generate_team_top_shot_contributor_map_plot(stored_match_data_json, player_stats_df_json, is_for_home_team=True)

    #     return dash_html.Div([ # Flex container
    #         dash_html.Div(home_contributor_map, style=common_plot_area_style),
    #         dash_html.Div([ # Comment Area
    #             dash_html.Hr(),
    #             dash_html.H6("Comments for Home Contributor Map:", className="mt-3 text-white"),
    #             dcc.Textarea(id="comment-home-top-shot-contributor-map", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
    #             dbc.Button("Save Comment", id="save-comment-home-top-shot-contributor-map", color="info", size="sm", className="me-2"),
    #             dash_html.Div(id="save-status-home-top-shot-contributor-map", className="small d-inline-block")
    #         ], style=common_comment_area_style)
    #     ], style=common_flex_column_style)

    # elif active_tab == "pa_away_shot_contributor_map":
    #     print("  Rendering content for 'pa_away_shot_contributor_map'")
    #     if not player_stats_df_json:
    #         return dash_html.P("Player stats not yet available for away contributor map.", style={"color": "orange"})
            
    #     away_contributor_map = generate_team_top_shot_contributor_map_plot(stored_match_data_json, player_stats_df_json, is_for_home_team=False)

    #     return dash_html.Div([ # Flex container
    #         dash_html.Div(away_contributor_map, style=common_plot_area_style),
    #         dash_html.Div([ # Comment Area
    #             dash_html.Hr(),
    #             dash_html.H6("Comments for Away Contributor Map:", className="mt-3 text-white"),
    #             dcc.Textarea(id="comment-away-top-shot-contributor-map", placeholder="Comments...", style=common_textarea_style, className="mb-2"),
    #             dbc.Button("Save Comment", id="save-comment-away-top-shot-contributor-map", color="info", size="sm", className="me-2"),
    #             dash_html.Div(id="save-status-away-top-shot-contributor-map", className="small d-inline-block")
    #         ], style=common_comment_area_style)
    #     ], style=common_flex_column_style)
    # return dash_html.P(f"Content for {active_tab} not found.")


# --- CALLBACK 3: For the DEFENDING secondary tabs ---
@app.callback(
    Output("defending-secondary-tab-content", "children"),
    Input("defending-secondary-tabs", "active_tab"),
    Input("store-player-stats-df", "data"), 
    State("store-df-match", "data"),
)
def render_defending_analysis_content(active_tab, player_stats_df_json, stored_match_data_json):
    if not player_stats_df_json:
        return dash_html.P("Player stats are loading...")

    # Reuse common styles and logic
    common_textarea_style = {'width': '100%', 'height': 100, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'}
    common_flex_column_style = {"display": "flex", "flexDirection": "column", "height": "calc(100vh - 280px)"}
    common_plot_area_style = {"flex": "1 1 75%", "minHeight": "350px", "overflow": "hidden"}
    common_comment_area_style = {"flex": "0 0 20%", "paddingTop": "15px", "overflowY": "auto"}

    if active_tab == "pa_defender_stats":
        print("  Rendering content for 'pa_defender_stats' (Simple Static Plot)")
        if not player_stats_df_json:
            return dash_html.P("Player stats data not available.", style={"color": "orange"})

        try:
            player_stats_df = pd.read_json(player_stats_df_json, orient='split')
            df_processed = pd.read_json(stored_match_data_json['df'], orient='split')
            match_info = json.loads(stored_match_data_json['match_info'])
            home_team_name = match_info.get('hteamName', '')

            fig = plot_defender_stats_bar_plotly(
                player_stats_df,
                df_processed,
                home_team_name,
                hcol=HCOL,
                acol=ACOL,
                violet_col=VIOLET,
                green_col=GREEN # Passa il nuovo colore per gli aerials
            )

            return dash_html.Div([
                dbc.Row(
                    dbc.Col(dcc.Graph(figure=fig), width=12)
                ),
                dbc.Row(
                    dbc.Col([
                        dash_html.Hr(),
                        dash_html.H5("Analyst Comments", className="mt-3"),
                        dcc.Textarea(
                            id="comment-defender-stats-bar",
                            placeholder="Enter your analysis on top defenders...",
                            style=common_textarea_style,
                            className="mb-2"
                        ),
                        dbc.Button("Save Comment", id="save-comment-defender-stats-bar", color="info", size="sm"),
                        dash_html.Div(id="save-status-defender-stats-bar", className="small d-inline-block ms-2 mt-2")
                    ], width=12),
                    className="mt-4"
                )
            ])

        except Exception as e:
            tb_str = traceback.format_exc()
            return dbc.Alert(f"Error generating defender stats plot: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})

    elif active_tab == "pa_home_defender_map":
        # Genera il layout iniziale con il dropdown
        layout_content, dropdown_options, top_defender = player_plots.generate_defender_layout_and_data(
            stored_match_data_json, player_stats_df_json, is_for_home_team=True
        )
        if not dropdown_options: return layout_content # Mostra solo il messaggio di errore

        return dash_html.Div([
            dbc.Row(dbc.Col(dcc.Dropdown(
                id='home-defender-dropdown',
                options=dropdown_options,
                value=top_defender,
                style={'color': 'black'}
            ), md=6), justify="center", className="mb-3"),
            dash_html.Div(id='home-defender-output', children=layout_content)
        ])

    elif active_tab == "pa_away_defender_map":
        # Genera il layout iniziale per il team away
        layout_content, dropdown_options, top_defender = player_plots.generate_defender_layout_and_data(
            stored_match_data_json, player_stats_df_json, is_for_home_team=False
        )
        if not dropdown_options: return layout_content

        return dash_html.Div([
            dbc.Row(dbc.Col(dcc.Dropdown(
                id='away-defender-dropdown',
                options=dropdown_options,
                value=top_defender,
                style={'color': 'black'}
            ), md=6), justify="center", className="mb-3"),
            dash_html.Div(id='away-defender-output', children=layout_content)
        ])
        
    return dash_html.P(f"Content for {active_tab} not found.")



# MODIFIED: This callback now ONLY populates the store with player_stats_df.
# It is triggered when the main "Player Analysis" tab becomes active via the URL.
@app.callback(
    Output("store-player-stats-df", "data"),
    Input("store-df-match", "data"),
    Input("url-match-page", "search") # Trigger when main tab might change
)
def calculate_and_store_player_stats(stored_data_json, search_query):
    print(f"--- calculate_and_store_player_stats TRIGGERED --- Search: {search_query}")
    current_main_tab = "overview"
    if search_query and isinstance(search_query, str) and search_query.startswith("?tab="):
        current_main_tab = search_query.split("?tab=")[1].split("&")[0]

    if current_main_tab == "player_analysis" and stored_data_json:
        print("  Player Analysis main tab active, calculating player stats for store...")
        try:
            df_json_str = stored_data_json.get('df')
            if not df_json_str: return None # Important to return None to clear/indicate no data
            df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
            if df_processed.empty: return None

            assist_qualifier_col_name = 'Assist' 
            prog_pass_exclusions = ['cross', 'Launch', 'ThrowIn']
            
            player_stats_df = player_metrics.calculate_player_stats(
                df_processed.copy(),
                assist_qualifier_col=assist_qualifier_col_name,
                prog_pass_exclusions=prog_pass_exclusions
            )
            if not player_stats_df.empty:
                print("  Player stats calculated and being stored.")
                return player_stats_df.to_json(orient='split')
            else:
                print("  Player stats calculation resulted in empty DataFrame.")
                return None
        except Exception as e:
            print(f"Error in calculate_and_store_player_stats: {e}")
            return None
    
    print(f"  Not Player Analysis main tab ({current_main_tab}), or no base data. No update to player_stats_df.")
    return no_update # Or None if you want to clear it when not on player_analysis tab

# Helper generate_top_passer_stats_plot now ONLY generates the image
# It will take player_stats_df_json as an input if render_player_analysis_nested_content passes it
# OR it could take stored_match_data_json and recalculate if you prefer to keep it fully independent
# For now, let's assume render_player_analysis_nested_content passes the necessary data

def generate_top_passer_stats_plot(player_stats_df_json_for_plot):
    print("--- Helper generate_top_passer_stats_plot (using pre-calculated stats) EXECUTING ---")
    if not player_stats_df_json_for_plot:
        return dash_html.P("⚠ Player stats data missing for bar chart.", style={"color": "orange"})
    try:
        player_stats_df = pd.read_json(player_stats_df_json_for_plot, orient='split')
        if player_stats_df.empty:
            return dash_html.P("⚠ Player stats DataFrame is empty.", style={"color": "orange"})
        
        # # --- *** START: Pre-calculate Flags on df_processed *** ---
        # # This ensures flags are available before other metric/processing steps
        # print("Pre-calculating key pass/assist flags...")
        # # *** IMPORTANT: Verify 'Assist' is the correct column name ***
        # assist_qualifier_col='Assist' # ADJUST IF NEEDED
        # key_pass_values=[13, 14, 15]; assist_values=[16] # Values from original code

        # if assist_qualifier_col not in player_stats_df.columns:
        #     print(f"Warning: Assist qualifier column '{assist_qualifier_col}' not found in player_stats_df. Key Pass/Assist flags cannot be determined.")
        #     # Create empty/False columns so downstream code doesn't break, but results will be inaccurate
        #     player_stats_df['is_key_pass'] = False
        #     player_stats_df['is_assist'] = False
        # else:
        #     assist_qual_numeric = pd.to_numeric(player_stats_df[assist_qualifier_col], errors='coerce')
        #     # Calculate and ensure flags are boolean
        #     if 'is_key_pass' not in player_stats_df.columns:
        #         print("Info: Adding 'is_key_pass' flag to df_processed.")
        #         player_stats_df['is_key_pass'] = assist_qual_numeric.isin(key_pass_values) & (player_stats_df['type_name'] == 'Pass')
        #     player_stats_df['is_key_pass'] = player_stats_df['is_key_pass'].fillna(False).astype(bool)

        #     if 'is_assist' not in player_stats_df.columns:
        #         print("Info: Adding 'is_assist' flag to df_processed.")
        #         player_stats_df['is_assist'] = assist_qual_numeric.isin(assist_values) & (player_stats_df['type_name'] == 'Pass')
        #     player_stats_df['is_assist'] = player_stats_df['is_assist'].fillna(False).astype(bool)
        # print("Flags pre-calculation complete.")
        # # --- *** END: Pre-calculate Flags *** ---

        fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR) # Adjusted figsize
        player_plots.plot_passer_stats_bar(ax, player_stats_df.copy(), num_players=10) # Pass a copy
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        encoded_img = base64.b64encode(buf.read()).decode('ascii')
        img_src = f"data:image/png;base64,{encoded_img}"
        plt.close(fig)
        return dash_html.Img(src=img_src, style={"width": "100%", "maxWidth": "750px", "display": "block", "margin": "auto", "objectFit":"contain"})
    except Exception as e:
        tb_str = traceback.format_exc()
        return dash_html.P(f"❌ Error generating Top Passer Stats: {e}\n{tb_str}", style={"color": "red", "whiteSpace": "pre-wrap"})
    
# --- HELPER: Generate Individual Player Pass Map ---
def generate_player_pass_map_plot(stored_data_json, target_player_name, is_target_away_team):
    print(f"--- Helper generate_player_pass_map_plot for {target_player_name} EXECUTING ---")
    if not stored_data_json: return dash_html.P("⚠ No data for Player Pass Map.", style={"color": "orange"})
    if not target_player_name or target_player_name == "N/A": return dash_html.P("No player selected for pass map.", style={"color": "orange"})

    try:
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')
        if not df_json_str or not match_info_json_str: return dash_html.P("Data missing.")
        
        df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)
        if df_processed.empty: return dash_html.P("DataFrame empty.")

        # --- *** START: Pre-calculate Flags on df_processed *** ---
        # This ensures flags are available before other metric/processing steps
        print("Pre-calculating key pass/assist flags...")
        # *** IMPORTANT: Verify 'Assist' is the correct column name ***
        assist_qualifier_col='Assist' # ADJUST IF NEEDED
        key_pass_values=[13, 14, 15]; assist_values=[16] # Values from original code

        if assist_qualifier_col not in df_processed.columns:
            print(f"Warning: Assist qualifier column '{assist_qualifier_col}' not found in df_processed. Key Pass/Assist flags cannot be determined.")
            # Create empty/False columns so downstream code doesn't break, but results will be inaccurate
            df_processed['is_key_pass'] = False
            df_processed['is_assist'] = False
        else:
            assist_qual_numeric = pd.to_numeric(df_processed[assist_qualifier_col], errors='coerce')
            # Calculate and ensure flags are boolean
            if 'is_key_pass' not in df_processed.columns:
                print("Info: Adding 'is_key_pass' flag to df_processed.")
                df_processed['is_key_pass'] = assist_qual_numeric.isin(key_pass_values) & (df_processed['type_name'] == 'Pass')
            df_processed['is_key_pass'] = df_processed['is_key_pass'].fillna(False).astype(bool)

            if 'is_assist' not in df_processed.columns:
                print("Info: Adding 'is_assist' flag to df_processed.")
                df_processed['is_assist'] = assist_qual_numeric.isin(assist_values) & (df_processed['type_name'] == 'Pass')
            df_processed['is_assist'] = df_processed['is_assist'].fillna(False).astype(bool)
        print("Flags pre-calculation complete.")
        # --- *** END: Pre-calculate Flags *** ---

        # Get all passes first (includes outcome, is_key_pass, is_assist from preprocess)
        all_passes_df = pass_processing.get_passes_df(df_processed.copy())
        if all_passes_df.empty: return dash_html.P(f"No pass data found at all.", style={"color": "orange"})

        df_player_passes = all_passes_df[all_passes_df['playerName'] == target_player_name].copy()

        if df_player_passes.empty:
            return dash_html.P(f"⚠ No passes found for player: {target_player_name}.", style={"color": "orange"})

        # Determine team color
        team_color = getattr(config, 'DEFAULT_HCOL', HCOL) # Default to home color
        if 'team_name' in df_player_passes.columns and not df_player_passes.empty:
            player_team_name = df_player_passes['team_name'].iloc[0]
            if player_team_name == match_info.get('ateamName'):
                team_color = getattr(config, 'DEFAULT_ACOL', ACOL)
        elif is_target_away_team: # Fallback if team_name not in player passes df
            team_color = getattr(config, 'DEFAULT_ACOL', ACOL)


        # Ensure 'is_key_pass' and 'is_assist' are boolean and present
        # These should be prepared by preprocess.process_opta_events
        if 'is_key_pass' not in df_player_passes.columns: df_player_passes['is_key_pass'] = False
        if 'is_assist' not in df_player_passes.columns: df_player_passes['is_assist'] = False
        df_player_passes['is_key_pass'] = df_player_passes['is_key_pass'].fillna(False).astype(bool)
        df_player_passes['is_assist'] = df_player_passes['is_assist'].fillna(False).astype(bool)
        # Ensure 'outcome' exists
        if 'outcome' not in df_player_passes.columns:
             return dash_html.P(f"⚠ 'outcome' column missing for player {target_player_name}'s passes.", style={"color": "red"})


        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG_COLOR) # Adjust figsize
        player_plots.plot_player_pass_map(ax, df_player_passes, target_player_name, team_color, is_target_away_team) # Uses global GREEN, VIOLET
        
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=90, bbox_inches='tight'); buf.seek(0)
        img_src = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"
        plt.close(fig)
        return dash_html.Img(src=img_src, style={"width": "100%", "maxWidth": "700px", "display": "block", "margin": "auto"})
    except Exception as e:
        tb_str = traceback.format_exc()
        return dash_html.P(f"❌ Error generating Pass Map for {target_player_name}: {e}\n{tb_str}", style={"color": "red", "whiteSpace": "pre-wrap"})


# --- Add Comment Callbacks for Player Analysis Plots ---
@app.callback(Output("store-comment-top-passers-bar", "data"), Output("save-status-top-passers-bar", "children"),
              Input("save-comment-top-passers-bar", "n_clicks"),
              State("comment-top-passers-bar", "value"), State("url-match-page", "pathname"), State("store-comment-top-passers-bar", "data"),
              prevent_initial_call=True)
def save_comment_top_passers(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_top_passers_stats"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.",color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-top-passers-bar", "value"), Input("store-comment-top-passers-bar", "data"), Input("url-match-page", "pathname"))
def load_comment_top_passers(data, pn):
    key = get_comment_key(pn, "pa_top_passers_stats")
    if not key or data is None: return ""
    return data.get(key, "")

@app.callback(Output("store-comment-home-top-passer-map", "data"), Output("save-status-home-top-passer-map", "children"),
              Input("save-comment-home-top-passer-map", "n_clicks"),
              State("comment-home-top-passer-map", "value"), State("url-match-page", "pathname"), State("store-comment-home-top-passer-map", "data"),
              prevent_initial_call=True)
def save_comment_home_pass_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_home_passer_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.",color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-home-top-passer-map", "value"), Input("store-comment-home-top-passer-map", "data"), Input("url-match-page", "pathname"))
def load_comment_home_pass_map(data, pn):
    key = get_comment_key(pn, "pa_home_passer_map")
    if not key or data is None: return ""
    return data.get(key, "")

@app.callback(Output("store-comment-away-top-passer-map", "data"), Output("save-status-away-top-passer-map", "children"),
              Input("save-comment-away-top-passer-map", "n_clicks"),
              State("comment-away-top-passer-map", "value"), State("url-match-page", "pathname"), State("store-comment-away-top-passer-map", "data"),
              prevent_initial_call=True)
def save_comment_away_pass_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_away_passer_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.",color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-away-top-passer-map", "value"), Input("store-comment-away-top-passer-map", "data"), Input("url-match-page", "pathname"))
def load_comment_away_pass_map(data, pn):
    key = get_comment_key(pn, "pa_away_passer_map")
    if not key or data is None: return ""
    return data.get(key, "")

# --------------------------------------

def generate_shot_sequence_bar_plot(player_stats_df_json_for_plot):
    """Generates the shot sequence involvement bar chart."""
    print("--- Helper generate_shot_sequence_bar_plot EXECUTING ---")
    if not player_stats_df_json_for_plot:
        return dash_html.P("⚠ Player stats data missing for shot sequence chart.", style={"color": "orange"})
    try:
        player_stats_df = pd.read_json(player_stats_df_json_for_plot, orient='split')
        if player_stats_df.empty:
            return dash_html.P("⚠ Player stats DataFrame is empty.", style={"color": "orange"})

        fig, ax = plt.subplots(figsize=(10, 7), facecolor=BG_COLOR)
        # Call the new plot function from your player_plots module
        player_plots.plot_shot_sequence_bar(ax, player_stats_df.copy(), num_players=10)
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
        buf.seek(0)
        encoded_img = base64.b64encode(buf.read()).decode('ascii')
        img_src = f"data:image/png;base64,{encoded_img}"
        plt.close(fig)
        return dash_html.Img(src=img_src, style={"width": "100%", "maxWidth": "750px", "display": "block", "margin": "auto", "objectFit":"contain"})
    except Exception as e:
        tb_str = traceback.format_exc()
        return dash_html.P(f"❌ Error generating Shot Sequence Stats: {e}\n{tb_str}", style={"color": "red", "whiteSpace": "pre-wrap"})

def generate_team_top_shot_contributor_map_plot(stored_data_json, player_stats_df_json, is_for_home_team):
    """Finds the top shot contributor for a specific team and plots their received passes."""
    team_type = "Home" if is_for_home_team else "Away"
    print(f"--- Helper generate_team_top_shot_contributor_map_plot for {team_type} Team EXECUTING ---")
    if not stored_data_json or not player_stats_df_json:
        return dash_html.P("⚠ Data missing for Top Contributor Map.", style={"color": "orange"})

    try:
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')
        df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)
        player_stats_df = pd.read_json(player_stats_df_json, orient='split')

        if df_processed.empty or player_stats_df.empty:
            return dash_html.P("⚠ DataFrame(s) empty.", style={"color": "orange"})
        
        # --- *** START: Pre-calculate Flags on df_processed *** ---
        # This ensures flags are available before other metric/processing steps
        print("Pre-calculating key pass/assist flags...")
        # *** IMPORTANT: Verify 'Assist' is the correct column name ***
        assist_qualifier_col='Assist' # ADJUST IF NEEDED
        key_pass_values=[13, 14, 15]; assist_values=[16] # Values from original code

        if assist_qualifier_col not in df_processed.columns:
            print(f"Warning: Assist qualifier column '{assist_qualifier_col}' not found in df_processed. Key Pass/Assist flags cannot be determined.")
            # Create empty/False columns so downstream code doesn't break, but results will be inaccurate
            df_processed['is_key_pass'] = False
            df_processed['is_assist'] = False
        else:
            assist_qual_numeric = pd.to_numeric(df_processed[assist_qualifier_col], errors='coerce')
            # Calculate and ensure flags are boolean
            if 'is_key_pass' not in df_processed.columns:
                print("Info: Adding 'is_key_pass' flag to df_processed.")
                df_processed['is_key_pass'] = assist_qual_numeric.isin(key_pass_values) & (df_processed['type_name'] == 'Pass')
            df_processed['is_key_pass'] = df_processed['is_key_pass'].fillna(False).astype(bool)

            if 'is_assist' not in df_processed.columns:
                print("Info: Adding 'is_assist' flag to df_processed.")
                df_processed['is_assist'] = assist_qual_numeric.isin(assist_values) & (df_processed['type_name'] == 'Pass')
            df_processed['is_assist'] = df_processed['is_assist'].fillna(False).astype(bool)
        print("Flags pre-calculation complete.")
        # --- *** END: Pre-calculate Flags *** ---

        # 1. Identify the target team and its players
        team_name = match_info.get('hteamName') if is_for_home_team else match_info.get('ateamName')
        if not team_name:
            return dash_html.P(f"Could not determine {team_type} team name.", style={"color":"red"})
            
        team_player_names = df_processed[df_processed['team_name'] == team_name]['playerName'].unique()
        team_player_stats = player_stats_df[player_stats_df.index.isin(team_player_names)]

        # 2. Find the top player from that team's shot sequence stats
        if 'Shooting Seq Total' not in team_player_stats.columns:
            return dash_html.P("⚠ 'Shooting Seq Total' column not found.", style={"color": "red"})
        
        top_players_df = team_player_stats.sort_values('Shooting Seq Total', ascending=False)
        if top_players_df.empty:
            return dash_html.P(f"Could not determine top shot contributor for {team_name}.", style={"color":"orange"})
        target_player_name = top_players_df.index[0]

        # 3. Get all passes for the plot function
        all_passes_df = pass_processing.get_passes_df(df_processed.copy())
        if all_passes_df.empty:
            return dash_html.P("No pass data found to generate map.", style={"color": "orange"})

        # 4. Determine team color and orientation
        team_color = HCOL if is_for_home_team else ACOL
        is_away_team = not is_for_home_team
        
        # 5. Generate the plot
        fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG_COLOR)
        player_plots.plot_player_received_passes(ax, all_passes_df.copy(), target_player_name, team_color, is_away_team)
        
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=90, bbox_inches='tight', facecolor=fig.get_facecolor()); buf.seek(0)
        img_src = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"
        plt.close(fig)
        
        return dash_html.Img(src=img_src, style={"width": "100%", "maxWidth": "700px", "display": "block", "margin": "auto"})
    except Exception as e:
        tb_str = traceback.format_exc()
        return dash_html.P(f"❌ Error generating {team_type} Top Contributor Map: {e}\n{tb_str}", style={"color": "red", "whiteSpace": "pre-wrap"})
    
def create_shot_contributor_layout(team_type, stored_match_data_json, player_stats_df_json):
    """
    Crea il layout (Dropdown + Grafico) per la mappa dei passaggi ricevuti
    dai giocatori di una squadra.
    """
    try:
        match_info = json.loads(stored_match_data_json['match_info'])
        df_processed = pd.read_json(io.StringIO(stored_match_data_json['df']), orient='split')
        player_stats_df = pd.read_json(io.StringIO(player_stats_df_json), orient='split')

        is_away = (team_type == 'away')
        team_name = match_info.get('ateamName') if is_away else match_info.get('hteamName')
        team_color = ACOL if is_away else HCOL
        
        team_players = df_processed[df_processed['team_name'] == team_name].dropna(subset=['playerName']).drop_duplicates('playerName')
        if team_players.empty:
            return dbc.Alert(f"No players found for {team_name}", color="warning")

        # player_jersey_map = team_players.set_index('playerName')['Mapped Jersey Number']
        # sorted_player_names = sorted(player_jersey_map.index.tolist())
        
        # dropdown_options = [{'label': f"#{int(player_jersey_map.get(name, '?')) if str(player_jersey_map.get(name, '?')).isdigit() else '?'} - {name}", 'value': name} for name in sorted_player_names]

        player_jersey_map = team_players.drop_duplicates('playerName').set_index('playerName')['Mapped Jersey Number']
        
        sorted_player_names = sorted(player_jersey_map.index.tolist())

        dropdown_options = []
        for name in sorted_player_names:
            jersey_raw = player_jersey_map.get(name)
            try:
                jersey = str(int(jersey_raw))
            except (ValueError, TypeError):
                jersey = '?'
            dropdown_options.append({'label': f"#{jersey} - {name}", 'value': name})

        # Filtra le statistiche solo per i giocatori di questa squadra
        team_player_stats = player_stats_df[player_stats_df.index.isin(player_jersey_map.index)].copy()
        
        top_contributor_name = None
        if not team_player_stats.empty:
            # **NUOVA LOGICA: CALCOLO DEL PUNTEGGIO PONDERATO**
            weights = {'Shots': 3, 'Shot Assists': 2, 'Buildup to Shot': 1}
            
            # Assicurati che le colonne esistano prima di calcolare
            for col in weights.keys():
                if col not in team_player_stats.columns:
                    team_player_stats[col] = 0 # Aggiungi la colonna con zeri se manca

            team_player_stats['Weighted Score'] = (
                team_player_stats['Shots'] * weights['Shots'] +
                team_player_stats['Shot Assists'] * weights['Shot Assists'] +
                team_player_stats['Buildup to Shot'] * weights['Buildup to Shot']
            )
            
            # Trova il giocatore con il punteggio ponderato più alto
            top_contributor_name = team_player_stats['Weighted Score'].idxmax()

        # --- LOGICA PER GENERARE IL GRAFICO INIZIALE (invariata) ---
        initial_graph = dash_html.Div(f"Select a player to see their received passes map. Top contributor is {top_contributor_name or 'N/A'}.")
        if top_contributor_name:
            all_passes = pass_processing.get_passes_df(df_processed.copy())
            received_passes = all_passes[(all_passes['receiver'] == top_contributor_name) & (all_passes['team_name'] == team_name)].copy()
            
            jersey_num_raw = player_jersey_map.get(top_contributor_name)
            try: jersey_num = str(int(jersey_num_raw))
            except (ValueError, TypeError): jersey_num = '?'
            
            fig = player_plots.plot_player_received_passes_plotly(received_passes, top_contributor_name, team_color, jersey_num, is_away)
            initial_graph = dcc.Graph(figure=fig)
        
        return dash_html.Div([
            dbc.Row(
                dbc.Col(dcc.Dropdown(
                    id=f'{team_type}-shot-contributor-dropdown',
                    options=dropdown_options,
                    value=top_contributor_name,
                    placeholder="Select a player...",
                    style={'color': 'black'}
                ), md=6),
                justify="center", className="my-3"
            ),
            dcc.Loading(
                dash_html.Div(id=f'shot-contributor-map-container-{team_type}', children=initial_graph)
            )
        ])

    except Exception as e:
        tb_str = traceback.format_exc()
        return dbc.Alert(f"Error creating layout for {team_type} shot contributor map: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})

# --- COMMENT CALLBACKS FOR SHOT SEQUENCE BAR CHART ---
@app.callback(Output("store-comment-shot-sequence-bar", "data"), Output("save-status-shot-sequence-bar", "children"),
              Input("save-comment-shot-sequence-bar", "n_clicks"),
              State("comment-shot-sequence-bar", "value"), State("url-match-page", "pathname"), State("store-comment-shot-sequence-bar", "data"),
              prevent_initial_call=True)
def save_comment_shot_seq_bar(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_shot_sequence_stats"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-shot-sequence-bar", "value"), Input("store-comment-shot-sequence-bar", "data"), Input("url-match-page", "pathname"))
def load_comment_shot_seq_bar(data, pn):
    key = get_comment_key(pn, "pa_shot_sequence_stats")
    if not key or data is None: return ""
    return data.get(key, "")

# --- COMMENT CALLBACKS FOR HOME TOP SHOT CONTRIBUTOR MAP ---
@app.callback(Output("store-comment-home-top-shot-contributor-map", "data"), Output("save-status-home-top-shot-contributor-map", "children"),
              Input("save-comment-home-top-shot-contributor-map", "n_clicks"),
              State("comment-home-top-shot-contributor-map", "value"), State("url-match-page", "pathname"), State("store-comment-home-top-shot-contributor-map", "data"),
              prevent_initial_call=True)
def save_comment_home_shot_contrib_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_home_shot_contributor_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-home-top-shot-contributor-map", "value"), Input("store-comment-home-top-shot-contributor-map", "data"), Input("url-match-page", "pathname"))
def load_comment_home_shot_contrib_map(data, pn):
    key = get_comment_key(pn, "pa_home_shot_contributor_map")
    if not key or data is None: return ""
    return data.get(key, "")

# --- COMMENT CALLBACKS FOR AWAY TOP SHOT CONTRIBUTOR MAP ---
@app.callback(Output("store-comment-away-top-shot-contributor-map", "data"), Output("save-status-away-top-shot-contributor-map", "children"),
              Input("save-comment-away-top-shot-contributor-map", "n_clicks"),
              State("comment-away-top-shot-contributor-map", "value"), State("url-match-page", "pathname"), State("store-comment-away-top-shot-contributor-map", "data"),
              prevent_initial_call=True)
def save_comment_away_shot_contrib_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_away_shot_contributor_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-away-top-shot-contributor-map", "value"), Input("store-comment-away-top-shot-contributor-map", "data"), Input("url-match-page", "pathname"))
def load_comment_away_shot_contrib_map(data, pn):
    key = get_comment_key(pn, "pa_away_shot_contributor_map")
    if not key or data is None: return ""
    return data.get(key, "")

# -----------------------------------------

def plot_defender_stats_bar_plotly(player_stats_df, df_processed, home_team_name, hcol='tomato', acol='skyblue', violet_col='#a369ff', green_col='#69f900', num_players=10):
    """
    Crea un bar chart Plotly interattivo per le statistiche difensive,
    con ranking ponderato e etichette colorate.
    """
    req_cols = ['Tackles Won', 'Interceptions', 'Clearances']
    if not all(col in player_stats_df.columns for col in req_cols):
        # ... gestione errore ...
        return go.Figure() # ... con messaggio di errore

    # --- Calcolo del Punteggio Ponderato ---
    weights = {
        'Tackles Won': 3,
        'Interceptions': 3,
        'Aerials Won': 2,
        'Ball recovery': 2,
        'Clearances': 1
    }
    df_with_score = player_stats_df.copy()
    
    # Assicura che le colonne esistano
    for col in weights.keys():
        if col not in df_with_score.columns:
            df_with_score[col] = 0
            
    df_with_score['Weighted Defensive Score'] = sum(df_with_score[col] * w for col, w in weights.items())

    # 1. Ordina per punteggio ponderato, poi inverti per il plot
    top_players_sorted = df_with_score.sort_values('Weighted Defensive Score', ascending=False).head(num_players)
    plot_df = top_players_sorted.iloc[::-1]

    # 2. Mappe per i dati dei giocatori
    player_to_team_map = df_processed.drop_duplicates('playerName').set_index('playerName')['team_name'].to_dict()
    player_jersey_map = df_processed.drop_duplicates('playerName').set_index('playerName')['Mapped Jersey Number'].to_dict()

    # --- Creazione della Figura ---
    fig = go.Figure()

    # Aggiungi le tracce
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Clearances'], name='Clearances', orientation='h', marker_color=acol))
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Ball recovery'], name='Ball Recoveries', orientation='h', marker_color='orange'))
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Aerials Won'], name='Aerials Won', orientation='h', marker_color=green_col))
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Interceptions'], name='Interceptions', orientation='h', marker_color=violet_col))
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Tackles Won'], name='Tackles Won', orientation='h', marker_color=hcol))
    
    # --- Configurazione del Layout ---
    fig.update_layout(
        title_text='Top Defenders by Weighted Score',
        barmode='stack',
        yaxis=dict(showticklabels=False), # Nascondi etichette, le creiamo con annotazioni
        xaxis=dict(title='Total Actions (raw count)'), # L'asse X mostra ancora il conteggio grezzo
        plot_bgcolor='#2E3439', paper_bgcolor='#2E3439',
        font_color='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=160, r=20, t=80, b=40),
        height=800,
        annotations=[]
    )
    
    # Aggiungi le etichette manualmente come annotazioni
    for player_name in plot_df.index:
        team_name = player_to_team_map.get(player_name)
        label_color = hcol if team_name == home_team_name else acol
        
        jersey_raw = player_jersey_map.get(player_name)
        try: jersey = str(int(jersey_raw))
        except (ValueError, TypeError): jersey = '?'
        
        label_text = f"<b>#{jersey} - {player_name}</b>"

        fig.add_annotation(
            x=0, y=player_name,
            xref="paper", yref="y",
            text=label_text,
            showarrow=False, xanchor="right", align="right",
            font=dict(color=label_color, size=12),
            xshift=-10
        )

    annotations = []
    for player_name in plot_df.index:
        # Inverti l'ordine per il calcolo degli offset (left)
        data_for_player = plot_df.loc[player_name]
        ordered_metrics = ['Clearances', 'Ball recovery', 'Aerials Won', 'Interceptions', 'Tackles Won']
        
        current_offset = 0
        for metric in ordered_metrics:
            value = data_for_player.get(metric, 0)
            if value > 0:
                # Posiziona l'annotazione al centro del segmento di barra
                annotations.append(dict(
                    x=current_offset + value / 2,
                    y=player_name,
                    text=f"<b>{int(value)}</b>",
                    showarrow=False,
                    font=dict(color='white', size=10)
                ))
            current_offset += value
            
    fig.update_layout(annotations=fig.layout.annotations + tuple(annotations))

    return fig


def generate_team_top_defender_map_plot(stored_data_json, player_stats_df_json, is_for_home_team, selected_player=None):
    """
    Finds a team's top defender (or uses a selected player), generates the interactive map
    and success rate table, and returns them along with a list of the team's defenders.
    """
    team_type = "Home" if is_for_home_team else "Away"
    print(f"--- Helper generate_team_top_defender_map_plot (Components) for {team_type} Team EXECUTING ---")
    
    try:
        # --- 1. Data Loading (same as before) ---
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')
        df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)
        player_stats_df = pd.read_json(player_stats_df_json, orient='split')
        
        team_name = match_info.get('hteamName') if is_for_home_team else match_info.get('ateamName')
        team_player_names_all = df_processed[df_processed['team_name'] == team_name]['playerName'].unique()

        # --- 2. Create the list of players for the dropdown ---
        # Filter for players who made at least one defensive action to populate the dropdown
        DEFENSIVE_ACTION_TYPES = ['Tackle', 'Interception', 'Ball recovery', 'Clearance', 'Foul', 'Aerial', 'Blocked pass']
        defensive_players_df = df_processed[
            (df_processed['playerName'].isin(team_player_names_all)) &
            (df_processed['type_name'].isin(DEFENSIVE_ACTION_TYPES))
        ]
        defensive_players_df = defensive_players_df.dropna(subset=['playerName'])
        # Get the unique names of players who made these actions
        if defensive_players_df.empty:
            players_for_dropdown = []
        else:
            # Create a mapping of playerName to jersey number.
            # We drop duplicates to get one entry per player.
            player_jersey_map = defensive_players_df[['playerName', 'Mapped Jersey Number']].drop_duplicates('playerName').set_index('playerName')['Mapped Jersey Number']
            
            # Sort the player names alphabetically
            sorted_player_names = sorted(player_jersey_map.index.tolist())
            
            # Build the list of dictionaries for the dropdown
            players_for_dropdown = [
                {
                    'label': f"#{player_jersey_map.get(name, '?')} - {name}", # Format the label
                    'value': name  # The value remains the name
                }
                for name in sorted_player_names
            ]

        # --- 3. Determine the Target Player ---
        if selected_player:
            target_player_name = selected_player
        else:
            # Default to the top defender
            team_player_stats = player_stats_df[player_stats_df.index.isin(team_player_names_all)]
            top_defenders_df = team_player_stats.sort_values('Defensive Actions Total', ascending=False)
            
            # Check if there are any defenders to select as default
            if top_defenders_df.empty:
                # If there are no defenders with stats, check if there are any in the dropdown list
                if not players_for_dropdown:
                    return dash_html.P(f"No defensive actions recorded for {team_name}."), None, []
                # Otherwise, default to the first player in the dropdown
                target_player_name = players_for_dropdown[0]['value']
            else:
                 target_player_name = top_defenders_df.index[0]

        # --- 4. Generate plots and tables for the target player ---
        df_player_def_actions = defensive_players_df[defensive_players_df['playerName'] == target_player_name].copy()
        
        stats_df = player_metrics.calculate_defensive_action_rates(df_player_def_actions)

        # Define the desired order for the table rows
        action_hierarchy_order = [
            'Tackle',
            'Interception',
            'Aerial',
            'Ball recovery',
            'Clearance',
            'Foul' # Keep foul at the bottom
        ]
        
        # Reorder the DataFrame based on the hierarchy.
        # We use pd.Categorical to enforce a custom sort order on the 'Action' column.
        if not stats_df.empty:
            stats_df['Action'] = pd.Categorical(stats_df['Action'], categories=action_hierarchy_order, ordered=True)
            stats_df = stats_df.sort_values('Action')
        
        stats_table = dash_table.DataTable(
            data=stats_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in stats_df.columns],
            style_cell={'backgroundColor': '#343A40', 'color': 'white', 'textAlign': 'center', 'border': '1px solid #454D55'},
            style_header={'backgroundColor': '#454D55', 'color': 'white', 'fontWeight': 'bold'},
            style_as_list_view=True,
        )
        
        team_color = HCOL if is_for_home_team else ACOL
        is_away_team = not is_for_home_team
        fig = player_plots.plot_player_defensive_actions_plotly(df_player_def_actions, target_player_name, team_color, is_away_team)
        interactive_map = dcc.Graph(figure=fig, config={'displayModeBar': False})

        # --- 5. Return all three components ---
        return interactive_map, stats_table, players_for_dropdown

    except Exception as e:
        tb_str = traceback.format_exc()
        error_message = dash_html.P(f"❌ Error generating {team_type} Top Defender Layout: {e}\n{tb_str}", style={"color": "red", "whiteSpace": "pre-wrap"})
        return error_message, None, []
    
# --- COMMENT CALLBACKS FOR DEFENDER STATS BAR CHART ---
@app.callback(Output("store-comment-defender-stats-bar", "data"), Output("save-status-defender-stats-bar", "children"),
              Input("save-comment-defender-stats-bar", "n_clicks"),
              State("comment-defender-stats-bar", "value"), State("url-match-page", "pathname"), State("store-comment-defender-stats-bar", "data"),
              prevent_initial_call=True)
def save_comment_defender_stats_bar(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_defender_stats"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-defender-stats-bar", "value"), Input("store-comment-defender-stats-bar", "data"), Input("url-match-page", "pathname"))
def load_comment_defender_stats_bar(data, pn):
    key = get_comment_key(pn, "pa_defender_stats")
    if not key or data is None: return ""
    return data.get(key, "")

@app.callback(
    Output('home-defender-output', 'children'),
    Input('home-defender-dropdown', 'value'),
    State('store-df-match', 'data'),
    State('store-player-stats-df', 'data'),
    prevent_initial_call=True
)
def update_home_defender_view(selected_player, stored_match_data_json, player_stats_df_json):
    if not selected_player:
        return html.P("Select a player from the dropdown to view their map.")

    # Chiama la stessa funzione helper, ma passando il giocatore selezionato
    layout_content, _, _ = player_plots.generate_defender_layout_and_data(
        stored_match_data_json, player_stats_df_json, is_for_home_team=True, selected_player=selected_player
    )
    return layout_content

@app.callback(
    Output('away-defender-output', 'children'),
    Input('away-defender-dropdown', 'value'),
    State('store-df-match', 'data'),
    State('store-player-stats-df', 'data'),
    prevent_initial_call=True
)
def update_away_defender_view(selected_player, stored_match_data_json, player_stats_df_json):
    if not selected_player:
        return html.P("Select a player from the dropdown to view their map.")
        
    layout_content, _, _ = player_plots.generate_defender_layout_and_data(
        stored_match_data_json, player_stats_df_json, is_for_home_team=False, selected_player=selected_player
    )
    return layout_content

# --- COMMENT CALLBACKS FOR HOME TOP DEFENDER MAP ---
@app.callback(Output("store-comment-home-top-defender-map", "data"), Output("save-status-home-top-defender-map", "children"),
              Input("save-comment-home-top-defender-map", "n_clicks"),
              State("comment-home-top-defender-map", "value"), State("url-match-page", "pathname"), State("store-comment-home-top-defender-map", "data"),
              prevent_initial_call=True)
def save_comment_home_defender_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_home_defender_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-home-top-defender-map", "value"), Input("store-comment-home-top-defender-map", "data"), Input("url-match-page", "pathname"))
def load_comment_home_defender_map(data, pn):
    key = get_comment_key(pn, "pa_home_defender_map")
    if not key or data is None: return ""
    return data.get(key, "")

# --- COMMENT CALLBACKS FOR AWAY TOP DEFENDER MAP ---
@app.callback(Output("store-comment-away-top-defender-map", "data"), Output("save-status-away-top-defender-map", "children"),
              Input("save-comment-away-top-defender-map", "n_clicks"),
              State("comment-away-top-defender-map", "value"), State("url-match-page", "pathname"), State("store-comment-away-top-defender-map", "data"),
              prevent_initial_call=True)
def save_comment_away_defender_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_away_defender_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-away-top-defender-map", "value"), Input("store-comment-away-top-defender-map", "data"), Input("url-match-page", "pathname"))
def load_comment_away_defender_map(data, pn):
    key = get_comment_key(pn, "pa_away_defender_map")
    if not key or data is None: return ""
    return data.get(key, "")

# -----------------------------------------

def generate_player_defensive_heatmap(stored_data_json, player_name):
    """Generates an interactive defensive action heatmap for a single player."""
    if not stored_data_json or not player_name:
        return go.Figure() # Return empty figure if no data

    try:
        df_processed = pd.read_json(stored_data_json['df'], orient='split')
        
        DEFENSIVE_ACTION_TYPES = ['Tackle', 'Interception', 'Ball recovery', 'Clearance', 'Foul', 'Aerial', 'Blocked pass']
        df_player_actions = df_processed[
            (df_processed['playerName'] == player_name) &
            (df_processed['type_name'].isin(DEFENSIVE_ACTION_TYPES))
        ]

        if df_player_actions.empty:
            fig = go.Figure()
            fig.add_annotation(x=50, y=50, text=f"No defensive actions for<br>{player_name}", showarrow=False, font=dict(size=14, color='white'))
        else:
            fig = go.Figure(go.Densitymapbox(
                lon=df_player_actions['x'],
                lat=df_player_actions['y'],
                radius=20, # Adjust radius for desired "blotchiness"
                colorscale="Viridis",
                showscale=False
            ))
        
        # --- Layout for the heatmap pitch ---
        fig.update_layout(
            mapbox_style="white-bg", # Use a blank background
            mapbox_layers=[{
                "below": 'traces',
                "sourcetype": "raster", # Not really used, just to enable layers
            }],
            mapbox_center={"lon": 50, "lat": 50},
            mapbox_zoom=4,
            plot_bgcolor='#2E3439',
            paper_bgcolor='#2E3439',
            margin={"r":0,"t":0,"l":0,"b":0},
            height=300 # A smaller pitch for the side view
        )
        return fig
        
    except Exception:
        return go.Figure() # Return empty figure on error
    
# ----------------------------------------

# @app.callback(
#     Output("store-buildup-filter", "data"),
#     Input({'type': 'buildup-filter', 'filter_type': ALL, 'value': ALL}, 'n_clicks'),
#     Input("buildup-reset-filter-btn", "n_clicks"),
#     prevent_initial_call=True
# )
# def update_buildup_filter(card_clicks, reset_click):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         return dash.no_update

#     triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

#     if triggered_id == "buildup-reset-filter-btn":
#         return None  # Reset filter

#     try:
#         triggered_dict = json.loads(triggered_id.replace("'", '"'))
#         filter_type = triggered_dict.get("filter_type")
#         value = triggered_dict.get("value")
#         return {"type": filter_type, "value": value}
#     except Exception as e:
#         print(f"[Errore filtro buildup] ID non valido: {triggered_id}, errore: {e}")
#         return dash.no_update  
    
@app.callback(
    Output("store-buildup-filter", "data"),
    Input({'type': 'buildup-filter', 'filter_type': ALL, 'value': ALL}, 'n_clicks'),
    Input("buildup-reset-filter-btn", "n_clicks"),
    State("store-buildup-filter", "data"),
    prevent_initial_call=True
)
def update_multi_filter(card_clicks, reset_click, current_filter):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    # RESET → empty filter
    if triggered_id == "buildup-reset-filter-btn":
        return {}

    # Click on a filter card
    try:
        triggered_dict = json.loads(triggered_id.replace("'", '"'))
        filter_type = triggered_dict.get("filter_type")
        value = triggered_dict.get("value")

        current_filter = current_filter or {}
        
        if current_filter.get(filter_type) == value:
            current_filter.pop(filter_type)
        else:
            current_filter[filter_type] = value

        return current_filter if current_filter else None
    except Exception as e:
        print(f"[Filtro multiplo] Errore nel parsing dell'ID: {triggered_id} → {e}")
        return dash.no_update
    
@app.callback(
    Output("buildup-tab-content", "children"),
    Input("buildup-primary-tabs", "active_tab"),
    Input("store-buildup-filter", "data"),
    State("store-df-match", "data")
)
def render_buildup_content(active_buildup_tab, active_filter, stored_data_json):
    """
    This single callback handles rendering the layout for both the Home and Away
    buildup sub-tabs. It sets up the structure for the interactive carousel.
    """
    if not stored_data_json:
        return dbc.Alert("Match data loading...", color="info", className="mt-3")

    try:
        df_processed = pd.read_json(stored_data_json['df'], orient='split')
        match_info = json.loads(stored_data_json['match_info'])
        HTEAM_NAME = match_info.get('hteamName')
        ATEAM_NAME = match_info.get('ateamName')

        # --- 1. Determine which team to analyze based on the active tab ---
        if active_buildup_tab == 'buildup_home':
            attacking_team, defending_team, team_color, is_away = HTEAM_NAME, ATEAM_NAME, HCOL, False
        else:  # 'buildup_away'
            attacking_team, defending_team, team_color, is_away = ATEAM_NAME, HTEAM_NAME, ACOL, True

        # --- 2. Find and prepare all buildup sequences for that team ---
        triggers = getattr(config, 'TRIGGER_TYPES_FOR_BUILDUPS', [])
        df_buildups = buildup_metrics.find_buildup_sequences(
            df_processed,
            attacking_team,
            defending_team,
            metric_to_analyze='buildup_phase',
            triggers_buildups=triggers
        )

        if df_buildups is None or df_buildups.empty:
            return dbc.Alert(f"No valid buildup sequences found for {attacking_team}.", color="warning", className="mt-3")

        all_sequences = [
            df_buildups[df_buildups['trigger_sequence_id'] == seq_id]
            for seq_id in df_buildups['trigger_sequence_id'].unique()
        ]

        if not all_sequences:
            return dbc.Alert(f"No sequences found for {attacking_team} after grouping.", color="warning", className="mt-3")

        # --- 3. Assign lb_type to each sequence ---
        sequences_with_type = []
        for seq_df in all_sequences:
            if seq_df.empty:
                sequences_with_type.append(seq_df)
                continue
            passes = seq_df[seq_df['type_name'] == 'Pass'].copy()
            if passes.empty:
                lb_type = 'Short-Short'
            else:
                first_pass = passes.iloc[0]
                if first_pass.get('lb') == 1:
                    lb_type = 'Long Ball'
                elif len(passes) > 1 and passes.iloc[1:].get('lb', pd.Series()).eq(1).any():
                    lb_type = 'Short-Long'
                else:
                    lb_type = 'Short-Short'
            seq_df = seq_df.copy()
            seq_df['lb_type'] = lb_type
            sequences_with_type.append(seq_df)
        sequences_with_type = buildup_metrics.assign_flank_to_sequences(sequences_with_type, is_away)

        filter_labels = {
            "outcomes": "Outcome",
            "flanks": "Flank",
            "types": "Buildup Type"
        }

        if active_filter:
            badges = [
                dbc.Badge(f"{filter_labels[k]}: {v}", color="info", className="me-2", pill=True)
                for k, v in active_filter.items()
            ]
            active_filters_badge = dash_html.Div([
                dash_html.Small("🎯 Active Filters:", className="text-muted me-2"),
                *badges
            ], className="mb-2")
        else:
            active_filters_badge = None

        # --- 4. Apply filter if active ---
        if active_filter:
            filter_key_map = {
                "outcomes": "sequence_outcome_type",
                "flanks": "dominant_flank",
                "types": "lb_type"
            }

            print(f"[Filtro multiplo] Attivi: {active_filter}")
            filtered_sequences = []

            for seq in sequences_with_type:
                if seq.empty:
                    continue

                match = True
                for ftype, fvalue in active_filter.items():
                    real_col = filter_key_map.get(ftype)
                    val = seq.iloc[-1].get(real_col)
                    if str(val).strip() != str(fvalue).strip():
                        match = False
                        break

                if match:
                    filtered_sequences.append(seq)

            if not filtered_sequences:
                print("[Filtro] Nessuna sequenza trovata, fallback su tutte.")
                filtered_sequences = sequences_with_type
        else:
            filtered_sequences = sequences_with_type

        # --- 5. Sort sequences by quality ---
        def get_quality_score(seq_df):
            if seq_df.empty or 'sequence_outcome_type' not in seq_df.columns:
                return 99
            outcome = seq_df['sequence_outcome_type'].iloc[-1]
            if outcome == 'Goals': return 0
            elif outcome == 'Shots': return 1
            elif outcome == 'Big Chances': return 2
            elif 'Lost' in outcome: return 3
            else: return 4

        sorted_sequences = sorted(filtered_sequences, key=get_quality_score)
        stored_sequence_data = [seq.to_json(orient='split') for seq in sorted_sequences]
        num_items = len(sorted_sequences)

        # --- 6. Compute stats and build layout ---
        buildup_stats = buildup_metrics.calculate_buildup_stats(filtered_sequences, not is_away)
        summary_layout = dash_html.Div([
            dbc.Button("❌ Reset Filter", id="buildup-reset-filter-btn", color="danger", size="sm", className="mb-2"),
            active_filters_badge,
            buildup_metrics.create_buildup_summary_cards(buildup_stats, active_filter)
        ])

        return dash_html.Div([
            dash_html.H4(f"Analysis for {attacking_team}", className="text-white mt-4"),
            dcc.Store(id='buildup-sequence-store', data={
                'sequences': stored_sequence_data,
                'team_color': team_color,
                'is_away': is_away
            }),
            dcc.Store(id='buildup-carousel-controller', data={'active_index': 0, 'total_items': num_items}),

            dbc.Button(
                [dash_html.I(className="fas fa-chart-bar me-2"), "Toggle Analysis Summary"],
                id="buildup-summary-toggle-button",
                className="mb-3 w-100",
                color="info",
                outline=True
            ),

            dbc.Collapse(
                summary_layout,
                id="buildup-summary-collapse",
                is_open=True,
            ),

            dash_html.Div([
                dcc.Store(id='buildup-sequence-store', data={
                    'sequences': stored_sequence_data, 'team_color': team_color, 'is_away': is_away
                }),
                dcc.Store(id='buildup-carousel-controller', data={'active_index': 0, 'total_items': num_items}),

                dash_html.Div(
                    id='carousel-content-wrapper',
                    style={'position': 'relative', 'minHeight': '550px'},
                    children=[dcc.Loading(type="circle", children=dash_html.Div(id='buildup-carousel-content'))]
                ),

                dbc.Row([
                    dbc.Col(dbc.Button("‹ Prev", id="buildup-prev-button", color="secondary", outline=True), width="auto"),
                    dbc.Col(dash_html.Div(id="buildup-indicator-text", className="text-center text-muted align-self-center"), width=True),
                    dbc.Col(dbc.Button("Next ›", id="buildup-next-button", color="secondary", outline=True), width="auto"),
                ], justify="between", align="center", className="mt-2"),
            ], className="mt-4"),

            dash_html.Hr(className="my-4"),
            dash_html.H6(f"Comments for {attacking_team} Buildup Analysis:", className="mt-3 text-white"),
            dcc.Textarea(
                id="comment-buildup",
                placeholder=f"Enter your analysis for {attacking_team}...",
                style={'width': '100%', 'height': 120, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'},
                className="mb-2"
            ),
            dbc.Button("Save Comment", id="save-comment-buildup", color="info", size="sm", className="me-2"),
            dash_html.Div(id="save-status-buildup", className="small d-inline-block")
        ])

    except Exception as e:
        tb_str = traceback.format_exc()
        return dbc.Alert(f"An error occurred during buildup analysis: {e}\n{tb_str}", color="danger", className="mt-3", style={"whiteSpace": "pre-wrap"})


# Callback 1: Handles the "Next" button click
@app.callback(
    Output('buildup-carousel-controller', 'data', allow_duplicate=True),
    Input('buildup-next-button', 'n_clicks'),
    State('buildup-carousel-controller', 'data'),
    prevent_initial_call=True
)
def next_slide(n_clicks, controller_data):
    if n_clicks and controller_data:
        new_index = (controller_data['active_index'] + 1) % controller_data['total_items']
        controller_data['active_index'] = new_index
        return controller_data
    return no_update

# Callback 2: Handles the "Previous" button click
@app.callback(
    Output('buildup-carousel-controller', 'data'),
    Input('buildup-prev-button', 'n_clicks'),
    State('buildup-carousel-controller', 'data'),
    prevent_initial_call=True
)
def prev_slide(n_clicks, controller_data):
    if n_clicks and controller_data:
        new_index = (controller_data['active_index'] - 1 + controller_data['total_items']) % controller_data['total_items']
        controller_data['active_index'] = new_index
        return controller_data
    return no_update

# Callback 3: Updates the plot and indicator text based on the controller's state
@app.callback(
    Output('buildup-carousel-content', 'children'),
    Output('buildup-indicator-text', 'children'),
    Input('buildup-carousel-controller', 'data'),
    State('buildup-sequence-store', 'data')
)
def update_buildup_plot_and_indicator(controller_data, stored_sequence_data):
    if not controller_data or not stored_sequence_data:
        return no_update, no_update
        
    active_index = controller_data.get('active_index', 0)
    total_items = controller_data.get('total_items', 0)
    
    indicator_text = f"Sequence {active_index + 1} of {total_items}"
    
    try:
        sequences_json = stored_sequence_data['sequences']
        team_color = stored_sequence_data['team_color']
        is_away = stored_sequence_data['is_away']
        
        if active_index >= len(sequences_json):
            return dbc.Alert("Invalid sequence index."), indicator_text
        
        seq_df = pd.read_json(sequences_json[active_index], orient='split')
        
        # Call the Plotly function
        # fig = buildup_plotly.plot_buildup_sequence_plotly(seq_df, team_color, is_away)
        fig = buildup_plotly.plot_opponent_buildup_after_loss_plotly(
            seq_df,
            team_that_lost_possession=None,  # Or actual value if available
            team_building_up=None,           # Or actual value if available
            color_for_buildup_team=team_color,
            loss_sequence_id=active_index + 1,
            loss_zone=None,                  # Or actual value if available
            is_buildup_team_away=is_away, 
            metric_to_analyze='buildup_phases',
        )
        
        plot_component = dcc.Graph(figure=fig, config={'displayModeBar': False}, style={'height': '550px'})
        return plot_component, indicator_text
        
    except Exception as e:
        tb_str = traceback.format_exc()
        error_alert = dbc.Alert(f"Error updating buildup plot: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})
        return error_alert, indicator_text
    
@app.callback(
    Output("buildup-summary-collapse", "is_open"),
    Input("buildup-summary-toggle-button", "n_clicks"),
    State("buildup-summary-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_buildup_summary(n, is_open):
    if n:
        return not is_open
    return is_open

@app.callback(
    Output("store-comment-buildup", "data"),
    Output("save-status-buildup", "children"),
    Input("save-comment-buildup", "n_clicks"),
    State("comment-buildup", "value"),
    State("url-match-page", "pathname"),
    State("store-comment-buildup", "data"),
    prevent_initial_call=True
)
def save_buildup_comment(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "buildup")
    store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(
    Output("comment-buildup", "value"),
    Input("store-comment-buildup", "data"),
    Input("url-match-page", "pathname")
)
def load_buildup_comment(data, pn):
    key = get_comment_key(pn, "buildup")
    if not key or data is None: return ""
    return data.get(key, "")
# -----------------------------------------

# ---- DEFENSIVE TRANSITIONS -----

@app.callback(
    Output("def-transition-tab-content", "children"),
    Input("def-transition-primary-tabs", "active_tab"),
    Input("store-def-transition-filter", "data"),
    State("store-df-match", "data"),
    # State("store-def-transition-filter", "data")
)
def render_def_transition_content(active_tab, active_filter, stored_data_json):
    if not stored_data_json:
        return dbc.Alert("Match data loading...", color="info", className="mt-3")

    try:
        df_processed = pd.read_json(stored_data_json['df'], orient='split')
        match_info = json.loads(stored_data_json['match_info'])
        HTEAM_NAME = match_info.get('hteamName')
        ATEAM_NAME = match_info.get('ateamName')

        if active_tab == 'def_shape':
            # Prepara i dati per entrambe le squadre usando la nuova funzione in transition_metrics
            df_home_def_actions, df_home_agg = defensive_metrics.get_defensive_block_data(df_processed, HTEAM_NAME)
            df_away_def_actions, df_away_agg = defensive_metrics.get_defensive_block_data(df_processed, ATEAM_NAME)
            
            # Crea i grafici interattivi
            fig_home = defensive_transitions_plotly.plot_defensive_block_plotly(df_home_def_actions, df_home_agg, HCOL, is_away=False)
            fig_away = defensive_transitions_plotly.plot_defensive_block_plotly(df_away_def_actions, df_away_agg, ACOL, is_away=True)

            return dbc.Row([
                dbc.Col([
                    dash_html.H5(f"{HTEAM_NAME} - Defensive Block", className="text-center text-white mt-3"),
                    dcc.Graph(figure=fig_home, config={'displayModeBar': False})
                ], md=6),
                dbc.Col([
                    dash_html.H5(f"{ATEAM_NAME} - Defensive Block", className="text-center text-white mt-3"),
                    dcc.Graph(figure=fig_away, config={'displayModeBar': False})
                ], md=6)
            ], className="mt-4")
        
        elif active_tab == 'def_hull':
            # Prepara i dati aggregati (la funzione è la stessa)
            _, df_home_agg = defensive_metrics.get_defensive_block_data(df_processed, HTEAM_NAME)
            _, df_away_agg = defensive_metrics.get_defensive_block_data(df_processed, ATEAM_NAME)

            # Crea i grafici con la nuova funzione per il Convex Hull
            fig_home_hull = defensive_transitions_plotly.plot_defensive_hull_plotly(df_home_agg, HCOL, is_away=False)
            fig_away_hull = defensive_transitions_plotly.plot_defensive_hull_plotly(df_away_agg, ACOL, is_away=True)

            return dbc.Row([
                dbc.Col([
                    dash_html.H5(f"{HTEAM_NAME} - Defensive Shape (Hull)", className="text-center text-white mt-3"),
                    dcc.Graph(figure=fig_home_hull, config={'displayModeBar': False})
                ], md=6),
                dbc.Col([
                    dash_html.H5(f"{ATEAM_NAME} - Defensive Shape (Hull)", className="text-center text-white mt-3"),
                    dcc.Graph(figure=fig_away_hull, config={'displayModeBar': False})
                ], md=6)
            ], className="mt-4")
        
        elif active_tab == 'def_ppda':
            # Calcola i dati per il Liverpool (Home Team)
            ppda_home, home_def, home_opp_pass, home_player_stats = defensive_metrics.calculate_ppda_data(df_processed, HTEAM_NAME, ATEAM_NAME)
            ppda_away, away_def, away_opp_pass, away_player_stats = defensive_metrics.calculate_ppda_data(df_processed, ATEAM_NAME, HTEAM_NAME)
            
            # Crea i grafici
            fig_home = defensive_transitions_plotly.plot_ppda_plotly(ppda_home, home_def, home_opp_pass, HTEAM_NAME, HCOL, ACOL, is_away=False)
            fig_away = defensive_transitions_plotly.plot_ppda_plotly(ppda_away, away_def, away_opp_pass, ATEAM_NAME, ACOL, HCOL, is_away=True)

            # Crea le tabelle
            table_home = dash_table.DataTable(
                data=home_player_stats.to_dict('records'),
                columns=[{"name": col.replace('_', ' '), "id": col} for col in home_player_stats.columns],
                style_table={"overflowX": "auto"},
                style_cell={'backgroundColor': '#343A40', 'color': 'white', 'textAlign': 'center', 'border': '1px solid #454D55'},
                style_header={'backgroundColor': '#454D55', 'color': 'white', 'fontWeight': 'bold'},
                style_as_list_view=True,
                sort_action="native",
            )
            table_away = dash_table.DataTable(
                data=away_player_stats.to_dict('records'),
                columns=[{"name": col.replace('_', ' '), "id": col} for col in away_player_stats.columns],
                style_as_list_view=True,
                style_table={"overflowX": "auto"},
                style_cell={'backgroundColor': '#343A40', 'color': 'white', 'textAlign': 'center', 'border': '1px solid #454D55'},
                style_header={'backgroundColor': '#454D55', 'color': 'white', 'fontWeight': 'bold'},
                sort_action="native",
            )

            # Layout finale con grafici e tabelle
            return dash_html.Div([
                dbc.Row([
                    dbc.Col(dcc.Graph(figure=fig_home), md=6),
                    dbc.Col(dcc.Graph(figure=fig_away), md=6)
                ]),
                dash_html.Hr(className="my-4"),
                dbc.Row([
                    dbc.Col([
                        dash_html.H5(f"{HTEAM_NAME} - Top Pressing Players", className="text-center text-white mb-2"),
                        table_home
                    ], md=6),
                    dbc.Col([
                        dash_html.H5(f"{ATEAM_NAME} - Top Pressing Players", className="text-center text-white mb-2"),
                        table_away
                    ], md=6)
                ])
            ])

        else:
            if active_tab == 'def_transitions_home':
                team_losing_ball = HTEAM_NAME
                team_building_up = ATEAM_NAME
                team_color = ACOL
                is_away = True
            else:
                team_losing_ball = ATEAM_NAME
                team_building_up = HTEAM_NAME
                team_color = HCOL
                is_away = False

            # Step 1 – Trova sequenze di transizione
            df_transitions = transition_metrics.find_buildup_after_possession_loss(
                df_processed,
                team_that_lost_possession=team_losing_ball,
                metric_to_analyze='defensive_transitions'
            )

            if df_transitions is None or df_transitions.empty:
                return dbc.Alert(f"No defensive transitions found for {team_losing_ball}.", color="warning", className="mt-3")

            # Step 2 – Raggruppa in sequenze singole
            all_sequences = [
                df_transitions[df_transitions['loss_sequence_id'] == seq_id]
                for seq_id in df_transitions['loss_sequence_id'].unique()
            ]

            # Step 3 – Assegna flank
            for seq in all_sequences:
                if not seq.empty:
                    seq = seq.copy()
                    seq["dominant_flank"] = transition_metrics.calculate_flank(seq['y'])

            # Step 4 – Applica filtro multiplo
            filter_key_map = {
                "outcomes": "sequence_outcome_type",
                "flanks": "dominant_flank",
                "types": "type_of_initial_loss"
            }

            filtered_sequences = []
            for seq in all_sequences:
                if seq.empty:
                    continue
                match = True
                if active_filter:
                    for key, val in active_filter.items():
                        col = filter_key_map.get(key)
                        value = str(seq.iloc[-1].get(col)) if col else None
                        if value != str(val):
                            match = False
                            break
                if match:
                    filtered_sequences.append(seq)

            if not filtered_sequences:
                filtered_sequences = all_sequences

            # Step 5 – Ordina per qualità
            def get_quality_score(seq_df):
                if seq_df.empty or 'sequence_outcome_type' not in seq_df.columns:
                    return 99
                outcome = seq_df['sequence_outcome_type'].iloc[-1]
                if outcome == 'Goals conceded': return 0
                elif outcome == 'Shots conceded': return 1
                elif outcome == 'Big Chances conceded': return 2
                elif 'Regained' in outcome: return 3
                else: return 4

            sorted_sequences = sorted(filtered_sequences, key=get_quality_score)
            stored_sequence_data = [seq.to_json(orient='split') for seq in sorted_sequences]
            num_items = len(sorted_sequences)

            # Step 6 – Layout
            return dash_html.Div([
                dash_html.H4(f"Analysis for {team_losing_ball}", className="text-white mt-4"),
                dcc.Store(id='def-transition-sequence-store', data={
                    'sequences': stored_sequence_data,
                    'team_color': team_color,
                    'is_away': is_away
                }),
                dcc.Store(id='def-transition-carousel-controller', data={
                    'active_index': 0,
                    'total_items': num_items
                }),
                dbc.Button(
                    [dash_html.I(className="fas fa-chart-bar me-2"), "Toggle Analysis Summary"],
                    id="def-transition-summary-toggle-button",
                    className="mb-3 w-100",
                    color="info",
                    outline=True
                ),
                dbc.Collapse(
                    dash_html.Div([
                        dash_html.Div(id="def-transition-filter-status", className="mb-2"),
                        dbc.Button("❌ Reset Filter", id="def-transition-reset-filter-btn", color="danger", size="sm", className="mb-3"),
                        dash_html.Div(id="def-transition-summary-content")
                    ]),
                    id="def-transition-summary-collapse",
                    is_open=True,
                ),

                # Heatmap section
                dash_html.Div([
                    dash_html.H5("🔍 Possession Loss Heatmap", className="mt-4"),
                    dcc.Loading(dcc.Graph(id="loss-heatmap-graph", config={"displayModeBar": False}))
                ]),
                dash_html.Div([
                    dash_html.H5("Defensive Sequence", className="mt-4"),
                    dash_html.Div(id='def-transition-carousel-content')
                ], className="mt-4"),
                dbc.Row([
                    dbc.Col(dbc.Button("‹ Prev", id="def-transition-prev-button", color="secondary", outline=True), width="auto"),
                    dbc.Col(dash_html.Div(id="def-transition-indicator-text", className="text-center text-muted align-self-center"), width=True),
                    dbc.Col(dbc.Button("Next ›", id="def-transition-next-button", color="secondary", outline=True), width="auto"),
                ], justify="between", align="center", className="mt-2"),

                dash_html.Hr(className="my-4"),
                dash_html.H6(f"Comments for {team_losing_ball} Defensive Transitions:", className="mt-3 text-white"),
                dcc.Textarea(id="comment-def-transition", placeholder=f"Enter your analysis for {team_losing_ball}...", style={'width': '100%', 'height': 120, 'backgroundColor': '#495057', 'color': 'white', 'borderColor': '#6c757d'}, className="mb-2"),
                dbc.Button("Save Comment", id="save-comment-def-transition", color="info", size="sm", className="me-2"),
                dash_html.Div(id="save-status-def-transition", className="small d-inline-block")
            ])

    except Exception as e:
        tb = traceback.format_exc()
        return dbc.Alert(f"Error in Def. Transition tab: {e}\n{tb}", color="danger", style={"whiteSpace": "pre-wrap"})
    

@app.callback(
    Output("def-transition-carousel-content", "children"),
    Output("def-transition-indicator-text", "children"),
    Input("def-transition-carousel-controller", "data"),
    Input("store-def-transition-filter", "data"),
    State("def-transition-sequence-store", "data")
)
def update_def_transition_plot(controller_data, active_filter, stored_data):
    if not controller_data or not stored_data:
        return no_update, no_update

    try:
        active_index = controller_data.get("active_index", 0)
        sequences_json = stored_data.get("sequences", [])
        team_color = stored_data.get("team_color", "#007BFF")
        is_away = stored_data.get("is_away", False)

        all_sequences = [pd.read_json(seq, orient="split") for seq in sequences_json]

        filtered_sequences = []
        for seq in all_sequences:
            if seq.empty:
                continue
            match = True
            if active_filter:
                for key, value in active_filter.items():
                    if key == "outcomes":
                        if str(seq.iloc[-1].get("sequence_outcome_type")) != str(value):
                            match = False
                    elif key == "flanks":
                        if transition_metrics.calculate_flank(seq['y']) != value:
                            match = False
                    elif key == "types":
                        if str(seq.iloc[0].get("type_of_initial_loss")) != str(value):
                            match = False
            if match:
                filtered_sequences.append(seq)

        if not filtered_sequences:
            filtered_sequences = all_sequences

        total_sequences = len(filtered_sequences)
        if active_index >= total_sequences:
            active_index = 0

        selected_seq = filtered_sequences[active_index]

        fig = buildup_plotly.plot_opponent_buildup_after_loss_plotly(
            selected_seq,
            team_that_lost_possession=None,
            team_building_up=None,
            color_for_buildup_team=team_color,
            loss_sequence_id=active_index + 1,
            loss_zone=selected_seq.iloc[0].get("loss_zone"),
            is_buildup_team_away=is_away,
            metric_to_analyze='defensive_transitions'
        )

        graph = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "550px"})
        indicator_text = f"Sequence {active_index + 1} of {total_sequences}"

        return graph, indicator_text

    except Exception as e:
        tb = traceback.format_exc()
        alert = dbc.Alert(f"Error rendering defensive transition plot: {e}\n{tb}", color="danger", style={"whiteSpace": "pre-wrap"})
        return alert, no_update

@app.callback(
    Output("store-def-transition-filter", "data"),
    Input({"type": "def-filter", "filter_type": ALL, "value": ALL}, "n_clicks"),
    Input("def-transition-reset-filter-btn", "n_clicks"),
    State("store-def-transition-filter", "data"),
    prevent_initial_call=True
)
def update_def_transition_filter(n_clicks_list, reset_clicks, current_filter):
    ctx = dash.callback_context
    if not ctx.triggered:
        return current_filter

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "def-transition-reset-filter-btn":
        return None

    try:
        triggered = json.loads(triggered_id)
        filter_type = triggered.get("filter_type")
        value = triggered.get("value")
    except Exception:
        return current_filter

    if not filter_type or not value:
        return current_filter

    if current_filter is None:
        current_filter = {}

    # Toggle filtro
    if current_filter.get(filter_type) == value:
        current_filter.pop(filter_type)
    else:
        current_filter[filter_type] = value

    return current_filter if current_filter else None


@app.callback(
    Output("def-transition-summary-content", "children"),
    Input("store-def-transition-filter", "data"),
    State("def-transition-sequence-store", "data")
)
def update_def_transition_summary_cards(active_filter, stored_data):
    if not stored_data:
        return dash_html.Div()

    sequences_json = stored_data.get("sequences", [])
    is_away = stored_data.get("is_away", False)
    all_sequences = [pd.read_json(io.StringIO(seq), orient="split") for seq in sequences_json]

    filtered_sequences = []
    for seq in all_sequences:
        if seq.empty:
            continue
        match = True
        if active_filter:
            for key, val in active_filter.items():
                if key == "outcomes" and str(seq.iloc[-1].get("sequence_outcome_type")) != str(val):
                    match = False
                elif key == "flanks" and transition_metrics.calculate_flank(seq['y']) != val:
                    match = False
                elif key == "types" and str(seq.iloc[0].get("type_of_initial_loss")) != str(val):
                    match = False
        if match:
            filtered_sequences.append(seq)

    stats = transition_metrics.calculate_def_transition_stats(filtered_sequences, is_away)
    # transition_profile_table = stats.get("transition_profile_table", pd.DataFrame())
    # # print("[DEBUG] Colonne disponibili:", transition_profile_table.columns)

    # transition_profile_table.sort_values(by="Num_Sequences", ascending=False, inplace=True)

    # if not transition_profile_table.empty:
    #     transition_profile_component = transition_metrics.generate_transition_profile_table(transition_profile_table)
    # else:
    #     transition_profile_component = dbc.Alert("No transition profile data available.", color="secondary")

    profile_df = stats.get("transition_profile_table", pd.DataFrame())
    if not profile_df.empty:
        profile_df = profile_df.sort_values(by="Num_Sequences", ascending=False)
        profile_table_component = dash_table.DataTable(
            data=profile_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in profile_df.columns],
            style_table={"overflowX": "auto"},
            style_cell={'backgroundColor': '#343A40', 'color': 'white', 'textAlign': 'center', 'border': '1px solid #454D55'},
            style_header={'backgroundColor': '#454D55', 'color': 'white', 'fontWeight': 'bold'},
            style_as_list_view=True,
        )
    else:
        profile_table_component = dbc.Alert("No transition profile data available.", color="secondary")
    

    filter_labels = {
        "outcomes": "Outcome",
        "flanks": "Flank",
        "types": "Type of Loss"
    }
    if active_filter:
        badges = [
            dbc.Badge(f"{filter_labels.get(k, k)}: {v}", color="info", className="me-2", pill=True)
            for k, v in active_filter.items()
        ]
        active_filters_badge = dash_html.Div([
            dash_html.Small("🎯 Active Filters:", className="text-muted me-2"),
            *badges
        ], className="mb-2")
    else:
        active_filters_badge = None

    return dash_html.Div([
        active_filters_badge,
        transition_metrics.create_def_transition_summary_cards(stats, active_filter),
        dash_html.Hr(),
        dash_html.H5("Transition Profile Summary", className="text-white mt-4 mb-3 text-center"),
        profile_table_component 
    ])

# Toggle per la sezione riassuntiva
@app.callback(
    Output("def-transition-summary-collapse", "is_open"),
    Input("def-transition-summary-toggle-button", "n_clicks"),
    State("def-transition-summary-collapse", "is_open"),
    prevent_initial_call=True
)
def toggle_def_transition_summary(n, is_open):
    return not is_open if n else is_open

@app.callback(
    Output("debug-def-filter", "children"),
    Input("store-def-transition-filter", "data")
)
def show_filter_state(data):
    return f"Filtro attivo: {data}" if data else "Nessun filtro attivo"

@app.callback(
    Output('def-transition-carousel-controller', 'data', allow_duplicate=True),
    Input('def-transition-next-button', 'n_clicks'),
    State('def-transition-carousel-controller', 'data'),
    prevent_initial_call=True
)
def def_transition_next_slide(n_clicks, controller_data):
    if n_clicks and controller_data:
        new_index = (controller_data['active_index'] + 1) % controller_data['total_items']
        controller_data['active_index'] = new_index
        return controller_data
    return dash.no_update

@app.callback(
    Output('def-transition-carousel-controller', 'data'),
    Input('def-transition-prev-button', 'n_clicks'),
    State('def-transition-carousel-controller', 'data'),
    prevent_initial_call=True
)
def def_transition_prev_slide(n_clicks, controller_data):
    if n_clicks and controller_data:
        new_index = (controller_data['active_index'] - 1 + controller_data['total_items']) % controller_data['total_items']
        controller_data['active_index'] = new_index
        return controller_data
    return dash.no_update

@app.callback(
    Output("loss-heatmap-graph", "figure"),
    Input("store-def-transition-filter", "data"),
    State("def-transition-sequence-store", "data")
)
def update_loss_heatmap(active_filter, stored_data):

    if not stored_data:
        return go.Figure()

    sequences_json = stored_data.get("sequences", [])
    is_away = stored_data.get("is_away", False)
    all_sequences = [pd.read_json(seq, orient="split") for seq in sequences_json]

    filtered_sequences = []
    for seq in all_sequences:
        if seq.empty:
            continue
        match = True
        if active_filter:
            for key, value in active_filter.items():
                if key == "outcomes" and str(seq.iloc[-1].get("sequence_outcome_type")) != str(value):
                    match = False
                elif key == "flanks" and transition_metrics.calculate_flank(seq) != value:
                    match = False
                elif key == "types" and str(seq.iloc[0].get("type_of_initial_loss")) != str(value):
                    match = False
        if match:
            filtered_sequences.append(seq)

    return defensive_transitions_plotly.plot_loss_heatmap_on_pitch(filtered_sequences, is_away=is_away)


# ---------------------------------------

# ---- OFFENSIVE TRANSITIONS -----

@app.callback(
    Output("off-transition-tab-content", "children"),
    Input("off-transition-primary-tabs", "active_tab"),
    Input("store-off-transition-filter", "data"),
    State("store-df-match", "data")
)
def render_off_transition_content(active_tab, active_filter, stored_data_json):
    if not stored_data_json:
        return dbc.Alert("Match data loading...", color="info", className="mt-3")

    try:
        df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        match_info = json.loads(stored_data_json['match_info'])
        
        if active_tab == 'off_transitions_home':
            team_recovering_ball = match_info.get('hteamName')
            team_losing_ball = match_info.get('ateamName')
            team_color = HCOL
            is_away = False
        else:
            team_recovering_ball = match_info.get('ateamName')
            team_losing_ball = match_info.get('hteamName')
            team_color = ACOL
            is_away = True

        # Usiamo la stessa funzione, ma invertiamo chi perde palla!
        df_transitions = transition_metrics.find_buildup_after_possession_loss(
            df_processed,
            team_that_lost_possession=team_losing_ball, 
            metric_to_analyze='offensive_transitions'
        )

        if df_transitions is None or df_transitions.empty:
            return dbc.Alert(f"No offensive transitions found for {team_recovering_ball}.", color="warning", className="mt-3")

        # Raggruppamento e ordinamento (logica identica)
        all_sequences = [df_transitions[df_transitions['loss_sequence_id'] == seq_id] for seq_id in df_transitions['loss_sequence_id'].unique()]
        
        # Step 4 – Applica filtro multiplo
        filter_key_map = {
            "outcomes": "sequence_outcome_type",
            "flanks": "dominant_flank",
            "types": "type_of_initial_loss"
        }

        filtered_sequences = []
        for seq in all_sequences:
            if seq.empty:
                continue
            match = True
            if active_filter:
                for key, val in active_filter.items():
                    col = filter_key_map.get(key)
                    value = str(seq.iloc[-1].get(col)) if col else None
                    if value != str(val):
                        match = False
                        break
            if match:
                filtered_sequences.append(seq)

        if not filtered_sequences:
            filtered_sequences = all_sequences

        # filtered_sequences = all_sequences # Per ora, mostriamo tutte
        
        def get_quality_score(seq_df):
            if seq_df.empty: return 99
            outcome = seq_df['sequence_outcome_type'].iloc[-1]
            if outcome == 'Goals': return 0
            if outcome == 'Shots': return 1
            if outcome == 'Big Chances': return 2
            return 4
        
        sorted_sequences = sorted(filtered_sequences, key=get_quality_score)
        stored_sequence_data = [seq.to_json(orient='split') for seq in sorted_sequences]
        num_items = len(sorted_sequences)

        # Creazione layout
        return dash_html.Div([
            dash_html.H4(f"Analysis for {team_recovering_ball}", className="text-white mt-4"),
            dcc.Store(id='off-transition-sequence-store', 
                      data={'sequences': stored_sequence_data, 
                            'team_color': team_color, 
                            'is_away': is_away}),
            dcc.Store(id='off-transition-carousel-controller', 
                      data={'active_index': 0, 'total_items': num_items}),

            # Summary Section
            dbc.Button(
                [dash_html.I(className="fas fa-chart-bar me-2"), "Toggle Analysis Summary"],
                id="off-transition-summary-toggle-button", className="mb-3 w-100", color="info", outline=True),
            dbc.Collapse(id="off-transition-summary-collapse", is_open=True, children=[
                dbc.Button("❌ Reset Filter", id="off-transition-reset-filter-btn", color="danger", size="sm", className="mb-3"),
                dash_html.Div(id="off-transition-summary-content")
            ]),

            # Heatmap Section
            dash_html.Div([
                dash_html.H5("🔍 Ball Recovery Heatmap", className="mt-4"),
                dcc.Loading(dcc.Graph(id="recovery-heatmap-graph", config={"displayModeBar": False}))
            ]),
            
            # Carousel Section
            dash_html.Div([
                dash_html.H5("Offensive Sequence", className="mt-4"),
                dcc.Loading(type="circle", children=dash_html.Div(id='off-transition-carousel-content'))
            ], className="mt-4"),
            dbc.Row([
                dbc.Col(dbc.Button("‹ Prev", id="off-transition-prev-button", color="secondary", outline=True), width="auto"),
                dbc.Col(dash_html.Div(id="off-transition-indicator-text", className="text-center text-muted align-self-center"), width=True),
                dbc.Col(dbc.Button("Next ›", id="off-transition-next-button", color="secondary", outline=True), width="auto"),
            ], justify="between", align="center", className="mt-2"),
        ])

    except Exception as e:
        return dbc.Alert(f"Error in Off. Transition tab: {traceback.format_exc()}", color="danger", style={"whiteSpace": "pre-wrap"})

# Callback per aggiornare il carosello
@app.callback(
    Output('off-transition-carousel-content', 'children'),
    Output('off-transition-indicator-text', 'children'),
    Input('off-transition-carousel-controller', 'data'),
    State('off-transition-sequence-store', 'data'),
    State("store-df-match", "data") # Aggiungiamo lo store dei dati della partita
)
def update_off_transition_plot(controller_data, stored_data, stored_match_data): # Aggiunto stored_match_data
    if not controller_data or not stored_data or not stored_match_data:
        return no_update, no_update
    
    try:
        # --- 1. Estrai i dati necessari ---
        active_index = controller_data['active_index']
        seq_df = pd.read_json(io.StringIO(stored_data['sequences'][active_index]), orient='split')
        
        if seq_df.empty:
            return dbc.Alert("Sequenza vuota, impossibile generare il plot."), "N/A"

        # Dati generali sulla sequenza
        team_color = stored_data['team_color']
        is_away = stored_data['is_away']
        
        # Dati specifici per la funzione di plot
        match_info = json.loads(stored_match_data['match_info'])
        
        # Determiniamo i nomi delle squadre
        if is_away:
            team_building_up = match_info.get('ateamName')
            team_that_lost_possession = match_info.get('hteamName')
        else:
            team_building_up = match_info.get('hteamName')
            team_that_lost_possession = match_info.get('ateamName')
            
        # Estraiamo i dati dalla prima riga della sequenza, dove sono salvati
        first_event = seq_df.iloc[0]
        loss_sequence_id = first_event.get('loss_sequence_id', active_index + 1)
        loss_zone = first_event.get('loss_zone', 'Unknown Zone')
        
        # --- 2. Chiama la funzione con TUTTI i parametri richiesti ---
        fig = buildup_plotly.plot_opponent_buildup_after_loss_plotly(
            sequence_data=seq_df,                                  # <--- Passato come argomento con nome per chiarezza
            team_that_lost_possession=team_that_lost_possession, # <--- PARAMETRO RICHIESTO
            team_building_up=team_building_up,                   # <--- PARAMETRO RICHIESTO
            color_for_buildup_team=team_color,
            loss_sequence_id=loss_sequence_id,                   # <--- PARAMETRO RICHIESTO
            loss_zone=loss_zone,                                 # <--- PARAMETRO RICHIESTO
            is_buildup_team_away=is_away,
            metric_to_analyze='offensive_transitions'
        )
        
        graph = dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "550px"})
        indicator = f"Sequence {active_index + 1} of {controller_data['total_items']}"
        
        return graph, indicator

    except Exception as e:
        # Forniamo un messaggio di errore più utile in caso di problemi
        error_message = f"Errore durante la generazione del plot della transizione offensiva: {e}"
        tb_str = traceback.format_exc()
        print(f"{error_message}\n{tb_str}")
        return dbc.Alert(error_message, color="danger"), "Error"

# Callback per i pulsanti del carosello
@app.callback(Output('off-transition-carousel-controller', 'data', allow_duplicate=True), Input('off-transition-next-button', 'n_clicks'), State('off-transition-carousel-controller', 'data'), prevent_initial_call=True)
def off_next(n, data): return {'active_index': (data['active_index'] + 1) % data['total_items'], 'total_items': data['total_items']}
@app.callback(Output('off-transition-carousel-controller', 'data'), Input('off-transition-prev-button', 'n_clicks'), State('off-transition-carousel-controller', 'data'), prevent_initial_call=True)
def off_prev(n, data): return {'active_index': (data['active_index'] - 1 + data['total_items']) % data['total_items'], 'total_items': data['total_items']}

# Callback per le summary cards e la heatmap
@app.callback(
    Output("off-transition-summary-content", "children"),
    Output("recovery-heatmap-graph", "figure"),
    Input("store-off-transition-filter", "data"),
    State("off-transition-sequence-store", "data")
)
def update_off_transition_summary_and_heatmap(active_filter, stored_data):
    if not stored_data:
        return no_update, go.Figure()

    all_sequences = [pd.read_json(io.StringIO(seq), orient="split") for seq in stored_data['sequences']]
    is_away = stored_data.get("is_away", False)
    
    filtered_sequences = []
    for seq in all_sequences:
        if seq.empty:
            continue
        match = True
        if active_filter:
            for key, val in active_filter.items():
                if key == "outcomes" and str(seq.iloc[-1].get("sequence_outcome_type")) != str(val):
                    match = False
                elif key == "flanks" and transition_metrics.calculate_flank(seq['y']) != val:
                    match = False
                elif key == "types" and str(seq.iloc[0].get("type_of_initial_loss")) != str(val):
                    match = False
        if match:
            filtered_sequences.append(seq)

    stats = transition_metrics.calculate_off_transition_stats(filtered_sequences)
    cards = transition_metrics.create_off_transition_summary_cards(stats, active_filter)
    heatmap_fig = offensive_transitions_plotly.plot_recovery_heatmap_on_pitch(filtered_sequences, is_away=is_away)

    profile_df = stats.get("transition_profile_table", pd.DataFrame())
    
    if not profile_df.empty:
        # Ordiniamo per numero di sequenze per vedere i pattern più comuni
        profile_df = profile_df.sort_values(by="Num_Sequences", ascending=False)
        
        profile_table_component = dash_table.DataTable(
            data=profile_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in profile_df.columns],
            style_table={"overflowX": "auto"},
            style_cell={'backgroundColor': '#343A40', 'color': 'white', 'textAlign': 'center', 'border': '1px solid #454D55'},
            style_header={'backgroundColor': '#454D55', 'color': 'white', 'fontWeight': 'bold'},
            style_as_list_view=True,
        )
    else:
        profile_table_component = dbc.Alert("No transition profile data available.", color="secondary")

    # --- Combiniamo le card e la tabella in un unico output ---
    summary_layout = dash_html.Div([
        cards, 
        dash_html.Hr(),
        dash_html.H5("Transition Profile Summary", className="text-white mt-4 mb-3 text-center"),
        profile_table_component
    ])
    
    return summary_layout, heatmap_fig

# Callback per gestire il filtro
@app.callback(
    Output("store-off-transition-filter", "data"),
    Input({"type": "off-filter", "filter_type": ALL, "value": ALL}, "n_clicks"),
    Input("off-transition-reset-filter-btn", "n_clicks"),
    State("store-off-transition-filter", "data"),
    prevent_initial_call=True
)
def update_off_transition_filter(n_clicks_list, reset_clicks, current_filter):
    # Logica identica a `update_def_transition_filter`, basta cambiare l'ID
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "off-transition-reset-filter-btn":
        return None
    try:
        triggered = json.loads(triggered_id)
        filter_type, value = triggered.get("filter_type"), triggered.get("value")
        if current_filter is None: current_filter = {}
        if current_filter.get(filter_type) == value: current_filter.pop(filter_type)
        else: current_filter[filter_type] = value
        return current_filter or None
    except:
        return current_filter

# Callback per il collapse
@app.callback(Output("off-transition-summary-collapse", "is_open"), Input("off-transition-summary-toggle-button", "n_clicks"), State("off-transition-summary-collapse", "is_open"), prevent_initial_call=True)
def toggle_off_transition_summary(n, is_open): return not is_open if n else is_open

# ---------------------------------------

# --- SET PIECE SECTION ---

@app.callback(
    Output("set-piece-tab-content", "children"),
    Input("set-piece-primary-tabs", "active_tab"),
    Input("store-set-piece-filter", "data"),
    State("store-df-match", "data")
)
def render_set_piece_interface(active_tab, active_filter, stored_data_json):
    if not stored_data_json:
        return dbc.Alert("Match data is loading...", color="info")

    try:
        # --- 1. Caricamento e analisi iniziale dei dati (ogni volta che la tab cambia) ---
        df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        match_info = json.loads(stored_data_json['match_info'])
        
        is_home = active_tab == 'set_piece_home'
        team_name = match_info.get('hteamName') if is_home else match_info.get('ateamName')
        defending_team = match_info.get('ateamName') if is_home else match_info.get('hteamName')
        team_color = HCOL if is_home else ACOL

        set_piece_triggers = ['Out', 'Foul', 'Corner Awarded']
        df_sequences_raw = buildup_metrics.find_buildup_sequences(
            df_processed, team_name, defending_team,
            metric_to_analyze='set_piece',
            triggers_buildups=set_piece_triggers,
            start_x=50
        )

        if df_sequences_raw is None or df_sequences_raw.empty:
            return dbc.Alert(f"No offensive set pieces found for {team_name}.", color="warning", className="mt-3")

        all_sequences = [df_sequences_raw[df_sequences_raw['trigger_sequence_id'] == seq_id] 
                         for seq_id in df_sequences_raw['trigger_sequence_id'].unique()]
        
        df_analyzed, full_stats = set_piece_metrics.analyze_and_summarize_set_pieces(all_sequences)
        player_jersey_map = df_processed.drop_duplicates(subset=['playerName'])[['playerName', 'Mapped Jersey Number']].set_index('playerName').to_dict()['Mapped Jersey Number']

        # --- 2. Logica dei Filtri a Cascata ---
        active_filter = active_filter or {}
        df_filtered = df_analyzed.copy()

        filter_map = {
            'action': 'Action Type', 
            'side': 'Side', 
            'delivery': 'Delivery', 
            'swing': 'Swing', 
            'outcome': 'Outcome',
            'foot': 'Foot', 
            'destination': 'Destination', 
            'taker': 'playerName'
        }
        for filter_key, filter_value in active_filter.items():
            column_name = filter_map.get(filter_key)
            if column_name and column_name in df_filtered.columns:
                df_filtered = df_filtered[df_filtered[column_name] == filter_value]

        # --- 3. Ricostruisci le statistiche per le card basandoti sui dati filtrati ---
        filtered_stats = {
            'total': len(df_filtered),
            'action_types': df_filtered['Action Type'].value_counts().to_dict(),
            'sides': df_filtered['Side'].value_counts().to_dict(),
            'deliveries': df_filtered['Delivery'].value_counts().to_dict(),
            'swings': df_filtered[df_filtered['Swing'] != 'N/A']['Swing'].value_counts().to_dict(),
            'feet': df_filtered[df_filtered['Foot'] != 'Unknown']['Foot'].value_counts().to_dict(),
            'destinations': df_filtered[df_filtered['Destination'] != 'N/A']['Destination'].value_counts().to_dict(),
            'outcomes': df_filtered['Outcome'].value_counts().to_dict()
        }
        cards = set_piece_metrics.create_set_piece_summary_cards(filtered_stats, active_filter)
        takers_card = set_piece_metrics.create_takers_card(df_filtered, player_jersey_map, active_filter)
        if takers_card and isinstance(cards, dash_html.Div) and cards.children:
            cards.children[0].children.append(takers_card)

        # --- 4. Prepara il Carosello ---
        filtered_seq_ids = df_filtered['sequence_id'].unique()
        sequences_for_carousel = [s for s in all_sequences if not s.empty and s.iloc[0]['trigger_sequence_id'] in filtered_seq_ids]
        
        # --- START: AGGIUNTA LOGICA DI ORDINAMENTO ---
        def get_set_piece_quality_score(seq_df):
            if seq_df.empty or 'sequence_outcome_type' not in seq_df.columns:
                return 99 # Manda in fondo le sequenze vuote/errate
            outcome = seq_df.iloc[-1]['sequence_outcome_type']
            # Assegna un punteggio numerico (più basso è meglio)
            if outcome == 'Goals': return 0
            elif outcome == 'Shots': return 1
            elif outcome == 'Big Chances': return 2
            elif outcome == 'Lost Possessions': return 3
            else: return 4

        sorted_sequences_for_carousel = sorted(sequences_for_carousel, key=get_set_piece_quality_score)
        num_items = len(sorted_sequences_for_carousel)

        carousel_section = dbc.Alert("No sequences match the current filter.", color="warning", className="mt-4")
        if num_items > 0:
            carousel_section = dash_html.Div([
                dcc.Store(id='set-piece-sequence-store', data={'sequences': [s.to_json(orient='split') for s in sorted_sequences_for_carousel], 'team_color': team_color, 'is_away': not is_home}),
                dcc.Store(id='set-piece-carousel-controller', data={'active_index': 0, 'total_items': num_items}),
                dash_html.H5("Sequence Explorer", className="text-white mt-4"),
                dcc.Loading(type="circle", children=dash_html.Div(id='set-piece-carousel-content')),
                dbc.Row([
                    dbc.Col(dbc.Button("‹ Prev", id="set-piece-prev-button", color="secondary"), width="auto"),
                    dbc.Col(dash_html.Div(id="set-piece-indicator-text", className="text-center text-muted align-self-center"), width=True),
                    dbc.Col(dbc.Button("Next ›", id="set-piece-next-button", color="secondary"), width="auto"),
                ], justify="between", align="center", className="mt-2"),
            ])
        
        # --- 5. Layout Finale (con indentazione corretta) ---
        return dash_html.Div([
            dash_html.H4(f"Analysis for {team_name}", className="text-white mt-4"),
            dbc.Button(
                [dash_html.I(className="fas fa-chart-bar me-2"), "Toggle Analysis Summary"],
                id="set-piece-toggle-button", # L'ID che il callback si aspetta
                className="mb-3 w-100",
                color="info",
                outline=True
            ),
            
            dbc.Collapse(
                dash_html.Div([
                    dbc.Row([
                        dbc.Col(dash_html.Div(), width='auto'), # Placeholder per allineare a destra
                        dbc.Col(dbc.Button("❌ Reset Filters", id={'type': 'reset-btn', 'section': 'set-piece'}, color="danger", size="sm"), width='auto')
                    ], justify="end", className="mb-3"),
                    cards,
                ]),
                id="set-piece-collapse", # ID del Collapse
                is_open=True,
            ),

            # dbc.Row([
            #     dbc.Col(dbc.Button("❌ Reset Filters", id={'type': 'reset-btn', 'section': 'set-piece'}, color="danger", size="sm"), width='auto')
            # ], justify="between", align="center", className="mb-3"),
            # cards,
            dash_html.Hr(),
            carousel_section
        ])

    except Exception as e:
        return dbc.Alert(f"Error in Set Piece tab: {traceback.format_exc()}", color="danger", style={"whiteSpace": "pre-wrap"})
    
@app.callback(
    Output("set-piece-collapse", "is_open"),
    Input("set-piece-toggle-button", "n_clicks"),
    State("set-piece-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_set_piece(n, is_open):
    if n:
        return not is_open
    return is_open


# # Callback 3: Gestisce i click sui filtri (rimane invariato)
# @app.callback(
#     Output("store-set-piece-filter", "data"),
#     Input({"type": "sp-filter", "filter_type": ALL, "value": ALL}, "n_clicks"),
#     Input("sp-reset-filter-btn", "n_clicks"),
#     State("store-set-piece-filter", "data"),
#     prevent_initial_call=True
# )
# def update_set_piece_filter(n_clicks_list, reset_clicks, current_filter):
#     ctx = dash.callback_context
#     triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

#     if triggered_id == "sp-reset-filter-btn":
#         return None

#     if not any(n > 0 for n in n_clicks_list):
#         return no_update

#     try:
#         triggered_info = ast.literal_eval(triggered_id)
#     except (ValueError, SyntaxError):
#         # Fallback nel caso la stringa sia malformata per qualche motivo
#         print(f"Errore nel parsing dell'ID con ast: {triggered_id}")
#         return no_update

#     current_filter = current_filter or {}
#     filter_type = triggered_info['filter_type']
#     value = triggered_info['value']
    
#     if current_filter.get(filter_type) == value:
#         current_filter.pop(filter_type)
#     else:
#         current_filter[filter_type] = value
        
#     return current_filter if current_filter else None

@app.callback(
    Output("store-set-piece-filter", "data"),
    Output("cross-filter-store", "data"),
    Input({"type": "sp-filter", "filter_type": ALL, "value": ALL}, "n_clicks"),
    Input({"type": "cross-filter", "filter_type": ALL, "value": ALL}, "n_clicks"),
    Input({"type": "reset-btn", "section": ALL}, "n_clicks"),
    State("store-set-piece-filter", "data"),
    State("cross-filter-store", "data"),
    prevent_initial_call=True
)
def update_specific_filters(sp_clicks, cross_clicks, reset_clicks, sp_filter, cross_filter):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    triggered_id_str = ctx.triggered[0]["prop_id"].split(".")[0]
    
    try:
        triggered_info = ast.literal_eval(triggered_id_str)
    except (ValueError, SyntaxError):
        return no_update, no_update

    # Determina quale filtro è stato attivato
    trigger_source = triggered_info.get('type')

    # Se è stato premuto un bottone di reset
    if trigger_source == 'reset-btn':
        section = triggered_info.get('section')
        if section == 'set-piece':
            return None, no_update
        elif section == 'crosses':
            return no_update, None
        else:
            return no_update, no_update

    # Se è stata cliccata una card del filtro "Set Piece"
    elif trigger_source == 'sp-filter':
        filter_type = triggered_info.get('filter_type')
        value = triggered_info.get('value')
        
        current_filter = sp_filter or {}
        if current_filter.get(filter_type) == value:
            current_filter.pop(filter_type, None)
        else:
            current_filter[filter_type] = value
        
        return current_filter if current_filter else None, no_update

    # Se è stata cliccata una card del filtro "Crosses"
    elif trigger_source == 'cross-filter':
        filter_type = triggered_info.get('filter_type')
        value = triggered_info.get('value')

        current_filter = cross_filter or {}
        if current_filter.get(filter_type) == value:
            current_filter.pop(filter_type, None)
        else:
            current_filter[filter_type] = value
            
        return no_update, current_filter if current_filter else None

    # Se non è nessuno dei casi precedenti, non fare nulla
    return no_update, no_update


@app.callback(
    Output('set-piece-carousel-controller', 'data'),
    Input('set-piece-prev-button', 'n_clicks'),
    Input('set-piece-next-button', 'n_clicks'),
    State('set-piece-carousel-controller', 'data'),
    prevent_initial_call=True
)
def update_set_piece_carousel_controller(prev_clicks, next_clicks, controller_data):
    ctx = dash.callback_context
    if not ctx.triggered or not controller_data:
        return no_update
        
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    active_index = controller_data['active_index']
    total_items = controller_data['total_items']
    
    if button_id == 'set-piece-next-button':
        new_index = (active_index + 1) % total_items
    elif button_id == 'set-piece-prev-button':
        new_index = (active_index - 1 + total_items) % total_items
    else:
        new_index = active_index

    controller_data['active_index'] = new_index
    return controller_data


@app.callback(
    Output('set-piece-carousel-content', 'children'),
    Output('set-piece-indicator-text', 'children'),
    Input('set-piece-carousel-controller', 'data'),
    State('set-piece-sequence-store', 'data'),
    State('store-df-match', 'data')
)
def update_set_piece_carousel_plot(controller_data, sequence_data, match_data):
    if not controller_data or not sequence_data or not match_data:
        return "Loading...", "..."

    active_index = controller_data['active_index']
    total_items = controller_data['total_items']
    
    seq_df = pd.read_json(io.StringIO(sequence_data['sequences'][active_index]), orient='split')
    team_color = sequence_data['team_color']
    is_away = sequence_data['is_away']

    match_info = json.loads(match_data['match_info'])
    attacking_team = match_info.get('hteamName') if not is_away else match_info.get('ateamName')
    defending_team = match_info.get('ateamName') if not is_away else match_info.get('hteamName')

    fig = buildup_plotly.plot_opponent_buildup_after_loss_plotly(
        sequence_data=seq_df,
        team_that_lost_possession=defending_team,
        team_building_up=attacking_team,
        color_for_buildup_team=team_color,
        loss_sequence_id=seq_df.iloc[0]['trigger_sequence_id'],
        loss_zone=seq_df.iloc[0]['trigger_zone'],
        is_buildup_team_away=is_away,
        metric_to_analyze='set_piece'
    )
    
    indicator = f"Sequence {active_index + 1} of {total_items}"
    return dcc.Graph(figure=fig), indicator


# ---------------------------------------

# --- CROSSES TAB CALLBACKS ---

# Callback 1: Genera il contenuto principale della tab "Crosses" (che ora contiene altre tab)
@app.callback(
    Output("crosses-content", "children"),
    Input("passes-nested-tabs", "active_tab")
)
def layout_crosses_tab(active_nested_tab):
    if active_nested_tab != "crosses":
        return None # Non mostrare nulla se non siamo in questa tab

    return dash_html.Div([
        dbc.Tabs(
            id="crosses-team-tabs",
            active_tab="crosses-home",
            children=[
                dbc.Tab(label="Home Team Crosses", tab_id="crosses-home"),
                dbc.Tab(label="Away Team Crosses", tab_id="crosses-away"),
            ]
        ),
        dcc.Loading(type="circle", children=dash_html.Div(id="crosses-team-content"))
    ])

# Callback 2: Genera il contenuto per la squadra selezionata (Home o Away)
@app.callback(
    Output("crosses-team-content", "children"),
    Input("crosses-team-tabs", "active_tab"),
    Input("cross-filter-store", "data"),
    State("store-df-match", "data")
)
def render_crosses_team_content(active_team_tab, active_filter, stored_data_json):
    if not stored_data_json:
        return dbc.Alert("Match data loading...", color="info")

    try:
        df_processed = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        match_info = json.loads(stored_data_json['match_info'])
        
        is_away = (active_team_tab == "crosses-away")
        team_name = match_info.get('ateamName') if is_away else match_info.get('hteamName')
        
        crosses_full = cross_metrics.analyze_crosses(df_processed, team_name)
        
        active_filter = active_filter or {}
        crosses_filtered = crosses_full.copy()
        filter_map = {'origin': 'Origin Zone', 'destination': 'Destination Zone', 'swing': 'Swing', 'outcome': 'Outcome', 'foot': 'Foot', 'taker': 'playerName', 'play_type': 'Play Type'}
        for key, value in active_filter.items():
            col = filter_map.get(key)
            if col and col in crosses_filtered.columns:
                crosses_filtered = crosses_filtered[crosses_filtered[col] == value]
        
        cards = cross_metrics.create_cross_summary_cards(crosses_filtered, active_filter)
        
        home_flow_table = cross_metrics.generate_cross_flow_table(crosses_filtered)
        sankey_plot = cross_plots.plot_cross_sankey(crosses_filtered)
        
        analysis_summary_content = dash_html.Div([
            dbc.Button("❌ Reset Filters & Selection", id="cross-reset-filter-btn", color="danger", size="sm", className="mb-3"),
            cards
        ])

        return dash_html.Div([
            dcc.Store(id='cross-data-store-current-team', data=crosses_filtered.to_json(orient='split')),
            dash_html.H4(f"Analysis for {team_name}", className="text-white mt-4"),
            dbc.Button(
                [dash_html.I(className="fas fa-chart-bar me-2"), "Toggle Analysis Summary"],
                id="cross-summary-toggle-button", # L'ID che il callback si aspetta
                className="mb-3 w-100",
                color="info",
                outline=True
            ),
            
            # Ora il Collapse ha un solo figlio: il Div che contiene tutto il resto
            dbc.Collapse(
                analysis_summary_content,
                id="cross-summary-collapse",
                is_open=False,
            ),
            dash_html.Hr(),
            dbc.Tabs([
                dbc.Tab(label="📍 Location Heatmaps", children=[
                    dbc.Row([
                        dbc.Col(dcc.Graph(id='cross-origin-map'), md=6),
                        dbc.Col(dcc.Graph(id='cross-dest-map'), md=6),
                    ], className="mt-4"),
                ]),
                dbc.Tab(label="🌊 Cross Flow Analysis", children=[
                    dash_html.Div(home_flow_table, className="mt-4"),
                    dcc.Graph(figure=sankey_plot)
                ]),
            ])
        ])
    except Exception as e:
        return dbc.Alert(f"Error rendering crosses: {traceback.format_exc()}", color="danger")
    
@app.callback(
    Output("cross-summary-collapse", "is_open"),
    Input("cross-summary-toggle-button", "n_clicks"),
    State("cross-summary-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_cross_summary(n, is_open):
    if n:
        return not is_open
    return is_open

# Callback 3: Gestisce la selezione/deselezione del punto e aggiorna lo store
@app.callback(
    Output('cross-selection-store', 'data'),
    Input('cross-origin-map', 'clickData'),
    Input('cross-dest-map', 'clickData'),
    Input("cross-reset-filter-btn", "n_clicks"),
    State('cross-selection-store', 'data'),
    prevent_initial_call=True
)
def update_cross_selection(origin_click, dest_click, reset_clicks, selected_cross_id):
    ctx = dash.callback_context
    triggered_id_str = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id_str == "cross-reset-filter-btn":
        return None # Resetta la selezione

    click_data = ctx.triggered[0]['value']
    if click_data and click_data['points']:
        clicked_id = click_data['points'][0].get('customdata')
        
        # Se clicco lo stesso punto, lo deseleziono. Altrimenti, lo seleziono.
        if selected_cross_id == clicked_id:
            return None 
        else:
            return clicked_id
    return no_update

# Callback 4: Aggiorna i grafici in base ai dati filtrati e alla selezione
@app.callback(
    Output('cross-origin-map', 'figure'),
    Output('cross-dest-map', 'figure'),
    Input('cross-data-store-current-team', 'data'),
    Input('cross-selection-store', 'data'),
    State("crosses-team-tabs", "active_tab")
)
def update_cross_plots_on_selection(cross_data_json, selected_cross_id, active_team_tab):
    if not cross_data_json:
        return go.Figure(layout={'title': 'No Data'}), go.Figure(layout={'title': 'No Data'})
        
    df_filtered = pd.read_json(io.StringIO(cross_data_json), orient='split')
    is_away = (active_team_tab == "crosses-away")
    
    origin_map = cross_plots.plot_cross_heatmap(df_filtered, 'origin', is_away, selected_cross_id=selected_cross_id)
    dest_map = cross_plots.plot_cross_heatmap(df_filtered, 'destination', is_away, selected_cross_id=selected_cross_id)
    
    return origin_map, dest_map

# # Callback 5: Gestisce i filtri delle card
# @app.callback(
#     Output("cross-filter-store", "data"),
#     Input({"type": "cross-filter", "filter_type": ALL, "value": ALL}, "n_clicks"),
#     Input("cross-reset-filter-btn", "n_clicks"),
#     State("passes-nested-tabs", "active_tab"),
#     State("cross-filter-store", "data"),
#     prevent_initial_call=True
# )
# def update_cross_filter(card_clicks, reset_clicks, active_nested_tab, current_filter):
#     # ... (questo callback ora è più semplice, il suo unico scopo è aggiornare lo store dei filtri)
#     if active_nested_tab != "crosses":
#         raise dash.exceptions.PreventUpdate

#     ctx = dash.callback_context
#     triggered_id_str = ctx.triggered[0]["prop_id"].split(".")[0]
    
#     if triggered_id_str == "cross-reset-filter-btn":
#         return None # Resetta il filtro



# ----------------------------------------

# --- CALLBACK TO GENERATE REPORT HTML ---
@app.callback(
    Output("report-html-content-store", "data"),
    Output("clientside-report-trigger-div", "children"), # Output simple trigger
    Input("generate-report-button", "n_clicks"),
    State("store-df-match", "data"),
    State("url-match-page", "pathname"),
    State("store-comment-formation", "data"),         # State 1
    State("store-comment-pass-network", "data"),      # State 2
    State("store-comment-progressive-passes", "data"),# State 3 - THIS IS THE ONE
    # ... other comment stores ...
    prevent_initial_call=True
)
def prepare_report_and_trigger_clientside(
    n_clicks, stored_match_data, pathname,
    formation_comments_data,
    pass_network_comments_data,
    progressive_passes_comments_data
):
    if n_clicks is None or not stored_match_data:
        return no_update, no_update # No update for both outputs
    
    print(f"--- prepare_report_and_trigger_clientside TRIGGERED (n_clicks: {n_clicks}) ---")

    report_html_elements = []
    match_info_dict = {}
    if stored_match_data.get('match_info'):
        match_info_dict = json.loads(stored_match_data['match_info'])
        # ... (extracting hteam, ateam, etc.) ...
        hteam = match_info_dict.get('hteamDisplayName', 'Home')
        ateam = match_info_dict.get('ateamDisplayName', 'Away')
        comp = match_info_dict.get('competitionName', '')
        round_n = match_info_dict.get('roundNameFromFilename', '') 
        date_val = match_info_dict.get('date_formatted', '') 
        hs = match_info_dict.get('home_score', '')
        aws = match_info_dict.get('away_score', '')
        score = f"{hs} - {aws}" if hs is not None and aws is not None else "vs"

        report_html_elements.append(f"<h1>Match Report: {hteam} {score} {ateam}</h1>")
        report_html_elements.append(f"<p>{comp} - {round_n} | {date_val}</p><hr>")


    # # --- Section for Formation ---
    # report_html_elements.append("<h2>Formation Analysis</h2>")
    # formation_img_component = show_match_formation(stored_match_data)
    # if isinstance(formation_img_component, dash_html.Img): # Use aliased dash_html
    #     report_html_elements.append(f"<img src='{formation_img_component.src}' style='width:90%; max-width:800px; display:block; margin:auto;'/>")
    # elif isinstance(formation_img_component, dash_html.P):
    #     report_html_elements.append(f"<p><em>Error generating formation plot: {str(formation_img_component.children)}</em></p>")
    # else:
    #     report_html_elements.append("<p>Formation plot could not be generated.</p>")
    
    # form_comment_key = get_comment_key(pathname, "formation")
    # if formation_comments_data and form_comment_key and formation_comments_data.get(form_comment_key):
    #     report_html_elements.append("<h4>Comments:</h4>")
    #     comment_text = html.escape(formation_comments_data.get(form_comment_key)) # <<<--- CORRECTED
    #     report_html_elements.append(f"<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc;'>{comment_text}</pre>")
    # report_html_elements.append("<hr>")


    # # --- Section for Pass Network ---
    # report_html_elements.append("<h2>Pass Network Analysis</h2>")
    # pass_network_img_component = show_pass_network_graph(stored_match_data)
    # if isinstance(pass_network_img_component, dash_html.Img): # Use aliased dash_html
    #     report_html_elements.append(f"<img src='{pass_network_img_component.src}' style='width:90%; max-width:800px; display:block; margin:auto;'/>")
    # elif isinstance(pass_network_img_component, dash_html.P):
    #     report_html_elements.append(f"<p><em>Error generating pass network plot: {str(pass_network_img_component.children)}</em></p>")
    # else:
    #     report_html_elements.append("<p>Pass Network plot could not be generated.</p>")

    # pn_comment_key = get_comment_key(pathname, "pass_network")
    # if pass_network_comments_data and pn_comment_key and pass_network_comments_data.get(pn_comment_key):
    #     report_html_elements.append("<h4>Comments:</h4>")
    #     comment_text = html.escape(pass_network_comments_data.get(pn_comment_key))
    #     report_html_elements.append(f"<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc;'>{comment_text}</pre>")
    # report_html_elements.append("<hr>")

    # # ... (rest of the function, including final_html_string) ...
    # final_html_string = f"""
    # <html>
    #     <head>
    #         <title>Match Report</title>
    #         <style>
    #             body {{ font-family: sans-serif; margin: 20px; }}
    #             h1, h2, h3, h4 {{ color: #333; }}
    #             hr {{ margin-top: 20px; margin-bottom: 20px; border: 0; border-top: 1px solid #eee; }}
    #             img {{ border: 1px solid #ddd; margin-bottom: 10px; padding: 5px; background-color: white; }}
    #             pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.9em; }}
    #         </style>
    #     </head>
    #     <body>
    #         {''.join(report_html_elements)}
    #     </body>
    # </html>
    # """
    # print(f"prepare_report_html: Generated HTML (first 200 chars): {final_html_string[:200]}")
    # print(f"prepare_report_html: Generated HTML (last 200 chars): {final_html_string[-200:]}")
    # print(f"prepare_report_html: Total length of HTML string: {len(final_html_string)}")
    # # ... (your logic to build final_html_string) ...
    # Example:
    report_html_elements = ["<h1>Test Report Version 2</h1>"]
    # ... (add plots and comments as before) ...
    final_html_string = f"<html><body>{''.join(report_html_elements)}</body></html>"
    # ...

    print(f"prepare_report_and_trigger_clientside: HTML length: {len(final_html_string)}")
    
    # Return HTML to its store, and a simple trigger (timestamp) to the dummy div
    trigger_value = datetime.now().timestamp()
    print(f"prepare_report_and_trigger_clientside: Setting trigger value: {trigger_value}")
    return final_html_string, trigger_value

# def relay_trigger_for_report_window(report_html):
#     if report_html:
#         return datetime.now().timestamp() # Or just a counter, anything to trigger the change
#     return no_update

# # --- NEW PYTHON CALLBACK TO TRIGGER CLIENTSIDE ACTION & CLEAR HTML STORE ---
# @app.callback(
#     Output("clientside-report-trigger-div", "children"), # Output to dummy div (acts as trigger)
#     Output("report-html-content-store", "data", allow_duplicate=True), # Output to clear the store
#     Input("report-html-content-store", "data"), # Input: when HTML is ready
#     prevent_initial_call=True
# )
# def trigger_clientside_and_clear_store(report_html_content):
#     if report_html_content:
#         print("trigger_clientside_and_clear_store: HTML ready, triggering clientside and clearing store.")
#         # The value passed to the dummy div's children can be anything that changes.
#         # The clientside callback will use the HTML from the store via State.
#         # We pass the HTML itself as the trigger data, so the clientside callback gets it directly.
#         return report_html_content, None # Trigger with HTML, then clear the store
#     print("trigger_clientside_and_clear_store: No HTML, no action.")
#     return no_update, no_update

# Callback 3: Clientside callback to open window

##################################################################
def create_graph_card(graph_id, title, height='550px'):
    return dbc.Card([
        dbc.CardHeader(title),
        dbc.CardBody(dcc.Loading(dcc.Graph(id=graph_id, style={'height': height})))
    ], className="mb-4")

def layout_league_analysis():
    """Crea il layout per la pagina di analisi della lega."""
    
    # Per ora, hardcodiamo il percorso del file. In futuro potresti renderlo dinamico.
    league_data_path = os.path.join("data", "estadisticas", "England_Premier_League", "2024-2025", "equipos", "equipos_seasonstats.csv")
    
    try:
        df_league = pd.read_csv(league_data_path)
    except FileNotFoundError:
        return dbc.Alert(f"Data file not found at: {league_data_path}", color="danger")
    
    quadrant_options = [
        {'label': 'Offensive Efficiency (Shots vs. Conversion)', 'value': 'goals_vs_shots'},
        {'label': 'Playing Style (Possession vs. Verticality)', 'value': 'style'},
        {'label': 'Defensive Solidity (Pressure vs. Shots Conceded)', 'value': 'defense'}
    ]
    
    return dbc.Container([
        dbc.Row([
            dbc.Col(dash_html.H1("League Analysis - Premier League 2024/2025"), width="auto"),
            dbc.Col(dbc.Button("Back to Home", href="/", color="secondary"), width="auto", className="ms-auto")
        ], align="center", className="mt-3 mb-4"),
        
        # Struttura a TAB principale
        dbc.Tabs(
            id="league-analysis-tabs",
            active_tab="tab-quadrant",
            children=[
                dbc.Tab(label="Quadrant Analysis", tab_id="tab-quadrant", children=[
                    dcc.Dropdown(
                        id='quadrant-metric-dropdown',
                        options=quadrant_options,
                        value='goals_vs_shots',
                        className="my-3"
                    ),
                    create_graph_card('league-quadrant-plot', "Team Positioning", height='700px')
                ]),
                dbc.Tab(label="Team Profile Radar", tab_id="tab-radar", children=[
                    dcc.Dropdown(
                        id='league-team-dropdown-multi',
                        options=[{'label': team, 'value': team} for team in sorted(df_league['equipo'].unique())],
                        value=[df_league['equipo'].iloc[0], df_league['equipo'].iloc[1]], # Default ai primi due team
                        multi=True,
                        className="my-3"
                    ),
                    create_graph_card('team-radar-plot-multi', "Statistical Profile Comparison", height='700px')
                ]),
            ],
        )
    ], fluid=True, className="py-4")

@app.callback(
    Output('league-comparison-barchart', 'figure'),
    Input('league-metric-dropdown', 'value')
)
def update_league_barchart(selected_metric):
    league_data_path = os.path.join("data", "estadisticas", "England_Premier_League", "2024-2025", "equipos", "equipos_seasonstats.csv")
    df_league = pd.read_csv(league_data_path)
    
    return league_plots.create_league_barchart(df_league, selected_metric)

@app.callback(
    Output('team-radar-plot', 'figure'),
    Input('league-team-dropdown', 'value')
)
def update_team_radar(selected_team):
    league_data_path = os.path.join("data", "estadisticas", "England_Premier_League", "2024-2025", "equipos", "equipos_seasonstats.csv")
    df_league = pd.read_csv(league_data_path)
    
    return league_plots.create_team_radar(df_league, selected_team)

@app.callback(
    Output('league-quadrant-plot', 'figure'),
    Input('quadrant-metric-dropdown', 'value')
)
def update_quadrant_plot(selected_view):
    if not selected_view:
        return go.Figure()
        
    league_data_path = os.path.join("data", "estadisticas", "England_Premier_League", "2024-2025", "equipos", "equipos_seasonstats.csv")
    df_league = pd.read_csv(league_data_path)
    df_league_adv = league_metrics.add_advanced_metrics(df_league)
    
    plot_template = 'plotly_white'
    
    if selected_view == 'goals_vs_shots':
        labels = ['Elite Attack', 'Wasteful Attack', 'Ineffective Attack', 'Clinical Attack']
        return league_plots.create_quadrant_plot(df_league_adv, 'Total Shots', 'Goal Conversion', quadrant_labels=labels, template=plot_template)
    
    elif selected_view == 'style':
        labels = ['Fast & Short', 'Fast & Direct', 'Slow & Direct', 'Slow & Methodical']
        return league_plots.create_quadrant_plot(df_league_adv, 'Passing Tempo', 'Short vs Long Ratio', quadrant_labels=labels, template=plot_template)
        
    elif selected_view == 'defense':
        labels = ['Proactive & Solid', 'Busy & Leaky', 'Passive & Vulnerable', 'Organized & Efficient']
        
        # Controlla esplicitamente che le colonne necessarie esistano
        required_cols = ['Defensive Actions', 'Shots Conceded per DA']
        if not all(col in df_league_adv.columns for col in required_cols):
            return go.Figure().update_layout(title_text="Required defensive metrics are missing.", template=plot_template)
            
        return league_plots.create_quadrant_plot(df_league_adv, 'Defensive Actions', 'Shots Conceded per DA', invert_y=True, quadrant_labels=labels, template=plot_template)
    
    return go.Figure()

@app.callback(
    Output('team-radar-plot-multi', 'figure'),
    Input('league-team-dropdown-multi', 'value')
)
def update_team_radar_multi(selected_teams):
    if not selected_teams:
        return go.Figure().update_layout(title_text="Select up to 2 teams to compare")
    
    teams_to_plot = selected_teams[:2]
    
    league_data_path = os.path.join("data", "estadisticas", "England_Premier_League", "2024-2025", "equipos", "equipos_seasonstats.csv")
    df_league = pd.read_csv(league_data_path)
    df_league_adv = league_metrics.add_advanced_metrics(df_league)
    
    # Passiamo il template 'plotly_white'
    return league_plots.create_team_radar(df_league_adv, teams_to_plot, template='plotly_white')

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    State("store-df-match", "data"), # Prendiamo i dati completi dallo store
    prevent_initial_call=True,
)
def download_csv(n_clicks, stored_data_json):
    if not n_clicks or not stored_data_json:
        return dash.no_update

    try:
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')

        if not df_json_str or not match_info_json_str:
            return dash.no_update
        
        df = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)

        # Creiamo un nome file significativo
        hteam = match_info.get('hteamDisplayName', 'Home')
        ateam = match_info.get('ateamDisplayName', 'Away')
        filename = f"match_events_{hteam}_vs_{ateam}.csv"

        # Usiamo dcc.send_data_frame per creare e inviare il file CSV
        return dcc.send_data_frame(df.to_csv, filename=filename, index=False)

    except Exception as e:
        print(f"Error during CSV download: {e}")
        return dash.no_update
    
@app.callback(
    Output("dropdown-team-filter", "options"),
    Output("dropdown-team-filter", "value"),
    Input("dropdown-league", "value"),
    Input("dropdown-season", "value"),
    prevent_initial_call=True
)
def update_team_filter_options(league, season):
    if not (league and season):
        return [], None

    all_matches_files = get_matches(league, season)
    if not all_matches_files:
        return [], None

    teams = set()
    base_path_matches = os.path.join("data", "matches", league, season, "partidos")

    for m_filename in all_matches_files:
        try:
            match_file_path = os.path.join(base_path_matches, m_filename)
            with open(match_file_path, 'r', encoding='utf-8') as f:
                match_data = json.load(f)
            
            # Estrarre i nomi delle squadre dal JSON per maggiore precisione
            match_info = config.extract_match_info(match_data)
            if match_info.get('hteamDisplayName'):
                teams.add(match_info['hteamDisplayName'])
            if match_info.get('ateamDisplayName'):
                teams.add(match_info['ateamDisplayName'])
        except Exception:
            # Fallback: estrai dai nomi dei file se il JSON fallisce
            parsed_info = parse_match(m_filename)
            if parsed_info:
                teams.add(parsed_info['home_team'].replace('_', ' '))
                teams.add(parsed_info['away_team'].replace('_', ' '))

    sorted_teams = sorted(list(teams))
    options = [{"label": team, "value": team} for team in sorted_teams]
    
    return options, None


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
# --- END OF FILE app.py ---