import os
import pandas as pd
from io import StringIO
from dash import html, dcc, callback, Input, Output, State, no_update, ctx, ALL
import dash_bootstrap_components as dbc

# Importa le funzioni di utilità
from src.utils.path_helpers import get_team_logo_path, get_player_photo_path
from src.visualization import player_plots
from src.utils.player_helpers import LEAGUES, METRIC_TOOLTIPS
from src.components.layout_components import app_signature

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = os.path.join("data", "processed")
MIN_AVG_MINUTES_PER_GAME = 60 # Soglia di minuti medi a partita per essere validi

def get_available_processed_seasons():
    if not os.path.isdir(PROCESSED_DATA_PATH): return []
    files = os.listdir(PROCESSED_DATA_PATH)
    seasons = [f.replace('player_stats_', '').replace('.parquet', '') for f in files if f.startswith('player_stats_') and f.endswith('.parquet')]
    return sorted(seasons, reverse=True)

# --- LAYOUT & CALLBACKS ---
def create_league_filter_buttons():
    # ... (invariato)
    button_style = {'width': '80px', 'height': '60px', 'padding': '0.25rem', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}
    logo_style = {'height': '40px', 'width': 'auto', 'object-fit': 'contain'}
    buttons = [dbc.Button(html.I(className="fas fa-globe fa-2x"), id="player-filter-all", n_clicks=0, color="light", outline=True, style=button_style)]
    buttons.append(dbc.Tooltip("All Leagues", target="player-filter-all"))
    for league_folder, league_info in LEAGUES.items():
        button_id = f"player-filter-{league_folder}"
        buttons.append(dbc.Button(html.Img(src=league_info['logo'], style=logo_style), id=button_id, n_clicks=0, color="light", outline=True, style=button_style))
        buttons.append(dbc.Tooltip(league_info['name'], target=button_id))
    return dbc.ButtonGroup(buttons, id="player-league-filter-buttons")

def create_player_ranking_rows(top_players_df, metric_col, unit="", format_spec="{:,.2f}"):
    rows = []
    for i, player in top_players_df.iterrows():
        is_first = (i == top_players_df.index[0])
        player_name, league_name, club_name = player.get('Player', 'N/A'), player.get('League', 'N/A'), player.get('Club', 'N/A')
        player_name_url = player_name.replace(' ', '_')
        nationality_code = player.get('Nationality_Code')
        photo_style = {'width': '60px', 'height': '60px', 'object-fit': 'cover', 'border-radius': '50%'} if is_first else {'width': '40px', 'height': '40px', 'object-fit': 'cover', 'border-radius': '50%'}
        name_class, metric_class = ("fs-5 fw-bold", "display-6 fw-bold") if is_first else ("fs-6", "fs-4")
        flag_img = html.Img(src=f"https://cdnjs.cloudflare.com/ajax/libs/flag-icon-css/7.2.1/flags/4x3/{nationality_code}.svg", style={'width': '16px', 'margin-right': '5px'}) if nationality_code else ""
        club_logo_img = html.Img(src=get_team_logo_path(league_name, club_name), style={'height': '16px', 'margin-right': '5px', 'object-fit': 'contain'})
        
        player_details = html.Div([
            html.Span([flag_img, player.get('Nationality', 'N/A')], className="me-2 d-inline-flex align-items-center"),
            html.Span([club_logo_img, club_name], className="d-inline-flex align-items-center")
        ], className="small text-muted")

        # --- MODIFICA CHIAVE: Da dcc.Link a dbc.Button con ID pattern ---
        player_link_button = dbc.Button(
            player_name,
            id={'type': 'player-redirect-button', 'index': player_name_url},
            color="link",
            className=f"text-white text-decoration-none p-0 {name_class} text-start"
        )
        
        row_content = dbc.Row([
            dbc.Col(html.Img(src=get_player_photo_path(player_name), style=photo_style), width="auto", className="pe-2"),
            dbc.Col([
                player_link_button,
                player_details
            ], width=True, className="align-self-center"),
            dbc.Col(html.Span(f"{format_spec.format(player[metric_col])}{unit}", className=f"fw-bold text-end {metric_class}"), width="auto", className="align-self-center")
        ], align="center", className="py-2")
        
        list_item_class = "border-top-0 bg-transparent" if is_first else "border-top pt-2 bg-transparent"
        rows.append(dbc.ListGroupItem(row_content, color="dark", className=list_item_class))
    return rows

# ... (tutte le altre funzioni create_metric_card, generate_tabs_content sono invariate) ...
def create_metric_card(df, title, metric_col, icon, ascending=False, **kwargs):
    if df.empty or metric_col not in df.columns or 'League_MP' not in df.columns:
        return dbc.Col(dbc.Alert(f"Data not available for '{title}'", color="warning", className="h-100"))
    
    df_copy = df.copy()
    df_copy = df_copy[df_copy['League_MP'] > 0]
    df_copy['min_threshold'] = df_copy['League_MP'] * MIN_AVG_MINUTES_PER_GAME
    df_to_use = df_copy[df_copy['Min'] >= df_copy['min_threshold']]
    
    if df_to_use.empty:
        return dbc.Col(dbc.Alert(f"No players meet the dynamic minute requirement for '{title}'", color="info", className="h-100"))
    
    top_5 = df_to_use.sort_values(by=metric_col, ascending=ascending).head(5)
    
    header_id = f"player-card-header-{metric_col.replace('/', '-')}"
    card_header = dbc.CardHeader(dbc.Row([dbc.Col(html.I(className=f"{icon} me-2"), width="auto"), dbc.Col(html.H5(title, className="m-0"), width=True)], align="center"), id=header_id, className="text-center")
    tooltip = dbc.Tooltip(METRIC_TOOLTIPS.get(metric_col, "No description available."), target=header_id, placement="top")
    ranking_rows = create_player_ranking_rows(top_5, metric_col, unit=kwargs.get('unit', ""), format_spec=kwargs.get('format_spec', "{:,.2f}"))
    return dbc.Col([dbc.Card([card_header, dbc.CardBody(dbc.ListGroup(ranking_rows, flush=True), className="p-0")], className="h-100 shadow"), tooltip], lg=3, md=6, className="mb-4")

def generate_tabs_content(df):
    if df.empty: return dbc.Alert("No data available for the current selection.", color="info", className="mt-4")
    df_outfield = df[~df['Pos'].str.contains('GK', na=False)] if 'Pos' in df.columns else df
    attacking_metrics = [{'title': 'Top Scorers', 'metric_col': 'Gls', 'icon': 'fa-solid fa-futbol', 'format_spec': '{:,.0f}'}, {'title': 'Goals - xG per 90', 'metric_col': 'G_minus_xG_per_90', 'icon': 'fa-solid fa-chart-line', 'requires_min_minutes': True}, {'title': 'Shot-Creating Actions p90', 'metric_col': 'SCA90', 'icon': 'fa-solid fa-wand-magic-sparkles', 'requires_min_minutes': True}, {'title': 'Shots on Target p90', 'metric_col': 'SoT/90', 'icon': 'fa-solid fa-bullseye', 'requires_min_minutes': True}]
    possession_metrics = [{'title': 'Top Playmakers', 'metric_col': 'Ast', 'icon': 'fa-solid fa-hands-helping', 'format_spec': '{:,.0f}'}, {'title': 'Passes into Final Third p90', 'metric_col': 'Passes_F3_per_90', 'icon': 'fa-solid fa-arrow-right-to-bracket', 'requires_min_minutes': True}, {'title': 'Progressive Passes p90', 'metric_col': 'PrgP_per_90', 'icon': 'fa-solid fa-angles-up', 'requires_min_minutes': True}, {'title': 'Carries into Final Third p90', 'metric_col': 'Carries_F3_per_90', 'icon': 'fa-solid fa-arrow-trend-up', 'requires_min_minutes': True}]
    defensive_metrics = [{'title': 'Tackles + Interceptions p90', 'metric_col': 'Tkl+Int_per_90', 'icon': 'fa-solid fa-shield-halved', 'requires_min_minutes': True}, {'title': 'Aerial Duels Won %', 'metric_col': 'Aerial_Duels_perc', 'icon': 'fa-solid fa-plane-up', 'unit': '%', 'format_spec': '{:,.1f}', 'requires_min_minutes': True}, {'title': 'Clearances per 90', 'metric_col': 'Clr_per_90', 'icon': 'fa-solid fa-broom', 'requires_min_minutes': True}, {'title': 'Blocks per 90', 'metric_col': 'Blocks_per_90', 'icon': 'fa-solid fa-person-falling-burst', 'requires_min_minutes': True}]
    df_gk = df[df['Pos'].str.contains('GK', na=False)] if 'Pos' in df.columns else pd.DataFrame()
    goalkeeping_metrics = [{'title': 'Save %', 'metric_col': 'Save%', 'icon': 'fa-solid fa-mitten', 'unit': '%', 'format_spec': '{:,.1f}', 'requires_min_minutes': True, 'df': df_gk}, {'title': 'PSxG - Goals Against', 'metric_col': 'PSxG+/-', 'icon': 'fa-solid fa-chart-line', 'requires_min_minutes': True, 'df': df_gk}, {'title': 'Crosses Stopped %', 'metric_col': 'Stp%', 'icon': 'fa-solid fa-plane-slash', 'unit': '%', 'format_spec': '{:,.1f}', 'requires_min_minutes': True, 'df': df_gk}, {'title': 'Sweeper Actions p90', 'metric_col': '#OPA/90', 'icon': 'fa-solid fa-shoe-prints', 'requires_min_minutes': True, 'df': df_gk}]
    return dbc.Tabs([dbc.Tab(label="⚔️ Attacking", tab_id="tab-attacking", children=dbc.Row([create_metric_card(df_outfield, **m) for m in attacking_metrics], className="mt-4")), dbc.Tab(label="⚽ Possession", tab_id="tab-possession", children=dbc.Row([create_metric_card(df_outfield, **m) for m in possession_metrics], className="mt-4")), dbc.Tab(label="🛡️ Defending", tab_id="tab-defending", children=dbc.Row([create_metric_card(df_outfield, **m) for m in defensive_metrics], className="mt-4")), dbc.Tab(label="🧤 Goalkeeping", tab_id="tab-goalkeeping", children=dbc.Row([create_metric_card(m.pop('df', df_gk), **m) for m in goalkeeping_metrics], className="mt-4"))], id="player-stats-tabs", active_tab="tab-attacking")

def layout(player_name_url=None):
    if player_name_url:
        from pages import player_profile
        return player_profile.layout(player_name_url)
    available_seasons = get_available_processed_seasons()
    default_season = available_seasons[0] if available_seasons else "2024-2025"
    return dbc.Container([
        dcc.Store(id='player-stats-df-store'),
        dcc.Store(id='player-active-league-filter-store', data='All'),
        dbc.Row([
            dbc.Col(dbc.Button([html.I(className="fas fa-arrow-left me-2"), "Back to Home"], href="/", color="secondary"), width="auto"),
            dbc.Col(html.H1("Player Statistics Hub", className="text-center text-white"), width=True),
            dbc.Col(width="auto")
        ], align="center", className="my-4"),
        dbc.Row([
            dbc.Col(dcc.Dropdown(id='player-search-dropdown', placeholder="Search for a player..."), md=4),
            dbc.Col(create_league_filter_buttons(), md="auto"),
            dbc.Col(dcc.Dropdown(id='player-season-filter', options=available_seasons, value=default_season, clearable=False, style={'color': 'black'}), md=3),
        ], className="mb-4 align-items-center justify-content-center"),
        dcc.Loading(id="loading-main-content", type="circle", children=[
            html.Div(id='player-stats-content'),
            html.Div(id='player-quadrant-plots', className="mt-5")
        ]),
        app_signature()
    ], fluid=True, className="p-4")

# ... (tutti gli altri callback sono invariati, tranne gli ultimi due) ...
@callback([Output('player-stats-df-store', 'data'), Output('player-search-dropdown', 'options')], Input('player-season-filter', 'value'), prevent_initial_call=False)
def load_data_for_season(selected_season):
    if not selected_season: return None, []
    file_path = os.path.join(PROCESSED_DATA_PATH, f"player_stats_{selected_season}.parquet")
    if not os.path.exists(file_path): return None, []
    df = pd.read_parquet(file_path)
    if df.empty: return None, []
    search_options = [{'label': name, 'value': name} for name in sorted(df['Player'].unique())]
    return df.to_json(orient='split'), search_options
@callback(Output('player-stats-content', 'children'), [Input('player-active-league-filter-store', 'data'), Input('player-stats-df-store', 'data')])
def update_hub_content(selected_league, df_json):
    if df_json is None: return dbc.Alert("No data available for the selected season. Please run the processing script `create_player_dataset.py`.", color="danger", className="mt-4")
    df = pd.read_json(StringIO(df_json), orient='split')
    if df.empty: return dbc.Alert("No player data loaded for the selected season.", color="warning", className="mt-4")
    df_filtered = df if selected_league == 'All' else df[df['League'] == selected_league]
    if df_filtered.empty: return dbc.Alert(f"No player data found for {selected_league} in the selected season.", color="warning", className="mt-4")
    return generate_tabs_content(df_filtered)
@callback(Output('player-active-league-filter-store', 'data'), [Input(f"player-filter-{lf}", 'n_clicks') for lf in LEAGUES.keys()] + [Input('player-filter-all', 'n_clicks')], prevent_initial_call=True)
def update_active_league_filter(*clicks):
    triggered_id = ctx.triggered_id
    if not triggered_id or triggered_id == 'player-filter-all': return 'All'
    league_folder = triggered_id.replace('player-filter-', '')
    return LEAGUES[league_folder]['name']
@callback(Output('player-quadrant-plots', 'children'), [Input('player-stats-tabs', 'active_tab')], [State('player-active-league-filter-store', 'data'), State('player-stats-df-store', 'data')])
def update_player_quadrant_plots(active_tab, selected_league, df_json):
    if not df_json: return None
    df = pd.read_json(StringIO(df_json), orient='split')
    title_prefix = selected_league if selected_league != 'All' else "Top 20 European"
    if selected_league == 'All': source_df = df
    else: source_df = df[df['League'] == selected_league].copy()
    if source_df.empty or 'Pos' not in source_df.columns: return None
    df_copy = source_df.copy()
    df_copy = df_copy[df_copy['League_MP'] > 0]
    if 'League_MP' in df_copy.columns:
        df_copy['min_threshold'] = df_copy['League_MP'] * MIN_AVG_MINUTES_PER_GAME
        source_df = df_copy[df_copy['Min'] >= df_copy['min_threshold']]
    df_outfield = source_df[~source_df['Pos'].str.contains('GK', na=False)]
    df_gk = source_df[source_df['Pos'].str.contains('GK', na=False)]
    plot_definitions = {"tab-attacking": {"df": df_outfield, "title": f"{title_prefix} Attacking Contribution", "x_metric": "SoT/90", "y_metric": "SCA90", "x_label": "Shots on Target p90", "y_label": "Shot Creating Actions p90", "quadrant_labels": ['Elite Attacker', 'Finisher', 'Low Output', 'Creator']}, "tab-possession": {"df": df_outfield, "title": f"{title_prefix} Possession & Progression", "x_metric": "Carries_F3_per_90", "y_metric": "PrgP_per_90", "x_label": "Carries into Final Third p90", "y_label": "Progressive Passes p90", "quadrant_labels": ['Dual Threat', 'Primary Passer', 'Low Progression', 'Primary Carrier']}, "tab-defending": {"df": df_outfield, "title": f"{title_prefix} Defensive Activity", "x_metric": "Tkl+Int_per_90", "y_metric": "Aerial_Duels_perc", "x_label": "Tackles + Interceptions p90", "y_label": "Aerial Duels Won %", "quadrant_labels": ['Dominant Defender', 'Ground Warrior', 'Low Activity', 'Aerial Specialist']}, "tab-goalkeeping": {"df": df_gk, "title": f"{title_prefix} Goalkeeping Performance", "x_metric": "Save%", "y_metric": "PSxG+/-", "x_label": "Save Percentage", "y_label": "Post-Shot xG - GA", "quadrant_labels": ['Elite Shot-Stopper', 'Reliable', 'Under-performing', 'Saves the Impossible']}}
    if active_tab not in plot_definitions: return None
    plot_params = plot_definitions[active_tab]
    plot_df = plot_params.pop('df')
    plot_df = plot_df.sort_values(by=plot_params['y_metric'], ascending=False).head(20)
    if plot_df.empty: return None
    graph = dcc.Graph(figure=player_plots.create_quadrant_plot(plot_df, **plot_params))
    return dbc.Container([html.H2(f"{title_prefix} Quadrant Analysis", className="text-white text-center mb-4"), dbc.Row(dbc.Col(graph))], fluid=True)

# --- CALLBACK PER LA RICERCA DAL DROPDOWN ---
@callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input('player-search-dropdown', 'value'),
    prevent_initial_call=True
)
def redirect_to_player_page_from_search(selected_player):
    if selected_player:
        return f"/player-stats/{selected_player.replace(' ', '_')}"
    return no_update

# --- NUOVO CALLBACK PER I CLICK SUI NOMI DEI GIOCATORI NELLE LISTE ---
@callback(
    Output('url', 'pathname', allow_duplicate=True),
    Input({'type': 'player-redirect-button', 'index': ALL}, 'n_clicks'),
    prevent_initial_call=True
)
def redirect_to_player_page_from_list(n_clicks):
    # Controlla se un bottone è stato effettivamente cliccato
    if not any(n_clicks):
        return no_update
        
    # Ottieni l'ID del bottone che ha scatenato il callback
    player_name_url = ctx.triggered_id['index']
    
    return f"/player-stats/{player_name_url}"