import os
import pandas as pd
from io import StringIO
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

# Importa le funzioni di utilità
from src.utils.path_helpers import get_team_logo_path, get_player_photo_path
from src.visualization import player_plots
from src.utils.player_helpers import LEAGUES, METRIC_TOOLTIPS, get_nationality_flag_code
from src.components.brand_components import app_footer, stats_page_hero
from src.metrics.sportmonks import (
    PLAYER_METRIC_GROUPS,
    PLAYER_QUADRANTS,
    card_definitions,
    uses_sportmonks,
)

# --- CONFIGURATION ---
PROCESSED_DATA_PATH = os.path.join("data", "processed")
# A full-season 60-minutes-per-team-match rule removes many genuine leaders.
# Thirty minutes per team match is the standard dynamic eligibility used here:
# it preserves a meaningful sample while allowing high-impact rotation players.
MIN_AVG_MINUTES_PER_GAME = 30

def get_available_processed_seasons():
    if not os.path.isdir(PROCESSED_DATA_PATH): return []
    files = os.listdir(PROCESSED_DATA_PATH)
    seasons = [f.replace('player_stats_', '').replace('.parquet', '') for f in files if f.startswith('player_stats_') and f.endswith('.parquet')]
    return sorted(seasons, reverse=True)


def filter_qualified_players(df):
    """Apply the same dynamic minutes threshold to cards and quadrant plots."""
    if df.empty or not {'League_MP', 'Min'}.issubset(df.columns):
        return df.iloc[0:0].copy()
    qualified = df.copy()
    qualified['League_MP'] = pd.to_numeric(qualified['League_MP'], errors='coerce')
    qualified['Min'] = pd.to_numeric(qualified['Min'], errors='coerce')
    qualified = qualified[qualified['League_MP'] > 0]
    threshold = qualified['League_MP'] * MIN_AVG_MINUTES_PER_GAME
    return qualified[qualified['Min'] >= threshold].copy()

# --- LAYOUT & CALLBACKS ---
def create_league_filter_buttons():
    button_style = {'width': '80px', 'height': '60px', 'padding': '0.25rem', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center'}
    logo_style = {'height': '40px', 'width': 'auto', 'object-fit': 'contain'}
    buttons = [dbc.Button(
        html.I(className="fas fa-globe fa-2x"),
        id="player-filter-all",
        n_clicks=0,
        color="primary",
        outline=True,
        className="player-league-filter-btn active",
        style=button_style,
    )]
    buttons.append(dbc.Tooltip("All Leagues", target="player-filter-all"))
    for league_folder, league_info in LEAGUES.items():
        button_id = f"player-filter-{league_folder}"
        logo = html.Div([
            html.Span(
                league_info['fallback'],
                className="player-league-fallback",
            ),
            html.Img(
                src=league_info['logo'],
                alt=league_info['name'],
                title=league_info['name'],
                style={**logo_style, 'position': 'relative', 'zIndex': 1},
            ),
        ], className="player-league-logo-wrap")
        buttons.append(dbc.Button(
            logo,
            id=button_id,
            n_clicks=0,
            color="secondary",
            outline=True,
            className="player-league-filter-btn",
            style=button_style,
        ))
        buttons.append(dbc.Tooltip(league_info['name'], target=button_id))
    return dbc.ButtonGroup(buttons, id="player-league-filter-buttons")

def create_player_ranking_rows(
    top_players_df,
    metric_col,
    unit="",
    format_spec="{:,.2f}",
    secondary_metric_col=None,
    secondary_label="",
    secondary_format_spec="{:,.0f}",
    season=None,
):
    rows = []
    for i, player in top_players_df.iterrows():
        is_first = (i == top_players_df.index[0])
        player_name = " ".join(str(player.get('Player', 'N/A')).split())
        league_name, club_name = player.get('League', 'N/A'), player.get('Club', 'N/A')
        nationality_code = get_nationality_flag_code(
            player.get('Nationality_Code'),
            player.get('Nationality'),
        )
        photo_style = {'width': '56px', 'height': '56px'} if is_first else {'width': '38px', 'height': '38px'}
        name_class, metric_class = ("fs-5 fw-bold", "display-6 fw-bold") if is_first else ("fs-6", "fs-4")
        flag_img = html.Img(src=f"https://cdnjs.cloudflare.com/ajax/libs/flag-icon-css/7.2.1/flags/4x3/{nationality_code}.svg", className="player-ranking-flag") if nationality_code else ""
        club_logo_img = html.Img(src=player.get('team_image_path') or get_team_logo_path(league_name, club_name), className="player-ranking-club-logo")

        player_details = html.Div([
            html.Span([flag_img, player.get('Nationality', 'N/A')], className="me-2 d-inline-flex align-items-center"),
            html.Span([club_logo_img, club_name], className="d-inline-flex align-items-center")
        ], className="player-ranking-meta small")

        player_link = dcc.Link(
            player_name,
            href=(
                f"/player-stats/{player_name.replace(' ', '_')}?season={season}"
                if season else f"/player-stats/{player_name.replace(' ', '_')}"
            ),
            className=f"player-ranking-name text-decoration-none {name_class}",
        )

        secondary_value = None
        if secondary_metric_col and secondary_metric_col in player.index:
            raw_secondary = pd.to_numeric(player.get(secondary_metric_col), errors='coerce')
            if pd.notna(raw_secondary):
                secondary_value = f"{secondary_format_spec.format(raw_secondary)}{secondary_label}"
        metric_value = html.Div([
            html.Span(
                f"{format_spec.format(player[metric_col])}{unit}",
                className=f"player-ranking-value fw-bold {metric_class}",
            ),
            html.Small(secondary_value, className="player-ranking-secondary") if secondary_value else None,
        ], className="player-ranking-value-wrap text-end")

        row_content = dbc.Row([
            dbc.Col(html.Img(src=player.get('image_path') or get_player_photo_path(player_name), style=photo_style, className="player-ranking-photo"), width="auto", className="pe-2"),
            dbc.Col([
                player_link,
                player_details
            ], width=True, className="align-self-center"),
            dbc.Col(metric_value, width="auto", className="align-self-center")
        ], align="center", className="py-2")

        list_item_class = "player-ranking-item ranking-winner border-top-0" if is_first else "player-ranking-item"
        rows.append(dbc.ListGroupItem(row_content, color="dark", className=list_item_class))
    return rows

# ... (tutte le altre funzioni create_metric_card, generate_tabs_content sono invariate) ...
def create_metric_card(df, title, metric_col, icon, ascending=False, season=None, **kwargs):
    if (df.empty or metric_col not in df.columns or 'League_MP' not in df.columns
            or pd.to_numeric(df[metric_col], errors='coerce').notna().sum() == 0):
        return dbc.Col(dbc.Alert(f"Data not available for '{title}'", color="warning", className="h-100"))

    df_to_use = filter_qualified_players(df)

    minimum_column = kwargs.get('minimum_column')
    minimum_value = kwargs.get('minimum_value')
    if minimum_column and minimum_value is not None:
        if minimum_column not in df_to_use.columns:
            return dbc.Col(dbc.Alert(
                f"Run the Sportmonks normalizer to calculate '{title}'.",
                color="warning",
                className="h-100",
            ))
        opportunities = pd.to_numeric(df_to_use[minimum_column], errors='coerce').fillna(0)
        df_to_use = df_to_use[opportunities >= minimum_value]

    if df_to_use.empty:
        return dbc.Col(dbc.Alert(f"No players meet the eligibility requirements for '{title}'", color="info", className="h-100"))

    top_5 = df_to_use.sort_values(by=metric_col, ascending=ascending).head(5)

    header_id = f"player-card-header-{metric_col.replace('/', '-')}"
    card_header = dbc.CardHeader(dbc.Row([dbc.Col(html.I(className=f"{icon} me-2"), width="auto"), dbc.Col(html.H5(title, className="m-0"), width=True)], align="center"), id=header_id, className="player-metric-card-header text-center")
    tooltip = dbc.Tooltip(METRIC_TOOLTIPS.get(metric_col, "No description available."), target=header_id, placement="top")
    ranking_rows = create_player_ranking_rows(
        top_5,
        metric_col,
        unit=kwargs.get('unit', ""),
        format_spec=kwargs.get('format_spec', "{:,.2f}"),
        secondary_metric_col=kwargs.get('secondary_metric_col'),
        secondary_label=kwargs.get('secondary_label', ""),
        secondary_format_spec=kwargs.get('secondary_format_spec', "{:,.0f}"),
        season=season,
    )
    return dbc.Col([dbc.Card([card_header, dbc.CardBody(dbc.ListGroup(ranking_rows, flush=True), className="p-0")], className="player-metric-card h-100"), tooltip], lg=3, md=6, sm=12, className="mb-4")

def filter_outfield_role(df, selected_role):
    """Filter the outfield comparison cohort without affecting goalkeepers."""
    if 'Pos' not in df.columns:
        return df.iloc[0:0].copy()
    outfield = df[~df['Pos'].astype(str).str.contains('GK', na=False)].copy()
    if selected_role in {'DF', 'MF', 'FW'}:
        outfield = outfield[outfield['Pos'].astype(str).str.contains(selected_role, na=False)]
    return outfield


def apply_metric_minimums(df, metric_columns):
    """Apply registry-defined opportunity thresholds to a comparison cohort."""
    filtered = df.copy()
    specs_by_column = {
        spec.column: spec
        for group in PLAYER_METRIC_GROUPS.values()
        for spec in group
    }
    for metric_column in metric_columns:
        spec = specs_by_column.get(metric_column)
        if not spec or spec.minimum_column is None or spec.minimum_value is None:
            continue
        if spec.minimum_column not in filtered.columns:
            return filtered.iloc[0:0].copy()
        opportunities = pd.to_numeric(filtered[spec.minimum_column], errors='coerce').fillna(0)
        filtered = filtered[opportunities >= spec.minimum_value]
    return filtered


def select_quadrant_leaders(df, x_metric, y_metric, invert_x=False, invert_y=False, limit=20):
    """Select players who rank strongly across both quadrant axes."""
    leaders = df.copy()
    leaders[x_metric] = pd.to_numeric(leaders[x_metric], errors='coerce')
    leaders[y_metric] = pd.to_numeric(leaders[y_metric], errors='coerce')
    leaders = leaders.dropna(subset=[x_metric, y_metric])
    if leaders.empty:
        return leaders
    x_percentile = leaders[x_metric].rank(pct=True, ascending=not invert_x)
    y_percentile = leaders[y_metric].rank(pct=True, ascending=not invert_y)
    leaders['_quadrant_score'] = (x_percentile + y_percentile) / 2
    return leaders.sort_values('_quadrant_score', ascending=False).head(limit).drop(columns='_quadrant_score')


def generate_card_sections(df, selected_role='ALL', season=None):
    if df.empty:
        return tuple(
            dbc.Alert("No data available for the current selection.", color="info", className="m-3")
            for _ in range(4)
        )
    df_outfield = filter_outfield_role(df, selected_role)
    has_sportmonks = uses_sportmonks(df)
    if has_sportmonks:
        attacking_metrics = card_definitions(PLAYER_METRIC_GROUPS['attacking'], 'metric_col')
        possession_metrics = card_definitions(PLAYER_METRIC_GROUPS['creation'], 'metric_col')
        defensive_metrics = card_definitions(PLAYER_METRIC_GROUPS['defending'], 'metric_col')
        goalkeeping_metrics = card_definitions(PLAYER_METRIC_GROUPS['goalkeeping'], 'metric_col')
    else:
        attacking_metrics = [{'title': 'Top Scorers', 'metric_col': 'Gls', 'icon': 'fa-solid fa-futbol', 'format_spec': '{:,.0f}'}, {'title': 'Goals - xG per 90', 'metric_col': 'G_minus_xG_per_90', 'icon': 'fa-solid fa-chart-line'}, {'title': 'Shot-Creating Actions p90', 'metric_col': 'SCA90', 'icon': 'fa-solid fa-wand-magic-sparkles'}, {'title': 'Shots on Target p90', 'metric_col': 'SoT/90', 'icon': 'fa-solid fa-bullseye'}]
        possession_metrics = [{'title': 'Top Playmakers', 'metric_col': 'Ast', 'icon': 'fa-solid fa-hands-helping', 'format_spec': '{:,.0f}'}, {'title': 'Passes into Final Third p90', 'metric_col': 'Passes_F3_per_90', 'icon': 'fa-solid fa-arrow-right-to-bracket', 'requires_min_minutes': True}, {'title': 'Progressive Passes p90', 'metric_col': 'PrgP_per_90', 'icon': 'fa-solid fa-angles-up', 'requires_min_minutes': True}, {'title': 'Carries into Final Third p90', 'metric_col': 'Carries_F3_per_90', 'icon': 'fa-solid fa-arrow-trend-up', 'requires_min_minutes': True}]
        defensive_metrics = [{'title': 'Tackles + Interceptions p90', 'metric_col': 'Tkl+Int_per_90', 'icon': 'fa-solid fa-shield-halved', 'requires_min_minutes': True}, {'title': 'Aerial Duels Won %', 'metric_col': 'Aerial_Duels_perc', 'icon': 'fa-solid fa-plane-up', 'unit': '%', 'format_spec': '{:,.1f}', 'requires_min_minutes': True}, {'title': 'Clearances per 90', 'metric_col': 'Clr_per_90', 'icon': 'fa-solid fa-broom', 'requires_min_minutes': True}, {'title': 'Blocks per 90', 'metric_col': 'Blocks_per_90', 'icon': 'fa-solid fa-person-falling-burst', 'requires_min_minutes': True}]
        goalkeeping_metrics = [{'title': 'Save %', 'metric_col': 'Save%', 'icon': 'fa-solid fa-mitten', 'unit': '%', 'format_spec': '{:,.1f}', 'requires_min_minutes': True}, {'title': 'PSxG - Goals Against', 'metric_col': 'PSxG+/-', 'icon': 'fa-solid fa-chart-line', 'requires_min_minutes': True}, {'title': 'Crosses Stopped %', 'metric_col': 'Stp%', 'icon': 'fa-solid fa-plane-slash', 'unit': '%', 'format_spec': '{:,.1f}', 'requires_min_minutes': True}, {'title': 'Sweeper Actions p90', 'metric_col': '#OPA/90', 'icon': 'fa-solid fa-shoe-prints', 'requires_min_minutes': True}]
    df_gk = df[df['Pos'].str.contains('GK', na=False)] if 'Pos' in df.columns else pd.DataFrame()
    return (
        dbc.Row([create_metric_card(df_outfield, season=season, **metric) for metric in attacking_metrics]),
        dbc.Row([create_metric_card(df_outfield, season=season, **metric) for metric in possession_metrics]),
        dbc.Row([create_metric_card(df_outfield, season=season, **metric) for metric in defensive_metrics]),
        dbc.Row([create_metric_card(df_gk, season=season, **metric) for metric in goalkeeping_metrics]),
    )


def create_quadrant_guide(labels, descriptions, comparison_note):
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
        cards.append(dbc.Col(
            html.Div([
                html.Div([
                    html.Span(className=f"player-quadrant-swatch {position}"),
                    html.Span(screen_label, className="player-quadrant-position"),
                ], className="d-flex align-items-center gap-2 mb-2"),
                html.H5(labels.get(position, ""), className="player-quadrant-guide-title"),
                html.P(descriptions.get(position, ""), className="player-quadrant-description mb-0"),
            ], className=f"player-quadrant-guide-card {position}"),
            lg=3,
            md=6,
            sm=12,
        ))

    return html.Div([
        html.P([
            html.I(className="fa-solid fa-circle-info me-2"),
            comparison_note,
            " Dashed axes are cohort medians, so every quadrant describes a relative player profile.",
        ], className="player-quadrant-intro"),
        dbc.Row(cards, className="g-3"),
    ], className="player-quadrant-guide mt-3")

def layout(player_name_url=None, initial_season=None):
    if player_name_url:
        from pages import player_profile
        return player_profile.layout(player_name_url)
    available_seasons = get_available_processed_seasons()
    default_season = (
        initial_season
        if initial_season in available_seasons
        else (available_seasons[0] if available_seasons else "2024-2025")
    )
    return html.Main([
        stats_page_hero(
            title="Player Statistics",
            subtitle="Discover role-adjusted individual performance across Europe’s top five leagues.",
            eyebrow="PLAYER INTELLIGENCE",
            icon="fa-solid fa-user-shield",
            variant="player",
        ),
        dbc.Container([
            dcc.Store(id='player-stats-df-store'),
            dcc.Store(id='player-active-league-filter-store', data='All'),
            html.Div([
                dbc.Row([
                    dbc.Col(
                        dcc.Dropdown(
                            id='player-search-dropdown',
                            placeholder="Search for a player...",
                            className="player-search-dropdown",
                        ),
                        lg=5,
                        md=12,
                        className="player-search-col",
                    ),
                    dbc.Col(create_league_filter_buttons(), width="auto", className="player-league-filter-col"),
                    dbc.Col(
                        dcc.Dropdown(
                            id='player-season-filter',
                            options=[{'label': season, 'value': season} for season in available_seasons],
                            value=default_season,
                            clearable=False,
                            className="player-season-dropdown",
                        ),
                        width="auto",
                        className="player-season-col",
                    ),
                ], className="g-3 align-items-center justify-content-center"),
                html.Div([
                    html.Span("OUTFIELD ROLE", className="player-role-filter-label"),
                    dbc.RadioItems(
                        id='player-role-filter',
                        options=[
                            {'label': 'All', 'value': 'ALL'},
                            {'label': 'Defenders', 'value': 'DF'},
                            {'label': 'Midfielders', 'value': 'MF'},
                            {'label': 'Forwards', 'value': 'FW'},
                        ],
                        value='ALL',
                        inline=True,
                        className="player-role-filter",
                        inputClassName="btn-check",
                        labelClassName="player-role-filter-btn",
                        labelCheckedClassName="active",
                    ),
                ], className="player-role-filter-wrap mt-3"),
            ], className="player-stats-filter-bar mb-4"),
            html.P(
                "Rankings include players with at least 30 minutes played per league match completed by their competition. The role filter applies to outfield tabs; Goalkeeping always compares goalkeepers.",
                className="player-qualification-note text-center mb-3",
            ),
            dcc.Loading(id="loading-main-content", type="circle", children=[
                dbc.Tabs(
                    id="player-stats-tabs",
                    active_tab="tab-attacking",
                    children=[
                        dbc.Tab(label="⚔️ Attacking", tab_id="tab-attacking", children=html.Div(id="player-attacking-cards", className="p-2 mt-3")),
                        dbc.Tab(label="⚽ Creation & Ball Use", tab_id="tab-possession", children=html.Div(id="player-creation-cards", className="p-2 mt-3")),
                        dbc.Tab(label="🛡️ Defending", tab_id="tab-defending", children=html.Div(id="player-defending-cards", className="p-2 mt-3")),
                        dbc.Tab(label="🧤 Goalkeeping", tab_id="tab-goalkeeping", children=html.Div(id="player-goalkeeping-cards", className="p-2 mt-3")),
                    ],
                    className="player-metric-tabs mt-4",
                ),
                html.Div(id='player-quadrant-plots', className="mt-5")
            ]),
        ], fluid=True, className="player-stats-content px-3 px-lg-4"),
        app_footer("Role-adjusted player performance and positional intelligence."),
    ], className="player-stats-page branded-analytics-page")

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
@callback(
    [Output('player-attacking-cards', 'children'),
     Output('player-creation-cards', 'children'),
     Output('player-defending-cards', 'children'),
     Output('player-goalkeeping-cards', 'children')],
    [Input('player-active-league-filter-store', 'data'),
     Input('player-stats-df-store', 'data'),
     Input('player-role-filter', 'value'),
     Input('player-season-filter', 'value')],
)
def update_hub_content(selected_league, df_json, selected_role, selected_season):
    if df_json is None:
        return tuple(
            dbc.Alert("No data available for the selected season.", color="danger", className="m-3")
            for _ in range(4)
        )
    df = pd.read_json(StringIO(df_json), orient='split')
    if df.empty:
        return tuple(
            dbc.Alert("No player data loaded for the selected season.", color="warning", className="m-3")
            for _ in range(4)
        )
    df_filtered = df if selected_league == 'All' else df[df['League'] == selected_league]
    if df_filtered.empty:
        return tuple(
            dbc.Alert(f"No player data found for {selected_league} in the selected season.", color="warning", className="m-3")
            for _ in range(4)
        )
    return generate_card_sections(df_filtered, selected_role, selected_season)
@callback(Output('player-active-league-filter-store', 'data'), [Input(f"player-filter-{lf}", 'n_clicks') for lf in LEAGUES.keys()] + [Input('player-filter-all', 'n_clicks')], prevent_initial_call=True)
def update_active_league_filter(*clicks):
    triggered_id = ctx.triggered_id
    if not triggered_id or triggered_id == 'player-filter-all': return 'All'
    league_folder = triggered_id.replace('player-filter-', '')
    return LEAGUES[league_folder]['name']


@callback(
    [Output('player-filter-all', 'className')]
    + [Output(f"player-filter-{league_folder}", 'className') for league_folder in LEAGUES],
    Input('player-active-league-filter-store', 'data'),
)
def style_active_player_league_filter(selected_league):
    base_class = "player-league-filter-btn"
    classes = [f"{base_class} active" if selected_league == 'All' else base_class]
    classes.extend(
        f"{base_class} active" if selected_league == league_info['name'] else base_class
        for league_info in LEAGUES.values()
    )
    return classes


@callback(
    Output('player-quadrant-plots', 'children'),
    [Input('player-stats-tabs', 'active_tab'),
     Input('player-active-league-filter-store', 'data'),
     Input('player-stats-df-store', 'data'),
     Input('player-role-filter', 'value')],
)
def update_player_quadrant_plots(active_tab, selected_league, df_json, selected_role):
    if not df_json: return None
    df = pd.read_json(StringIO(df_json), orient='split')
    title_prefix = selected_league if selected_league != 'All' else "Top 20 European"
    if selected_league == 'All': source_df = df
    else: source_df = df[df['League'] == selected_league].copy()
    if source_df.empty or 'Pos' not in source_df.columns: return None
    source_df = filter_qualified_players(source_df)
    df_outfield = filter_outfield_role(source_df, selected_role)
    df_gk = source_df[source_df['Pos'].str.contains('GK', na=False)]
    has_sportmonks = uses_sportmonks(source_df)
    if has_sportmonks:
        plot_definitions = {
            key: {**dict(value), 'df': df_gk if key == 'tab-goalkeeping' else df_outfield}
            for key, value in PLAYER_QUADRANTS.items()
        }
    else:
        plot_definitions = {"tab-attacking": {"df": df_outfield, "title": "Attacking Contribution", "x_metric": "SoT/90", "y_metric": "SCA90", "x_label": "Shots on Target p90", "y_label": "Shot Creating Actions p90", "quadrant_labels": ['Elite Attacker', 'Finisher', 'Low Output', 'Creator']}, "tab-possession": {"df": df_outfield, "title": "Possession & Progression", "x_metric": "Carries_F3_per_90", "y_metric": "PrgP_per_90", "x_label": "Carries into Final Third p90", "y_label": "Progressive Passes p90", "quadrant_labels": ['Dual Threat', 'Primary Passer', 'Low Progression', 'Primary Carrier']}, "tab-defending": {"df": df_outfield, "title": "Defensive Activity", "x_metric": "Tkl+Int_per_90", "y_metric": "Aerial_Duels_perc", "x_label": "Tackles + Interceptions p90", "y_label": "Aerial Duels Won %", "quadrant_labels": ['Dominant Defender', 'Ground Warrior', 'Low Activity', 'Aerial Specialist']}, "tab-goalkeeping": {"df": df_gk, "title": "Goalkeeping Performance", "x_metric": "Save%", "y_metric": "PSxG+/-", "x_label": "Save Percentage", "y_label": "Post-Shot xG - GA", "quadrant_labels": ['Elite Shot-Stopper', 'Reliable', 'Under-performing', 'Saves the Impossible']}}
    if active_tab not in plot_definitions: return None
    plot_params = dict(plot_definitions[active_tab])
    plot_df = plot_params.pop('df')
    guide_descriptions = plot_params.pop('quadrant_guide', {})
    guide_labels = plot_params.get('quadrant_labels', {})
    plot_params['title'] = f"{title_prefix} {plot_params['title']}"
    if any(metric not in plot_df.columns or pd.to_numeric(plot_df[metric], errors='coerce').notna().sum() == 0 for metric in (plot_params['x_metric'], plot_params['y_metric'])): return None
    plot_df = apply_metric_minimums(plot_df, (plot_params['x_metric'], plot_params['y_metric']))
    plot_df = select_quadrant_leaders(
        plot_df,
        plot_params['x_metric'],
        plot_params['y_metric'],
        invert_x=plot_params.get('invert_x', False),
        invert_y=plot_params.get('invert_y', False),
    )
    if plot_df.empty: return None
    graph = dcc.Graph(
        figure=player_plots.create_quadrant_plot(plot_df, **plot_params),
        config={'displayModeBar': False, 'responsive': True},
        className="player-quadrant-graph",
    )
    comparison_note = (
        "The chart shows the 20 qualified players with the strongest combined percentile across both axes in the top five leagues."
        if selected_league == 'All'
        else f"The chart shows the 20 qualified {selected_league} players with the strongest combined percentile across both axes."
    )
    guide = create_quadrant_guide(guide_labels, guide_descriptions, comparison_note)
    return dbc.Container([
        html.H2(f"{title_prefix} Quadrant Analysis", className="player-quadrant-section-title text-center mb-4"),
        dbc.Row(dbc.Col(graph)),
        guide,
    ], fluid=True, className="player-quadrant-section")

# --- CALLBACK PER LA RICERCA DAL DROPDOWN ---
@callback(
    Output('url', 'pathname', allow_duplicate=True),
    Output('url', 'search', allow_duplicate=True),
    Input('player-search-dropdown', 'value'),
    State('player-season-filter', 'value'),
    prevent_initial_call=True
)
def redirect_to_player_page_from_search(selected_player, selected_season):
    if selected_player:
        return (
            f"/player-stats/{selected_player.replace(' ', '_')}",
            f"?season={selected_season}" if selected_season else "",
        )
    return no_update, no_update
