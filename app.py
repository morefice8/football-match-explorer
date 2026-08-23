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
from dash import Dash, html as dash_html, dcc, Input, Output, State, dash_table, no_update, callback, ctx, ALL
from dash.dependencies import State
import dash_bootstrap_components as dbc
import traceback
from datetime import datetime
import html
import dash
from dash.dependencies import ALL
import uuid
from dash import ctx
from flask import send_from_directory
from urllib.parse import parse_qs, unquote
from src.utils.path_helpers import get_team_logo_path
from src.utils.sequence_filtering import (
    filter_sequences_exact,
    make_carousel_controller,
    step_carousel,
)


# Import dei layout dalle pagine separate
from pages import home, player_stats, upload, database, match_analysis, team_stats, team_profile, player_profile

# Import delle funzioni di logica
from src.config import TEAM_NAME_TO_LOGO_CODE, LOGO_PREFIX, LOGO_EXTENSION, DEFAULT_LOGO_PATH
from src.visualization import pitch_plots, player_plots, buildup_plotly, defensive_transitions_plotly, offensive_transitions_plotly, set_piece_plotly, cross_plots, league_plots, formation_plotly, formations, pass_plotly
from src.data_processing import preprocess, pass_processing
from src.utils import mapping_loader
from src import config
from src.metrics import (
    pass_metrics,
    player_metrics,
    buildup_metrics,
    transition_metrics,
    sequence_outcome_metrics,
    set_piece_metrics,
    restart_metrics,
    cross_metrics,
    league_metrics,
    defensive_metrics,
    data_quality,
)

# Define colors
HCOL = getattr(config, 'DEFAULT_HCOL', 'tomato')
ACOL = getattr(config, 'DEFAULT_ACOL', 'skyblue')
VIOLET = getattr(config, 'VIOLET', '#a369ff')
GREEN = getattr(config, 'GREEN', '#69f900')
BG_COLOR = getattr(config, 'BG_COLOR', 'white')
LINE_COLOR = getattr(config, 'LINE_COLOR', 'black')

# App configuration
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True
)
server = app.server

# --- NUOVA SEZIONE: SERVIRE I FILE DALLA CARTELLA 'data' ---
# Questa rotta permette di accedere ai loghi tramite URL come /data/fbref/serie-a/.../logo.png
@app.server.route('/data/<path:filepath>')
def serve_data_files(filepath):
    """Serve un file dalla directory 'data' del progetto."""
    return send_from_directory('data', filepath)

# -----------------------------------------------------------------------------
# Main App Layout (Contenitore principale)
# -----------------------------------------------------------------------------
app.layout = dash_html.Div([
    dcc.Location(id='url', refresh=False),

    # Stores per i dati
    dcc.Store(id="store-df-match"),
    dcc.Store(id='store-uploaded-data', storage_type='session'),
    dcc.Store(id="store-player-stats-df"),
    dcc.Store(id="team-stats-full-df-store"),

    # Stores per i commenti e filtri
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
    dcc.Store(id="store-buildup-filter", storage_type="memory"),
    dcc.Store(id="store-def-transition-filter", data=None),
    dcc.Store(id="store-off-transition-filter", data=None),
    dcc.Store(id="store-set-piece-filter", data=None),
    dcc.Store(id="cross-filter-store", data=None),
    dcc.Store(id="cross-selection-store", data=None),
    dcc.Store(id="report-html-content-store"),

    # Contenitore dove verranno caricate le pagine
    dash_html.Div(id='page-content')
])

# -----------------------------------------------------------------------------
# Helper Functions (solo quelle usate nei callback di questo file)
# -----------------------------------------------------------------------------
def get_seasons(league):
    path = os.path.join("data", "matches", league)
    if not os.path.exists(path): return []
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

def get_matches(league, season):
    path = os.path.join("data", "matches", league, season, "partidos")
    if not os.path.exists(path): return []
    return [f for f in os.listdir(path) if f.endswith(".json")]

def parse_upload_contents(contents, filename):
    if not contents or not filename: return None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'json' in filename:
            return json.loads(decoded.decode('utf-8'))
        return None
    except Exception as e:
        print(f"Error parsing file {filename}: {e}")
        return None

def parse_match(filename):
    parts = filename.replace(".json", "").split("_")
    if len(parts) < 4: return None
    return {"round": parts[0], "home_team": parts[1], "away_team": "_".join(parts[2:-1]), "id": parts[-1], "file": filename}

def enrich_match_info_with_raw_metadata(
    json_data,
    match_info,
):
    """
    Add useful match metadata from the raw Opta payload
    to the compact match_info dictionary stored by the app.
    """

    if match_info is None:
        match_info = {}

    # ---------------------------------------------------------
    # FIND MATCH ROOT
    # ---------------------------------------------------------

    raw_match = {}

    if isinstance(json_data, dict):

        # Match-details feed:
        # {
        #     "match": [
        #         {
        #             "matchInfo": ...,
        #             "liveData": ...
        #         }
        #     ]
        # }
        match_list = json_data.get(
            'match'
        )

        if (
            isinstance(match_list, list)
            and match_list
            and isinstance(
                match_list[0],
                dict,
            )
        ):
            raw_match = match_list[0]

        # Single-match feed:
        # {
        #     "matchInfo": ...,
        #     "liveData": ...
        # }
        elif 'matchInfo' in json_data:
            raw_match = json_data

    if not raw_match:
        return match_info

    raw_match_info = (
        raw_match.get(
            'matchInfo',
            {}
        )
        or {}
    )

    live_data = (
        raw_match.get(
            'liveData',
            {}
        )
        or {}
    )

    # ---------------------------------------------------------
    # MATCHWEEK
    # ---------------------------------------------------------

    week = raw_match_info.get(
        'week'
    )

    if week not in (
        None,
        '',
    ):
        match_info['week'] = week

    # ---------------------------------------------------------
    # VENUE
    # ---------------------------------------------------------

    venue = (
        raw_match_info.get(
            'venue',
            {}
        )
        or {}
    )

    venue_name = (
        venue.get('longName')
        or venue.get('shortName')
        or ''
    )

    if venue_name:
        match_info[
            'venue_name'
        ] = venue_name

    # ---------------------------------------------------------
    # REFEREE
    # ---------------------------------------------------------

    match_details_extra = (
        live_data.get(
            'matchDetailsExtra'
        )
        or {}
    )

    # Defensive fallback in case another Opta payload
    # nests it inside matchDetails.
    if not match_details_extra:
        match_details_extra = (
            live_data
            .get(
                'matchDetails',
                {}
            )
            .get(
                'matchDetailsExtra',
                {}
            )
            or {}
        )

    officials = (
        match_details_extra.get(
            'matchOfficial',
            []
        )
        or []
    )

    main_referee = next(
        (
            official
            for official in officials
            if str(
                official.get(
                    'type',
                    ''
                )
            ).lower()
            == 'main'
        ),
        None,
    )

    if main_referee:

        referee_name = " ".join(
            part
            for part in [
                main_referee.get(
                    'firstName'
                ),
                main_referee.get(
                    'lastName'
                ),
            ]
            if part
        ).strip()

        if referee_name:
            match_info[
                'referee_name'
            ] = referee_name

    # ---------------------------------------------------------
    # HOME / AWAY CONTESTANT IDS
    # ---------------------------------------------------------

    contestant_positions = {}

    for contestant in (
        raw_match_info.get(
            'contestant',
            []
        )
        or []
    ):
        contestant_id = (
            contestant.get('id')
        )

        position = (
            contestant.get(
                'position'
            )
        )

        if (
            contestant_id
            and position
            in (
                'home',
                'away',
            )
        ):
            contestant_positions[
                contestant_id
            ] = position

    # ---------------------------------------------------------
    # GOALS
    # ---------------------------------------------------------
    # Goal extraction must run only after the full contestant map
    # is available. This is especially important for away goals and
    # own goals, where the benefiting team can differ from the scorer's
    # contestantId.

    goals = []
    seen_goals = set()

    def opposite_position(position):
        if position == 'home':
            return 'away'
        if position == 'away':
            return 'home'
        return position

    def event_has_qualifier(event, qualifier_id):
        for qualifier in (
            event.get('qualifier', [])
            or []
        ):
            try:
                current_id = int(
                    qualifier.get('qualifierId')
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if current_id == qualifier_id:
                return True

        return False

    def add_goal(goal_data, dedupe_key):
        if dedupe_key in seen_goals:
            return

        seen_goals.add(dedupe_key)
        goals.append(goal_data)

    # ---------------------------------------------------------
    # CASE 1:
    # Dedicated Opta goal feed when available.
    # ---------------------------------------------------------

    raw_goals = (
        live_data.get(
            'goal',
            []
        )
        or []
    )

    for goal in raw_goals:
        contestant_id = goal.get(
            'contestantId'
        )

        goal_type = str(
            goal.get(
                'type',
                'G',
            )
            or 'G'
        ).upper()

        team_position = (
            contestant_positions.get(
                contestant_id
            )
        )

        # Opta associates an own-goal record with the scorer's
        # contestant. For the match overview we need the team that
        # benefited from the goal, so flip home/away for OG events.
        if goal_type == 'OG':
            team_position = opposite_position(
                team_position
            )

        source_id = (
            goal.get('optaEventId')
            or goal.get('id')
        )

        if source_id is not None:
            dedupe_key = (
                'goal-feed',
                str(source_id),
            )
        else:
            dedupe_key = (
                'goal-feed-fallback',
                contestant_id,
                goal.get('scorerId'),
                goal.get('scorerName'),
                goal.get('periodId'),
                goal.get('timeMin'),
                goal.get('timeMinSec'),
                goal_type,
            )

        add_goal(
            {
                'team_position':
                    team_position,

                'scorer':
                    goal.get(
                        'scorerName'
                    )
                    or 'Unknown',

                'timeMin':
                    goal.get(
                        'timeMin'
                    ),

                'timeMinSec':
                    goal.get(
                        'timeMinSec'
                    ),

                'periodId':
                    goal.get(
                        'periodId'
                    ),

                'goal_type':
                    goal_type,
            },
            dedupe_key,
        )

    # ---------------------------------------------------------
    # CASE 2:
    # Eventing feed.
    #
    # In Match Eventing, goals may only exist as typeId = 16
    # events rather than in liveData.goal. Qualifier 9 marks a
    # penalty and qualifier 28 marks an own goal.
    # ---------------------------------------------------------

    if not goals:
        for event in (
            live_data.get(
                'event',
                []
            )
            or []
        ):
            try:
                event_type_id = int(
                    event.get('typeId')
                )
            except (
                TypeError,
                ValueError,
            ):
                continue

            if event_type_id != 16:
                continue

            contestant_id = event.get(
                'contestantId'
            )

            if event_has_qualifier(
                event,
                28,
            ):
                goal_type = 'OG'
            elif event_has_qualifier(
                event,
                9,
            ):
                goal_type = 'PG'
            else:
                goal_type = 'G'

            team_position = (
                contestant_positions.get(
                    contestant_id
                )
            )

            if goal_type == 'OG':
                team_position = opposite_position(
                    team_position
                )

            minute = event.get('timeMin')
            second = event.get('timeSec')

            try:
                time_min_sec = (
                    f"{int(minute)}:"
                    f"{int(second):02d}"
                )
            except (
                TypeError,
                ValueError,
            ):
                time_min_sec = None

            source_id = (
                event.get('id')
                or event.get('eventId')
            )

            if source_id is not None:
                dedupe_key = (
                    'eventing',
                    contestant_id,
                    str(source_id),
                )
            else:
                dedupe_key = (
                    'eventing-fallback',
                    contestant_id,
                    event.get('playerId'),
                    event.get('playerName'),
                    event.get('periodId'),
                    minute,
                    second,
                    goal_type,
                )

            add_goal(
                {
                    'team_position':
                        team_position,

                    'scorer':
                        event.get(
                            'playerName'
                        )
                        or 'Unknown',

                    'timeMin':
                        minute,

                    'timeMinSec':
                        time_min_sec,

                    'periodId':
                        event.get(
                            'periodId'
                        ),

                    'goal_type':
                        goal_type,
                },
                dedupe_key,
            )

    if goals:
        match_info[
            'goals'
        ] = goals

    return match_info

def get_team_logo_src_by_code(team_short_code):
    if not team_short_code: return DEFAULT_LOGO_PATH
    logo_filename = f"{LOGO_PREFIX}{str(team_short_code).upper()}{LOGO_EXTENSION}"
    return f"/assets/logos/{logo_filename}"

def match_section_header(title, description, icon, eyebrow="MATCH MODULE", actions=None):
    """Shared editorial header used by every Match Analysis module."""
    return dash_html.Div([
        dash_html.Div([
            dash_html.Div([
                dash_html.I(className=f"{icon} match-module-heading-icon"),
                dash_html.Div([
                    dash_html.Span(eyebrow, className="match-module-eyebrow"),
                    dash_html.H2(title, className="match-module-title"),
                    dash_html.P(description, className="match-module-description"),
                ]),
            ], className="match-module-heading-copy"),
            dash_html.Div(actions, className="match-module-heading-actions") if actions else None,
        ], className="match-module-heading-row")
    ], className="match-module-heading")


def match_kpi_card(icon, label, value, note=None, accent="blue"):
    """Small, reusable KPI card for the match workspace."""
    return dash_html.Div([
        dash_html.Div(dash_html.I(className=icon), className=f"match-kpi-icon match-kpi-icon--{accent}"),
        dash_html.Div([
            dash_html.Span(label, className="match-kpi-label"),
            dash_html.Strong(value, className="match-kpi-value"),
            dash_html.Small(note, className="match-kpi-note") if note else None,
        ], className="match-kpi-copy")
    ], className="match-kpi-card")

def sequence_filter_zero_state(
    sequence_label="sequences",
):
    """Shared zero-state for sequence filters with no exact matches."""
    return dbc.Alert(
        [
            dash_html.Strong(
                f"No {sequence_label} match the current filters."
            ),
            dash_html.Br(),
            dash_html.Span(
                "Adjust or reset one or more filters to continue."
            ),
        ],
        color="secondary",
        className="m-3",
    )


def render_sequence_comparison_panel(
    comparison,
    home_color,
    away_color,
    title="Sequence progression",
    description=None,
    funnel_keys=None,
    funnel_label_overrides=None,
    profile_labels=None,
    hint_text=None,
    funnel_tooltip_overrides=None,
):
    """
    Render a Home vs Away sequence comparison.

    The component is intentionally presentation-only:
    all calculations come from sequence_outcome_metrics.
    """

    if not comparison:
        return None

    funnel_label_overrides = (
        funnel_label_overrides
        or {}
    )

    funnel_tooltip_overrides = (
        funnel_tooltip_overrides
        or {}
    )

    default_funnel_tooltips = {
        'total_sequences': (
            "All detected sequences included in the comparison. "
            "Every percentage in this panel uses this total "
            "as its denominator."
        ),

        'reached_middle_third': (
            "The sequence reached x ≥ 33.33 through a controlled "
            "ball location."
        ),

        'reached_opposition_half': (
            "The sequence reached x ≥ 50 through a controlled "
            "ball location."
        ),

        'reached_final_third': (
            "The sequence reached x ≥ 66.67 through a controlled "
            "ball location."
        ),

        'entered_penalty_area': (
            "The sequence reached the penalty area at x ≥ 83 "
            "and y between 21.1 and 78.9. Successful action "
            "destinations count; unsuccessful destinations do not."
        ),

        'produced_shot': (
            "The sequence contained a Goal, Miss, "
            "Attempt Saved or Post event."
        ),

        'produced_goal': (
            "The sequence produced a goal."
        ),
    }

    default_funnel_tooltips.update(
        funnel_tooltip_overrides
    )

    profile_labels = (
        profile_labels
        or {}
    )

    if funnel_keys is None:
        funnel_keys = [
            'total_sequences',
            'reached_opposition_half',
            'reached_final_third',
            'entered_penalty_area',
            'produced_shot',
            'produced_goal',
        ]

    home_team = comparison.get(
        'home_team',
        'Home',
    )

    away_team = comparison.get(
        'away_team',
        'Away',
    )

    funnel_rows = [
        row
        for row in comparison.get(
            'funnel',
            []
        )
        if row.get('key') in funnel_keys
    ]

    total_row = next(
        (
            row
            for row in comparison.get(
                'funnel',
                []
            )
            if row.get('key')
            == 'total_sequences'
        ),
        {},
    )

    home_sample_size = int(
        total_row.get(
            'home_count',
            0,
        )
        or 0
    )

    away_sample_size = int(
        total_row.get(
            'away_count',
            0,
        )
        or 0
    )

    def format_percentage(value):
        try:
            return f"{float(value):.0f}%"
        except (TypeError, ValueError):
            return "0%"

    def metric_label(
        row,
    ):
        key = row.get('key')

        label = funnel_label_overrides.get(
            key,
            row.get(
                'label',
                key or '',
            ),
        )

        tooltip_text = (
            default_funnel_tooltips.get(
                key
            )
        )

        if not tooltip_text:
            return dash_html.Span(
                label,
                className="sequence-comparison-label-text",
            )

        tooltip_id = (
            "sequence-info-"
            f"{uuid.uuid4().hex}"
        )

        return dash_html.Div([

            dash_html.Span(
                label,
                className="sequence-comparison-label-text",
            ),

            dash_html.I(
                id=tooltip_id,
                className=(
                    "fa-regular "
                    "fa-circle-question "
                    "sequence-definition-icon"
                ),
            ),

            dbc.Tooltip(
                tooltip_text,
                target=tooltip_id,
                placement="top",
                delay={
                    "show": 250,
                    "hide": 80,
                },
            ),

        ], className="sequence-comparison-label-content")

    def funnel_row(row):
        home_percentage = float(
            row.get(
                'home_percentage',
                0,
            )
            or 0
        )

        away_percentage = float(
            row.get(
                'away_percentage',
                0,
            )
            or 0
        )

        return dash_html.Div([
            dash_html.Div([
                dash_html.Div([
                    dash_html.Strong(
                        str(
                            row.get(
                                'home_count',
                                0,
                            )
                        ),
                        className=(
                            "sequence-comparison-count"
                        ),
                    ),
                    dash_html.Span(
                        format_percentage(
                            home_percentage
                        ),
                        className=(
                            "sequence-comparison-percent"
                        ),
                    ),
                ], className="sequence-comparison-value"),

                dash_html.Div([
                    dash_html.Div(
                        className=(
                            "sequence-comparison-bar "
                            "sequence-comparison-bar--home"
                        ),
                        style={
                            "width":
                                f"{min(home_percentage, 100):.1f}%",
                            "backgroundColor":
                                home_color,
                        },
                    ),
                ], className="sequence-comparison-track"),
            ], className="sequence-comparison-side"),

            dash_html.Div(
                metric_label(row),
                className="sequence-comparison-label",
            ),

            dash_html.Div([
                dash_html.Div([
                    dash_html.Strong(
                        str(
                            row.get(
                                'away_count',
                                0,
                            )
                        ),
                        className=(
                            "sequence-comparison-count"
                        ),
                    ),
                    dash_html.Span(
                        format_percentage(
                            away_percentage
                        ),
                        className=(
                            "sequence-comparison-percent"
                        ),
                    ),
                ], className=(
                    "sequence-comparison-value "
                    "sequence-comparison-value--away"
                )),

                dash_html.Div([
                    dash_html.Div(
                        className=(
                            "sequence-comparison-bar "
                            "sequence-comparison-bar--away"
                        ),
                        style={
                            "width":
                                f"{min(away_percentage, 100):.1f}%",
                            "backgroundColor":
                                away_color,
                        },
                    ),
                ], className="sequence-comparison-track"),
            ], className="sequence-comparison-side"),
        ], className="sequence-comparison-row")

    outcome_rows = []

    for outcome in comparison.get(
        'outcomes',
        []
    ):
        outcome_rows.append(
            dash_html.Div([

                # -----------------------------------------
                # HOME
                # -----------------------------------------

                dash_html.Div([
                    dash_html.Strong(
                        str(
                            outcome.get(
                                'home_count',
                                0,
                            )
                        ),
                        className=(
                            "sequence-outcome-count"
                        ),
                    ),

                    dash_html.Span(
                        format_percentage(
                            outcome.get(
                                'home_percentage',
                                0,
                            )
                        ),
                        className=(
                            "sequence-outcome-percent"
                        ),
                    ),

                ], className=(
                    "sequence-outcome-value"
                )),

                # -----------------------------------------
                # OUTCOME
                # -----------------------------------------

                dash_html.Span(
                    outcome.get(
                        'label',
                        outcome.get(
                            'outcome',
                            '',
                        ),
                    ),
                    className=(
                        "sequence-outcome-label"
                    ),
                ),

                # -----------------------------------------
                # AWAY
                # -----------------------------------------

                dash_html.Div([
                    dash_html.Strong(
                        str(
                            outcome.get(
                                'away_count',
                                0,
                            )
                        ),
                        className=(
                            "sequence-outcome-count"
                        ),
                    ),

                    dash_html.Span(
                        format_percentage(
                            outcome.get(
                                'away_percentage',
                                0,
                            )
                        ),
                        className=(
                            "sequence-outcome-percent"
                        ),
                    ),

                ], className=(
                    "sequence-outcome-value "
                    "sequence-outcome-value--away"
                )),

            ], className="sequence-outcome-row")
        )

    profile = comparison.get(
        'profile',
        {},
    )

    home_profile = profile.get(
        'home',
        {},
    )

    away_profile = profile.get(
        'away',
        {},
    )

    def format_number(
        value,
        suffix="",
    ):
        try:
            if pd.isna(value):
                return "—"

            return (
                f"{float(value):.1f}"
                f"{suffix}"
            )
        except (
            TypeError,
            ValueError,
        ):
            return "—"

    def profile_metric(
        label,
        home_value,
        away_value,
    ):
        return dash_html.Div([
            dash_html.Strong(
                home_value,
                className="sequence-profile-value",
            ),
            dash_html.Span(
                label,
                className="sequence-profile-label",
            ),
            dash_html.Strong(
                away_value,
                className="sequence-profile-value",
            ),
        ], className="sequence-profile-row")

    def detail_header(
        middle_label,
    ):
        return dash_html.Div([
            dash_html.Strong(
                home_team,
                className=(
                    "sequence-detail-team "
                    "sequence-detail-team--home"
                ),
                style={
                    "color": home_color,
                },
            ),

            dash_html.Span(
                middle_label,
                className="sequence-detail-middle",
            ),

            dash_html.Strong(
                away_team,
                className=(
                    "sequence-detail-team "
                    "sequence-detail-team--away"
                ),
                style={
                    "color": away_color,
                },
            ),
        ], className="sequence-detail-header")

    return dash_html.Section([
        dash_html.Div([
            dash_html.Div([
                dash_html.Span(
                    "MATCH COMPARISON",
                    className="match-panel-eyebrow",
                ),
                dash_html.H3(
                    title,
                    className="match-panel-title",
                ),
                dash_html.P(
                    description
                    or (
                        "Independent milestones show how "
                        "far each sequence progressed."
                    ),
                    className="match-panel-description",
                ),
            ]),
            dash_html.Div([
                dash_html.I(
                    className="fa-solid fa-circle-info"
                ),
                dash_html.Span(
                    hint_text
                    or (
                        "Percentages use all detected "
                        "sequences as the denominator."
                    )
                ),
            ], className="match-panel-hint"),
        ], className="match-panel-header"),

        dash_html.Div([
            dash_html.Div([

                dash_html.Div([
                    dash_html.Strong(
                        home_team,
                        className="sequence-team-name",
                        style={
                            "color": home_color,
                        },
                    ),

                    dash_html.Small(
                        f"n = {home_sample_size}",
                        className="sequence-team-sample",
                    ),

                ], className="sequence-team-identity"),

                dash_html.Span(
                    "Milestone",
                    className="sequence-team-middle",
                ),

                dash_html.Div([
                    dash_html.Strong(
                        away_team,
                        className=(
                            "sequence-team-name "
                            "sequence-team-name--away"
                        ),
                        style={
                            "color": away_color,
                        },
                    ),

                    dash_html.Small(
                        f"n = {away_sample_size}",
                        className="sequence-team-sample",
                    ),

                ], className=(
                    "sequence-team-identity "
                    "sequence-team-identity--away"
                )),

            ], className="sequence-team-header"),

            *[
                funnel_row(row)
                for row in funnel_rows
            ],
        ], className="sequence-funnel"),

        dash_html.Div([

            dash_html.Div([
                dash_html.Span(
                    "SEQUENCE PROFILE",
                    className="match-panel-eyebrow",
                ),

                detail_header("Metric"),

                profile_metric(
                    profile_labels.get(
                        'duration',
                        "Avg active duration",
                    ),
                    format_number(
                        home_profile.get(
                            'avg_duration_seconds'
                        ),
                        "s",
                    ),
                    format_number(
                        away_profile.get(
                            'avg_duration_seconds'
                        ),
                        "s",
                    ),
                ),

                profile_metric(
                    profile_labels.get(
                        'passes',
                        "Avg completed passes",
                    ),
                    format_number(
                        home_profile.get(
                            'avg_completed_passes'
                        ),
                    ),
                    format_number(
                        away_profile.get(
                            'avg_completed_passes'
                        ),
                    ),
                ),
            ], className="sequence-profile-block"),

            dash_html.Div([
                dash_html.Span(
                    "HOW SEQUENCES ENDED",
                    className="match-panel-eyebrow",
                ),

                detail_header("Outcome"),

                dash_html.Div(
                    outcome_rows,
                    className="sequence-outcome-list",
                ),
            ], className="sequence-outcome-block"),
        ], className="sequence-comparison-detail"),
    ], className=(
        "match-panel "
        "sequence-comparison-panel"
    ))

# ---------------------------------------------------------------------------
# REL-10B — shared Data Coverage UI helpers
# ---------------------------------------------------------------------------

def _data_coverage_thresholds():
    return getattr(config, "DATA_COVERAGE_THRESHOLDS", {}) or {}


def _receiver_coverage_item(passes_df, *, label="Receiver coverage"):
    coverage = data_quality.receiver_coverage(passes_df)
    if coverage["eligible"] <= 0:
        return None

    return data_quality.coverage_item(
        key="receiver_coverage_pct",
        label=label,
        value=f"{coverage['resolved']} / {coverage['eligible']} reliable",
        detail=(
            f"{coverage['coverage_pct']:.1f}% · "
            f"{coverage['high']} high · {coverage['medium']} medium"
        ),
        threshold_value=coverage["coverage_pct"],
        thresholds=_data_coverage_thresholds(),
    )


def _coordinate_coverage_item(
    df,
    *,
    label="Valid coordinates",
    columns=("x", "y"),
):
    coverage = data_quality.coordinate_coverage(df, columns)
    if coverage["total"] <= 0:
        return None

    return data_quality.coverage_item(
        key="valid_coordinate_pct",
        label=label,
        value=f"{coverage['valid']} / {coverage['total']} valid",
        detail=(
            f"{coverage['coverage_pct']:.1f}% · "
            f"{coverage['invalid']} invalid"
        ),
        threshold_value=coverage["coverage_pct"],
        thresholds=_data_coverage_thresholds(),
    )


def _outcome_coverage_item(
    df,
    *,
    label="Known outcomes",
    column="outcome",
):
    coverage = data_quality.outcome_coverage(df, column=column)
    if coverage["total"] <= 0:
        return None

    return data_quality.coverage_item(
        key="known_outcome_pct",
        label=label,
        value=f"{coverage['unknown']} unknown",
        detail=(
            f"{coverage['known']} / {coverage['total']} known · "
            f"{coverage['known_pct']:.1f}%"
        ),
        threshold_value=coverage["known_pct"],
        thresholds=_data_coverage_thresholds(),
    )


def _sequence_outcome_coverage_item(
    sequence_df,
    *,
    sequence_id_column,
    label="Sequence outcomes",
    outcome_column="sequence_outcome_type",
):
    coverage = data_quality.sequence_outcome_coverage(
        sequence_df,
        sequence_id_column=sequence_id_column,
        outcome_column=outcome_column,
    )
    if coverage["total"] <= 0:
        return None

    return data_quality.coverage_item(
        key="known_outcome_pct",
        label=label,
        value=f"{coverage['unknown']} unknown",
        detail=(
            f"{coverage['known']} / {coverage['total']} known · "
            f"{coverage['known_pct']:.1f}%"
        ),
        threshold_value=coverage["known_pct"],
        thresholds=_data_coverage_thresholds(),
    )


def _sequence_retention_coverage_item(
    sequence_df,
    *,
    label="Sequence retention",
    metric_key="sequence_retention_pct",
):
    if not isinstance(sequence_df, pd.DataFrame):
        return None

    metadata = (
        getattr(sequence_df, "attrs", {})
        .get("data_coverage", {})
        or {}
    )
    if not metadata:
        return None

    candidates = int(metadata.get("sequence_candidates", 0) or 0)
    built = int(metadata.get("sequence_built", 0) or 0)
    if candidates <= 0:
        return None

    coverage = data_quality.sequence_retention(
        candidates=candidates,
        built=built,
    )

    return data_quality.coverage_item(
        key=metric_key,
        label=label,
        value=f"{coverage['built']} / {coverage['candidates']} built",
        detail=(
            f"{coverage['discarded']} discarded · "
            f"{coverage['retention_pct']:.1f}% retained"
        ),
        threshold_value=coverage["retention_pct"],
        thresholds=_data_coverage_thresholds(),
    )


def _carry_coverage_item(stats, *, label="Carry candidates"):
    coverage = data_quality.carry_candidate_coverage_from_stats(stats)
    if coverage["candidates"] <= 0:
        return None

    return data_quality.coverage_item(
        key="carry_inclusion_pct",
        label=label,
        value=f"{coverage['included']} / {coverage['candidates']} included",
        detail=(
            f"{coverage['excluded']} excluded · "
            f"{coverage['inclusion_pct']:.1f}% included"
        ),
        threshold_value=coverage["inclusion_pct"],
        thresholds=_data_coverage_thresholds(),
    )


def render_data_coverage_panel(items, *, note=None):
    """
    Small collapsed REL-10 panel.

    Only configured threshold breaches receive warning styling.
    Neutral metrics remain informative without being presented as failures.
    """
    items = [item for item in (items or []) if item]
    if not items:
        return None

    warning_count = sum(
        item.get("status") == "warning"
        for item in items
    )

    if warning_count:
        status_text = (
            f"{warning_count} threshold warning"
            f"{'s' if warning_count != 1 else ''}"
        )
        status_class = "data-coverage-summary-status is-warning"
    else:
        status_text = "No threshold warnings"
        status_class = "data-coverage-summary-status"

    metric_components = []

    for item in items:
        item_class = "data-coverage-item"
        if item.get("status") == "warning":
            item_class += " is-warning"

        metric_components.append(
            dash_html.Div([
                dash_html.Span(
                    item.get("label", ""),
                    className="data-coverage-label",
                ),
                dash_html.Strong(
                    item.get("value", "—"),
                    className="data-coverage-value",
                ),
                dash_html.Small(
                    item.get("detail", ""),
                    className="data-coverage-detail",
                ),
            ], className=item_class)
        )

    return dash_html.Details([
        dash_html.Summary([
            dash_html.Span([
                dash_html.I(
                    className="fa-solid fa-shield-halved",
                ),
                dash_html.Span(
                    "Data coverage",
                    className="data-coverage-title",
                ),
            ], className="data-coverage-summary-main"),
            dash_html.Span(
                status_text,
                className=status_class,
            ),
        ], className="data-coverage-summary"),

        dash_html.Div(
            metric_components,
            className="data-coverage-grid",
        ),

        (
            dash_html.P(
                note,
                className="data-coverage-note",
            )
            if note
            else None
        ),
    ], className=(
        "data-coverage-panel"
        + (" has-warning" if warning_count else "")
    ))

# -----------------------------------------------------------------------------
# CALLBACKS DI ROUTING E CARICAMENTO DATI
# -----------------------------------------------------------------------------

@callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    Input("url", "search")
)
def render_page_content(pathname, search):
    print(f"--- Router rendering for path: '{pathname}' ---")

    # Decodifica l'intero percorso per gestire caratteri speciali ovunque
    decoded_pathname = unquote(pathname)

    if decoded_pathname == "/upload":
        return upload.layout()
    elif decoded_pathname == "/database":
        return database.layout()
    elif decoded_pathname == "/team-stats":
        return team_stats.layout()
    elif decoded_pathname.startswith("/team-stats/team/"):
        team_name_url = decoded_pathname.split("/")[-1]

        available_team_seasons = team_stats.get_available_seasons()
        season = available_team_seasons[0] if available_team_seasons else "2024-2025"
        if search:
            query_params = parse_qs(search.lstrip('?'))
            if 'season' in query_params:
                season = query_params['season'][0]

        return team_profile.layout(team_name_url, season)

    elif decoded_pathname.startswith("/team-stats/league/"):
        league_name = decoded_pathname.split("/")[-1].replace('_', ' ')
        return dash_html.Div([
            dash_html.H1(f"League Detail Page: {league_name}"),
            dbc.Alert("This page is under construction.", color="info")
        ])
    elif decoded_pathname == "/player-stats":
        season = None
        if search:
            query_params = parse_qs(search.lstrip('?'))
            if 'season' in query_params:
                season = query_params['season'][0]
        return player_stats.layout(initial_season=season)

    elif decoded_pathname.startswith("/player-stats/"):
        parts = decoded_pathname.strip('/').split('/')
        if len(parts) > 1:
            player_name_url = parts[1]
            season = None
            if search:
                query_params = parse_qs(search.lstrip('?'))
                if 'season' in query_params:
                    season = query_params['season'][0]
            return player_profile.layout(player_name_url, season)
        else:
            return player_stats.layout()

    elif decoded_pathname and decoded_pathname.startswith("/match/"):
        match_id = decoded_pathname.split("/")[-1]
        return match_analysis.layout(match_id)

    return home.layout()

@callback(
    Output('store-uploaded-data', 'data'),
    Output('url', 'pathname', allow_duplicate=True),
    Output('upload-status-output', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def handle_upload(contents, filename):
    if not contents: return no_update, no_update, no_update
    print(f"--- Upload Handler: Processing '{filename}' ---")
    json_data = parse_upload_contents(contents, filename)
    if json_data is None: return no_update, no_update, dbc.Alert("Error parsing file. Please ensure it is a valid JSON.", color="danger", duration=4000)

    try:
        event_map = mapping_loader.load_opta_event_mapping(config.OPTA_EVENTS_XLSX)
        qualifier_map = mapping_loader.load_opta_qualifier_mapping(config.OPTA_QUALIFIERS_JSON)
        match_info = config.extract_match_info(json_data)
        match_info = (
            enrich_match_info_with_raw_metadata(
                json_data,
                match_info,
            )
        )
        df, _, _, _ = preprocess.process_opta_events(json_data, event_map, qualifier_map, match_info)
        if df is None or df.empty: return no_update, no_update, dbc.Alert("Processing resulted in empty data.", color="warning", duration=4000)

        match_id = f"upload-{uuid.uuid4().hex[:12]}"
        match_info['id'] = match_id
        data_to_store = {'df': df.to_json(date_format='iso', orient='split'), 'match_info': json.dumps(match_info)}
        new_pathname = f"/match/{match_id}"
        print(f"  Upload successful. Populating session store and redirecting to {new_pathname}")
        return data_to_store, new_pathname, dbc.Alert(f"Successfully processed {filename}!", color="success", duration=3000)
    except Exception as e:
        print(f"ERROR during processing: {traceback.format_exc()}")
        return no_update, no_update, dbc.Alert(f"An error occurred: {e}", color="danger", duration=5000)

@callback(
    Output('store-df-match', 'data'),
    Output('store-player-stats-df', 'data', allow_duplicate=True),
    Input('url', 'pathname'),
    State('store-uploaded-data', 'data'),
    prevent_initial_call=True
)
def populate_main_store(pathname, uploaded_data):
    print(f"--- Main Data Loader triggered for path: {pathname} ---")

    # Se andiamo a una pagina che NON è di analisi, puliamo gli store per sicurezza
    if not (pathname and pathname.startswith('/match/')):
        print("  Navigated to a non-match page. Clearing match-specific stores.")
        return None, no_update

    # Caso Upload: i dati sono nello store di sessione
    if 'upload-' in pathname:
        if uploaded_data:
            print(f"  Populating main store for {pathname} from session data.")
            return uploaded_data, None
        return no_update, no_update

    # Caso Database
    match_id = pathname.split('/')[-1]
    print(f"  Attempting to load match {match_id} from DB.")
    for league in database.get_leagues():
        for season in get_seasons(league):
            path = os.path.join("data", "matches", league, season, "partidos")
            if not os.path.exists(path): continue
            for file_name in os.listdir(path):
                if file_name.endswith(f"_{match_id}.json"):
                    try:
                        with open(os.path.join(path, file_name), 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                        event_map = mapping_loader.load_opta_event_mapping(config.OPTA_EVENTS_XLSX)
                        qualifier_map = mapping_loader.load_opta_qualifier_mapping(config.OPTA_QUALIFIERS_JSON)
                        match_info = config.extract_match_info(json_data)
                        match_info = (
                            enrich_match_info_with_raw_metadata(
                                json_data,
                                match_info,
                            )
                        )
                        parsed_info = parse_match(file_name)
                        if parsed_info: match_info['roundNameFromFilename'] = parsed_info['round']
                        df, _, _, _ = preprocess.process_opta_events(json_data, event_map, qualifier_map, match_info)
                        if df is None or df.empty: return None, None
                        print(f"  DB load successful for {match_id}")
                        return {'df': df.to_json(date_format='iso', orient='split'), 'match_info': json.dumps(match_info)}, None
                    except Exception:
                        print(f"  Error processing DB file.")
                        return None, None
    print(f"  Match ID {match_id} not found in DB. Clearing stores.")
    return None, None

# -----------------------------------------------------------------------------
# CALLBACKS PER LA PAGINA DEL DATABASE
# -----------------------------------------------------------------------------

@callback(
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

@callback(
    Output("dropdown-round", "options"),
    Output("dropdown-round", "value"),
    Input("dropdown-league", "value"),
    Input("dropdown-season", "value"),
    prevent_initial_call=True
)
def update_rounds(league, season):
    if league and season:
        matches = get_matches(league, season)
        # La funzione extract_rounds non è definita qui, va aggiunta alle helper o importata
        # Per ora la includo per completezza
        def extract_rounds(matches_list):
            rounds_data = set()
            for file_name in matches_list:
                original_round_name = file_name.split("_")[0]
                numeric_part_match = re.match(r"(\d+)", original_round_name)
                if numeric_part_match:
                    rounds_data.add((int(numeric_part_match.group(1)), original_round_name))
                else:
                    rounds_data.add((float('inf'), original_round_name))
            sorted_rounds_data = sorted(list(rounds_data), key=lambda x: (x[0], x[1]))
            return [name for _, name in sorted_rounds_data]

        rounds = extract_rounds(matches)
        options = [{"label": round_name, "value": round_name} for round_name in rounds]
        return options, None
    return [], None

@callback(
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
            match_info = config.extract_match_info(match_data)
            if match_info.get('hteamDisplayName'): teams.add(match_info['hteamDisplayName'])
            if match_info.get('ateamDisplayName'): teams.add(match_info['ateamDisplayName'])
        except Exception:
            parsed_info = parse_match(m_filename)
            if parsed_info:
                teams.add(parsed_info['home_team'].replace('_', ' '))
                teams.add(parsed_info['away_team'].replace('_', ' '))
    sorted_teams = sorted(list(teams))
    options = [{"label": team, "value": team} for team in sorted_teams]
    return options, None

@callback(
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
            filename_parsed_info = parse_match(m_filename)
            if not filename_parsed_info: continue
            if round_name_filter and filename_parsed_info['round'] != round_name_filter: continue

            match_file_path = os.path.join(base_path_matches, m_filename)
            with open(match_file_path, 'r', encoding='utf-8') as f:
                json_data_for_card = json.load(f)

            temp_match_info = config.extract_match_info(json_data_for_card)
            home_team_display = temp_match_info.get('hteamDisplayName')
            away_team_display = temp_match_info.get('ateamDisplayName')

            if team_filter and (team_filter not in [home_team_display, away_team_display]): continue

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
                try: match_details_for_card['numeric_round_sort_key'] = int(numeric_parts_round[0])
                except ValueError: pass

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
            continue

    if not matches_data_for_cards:
        return dbc.Alert("No matches found for this selection.", color="warning")

    def sort_key_for_card(match_data):
        date_for_sort = match_data.get('date_iso_for_sort', datetime.max)
        if date_for_sort is None: date_for_sort = datetime.max
        return (match_data['numeric_round_sort_key'], match_data['original_round_name'], date_for_sort, match_data.get('home_team_display_name', ''))

    sorted_matches_data = sorted(matches_data_for_cards, key=sort_key_for_card)

    cards = []
    logo_style = {"height": "40px", "width": "40px", "objectFit": "contain", "marginRight": "8px", "marginLeft": "8px"}
    for match_data in sorted_matches_data:
        pbi = match_data['parsed_base_info']
        home_logo_src = get_team_logo_src_by_code(match_data['home_team_code_for_logo'])
        away_logo_src = get_team_logo_src_by_code(match_data['away_team_code_for_logo'])

        score_display = [dash_html.Span("vs", className="mx-2")]
        if match_data['home_score'] is not None and match_data['away_score'] is not None:
            score_display = [dash_html.Span(f"{match_data['home_score']}", className="fw-bold fs-5"), dash_html.Span("-", className="mx-2"), dash_html.Span(f"{match_data['away_score']}", className="fw-bold fs-5")]

        header_content = [dash_html.Span(f"Round: {match_data['original_round_name']}", className="me-3")]
        if match_data['date_formatted_for_display'] != "N/A": header_content.append(dash_html.Span(f"{match_data['date_formatted_for_display']}"))

        card = dbc.Col(
            dbc.Card([
                dbc.CardHeader(dash_html.Div(header_content, className="small text-muted text-center")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([dash_html.Img(src=home_logo_src, style=logo_style), dash_html.Span(match_data['home_team_display_name'], className="fw-bold")], width="auto", className="d-flex align-items-center justify-content-end"),
                        dbc.Col(score_display, width="auto", className="d-flex align-items-center justify-content-center px-0"),
                        dbc.Col([dash_html.Img(src=away_logo_src, style=logo_style), dash_html.Span(match_data['away_team_display_name'], className="fw-bold")], width="auto", className="d-flex align-items-center justify-content-start")
                    ], justify="center", align="center", className="my-3"),
                    dbc.Button("View Match", color="primary", href=f"/match/{pbi['id']}", className="w-100 mt-auto")
                ], className="d-flex flex-column")
            ], className="mb-4 shadow-sm h-100"),
            lg=4, md=6, sm=12
        )
        cards.append(card)
    return dbc.Row(cards)

# -----------------------------------------------------------------------------
# CALLBACKS PER LA PAGINA DI ANALISI
# -----------------------------------------------------------------------------

@app.callback(
    Output("sidebar-match-header", "children"),
    Input("store-df-match", "data"),
    Input("url", "pathname")
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

    if not stored_data_json:
        return dash_html.Div(default_header_content)

    try:
        match_info_json_str = stored_data_json.get('match_info')
        if not match_info_json_str:
            return dash_html.Div(default_header_content)

        match_info = json.loads(match_info_json_str)

        hteam_display_name = match_info.get('hteamDisplayName', 'Home')
        ateam_display_name = match_info.get('ateamDisplayName', 'Away')

        home_score = match_info.get('home_score')
        away_score = match_info.get('away_score')

        competition = match_info.get('competitionName', '')
        round_name_from_file = match_info.get('roundNameFromFilename', '')
        game_date = match_info.get('date_formatted', '')

        # --- MODIFICA CHIAVE QUI ---
        # Usiamo la nuova funzione helper che prende il nome della competizione e il nome della squadra
        home_logo_src = get_team_logo_path(competition, hteam_display_name)
        away_logo_src = get_team_logo_path(competition, ateam_display_name)
        # ---------------------------

        sidebar_logo_style = {"height": "28px", "width": "28px", "objectFit": "contain"}
        team_name_style = {"fontSize": "0.9rem"}

        line1_display_text = f"{competition} - {round_name_from_file}" if competition and round_name_from_file else competition or round_name_from_file

        home_team_elements = [
            dbc.Col(dash_html.Img(src=home_logo_src, style=sidebar_logo_style), width="auto", className="pe-2 align-self-center"),
            dbc.Col(dash_html.Span(hteam_display_name, className="fw-bold", style=team_name_style), width=True, className="align-self-center text-start"),
        ]
        if home_score is not None:
            home_team_elements.append(dbc.Col(dash_html.Span(str(home_score), className="fw-bold fs-5"), width="auto", className="ps-2 align-self-center"))
        home_team_display_row = dbc.Row(home_team_elements, align="center", className="mb-1 gx-2")

        away_team_elements = [
            dbc.Col(dash_html.Img(src=away_logo_src, style=sidebar_logo_style), width="auto", className="pe-2 align-self-center"),
            dbc.Col(dash_html.Span(ateam_display_name, className="fw-bold", style=team_name_style), width=True, className="align-self-center text-start"),
        ]
        if away_score is not None:
            away_team_elements.append(dbc.Col(dash_html.Span(str(away_score), className="fw-bold fs-5"), width="auto", className="ps-2 align-self-center"))
        away_team_display_row = dbc.Row(away_team_elements, align="center", className="gx-2")

        header_content_list = []
        if line1_display_text:
            header_content_list.append(dash_html.P(line1_display_text, className="mb-2 small text-muted opacity-75 text-center"))

        header_content_list.append(home_team_display_row)
        header_content_list.append(away_team_display_row)

        if game_date:
            header_content_list.append(dash_html.P(game_date, className="mt-2 small text-muted opacity-75 text-center mb-0"))

        return dash_html.Div(header_content_list)

    except Exception as e:
        tb_str = traceback.format_exc()
        print(f"Error updating sidebar header: {e}\n{tb_str}")
        return dash_html.Div([
            dash_html.H5(f"Match ID: {match_id_from_url}", className="mb-1"),
            dash_html.P("Error loading details.", className="small text-danger")
        ])

@app.callback(
    Output("match-tab-content", "children"),
    Input("url", "search"),  # Listen to query parameters like ?tab=formation
    Input("store-df-match", "data")   # Depends on the match data being loaded
)
def render_match_tab_content(search_query, stored_data_json):
    if not stored_data_json:
            return dbc.Row(dbc.Col(dbc.Spinner(color="primary"), className="text-center mt-5"))

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
        if not stored_data_json:
            return dash_html.P(
                "No data available for overview.",
                style={"color": "orange"},
            )

        try:
            # -------------------------------------------------
            # LOAD DATA
            # -------------------------------------------------

            df_json_str = stored_data_json.get('df')

            if not df_json_str:
                return dash_html.P(
                    "DataFrame missing in stored data.",
                    style={"color": "orange"},
                )

            df = pd.read_json(
                io.StringIO(df_json_str),
                orient='split',
            )

            if df.empty:
                return dash_html.P(
                    "The DataFrame is empty.",
                    style={"color": "orange"},
                )

            match_info = json.loads(
                stored_data_json.get(
                    'match_info',
                    '{}',
                )
            )

            # Use internal team names for filtering.
            home_team = match_info.get(
                'hteamName',
                'Home',
            )

            away_team = match_info.get(
                'ateamName',
                'Away',
            )

            # Use display names in the UI where available.
            home_display = match_info.get(
                'hteamDisplayName',
                home_team,
            )

            away_display = match_info.get(
                'ateamDisplayName',
                away_team,
            )

            home_score = match_info.get(
                'home_score',
                0,
            )

            away_score = match_info.get(
                'away_score',
                0,
            )

            competition = match_info.get(
                'competitionName',
                '',
            )

            game_date = match_info.get(
                'date_formatted',
                '',
            )

            round_name = match_info.get(
                'roundNameFromFilename',
                '',
            )

            week = match_info.get(
                'week'
            )

            venue_name = match_info.get(
                'venue_name',
                '',
            )

            referee_name = match_info.get(
                'referee_name',
                '',
            )

            match_goals = match_info.get(
                'goals',
                [],
            ) or []

            if week not in (
                None,
                '',
            ):
                round_label = (
                    f"Matchweek {week}"
                )

            elif round_name:
                round_label = (
                    f"Round {round_name}"
                )

            else:
                round_label = ''

            def format_goal_minute(
                goal,
            ):
                try:
                    minute = int(
                        goal.get(
                            'timeMin'
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    return "—"

                try:
                    period_id = int(
                        goal.get(
                            'periodId'
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    period_id = None

                # Football-style injury time.
                if (
                    period_id == 1
                    and minute > 45
                ):
                    return (
                        f"45+{minute - 45}'"
                    )

                if (
                    period_id == 2
                    and minute > 90
                ):
                    return (
                        f"90+{minute - 90}'"
                    )

                return f"{minute}'"


            def goal_suffix(
                goal,
            ):
                goal_type = str(
                    goal.get(
                        'goal_type',
                        'G',
                    )
                ).upper()

                if goal_type == 'PG':
                    return " (P)"

                if goal_type == 'OG':
                    return " (OG)"

                return ""


            home_goals = [
                goal
                for goal in match_goals
                if goal.get(
                    'team_position'
                ) == 'home'
            ]

            away_goals = [
                goal
                for goal in match_goals
                if goal.get(
                    'team_position'
                ) == 'away'
            ]

            def scorer_list(
                goals,
                is_away=False,
            ):
                if not goals:
                    return None

                return dash_html.Div(
                    [
                        dash_html.Div([

                            dash_html.Span(
                                (
                                    f"{goal.get('scorer', 'Unknown')}"
                                    f"{goal_suffix(goal)}"
                                ),
                                className=(
                                    "overview-scorer-name"
                                ),
                            ),

                            dash_html.Span(
                                format_goal_minute(
                                    goal
                                ),
                                className=(
                                    "overview-scorer-minute"
                                ),
                            ),

                        ], className=(
                            "overview-scorer "
                            + (
                                "overview-scorer--away"
                                if is_away
                                else ""
                            )
                        ))

                        for goal in goals
                    ],
                    className=(
                        "overview-scorers "
                        + (
                            "overview-scorers--away"
                            if is_away
                            else ""
                        )
                    ),
                )

            # -------------------------------------------------
            # LOGOS
            # -------------------------------------------------

            home_logo = get_team_logo_path(
                competition,
                home_display,
            )

            away_logo = get_team_logo_path(
                competition,
                away_display,
            )

            # -------------------------------------------------
            # BASIC EVENTS
            # -------------------------------------------------

            event_names = df.get(
                'type_name',
                pd.Series(
                    index=df.index,
                    dtype='object',
                ),
            )

            teams = df.get(
                'team_name',
                pd.Series(
                    index=df.index,
                    dtype='object',
                ),
            )

            shot_names = [
                'Goal',
                'Miss',
                'Attempt Saved',
                'Post',
            ]

            home_shots = int(
                (
                    (teams == home_team)
                    & event_names.isin(
                        shot_names
                    )
                ).sum()
            )

            away_shots = int(
                (
                    (teams == away_team)
                    & event_names.isin(
                        shot_names
                    )
                ).sum()
            )

            home_recoveries = int(
                (
                    (teams == home_team)
                    & (
                        event_names
                        == 'Ball recovery'
                    )
                ).sum()
            )

            away_recoveries = int(
                (
                    (teams == away_team)
                    & (
                        event_names
                        == 'Ball recovery'
                    )
                ).sum()
            )

            # -------------------------------------------------
            # PASSING
            # -------------------------------------------------

            all_passes = (
                pass_processing.get_passes_df(
                    df.copy()
                )
            )

            home_pass_df = all_passes[
                all_passes['team_name']
                == home_team
            ].copy()

            away_pass_df = all_passes[
                all_passes['team_name']
                == away_team
            ].copy()

            home_passes = len(
                home_pass_df
            )

            away_passes = len(
                away_pass_df
            )

            home_completed_passes = int(
                (
                    home_pass_df['outcome']
                    == 'Successful'
                ).sum()
            )

            away_completed_passes = int(
                (
                    away_pass_df['outcome']
                    == 'Successful'
                ).sum()
            )

            def percentage(
                numerator,
                denominator,
            ):
                if not denominator:
                    return 0.0

                return (
                    numerator
                    / denominator
                    * 100
                )

            home_pass_completion = percentage(
                home_completed_passes,
                home_passes,
            )

            away_pass_completion = percentage(
                away_completed_passes,
                away_passes,
            )

            # -------------------------------------------------
            # PROGRESSIVE PASSES
            # -------------------------------------------------

            if (
                'is_progressive_attempt'
                in all_passes.columns
            ):
                progressive_mask = (
                    all_passes[
                        'is_progressive_attempt'
                    ]
                    .fillna(False)
                    .astype(bool)
                )

                progressive_passes = (
                    all_passes[
                        progressive_mask
                    ]
                    .copy()
                )

                home_progressive = int(
                    (
                        (
                            progressive_passes[
                                'team_name'
                            ]
                            == home_team
                        )
                        & (
                            progressive_passes[
                                'outcome'
                            ]
                            == 'Successful'
                        )
                    ).sum()
                )

                away_progressive = int(
                    (
                        (
                            progressive_passes[
                                'team_name'
                            ]
                            == away_team
                        )
                        & (
                            progressive_passes[
                                'outcome'
                            ]
                            == 'Successful'
                        )
                    ).sum()
                )

            else:
                home_progressive = 0
                away_progressive = 0

            # -------------------------------------------------
            # FINAL-THIRD ENTRIES
            # -------------------------------------------------

            successful_passes = (
                all_passes[
                    all_passes['outcome']
                    == 'Successful'
                ]
                .copy()
            )

            carries_df = (
                pass_processing.infer_carries(
                    df.copy()
                )
            )

            def final_third_entries_for_team(
                team_name,
            ):
                team_passes = (
                    successful_passes[
                        successful_passes[
                            'team_name'
                        ]
                        == team_name
                    ]
                    .copy()
                )

                if (
                    carries_df is not None
                    and not carries_df.empty
                    and 'team_name'
                    in carries_df.columns
                ):
                    team_carries = (
                        carries_df[
                            carries_df[
                                'team_name'
                            ]
                            == team_name
                        ]
                        .copy()
                    )
                else:
                    team_carries = (
                        pd.DataFrame()
                    )

                _, stats = (
                    pass_metrics
                    .analyze_final_third_entries(
                        team_passes,
                        team_carries,
                    )
                )

                return int(
                    stats.get(
                        'total_final_third',
                        0,
                    )
                )

            home_final_third = (
                final_third_entries_for_team(
                    home_team
                )
            )

            away_final_third = (
                final_third_entries_for_team(
                    away_team
                )
            )

            # -------------------------------------------------
            # CROSSES
            # -------------------------------------------------

            home_crosses_df = (
                cross_metrics.analyze_crosses(
                    df,
                    home_team,
                )
            )

            away_crosses_df = (
                cross_metrics.analyze_crosses(
                    df,
                    away_team,
                )
            )

            home_crosses = (
                len(home_crosses_df)
                if home_crosses_df
                is not None
                else 0
            )

            away_crosses = (
                len(away_crosses_df)
                if away_crosses_df
                is not None
                else 0
            )

            # -------------------------------------------------
            # COMPARISON COMPONENT
            # -------------------------------------------------

            def comparison_row(
                label,
                home_value,
                away_value,
                *,
                home_display_value=None,
                away_display_value=None,
                description=None,
            ):
                home_numeric = float(
                    home_value or 0
                )

                away_numeric = float(
                    away_value or 0
                )

                maximum = max(
                    home_numeric,
                    away_numeric,
                    1,
                )

                home_width = (
                    home_numeric
                    / maximum
                    * 100
                )

                away_width = (
                    away_numeric
                    / maximum
                    * 100
                )

                if home_display_value is None:
                    home_display_value = (
                        str(home_value)
                    )

                if away_display_value is None:
                    away_display_value = (
                        str(away_value)
                    )

                return dash_html.Div([

                    # HOME
                    dash_html.Div([

                        dash_html.Strong(
                            home_display_value,
                            className=(
                                "overview-comparison-value"
                            ),
                        ),

                        dash_html.Div(
                            dash_html.Span(
                                style={
                                    "width":
                                        f"{home_width:.1f}%",
                                    "backgroundColor":
                                        HCOL,
                                }
                            ),
                            className=(
                                "overview-comparison-track "
                                "overview-comparison-track--home"
                            ),
                        ),

                    ], className=(
                        "overview-comparison-side"
                    )),

                    # METRIC
                    dash_html.Div([

                        dash_html.Strong(
                            label,
                            className=(
                                "overview-comparison-label"
                            ),
                        ),

                        dash_html.Small(
                            description,
                            className=(
                                "overview-comparison-description"
                            ),
                        )
                        if description
                        else None,

                    ], className=(
                        "overview-comparison-middle"
                    )),

                    # AWAY
                    dash_html.Div([

                        dash_html.Strong(
                            away_display_value,
                            className=(
                                "overview-comparison-value "
                                "overview-comparison-value--away"
                            ),
                        ),

                        dash_html.Div(
                            dash_html.Span(
                                style={
                                    "width":
                                        f"{away_width:.1f}%",
                                    "backgroundColor":
                                        ACOL,
                                }
                            ),
                            className=(
                                "overview-comparison-track "
                                "overview-comparison-track--away"
                            ),
                        ),

                    ], className=(
                        "overview-comparison-side"
                    )),

                ], className=(
                    "overview-comparison-row"
                ))

            # -------------------------------------------------
            # MATCH META
            # -------------------------------------------------

            def overview_meta_item(
                icon,
                text,
            ):
                if not text:
                    return None

                return dash_html.Div([
                    dash_html.I(
                        className=icon
                    ),

                    dash_html.Span(
                        text
                    ),

                ], className=(
                    "overview-meta-item"
                ))


            meta_items = [
                overview_meta_item(
                    "fa-solid fa-trophy",
                    competition,
                ),

                overview_meta_item(
                    "fa-regular fa-calendar",
                    round_label,
                ),

                overview_meta_item(
                    "fa-regular fa-calendar-days",
                    game_date,
                ),

                overview_meta_item(
                    "fa-solid fa-location-dot",
                    venue_name,
                ),

                overview_meta_item(
                    "fa-solid fa-user-tie",
                    (
                        f"Referee: {referee_name}"
                        if referee_name
                        else ''
                    ),
                ),
            ]

            meta_items = [
                item
                for item in meta_items
                if item is not None
            ]

            # -------------------------------------------------
            # RETURN
            # -------------------------------------------------

            return dash_html.Div([

                match_section_header(
                    "Match overview",
                    (
                        "The key match indicators at a "
                        "glance before exploring the "
                        "detailed analysis."
                    ),
                    "fa-solid fa-chart-simple",
                    eyebrow="GAME STATE",
                    actions=[
                        dbc.Button(
                            [
                                dash_html.I(
                                    className=(
                                        "fas fa-download me-2"
                                    )
                                ),
                                "Download full CSV",
                            ],
                            id="btn-download-csv",
                            className=(
                                "match-action-button"
                            ),
                            size="sm",
                        ),
                    ],
                ),

                # =============================================
                # SCORE HERO
                # =============================================

                dash_html.Section([

                    dash_html.Div([

                        # HOME
                        dash_html.Div([

                            dash_html.Img(
                                src=home_logo,
                                className=(
                                    "overview-team-logo"
                                ),
                            ),

                            dash_html.Div([

                                dash_html.Span(
                                    "HOME",
                                    className="overview-team-role",
                                ),

                                dash_html.Strong(
                                    home_display,
                                    className="overview-team-name",
                                ),

                                scorer_list(
                                    home_goals,
                                    is_away=False,
                                ),

                            ]),

                        ], className=(
                            "overview-team "
                            "overview-team--home"
                        )),

                        # SCORE
                        dash_html.Div([

                            dash_html.Strong(
                                (
                                    f"{home_score}"
                                    f" – "
                                    f"{away_score}"
                                ),
                                className=(
                                    "overview-score"
                                ),
                            ),

                            dash_html.Span(
                                "FULL TIME",
                                className=(
                                    "overview-score-label"
                                ),
                            ),

                        ], className=(
                            "overview-score-block"
                        )),

                        # AWAY
                        dash_html.Div([

                            dash_html.Div([

                                dash_html.Span(
                                    "AWAY",
                                    className="overview-team-role",
                                ),

                                dash_html.Strong(
                                    away_display,
                                    className="overview-team-name",
                                ),

                                scorer_list(
                                    away_goals,
                                    is_away=True,
                                ),

                            ]),

                            dash_html.Img(
                                src=away_logo,
                                className=(
                                    "overview-team-logo"
                                ),
                            ),

                        ], className=(
                            "overview-team "
                            "overview-team--away"
                        )),

                    ], className="overview-score-hero"),

                    dash_html.Div(
                        meta_items,
                        className="overview-match-meta",
                    )
                    if meta_items
                    else None,

                ], className=(
                    "match-panel "
                    "overview-score-panel"
                )),

                # =============================================
                # QUICK KPIs
                # =============================================

                dash_html.Div([

                    match_kpi_card(
                        "fa-solid fa-arrow-right-arrow-left",
                        "Passes",
                        (
                            f"{home_passes} – "
                            f"{away_passes}"
                        ),
                        "Home – Away",
                        "blue",
                    ),

                    match_kpi_card(
                        "fa-solid fa-bullseye",
                        "Shots",
                        (
                            f"{home_shots} – "
                            f"{away_shots}"
                        ),
                        "Home – Away",
                        "gold",
                    ),

                    match_kpi_card(
                        "fa-solid fa-rotate",
                        "Ball recoveries",
                        (
                            f"{home_recoveries} – "
                            f"{away_recoveries}"
                        ),
                        "Home – Away",
                        "green",
                    ),

                    match_kpi_card(
                        "fa-solid fa-location-crosshairs",
                        "Final-third entries",
                        (
                            f"{home_final_third} – "
                            f"{away_final_third}"
                        ),
                        "Passes + high-confidence inferred carries",
                        "coral",
                    ),

                ], className="match-kpi-grid"),

                # =============================================
                # MATCH COMPARISON
                # =============================================

                dash_html.Section([

                    dash_html.Div([

                        dash_html.Div([

                            dash_html.Span(
                                "MATCH COMPARISON",
                                className=(
                                    "match-panel-eyebrow"
                                ),
                            ),

                            dash_html.H3(
                                "Game profile",
                                className=(
                                    "match-panel-title"
                                ),
                            ),

                            dash_html.P(
                                (
                                    "A compact comparison of "
                                    "the main attacking and "
                                    "possession indicators."
                                ),
                                className=(
                                    "match-panel-description"
                                ),
                            ),

                        ]),

                        dash_html.Div([
                            dash_html.I(
                                className=(
                                    "fa-solid "
                                    "fa-circle-info"
                                )
                            ),
                            dash_html.Span(
                                (
                                    "Bars compare the two "
                                    "teams within each metric; "
                                    "they do not imply that "
                                    "higher is always better."
                                )
                            ),
                        ], className="match-panel-hint"),

                    ], className="match-panel-header"),

                    # Team header
                    dash_html.Div([

                        dash_html.Strong(
                            home_display,
                            style={
                                "color": HCOL,
                            },
                        ),

                        dash_html.Span(
                            "Metric",
                        ),

                        dash_html.Strong(
                            away_display,
                            style={
                                "color": ACOL,
                            },
                        ),

                    ], className=(
                        "overview-comparison-header"
                    )),

                    comparison_row(
                        "Passes",
                        home_passes,
                        away_passes,
                    ),

                    comparison_row(
                        "Pass completion",
                        home_pass_completion,
                        away_pass_completion,
                        home_display_value=(
                            f"{home_pass_completion:.1f}%"
                        ),
                        away_display_value=(
                            f"{away_pass_completion:.1f}%"
                        ),
                    ),

                    comparison_row(
                        "Shots",
                        home_shots,
                        away_shots,
                    ),

                    comparison_row(
                        "Progressive passes",
                        home_progressive,
                        away_progressive,
                        description=(
                            "Completed open-play "
                            "progressive passes"
                        ),
                    ),

                    comparison_row(
                        "Final-third entries",
                        home_final_third,
                        away_final_third,
                        description=(
                            "Completed passes + high-confidence "
                            "inferred carries"
                        ),
                    ),

                    comparison_row(
                        "Crosses",
                        home_crosses,
                        away_crosses,
                    ),

                    comparison_row(
                        "Ball recoveries",
                        home_recoveries,
                        away_recoveries,
                    ),

                ], className=(
                    "match-panel "
                    "overview-comparison-panel"
                )),

                # Download target remains available.
                dcc.Download(
                    id="download-dataframe-csv"
                ),

            ], className=(
                "match-module "
                "match-overview-module"
            ))

        except Exception as e:
            tb_str = traceback.format_exc()

            return dbc.Alert(
                (
                    f"Error loading overview: {e}"
                    f"\n{tb_str}"
                ),
                color="danger",
                style={
                    "whiteSpace": "pre-wrap",
                },
            )

    elif active_tab == "formation":
            return dash_html.Div([
                match_section_header(
                    "Formation & shape",
                    "Compare the starting structures, tactical changes and territorial occupation of both teams.",
                    "fa-solid fa-people-group",
                    eyebrow="TEAM STRUCTURE",
                ),
                dbc.Tabs(
                    id="formation-primary-tabs", # Un ID per questo gruppo di tab
                    active_tab="formation_timeline", # La tab predefinita
                    children=[
                        dbc.Tab(label="Formation Timeline", tab_id="formation_timeline"),
                        dbc.Tab(label="Mean Positions", tab_id="mean_positions")
                    ],
                    className="match-analysis-tabs"
                ),
                # Un contenitore vuoto che verrà riempito dal callback sottostante
                dcc.Loading(
                    type="circle",
                    children=dash_html.Div(id="formation-tab-content")
                )
            ], className="match-module")

    elif active_tab == "passes":
        passes_content = dash_html.Div([
            match_section_header(
                "Passing analysis",
                "Explore connections, progression, territorial access and delivery patterns.",
                "fa-solid fa-arrow-right-arrow-left",
                eyebrow="IN POSSESSION",
            ),
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
                    dbc.Tab(
    label="Final Third Entries",
    tab_id="final_third_entries",
    children=[
        dcc.Loading(
            type="circle",
            children=dash_html.Div(
                id="div-final-third-content"
            )
        )
    ]
),
                    dbc.Tab(label="Pass Locations", tab_id="pass_locations", children=[
                        dash_html.Div([ # Main container for this tab's content
                            dcc.Loading(type="circle", children=dash_html.Div(id="div-pass-density-content")),
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
                className="match-analysis-tabs"
            )
        ], className="match-module")
        return passes_content

    elif active_tab == "player_analysis":
        print("--- render_match_tab_content: RENDERING NEW 'player_analysis' PRIMARY TAB STRUCTURE ---")
        return dash_html.Div([
            match_section_header(
                "Player analysis",
                "Identify the main individual contributors in possession, shooting and defending.",
                "fa-solid fa-user-group",
                eyebrow="INDIVIDUAL PERFORMANCE",
            ),

            # 1. The new PRIMARY tabs
            dbc.Tabs(
                id="player-analysis-primary-tabs",
                active_tab="pa_primary_passing", # Default to passing analysis
                children=[
                    dbc.Tab(label="Passing Analysis", tab_id="pa_primary_passing"),
                    dbc.Tab(label="Shooting Analysis", tab_id="pa_primary_shooting"),
                    dbc.Tab(label="Defending Analysis", tab_id="pa_primary_defending"),
                ],
                className="match-analysis-tabs"
            ),

            # 2. A single content area that will be filled by our new "router" callback
            dcc.Loading(type="circle", children=dash_html.Div(id="player-analysis-primary-tab-content"))
        ], className="match-module")

    elif active_tab == "buildup":
        return dash_html.Div([
             match_section_header(
                 "Buildup analysis",
                 "Review how each team progresses from the first phase and where its possessions break down.",
                 "fa-solid fa-diagram-project",
                 eyebrow="FIRST PHASE",
             ),
             # Primary tabs for Home/Away
             dbc.Tabs(
                 id="buildup-primary-tabs",
                 active_tab="buildup_home",
                 children=[
                     dbc.Tab(label="Home Team Buildups", tab_id="buildup_home"),
                     dbc.Tab(label="Away Team Buildups", tab_id="buildup_away"),
                 ],
                 className="match-analysis-tabs"
             ),
             # A single content area to be filled by the new callback
             dcc.Loading(
                 type="circle",
                 children=dash_html.Div(id="buildup-tab-content")
             )
        ], className="match-module")

    elif active_tab == "defensive-transition":
        return dash_html.Div([
            match_section_header(
                "Defending & transition",
                "Assess defensive height, pressure and the response immediately after losing possession.",
                "fa-solid fa-shield-halved",
                eyebrow="OUT OF POSSESSION",
            ),
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
                 className="match-analysis-tabs"
            ),
            dcc.Loading(
                 type="circle",
                 children=dash_html.Div(id="def-transition-tab-content")
             )
        ], className="match-module")

    elif active_tab == "offensive-transition":
        return dash_html.Div([
            match_section_header(
                "Offensive transition",
                "Explore the speed, direction and outcome of attacks launched after regaining possession.",
                "fa-solid fa-bolt",
                eyebrow="CHANGE OF POSSESSION",
            ),
            dbc.Tabs(
                id="off-transition-primary-tabs",
                active_tab="off_transitions_home",
                children=[
                     dbc.Tab(label="Home Offensive Transitions", tab_id="off_transitions_home"),
                     dbc.Tab(label="Away Offensive Transitions", tab_id="off_transitions_away"),
                 ],
                 className="match-analysis-tabs"
            ),
            dcc.Loading(
                 type="circle",
                 children=dash_html.Div(id="off-transition-tab-content")
             )
        ], className="match-module")

    elif active_tab == "set-piece":
        return dash_html.Div([
            match_section_header(
                "Restart analysis",
                (
                    "Inspect corners, free kicks, throw-ins, goal kicks "
                    "and penalties from the actual restart delivery."
                ),
                "fa-solid fa-flag",
                eyebrow="RESTARTS",
            ),
            dbc.Tabs(
                id="set-piece-primary-tabs",
                active_tab="set_piece_home",
                children=[
                     dbc.Tab(label="Home Restarts", tab_id="set_piece_home"),
                     dbc.Tab(label="Away Restarts", tab_id="set_piece_away"),
                 ],
                 className="match-analysis-tabs"
            ),
            dcc.Loading(
                 type="circle",
                 children=dash_html.Div(id="set-piece-tab-content")
             )
        ], className="match-module")

### Formaion Tab Content Callbacks
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
                timeline_items.append(
                    dbc.ListGroupItem(
                        [
                            dash_html.Span("MATCH TIMELINE", className="match-panel-eyebrow"),
                            dash_html.Strong("0' · Kick off"),
                        ],
                        className="formation-event-item formation-event-item--kickoff",
                    )
                )



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
                    timeline_items.append(
                        dbc.ListGroupItem(
                            [
                                dash_html.Span(event['time_str'], className="formation-event-time"),
                                dash_html.Div(event['description_component'], className="formation-event-description"),
                            ],
                            className="formation-event-item",
                        )
                    )

                # --- 3. COSTRUZIONE LAYOUT FINALE ---
                final_layout = dash_html.Div([
                    dash_html.Div([
                        dash_html.I(className="fa-solid fa-circle-info"),
                        dash_html.Span("Cyan highlights identify players whose formation slot changed at that moment."),
                    ], className="match-analysis-note"),
                    dash_html.Div([
                        dash_html.Section([
                            dash_html.Div([
                                dash_html.Span("HOME TEAM", className="match-panel-eyebrow"),
                                dash_html.H3(home_name, className="match-panel-title"),
                                dash_html.P("Shape snapshots throughout the match.", className="match-panel-description"),
                            ], className="match-panel-header"),
                            dash_html.Div(home_plots, className="formation-snapshot-stack"),
                        ], className="match-panel formation-team-panel"),
                        dash_html.Section([
                            dash_html.Div([
                                dash_html.Span("MATCH FLOW", className="match-panel-eyebrow"),
                                dash_html.H3("Key events", className="match-panel-title"),
                                dash_html.P("Goals, substitutions and structural changes.", className="match-panel-description"),
                            ], className="match-panel-header"),
                            dbc.ListGroup(timeline_items, flush=True, className="formation-event-list"),
                        ], className="match-panel formation-timeline-panel"),
                        dash_html.Section([
                            dash_html.Div([
                                dash_html.Span("AWAY TEAM", className="match-panel-eyebrow"),
                                dash_html.H3(away_name, className="match-panel-title"),
                                dash_html.P("Shape snapshots throughout the match.", className="match-panel-description"),
                            ], className="match-panel-header"),
                            dash_html.Div(away_plots, className="formation-snapshot-stack"),
                        ], className="match-panel formation-team-panel"),
                    ], className="formation-timeline-grid"),
                    dash_html.Section([
                        dash_html.Div([
                            dash_html.I(className="fa-regular fa-note-sticky"),
                            dash_html.Div([
                                dash_html.H3("Analyst notes", className="match-panel-title"),
                                dash_html.P("Summarise the most meaningful structural changes.", className="match-panel-description"),
                            ]),
                        ], className="match-comment-heading"),
                        dcc.Textarea(
                            id="comment-formation",
                            placeholder="Write your formation analysis...",
                            className="match-comment-input",
                        ),
                        dash_html.Div([
                            dbc.Button(
                                [dash_html.I(className="fa-solid fa-floppy-disk me-2"), "Save note"],
                                id="save-comment-formation",
                                className="match-action-button",
                                size="sm",
                            ),
                            dash_html.Div(id="save-status-formation", className="small"),
                        ], className="match-comment-actions"),
                    ], className="match-panel match-comment-panel"),
                ], className="match-tab-body")
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

            return dash_html.Div([
                dash_html.Div([
                    dash_html.I(className="fa-solid fa-circle-info"),
                    dash_html.Span("Circles represent starters; diamonds represent substitutes. The dashed line marks the average team height."),
                ], className="match-analysis-note"),
                dbc.Row([
                    dbc.Col([
                        dash_html.Section([
                            dash_html.Div([
                                dash_html.Span("HOME TEAM", className="match-panel-eyebrow"),
                                dash_html.H3(HTEAM_NAME, className="match-panel-title"),
                                dash_html.P("Average player locations with the team's territorial touch density.", className="match-panel-description"),
                            ], className="match-panel-header"),
                            dcc.Graph(
                                figure=fig_home,
                                config={'displayModeBar': False, 'responsive': True},
                                className="match-analysis-graph",
                            )
                        ], className="match-panel match-viz-panel")
                    ], lg=6),
                    dbc.Col([
                        dash_html.Section([
                            dash_html.Div([
                                dash_html.Span("AWAY TEAM", className="match-panel-eyebrow"),
                                dash_html.H3(ATEAM_NAME, className="match-panel-title"),
                                dash_html.P("Average player locations with the team's territorial touch density.", className="match-panel-description"),
                            ], className="match-panel-header"),
                            dcc.Graph(
                                figure=fig_away,
                                config={'displayModeBar': False, 'responsive': True},
                                className="match-analysis-graph",
                            )
                        ], className="match-panel match-viz-panel")
                    ], lg=6)
                ], className="g-3")
            ], className="match-tab-body")

    except Exception as e:
        return dbc.Alert(f"Error rendering formation/shape content: {traceback.format_exc()}", color="danger", style={"whiteSpace": "pre-wrap"})

    return dash_html.P("Select a sub-tab.")



# --- CALLBACKS FOR FORMATION COMMENTS ---
@app.callback(
    Output("store-comment-formation", "data"),
    Output("save-status-formation", "children"),
    Input("save-comment-formation", "n_clicks"),
    State("comment-formation", "value"),
    State("url", "pathname"),
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
    Input("url", "pathname")
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

        home_receiver_coverage = pass_processing.receiver_coverage_summary(
            successful_passes[successful_passes['team_name'] == HTEAM_NAME]
        )
        away_receiver_coverage = pass_processing.receiver_coverage_summary(
            successful_passes[successful_passes['team_name'] == ATEAM_NAME]
        )

        # **Ottieni la lista dei subentrati per ogni squadra**
        home_subs = pass_processing.get_sub_list(df_processed[df_processed['team_name'] == HTEAM_NAME])
        away_subs = pass_processing.get_sub_list(df_processed[df_processed['team_name'] == ATEAM_NAME])

        # --- Dati per Home Team ---
        home_passes_between, home_avg_locs = pass_metrics.calculate_pass_network_data(successful_passes, HTEAM_NAME)
        fig_home = pass_plotly.plot_pass_network_plotly(home_passes_between, home_avg_locs, HTEAM_NAME, HCOL, home_subs, is_away=False)

        # **MODIFICA TABELLA: Aggiungi numeri di maglia**

        # home_table_df = home_passes_between[['player1', 'player2', 'pass_count']].copy()
        # home_table_df['player1'] = home_table_df['player1'].apply(lambda name: f"#{int(home_jersey_map.get(name, '?')) if pd.notna(home_jersey_map.get(name)) else '?'} - {name}")
        # home_table_df['player2'] = home_table_df['player2'].apply(lambda name: f"#{int(home_jersey_map.get(name, '?')) if pd.notna(home_jersey_map.get(name)) else '?'} - {name}")
        # home_table = dbc.Table.from_dataframe(home_table_df.sort_values('pass_count', ascending=False).head(10), striped=True, bordered=True, hover=True, color="dark")

        # --- Dati per Away Team ---
        away_passes_between, away_avg_locs = pass_metrics.calculate_pass_network_data(successful_passes, ATEAM_NAME)
        fig_away = pass_plotly.plot_pass_network_plotly(away_passes_between, away_avg_locs, ATEAM_NAME, ACOL, away_subs, is_away=True)

        # **MODIFICA TABELLA: Aggiungi numeri di maglia**
        # away_table_df = away_passes_between[['player1', 'player2', 'pass_count']].copy()
        # away_table_df['player1'] = away_table_df['player1'].apply(lambda name: f"#{int(away_jersey_map.get(name, '?')) if pd.notna(away_jersey_map.get(name)) else '?'} - {name}")
        # away_table_df['player2'] = away_table_df['player2'].apply(lambda name: f"#{int(away_jersey_map.get(name, '?')) if pd.notna(away_jersey_map.get(name)) else '?'} - {name}")
        # away_table = dbc.Table.from_dataframe(away_table_df.sort_values('pass_count', ascending=False).head(10), striped=True, bordered=True, hover=True, color="dark")

        def build_top_connections(
            passes_between,
            jersey_map,
            team_color,
            limit=6,
        ):
            if (
                passes_between is None
                or passes_between.empty
            ):
                return dash_html.Div(
                    "No reliable connections available.",
                    className="pass-network-empty",
                )

            top_connections = (
                passes_between
                .sort_values(
                    'pass_count',
                    ascending=False,
                )
                .head(limit)
                .copy()
            )

            max_count = max(
                int(
                    top_connections[
                        'pass_count'
                    ].max()
                ),
                1,
            )

            def player_label(
                player_name,
            ):
                jersey_raw = jersey_map.get(
                    player_name
                )

                try:
                    jersey = str(
                        int(float(jersey_raw))
                    )
                except (
                    ValueError,
                    TypeError,
                ):
                    jersey = '?'

                return (
                    f"#{jersey} · {player_name}"
                )

            rows = []

            for rank, (_, row) in enumerate(
                top_connections.iterrows(),
                start=1,
            ):
                count = int(
                    row['pass_count']
                )

                width = (
                    count / max_count * 100
                )

                rows.append(
                    dash_html.Div([

                        dash_html.Span(
                            str(rank),
                            className=(
                                "pass-connection-rank"
                            ),
                        ),

                        dash_html.Div([

                            dash_html.Div(
                                [
                                    dash_html.Span(
                                        player_label(
                                            row['player1']
                                        )
                                    ),

                                    dash_html.I(
                                        className=(
                                            "fa-solid "
                                            "fa-arrow-right-arrow-left"
                                        )
                                    ),

                                    dash_html.Span(
                                        player_label(
                                            row['player2']
                                        )
                                    ),
                                ],
                                className=(
                                    "pass-connection-pair"
                                ),
                            ),

                            dash_html.Div(
                                dash_html.Span(
                                    style={
                                        "width":
                                            f"{width:.1f}%",
                                        "backgroundColor":
                                            team_color,
                                    }
                                ),
                                className=(
                                    "pass-connection-track"
                                ),
                            ),

                        ], className=(
                            "pass-connection-main"
                        )),

                        dash_html.Strong(
                            str(count),
                            className=(
                                "pass-connection-count"
                            ),
                        ),

                    ], className=(
                        "pass-connection-row"
                    ))
                )

            return dash_html.Div(
                rows,
                className="pass-connection-list",
            )

        home_jersey_map = home_avg_locs.set_index('playerName')['jersey_number'].to_dict()
        home_connections = build_top_connections(
            home_passes_between,
            home_jersey_map,
            HCOL,
        )

        away_jersey_map = away_avg_locs.set_index('playerName')['jersey_number'].to_dict()
        away_connections = build_top_connections(
            away_passes_between,
            away_jersey_map,
            ACOL,
        )


        # Layout a due colonne per mostrare i grafici affiancati
        pass_network_coverage_panel = render_data_coverage_panel(
            [
                _receiver_coverage_item(
                    passes_df[
                        passes_df["team_name"] == HTEAM_NAME
                    ],
                    label=f"{HTEAM_NAME} receiver coverage",
                ),
                _receiver_coverage_item(
                    passes_df[
                        passes_df["team_name"] == ATEAM_NAME
                    ],
                    label=f"{ATEAM_NAME} receiver coverage",
                ),
                _coordinate_coverage_item(
                    passes_df,
                    label="Pass coordinates",
                    columns=("x", "y", "end_x", "end_y"),
                ),
                _outcome_coverage_item(
                    passes_df,
                    label="Pass outcomes",
                ),
            ],
            note=(
                "Network links use only reliable inferred receivers. "
                "Unresolved successful passes stay outside the links."
            ),
        )

        return dash_html.Div([

            pass_network_coverage_panel,


            # ---------------------------------------------------------
            # NETWORK PANELS
            # ---------------------------------------------------------
            dash_html.Div([

                # HOME
                dash_html.Section([

                    dcc.Graph(
                        figure=fig_home,
                        config={
                            'displayModeBar': False,
                            'responsive': True,
                        },
                        className=(
                            "pass-network-graph"
                        ),
                    ),

                    dash_html.Div([

                        dash_html.Div([
                            dash_html.Span(
                                "TOP CONNECTIONS",
                                className=(
                                    "match-panel-eyebrow"
                                ),
                            ),

                            dash_html.H4(
                                HTEAM_NAME,
                                className=(
                                    "pass-network-connections-title"
                                ),
                            ),

                            dash_html.P(
                                (
                                    "Highest-volume player pairs. "
                                    "Passes in both directions are "
                                    "combined."
                                ),
                                className=(
                                    "match-panel-description"
                                ),
                            ),

                        ]),

                        home_connections,

                    ], className=(
                        "pass-network-connections"
                    )),

                ], className=(
                    "match-panel "
                    "pass-network-team-panel"
                )),


                # AWAY
                dash_html.Section([

                    dcc.Graph(
                        figure=fig_away,
                        config={
                            'displayModeBar': False,
                            'responsive': True,
                        },
                        className=(
                            "pass-network-graph"
                        ),
                    ),

                    dash_html.Div([

                        dash_html.Div([
                            dash_html.Span(
                                "TOP CONNECTIONS",
                                className=(
                                    "match-panel-eyebrow"
                                ),
                            ),

                            dash_html.H4(
                                ATEAM_NAME,
                                className=(
                                    "pass-network-connections-title"
                                ),
                            ),

                            dash_html.P(
                                (
                                    "Highest-volume player pairs. "
                                    "Passes in both directions are "
                                    "combined."
                                ),
                                className=(
                                    "match-panel-description"
                                ),
                            ),

                        ]),

                        away_connections,

                    ], className=(
                        "pass-network-connections"
                    )),

                ], className=(
                    "match-panel "
                    "pass-network-team-panel"
                )),

            ], className="pass-network-grid"),

        ], className="match-tab-body")

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
    State("url", "pathname"), # To get match_id for unique storage key
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
    Input("url", "pathname")      # Trigger when page/match changes
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
        prog_pass_exclusions = None  # Standard open-play exclusions are applied centrally.
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

        # Keep attempted and completed progressive passes separate: volume alone
        # must not be presented as passing quality.
        all_passes = pass_processing.get_passes_df(df_processed)
        prog_passes = all_passes[
            all_passes['is_progressive_attempt'].fillna(False).astype(bool)
        ].copy()

        if prog_passes.empty:
            return dbc.Alert(
                "No open-play progressive pass attempts found in the match.",
                color="warning",
            )

        def create_prog_pass_layout_for_team(team_name, team_color, is_away):
            team_passes = prog_passes[prog_passes['team_name'] == team_name]
            summary = pass_metrics.progressive_pass_summary(team_passes)
            fig = pass_plotly.plot_progressive_passes_plotly(
                team_passes, team_name, team_color, is_away
            )
            graph_component = dcc.Graph(
                figure=fig,
                config={'displayModeBar': False, 'responsive': True},
                className='progressive-map-graph',
            )

            def metric_card(
                label,
                value,
                detail,
                tooltip=None,
            ):
                if tooltip:
                    tooltip_id = (
                        "progressive-kpi-info-"
                        f"{uuid.uuid4().hex}"
                    )

                    label_component = dash_html.Div([

                        dash_html.Span(
                            label,
                            className='progressive-kpi-label',
                        ),

                        dash_html.I(
                            id=tooltip_id,
                            className=(
                                "fa-regular "
                                "fa-circle-question "
                                "metric-definition-icon"
                            ),
                        ),

                        dbc.Tooltip(
                            tooltip,
                            target=tooltip_id,
                            placement="top",
                            delay={
                                "show": 250,
                                "hide": 80,
                            },
                        ),

                    ], className="metric-label-with-info")

                else:
                    label_component = dash_html.Span(
                        label,
                        className='progressive-kpi-label',
                    )

                return dash_html.Div([

                    label_component,

                    dash_html.Strong(
                        value,
                        className='progressive-kpi-value',
                    ),

                    dash_html.Small(
                        detail,
                        className='progressive-kpi-detail',
                    ),

                ], className='progressive-kpi-card')

            main_channel_count = summary['channel_counts'].get(
                summary['main_channel'], 0
            )

            kpis = dash_html.Div([

                metric_card(
                    'Completed / attempted',
                    (
                        f"{summary['successful']} "
                        f"/ {summary['attempted']}"
                    ),
                    'Open-play progressive passes',
                    (
                        "Attempted is the total number of open-play "
                        "passes that satisfy the progressive-pass "
                        "criterion. Completed counts only those with "
                        "a successful outcome."
                    ),
                ),

                metric_card(
                    'Completion',
                    f"{summary['completion_pct']:.1f}%",
                    'Completed ÷ attempted',
                    (
                        "Progressive-pass completion rate. "
                        "The denominator is all qualifying "
                        "progressive-pass attempts."
                    ),
                ),

                metric_card(
                    'Progression gained',
                    f"{summary['total_progression_m']:.0f} m",
                    'Completed progressive passes only',
                    (
                        "Only completed progressive passes contribute "
                        "to this total. Failed attempts contribute "
                        "zero metres."
                    ),
                ),

                metric_card(
                    'Main origin channel',
                    summary['main_channel'],
                    (
                        f"{main_channel_count} of "
                        f"{summary['attempted']} attempts"
                    ),
                    (
                        "The pitch is divided into Left, Central and "
                        "Right origin channels using the starting "
                        "location of each progressive-pass attempt. "
                        "The denominator is all progressive attempts."
                    ),
                ),

            ], className='progressive-kpi-grid')

            channel_total = max(summary['attempted'], 1)
            channel_profile = dash_html.Div([
                dash_html.Div([
                    dash_html.Span(channel),
                    dash_html.Div(
                        dash_html.Span(style={
                            'width': (
                                f"{summary['channel_counts'][channel] / channel_total * 100:.1f}%"
                            )
                        }),
                        className='progressive-channel-track',
                    ),
                    dash_html.Strong(str(summary['channel_counts'][channel])),
                ], className='progressive-channel-row')
                for channel in ('Left', 'Central', 'Right')
            ], className='progressive-channel-profile')

            top_passers = pass_metrics.progressive_pass_player_summary(team_passes)
            if top_passers.empty:
                table_content = dbc.Alert('No player data', color='secondary')
            else:
                player_jersey_map = (
                    team_passes.drop_duplicates('playerName')
                    .set_index('playerName')['Mapped Jersey Number']
                )

                def format_player_name_with_jersey(player_name):
                    jersey_raw = player_jersey_map.get(player_name)
                    try:
                        jersey = str(int(jersey_raw))
                    except (ValueError, TypeError):
                        jersey = '?'
                    return f"#{jersey} · {player_name}"

                top_passers['Player'] = top_passers['Player'].apply(
                    format_player_name_with_jersey
                )
                top_passers['Completed'] = (
                    top_passers['Successful'].astype(str)
                    + ' / '
                    + top_passers['Attempted'].astype(str)
                )
                top_passers['Rate'] = top_passers['Completion %'].astype(str) + '%'
                top_passers['Gain'] = top_passers['Progression m'].astype(str) + ' m'
                display_table = top_passers[['Player', 'Completed', 'Rate', 'Gain']]
                table_content = dbc.Table.from_dataframe(
                    display_table,
                    striped=False,
                    bordered=False,
                    hover=True,
                    responsive=True,
                    className='progressive-player-table',
                )

            sidebar = dash_html.Div([
                kpis,
                dash_html.Div([
                    dash_html.H6('Origin-channel profile'),
                    channel_profile,
                ], className='progressive-sidebar-section'),
                dash_html.Div([
                    dash_html.H6('Top progressive passers'),
                    table_content,
                ], className='progressive-sidebar-section'),
            ], className='progressive-sidebar')

            return dash_html.Section([
                dash_html.Div([
                    dash_html.Div([
                        dash_html.Span(
                            'HOME TEAM' if not is_away else 'AWAY TEAM',
                            className='match-panel-eyebrow',
                        ),
                        dash_html.H4(team_name,className="match-team-name"),
                    ]),
                    dash_html.Div([

                        dash_html.Span(
                            f"n = {summary['attempted']} attempts",
                            className="progressive-sample-size",
                        ),

                        dash_html.Span(
                            "All teams attack left to right",
                            className='match-panel-hint',
                        ),

                    ], className="progressive-panel-meta"),
                ], className='match-panel-header'),
                dbc.Row([
                    dbc.Col(graph_component, lg=8),
                    dbc.Col(sidebar, lg=4),
                ], className='g-0'),
            ], className='match-panel progressive-team-panel')

        home_layout = create_prog_pass_layout_for_team(HTEAM_NAME, HCOL, is_away=False)
        away_layout = create_prog_pass_layout_for_team(ATEAM_NAME, ACOL, is_away=True)

        return dash_html.Div([
            dash_html.Div([
                dash_html.I(className='fas fa-info-circle'),
                dash_html.Span(
                    'Open-play only. A pass is progressive when it reduces the '
                    'distance to the centre of goal by at least 30 m in the own '
                    'half, 15 m across halfway or 10 m in the opposition half. '
                    'Crosses and restarts are excluded.'
                ),
            ], className='match-analysis-note progressive-definition-note'),
            home_layout,
            away_layout
        ], className='progressive-analysis')

    except Exception as e:
        tb_str = traceback.format_exc()
        return dbc.Alert(f"Error generating progressive passes plot: {e}\n{tb_str}", color="danger", style={"whiteSpace": "pre-wrap"})

# Example for progressive passes save:
@app.callback(
    Output("store-comment-progressive-passes", "data"),
    Output("save-status-progressive-passes", "children"),
    Input("save-comment-progressive-passes", "n_clicks"),
    State("comment-progressive-passes", "value"),
    State("url", "pathname"),
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
    Input("url", "pathname")
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

@app.callback(
    Output("div-final-third-content", "children"),
    Input("store-df-match", "data"),
    Input("passes-nested-tabs", "active_tab")
)
def show_final_third_content_callback(stored_data_json, active_nested_tab):
    if active_nested_tab != "final_third_entries" or not stored_data_json:
        return no_update

    try:
        df_processed = pd.read_json(
            io.StringIO(stored_data_json['df']),
            orient='split',
        )
        match_info = json.loads(stored_data_json['match_info'])

        HTEAM_NAME = match_info.get('hteamName', 'Home')
        ATEAM_NAME = match_info.get('ateamName', 'Away')

        # -----------------------------------------------------
        # PASSES + CARRIES
        # -----------------------------------------------------
        passes_df = pass_processing.get_passes_df(df_processed.copy())

        successful_passes = passes_df[
            passes_df['outcome'] == 'Successful'
        ].copy()

        carries_df = pass_processing.infer_carries(
            df_processed.copy()
        )

        if successful_passes.empty and carries_df.empty:
            return dbc.Alert(
                "No pass or carry data available for Final Third analysis.",
                color="warning",
            )

        # -----------------------------------------------------
        # TEAM PANEL
        # -----------------------------------------------------
        def create_final_third_layout_for_team(
            team_name,
            team_color,
            is_away,
        ):
            team_passes = successful_passes[
                successful_passes['team_name'] == team_name
            ].copy()

            if (
                carries_df is not None
                and not carries_df.empty
                and 'team_name' in carries_df.columns
            ):
                team_carries = carries_df[
                    carries_df['team_name'] == team_name
                ].copy()
            else:
                team_carries = pd.DataFrame()

            entries_df, stats = (
                pass_metrics.analyze_final_third_entries(
                    team_passes,
                    team_carries,
                )
            )

            # -------------------------------------------------
            # PITCH
            # -------------------------------------------------
            fig = pass_plotly.plot_final_third_entries_plotly(
                entries_df,
                stats,
                team_name,
                team_color,
                is_away=is_away,
            )

            graph_component = dcc.Graph(
                figure=fig,
                config={
                    'displayModeBar': False,
                    'responsive': True,
                },
                className='progressive-map-graph',
            )

            # -------------------------------------------------
            # KPI
            # -------------------------------------------------
            def metric_card(
                label,
                value,
                detail,
                tooltip=None,
            ):
                if tooltip:
                    tooltip_id = (
                        "final-third-kpi-info-"
                        f"{uuid.uuid4().hex}"
                    )

                    label_component = dash_html.Div([

                        dash_html.Span(
                            label,
                            className='progressive-kpi-label',
                        ),

                        dash_html.I(
                            id=tooltip_id,
                            className=(
                                "fa-regular "
                                "fa-circle-question "
                                "metric-definition-icon"
                            ),
                        ),

                        dbc.Tooltip(
                            tooltip,
                            target=tooltip_id,
                            placement="top",
                            delay={
                                "show": 250,
                                "hide": 80,
                            },
                        ),

                    ], className="metric-label-with-info")

                else:
                    label_component = dash_html.Span(
                        label,
                        className='progressive-kpi-label',
                    )

                return dash_html.Div([

                    label_component,

                    dash_html.Strong(
                        value,
                        className='progressive-kpi-value',
                    ),

                    dash_html.Small(
                        detail,
                        className='progressive-kpi-detail',
                    ),

                ], className='progressive-kpi-card')

            total_entries = stats.get('total_final_third', 0)
            pass_count = stats.get('pass_entries', 0)
            carry_count = stats.get('carry_entries', 0)
            carry_candidate_count = stats.get(
                'carry_entry_candidates',
                0,
            )
            excluded_carry_count = stats.get(
                'carry_entries_excluded_total',
                0,
            )

            channel_counts = {
                'Left': stats.get('channel_left', 0),
                'Central': stats.get('channel_central', 0),
                'Right': stats.get('channel_right', 0),
            }

            if total_entries:
                main_channel = max(
                    channel_counts,
                    key=channel_counts.get,
                )
                main_channel_count = channel_counts[main_channel]
            else:
                main_channel = '—'
                main_channel_count = 0

            inside_entries = (
                stats.get('zone14', 0)
                + stats.get('hs_left', 0)
                + stats.get('hs_right', 0)
            )

            inside_pct = (
                inside_entries / total_entries * 100
                if total_entries
                else 0.0
            )

            kpis = dash_html.Div([

                metric_card(
                    'Total entries',
                    str(total_entries),
                    (
                        'Completed passes + high-confidence '
                        'inferred carries'
                    ),
                    (
                        "Passes use the exact final-third boundary. "
                        "Inferred carries also require a robust "
                        "crossing and high confidence."
                    ),
                ),

                metric_card(
                    'Pass / inferred carry',
                    f"{pass_count} / {carry_count}",
                    (
                        f"{excluded_carry_count} of "
                        f"{carry_candidate_count} inferred carry "
                        "candidates excluded"
                        if carry_candidate_count
                        else "No inferred carry candidates"
                    ),
                    (
                        "Only high-confidence inferred carries are "
                        "included. They must start at least 1 Opta "
                        "point before the boundary, finish at least "
                        "1 point beyond it and advance at least 3 "
                        "points longitudinally."
                    ),
                ),

                metric_card(
                    'Main entry channel',
                    main_channel,
                    (
                        f"{main_channel_count} of "
                        f"{total_entries} entries"
                    ),
                    (
                        "Left, Central and Right are based on the "
                        "entry destination y-coordinate. "
                        "The denominator is all final-third entries."
                    ),
                ),

                metric_card(
                    'Inside channels',
                    str(inside_entries),
                    (
                        f"{inside_pct:.1f}% of "
                        f"{total_entries} entries"
                    ),
                    (
                        "Counts entries ending in Zone 14, "
                        "the Left Half-Space or the Right Half-Space. "
                        "The denominator is all final-third entries."
                    ),
                ),

            ], className='progressive-kpi-grid')

            # -------------------------------------------------
            # PROFILE BARS
            # -------------------------------------------------
            profile_total = max(total_entries, 1)

            def build_profile(counts):
                return dash_html.Div([
                    dash_html.Div([
                        dash_html.Span(label),

                        dash_html.Div(
                            dash_html.Span(
                                style={
                                    'width': (
                                        f"{value / profile_total * 100:.1f}%"
                                    )
                                }
                            ),
                            className='progressive-channel-track',
                        ),

                        dash_html.Strong(str(value)),

                    ], className='progressive-channel-row')

                    for label, value in counts.items()

                ], className='progressive-channel-profile')

            channel_profile = build_profile(
                channel_counts
            )

            destination_counts = {
                'Zone 14': stats.get('zone14', 0),
                'Left HS': stats.get('hs_left', 0),
                'Right HS': stats.get('hs_right', 0),
                'Other': stats.get('other', 0),
            }

            destination_profile = build_profile(
                destination_counts
            )

            # -------------------------------------------------
            # TOP CONTRIBUTORS
            # -------------------------------------------------
            if entries_df is None or entries_df.empty:
                table_content = dbc.Alert(
                    'No player data',
                    color='secondary',
                )

            else:
                pass_entries = entries_df[
                    entries_df['entry_type'] == 'Pass'
                ]

                carry_entries = entries_df[
                    entries_df['entry_type'] == 'Carry'
                ]

                pass_contributors = (
                    pass_entries['playerName']
                    .dropna()
                    .value_counts()
                    .rename('Pass')
                )

                carry_contributors = (
                    carry_entries['playerName']
                    .dropna()
                    .value_counts()
                    .rename('Carry')
                )

                contributors = pd.concat(
                    [
                        pass_contributors,
                        carry_contributors,
                    ],
                    axis=1,
                ).fillna(0)

                if contributors.empty:
                    table_content = dbc.Alert(
                        'No player data',
                        color='secondary',
                    )

                else:
                    contributors['Pass'] = (
                        contributors['Pass'].astype(int)
                    )
                    contributors['Carry'] = (
                        contributors['Carry'].astype(int)
                    )
                    contributors['Total'] = (
                        contributors['Pass']
                        + contributors['Carry']
                    )

                    contributors = (
                        contributors
                        .sort_values(
                            ['Total', 'Pass'],
                            ascending=False,
                        )
                        .head(5)
                        .reset_index()
                    )

                    if 'playerName' in contributors.columns:
                        contributors = contributors.rename(
                            columns={'playerName': 'Player'}
                        )
                    elif 'index' in contributors.columns:
                        contributors = contributors.rename(
                            columns={'index': 'Player'}
                        )

                    # Jersey numbers
                    jersey_source = (
                        df_processed[
                            df_processed['team_name'] == team_name
                        ]
                        .dropna(subset=['playerName'])
                        .drop_duplicates(
                            'playerName',
                            keep='last',
                        )
                    )

                    if (
                        not jersey_source.empty
                        and 'Mapped Jersey Number'
                        in jersey_source.columns
                    ):
                        jersey_map = jersey_source.set_index(
                            'playerName'
                        )['Mapped Jersey Number']
                    else:
                        jersey_map = pd.Series(dtype='object')

                    def format_player_name(player_name):
                        jersey_raw = jersey_map.get(player_name)

                        try:
                            jersey = str(
                                int(float(jersey_raw))
                            )
                        except (ValueError, TypeError):
                            jersey = '?'

                        return f"#{jersey} · {player_name}"

                    contributors['Player'] = (
                        contributors['Player']
                        .apply(format_player_name)
                    )

                    display_table = contributors[
                        [
                            'Player',
                            'Pass',
                            'Carry',
                            'Total',
                        ]
                    ].rename(columns={
                        'Carry': 'Inferred carry',
                    })

                    table_content = dbc.Table.from_dataframe(
                        display_table,
                        striped=False,
                        bordered=False,
                        hover=True,
                        responsive=True,
                        className='progressive-player-table',
                    )

            # -------------------------------------------------
            # SIDEBAR
            # -------------------------------------------------
            sidebar = dash_html.Div([
                kpis,

                dash_html.Div([
                    dash_html.H6(
                        'Entry-channel profile'
                    ),
                    channel_profile,
                ], className='progressive-sidebar-section'),

                dash_html.Div([
                    dash_html.H6(
                        'Destination profile'
                    ),
                    destination_profile,
                ], className='progressive-sidebar-section'),

                dash_html.Div([
                    dash_html.H6(
                        'Top entry contributors'
                    ),
                    table_content,
                ], className='progressive-sidebar-section'),

            ], className='progressive-sidebar')

            # -------------------------------------------------
            # PANEL — SAME STRUCTURE AS PROGRESSIVE PASSES
            # -------------------------------------------------
            return dash_html.Section([
                dash_html.Div([
                    dash_html.Div([
                        dash_html.Span(
                            'HOME TEAM' if not is_away else 'AWAY TEAM',
                            className='match-panel-eyebrow',
                        ),
                        dash_html.H4(team_name,className="match-team-name"),
                    ]),

                    dash_html.Div([

                        dash_html.Span(
                            f"n = {total_entries} entries",
                            className="progressive-sample-size",
                        ),

                        dash_html.Span(
                            "All teams attack left to right",
                            className='match-panel-hint',
                        ),

                    ], className="progressive-panel-meta"),

                ], className='match-panel-header'),

                dbc.Row([
                    dbc.Col(
                        graph_component,
                        lg=8,
                    ),
                    dbc.Col(
                        sidebar,
                        lg=4,
                    ),
                ], className='g-0'),

            ], className='match-panel progressive-team-panel')

        # -----------------------------------------------------
        # HOME / AWAY
        # -----------------------------------------------------
        home_layout = create_final_third_layout_for_team(
            HTEAM_NAME,
            HCOL,
            is_away=False,
        )

        away_layout = create_final_third_layout_for_team(
            ATEAM_NAME,
            ACOL,
            is_away=True,
        )

        def _final_third_stats_for_coverage(team_name):
            coverage_passes = successful_passes[
                successful_passes["team_name"] == team_name
            ].copy()

            if (
                carries_df is not None
                and not carries_df.empty
                and "team_name" in carries_df.columns
            ):
                coverage_carries = carries_df[
                    carries_df["team_name"] == team_name
                ].copy()
            else:
                coverage_carries = pd.DataFrame()

            _, coverage_stats = pass_metrics.analyze_final_third_entries(
                coverage_passes,
                coverage_carries,
            )
            return coverage_stats

        home_final_third_coverage_stats = (
            _final_third_stats_for_coverage(HTEAM_NAME)
        )
        away_final_third_coverage_stats = (
            _final_third_stats_for_coverage(ATEAM_NAME)
        )

        final_third_coverage_panel = render_data_coverage_panel(
            [
                _carry_coverage_item(
                    home_final_third_coverage_stats,
                    label=f"{HTEAM_NAME} carry candidates",
                ),
                _carry_coverage_item(
                    away_final_third_coverage_stats,
                    label=f"{ATEAM_NAME} carry candidates",
                ),
                _coordinate_coverage_item(
                    passes_df,
                    label="Pass coordinates",
                    columns=("x", "y", "end_x", "end_y"),
                ),
                _outcome_coverage_item(
                    passes_df,
                    label="Pass outcomes",
                ),
            ],
            note=(
                "Carry inclusion is informative by default: a conservative "
                "inference can legitimately exclude ambiguous candidates."
            ),
        )

        # -----------------------------------------------------
        # DEFINITION
        # -----------------------------------------------------
        definition_note = dash_html.Div([
            dash_html.I(
                className='fas fa-info-circle'
            ),
            dash_html.Span(
                (
                    'A pass entry is recorded when the ball moves from '
                    'x < 66.67 to x ≥ 66.67. Carry entries are inferred '
                    'from consecutive events rather than observed '
                    'directly: the primary KPI includes only '
                    'high-confidence candidates that start at least 1 '
                    'Opta point before the boundary, finish at least 1 '
                    'point beyond it and advance at least 3 points '
                    'longitudinally. Rejected candidates are disclosed '
                    'in the KPI card. Entry channels are classified from '
                    'the destination y-coordinate. Zone 14 and the '
                    'half-spaces describe the destination of the entry, '
                    'not the definition of the metric.'
                )
            ),
        ], className=(
            'match-analysis-note '
            'progressive-definition-note'
        ))

        # -----------------------------------------------------
        # COMMENTS
        # -----------------------------------------------------
        comment_panel = dash_html.Section([
            dash_html.Div([
                dash_html.I(
                    className='fa-regular fa-note-sticky'
                ),
                dash_html.Div([
                    dash_html.H3(
                        'Analyst notes',
                        className='match-panel-title',
                    ),
                    dash_html.P(
                        'Summarise the most meaningful final-third access patterns.',
                        className='match-panel-description',
                    ),
                ]),
            ], className='match-comment-heading'),

            dcc.Textarea(
                id='comment-final-third',
                placeholder='Write your Final Third Entries analysis...',
                className='match-comment-input',
            ),

            dash_html.Div([
                dbc.Button(
                    [
                        dash_html.I(
                            className='fa-solid fa-floppy-disk me-2'
                        ),
                        'Save note',
                    ],
                    id='save-comment-final-third',
                    className='match-action-button',
                    size='sm',
                ),
                dash_html.Div(
                    id='save-status-final-third',
                    className='small',
                ),
            ], className='match-comment-actions'),

        ], className='match-panel match-comment-panel')

        return dash_html.Div([
            definition_note,
            final_third_coverage_panel,
            home_layout,
            away_layout,
            comment_panel,
        ], className='progressive-analysis')

    except Exception as e:
        tb_str = traceback.format_exc()

        return dbc.Alert(
            f"Error generating final third plot: {e}\n{tb_str}",
            color="danger",
            style={"whiteSpace": "pre-wrap"},
        )

# --- COMMENT CALLBACKS FOR FINAL THIRD ENTRIES ---
@app.callback(
    Output("store-comment-final-third", "data"),
    Output("save-status-final-third", "children"),
    Input("save-comment-final-third", "n_clicks"),
    State("comment-final-third", "value"),
    State("url", "pathname"),
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
    Input("url", "pathname")
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

        def team_pass_location_panel(
            team_name,
            passes,
            density_fig,
            heatmap_fig,
            is_away,
        ):
            return dash_html.Section([

                # -----------------------------------------
                # TEAM HEADER
                # -----------------------------------------
                dash_html.Div([

                    dash_html.Div([

                        dash_html.Span(
                            (
                                "AWAY TEAM"
                                if is_away
                                else "HOME TEAM"
                            ),
                            className=(
                                "match-panel-eyebrow"
                            ),
                        ),

                        dash_html.H4(
                            team_name,
                            className="match-team-name",
                        ),

                    ]),

                    dash_html.Div([

                        dash_html.Span(
                            (
                                f"n = {len(passes)} "
                                "pass attempts"
                            ),
                            className=(
                                "progressive-sample-size"
                            ),
                        ),

                        dash_html.Span(
                            (
                                "All teams attack "
                                "left to right"
                            ),
                            className=(
                                "match-panel-hint"
                            ),
                        ),

                    ], className=(
                        "progressive-panel-meta"
                    )),

                ], className="match-panel-header"),


                # -----------------------------------------
                # TWO-PLOT WORKSPACE
                # -----------------------------------------
                dash_html.Div([

                    dash_html.Div(
                        dcc.Graph(
                            figure=density_fig,
                            config={
                                'displayModeBar':
                                    False,
                                'responsive':
                                    True,
                            },
                            className=(
                                "pass-location-graph"
                            ),
                        ),
                        className=(
                            "pass-location-plot-shell"
                        ),
                    ),

                    dash_html.Div(
                        dcc.Graph(
                            figure=heatmap_fig,
                            config={
                                'displayModeBar':
                                    False,
                                'responsive':
                                    True,
                            },
                            className=(
                                "pass-location-graph"
                            ),
                        ),
                        className=(
                            "pass-location-plot-shell"
                        ),
                    ),

                ], className=(
                    "pass-location-plot-grid"
                )),

            ], className=(
                "match-panel "
                "pass-location-team-panel"
            ))


        home_panel = team_pass_location_panel(
            HTEAM_NAME,
            home_passes,
            fig_home_density,
            fig_home_heatmap,
            is_away=False,
        )

        away_panel = team_pass_location_panel(
            ATEAM_NAME,
            away_passes,
            fig_away_density,
            fig_away_heatmap,
            is_away=True,
        )

        return dash_html.Div(
            [
                home_panel,
                away_panel,
            ],
            className="pass-location-analysis",
        )

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
              State("comment-pass-density", "value"), State("url", "pathname"), State("store-comment-pass-density", "data"),
              prevent_initial_call=True)
def save_pass_density_comment(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pass_density"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-pass-density", "value"),
              Input("store-comment-pass-density", "data"), Input("url", "pathname"))
def load_pass_density_comment(data, pn):
    key = get_comment_key(pn, "pass_density")
    if not key or data is None: return ""
    return data.get(key, "")

# --- COMMENT CALLBACKS FOR PASS HEATMAP ---
@app.callback(Output("store-comment-pass-heatmap", "data"), Output("save-status-pass-heatmap", "children"),
              Input("save-comment-pass-heatmap", "n_clicks"),
              State("comment-pass-heatmap", "value"), State("url", "pathname"), State("store-comment-pass-heatmap", "data"),
              prevent_initial_call=True)
def save_pass_heatmap_comment(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pass_heatmap"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-pass-heatmap", "value"),
              Input("store-comment-pass-heatmap", "data"), Input("url", "pathname"))
def load_pass_heatmap_comment(data, pn):
    key = get_comment_key(pn, "pass_heatmap")
    if not key or data is None: return ""
    return data.get(key, "")


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
            top_passer_name = (
                team_player_stats[
                'Offensive Pass Contributions'
                ].idxmax()
            )

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

            ranking_component = (
                player_plots.create_offensive_pass_contributions_table(
                    player_stats_df,
                    df_processed,
                    home_team_name,
                    hcol=HCOL,
                    acol=ACOL,
                    num_players=10,
                    )
                )

            return dash_html.Div([

    dash_html.Div([
        dash_html.I(
            className='fa-solid fa-circle-info'
        ),
        dash_html.Span(
            'Players are ranked by unique qualifying pass events. '
            'Progressive passes, completed passes into the box, '
            'key passes and assists all contribute; overlaps count '
            'once. Assists are included in UNIQUE even though they '
            'are not shown as a separate column.'
        ),
    ], className='match-analysis-note'),

    dash_html.Section([

        dash_html.Div([
            dash_html.Div([
                dash_html.Span(
                    'PASSING PROFILE',
                    className='match-panel-eyebrow',
                ),
                dash_html.H3(
                    'Offensive passing contributions',
                    className='match-panel-title',
                ),
                dash_html.P(
                    'Compare the players most involved in '
                    'meaningful ball progression and chance creation.',
                    className='match-panel-description',
                ),
            ]),

            dash_html.Span(
                'Ranked by unique qualifying passes',
                className='match-panel-hint',
            ),

        ], className='match-panel-header'),

        dash_html.Div(
            ranking_component,
            className='passing-ranking-wrapper',
        ),

    ], className='match-panel'),

    dash_html.Section([

        dash_html.Div([
            dash_html.I(
                className='fa-regular fa-note-sticky'
            ),

            dash_html.Div([
                dash_html.H3(
                    'Analyst notes',
                    className='match-panel-title',
                ),
                dash_html.P(
                    'Summarise the main individual '
                    'passing contributions.',
                    className='match-panel-description',
                ),
            ]),

        ], className='match-comment-heading'),

        dcc.Textarea(
            id='comment-top-passers-bar',
            placeholder=(
                'Write your player passing analysis...'
            ),
            className='match-comment-input',
        ),

        dash_html.Div([
            dbc.Button(
                [
                    dash_html.I(
                        className=(
                            'fa-solid '
                            'fa-floppy-disk me-2'
                        )
                    ),
                    'Save note',
                ],
                id='save-comment-top-passers-bar',
                className='match-action-button',
                size='sm',
            ),

            dash_html.Div(
                id='save-status-top-passers-bar',
                className='small',
            ),

        ], className='match-comment-actions'),

    ], className='match-panel match-comment-panel'),

], className='match-tab-body')
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
            away_team_name = match_info.get('ateamName', '')

            shot_sequence_passes = pass_processing.get_passes_df(
                df_processed.copy()
            )

            shot_sequence_coverage_panel = render_data_coverage_panel(
                [
                    _receiver_coverage_item(
                        shot_sequence_passes[
                            shot_sequence_passes["team_name"]
                            == home_team_name
                        ],
                        label=f"{home_team_name} receiver coverage",
                    ),
                    _receiver_coverage_item(
                        shot_sequence_passes[
                            shot_sequence_passes["team_name"]
                            == away_team_name
                        ],
                        label=f"{away_team_name} receiver coverage",
                    ),
                    _coordinate_coverage_item(
                        shot_sequence_passes,
                        label="Pass coordinates",
                        columns=("x", "y", "end_x", "end_y"),
                    ),
                    _outcome_coverage_item(
                        shot_sequence_passes,
                        label="Pass outcomes",
                    ),
                ],
                note=(
                    "Pre-Assist attribution uses only reliable inferred "
                    "receivers; ambiguous receiver links are not promoted."
                ),
            )



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
                shot_sequence_coverage_panel,
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

    return dash_html.P(f"Content for {active_tab} not found.")

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
    Input("url", "search")
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
            prog_pass_exclusions = None

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
              State("comment-top-passers-bar", "value"), State("url", "pathname"), State("store-comment-top-passers-bar", "data"),
              prevent_initial_call=True)
def save_comment_top_passers(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_top_passers_stats"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.",color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-top-passers-bar", "value"), Input("store-comment-top-passers-bar", "data"), Input("url", "pathname"))
def load_comment_top_passers(data, pn):
    key = get_comment_key(pn, "pa_top_passers_stats")
    if not key or data is None: return ""
    return data.get(key, "")

@app.callback(Output("store-comment-home-top-passer-map", "data"), Output("save-status-home-top-passer-map", "children"),
              Input("save-comment-home-top-passer-map", "n_clicks"),
              State("comment-home-top-passer-map", "value"), State("url", "pathname"), State("store-comment-home-top-passer-map", "data"),
              prevent_initial_call=True)
def save_comment_home_pass_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_home_passer_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.",color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-home-top-passer-map", "value"), Input("store-comment-home-top-passer-map", "data"), Input("url", "pathname"))
def load_comment_home_pass_map(data, pn):
    key = get_comment_key(pn, "pa_home_passer_map")
    if not key or data is None: return ""
    return data.get(key, "")

@app.callback(Output("store-comment-away-top-passer-map", "data"), Output("save-status-away-top-passer-map", "children"),
              Input("save-comment-away-top-passer-map", "n_clicks"),
              State("comment-away-top-passer-map", "value"), State("url", "pathname"), State("store-comment-away-top-passer-map", "data"),
              prevent_initial_call=True)
def save_comment_away_pass_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_away_passer_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.",color="danger")
    if existing is None: existing = {}; existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-away-top-passer-map", "value"), Input("store-comment-away-top-passer-map", "data"), Input("url", "pathname"))
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
              State("comment-shot-sequence-bar", "value"), State("url", "pathname"), State("store-comment-shot-sequence-bar", "data"),
              prevent_initial_call=True)
def save_comment_shot_seq_bar(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_shot_sequence_stats"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-shot-sequence-bar", "value"), Input("store-comment-shot-sequence-bar", "data"), Input("url", "pathname"))
def load_comment_shot_seq_bar(data, pn):
    key = get_comment_key(pn, "pa_shot_sequence_stats")
    if not key or data is None: return ""
    return data.get(key, "")

# --- COMMENT CALLBACKS FOR HOME TOP SHOT CONTRIBUTOR MAP ---
@app.callback(Output("store-comment-home-top-shot-contributor-map", "data"), Output("save-status-home-top-shot-contributor-map", "children"),
              Input("save-comment-home-top-shot-contributor-map", "n_clicks"),
              State("comment-home-top-shot-contributor-map", "value"), State("url", "pathname"), State("store-comment-home-top-shot-contributor-map", "data"),
              prevent_initial_call=True)
def save_comment_home_shot_contrib_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_home_shot_contributor_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-home-top-shot-contributor-map", "value"), Input("store-comment-home-top-shot-contributor-map", "data"), Input("url", "pathname"))
def load_comment_home_shot_contrib_map(data, pn):
    key = get_comment_key(pn, "pa_home_shot_contributor_map")
    if not key or data is None: return ""
    return data.get(key, "")

# --- COMMENT CALLBACKS FOR AWAY TOP SHOT CONTRIBUTOR MAP ---
@app.callback(Output("store-comment-away-top-shot-contributor-map", "data"), Output("save-status-away-top-shot-contributor-map", "children"),
              Input("save-comment-away-top-shot-contributor-map", "n_clicks"),
              State("comment-away-top-shot-contributor-map", "value"), State("url", "pathname"), State("store-comment-away-top-shot-contributor-map", "data"),
              prevent_initial_call=True)
def save_comment_away_shot_contrib_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_away_shot_contributor_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-away-top-shot-contributor-map", "value"), Input("store-comment-away-top-shot-contributor-map", "data"), Input("url", "pathname"))
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
              State("comment-defender-stats-bar", "value"), State("url", "pathname"), State("store-comment-defender-stats-bar", "data"),
              prevent_initial_call=True)
def save_comment_defender_stats_bar(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_defender_stats"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-defender-stats-bar", "value"), Input("store-comment-defender-stats-bar", "data"), Input("url", "pathname"))
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
        return dash_html.P("Select a player from the dropdown to view their map.")

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
        return dash_html.P("Select a player from the dropdown to view their map.")

    layout_content, _, _ = player_plots.generate_defender_layout_and_data(
        stored_match_data_json, player_stats_df_json, is_for_home_team=False, selected_player=selected_player
    )
    return layout_content

# --- COMMENT CALLBACKS FOR HOME TOP DEFENDER MAP ---
@app.callback(Output("store-comment-home-top-defender-map", "data"), Output("save-status-home-top-defender-map", "children"),
              Input("save-comment-home-top-defender-map", "n_clicks"),
              State("comment-home-top-defender-map", "value"), State("url", "pathname"), State("store-comment-home-top-defender-map", "data"),
              prevent_initial_call=True)
def save_comment_home_defender_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_home_defender_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-home-top-defender-map", "value"), Input("store-comment-home-top-defender-map", "data"), Input("url", "pathname"))
def load_comment_home_defender_map(data, pn):
    key = get_comment_key(pn, "pa_home_defender_map")
    if not key or data is None: return ""
    return data.get(key, "")

# --- COMMENT CALLBACKS FOR AWAY TOP DEFENDER MAP ---
@app.callback(Output("store-comment-away-top-defender-map", "data"), Output("save-status-away-top-defender-map", "children"),
              Input("save-comment-away-top-defender-map", "n_clicks"),
              State("comment-away-top-defender-map", "value"), State("url", "pathname"), State("store-comment-away-top-defender-map", "data"),
              prevent_initial_call=True)
def save_comment_away_defender_map(n_clicks, value, pn, existing):
    if not n_clicks: return no_update, ""
    key = get_comment_key(pn, "pa_away_defender_map"); store_val = existing if existing is not None else no_update
    if not key: return store_val, dbc.Alert("Context error.", color="danger")
    if existing is None: existing = {}
    existing[key] = value
    return existing, dbc.Alert("Saved!", color="success", duration=2000)

@app.callback(Output("comment-away-top-defender-map", "value"), Input("store-comment-away-top-defender-map", "data"), Input("url", "pathname"))
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

        buildup_coverage_panel = render_data_coverage_panel(
            [
                _sequence_retention_coverage_item(
                    df_buildups,
                    label="Buildup candidates",
                    metric_key="buildup_sequence_retention_pct",
                ),
                _coordinate_coverage_item(
                    df_buildups,
                    label="Sequence event coordinates",
                    columns=("x", "y"),
                ),
                _sequence_outcome_coverage_item(
                    df_buildups,
                    sequence_id_column="trigger_sequence_id",
                    label="Buildup outcomes",
                ),
            ],
            note=(
                "Discarded means an eligible detector candidate did not "
                "produce a valid first-phase sequence. UI-filtered "
                "sequences are not counted as discarded."
            ),
        )

        if df_buildups is None or df_buildups.empty:
            return dash_html.Div([
                buildup_coverage_panel,
                dbc.Alert(
                    f"No valid buildup sequences found for {attacking_team}.",
                    color="warning",
                    className="mt-3",
                ),
            ], className="match-tab-body")

        # ---------------------------------------------------------
        # HOME vs AWAY BUILDUP COMPARISON
        # ---------------------------------------------------------

        if active_buildup_tab == 'buildup_home':
            df_home_buildups = df_buildups

            df_away_buildups = (
                buildup_metrics.find_buildup_sequences(
                    df_processed,
                    ATEAM_NAME,
                    HTEAM_NAME,
                    metric_to_analyze='buildup_phase',
                    triggers_buildups=triggers,
                )
            )

        else:
            df_away_buildups = df_buildups

            df_home_buildups = (
                buildup_metrics.find_buildup_sequences(
                    df_processed,
                    HTEAM_NAME,
                    ATEAM_NAME,
                    metric_to_analyze='buildup_phase',
                    triggers_buildups=triggers,
                )
            )


        if df_home_buildups is None:
            df_home_buildups = pd.DataFrame()

        if df_away_buildups is None:
            df_away_buildups = pd.DataFrame()


        home_buildup_summary = (
            sequence_outcome_metrics
            .summarize_sequences(
                df_home_buildups,
                sequence_kind='buildup',
            )
        )

        away_buildup_summary = (
            sequence_outcome_metrics
            .summarize_sequences(
                df_away_buildups,
                sequence_kind='buildup',
            )
        )


        buildup_comparison = (
            sequence_outcome_metrics
            .build_sequence_comparison(
                home_buildup_summary,
                away_buildup_summary,
                home_team=HTEAM_NAME,
                away_team=ATEAM_NAME,
            )
        )


        buildup_comparison_panel = (
            render_sequence_comparison_panel(
                buildup_comparison,
                HCOL,
                ACOL,
                title="Buildup progression",
                description=(
                    "First phase starts from a controlled own-half origin, including the ""actual throw-in, free-kick or goal-kick delivery, "
                    "ends when controlled possession reaches the "
                    "opposition half or the "
                    f"{buildup_metrics.MAX_ACTIVE_BUILDUP_SECONDS:.0f}s "
                    "active window expires, and excludes restart "
                    "dead time from duration."
                ),
            )
        )

        all_sequences = [
            df_buildups[df_buildups['trigger_sequence_id'] == seq_id]
            for seq_id in df_buildups['trigger_sequence_id'].unique()
        ]

        if not all_sequences:
            return dbc.Alert(f"No sequences found for {attacking_team} after grouping.", color="warning", className="mt-3")

        # --- 3. Use the canonical first-phase buildup type ---
        sequences_with_type = []

        for seq_df in all_sequences:
            if seq_df.empty:
                sequences_with_type.append(
                    seq_df
                )
                continue

            buildup_type = seq_df.iloc[-1].get(
                'buildup_type'
            )

            if not buildup_type:
                buildup_type = (
                    buildup_metrics
                    .classify_buildup_type(
                        seq_df
                    )
                )

            seq_df = seq_df.copy()
            # Keep the historical UI/filter key while deriving it from the
            # canonical detector rather than recalculating it in app.py.
            seq_df['lb_type'] = buildup_type
            sequences_with_type.append(
                seq_df
            )
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

        # --- 4. Apply filters strictly: no implicit fallback ---
        filtered_sequences = filter_sequences_exact(
            sequences_with_type,
            active_filter,
            {
                "outcomes": (
                    lambda seq:
                    seq.iloc[-1].get(
                        "sequence_outcome_type"
                    )
                ),
                "flanks": (
                    lambda seq:
                    seq.iloc[-1].get(
                        "dominant_flank"
                    )
                ),
                "types": (
                    lambda seq:
                    seq.iloc[-1].get(
                        "lb_type"
                    )
                ),
            },
        )

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
        if filtered_sequences:
            buildup_stats = (
                buildup_metrics.calculate_buildup_stats(
                    filtered_sequences,
                    not is_away,
                )
            )
            summary_content = (
                buildup_metrics.create_buildup_summary_cards(
                    buildup_stats,
                    active_filter,
                )
            )
        else:
            summary_content = sequence_filter_zero_state(
                "build-up sequences"
            )

        summary_layout = dash_html.Div([
            dash_html.Div([
                dash_html.Div([
                    dash_html.Span("SEQUENCE PROFILE", className="match-panel-eyebrow"),
                    dash_html.H3("Buildup summary", className="match-panel-title"),
                    dash_html.P("Select a value to filter the sequence explorer below.", className="match-panel-description"),
                ]),
                dbc.Button(
                    [dash_html.I(className="fa-solid fa-filter-circle-xmark me-2"), "Reset filters"],
                    id="buildup-reset-filter-btn",
                    size="sm",
                    className="match-secondary-button",
                ),
            ], className="match-panel-header"),
            active_filters_badge,
            summary_content,
        ], className="match-summary-content")

        return dash_html.Div([
            dcc.Store(id='buildup-sequence-store', data={
                'sequences': stored_sequence_data,
                'team_color': team_color,
                'is_away': is_away
            }),
            dcc.Store(
                id='buildup-carousel-controller',
                data=make_carousel_controller(
                    num_items
                ),
            ),

            buildup_comparison_panel,
            buildup_coverage_panel,

            dash_html.Div([
                dash_html.Div([
                    dash_html.Span("TEAM IN POSSESSION", className="match-panel-eyebrow"),
                    dash_html.H3(attacking_team, className="match-panel-title"),
                    dash_html.P(
                        f"{num_items} sequences available after the current filters.",
                        className="match-panel-description",
                    ),
                ]),
                dbc.Button(
                    [dash_html.I(className="fas fa-chart-bar me-2"), "Show / hide summary"],
                    id="buildup-summary-toggle-button",
                    className="match-secondary-button",
                    size="sm",
                ),
            ], className="match-tab-intro"),

            dbc.Collapse(
                dash_html.Section(summary_layout, className="match-panel match-summary-panel"),
                id="buildup-summary-collapse",
                is_open=True,
            ),

            dash_html.Section([
                dash_html.Div([
                    dash_html.Span("SEQUENCE EXPLORER", className="match-panel-eyebrow"),
                    dash_html.H3("Possession chain", className="match-panel-title"),
                    dash_html.P("Use the controls to inspect every selected buildup in chronological detail.", className="match-panel-description"),
                ], className="match-panel-header"),
                dash_html.Div(
                    id='carousel-content-wrapper',
                    className="match-sequence-plot",
                    children=[dcc.Loading(type="circle", children=dash_html.Div(id='buildup-carousel-content'))]
                ),
                dash_html.Div([
                    dbc.Button(
                        [dash_html.I(className="fa-solid fa-chevron-left me-2"), "Previous"],
                        id="buildup-prev-button",
                        className="match-carousel-button",
                        size="sm",
                        disabled=(num_items == 0),
                    ),
                    dash_html.Div(id="buildup-indicator-text", className="match-carousel-indicator"),
                    dbc.Button(
                        ["Next", dash_html.I(className="fa-solid fa-chevron-right ms-2")],
                        id="buildup-next-button",
                        className="match-carousel-button",
                        size="sm",
                        disabled=(num_items == 0),
                    ),
                ], className="match-carousel-controls"),
            ], className="match-panel match-sequence-panel"),

            dash_html.Section([
                dash_html.Div([
                    dash_html.I(className="fa-regular fa-note-sticky"),
                    dash_html.Div([
                        dash_html.H3("Analyst notes", className="match-panel-title"),
                        dash_html.P(f"Save your interpretation of {attacking_team}'s buildup.", className="match-panel-description"),
                    ]),
                ], className="match-comment-heading"),
                dcc.Textarea(
                    id="comment-buildup",
                    placeholder=f"Write your analysis for {attacking_team}...",
                    className="match-comment-input",
                ),
                dash_html.Div([
                    dbc.Button(
                        [dash_html.I(className="fa-solid fa-floppy-disk me-2"), "Save note"],
                        id="save-comment-buildup", className="match-action-button", size="sm",
                    ),
                    dash_html.Div(id="save-status-buildup", className="small"),
                ], className="match-comment-actions"),
            ], className="match-panel match-comment-panel")
        ], className="match-tab-body")

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
    if (
        not n_clicks
        or not controller_data
    ):
        return no_update

    if int(
        controller_data.get(
            "total_items",
            0,
        )
        or 0
    ) <= 0:
        return no_update

    return step_carousel(
        controller_data,
        1,
    )

# Callback 2: Handles the "Previous" button click
@app.callback(
    Output('buildup-carousel-controller', 'data'),
    Input('buildup-prev-button', 'n_clicks'),
    State('buildup-carousel-controller', 'data'),
    prevent_initial_call=True
)
def prev_slide(n_clicks, controller_data):
    if (
        not n_clicks
        or not controller_data
    ):
        return no_update

    if int(
        controller_data.get(
            "total_items",
            0,
        )
        or 0
    ) <= 0:
        return no_update

    return step_carousel(
        controller_data,
        -1,
    )

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

    sequences_json = (
        stored_sequence_data.get(
            "sequences",
            [],
        )
    )

    total_items = int(
        controller_data.get(
            "total_items",
            len(sequences_json),
        )
        or 0
    )

    if (
        total_items <= 0
        or not sequences_json
    ):
        return (
            sequence_filter_zero_state(
                "build-up sequences"
            ),
            "0 sequences",
        )

    active_index = int(
        controller_data.get(
            "active_index",
            0,
        )
        or 0
    )

    if active_index >= len(sequences_json):
        active_index = 0

    indicator_text = (
        f"Sequence {active_index + 1} "
        f"of {total_items}"
    )

    try:
        team_color = stored_sequence_data['team_color']
        is_away = stored_sequence_data['is_away']

        seq_df = pd.read_json(
            sequences_json[active_index],
            orient='split',
        )

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

        plot_component = dcc.Graph(
            figure=fig,
            config={'displayModeBar': False, 'responsive': True},
            className="match-analysis-graph match-sequence-graph",
        )
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
    State("url", "pathname"),
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
    Input("url", "pathname")
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
            home_profile = defensive_metrics.calculate_ppda_profile(
                df_processed, HTEAM_NAME, ATEAM_NAME
            )
            away_profile = defensive_metrics.calculate_ppda_profile(
                df_processed, ATEAM_NAME, HTEAM_NAME
            )
            key_events = defensive_metrics.extract_ppda_key_events(df_processed)

            fig_timeline = defensive_transitions_plotly.plot_ppda_timeline(
                home_profile,
                away_profile,
                key_events,
                HTEAM_NAME,
                ATEAM_NAME,
                HCOL,
                ACOL,
            )
            fig_home = defensive_transitions_plotly.plot_ppda_plotly(
                home_profile['overall']['ppda'],
                home_profile['overall']['df_defensive_actions'],
                home_profile['overall']['df_opponent_passes'],
                HTEAM_NAME,
                HCOL,
                ACOL,
                is_away=False,
                pass_zone_threshold=home_profile['pass_zone_threshold'],
                defensive_zone_threshold=home_profile['defensive_zone_threshold'],
            )
            fig_away = defensive_transitions_plotly.plot_ppda_plotly(
                away_profile['overall']['ppda'],
                away_profile['overall']['df_defensive_actions'],
                away_profile['overall']['df_opponent_passes'],
                ATEAM_NAME,
                ACOL,
                HCOL,
                is_away=True,
                pass_zone_threshold=away_profile['pass_zone_threshold'],
                defensive_zone_threshold=away_profile['defensive_zone_threshold'],
            )

            def format_ppda(value):
                return f"{value:.2f}" if pd.notna(value) and np.isfinite(value) else "N/A"

            def period_metric(label, snapshot):
                return dash_html.Div([
                    dash_html.Span(label, className="ppda-period-label"),
                    dash_html.Strong(format_ppda(snapshot['ppda']), className="ppda-period-value"),
                    dash_html.Small(
                        (
                            f"{snapshot['opponent_passes']} opp. passes "
                            f"÷ {snapshot['defensive_actions']} "
                            "pressing actions"
                        ),
                        className="ppda-period-detail",
                    ),
                ], className="ppda-period-metric")

            def summary_card(team_name, profile, team_color):
                return dash_html.Section([
                    dash_html.Div([
                        dash_html.Div([
                            dash_html.Span("PRESSING INTENSITY", className="match-panel-eyebrow"),
                            dash_html.H3(team_name, className="match-panel-title"),
                        ]),
                        dash_html.Span("Lower is more intense", className="ppda-direction-badge"),
                    ], className="match-panel-header"),
                    dash_html.Div([
                        period_metric("Full match", profile['overall']),
                        period_metric("First half", profile['first_half']),
                        period_metric("Second half", profile['second_half']),
                    ], className="ppda-period-grid"),
                ], className="match-panel ppda-summary-card", style={"borderTop": f"4px solid {team_color}"})

            def pressing_table(dataframe):
                return dash_table.DataTable(
                    data=dataframe.to_dict('records'),
                    columns=[{"name": col, "id": col} for col in dataframe.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={
                        'backgroundColor': '#ffffff',
                        'color': '#29465d',
                        'textAlign': 'center',
                        'border': '0',
                        'borderBottom': '1px solid #e4edf2',
                        'fontFamily': 'Arial',
                        'fontSize': '12px',
                        'padding': '10px 8px',
                    },
                    style_cell_conditional=[
                        {'if': {'column_id': 'Player'}, 'textAlign': 'left', 'fontWeight': '600'},
                    ],
                    style_header={
                        'backgroundColor': '#f1f7f9',
                        'color': '#18344d',
                        'fontWeight': '800',
                        'border': '0',
                        'borderBottom': '1px solid #d7e5eb',
                    },
                    style_as_list_view=True,
                    sort_action="native",
                    page_action="none",
                )

            return dash_html.Div([
                dash_html.Div([
                    dash_html.I(
                        className="fa-solid fa-circle-info"
                    ),

                    dash_html.Span([
                        dash_html.Strong(
                            "PPDA = opponent passes ÷ pressing actions. "
                        ),

                        (
                            "The numerator includes opponent pass attempts "
                            "starting in their first 60% of the pitch "
                            "(x < 60). The denominator includes tackles, "
                            "challenges, interceptions, blocked passes and "
                            "fouls committed from x ≥ 40. "
                        ),

                        dash_html.Strong(
                            "Lower PPDA = more intense pressure."
                        ),
                    ]),

                ], className=(
                    "match-analysis-note "
                    "ppda-definition-note"
                )),

                dash_html.Div([
                    summary_card(HTEAM_NAME, home_profile, HCOL),
                    summary_card(ATEAM_NAME, away_profile, ACOL),
                ], className="ppda-summary-grid"),

                dash_html.Section([
                    dash_html.Div([
                        dash_html.Div([
                            dash_html.Span("MATCH FLOW", className="match-panel-eyebrow"),
                            dash_html.H3("Pressing intensity by match phase", className="match-panel-title"),
                            dash_html.P(
                                "Each bar is an independent 15-minute phase. Height shows pressing actions per 100 opponent passes; official PPDA remains available in the hover.",
                                className="match-panel-description",
                            ),
                        ]),
                        dash_html.Span("Higher bar = more intense pressure", className="match-panel-hint"),
                    ], className="match-panel-header"),
                    dcc.Graph(
                        figure=fig_timeline,
                        config={'displayModeBar': False, 'responsive': True},
                        className="ppda-timeline-graph",
                    ),
                ], className="match-panel ppda-timeline-panel"),

                dash_html.Div([
                    dash_html.Section([
                        dcc.Graph(
                            figure=fig_home,
                            config={'displayModeBar': False, 'responsive': True},
                        )
                    ], className="match-panel ppda-map-panel"),
                    dash_html.Section([
                        dcc.Graph(
                            figure=fig_away,
                            config={'displayModeBar': False, 'responsive': True},
                        )
                    ], className="match-panel ppda-map-panel"),
                ], className="ppda-map-grid"),

                dash_html.Section([
                    dash_html.Div([
                        dash_html.Div([
                            dash_html.P(
                                (
                                    "Counts tackles, challenges, interceptions, "
                                    "blocked passes and fouls committed from x ≥ 40. "
                                    "Fouls suffered are excluded."
                                ),
                                className="match-panel-description",
                            ),
                        ]),
                    ], className="match-panel-header"),
                    dash_html.Div([
                        dash_html.Div([
                            dash_html.H4(HTEAM_NAME, className="ppda-table-team"),
                            pressing_table(home_profile['player_stats']),
                        ]),
                        dash_html.Div([
                            dash_html.H4(ATEAM_NAME, className="ppda-table-team"),
                            pressing_table(away_profile['player_stats']),
                        ]),
                    ], className="ppda-table-grid"),
                ], className="match-panel ppda-player-panel"),
            ], className="ppda-analysis")

        else:
                        # ---------------------------------------------------------
            # ACTIVE DEFENSIVE TRANSITION
            # ---------------------------------------------------------

            if active_tab == 'def_transitions_home':
                team_losing_ball = HTEAM_NAME
                team_building_up = ATEAM_NAME

                # The sequence being plotted belongs
                # to the opponent.
                team_color = ACOL
                is_away = True

            else:
                team_losing_ball = ATEAM_NAME
                team_building_up = HTEAM_NAME

                team_color = HCOL
                is_away = False


            df_transitions = (
                transition_metrics
                .find_buildup_after_possession_loss(
                    df_processed,
                    team_that_lost_possession=
                        team_losing_ball,
                    metric_to_analyze=
                        'defensive_transitions',
                )
            )


            # ---------------------------------------------------------
            # HOME vs AWAY DEFENSIVE COMPARISON
            # ---------------------------------------------------------
            #
            # Home defensive transition:
            # Home loses possession, Away attacks.
            #
            # Away defensive transition:
            # Away loses possession, Home attacks.
            # ---------------------------------------------------------

            if active_tab == 'def_transitions_home':

                df_home_def_transitions = (
                    df_transitions
                    if df_transitions is not None
                    else pd.DataFrame()
                )

                df_away_def_transitions = (
                    transition_metrics
                    .find_buildup_after_possession_loss(
                        df_processed,
                        team_that_lost_possession=
                            ATEAM_NAME,
                        metric_to_analyze=
                            'defensive_transitions',
                    )
                )

            else:

                df_away_def_transitions = (
                    df_transitions
                    if df_transitions is not None
                    else pd.DataFrame()
                )

                df_home_def_transitions = (
                    transition_metrics
                    .find_buildup_after_possession_loss(
                        df_processed,
                        team_that_lost_possession=
                            HTEAM_NAME,
                        metric_to_analyze=
                            'defensive_transitions',
                    )
                )


            if df_home_def_transitions is None:
                df_home_def_transitions = (
                    pd.DataFrame()
                )

            if df_away_def_transitions is None:
                df_away_def_transitions = (
                    pd.DataFrame()
                )


            # ---------------------------------------------------------
            # ONE ROW PER DEFENSIVE TRANSITION
            # ---------------------------------------------------------

            home_def_transition_summary = (
                sequence_outcome_metrics
                .summarize_sequences(
                    df_home_def_transitions,
                    sequence_kind=
                        'defensive_transition',
                )
            )

            away_def_transition_summary = (
                sequence_outcome_metrics
                .summarize_sequences(
                    df_away_def_transitions,
                    sequence_kind=
                        'defensive_transition',
                )
            )


            def_transition_comparison = (
                sequence_outcome_metrics
                .build_sequence_comparison(
                    home_def_transition_summary,
                    away_def_transition_summary,
                    home_team=HTEAM_NAME,
                    away_team=ATEAM_NAME,
                )
            )


            # ---------------------------------------------------------
            # DEFENSIVE SEMANTICS
            # ---------------------------------------------------------
            #
            # The milestones describe what THE OPPONENT achieved
            # after the named team lost possession.
            # ---------------------------------------------------------

            def_transition_comparison_panel = (
                render_sequence_comparison_panel(
                    def_transition_comparison,
                    HCOL,
                    ACOL,

                    title=(
                        "Defensive transition containment"
                    ),

                    description=(
                        "Compare what opponents achieved during the 12-second "
                        "active transition window after each team lost "
                        "possession. An already-advanced attack can complete "
                        "with a shot or goal during a short terminal grace period."
                    ),

                    funnel_keys=[
                        'total_sequences',
                        'reached_opposition_half',
                        'reached_final_third',
                        'entered_penalty_area',
                        'produced_shot',
                        'produced_goal',
                    ],

                    funnel_label_overrides={
                        'total_sequences':
                            'Transitions defended',

                        'reached_opposition_half':
                            'Opp. reached our half',

                        'reached_final_third':
                            'Opp. reached our final third',

                        'entered_penalty_area':
                            'Opp. entered our box',

                        'produced_shot':
                            'Shot conceded',

                        'produced_goal':
                            'Goal conceded',
                    },

                    profile_labels={
                        'duration':
                            'Opp. avg active duration',

                        'passes':
                            'Opp. avg completed passes',
                    },

                    funnel_tooltip_overrides={
                        'reached_opposition_half': (
                            "The opponent reached the "
                            "defending team's half during "
                            "the 12-second transition window."
                        ),

                        'reached_final_third': (
                            "The opponent reached the "
                            "defending team's final third "
                            "during the 12-second transition "
                            "window."
                        ),

                        'entered_penalty_area': (
                            "The opponent reached the "
                            "defending team's penalty area "
                            "during the 12-second transition "
                            "window."
                        ),

                        'produced_shot': (
                            "The opponent produced a shot "
                            "before the defensive transition "
                            "phase ended."
                        ),

                        'produced_goal': (
                            "The opponent scored before the "
                            "defensive transition phase ended."
                        ),
                    },

                    hint_text=(
                        "Lower opponent-progression percentages "
                        "indicate better containment."
                    ),
                )
            )


            # Still show the comparison even if the selected
            # team has no defensive transitions.
            def_transition_coverage_panel = render_data_coverage_panel(
                [
                    _sequence_retention_coverage_item(
                        df_transitions,
                        label="Transition candidates",
                        metric_key="transition_sequence_retention_pct",
                    ),
                    _coordinate_coverage_item(
                        df_transitions,
                        label="Sequence event coordinates",
                        columns=("x", "y"),
                    ),
                    _sequence_outcome_coverage_item(
                        df_transitions,
                        sequence_id_column="loss_sequence_id",
                        label="Transition outcomes",
                    ),
                ],
                note=(
                    "Candidate retention is measured before explorer filters. "
                    "Many eligible losses legitimately do not become transition sequences,"
                    " so retention is informative rather than a warning by default."
                ),
            )

            if (
                df_transitions is None
                or df_transitions.empty
            ):
                return dash_html.Div([

                    def_transition_comparison_panel,
                def_transition_coverage_panel,

                    dbc.Alert(
                        (
                            "No defensive transitions "
                            "found for "
                            f"{team_losing_ball}."
                        ),
                        color="warning",
                    ),

                ], className="match-tab-body")

            # Step 2 – Raggruppa in sequenze singole
            all_sequences = [
                df_transitions[df_transitions['loss_sequence_id'] == seq_id]
                for seq_id in df_transitions['loss_sequence_id'].unique()
            ]

            # Step 3/4 – Apply filters strictly: no implicit fallback
            filtered_sequences = (
                filter_sequences_exact(
                    all_sequences,
                    active_filter,
                    {
                        "outcomes": (
                            lambda seq:
                            seq.iloc[-1].get(
                                "sequence_outcome_type"
                            )
                        ),
                        "flanks": (
                            lambda seq:
                            transition_metrics.calculate_flank(
                                seq["y"]
                            )
                        ),
                        "types": (
                            lambda seq:
                            seq.iloc[0].get(
                                "type_of_initial_loss"
                            )
                        ),
                    },
                )
            )

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

                # -------------------------------------------------
                # STORES
                # -------------------------------------------------

                dcc.Store(
                    id='def-transition-sequence-store',
                    data={
                        'sequences': stored_sequence_data,
                        'team_color': team_color,
                        'is_away': is_away,
                    },
                ),

                dcc.Store(
                    id='def-transition-carousel-controller',
                    data=make_carousel_controller(
                        num_items
                    ),
                ),

                # -------------------------------------------------
                # HOME vs AWAY COMPARISON
                # -------------------------------------------------

                def_transition_comparison_panel,
                def_transition_coverage_panel,

                # -------------------------------------------------
                # ACTIVE TEAM
                # -------------------------------------------------

                dash_html.Div([
                    dash_html.Div([

                        dash_html.Span(
                            "TEAM DEFENDING TRANSITION",
                            className="match-panel-eyebrow",
                        ),

                        dash_html.H3(
                            team_losing_ball,
                            className="match-panel-title",
                        ),

                        dash_html.P(
                            (
                                f"{num_items} defensive transitions "
                                "available after the current filters."
                            ),
                            className="match-panel-description",
                        ),

                    ]),
                ], className="match-tab-intro"),

                # -------------------------------------------------
                # FILTERABLE DEFENSIVE PROFILE
                # -------------------------------------------------

                dash_html.Section([

                    dash_html.Div([

                        dash_html.Div([
                            dash_html.Span(
                                "FILTERABLE PROFILE",
                                className="match-panel-eyebrow",
                            ),

                            dash_html.H3(
                                "Defensive transition profile",
                                className="match-panel-title",
                            ),

                            dash_html.P(
                                (
                                    "Filter the sequences by how the opponent's "
                                    "transition ended, where possession was lost "
                                    "or the type of turnover."
                                ),
                                className="match-panel-description",
                            ),
                        ]),

                        dash_html.Div([

                            dbc.Button(
                                [
                                    dash_html.I(
                                        className="fa-solid fa-sliders me-2"
                                    ),
                                    "Show / hide profile",
                                ],
                                id="def-transition-summary-toggle-button",
                                className="match-secondary-button",
                                size="sm",
                            ),

                            dbc.Button(
                                [
                                    dash_html.I(
                                        className="fa-solid fa-rotate-left me-2"
                                    ),
                                    "Reset filters",
                                ],
                                id="def-transition-reset-filter-btn",
                                className="match-secondary-button",
                                size="sm",
                            ),

                        ], className="def-transition-profile-actions"),

                    ], className="match-panel-header"),

                    dbc.Collapse(

                        dash_html.Div([

                            # Kept for callback compatibility.
                            dash_html.Div(
                                id="def-transition-filter-status",
                            ),

                            dash_html.Div(
                                id="def-transition-summary-content",
                                className="def-transition-summary-content",
                            ),

                        ]),

                        id="def-transition-summary-collapse",
                        is_open=True,
                    ),

                ], className="match-panel def-transition-summary-panel"),


                # -------------------------------------------------
                # DEFENSIVE TRANSITION EXPLORER
                # -------------------------------------------------

                dash_html.Div([

                    # ---------------------------------------------
                    # POSSESSION LOSS MAP
                    # ---------------------------------------------

                    dash_html.Section([

                        dash_html.Div([

                            dash_html.Div([
                                dash_html.Span(
                                    "POSSESSION LOSS LOCATIONS",
                                    className="match-panel-eyebrow",
                                ),

                                dash_html.H3(
                                    "Where possession was lost",
                                    className="match-panel-title",
                                ),

                                dash_html.P(
                                    (
                                        "Locate the turnovers that triggered "
                                        "the opponent's transition."
                                    ),
                                    className="match-panel-description",
                                ),
                            ]),

                        ], className="match-panel-header"),

                        dash_html.Div([

                            dcc.Loading(
                                type="circle",
                                children=dcc.Graph(
                                    id="loss-heatmap-graph",
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                    className="def-transition-map-graph",
                                ),
                            ),

                        ], className="def-transition-map-body"),

                    ], className=(
                        "match-panel "
                        "def-transition-map-panel"
                    )),


                    # ---------------------------------------------
                    # SEQUENCE EXPLORER
                    # ---------------------------------------------

                    dash_html.Section([

                        dash_html.Div([

                            dash_html.Div([
                                dash_html.Span(
                                    "SEQUENCE EXPLORER",
                                    className="match-panel-eyebrow",
                                ),

                                dash_html.H3(
                                    "Defensive transition sequence",
                                    className="match-panel-title",
                                ),

                                dash_html.P(
                                    (
                                        "Inspect what the opponent did during "
                                        "the 12-second window after possession "
                                        "was lost."
                                    ),
                                    className="match-panel-description",
                                ),
                            ]),

                        ], className="match-panel-header"),

                        dash_html.Div([

                            dcc.Loading(
                                type="circle",
                                children=dash_html.Div(
                                    id="def-transition-carousel-content"
                                ),
                            ),

                        ], className="def-transition-sequence-body"),

                        dash_html.Div([

                            dbc.Button(
                                [
                                    dash_html.I(
                                        className="fa-solid fa-chevron-left me-2"
                                    ),
                                    "Previous",
                                ],
                                id="def-transition-prev-button",
                                className="match-carousel-button",
                                size="sm",
                                disabled=(num_items == 0),
                            ),

                            dash_html.Div(
                                id="def-transition-indicator-text",
                                className="match-carousel-indicator",
                            ),

                            dbc.Button(
                                [
                                    "Next",
                                    dash_html.I(
                                        className="fa-solid fa-chevron-right ms-2"
                                    ),
                                ],
                                id="def-transition-next-button",
                                className="match-carousel-button",
                                size="sm",
                                disabled=(num_items == 0),
                            ),

                        ], className="match-carousel-controls"),

                    ], className=(
                        "match-panel "
                        "def-transition-sequence-panel"
                    )),

                ], className="def-transition-explorer-grid"),


                # -------------------------------------------------
                # ANALYST NOTES
                # -------------------------------------------------

                dash_html.Section([

                    dash_html.Div([

                        dash_html.I(
                            className=(
                                "fa-regular fa-note-sticky "
                                "match-comment-icon"
                            ),
                        ),

                        dash_html.Div([
                            dash_html.H3(
                                "Analyst notes",
                                className="match-comment-title",
                            ),

                            dash_html.P(
                                (
                                    f"Summarise {team_losing_ball}'s response "
                                    "immediately after losing possession."
                                ),
                                className="match-comment-description",
                            ),
                        ]),

                    ], className="match-comment-heading"),

                    dcc.Textarea(
                        id="comment-def-transition",
                        placeholder=(
                            f"Write your defensive transition analysis "
                            f"for {team_losing_ball}..."
                        ),
                        className="match-comment-input",
                    ),

                    dash_html.Div([

                        dbc.Button(
                            [
                                dash_html.I(
                                    className="fa-regular fa-floppy-disk me-2"
                                ),
                                "Save note",
                            ],
                            id="save-comment-def-transition",
                            className="match-action-button",
                            size="sm",
                        ),

                        dash_html.Div(
                            id="save-status-def-transition",
                            className="small",
                        ),

                    ], className="match-comment-actions"),

                ], className="match-panel match-comment-panel"),

                # -------------------------------------------------
                # ANALYST NOTES — LEGACY FOR NOW
                # -------------------------------------------------

                dash_html.Hr(
                    className="my-4"
                ),

                dash_html.H6(
                    (
                        f"Comments for {team_losing_ball} "
                        "Defensive Transitions:"
                    ),
                    className="mt-3 text-white",
                ),

                dcc.Textarea(
                    id="comment-def-transition",
                    placeholder=(
                        f"Enter your analysis for "
                        f"{team_losing_ball}..."
                    ),
                    style={
                        'width': '100%',
                        'height': 120,
                        'backgroundColor': '#495057',
                        'color': 'white',
                        'borderColor': '#6c757d',
                    },
                    className="mb-2",
                ),

                dbc.Button(
                    "Save Comment",
                    id="save-comment-def-transition",
                    color="info",
                    size="sm",
                    className="me-2",
                ),

                dash_html.Div(
                    id="save-status-def-transition",
                    className="small d-inline-block",
                ),

            ], className="match-tab-body")

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

        all_sequences = [
            pd.read_json(
                seq,
                orient="split",
            )
            for seq in sequences_json
        ]

        filtered_sequences = (
            filter_sequences_exact(
                all_sequences,
                active_filter,
                {
                    "outcomes": (
                        lambda seq:
                        seq.iloc[-1].get(
                            "sequence_outcome_type"
                        )
                    ),
                    "flanks": (
                        lambda seq:
                        transition_metrics.calculate_flank(
                            seq["y"]
                        )
                    ),
                    "types": (
                        lambda seq:
                        seq.iloc[0].get(
                            "type_of_initial_loss"
                        )
                    ),
                },
            )
        )

        total_sequences = len(
            filtered_sequences
        )

        if total_sequences == 0:
            return (
                sequence_filter_zero_state(
                    "defensive transitions"
                ),
                "0 sequences",
            )

        if active_index >= total_sequences:
            active_index = 0

        selected_seq = filtered_sequences[
            active_index
        ]

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

        graph = dcc.Graph(
            figure=fig,
            config={
                "displayModeBar": False,
                "responsive": True,
            },
            className="def-transition-sequence-graph",
        )
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

    filtered_sequences = (
        filter_sequences_exact(
            all_sequences,
            active_filter,
            {
                "outcomes": (
                    lambda seq:
                    seq.iloc[-1].get(
                        "sequence_outcome_type"
                    )
                ),
                "flanks": (
                    lambda seq:
                    transition_metrics.calculate_flank(
                        seq["y"]
                    )
                ),
                "types": (
                    lambda seq:
                    seq.iloc[0].get(
                        "type_of_initial_loss"
                    )
                ),
            },
        )
    )

    if not filtered_sequences:
        return sequence_filter_zero_state(
            "defensive transitions"
        )

    stats = transition_metrics.calculate_def_transition_stats(filtered_sequences, is_away)
    # transition_profile_table = stats.get("transition_profile_table", pd.DataFrame())
    # # print("[DEBUG] Colonne disponibili:", transition_profile_table.columns)

    # transition_profile_table.sort_values(by="Num_Sequences", ascending=False, inplace=True)

    # if not transition_profile_table.empty:
    #     transition_profile_component = transition_metrics.generate_transition_profile_table(transition_profile_table)
    # else:
    #     transition_profile_component = dbc.Alert("No transition profile data available.", color="secondary")

    filter_labels = {
        "outcomes": "Outcome",
        "flanks": "Loss side",
        "types": "Type of loss",
    }

    if active_filter:
        badges = [
            dbc.Badge(
                f"{filter_labels.get(k, k)}: {v}",
                color="info",
                className="me-2",
                pill=True,
            )
            for k, v in active_filter.items()
        ]

        active_filters_badge = dash_html.Div([
            dash_html.Small(
                "Active filters:",
                className="me-2",
            ),
            *badges,
        ], className="def-transition-active-filters")

    else:
        active_filters_badge = None

    return dash_html.Div([
        active_filters_badge,

        transition_metrics.create_def_transition_summary_cards(
            stats,
            active_filter,
        ),

    ], className="def-transition-filter-cards")

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
    if (
        not n_clicks
        or not controller_data
    ):
        return dash.no_update

    if int(
        controller_data.get(
            "total_items",
            0,
        )
        or 0
    ) <= 0:
        return dash.no_update

    return step_carousel(
        controller_data,
        1,
    )

@app.callback(
    Output('def-transition-carousel-controller', 'data'),
    Input('def-transition-prev-button', 'n_clicks'),
    State('def-transition-carousel-controller', 'data'),
    prevent_initial_call=True
)
def def_transition_prev_slide(n_clicks, controller_data):
    if (
        not n_clicks
        or not controller_data
    ):
        return dash.no_update

    if int(
        controller_data.get(
            "total_items",
            0,
        )
        or 0
    ) <= 0:
        return dash.no_update

    return step_carousel(
        controller_data,
        -1,
    )

@app.callback(
    Output("loss-heatmap-graph", "figure"),
    Input("store-def-transition-filter", "data"),
    State("def-transition-sequence-store", "data")
)
def update_loss_heatmap(active_filter, stored_data):

    if not stored_data:
        return go.Figure()

    sequences_json = stored_data.get("sequences", [])
    opponent_is_away = stored_data.get(
        "is_away",
        False,
    )

    losing_team_is_away = (
        not opponent_is_away
    )

    all_sequences = [pd.read_json(seq, orient="split") for seq in sequences_json]

    filtered_sequences = (
        filter_sequences_exact(
            all_sequences,
            active_filter,
            {
                "outcomes": (
                    lambda seq:
                    seq.iloc[-1].get(
                        "sequence_outcome_type"
                    )
                ),
                "flanks": (
                    lambda seq:
                    transition_metrics.calculate_flank(
                        seq["y"]
                    )
                ),
                "types": (
                    lambda seq:
                    seq.iloc[0].get(
                        "type_of_initial_loss"
                    )
                ),
            },
        )
    )

    return (
        defensive_transitions_plotly
        .plot_loss_heatmap_on_pitch(
            filtered_sequences,
            losing_team_is_away=
                losing_team_is_away,
        )
    )


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
        HTEAM_NAME = match_info.get(
            'hteamName'
        )

        ATEAM_NAME = match_info.get(
            'ateamName'
        )

        if active_tab == 'off_transitions_home':
            team_recovering_ball = HTEAM_NAME
            team_losing_ball = ATEAM_NAME
            team_color = HCOL
            is_away = False
        else:
            team_recovering_ball = ATEAM_NAME
            team_losing_ball = HTEAM_NAME
            team_color = ACOL
            is_away = True

        # ---------------------------------------------------------
        # ACTIVE TEAM TRANSITIONS
        # ---------------------------------------------------------

        df_transitions = (
            transition_metrics
            .find_buildup_after_possession_loss(
                df_processed,
                team_that_lost_possession=team_losing_ball,
                metric_to_analyze='offensive_transitions',
            )
        )


        # ---------------------------------------------------------
        # HOME vs AWAY COMPARISON
        # ---------------------------------------------------------
        #
        # Important:
        # team_that_lost_possession is the OPPOSITE team.
        #
        # Napoli offensive transitions, for example, start after
        # Udinese loses possession.

        if active_tab == 'off_transitions_home':
            df_home_transitions = (
                df_transitions
                if df_transitions is not None
                else pd.DataFrame()
            )

            df_away_transitions = (
                transition_metrics
                .find_buildup_after_possession_loss(
                    df_processed,
                    team_that_lost_possession=HTEAM_NAME,
                    metric_to_analyze='offensive_transitions',
                )
            )

        else:
            df_away_transitions = (
                df_transitions
                if df_transitions is not None
                else pd.DataFrame()
            )

            df_home_transitions = (
                transition_metrics
                .find_buildup_after_possession_loss(
                    df_processed,
                    team_that_lost_possession=ATEAM_NAME,
                    metric_to_analyze='offensive_transitions',
                )
            )


        if df_home_transitions is None:
            df_home_transitions = pd.DataFrame()

        if df_away_transitions is None:
            df_away_transitions = pd.DataFrame()


        home_transition_summary = (
            sequence_outcome_metrics
            .summarize_sequences(
                df_home_transitions,
                sequence_kind='offensive_transition',
            )
        )

        away_transition_summary = (
            sequence_outcome_metrics
            .summarize_sequences(
                df_away_transitions,
                sequence_kind='offensive_transition',
            )
        )


        off_transition_comparison = (
            sequence_outcome_metrics
            .build_sequence_comparison(
                home_transition_summary,
                away_transition_summary,
                home_team=HTEAM_NAME,
                away_team=ATEAM_NAME,
            )
        )


        off_transition_comparison_panel = (
            render_sequence_comparison_panel(
                off_transition_comparison,
                HCOL,
                ACOL,
                title="Offensive transition progression",
                description=(
                    "Compare how far each team progressed during the "
                    "12-second active transition window after regaining "
                    "possession. An already-advanced attack can complete "
                    "with a shot or goal during a short terminal grace period."
                ),
                funnel_keys=[
                    'total_sequences',
                    'reached_opposition_half',
                    'reached_final_third',
                    'entered_penalty_area',
                    'produced_shot',
                    'produced_goal',
                ],
            )
        )

        off_transition_coverage_panel = render_data_coverage_panel(
            [
                _sequence_retention_coverage_item(
                    df_transitions,
                    label="Transition candidates",
                    metric_key="transition_sequence_retention_pct",
                ),
                _coordinate_coverage_item(
                    df_transitions,
                    label="Sequence event coordinates",
                    columns=("x", "y"),
                ),
                _sequence_outcome_coverage_item(
                    df_transitions,
                    sequence_id_column="loss_sequence_id",
                    label="Transition outcomes",
                ),
            ],
            note=(
                "Candidate retention is measured before explorer filters. "
                "Many eligible losses legitimately do not become transition sequences,"
                " so retention is informative rather than a warning by default."
            ),
        )

        if (
            df_transitions is None
            or df_transitions.empty
        ):
            return dash_html.Div([
                off_transition_comparison_panel,
            off_transition_coverage_panel,

                dbc.Alert(
                    (
                        "No offensive transitions found for "
                        f"{team_recovering_ball}."
                    ),
                    color="warning",
                ),
            ], className="match-tab-body")

        # Raggruppamento e ordinamento (logica identica)
        all_sequences = [df_transitions[df_transitions['loss_sequence_id'] == seq_id] for seq_id in df_transitions['loss_sequence_id'].unique()]

        # Step 4 – Apply filters strictly: no implicit fallback
        filtered_sequences = (
            filter_sequences_exact(
                all_sequences,
                active_filter,
                {
                    "outcomes": (
                        lambda seq:
                        seq.iloc[-1].get(
                            "sequence_outcome_type"
                        )
                    ),
                    "flanks": (
                        lambda seq:
                        transition_metrics.calculate_flank(
                            seq["y"]
                        )
                    ),
                    "types": (
                        lambda seq:
                        seq.iloc[0].get(
                            "type_of_initial_loss"
                        )
                    ),
                },
            )
        )

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
            dcc.Store(
                id='off-transition-sequence-store',
                data={
                    'sequences':
                        stored_sequence_data,
                    'team_color':
                        team_color,
                    'is_away':
                        is_away,
                },
            ),

            dcc.Store(
                id='off-transition-carousel-controller',
                data=make_carousel_controller(
                    num_items
                ),
            ),

            off_transition_comparison_panel,
            off_transition_coverage_panel,

            # -----------------------------------------------------
            # ACTIVE TEAM
            # -----------------------------------------------------

            dash_html.Div([
                dash_html.Div([
                    dash_html.Span(
                        "TEAM IN TRANSITION",
                        className="match-panel-eyebrow",
                    ),

                    dash_html.H3(
                        team_recovering_ball,
                        className="match-panel-title",
                    ),

                    dash_html.P(
                        (
                            f"{num_items} offensive transitions "
                            "available after the current filters."
                        ),
                        className="match-panel-description",
                    ),
                ]),

                dbc.Button(
                    [
                        dash_html.I(
                            className="fas fa-chart-bar me-2"
                        ),
                        "Show / hide profile",
                    ],
                    id="off-transition-summary-toggle-button",
                    className="match-secondary-button",
                    size="sm",
                ),

            ], className="match-tab-intro"),


            # -----------------------------------------------------
            # FILTERABLE TRANSITION PROFILE
            # -----------------------------------------------------

            dash_html.Section([

                dash_html.Div([
                    dash_html.Div([
                        dash_html.Span(
                            "FILTERABLE PROFILE",
                            className="match-panel-eyebrow",
                        ),

                        dash_html.H3(
                            "Transition profile",
                            className="match-panel-title",
                        ),

                        dash_html.P(
                            (
                                "Select an outcome, recovery pattern "
                                "or initial action to filter the "
                                "transition explorer below."
                            ),
                            className="match-panel-description",
                        ),
                    ]),

                    dbc.Button(
                        [
                            dash_html.I(
                                className=(
                                    "fa-solid "
                                    "fa-filter-circle-xmark me-2"
                                )
                            ),
                            "Reset filters",
                        ],
                        id="off-transition-reset-filter-btn",
                        className="match-secondary-button",
                        size="sm",
                    ),

                ], className="match-panel-header"),

                dbc.Collapse(
                    dash_html.Div(
                        id="off-transition-summary-content",
                        className="off-transition-summary-content",
                    ),
                    id="off-transition-summary-collapse",
                    is_open=True,
                ),

            ], className=(
                "match-panel "
                "off-transition-summary-panel"
            )),


            # -----------------------------------------------------
            # EXPLORER WORKSPACE
            # -----------------------------------------------------

            dash_html.Div([

                # -------------------------------------------------
                # RECOVERY MAP
                # -------------------------------------------------

                dash_html.Section([

                    dash_html.Div([
                        dash_html.Div([
                            dash_html.Span(
                                "RECOVERY LOCATIONS",
                                className="match-panel-eyebrow",
                            ),

                            dash_html.H3(
                                "Where transitions start",
                                className="match-panel-title",
                            ),

                            dash_html.P(
                                (
                                    "Recovery locations for the "
                                    "transitions matching the "
                                    "current filters."
                                ),
                                className="match-panel-description",
                            ),
                        ]),
                    ], className="match-panel-header"),

                    dash_html.Div([
                        dcc.Loading(
                            type="circle",
                            children=dcc.Graph(
                                id="recovery-heatmap-graph",
                                config={
                                    "displayModeBar": False,
                                    "responsive": True,
                                },
                                className=(
                                    "off-transition-map-graph"
                                ),
                            ),
                        ),
                    ], className="off-transition-map-body"),

                ], className=(
                    "match-panel "
                    "off-transition-map-panel"
                )),


                # -------------------------------------------------
                # SEQUENCE EXPLORER
                # -------------------------------------------------

                dash_html.Section([

                    dash_html.Div([
                        dash_html.Div([
                            dash_html.Span(
                                "SEQUENCE EXPLORER",
                                className="match-panel-eyebrow",
                            ),

                            dash_html.H3(
                                "Transition sequence",
                                className="match-panel-title",
                            ),

                            dash_html.P(
                                (
                                    "Inspect each selected transition "
                                    "in chronological detail."
                                ),
                                className="match-panel-description",
                            ),
                        ]),
                    ], className="match-panel-header"),

                    dash_html.Div([
                        dcc.Loading(
                            type="circle",
                            children=dash_html.Div(
                                id="off-transition-carousel-content"
                            ),
                        ),
                    ], className="off-transition-sequence-body"),

                    dash_html.Div([

                        dbc.Button(
                            [
                                dash_html.I(
                                    className="fa-solid fa-chevron-left me-2"
                                ),
                                "Previous",
                            ],
                            id="off-transition-prev-button",
                            className="match-carousel-button",
                            size="sm",
                            disabled=(num_items == 0),
                        ),

                        dash_html.Div(
                            id="off-transition-indicator-text",
                            className="match-carousel-indicator",
                        ),

                        dbc.Button(
                            [
                                "Next",
                                dash_html.I(
                                    className="fa-solid fa-chevron-right ms-2"
                                ),
                            ],
                            id="off-transition-next-button",
                            className="match-carousel-button",
                            size="sm",
                            disabled=(num_items == 0),
                        ),

                    ], className="match-carousel-controls"),

                ], className=(
                    "match-panel "
                    "off-transition-sequence-panel"
                )),

            ], className="off-transition-explorer-grid"),

        ], className="match-tab-body")

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
        sequences_json = (
            stored_data.get(
                "sequences",
                [],
            )
        )

        total_items = int(
            controller_data.get(
                "total_items",
                len(sequences_json),
            )
            or 0
        )

        if (
            total_items <= 0
            or not sequences_json
        ):
            return (
                sequence_filter_zero_state(
                    "offensive transitions"
                ),
                "0 sequences",
            )

        active_index = int(
            controller_data.get(
                "active_index",
                0,
            )
            or 0
        )

        if active_index >= len(
            sequences_json
        ):
            active_index = 0

        seq_df = pd.read_json(
            io.StringIO(
                sequences_json[
                    active_index
                ]
            ),
            orient='split',
        )

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
        indicator = (
            f"Sequence {active_index + 1} "
            f"of {total_items}"
        )

        return graph, indicator

    except Exception as e:
        # Forniamo un messaggio di errore più utile in caso di problemi
        error_message = f"Errore durante la generazione del plot della transizione offensiva: {e}"
        tb_str = traceback.format_exc()
        print(f"{error_message}\n{tb_str}")
        return dbc.Alert(error_message, color="danger"), "Error"

# Callback per i pulsanti del carosello
@app.callback(Output('off-transition-carousel-controller', 'data', allow_duplicate=True), Input('off-transition-next-button', 'n_clicks'), State('off-transition-carousel-controller', 'data'), prevent_initial_call=True)
def off_next(n, data):
    if (
        not n
        or not data
        or int(
            data.get(
                "total_items",
                0,
            )
            or 0
        ) <= 0
    ):
        return no_update

    return step_carousel(
        data,
        1,
    )
@app.callback(Output('off-transition-carousel-controller', 'data'), Input('off-transition-prev-button', 'n_clicks'), State('off-transition-carousel-controller', 'data'), prevent_initial_call=True)
def off_prev(n, data):
    if (
        not n
        or not data
        or int(
            data.get(
                "total_items",
                0,
            )
            or 0
        ) <= 0
    ):
        return no_update

    return step_carousel(
        data,
        -1,
    )

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

    filtered_sequences = (
        filter_sequences_exact(
            all_sequences,
            active_filter,
            {
                "outcomes": (
                    lambda seq:
                    seq.iloc[-1].get(
                        "sequence_outcome_type"
                    )
                ),
                "flanks": (
                    lambda seq:
                    transition_metrics.calculate_flank(
                        seq["y"]
                    )
                ),
                "types": (
                    lambda seq:
                    seq.iloc[0].get(
                        "type_of_initial_loss"
                    )
                ),
            },
        )
    )

    if not filtered_sequences:
        return (
            sequence_filter_zero_state(
                "offensive transitions"
            ),
            go.Figure(),
        )

    stats = transition_metrics.calculate_off_transition_stats(filtered_sequences)
    cards = transition_metrics.create_off_transition_summary_cards(stats, active_filter)
    heatmap_fig = offensive_transitions_plotly.plot_recovery_heatmap_on_pitch(filtered_sequences, is_away=is_away)
    heatmap_fig.update_layout(
        title=None,
        margin=dict(
            l=10,
            r=10,
            t=15,
            b=10,
        ),
    )

    # ---------------------------------------------------------
    # COMMON TRANSITION PATTERNS
    # ---------------------------------------------------------
    #
    # Keep this block descriptive only.
    #
    # Avg duration and avg completed passes are intentionally
    # NOT taken from the legacy transition profile table:
    # canonical sequence-level definitions are already shown
    # in the comparison panel above.

    profile_df = stats.get(
        "transition_profile_table",
        pd.DataFrame(),
    ).copy()


    if not profile_df.empty:

        profile_df = profile_df.sort_values(
            by="Num_Sequences",
            ascending=False,
        )

        total_profile_sequences = int(
            profile_df["Num_Sequences"].sum()
        )

        profile_df["Share"] = (
            profile_df["Num_Sequences"]
            / max(total_profile_sequences, 1)
            * 100
        )

        profile_df["Share"] = (
            profile_df["Share"]
            .round(0)
            .astype(int)
            .astype(str)
            + "%"
        )

        profile_df = profile_df.rename(
            columns={
                "Recovery Zone":
                    "Recovery zone",

                "Attack Side":
                    "Attack side",

                "Num_Sequences":
                    "Sequences",
            }
        )

        profile_df = profile_df[
            [
                "Recovery zone",
                "Attack side",
                "Sequences",
                "Share",
            ]
        ]


        profile_table_component = (
            dbc.Table.from_dataframe(
                profile_df,
                striped=False,
                bordered=False,
                hover=True,
                responsive=True,
                index=False,
                className=(
                    "off-transition-pattern-table "
                    "mb-0"
                ),
            )
        )

    else:

        profile_table_component = (
            dbc.Alert(
                (
                    "No transition pattern "
                    "data available."
                ),
                color="secondary",
            )
        )


    # ---------------------------------------------------------
    # SUMMARY LAYOUT
    # ---------------------------------------------------------

    summary_layout = dash_html.Div([

        dash_html.Div(
            cards,
            className=(
                "off-transition-filter-cards"
            ),
        ),

        dash_html.Div([

            dash_html.Div([
                dash_html.Span(
                    "COMMON PATTERNS",
                    className="match-panel-eyebrow",
                ),

                dash_html.H4(
                    "Recovery zone × attack side",
                    className=(
                        "off-transition-pattern-title"
                    ),
                ),

                dash_html.P(
                    (
                        "The most frequent combinations "
                        "of recovery zone and direction "
                        "of the subsequent attack."
                    ),
                    className="match-panel-description",
                ),

            ], className=(
                "off-transition-pattern-header"
            )),

            profile_table_component,

        ], className=(
            "off-transition-pattern-block"
        )),

    ], className=(
        "off-transition-summary-layout"
    ))

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

        # Penalty kicks are extracted directly from their shot event.
        # The award and the kick can be separated by several minutes, so
        # tracing from the foul is not a reliable way to model penalties.
        penalty_sequences = (
            set_piece_metrics.extract_penalty_set_piece_sequences(
                df_processed,
                team_name,
            )
        )

        # Keep penalty-award fouls out of the regular free-kick pipeline;
        # otherwise the same penalty can be counted once as a free kick and
        # once again from the direct penalty-shot extraction above.
        set_piece_source = df_processed
        if 'Penalty' in df_processed.columns:
            penalty_award_mask = (
                (df_processed['type_name'] == 'Foul')
                & df_processed['Penalty'].isin([1, '1', True])
            )
            set_piece_source = df_processed.loc[
                ~penalty_award_mask
            ].copy()

        # REL-09: classify the actual restart delivery. Generic
        # Out/Foul/Corner Awarded events are not the restart taxonomy.
        restart_sequences = (
            restart_metrics.extract_restart_sequences(
                set_piece_source,
                team_name,
            )
        )

        all_sequences = list(restart_sequences)
        all_sequences.extend(penalty_sequences)

        if not all_sequences:
            return dbc.Alert(
                f"No offensive restarts found for {team_name}.",
                color="warning",
                className="mt-3",
            )

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
            if outcome == 'Penalty Goal': return 0
            elif outcome == 'Goals': return 1
            elif outcome in ('Penalty Saved', 'Penalty Missed'): return 2
            elif outcome == 'Shots': return 3
            elif outcome == 'Big Chances': return 4
            elif outcome == 'Lost Possessions': return 5
            else: return 6

        sorted_sequences_for_carousel = sorted(sequences_for_carousel, key=get_set_piece_quality_score)
        num_items = len(sorted_sequences_for_carousel)

        carousel_section = dbc.Alert("No sequences match the current filter.", color="warning", className="mt-4")
        if num_items > 0:
            carousel_section = dash_html.Div([
                dcc.Store(id='set-piece-sequence-store', data={'sequences': [s.to_json(orient='split') for s in sorted_sequences_for_carousel], 'team_color': team_color, 'is_away': not is_home}),
                dcc.Store(id='set-piece-carousel-controller', data={'active_index': 0, 'total_items': num_items}),
                dash_html.H5("Restart Explorer", className="text-white mt-4"),
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
                [dash_html.I(className="fas fa-chart-bar me-2"), "Toggle Restart Summary"],
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
        return dbc.Alert(f"Error in Restart tab: {traceback.format_exc()}", color="danger", style={"whiteSpace": "pre-wrap"})

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
    Output("cross-filter-store", "data", allow_duplicate=True),
    Output("cross-selection-store", "data", allow_duplicate=True), # <-- NUOVO OUTPUT
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
        return no_update, no_update, no_update

    trigger_source = triggered_info.get('type')

    if trigger_source == 'reset-btn':
        section = triggered_info.get('section')
        if section == 'set-piece':
            return None, no_update, no_update
        elif section == 'crosses':
            # Resetta sia il filtro che la selezione
            return no_update, None, None # <-- MODIFICATO QUI
        else:
            return no_update, no_update, no_update

    elif trigger_source == 'sp-filter':
        filter_type = triggered_info.get('filter_type')
        value = triggered_info.get('value')
        current_filter = sp_filter or {}
        if current_filter.get(filter_type) == value:
            current_filter.pop(filter_type, None)
        else:
            current_filter[filter_type] = value
        return current_filter if current_filter else None, no_update, no_update

    elif trigger_source == 'cross-filter':
        filter_type = triggered_info.get('filter_type')
        value = triggered_info.get('value')
        current_filter = cross_filter or {}
        if current_filter.get(filter_type) == value:
            current_filter.pop(filter_type, None)
        else:
            current_filter[filter_type] = value
        # Quando applico un filtro, resetto anche la selezione del singolo cross
        return no_update, current_filter if current_filter else None, None # <-- MODIFICATO QUI

    return no_update, no_update, no_update


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
                dbc.Tab(
                    label="Home Crosses",
                    tab_id="crosses-home",
                ),
                dbc.Tab(
                    label="Away Crosses",
                    tab_id="crosses-away",
                ),
            ],
            className="cross-team-tabs",
        ),

        dcc.Loading(
            type="circle",
            children=dash_html.Div(
                id="crosses-team-content"
            ),
        ),

    ], className="cross-analysis")

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

        flow_summary, flow_routes = (
            cross_metrics.build_cross_flow_profile(
                crosses_filtered,
                limit=8,
            )
        )

        def build_cross_flow_component(
            summary,
            routes,
            team_color,
        ):
            if (
                routes is None
                or routes.empty
            ):
                return dash_html.Div(
                    "No cross-flow data available.",
                    className="cross-flow-empty",
                )

            max_count = max(
                int(
                    routes['Crosses'].max()
                ),
                1,
            )

            route_rows = []

            for rank, (_, row) in enumerate(
                routes.iterrows(),
                start=1,
            ):
                count = int(
                    row['Crosses']
                )

                share = float(
                    row['Share %']
                )

                completed = int(
                    row['Completed']
                )

                completion_pct = float(
                    row['Completion %']
                )

                bar_width = (
                    count
                    / max_count
                    * 100
                )

                route_rows.append(
                    dash_html.Div([

                        dash_html.Span(
                            str(rank),
                            className="cross-flow-rank",
                        ),

                        dash_html.Div([

                            dash_html.Div([

                                dash_html.Span(
                                    row['Origin Zone'],
                                    className=(
                                        "cross-flow-zone "
                                        "cross-flow-zone--origin"
                                    ),
                                ),

                                dash_html.I(
                                    className=(
                                        "fa-solid "
                                        "fa-arrow-right "
                                        "cross-flow-arrow"
                                    ),
                                ),

                                dash_html.Span(
                                    row['Destination Zone'],
                                    className=(
                                        "cross-flow-zone "
                                        "cross-flow-zone--destination"
                                    ),
                                ),

                            ], className="cross-flow-route"),

                            dash_html.Div(
                                dash_html.Span(
                                    style={
                                        "width":
                                            f"{bar_width:.1f}%",
                                        "backgroundColor":
                                            team_color,
                                    }
                                ),
                                className="cross-flow-track",
                            ),

                            dash_html.Div([

                                dash_html.Span(
                                    f"{share:.0f}% of crosses",
                                ),

                                dash_html.Span(
                                    "·",
                                ),

                                dash_html.Span(
                                    (
                                        f"{completed}/{count} "
                                        f"completed "
                                        f"({completion_pct:.0f}%)"
                                    ),
                                ),

                            ], className="cross-flow-detail"),

                        ], className="cross-flow-main"),

                        dash_html.Strong(
                            str(count),
                            className="cross-flow-count",
                        ),

                    ], className="cross-flow-row")
                )

            return dash_html.Div([

                # -----------------------------------------------------
                # SUMMARY
                # -----------------------------------------------------

                dash_html.Div([

                    dash_html.Div([
                        dash_html.Span(
                            "MOST USED ROUTE",
                            className="cross-flow-kpi-label",
                        ),

                        dash_html.Strong(
                            summary['top_route'],
                            className="cross-flow-kpi-value",
                        ),

                        dash_html.Small(
                            (
                                f"{summary['top_route_count']} crosses · "
                                f"{summary['top_route_pct']:.0f}% "
                                "of the sample"
                            ),
                            className="cross-flow-kpi-detail",
                        ),
                    ], className="cross-flow-kpi"),

                    dash_html.Div([
                        dash_html.Span(
                            "TOP 3 CONCENTRATION",
                            className="cross-flow-kpi-label",
                        ),

                        dash_html.Strong(
                            f"{summary['top_three_pct']:.0f}%",
                            className="cross-flow-kpi-value",
                        ),

                        dash_html.Small(
                            "Share of crosses using the three most common routes",
                            className="cross-flow-kpi-detail",
                        ),
                    ], className="cross-flow-kpi"),

                    dash_html.Div([
                        dash_html.Span(
                            "SAMPLE",
                            className="cross-flow-kpi-label",
                        ),

                        dash_html.Strong(
                            str(
                                summary['total_crosses']
                            ),
                            className="cross-flow-kpi-value",
                        ),

                        dash_html.Small(
                            "Currently filtered crosses",
                            className="cross-flow-kpi-detail",
                        ),
                    ], className="cross-flow-kpi"),

                ], className="cross-flow-kpi-grid"),

                # -----------------------------------------------------
                # ROUTES
                # -----------------------------------------------------

                dash_html.Div(
                    route_rows,
                    className="cross-flow-route-list",
                ),

            ])

        flow_component = (
            build_cross_flow_component(
                flow_summary,
                flow_routes,
                HCOL if not is_away else ACOL,
            )
        )

        # -----------------------------------------------------
        # TEAM HEADER
        # -----------------------------------------------------

        team_header = dash_html.Div([

            dash_html.Div([

                dash_html.Span(
                    (
                        "AWAY TEAM"
                        if is_away
                        else "HOME TEAM"
                    ),
                    className="match-panel-eyebrow",
                ),

                dash_html.H4(
                    team_name,
                    className="match-team-name",
                ),

            ]),

            dash_html.Div([

                dash_html.Span(
                    (
                        f"n = {len(crosses_filtered)} "
                        "crosses"
                    ),
                    className="progressive-sample-size",
                ),

                dash_html.Span(
                    "All teams attack left to right",
                    className="match-panel-hint",
                ),

            ], className="progressive-panel-meta"),

        ], className=(
            "match-panel-header "
            "cross-team-header"
        ))


        # -----------------------------------------------------
        # FILTERABLE PROFILE
        # -----------------------------------------------------

        summary_panel = dash_html.Section([

            dash_html.Div([

                dash_html.Div([

                    dash_html.Span(
                        "FILTERABLE PROFILE",
                        className="match-panel-eyebrow",
                    ),

                    dash_html.H3(
                        "Cross profile",
                        className="match-panel-title",
                    ),

                    dash_html.P(
                        (
                            "Explore delivery patterns by "
                            "origin, destination, swing, "
                            "outcome, foot, taker and "
                            "play type."
                        ),
                        className="match-panel-description",
                    ),

                ]),

                dash_html.Div([

                    dbc.Button(
                        [
                            dash_html.I(
                                className=(
                                    "fa-solid "
                                    "fa-sliders me-2"
                                )
                            ),
                            "Show / hide profile",
                        ],
                        id=(
                            "cross-summary-toggle-button"
                        ),
                        className="match-secondary-button",
                        size="sm",
                    ),

                    dbc.Button(
                        [
                            dash_html.I(
                                className=(
                                    "fa-solid "
                                    "fa-rotate-left me-2"
                                )
                            ),
                            "Reset filters",
                        ],
                        id={
                            'type': 'reset-btn',
                            'section': 'crosses',
                        },
                        className="match-secondary-button",
                        size="sm",
                    ),

                ], className="cross-profile-actions"),

            ], className="match-panel-header"),

            dbc.Collapse(

                dash_html.Div(
                    cards,
                    className="cross-summary-content",
                ),

                id="cross-summary-collapse",
                is_open=True,
            ),

        ], className=(
            "match-panel "
            "cross-summary-panel"
        ))


        # -----------------------------------------------------
        # ANALYSIS WORKSPACE
        # -----------------------------------------------------

        analysis_tabs = dbc.Tabs(

            [

                # ---------------------------------------------
                # LOCATION MAPS
                # ---------------------------------------------
                dbc.Tab(

                    label="Location maps",
                    tab_id="heatmaps-tab",

                    children=[

                        dash_html.Div([

                            dash_html.Section([

                                dash_html.Div([

                                    dash_html.Span(
                                        "CROSS ORIGINS",
                                        className=(
                                            "match-panel-eyebrow"
                                        ),
                                    ),

                                    dash_html.H3(
                                        "Where crosses started",
                                        className=(
                                            "match-panel-title"
                                        ),
                                    ),

                                    dash_html.P(
                                        (
                                            "Click a cross to "
                                            "highlight the same "
                                            "delivery on both maps."
                                        ),
                                        className=(
                                            "match-panel-description"
                                        ),
                                    ),

                                ], className=(
                                    "cross-map-header"
                                )),

                                dcc.Graph(
                                    id="cross-origin-map",
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                    className="cross-map-graph",
                                ),

                            ], className=(
                                "match-panel "
                                "cross-map-panel"
                            )),


                            dash_html.Section([

                                dash_html.Div([

                                    dash_html.Span(
                                        "CROSS DESTINATIONS",
                                        className=(
                                            "match-panel-eyebrow"
                                        ),
                                    ),

                                    dash_html.H3(
                                        "Where crosses arrived",
                                        className=(
                                            "match-panel-title"
                                        ),
                                    ),

                                    dash_html.P(
                                        (
                                            "Destination locations "
                                            "for the currently "
                                            "filtered deliveries."
                                        ),
                                        className=(
                                            "match-panel-description"
                                        ),
                                    ),

                                ], className=(
                                    "cross-map-header"
                                )),

                                dcc.Graph(
                                    id="cross-dest-map",
                                    config={
                                        "displayModeBar": False,
                                        "responsive": True,
                                    },
                                    className="cross-map-graph",
                                ),

                            ], className=(
                                "match-panel "
                                "cross-map-panel"
                            )),

                        ], className="cross-map-grid"),

                    ],

                ),


                # ---------------------------------------------
                # FLOW
                # ---------------------------------------------
                dbc.Tab(

                    label="Flow analysis",
                    tab_id="flow-tab",

                    children=[

                        dash_html.Section([

                            dash_html.Div([

                                dash_html.Div([

                                    dash_html.Span(
                                        "DELIVERY PATHWAYS",
                                        className="match-panel-eyebrow",
                                    ),

                                    dash_html.H3(
                                        "Cross flow profile",
                                        className="match-panel-title",
                                    ),

                                    dash_html.P(
                                        (
                                            "Rank the most common routes from "
                                            "cross origin to destination. "
                                            "Percentages use all currently "
                                            "filtered crosses."
                                        ),
                                        className="match-panel-description",
                                    ),

                                ]),

                                dash_html.Div(
                                    dash_html.Span(
                                        (
                                            "Routes combine crosses with the "
                                            "same origin and destination zones."
                                        ),
                                        className="match-panel-hint",
                                    ),
                                ),

                            ], className="match-panel-header"),

                            dash_html.Div(
                                flow_component,
                                className="cross-flow-profile-body",
                            ),

                        ], className=(
                            "match-panel "
                            "cross-flow-profile-panel"
                        )),

                    ],

                ),

            ],

            id="cross-analysis-subtabs",
            active_tab="heatmaps-tab",
            className="cross-analysis-tabs",
        )


        # -----------------------------------------------------
        # RETURN
        # -----------------------------------------------------

        return dash_html.Div([

            dcc.Store(
                id="cross-data-store-current-team",
                data=crosses_filtered.to_json(
                    orient="split"
                ),
            ),

            dash_html.Section([
                team_header,
            ], className=(
                "match-panel "
                "cross-team-panel"
            )),

            summary_panel,

            analysis_tabs,

        ], className="cross-team-analysis")

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

# --- NEW: Callback to reset cross filters when switching between Home/Away ---
@app.callback(
    Output("cross-filter-store", "data", allow_duplicate=True),
    Input("crosses-team-tabs", "active_tab"),
    prevent_initial_call=True
)
def reset_cross_filter_on_team_switch(active_tab):
    """
    This callback fires whenever the user switches between the Home and Away
    crosses tabs. Its only job is to reset the filter to ensure no old,
    stale filters are applied to the new view.
    """
    print(f"Crosses team tab changed to '{active_tab}'. Resetting cross filter store.")
    return None # Returning None effectively clears the store

# Callback 3: Gestisce la selezione/deselezione del punto e aggiorna lo store
@app.callback(
    Output('cross-selection-store', 'data'),
    Input('cross-origin-map', 'clickData'),
    Input('cross-dest-map', 'clickData'),
    State('cross-selection-store', 'data'),
    prevent_initial_call=True
)
def update_cross_selection_on_map_click(origin_click, dest_click, selected_cross_id):
    ctx = dash.callback_context
    # Se non c'è stato un click, non fare nulla
    if not ctx.triggered:
        return no_update

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
# @app.callback(
#     Output("report-html-content-store", "data"),
#     Output("clientside-report-trigger-div", "children"), # Output simple trigger
#     Input("generate-report-button", "n_clicks"),
#     State("store-df-match", "data"),
#     State("url", "pathname"),
#     State("store-comment-formation", "data"),         # State 1
#     State("store-comment-pass-network", "data"),      # State 2
#     State("store-comment-progressive-passes", "data"),# State 3 - THIS IS THE ONE
#     # ... other comment stores ...
#     prevent_initial_call=True
# )
# def prepare_report_and_trigger_clientside(
#     n_clicks, stored_match_data, pathname,
#     formation_comments_data,
#     pass_network_comments_data,
#     progressive_passes_comments_data
# ):
#     if n_clicks is None or not stored_match_data:
#         return no_update, no_update # No update for both outputs

#     print(f"--- prepare_report_and_trigger_clientside TRIGGERED (n_clicks: {n_clicks}) ---")

#     report_html_elements = []
#     match_info_dict = {}
#     if stored_match_data.get('match_info'):
#         match_info_dict = json.loads(stored_match_data['match_info'])
#         # ... (extracting hteam, ateam, etc.) ...
#         hteam = match_info_dict.get('hteamDisplayName', 'Home')
#         ateam = match_info_dict.get('ateamDisplayName', 'Away')
#         comp = match_info_dict.get('competitionName', '')
#         round_n = match_info_dict.get('roundNameFromFilename', '')
#         date_val = match_info_dict.get('date_formatted', '')
#         hs = match_info_dict.get('home_score', '')
#         aws = match_info_dict.get('away_score', '')
#         score = f"{hs} - {aws}" if hs is not None and aws is not None else "vs"

#         report_html_elements.append(f"<h1>Match Report: {hteam} {score} {ateam}</h1>")
#         report_html_elements.append(f"<p>{comp} - {round_n} | {date_val}</p><hr>")


#     # # --- Section for Formation ---
#     # report_html_elements.append("<h2>Formation Analysis</h2>")
#     # formation_img_component = show_match_formation(stored_match_data)
#     # if isinstance(formation_img_component, dash_html.Img): # Use aliased dash_html
#     #     report_html_elements.append(f"<img src='{formation_img_component.src}' style='width:90%; max-width:800px; display:block; margin:auto;'/>")
#     # elif isinstance(formation_img_component, dash_html.P):
#     #     report_html_elements.append(f"<p><em>Error generating formation plot: {str(formation_img_component.children)}</em></p>")
#     # else:
#     #     report_html_elements.append("<p>Formation plot could not be generated.</p>")

#     # form_comment_key = get_comment_key(pathname, "formation")
#     # if formation_comments_data and form_comment_key and formation_comments_data.get(form_comment_key):
#     #     report_html_elements.append("<h4>Comments:</h4>")
#     #     comment_text = dash_html.escape(formation_comments_data.get(form_comment_key)) # <<<--- CORRECTED
#     #     report_html_elements.append(f"<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc;'>{comment_text}</pre>")
#     # report_html_elements.append("<hr>")


#     # # --- Section for Pass Network ---
#     # report_html_elements.append("<h2>Pass Network Analysis</h2>")
#     # pass_network_img_component = show_pass_network_graph(stored_match_data)
#     # if isinstance(pass_network_img_component, dash_html.Img): # Use aliased dash_html
#     #     report_html_elements.append(f"<img src='{pass_network_img_component.src}' style='width:90%; max-width:800px; display:block; margin:auto;'/>")
#     # elif isinstance(pass_network_img_component, dash_html.P):
#     #     report_html_elements.append(f"<p><em>Error generating pass network plot: {str(pass_network_img_component.children)}</em></p>")
#     # else:
#     #     report_html_elements.append("<p>Pass Network plot could not be generated.</p>")

#     # pn_comment_key = get_comment_key(pathname, "pass_network")
#     # if pass_network_comments_data and pn_comment_key and pass_network_comments_data.get(pn_comment_key):
#     #     report_html_elements.append("<h4>Comments:</h4>")
#     #     comment_text = dash_html.escape(pass_network_comments_data.get(pn_comment_key))
#     #     report_html_elements.append(f"<pre style='white-space: pre-wrap; word-wrap: break-word; background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc;'>{comment_text}</pre>")
#     # report_html_elements.append("<hr>")

#     # # ... (rest of the function, including final_html_string) ...
#     # final_html_string = f"""
#     # <html>
#     #     <head>
#     #         <title>Match Report</title>
#     #         <style>
#     #             body {{ font-family: sans-serif; margin: 20px; }}
#     #             h1, h2, h3, h4 {{ color: #333; }}
#     #             hr {{ margin-top: 20px; margin-bottom: 20px; border: 0; border-top: 1px solid #eee; }}
#     #             img {{ border: 1px solid #ddd; margin-bottom: 10px; padding: 5px; background-color: white; }}
#     #             pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f0f0f0; padding: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 0.9em; }}
#     #         </style>
#     #     </head>
#     #     <body>
#     #         {''.join(report_html_elements)}
#     #     </body>
#     # </html>
#     # """
#     # print(f"prepare_report_html: Generated HTML (first 200 chars): {final_html_string[:200]}")
#     # print(f"prepare_report_html: Generated HTML (last 200 chars): {final_html_string[-200:]}")
#     # print(f"prepare_report_html: Total length of HTML string: {len(final_html_string)}")
#     # # ... (your logic to build final_html_string) ...
#     # Example:
#     report_html_elements = ["<h1>Test Report Version 2</h1>"]
#     # ... (add plots and comments as before) ...
#     final_html_string = f"<html><body>{''.join(report_html_elements)}</body></html>"
#     # ...

#     print(f"prepare_report_and_trigger_clientside: HTML length: {len(final_html_string)}")

#     # Return HTML to its store, and a simple trigger (timestamp) to the dummy div
#     trigger_value = datetime.now().timestamp()
#     print(f"prepare_report_and_trigger_clientside: Setting trigger value: {trigger_value}")
#     return final_html_string, trigger_value

# # def relay_trigger_for_report_window(report_html):
# #     if report_html:
# #         return datetime.now().timestamp() # Or just a counter, anything to trigger the change
# #     return no_update

# # # --- NEW PYTHON CALLBACK TO TRIGGER CLIENTSIDE ACTION & CLEAR HTML STORE ---
# # @app.callback(
# #     Output("clientside-report-trigger-div", "children"), # Output to dummy div (acts as trigger)
# #     Output("report-html-content-store", "data", allow_duplicate=True), # Output to clear the store
# #     Input("report-html-content-store", "data"), # Input: when HTML is ready
# #     prevent_initial_call=True
# # )
# # def trigger_clientside_and_clear_store(report_html_content):
# #     if report_html_content:
# #         print("trigger_clientside_and_clear_store: HTML ready, triggering clientside and clearing store.")
# #         # The value passed to the dummy div's children can be anything that changes.
# #         # The clientside callback will use the HTML from the store via State.
# #         # We pass the HTML itself as the trigger data, so the clientside callback gets it directly.
# #         return report_html_content, None # Trigger with HTML, then clear the store
# #     print("trigger_clientside_and_clear_store: No HTML, no action.")
# #     return no_update, no_update

# # Callback 3: Clientside callback to open window

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


@callback(
    Output("download-dataframe-csv", "data"),
    Input("btn-download-csv", "n_clicks"),
    State("store-df-match", "data"),
    prevent_initial_call=True,
)
def download_csv(n_clicks, stored_data_json):
    if not n_clicks or not stored_data_json:
        return no_update
    try:
        df = pd.read_json(io.StringIO(stored_data_json['df']), orient='split')
        match_info = json.loads(stored_data_json['match_info'])
        hteam = match_info.get('hteamDisplayName', 'Home')
        ateam = match_info.get('ateamDisplayName', 'Away')
        filename = f"match_events_{hteam}_vs_{ateam}.csv"
        return dcc.send_data_frame(df.to_csv, filename=filename, index=False)
    except Exception as e:
        print(f"Error during CSV download: {e}")
        return no_update

# -----------------------------------------------------------------------------
# Esecuzione dell'App
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
