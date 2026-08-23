"""PLOT-04 Pass Network UI component."""

from __future__ import annotations

import io
import json

import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.data_processing import pass_processing
from src.metrics import pass_network_metrics
from src.visualization import pass_plotly


PERIOD_LABELS = {
    "full": "Full Match",
    "1h": "First Half",
    "2h": "Second Half",
}


def controls():
    return html.Section([
        html.Div([
            html.Div([
                html.Span("TIME WINDOW", className="match-panel-eyebrow"),
                html.Strong("Network period", className="pass-network-control-title"),
            ]),
            dbc.RadioItems(
                id="pass-network-period-selector",
                options=[
                    {"label": "Full Match", "value": "full"},
                    {"label": "1H", "value": "1h"},
                    {"label": "2H", "value": "2h"},
                ],
                value="full", inline=True,
                className="pass-network-radio",
                inputClassName="btn-check",
                labelClassName="pass-network-radio-label",
                labelCheckedClassName="is-active",
            ),
        ], className="pass-network-control-group"),
        html.Div([
            html.Div([
                html.Span("EDGE FILTER", className="match-panel-eyebrow"),
                html.Strong("Minimum connection", className="pass-network-control-title"),
            ]),
            dbc.RadioItems(
                id="pass-network-threshold-selector",
                options=[
                    {"label": "3+", "value": 3},
                    {"label": "5+", "value": 5},
                    {"label": "8+", "value": 8},
                ],
                value=pass_network_metrics.PASS_NETWORK_MIN_CONNECTION,
                inline=True,
                className="pass-network-radio",
                inputClassName="btn-check",
                labelClassName="pass-network-radio-label",
                labelCheckedClassName="is-active",
            ),
        ], className="pass-network-control-group"),
        html.Div([
            html.Span("VIEW RULES", className="match-panel-eyebrow"),
            html.Strong(
                f"Top {pass_network_metrics.PASS_NETWORK_TOP_CONNECTIONS}",
                className="pass-network-control-title",
            ),
            html.Small(
                f"Minimum sample {int(pass_network_metrics.PASS_NETWORK_MIN_MINUTES)} min",
                className="pass-network-control-note",
            ),
        ], className="pass-network-control-group pass-network-control-group--rules"),
    ], className="pass-network-controls")


def _label(player, jersey_map):
    try:
        jersey = str(int(float(jersey_map.get(player))))
    except (TypeError, ValueError):
        jersey = "?"
    return f"#{jersey} · {player}"


def _connections(edges, nodes, color):
    if edges is None or edges.empty:
        return html.Div(
            "No connection reaches the current threshold.",
            className="pass-network-empty",
        )
    jersey_map = nodes.set_index("playerName")["jersey_number"].to_dict()
    maximum = max(int(edges["pass_count"].max()), 1)
    rows = []
    for rank, (_, row) in enumerate(edges.iterrows(), start=1):
        total = int(row["pass_count"])
        rows.append(html.Div([
            html.Span(str(rank), className="pass-connection-rank"),
            html.Div([
                html.Div([
                    html.Span(_label(row["player1"], jersey_map)),
                    html.Span("↔", className="pass-connection-arrow"),
                    html.Span(_label(row["player2"], jersey_map)),
                ], className="pass-connection-pair"),
                html.Div([
                    html.Span(
                        f"{row['player1']} → {row['player2']} · "
                        f"{int(row['player1_to_player2'])}"
                    ),
                    html.Span(
                        f"{row['player2']} → {row['player1']} · "
                        f"{int(row['player2_to_player1'])}"
                    ),
                ], className="pass-connection-directions"),
                html.Div(
                    html.Span(style={
                        "width": f"{total / maximum * 100:.1f}%",
                        "backgroundColor": color,
                    }),
                    className="pass-connection-track",
                ),
            ], className="pass-connection-main"),
            html.Strong(str(total), className="pass-connection-count"),
        ], className="pass-connection-row"))
    return html.Div(rows, className="pass-connection-list")


def _team_panel(team, color, is_away, edges, nodes, summary):
    figure = pass_plotly.plot_pass_network_profile_plotly(
        edges, nodes, team, is_away=is_away,
    )
    qualifying = int(summary.get("qualifying_connections", 0))
    shown = int(summary.get("shown_connections", 0))
    return html.Section([
        html.Div([
            html.Div([
                html.Span(
                    "AWAY TEAM" if is_away else "HOME TEAM",
                    className="match-panel-eyebrow",
                ),
                html.H3(team, className="pass-network-team-name"),
            ]),
            html.Div([
                html.Span(
                    f"{summary.get('eligible_players', 0)} eligible players",
                    className="pass-network-sample-pill",
                ),
                html.Span(
                    f"{summary.get('included_passes', 0)} reliable passes",
                    className="pass-network-sample-pill",
                ),
            ], className="pass-network-sample-pills"),
        ], className="pass-network-team-header"),
        html.Div([
            html.Span("○ Starter"),
            html.Span("◇ Substitute"),
            html.Span("Node size = pass involvement"),
            html.Span("Edge width = connection volume"),
        ], className="pass-network-visual-key"),
        dcc.Graph(
            figure=figure,
            config={"displayModeBar": False, "responsive": True},
            className="pass-network-graph",
        ),
        html.Div([
            html.Span("TOP CONNECTIONS", className="match-panel-eyebrow"),
            html.H4(
                f"{shown} shown" + (f" · {qualifying} qualify" if qualifying != shown else ""),
                className="pass-network-connections-title",
            ),
            html.P(
                "Reliable completed passes in both directions. Hover a pitch edge for the split.",
                className="match-panel-description",
            ),
            _connections(edges, nodes, color),
        ], className="pass-network-connections"),
    ], className="match-panel pass-network-team-panel")


def render_period_view(stored_data_json, period, threshold, home_color, away_color):
    if not stored_data_json:
        return html.Div("No data available.", className="pass-network-empty")

    df_processed = pd.read_json(io.StringIO(stored_data_json["df"]), orient="split")
    match_info = json.loads(stored_data_json.get("match_info", "{}"))
    home_team = match_info.get("hteamName", "Home")
    away_team = match_info.get("ateamName", "Away")
    passes_df = pass_processing.get_passes_df(df_processed.copy())
    threshold = int(threshold or pass_network_metrics.PASS_NETWORK_MIN_CONNECTION)

    kwargs = dict(
        period=period or "full",
        min_minutes=pass_network_metrics.PASS_NETWORK_MIN_MINUTES,
        min_connection=threshold,
        top_n=pass_network_metrics.PASS_NETWORK_TOP_CONNECTIONS,
    )
    h_edges, h_nodes, h_summary = pass_network_metrics.build_pass_network_profile(
        passes_df, df_processed, home_team, **kwargs,
    )
    a_edges, a_nodes, a_summary = pass_network_metrics.build_pass_network_profile(
        passes_df, df_processed, away_team, **kwargs,
    )

    return html.Div([
        html.Div([
            html.Div([
                html.Span("ACTIVE WINDOW", className="match-panel-eyebrow"),
                html.Strong(
                    PERIOD_LABELS.get(str(period or "full").lower(), "Full Match"),
                    className="pass-network-window-value",
                ),
            ]),
            html.Div([
                html.Span(
                    f"≥ {int(pass_network_metrics.PASS_NETWORK_MIN_MINUTES)} min",
                    className="pass-network-rule-pill",
                ),
                html.Span(
                    f"≥ {threshold} passes / connection",
                    className="pass-network-rule-pill",
                ),
                html.Span(
                    f"Top {pass_network_metrics.PASS_NETWORK_TOP_CONNECTIONS}",
                    className="pass-network-rule-pill",
                ),
            ], className="pass-network-rule-pills"),
        ], className="pass-network-active-summary"),
        html.Div([
            _team_panel(home_team, home_color, False, h_edges, h_nodes, h_summary),
            _team_panel(away_team, away_color, True, a_edges, a_nodes, a_summary),
        ], className="pass-network-grid"),
    ], className="pass-network-period-view")
