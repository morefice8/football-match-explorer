"""PLOT-05 Progressive Passes Summary / All attempts UI."""

from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.components.match_graph_shell import match_graph_panel
from src.metrics import pass_metrics
from src.visualization import pass_plotly


VIEW_SUMMARY = "summary"
VIEW_ALL_ATTEMPTS = "all_attempts"


def controls():
    return html.Section([
        html.Div([
            html.Div([
                html.Span(
                    "VIEW MODE",
                    className="match-panel-eyebrow",
                ),
                html.Strong(
                    "Progressive pass detail",
                    className="progressive-summary-control-title",
                ),
                html.Small(
                    (
                        "Summary is the default analytical view. "
                        "Individual lines appear only on request."
                    ),
                    className="progressive-summary-control-note",
                ),
            ]),
            dbc.RadioItems(
                id="progressive-view-toggle",
                options=[
                    {
                        "label": "Summary",
                        "value": VIEW_SUMMARY,
                    },
                    {
                        "label": "All attempts",
                        "value": VIEW_ALL_ATTEMPTS,
                    },
                ],
                value=VIEW_SUMMARY,
                inline=True,
                className="progressive-summary-toggle",
                inputClassName="btn-check",
                labelClassName="progressive-summary-toggle-label",
                labelCheckedClassName="is-active",
            ),
        ], className="progressive-summary-control-row"),
    ], className="match-panel progressive-summary-controls")


def _metric_card(label, value, detail):
    return html.Div([
        html.Span(
            label,
            className="progressive-summary-kpi-label",
        ),
        html.Strong(
            value,
            className="progressive-summary-kpi-value",
        ),
        html.Small(
            detail,
            className="progressive-summary-kpi-detail",
        ),
    ], className="progressive-summary-kpi")


def _channel_profile(summary, team_color):
    attempted = max(
        int(summary.get("attempted", 0)),
        1,
    )
    rows = []

    for channel in ("Left", "Central", "Right"):
        count = int(
            summary.get(
                "channel_counts",
                {},
            ).get(channel, 0)
        )
        share = (
            count / attempted * 100.0
        )

        rows.append(
            html.Div([
                html.Div([
                    html.Span(channel),
                    html.Strong(
                        f"{count} · {share:.0f}%"
                    ),
                ], className="progressive-channel-head"),
                html.Div(
                    html.Span(
                        style={
                            "width": f"{share:.1f}%",
                            "backgroundColor": team_color,
                        }
                    ),
                    className="progressive-summary-channel-track",
                ),
            ], className="progressive-summary-channel-row")
        )

    return html.Div(
        rows,
        className="progressive-summary-channels",
    )


def _top_progressor(team_passes):
    ranking = (
        pass_metrics
        .progressive_pass_player_summary(
            team_passes,
            limit=1,
        )
    )

    if ranking.empty:
        return html.Div(
            "No qualifying progressor.",
            className="progressive-summary-empty",
        )

    top = ranking.iloc[0]

    return html.Div([
        html.Div([
            html.Span(
                "TOP PROGRESSOR",
                className="match-panel-eyebrow",
            ),
            html.Strong(
                str(top["Player"]),
                className="progressive-top-player",
            ),
        ]),
        html.Div([
            html.Div([
                html.Span("Completed"),
                html.Strong(
                    f"{int(top['Successful'])} / {int(top['Attempted'])}"
                ),
            ]),
            html.Div([
                html.Span("Completion"),
                html.Strong(
                    f"{int(top['Completion %'])}%"
                ),
            ]),
            html.Div([
                html.Span("Progression"),
                html.Strong(
                    f"{int(top['Progression m'])} m"
                ),
            ]),
        ], className="progressive-top-stats"),
    ], className="progressive-top-progressor")


def _sidebar(team_passes, summary, team_color):
    kpis = html.Div([
        _metric_card(
            "Completed / attempted",
            f"{summary['successful']} / {summary['attempted']}",
            "Open-play progressive passes",
        ),
        _metric_card(
            "Completion",
            f"{summary['completion_pct']:.0f}%",
            "Completed attempts",
        ),
        _metric_card(
            "Total progression",
            f"{summary['total_progression_m']:.0f} m",
            "Completed passes only",
        ),
        _metric_card(
            "Average gain",
            f"{summary['average_progression_m']:.1f} m",
            "Per completed progressive pass",
        ),
    ], className="progressive-summary-kpi-grid")

    return html.Div([
        kpis,
        html.Div([
            html.Div([
                html.Span(
                    "CHANNEL PROFILE",
                    className="match-panel-eyebrow",
                ),
                html.Strong(
                    "Where progression starts",
                    className="progressive-summary-section-title",
                ),
            ]),
            _channel_profile(
                summary,
                team_color,
            ),
        ], className="progressive-summary-section"),
        _top_progressor(team_passes),
    ], className="progressive-summary-sidebar")


def team_panel(
    team_passes,
    team_name,
    team_color,
    *,
    is_away=False,
    view_mode=VIEW_SUMMARY,
):
    summary = (
        pass_metrics
        .progressive_pass_summary(
            team_passes
        )
    )

    if view_mode == VIEW_ALL_ATTEMPTS:
        figure = (
            pass_plotly
            .plot_progressive_passes_plotly(
                team_passes,
                team_name,
                team_color,
                is_away,
            )
        )
    else:
        figure = (
            pass_plotly
            .plot_progressive_pass_summary_plotly(
                team_passes,
                team_name,
                team_color,
                is_away,
            )
        )

    graph = dcc.Graph(
        figure=figure,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        className="progressive-summary-map-graph",
    )

    sidebar = _sidebar(
        team_passes,
        summary,
        team_color,
    )

    empty_message = (
        "No open-play progressive pass attempts for this team."
        if summary["attempted"] == 0
        else None
    )

    return match_graph_panel(
        team_name=team_name,
        is_away=is_away,
        sample_size=f"n = {summary['attempted']} attempts",
        graph=graph,
        sidebar=sidebar,
        empty_message=empty_message,
    )


def panels(
    prog_passes,
    home_team,
    away_team,
    home_color,
    away_color,
    *,
    view_mode=VIEW_SUMMARY,
):
    data = (
        prog_passes.copy()
        if prog_passes is not None
        else pd.DataFrame()
    )

    if "team_name" in data.columns:
        home_passes = data[
            data["team_name"].eq(home_team)
        ].copy()
        away_passes = data[
            data["team_name"].eq(away_team)
        ].copy()
    else:
        home_passes = data.iloc[0:0].copy()
        away_passes = data.iloc[0:0].copy()

    return [
        team_panel(
            home_passes,
            home_team,
            home_color,
            is_away=False,
            view_mode=view_mode,
        ),
        team_panel(
            away_passes,
            away_team,
            away_color,
            is_away=True,
            view_mode=view_mode,
        ),
    ]

def workspace(
    shell_component,
):
    """
    Stable PLOT-05 workspace contract.

    The mode selector deliberately lives outside the Graph Shell so it cannot
    be swallowed by panel composition or dynamic panel replacement.
    """
    return html.Div([
        controls(),
        shell_component,
    ], className=(
        "progressive-summary-analysis "
        "progressive-summary-workspace"
    ))
