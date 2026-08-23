from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

from src.components.match_graph_shell import match_graph_panel
from src.visualization import pass_plotly


VIEW_DENSITY = "density"
VIEW_GRID = "grid"


def controls():
    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "VIEW MODE",
                                className="match-panel-eyebrow",
                            ),
                            html.Strong(
                                "Pass origin structure",
                                className="pass-location-control-title",
                            ),
                            html.Small(
                                (
                                    "Both views use each team's pass origins "
                                    "as a 100% denominator. Grid hover also "
                                    "shows the raw pass count."
                                ),
                                className="pass-location-control-note",
                            ),
                        ]
                    ),
                    dbc.RadioItems(
                        id="pass-location-view-toggle",
                        options=[
                            {
                                "label": "Density",
                                "value": VIEW_DENSITY,
                            },
                            {
                                "label": "Grid",
                                "value": VIEW_GRID,
                            },
                        ],
                        value=VIEW_DENSITY,
                        inline=True,
                        className="pass-location-view-toggle",
                        inputClassName="btn-check",
                        labelClassName="pass-location-toggle-label",
                        labelCheckedClassName="is-active",
                    ),
                ],
                className="pass-location-control-row",
            ),
        ],
        className="match-panel pass-location-controls",
    )


def workspace(shell_component):
    return html.Div(
        [
            controls(),
            shell_component,
        ],
        className="pass-location-workspace",
    )


def _team_panel(
    team_passes,
    team_name,
    *,
    is_away,
    view_mode,
    shared_scales,
):
    team_passes = (
        team_passes.copy()
        if team_passes is not None
        else pd.DataFrame()
    )

    view_mode = (
        view_mode
        if view_mode in {VIEW_DENSITY, VIEW_GRID}
        else VIEW_DENSITY
    )

    if view_mode == VIEW_GRID:
        figure = pass_plotly.plot_pass_heatmap_plotly(
            team_passes,
            team_name,
            is_away=is_away,
            shared_zmax=shared_scales["grid"],
        )
    else:
        figure = pass_plotly.plot_pass_density_plotly(
            team_passes,
            team_name,
            is_away=is_away,
            shared_zmax=shared_scales["density"],
        )

    graph = dcc.Graph(
        figure=figure,
        config={
            "displayModeBar": False,
            "responsive": True,
        },
        className="pass-location-graph",
    )

    empty_message = (
        "No valid pass-origin coordinates for this team."
        if len(team_passes) == 0
        else None
    )

    return match_graph_panel(
        team_name=team_name,
        is_away=is_away,
        sample_size=f"n = {len(team_passes)} pass attempts",
        graph=graph,
        empty_message=empty_message,
    )


def panels(
    passes_df,
    home_team,
    away_team,
    *,
    view_mode=VIEW_DENSITY,
):
    passes_df = (
        passes_df.copy()
        if passes_df is not None
        else pd.DataFrame()
    )

    if (
        passes_df.empty
        or "team_name" not in passes_df.columns
    ):
        home_passes = pd.DataFrame()
        away_passes = pd.DataFrame()
    else:
        home_passes = passes_df[
            passes_df["team_name"].eq(home_team)
        ].copy()

        away_passes = passes_df[
            passes_df["team_name"].eq(away_team)
        ].copy()

    shared_scales = (
        pass_plotly.pass_location_shared_scales(
            home_passes,
            away_passes,
        )
    )

    return html.Div(
        [
            _team_panel(
                home_passes,
                home_team,
                is_away=False,
                view_mode=view_mode,
                shared_scales=shared_scales,
            ),
            _team_panel(
                away_passes,
                away_team,
                is_away=True,
                view_mode=view_mode,
                shared_scales=shared_scales,
            ),
        ],
        className="pass-location-panel-grid",
    )
