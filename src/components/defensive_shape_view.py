from __future__ import annotations

from dash import dcc, html
import dash_bootstrap_components as dbc


PERIOD_OPTIONS = [
    {
        "label": "Full Match",
        "value": "full",
    },
    {
        "label": "1H",
        "value": "1h",
    },
    {
        "label": "2H",
        "value": "2h",
    },
]

MODE_OPTIONS = [
    {
        "label": "Density",
        "value": "density",
    },
    {
        "label": "Shape",
        "value": "shape",
    },
]


def controls():
    return html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "TIME WINDOW",
                        className=
                            "match-panel-eyebrow",
                    ),
                    html.Strong(
                        "Defensive sample",
                        className=
                            "defensive-shape-control-title",
                    ),
                    dbc.RadioItems(
                        id=
                            "defensive-shape-period",
                        options=
                            PERIOD_OPTIONS,
                        value="full",
                        inline=True,
                        className=
                            "defensive-shape-selector",
                        inputClassName=
                            "btn-check",
                        labelClassName=
                            "defensive-shape-option",
                        labelCheckedClassName=
                            "defensive-shape-option--active",
                    ),
                ],
                className=
                    "defensive-shape-control-group",
            ),
            html.Div(
                [
                    html.Span(
                        "VIEW MODE",
                        className=
                            "match-panel-eyebrow",
                    ),
                    html.Strong(
                        "Defensive Shape",
                        className=
                            "defensive-shape-control-title",
                    ),
                    dbc.RadioItems(
                        id=
                            "defensive-shape-mode",
                        options=
                            MODE_OPTIONS,
                        value="density",
                        inline=True,
                        className=
                            "defensive-shape-selector",
                        inputClassName=
                            "btn-check",
                        labelClassName=
                            "defensive-shape-option",
                        labelCheckedClassName=
                            "defensive-shape-option--active",
                    ),
                ],
                className=
                    "defensive-shape-control-group",
            ),
        ],
        className=
            "defensive-shape-controls",
    )


def workspace():
    return html.Div(
        [
            controls(),
            html.Div(
                [
                    html.I(
                        className=
                            "fa-solid fa-circle-info",
                    ),
                    html.Span(
                        (
                            "Defensive Shape is event-derived, not tracking data. "
                            "Density uses all defensive actions in the selected period. "
                            "Shape never builds a whole-match hull: it uses 15-minute "
                            "stable windows and displays one real representative "
                            "window closest to the period median structure."
                        )
                    ),
                ],
                className=
                    "match-analysis-note",
            ),
            dcc.Loading(
                type="circle",
                children=
                    html.Div(
                        id=
                            "defensive-shape-content",
                    ),
            ),
        ],
        className=
            "defensive-shape-workspace",
    )


def _kpi(
    label,
    value,
    caption,
):
    return html.Div(
        [
            html.Span(
                label,
                className=
                    "defensive-shape-kpi-label",
            ),
            html.Strong(
                value,
                className=
                    "defensive-shape-kpi-value",
            ),
            html.Small(
                caption,
                className=
                    "defensive-shape-kpi-caption",
            ),
        ],
        className=
            "defensive-shape-kpi",
    )


def team_panel(
    team_name,
    figure,
    profile,
    side,
    mode,
):
    block_height = profile.get(
        "block_height_m"
    )
    width = profile.get(
        "width_m"
    )
    compactness = profile.get(
        "compactness_m"
    )
    action_count = int(
        profile.get(
            "action_count",
            0,
        )
    )
    snapshot_count = int(
        profile.get(
            "snapshot_count",
            0,
        )
    )

    block_value = (
        f"{block_height:.1f} m"
        if block_height
        is not None
        else "—"
    )
    width_value = (
        f"{width:.1f} m"
        if width is not None
        else "—"
    )
    compact_value = (
        f"{compactness:.1f} m"
        if compactness
        is not None
        else "—"
    )

    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                (
                                    "HOME TEAM"
                                    if side
                                    == "home"
                                    else "AWAY TEAM"
                                ),
                                className=
                                    "match-panel-eyebrow",
                            ),
                            html.H3(
                                team_name,
                                className=
                                    "match-panel-title",
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Span(
                                (
                                    f"{action_count} "
                                    "defensive actions"
                                ),
                                className=
                                    "match-panel-chip",
                            ),
                            html.Span(
                                (
                                    f"{snapshot_count} "
                                    "stable windows"
                                ),
                                className=
                                    "match-panel-chip",
                            ),
                        ],
                        className=
                            "defensive-shape-panel-chips",
                    ),
                ],
                className=
                    "match-panel-header",
            ),
            html.Div(
                [
                    _kpi(
                        "ACTION DENSITY",
                        str(
                            action_count
                        ),
                        (
                            "Selected-period "
                            "defensive events"
                        ),
                    ),
                    _kpi(
                        "BLOCK HEIGHT",
                        block_value,
                        (
                            "Median coherent-window "
                            "height from own goal"
                        ),
                    ),
                    _kpi(
                        "WIDTH",
                        width_value,
                        (
                            "Median coherent-window "
                            "lateral span"
                        ),
                    ),
                    _kpi(
                        "COMPACTNESS",
                        compact_value,
                        (
                            "Typical player spread from block centre · lower = tighter"
                        ),
                    ),
                ],
                className=
                    "defensive-shape-kpis",
            ),
            dcc.Graph(
                figure=figure,
                config={
                    "displayModeBar":
                        False,
                    "responsive":
                        True,
                },
                className=
                    "defensive-shape-graph",
            ),
        ],
        className=(
            "match-panel "
            "defensive-shape-panel "
            f"defensive-shape-panel--{mode}"
        ),
    )


def comparison(
    home_panel,
    away_panel,
    period_label,
    mode_label,
):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "ACTIVE VIEW",
                                className=
                                    "match-panel-eyebrow",
                            ),
                            html.Strong(
                                (
                                    f"{period_label} "
                                    f"· {mode_label}"
                                ),
                                className=
                                    "defensive-shape-active-view",
                            ),
                        ]
                    ),
                    html.Span(
                        (
                            "Shape = representative stable-lineup window · up to 15 min"
                        ),
                        className=
                            "match-panel-chip",
                    ),
                ],
                className=
                    "defensive-shape-active-bar",
            ),
            html.Div(
                [
                    home_panel,
                    away_panel,
                ],
                className=
                    "defensive-shape-grid",
            ),
        ]
    )
