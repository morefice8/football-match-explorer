from __future__ import annotations

from dash import (
    dcc,
    html,
)


def _metric(
    label,
    value,
    caption,
):
    return html.Div(
        [
            html.Span(
                label,
                className=
                    "shot-contributor-kpi-label",
            ),
            html.Strong(
                str(value),
                className=
                    "shot-contributor-kpi-value",
            ),
            html.Small(
                caption,
                className=
                    "shot-contributor-kpi-caption",
            ),
        ],
        className=
            "shot-contributor-kpi",
    )


def player_panel(
    *,
    selected_player,
    jersey,
    team_name,
    team_color,
    summary,
    figure,
):
    summary = summary or {}

    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "ATTACKING RECEPTION PROFILE",
                                className=
                                    "match-panel-eyebrow",
                            ),
                            html.H3(
                                (
                                    f"#{jersey} · "
                                    f"{selected_player}"
                                ),
                                className=
                                    "match-panel-title",
                                style={
                                    "color":
                                        team_color,
                                },
                            ),
                            html.P(
                                (
                                    "Where the selected player "
                                    "received completed passes. "
                                    "Gold links highlight receptions "
                                    "inside shot-ending sequences in "
                                    "which the player was directly "
                                    "involved."
                                ),
                                className=
                                    "match-panel-description",
                            ),
                        ]
                    ),
                    html.Div(
                        team_name,
                        className=
                            "shot-contributor-team-pill",
                    ),
                ],
                className=
                    "match-panel-header",
            ),

            html.Div(
                [
                    _metric(
                        "RECEPTIONS",
                        summary.get(
                            "receptions",
                            0,
                        ),
                        (
                            "Reliable completed "
                            "passes received"
                        ),
                    ),
                    _metric(
                        "FINAL THIRD",
                        summary.get(
                            "final_third",
                            0,
                        ),
                        (
                            "Receptions ending "
                            "in the final third"
                        ),
                    ),
                    _metric(
                        "BOX",
                        summary.get(
                            "box",
                            0,
                        ),
                        (
                            "Receptions inside "
                            "the penalty area"
                        ),
                    ),
                    _metric(
                        "PASSERS",
                        summary.get(
                            "passers",
                            0,
                        ),
                        (
                            "Different teammates "
                            "who found the player"
                        ),
                    ),
                ],
                className=
                    "shot-contributor-kpis",
            ),

            html.Div(
                [
                    html.I(
                        className=(
                            "fa-solid "
                            "fa-circle-info"
                        )
                    ),
                    html.Span(
                        (
                            "Hollow points show pass origins; filled points show receptions. The numbered marker is the median reception position, while the shaded area represents the typical reception zone. Gold links identify shot-linked receptions."
                        )
                    ),
                ],
                className=
                    "match-analysis-note",
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
                    "shot-contributor-graph",
            ),
        ],
        className=(
            "match-panel "
            "shot-contributor-panel"
        ),
    )


def selector_layout(
    *,
    team_type,
    team_name,
    dropdown_options,
    selected_player,
    initial_panel,
):
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                (
                                    "HOME TEAM"
                                    if team_type
                                    == "home"
                                    else "AWAY TEAM"
                                ),
                                className=
                                    "match-panel-eyebrow",
                            ),
                            html.H3(
                                "Reception profile",
                                className=
                                    "match-panel-title",
                            ),
                            html.P(
                                (
                                    "Select a high-impact shooting "
                                    "contributor to inspect where "
                                    "and how they received the ball."
                                ),
                                className=
                                    "match-panel-description",
                            ),
                        ]
                    ),
                    html.Div(
                        team_name,
                        className=
                            "shot-contributor-team-pill",
                    ),
                ],
                className=(
                    "match-panel "
                    "match-panel-header "
                    "shot-contributor-selector-header"
                ),
            ),

            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Player",
                                className=
                                    "shot-contributor-select-label",
                            ),
                            dcc.Dropdown(
                                id=(
                                    f"{team_type}-"
                                    "shot-contributor-dropdown"
                                ),
                                options=
                                    dropdown_options,
                                value=
                                    selected_player,
                                clearable=False,
                                searchable=True,
                                placeholder=
                                    "Select a player...",
                                className=
                                    "shot-contributor-select",
                            ),
                        ],
                        className=
                            "shot-contributor-selector",
                        style={
                            "maxWidth":
                                "820px",
                            "margin":
                                "0 auto",
                            "width":
                                "100%",
                        },
                    ),
                ]
            ),

            html.Div(
                initial_panel,
                id=(
                    "shot-contributor-map-"
                    f"container-{team_type}"
                ),
            ),
        ],
        className=
            "shot-contributor-layout",
    )
