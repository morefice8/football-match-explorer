from __future__ import annotations

from dash import dcc, html


def _metric(
    label,
    value,
    caption,
):
    return html.Div(
        [
            html.Span(
                label,
                className="defender-action-kpi-label",
            ),
            html.Strong(
                str(value),
                className="defender-action-kpi-value",
            ),
            html.Small(
                caption,
                className="defender-action-kpi-caption",
            ),
        ],
        className="defender-action-kpi",
    )


def player_panel(
    *,
    selected_player,
    jersey,
    team_name,
    team_color,
    profile,
    figure,
):
    profile = profile or {}

    tackles = (
        f"{int(profile.get('tackles_won', 0) or 0)}"
        f"/{int(profile.get('tackles_attempted', 0) or 0)}"
    )

    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "DEFENSIVE ACTION PROFILE",
                                className="match-panel-eyebrow",
                            ),
                            html.H3(
                                f"#{jersey} · {selected_player}",
                                className="match-panel-title",
                                style={"color": team_color},
                            ),
                            html.P(
                                (
                                    "Where the selected player intervened out "
                                    "of possession and which defensive actions "
                                    "they produced."
                                ),
                                className="match-panel-description",
                            ),
                        ]
                    ),
                    html.Div(
                        team_name,
                        className="shot-contributor-team-pill",
                    ),
                ],
                className="match-panel-header",
            ),
            html.Div(
                [
                    _metric(
                        "TACKLES",
                        tackles,
                        "Successful / attempted",
                    ),
                    _metric(
                        "INTERCEPTIONS",
                        int(profile.get("interceptions", 0) or 0),
                        "Passing lanes cut out",
                    ),
                    _metric(
                        "RECOVERIES",
                        int(profile.get("recoveries", 0) or 0),
                        "Loose balls regained",
                    ),
                    _metric(
                        "CLEARANCES",
                        int(profile.get("clearances", 0) or 0),
                        "Danger cleared",
                    ),
                    _metric(
                        "BLOCKS",
                        int(profile.get("blocks", 0) or 0),
                        "Passes and shots blocked",
                    ),
                    _metric(
                        "FOULS",
                        int(profile.get("fouls", 0) or 0),
                        "Fouls committed",
                    ),
                ],
                className="defender-action-kpis",
            ),
            html.Div(
                [
                    html.I(
                        className="fa-solid fa-circle-info"
                    ),
                    html.Span(
                        (
                            "Marker shape identifies the action type. Filled "
                            "tackle markers are won tackles; hollow tackle "
                            "markers are unsuccessful attempts. The shaded "
                            "ellipse shows the typical defensive intervention "
                            "area and the numbered marker its median position. "
                            "Fouls mean fouls committed only."
                        )
                    ),
                ],
                className="match-analysis-note",
            ),
            dcc.Graph(
                figure=figure,
                config={
                    "displayModeBar": False,
                    "responsive": True,
                },
                className="defender-action-graph",
            ),
        ],
        className="match-panel defender-action-panel",
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
                                    if team_type == "home"
                                    else "AWAY TEAM"
                                ),
                                className="match-panel-eyebrow",
                            ),
                            html.H3(
                                "Defensive action map",
                                className="match-panel-title",
                            ),
                            html.P(
                                (
                                    "Select a defender to inspect where their "
                                    "tackles, interceptions, recoveries, "
                                    "clearances, blocks and fouls occurred."
                                ),
                                className="match-panel-description",
                            ),
                        ]
                    ),
                    html.Div(
                        team_name,
                        className="shot-contributor-team-pill",
                    ),
                ],
                className=(
                    "match-panel match-panel-header "
                    "shot-contributor-selector-header"
                ),
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Player",
                                className="shot-contributor-select-label",
                            ),
                            dcc.Dropdown(
                                id=f"{team_type}-defender-dropdown",
                                options=dropdown_options,
                                value=selected_player,
                                clearable=False,
                                searchable=True,
                                placeholder="Select a player...",
                                className="shot-contributor-select",
                            ),
                        ],
                        className="shot-contributor-selector",
                        style={
                            "maxWidth": "820px",
                            "margin": "0 auto",
                            "width": "100%",
                        },
                    ),
                ]
            ),
            html.Div(
                initial_panel,
                id=f"defender-map-container-{team_type}",
            ),
        ],
        className="shot-contributor-layout defender-action-layout",
    )
