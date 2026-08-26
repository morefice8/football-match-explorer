from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html


def _info_icon(
    tooltip,
):
    return html.I(
        className=(
            "fa-solid "
            "fa-circle-info "
            "ms-1"
        ),
        title=tooltip,
    )


def panel(
    ranking,
    *,
    home_team_name,
    hcol,
    acol,
):
    if (
        ranking is None
        or ranking.empty
    ):
        return dbc.Alert(
            "No defensive actions found.",
            color="secondary",
            className="m-3",
        )

    max_unique = max(
        int(
            ranking[
                "unique"
            ].max()
        ),
        1,
    )

    rows = []

    for rank, row in enumerate(
        ranking.to_dict(
            orient="records"
        ),
        start=1,
    ):
        team_name = (
            row.get(
                "team_name"
            )
            or ""
        )

        team_color = (
            hcol
            if team_name
            == home_team_name
            else acol
        )

        unique = int(
            row.get(
                "unique",
                0,
            )
            or 0
        )

        bar_width = (
            unique
            / max_unique
            * 100.0
        )

        tackle_value = (
            f"{int(row.get('tackles_won', 0) or 0)}"
            f"/{int(row.get('tackles_attempted', 0) or 0)}"
        )

        rows.append(
            html.Div(
                [
                    html.Div(
                        f"{rank:02d}",
                        className=
                            "defensive-rank",
                    ),
                    html.Div(
                        [
                            html.Strong(
                                (
                                    f"#{row.get('jersey', '?')} · "
                                    f"{row.get('playerName', '')}"
                                ),
                                style={
                                    "color":
                                        team_color,
                                },
                            ),
                            html.Small(
                                team_name,
                                className=
                                    "defensive-player-team",
                            ),
                        ],
                        className=
                            "defensive-player",
                    ),
                    html.Div(
                        [
                            html.Span(
                                style={
                                    "width":
                                        f"{bar_width:.1f}%"
                                }
                            ),
                        ],
                        className=
                            "defensive-contribution-track",
                    ),
                    html.Div(
                        str(unique),
                        className=
                            "defensive-total-value",
                    ),
                    html.Div(
                        tackle_value,
                        className=
                            "defensive-metric-value",
                    ),
                    html.Div(
                        str(
                            int(
                                row.get(
                                    "interceptions",
                                    0,
                                )
                                or 0
                            )
                        ),
                        className=
                            "defensive-metric-value",
                    ),
                    html.Div(
                        str(
                            int(
                                row.get(
                                    "recoveries",
                                    0,
                                )
                                or 0
                            )
                        ),
                        className=
                            "defensive-metric-value",
                    ),
                    html.Div(
                        str(
                            int(
                                row.get(
                                    "clearances",
                                    0,
                                )
                                or 0
                            )
                        ),
                        className=
                            "defensive-metric-value",
                    ),
                    html.Div(
                        str(
                            int(
                                row.get(
                                    "blocks",
                                    0,
                                )
                                or 0
                            )
                        ),
                        className=
                            "defensive-metric-value",
                    ),
                    html.Div(
                        str(
                            int(
                                row.get(
                                    "fouls",
                                    0,
                                )
                                or 0
                            )
                        ),
                        className=(
                            "defensive-metric-value "
                            "defensive-foul-value"
                        ),
                    ),
                ],
                className=
                    "defensive-player-row",
            )
        )

    header = html.Div(
        [
            html.Div("#"),
            html.Div("Player"),
            html.Div(
                "Defensive involvement"
            ),
            html.Div(
                [
                    "Unique",
                    _info_icon(
                        "Different positive defensive events: "
                        "successful tackles, interceptions, recoveries, "
                        "clearances and blocks. Each event counts once."
                    ),
                ]
            ),
            html.Div(
                [
                    "Tackles",
                    _info_icon(
                        "Successful / attempted tackles. "
                        "Failed tackle attempts do not increase Unique."
                    ),
                ]
            ),
            html.Div(
                "Interceptions"
            ),
            html.Div(
                "Recoveries"
            ),
            html.Div(
                "Clearances"
            ),
            html.Div(
                [
                    "Blocks",
                    _info_icon(
                        "Blocked passes plus outfield shot blocks."
                    ),
                ]
            ),
            html.Div(
                [
                    "Fouls",
                    _info_icon(
                        "Fouls committed only. Fouls suffered are excluded "
                        "and fouls never increase the Unique ranking."
                    ),
                ]
            ),
        ],
        className=
            "defensive-player-header",
    )

    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "DEFENSIVE PROFILE",
                                className=
                                    "match-panel-eyebrow",
                            ),
                            html.H3(
                                "Defensive contributions",
                                className=
                                    "match-panel-title",
                            ),
                            html.P(
                                (
                                    "Compare the players most involved in "
                                    "positive defensive actions and see how "
                                    "that contribution was built."
                                ),
                                className=
                                    "match-panel-description",
                            ),
                        ]
                    ),
                    html.Div(
                        "Ranked by unique defensive actions",
                        className=
                            "defensive-ranking-badge",
                    ),
                ],
                className=
                    "match-panel-header",
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
                            "Successful tackles, interceptions, recoveries, "
                            "clearances and blocks contribute to the ranking. "
                            "Tackles are successful/attempted. Blocks include "
                            "blocked passes and defender shot blocks. Fouls "
                            "mean fouls committed only and never improve rank."
                        )
                    ),
                ],
                className=
                    "match-analysis-note",
            ),
            html.Div(
                [
                    header,
                    *rows,
                ],
                className=
                    "defensive-player-ranking",
            ),
        ],
        className=(
            "match-panel "
            "defensive-ranking-panel"
        ),
    )
