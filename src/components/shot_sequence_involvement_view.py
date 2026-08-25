from __future__ import annotations

import pandas as pd
import dash_bootstrap_components as dbc
from dash import html


def _jersey_label(
    value,
):
    try:
        if pd.isna(
            value
        ):
            return "?"
    except (
        TypeError,
        ValueError,
    ):
        pass

    try:
        return str(
            int(
                float(
                    value
                )
            )
        )
    except (
        TypeError,
        ValueError,
    ):
        return "?"


def _player_meta(
    df_processed,
):
    if (
        df_processed is None
        or df_processed.empty
        or "playerName"
        not in df_processed.columns
    ):
        return pd.DataFrame()

    return (
        df_processed
        .dropna(
            subset=[
                "playerName"
            ]
        )
        .drop_duplicates(
            "playerName",
            keep="last",
        )
        .set_index(
            "playerName"
        )
    )


def table(
    ranking,
    df_processed,
    home_team_name,
    *,
    hcol,
    acol,
):
    if (
        ranking is None
        or ranking.empty
    ):
        return dbc.Alert(
            "No shot-sequence involvements found.",
            color="secondary",
            className="m-3",
        )

    meta = _player_meta(
        df_processed
    )

    max_involvement = max(
        int(
            ranking[
                "Shot Sequence Involvements"
            ]
            .max()
        ),
        1,
    )

    rows = []

    for (
        rank,
        (
            player_name,
            row,
        ),
    ) in enumerate(
        ranking.iterrows(),
        start=1,
    ):
        team_name = None
        jersey = "?"

        if (
            player_name
            in meta.index
        ):
            if (
                "team_name"
                in meta.columns
            ):
                team_name = meta.at[
                    player_name,
                    "team_name",
                ]

            if (
                "Mapped Jersey Number"
                in meta.columns
            ):
                jersey = (
                    _jersey_label(
                        meta.at[
                            player_name,
                            "Mapped Jersey Number",
                        ]
                    )
                )

        team_color = (
            hcol
            if team_name
            == home_team_name
            else acol
        )

        involvements = int(
            row[
                "Shot Sequence Involvements"
            ]
        )
        shots = int(
            row[
                "Shot Sequence Shots"
            ]
        )
        shot_assists = int(
            row[
                "Shot Sequence Shot Assists"
            ]
        )
        pre_assists = int(
            row[
                "Shot Sequence Pre-Assists"
            ]
        )

        bar_width = (
            involvements
            / max_involvement
            * 100.0
        )

        rows.append(
            html.Div(
                [
                    html.Div(
                        f"{rank:02d}",
                        className=
                            "passing-rank",
                    ),

                    html.Div(
                        [
                            html.Strong(
                                (
                                    f"#{jersey} · "
                                    f"{player_name}"
                                ),
                                style={
                                    "color":
                                        team_color,
                                },
                            ),
                            html.Small(
                                team_name
                                or "",
                                className=
                                    "passing-player-team",
                            ),
                        ],
                        className=
                            "passing-player",
                    ),

                    html.Div(
                        [
                            html.Span(
                                style={
                                    "width":
                                        (
                                            f"{bar_width:.1f}%"
                                        )
                                }
                            ),
                        ],
                        className=
                            "passing-contribution-track",
                    ),

                    html.Div(
                        str(
                            involvements
                        ),
                        className=
                            "passing-total-value",
                    ),

                    html.Div(
                        str(
                            shots
                        ),
                        className=
                            "passing-metric-value",
                    ),

                    html.Div(
                        str(
                            shot_assists
                        ),
                        className=
                            "passing-metric-value",
                    ),

                    html.Div(
                        str(
                            pre_assists
                        ),
                        className=
                            "passing-metric-value",
                    ),
                ],
                className=
                    "passing-player-row",
            )
        )

    header = html.Div(
        [
            html.Div(
                "#"
            ),
            html.Div(
                "Player"
            ),
            html.Div(
                "Shot sequence involvement"
            ),
            html.Div(
                [
                    "Unique",
                    html.I(
                        className=(
                            "fa-solid "
                            "fa-circle-info "
                            "ms-1"
                        ),
                        title=(
                            "Number of different shot sequences "
                            "in which the player was involved."
                        ),
                    ),
                ]
            ),
            html.Div(
                "Shots"
            ),
            html.Div(
                [
                    "Shot assists",
                    html.I(
                        className=(
                            "fa-solid "
                            "fa-circle-info "
                            "ms-1"
                        ),
                        title=(
                            "Completed pass directly "
                            "creating the shot."
                        ),
                    ),
                ]
            ),
            html.Div(
                [
                    "Pre-assists",
                    html.I(
                        className=(
                            "fa-solid "
                            "fa-circle-info "
                            "ms-1"
                        ),
                        title=(
                            "Reliable completed pass "
                            "received by the player who "
                            "then makes the shot assist."
                        ),
                    ),
                ]
            ),
        ],
        className=
            "passing-player-header",
    )

    ranking_table = html.Div(
        [
            header,
            *rows,
        ],
        className=(
            "passing-player-ranking "
            "shot-sequence-player-ranking"
        ),
    )

    return html.Section(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "SHOOTING PROFILE",
                                className=
                                    "match-panel-eyebrow",
                            ),
                            html.H3(
                                "Shot sequence involvement",
                                className=
                                    "match-panel-title",
                            ),
                            html.P(
                                (
                                    "Compare how players contribute "
                                    "to attacking sequences that end "
                                    "in a shot."
                                ),
                                className=
                                    "match-panel-description",
                            ),
                        ]
                    ),

                    html.Div(
                        (
                            "Ranked by unique "
                            "shot-sequence involvements"
                        ),
                        className=
                            "shot-sequence-ranking-badge",
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
                            "Each shot sequence counts once per "
                            "player. Shots, shot assists and pre-assists "
                            "are shown separately; no weighted score "
                            "is used."
                        )
                    ),
                ],
                className=
                    "match-analysis-note",
            ),

            ranking_table,
        ],
        className=(
            "match-panel "
            "shot-sequence-ranking-panel"
        ),
    )
