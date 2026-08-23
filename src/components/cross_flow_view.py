from __future__ import annotations

from dash import html


def _pct(value):
    try:
        return f"{float(value):.0f}%"
    except (TypeError, ValueError):
        return "0%"


def flow_panel(
    summary,
    routes,
    team_color,
):
    summary = summary or {}

    if (
        routes is None
        or routes.empty
    ):
        return html.Div(
            "No cross-flow data available.",
            className="cross-flow-empty",
        )

    max_count = max(
        int(
            routes[
                "Crosses"
            ].max()
        ),
        1,
    )

    kpis = html.Div(
        [
            html.Div(
                [
                    html.Span(
                        "VOLUME",
                        className="plot08-cross-flow-kpi-label",
                    ),
                    html.Strong(
                        str(
                            summary.get(
                                "total_crosses",
                                0,
                            )
                        ),
                        className="plot08-cross-flow-kpi-value",
                    ),
                    html.Small(
                        "Currently filtered crosses",
                        className="plot08-cross-flow-kpi-detail",
                    ),
                ],
                className="plot08-cross-flow-kpi",
            ),
            html.Div(
                [
                    html.Span(
                        "RETENTION RATE",
                        className="plot08-cross-flow-kpi-label",
                    ),
                    html.Strong(
                        _pct(
                            summary.get(
                                "retention_pct",
                                0,
                            )
                        ),
                        className="plot08-cross-flow-kpi-value",
                    ),
                    html.Small(
                        (
                            f"{summary.get('retained_crosses', 0)} "
                            "retained deliveries / second balls"
                        ),
                        className="plot08-cross-flow-kpi-detail",
                    ),
                ],
                className="plot08-cross-flow-kpi",
            ),
            html.Div(
                [
                    html.Span(
                        "SHOT RATE",
                        className="plot08-cross-flow-kpi-label",
                    ),
                    html.Strong(
                        _pct(
                            summary.get(
                                "shot_rate_pct",
                                0,
                            )
                        ),
                        className="plot08-cross-flow-kpi-value",
                    ),
                    html.Small(
                        (
                            f"{summary.get('shot_crosses', 0)} "
                            "crosses generated a shot"
                        ),
                        className="plot08-cross-flow-kpi-detail",
                    ),
                ],
                className="plot08-cross-flow-kpi",
            ),
            html.Div(
                [
                    html.Span(
                        "TOP CROSSER",
                        className="plot08-cross-flow-kpi-label",
                    ),
                    html.Strong(
                        str(
                            summary.get(
                                "top_crosser",
                                "N/A",
                            )
                        ),
                        className=(
                            "plot08-cross-flow-kpi-value "
                            "plot08-cross-flow-kpi-value--name"
                        ),
                    ),
                    html.Small(
                        (
                            f"{summary.get('top_crosser_count', 0)} "
                            "crosses in current sample"
                        ),
                        className="plot08-cross-flow-kpi-detail",
                    ),
                ],
                className="plot08-cross-flow-kpi",
            ),
        ],
        className="plot08-cross-flow-kpi-grid",
    )

    route_rows = []

    for rank, (_, row) in enumerate(
        routes.iterrows(),
        start=1,
    ):
        count = int(
            row["Crosses"]
        )

        share = float(
            row["Share %"]
        )

        retention = float(
            row["Retention %"]
        )

        shot_rate = float(
            row["Shot Rate %"]
        )

        bar_width = (
            count
            / max_count
            * 100.0
        )

        origin = str(
            row[
                "Origin Zone"
            ]
        )

        destination = str(
            row[
                "Destination Zone"
            ]
        )

        route_rows.append(
            html.Button(
                [
                    html.Span(
                        str(rank),
                        className="plot08-cross-flow-rank",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Span(
                                        origin,
                                        className="plot08-cross-flow-zone",
                                    ),
                                    html.I(
                                        className=(
                                            "fa-solid "
                                            "fa-arrow-right "
                                            "plot08-cross-flow-arrow"
                                        )
                                    ),
                                    html.Span(
                                        destination,
                                        className=(
                                            "plot08-cross-flow-zone "
                                            "plot08-cross-flow-zone--destination"
                                        ),
                                    ),
                                ],
                                className="plot08-cross-flow-route",
                            ),
                            html.Div(
                                html.Span(
                                    style={
                                        "width":
                                            f"{bar_width:.1f}%",
                                        "backgroundColor":
                                            team_color,
                                    },
                                ),
                                className="plot08-cross-flow-track",
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        (
                                            f"{count} crosses · "
                                            f"{share:.0f}% of sample"
                                        )
                                    ),
                                    html.Span("·"),
                                    html.Span(
                                        f"Retention {retention:.0f}%"
                                    ),
                                    html.Span("·"),
                                    html.Span(
                                        f"Shot rate {shot_rate:.0f}%"
                                    ),
                                ],
                                className="plot08-cross-flow-detail",
                            ),
                        ],
                        className="plot08-cross-flow-main",
                    ),
                    html.Strong(
                        str(count),
                        className="plot08-cross-flow-count",
                    ),
                ],
                id={
                    "type":
                        "cross-flow-route",
                    "origin":
                        origin,
                    "destination":
                        destination,
                },
                n_clicks=0,
                type="button",
                className="plot08-cross-flow-row",
                title=(
                    "Select this route to highlight "
                    "its crosses on both location maps"
                ),
            )
        )

    return html.Div(
        [
            kpis,
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                "TOP ROUTES",
                                className="match-panel-eyebrow",
                            ),
                            html.Strong(
                                "Origin → destination",
                                className="plot08-cross-flow-list-title",
                            ),
                            html.Small(
                                (
                                    "Select a pathway to open the "
                                    "location maps with matching "
                                    "crosses highlighted."
                                ),
                                className="plot08-cross-flow-list-note",
                            ),
                        ]
                    ),
                    html.Div(
                        route_rows,
                        className="plot08-cross-flow-route-list",
                    ),
                ],
                className="plot08-cross-flow-list",
            ),
            html.Div(
                [
                    html.I(
                        className="fa-solid fa-circle-info"
                    ),
                    html.Span(
                        (
                            "Retention counts a completed cross or "
                            "same-team control/second ball within "
                            "10 seconds before opponent control. "
                            "Shot rate counts a same-team shot in "
                            "that same post-cross window. "
                            "The window never crosses periods."
                        )
                    ),
                ],
                className="plot08-cross-flow-method",
            ),
        ],
        className="plot08-cross-flow",
    )
