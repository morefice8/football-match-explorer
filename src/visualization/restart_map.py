from __future__ import annotations

import math

import plotly.graph_objects as go

from src.visualization.buildup_plotly import (
    draw_plotly_pitch,
)
from src.visualization.plotly_branding import (
    add_attacking_direction,
)


SYMBOLS = {
    "Corner": "diamond",
    "Free Kick": "square",
    "Throw-in": "circle",
    "Goal Kick": "triangle-up",
    "Penalty": "star",
}

OUTCOME_ENDPOINT = {
    "Goal": "#f0b44c",
    "Shot": "#e8edf0",
    "Successful Delivery": "#9be2ca",
    "Unsuccessful Delivery": "#d8a18c",
}


def _time_label(
    value,
):
    try:
        second = float(value)
    except (TypeError, ValueError):
        return "—"

    if not math.isfinite(second):
        return "—"

    minute = int(
        second // 60
    )
    remaining = int(
        round(
            second
            - minute * 60
        )
    )

    if remaining >= 60:
        minute += 1
        remaining = 0

    return (
        f"{minute}'{remaining:02d}\""
    )


def _hover(
    record,
):
    player = record.get(
        "player_name",
        "Unknown player",
    )

    jersey = record.get(
        "jersey_number"
    )

    player_label = (
        f"{player} · {jersey}"
        if jersey not in (
            None,
            "",
        )
        else str(player)
    )

    return (
        f"<b>{player_label}</b>"
        f"<br>{record.get('restart_type', 'Restart')}"
        f" · {record.get('side', 'Unknown')}"
        f"<br>Delivery: {record.get('delivery', 'Unknown')}"
        f"<br>Destination: {record.get('destination', 'N/A')}"
        f"<br>Execution: {record.get('outcome', 'Unknown')}"
        f"<br>Development: {record.get('development_outcome', 'Unknown')}"
        f"<br>Match time: {_time_label(record.get('match_second'))}"
        "<br><i>Click to inspect this restart</i>"
    )


def plot_restart_map(
    records,
    *,
    team_color,
    selected_sequence_id=None,
):
    fig = go.Figure()

    draw_plotly_pitch(
        fig
    )

    add_attacking_direction(
        fig,
        dark=True,
    )

    if not records:
        fig.add_annotation(
            x=50,
            y=50,
            text="No restarts match the current filters",
            showarrow=False,
            font=dict(
                color="#dbe7ec",
                size=14,
                family="Inter, Arial",
            ),
        )

    legend_seen = set()

    record_count = len(
        records
        or []
    )

    aggregate_opacity = (
        0.40
        if record_count >= 20
        else (
            0.58
            if record_count >= 10
            else 0.74
        )
    )

    aggregate_width = (
        1.35
        if record_count >= 20
        else (
            1.65
            if record_count >= 10
            else 2.0
        )
    )

    for record in records or []:
        x0 = record.get(
            "start_x"
        )
        y0 = record.get(
            "start_y"
        )
        x1 = record.get(
            "end_x"
        )
        y1 = record.get(
            "end_y"
        )

        if any(
            value is None
            for value
            in (
                x0,
                y0,
                x1,
                y1,
            )
        ):
            continue

        sequence_id = str(
            record.get(
                "sequence_id"
            )
        )

        selected = (
            selected_sequence_id
            is not None
            and sequence_id
            == str(
                selected_sequence_id
            )
        )

        restart_type = str(
            record.get(
                "restart_type",
                "Restart",
            )
        )

        outcome = str(
            record.get(
                "outcome",
                "Unknown",
            )
        )

        unsuccessful = (
            "unsuccessful"
            in outcome.lower()
            or "lost"
            in outcome.lower()
            or "miss"
            in outcome.lower()
            or "saved"
            in outcome.lower()
        )

        endpoint_color = (
            OUTCOME_ENDPOINT.get(
                outcome,
                "#edf5f7",
            )
        )

        custom = [
            [
                sequence_id,
                restart_type,
                record.get(
                    "side"
                ),
                record.get(
                    "delivery"
                ),
                record.get(
                    "destination"
                ),
                outcome,
            ],
            [
                sequence_id,
                restart_type,
                record.get(
                    "side"
                ),
                record.get(
                    "delivery"
                ),
                record.get(
                    "destination"
                ),
                outcome,
            ],
        ]

        fig.add_trace(
            go.Scatter(
                x=[
                    float(x0),
                    float(x1),
                ],
                y=[
                    float(y0),
                    float(y1),
                ],
                mode="lines+markers",
                line=dict(
                    color=
                        team_color,
                    width=(
                        4.6
                        if selected
                        else aggregate_width
                    ),
                    dash=(
                        "dot"
                        if unsuccessful
                        else "solid"
                    ),
                ),
                marker=dict(
                    size=[
                        (
                            15
                            if selected
                            else (
                                7
                                if record_count >= 20
                                else 9
                            )
                        ),
                        (
                            11
                            if selected
                            else (
                                5
                                if record_count >= 20
                                else 7
                            )
                        ),
                    ],
                    symbol=[
                        SYMBOLS.get(
                            restart_type,
                            "circle",
                        ),
                        "circle",
                    ],
                    color=[
                        team_color,
                        endpoint_color,
                    ],
                    line=dict(
                        color="white",
                        width=(
                            1.2
                            if selected
                            else 0.75
                        ),
                    ),
                ),
                opacity=(
                    1.0
                    if selected
                    else (
                        0.24
                        if (
                            selected_sequence_id
                            is not None
                        )
                        else aggregate_opacity
                    )
                ),
                text=[
                    _hover(
                        record
                    ),
                    _hover(
                        record
                    ),
                ],
                customdata=
                    custom,
                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                ),
                name=
                    restart_type,
                legendgroup=
                    restart_type,
                showlegend=(
                    restart_type
                    not in legend_seen
                ),
            )
        )

        legend_seen.add(
            restart_type
        )

    fig.add_annotation(
        x=1.0,
        y=1.055,
        xref="paper",
        yref="paper",
        text=(
            "<b>Solid</b> = successful"
            " · "
            "<b>Dotted</b> = unsuccessful"
        ),
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        font=dict(
            color="#dce9ee",
            size=10,
            family="Inter, Arial",
        ),
        bgcolor="rgba(39,52,62,0.82)",
        bordercolor="#506875",
        borderwidth=1,
        borderpad=4,
    )

    fig.update_layout(
        title=None,
        height=600,
        autosize=True,
        paper_bgcolor="#27343e",
        plot_bgcolor="#27343e",
        margin=dict(
            l=12,
            r=12,
            t=58,
            b=12,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="left",
            x=0,
            bgcolor=
                "rgba(39,52,62,0.78)",
            font=dict(
                color="white",
                size=11,
                family="Inter, Arial",
            ),
        ),
        hoverlabel=dict(
            bgcolor="#11344c",
            bordercolor="#4f7187",
            font=dict(
                color="white",
                family="Inter, Arial",
                size=12,
            ),
        ),
        clickmode="event+select",
    )

    fig.update_xaxes(
        range=[-2, 102],
        visible=False,
        fixedrange=True,
    )

    fig.update_yaxes(
        range=[-5, 105],
        visible=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=0.68,
    )

    return fig
