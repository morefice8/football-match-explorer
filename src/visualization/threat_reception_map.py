from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from src.visualization import pitch_plots
from src.visualization.plotly_branding import (
    add_attacking_direction,
)

PITCH_BG = "#27343e"
TEXT = "#edf5f7"
MUTED = "#9fb3bf"
FOCUS = "#f0b44c"


def _num(value):
    try:
        number = float(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(
        number
    ):
        return None

    return number


def _time_label(row):
    minute = _num(
        row.get(
            "timeMin"
        )
    )
    second = _num(
        row.get(
            "timeSec"
        )
    )

    if minute is None:
        return "Time unavailable"

    if second is None:
        second = 0

    return (
        f"{int(minute)}:"
        f"{int(second):02d}"
    )


def _hex_to_rgba(
    color,
    alpha,
):
    text = (
        str(
            color
            or ""
        )
        .strip()
        .lstrip("#")
    )

    if len(text) != 6:
        return (
            f"rgba(255,255,255,{alpha})"
        )

    try:
        red = int(
            text[0:2],
            16,
        )
        green = int(
            text[2:4],
            16,
        )
        blue = int(
            text[4:6],
            16,
        )
    except ValueError:
        return (
            f"rgba(255,255,255,{alpha})"
        )

    return (
        f"rgba({red},{green},{blue},{alpha})"
    )


def _base_layout(
    fig,
    *,
    showlegend=True,
):
    fig.update_layout(
        title=None,
        plot_bgcolor=PITCH_BG,
        paper_bgcolor=PITCH_BG,
        font=dict(
            color=TEXT,
            family="Inter, Arial",
        ),
        xaxis=dict(
            range=[
                0,
                100,
            ],
            visible=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[
                0,
                100,
            ],
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=0.68,
        ),
        height=560,
        margin=dict(
            l=8,
            r=8,
            t=54 if showlegend else 16,
            b=8,
        ),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0,
            font=dict(
                color="#dbe7ec",
                size=10,
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#dbe7ed",
            font=dict(
                color="#17354d",
                size=12,
            ),
        ),
    )


def _typical_reception_zone(
    frame,
    team_color,
):
    """
    Robust dispersion ellipse around the central reception cluster.

    Unlike a convex hull, this deliberately does not stretch to every outlier.
    It summarises the typical reception territory, similar in spirit to the
    dispersion footprints used by Mean Positions.
    """
    end_x = pd.to_numeric(
        frame["end_x"],
        errors="coerce",
    ).dropna()

    end_y = pd.to_numeric(
        frame["end_y"],
        errors="coerce",
    ).dropna()

    if (
        end_x.empty
        or end_y.empty
    ):
        return None

    center_x = float(
        end_x.median()
    )
    center_y = float(
        end_y.median()
    )

    q25_x = float(
        end_x.quantile(
            0.25
        )
    )
    q75_x = float(
        end_x.quantile(
            0.75
        )
    )
    q25_y = float(
        end_y.quantile(
            0.25
        )
    )
    q75_y = float(
        end_y.quantile(
            0.75
        )
    )

    radius_x = max(
        4.5,
        min(
            14.0,
            (
                q75_x
                - q25_x
            )
            * 0.95,
        ),
    )

    radius_y = max(
        4.5,
        min(
            14.0,
            (
                q75_y
                - q25_y
            )
            * 0.95,
        ),
    )

    points = 48

    theta = [
        (
            index
            * 2
            * math.pi
            / points
        )
        for index
        in range(
            points
            + 1
        )
    ]

    x_coords = [
        center_x
        + radius_x
        * math.cos(
            value
        )
        for value
        in theta
    ]

    y_coords = [
        center_y
        + radius_y
        * math.sin(
            value
        )
        for value
        in theta
    ]

    return go.Scatter(
        x=x_coords,
        y=y_coords,
        mode="lines",
        name="Typical reception zone",
        fill="toself",
        line=dict(
            color=_hex_to_rgba(
                team_color,
                0.42,
            ),
            width=1.0,
        ),
        fillcolor=_hex_to_rgba(
            team_color,
            0.11,
        ),
        hoverinfo="skip",
    )


def plot_reception_profile(
    receptions,
    *,
    selected_player,
    jersey,
    team_color,
    summary,
    is_away=False,
):
    del is_away

    fig = go.Figure()

    fig.update_layout(
        shapes=(
            pitch_plots
            .get_plotly_pitch_shapes(
                "rgba(255,255,255,0.22)",
                "#d7e4eb",
            )
        )
    )

    add_attacking_direction(
        fig,
        dark=True,
    )

    if (
        receptions is None
        or receptions.empty
    ):
        fig.add_annotation(
            x=50,
            y=50,
            text=(
                "No reliable completed receptions "
                "for the selected player"
            ),
            showarrow=False,
            font=dict(
                color="#dbe7ec",
                size=14,
            ),
        )

        _base_layout(
            fig,
            showlegend=False,
        )

        return fig

    frame = receptions.copy()

    for column in (
        "x",
        "y",
        "end_x",
        "end_y",
    ):
        frame[
            column
        ] = pd.to_numeric(
            frame[
                column
            ],
            errors="coerce",
        )

    if (
        "_in_shot_sequence"
        not in frame.columns
    ):
        frame[
            "_in_shot_sequence"
        ] = False

    frame = frame.dropna(
        subset=[
            "x",
            "y",
            "end_x",
            "end_y",
        ]
    )

    if frame.empty:
        _base_layout(
            fig,
            showlegend=False,
        )
        return fig

    zone_trace = (
        _typical_reception_zone(
            frame,
            team_color,
        )
    )

    if (
        zone_trace is not None
    ):
        fig.add_trace(
            zone_trace
        )

    line_x = []
    line_y = []

    for _, row in frame.iterrows():
        line_x.extend([
            row["x"],
            row["end_x"],
            None,
        ])
        line_y.extend([
            row["y"],
            row["end_y"],
            None,
        ])

    fig.add_trace(
        go.Scattergl(
            x=line_x,
            y=line_y,
            mode="lines",
            name="Received pass",
            line=dict(
                color=team_color,
                width=1.55,
            ),
            opacity=0.20,
            hoverinfo="skip",
        )
    )

    focus = (
        frame[
            frame[
                "_in_shot_sequence"
            ]
            .fillna(False)
            .astype(bool)
        ]
        .copy()
    )

    if not focus.empty:
        focus_x = []
        focus_y = []

        for _, row in focus.iterrows():
            focus_x.extend([
                row["x"],
                row["end_x"],
                None,
            ])
            focus_y.extend([
                row["y"],
                row["end_y"],
                None,
            ])

        fig.add_trace(
            go.Scattergl(
                x=focus_x,
                y=focus_y,
                mode="lines",
                name="Shot-linked reception",
                line=dict(
                    color=FOCUS,
                    width=3.1,
                ),
                opacity=0.96,
                hoverinfo="skip",
            )
        )

    origin_hover = []

    for _, row in frame.iterrows():
        passer = str(
            row.get(
                "playerName",
                "Unknown passer",
            )
        )

        origin_hover.append(
            (
                f"<b>{passer}</b>"
                f"<br>Pass origin"
                f"<br>{_time_label(row)}"
                f"<br>To: {selected_player}"
            )
        )

    fig.add_trace(
        go.Scattergl(
            x=frame["x"],
            y=frame["y"],
            mode="markers",
            name="Pass origin",
            marker=dict(
                symbol="circle-open",
                size=7,
                color=MUTED,
                line=dict(
                    color=MUTED,
                    width=1.2,
                ),
            ),
            opacity=0.70,
            text=origin_hover,
            hovertemplate=(
                "%{text}"
                "<extra></extra>"
            ),
        )
    )

    reception_hover = []

    for _, row in frame.iterrows():
        passer = str(
            row.get(
                "playerName",
                "Unknown passer",
            )
        )

        reception_hover.append(
            (
                f"<b>{selected_player}</b>"
                f"<br>Reception"
                f"<br>{_time_label(row)}"
                f"<br>Pass from: {passer}"
                f"<br>Location: "
                f"({row['end_x']:.1f}, {row['end_y']:.1f})"
            )
        )

    fig.add_trace(
        go.Scattergl(
            x=frame["end_x"],
            y=frame["end_y"],
            mode="markers",
            name="Reception",
            marker=dict(
                symbol="circle",
                size=8.5,
                color=team_color,
                line=dict(
                    color=PITCH_BG,
                    width=1.0,
                ),
            ),
            opacity=0.84,
            text=reception_hover,
            hovertemplate=(
                "%{text}"
                "<extra></extra>"
            ),
        )
    )

    if not focus.empty:
        focus_hover = []

        for _, row in focus.iterrows():
            passer = str(
                row.get(
                    "playerName",
                    "Unknown passer",
                )
            )

            focus_hover.append(
                (
                    f"<b>{selected_player}</b>"
                    f"<br>Shot-linked reception"
                    f"<br>{_time_label(row)}"
                    f"<br>Pass from: {passer}"
                    f"<br>Location: "
                    f"({row['end_x']:.1f}, {row['end_y']:.1f})"
                )
            )

        fig.add_trace(
            go.Scattergl(
                x=focus["end_x"],
                y=focus["end_y"],
                mode="markers",
                name="Highlighted reception",
                showlegend=False,
                marker=dict(
                    symbol="circle",
                    size=12,
                    color=team_color,
                    line=dict(
                        color=FOCUS,
                        width=2.5,
                    ),
                ),
                opacity=1.0,
                text=focus_hover,
                hovertemplate=(
                    "%{text}"
                    "<extra></extra>"
                ),
            )
        )

    median_x = (
        summary
        or {}
    ).get(
        "median_x"
    )

    median_y = (
        summary
        or {}
    ).get(
        "median_y"
    )

    if (
        median_x is not None
        and median_y is not None
    ):
        fig.add_trace(
            go.Scatter(
                x=[
                    median_x
                ],
                y=[
                    median_y
                ],
                mode="markers+text",
                name="Median reception position",
                marker=dict(
                    symbol="circle",
                    size=31,
                    color=team_color,
                    line=dict(
                        color="#ffffff",
                        width=2.2,
                    ),
                ),
                text=[
                    str(jersey)
                ],
                textposition="middle center",
                textfont=dict(
                    color="#ffffff",
                    size=12,
                    family="Inter, Arial",
                ),
                hovertemplate=(
                    "<b>Median reception position</b>"
                    f"<br>{selected_player}"
                    f"<br>Location: "
                    f"({median_x:.1f}, {median_y:.1f})"
                    "<extra></extra>"
                ),
            )
        )

    _base_layout(
        fig,
        showlegend=True,
    )

    return fig
