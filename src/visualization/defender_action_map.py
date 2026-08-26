from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go

from src.visualization import pitch_plots
from src.visualization.plotly_branding import (
    MATCH_COMPARE_HEIGHT,
    MATCH_HOVER_BG,
    MATCH_PITCH_BG,
    MATCH_PITCH_MUTED,
    MATCH_PITCH_TEXT,
    MATCH_TEXT_FONT,
    MATCH_WARNING,
    add_attacking_direction,
)


MUTED = "#9fb3bf"


ACTION_STYLE = {
    "Interception": {
        "symbol": "diamond",
        "size": 12,
    },
    "Recovery": {
        "symbol": "hexagon",
        "size": 12,
    },
    "Clearance": {
        "symbol": "triangle-up",
        "size": 13,
    },
    "Block": {
        "symbol": "square",
        "size": 11,
    },
    "Foul": {
        "symbol": "x",
        "size": 12,
    },
}


LEGEND_ORDER = (
    "Tackle",
    "Interception",
    "Recovery",
    "Clearance",
    "Block",
    "Foul",
)


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number):
        return None

    return number


def _time_label(row):
    minute = _num(row.get("timeMin"))
    second = _num(row.get("timeSec"))

    if minute is None:
        return "Time unavailable"

    if second is None:
        second = 0

    return f"{int(minute)}:{int(second):02d}"


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
    """Match the established Player Analysis pitch-map presentation."""
    fig.update_layout(
        title=None,
        plot_bgcolor=MATCH_PITCH_BG,
        paper_bgcolor=MATCH_PITCH_BG,
        font=dict(
            color=MATCH_PITCH_TEXT,
            family=MATCH_TEXT_FONT,
        ),
        xaxis=dict(
            range=[0, 100],
            visible=False,
            fixedrange=True,
        ),
        yaxis=dict(
            range=[0, 100],
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=0.68,
        ),
        height=MATCH_COMPARE_HEIGHT,
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
            traceorder="normal",
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor=MATCH_HOVER_BG,
            bordercolor="rgba(255,255,255,0.18)",
            font=dict(
                color="#ffffff",
                size=12,
                family=MATCH_TEXT_FONT,
            ),
            align="left",
        ),
    )


def _typical_defensive_zone(
    frame,
    team_color,
):
    """
    Robust typical defensive-intervention area.

    This deliberately mirrors the visual grammar used by Shooting Analysis:
    the ellipse is centred on the median location and sized from the IQR, so
    isolated actions do not stretch the footprint across the whole pitch.

    Fouls are excluded before this helper is called. Failed tackle attempts
    remain included because they still describe where the player defended.
    """
    if (
        frame is None
        or frame.empty
        or len(frame) < 4
    ):
        return None

    x = pd.to_numeric(
        frame["x"],
        errors="coerce",
    ).dropna()

    y = pd.to_numeric(
        frame["y"],
        errors="coerce",
    ).dropna()

    if (
        len(x) < 4
        or len(y) < 4
    ):
        return None

    center_x = float(
        x.median()
    )
    center_y = float(
        y.median()
    )

    q25_x = float(
        x.quantile(
            0.25
        )
    )
    q75_x = float(
        x.quantile(
            0.75
        )
    )
    q25_y = float(
        y.quantile(
            0.25
        )
    )
    q75_y = float(
        y.quantile(
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
        name="Typical defensive zone",
        showlegend=False,
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


def _hover_text(row):
    detail = str(
        row.get(
            "defensive_detail",
            row.get(
                "defensive_category",
                "Defensive action",
            ),
        )
    )

    return (
        f"<b>{detail}</b>"
        f"<br>Match time: {_time_label(row)}"
    )


def _add_action_trace(
    fig,
    frame,
    *,
    name,
    symbol,
    color,
    size,
    open_marker=False,
    showlegend=True,
):
    if frame.empty:
        return

    marker_symbol = (
        f"{symbol}-open"
        if open_marker
        else symbol
    )

    marker = dict(
        symbol=marker_symbol,
        size=size,
        color=color,
        line=dict(
            color=(
                color
                if open_marker
                else "#ffffff"
            ),
            width=(
                2.0
                if open_marker
                else 1.0
            ),
        ),
    )

    fig.add_trace(
        go.Scattergl(
            x=frame["x"],
            y=frame["y"],
            mode="markers",
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            marker=marker,
            opacity=(
                0.78
                if open_marker
                else 0.96
            ),
            text=[
                _hover_text(row)
                for _, row in frame.iterrows()
            ],
            hovertemplate=(
                "%{text}"
                "<extra></extra>"
            ),
        )
    )


def plot_defensive_action_profile(
    defensive_events,
    *,
    selected_player,
    jersey,
    team_color,
    is_away=False,
):
    """Plot the canonical PLOT-17 player defensive-action map.

    Processed coordinates are already team-relative. ``is_away`` is accepted
    for UI symmetry with the other player maps and never mirrors geometry.
    """
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
        defensive_events is None
        or defensive_events.empty
    ):
        fig.add_annotation(
            x=50,
            y=50,
            text=(
                "No defensive actions for the selected player"
            ),
            showarrow=False,
            font=dict(
                color=MATCH_PITCH_MUTED,
                size=14,
                family=MATCH_TEXT_FONT,
            ),
        )
        _base_layout(
            fig,
            showlegend=False,
        )
        return fig

    frame = defensive_events.copy()

    for column in (
        "x",
        "y",
    ):
        if column not in frame.columns:
            frame[column] = pd.NA
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame = frame.dropna(
        subset=[
            "x",
            "y",
        ]
    )

    if frame.empty:
        fig.add_annotation(
            x=50,
            y=50,
            text=(
                "Defensive actions are recorded but locations are unavailable"
            ),
            showarrow=False,
            font=dict(
                color=MATCH_PITCH_MUTED,
                size=14,
                family=MATCH_TEXT_FONT,
            ),
        )
        _base_layout(
            fig,
            showlegend=False,
        )
        return fig

    # The coverage footprint describes defensive intervention territory,
    # not disciplinary events. Unsuccessful tackle attempts remain valid
    # spatial evidence; fouls are intentionally excluded.
    coverage_frame = frame.loc[
        ~frame["defensive_category"].eq(
            "Foul"
        )
    ].copy()

    zone_trace = _typical_defensive_zone(
        coverage_frame,
        team_color,
    )

    if zone_trace is not None:
        fig.add_trace(
            zone_trace
        )

    tackle = frame.loc[
        frame["defensive_category"].eq(
            "Tackle"
        )
    ]
    tackle_won = tackle.loc[
        tackle["tackle_won"]
        .fillna(False)
        .astype(bool)
    ]
    tackle_unsuccessful = tackle.loc[
        ~tackle["tackle_won"]
        .fillna(False)
        .astype(bool)
    ]

    _add_action_trace(
        fig,
        tackle_won,
        name="Tackle",
        symbol="circle",
        color=team_color,
        size=12,
    )
    _add_action_trace(
        fig,
        tackle_unsuccessful,
        name="Tackle",
        symbol="circle",
        color=MUTED,
        size=12,
        open_marker=True,
        showlegend=tackle_won.empty,
    )

    for category in (
        "Interception",
        "Recovery",
        "Clearance",
        "Block",
        "Foul",
    ):
        subset = frame.loc[
            frame["defensive_category"].eq(
                category
            )
        ]
        style = ACTION_STYLE[category]
        _add_action_trace(
            fig,
            subset,
            name=category,
            symbol=style["symbol"],
            color=(
                MATCH_WARNING
                if category == "Foul"
                else team_color
            ),
            size=style["size"],
        )

    median_x = (
        float(
            coverage_frame["x"].median()
        )
        if not coverage_frame.empty
        else None
    )
    median_y = (
        float(
            coverage_frame["y"].median()
        )
        if not coverage_frame.empty
        else None
    )

    if (
        median_x is not None
        and median_y is not None
        and math.isfinite(median_x)
        and math.isfinite(median_y)
    ):
        fig.add_trace(
            go.Scatter(
                x=[median_x],
                y=[median_y],
                mode="markers+text",
                name="Median defensive-action position",
                showlegend=False,
                marker=dict(
                    symbol="circle",
                    size=31,
                    color=team_color,
                    line=dict(
                        color="#ffffff",
                        width=2.2,
                    ),
                ),
                text=[str(jersey)],
                textposition="middle center",
                textfont=dict(
                    color="#ffffff",
                    size=12,
                    family=MATCH_TEXT_FONT,
                ),
                hovertemplate=(
                    "<b>Median defensive-action position</b>"
                    f"<br>{selected_player}"
                    f"<br>Location: ({median_x:.1f}, {median_y:.1f})"
                    "<extra></extra>"
                ),
            )
        )

    # The traces are added in the same semantic order as the legend contract.
    # This explicit rank keeps the presentation stable if Plotly changes trace
    # ordering internals.
    rank = {
        "Typical defensive zone": -1,
        **{
            name: index
            for index, name in enumerate(
                LEGEND_ORDER
            )
        },
    }
    fig.data = tuple(
        sorted(
            fig.data,
            key=lambda trace: rank.get(
                trace.name,
                len(rank),
            ),
        )
    )

    _base_layout(
        fig,
        showlegend=True,
    )

    return fig
