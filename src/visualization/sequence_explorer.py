"""Shared Plotly renderer for normalized football sequences."""

from __future__ import annotations

from collections import Counter
import html
import math

import plotly.graph_objects as go


PITCH_BG = "#283640"
PITCH_LINE = "rgba(215, 228, 235, 0.34)"
TEXT = "#18344d"

PASS_FAILURE = "#f0b44c"
CARRY_COLOR = "#f3c870"
SHOT_COLOR = "#e9edf0"
TURNOVER_COLOR = "#f0b44c"
RESTART_COLOR = "#d8e9ef"


SEQUENCE_LABELS = {
    "buildup": "Build-up",
    "defensive_transition":
        "Defensive transition",
    "offensive_transition":
        "Offensive transition",
    "set_piece": "Restart",
}


def _pitch_shapes():
    return [
        dict(
            type="rect",
            x0=0,
            y0=0,
            x1=100,
            y1=100,
            line=dict(
                color=PITCH_LINE,
                width=1.5,
            ),
        ),
        dict(
            type="line",
            x0=50,
            y0=0,
            x1=50,
            y1=100,
            line=dict(
                color=PITCH_LINE,
                width=1.2,
            ),
        ),
        dict(
            type="circle",
            x0=41.5,
            y0=36.5,
            x1=58.5,
            y1=63.5,
            line=dict(
                color=PITCH_LINE,
                width=1.2,
            ),
        ),
        dict(
            type="rect",
            x0=0,
            y0=21.1,
            x1=16.5,
            y1=78.9,
            line=dict(
                color=PITCH_LINE,
                width=1.2,
            ),
        ),
        dict(
            type="rect",
            x0=83.5,
            y0=21.1,
            x1=100,
            y1=78.9,
            line=dict(
                color=PITCH_LINE,
                width=1.2,
            ),
        ),
        dict(
            type="rect",
            x0=0,
            y0=36.8,
            x1=5.5,
            y1=63.2,
            line=dict(
                color=PITCH_LINE,
                width=1.2,
            ),
        ),
        dict(
            type="rect",
            x0=94.5,
            y0=36.8,
            x1=100,
            y1=63.2,
            line=dict(
                color=PITCH_LINE,
                width=1.2,
            ),
        ),
    ]


def _safe_text(value, fallback=""):
    if value is None:
        return fallback

    return html.escape(
        str(value)
    )


def _duration_label(value):
    if value is None:
        return "—"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"

    if not math.isfinite(value):
        return "—"

    return f"{value:.1f}s"


def _time_label(value):
    if value is None:
        return "—"

    try:
        total = int(
            round(float(value))
        )
    except (TypeError, ValueError):
        return "—"

    minute = total // 60
    second = total % 60

    return (
        f"{minute}'{second:02d}\""
    )


def _valid_line(event):
    return all(
        event.get(key) is not None
        for key in (
            "x",
            "y",
            "end_x",
            "end_y",
        )
    )


def _hover(event):
    player = _safe_text(
        event.get("player_name"),
        "Unknown player",
    )

    raw_type = _safe_text(
        event.get(
            "raw_event_type",
            event.get(
                "event_type",
                "Event",
            ),
        )
    )

    outcome = _safe_text(
        event.get("outcome"),
        "—",
    )

    time_value = _time_label(
        event.get("second")
    )

    parts = [
        f"<b>{player}</b>",
        raw_type,
        f"Time: {time_value}",
        f"Outcome: {outcome}",
    ]

    metadata = (
        event.get("metadata")
        or {}
    )

    if (
        event.get("event_type")
        == "restart"
    ):
        restart_type = metadata.get(
            "restart_type"
        )
        delivery = metadata.get(
            "restart_delivery_type"
        )
        length = metadata.get(
            "restart_length_m"
        )

        if restart_type:
            parts.append(
                "Restart: "
                + _safe_text(
                    restart_type
                )
            )

        if delivery:
            parts.append(
                "Delivery: "
                + _safe_text(
                    delivery
                )
            )

        if length is not None:
            try:
                parts.append(
                    "Length: "
                    f"{float(length):.1f} m"
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    if (
        event.get("event_type")
        == "carry"
    ):
        distance = metadata.get(
            "distance_coordinate_units"
        )

        if distance is not None:
            try:
                parts.append(
                    "Spatial carry: "
                    f"{float(distance):.1f}"
                    " coordinate units"
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    return "<br>".join(
        parts
    )


def _marker_label(
    event,
    step,
):
    jersey = event.get(
        "jersey_number"
    )

    if jersey is None:
        metadata = (
            event.get("metadata")
            or {}
        )
        jersey = metadata.get(
            "Mapped Jersey Number"
        )

    if jersey is None:
        return str(step)

    try:
        number = float(jersey)
        if number.is_integer():
            return str(int(number))
    except (
        TypeError,
        ValueError,
    ):
        pass

    text = str(jersey).strip()
    return text or str(step)


def _add_step_marker(
    fig,
    event,
    step,
):
    x = event.get("x")
    y = event.get("y")

    if x is None or y is None:
        return

    is_turnover = (
        event.get("event_type")
        == "turnover"
    )

    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            marker=dict(
                symbol=(
                    "diamond"
                    if is_turnover
                    else "circle"
                ),
                size=(
                    22
                    if is_turnover
                    else 20
                ),
                color=(
                    TURNOVER_COLOR
                    if is_turnover
                    else "#11344c"
                ),
                line=dict(
                    color=(
                        "#fff4d6"
                        if is_turnover
                        else "#f2f7f9"
                    ),
                    width=(
                        2.0
                        if is_turnover
                        else 1.5
                    ),
                ),
            ),
            text=[
                _marker_label(
                    event,
                    step,
                )
            ],
            textposition="middle center",
            textfont=dict(
                size=(
                    10
                    if is_turnover
                    else 9
                ),
                color=(
                    "#11344c"
                    if is_turnover
                    else "white"
                ),
                family="Inter, Arial",
            ),
            hovertext=[
                _hover(event)
                + "<br>Step: "
                + str(step)
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )


def _add_line_event(
    fig,
    event,
    *,
    name,
    color,
    dash,
    width,
    showlegend,
):
    if not _valid_line(
        event
    ):
        return

    fig.add_trace(
        go.Scatter(
            x=[
                event["x"],
                event["end_x"],
            ],
            y=[
                event["y"],
                event["end_y"],
            ],
            mode="lines",
            line=dict(
                color=color,
                width=width,
                dash=dash,
            ),
            opacity=0.94,
            name=name,
            legendgroup=name,
            showlegend=showlegend,
            hovertext=[
                _hover(event),
                _hover(event),
            ],
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[event["end_x"]],
            y=[event["end_y"]],
            mode="markers",
            marker=dict(
                size=8,
                symbol="triangle-right",
                color=color,
                line=dict(
                    width=0,
                ),
            ),
            hovertext=[
                _hover(event)
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )


def _add_shot(
    fig,
    event,
    *,
    showlegend,
):
    x = event.get("x")
    y = event.get("y")

    if x is None or y is None:
        return

    end_x = event.get("end_x")
    end_y = event.get("end_y")

    if end_x is None:
        end_x = 100.0

    if end_y is None:
        end_y = 50.0

    fig.add_trace(
        go.Scatter(
            x=[x, end_x],
            y=[y, end_y],
            mode="lines",
            line=dict(
                color=SHOT_COLOR,
                width=3.2,
            ),
            name="Shot",
            legendgroup="Shot",
            showlegend=showlegend,
            hovertext=[
                _hover(event),
                _hover(event),
            ],
            hoverinfo="text",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(
                symbol="star",
                size=13,
                color=SHOT_COLOR,
                line=dict(
                    color="#11344c",
                    width=1,
                ),
            ),
            hovertext=[
                _hover(event)
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )


def _add_turnover(
    fig,
    event,
    *,
    showlegend,
):
    x = event.get("x")
    y = event.get("y")

    if x is None or y is None:
        return

    fig.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=15,
                color=TURNOVER_COLOR,
                line=dict(
                    color="white",
                    width=1.5,
                ),
            ),
            name="Turnover",
            legendgroup="Turnover",
            showlegend=showlegend,
            hovertext=[
                _hover(event)
            ],
            hoverinfo="text",
        )
    )


def _add_incomplete_endpoint(
    fig,
    event,
):
    if (
        event.get("event_type")
        != "pass"
        or event.get("successful")
        is not False
        or event.get("end_x")
        is None
        or event.get("end_y")
        is None
    ):
        return

    fig.add_trace(
        go.Scatter(
            x=[
                event["end_x"]
            ],
            y=[
                event["end_y"]
            ],
            mode="markers",
            marker=dict(
                symbol="x",
                size=11,
                color=PASS_FAILURE,
                line=dict(
                    width=2,
                ),
            ),
            hovertext=[
                _hover(event)
            ],
            hoverinfo="text",
            showlegend=False,
        )
    )


def _restart_detail_line(sequence):
    events = (
        sequence.get("events")
        or []
    )

    restart_event = next(
        (
            event
            for event in events
            if event.get("event_type")
            == "restart"
        ),
        None,
    )

    metadata = (
        restart_event.get("metadata")
        if restart_event
        else {}
    ) or {}

    delivery = metadata.get(
        "restart_delivery_type"
    )
    length = metadata.get(
        "restart_length_m"
    )
    outcome = _safe_text(
        sequence.get("outcome"),
        "Unknown",
    )

    details = []

    if delivery:
        details.append(
            _safe_text(delivery)
        )

    if length is not None:
        try:
            details.append(
                f"{float(length):.1f} m"
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    details.append(outcome)

    return " · ".join(details)


def _summary_annotation(
    sequence,
):
    events = (
        sequence.get("events")
        or []
    )

    counts = Counter(
        event.get("event_type")
        for event in events
    )

    sequence_label = (
        SEQUENCE_LABELS.get(
            sequence.get(
                "sequence_type"
            ),
            "Sequence",
        )
    )

    trigger = sequence.get(
        "trigger"
    )

    outcome = _safe_text(
        sequence.get("outcome"),
        "Unknown",
    )

    duration = _duration_label(
        sequence.get(
            "duration_seconds"
        )
    )

    line_one = (
        f"<b>{_safe_text(sequence_label)}</b>"
    )

    if trigger:
        line_one += (
            " · "
            + _safe_text(trigger)
        )

    if (
        sequence.get("sequence_type")
        == "set_piece"
    ):
        # Restart Analysis ends at delivery; do not present it as a
        # possession with a missing duration.
        line_two = _restart_detail_line(
            sequence
        )
    else:
        line_two = (
            f"Duration {duration}"
            f" · Outcome: {outcome}"
        )

    line_three = (
        f"Passes {counts.get('pass', 0)}"
        f" · Carries {counts.get('carry', 0)}"
        f" · Shots {counts.get('shot', 0)}"
        f" · Turnovers {counts.get('turnover', 0)}"
    )

    if (
        sequence.get("sequence_type")
        == "set_piece"
    ):
        # Restart Analysis stops at the delivery. Zero possession-action
        # counters add noise without adding analytical value.
        return (
            line_one
            + "<br>"
            + line_two
        )

    return (
        line_one
        + "<br>"
        + line_two
        + "<br>"
        + line_three
    )


def plot_sequence_explorer(
    sequence,
    *,
    team_color,
    height=520,
):
    """Render any normalized Match Analysis sequence with one visual grammar."""

    fig = go.Figure()

    events = (
        sequence.get("events")
        if sequence
        else []
    ) or []

    if not events:
        fig.add_annotation(
            x=50,
            y=50,
            text="No sequence events",
            showarrow=False,
            font=dict(
                size=16,
                color="#dbe7ec",
                family="Inter, Arial",
            ),
        )

    legend_seen = set()
    action_step = 0

    for event in events:
        event_type = event.get(
            "event_type"
        )

        if event_type == "carry":
            _add_line_event(
                fig,
                event,
                name="Carry",
                color=CARRY_COLOR,
                dash="dash",
                width=2.3,
                showlegend=(
                    "Carry"
                    not in legend_seen
                ),
            )
            legend_seen.add(
                "Carry"
            )
            continue

        action_step += 1

        if event_type == "pass":
            successful = event.get(
                "successful"
            )

            color = (
                team_color
                if successful is not False
                else PASS_FAILURE
            )

            dash = (
                "solid"
                if successful is not False
                else "dot"
            )

            _add_line_event(
                fig,
                event,
                name="Pass",
                color=color,
                dash=dash,
                width=3.0,
                showlegend=(
                    "Pass"
                    not in legend_seen
                ),
            )
            legend_seen.add(
                "Pass"
            )

            _add_incomplete_endpoint(
                fig,
                event,
            )

        elif event_type == "restart":
            _add_line_event(
                fig,
                event,
                name="Restart",
                color=RESTART_COLOR,
                dash="dashdot",
                width=3.2,
                showlegend=(
                    "Restart"
                    not in legend_seen
                ),
            )
            legend_seen.add(
                "Restart"
            )

        elif event_type == "shot":
            _add_shot(
                fig,
                event,
                showlegend=(
                    "Shot"
                    not in legend_seen
                ),
            )
            legend_seen.add(
                "Shot"
            )

        elif event_type == "turnover":
            _add_turnover(
                fig,
                event,
                showlegend=(
                    "Turnover"
                    not in legend_seen
                ),
            )
            legend_seen.add(
                "Turnover"
            )

        _add_step_marker(
            fig,
            event,
            action_step,
        )

    fig.update_layout(
        shapes=_pitch_shapes(),
        height=height,
        autosize=True,
        paper_bgcolor="white",
        plot_bgcolor=PITCH_BG,
        margin=dict(
            l=18,
            r=18,
            t=122,
            b=18,
        ),
        font=dict(
            family="Inter, Arial",
            color=TEXT,
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255,255,255,0)",
            font=dict(
                family="Inter, Arial",
                size=11,
                color=TEXT,
            ),
        ),
        showlegend=True,
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

    fig.add_annotation(
        x=0,
        y=1.07,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="bottom",
        align="left",
        text=_summary_annotation(
            sequence or {}
        ),
        showarrow=False,
        font=dict(
            family="Inter, Arial",
            size=13,
            color=TEXT,
        ),
    )

    fig.add_annotation(
        x=0.99,
        y=0.02,
        xref="paper",
        yref="paper",
        xanchor="right",
        yanchor="bottom",
        text="<b>Attacking →</b>",
        showarrow=False,
        bgcolor="#11344c",
        bordercolor="#31566d",
        borderwidth=1,
        font=dict(
            family="Inter, Arial",
            size=10,
            color="white",
        ),
    )

    return fig
