from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.metrics.transition_heatmap_metrics import (
    sequence_display_location,
)
from src.visualization.plotly_branding import add_attacking_direction

GRID_SIZE = 6
SCALE_MAX_PCT = 30.0

# Same palette and absolute percentage scale for both loss and recovery maps.
COLORSCALE = [
    [0.00, "#27343e"],
    [0.12, "#31505d"],
    [0.28, "#3b7181"],
    [0.48, "#3194aa"],
    [0.70, "#13a5c4"],
    [1.00, "#06718e"],
]


def _draw_pitch(fig):
    line = "#78909f"
    shapes = [
        go.layout.Shape(
            type="rect", x0=0, y0=0, x1=100, y1=100,
            line=dict(color=line, width=1.25),
        ),
        go.layout.Shape(
            type="line", x0=50, y0=0, x1=50, y1=100,
            line=dict(color=line, width=1.25),
        ),
        go.layout.Shape(
            type="circle", x0=42, y0=42, x1=58, y1=58,
            line=dict(color=line, width=1.15),
        ),
        go.layout.Shape(
            type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9,
            line=dict(color=line, width=1.15),
        ),
        go.layout.Shape(
            type="rect", x0=83.5, y0=21.1, x1=100, y1=78.9,
            line=dict(color=line, width=1.15),
        ),
        go.layout.Shape(
            type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2,
            line=dict(color=line, width=1.15),
        ),
        go.layout.Shape(
            type="rect", x0=94.5, y0=36.8, x1=100, y1=63.2,
            line=dict(color=line, width=1.15),
        ),
    ]
    fig.update_layout(shapes=shapes)
    return fig


def _point_hover(sequence, location_kind):
    first = sequence.iloc[0]
    outcome = first.get("sequence_outcome_type", "Unknown")

    if location_kind == "loss":
        loss_type = first.get("type_of_initial_loss", "Possession loss")
        zone = first.get("loss_zone", "Unknown zone")
        return (
            f"<b>{loss_type}</b>"
            f"<br>Zone: {zone}"
            f"<br>Outcome: {outcome}"
        )

    recovery_type = first.get("type_of_initial_loss", "Recovery")
    return f"<b>{recovery_type}</b><br>Outcome: {outcome}"


def plot_transition_heatmap(
    sequences,
    *,
    location_kind,
    is_away=False,
    loss_to_transition_frame=False,
    grid_size=GRID_SIZE,
):
    if location_kind not in {"loss", "recovery"}:
        raise ValueError(
            "location_kind must be 'loss' or 'recovery'"
        )

    x_coords = []
    y_coords = []
    hover_texts = []

    for sequence in (sequences or []):
        if sequence is None or sequence.empty:
            continue

        location = sequence_display_location(
            sequence,
            location_kind=location_kind,
            is_away=is_away,
            loss_to_transition_frame=
                loss_to_transition_frame,
        )
        if location is None:
            continue

        x, y = location
        x_coords.append(x)
        y_coords.append(y)
        hover_texts.append(_point_hover(sequence, location_kind))

    fig = go.Figure()
    _draw_pitch(fig)

    if x_coords:
        bin_edges = np.linspace(0, 100, int(grid_size) + 1)
        heatmap, _, _ = np.histogram2d(
            x_coords,
            y_coords,
            bins=[bin_edges, bin_edges],
        )

        total = float(heatmap.sum())
        heatmap_pct = (
            heatmap / total * 100.0
            if total > 0
            else heatmap
        )
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        customdata = []
        for j in range(len(centers)):
            row = []
            for i in range(len(centers)):
                row.append([
                    float(bin_edges[i]),
                    float(bin_edges[i + 1]),
                    float(bin_edges[j]),
                    float(bin_edges[j + 1]),
                    int(heatmap[i, j]),
                ])
            customdata.append(row)

        fig.add_trace(
            go.Heatmap(
                x=centers,
                y=centers,
                z=heatmap_pct.T,
                customdata=customdata,
                zmin=0,
                zmax=SCALE_MAX_PCT,
                colorscale=COLORSCALE,
                showscale=False,
                xgap=1,
                ygap=1,
                hoverongaps=False,
                hovertemplate=(
                    "<b>%{customdata[4]} transitions</b>"
                    "<br>Share: %{z:.1f}%"
                    "<br>x: %{customdata[0]:.0f}–%{customdata[1]:.0f}"
                    "<br>y: %{customdata[2]:.0f}–%{customdata[3]:.0f}"
                    "<br><i>Click to focus Sequence Explorer</i>"
                    "<extra></extra>"
                ),
                name=(
                    "Loss grid"
                    if location_kind == "loss"
                    else "Recovery grid"
                ),
            )
        )

        label_x = []
        label_y = []
        label_text = []
        label_custom = []

        for i, x in enumerate(centers):
            for j, y in enumerate(centers):
                count = int(heatmap[i, j])
                if count < 2:
                    continue

                label_x.append(float(x))
                label_y.append(float(y))
                label_text.append(f"{heatmap_pct[i, j]:.1f}%")
                label_custom.append([
                    float(bin_edges[i]),
                    float(bin_edges[i + 1]),
                    float(bin_edges[j]),
                    float(bin_edges[j + 1]),
                    count,
                ])

        if label_x:
            fig.add_trace(
                go.Scatter(
                    x=label_x,
                    y=label_y,
                    mode="text",
                    text=label_text,
                    customdata=label_custom,
                    textfont=dict(
                        color="white",
                        size=12,
                        family="Inter, Arial",
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    name="Cell labels",
                )
            )

        point_custom = []
        cell_size = 100.0 / int(grid_size)
        for x, y in zip(x_coords, y_coords):
            i = min(int(x / cell_size), int(grid_size) - 1)
            j = min(int(y / cell_size), int(grid_size) - 1)
            point_custom.append([
                float(bin_edges[i]),
                float(bin_edges[i + 1]),
                float(bin_edges[j]),
                float(bin_edges[j + 1]),
                int(heatmap[i, j]),
            ])

        fig.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers",
                marker=dict(
                    size=6,
                    color="#f4f8fa",
                    opacity=0.58,
                    line=dict(color="#17354d", width=0.7),
                ),
                text=hover_texts,
                customdata=point_custom,
                hovertemplate=(
                    "%{text}"
                    "<br>X: %{x:.1f}"
                    "<br>Y: %{y:.1f}"
                    "<br><i>Click to focus Sequence Explorer</i>"
                    "<extra></extra>"
                ),
                name=(
                    "Possession loss"
                    if location_kind == "loss"
                    else "Recovery"
                ),
                showlegend=False,
            )
        )
    else:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No transition locations to plot",
            showarrow=False,
            font=dict(color="#d8e5eb", size=13),
        )

    add_attacking_direction(fig, dark=True)

    fig.update_layout(
        title=None,
        plot_bgcolor="#27343e",
        paper_bgcolor="#27343e",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            range=[0, 100],
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False,
            fixedrange=True,
            range=[0, 100],
            scaleanchor="x",
            scaleratio=0.68,
        ),
        margin=dict(l=8, r=8, t=12, b=8),
        height=540,
        showlegend=False,
        clickmode="event+select",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="#dbe7ed",
            font=dict(color="#17354d", size=12),
        ),
    )

    return fig
