# src/visualization/offensive_transitions_plotly.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from .defensive_transitions_plotly import draw_plotly_pitch # Riusiamo il disegnatore del campo
from ..config import BG_COLOR
from src.visualization.plotly_branding import add_attacking_direction
from src.visualization.coordinate_contract import orient_point

def plot_recovery_heatmap_on_pitch(
    sequences,
    is_away=False,
    grid_size=6,
):
    """
    Plot where offensive transitions begin.

    Recovery locations are approximated using the starting
    location of the first action in each detected transition.

    Heatmap labels are shown only for bins containing at
    least two transitions to reduce visual noise.
    """

    x_coords = []
    y_coords = []
    hover_texts = []

    # ---------------------------------------------------------
    # RECOVERY LOCATIONS
    # ---------------------------------------------------------

    for seq in sequences:
        if seq.empty:
            continue

        first_event = seq.iloc[0]

        x = pd.to_numeric(
            first_event.get('x'),
            errors='coerce',
        )

        y = pd.to_numeric(
            first_event.get('y'),
            errors='coerce',
        )

        if pd.isna(x) or pd.isna(y):
            continue

        x, y = orient_point(
            float(x),
            float(y),
            is_away=is_away,
        )
        x_coords.append(float(x))
        y_coords.append(float(y))

        hover_texts.append(
            first_event.get(
                'sequence_outcome_type',
                'Unknown',
            )
        )

    # ---------------------------------------------------------
    # EMPTY STATE
    # ---------------------------------------------------------

    if not x_coords:
        fig = go.Figure()

        draw_plotly_pitch(fig)
        add_attacking_direction(fig, dark=False)

        fig.update_shapes(
            line_color='#708696',
            line_width=1.2,
        )

        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref='paper',
            yref='paper',
            text='No recovery data to plot',
            showarrow=False,
            font=dict(
                color='#647c8e',
                size=13,
            ),
        )

        return fig

    # ---------------------------------------------------------
    # HEATMAP
    # ---------------------------------------------------------

    bin_edges = np.linspace(
        0,
        100,
        grid_size + 1,
    )

    heatmap, _, _ = np.histogram2d(
        x_coords,
        y_coords,
        bins=[
            bin_edges,
            bin_edges,
        ],
    )

    total = heatmap.sum()

    heatmap_pct = (
        heatmap / total * 100
        if total > 0
        else heatmap
    )

    max_val = (
        heatmap_pct.max()
        if heatmap_pct.max() > 0
        else 1
    )

    # Match Analysis palette.
    if is_away:
        colorscale = [
            [0.0, '#edf9fc'],
            [0.5, '#74cbdc'],
            [1.0, '#0b88a8'],
        ]
    else:
        colorscale = [
            [0.0, '#fff1ed'],
            [0.5, '#f3a08d'],
            [1.0, '#e7644a'],
        ]

    def rectangle(
        x0,
        x1,
        y0,
        y1,
    ):
        return {
            'x': [
                x0,
                x1,
                x1,
                x0,
                x0,
            ],
            'y': [
                y0,
                y0,
                y1,
                y1,
                y0,
            ],
        }

    fig = go.Figure()

    draw_plotly_pitch(fig)
    add_attacking_direction(fig, dark=False)

    # Softer pitch lines, closer to the Match Analysis UI.
    fig.update_shapes(
        line_color='#718797',
        line_width=1.15,
    )

    for i, x0 in enumerate(
        bin_edges[:-1]
    ):
        x1 = bin_edges[i + 1]

        for j, y0 in enumerate(
            bin_edges[:-1]
        ):
            y1 = bin_edges[j + 1]

            count = int(
                heatmap[i, j]
            )

            perc = float(
                heatmap_pct[i, j]
            )

            if count == 0:
                continue

            intensity = min(
                perc / max_val,
                1.0,
            )

            color = sample_colorscale(
                colorscale,
                [intensity],
            )[0]

            poly = rectangle(
                x0,
                x1,
                y0,
                y1,
            )

            # -------------------------------------------------
            # BIN
            # -------------------------------------------------

            fig.add_trace(
                go.Scatter(
                    x=poly['x'],
                    y=poly['y'],
                    fill='toself',
                    mode='lines',
                    fillcolor=color,
                    line=dict(
                        color='rgba(75, 101, 119, 0.22)',
                        width=1,
                    ),
                    text=(
                        f'{count} transitions'
                        f'<br>{perc:.1f}%'
                    ),
                    hovertemplate=(
                        '%{text}'
                        '<extra></extra>'
                    ),
                    showlegend=False,
                )
            )

            # -------------------------------------------------
            # LABEL
            #
            # A single transition does not need a permanent
            # percentage printed inside the cell. It remains
            # available on hover.
            # -------------------------------------------------

            if count >= 2:

                cx = (
                    x0 + x1
                ) / 2

                cy = (
                    y0 + y1
                ) / 2

                # Dynamic contrast.
                text_color = (
                    '#ffffff'
                    if intensity >= 0.48
                    else '#17354d'
                )

                fig.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],
                        mode='text',
                        text=[
                            f'{perc:.1f}%'
                        ],
                        textfont=dict(
                            size=13,
                            color=text_color,
                            weight='bold',
                        ),
                        showlegend=False,
                        hoverinfo='skip',
                    )
                )

    # ---------------------------------------------------------
    # INDIVIDUAL RECOVERIES
    # ---------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(
                size=6,
                color='#17354d',
                opacity=0.72,
                line=dict(
                    color='#ffffff',
                    width=0.8,
                ),
            ),
            text=hover_texts,
            hovertemplate=(
                'Outcome: %{text}'
                '<br>X: %{x:.1f}'
                '<br>Y: %{y:.1f}'
                '<extra></extra>'
            ),
            name='Recovery',
            showlegend=False,
        )
    )

    # ---------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------

    fig.update_layout(
        title=None,

        plot_bgcolor='#ffffff',
        paper_bgcolor='#ffffff',

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
            scaleanchor='x',
            scaleratio=0.68,
        ),

        margin=dict(
            l=8,
            r=8,
            t=12,
            b=8,
        ),

        height=540,

        showlegend=False,
        hoverlabel=dict(
            bgcolor='#ffffff',
            bordercolor='#dbe7ed',
            font=dict(
                color='#17354d',
                size=12,
            ),
        ),
    )

    return fig