# src/visualization/cross_plots.py
import plotly.graph_objects as go
from .buildup_plotly import draw_plotly_pitch # Riusiamo il disegnatore
import pandas as pd
import numpy as np
from plotly.colors import sample_colorscale
from .plotly_branding import (
    apply_dark_pitch_layout,
)

def plot_cross_heatmap(
    df_analyzed,
    location_type='origin',
    is_away=False,
    grid_size=6,
    selected_cross_id=None,
    selected_flow_route=None,
):
    """
    Plot cross origin or destination locations.

    All teams are displayed attacking towards x=100.
    Home uses the coral palette and Away the cyan palette.

    The heatmap always represents the full filtered sample.
    When one cross is selected, it is highlighted on top
    without removing the distribution context.
    """

    fig = go.Figure()

    draw_plotly_pitch(fig)

    # Make the pitch compatible with the dark branded canvas.
    fig.update_shapes(
        line_color='rgba(255,255,255,0.48)',
        line_width=1.1,
    )

    if (
        df_analyzed is None
        or df_analyzed.empty
    ):
        fig.add_annotation(
            x=50,
            y=50,
            text="No cross data to plot",
            showarrow=False,
            font=dict(
                color='rgba(255,255,255,0.68)',
                size=14,
            ),
        )

        fig.update_layout(
            xaxis=dict(
                range=[0, 100],
                visible=False,
                fixedrange=True,
            ),
            yaxis=dict(
                range=[0, 100],
                visible=False,
                fixedrange=True,
            ),
        )

        apply_dark_pitch_layout(
            fig,
            height=520,
            top_margin=35,
            showlegend=False,
        )

        return fig

    # ---------------------------------------------------------
    # COORDINATES
    # ---------------------------------------------------------

    if location_type == 'origin':
        x_column = 'x'
        y_column = 'y'
    else:
        x_column = 'end_x'
        y_column = 'end_y'

    df_plot = df_analyzed.copy()

    df_plot[x_column] = pd.to_numeric(
        df_plot[x_column],
        errors='coerce',
    )

    df_plot[y_column] = pd.to_numeric(
        df_plot[y_column],
        errors='coerce',
    )

    df_plot = df_plot[
        df_plot[x_column].between(
            0,
            100,
            inclusive='both',
        )
        & df_plot[y_column].between(
            0,
            100,
            inclusive='both',
        )
    ].copy()

    if df_plot.empty:
        fig.add_annotation(
            x=50,
            y=50,
            text="No valid cross locations",
            showarrow=False,
            font=dict(
                color='rgba(255,255,255,0.68)',
                size=14,
            ),
        )

        fig.update_layout(
            xaxis=dict(
                range=[0, 100],
                visible=False,
                fixedrange=True,
            ),
            yaxis=dict(
                range=[0, 100],
                visible=False,
                fixedrange=True,
            ),
        )

        apply_dark_pitch_layout(
            fig,
            height=520,
            top_margin=35,
            showlegend=False,
        )

        return fig

    # Coordinates are already team-normalised.
    # Do not mirror the Away team.

    # ---------------------------------------------------------
    # TEAM PALETTE
    # ---------------------------------------------------------

    if is_away:
        colorscale = [
            [0.00, '#eaf8fb'],
            [0.45, '#80d0df'],
            [0.72, '#36b4cf'],
            [1.00, '#087f9f'],
        ]

        team_color = '#13a7c7'

    else:
        colorscale = [
            [0.00, '#fff0ec'],
            [0.45, '#f3aa99'],
            [0.72, '#ee765f'],
            [1.00, '#d84f39'],
        ]

        team_color = '#ef6652'

    # ---------------------------------------------------------
    # BINNED DISTRIBUTION
    # ---------------------------------------------------------

    bin_edges = np.linspace(
        0,
        100,
        grid_size + 1,
    )

    heatmap, _, _ = np.histogram2d(
        df_plot[x_column],
        df_plot[y_column],
        bins=[
            bin_edges,
            bin_edges,
        ],
    )

    total = int(
        heatmap.sum()
    )

    heatmap_pct = (
        heatmap / total * 100
        if total > 0
        else heatmap
    )

    max_pct = max(
        float(
            heatmap_pct.max()
        ),
        1.0,
    )

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

            if count == 0:
                continue

            percentage = float(
                heatmap_pct[i, j]
            )

            intensity = min(
                percentage / max_pct,
                1.0,
            )

            color = sample_colorscale(
                colorscale,
                [intensity],
            )[0]

            polygon = rectangle(
                x0,
                x1,
                y0,
                y1,
            )

            fig.add_trace(
                go.Scatter(
                    x=polygon['x'],
                    y=polygon['y'],

                    fill='toself',

                    mode='lines',

                    fillcolor=color,

                    line=dict(
                        color=(
                            'rgba(255,255,255,0.12)'
                        ),
                        width=1,
                    ),

                    text=(
                        f"{count} crosses"
                        f"<br>{percentage:.1f}% "
                        "of filtered crosses"
                    ),

                    hovertemplate=(
                        "%{text}"
                        "<extra></extra>"
                    ),

                    showlegend=False,
                )
            )

            # One cross remains visible as an individual
            # point only. Label cells from 2 crosses upward.
            if count >= 2:

                cx = (
                    x0 + x1
                ) / 2

                cy = (
                    y0 + y1
                ) / 2

                text_color = (
                    '#ffffff'
                    if intensity >= 0.45
                    else '#17354d'
                )

                fig.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],

                        mode='text',

                        text=[
                            f"{percentage:.0f}%"
                        ],

                        textfont=dict(
                            size=12,
                            color=text_color,
                            weight='bold',
                        ),

                        hoverinfo='skip',
                        showlegend=False,
                    )
                )

    # ---------------------------------------------------------
    # INDIVIDUAL CROSSES
    # ---------------------------------------------------------

    hover_texts = []

    for _, row in df_plot.iterrows():

        player = row.get(
            'playerName',
            'Unknown',
        )

        foot = row.get(
            'Foot',
            'Unknown',
        )

        swing = row.get(
            'Swing',
            'N/A',
        )

        play_type = row.get(
            'Play Type',
            'Unknown',
        )

        outcome = row.get(
            'Outcome',
            'Unknown',
        )

        origin = row.get(
            'Origin Zone',
            'Unknown',
        )

        destination = row.get(
            'Destination Zone',
            'Unknown',
        )

        hover_texts.append(
            (
                f"<b>{player}</b>"
                f"<br>{foot} foot · {swing}"
                f"<br>{play_type}"
                f"<br>{origin} → {destination}"
                f"<br>Outcome: {outcome}"
            )
        )

    fig.add_trace(
        go.Scatter(
            x=df_plot[x_column],
            y=df_plot[y_column],

            mode='markers',

            marker=dict(
                color='rgba(235,243,247,0.72)',
                size=6,
                opacity=0.80,

                line=dict(
                    color=team_color,
                    width=1.0,
                ),
            ),

            text=hover_texts,

            hovertemplate=(
                "%{text}"
                "<extra></extra>"
            ),

            customdata=df_plot['cross_id'],

            name='cross_points',

            showlegend=False,
        )
    )


    # ---------------------------------------------------------
    # SELECTED FLOW
    # ---------------------------------------------------------

    if (
        selected_flow_route
        and isinstance(
            selected_flow_route,
            dict,
        )
    ):
        selected_origin = (
            selected_flow_route.get(
                "origin"
            )
        )

        selected_destination = (
            selected_flow_route.get(
                "destination"
            )
        )

        if (
            selected_origin
            and selected_destination
            and "Origin Zone"
                in df_plot.columns
            and "Destination Zone"
                in df_plot.columns
        ):
            selected_flow = (
                df_plot[
                    (
                        df_plot[
                            "Origin Zone"
                        ]
                        == selected_origin
                    )
                    & (
                        df_plot[
                            "Destination Zone"
                        ]
                        == selected_destination
                    )
                ]
                .copy()
            )

            if not selected_flow.empty:
                fig.add_trace(
                    go.Scatter(
                        x=selected_flow[
                            x_column
                        ],
                        y=selected_flow[
                            y_column
                        ],
                        mode="markers",
                        marker=dict(
                            size=15,
                            color=(
                                "rgba(245,196,81,0.20)"
                            ),
                            line=dict(
                                color="#f5c451",
                                width=2.6,
                            ),
                        ),
                        customdata=(
                            selected_flow[
                                "cross_id"
                            ]
                        ),
                        hovertemplate=(
                            "<b>Selected cross flow</b>"
                            "<br>"
                            + str(
                                selected_origin
                            )
                            + " → "
                            + str(
                                selected_destination
                            )
                            + "<extra></extra>"
                        ),
                        name="selected_flow_points",
                        showlegend=False,
                    )
                )

                fig.add_annotation(
                    x=2,
                    y=103,
                    text=(
                        "<b>SELECTED FLOW</b> · "
                        + str(
                            selected_origin
                        )
                        + " → "
                        + str(
                            selected_destination
                        )
                        + " · "
                        + str(
                            len(
                                selected_flow
                            )
                        )
                        + " crosses"
                    ),
                    showarrow=False,
                    xanchor="left",
                    font=dict(
                        color="#f5c451",
                        size=10,
                    ),
                    bgcolor=(
                        "rgba(10,52,78,0.86)"
                    ),
                    bordercolor=(
                        "rgba(245,196,81,0.55)"
                    ),
                    borderwidth=1,
                    borderpad=4,
                )

    # ---------------------------------------------------------
    # SELECTED CROSS
    # ---------------------------------------------------------

    if selected_cross_id is not None:

        selected = df_plot[
            df_plot['cross_id']
            == selected_cross_id
        ]

        if not selected.empty:

            selected_row = (
                selected.iloc[0]
            )

            fig.add_trace(
                go.Scatter(
                    x=[
                        selected_row[
                            x_column
                        ]
                    ],
                    y=[
                        selected_row[
                            y_column
                        ]
                    ],

                    mode='markers',

                    marker=dict(
                        size=14,
                        color='#f5c451',

                        line=dict(
                            color='#17354d',
                            width=2.2,
                        ),
                    ),

                    hoverinfo='skip',

                    showlegend=False,
                )
            )

    # ---------------------------------------------------------
    # ATTACKING DIRECTION
    # ---------------------------------------------------------

    fig.add_annotation(
        x=98,
        y=103,
        text='<b>ATTACKING →</b>',
        showarrow=False,
        xanchor='right',

        font=dict(
            color='#94dbea',
            size=10,
        ),
    )

    # ---------------------------------------------------------
    # LAYOUT
    # ---------------------------------------------------------

    fig.update_layout(
        title=None,

        xaxis=dict(
            range=[0, 100],
            visible=False,
            fixedrange=True,
        ),

        yaxis=dict(
            range=[0, 100],
            visible=False,
            fixedrange=True,
        ),

        showlegend=False,
    )

    apply_dark_pitch_layout(
        fig,
        height=520,
        top_margin=32,
        showlegend=False,
    )

    return fig
