# In src/visualization/team_plots.py
from collections import Counter

import plotly.graph_objects as go
import pandas as pd
from src.utils import formation_layouts
from src.metrics.sportmonks import LOWER_IS_BETTER, TEAM_RADAR_GROUPS, uses_sportmonks


def create_team_profile_radar(
    df_for_normalization,
    primary_team_name,
    comparison_team_name=None,
    template="plotly_white",
):
    """Create a true percentile radar against the supplied comparison cohort."""
    if uses_sportmonks(df_for_normalization):
        categories = TEAM_RADAR_GROUPS
        inverted_columns = LOWER_IS_BETTER
    else:
        categories = {
            'Attacking': {'Goals': 'Gls', 'Goal Conversion %': 'Goal_Conversion', 'Total Shots': 'Sh', 'xG per Shot': 'xG_per_Shot'},
            'Possession & Style': {'Possession %': 'Poss', 'Passing Tempo': 'Passing_Tempo', 'Progressions / Touch': 'Progressions_per_Touch', 'Take-On Success %': 'TakeOn_Success_Rate'},
            'Defending': {'Goals Conceded': 'GA', 'Tackles Won': 'TklW', 'Interceptions': 'Int', 'Aerial Duels Won %': 'Aerial_Duels_Won_Perc'}
        }
        inverted_columns = {'GA'}

    metric_pairs = [item for group in categories.values() for item in group.items()]
    radar_metrics_ordered = [display_name for display_name, _ in metric_pairs]
    df_norm = df_for_normalization.copy()
    raw_columns = {}
    for display_name, column_name in metric_pairs:
        raw_columns[display_name] = column_name
        if column_name not in df_norm.columns:
            df_norm[display_name] = float("nan")
            continue
        values = pd.to_numeric(df_norm[column_name], errors="coerce")
        # rank(pct=True) gives a genuine cohort percentile. Reversing the
        # ranking direction makes a lower raw value a higher percentile.
        df_norm[display_name] = values.rank(
            method="average",
            pct=True,
            ascending=column_name not in inverted_columns,
        ) * 100

    fig = go.Figure()
    category_colors = {
        "Attacking": "rgba(233, 106, 74, 0.10)",
        "Possession & Territory": "rgba(21, 151, 194, 0.10)",
        "Possession & Style": "rgba(21, 151, 194, 0.10)",
        "Defending": "rgba(45, 155, 112, 0.10)",
        "Defending & Pressing": "rgba(45, 155, 112, 0.10)",
        "Set Pieces": "rgba(139, 107, 214, 0.10)",
    }
    bar_widths = []
    bar_colors = []
    for category, metrics in categories.items():
        bar_widths.extend([1] * len(metrics))
        bar_colors.extend([category_colors.get(category, "rgba(98, 116, 138, 0.08)")] * len(metrics))

    fig.add_trace(go.Barpolar(
        r=[100] * len(radar_metrics_ordered),
        theta=radar_metrics_ordered,
        width=bar_widths,
        marker_color=bar_colors,
        marker_line=dict(color="rgba(220, 229, 238, 0.9)", width=1),
        hoverinfo='none',
        showlegend=False,
    ))

    teams_to_plot = [primary_team_name]
    if comparison_team_name:
        teams_to_plot.append(comparison_team_name)

    trace_styles = (
        {"color": "#087ea4", "fillcolor": "rgba(8, 126, 164, 0.18)"},
        {"color": "#e96a4a", "fillcolor": "rgba(233, 106, 74, 0.13)"},
    )
    for index, team_name in enumerate(teams_to_plot):
        team_data = df_norm[df_norm['Squad'] == team_name]
        if team_data.empty:
            continue
        row = team_data.iloc[0]
        values = [row.get(metric) for metric in radar_metrics_ordered]
        values = [50.0 if pd.isna(value) else float(value) for value in values]
        raw_values = [row.get(raw_columns[metric]) for metric in radar_metrics_ordered]
        raw_labels = ["N/A" if pd.isna(value) else f"{float(value):,.2f}" for value in raw_values]
        style = trace_styles[min(index, len(trace_styles) - 1)]
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=radar_metrics_ordered + [radar_metrics_ordered[0]],
            customdata=raw_labels + [raw_labels[0]],
            fill="toself",
            name=team_name,
            line=dict(color=style["color"], width=3),
            fillcolor=style["fillcolor"],
            marker=dict(size=5, color=style["color"]),
            hovertemplate=(
                "<b>%{theta}</b><br>"
                "Percentile: %{r:.0f}<br>"
                "Raw value: %{customdata}<extra>%{fullData.name}</extra>"
            ),
        ))

    fig.update_layout(
        height=680,
        template=template,
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=90, r=90, t=20, b=30),
        font=dict(family="Inter, Segoe UI, sans-serif", color="#172b4d"),
        polar=dict(
            # Reserve a dedicated top band for title and legend. Without an
            # explicit domain they occupy the same space as the top metric.
            domain=dict(x=[0, 1], y=[0, 0.76]),
            bgcolor="#ffffff",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                ticks="",
                gridcolor="#dce5ee",
                linecolor="#dce5ee",
                angle=90,
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,
                gridcolor="#dce5ee",
                linecolor="#dce5ee",
                tickfont=dict(size=11, color="#40556d"),
            ),
        ),
        legend=dict(
            yanchor="top",
            y=0.91,
            xanchor="center",
            x=0.5,
            orientation="h",
            font=dict(size=13, color="#172b4d"),
        ),
        title=dict(
            text="Percentile profile",
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
            font=dict(size=18, color="#172b4d"),
        ),
    )
    return fig


def plot_typical_formation_plotly(
    formation_name,
    lineup,
    team_color="#e96a4a",
    height=610,
):
    """Plot a representative starting XI using Sportmonks formation slots."""
    pitch_width = 68
    pitch_length = 105
    pitch_color = "#123f49"
    line_color = "rgba(255,255,255,0.55)"
    figure = go.Figure()

    shapes = [
        dict(type="rect", x0=0, y0=0, x1=pitch_width, y1=pitch_length),
        dict(type="line", x0=0, y0=pitch_length / 2, x1=pitch_width, y1=pitch_length / 2),
        dict(type="circle", x0=24.85, y0=43.35, x1=43.15, y1=61.65),
        dict(type="rect", x0=13.84, y0=0, x1=54.16, y1=16.5),
        dict(type="rect", x0=24.84, y0=0, x1=43.16, y1=5.5),
        dict(type="rect", x0=13.84, y0=88.5, x1=54.16, y1=105),
        dict(type="rect", x0=24.84, y0=99.5, x1=43.16, y1=105),
    ]
    for shape in shapes:
        shape["line"] = dict(color=line_color, width=1.4)
    figure.update_layout(shapes=shapes)

    if lineup:
        maximum_row = max(player.get("row", 1) for player in lineup)
        row_sizes = Counter(player.get("row", 1) for player in lineup)
        x_values = []
        y_values = []
        labels = []
        hover_names = []
        starts = []
        for player in lineup:
            row_number = player.get("row", 1)
            column_number = player.get("column", 1)
            row_size = row_sizes[row_number]
            x_values.append(pitch_width * column_number / (row_size + 1))
            y_values.append(
                8
                if maximum_row <= 1
                else 8 + ((row_number - 1) / (maximum_row - 1)) * 88
            )
            full_name = str(player.get("player_name", ""))
            labels.append(full_name.replace("\u00a0", " ").split()[-1] if full_name else "")
            hover_names.append(full_name)
            starts.append(player.get("starts", 0))
        figure.add_trace(
            go.Scatter(
                x=x_values,
                y=y_values,
                mode="markers+text",
                text=labels,
                textposition="bottom center",
                textfont=dict(color="#ffffff", size=11),
                customdata=list(zip(hover_names, starts)),
                marker=dict(
                    size=32,
                    color=team_color,
                    line=dict(color="#ffffff", width=2.5),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Starts in this slot: %{customdata[1]}"
                    "<extra></extra>"
                ),
                cliponaxis=False,
                showlegend=False,
            )
        )

    figure.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=12, b=12),
        paper_bgcolor=pitch_color,
        plot_bgcolor=pitch_color,
        font=dict(family="Inter, Segoe UI, sans-serif", color="#ffffff"),
        xaxis=dict(
            range=[-5, pitch_width + 5],
            visible=False,
            fixedrange=True,
            constrain="domain",
        ),
        yaxis=dict(
            range=[-4, pitch_length + 4],
            visible=False,
            fixedrange=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        annotations=[
            dict(
                x=pitch_width - 2,
                y=2,
                text=formation_name,
                showarrow=False,
                xanchor="right",
                yanchor="bottom",
                font=dict(size=14, color="rgba(255,255,255,0.55)"),
            )
        ],
    )
    return figure



def _draw_pitch(fig, bg_color='#2E3439', line_color='rgba(255,255,255,0.5)'):
    """Adds football pitch shapes to a Plotly figure."""
    PITCH_WIDTH_OPTA, PITCH_HEIGHT_OPTA = 100, 100

    # Bordi e linea di metà campo
    fig.add_shape(type="rect", x0=0, y0=0, x1=PITCH_WIDTH_OPTA, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2))
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2))

    # Cerchio di centrocampo
    fig.add_shape(type="circle", x0=40.5, y0=40.5, x1=59.5, y1=59.5, line=dict(color=line_color, width=2))

    # Aree di rigore
    fig.add_shape(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=100, y0=21.1, x1=83.5, y1=78.9, line=dict(color=line_color, width=2))

    # Aree piccole
    fig.add_shape(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=line_color, width=2))
    fig.add_shape(type="rect", x0=100, y0=36.8, x1=94.5, y1=63.2, line=dict(color=line_color, width=2))

    fig.update_layout(
        xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        showlegend=False,
    )
    return fig

# --- FUNZIONE DI PLOT FORMAZIONE AGGIORNATA ---
def plot_most_used_formation_plotly(formation_id, team_color='skyblue', template="plotly_dark", height=400):
    """
    Creates a Plotly figure of a football pitch with dots representing a formation.
    Accepts a height parameter for flexible sizing.
    """

    # ... (il corpo della funzione è quasi identico, cambia solo l'uso di 'height')
    fig = go.Figure()
    fig = _draw_pitch(fig)

    coords = []
    if formation_id in formation_layouts.FORMATION_COORDINATES:
        formation_map = formation_layouts.FORMATION_COORDINATES[formation_id]
        for pos_num in range(1, 12):
            coord = formation_map.get(pos_num)
            if coord:
                coords.append(coord)

    if coords:
        x_coords, y_coords = zip(*coords)
        fig.add_trace(go.Scatter(
            x=x_coords,
            y=y_coords,
            mode='markers',
            marker=dict(color=team_color, size=20, line=dict(width=2, color='white')),
            hoverinfo='none'
        ))

    fig.update_layout(
        height=height, # <--- Usa il parametro height
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig
