# src/visualization/set_piece_plots.py

import plotly.graph_objects as go

from .buildup_plotly import draw_plotly_pitch
from src.visualization.plotly_branding import (
    MATCH_AWAY_CYAN,
    MATCH_PITCH_LINE,
    MATCH_PITCH_MUTED,
    MATCH_WARNING,
    add_attacking_direction,
    add_zero_state,
    apply_match_pitch_layout,
)

# Outcome-specific accents; anything not listed here (e.g. "Possession
# Retained") falls back to the team's own colour.
_OUTCOME_COLORS = {
    "Goal": MATCH_WARNING,
    "Shot": MATCH_AWAY_CYAN,
    "Possession Lost": MATCH_PITCH_MUTED,
}


def plot_set_piece_map(df_analyzed, team_color):
    """
    Creates a Plotly map of offensive set pieces.
    """
    fig = go.Figure()
    draw_plotly_pitch(fig, line_color=MATCH_PITCH_LINE)
    add_attacking_direction(fig, dark=True)

    if df_analyzed.empty:
        add_zero_state(fig, "No set piece data to display")
    else:
        seen_outcomes = set()
        for _, sp in df_analyzed.iterrows():
            outcome = sp['Outcome']
            color = _OUTCOME_COLORS.get(outcome, team_color)
            # Only the first delivery of each outcome gets a legend entry --
            # one row per set piece would otherwise repeat the same label.
            show_legend_entry = outcome not in seen_outcomes
            seen_outcomes.add(outcome)

            fig.add_trace(go.Scatter(
                x=[sp['x_start'], sp['x_end']],
                y=[sp['y_start'], sp['y_end']],
                mode='lines',
                line=dict(color=color, width=2.5, dash='solid' if outcome != 'Possession Lost' else 'dot'),
                hoverinfo='text',
                hovertext=f"<b>{sp['Player']}</b><br>{sp['Action Type']} to {outcome}",
                name=outcome,
                showlegend=show_legend_entry,
            ))
            # Draw marker at the start
            fig.add_trace(go.Scatter(
                x=[sp['x_start']],
                y=[sp['y_start']],
                mode='markers',
                marker=dict(color=color, size=7, line=dict(width=1, color='#ffffff')),
                hoverinfo='skip',
                showlegend=False
            ))

    apply_match_pitch_layout(
        fig,
        x_range=(48, 102),
        y_range=(-2, 102),
    )
    return fig
