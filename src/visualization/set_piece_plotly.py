# src/visualization/set_piece_plots.py

import plotly.graph_objects as go
from .buildup_plotly import draw_plotly_pitch
from ..config import BG_COLOR, LINE_COLOR

def plot_set_piece_map(df_analyzed, team_color):
    """
    Creates a Plotly map of offensive set pieces.
    """
    fig = go.Figure()
    draw_plotly_pitch(fig)

    if df_analyzed.empty:
        fig.add_annotation(text="No set piece data to display", showarrow=False, font=dict(color='white'))
    else:
        # Define colors for outcomes
        outcome_colors = {
            'Goal': 'gold',
            'Shot': 'orange',
            'Possession Retained': team_color,
            'Possession Lost': 'grey'
        }
        
        for _, sp in df_analyzed.iterrows():
            color = outcome_colors.get(sp['Outcome'], 'white')
            
            # Draw line for the delivery
            fig.add_trace(go.Scatter(
                x=[sp['x_start'], sp['x_end']],
                y=[sp['y_start'], sp['y_end']],
                mode='lines',
                line=dict(color=color, width=2.5, dash='solid' if sp['Outcome'] != 'Possession Lost' else 'dot'),
                hoverinfo='text',
                hovertext=f"<b>{sp['Player']}</b><br>{sp['Action Type']} to {sp['Outcome']}",
                name=sp['Outcome'] # For legend grouping
            ))
            # Draw marker at the start
            fig.add_trace(go.Scatter(
                x=[sp['x_start']],
                y=[sp['y_start']],
                mode='markers',
                marker=dict(color=color, size=7, line=dict(width=1, color='black')),
                hoverinfo='skip',
                showlegend=False
            ))

    fig.update_layout(
        title="<b>Offensive Set Piece Map</b>",
        title_font_color='white', title_x=0.5,
        plot_bgcolor=BG_COLOR,
        paper_bgcolor=BG_COLOR,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(color='white'),
            bgcolor='rgba(0,0,0,0.3)'
        ),
        xaxis=dict(range=[48, 102]), # Focus on the offensive half
        yaxis=dict(range=[-2, 102])
    )
    return fig