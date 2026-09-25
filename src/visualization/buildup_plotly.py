import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ..config import LINE_COLOR, GREEN, VIOLET, CARRY_COLOR, SHOT_TYPES
from src.visualization.plotly_branding import (
    MATCH_PITCH_BG,
    MATCH_PITCH_LINE,
    MATCH_PITCH_TEXT,
    add_attacking_direction,
)

# UNSUCCESSFUL_COLOR (config.py) is black -- fine on the light pitches it
# was designed for, invisible on this chart's dark pitch. Local override
# for the one function below that's migrated to the dark convention.
_UNSUCCESSFUL_DARK = "#c9d6de"

# This is the helper function we created for the defender map. We can reuse it.
def draw_plotly_pitch(fig, *, line_color=None):
    """Helper function to draw an Opta pitch using Plotly shapes.

    ``line_color`` defaults to the legacy black-on-white ``LINE_COLOR`` so
    callers that haven't migrated to the shared dark Match Plot design
    system yet keep their current look. Migrated callers pass
    ``plotly_branding.MATCH_PITCH_LINE`` explicitly.
    """
    line_color = line_color or LINE_COLOR
    pitch_shapes = [
        # Outer lines & halfway line
        go.layout.Shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color=line_color, width=2)),
        go.layout.Shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color=line_color, width=2)),
        # Center circle
        go.layout.Shape(type="circle", x0=42, y0=42, x1=58, y1=58, line=dict(color=line_color, width=2)),
        go.layout.Shape(type="circle", x0=49.5, y0=49.5, x1=50.5, y1=50.5, line=dict(color=line_color, width=2), fillcolor=line_color),
        # Penalty Areas
        go.layout.Shape(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=line_color, width=2)),
        go.layout.Shape(type="rect", x0=83.5, y0=21.1, x1=100, y1=78.9, line=dict(color=line_color, width=2)),
        # 6-yard boxes
        go.layout.Shape(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=line_color, width=2)),
        go.layout.Shape(type="rect", x0=94.5, y0=36.8, x1=100, y1=63.2, line=dict(color=line_color, width=2)),
    ]
    fig.update_layout(shapes=pitch_shapes)
    return fig

def plot_buildup_sequence_plotly(sequence_data, team_color, is_away):
    """
    Plots a single, complete buildup sequence interactively using Plotly.
    """
    fig = go.Figure()
    fig = draw_plotly_pitch(fig, line_color=MATCH_PITCH_LINE)
    add_attacking_direction(fig, dark=True)

    if sequence_data is None or sequence_data.empty:
        fig.add_annotation(x=50, y=50, text="No Sequence Data", showarrow=False, font=dict(size=16, color="#e07a6a"))
        return fig # Return the empty figure

    df = sequence_data.copy()
    # Data cleaning
    for col in ['x', 'y', 'end_x', 'end_y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['x', 'y'], inplace=True)
    if df.empty:
        fig.add_annotation(x=50, y=50, text="Invalid Coordinates", showarrow=False, font=dict(size=16, color="#e07a6a"))
        return fig

    # --- Plot Player Nodes First ---
    # Filter for events that should have a starting node
    node_events = df[df['type_name'].isin(['Pass'] + SHOT_TYPES)]
    for i, row in node_events.iterrows():
        jersey = row.get('Mapped Jersey Number', '')
        jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
        hover_text = f"<b>{row.playerName} (#{jersey_text})</b><br>{row.type_name}"
        fig.add_trace(go.Scatter(
            x=[row.x], y=[row.y], mode='markers+text',
            marker=dict(size=25, color=team_color, line=dict(color='white', width=2)),
            text=jersey_text, textfont=dict(color='white', size=10, family='Arial Black'),
            hovertext=hover_text, hoverinfo='text', showlegend=False
        ))

    # --- Plot Event Lines (Passes, Carries, Shots) ---
    last_x, last_y = None, None
    for i, row in df.iterrows():
        # Draw Carry Line
        if last_x is not None and pd.notna(row.x) and np.sqrt((row.x - last_x)**2 + (row.y - last_y)**2) > 1:
            fig.add_trace(go.Scatter(
                x=[last_x, row.x], y=[last_y, row.y], mode='lines',
                line=dict(color=CARRY_COLOR, width=2, dash='dash'),
                hoverinfo='none', showlegend=False
            ))

        # --- Draw Main Event Line ---
        if pd.notna(row.end_x) and pd.notna(row.end_y):
            event_type = row.type_name
            is_successful = row.outcome == 'Successful'
            is_last_event = (i == df.index[-1])
            
            # Default style
            line_color = '#a3a3a3'
            line_width = 3
            
            if event_type == 'Pass':
                line_color = team_color if is_successful else _UNSUCCESSFUL_DARK
                # Add end-of-pass marker if it's not the final event
                if is_successful and not is_last_event:
                     fig.add_trace(go.Scatter(
                        x=[row.end_x], y=[row.end_y], mode='markers',
                        marker=dict(size=5, color=line_color),
                        hoverinfo='none', showlegend=False
                    ))
            elif event_type in SHOT_TYPES:
                outcome_colors = {'Miss': 'grey', 'Attempt Saved': 'blue', 'Goal': GREEN, 'Post': 'orange'}
                line_color = outcome_colors.get(event_type, 'red')
                line_width = 5
            elif event_type == 'Offside Pass':
                line_color = VIOLET

            # Add arrow for the event
            fig.add_annotation(
                x=row.end_x, y=row.end_y, ax=row.x, ay=row.y,
                xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=line_width,
                arrowcolor=line_color
            )
            
            # Add special markers for terminating events
            if not is_successful or is_last_event:
                marker_symbol = 'x' if not is_successful else 'circle-open'
                marker_color = _UNSUCCESSFUL_DARK if not is_successful else line_color
                # Add terminating marker
                fig.add_trace(go.Scatter(
                    x=[row.end_x], y=[row.end_y], mode='markers',
                    marker=dict(symbol=marker_symbol, size=10, color=marker_color, line=dict(width=2, color=MATCH_PITCH_BG)),
                    hoverinfo='none', showlegend=False
                ))

        last_x, last_y = row.end_x, row.end_y
        
    # --- Final Layout ---
    outcome = df['sequence_outcome_type'].iloc[-1]
    pass_count = df['buildup_pass_count'].iloc[-1]
    
    fig.update_layout(
        title=f"Outcome: {outcome} ({pass_count} Passes)",
        title_font_color=MATCH_PITCH_TEXT, title_x=0.5,
        plot_bgcolor=MATCH_PITCH_BG, paper_bgcolor=MATCH_PITCH_BG,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor="x", scaleratio=0.68),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    # Match Analysis coordinates are already team-relative.
    # Home/away never changes pitch geometry.
    del is_away
    fig.update_xaxes(range=[0, 100])
    fig.update_yaxes(range=[0, 100])
        
    return fig

