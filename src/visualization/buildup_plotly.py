import pandas as pd
import numpy as np
import plotly.graph_objects as go
from ..config import BG_COLOR, LINE_COLOR, GREEN, VIOLET, CARRY_COLOR, SHOT_TYPES, UNSUCCESSFUL_COLOR

# This is the helper function we created for the defender map. We can reuse it.
def draw_plotly_pitch(fig):
    """Helper function to draw an Opta pitch using Plotly shapes."""
    pitch_shapes = [
        # Outer lines & halfway line
        go.layout.Shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color=LINE_COLOR, width=2)),
        go.layout.Shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color=LINE_COLOR, width=2)),
        # Center circle
        go.layout.Shape(type="circle", x0=42, y0=42, x1=58, y1=58, line=dict(color=LINE_COLOR, width=2)),
        go.layout.Shape(type="circle", x0=49.5, y0=49.5, x1=50.5, y1=50.5, line=dict(color=LINE_COLOR, width=2), fillcolor=LINE_COLOR),
        # Penalty Areas
        go.layout.Shape(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=LINE_COLOR, width=2)),
        go.layout.Shape(type="rect", x0=83.5, y0=21.1, x1=100, y1=78.9, line=dict(color=LINE_COLOR, width=2)),
        # 6-yard boxes
        go.layout.Shape(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=LINE_COLOR, width=2)),
        go.layout.Shape(type="rect", x0=94.5, y0=36.8, x1=100, y1=63.2, line=dict(color=LINE_COLOR, width=2)),
    ]
    fig.update_layout(shapes=pitch_shapes)
    return fig

def plot_buildup_sequence_plotly(sequence_data, team_color, is_away):
    """
    Plots a single, complete buildup sequence interactively using Plotly.
    """
    fig = go.Figure()
    fig = draw_plotly_pitch(fig)

    if sequence_data is None or sequence_data.empty:
        fig.add_annotation(x=50, y=50, text="No Sequence Data", showarrow=False, font=dict(size=16, color="red"))
        return fig # Return the empty figure

    df = sequence_data.copy()
    # Data cleaning
    for col in ['x', 'y', 'end_x', 'end_y']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['x', 'y'], inplace=True)
    if df.empty:
        fig.add_annotation(x=50, y=50, text="Invalid Coordinates", showarrow=False, font=dict(size=16, color="red"))
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
                line_color = team_color if is_successful else UNSUCCESSFUL_COLOR
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
                marker_color = 'black' if not is_successful else line_color
                # Add terminating marker
                fig.add_trace(go.Scatter(
                    x=[row.end_x], y=[row.end_y], mode='markers',
                    marker=dict(symbol=marker_symbol, size=10, color=marker_color, line=dict(width=2)),
                    hoverinfo='none', showlegend=False
                ))

        last_x, last_y = row.end_x, row.end_y
        
    # --- Final Layout ---
    outcome = df['sequence_outcome_type'].iloc[-1]
    pass_count = df['buildup_pass_count'].iloc[-1]
    
    fig.update_layout(
        title=f"Outcome: {outcome} ({pass_count} Passes)",
        title_font_color='black', title_x=0.5,
        plot_bgcolor=BG_COLOR, paper_bgcolor=BG_COLOR,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor="x", scaleratio=0.68),
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10)
    )

    if is_away:
        fig.update_xaxes(range=[100, 0])
        fig.update_yaxes(range=[100, 0])
    else:
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])
        
    return fig


# def plot_opponent_buildup_after_loss_plotly(
#     sequence_data,
#     team_that_lost_possession,
#     team_building_up,
#     color_for_buildup_team,
#     loss_sequence_id,
#     loss_zone,
#     is_buildup_team_away,
#     bg_color=BG_COLOR,
#     line_color=LINE_COLOR,
#     unsuccessful_pass_color=UNSUCCESSFUL_COLOR,
#     metric_to_analyze='defensive_actions' # Default value
# ):
#     import plotly.graph_objects as go
#     import numpy as np
#     import pandas as pd

#     # --- Draw pitch ---
#     fig = go.Figure()
#     fig = draw_plotly_pitch(fig)
#     # Add thirds lines
#     thirds = [100/3, 2*100/3]
#     for x in thirds:
#         fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=100, line=dict(color="grey", width=1, dash="dash"))

#     # The plot is ALWAYS oriented L->R. Data coordinates are normalized before plotting.
#     df = sequence_data.copy()
#     if is_buildup_team_away:
#         for col in ['x', 'end_x']:
#             df[col] = 100 - df[col]
#         for col in ['y', 'end_y', 'shot_end_y']:
#             if col in df.columns:
#                 df[col] = 100 - df[col]

#     # --- Data cleaning ---
#     if df.empty:
#         fig.add_annotation(x=50, y=50, text="No Sequence Data", showarrow=False, font=dict(size=16, color="red"))
#         return fig

#     for col in ['x', 'y', 'end_x', 'end_y']:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce')
#     df.dropna(subset=['x', 'y'], inplace=True)
    
#     if df.empty:
#         fig.add_annotation(x=50, y=50, text="Invalid Coordinates", showarrow=False, font=dict(size=16, color="red"))
#         return fig

#     # --- Extract time and trigger type for title ---
#     first_event = df.iloc[0]
#     trigger_col_prefix = 'trigger' if metric_to_analyze in ['buildup_phases', 'set_piece'] else 'loss'
    
#     time_min = first_event.get(f'timeMin_at_{trigger_col_prefix}')
#     time_sec = first_event.get(f'timeSec_at_{trigger_col_prefix}')
#     type_of_trigger_str = first_event.get(f'type_of_initial_{trigger_col_prefix}', 'Unknown Event')
    
#     time_str = "Time N/A"
#     if pd.notna(time_min) and pd.notna(time_sec):
#         time_str = f"{int(time_min)}'{int(time_sec):02d}\""

#     # --- Plot events ---
#     node_size = 22
#     pass_lw = 3
#     pass_alpha = 0.7
#     last_event_end_x, last_event_end_y = None, None
#     outcome_text = df.iloc[-1].get('sequence_outcome_type', 'Sequence End')
    
#     # --- START: Set Piece Specific Logic ---
#     if metric_to_analyze == 'set_piece':
#         icon_symbol, icon_text = 'circle-open', ''
#         if type_of_trigger_str == 'Corner Awarded':
#             icon_symbol, icon_text = 'star-open', 'C'
#         elif type_of_trigger_str == 'Foul':
#             icon_symbol, icon_text = 'cross-open', 'F'
#         elif type_of_trigger_str == 'Out':
#             icon_symbol, icon_text = 'square-open', 'T'
        
#         # Draw the special marker for the set piece location
#         fig.add_trace(go.Scatter(
#             x=[first_event['x']], y=[first_event['y']],
#             mode='markers+text',
#             marker=dict(symbol=icon_symbol, color=color_for_buildup_team, size=22, line=dict(width=2, color='white')),
#             text=icon_text, textfont=dict(color='white', size=11, family='Arial Black'),
#             hoverinfo='text', hovertext=f"<b>Set Piece Start</b><br>{type_of_trigger_str}",
#             showlegend=False
#         ))
#         # The first "action" of the sequence is the set piece delivery itself,
#         # so we will use its end coordinates as the start of the next carry.
#         if pd.notna(first_event.get('end_x')) and pd.notna(first_event.get('end_y')):
#             last_event_end_x = first_event['end_x']
#             last_event_end_y = first_event['end_y']
#     # --- END: Set Piece Specific Logic ---

#     for i, row in df.iterrows():
#         start_x, start_y = row['x'], row['y']
#         end_x, end_y = row.get('end_x'), row.get('end_y')
#         event_type = row['type_name']
#         is_successful = row.get('outcome') == 'Successful'
        
#         # Draw carry line from previous event to current
#         if last_event_end_x is not None:
#             dist = np.sqrt((start_x - last_event_end_x)**2 + (start_y - last_event_end_y)**2)
#             if dist > 0.5: # Only draw if it's a significant carry
#                 fig.add_trace(go.Scatter(
#                     x=[last_event_end_x, start_x], y=[last_event_end_y, start_y],
#                     mode='lines', line=dict(color=CARRY_COLOR, width=2, dash='dash'),
#                     hoverinfo='none', showlegend=False
#                 ))
        
#         # Player node
#         jersey = row.get('Mapped Jersey Number', '')
#         jersey_text = str(int(jersey)) if pd.notna(jersey) else ''
#         hover_text = f"<b>{row.get('playerName','')} (#{jersey_text})</b><br>{event_type}"
        
#         fig.add_trace(go.Scatter(
#             x=[start_x], y=[start_y], mode='markers+text',
#             marker=dict(size=node_size, color=color_for_buildup_team, line=dict(color='white', width=1.5)),
#             text=jersey_text, textfont=dict(color='white', size=10),
#             hovertext=hover_text, hoverinfo='text', showlegend=False
#         ))

#         # Event lines (Passes, Shots)
#         if event_type == 'Pass' and pd.notna(end_x):
#             line_color_pass = color_for_buildup_team if is_successful else unsuccessful_pass_color
#             fig.add_trace(go.Scatter(x=[start_x, end_x], y=[start_y, end_y], mode='lines', line=dict(color=line_color_pass, width=pass_lw), opacity=pass_alpha, hoverinfo='none', showlegend=False))
            
#             if not is_successful:
#                 fig.add_trace(go.Scatter(x=[end_x], y=[end_y], mode='markers', marker=dict(symbol='x', size=12, color='black'), hoverinfo='none', showlegend=False))

#         elif event_type in SHOT_TYPES:
#             shot_color = GREEN if event_type == 'Goal' else VIOLET
#             shot_end_x, shot_end_y = 100.0, row.get('shot_end_y', 50.0)
#             fig.add_trace(go.Scatter(x=[start_x, shot_end_x], y=[start_y, shot_end_y], mode='lines', line=dict(color=shot_color, width=3, dash='dash'), hoverinfo='none', showlegend=False))
#             fig.add_trace(go.Scatter(x=[shot_end_x], y=[shot_end_y], mode='markers', marker=dict(symbol='star', size=16, color=shot_color), hoverinfo='none', showlegend=False))

#         # Update position for next carry line
#         if is_successful and pd.notna(end_x):
#             last_event_end_x, last_event_end_y = end_x, end_y
#         else:
#             last_event_end_x, last_event_end_y = None, None # End of chain

#     # --- Title and Layout ---
#     title_map = {
#         'defensive_transitions': f"Defensive Transition from {type_of_trigger_str}",
#         'offensive_transitions': f"Offensive Transition from {type_of_trigger_str}",
#         'buildup_phases': f"Buildup from {type_of_trigger_str}",
#         'set_piece': f"Set Piece: {type_of_trigger_str}"
#     }
#     title_line1 = title_map.get(metric_to_analyze, "Sequence Analysis")
    
#     fig.update_layout(
#         title=f"<b>{title_line1}</b><br>Time: {time_str} | Outcome: {outcome_text}",
#         title_font=dict(color='white', size=16),
#         title_x=0.5,
#         plot_bgcolor=bg_color,
#         paper_bgcolor=bg_color,
#         showlegend=False,
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, range=[-5, 105]),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, range=[-5, 105], scaleanchor="x", scaleratio=0.68),
#         margin=dict(l=10, r=10, t=80, b=10)
#     )

#     return fig


def plot_opponent_buildup_after_loss_plotly(
    sequence_data,
    team_that_lost_possession,
    team_building_up,
    color_for_buildup_team,
    loss_sequence_id,
    loss_zone,
    is_buildup_team_away,
    bg_color=BG_COLOR,
    line_color=LINE_COLOR,
    unsuccessful_pass_color=UNSUCCESSFUL_COLOR,
    metric_to_analyze='defensive_actions'
):
    import plotly.graph_objects as go
    import numpy as np
    import pandas as pd

    # --- Draw pitch ---
    fig = go.Figure()
    # Draw pitch lines, thirds, etc.
    # (You can use your draw_plotly_pitch helper, then add thirds lines)
    fig = draw_plotly_pitch(fig)
    # Add thirds lines
    thirds = [100/3, 2*100/3]
    for x in thirds:
        fig.add_shape(type="line", x0=x, y0=0, x1=x, y1=100, line=dict(color="grey", width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=x, x1=100, y1=x, line=dict(color="grey", width=1, dash="dash"))

    # Invert axes for away team
    if is_buildup_team_away:
        x_range = [100, 0]
        y_range = [100, 0]
    else:
        x_range = [0, 100]
        y_range = [0, 100]

    fig.update_xaxes(
        range=x_range,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        fixedrange=True
    )
    fig.update_yaxes(
        range=y_range,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        fixedrange=True,
        scaleanchor="x",
        scaleratio=0.68
    )

    # --- Data cleaning ---
    if sequence_data is None or sequence_data.empty:
        fig.add_annotation(x=50, y=50, text="No Sequence Data", showarrow=False, font=dict(size=16, color="red"))
        return fig

    df = sequence_data.copy()
    time_cols_to_check = ['x', 'y', 'end_x', 'end_y', 'timeMin', 'timeSec']
    if metric_to_analyze in ('buildup_phases', 'set_piece'):
        time_cols_to_check.extend(['timeMin_at_trigger', 'timeSec_at_trigger'])
    else: # per defensive/offensive transitions
        time_cols_to_check.extend(['timeMin_at_loss', 'timeSec_at_loss'])

    for col in time_cols_to_check:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['x', 'y'], inplace=True)
    if df.empty:
        fig.add_annotation(x=50, y=50, text="Invalid Coordinates", showarrow=False, font=dict(size=16, color="red"))
        return fig
    
    # --- START: CALCOLO DURATA TRANSIZIONE ---
    duration_str = "N/A"
    time_str = "Time N/A"
    type_of_trigger_str = "Unknown"
    
    if not df.empty:
        first_event = df.iloc[0]
        last_event = df.iloc[-1]
        
        # --- START: LOGICA DI INIZIO FLESSIBILE ---
        if metric_to_analyze in ('buildup_phases', 'set_piece'):
            start_min = first_event.get('timeMin_at_trigger')
            start_sec = first_event.get('timeSec_at_trigger')
            type_of_trigger_str = first_event.get('type_of_initial_trigger', 'Unknown Event')
        else: # transitions
            start_min = first_event.get('timeMin_at_loss')
            start_sec = first_event.get('timeSec_at_loss')
            type_of_trigger_str = first_event.get('type_of_initial_loss', 'Unknown Loss')
        # --- END: LOGICA DI INIZIO FLESSIBILE ---

        end_min = last_event.get('timeMin')
        end_sec = last_event.get('timeSec')
        
        if pd.notna(start_min) and pd.notna(start_sec) and pd.notna(end_min) and pd.notna(end_sec):
            start_total_seconds = start_min * 60 + start_sec
            end_total_seconds = end_min * 60 + end_sec
            duration_seconds = end_total_seconds - start_total_seconds
            duration_str = f"{duration_seconds:.2f}s"
        
        if pd.notna(start_min) and pd.notna(start_sec):
            try:
                time_str = f"{int(start_min)}'{int(start_sec):02d}\""
            except Exception:
                pass
    # --- END: CALCOLO DURATA TRANSIZIONE ---

    # --- Extract time and loss type for title ---
    time_str = "Time N/A"
    type_of_loss_str = "Unknown Loss"
    type_of_trigger_str = 'Unknown Event'
    if not df.empty:
        first_event = df.iloc[0]
        if metric_to_analyze in ('buildup_phases', 'set_piece'):
            time_min_loss = first_event.get('timeMin_at_trigger')
            time_sec_loss = first_event.get('timeSec_at_trigger')
            type_of_loss_str = first_event.get('type_of_initial_trigger', 'Unknown Loss')
        else:
            time_min_loss = first_event.get('timeMin_at_loss')
            time_sec_loss = first_event.get('timeSec_at_loss')
            type_of_loss_str = first_event.get('type_of_initial_loss', 'Unknown Loss')
        if pd.notna(time_min_loss) and pd.notna(time_sec_loss):
            try:
                time_str = f"{int(time_min_loss)}'{int(time_sec_loss):02d}\""
            except Exception:
                pass

    # --- Plot events ---
    node_size = 22
    pass_lw = 3
    pass_alpha = 0.7
    last_event_end_x = None
    last_event_end_y = None
    outcome_text = "Sequence End"
    something_was_plotted = False

    # --- START: Set Piece Specific Logic ---
    if metric_to_analyze == 'set_piece':
        icon_symbol, icon_text = 'circle-open', ''
        if type_of_trigger_str == 'Corner Awarded':
            icon_symbol, icon_text = 'star-open', 'C'
        elif type_of_trigger_str == 'Foul':
            icon_symbol, icon_text = 'cross-open', 'F'
        elif type_of_trigger_str == 'Out':
            icon_symbol, icon_text = 'square-open', 'T'
        
        # Draw the special marker for the set piece location
        fig.add_trace(go.Scatter(
            x=[first_event['x']], y=[first_event['y']],
            mode='markers+text',
            marker=dict(symbol=icon_symbol, color=color_for_buildup_team, size=22, line=dict(width=2, color='white')),
            text=icon_text, textfont=dict(color='white', size=11, family='Arial Black'),
            hoverinfo='text', hovertext=f"<b>Set Piece Start</b><br>{type_of_trigger_str}",
            showlegend=False
        ))
        # The first "action" of the sequence is the set piece delivery itself,
        # so we will use its end coordinates as the start of the next carry.
        if pd.notna(first_event.get('end_x')) and pd.notna(first_event.get('end_y')):
            last_event_end_x = first_event['end_x']
            last_event_end_y = first_event['end_y']
    # --- END: Set Piece Specific Logic ---

    for i, row in df.iterrows():
        start_x, start_y = row['x'], row['y']
        end_x, end_y = row.get('end_x'), row.get('end_y')
        event_type = row['type_name']
        is_successful = row.get('outcome') == 'Successful'
        is_last_event = (i == df.index[-1])

        # Carry line from previous event's end to this event's start
        if last_event_end_x is not None and last_event_end_y is not None:
            dist = np.sqrt((start_x - last_event_end_x)**2 + (start_y - last_event_end_y)**2)
            if dist >= 0.5:
                fig.add_trace(go.Scatter(
                    x=[last_event_end_x, start_x], y=[last_event_end_y, start_y],
                    mode='lines',
                    line=dict(color=CARRY_COLOR, width=2, dash='dash'),
                    hoverinfo='none', showlegend=False
                ))

        # Node at start of each pass/shot
        jersey = row.get('Mapped Jersey Number', '')
        jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
        hover_text = f"<b>{row.get('playerName','')} (#{jersey_text})</b><br>{event_type}"
        if event_type == 'Pass' or event_type in SHOT_TYPES:
            fig.add_trace(go.Scatter(
                x=[start_x], y=[start_y], mode='markers+text',
                marker=dict(size=node_size, color=color_for_buildup_team, line=dict(color='white', width=1.5)),
                text=jersey_text, textfont=dict(color='white', size=10),
                hovertext=hover_text, hoverinfo='text', showlegend=False
            ))

        # Pass lines
        if event_type == 'Pass' and pd.notna(end_x) and pd.notna(end_y):
            line_color = color_for_buildup_team if is_successful else unsuccessful_pass_color
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[start_y, end_y],
                mode='lines',
                line=dict(color=line_color, width=pass_lw, dash='solid'),
                opacity=pass_alpha,
                hoverinfo='none', showlegend=False
            ))
            # End marker
            if is_successful and not is_last_event:
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[end_y], mode='markers',
                    marker=dict(size=7, color=line_color),
                    hoverinfo='none', showlegend=False
                ))
            elif not is_successful:
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[end_y], mode='markers',
                    marker=dict(symbol='x', size=12, color='black', line=dict(width=2)),
                    hoverinfo='none', showlegend=False
                ))
                if end_x >= 83 and (21.1 <= end_y <= 78.9): # If in the goal area
                        if metric_to_analyze == 'defensive_actions':
                            outcome_text = f"Chance conceded to {team_building_up}"
                        else: 
                            outcome_text = f"Big Chance"
                else:
                    if metric_to_analyze == 'defensive_actions': 
                        outcome_text = f"Possession regained"
                    else:
                        outcome_text = f"Possession lost"
                        
            if is_successful and is_last_event:
                receiver_jersey = row.get('receiver_jersey_number', '')
                receiver_text = str(int(receiver_jersey)) if pd.notna(receiver_jersey) and receiver_jersey != '' else ''
                fig.add_trace(go.Scatter(
                    x=[end_x], y=[end_y], mode='markers+text',
                    marker=dict(size=node_size, color=color_for_buildup_team, line=dict(color='white', width=1.5)),
                    text=receiver_text, textfont=dict(color='white', size=10),
                    hoverinfo='skip', showlegend=False
                ))
            something_was_plotted = True

        # Shots
        elif event_type in SHOT_TYPES:
            is_own_goal = row.get('Own goal') == 1
            if is_own_goal:
                shot_color = 'magenta' # Colore speciale per evidenziare l'autogol
            else: 
                outcome_colors = {'Miss': 'grey', 'Attempt Saved': 'blue', 'Goal': GREEN, 'Post': 'orange'}
                shot_color = outcome_colors.get(event_type, 'red')
            if is_own_goal:
                shot_end_x = 0.0
            else:
                shot_end_x = 100.0
            shot_end_y = row.get('shot_end_y', row.get('end_y', 50.0))
            fig.add_trace(go.Scatter(
                x=[start_x, shot_end_x], y=[start_y, shot_end_y],
                mode='lines',
                line=dict(color=shot_color, width=3),
                hoverinfo='none', showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=[shot_end_x], y=[shot_end_y], mode='markers',
                marker=dict(symbol='x', size=16, color=shot_color, line=dict(width=2)),
                hoverinfo='none', showlegend=False
            ))
            if event_type == 'Goal':
                if is_own_goal:
                    outcome_text = f"OWN GOAL"
                else:
                    outcome_text = f"Goal by {team_building_up}" if metric_to_analyze == 'defensive_actions' else "Goal"
            else:
                outcome_text = f"Shot conceded" if metric_to_analyze == 'defensive_actions' else "Shot"
            something_was_plotted = True

        elif event_type == 'Offside Pass':
            jersey = row.get('Mapped Jersey Number', '')
            jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
            hover_text = f"<b>{row.get('playerName','')} (#{jersey_text})</b><br>{event_type}"
            if jersey_text:
                fig.add_trace(go.Scatter(
                    x=[start_x], y=[start_y], mode='markers+text',
                    marker=dict(size=node_size, color=color_for_buildup_team, line=dict(color='white', width=1.5)),
                    text=jersey_text, textfont=dict(color='white', size=10),
                    hovertext=hover_text, hoverinfo='text', showlegend=False
                ))
            # Special color for offside
            fig.add_trace(go.Scatter(
                x=[start_x, end_x], y=[start_y, end_y],
                mode='lines',
                line=dict(color=VIOLET, width=pass_lw, dash='dot'),
                opacity=pass_alpha,
                hoverinfo='text',
                hovertext="Offside Pass",
                showlegend=False
            ))
            # Mark end with X
            fig.add_trace(go.Scatter(
                x=[end_x], y=[end_y], mode='markers',
                marker=dict(symbol='x', size=14, color=VIOLET, line=dict(width=2)),
                hovertext="Offside", hoverinfo='text', showlegend=False
            ))
            outcome_text = "Offside"

        elif event_type == 'Out':
            # Mark end with X
            fig.add_trace(go.Scatter(
                x=[end_x], y=[end_y], mode='markers',
                marker=dict(symbol='x', size=14, color='black', line=dict(width=2)),
                hovertext="Out", hoverinfo='text', showlegend=False
            ))
            outcome_text = "Out"

        elif event_type in ('Dispossessed', 'Ball touch', 'Take On'):
            jersey = row.get('Mapped Jersey Number', '')
            jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
            hover_text = f"<b>{row.get('playerName','')} (#{jersey_text})</b><br>{event_type}"
            if jersey_text:
                fig.add_trace(go.Scatter(
                    x=[start_x], y=[start_y], mode='markers+text',
                    marker=dict(symbol='square', size=node_size, color='grey', line=dict(color='white', width=1.5)),
                    text=jersey_text, textfont=dict(color='white', size=10),
                    hovertext=hover_text, hoverinfo='text', showlegend=False
                ))
            if metric_to_analyze == 'defensive_actions':
                outcome_text = f"Possession regained"
            else: 
                outcome_text = f"Possession lost"

        elif is_last_event:
            # Other terminating event (e.g., Foul)
            jersey = row.get('Mapped Jersey Number', '')
            jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
            hover_text = f"<b>{row.get('playerName','')} (#{jersey_text})</b><br>{event_type}"
            if jersey_text:
                fig.add_trace(go.Scatter(
                    x=[start_x], y=[start_y], mode='markers+text',
                    marker=dict(symbol='square', size=node_size, color='grey', line=dict(color='white', width=1.5)),
                    text=jersey_text, textfont=dict(color='white', size=10),
                    hovertext=hover_text, hoverinfo='text', showlegend=False
                ))
            fig.add_trace(go.Scatter(
                x=[start_x], y=[start_y], mode='markers',
                marker=dict(symbol='square', size=18, color='grey', line=dict(width=2)),
                hovertext=event_type, hoverinfo='text', showlegend=False
            ))
            outcome_text = event_type

        # Update last_event_end_x/y for next iteration's carry check
        if event_type == 'Pass' and is_successful and pd.notna(end_x) and pd.notna(end_y):
            last_event_end_x = end_x
            last_event_end_y = end_y
        else: # Sequence terminated or non-pass event
            last_event_end_x = None
            last_event_end_y = None

    # # --- Title ---
    # if metric_to_analyze == 'defensive_actions':
    #     title_each_plot = f"Defensive Transition due to {type_of_loss_str}"
    # elif metric_to_analyze == 'buildup_phases':
    #     title_each_plot = f"Offensive Buildup starting from {type_of_loss_str}"
    # else:
    #     title_each_plot = f"Offensive Transition thanks to {type_of_loss_str}"

    # title_parts = [
    #     f"Time: {time_str} - {title_each_plot}",
    #     f"Outcome: {outcome_text}",
    # ]
    # fig.update_layout(
    #     title="<br>".join(title_parts),
    #     title_font=dict(color='black', size=20),  # or 'white' if your background is dark
    #     title_x=0.5,
    #     plot_bgcolor=bg_color,
    #     paper_bgcolor=bg_color,
    #     showlegend=False,
    #     margin=dict(l=10, r=10, t=60, b=10)
    # )

    # --- Title and Layout (con aggiunta della durata) ---
    title_map = {
        'defensive_transitions': f"Transition Conceded after {type_of_loss_str}",
        'offensive_transitions': f"Offensive Transition from {type_of_loss_str}",
        'buildup_phases': f"Buildup from {type_of_loss_str}",
        'set_piece': f"Set Piece: {type_of_loss_str}"
    }
    title_line1 = title_map.get(metric_to_analyze, "Sequence Analysis")
    
    # --- START: AGGIORNAMENTO TITOLO ---
    fig.update_layout(
        title=f"<b>{title_line1}</b><br>Time: {time_str} | Duration: {duration_str} | Outcome: {outcome_text}",
        title_font=dict(color='black', size=18),
        title_x=0.5,
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        showlegend=False,
        margin=dict(l=10, r=10, t=80, b=10)
    )
    # --- END: AGGIORNAMENTO TITOLO ---

    return fig


import plotly.graph_objects as go
import pandas as pd


def plot_buildup_sequence_animated(df, team_color="#007BFF"):
    frames = []
    slider_steps = []
    all_traces = []

    fig = go.Figure()

    # --- 1. Disegna il campo (campo standard 100x100) ---
    fig.update_layout(
        xaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 100], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=0.68),
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "⏹ Stop",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                }
            ],
            "x": 0.05,
            "y": 1.05,
            "xanchor": "left",
            "yanchor": "top"
        }]
    )

    # --- 2. Eventi e animazione passo-passo ---
    for i in range(len(df)):
        current = df.iloc[:i+1]

        trace = go.Scatter(
            x=current["x"],
            y=current["y"],
            mode="lines+markers+text",
            line=dict(color=team_color, width=3),
            marker=dict(color=team_color, size=12, line=dict(color="white", width=2)),
            text=[
                str(int(j)) if not pd.isna(j) else ""
                for j in current.get("Mapped Jersey Number", [""])
            ],
            textposition="middle center",
            hoverinfo="text",
            hovertext=[
                f"{row['type_name']} ({row['playerName']})" if 'playerName' in row else row['type_name']
                for _, row in current.iterrows()
            ],
            showlegend=False
        )

        frames.append(go.Frame(data=[trace], name=str(i)))

        slider_steps.append({
            "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
            "label": f"{i+1}",
            "method": "animate"
        })

    # --- 3. Slider ---
    fig.update_layout(
        sliders=[{
            "steps": slider_steps,
            "transition": {"duration": 0},
            "x": 0.05,
            "y": -0.08,
            "len": 0.9
        }]
    )

    if frames:
        fig.add_trace(frames[0].data[0])
        fig.frames = frames

    return fig