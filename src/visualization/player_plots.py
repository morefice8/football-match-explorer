# src/visualization/player_plots.py
from dash import Dash, html as dash_html, dcc, Input, Output, dash_table, no_update
from dash.dependencies import State
import dash_bootstrap_components as dbc
import traceback
from datetime import datetime
import html
import dash
from dash.dependencies import ALL

import json
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mplsoccer import Pitch, FontManager
import matplotlib.patheffects as path_effects
import plotly.graph_objects as go
from . import pitch_plots
from src.metrics import player_metrics 

# Import config for colors if not passed directly
from src import config
# Import helper if needed (e.g., get_shorter_name)
# from src.data_processing.preprocess import get_shorter_name # Or wherever it lives

# Define colors (get from config or define defaults)
VIOLET = config.VIOLET
GREEN = config.GREEN
HCOL = config.DEFAULT_HCOL # Assuming these exist in config
ACOL = config.DEFAULT_ACOL
BG_COLOR = config.BG_COLOR
LINE_COLOR = config.LINE_COLOR

# --- Plotting Functions for Dashboard Components ---

def plot_shot_sequence_bar(ax, player_stats_df, num_players=10, title="Shot Sequence Involvement"):
    """Plots a stacked bar chart for shot sequence involvement for top players."""
    print(f"Plotting {title} bar chart...")
    # Ensure required columns exist
    req_cols = ['Shots', 'Shot Assists', 'Buildup to Shot', 'Shooting Seq Total']
    if not all(col in player_stats_df.columns for col in req_cols):
        print(f"Warning: Missing required columns for shot sequence plot. Skipping.")
        ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center', fontsize=12, color='red')
        return

    # Get top N players based on the total shooting sequence involvement
    top_players_df = player_stats_df.sort_values('Shooting Seq Total', ascending=False).head(num_players)
    # Sort back by index/original order potentially? Or keep sorted by total? Let's keep sorted for ranking.
    # Reverse for plotting (top player at the top of the bar chart)
    top_players_df = top_players_df.iloc[::-1]

    players = top_players_df.index.tolist()
    shots = top_players_df['Shots'].tolist()
    shot_assists = top_players_df['Shot Assists'].tolist()
    buildup = top_players_df['Buildup to Shot'].tolist()

    # Calculate left offsets for stacking
    left_sa = np.array(shots)
    left_buildup = left_sa + np.array(shot_assists)

    # Plot bars
    ax.barh(players, shots, label='Shot', color=HCOL, left=0, zorder=3) # Use a distinct color
    ax.barh(players, shot_assists, label='Shot Assist', color=VIOLET, left=left_sa, zorder=3)
    ax.barh(players, buildup, label='Buildup', color=ACOL, left=left_buildup, zorder=3) # Use another distinct color

    # Add counts inside bars
    for i, player in enumerate(players):
        counts = [shots[i], shot_assists[i], buildup[i]]
        lefts = [0, left_sa[i], left_buildup[i]]
        for j, count in enumerate(counts):
            if count > 0:
                x_pos = lefts[j] + count / 2
                ax.text(x_pos, i, str(int(count)), ha='center', va='center', color=BG_COLOR, fontsize=10, fontweight='bold', zorder=4)

    # Styling
    max_total = top_players_df['Shooting Seq Total'].max()
    ax.set_xlim(0, max_total * 1.05) # Add padding
    # Add grid lines
    x_ticks = np.arange(0, max_total + 1, 2) # Adjust step if needed
    for x in x_ticks:
        if x > 0: ax.axvline(x=x, color='grey', linestyle='--', zorder=2, alpha=0.5)

    ax.set_facecolor(BG_COLOR)
    ax.tick_params(axis='x', colors=LINE_COLOR, labelsize=12)
    ax.tick_params(axis='y', colors=LINE_COLOR, labelsize=12)
    for spine in ax.spines.values(): spine.set_edgecolor(BG_COLOR)
    ax.legend(fontsize=10)
    ax.set_title(title, color=LINE_COLOR, fontsize=16, fontweight='bold')


def plot_passer_stats_bar(ax, player_stats_df, num_players=10, title="Top Passers Stats"):
    """Plots a stacked bar chart for key passing stats for top players."""
    print(f"Plotting {title} bar chart...")
    # Ensure required columns exist
    req_cols = ['Progressive Passes', 'Passes into Box', 'Shot Assists', 'Offensive Pass Total'] # Assuming KeyPasses = Shot Assists
    if not all(col in player_stats_df.columns for col in req_cols):
        print(f"Warning: Missing required columns for passer stats plot. Skipping.")
        ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center', fontsize=12, color='red')
        return

    top_players_df = player_stats_df.sort_values('Offensive Pass Total', ascending=False).head(num_players).iloc[::-1]

    players = top_players_df.index.tolist()
    prog_p = top_players_df['Progressive Passes'].tolist()
    box_p = top_players_df['Passes into Box'].tolist()
    key_p = top_players_df['Shot Assists'].tolist() # Use Shot Assists as Key Passes

    left_box = np.array(prog_p)
    left_key = left_box + np.array(box_p)

    ax.barh(players, prog_p, label='Prog. Pass', color=HCOL, left=0, zorder=3)
    ax.barh(players, box_p, label='Pass into Box', color=ACOL, left=left_box, zorder=3)
    ax.barh(players, key_p, label='Key Pass', color=VIOLET, left=left_key, zorder=3)

    for i, player in enumerate(players):
        counts = [prog_p[i], box_p[i], key_p[i]]
        lefts = [0, left_box[i], left_key[i]]
        for j, count in enumerate(counts):
            if count > 0:
                x_pos = lefts[j] + count / 2
                ax.text(x_pos, i, str(int(count)), ha='center', va='center', color=BG_COLOR, fontsize=10, fontweight='bold', zorder=4)

    max_total = top_players_df['Offensive Pass Total'].max()
    ax.set_xlim(0, max_total * 1.05)
    x_ticks = np.arange(0, max_total + 1, 2)
    for x in x_ticks:
        if x > 0: ax.axvline(x=x, color='grey', linestyle='--', zorder=2, alpha=0.5)

    ax.set_facecolor(BG_COLOR); ax.tick_params(axis='both', colors=LINE_COLOR, labelsize=12)
    for spine in ax.spines.values(): spine.set_edgecolor(BG_COLOR)
    ax.legend(fontsize=10)
    ax.set_title(title, color=LINE_COLOR, fontsize=16, fontweight='bold')


def plot_defender_stats_bar(ax, player_stats_df, num_players=10, title="Top Defenders Stats"):
    """Plots a stacked bar chart for key defensive stats for top players."""
    print(f"Plotting {title} bar chart...")
    req_cols = ['Tackles Won', 'Interceptions', 'Clearances', 'Defensive Actions Total']
    if not all(col in player_stats_df.columns for col in req_cols):
        print(f"Warning: Missing required columns for defender stats plot. Skipping.")
        ax.text(0.5, 0.5, "Data Unavailable", ha='center', va='center', fontsize=12, color='red')
        return

    top_players_df = player_stats_df.sort_values('Defensive Actions Total', ascending=False).head(num_players).iloc[::-1]

    players = top_players_df.index.tolist()
    tackles = top_players_df['Tackles Won'].tolist()
    intercept = top_players_df['Interceptions'].tolist()
    clear = top_players_df['Clearances'].tolist()

    left_int = np.array(tackles)
    left_clr = left_int + np.array(intercept)

    ax.barh(players, tackles, label='Tackles Won', color=HCOL, left=0, zorder=3)
    ax.barh(players, intercept, label='Interceptions', color=VIOLET, left=left_int, zorder=3)
    ax.barh(players, clear, label='Clearances', color=ACOL, left=left_clr, zorder=3)

    for i, player in enumerate(players):
        counts = [tackles[i], intercept[i], clear[i]]
        lefts = [0, left_int[i], left_clr[i]]
        for j, count in enumerate(counts):
            if count > 0:
                x_pos = lefts[j] + count / 2
                ax.text(x_pos, i, str(int(count)), ha='center', va='center', color=BG_COLOR, fontsize=10, fontweight='bold', zorder=4)

    max_total = top_players_df['Defensive Actions Total'].max()
    ax.set_xlim(0, max_total * 1.05)
    x_ticks = np.arange(0, max_total + 1, 2)
    for x in x_ticks:
         if x > 0: ax.axvline(x=x, color='grey', linestyle='--', zorder=2, alpha=0.5)

    ax.set_facecolor(BG_COLOR); ax.tick_params(axis='both', colors=LINE_COLOR, labelsize=12)
    for spine in ax.spines.values(): spine.set_edgecolor(BG_COLOR)
    ax.legend(fontsize=10)
    ax.set_title(title, color=LINE_COLOR, fontsize=16, fontweight='bold')


def plot_player_pass_map(ax, df_player_passes, player_name, team_color, is_away):
    """Plots a pass map for a single player."""
    print(f"Plotting pass map for {player_name}...")
    pitch = Pitch(pitch_type='opta', corner_arcs=True, pitch_color=BG_COLOR, line_color=LINE_COLOR, linewidth=2)
    pitch.draw(ax=ax)
    if is_away: ax.invert_xaxis(); ax.invert_yaxis()

    if df_player_passes.empty:
        print(f"Warning: No pass data for {player_name}.")
        ax.set_title(f"{player_name} Pass Map (No Data)", color=team_color, fontsize=18, fontweight='bold')
        return

    # Assume 'is_key_pass', 'is_assist' columns exist
    pass_comp = df_player_passes[df_player_passes['outcome'] == 'Successful']
    pass_incomp = df_player_passes[df_player_passes['outcome'] == 'Unsuccessful']
    kp = df_player_passes[df_player_passes['is_key_pass']]
    assist = df_player_passes[df_player_passes['is_assist']]

    # Plot lines and points (use team color for successful)
    pitch.lines(pass_comp.x, pass_comp.y, pass_comp.end_x, pass_comp.end_y, lw=3, transparent=True, comet=True, color=team_color, ax=ax, alpha=0.6)
    pitch.lines(pass_incomp.x, pass_incomp.y, pass_incomp.end_x, pass_incomp.end_y, lw=3, transparent=True, comet=True, color='grey', ax=ax, alpha=0.2)
    pitch.lines(kp.x, kp.y, kp.end_x, kp.end_y, lw=4, transparent=True, comet=True, color=VIOLET, ax=ax, alpha=0.8)
    pitch.lines(assist.x, assist.y, assist.end_x, assist.end_y, lw=4, transparent=True, comet=True, color=GREEN, ax=ax, alpha=0.9)

    pitch.scatter(pass_comp.end_x, pass_comp.end_y, s=30, fc=BG_COLOR, edgecolor=team_color, zorder=2, ax=ax) 
    pitch.scatter(pass_incomp.end_x, pass_incomp.end_y, s=30, fc=BG_COLOR, edgecolor='grey', alpha=0.2, zorder=2, ax=ax) 
    pitch.scatter(kp.end_x, kp.end_y, s=40, fc=BG_COLOR, edgecolor=VIOLET, lw=1.5, zorder=2, ax=ax) 
    pitch.scatter(assist.end_x, assist.end_y, s=50, fc=BG_COLOR, edgecolors=GREEN, linewidths=1.5, marker='football', zorder=2, ax=ax)

    # --- ADD Average Start Position and Jersey Annotation ---
    if 'x' in df_player_passes.columns and 'y' in df_player_passes.columns:
        avg_x = df_player_passes['x'].median()
        avg_y = df_player_passes['y'].median()
        # Get jersey number (assuming it's consistent for the player in this df)
        jersey = df_player_passes['Mapped Jersey Number'].iloc[0] if 'Mapped Jersey Number' in df_player_passes.columns and not df_player_passes.empty else ''

        if pd.notna(avg_x) and pd.notna(avg_y):
            # Plot a marker for average start location
            pitch.scatter(avg_x, avg_y, marker='o', s=800, # Adjust size
                          facecolor=team_color, edgecolor='black', linewidth=1.5, alpha=0.8,
                          ax=ax, zorder=3) # Make sure zorder is high enough

            # Annotate with jersey number
            try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
            except (ValueError, TypeError): jersey_text = str(jersey) if pd.notna(jersey) else ''
            pitch.annotate(jersey_text, xy=(avg_x, avg_y), c='white', # Text color white for contrast
                           ha='center', va='center', size=10, weight='bold', ax=ax, zorder=4)

    # Add annotations
    ax.text(0.98 if not is_away else 0.02, -0.1, f'Successful: {len(pass_comp)}\nUnsuccessful: {len(pass_incomp)}', color=LINE_COLOR, va='bottom', ha='right' if not is_away else 'left', fontsize=10, transform=ax.transAxes)
    ax.text(0.02 if not is_away else 0.98, -0.1, f'Key Pass: {len(kp)}\nAssist: {len(assist)}', color=LINE_COLOR, va='bottom', ha='left' if not is_away else 'right', fontsize=10, transform=ax.transAxes)
    ax.set_title(f"{player_name} Pass Map", color=team_color, fontsize=18, fontweight='bold')


def plot_player_received_passes(ax, df_all_passes, target_player_name, team_color, is_away):
    """Plots passes received by a specific target player."""
    print(f"Plotting passes received by {target_player_name}...")
    pitch = Pitch(pitch_type='opta', corner_arcs=True, pitch_color=BG_COLOR, line_color=LINE_COLOR, linewidth=2)
    pitch.draw(ax=ax)
    if is_away: ax.invert_xaxis(); ax.invert_yaxis()

    # Filter successful passes WHERE the NEXT event's player is the target player
    # This uses the potentially fragile shift logic from original code
    received_filter = (df_all_passes['outcome'] == 'Successful') & \
                      (df_all_passes['receiver'] == target_player_name)
    filtered_rows = df_all_passes[received_filter].copy()

    if filtered_rows.empty:
         print(f"Warning: No passes recorded as received by {target_player_name} (using shift logic).")
         ax.set_title(f"{target_player_name} Passes Received (No Data)", color=team_color, fontsize=18, fontweight='bold')
         return

    # Check if key pass/assist columns exist on the PASS event itself
    # Note: This shows if the PASS was a KP/Assist, not if the received pass LED to one immediately
    kp_received = filtered_rows[filtered_rows.get('is_key_pass', False)] # Use .get for safety
    as_received = filtered_rows[filtered_rows.get('is_assist', False)]

    # Plot lines and points
    pitch.lines(filtered_rows.x, filtered_rows.y, filtered_rows.end_x, filtered_rows.end_y, lw=3, transparent=True, comet=True, color=team_color, ax=ax, alpha=0.5)
    pitch.lines(kp_received.x, kp_received.y, kp_received.end_x, kp_received.end_y, lw=4, transparent=True, comet=True, color=VIOLET, ax=ax, alpha=0.75)
    pitch.lines(as_received.x, as_received.y, as_received.end_x, as_received.end_y, lw=4, transparent=True, comet=True, color=GREEN, ax=ax, alpha=0.75)

    pitch.scatter(filtered_rows.end_x, filtered_rows.end_y, s=30, edgecolor=team_color, fc=BG_COLOR, zorder=2, ax=ax) 
    pitch.scatter(kp_received.end_x, kp_received.end_y, s=40, edgecolor=VIOLET, fc=BG_COLOR, lw=1.5, zorder=2, ax=ax) 
    pitch.scatter(as_received.end_x, as_received.end_y, s=50, edgecolors=GREEN, fc=BG_COLOR, marker='football', linewidths=1.5, zorder=2, ax=ax)

    # --- Add Average End Position and Jersey Annotation ---
    if 'end_x' in filtered_rows.columns and 'end_y' in filtered_rows.columns:
        avg_end_x = filtered_rows['end_x'].median()
        avg_end_y = filtered_rows['end_y'].median()
        # Get receiver jersey number (should be consistent for the target player)
        # Need to look at the *next* event's jersey if using shift logic,
        # OR if we have a 'receiver_jersey_number' column from get_passes_df:
        jersey = filtered_rows['receiver_jersey_number'].iloc[0] if 'receiver_jersey_number' in filtered_rows.columns and not filtered_rows.empty else ''

        if pd.notna(avg_end_x) and pd.notna(avg_end_y):
            # Plot marker for average reception location
            pitch.scatter(avg_end_x, avg_end_y, marker='o', s=800,
                          facecolor=team_color, edgecolor='black', linewidth=1.5, alpha=0.8,
                          ax=ax, zorder=3)

            # Annotate with jersey number
            try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
            except (ValueError, TypeError): jersey_text = str(jersey) if pd.notna(jersey) else ''
            pitch.annotate(jersey_text, xy=(avg_end_x, avg_end_y), c='white',
                           ha='center', va='center', size=10, weight='bold', ax=ax, zorder=4)

    # Add annotations
    total_received = len(filtered_rows)
    kp_count = len(kp_received) # Count of received passes that *were* key passes
    # Note: assist count here likely 0 unless an assist directly received by target?
    # If you want key passes *resulting* from the reception, that's different logic.

    title_name = target_player_name # Use full name unless shorter name is needed/available
    ax.set_title(f"{title_name} Passes Received", color=team_color, fontsize=18, fontweight='bold')
    # Adjust text annotation - original was complex
    ax.text(0.5, 0.01, f'Total Passes Received: {total_received}', color=LINE_COLOR, fontsize=12, ha='center', va='top', transform=ax.transAxes)
    # Add KP/Assist counts if meaningful
    if kp_count > 0 or len(as_received) > 0:
        ax.text(0.5, -0.06, f'(Included {kp_count} Key Passes, {len(as_received)} Assists)', color=LINE_COLOR, fontsize=10, ha='center', va='top', transform=ax.transAxes)


def plot_player_defensive_actions(ax, df_player_def_actions, player_name, team_color, is_away):
    """Plots defensive actions for a single player with different markers."""
    print(f"Plotting defensive actions for {player_name}...")
    pitch = Pitch(pitch_type='opta', corner_arcs=True, pitch_color=BG_COLOR, line_color=LINE_COLOR, line_zorder=1, linewidth=2)
    pitch.draw(ax=ax)
    if is_away: ax.invert_xaxis(); ax.invert_yaxis()

    if df_player_def_actions.empty:
        print(f"Warning: No defensive actions for {player_name}.")
        ax.set_title(f"{player_name} Def. Actions (No Data)", color=team_color, fontsize=18, fontweight='bold')
        return

    # Define markers and filters
    action_styles = {
        'Tackle': {'marker': '+', 'facecolor': team_color, 'edgecolor': team_color, 'hatch': '/////'},
        'Interception': {'marker': 's', 'facecolor': 'None', 'edgecolor': team_color, 'hatch': '/////'},
        'Ball recovery': {'marker': 'o', 'facecolor': 'None', 'edgecolor': team_color, 'hatch': '/////'},
        'Clearance': {'marker': 'd', 'facecolor': 'None', 'edgecolor': team_color, 'hatch': '/////'},
        'Foul': {'marker': 'x', 'facecolor': team_color, 'edgecolor': team_color, 'hatch': '/////'},
        'Aerial': {'marker': '^', 'facecolor': 'None', 'edgecolor': team_color, 'hatch': '/////'},
        # Add others if needed
    }
    action_size = 250

    # Plot each action type
    for action_type, style in action_styles.items():
        df_action = df_player_def_actions[df_player_def_actions['type_name'] == action_type]
        if not df_action.empty:
            pitch.scatter(df_action.x, df_action.y, s=action_size,
                          marker=style['marker'], facecolor=style['facecolor'],
                          edgecolor=style['edgecolor'], hatch=style['hatch'],
                          linewidth=style.get('linewidth', 1.5),
                          ax=ax, label=action_type, zorder=2) # Add label for legend
            
    # --- Add Median Position and Jersey Annotation ---
    if 'x' in df_player_def_actions.columns and 'y' in df_player_def_actions.columns:
         median_x = df_player_def_actions['x'].median()
         median_y = df_player_def_actions['y'].median()
         # Get jersey number (should be consistent for this player)
         jersey = df_player_def_actions['Mapped Jersey Number'].iloc[0] if 'Mapped Jersey Number' in df_player_def_actions.columns and not df_player_def_actions.empty else ''

         if pd.notna(median_x) and pd.notna(median_y):
             # Plot a distinct marker for the median position
             pitch.scatter(median_x, median_y, marker='o', s=1000, # Larger marker
                           facecolor=team_color, edgecolor='white', linewidth=2, alpha=0.9,
                           ax=ax, zorder=3) # Higher zorder

             # Annotate with jersey number
             try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
             except (ValueError, TypeError): jersey_text = str(jersey) if pd.notna(jersey) else ''
             pitch.annotate(jersey_text, xy=(median_x, median_y), c='white',
                            ha='center', va='center', size=12, weight='bold', ax=ax, zorder=4)
    
    # Position legend outside the pitch area
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.04), # Position below axes
              ncol=3, # Arrange in columns
              fontsize=10, frameon=False) # Styling

    # Titles
    ax.set_title(f"{player_name} Defensive Actions", color=team_color, fontsize=18, fontweight='bold')


def draw_plotly_pitch(fig):
    """Helper function to draw an Opta pitch using Plotly shapes."""
    # Outer lines
    fig.add_shape(type="rect", x0=0, y0=0, x1=100, y1=100, line=dict(color=LINE_COLOR, width=2))
    # Halfway line
    fig.add_shape(type="line", x0=50, y0=0, x1=50, y1=100, line=dict(color=LINE_COLOR, width=2))
    # Center circle
    fig.add_shape(type="circle", x0=42, y0=42, x1=58, y1=58, line=dict(color=LINE_COLOR, width=2))
    fig.add_shape(type="circle", x0=49.5, y0=49.5, x1=50.5, y1=50.5, line=dict(color=LINE_COLOR, width=2), fillcolor=LINE_COLOR)
    # Penalty Areas
    fig.add_shape(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=LINE_COLOR, width=2))
    fig.add_shape(type="rect", x0=83.5, y0=21.1, x1=100, y1=78.9, line=dict(color=LINE_COLOR, width=2))
    # 6-yard boxes
    fig.add_shape(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=LINE_COLOR, width=2))
    fig.add_shape(type="rect", x0=94.5, y0=36.8, x1=100, y1=63.2, line=dict(color=LINE_COLOR, width=2))
    return fig

# def plot_player_defensive_actions_plotly(df_player_def_actions, player_name, team_color, is_away):
#     """Plots defensive actions for a single player using Plotly for interactivity."""
#     print(f"Plotting interactive defensive actions for {player_name}...")
    
#     fig = go.Figure()
#     fig = draw_plotly_pitch(fig)

#     if df_player_def_actions.empty:
#         print(f"Warning: No defensive actions for {player_name}.")
#         # You can add an annotation to the empty plot if you wish
#         fig.add_annotation(x=50, y=50, text="No Defensive Actions Data", showarrow=False, font=dict(size=16, color="red"))
#     else:
#         action_styles = {
#             # Plotly marker symbols: https://plotly.com/python/marker-style/
#             'Tackle': {'symbol': 'cross', 'color': team_color},
#             'Interception': {'symbol': 'square-open', 'color': team_color},
#             'Ball recovery': {'symbol': 'circle-open', 'color': team_color},
#             'Clearance': {'symbol': 'diamond-open', 'color': team_color},
#             'Foul': {'symbol': 'x', 'color': 'red'},
#             'Aerial': {'symbol': 'triangle-up-open', 'color': team_color},
#             'Blocked pass': {'symbol': 'pentagon', 'color': team_color},
#         }

#         for action_type, style in action_styles.items():
#             df_action = df_player_def_actions[df_player_def_actions['type_name'] == action_type].copy()
#             if not df_action.empty:
#                 # Create custom hover text
#                 time_cols_exist = 'timeMin' in df_action.columns and 'timeSec' in df_action.columns
                
#                 def create_hover_text(row):
#                     # Basic info that should always be present
#                     text = f"<b>{action_type}</b><br>Location: ({row['x']:.1f}, {row['y']:.1f})"
#                     # Safely add the time if the columns exist
#                     if time_cols_exist:
#                         # Use .get() for an extra layer of safety, providing a default if the value is somehow missing for a row
#                         minute = row.get('timeMin', '?')
#                         second = row.get('timeSec', '?')
#                         text += f"<br>Time: {minute}:{second}"
#                     return text

#                 df_action['hover_text'] = df_action.apply(create_hover_text, axis=1)
#                 fig.add_trace(go.Scatter(
#                     x=df_action['x'],
#                     y=df_action['y'],
#                     mode='markers',
#                     name=action_type,
#                     marker=dict(
#                         symbol=style['symbol'],
#                         color=style['color'],
#                         size=12,
#                         line=dict(width=2)
#                     ),
#                     hoverinfo='text',
#                     hovertext=df_action['hover_text']
#                 ))

#         # --- Add Median Position and Jersey Annotation ---
#         if 'x' in df_player_def_actions.columns and 'y' in df_player_def_actions.columns:
#             median_x = df_player_def_actions['x'].median()
#             median_y = df_player_def_actions['y'].median()
#             jersey = df_player_def_actions['Mapped Jersey Number'].iloc[0] if 'Mapped Jersey Number' in df_player_def_actions.columns and not df_player_def_actions.empty else ''

#             if pd.notna(median_x) and pd.notna(median_y):
#                 # Add large marker for median position
#                 fig.add_trace(go.Scatter(
#                     x=[median_x], y=[median_y],
#                     mode='markers',
#                     name='Avg. Position',
#                     marker=dict(color=team_color, size=35, line=dict(color='white', width=2)),
#                     hoverinfo='none'
#                 ))
#                 # Add jersey number annotation
#                 try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
#                 except (ValueError, TypeError): jersey_text = str(jersey) if pd.notna(jersey) else ''
#                 fig.add_annotation(
#                     x=median_x, y=median_y, text=jersey_text, showarrow=False,
#                     font=dict(color='white', size=14, family="Arial Black")
#                 )
    
#     # --- Final Layout Updates ---
#     fig.update_layout(
#         title=f"{player_name} Defensive Actions",
#         title_font_color=team_color,
#         plot_bgcolor=BG_COLOR,
#         paper_bgcolor=BG_COLOR,
#         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
#         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
#         showlegend=True,
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=-0.1,
#             xanchor="center",
#             x=0.5,
#             font=dict(color=LINE_COLOR),
#             bgcolor=BG_COLOR
#         ),
#         # This forces the aspect ratio to look like a football pitch
#         yaxis_scaleanchor="x",
#         yaxis_scaleratio=0.68,
#         margin=dict(l=10, r=10, t=40, b=40)
#     )

#     # Invert axis for away team view
#     if is_away:
#         fig.update_xaxes(autorange="reversed")
#         fig.update_yaxes(autorange="reversed")

#     return fig

def plot_player_defensive_actions_plotly(df_player_def_actions, player_name, team_color, is_away):
    """
    Plots defensive actions for a single player using Plotly,
    distinguishing between successful and unsuccessful outcomes.
    Includes visual enhancements and fixes for layout and interaction.
    """
    print(f"Plotting enhanced interactive defensive actions for {player_name}...")
    
    fig = go.Figure()
    fig = draw_plotly_pitch(fig) # Your existing pitch drawing function is fine

    # --- CHANGE: Define a new, more appealing background color ---
    # We will use this in the layout update at the end.
    # It's a dark slate grey that works well with the default Dash theme.
    PLOT_BG_COLOR = '#2E3439'

    if df_player_def_actions.empty:
        fig.add_annotation(x=50, y=50, text="No Defensive Actions Data", showarrow=False, font=dict(size=16, color="red"))
    else:
        # Define base symbols for each action
        action_symbols = {
            'Tackle': 'cross', 'Interception': 'square', 'Ball recovery': 'circle',
            'Clearance': 'diamond', 'Foul': 'x', 'Aerial': 'triangle-up', 'Blocked pass': 'pentagon'
        }
        
        # Plot successful actions first, then unsuccessful on top
        outcomes_to_plot = [('Successful', team_color, 'solid'), ('Unsuccessful', '#B2B2B2', 'open')]

        for outcome, color, style in outcomes_to_plot:
            for action_type, symbol in action_symbols.items():
                # Filter for the specific action and outcome
                df_plot = df_player_def_actions[
                    (df_player_def_actions['type_name'] == action_type) &
                    (df_player_def_actions['outcome'] == outcome)
                ]

                # For actions that are always successful, only plot them once
                if action_type in ['Interception', 'Clearance', 'Ball recovery', 'Blocked pass'] and outcome == 'Unsuccessful':
                    continue
                # For fouls, which are always unsuccessful
                if action_type == 'Foul' and outcome == 'Successful':
                    continue

                if not df_plot.empty:
                    # Create custom hover text
                    df_plot['hover_text'] = df_plot.apply(
                        lambda row: f"<b>{action_type} ({row['outcome']})</b><br>Time: {row.get('timeMin','?')}:{row.get('timeSec','?')}<br>Location: ({row['x']:.1f}, {row['y']:.1f})",
                        axis=1
                    )
                    
                    marker_symbol = f"{symbol}-open" if style == 'open' else symbol
                    
                    fig.add_trace(go.Scatter(
                        x=df_plot['x'], y=df_plot['y'], mode='markers',
                        name=f"{action_type} ({outcome})",
                        legendgroup=action_type, # Group success/fail of same action type in legend
                        marker=dict(symbol=marker_symbol, color=color, size=12, line=dict(width=2)),
                        hoverinfo='text', hovertext=df_plot['hover_text']
                    ))
        
        # --- Median Position and Jersey (no change here) ---
        median_x = df_player_def_actions['x'].median()
        median_y = df_player_def_actions['y'].median()
        jersey = df_player_def_actions['Mapped Jersey Number'].iloc[0] if 'Mapped Jersey Number' in df_player_def_actions.columns and not df_player_def_actions.empty else ''

        if pd.notna(median_x) and pd.notna(median_y):
            fig.add_trace(go.Scatter(
                x=[median_x], y=[median_y], mode='markers', name='Avg. Position',
                marker=dict(color=team_color, size=35, line=dict(color='white', width=2)),
                hoverinfo='none', showlegend=False
            ))
            try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
            except (ValueError, TypeError): jersey_text = str(jersey) if pd.notna(jersey) else ''
            fig.add_annotation(
                x=median_x, y=median_y, text=jersey_text, showarrow=False,
                font=dict(color='white', size=14, family="Arial Black")
            )

    # --- Final Layout Updates (no major changes) ---
    fig.update_layout(
        # --- CHANGE: Center the title ---
        title={
            'text': f"{player_name} Defensive Actions",
            'y':0.97,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'color': 'white', 'size': 20} # Use white text for dark background
        },
        # --- CHANGE: Set new background color and increased height ---
        plot_bgcolor=PLOT_BG_COLOR,
        paper_bgcolor=PLOT_BG_COLOR,
        height=650, # Increased height to ensure legend doesn't overlap pitch
        
        # --- CHANGE: Lock the axes to prevent zoom/pan/reset ---
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, 
            fixedrange=True # This is the key to disabling zoom/pan on this axis
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False, 
            fixedrange=True # This is the key to disabling zoom/pan on this axis
        ),
        
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15, # Adjusted position slightly for new height
            xanchor="center",
            x=0.5,
            font=dict(color='white'), # White text for legend
            bgcolor=PLOT_BG_COLOR,
            bordercolor=LINE_COLOR
        ),
        # Use the scaleratio to maintain the pitch shape now that height is fixed
        yaxis_scaleanchor="x",
        yaxis_scaleratio=0.68,
        margin=dict(l=10, r=10, t=50, b=50)
    )
    
    # --- CHANGE: Update the drawing function to use a better background ---
    # We will add a radial gradient to the background for a nice spotlight effect.
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=100, y1=100,
        line_width=0,
        fillcolor=PLOT_BG_COLOR,
        layer="below"
    )
    fig.update_layout(
        shapes=[
            dict(
                type='rect',
                xref='paper', yref='paper',
                x0=0, y0=0, x1=1, y1=1,
                line_width=0,
                fillcolor=f'rgba({int(PLOT_BG_COLOR[1:3], 16)}, {int(PLOT_BG_COLOR[3:5], 16)}, {int(PLOT_BG_COLOR[5:7], 16)}, 1)',
                layer='below'
            ),
            # This shape creates the subtle radial gradient
            dict(
                type='circle',
                xref='paper', yref='paper',
                x0=-0.6, y0=-0.5, x1=1.6, y1=1.5,
                line_width=0,
                fillcolor='rgba(255, 255, 255, 0.05)',
                layer='below'
            )
        ] + list(fig.layout.shapes) 
    )

    # Invert axis for away team view - the range setting is the most robust way
    if is_away:
        fig.update_xaxes(range=[100, 0])
        fig.update_yaxes(range=[100, 0])
    else:
        fig.update_xaxes(range=[0, 100])
        fig.update_yaxes(range=[0, 100])

    return fig

def plot_defender_stats_bar_by_team(ax, player_stats_df, df_processed, home_team_name, num_players=10, hcol=HCOL, acol=ACOL, title="Top Defenders Stats"):
    """
    Plots a comprehensive and readable stacked bar chart for key defensive stats.
    This version dynamically plots only the available stat columns.
    """
    print(f"Plotting improved and resilient {title} bar chart by team...")
    
    # --- FIX: Dynamically build the list of actions to plot ---
    # Define all possible actions and their desired colors
    possible_actions = {
        'Tackles Won': '#d9534f', 'Interceptions': '#5bc0de',
        'Ball recovery': '#5cb85c', 'Aerials Won': '#f0ad4e', 'Clearances': '#777777'
    }
    
    # Filter for actions that actually exist as columns in the provided dataframe
    actions_to_plot = {
        action: color for action, color in possible_actions.items() if action in player_stats_df.columns
    }
    
    if not actions_to_plot or 'Defensive Actions Total' not in player_stats_df.columns:
        ax.text(0.5, 0.5, "Required defensive stats are not available.", ha='center', va='center', fontsize=12, color='red')
        return

    # Define the sorting hierarchy
    sort_hierarchy = [
        'Defensive Actions Total', # Primary sort: total volume
        'Tackles Won',             # Tie-breaker 1: most valuable action
        'Interceptions',           # Tie-breaker 2
        'Aerials Won',             # Tie-breaker 3
        'Ball recovery',           # Tie-breaker 4
        'Clearances'               # Tie-breaker 5
    ]
    
    # Filter the hierarchy list to only include columns that actually exist in the dataframe
    # This makes the sorting robust even if some columns are missing.
    available_sort_columns = [col for col in sort_hierarchy if col in player_stats_df.columns]

    # Sort players by the defined hierarchy. All are sorted descending.
    top_players_df = player_stats_df.sort_values(
        by=available_sort_columns,
        ascending=False
    ).head(num_players).iloc[::-1] # .iloc[::-1] reverses the order for plotting top-to-bottom
    
    # Get Team Colors for Y-Axis Labels
    player_team_map = df_processed[['playerName', 'team_name']].drop_duplicates('playerName').set_index('playerName')['team_name']
    top_players_df['team_name'] = top_players_df.index.map(player_team_map)
    label_colors = [hcol if team == home_team_name else acol for team in top_players_df['team_name']]
    
    players = top_players_df.index.tolist()
    
    # Plot ALL available actions in a loop
    left_offset = np.zeros(len(players))
    for action, color in actions_to_plot.items():
        values = top_players_df[action].values
        ax.barh(players, values, color=color, left=left_offset, zorder=3, label=action)
        
        for i, (value, player) in enumerate(zip(values, players)):
            if value > 0:
                ax.text(left_offset[i] + value / 2, i, str(int(value)),
                        ha='center', va='center', color='white', fontsize=10, fontweight='bold', zorder=4)
        
        left_offset += values

    # Styling (same as before)
    TEXT_COLOR_DARK_THEME = 'white'
    ax.set_yticks(np.arange(len(players)))
    ax.set_yticklabels(players)
    for ticklabel, color in zip(ax.get_yticklabels(), label_colors):
        ticklabel.set_color(color)
        ticklabel.set_fontsize(12)
    
    max_total = top_players_df['Defensive Actions Total'].max()
    ax.set_xlim(0, max_total * 1.05)
    ax.tick_params(axis='x', colors=TEXT_COLOR_DARK_THEME)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color(TEXT_COLOR_DARK_THEME)

    ax.legend(fontsize=10, labelcolor=TEXT_COLOR_DARK_THEME, frameon=False)
    ax.set_title(title, color=TEXT_COLOR_DARK_THEME, fontsize=18, fontweight='bold')



############# PLOTLY PASSER STATS BAR PLOT #############
def plot_passer_stats_bar_plotly(player_stats_df, df_processed, home_team_name, hcol='tomato', acol='skyblue', violet_col='#a369ff', num_players=10):
    """
    Versione Definitiva: Usa dati invertiti per l'ordinamento e annotazioni per etichette colorate.
    """
    req_cols = ['Progressive Passes', 'Passes into Box', 'Shot Assists', 'Offensive Pass Total']
    if not all(col in player_stats_df.columns for col in req_cols):
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='#2E3439', paper_bgcolor='#2E3439', font_color='white',
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[dict(text="Player Passing Data Unavailable", showarrow=False, font=dict(size=16, color="orange"))]
        )
        return fig

    # 1. Ordina per trovare i migliori, poi inverti l'ordine per il plotting
    top_players_sorted = player_stats_df.sort_values('Offensive Pass Total', ascending=False).head(num_players)
    plot_df = top_players_sorted.iloc[::-1]

    # 2. Mappe per i dati dei giocatori
    player_to_team_map = df_processed.drop_duplicates('playerName').set_index('playerName')['team_name'].to_dict()
    player_jersey_map = df_processed.drop_duplicates('playerName').set_index('playerName')['Mapped Jersey Number'].to_dict()

    # --- Creazione della Figura ---
    fig = go.Figure()

    # Aggiungi le tracce usando il DataFrame invertito `plot_df`
    # L'asse Y userà l'indice di plot_df (i nomi dei giocatori)
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Progressive Passes'], name='Progressive Passes', orientation='h', marker_color=hcol, text=plot_df['Progressive Passes'], hoverinfo='y+x', hovertemplate='<b>%{y}</b><br>Prog. Passes: %{x}<extra></extra>'))
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Passes into Box'], name='Passes into Box', orientation='h', marker_color=acol, text=plot_df['Passes into Box'], hoverinfo='y+x', hovertemplate='<b>%{y}</b><br>Passes into Box: %{x}<extra></extra>'))
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Shot Assists'], name='Key Passes', orientation='h', marker_color=violet_col, text=plot_df['Shot Assists'], hoverinfo='y+x', hovertemplate='<b>%{y}</b><br>Key Passes: %{x}<extra></extra>'))
    
    # --- Configurazione del Layout ---
    fig.update_layout(
        title_text='Top Players by Offensive Passes',
        barmode='stack',
        yaxis=dict(
            # Non impostiamo l'ordine qui, lasciamo che Plotly usi l'ordine dei dati.
            # Nascondiamo le etichette di default perché le creeremo noi.
            showticklabels=False
        ),
        xaxis=dict(title='Total Offensive Passes'),
        plot_bgcolor='#2E3439', paper_bgcolor='#2E3439',
        font_color='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=180, r=15, t=80, b=40),
        height=800,
        annotations=[]
    )
    
    # --- Aggiungi le etichette manualmente come annotazioni ---
    # Itera sull'indice di plot_df per mantenere l'ordine corretto
    for player_name in plot_df.index:
        team_name = player_to_team_map.get(player_name)
        label_color = hcol if team_name == home_team_name else acol
        
        jersey_raw = player_jersey_map.get(player_name)
        try: jersey = str(int(jersey_raw))
        except (ValueError, TypeError): jersey = '?'
        
        label_text = f"<b>#{jersey} - {player_name}</b>"

        fig.add_annotation(
            x=0, y=player_name,
            xref="paper", yref="y",
            text=label_text,
            showarrow=False,
            xanchor="right",
            align="right",
            font=dict(color=label_color, size=12),
            xshift=-10
        )

    return fig

def plot_player_pass_map_plotly(df_player_passes, player_name, team_color, player_jersey='?', is_away_team=False):
    """
    Crea una mappa interattiva dei passaggi di un giocatore usando Plotly.
    """
    fig = go.Figure()
    
    # Prepara lo sfondo del campo
    pitch_shapes = pitch_plots.get_plotly_pitch_shapes("rgba(255,255,255,0.2)", "white")
    
    if df_player_passes.empty:
        fig.update_layout(
            title=f"No passes recorded for {player_name}",
            plot_bgcolor='#2E3439', paper_bgcolor='#2E3439', font_color='white',
            shapes=pitch_shapes,
            xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig
        
    # Inverti coordinate per away team
    if is_away_team:
        df_player_passes = df_player_passes.copy()
        df_player_passes[['x', 'end_x']] = 100 - df_player_passes[['x', 'end_x']]
        df_player_passes[['y', 'end_y']] = 100 - df_player_passes[['y', 'end_y']]

    # Colori e gerarchia
    COLORS = {
        'Assist': '#69f900', 'Key Pass': '#a369ff', 'Pass into Box': '#00a0de',
        'Progressive Pass': '#ff4b44', 'Successful': 'rgba(128, 128, 128, 0.7)',
        'Unsuccessful': 'rgba(220, 53, 69, 0.5)'
    }
    plot_order = ['Unsuccessful', 'Successful', 'Progressive Pass', 'Pass into Box', 'Key Pass', 'Assist']

    def get_pass_category(p):
        if p.get('is_assist', False): return 'Assist'
        if p.get('is_key_pass', False): return 'Key Pass'
        if p.get('is_into_box', False): return 'Pass into Box'
        if p.get('is_progressive', False): return 'Progressive Pass'
        return 'Successful' if p.get('outcome') == 'Successful' else 'Unsuccessful'

    df_player_passes['category'] = df_player_passes.apply(get_pass_category, axis=1)

    # --- NUOVA LOGICA: UNA TRACCIA PER CATEGORIA ---
    
    for category in plot_order:
        category_passes = df_player_passes[df_player_passes['category'] == category]
        if category_passes.empty:
            continue
            
        is_fail = (category == 'Unsuccessful')
        
        # Prepara i dati per una singola traccia Scattergl (più performante per tante forme)
        x_coords, y_coords = [], []
        hover_texts = []
        for _, p in category_passes.iterrows():
            x_coords.extend([p['x'], p['end_x'], None]) # None per spezzare le linee
            y_coords.extend([p['y'], p['end_y'], None])
            # Duplichiamo l'hovertext per coprire entrambi i punti della linea
            hover_text = f"<b>{category}</b><br>Min {p.get('timeMin', '?')}': {p['outcome']}"
            hover_texts.extend([hover_text, hover_text, ''])

        # Aggiungi una sola traccia per l'intera categoria
        fig.add_trace(go.Scattergl(
            x=x_coords,
            y=y_coords,
            mode='lines',
            line=dict(
                color=COLORS[category], 
                width=2.5 if not is_fail else 1.5, 
                dash='solid' if not is_fail else 'dot'
            ),
            name=category, # Questo nome apparirà nella legenda
            hoverinfo='text',
            hovertext=hover_texts
        ))
        
        # Aggiungi i marker di fine passaggio separatamente per non interferire con le linee
        fig.add_trace(go.Scattergl(
            x=category_passes['end_x'],
            y=category_passes['end_y'],
            mode='markers',
            marker=dict(size=6, color=COLORS[category]),
            showlegend=False, # Non mostrare i marker nella legenda
            hoverinfo='none'
        ))

    # Calcola la posizione media
    avg_x, avg_y = df_player_passes['x'].mean(), df_player_passes['y'].mean()  

    # --- Layout finale con l'asse secondario configurato ---
    fig.update_layout(
        title=f"Pass Map for {player_name}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, traceorder='reversed'),
        shapes=pitch_shapes,
        xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        
        # **MODIFICA CHIAVE: Configura l'asse Y secondario**
        yaxis2=dict(
            range=[-2, 102],
            overlaying="y", # Sovrapponi all'asse y primario
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
        
        plot_bgcolor='#2E3439', paper_bgcolor='#2E3439', font_color='white', height=800
    )

    fig.add_trace(go.Scatter(
        x=[avg_x], y=[avg_y],
        mode='markers+text',
        marker=dict(symbol='circle', size=40, color=team_color, line=dict(color='white', width=2)),
        text=[f"<b>{player_jersey}</b>"],
        textposition="middle center",
        textfont=dict(color='white', size=14),
        hoverinfo='text', hovertext="Avg. Position",
        showlegend=False,
        yaxis="y2" # <-- Assegna a y2
    ))

    return fig

################################ SHOT MAPS ################################
def plot_shot_sequence_bar_plotly(player_stats_df, df_processed, home_team_name, hcol='tomato', acol='skyblue', violet_col='#a369ff', num_players=10):
    """
    Crea un bar chart Plotly interattivo per il coinvolgimento nelle sequenze di tiro.
    """
    req_cols = ['Shots', 'Shot Assists', 'Buildup to Shot', 'Shooting Seq Total']
    if not all(col in player_stats_df.columns for col in req_cols):
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor='#2E3439', paper_bgcolor='#2E3439', font_color='white',
            xaxis={'visible': False}, yaxis={'visible': False},
            annotations=[dict(text="Shooting Sequence Data Unavailable", showarrow=False, font=dict(size=16, color="orange"))]
        )
        return fig
    
    weights = {'Shots': 3, 'Shot Assists': 2, 'Buildup to Shot': 1}
    df_with_score = player_stats_df.copy()
    df_with_score['Weighted Score'] = (
        df_with_score['Shots'] * weights['Shots'] +
        df_with_score['Shot Assists'] * weights['Shot Assists'] +
        df_with_score['Buildup to Shot'] * weights['Buildup to Shot']
    )

    # 1. Prepara i dati dei top players, ordinamento DISCENDENTE e inversione per il plot
    top_players_sorted = df_with_score.sort_values('Weighted Score', ascending=False).head(num_players)
    plot_df = top_players_sorted.iloc[::-1]

    # 2. Mappe per i dati dei giocatori
    player_to_team_map = df_processed.drop_duplicates('playerName').set_index('playerName')['team_name'].to_dict()
    player_jersey_map = df_processed.drop_duplicates('playerName').set_index('playerName')['Mapped Jersey Number'].to_dict()

    # --- Creazione della Figura ---
    fig = go.Figure()

    # Aggiungi le tracce usando il DataFrame invertito
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Buildup to Shot'], name='Buildup', orientation='h', marker_color=acol, text=plot_df['Buildup to Shot'], hoverinfo='y+x', hovertemplate='<b>%{y}</b><br>Buildup to Shot: %{x}<extra></extra>'))
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Shot Assists'], name='Shot Assist', orientation='h', marker_color=violet_col, text=plot_df['Shot Assists'], hoverinfo='y+x', hovertemplate='<b>%{y}</b><br>Shot Assist: %{x}<extra></extra>'))
    fig.add_trace(go.Bar(y=plot_df.index, x=plot_df['Shots'], name='Shot', orientation='h', marker_color=hcol, text=plot_df['Shots'], hoverinfo='y+x', hovertemplate='<b>%{y}</b><br>Shots: %{x}<extra></extra>'))
    
    # --- Configurazione del Layout ---
    fig.update_layout(
        title_text='Shot Sequence Involvement',
        barmode='stack',
        yaxis=dict(showticklabels=False), # Nascondiamo le etichette, le creiamo con le annotazioni
        xaxis=dict(title='Total Involvements in Shot Sequences'),
        plot_bgcolor='#2E3439', paper_bgcolor='#2E3439',
        font_color='white',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=180, r=15, t=80, b=40),
        height=800,
        annotations=[]
    )
    
    # Aggiungi le etichette manualmente come annotazioni
    for player_name in plot_df.index:
        team_name = player_to_team_map.get(player_name)
        label_color = hcol if team_name == home_team_name else acol
        
        jersey_raw = player_jersey_map.get(player_name)
        try: jersey = str(int(jersey_raw))
        except (ValueError, TypeError): jersey = '?'
        
        label_text = f"<b>#{jersey} - {player_name}</b>"

        fig.add_annotation(
            x=0, y=player_name,
            xref="paper", yref="y",
            text=label_text,
            showarrow=False,
            xanchor="right",
            align="right",
            font=dict(color=label_color, size=12),
            xshift=-10
        )

    return fig

def plot_player_received_passes_plotly(df_received_passes, player_name, team_color, player_jersey='?', is_away_team=False):
    """
    Crea una mappa interattiva dei passaggi ricevuti da un giocatore.
    """
    fig = go.Figure()
    pitch_shapes = pitch_plots.get_plotly_pitch_shapes("rgba(255,255,255,0.2)", "white")
    
    if df_received_passes.empty:
        # ... gestione errore ...
        fig.update_layout(title=f"No successful passes recorded as received by {player_name}")
        return fig
        
    # Inverti coordinate per away team (visualizzazione da sinistra a destra)
    if is_away_team:
        df_received_passes = df_received_passes.copy()
        df_received_passes[['x', 'end_x']] = 100 - df_received_passes[['x', 'end_x']]
        df_received_passes[['y', 'end_y']] = 100 - df_received_passes[['y', 'end_y']]

    # Colori per tipo di passaggio (il passaggio che è stato ricevuto)
    COLORS = {
        'Assist': '#69f900', 'Key Pass': '#a369ff',
        'Standard Pass': team_color
    }
    
    # Gerarchia: un assist è anche un key pass, ma vogliamo mostrarlo come assist
    def get_received_pass_type(p):
        if p.get('is_assist', False): return 'Assist'
        if p.get('is_key_pass', False): return 'Key Pass'
        return 'Standard Pass'

    df_received_passes['pass_type'] = df_received_passes.apply(get_received_pass_type, axis=1)

    # Plotta per categoria per avere una legenda funzionante
    for pass_type, color in COLORS.items():
        subset = df_received_passes[df_received_passes['pass_type'] == pass_type]
        if subset.empty:
            continue
        
        x_coords, y_coords, hover_texts = [], [], []
        for _, p in subset.iterrows():
            x_coords.extend([p['x'], p['end_x'], None])
            y_coords.extend([p['y'], p['end_y'], None])
            hover_texts.extend([f"From: {p['playerName']}", f"From: {p['playerName']}", None])

        fig.add_trace(go.Scattergl(
            x=x_coords, y=y_coords, mode='lines',
            line=dict(color=color, width=2.5),
            name=pass_type, hoverinfo='text', hovertext=hover_texts
        ))
        fig.add_trace(go.Scattergl(
            x=subset['end_x'], y=subset['end_y'], mode='markers',
            marker=dict(size=6, color=color), showlegend=False, hoverinfo='none'
        ))

    # Calcola e aggiungi la posizione media di RICEZIONE
    avg_end_x, avg_end_y = df_received_passes['end_x'].mean(), df_received_passes['end_y'].mean()
    fig.add_trace(go.Scatter(
        x=[avg_end_x], y=[avg_end_y], mode='markers+text',
        marker=dict(symbol='circle', size=40, color=team_color, line=dict(color='white', width=2)),
        text=[f"<b>{player_jersey}</b>"], textposition="middle center", textfont=dict(color='white', size=14),
        hoverinfo='text', hovertext=f"Avg. Reception Spot", showlegend=False
    ))

    # Layout finale
    fig.update_layout(
        title=f"Passes Received by {player_name}",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=pitch_shapes,
        xaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-2, 102], showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='#2E3439', paper_bgcolor='#2E3439', font_color='white', height=800
    )
    
    return fig

def generate_defender_layout_and_data(stored_data_json, player_stats_df_json, is_for_home_team, selected_player=None):
    """
    Versione 2: Usa un ranking ponderato e restituisce tutti i componenti necessari per il layout.
    """
    team_type = "Home" if is_for_home_team else "Away"
    
    try:
        # --- 1. Caricamento e Setup ---
        df_json_str = stored_data_json.get('df')
        match_info_json_str = stored_data_json.get('match_info')
        df_processed = pd.read_json(io.StringIO(df_json_str), orient='split')
        match_info = json.loads(match_info_json_str)
        player_stats_df = pd.read_json(player_stats_df_json, orient='split')
        
        team_name = match_info.get('hteamName') if is_for_home_team else match_info.get('ateamName')
        team_color = HCOL if is_for_home_team else ACOL

        # --- 2. Trova tutti i giocatori difensivi per il dropdown ---
        DEFENSIVE_ACTION_TYPES = ['Tackle', 'Interception', 'Ball recovery', 'Clearance', 'Foul', 'Aerial', 'Blocked pass']
        defensive_players_df = df_processed[
            (df_processed['team_name'] == team_name) &
            (df_processed['type_name'].isin(DEFENSIVE_ACTION_TYPES))
        ].dropna(subset=['playerName'])
        
        if defensive_players_df.empty:
            # Ritorna solo un messaggio di errore e una lista vuota per il dropdown
            return dbc.Alert(f"No defensive actions recorded for {team_name}."), [], None

        unique_defensive_players = defensive_players_df.drop_duplicates('playerName')
        player_jersey_map = unique_defensive_players.set_index('playerName')['Mapped Jersey Number']
        sorted_player_names = sorted(player_jersey_map.index.tolist())
        
        dropdown_options = []
        for name in sorted_player_names:
            jersey_raw = player_jersey_map.get(name)
            try:
                jersey = str(int(jersey_raw))
            except (ValueError, TypeError):
                jersey = '?'
            dropdown_options.append({'label': f"#{jersey} - {name}", 'value': name})
        
        # --- 3. Determina il Giocatore Target (con ranking ponderato) ---
        if selected_player:
            target_player_name = selected_player
        else:
            # Default al top defender basato sul punteggio ponderato
            team_player_stats = player_stats_df[player_stats_df.index.isin(player_jersey_map.index)].copy()
            weights = {'Tackles Won': 3, 'Interceptions': 3, 'Aerials Won': 2, 'Ball recovery': 2, 'Clearances': 1}
            
            for col in weights.keys():
                if col not in team_player_stats.columns:
                    team_player_stats[col] = 0
            
            team_player_stats['Weighted Score'] = sum(team_player_stats[col] * w for col, w in weights.items())
            
            if not team_player_stats.empty:
                target_player_name = team_player_stats['Weighted Score'].idxmax()
            else:
                target_player_name = sorted_player_names[0] if sorted_player_names else None
        
        if not target_player_name:
             return dbc.Alert(f"Could not determine a target player for {team_name}."), dropdown_options, None

        # --- 4. Genera Grafico e Tabella per il giocatore target ---
        df_player_actions = defensive_players_df[defensive_players_df['playerName'] == target_player_name]
        
        stats_df = player_metrics.calculate_defensive_action_rates(df_player_actions)
        # ... (la tua logica per ordinare la tabella stats_df può rimanere qui) ...
        
        stats_table = dash_table.DataTable(
            data=stats_df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in stats_df.columns],
            style_cell={'backgroundColor': '#343A40', 'color': 'white', 'textAlign': 'center'},
            style_header={'backgroundColor': '#454D55', 'fontWeight': 'bold'},
            style_as_list_view=True,
        )

        is_away_team = not is_for_home_team
        
        fig = plot_player_defensive_actions_plotly(df_player_actions, target_player_name, team_color, is_away_team)
        interactive_map = dcc.Graph(figure=fig, config={'displayModeBar': False})
        
        # Combina mappa e tabella in un unico componente
        layout_content = dash_html.Div([
            dbc.Row(dbc.Col(interactive_map)),
            dbc.Row([
                dbc.Col([
                    dash_html.H5("Action Summary", className="text-center text-white mt-4"),
                    stats_table
                ], md=6),
                dbc.Col([
                    dash_html.H5("Analyst Comments", className="text-center text-white mt-4"),
                    dcc.Textarea(placeholder=f"Comments for {target_player_name}...", style={'width': '100%', 'height': 100}, disabled=True)
                ], md=6)
            ], className="mt-3")
        ])

        return layout_content, dropdown_options, target_player_name

    except Exception as e:
        tb_str = traceback.format_exc()
        return dbc.Alert(f"Error in defender analysis: {e}\n{tb_str}", color="danger"), [], None
