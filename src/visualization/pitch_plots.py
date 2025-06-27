# src/visualization/pitch_plots.py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mplsoccer import VerticalPitch, Pitch, FontManager
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.patches import FancyArrow
import numpy as np
import pandas as pd 
from highlight_text import ax_text, fig_text
from src import config
from scipy.spatial import ConvexHull
from scipy.spatial import Voronoi, voronoi_plot_2d
from src.utils import formation_layouts 
# Optional: Load font manager if using custom fonts
# try:
#     robotto_regular = FontManager() # Add URL or path if needed
# except Exception as e:
#     print(f"Warning: FontManager failed - {e}. Using default fonts.")
#     robotto_regular = None # Use default font

# Path effects for heatmap labels
PATH_EFFECTS_HEATMAP = [path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()] # Slightly thinner stroke
PATH_EFFECTS_TEXT = [path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()] # For text on pitch

# Define colors used in the plots
VIOLET = config.VIOLET # Get from config if defined there, otherwise define here
GREEN = config.GREEN   # Get from config if defined there, otherwise define here
BG_COLOR = config.BG_COLOR if hasattr(config, 'BG_COLOR') else '#FAFAFA'
LINE_COLOR = config.LINE_COLOR if hasattr(config, 'LINE_COLOR') else '#222222'
CARRY_COLOR = 'darkgrey'
CARRY_LINESTYLE = (0, (3, 3)) # Adjust dash pattern
UNSUCCESSFUL_PASS_COLOR = 'black'
SHOT_TYPES = config.DEFAULT_SHOT_TYPES if hasattr(config, 'DEFAULT_SHOT_TYPES') else ['Goal', 'Miss', 'Attempt Saved', 'Post']

# --- Define Semicircle Parameters (should be globally consistent or passed) ---
# These constants define the "big chance" area and MUST MATCH those used in
# your `find_buildup_sequences` function and its `is_in_big_chance_area` helper.
# Assuming attacking L->R (goal at x=100) by default for these base definitions.
# Penalty box implied by original check: x >= 83, y between 21.1 and 78.9
BC_STD_BOX_EDGE_X_FURTHEST_FROM_GOAL = 83.0  # X-coord of penalty box edge (e.g., 18-yard line)
BC_STD_BOX_EDGE_X_GOAL_LINE = 100.0
BC_STD_BOX_EDGE_Y_MIN = 21.1
BC_STD_BOX_EDGE_Y_MAX = 78.9
BC_STD_BOX_CENTER_Y = (BC_STD_BOX_EDGE_Y_MIN + BC_STD_BOX_EDGE_Y_MAX) / 2.0 # 50.0

# Semicircle's flat diameter is on the line x = BC_STD_BOX_EDGE_X_FURTHEST_FROM_GOAL
# and it's centered vertically at BC_STD_BOX_CENTER_Y.
BC_SEMICIRCLE_DIAMETER_X_STD = BC_STD_BOX_EDGE_X_FURTHEST_FROM_GOAL # Center of the full circle for Arc/Wedge
BC_SEMICIRCLE_CENTER_Y_STD = BC_STD_BOX_CENTER_Y

# Calculate the radius needed for the semicircle to encompass the box.
_dx = BC_STD_BOX_EDGE_X_GOAL_LINE - BC_SEMICIRCLE_DIAMETER_X_STD
_dy = BC_STD_BOX_EDGE_Y_MAX - BC_SEMICIRCLE_CENTER_Y_STD # or use BC_STD_BOX_EDGE_Y_MIN
BC_SEMICIRCLE_RADIUS_STD = np.sqrt(_dx**2 + _dy**2)
BC_SEMICIRCLE_RADIUS_SQUARED_STD = BC_SEMICIRCLE_RADIUS_STD**2

# --- Plot Pass Density ---
# This function plots the Kernel Density Estimate (KDE) of pass start locations.
def plot_pass_density(ax, passes_df, team_name, cmap='viridis', is_away_team=False): # Added is_away_team
    """Plots a Kernel Density Estimate (KDE) of pass start locations."""
    print(f"Plotting pass density for {team_name}...")
    # Using VerticalPitch for this example plot
    pitch = VerticalPitch(pitch_type='opta', line_color='#000009', line_zorder=2, corner_arcs=True)
    pitch.draw(ax=ax)

    # Plot KDE if there's data
    if not passes_df.empty and 'x' in passes_df.columns and 'y' in passes_df.columns:
        pitch.kdeplot(passes_df.x, passes_df.y, ax=ax,
                      fill=True, levels=100, thresh=0, cut=4, cmap=cmap, zorder=1) # Lower zorder

        # --- Axis Inversion and Arrow Logic ---
        if is_away_team:
            ax.invert_xaxis() # Invert X axis for away team
            ax.invert_yaxis() # Invert Y axis for away team
            ax.annotate('', xy=(pitch.dim.left + 10, 65), xytext=(pitch.dim.left + 10, 35),
                         arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
        else:
            ax.annotate('', xy=(pitch.dim.left + 10, 65), xytext=(pitch.dim.left + 10, 35),
                         arrowprops=dict(facecolor='white', edgecolor='white', arrowstyle='->', lw=2))
        # --- End Inversion Logic ---
    else:
        print(f"Warning: No pass data or missing coordinates for density plot of {team_name}.")

    ax.set_title(f"Pass Density: {team_name}", fontsize=14, color='black')

# --- Plot Pass Heatmap ---
# This function plots a positional heatmap of pass start locations, including scatter points.
def plot_pass_heatmap(ax, passes_df, team_name, cmap='viridis', is_away_team=False):
    """Plots a positional heatmap of pass start locations, including scatter points."""
    print(f"Plotting pass heatmap for {team_name}...")
    # Using VerticalPitch for this plot
    pitch = VerticalPitch(pitch_type='opta', line_zorder=3, pitch_color='#FAFAFA', line_color='black', corner_arcs=True) # Lighter pitch color
    pitch.draw(ax=ax)

    # Calculate and plot heatmap if there's data
    if not passes_df.empty and 'x' in passes_df.columns and 'y' in passes_df.columns:
        # Calculate positional statistics
        bin_statistic = pitch.bin_statistic_positional(passes_df.x, passes_df.y, statistic='count',
                                                        positional='full', normalize=True)
        # Plot the heatmap
        pitch.heatmap_positional(bin_statistic, ax=ax, cmap=cmap, edgecolors='#AAAAAA', zorder=1) # Edges slightly visible

        # Scatter plot for individual pass starts
        pitch.scatter(passes_df.x, passes_df.y, c='black', s=8, ax=ax, alpha=0.2, zorder=2) # Small black dots with transparency

        # Label the heatmap bins with percentages
        pitch.label_heatmap(bin_statistic, color='white', fontsize=10, # Adjusted size and color
                              ax=ax, ha='center', va='center', str_format='{:.0%}', # Percentage format
                              path_effects=PATH_EFFECTS_HEATMAP, zorder=4) # Ensure labels are on top

        # --- Axis Inversion ---
        if is_away_team:
            ax.invert_xaxis() # Invert X axis for away team
            ax.invert_yaxis() # Invert Y axis for away team
        # --- End Inversion ---
    else:
        print(f"Warning: No pass data or missing coordinates for heatmap plot of {team_name}.")

    ax.set_title(f"Pass Heatmap: {team_name}", fontsize=14, color='black')


# --- Plot Pass Network ---
# This function visualizes the pass network for a team, including player nodes and pass lines.
def plot_pass_network(ax, passes_between_df, average_locs_and_count_df,
                      team_color, team_name, sub_list=[], is_away_team=False):
    """Visualizes the pass network for a team."""
    print(f"Plotting pass network for {team_name}...")
    # Constants
    MAX_LINE_WIDTH = 15
    MAX_MARKER_SIZE = 2000
    MIN_TRANSPARENCY = 0.1
    MAX_TRANSPARENCY = 0.9
    NODE_SIZE_FACTOR = 100 # Base size factor for nodes

    # Setup Pitch (Standard Horizontal for Network)
    pitch = Pitch(pitch_type='opta', corner_arcs=True, pitch_color='white', line_color='black', linewidth=2)
    pitch.draw(ax=ax)

    # Exit function gracefully if no data to plot
    if passes_between_df.empty or average_locs_and_count_df.empty:
        print(f"Warning: No pass network data to plot for {team_name}.")
        ax.set_title(f"{team_name}\nPassing Network (No Data)", color='black', size=15, fontweight='bold')
        return

    # Prepare Line Widths and Colors
    max_pass_count = passes_between_df.pass_count.max()
    if max_pass_count > 0:
        passes_between_df['width'] = (passes_between_df.pass_count / max_pass_count * MAX_LINE_WIDTH)
    else:
        passes_between_df['width'] = MAX_LINE_WIDTH / 2

    if max_pass_count > 0:
        c_transparency = passes_between_df.pass_count / max_pass_count
        c_transparency = (c_transparency * (MAX_TRANSPARENCY - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    else:
        c_transparency = (MAX_TRANSPARENCY + MIN_TRANSPARENCY) / 2

    line_color_rgba = np.array(to_rgba(team_color))
    line_colors_array = np.tile(line_color_rgba, (len(passes_between_df), 1))
    line_colors_array[:, 3] = c_transparency

    # Draw Pass Lines (using Opta coordinates for plotting)
    required_cols = ['pass_avg_x', 'pass_avg_y', 'pass_avg_x_end', 'pass_avg_y_end', 'width']
    if not all(col in passes_between_df.columns for col in required_cols):
         print(f"Error: Missing required columns for drawing lines in pass network for {team_name}.")
         return
    pitch.lines(passes_between_df.pass_avg_x, passes_between_df.pass_avg_y,
                 passes_between_df.pass_avg_x_end, passes_between_df.pass_avg_y_end,
                 lw=passes_between_df.width, color=line_colors_array, zorder=1, ax=ax)

    # Draw Player Nodes (using Opta coordinates for plotting)
    max_player_passes = average_locs_and_count_df['pass_count'].max()
    if max_player_passes > 0:
        # Scale marker size by pass count, adding a base size
        average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['pass_count'] / max_player_passes * (MAX_MARKER_SIZE - NODE_SIZE_FACTOR)) + NODE_SIZE_FACTOR
    else:
         average_locs_and_count_df['marker_size'] = NODE_SIZE_FACTOR

    required_node_cols = ['playerName', 'pass_avg_x', 'pass_avg_y', 'marker_size', 'jersey_number']
    if not all(col in average_locs_and_count_df.columns for col in required_node_cols):
        print(f"Error: Missing required columns for drawing nodes in pass network for {team_name}.")
        print(f"Available columns: {average_locs_and_count_df.columns.tolist()}")
        return

    for index, row in average_locs_and_count_df.iterrows():
        player_name = row['playerName']
        x, y = row['pass_avg_x'], row['pass_avg_y']
        size = row['marker_size']
        jersey = row['jersey_number'] if pd.notna(row['jersey_number']) else ''
        marker = 's' if player_name in sub_list else 'o'

        pitch.scatter(x, y, s=size, marker=marker, color='white', edgecolors='black', linewidth=1.5, alpha=1, ax=ax, zorder=2)

        try: # Format jersey number
             jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
        except (ValueError, TypeError):
             jersey_text = str(jersey) if pd.notna(jersey) else ''

        pitch.annotate(jersey_text, xy=(x, y), c='black', ha='center', va='center', size=10, weight='bold', ax=ax, zorder=3)

    # --- Add Annotations (Average Line, Titles, Legend, Direction) ---
    # Calculate median X position IN OPTA COORDINATES (0-100)
    avg_position_x_opta = average_locs_and_count_df['pass_avg_x'].median()

    # Draw vertical line and annotate with METERS if position is valid
    if pd.notna(avg_position_x_opta):
        # Draw the line using the OPTA coordinate
        ax.axvline(x=avg_position_x_opta, color='grey', linestyle='--', alpha=0.7, linewidth=1.5, zorder=0)

        # --- *** CONVERSION TO METERS FOR DISPLAY *** ---
        # Define standard pitch dimensions (adjust if your source uses different ones)
        PITCH_LENGTH_METERS = 105.0
        PITCH_WIDTH_METERS = 68.0 # Less relevant for X coordinate display

        # Convert the Opta X coordinate (0-100) to meters for the text label
        avg_position_meters = avg_position_x_opta * (PITCH_LENGTH_METERS / 100.0)
        # --- *** END CONVERSION *** ---

        # Adjust text position and alignment based on whether axis is inverted
        # Position text vertically near the top/bottom edge depending on inversion
        text_y_pos = pitch.dim.top + 3 if is_away_team else pitch.dim.bottom - 3 # Position near top/bottom edge
        horizontal_alignment = 'right' if is_away_team else 'left'

        # Display the value in METERS, formatted to one decimal place
        ax.text(avg_position_x_opta + 1, # Position text slightly right of the line (using Opta X for positioning)
                text_y_pos,              # Use calculated Y position
                f"{avg_position_meters:.1f}m", # Display the METERS value
                fontsize=11, color='grey', ha=horizontal_alignment, va='center') # Adjust alignment/va if needed
    else:
        print(f"Info: Could not calculate median pass x-position for {team_name}.")

    # Add main title
    ax.set_title(f"{team_name}\nPassing Network", color='black', size=18, fontweight='bold')    

    # --- Axis Inversion (X and Y) and Direction Arrow Logic ---
    if is_away_team:
        ax.invert_xaxis() # Invert X axis for away team plot
        ax.invert_yaxis() # Invert Y axis for away team plot
        # Place attacking direction text on the left side after inversion
        ax.text(2, 2, "○ = Starter\n□ = Substitute", color=team_color, size=10,
            ha='right', va='top')
        ax.text(0.98, -0.01, "← Attacking Direction", color=team_color, size=12, ha='right', va='bottom', transform=ax.transAxes)

    else:
        # Default: Place attacking direction text on the right side
        ax.text(2, 98, "○ = Starter\n□ = Substitute", color=team_color, size=10,
            ha='left', va='top')
        ax.text(0.02, -0.01, "Attacking Direction →", color=team_color, size=12, ha='left', va='bottom', transform=ax.transAxes)

    # --- End Inversion Logic ---

# --- Progressive Pass Maps ---
# This function visualizes progressive passes for a specific team on a pitch map.
def plot_progressive_passes(ax, df_prog_passes_team, zone_counts_team,
                            team_name, team_color, is_away_team=False,
                            excluded_qualifiers_list=None):
    """
    Visualizes progressive passes for a specific team on a pitch map.
    ... (Args remain the same) ...
    """
    print(f"Plotting progressive passes for {team_name}...")
    # Constants
    PITCH_COLOR = '#FAFAFA'
    LINE_COLOR = '#222222'

    # Setup Pitch (Standard Horizontal)
    pitch = Pitch(pitch_type='opta', pitch_color=PITCH_COLOR, line_color=LINE_COLOR,
                  linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)

    total_team_prog_count = zone_counts_team.get('total', 0)

    # Plot lines and points
    if total_team_prog_count > 0 and not df_prog_passes_team.empty:
        # ... (plotting pitch.lines and pitch.scatter remains the same) ...
        if not all(c in df_prog_passes_team for c in ['x', 'y', 'end_x', 'end_y']):
            print(f"Error: Missing coordinate columns for progressive pass lines for {team_name}.")
        else:
            pitch.lines(df_prog_passes_team.x, df_prog_passes_team.y,
                        df_prog_passes_team.end_x, df_prog_passes_team.end_y,
                        lw=3, transparent=True, comet=True, color=team_color,
                        ax=ax, alpha=0.6, zorder=1)
            pitch.scatter(df_prog_passes_team.end_x, df_prog_passes_team.end_y,
                        s=40, edgecolor=team_color, linewidth=1, color=PITCH_COLOR,
                        zorder=2, ax=ax, alpha=0.8)
    else:
        print(f"Info: No progressive passes to plot for {team_name}.")


    # --- Add Zone Lines and Annotations ---
    # Draw dashed lines dividing the pitch vertically
    ax.hlines(33.33, xmin=pitch.dim.left, xmax=pitch.dim.right, color=LINE_COLOR, linestyle='--', alpha=0.4, zorder=0)
    ax.hlines(66.67, xmin=pitch.dim.left, xmax=pitch.dim.right, color=LINE_COLOR, linestyle='--', alpha=0.4, zorder=0)

    # Calculate percentages
    # ... (percentage calculations remain the same) ...
    left_count = zone_counts_team.get('left', 0)
    mid_count = zone_counts_team.get('mid', 0)
    right_count = zone_counts_team.get('right', 0)
    left_perc = f"({(left_count / total_team_prog_count * 100):.0f}%)" if total_team_prog_count > 0 else "(0%)"
    mid_perc = f"({(mid_count / total_team_prog_count * 100):.0f}%)" if total_team_prog_count > 0 else "(0%)"
    right_perc = f"({(right_count / total_team_prog_count * 100):.0f}%)" if total_team_prog_count > 0 else "(0%)"


    # --- *** CALCULATE PITCH DIMENSIONS *** ---
    pitch_center_x = (pitch.dim.left + pitch.dim.right) / 2.0
    pitch_center_y = (pitch.dim.bottom + pitch.dim.top) / 2.0
    pitch_height = pitch.dim.top - pitch.dim.bottom # Calculate height
    # --- *** END CALCULATION *** ---

    # Add text annotations within each zone
    # Position text horizontally slightly into the defensive half relative to center
    text_x_pos = pitch_center_x * 0.6

    # Annotate Right Channel (Bottom Third: Opta Y usually 0-100)
    # Position Y = bottom edge + 1/6th of the height
    ax.text(text_x_pos, pitch.dim.bottom + (pitch_height * 1/6),
            f'{right_count}\n{right_perc}', color=team_color, fontsize=18, weight='bold', va='center', ha='center')
    # Annotate Middle Channel (Middle Third)
    ax.text(text_x_pos, pitch_center_y, # Use calculated center Y
            f'{mid_count}\n{mid_perc}', color=team_color, fontsize=18, weight='bold', va='center', ha='center')
    # Annotate Left Channel (Top Third)
    # Position Y = top edge - 1/6th of the height
    ax.text(text_x_pos, pitch.dim.top - (pitch_height * 1/6),
            f'{left_count}\n{left_perc}', color=team_color, fontsize=18, weight='bold', va='center', ha='center')

    # --- Title and Direction Text ---
    # ... (Title setting remains the same) ...
    count_text = f"{total_team_prog_count} Progressive Passes"
    ax.set_title(f"{team_name}\n{count_text}", color=LINE_COLOR, fontsize=20, fontweight='bold')

    # Get the figure object to add figure-level text
    fig = ax.figure
    exclusion_text = ""
    if excluded_qualifiers_list: # Check if the list is not None or empty
        # Format the list for display (e.g., capitalize, join with commas)
        formatted_exclusions = [q.capitalize() for q in excluded_qualifiers_list]
        exclusion_text = f"Excluding: {', '.join(formatted_exclusions)}"
        # Add text below the title using fig.text for better positioning relative to the figure
        fig.text(0.5, 0.90, exclusion_text, # Adjust y=0.90 as needed
                 color=LINE_COLOR, fontsize=10, ha='center', va='top')

    # --- Axis Inversion and Direction Arrow Logic ---
    # ... (Inversion logic remains the same) ...
    if is_away_team:
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.text(0.98, -0.01, "← Attacking Direction", color=team_color, size=12, ha='right', va='bottom', transform=ax.transAxes)
    else:
        ax.text(0.02, -0.01, "Attacking Direction →", color=team_color, size=12, ha='left', va='bottom', transform=ax.transAxes)
    # --- End Inversion Logic ---

# --- Shot Map and Stats Bar ---
# This function plots a shot map showing shots from both teams attacking the same goal,
# combined with a horizontal bar chart comparing key shooting stats.
def plot_shot_map_and_stats(ax, shots_df, home_stats, away_stats,
                            hteamName, ateamName, hcol, acol,
                            gw, league, date, # Match info needed for title
                            bg_color='#FAFAFA', line_color='#222222'):
    """
    Plots a shot map showing shots from both teams attacking the same goal,
    combined with a horizontal bar chart comparing key shooting stats.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        shots_df (pd.DataFrame): DataFrame containing ONLY shot events.
        home_stats (dict): Calculated stats for the home team.
        away_stats (dict): Calculated stats for the away team.
        hteamName (str): Home team name.
        ateamName (str): Away team name.
        hcol (str): Home team color hex.
        acol (str): Away team color hex.
        gw (str/int): Gameweek.
        league (str): League name.
        date (str): Formatted match date.
        bg_color (str, optional): Background color. Defaults to '#FAFAFA'.
        line_color (str, optional): Pitch line color. Defaults to '#222222'.
    """
    print("Plotting shot map and stats bar...")

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta', corner_arcs=True, pitch_color=bg_color,
                  linewidth=2, line_color=line_color)
    pitch.draw(ax=ax)
    # Adjust limits slightly for padding if needed, but default usually fine
    # ax.set_ylim(-0.5, 100.5)
    # ax.set_xlim(-0.5, 100.5)

    # --- Plot Shots (Both teams attacking the RIGHT goal visually) ---
    # Filter shots by team and outcome
    hGoalData = shots_df[(shots_df['team_name'] == hteamName) & (shots_df['type_name'] == 'Goal')]
    hPostData = shots_df[(shots_df['team_name'] == hteamName) & (shots_df['type_name'] == 'Post')]
    hSaveData = shots_df[(shots_df['team_name'] == hteamName) & (shots_df['type_name'] == 'Attempt Saved')]
    hMissData = shots_df[(shots_df['team_name'] == hteamName) & (shots_df['type_name'] == 'Miss')]

    aGoalData = shots_df[(shots_df['team_name'] == ateamName) & (shots_df['type_name'] == 'Goal')]
    aPostData = shots_df[(shots_df['team_name'] == ateamName) & (shots_df['type_name'] == 'Post')]
    aSaveData = shots_df[(shots_df['team_name'] == ateamName) & (shots_df['type_name'] == 'Attempt Saved')]
    aMissData = shots_df[(shots_df['team_name'] == ateamName) & (shots_df['type_name'] == 'Miss')]

    # Plot Home Team Shots (Inverting coordinates: 100-x, 100-y to attack right goal)
    if not hPostData.empty:
        pitch.scatter(100 - hPostData.x, 100 - hPostData.y, s=200, edgecolors=hcol, c=hcol, marker='o', ax=ax, label='Home Post/Bar')
    if not hSaveData.empty:
        pitch.scatter(100 - hSaveData.x, 100 - hSaveData.y, s=200, edgecolors=hcol, c='None', hatch='///////', marker='o', ax=ax, label='Home Saved')
    if not hMissData.empty:
        pitch.scatter(100 - hMissData.x, 100 - hMissData.y, s=200, edgecolors=hcol, c='None', marker='o', ax=ax, label='Home Miss')
    if not hGoalData.empty:
        pitch.scatter(100 - hGoalData.x, 100 - hGoalData.y, s=350, edgecolors='green', linewidths=0.8, c=bg_color, marker='football', zorder=3, ax=ax, label='Home Goal')

    # Plot Away Team Shots (Using original coordinates x, y to attack right goal)
    if not aPostData.empty:
        pitch.scatter(aPostData.x, aPostData.y, s=200, edgecolors=acol, c=acol, marker='o', ax=ax, label='Away Post/Bar')
    if not aSaveData.empty:
        pitch.scatter(aSaveData.x, aSaveData.y, s=200, edgecolors=acol, c='None', hatch='///////', marker='o', ax=ax, label='Away Saved')
    if not aMissData.empty:
        pitch.scatter(aMissData.x, aMissData.y, s=200, edgecolors=acol, c='None', marker='o', ax=ax, label='Away Miss')
    if not aGoalData.empty:
        pitch.scatter(aGoalData.x, aGoalData.y, s=350, edgecolors='green', linewidths=0.8, c=bg_color, marker='football', zorder=3, ax=ax, label='Away Goal')

    # Add a simple legend for shot types (optional)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)


    # --- Prepare Stats Bar Data ---
    # Define vertical positions for bars and text (more dynamic)
    num_stats = 7
    bar_height = 5
    gap = 2 # Gap between bars
    total_bar_height = num_stats * bar_height + (num_stats - 1) * gap
    bottom_y = (100 - total_bar_height) / 2 # Center the block vertically
    stat_labels = ["Goals", "xG", "xGOT", "Shots", "On Target", "xG/Shot", "Avg.Dist(m)"] # Note Avg.Dist in meters now
    stat_y_positions = [bottom_y + i * (bar_height + gap) for i in range(num_stats)][::-1] # Reverse for top-to-bottom

    # Get stats from dictionaries, handle potential NaN/None for formatting
    h_stats_values = [
        home_stats.get('goals', 0), home_stats.get('xg', 0), home_stats.get('xgot', 0),
        home_stats.get('total_shots', 0), home_stats.get('shots_on_target', 0),
        home_stats.get('xg_per_shot', 0),
        home_stats.get('avg_shot_distance', np.nan) # Keep NaN for formatting check
    ]
    a_stats_values = [
        away_stats.get('goals', 0), away_stats.get('xg', 0), away_stats.get('xgot', 0),
        away_stats.get('total_shots', 0), away_stats.get('shots_on_target', 0),
        away_stats.get('xg_per_shot', 0),
        away_stats.get('avg_shot_distance', np.nan) # Keep NaN for formatting check
    ]

    # --- Normalize Stats for Bar Widths (0-20 range, centered around x=50) ---
    # Define total bar width and start points
    total_bar_width = 20 # Max combined width for home+away
    center_x = 50
    bar_start_x = center_x - (total_bar_width / 2) # = 40

    norm_h = []
    norm_a = []
    for i in range(num_stats):
        h_val = h_stats_values[i]
        a_val = a_stats_values[i]
        # Handle NaN distance for normalization
        if pd.isna(h_val) and pd.isna(a_val):
            total = 0
        elif pd.isna(h_val):
            h_val = 0
            total = a_val
        elif pd.isna(a_val):
            a_val = 0
            total = h_val
        else:
            total = h_val + a_val

        # Special case for 0-0 goals
        if i == 0 and total == 0: # Goals stat index
            norm_h.append(total_bar_width / 2) # Equal split
            norm_a.append(total_bar_width / 2)
        elif total == 0:
            norm_h.append(total_bar_width / 2) # Equal split if both zero
            norm_a.append(total_bar_width / 2)
        else:
            norm_h.append((h_val / total) * total_bar_width if pd.notna(h_val) else 0)
            norm_a.append((a_val / total) * total_bar_width if pd.notna(a_val) else 0)

    # Calculate left offsets for away bars
    away_bar_left = [bar_start_x + h for h in norm_h]

    # --- Draw Stats Bars ---
    ax.barh(stat_y_positions, norm_h, height=bar_height, color=hcol, left=bar_start_x, zorder=4)
    ax.barh(stat_y_positions, norm_a, height=bar_height, left=away_bar_left, color=acol, zorder=4)

    # --- Add Stats Text ---
    text_color_on_bar = 'white' # Color for text labels inside bars
    text_color_outside = line_color # Color for numbers outside bars
    font_size = 11 # Smaller font size for stats
    font_weight = 'bold'

    for i in range(num_stats):
        y_pos = stat_y_positions[i]
        # Stat Label (centered)
        ax.text(center_x, y_pos, stat_labels[i], color=text_color_on_bar, fontsize=font_size + 1,
                ha='center', va='center', fontweight=font_weight, zorder=5)

        # Home Stat Value (right-aligned before bar starts)
        # Format differently for distance (1 decimal) vs others (int or 2 decimals)
        if i == 6: # Avg Distance
            h_text = f"{h_stats_values[i]:.1f}" if pd.notna(h_stats_values[i]) else "N/A"
        elif i == 5: # xG/Shot
             h_text = f"{h_stats_values[i]:.2f}"
        else: # Goals, xG, xGOT, Shots, SOT
             h_text = f"{h_stats_values[i]:.2f}" if isinstance(h_stats_values[i], float) else f"{int(h_stats_values[i])}"

        ax.text(bar_start_x - 1, y_pos, h_text, color=text_color_outside, fontsize=font_size,
                ha='right', va='center', fontweight=font_weight)

        # Away Stat Value (left-aligned after bars end)
        if i == 6: # Avg Distance
            a_text = f"{a_stats_values[i]:.1f}" if pd.notna(a_stats_values[i]) else "N/A"
        elif i == 5: # xG/Shot
             a_text = f"{a_stats_values[i]:.2f}"
        else: # Goals, xG, xGOT, Shots, SOT
             a_text = f"{a_stats_values[i]:.2f}" if isinstance(a_stats_values[i], float) else f"{int(a_stats_values[i])}"

        ax.text(bar_start_x + total_bar_width + 1, y_pos, a_text, color=text_color_outside, fontsize=font_size,
                ha='left', va='center', fontweight=font_weight)


    # --- Clean up Axes ---
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    # ax.set_xticks([]) # Already handled by tick_params
    # ax.set_yticks([]) # Already handled by tick_params


    # --- Titles and Match Info ---
    # Use fig_text for main title to place relative to figure, not axes
    # This requires passing the figure object to the function or accessing ax.figure
    fig = ax.figure # Get the figure object from the axes
    home_goals_final = home_stats.get('goals', 0)
    away_goals_final = away_stats.get('goals', 0)
    title_text = f"<{hteamName} {home_goals_final}> - <{(away_goals_final)} {ateamName}>"
    # Define the properties for each highlighted section (colors match order in string)
    highlight_props = [{'color': hcol}, {'color': acol}]

    fig_text(x=0.5, y=0.97, s=title_text,
             highlight_textprops=highlight_props, # Uncomment if highlight_text installed
             color=line_color, fontsize=22, fontweight='bold',
             ha='center', va='top', fig=fig) # Pass fig object

    subtitle_text = f"Gameweek {gw} | {league} | {date}\nShot Map & Stats Comparison"
    fig.text(0.5, 0.92, subtitle_text, color=line_color, fontsize=12, ha='center', va='top')

    # Add team labels near shots (optional)
    ax.text(0.2, 0.9, f"{hteamName}\nShots", color=hcol, size=12, ha='center', va='center', transform=ax.transAxes, fontweight='bold')
    ax.text(0.8, 0.9, f"{ateamName}\nShots", color=acol, size=12, ha='center', va='center', transform=ax.transAxes, fontweight='bold')

# --- Zone 14 / Half-Space Maps ---
# This function is called separately to plot Zone 14 and Half-Space passes
def plot_zone14_halfspace_map(ax, df_zone14, df_lhs, df_rhs, zone_stats_dict,
                              team_name, team_color, is_away_team=False,
                              zone14_color='orange', halfspace_color=None,
                              bg_color='#FAFAFA', line_color='#222222'):
    """
    Visualizes passes ending in Zone 14 and Half-Spaces using arrows,
    shaded zones, and hexagon markers (in defensive half) with counts and names,
    inspired by the user's original draw_pass_map function.
    """
    print(f"Plotting Zone 14 / Half-Space map for {team_name}...")

    # Set default halfspace color if not provided
    if halfspace_color is None: halfspace_color = team_color

    # Setup Pitch (Standard Horizontal)
    pitch = Pitch(pitch_type='opta', pitch_color=bg_color, line_color=line_color,
                  linewidth=2, corner_arcs=True)
    pitch.draw(ax=ax)

    # --- Zone Definitions (inspired by user's original code) ---
    # Zone 14
    zone14_x_min, zone14_x_max = 66.67, 82.0 # Similar X range
    zone14_y_min, zone14_y_max = 100/3, 200/3 # Y range from original code

    # Half-Spaces
    halfspace_x_min = 66.67 # Start of final third
    # Right Half-Space (Bottom of pitch visually) - coordinates from original code
    rhs_y_min, rhs_y_max = 100/6, 100/3
    # Left Half-Space (Top of pitch visually) - coordinates from original code
    lhs_y_min, lhs_y_max = 66.67, 83.33 # Note: original code used 66.67 to 83.33 here

    # Shade Attacking Zones
    ax.fill_betweenx(y=[zone14_y_min, zone14_y_max], x1=zone14_x_min, x2=zone14_x_max,
                     color=zone14_color, alpha=0.2, zorder=0, label='Zone 14')
    ax.fill_betweenx(y=[rhs_y_min, rhs_y_max], x1=halfspace_x_min, x2=pitch.dim.right,
                     color=halfspace_color, alpha=0.2, zorder=0, label='Right Half-Space')
    ax.fill_betweenx(y=[lhs_y_min, lhs_y_max], x1=halfspace_x_min, x2=pitch.dim.right,
                     color=halfspace_color, alpha=0.2, zorder=0, label='Left Half-Space')

    # Plot Pass Arrows into the zones (using mplsoccer's arrows for efficiency)
    arrow_props = {'alpha': 0.75, 'width': 1.5, 'headwidth': 4, 'headlength': 4, 'zorder': 1}
    if not df_zone14.empty and all(c in df_zone14 for c in ['x', 'y', 'end_x', 'end_y']):
        pitch.arrows(df_zone14.x, df_zone14.y, df_zone14.end_x, df_zone14.end_y,
                     color=zone14_color, ax=ax, **arrow_props)
    if not df_lhs.empty and all(c in df_lhs for c in ['x', 'y', 'end_x', 'end_y']):
        pitch.arrows(df_lhs.x, df_lhs.y, df_lhs.end_x, df_lhs.end_y,
                     color=halfspace_color, ax=ax, **arrow_props)
    if not df_rhs.empty and all(c in df_rhs for c in ['x', 'y', 'end_x', 'end_y']):
        pitch.arrows(df_rhs.x, df_rhs.y, df_rhs.end_x, df_rhs.end_y,
                     color=halfspace_color, ax=ax, **arrow_props)


    # --- Hexagon Annotations in Defensive Half (inspired by original code) ---
    # Fixed Coordinates for Hexagons
    hex_x_pos = 100 / 3  # Approx X=33.3, firmly in defensive half
    hex_y_z14 = 200 / 3  # Y-coordinate for Zone 14 Hex (higher on pitch)
    hex_y_hs_combined = 100 / 3 # Y-coordinate for Half-Space Hex (lower on pitch)

    # Appearance
    hex_size = 15000 # From original code
    hex_linewidth = 2 # From original code
    hex_alpha = 1 # From original code
    hex_text_fontsize_name = 20 # From original code
    hex_text_fontsize_count = 20 # From original code
    hex_text_color = line_color # Black text from original code
    hex_text_offset = 3 # Offset for count text from original code

    # Get Counts and Calculate Total Half-Space Count
    z14_count = zone_stats_dict.get('zone14', 0)
    lhs_count = zone_stats_dict.get('hs_left', 0)
    rhs_count = zone_stats_dict.get('hs_right', 0)
    total_hs_count = lhs_count + rhs_count

    # --- Draw Hexagon for Zone 14 ---
    pitch.scatter(hex_x_pos, hex_y_z14, marker='h', s=hex_size,
                  edgecolor=line_color, facecolor=zone14_color,
                  alpha=hex_alpha, linewidth=hex_linewidth, ax=ax, zorder=2)
    # Annotate Name (slightly above center)
    pitch.annotate("Zone14", xy=(hex_x_pos, hex_y_z14 + hex_text_offset),
                   color=hex_text_color, fontsize=hex_text_fontsize_name,
                   va='center', ha='center', ax=ax, zorder=3)
    # Annotate Count (slightly below center)
    pitch.annotate(f"{z14_count}", xy=(hex_x_pos, hex_y_z14 - hex_text_offset),
                   color=hex_text_color, fontsize=hex_text_fontsize_count,
                   va='center', ha='center', ax=ax, zorder=3)


    # --- Draw ONE Hexagon for Combined Half-Spaces ---
    pitch.scatter(hex_x_pos, hex_y_hs_combined, marker='h', s=hex_size,
                  edgecolor=line_color, facecolor=halfspace_color, # Use HS color
                  alpha=hex_alpha, linewidth=hex_linewidth, ax=ax, zorder=2)
    # Annotate Name (slightly above center)
    pitch.annotate("HalfSpace", xy=(hex_x_pos, hex_y_hs_combined + hex_text_offset),
                   color=hex_text_color, fontsize=hex_text_fontsize_name,
                   va='center', ha='center', ax=ax, zorder=3)
    # Annotate Count (slightly below center)
    pitch.annotate(f"{total_hs_count}", xy=(hex_x_pos, hex_y_hs_combined - hex_text_offset),
                   color=hex_text_color, fontsize=hex_text_fontsize_count,
                   va='center', ha='center', ax=ax, zorder=3)


    # --- Title and Direction Text (using current function's style) ---
    ax.set_title(f"{team_name}\nZone 14 & Half-Space", color=line_color, fontsize=20, fontweight='bold')

    # Axis inversion and attacking direction arrow
    if is_away_team:
        ax.invert_xaxis(); ax.invert_yaxis()
        # Place arrow on the left side for away team (attacking left after inversion)
        ax.text(0.98, -0.01, "← Attacking Direction", color=team_color, size=12, ha='right', va='bottom', transform=ax.transAxes) # Bottom left
    else:
        # Place arrow on the right side for home team (attacking right)
        ax.text(0.02, -0.01, "Attacking Direction →", color=team_color, size=12, ha='left', va='bottom', transform=ax.transAxes) # Bottom right

    # --- End Inversion Logic ---

# --- High Turnover Plot Function *** ---
# This function plots high turnover locations for both teams on a single pitch,
def plot_high_turnovers(ax, home_high_to_df_opta, away_high_to_df_opta, # Input DFs have Opta coords
                        hto_count, ato_count,
                        hteamName, ateamName, hcol, acol,
                        radius_meters=40.0, # Use meter radius directly
                        pitch_length_meters=105.0, pitch_width_meters=68.0, # Define pitch dims
                        bg_color='#FAFAFA', line_color='#222222'):
    """
    Plots high turnover locations using a meter-based pitch coordinate system.
    Converts input Opta coordinates to meters for plotting.
    Draws circles indicating the turnover zone radius in meters.
    """
    print(f"Plotting high turnovers (using {pitch_length_meters}x{pitch_width_meters}m pitch)...")

    # --- Setup Pitch with Meter Dimensions ---
    pitch = Pitch(pitch_type='custom', # Use 'custom' for specified dimensions
                  pitch_length=pitch_length_meters,
                  pitch_width=pitch_width_meters,
                  pitch_color=bg_color, line_color=line_color,
                  linewidth=2, corner_arcs=True, line_zorder=1) # Ensure lines are behind points/circles
    pitch.draw(ax=ax)
    ax.set_ylim(0, pitch_width_meters)
    ax.set_xlim(0, pitch_length_meters)

    # --- Define Goal Centers in METERS ---
    home_goal_center_x_m = pitch_length_meters
    away_goal_center_x_m = 0
    goal_center_y_m = pitch_width_meters / 2.0

    # --- Convert Turnover Coordinates to Meters for Plotting ---
    def convert_opta_to_meters(df_opta, length_m, width_m):
        if df_opta.empty:
            return pd.DataFrame(columns=['x_m', 'y_m']) # Return empty if no data
        df_m = pd.DataFrame()
        # Apply conversion only if columns exist
        if 'x' in df_opta.columns:
            df_m['x_m'] = df_opta['x'] * (length_m / 100.0)
        if 'y' in df_opta.columns:
            # Opta Y corresponds to pitch width
            df_m['y_m'] = df_opta['y'] * (width_m / 100.0)
        return df_m

    home_high_to_df_m = convert_opta_to_meters(home_high_to_df_opta, pitch_length_meters, pitch_width_meters)
    away_high_to_df_m = convert_opta_to_meters(away_high_to_df_opta, pitch_length_meters, pitch_width_meters)

    # --- Draw Circles using plt.Circle and METERS ---
    circle_style = dict(fill=True, alpha=0.15, linestyle='--', linewidth=2, zorder=0)

    # Circle around Home Goal (Zone where Away Team makes High TOs)
    # Center is in meters, radius is in meters
    away_zone_circle = patches.Circle((home_goal_center_x_m, goal_center_y_m), radius_meters,
                                      color=acol, **circle_style)
    ax.add_patch(away_zone_circle)

    # Circle around Away Goal (Zone where Home Team makes High TOs)
    # Center is in meters, radius is in meters
    home_zone_circle = patches.Circle((away_goal_center_x_m, goal_center_y_m), radius_meters,
                                      color=hcol, **circle_style)
    ax.add_patch(home_zone_circle)
    # --- End Circle Drawing ---

    # --- Plot Turnover Points using METER Coordinates ---
    scatter_style = dict(s=250, marker='o', linewidth=1.5, zorder=2) # Zorder higher than circles
    if not home_high_to_df_m.empty and 'x_m' in home_high_to_df_m.columns and 'y_m' in home_high_to_df_m.columns:
        pitch.scatter(home_high_to_df_m.x_m, home_high_to_df_m.y_m,
                      c=hcol, edgecolor=line_color, ax=ax, **scatter_style)
    else:
        print(f"Info: No home high turnovers to plot for {hteamName}.")

    if not away_high_to_df_m.empty and 'x_m' in away_high_to_df_m.columns and 'y_m' in away_high_to_df_m.columns:
        pitch.scatter(away_high_to_df_m.x_m, away_high_to_df_m.y_m,
                      c=acol, edgecolor=line_color, ax=ax, **scatter_style)
    else:
         print(f"Info: No away high turnovers to plot for {ateamName}.")

    # --- Add Count Annotations (Position using Meter Coordinates) ---
    # Get pitch dimensions in meters from the pitch object for positioning
    top_edge_m = pitch.dim.top # Should be pitch_width_meters
    bottom_edge_m = pitch.dim.bottom # Should be 0
    left_edge_m = pitch.dim.left # Should be 0
    right_edge_m = pitch.dim.right # Should be pitch_length_meters

    # Position text near the top corners using meter coordinates
    ax.text(left_edge_m + 5, top_edge_m - 3, # Near top-left
            f"{hteamName}\nHigh Turnovers: {hto_count}",
            color=hcol, size=16, ha='left', va='top', fontweight='bold')
    ax.text(right_edge_m - 5, top_edge_m - 3, # Near top-right
            f"{ateamName}\nHigh Turnovers: {ato_count}",
            color=acol, size=16, ha='right', va='top', fontweight='bold')

    # Optional: Add title
    # ax.set_title(f"High Turnovers ({radius_meters}m Radius)", fontsize=20, fontweight='bold')

# --- Chance Creation Plot Function ---
# This function visualizes chance creation zones using a heatmap of pass origins
# and colored arrows for key passes and assists.
def plot_chance_creation(ax, df_chances_team, team_name, team_color, is_away_team=False,
                         bg_color='#FAFAFA', line_color='#222222',
                         kp_color=VIOLET, assist_color=GREEN):
    """
    Visualizes chance creation zones using a heatmap of pass origins and
    colored arrows for key passes and assists.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        df_chances_team (pd.DataFrame): DataFrame of the team's chance-creating passes
                                       (must include 'x', 'y', 'end_x', 'end_y',
                                       'is_key_pass', 'is_assist' columns).
        team_name (str): Name of the team.
        team_color (str): Primary color hex for the team (used for heatmap).
        is_away_team (bool, optional): Flag to invert axes. Defaults to False.
        bg_color (str, optional): Background color. Defaults to '#FAFAFA'.
        line_color (str, optional): Pitch line color. Defaults to '#222222'.
        kp_color (str, optional): Color for key pass arrows. Defaults to VIOLET.
        assist_color (str, optional): Color for assist arrows. Defaults to GREEN.
    """
    print(f"Plotting chance creation map for {team_name}...")

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta', line_color=line_color, corner_arcs=True,
                  line_zorder=2, pitch_color=bg_color, linewidth=2)
    pitch.draw(ax=ax)

    # Invert axes if it's the away team
    if is_away_team:
        ax.invert_xaxis()
        ax.invert_yaxis()

    # Exit if no data
    if df_chances_team.empty:
        print(f"Info: No chance creation data to plot for {team_name}.")
        ax.set_title(f"{team_name}\nChance Creation Zones (No Data)",
                     color=line_color, fontsize=18, fontweight='bold')
        return

    # --- Heatmap of Pass Origins ---
    # Create a colormap based on the team color
    team_cmap = LinearSegmentedColormap.from_list(f"{team_name}_cmap", [bg_color, team_color], N=20)
    # Define bins (e.g., 7x6 as in original code)
    bins = (7, 6)
    # Calculate heatmap statistics (count in each bin)
    bin_statistic = pitch.bin_statistic(df_chances_team.x, df_chances_team.y, bins=bins, statistic='count', normalize=False)
    # Draw heatmap
    pitch.heatmap(bin_statistic, ax=ax, cmap=team_cmap, edgecolors='#d9d9d9', alpha=0.6, zorder=0)
    # Draw scatter points for individual origins
    pitch.scatter(df_chances_team.x, df_chances_team.y, c='grey', s=8, ax=ax, alpha=0.3, zorder=1)
    # Label heatmap bins
    path_eff = [path_effects.Stroke(linewidth=2, foreground=bg_color), path_effects.Normal()]
    pitch.label_heatmap(bin_statistic, color=line_color, fontsize=20, # Adjust fontsize
                          ax=ax, ha='center', va='center', str_format='{:.0f}',
                          exclude_zeros=True, path_effects=path_eff, zorder=4)

    # --- Plot Arrows for Key Passes and Assists ---
    # Filter df for key passes and assists
    df_kp = df_chances_team[df_chances_team['is_key_pass']].copy()
    df_as = df_chances_team[df_chances_team['is_assist']].copy()

    arrow_props = {'width': 1.5, 'headwidth': 5, 'headlength': 5, 'alpha': 0.8, 'zorder': 3}

    if not df_kp.empty:
        pitch.arrows(df_kp.x, df_kp.y, df_kp.end_x, df_kp.end_y,
                     color=kp_color, ax=ax, **arrow_props)
    if not df_as.empty:
        pitch.arrows(df_as.x, df_as.y, df_as.end_x, df_as.end_y,
                     color=assist_color, ax=ax, **arrow_props)

    # --- Titles and Annotations ---
    total_chances = len(df_chances_team)
    # Legend text
    legend_text = f"Violet Arrow = Key Pass ({len(df_kp)})\nGreen Arrow = Assist ({len(df_as)})"
    # Title text
    title = f"{team_name}\nChance Creation Zones"
    # Total count text
    count_text = f"Total Chances Created = {total_chances}"

    # Position annotations based on home/away
    if is_away_team:
        ax.text(0.02, 0.02, legend_text, color=line_color, size=12, ha='left', va='top', transform=ax.transAxes)
        ax.text(0.98, 0, "← Attacking Direction", color=team_color, size=12, ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.50, 0.97, count_text, color=team_color, fontsize=12, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)
    else:
        ax.text(0.98, 0.02, legend_text, color=line_color, size=12, ha='right', va='top', transform=ax.transAxes)
        ax.text(0.02, 0, "Attacking Direction →", color=team_color, size=12, ha='left', va='bottom', transform=ax.transAxes)
        ax.text(0.50, 0.97, count_text, color=team_color, fontsize=12, fontweight='bold', ha='center', va='bottom', transform=ax.transAxes)

    ax.set_title(title, color=line_color, fontsize=18, fontweight='bold') # Slightly smaller title fontsize

# --- Defensive Block Plot Function ---
# This function visualizes a team's defensive block using a KDE heatmap of action locations
# and player nodes sized by action count at their median location.
def plot_defensive_block(ax, df_defensive_actions_team, df_player_agg_team,
                         team_name, team_color, sub_list=[], is_away_team=False,
                         bg_color='#FAFAFA', line_color='#222222', scatter_actions=True):
    """
    Visualizes a team's defensive block using a KDE heatmap of action locations
    and player nodes sized by action count at their median location.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        df_defensive_actions_team (pd.DataFrame): All defensive actions for the team.
        df_player_agg_team (pd.DataFrame): Aggregated stats (median_x, median_y, action_count)
                                           per player for the team.
        team_name (str): Name of the team.
        team_color (str): Primary color hex for the team.
        sub_list (list, optional): List of player names who were substitutes. Defaults to [].
        is_away_team (bool, optional): Flag to invert axes. Defaults to False.
        bg_color (str, optional): Background color. Defaults to '#FAFAFA'.
        line_color (str, optional): Pitch line color. Defaults to '#222222'.
        scatter_actions (bool, optional): Whether to plot individual actions as small
                                          markers. Defaults to True.
    """
    print(f"Plotting defensive block for {team_name}...")

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta', pitch_color=bg_color, line_color=line_color,
                  linewidth=2, line_zorder=2, corner_arcs=True)
    pitch.draw(ax=ax)

    # Invert axes if away team
    if is_away_team:
        ax.invert_xaxis()
        ax.invert_yaxis()

    # Exit if no aggregated player data
    if df_player_agg_team.empty:
        print(f"Warning: No aggregated player data to plot defensive block for {team_name}.")
        ax.set_title(f"{team_name}\nDefensive Block (No Player Data)", color=line_color, fontsize=18, fontweight='bold')
        return

    # --- Plot KDE Heatmap ---
    if not df_defensive_actions_team.empty:
        # Create team-specific colormap fading to background
        team_cmap = LinearSegmentedColormap.from_list(f"{team_name}_kde_cmap", [bg_color, team_color], N=100)
        pitch.kdeplot(df_defensive_actions_team.x, df_defensive_actions_team.y, ax=ax,
                      fill=True, levels=100, # Fewer levels might look cleaner
                      thresh=0.05, # Adjust threshold to show relevant areas
                      cut=4, cmap=team_cmap, alpha=0.5, zorder=0) # Lower alpha for subtlety
        if scatter_actions:
            pitch.scatter(df_defensive_actions_team.x, df_defensive_actions_team.y,
                          s=15, marker='o', color='yellow', edgecolors='black', linewidth=0.5, # Adjusted style
                          alpha=0.3, ax=ax, zorder=1) # zorder=1 (above heatmap)
            print("  Included scatter plot of individual defensive actions.")
    else:
        print(f"Warning: No individual defensive actions to plot heatmap for {team_name}.")


    # --- Plot Player Nodes ---
    # Calculate marker size based on action count
    MAX_MARKER_SIZE = 3000 # Adjusted max size
    NODE_BASE_SIZE = 200   # Minimum size for visibility
    max_count = df_player_agg_team['action_count'].max()
    if max_count > 0:
        df_player_agg_team['marker_size'] = ((df_player_agg_team['action_count'] / max_count) * (MAX_MARKER_SIZE - NODE_BASE_SIZE)) + NODE_BASE_SIZE
    else:
        df_player_agg_team['marker_size'] = NODE_BASE_SIZE

    # Plot nodes and annotate with jersey numbers
    for index, row in df_player_agg_team.iterrows():
        x, y = row['median_x'], row['median_y']
        size = row['marker_size']
        jersey = row['jersey_number'] if pd.notna(row['jersey_number']) else ''
        marker = 's' if row['playerName'] in sub_list else 'o'

        pitch.scatter(x, y, s=size, marker=marker,
                      color=bg_color, edgecolor=line_color, linewidth=1.5, alpha=0.9, zorder=3, ax=ax)

        try: # Format jersey number
             jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
        except (ValueError, TypeError):
             jersey_text = str(jersey) if pd.notna(jersey) else ''

        pitch.annotate(jersey_text, xy=(x, y), c=line_color, ha='center', va='center', size=12, weight='bold', ax=ax, zorder=4)


    # --- Add Average Defensive Line Height ---
    # Calculate mean X position (Opta coordinates)
    avg_def_line_x_opta = df_player_agg_team['median_x'].mean()

    if pd.notna(avg_def_line_x_opta):
        # Draw line using Opta coordinate
        ax.axvline(x=avg_def_line_x_opta, color='dimgray', linestyle='--', alpha=0.8, linewidth=2, zorder=1)

        # Convert to meters for display (using standard 105m length)
        pitch_length_meters = 105.0
        avg_def_line_meters = avg_def_line_x_opta * (pitch_length_meters / 100.0)

        # Position text based on team (adjust Y offset as needed)
        text_y_pos = pitch.dim.top + 3 if is_away_team else pitch.dim.bottom - 3 # Position near top/bottom edge
        horizontal_alignment = 'right' if is_away_team else 'left'

        ax.text(avg_def_line_x_opta + 1, # Offset text slightly from line
                text_y_pos,
                f"Avg Line: {avg_def_line_meters:.1f}m",
                fontsize=12, color='dimgray', ha=horizontal_alignment, va='center')
    else:
        print(f"Info: Could not calculate average defensive line for {team_name}.")


    # --- Titles and Annotations ---
    ax.set_title(f"{team_name}\nDefensive Block", color=line_color, fontsize=18, fontweight='bold')

    # Add legend/direction text
    legend_text = "○ = Starter | □ = Substitute\nSize = Def. Action Count"
    if is_away_team:
        ax.text(0.02, -0.015, legend_text, color='black', size=10, ha='left', va='bottom', transform=ax.transAxes)
        ax.text(0.98, 0, "← Attacking Direction", color=team_color, size=12, ha='right', va='bottom', transform=ax.transAxes)
    else:
        ax.text(0.98, -0.015, legend_text, color='black', size=10, ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.02, 0, "Attacking Direction →", color=team_color, size=12, ha='left', va='bottom', transform=ax.transAxes)

# --- NEW Defensive Shape Plot (Convex Hull) ---
def plot_defensive_hull(ax, df_player_agg_team,
                        team_name, team_color, sub_list=[], is_away_team=False,
                        bg_color='#FAFAFA', line_color='#222222'):
    """
    Visualizes a team's defensive shape using the convex hull around average
    player defensive positions.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        df_player_agg_team (pd.DataFrame): Aggregated stats (median_x, median_y, action_count)
                                           per player for the team.
        team_name (str): Name of the team.
        team_color (str): Primary color hex for the team.
        sub_list (list, optional): List of player names who were substitutes. Defaults to [].
        is_away_team (bool, optional): Flag to invert axes. Defaults to False.
        bg_color (str, optional): Background color. Defaults to '#FAFAFA'.
        line_color (str, optional): Pitch line color. Defaults to '#222222'.
    """
    print(f"Plotting defensive hull for {team_name}...")

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta', pitch_color=bg_color, line_color=line_color,
                  linewidth=2, line_zorder=1, corner_arcs=True)
    pitch.draw(ax=ax)

    # Invert axes if away team
    if is_away_team:
        ax.invert_xaxis()
        ax.invert_yaxis()

    # Exit if no player data
    if df_player_agg_team.empty or len(df_player_agg_team) < 3: # Need at least 3 points for a hull
        print(f"Warning: Not enough player data ({len(df_player_agg_team)}) to plot hull for {team_name}.")
        ax.set_title(f"{team_name}\nDefensive Shape (Hull - No Data)", color=line_color, fontsize=18, fontweight='bold')
        return

    # --- Plot Player Nodes ---
    # (Using fixed size for hull plot for clarity, or keep size by count if preferred)
    NODE_SIZE = 800
    for index, row in df_player_agg_team.iterrows():
        x, y = row['median_x'], row['median_y']
        jersey = row['jersey_number'] if pd.notna(row['jersey_number']) else ''
        marker = 's' if row['playerName'] in sub_list else 'o'

        pitch.scatter(x, y, s=NODE_SIZE, marker=marker,
                      color=bg_color, edgecolor=line_color, linewidth=1.5, alpha=0.9, zorder=3, ax=ax)

        try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
        except (ValueError, TypeError): jersey_text = str(jersey) if pd.notna(jersey) else ''
        pitch.annotate(jersey_text, xy=(x, y), c=line_color, ha='center', va='center', size=10, weight='bold', ax=ax, zorder=4)

    # --- Calculate and Plot Convex Hull ---
    # Prepare points for ConvexHull function (needs Nx2 array)
    # Consider filtering out the GK here if possible/desired
    # For now, use all players provided
    points = df_player_agg_team[['median_x', 'median_y']].values

    try:
        hull = ConvexHull(points)
        # Get the vertices forming the hull
        hull_points = points[hull.vertices]

        # Draw the hull using pitch.polygon
        pitch.polygon([hull_points], ax=ax, color=team_color, fill=True, alpha=0.2, zorder=2)
        # Optionally draw the hull outline
        pitch.polygon([hull_points], ax=ax, color=team_color, fill=False, lw=2, ls='--', zorder=2)

    except Exception as e:
        # Catch potential errors during hull calculation (e.g., collinear points)
        print(f"Warning: Could not calculate or plot Convex Hull for {team_name}: {e}")


    # --- Titles and Annotations ---
    ax.set_title(f"{team_name}\nDefensive Shape (Convex Hull)", color=line_color, fontsize=18, fontweight='bold')

    # Add legend/direction text
    legend_text = "○ = Starter | □ = Substitute"
    if is_away_team:
        ax.text(0.02, 0.02, legend_text, color='black', size=10, ha='left', va='bottom', transform=ax.transAxes)
        ax.text(0.98, 0.02, "← Attacking Direction", color=team_color, size=12, ha='right', va='bottom', transform=ax.transAxes)
    else:
        ax.text(0.98, 0.02, legend_text, color='black', size=10, ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.02, 0.02, "Attacking Direction →", color=team_color, size=12, ha='left', va='bottom', transform=ax.transAxes)


# --- Defensive Shape Plot (Voronoi) ---
def plot_defensive_voronoi(ax, df_player_agg_team,
                           team_name, team_color, sub_list=[], is_away_team=False,
                           bg_color='#FAFAFA', line_color='#222222'):
    """
    Visualizes a team's defensive coverage using a Voronoi diagram based on
    average player defensive positions, plotted using scipy.spatial.voronoi_plot_2d.
    """
    print(f"Plotting defensive Voronoi for {team_name}...")

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta', pitch_color=bg_color, line_color=line_color,
                  linewidth=1, line_zorder=1, corner_arcs=True)
    pitch.draw(ax=ax)

    # Invert axes if away team
    if is_away_team:
        ax.invert_xaxis()
        ax.invert_yaxis()

    # Exit if no player data
    if df_player_agg_team.empty or len(df_player_agg_team) < 3: # Voronoi often needs >= 3 points
        print(f"Warning: Not enough player data ({len(df_player_agg_team)}) to plot Voronoi for {team_name}.")
        ax.set_title(f"{team_name}\nDefensive Coverage (Voronoi - No Data)", color=line_color, fontsize=18, fontweight='bold')
        return

    # --- Prepare Points for Voronoi Calculation ---
    points_x = df_player_agg_team['median_x'].values
    points_y = df_player_agg_team['median_y'].values
    points = np.column_stack((points_x, points_y))

    # --- Calculate and Plot Voronoi using scipy ---
    try:
        vor = Voronoi(points) # Calculate using scipy

        # --- Plot using voronoi_plot_2d ---
        voronoi_plot_2d(vor, ax=ax,
                        show_vertices=False, # Don't plot the Voronoi vertices
                        show_points=False,    # Don't plot the input points again (we'll do it better)
                        line_colors='black', # Color for the Voronoi cell edges
                        line_width=1,
                        line_alpha=0.6,
                        point_size=0) # Set point size to 0 as we plot points separately
        # --- Style Note: voronoi_plot_2d has limited direct styling options for lines (color, width, alpha) ---
        # We may need to manually plot lines if more customization is needed.

        # Ensure plot limits are set by the pitch dimensions
        ax.set_xlim(pitch.dim.left, pitch.dim.right)
        ax.set_ylim(pitch.dim.bottom, pitch.dim.top)

    except Exception as e:
        print(f"Warning: Could not calculate or plot Voronoi diagram for {team_name}: {e}")
        # Continue to plot points even if Voronoi fails

    # --- Plot Player Nodes ---
    # Plot nodes ON TOP of Voronoi lines (higher zorder)
    NODE_SIZE = 800
    for index, row in df_player_agg_team.iterrows():
        x, y = row['median_x'], row['median_y']
        jersey = row['jersey_number'] if pd.notna(row['jersey_number']) else ''
        marker = 's' if row['playerName'] in sub_list else 'o'
        # Use zorder=3 or higher to ensure nodes are above Voronoi lines (default zorder is low)
        pitch.scatter(x, y, s=NODE_SIZE, marker=marker, color=bg_color, edgecolor=line_color, linewidth=1.5, alpha=1, zorder=3, ax=ax)
        try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
        except (ValueError, TypeError): jersey_text = str(jersey) if pd.notna(jersey) else ''
        # Use zorder=4 for text
        pitch.annotate(jersey_text, xy=(x, y), c=line_color, ha='center', va='center', size=10, weight='bold', ax=ax, zorder=4)


    # --- Titles and Annotations ---
    # (Title and legend logic remains the same)
    ax.set_title(f"{team_name}\nDefensive Coverage (Voronoi)", color=line_color, fontsize=18, fontweight='bold')
    legend_text = "○ = Starter | □ = Substitute"
    if is_away_team:
        ax.text(0.02, 0.02, legend_text, color='black', size=10, ha='left', va='bottom', transform=ax.transAxes)
        ax.text(0.98, 0.02, "← Attacking Direction", color=team_color, size=12, ha='right', va='bottom', transform=ax.transAxes)
    else:
        ax.text(0.98, 0.02, legend_text, color='black', size=10, ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.02, 0.02, "Attacking Direction →", color=team_color, size=12, ha='left', va='bottom', transform=ax.transAxes)

# --- Pass-to-Shot Sequence Plot Function ---
# This function visualizes a single pass-to-shot sequence on a pitch.
# It draws the pass lines, shot line, and player markers at the start of each action.
def plot_individual_shot_sequence(ax, sequence_data, team_name, team_color, sequence_id,
                                  carry_threshold_meters=10.0, # Min distance for a carry
                                  pitch_length_meters=105.0,  # For converting threshold
                                  bg_color=BG_COLOR, line_color=LINE_COLOR):
    """
    Plots a single pass-to-shot sequence with a detailed title including time,
    last pass type, play type, and shot body part.
    """
    # print(f"Plotting sequence {sequence_id} for {team_name}...") # Optional debug

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta',
                  pitch_color=bg_color, line_color=line_color,
                  linewidth=1.5, corner_arcs=True,
                  pad_top=2, pad_bottom=2, pad_left=2, pad_right=2) # Adjust padding values (default is usually larger)
    pitch.draw(ax=ax)

     # --- *** Try Setting Limits AFTER Drawing *** ---
    ax.set_xlim(pitch.dim.left - 2, pitch.dim.right + 2) # Slightly beyond pitch edge
    ax.set_ylim(pitch.dim.bottom - 3, pitch.dim.top + 3) # Slightly beyond pitch edge
    # --- *** End Setting Limits *** ---

    if sequence_data is None or sequence_data.empty:
        print(f"Warning: No data for sequence {sequence_id}")
        ax.set_title(f"Seq {sequence_id}\n(No Data)", color=line_color, fontsize=10) # Smaller font
        ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
        return

    # --- Ensure Coordinate Columns are Numeric ---
    coord_cols = ['x', 'y', 'end_x', 'end_y']; sequence_data_numeric = sequence_data.copy()
    for col in coord_cols:
        if col in sequence_data_numeric.columns: sequence_data_numeric[col] = pd.to_numeric(sequence_data_numeric[col], errors='coerce')
        else: sequence_data_numeric[col] = np.nan
        sequence_data_numeric[col] = sequence_data_numeric[col].astype(float)

    passes = sequence_data_numeric[sequence_data_numeric['type_name'] == 'Pass'].copy()
    shot_event_df = sequence_data_numeric[~sequence_data_numeric['type_name'].isin(['Pass', None])].iloc[-1:].copy()
    is_shot_sequence = not shot_event_df.empty

    # --- Extract Information for Detailed Title ---
    time_str = "Time N/A"
    last_pass_desc = ""
    play_type_desc = "" # Default
    shot_by_desc = ""
    shot_outcome_desc = "Unknown"

    # --- Calculate Carry Threshold in Opta Units ---
    opta_units_per_meter_x = 100.0 / pitch_length_meters
    carry_threshold_opta = carry_threshold_meters * opta_units_per_meter_x

    if is_shot_sequence:
        shot_event = shot_event_df.iloc[0] # Get the Series
        shot_outcome_desc = shot_event.get('type_name', 'Unknown')
        # print(f"Info: Shot outcome detected for seq {sequence_id}: {shot_outcome_desc}.")

        # Time of shot
        time_min = shot_event.get('timeMin')
        time_sec = shot_event.get('timeSec')
        if pd.notna(time_min) and pd.notna(time_sec):
            try: time_str = f"{int(time_min)}'{int(time_sec):02d}\""
            except ValueError: print(f"Warning: Non-integer timeMin/Sec for seq {sequence_id}")

        # Play Type of Shot (Check qualifiers on the SHOT event itself first)
        # Ensure qualifier columns are checked as strings '1' if that's how they are stored
        if str(shot_event.get('Set piece')) == '1': play_type_desc = "Set Piece" # Q24 - Shot occurred from a crossed free kick
        elif str(shot_event.get('Regular play')) == '1': play_type_desc = "Open play" # Q22 - Shot occurred from regular play
        elif str(shot_event.get('Fast break')) == '1': play_type_desc = "Fast break" # Q23 - Shot occurred from a fast break
        elif str(shot_event.get('From corner')) == '1': play_type_desc = "Corner" # Q25 - Shot occurred from a corner kick
        elif str(shot_event.get('Free kick')) == '1': play_type_desc = "Direct Free Kick" # Q26 - Shot occurred from a direct free kick
        elif str(shot_event.get('Throw-in set piece')) == '1': play_type_desc = "Throw-in set piece" # Q160 - Shot occurred from a throw-in set piece
        elif str(shot_event.get('Corner situation')) == '1': play_type_desc = "2nd Phase Corner" # Q96 - Shot occurred from a 2nd phase attack following a corner situation
        elif str(shot_event.get('Penalty')) == '1': play_type_desc = "Penalty" # Q9 - Shot occurred from a penalty kick

        # Shot Body Part
        if str(shot_event.get('Head')) == '1': shot_by_desc = "Head" # Q15
        elif str(shot_event.get('Right footed')) == '1': shot_by_desc = "Right F." # Q72
        elif str(shot_event.get('Left footed')) == '1': shot_by_desc = "Left F." # Q73
        elif str(shot_event.get('Other body part')) == '1': shot_by_desc = "Other" # Q21
        # Else: shot_by_desc remains "" (e.g. foot if not specified, common default)

    # --- Extract Information from the LAST PASS in the sequence (if any) ---
    if not passes.empty: # If there are any passes in the sequence
        successful_passes = passes[passes['outcome'] == 'Successful']  # Filter for successful passes
        if not successful_passes.empty:  # Check if there are any successful passes
            last_pass_event = successful_passes.iloc[-1]  # Get the last successful pass event
        else: last_pass_event = passes.iloc[-1] # Get the last pass event
        if str(last_pass_event.get('cross')) == '1': last_pass_desc = "Cross"
        elif str(last_pass_event.get('lb')) == '1': last_pass_desc = "Long Ball"
        elif str(last_pass_event.get('Corner taken')) == '1': last_pass_desc = "Corner Pass"
        elif str(last_pass_event.get('Free kick taken')) == '1': last_pass_desc = "FK Pass"
        else: last_pass_desc = "Pass" # Default if no specific type qualifier found

        if last_pass_desc: last_pass_desc = f"{last_pass_desc}"
    elif is_shot_sequence and len(sequence_data_numeric) == 1: # Shot only, no passes
        last_pass_desc = "No Preceding Pass"

    # --- Construct Title ---
    title_line1 = f"Time: {time_str} -> {shot_outcome_desc}"
    title_line2_parts = [part for part in [play_type_desc, last_pass_desc, shot_by_desc] if part]
    title_line2 = " | ".join(title_line2_parts) if title_line2_parts else "Details N/A"

    # --- Plotting Logic (Passes, Shot, Nodes) ---
    # ... (Plotting logic should be mostly the same, using current_shot_color for the shot line) ...
    outcome_colors = {'Miss': 'grey', 'Attempt Saved': 'blue', 'Goal': GREEN, 'Post': 'orange'}
    current_shot_color = outcome_colors.get(shot_outcome_desc, 'red')

    # Iterate through all events in the sequence to plot them
    # Keep track of the last pass's end point to detect carries
    last_pass_end_x = None
    last_pass_end_y = None

    # --- Plot Player Markers at START of each action ---
    node_size = 350 # Slightly smaller nodes
    required_node_cols = ['x', 'y', 'playerName', 'Mapped Jersey Number']
    if all(c in sequence_data_numeric.columns for c in required_node_cols):
        valid_starts = sequence_data_numeric.dropna(subset=['x', 'y'])
        for index, event in valid_starts.iterrows():
            jersey = event.get('Mapped Jersey Number', '')
            try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
            except: jersey_text = str(jersey) if pd.notna(jersey) else ''
            event_x, event_y = event['x'], event['y']

            pitch.scatter(event_x, event_y, s=node_size, marker='o',
                          facecolor=team_color, edgecolor=line_color, linewidth=1.5, alpha=1, ax=ax, zorder=3)
            pitch.annotate(jersey_text, xy=(event_x, event_y), c='white', ha='center', va='center', size=9, weight='bold', ax=ax, zorder=4) # Smaller font size
    else:
        print(f"Warning: Missing columns for node plotting in sequence {sequence_id}")

    # Now plot lines (passes, carries, shot)
    for index, event in passes.iterrows():
        event_x, event_y = event.get('x'), event.get('y')
        event_end_x, event_end_y = event.get('end_x'), event.get('end_y')

        if pd.isna(event_x) or pd.isna(event_y): continue # Skip if start coords are NaN

        # # --- Check for and Plot Carry from previous pass ---
        # if last_pass_end_x is not None and last_pass_end_y is not None:
        #     # Calculate distance between last pass end and current event start
        #     dx_carry = event_x - last_pass_end_x
        #     dy_carry = event_y - last_pass_end_y
        #     dist_carry_opta = np.sqrt(dx_carry**2 + dy_carry**2)

        #     if dist_carry_opta >= carry_threshold_opta:
        #         # Plot as a carry line
        #         pitch.lines(last_pass_end_x, last_pass_end_y, event_x, event_y,
        #                     lw=2, linestyle=CARRY_LINESTYLE, color=CARRY_COLOR,
        #                     ax=ax, alpha=0.7, zorder=1)
        # # --- End Carry Plot ---

        if pd.notna(event_end_x) and pd.notna(event_end_y):
            pitch.lines(event_x, event_y, event_end_x, event_end_y,
                        lw=2.5, transparent=True, comet=True, color=team_color,
                        ax=ax, alpha=0.6, zorder=1)
            # Plot end point of pass
            pitch.scatter(event_end_x, event_end_y, s=30, edgecolor=team_color, linewidth=1, facecolor=bg_color, zorder=2, ax=ax)
        # Update last_pass_end_x/y for next iteration's carry check
        last_pass_end_x = event_end_x if pd.notna(event_end_x) else None
        last_pass_end_y = event_end_y if pd.notna(event_end_y) else None

    if is_shot_sequence: # Plot shot
        valid_shot_line = shot_event_df.dropna(subset=['x','y','end_x','end_y'])
        if not valid_shot_line.empty:
            s_event = valid_shot_line.iloc[0]
            pitch.lines(s_event.x, s_event.y, s_event.end_x, s_event.end_y, lw=3, transparent=True, comet=True, color=current_shot_color, ax=ax, alpha=0.8, zorder=1)
            pitch.scatter(s_event.end_x, s_event.end_y, s=80, marker='x', color=current_shot_color, linewidth=2, ax=ax, zorder=2)
            last_pass_end_x = None 
            last_pass_end_y = None

    # --- Title ---
    # Simpler title directly on the axes
    ax.set_title(title_line1 + "\n" + title_line2, color=line_color, fontsize=8, fontweight='bold', pad=4) # Smaller font for two lines

    # --- Remove Ticks ---
    ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
    # Optionally remove spines for cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)

# --- Binned Sequence Flow Plot Function with Role Annotation ---
def plot_binned_sequence_flow(ax, df_bin_transitions, df_shot_origins, # Use the DataFrame name returned by the fixed metrics function
                             team_name, team_color, bins=(7, 6),
                             min_transition_count=1,
                             bg_color=BG_COLOR, line_color=LINE_COLOR,
                             shot_origin_cmap='Reds',
                             annotate_roles=True): # Flag to control role annotation
    """
    Visualizes aggregated pass flow between bins leading to shots,
    a heatmap of shot origin bins, and annotates arrows with the dominant passer role.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        df_bin_transitions (pd.DataFrame): Counts of transitions including dominant role
                                           (from calculate_binned_sequence_stats).
        df_shot_origins (pd.DataFrame): Counts of shots originating from bins.
        team_name (str): Name of the team.
        team_color (str): Primary color hex for the team (used for arrows).
        bins (tuple): Grid size (width_bins, height_bins). Defaults to (7, 6).
        min_transition_count (int): Minimum frequency for a transition arrow to be plotted.
        bg_color (str, optional): Background color.
        line_color (str, optional): Pitch line color.
        shot_origin_cmap (str): Colormap for the shot origin heatmap.
        annotate_roles (bool): Whether to add dominant role text to arrows.
    """
    print(f"Plotting binned sequence flow for {team_name}...")

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta', pitch_color=bg_color, line_color=line_color,
                  linewidth=1, line_zorder=1, corner_arcs=True)
    pitch.draw(ax=ax)

    has_transitions = not df_bin_transitions.empty
    has_origins = not df_shot_origins.empty

    if not has_transitions and not has_origins:
        print(f"Warning: No transition or origin data to plot for {team_name}.")
        ax.set_title(f"{team_name}\nShot Buildup Flow (No Data)", color=line_color, fontsize=16, fontweight='bold')
        return

    # --- Plot Heatmap of Shot Origins ---
    if has_origins:
        bin_rows, bin_cols = bins[1], bins[0]
        heatmap_statistic = np.zeros((bin_rows, bin_cols))
        # Ensure start_bin column exists before iterating
        if 'start_bin' in df_shot_origins.columns:
            for index, row in df_shot_origins.iterrows():
                bin_idx = row['start_bin']; count = row['shot_origin_count']
                if isinstance(bin_idx, tuple) and len(bin_idx) == 2 and pd.notna(bin_idx[0]) and pd.notna(bin_idx[1]):
                     heatmap_statistic[int(bin_idx[1]), int(bin_idx[0])] = count
                else: print(f"Warning: Invalid bin index {bin_idx} in shot origins.")
        else:
            print("Warning: 'start_bin' column missing in df_shot_origins.")

        if np.sum(heatmap_statistic) > 0:
             x_bin_edges = np.linspace(pitch.dim.left, pitch.dim.right, bins[0] + 1)
             y_bin_edges = np.linspace(pitch.dim.bottom, pitch.dim.top, bins[1] + 1)
             heatmap_dict = {'statistic': heatmap_statistic, 'x_grid': x_bin_edges, 'y_grid': y_bin_edges}
             heatmap = pitch.heatmap(heatmap_dict, ax=ax, cmap=shot_origin_cmap, edgecolor=bg_color, alpha=0.6, zorder=0)
        else: print("Info: No valid counts found for shot origin heatmap.")


    # --- Plot Transition Arrows and Annotate Roles ---
    if has_transitions:
        required_cols = ['start_bin', 'end_bin', 'total_transition_count', 'dominant_passer_role', 'dominant_receiver_role']
        if not all(col in df_bin_transitions.columns for col in required_cols):
             print(f"Warning: Missing columns in transition DataFrame. Cannot plot roles.")
             df_plot_transitions = df_bin_transitions[df_bin_transitions['total_transition_count'] >= min_transition_count].copy() # Plot arrows anyway if possible
             annotate_roles = False # Disable annotation
        else:
            df_plot_transitions = df_bin_transitions[df_bin_transitions['total_transition_count'] >= min_transition_count].copy()
            annotate_roles = True

        if not df_plot_transitions.empty:
            print(f"Plotting {len(df_plot_transitions)} transitions (count >= {min_transition_count}) with role pairs...")
            max_count = df_plot_transitions['total_transition_count'].max()
            base_lw = 0.5; max_extra_lw = 2.5 # Slightly thinner max arrow width
            node_base_size = 100; node_max_extra = 600 # Node size scales with count

            pitch_width_coords = pitch.dim.right - pitch.dim.left # Use left/right
            pitch_height_coords = pitch.dim.top - pitch.dim.bottom # Use top/bottom
            bin_width = pitch_width_coords / bins[0]
            bin_height = pitch_height_coords / bins[1]

            # --- Loop through transitions to plot arrow AND node ---
            for index, row in df_plot_transitions.iterrows():
                start_bin = row['start_bin']; end_bin = row['end_bin']
                count = row['total_transition_count']
                passer_role = row['dominant_passer_role'] if annotate_roles else '?'
                receiver_role = row['dominant_receiver_role'] if annotate_roles else '?'

                if not (isinstance(start_bin, tuple) and len(start_bin)==2 and pd.notna(start_bin[0])) or \
                   not (isinstance(end_bin, tuple) and len(end_bin)==2 and pd.notna(end_bin[0])):
                   continue

                # Calculate centers and midpoint
                start_x = pitch.dim.left + (start_bin[0] + 0.5) * bin_width
                start_y = pitch.dim.bottom + (start_bin[1] + 0.5) * bin_height
                end_x = pitch.dim.left + (end_bin[0] + 0.5) * bin_width
                end_y = pitch.dim.bottom + (end_bin[1] + 0.5) * bin_height
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2

                # Calculate style based on TOTAL count
                lw = base_lw + (count / max_count) * max_extra_lw if max_count > 0 else base_lw + max_extra_lw / 2
                alpha = 0.3 + (count / max_count) * 0.6 if max_count > 0 else 0.6 # Arrow slightly more transparent
                node_size = node_base_size + (count / max_count) * node_max_extra if max_count > 0 else node_base_size + node_max_extra / 2

                # Plot arrow
                pitch.arrows(start_x, start_y, end_x, end_y, width=0.7, headwidth=3, headlength=3, color=team_color, lw=lw, alpha=alpha, ax=ax, zorder=3) # Arrow shaft thinner

                # --- Plot Node at Midpoint ---
                if annotate_roles and pd.notna(passer_role) and pd.notna(receiver_role):
                    pitch.scatter(mid_x, mid_y, s=node_size, marker='o',
                                  facecolor=team_color, # Use team color for node
                                  edgecolor='white', linewidth=1.0, alpha=0.9,
                                  ax=ax, zorder=4) # Node above arrow

                    # --- Annotate Node with Role Pair ---
                    role_text = f"{passer_role}\n->\n{receiver_role}" # Multi-line text
                    pitch.annotate(role_text, xy=(mid_x, mid_y), color='white', # White text on node
                                   fontsize=6 if len(passer_role)>3 or len(receiver_role)>3 else 7, # Adjust size based on role name length
                                   va='center', ha='center', fontweight='bold', ax=ax, zorder=5)
            # --- End Loop ---
        else:
            print(f"Info: No transitions meet the minimum count ({min_transition_count}) threshold.")


    # --- Title & Explanatory Text ---
    ax.set_title(f"{team_name} - Shot Buildup Flow ({bins[0]}x{bins[1]} Bins)", color=line_color, fontsize=16, fontweight='bold')
    annotation_explanation = "Node=Dominant Passer->Receiver Role" if annotate_roles else ""
    ax.text(0.01, 0.01, f"Arrows show avg. pass transition (count >= {min_transition_count})\nHeatmap=Shot Origin Freq. | {annotation_explanation}",
            ha='left', va='bottom', fontsize=8, color=line_color, transform=ax.transAxes)
    
# --- Mean Positions Plot Function ---
def plot_mean_positions(ax, df_player_loc_agg, df_all_touches_team, # Pass aggregated and all touches
                        team_name, team_color, sub_list=[], is_away_team=False,
                        bg_color='#FAFAFA', line_color='#222222',
                        annotate_role=True): # Flag to control role annotation
    """
    Visualizes average player positions based on median touch location,
    overlaid on a KDE heatmap of all team touches. Annotates nodes with
    jersey number and optionally positional role. Includes average line height.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        df_player_loc_agg (pd.DataFrame): Aggregated median location, count, jersey, role per player.
        df_all_touches_team (pd.DataFrame): All touch events for this team (for KDE).
        team_name (str): Name of the team.
        team_color (str): Primary color hex for the team.
        sub_list (list, optional): List of player names who were substitutes.
        is_away_team (bool, optional): Flag to invert axes.
        bg_color (str, optional): Background color.
        line_color (str, optional): Pitch line color.
        annotate_role (bool): Whether to add positional role below the node.
    """
    print(f"Plotting mean positions for {team_name}...")

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta', pitch_color=bg_color, line_color=line_color,
                  linewidth=2, line_zorder=2, corner_arcs=True)
    pitch.draw(ax=ax)

    # Invert axes if away team
    if is_away_team:
        ax.invert_xaxis()
        ax.invert_yaxis()

    # Exit if no aggregated player data
    if df_player_loc_agg.empty:
        print(f"Warning: No aggregated player data to plot mean positions for {team_name}.")
        ax.set_title(f"{team_name}\nMean Positions (No Data)", color=line_color, fontsize=18, fontweight='bold')
        return

    # --- Plot KDE Heatmap of All Touches ---
    if not df_all_touches_team.empty and 'x' in df_all_touches_team.columns and 'y' in df_all_touches_team.columns:
        # Use a light version of the team color for the heatmap
        team_cmap_light = LinearSegmentedColormap.from_list(f"{team_name}_light_cmap", [bg_color, team_color], N=100)
        pitch.kdeplot(df_all_touches_team.x, df_all_touches_team.y, ax=ax,
                      fill=True, levels=100, # Adjust levels/thresh
                      thresh=0.02, cut=4, cmap=team_cmap_light, alpha=0.5, zorder=0) # Lower alpha
    else:
        print(f"Warning: No individual touch data to plot heatmap for {team_name}.")


    # --- Plot Player Nodes (Median Position) ---
    node_size = 2000 # Fixed size as per original code
    required_node_cols = ['playerName', 'median_x', 'median_y', 'jersey_number', 'positional_role']
    if not all(col in df_player_loc_agg.columns for col in required_node_cols):
        print(f"Warning: Missing columns in aggregated data for node plotting.")
        # Attempt to plot with available columns
    else:
        for index, row in df_player_loc_agg.iterrows():
            x, y = row['median_x'], row['median_y']
            jersey = row['jersey_number']
            role = row['positional_role']
            player_name = row['playerName']
            marker = 's' if player_name in sub_list else 'o'

            # Plot node only if coordinates are valid
            if pd.notna(x) and pd.notna(y):
                pitch.scatter(x, y, s=node_size, marker=marker,
                              facecolor=team_color, # Use team color fill
                              edgecolor=line_color, linewidth=1.5, alpha=1, zorder=3, ax=ax)

                # Annotate with Jersey Number (inside node)
                try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
                except: jersey_text = str(jersey) if pd.notna(jersey) else ''
                pitch.annotate(jersey_text, xy=(x, y), c='white', ha='center', va='center', size=12, weight='bold', ax=ax, zorder=4) # Smaller size

                # # --- Annotate with Positional Role (below node) ---
                # if annotate_role and pd.notna(role) and role not in ['Sub/Unknown', 'UnknownFormation', 'UnknownPosNum']:
                #     pitch.annotate(role, xy=(x, y - 5), # Offset below the node center
                #                    c=line_color, ha='center', va='top', # Align top of text to xy
                #                    size=8, weight='normal', ax=ax, zorder=4) # Smaller size, normal weight


    # --- Add Average Line Height ---
    # Calculate MEAN X position of the MEDIAN player positions
    avg_line_x_opta = df_player_loc_agg['median_x'].mean()

    if pd.notna(avg_line_x_opta):
        ax.axvline(x=avg_line_x_opta, color='grey', linestyle='--', alpha=0.75, linewidth=2, zorder=1) # Behind nodes
        # Convert to meters for display
        pitch_length_meters = 105.0 # Standard length assumption
        avg_line_meters = avg_line_x_opta * (pitch_length_meters / 100.0)
        # Position text based on team
        text_y_pos = pitch.dim.top + 2.2 if is_away_team else pitch.dim.bottom - 2.2 # Offset from top/bottom edge
        horizontal_alignment = 'right' if is_away_team else 'left'
        ax.text(avg_line_x_opta + 1, # Offset text from line
                text_y_pos, f"Avg Line: {avg_line_meters:.1f}m",
                fontsize=12, color='dimgray', ha=horizontal_alignment, va='center')
    else:
        print(f"Info: Could not calculate average line height for {team_name}.")


    # --- Titles and Annotations ---
    ax.set_title(f"{team_name}\nMean Positions", color=line_color, fontsize=20, fontweight='bold') # Smaller title

    # Add legend/direction text
    legend_text = "○ = Starter | □ = Substitute"
    if is_away_team:
        ax.text(0.02, 0, legend_text, color='black', size=10, ha='left', va='bottom', transform=ax.transAxes)
        ax.text(0.98, 0, "← Attacking Direction", color=team_color, size=12, ha='right', va='bottom', transform=ax.transAxes)
    else:
        ax.text(0.98, 0, legend_text, color='black', size=10, ha='right', va='bottom', transform=ax.transAxes)
        ax.text(0.02, 0, "Attacking Direction →", color=team_color, size=12, ha='left', va='bottom', transform=ax.transAxes)

# --- Pressure Map Plot Function ---
# This function visualizes pressure events for a specific team on a football pitch.
def plot_ppda_actions(ax, df, team_name, team_color,
                      action_ids_to_plot, # REQUIRED list of IDs used for PPDA denom
                      def_action_zone_thresh, # REQUIRED threshold for where actions count
                      event_id_col='typeId',
                      plot_type='kde', # Options: 'kde', 'scatter', 'hexbin', 'both'
                      pitch_color=BG_COLOR, line_color=LINE_COLOR,
                      scatter_size=50, scatter_alpha=0.5,
                      kde_cmap='Reds', kde_levels=100, kde_thresh=0.01, kde_alpha=0.6,
                      is_away_team=False,
                      event_mapping=None):
    """
    Plots the locations of a team's specified defensive actions occurring
    within the zone relevant for PPDA calculation (x >= def_action_zone_thresh).
    Ensures heatmap/scatter only use data from this zone. Corrects away team shading.

    Args:
        ax: Matplotlib axes object.
        df (pd.DataFrame): DataFrame with Opta event data.
        team_name (str): Name of the team whose actions to plot.
        team_color (str): Primary color for scatter points or heatmap base.
        action_ids_to_plot (list): List of typeIds counting as defensive actions for PPDA.
        def_action_zone_thresh (float): Min x-coordinate for defensive actions to be included.
        # ... (other args remain mostly the same: event_id_col, plot_type, styling, event_mapping) ...
        is_away_team (bool): If True, inverts pitch axes.
    """
    # --- Dynamically Generate Title ---
    action_names = []
    if event_mapping:
        try:
             event_mapping_int_keys = {int(k): v for k, v in event_mapping.items()}
             action_names = [event_mapping_int_keys.get(id_, f"ID {id_}") for id_ in action_ids_to_plot]
        except ValueError: action_names = [f"ID {id_}" for id_ in action_ids_to_plot]
    else: action_names = [f"ID {id_}" for id_ in action_ids_to_plot]
    actions_str = ', '.join(sorted(list(set(action_names))))
    # Update title to reflect PPDA context
    title = f"{team_name} - Def. Actions in Pressing Zone (x>={def_action_zone_thresh:.0f})\n({actions_str})"
    print(f"Plotting '{plot_type}' map for {team_name} | Events: {actions_str} in Zone x>={def_action_zone_thresh:.0f}")
    # --- End Title Generation ---

    # --- Input Validation ---
    required_cols = ['team_name', event_id_col, 'x', 'y']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Warning [Pressure Plot]: DataFrame missing required columns: {missing}. Skipping plot.")
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes)
        ax.set_title(f"{team_name} - Pressure Events (Data Missing)", color=line_color)
        return

    # Filter for the team and pressure events
    # Ensure event_id_col is numeric first
    try:
        numeric_event_ids = pd.to_numeric(df[event_id_col], errors='coerce')
    except Exception as e:
         print(f"Error converting event ID column '{event_id_col}' to numeric: {e}. Skipping plot.")
         ax.set_title(title + " (Error)", color='red')
         return

    pressure_df = df[
        (df['team_name'] == team_name) &
        (numeric_event_ids.isin(action_ids_to_plot)) &
        (df['x'].notna()) &
        (df['y'].notna()) &
        (df['x'].fillna(def_action_zone_thresh - 1) >= float(def_action_zone_thresh))
    ].copy()

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta', pitch_color=pitch_color, line_color=line_color,
                  line_zorder=2)
    pitch.draw(ax=ax)

    # Invert axes if away team BEFORE plotting data
    if is_away_team:
        ax.invert_xaxis()
        ax.invert_yaxis()

    if pressure_df.empty:
        print(f"No pressure events found for {team_name} with IDs {action_ids_to_plot}.")
        ax.set_title(f"{team_name} - Pressure Events (None Found)", color=line_color, fontsize=14)
        return
    
    # --- Optional: Shade the Pressing Zone ---
    # Shade the area where these actions are counted
    zone_start_x = float(def_action_zone_thresh)
    zone_end_x = pitch.dim.right # Assumes threshold is on Opta 0-100 scale
    shade_start = zone_start_x
    shade_end = pitch.dim.right
    ax.axvspan(shade_start, shade_end, color=team_color, alpha=0.08, zorder=0) # Light shading 
    # --- End Shading ---

    # --- Plotting ---
    # Create team-specific colormap fading to background for KDE/Hexbin
    team_kde_cmap = LinearSegmentedColormap.from_list(f"{team_name}_pressure_cmap", [pitch_color, team_color], N=100)

    if plot_type in ['kde', 'both']:
        try:
            pitch.kdeplot(pressure_df['x'], pressure_df['y'], ax=ax,
                          cmap=team_kde_cmap, levels=kde_levels, # Use team cmap
                          fill=True, thresh=kde_thresh, cut=4,
                          alpha=kde_alpha, zorder=1)
        except Exception as e:
            print(f"Could not generate KDE plot for {team_name}: {e}")

    if plot_type in ['scatter', 'both']:
         pitch.scatter(pressure_df['x'], pressure_df['y'], ax=ax,
                       s=scatter_size, color=team_color, # Use team color for points
                       edgecolors='black', linewidth=0.5, alpha=scatter_alpha,
                       zorder=3)

    if plot_type == 'hexbin':
         try:
             pitch.hexbin(pressure_df['x'], pressure_df['y'], ax=ax,
                          cmap=team_kde_cmap, gridsize=(12, 8), # Example gridsize
                          edgecolors=pitch_color, alpha=0.8, zorder=1) # Use pitch color for edges
         except Exception as e:
            print(f"Could not generate Hexbin plot for {team_name}: {e}")

    # Add Title
    ax.set_title(title, color=line_color, fontsize=14) # Use generated title

# --- Buildup Sequence Plot Function ---
def plot_buildup_sequence(ax, sequence_data, team_name, team_color, opponent_color, sequence_id,
                          target_zone_x_min=None, target_zone_y_bounds=None,
                         target_zone_color='lightcoral',
                         bg_color=BG_COLOR, line_color=LINE_COLOR, unsuccessful_color=VIOLET):
    """
    Plots a single buildup sequence starting from deep.
    Highlights successful and unsuccessful passes.
    Marks player nodes at start/end of the sequence.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        sequence_data (pd.DataFrame): DataFrame for ONE buildup sequence.
        team_name (str): Name of the team.
        team_color (str): Primary color for successful passes.
        opponent_color (str): Color for the node if the last pass is unsuccessful.
        sequence_id (int): ID of the sequence.
        bg_color (str, optional): Background color.
        line_color (str, optional): Pitch line color.
        unsuccessful_color (str, optional): Color for unsuccessful pass lines.
        target_zone_x_min (float, optional): Minimum x-coordinate of the target dangerous zone.
        target_zone_y_bounds (tuple, optional): (y_min, y_max) of the target dangerous zone.
        target_zone_color (str, optional): Color for shading the target zone.
    """
    # print(f"Plotting buildup sequence {sequence_id} for {team_name}...") # Optional

     # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta',
                  pitch_color=bg_color, line_color=line_color,
                  linewidth=1.5, corner_arcs=True,
                  pad_top=2, pad_bottom=2, pad_left=2, pad_right=2) # Adjust padding values (default is usually larger)
    pitch.draw(ax=ax)

     # --- *** Try Setting Limits AFTER Drawing *** ---
    ax.set_xlim(pitch.dim.left - 1, pitch.dim.right + 1) # Slightly beyond pitch edge
    ax.set_ylim(pitch.dim.bottom - 1, pitch.dim.top + 1) # Slightly beyond pitch edge
    # --- *** End Setting Limits *** ---

    if sequence_data is None or sequence_data.empty:
        print(f"Warning: No data for buildup sequence {sequence_id}")
        ax.set_title(f"Buildup Seq {sequence_id}\n(No Data)", color=line_color, fontsize=10)
        ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
        return

    # --- Ensure Coordinates are Numeric ---
    coord_cols = ['x', 'y', 'end_x', 'end_y']
    numeric_sequence_data = sequence_data.copy()
    for col in coord_cols:
        if col in numeric_sequence_data.columns:
            numeric_sequence_data[col] = pd.to_numeric(numeric_sequence_data[col], errors='coerce')
        else: numeric_sequence_data[col] = np.nan
        numeric_sequence_data[col] = numeric_sequence_data[col].astype(float)

    # --- *** Shade Target Dangerous Zone (if defined) *** ---
    if target_zone_x_min is not None and target_zone_y_bounds is not None:
        y_min, y_max = target_zone_y_bounds
        # ax.fill_betweenx defines a horizontal band
        # ax.fill_between defines a vertical band
        # We want a rectangle, so use ax.fill or pitch.polygon
        # Using ax.fill for simplicity here:
        zone_vertices = [
            (target_zone_x_min, y_min), # Bottom-left of zone
            (pitch.dim.right, y_min),   # Bottom-right of zone (to pitch edge)
            (pitch.dim.right, y_max),   # Top-right of zone
            (target_zone_x_min, y_max)    # Top-left of zone
        ]
        # Use pitch.polygon for consistency with mplsoccer's coordinate handling
        # Ensure vertices are in a list for pitch.polygon
        pitch.polygon([zone_vertices], ax=ax, color=target_zone_color, alpha=0.2, zorder=0)
        # Alternatively, using axvspan for a vertical band and clipping:
        # ax.axvspan(target_zone_x_min, pitch.dim.right, ymin=y_min/pitch.dim.height, ymax=y_max/pitch.dim.height,
        #            color=target_zone_color, alpha=0.2, zorder=0)
    # --- *** End Shading Zone *** ---

    # --- Separate Successful and Unsuccessful Passes ---
    df_succ = numeric_sequence_data[numeric_sequence_data['outcome'] == 'Successful'].dropna(subset=coord_cols)
    df_unsucc = numeric_sequence_data[numeric_sequence_data['outcome'] == 'Unsuccessful'].dropna(subset=coord_cols)

    # --- Plot Pass Lines ---
    if not df_succ.empty:
        pitch.lines(df_succ.x, df_succ.y, df_succ.end_x, df_succ.end_y,
                    lw=3, transparent=True, comet=True, color=team_color, ax=ax, alpha=0.6, zorder=1)
        pitch.scatter(df_succ.end_x, df_succ.end_y, s=30, edgecolor=team_color, linewidth=1, facecolor=bg_color, zorder=2, ax=ax)

    if not df_unsucc.empty:
        pitch.lines(df_unsucc.x, df_unsucc.y, df_unsucc.end_x, df_unsucc.end_y,
                    lw=3, transparent=True, comet=True, color=unsuccessful_color, ax=ax, alpha=0.7, zorder=1)
        pitch.scatter(df_unsucc.end_x, df_unsucc.end_y, s=30, edgecolor=unsuccessful_color, linewidth=1, facecolor=bg_color, zorder=2, ax=ax)

    # --- Plot Player Nodes at Start of Each Pass and End of Last Pass ---
    node_size = 350
    if not numeric_sequence_data.empty:
        valid_events = numeric_sequence_data.dropna(subset=['x', 'y'])
        for i, event in valid_events.iterrows():
            jersey = event.get('Mapped Jersey Number', '')
            try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
            except: jersey_text = str(jersey) if pd.notna(jersey) else ''
            event_x, event_y = event['x'], event['y']

            # Plot node at the start of each pass in the sequence
            pitch.scatter(event_x, event_y, s=node_size, marker='o',
                          facecolor=team_color, edgecolor=line_color, linewidth=1.5, alpha=1, ax=ax, zorder=3)
            pitch.annotate(jersey_text, xy=(event_x, event_y), c='white', ha='center', va='center', size=10, weight='bold', ax=ax, zorder=4)

        # Plot node at the end of the VERY LAST pass in the sequence
        last_event = numeric_sequence_data.iloc[-1]
        last_event_end_x = last_event.get('end_x')
        last_event_end_y = last_event.get('end_y')
        # Use receiver jersey if available for the last event's end node
        receiver_jersey = last_event.get('receiver_jersey_number', '') # From get_passes_df
        try: receiver_jersey_text = str(int(receiver_jersey)) if pd.notna(receiver_jersey) and receiver_jersey != '' else ''
        except: receiver_jersey_text = str(receiver_jersey) if pd.notna(receiver_jersey) else ''

        if pd.notna(last_event_end_x) and pd.notna(last_event_end_y):
            # Color end node based on outcome of the last pass
            end_node_color = opponent_color if last_event.get('outcome') == 'Unsuccessful' else team_color
            pitch.scatter(last_event_end_x, last_event_end_y, s=node_size, marker='o',
                          facecolor=end_node_color, edgecolor=line_color, linewidth=1.5, alpha=1, ax=ax, zorder=3)
            if receiver_jersey_text: # Only annotate if receiver jersey is known
                pitch.annotate(receiver_jersey_text, xy=(last_event_end_x, last_event_end_y),
                               c='white', ha='center', va='center', size=10, weight='bold', ax=ax, zorder=4)


    # --- Title and Direction ---
    ax.set_title(f"Buildup Seq. {sequence_id}", color=line_color, fontsize=12, fontweight='bold', pad=5)
    # Attacking direction will depend on is_away_team flag in main script
    ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
    for spine in ax.spines.values(): spine.set_visible(False)

# --- Formation and Substitutions Plot Function ---
def plot_formations_and_subs(ax, # Pass the figure object
                             df_processed_full_match, # Full event data for getting subs
                             df_starters_team, # DataFrame of just the 11 starters for this team
                             team_name, team_color, team_starting_formation_id, # Starting formation
                             formation_changes=None, # List of strings for formation changes
                             is_away_team=False,
                             bg_color=BG_COLOR, line_color=LINE_COLOR):
    """
    Plots a team's starting formation with player numbers and roles,
    and lists substitutions & formation changes below the pitch.
    Uses a dedicated subplot layout within the provided figure.

    Args:
        ax: The figure to plot on.
        df_processed_full_match (pd.DataFrame): Full event DataFrame to find substitutions
                                                and map player IDs to names.
        df_starters_team (pd.DataFrame): DataFrame containing 'playerName',
                                         'Mapped Jersey Number', 'Mapped Position Number',
                                         'positional_role' for the 11 starters of ONE TEAM.
        team_name (str): Name of the team.
        team_color (str): Color for the team.
        team_starting_formation_id (int): Opta ID of the team's starting formation.
        formation_changes (list, optional): List of strings describing formation changes
                                            (e.g., ["40' -> Form. 442"]). Defaults to [].
        is_away_team (bool): Flag to invert X-axis for away team.
        bg_color (str): Background color.
        line_color (str): Pitch line color.
    """
    print(f"Plotting formation for {team_name} (Initial Formation ID: {team_starting_formation_id})...")
    if formation_changes is None: formation_changes = []

    # --- *** Get Formation Name *** ---
    formation_name_str = formation_layouts.get_formation_name(team_starting_formation_id)
    # --- *** End Get Formation Name *** ---


    # --- Setup Pitch on the provided Axes ---
    pitch = Pitch(pitch_type='opta', pitch_color=bg_color, line_color=line_color,
                  linewidth=1.5, line_zorder=1, corner_arcs=True,
                  # Reduce padding to maximize pitch size within the axes
                  pad_left=2, pad_right=2, pad_top=5, pad_bottom=12) # Increased bottom pad for text
    pitch.draw(ax=ax)
    ax.set_title(f"{team_name} Starting Formation ({formation_name_str})",
                 color=line_color, fontsize=12, fontweight='bold', pad=8) # Reduced font, pad
    
    if is_away_team: 
        ax.invert_xaxis()
        ax.set_ylim(pitch.dim.bottom - 1, pitch.dim.top + 1) # Adjust y-limits for away team
        ax.invert_yaxis()
    else: ax.set_ylim(pitch.dim.bottom - 1, pitch.dim.top + 1) # Adjust y-limits for home team



    # --- Plot Starters on Pitch ---
    # (Starter plotting logic - same as before, using 'ax' passed in)
    node_size = 900 # Slightly smaller nodes
    if team_starting_formation_id is not None and not df_starters_team.empty:
        # ... (loop through df_starters_team to plot nodes with jersey and role on 'ax') ...
        df_starters_team['Mapped Position Number'] = pd.to_numeric(df_starters_team['Mapped Position Number'], errors='coerce').fillna(0).astype(int)
        for idx, player_row in df_starters_team.iterrows():
            pos_num = player_row.get('Mapped Position Number')
            plot_x, plot_y = formation_layouts.get_formation_layout_coords(team_starting_formation_id, pos_num)
            if plot_x is not None and plot_y is not None:
                pitch.scatter(plot_x, plot_y, s=node_size, marker='o', facecolor=team_color, edgecolor=line_color, linewidth=1.5, ax=ax, zorder=2)
                jersey = player_row.get('Mapped Jersey Number', ''); role = player_row.get('positional_role', '')
                try: jersey_text = str(int(jersey)) if pd.notna(jersey) and jersey != '' else ''
                except: jersey_text = str(jersey) if pd.notna(jersey) else ''
                pitch.annotate(jersey_text, xy=(plot_x, plot_y), color='white', ha='center', va='center', fontsize=8, weight='bold', ax=ax, zorder=3)
                if role and role not in ['Sub/Unknown', 'UnknownFormation', 'UnknownPosNum']:
                    if is_away_team:
                        pitch.annotate(role, xy=(plot_x, plot_y + 7), color=line_color, ha='center', va='bottom', fontsize=6, weight='bold', ax=ax, zorder=3, path_effects=[path_effects.withStroke(linewidth=1.0, foreground=bg_color)])
                    else:
                        pitch.annotate(role, xy=(plot_x, plot_y - 5), color=line_color, ha='center', va='top', fontsize=6, weight='bold', ax=ax, zorder=3, path_effects=[path_effects.withStroke(linewidth=1.0, foreground=bg_color)])
    elif team_starting_formation_id is None: print(f"Warning: No formation ID for {team_name}.")
    elif df_starters_team.empty: print(f"Warning: No starter data for {team_name}.")


    # --- Prepare Text Area Content ---
    text_area_content = []
    # ... (Starting XI listing - same as before) ...
    text_area_content.append(f"Starting XI ({formation_name_str if formation_name_str else team_starting_formation_id}):")
    if not df_starters_team.empty:
        df_starters_team_sorted = df_starters_team.sort_values(by='Mapped Position Number')
        for idx, p_row in df_starters_team_sorted.iterrows():
            jersey_s = p_row.get('Mapped Jersey Number', ''); name_s = p_row.get('playerName', 'N/A'); role_s = p_row.get('positional_role', 'N/A')
            text_area_content.append(f"  {str(jersey_s):>2}. {name_s} ({role_s})")
    else: text_area_content.append("  (Starter data not available)")


    text_area_content.append("\nSubstitutions:")

    # --- *** REVISED SUBSTITUTION LOGIC (Pairing Player Off/On by Time) *** ---
    # Ensure typeId, timeMin, timeSec are numeric for correct operations
    df_processed_full_match['typeId'] = pd.to_numeric(df_processed_full_match['typeId'], errors='coerce')
    df_processed_full_match['timeMin'] = pd.to_numeric(df_processed_full_match['timeMin'], errors='coerce')
    df_processed_full_match['timeSec'] = pd.to_numeric(df_processed_full_match['timeSec'], errors='coerce')


    # Filter for Player Off (typeId 18) and Player On (typeId 19) events FOR THIS TEAM
    player_off_events = df_processed_full_match[
        (df_processed_full_match['typeId'] == 18) &
        (df_processed_full_match['team_name'] == team_name)
    ].sort_values(['timeMin', 'timeSec', 'eventId'])

    player_on_events = df_processed_full_match[
        (df_processed_full_match['typeId'] == 19) &
        (df_processed_full_match['team_name'] == team_name)
    ].sort_values(['timeMin', 'timeSec', 'eventId'])

    subs_paired = []
    used_on_indices = set() # To ensure a "player on" is used only once

    if not player_off_events.empty and not player_on_events.empty:
        print(f"  Attempting to pair {len(player_off_events)} Player Off events with {len(player_on_events)} Player On events for {team_name}...")
        for off_idx, off_event in player_off_events.iterrows():
            time_min_off = off_event.get('timeMin')
            time_sec_off = off_event.get('timeSec')
            player_off_name = off_event.get('playerName', 'Unknown Off')
            player_off_jersey = off_event.get('Mapped Jersey Number', '')

            # Find matching "Player On" event(s) at the same time
            matching_on_events = player_on_events[
                (player_on_events['timeMin'] == time_min_off) &
                (player_on_events['timeSec'] == time_sec_off) &
                (~player_on_events.index.isin(used_on_indices)) # Not already used
            ]

            if not matching_on_events.empty:
                # Take the first available matching "Player On" event
                on_event = matching_on_events.iloc[0]
                player_on_name = on_event.get('playerName', 'Unknown On')
                player_on_jersey = on_event.get('Mapped Jersey Number', '')
                used_on_indices.add(on_event.name) # Mark this "on" event as used
                time_str = f"{int(time_min_off)}'" if pd.notna(time_min_off) else "?'"
                subs_paired.append(f"  {time_str:>3} : {player_off_jersey}. {player_off_name} OFF ↩ | {player_on_jersey}. {player_on_name} ON ↪")
            else:
                # Player Off event without a perfectly matching Player On event at same time
                time_str = f"{int(time_min_off)}'" if pd.notna(time_min_off) else "?'"
                subs_paired.append(f"  {time_str:>3} : {player_off_name} OFF ↩ | (No matching Player On found at same time)")

        # Add any remaining "Player On" events that weren't paired (e.g., if more "on" than "off" at a given time)
        remaining_on_events = player_on_events[~player_on_events.index.isin(used_on_indices)]
        for on_idx, on_event in remaining_on_events.iterrows():
            time_min_on = on_event.get('timeMin')
            player_on_name = on_event.get('playerName', 'Unknown On')
            time_str = f"{int(time_min_on)}'" if pd.notna(time_min_on) else "?'"
            subs_paired.append(f"  {time_str:>3} (No matching Player Off) | {player_on_name} ON ↪")

    elif not player_off_events.empty: # Only Player Off events found
        for off_idx, off_event in player_off_events.iterrows():
             time_min_off = off_event.get('timeMin')
             player_off_name = off_event.get('playerName', 'Unknown Off')
             time_str = f"{int(time_min_off)}'" if pd.notna(time_min_off) else "?'"
             subs_paired.append(f"  {time_str:>3} {player_off_name} OFF ↩")
    elif not player_on_events.empty: # Only Player On events found
        for on_idx, on_event in player_on_events.iterrows():
            time_min_on = on_event.get('timeMin')
            player_on_name = on_event.get('playerName', 'Unknown On')
            time_str = f"{int(time_min_on)}'" if pd.notna(time_min_on) else "?'"
            subs_paired.append(f"  {time_str:>3} {player_on_name} ON ↪")


    if subs_paired:
        text_area_content.extend(subs_paired)
    else:
        text_area_content.append("  None")
    # --- *** END REVISED SUBSTITUTION LOGIC *** ---


    # --- Display Formation Changes ---
    # ... (Formation changes listing - same as before) ...
    if formation_changes:
        text_area_content.append("\nFormation Changes:")
        for change_desc in formation_changes:
            text_area_content.append(f"  {change_desc}")

    # --- Display Text in the Text Area ---
    # ... (ax_text_area.text() call - same as before) ...
    text_start_y_axes_coords = -0.02 # Start text just below the axes frame for the pitch (adjust)
    ax.text(0.02, text_start_y_axes_coords, "\n".join(text_area_content),
            ha='left', va='top', fontsize=12, color=line_color, # Smaller font for text block
            wrap=True, transform=ax.transAxes, family='monospace') # transform=ax.transAxes is key
    
# --- Recovery to First Pass Plot Function ---
def plot_recovery_first_pass(ax, df_recovery_sequences, team_name, team_color,
                             zone_name, # e.g., "Defensive Third"
                             bg_color=BG_COLOR, line_color=LINE_COLOR):
    """
    Plots the first pass made after recoveries in a specific zone, styled
    with player nodes (circles) and comet pass lines colored by outcome.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        df_recovery_sequences (pd.DataFrame): DataFrame from find_recovery_to_first_pass,
                                             already filtered for the specific zone.
                                             MUST include 'first_pass_outcome' for coloring.
        team_name (str): Name of the team.
        team_color (str): Primary color for the team (used for successful passes/nodes).
        zone_name (str): Name of the zone being plotted (for title).
        # ... (other styling args) ...
    """
    print(f"Plotting recovery-first-pass for {team_name} in {zone_name} ({len(df_recovery_sequences)} sequences)...")

    # --- Setup Pitch ---
    pitch = Pitch(pitch_type='opta', pitch_color=bg_color, line_color=line_color,
                  linewidth=1.5, line_zorder=1, corner_arcs=True,
                  pad_left=2, pad_right=2, pad_top=2, pad_bottom=2)
    pitch.draw(ax=ax)

    # --- *** ADD VERTICAL LINES FOR PITCH THIRDS *** ---
    # X-coordinates for the lines dividing the thirds (on Opta 0-100 scale)
    x_third_1 = 100/3  # Approx 33.33
    x_third_2 = 2 * (100/3) # Approx 66.67
    line_style_thirds = {'color': 'grey', 'linestyle': '--', 'linewidth': 2, 'alpha': 0.7, 'zorder': 0.5}

    ax.axvline(x=x_third_1, **line_style_thirds)
    ax.axvline(x=x_third_2, **line_style_thirds)
    # --- *** END THIRDS LINES *** ---

    if df_recovery_sequences.empty:
        ax.set_title(f"{team_name}\nFirst Pass after Recovery in {zone_name}\n(No Data)",
                     color=line_color, fontsize=10)
        ax.xaxis.set_visible(False); ax.yaxis.set_visible(False); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['bottom'].set_visible(False); ax.spines['left'].set_visible(False)
        return

    # --- Ensure Coordinate and Outcome Columns are Numeric/Valid ---
    coord_cols = ['recovery_x', 'recovery_y', 'first_pass_x', 'first_pass_y', 'first_pass_end_x', 'first_pass_end_y']
    df_plot_data = df_recovery_sequences.copy()
    for col in coord_cols:
        df_plot_data[col] = pd.to_numeric(df_plot_data[col], errors='coerce')
    # Ensure 'first_pass_outcome' column exists (should be 'Successful' or 'Unsuccessful')
    if 'first_pass_outcome' not in df_plot_data.columns:
        print(f"Warning: 'first_pass_outcome' column missing. Cannot color passes by outcome.")
        df_plot_data['first_pass_outcome'] = 'Successful' # Default to successful for coloring

    # Filter out rows where essential coordinates for plotting are NaN
    df_plot_data.dropna(subset=['recovery_x', 'recovery_y', 'first_pass_x', 'first_pass_y', 'first_pass_end_x', 'first_pass_end_y'], inplace=True)


    # --- Plot Player Nodes and Pass Lines ---
    node_size = 400 # Adjust as needed
    pass_lw = 2.5
    pass_alpha = 0.6

    for idx, row in df_plot_data.iterrows():
        # Recovery point (same as start of first pass)
        rec_x, rec_y = row['recovery_x'], row['recovery_y']
        passer_jersey = row.get('recovery_jersey', '') # Jersey of player who regained/made first pass

        # First pass end point
        pass_end_x, pass_end_y = row['first_pass_end_x'], row['first_pass_end_y']
        receiver_jersey = row.get('first_pass_jersey', '') # Jersey of receiver of first pass

        # Determine pass color based on outcome
        pass_color = team_color if row['first_pass_outcome'] == 'Successful' else "black"

        # Plot Pass Line (Comet)
        pitch.lines(rec_x, rec_y, pass_end_x, pass_end_y,
                    lw=pass_lw, transparent=True, comet=True, color=pass_color,
                    ax=ax, alpha=pass_alpha, zorder=2)

        # Plot Node at Recovery/Pass Start
        pitch.scatter(rec_x, rec_y, s=node_size, marker='o',
                      facecolor=team_color, edgecolor=line_color, # Always team color for player
                      linewidth=1.0, alpha=1, ax=ax, zorder=3)
        try: passer_jersey_text = str(int(passer_jersey)) if pd.notna(passer_jersey) and passer_jersey != '' else ''
        except: passer_jersey_text = str(passer_jersey) if pd.notna(passer_jersey) else ''
        pitch.annotate(passer_jersey_text, xy=(rec_x, rec_y), c='white', ha='center', va='center', size=8, weight='bold', ax=ax, zorder=4)

        # Plot Node at Pass End (Receiver)
        pitch.scatter(pass_end_x, pass_end_y, s=0.1*node_size, marker='o',
                      facecolor="white", edgecolor="black", # Always team color for player
                      linewidth=1.0, alpha=1, ax=ax, zorder=3)
        # try: receiver_jersey_text = str(int(receiver_jersey)) if pd.notna(receiver_jersey) and receiver_jersey != '' else ''
        # except: receiver_jersey_text = str(receiver_jersey) if pd.notna(receiver_jersey) else ''
        # pitch.annotate(receiver_jersey_text, xy=(pass_end_x, pass_end_y), c='white', ha='center', va='center', size=8, weight='bold', ax=ax, zorder=4)


    # --- Title ---
    if len(df_recovery_sequences) == 0:
        ax.set_title(f"{team_name} - First Pass after Recovery\nZone: {zone_name} (No Data)",
                     color=line_color, fontsize=10, fontweight='bold', pad=5)
    else:
        ax.set_title(f"{team_name} - First Pass after Recovery\nZone: {zone_name} ({len(df_recovery_sequences)} Rec.)",
                 color=line_color, fontsize=10, fontweight='bold', pad=5)

    # Remove ticks for cleaner look
    ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
    for spine in ax.spines.values(): spine.set_visible(False)


# --- Helper function for plot text consistency (optional, if sequence_outcome_type isn't enough) ---
def _is_point_in_plot_big_chance_area(point_x, point_y, is_attacking_right_to_left):
    """
    Checks if a point is in the big chance area, considering attack direction.
    Uses the global-like BC_... constants.
    """
    if is_attacking_right_to_left: # Attacking R->L (goal at x=0 in data)
        # Semicircle flat diameter at data x_val = 100 - BC_SEMICIRCLE_DIAMETER_X_STD (e.g., 17)
        # Bulges towards x=0. Point must be to the left of or on the diameter.
        ref_x = 100.0 - BC_SEMICIRCLE_DIAMETER_X_STD
        if point_x > ref_x: # point_x is to the right of diameter, not in bulge
            return False
        # Check distance from center of diameter (ref_x, BC_SEMICIRCLE_CENTER_Y_STD)
        dist_sq = (point_x - ref_x)**2 + (point_y - BC_SEMICIRCLE_CENTER_Y_STD)**2
        return dist_sq <= BC_SEMICIRCLE_RADIUS_SQUARED_STD
    else: # Attacking L->R (goal at x=100 in data)
        # Semicircle flat diameter at data x_val = BC_SEMICIRCLE_DIAMETER_X_STD (e.g., 83)
        # Bulges towards x=100. Point must be to the right of or on the diameter.
        ref_x = BC_SEMICIRCLE_DIAMETER_X_STD
        if point_x < ref_x: # point_x is to the left of diameter, not in bulge
            return False
        # Check distance from center of diameter (ref_x, BC_SEMICIRCLE_CENTER_Y_STD)
        dist_sq = (point_x - ref_x)**2 + (point_y - BC_SEMICIRCLE_CENTER_Y_STD)**2
        return dist_sq <= BC_SEMICIRCLE_RADIUS_SQUARED_STD

# --- Opponent Buildup After Loss Plot Function ---
def plot_opponent_buildup_after_loss(ax, sequence_data,
                                     team_that_lost_possession,
                                     team_building_up,
                                     color_for_buildup_team, 
                                     loss_sequence_id,
                                     loss_zone,
                                     is_buildup_team_away,
                                     bg_color=BG_COLOR, line_color=LINE_COLOR,
                                     unsuccessful_pass_color=UNSUCCESSFUL_PASS_COLOR,
                                     metric_to_analyze='defensive_actions'):
    """
    Plots a single sequence of opponent passes after your team lost possession
    in a specific zone.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        sequence_data (pd.DataFrame): DataFrame for ONE opponent buildup sequence.
        your_team_name (str): Name of the team that lost possession.
        opponent_team_name (str): Name of the team building up.
        opponent_color (str): Color for the opponent's passes and nodes.
        loss_sequence_id (int): ID of this sequence.
        loss_zone (str): The zone where 'your_team_name' lost possession.
        is_buildup_team_away (bool): Flag to invert X-axis for away team.
        bg_color (str, optional): Background color.
        line_color (str, optional): Pitch line color.
        unsuccessful_pass_color (str, optional): Color for unsuccessful pass lines.
        metric_to_analyze (str): Metric to analyze for the buildup sequence.
    """
    print(f"Plotting {team_building_up} buildup (seq {loss_sequence_id}) after {team_that_lost_possession} loss in {loss_zone}...")

    pitch = Pitch(pitch_type='opta', corner_arcs=True, pitch_color=bg_color, line_color=line_color, linewidth=1.5,
                  pad_left=2, pad_right=2, pad_top=2, pad_bottom=2)
    pitch.draw(ax=ax)

    # --- *** ADD VERTICAL LINES FOR PITCH THIRDS *** ---
    # X-coordinates for the lines dividing the thirds (on Opta 0-100 scale)
    x_third_1 = 100/3  # Approx 33.33
    x_third_2 = 2 * (100/3) # Approx 66.67
    y_third_1 = 100/3  # Approx 33.33
    y_third_2 = 2 * (100/3) # Approx 66.67
    # Define a consistent style for these lines
    thirds_line_style = {'color': 'grey', 'linestyle': '--', 'linewidth': 0.8, 'alpha': 0.6, 'zorder': 0.5}

    ax.axvline(x=x_third_1, **thirds_line_style)
    ax.axvline(x=x_third_2, **thirds_line_style)
    ax.axhline(y=y_third_1, **thirds_line_style)
    ax.axhline(y=y_third_2, **thirds_line_style)
    # --- *** END THIRDS LINES *** ---

    # --- *** ADD BIG CHANCE SEMICIRCLE VISUALIZATION *** ---
    # wedge_center_x_data = 100
    # # Y-center for wedge is always BC_SEMICIRCLE_CENTER_Y_STD (e.g. 50.0) as Opta y-coords are 0-100 bottom-up
    # # and ax.invert_yaxis() handles the visual flip correctly for this midpoint.
    # wedge_center_y_data = BC_SEMICIRCLE_CENTER_Y_STD
    # wedge_radius = BC_SEMICIRCLE_RADIUS_STD
    # wedge_theta1, wedge_theta2 = 0.0, 0.0

    # if is_buildup_team_away: # Team building up attacks R->L (goal at data x=0)
    #     # Semicircle's flat diameter is at data x = 100.0 - BC_SEMICIRCLE_DIAMETER_X_STD (e.g., 17.0)
    #     # The Wedge is centered on this diameter line. It bulges towards smaller x values (left).
    #     wedge_center_x_data = 100.0
    #     wedge_theta1 = 90  # Start angle for left-bulging semicircle
    #     wedge_theta2 = 270 # End angle for left-bulging semicircle
    # else: # Team building up attacks L->R (goal at data x=100)
    #     # Semicircle's flat diameter is at data x = BC_SEMICIRCLE_DIAMETER_X_STD (e.g., 83.0)
    #      # The Wedge is centered on this diameter line. It bulges towards larger x values (right).
    #     wedge_center_x_data = 100
    #     wedge_theta1 = 90 # Start angle for right-bulging semicircle (or 270)
    #     wedge_theta2 = 270  # End angle for right-bulging semicircle

    # big_chance_wedge = patches.Wedge(center=(wedge_center_x_data, wedge_center_y_data),
    #                                  r=wedge_radius,
    #                                  theta1=wedge_theta1,
    #                                  theta2=wedge_theta2,
    #                                  facecolor='lightcoral', # Choose your preferred color
    #                                  alpha=0.25,            # Transparency
    #                                  edgecolor='none',      # Or a subtle edge like 'darkred'
    #                                  zorder=0.8)            # Draw below events, above thirds lines
    # ax.add_patch(big_chance_wedge)
    # --- *** END BIG CHANCE SEMICIRCLE *** ---

    if is_buildup_team_away:
        ax.invert_xaxis()
        ax.invert_yaxis()
    
    # --- Extract Time of Loss from the FIRST event in sequence_data for the title ---
    time_str = "Time N/A"
    if sequence_data is not None and not sequence_data.empty:
        first_event_in_seq = sequence_data.iloc[0]
        if metric_to_analyze == 'buildup_phases':
            time_min_loss = first_event_in_seq.get('timeMin_at_trigger')
            time_sec_loss = first_event_in_seq.get('timeSec_at_trigger')
            type_of_loss_str = first_event_in_seq.get('type_of_initial_trigger', 'Unknown Loss')
        else:
            time_min_loss = first_event_in_seq.get('timeMin_at_loss') # Get the stored loss time
            time_sec_loss = first_event_in_seq.get('timeSec_at_loss')
            type_of_loss_str = first_event_in_seq.get('type_of_initial_loss', 'Unknown Loss')
        if pd.notna(time_min_loss) and pd.notna(time_sec_loss):
            try: time_str = f"{int(time_min_loss)}'{int(time_sec_loss):02d}\""
            except (ValueError, TypeError): print(f"Warning: Non-integer time for loss in seq {loss_sequence_id}")
    # --- End Extract Time ---

    if sequence_data is None or sequence_data.empty:
        ax.set_title(f"Opp. Seq {loss_sequence_id} after Loss in {loss_zone}\n(No Opp. Passes)", color=line_color, fontsize=9)
        ax.xaxis.set_visible(False); ax.yaxis.set_visible(False); 
        return

    # Ensure coordinates are numeric
    coord_cols = ['x', 'y', 'end_x', 'end_y']; numeric_sequence_data = sequence_data.copy()
    for col in coord_cols: # ... (convert to numeric) ...
        if col in numeric_sequence_data.columns: numeric_sequence_data[col] = pd.to_numeric(numeric_sequence_data[col], errors='coerce')
        else: numeric_sequence_data[col] = np.nan; numeric_sequence_data[col] = numeric_sequence_data[col].astype(float)
    # numeric_sequence_data.sort_values('eventId', inplace=True) # IMPORTANT for carries
    numeric_sequence_data.dropna(subset=['x', 'y'], inplace=True) # Drop if start coords are NaN

    # --- Plot Pass Lines and Nodes Iteratively ---
    node_size = 400; pass_lw = 2.5; pass_alpha = 0.6
    last_event_end_x = None; last_event_end_y = None

    # Plot all start nodes first
    for index, event_row in numeric_sequence_data.iterrows():
        start_x, start_y = event_row['x'], event_row['y']
        passer_jersey = event_row.get('Mapped Jersey Number', '')
        try: jersey_text = str(int(passer_jersey)) if pd.notna(passer_jersey) and passer_jersey != '' else ''
        except: jersey_text = str(passer_jersey) if pd.notna(passer_jersey) else ''
        # Only plot node if it's a Pass or a Shot by the building team
        if event_row['type_name'] == 'Pass' or event_row['type_name'] in SHOT_TYPES:
            pitch.scatter(start_x, start_y, s=node_size, marker='o', facecolor=color_for_buildup_team, edgecolor=line_color, linewidth=1.0, alpha=1, ax=ax, zorder=3)
            pitch.annotate(jersey_text, xy=(start_x, start_y), c='white', ha='center', va='center', size=7, weight='bold', ax=ax, zorder=4)

    # Iterate to plot lines and end markers
    for i, event_row in numeric_sequence_data.iterrows():
        start_x, start_y = event_row['x'], event_row['y']
        current_end_x, current_end_y = event_row.get('end_x'), event_row.get('end_y')

        # Plot Carry from previous event's end to this event's start
        if last_event_end_x is not None and last_event_end_y is not None:
            if pd.notna(start_x) and pd.notna(start_y):
                dist_carry_opta = np.sqrt((start_x - last_event_end_x)**2 + (start_y - last_event_end_y)**2)
                if dist_carry_opta >= 0.5: # Only plot carry if distance is significant
                    pitch.lines(last_event_end_x, last_event_end_y, start_x, start_y, lw=1.5, linestyle=CARRY_LINESTYLE, color=CARRY_COLOR, ax=ax, alpha=0.7, zorder=1)

        # --- Determine current event type and outcome for plotting ---
        event_type = event_row['type_name']
        event_outcome_is_successful = event_row['outcome'] == 'Successful'
        is_last_event_in_plotted_sequence = (i == numeric_sequence_data.index[-1])
        outcome_text = ""

        something_was_plotted = False
        outcome_text = "Sequence End" # Default outcome text

        if event_type == 'Pass':
            line_color_current = color_for_buildup_team if event_outcome_is_successful else unsuccessful_pass_color
            if pd.notna(current_end_x) and pd.notna(current_end_y):
                pitch.lines(start_x, start_y, current_end_x, current_end_y,
                            lw=pass_lw, transparent=True, comet=True, color=line_color_current,
                            ax=ax, alpha=pass_alpha, zorder=1)
                
                something_was_plotted = True

                if event_outcome_is_successful:
                    if is_last_event_in_plotted_sequence: # Last event of sequence IS a successful pass
                        receiver_jersey = event_row.get('receiver_jersey_number', '')
                        try: receiver_text = str(int(receiver_jersey)) if pd.notna(receiver_jersey) and receiver_jersey != '' else ''
                        except: receiver_text = str(receiver_jersey) if pd.notna(receiver_jersey) else ''
                        print(f"DEBUG MICHELE: last event: {event_row}")
                        # if event_row['team_name'] == team_that_lost_possession: # defending team regained possession                        
                        #     pitch.scatter(current_end_x, current_end_y, s=node_size, marker='s', facecolor='grey', edgecolor=unsuccessful_pass_color, linewidth=1.0, alpha=1, ax=ax, zorder=3)
                        #     outcome_text = f"Possession lost"
                        # else:
                        pitch.scatter(current_end_x, current_end_y, s=node_size, marker='o', facecolor=color_for_buildup_team, edgecolor=line_color, linewidth=1.0, alpha=1, ax=ax, zorder=3)
                        if receiver_text: pitch.annotate(receiver_text, xy=(current_end_x, current_end_y), c='white', ha='center', va='center', size=7, weight='bold', ax=ax, zorder=4)
                    else: # Successful pass, but not the last, plot small end marker
                        pitch.scatter(current_end_x, current_end_y, s=30, edgecolor=line_color_current, linewidth=1, facecolor=bg_color, zorder=2, ax=ax)
                else: # Unsuccessful Pass - this IS the terminating event
                    pitch.scatter(current_end_x, current_end_y, s=100, marker='X', color='black', linewidth=1.5, ax=ax, zorder=2)
                    if current_end_x >= 83 and (21.1 <= current_end_y <= 78.9): # If in the goal area
                        if metric_to_analyze == 'defensive_actions':
                            outcome_text = f"Chance conceded to {team_building_up}"
                        else: 
                            outcome_text = f"Big Chance"
                    else:
                        if metric_to_analyze == 'defensive_actions': 
                            outcome_text = f"Possession regained by {team_that_lost_possession}"
                        else:
                            outcome_text = f"Possession lost"

        elif event_type == 'Offside Pass':
            player_jersey = event_row.get('Mapped Jersey Number', '')
            try: player_text = str(int(player_jersey)) if pd.notna(player_jersey) and player_jersey != '' else ''
            except: player_text = str(player_jersey) if pd.notna(player_jersey) else ''
            if player_text: 
                pitch.scatter(start_x, start_y, s=node_size, marker='o', facecolor=color_for_buildup_team, edgecolor=line_color, linewidth=1.0, alpha=1, ax=ax, zorder=3)
                pitch.annotate(player_text, xy=(start_x, start_y), c='white', ha='center', va='center', size=7, weight='bold', ax=ax, zorder=4)
            if pd.notna(start_x) and pd.notna(start_y):
                pitch.lines(start_x, start_y, current_end_x, current_end_y,
                            lw=pass_lw, transparent=True, comet=True, color=VIOLET,
                            ax=ax, alpha=pass_alpha, zorder=1)
                pitch.scatter(current_end_x, current_end_y, s=100, marker='X', color=VIOLET, linewidth=1.5, ax=ax, zorder=2)
                something_was_plotted = True
            outcome_text = f"Offside"
            
        elif event_type in SHOT_TYPES: # If the terminating event is a shot
            outcome_colors = {'Miss': 'grey', 'Attempt Saved': 'blue', 'Goal': GREEN, 'Post': 'orange'}
            shot_line_color = outcome_colors.get(event_type, 'red') # Color based on shot type
            # Shot end_x, end_y should be adjusted by find_buildup_after_possession_loss
            # or we adjust them here if they are still pass-like
            shot_viz_end_x = 100.0
            final_shot_end_y_val = 50.0 # Default
            if 'shot_end_y' in event_row and pd.notna(event_row['shot_end_y']):
                final_shot_end_y_val = event_row['shot_end_y']
            elif 'Goal mouth y co-ordinate' in event_row and pd.notna(event_row['Goal mouth y co-ordinate']): # Fallback
                final_shot_end_y_val = event_row['Goal mouth y co-ordinate']
            elif 'end_y' in event_row and pd.notna(event_row['end_y']): # Further fallback
                 final_shot_end_y_val = event_row['end_y']
            
            temp_numeric_y = pd.to_numeric(final_shot_end_y_val, errors='coerce')
            final_shot_end_y = temp_numeric_y if pd.notna(temp_numeric_y) else 50.0

            if pd.notna(start_x) and pd.notna(start_y):
                pitch.lines(start_x, start_y, shot_viz_end_x, final_shot_end_y,
                            lw=3, transparent=True, comet=True, color=shot_line_color,
                            ax=ax, alpha=0.8, zorder=1)
                pitch.scatter(shot_viz_end_x, final_shot_end_y, s=80, marker='x',
                              color=shot_line_color, linewidth=2, ax=ax, zorder=2)
                something_was_plotted = True
            else:
                print(f"  Warning: Shot for sequence {loss_sequence_id} has no valid start coordinates.")
            
            if event_type == 'Goal':
                if metric_to_analyze == 'defensive_actions':
                    outcome_text = f"Goal by {team_building_up}"
                else:
                    outcome_text = f"Goal"
            else:
                if metric_to_analyze == 'defensive_actions':
                    outcome_text = f"Shot conceded to {team_building_up}"
                else:
                    outcome_text = f"Shot"

        elif event_type == 'Out':
            pitch.scatter(current_end_x, current_end_y, s=100, marker='X', color='black', linewidth=1.5, ax=ax, zorder=2)
            outcome_text = f"Out"
            something_was_plotted = True
        
        elif is_last_event_in_plotted_sequence: # Some other terminating event by team_building_up
            # e.g., Foul, ...
            player_jersey = event_row.get('Mapped Jersey Number', '')
            try: player_text = str(int(player_jersey)) if pd.notna(player_jersey) and player_jersey != '' else ''
            except: player_text = str(player_jersey) if pd.notna(player_jersey) else ''
            if player_text: pitch.annotate(player_text, xy=(start_x, start_y), c='white', ha='center', va='center', size=7, weight='bold', ax=ax, zorder=4)
            if pd.notna(current_end_x) and pd.notna(current_end_y): # If event has an end_x, end_y 
                pitch.scatter(current_end_x, current_end_y, s=0.9*node_size, marker='s', # Square for other end
                              facecolor='grey', edgecolor='black', ax=ax, zorder=2)
                something_was_plotted = True
            elif pd.notna(start_x) and pd.notna(start_y): # If only start_x, start_y (like a foul at location)
                 pitch.scatter(start_x, start_y, s=0.9*node_size, marker='s', facecolor='grey', edgecolor='black', ax=ax, zorder=2)
            outcome_text = event_type
            something_was_plotted = True
            
        if event_type in ('Dispossessed', 'Ball touch', 'Take On'):
            if metric_to_analyze == 'defensive_actions':
                outcome_text = f"Possession regained by {team_that_lost_possession}"
            else: 
                outcome_text = f"Possession lost"

        # Update last_event_end_x/y for next iteration's carry check
        if event_type == 'Pass' and event_outcome_is_successful and pd.notna(current_end_x) and pd.notna(current_end_y):
            last_event_end_x = current_end_x
            last_event_end_y = current_end_y
        else: # Sequence terminated or non-pass event
            last_event_end_x = None
            last_event_end_y = None

    if not something_was_plotted and not numeric_sequence_data.empty:
        # This code now correctly executes only if the entire loop completed without plotting.
        ax.text(50, 50, "Unplotted Sequence Type",
                ha='center', va='center', color='yellow', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.5))
        print(f"Warning: Sequence {loss_sequence_id} had data but the plotting logic did not handle it.")


    # --- Title ---
    if metric_to_analyze == 'defensive_actions':
        title_each_plot = f"Defensive Transition due to {type_of_loss_str}"  
    elif metric_to_analyze == 'buildup_phases':
        title_each_plot = f"Offensive Buildup starting from {type_of_loss_str}"
    else:
        title_each_plot = f"Offensive Transition thanks to {type_of_loss_str}"

    title_parts = [
        f"Time: {time_str} - {title_each_plot}",
        f"Outcome: {outcome_text}",
    ]
    ax.set_title("\n".join(title_parts), color=line_color, fontsize=8, fontweight='bold', pad=5, linespacing=1.3)
    ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
    for spine in ax.spines.values(): spine.set_visible(False)

# --- Plot Transition Sequence Function ---
def plot_transition_sequence(df_sequence, ax=None, title=None, team_name=None):
    """
    Plots a single offensive transition sequence on a football pitch.
    
    Args:
        df_sequence (pd.DataFrame): DataFrame containing one transition sequence.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates one if not provided.
        title (str, optional): Title to show above the pitch.
        team_name (str, optional): Name of the team (used in title or annotations).
    
    Returns:
        matplotlib.axes.Axes: The axis containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # --- Draw pitch background ---
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    ax.set_aspect("equal")

    # Draw pitch lines (optional: simplify if pitch is drawn elsewhere)
    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color="black", linewidth=1)
    ax.axvline(50, color="black", linewidth=0.5, linestyle="--")  # Halfway line
    ax.add_patch(patches.Arc((100, 50), 20, 20, angle=0, theta1=270, theta2=90, color="black"))
    ax.add_patch(patches.Arc((0, 50), 20, 20, angle=0, theta1=90, theta2=270, color="black"))

    # --- Plot pass arrows ---
    for i, row in df_sequence.iterrows():
        x, y = row['x'], row['y']
        end_x, end_y = row.get('end_x'), row.get('end_y')
        is_pass = row['type_name'] == 'Pass'
        is_success = row['outcome'] == 'Successful'

        if is_pass and not pd.isna(end_x) and not pd.isna(end_y):
            color = 'green' if is_success else 'red'
            ax.annotate("",
                        xy=(end_x, end_y), xytext=(x, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=2))

    # --- Plot player markers with jersey numbers ---
    for i, row in df_sequence.iterrows():
        x, y = row['x'], row['y']
        jersey = row.get("Mapped Jersey Number", "")
        ax.add_patch(plt.Circle((x, y), 1.5, color='blue', alpha=0.8))
        if jersey != "":
            ax.text(x, y, str(int(jersey)), color='white', ha='center', va='center', fontsize=8)

    # --- Add title ---
    if title:
        ax.set_title(title, fontsize=10)

    return ax

# --- NEW: Cross Analysis Plot Function ---
def plot_cross_heatmap_and_summary(fig, ax_pitch, ax_table, # Added ax_table
                                   team_crosses_df, total_crosses,
                                   attacking_team_name, summary_data, summary_cols,
                                   grid_bins_x=9, grid_bins_y=9,
                                   pitch_type='opta', # Default to Opta
                                   pitch_background_color=BG_COLOR,
                                   pitch_line_color=LINE_COLOR,
                                   heatmap_cmap='viridis',
                                   text_color=LINE_COLOR,
                                   table_header_bg_color='#E0E0E0',
                                   table_cell_bg_color='#F0F0F0'
                                   ):
    """
    Plots a heatmap of cross origination zones and a summary table on provided axes.
    Args:
        # ... (other args same)
        ax_table (matplotlib.axes.Axes): The axes pre-configured for the summary table.
        pitch_type (str): Type of pitch for mplsoccer (default 'opta').
    """
    print(f"  Plotting cross heatmap for {attacking_team_name} on provided axes...")
    print(f"  DEBUG: pitch_type received by plot_cross_heatmap_and_summary: {pitch_type}") # Check arg

    pitch = Pitch(pitch_type=pitch_type, # Uses arg, defaults to 'opta'
                  pitch_length=100, # Explicitly for Opta-like 0-100 data
                  pitch_width=100,  # Explicitly for Opta-like 0-100 data
                  pitch_color=pitch_background_color,
                  line_color=pitch_line_color,
                  goal_type='box',
                  line_zorder=1)
    pitch.draw(ax=ax_pitch) # Draw on the provided ax_pitch

    plot_df = team_crosses_df.copy()
    plot_df['x'] = pd.to_numeric(plot_df['x'], errors='coerce')
    plot_df['y'] = pd.to_numeric(plot_df['y'], errors='coerce')
    plot_df.dropna(subset=['x', 'y'], inplace=True) # Crucial: only plot valid coordinates

    if plot_df.empty:
        print(f"  No valid cross coordinates for heatmap for {attacking_team_name}.")
        ax_pitch.text(0.5, 0.5, "No cross data for heatmap", ha='center', va='center',
                      fontsize=12, color=text_color, transform=ax_pitch.transAxes)
    else:
        current_bins_arg = (grid_bins_x, grid_bins_y)
        print(f"  DEBUG: Type of pitch object: {type(pitch)}")
        print(f"  DEBUG: Is pitch an mplsoccer.Pitch instance? {isinstance(pitch, Pitch)}")
        print(f"  DEBUG: x data for bin_statistic (head): \n{plot_df['x'].head()}")
        print(f"  DEBUG: y data for bin_statistic (head): \n{plot_df['y'].head()}")
        print(f"  DEBUG: bins argument for bin_statistic: {current_bins_arg}")
        print(f"  DEBUG: Values for x: min={plot_df['x'].min()}, max={plot_df['x'].max()}")
        print(f"  DEBUG: Values for y: min={plot_df['y'].min()}, max={plot_df['y'].max()}")
        print(f"  DEBUG: Pitch dimensions (check consistency with data): length={pitch.dim.pitch_length}, width={pitch.dim.pitch_width}")


        # Ensure you are calling the method from the Pitch class you expect
        # This should be mplsoccer.pitch.Pitch.bin_statistic
        print(f"  DEBUG: pitch.bin_statistic method: {pitch.bin_statistic}")
        stats = pitch.bin_statistic(
            plot_df['x'], plot_df['y'],
            values=None, statistic='count',
            bins=(grid_bins_x, grid_bins_y)
        )

        # --- DEBUGGING: Print keys of stats ---
        print("DEBUG: Keys in stats dictionary:", stats.keys())
        # You can also print the whole dictionary if it's not too large:
        # print("DEBUG: Full stats dictionary:", stats)
        # --- END DEBUGGING ---

        if total_crosses > 0 : # total_crosses should be count of non-NaN coord crosses
            stats['statistic'] = (stats['statistic'] / total_crosses) * 100
        else:
            stats['statistic'] = np.zeros_like(stats['statistic'])

        pcm = pitch.heatmap(stats, ax=ax_pitch, cmap=heatmap_cmap, edgecolors=pitch_background_color, lw=0.5, zorder=0.8)

        path_effects_to_use = PATH_EFFECTS_HEATMAP if 'PATH_EFFECTS_HEATMAP' in globals() else None
        text_on_map_color = 'white'
        if heatmap_cmap.lower() in ['reds', 'blues', 'greens', 'greys', 'oranges', 'purples', 'hot_r', 'coolwarm_r', 'summer', 'spring', 'autumn', 'winter', 'bone', 'pink', 'copper']:
             text_on_map_color = 'black'

        # --- START OF CORRECTED ANNOTATION LOGIC ---
        if 'x_bin_count' in stats and 'y_bin_count' in stats:
            print("  DEBUG: Annotating using x_bin_count and y_bin_count (Standard Path).")
            for i in range(stats['x_bin_count']):
                for j in range(stats['y_bin_count']):
                    if stats['statistic'][j, i] > 0: # Assuming statistic is (y_bins, x_bins)
                        pitch.annotate(f"{stats['statistic'][j, i]:.1f}%",
                                       xy=(stats['x_grid'][0, i] + stats['width'][i] / 2,
                                           stats['y_grid'][j, 0] + stats['height'][j] / 2),
                                       ax=ax_pitch, color=text_on_map_color,
                                       va='center', ha='center', fontsize=8, weight='bold',
                                       path_effects=path_effects_to_use, zorder=2)

        elif 'cx' in stats and 'cy' in stats and \
             'statistic' in stats and isinstance(stats['statistic'], np.ndarray) and \
             stats['statistic'].ndim == 2:
            print("  DEBUG: Annotating using 'cx' and 'cy' keys for centers (Workaround Path).")
            num_y_bins_observed, num_x_bins_observed = stats['statistic'].shape
            if stats['cx'].shape == stats['statistic'].shape and stats['cy'].shape == stats['statistic'].shape:
                for j_idx in range(num_y_bins_observed):
                    for i_idx in range(num_x_bins_observed):
                        if stats['statistic'][j_idx, i_idx] > 0:
                            try:
                                center_x = stats['cx'][j_idx, i_idx]
                                center_y = stats['cy'][j_idx, i_idx]
                                pitch.annotate(f"{stats['statistic'][j_idx, i_idx]:.1f}%",
                                               xy=(center_x, center_y),
                                               ax=ax_pitch, color=text_on_map_color,
                                               va='center', ha='center', fontsize=8, weight='bold',
                                               path_effects=path_effects_to_use, zorder=2)
                            except IndexError:
                                print(f"    DEBUG: IndexError accessing cx/cy at [{j_idx},{i_idx}] while annotating.")
                            except Exception as e_annotate:
                                print(f"    DEBUG: Failed to annotate using cx/cy at [{j_idx},{i_idx}]: {e_annotate}")
            else:
                print(f"  DEBUG: Shape mismatch for cx/cy and statistic. cx: {stats['cx'].shape}, cy: {stats['cy'].shape}, statistic: {stats['statistic'].shape}. Cannot annotate with cx/cy.")

        elif 'xr' in stats and 'yr' in stats: # Fallback for very old mplsoccer
            print("  DEBUG: Annotating using 'xr' and 'yr' (Older mplsoccer Path).")
            num_x_bins_old = stats['xr']
            num_y_bins_old = stats['yr']
            for i in range(num_x_bins_old):
                for j in range(num_y_bins_old):
                    if stats['statistic'][j, i] > 0:
                        bin_center_x = (stats['x_grid'][i] + stats['x_grid'][i+1]) / 2
                        bin_center_y = (stats['y_grid'][j] + stats['y_grid'][j+1]) / 2
                        pitch.annotate(f"{stats['statistic'][j, i]:.1f}%",
                                       xy=(bin_center_x, bin_center_y),
                                       ax=ax_pitch, color=text_on_map_color,
                                       va='center', ha='center', fontsize=8, weight='bold',
                                       path_effects=path_effects_to_use, zorder=2)
        else:
            print("  ERROR-DEBUG: Suitable keys for heatmap annotation NOT FOUND IN STATS. No annotation will be performed.")
        # --- END OF CORRECTED ANNOTATION LOGIC ---

        cbar = fig.colorbar(pcm, ax=ax_pitch, orientation='vertical', shrink=0.7, pad=0.02)
        cbar.set_label('Percentage of Crosses from Zone (%)', color=text_color, fontsize=10)
        cbar.ax.tick_params(colors=text_color, labelsize=8)

    ax_pitch.set_title(f"{attacking_team_name} - Cross Origination Zones",
                       fontsize=16, color=text_color, pad=10, fontweight='bold')

    # Plot Summary Table on ax_table (ax_table should have axis('off') by caller)
    if summary_data and summary_cols:
        table = ax_table.table(cellText=summary_data,
                               colLabels=summary_cols,
                               loc='center', cellLoc='left',
                               colWidths=[0.6, 0.4])
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.3)
        for key, cell in table.get_celld().items():
            cell.set_edgecolor('grey'); cell.set_text_props(color=text_color)
            if key[0] == 0:
                cell.set_text_props(weight='bold', color=text_color); cell.set_facecolor(table_header_bg_color)
            else: cell.set_facecolor(table_cell_bg_color)
            if key[1] == 1: cell.set_text_props(ha='right')
    else:
        ax_table.text(0.5, 0.5, "No summary data", ha='center', va='center', color=text_color, transform=ax_table.transAxes)



# In src/visualization/pitch_plots.py

# ... (i tuoi import e le costanti come PITCH_WIDTH_OPTA, PITCH_HEIGHT_OPTA, LINE_COLOR, BG_COLOR) ...

def get_plotly_pitch_shapes(line_color="rgba(0, 0, 0, 0.5)", background_color=None):
    """
    Restituisce una lista di 'shapes' di Plotly per disegnare un campo da calcio.
    Questa funzione NON modifica una figura, ma restituisce solo gli oggetti.
    """
    PITCH_WIDTH_OPTA = 100
    PITCH_HEIGHT_OPTA = 100
    
    # shapes = [
    #     # Bordo campo e linea di metà campo
    #     dict(type="rect", x0=0, y0=0, x1=PITCH_WIDTH_OPTA, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2), layer="below"),
    #     dict(type="line", x0=50, y0=0, x1=50, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2), layer="below"),
    #     # Cerchio di centrocampo
    #     dict(type="circle", x0=41.5, y0=41.5, x1=58.5, y1=58.5, line=dict(color=line_color, width=2), layer="below"),
    #     # Aree di rigore
    #     dict(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=line_color, width=2), layer="below"),
    #     dict(type="rect", x0=100, y0=21.1, x1=83.5, y1=78.9, line=dict(color=line_color, width=2), layer="below"),
    #     # Aree piccole
    #     dict(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=line_color, width=2), layer="below"),
    #     dict(type="rect", x0=100, y0=36.8, x1=94.5, y1=63.2, line=dict(color=line_color, width=2), layer="below"),
    #     # Archi di rigore
    #     dict(type="path", path="M 16.5,34.9 C 22.5,42 22.5,58 16.5,65.1", line_color=line_color, layer="below"),
    #     dict(type="path", path="M 83.5,34.9 C 77.5,42 77.5,58 83.5,65.1", line_color=line_color, layer="below"),
    # ]
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=PITCH_WIDTH_OPTA, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2), layer="above"),
        dict(type="line", x0=50, y0=0, x1=50, y1=PITCH_HEIGHT_OPTA, line=dict(color=line_color, width=2), layer="above"),
        dict(type="circle", x0=41.5, y0=41.5, x1=58.5, y1=58.5, line=dict(color=line_color, width=2), layer="above"),
        dict(type="rect", x0=0, y0=21.1, x1=16.5, y1=78.9, line=dict(color=line_color, width=2), layer="above"),
        dict(type="rect", x0=100, y0=21.1, x1=83.5, y1=78.9, line=dict(color=line_color, width=2), layer="above"),
        dict(type="rect", x0=0, y0=36.8, x1=5.5, y1=63.2, line=dict(color=line_color, width=2), layer="above"),
        dict(type="rect", x0=100, y0=36.8, x1=94.5, y1=63.2, line=dict(color=line_color, width=2), layer="above"),
        dict(type="path", path="M 16.5,34.9 C 22.5,42 22.5,58 16.5,65.1", line_color=line_color, layer="above"),
        dict(type="path", path="M 83.5,34.9 C 77.5,42 77.5,58 83.5,65.1", line_color=line_color, layer="above"),
    ]
    return shapes