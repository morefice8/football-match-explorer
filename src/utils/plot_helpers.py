# src/utils/plot_helpers.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import config # For default colors if needed

# Define default colors or import from config
BG_COLOR = config.BG_COLOR if hasattr(config, 'BG_COLOR') else '#FAFAFA'
LINE_COLOR = config.LINE_COLOR if hasattr(config, 'LINE_COLOR') else '#222222'
# Define colors for success/failure
SUCCESS_BAR_COLOR = config.GREEN if hasattr(config, 'GREEN') else 'limegreen'
FAILURE_BAR_COLOR = config.LOSS_COLOR if hasattr(config, 'LOSS_COLOR') else 'salmon'
NEUTRAL_BAR_COLOR = 'lightgrey'

# Default outcome color map for donuts/tables (can be overridden)
DEFAULT_OFFENSIVE_COLOR_MAP = {
    'Goals': 'green',
    'Shots': 'lightgreen', # For non-goal shots
    'Big Chances': 'skyblue', # If you distinguish this
    'Chance Created': config.VIOLET if hasattr(config, 'VIOLET') else 'mediumpurple', # For KPs/Assists not shot
    'Lost Possessions': 'lightcoral',
    'Out': 'lightgrey', # Ball out of play by attacking team
    'Foul': 'lightgrey', # Sequence ended without clear outcome
    'Offside': 'lightgrey', # Sequence ended without clear outcome
    # Add any other outcome strings your metrics might produce
}
DEFAULT_DEFENSIVE_COLOR_MAP = {
    'Goals conceded': 'salmon',
    'Shots conceded': 'orange', # For non-goal shots
    'Big Chances conceded': 'yellow', # If you distinguish this
    'Regained Possessions': 'lightgreen',
    'Out': 'lightgrey', # Ball out of play by attacking team
    'Foul': 'lightgrey', # Sequence ended without clear outcome
    'Offside': 'lightgrey', # Sequence ended without clear outcome
    # Add any other outcome strings your metrics might produce
}

# --- Helper: Prioritize outcome types for custom table sorting ---
def get_transition_outcome_priority(outcome_text: str, custom_priority: list = None) -> int:
    """
    Assigns a sorting rank to an outcome text for better visual priority in transition summary tables.
    Lower number = higher priority.

    Args:
        outcome_text (str): The outcome description (e.g., "Shot conceded to", "Goal by...").
        custom_priority (list, optional): List of keywords in desired priority order.

    Returns:
        int: Sorting rank (0 = highest priority).
    """
    keywords_priority = custom_priority if custom_priority else [
        "goal", "shot", "chance", "regain", "offside", "foul", "out"
    ]

    outcome_text = str(outcome_text).lower()
    for idx, keyword in enumerate(keywords_priority):
        if keyword in outcome_text:
            return idx
    return len(keywords_priority)

def create_donut_chart(ax, labels, values, title="", center_label=""):
    """
    Plots a donut chart showing the distribution of transition outcomes.
    
    Args:
        ax (matplotlib.axes.Axes): The axes on which to plot.
        labels (list): List of outcome category labels.
        values (list): List of values (frequencies) corresponding to each label.
        title (str): Title of the plot.
        center_label (str): Text to display in the center of the donut (usually the total).
    """
    colors = plt.cm.tab20.colors[:len(labels)]

    wedges, texts = ax.pie(
        values,
        labels=labels,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.4),
        colors=colors
    )

    ax.set(aspect="equal", title=title)
    ax.text(0, 0, center_label, ha='center', va='center', fontsize=14, fontweight='bold')

def create_zone_bar_chart(ax, zone_perc: dict, title=""):
    """
    Plots a horizontal bar chart showing % of transitions with a positive outcome by zone.
    """
    zones = list(zone_perc.keys())
    values = [zone_perc[z] for z in zones]

    bars = ax.barh(zones, values, color='skyblue')
    ax.set_xlim(0, 100)
    ax.set_xlabel('% with Goal/Shot/Chance')
    ax.set_title(title, fontsize=12)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", va='center', fontsize=10)

    ax.invert_yaxis()


def plot_donut_chart(ax, values, labels, title,
                     colors=None, # List of colors for segments
                     outcome_color_map=None, # Optional: dict to map labels to colors
                     total_in_center=True,
                     explode_values=None,
                     wedge_width=0.5, # Controls thickness (1.0 - inner_radius_ratio)
                     edge_color='white', edge_width=1.5,
                     autopct_format=lambda p, v_sum: '{:.0f}'.format(p * v_sum / 100) if p > 0 else '', # Shows count
                     pct_distance=0.75,
                     number_color='white', number_fontsize=8, number_fontweight='bold',
                     center_total_fontsize=16, center_total_fontweight='bold',
                     title_fontsize=12, # Increased default fontsize
                     title_fontweight='bold', # Default to bold
                     title_pad=10,      # Default padding
                     legend_kwargs=None):
    """
    Plots a flexible donut chart on the given axes.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        values (list): List of numerical values for each segment.
        labels (list): List of string labels for each segment.
        title (str): Title for the donut chart.
        colors (list, optional): A list of colors for segments. If None, uses outcome_color_map.
        outcome_color_map (dict, optional): Maps labels to colors. Used if 'colors' is None.
        total_in_center (bool, optional): If True, displays the sum of values in the center. Default is True.
        explode_values (list, optional): List of explode factors for each segment.
        wedge_width (float, optional): Width of the donut ring (0.0 to 1.0). Default is 0.5.
        edge_color (str, optional): Color of the lines between wedges. Default is 'white'.
        edge_width (float, optional): Width of the lines between wedges. Default is 1.5.
        autopct_format (function, optional): Function for formatting wedge labels. Default shows count.
        pct_distance (float, optional): Distance of autopct labels from the center. Default is 0.75.
        number_color (str, optional): Color for the numbers inside wedges. Default is 'white'.
        number_fontsize (int, optional): Fontsize for numbers inside wedges. Default is 8.
        number_fontweight (str, optional): Font weight for numbers inside wedges. Default is 'bold'.
        center_total_fontsize (int, optional): Fontsize for the total number in the center. Default is 16.
        center_total_fontweight (str, optional): Font weight for the total number in the center. Default is 'bold'.
        title_fontsize (int, optional): Fontsize for the title. Default is 12.
        title_fontweight (str, optional): Font weight for the title. Default is 'bold'.
        title_pad (int, optional): Padding for the title. Default is 10.
        legend_kwargs (dict, optional): Dictionary of keyword arguments for ax.legend().
            Example: {'title': "Outcomes", 'loc': "center left", 'bbox_to_anchor': (1, 0.5), 'ncol': 1}
            If None, no legend is plotted.
    """
    ax.axis('equal')  # Equal aspect ratio ensures circle.
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight=title_fontweight,
                     color=LINE_COLOR, pad=title_pad)

    if not values or sum(values) == 0:
        ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, fontsize=10)
        return

    # Determine segment colors
    segment_colors = []
    if colors:
        segment_colors = colors
    elif outcome_color_map:
        segment_colors = [outcome_color_map.get(lbl, 'grey') for lbl in labels]
    else: # Fallback
        cmap = plt.get_cmap("tab20") # Default qualitative cmap
        segment_colors = [cmap(i % cmap.N) for i in range(len(values))]


    outer_radius = 1.0
    inner_radius = outer_radius - wedge_width

    wedges, texts, autotexts = ax.pie(
        values,
        radius=outer_radius,
        wedgeprops=dict(width=outer_radius - inner_radius, edgecolor=edge_color, linewidth=edge_width),
        labels=None, # We handle labels via legend if provided
        colors=segment_colors,
        autopct=lambda p: autopct_format(p, sum(values)) if autopct_format else None,
        pctdistance=pct_distance,
        startangle=90,
        counterclock=False,
        explode=explode_values
    )

    for autotext in autotexts:
        autotext.set_color(number_color)
        autotext.set_fontsize(number_fontsize)
        autotext.set_fontweight(number_fontweight)

    if total_in_center:
        ax.text(0, 0, str(sum(values)), ha='center', va='center',
                fontsize=center_total_fontsize, fontweight=center_total_fontweight,
                color=LINE_COLOR)

    if legend_kwargs and labels:
        # Ensure labels match the plotted wedges (values > 0)
        valid_labels_for_legend = [labels[i] for i, val in enumerate(values) if val > 0]
        valid_wedges_for_legend = [wedges[i] for i, val in enumerate(values) if val > 0]
        if valid_wedges_for_legend:
             ax.legend(valid_wedges_for_legend, valid_labels_for_legend, **legend_kwargs)


def plot_transition_success_rate_bars(ax, df_zone_success_summary, title):
    """
    Plots a grouped or stacked bar chart showing transition success rates by recovery zone.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        df_zone_success_summary (pd.DataFrame): DataFrame with columns like
                                                ['recovery_zone', 'successful_transitions', 'failed_transitions', 'neutral_transitions', 'total_transitions']
                                                OR ['recovery_zone', 'success_rate_percentage']
                                                (Adapt based on your metrics function output).
        title (str): Title for the plot.
    """
    ax.set_title(title, fontsize=10, color=config.LINE_COLOR, pad=10, loc='center') # loc='center'

    if df_zone_success_summary is None or df_zone_success_summary.empty:
        ax.text(0.5, 0.5, "No Transition\nSuccess Data", ha='center', va='center', transform=ax.transAxes, fontsize=9)
        ax.axis('off')
        return

    # --- Option 1: Plotting direct success_rate_percentage (Simpler) ---
    # This assumes your metrics function calculates 'success_rate_percentage' (0-100)
    if 'success_rate_percentage' in df_zone_success_summary.columns and 'recovery_zone' in df_zone_success_summary.columns:
        # Ensure zones are in logical order
        zone_order = ['Defensive Third', 'Middle Third', 'Attacking Third']
        plot_data = df_zone_success_summary.set_index('recovery_zone').reindex(zone_order).fillna(0)
        plot_data = plot_data.reset_index() # Make recovery_zone a column again for plotting

        bars = ax.bar(plot_data['recovery_zone'], plot_data['success_rate_percentage'],
                      color=SUCCESS_BAR_COLOR, width=0.6, edgecolor=config.LINE_COLOR, linewidth=0.5)

        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height + 2, # Position above bar
                        f'{height:.0f}%', ha='center', va='bottom', color=config.LINE_COLOR, fontsize=8)
        ax.set_ylim(0, 110) # Y-axis from 0 to 100 (or slightly more for labels)
        ax.set_ylabel("Success Rate (%)", color=config.LINE_COLOR, fontsize=9)

    # --- Option 2: Plotting stacked bar for Successful/Failed/Neutral (More Detailed) ---
    elif all(col in df_zone_success_summary.columns for col in ['recovery_zone', 'successful_transitions', 'failed_transitions', 'total_transitions']):
        zone_order = ['Defensive Third', 'Middle Third', 'Attacking Third']
        plot_data = df_zone_success_summary.set_index('recovery_zone').reindex(zone_order).fillna(0)

        # Calculate percentages for stacking
        plot_data['success_perc'] = (plot_data['successful_transitions'] / plot_data['total_transitions'] * 100).fillna(0)
        plot_data['failure_perc'] = (plot_data['failed_transitions'] / plot_data['total_transitions'] * 100).fillna(0)
        # Assume neutral is the remainder if not explicitly provided
        if 'neutral_transitions' in plot_data.columns:
            plot_data['neutral_perc'] = (plot_data['neutral_transitions'] / plot_data['total_transitions'] * 100).fillna(0)
        else:
            plot_data['neutral_perc'] = 100 - plot_data['success_perc'] - plot_data['failure_perc']
            plot_data['neutral_perc'] = plot_data['neutral_perc'].clip(lower=0) # Ensure non-negative

        bottom = np.zeros(len(plot_data))
        categories_to_plot = {
            'Success': ('success_perc', SUCCESS_BAR_COLOR),
            'Neutral': ('neutral_perc', NEUTRAL_BAR_COLOR),
            'Failure': ('failure_perc', FAILURE_BAR_COLOR),
        }

        for label, (perc_col, color) in categories_to_plot.items():
            if perc_col in plot_data.columns:
                percentages = plot_data[perc_col]
                bars = ax.bar(plot_data.index, percentages, bottom=bottom, label=label, color=color, width=0.7)
                for bar_idx, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 5: # Only label significant segments
                        ax.text(bar.get_x() + bar.get_width() / 2., bottom[bar_idx] + height / 2.,
                                f'{height:.0f}%', ha='center', va='center', color='white', fontsize=7, fontweight='bold')
                bottom += percentages
        ax.set_ylim(0, 100)
        ax.set_ylabel("Outcome Percentage", color=config.LINE_COLOR, fontsize=9)
        ax.legend(title="Transition Outcome", fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    else:
        ax.text(0.5, 0.5, "Data Format Error", ha='center', va='center', transform=ax.transAxes)


    # General Styling
    ax.tick_params(axis='x', labelrotation=0, colors=config.LINE_COLOR, labelsize=8) # Smaller labels
    ax.tick_params(axis='y', colors=config.LINE_COLOR, labelsize=8)
    ax.set_facecolor(config.BG_COLOR)
    for spine in ['top', 'right']: ax.spines[spine].set_visible(False)
    ax.spines['left'].set_color(config.LINE_COLOR)
    ax.spines['bottom'].set_color(config.LINE_COLOR)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color=config.LINE_COLOR, zorder=0)