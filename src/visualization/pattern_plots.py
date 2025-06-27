# src/visualization/pattern_plots.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Import config for colors if not passed directly
from src import config

# Define colors (get from config or define defaults)
BG_COLOR = config.BG_COLOR if hasattr(config, 'BG_COLOR') else '#FAFAFA'
LINE_COLOR = config.LINE_COLOR if hasattr(config, 'LINE_COLOR') else '#222222'

def format_pattern_label(pattern_tuple):
    """Formats a pattern tuple into a readable string label."""
    return ' -> '.join(map(str, pattern_tuple)) # Join elements with arrow

def plot_pattern_bar_chart(ax, patterns_series, title, color, top_n=5):
    """
    Plots a horizontal bar chart showing the frequency of top patterns.

    Args:
        ax (matplotlib.axes.Axes): The axes to plot on.
        patterns_series (pd.Series): Series with patterns (tuples) as index
                                     and counts as values, sorted descending.
        title (str): The title for the subplot.
        color (str): The color for the bars.
        top_n (int): Number of top patterns to display.
    """
    if patterns_series is None or patterns_series.empty:
        ax.text(0.5, 0.5, "No Patterns Found", ha='center', va='center', fontsize=12, color='red', transform=ax.transAxes)
        ax.set_title(title, color=LINE_COLOR, fontsize=14, fontweight='bold')
        ax.axis('off') # Hide axes if no data
        return

    # Get top N patterns and reverse for plotting (top item at top)
    top_patterns = patterns_series.head(top_n).iloc[::-1]
    counts = top_patterns.values
    # Format labels from tuples
    labels = [format_pattern_label(pattern) for pattern in top_patterns.index]

    # Plot bars
    bars = ax.barh(labels, counts, color=color, edgecolor=LINE_COLOR, linewidth=0.5, zorder=3)

    # Add labels to bars
    ax.bar_label(bars, labels=[f'{c}' for c in counts], padding=3, color=LINE_COLOR, fontsize=10, zorder=4)

    # Styling
    max_count = top_patterns.max()
    ax.set_xlim(0, max_count * 1.15) # Add padding for labels

    # Optional: Add light grid lines
    ax.xaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3, zorder=0)
    ax.set_axisbelow(True) # Ensure grid is behind bars

    ax.set_facecolor(BG_COLOR)
    ax.tick_params(axis='x', colors=LINE_COLOR, labelsize=10)
    ax.tick_params(axis='y', colors=LINE_COLOR, labelsize=10) # Adjust label size if needed
    # Hide spines
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    # Hide x-axis ticks if counts are labeled on bars
    ax.xaxis.set_ticks_position('none')
    ax.set_xlabel("Frequency", color=LINE_COLOR, fontsize=11)

    ax.set_title(title, color=LINE_COLOR, fontsize=14, fontweight='bold', pad=10)