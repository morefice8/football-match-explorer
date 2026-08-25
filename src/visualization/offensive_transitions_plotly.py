# src/visualization/offensive_transitions_plotly.py

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from .defensive_transitions_plotly import draw_plotly_pitch # Riusiamo il disegnatore del campo
from ..config import BG_COLOR
from src.visualization.plotly_branding import add_attacking_direction
from src.visualization.coordinate_contract import orient_point

def plot_recovery_heatmap_on_pitch(
    sequences,
    is_away=False,
    grid_size=6,
):
    """Shared PLOT-12 recovery-location heatmap."""
    from src.visualization.transition_heatmap import plot_transition_heatmap

    return plot_transition_heatmap(
        sequences,
        location_kind="recovery",
        is_away=is_away,
        grid_size=grid_size,
    )

