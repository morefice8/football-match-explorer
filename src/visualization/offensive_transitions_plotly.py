# src/visualization/offensive_transitions_plotly.py

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

