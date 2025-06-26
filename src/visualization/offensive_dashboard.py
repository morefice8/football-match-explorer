# src/visualization/offensive_dashboard.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os

from .pitch_plots import plot_transition_sequence
from ..utils.plot_helpers import get_transition_outcome_priority, create_donut_chart, create_zone_bar_chart


def plot_offensive_dashboard_with_sequences_multipage(
    df_sequences: pd.DataFrame,
    summary_table_df: pd.DataFrame,
    team_name: str,
    opponent_name: str,
    save_path: str = "output/plots",
    save_plots: bool = True,
    max_sequences_per_page: int = 6,
    keywords_priority: list[str] = None
):
    """
    Plots a multipage dashboard for offensive transitions.
    Each page includes: Donut (top left), Summary Table (top center), Bar chart (top right), and up to 6 pitch sequences below.
    """
    os.makedirs(save_path, exist_ok=True)

    # Define default keywords if none provided
    if keywords_priority is None:
        keywords_priority = ["goal", "shot", "chance"]

    # 1. Prepare global stats
    total_chances = df_sequences[df_sequences['sequence_outcome_type'].str.lower().str.contains('|'.join(keywords_priority), na=False)]
    outcome_counts = total_chances['sequence_outcome_type'].value_counts().to_dict()

    # 2. Prepare zone % chart
    zone_counts = df_sequences[df_sequences['sequence_outcome_type'].str.lower().str.contains('|'.join(keywords_priority), na=False)]
    zone_perc = (zone_counts.groupby('recovery_zone').size() / df_sequences.groupby('recovery_zone').size()) * 100
    zone_perc = zone_perc.fillna(0).round(1)

    # 3. Sort the summary table if needed
    summary_table_df = summary_table_df.copy()
    summary_table_df["sort_rank"] = summary_table_df["sequence_outcome_type"].apply(
        lambda x: get_transition_outcome_priority(x, keywords_priority)
    )
    summary_table_df = summary_table_df.sort_values("sort_rank").drop(columns="sort_rank")

    # 4. Paginate sequences
    unique_sequences = df_sequences['loss_sequence_id'].unique()
    total_pages = int(np.ceil(len(unique_sequences) / max_sequences_per_page))

    for page_idx in range(total_pages):
        fig = plt.figure(figsize=(16, 10))
        spec = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 3])

        # --- Top row ---
        ax_donut = fig.add_subplot(spec[0, 0])
        ax_table = fig.add_subplot(spec[0, 1:3])
        ax_bar = fig.add_subplot(spec[0, 3])

        create_donut_chart(
            ax_donut,
            labels=list(outcome_counts.keys()),
            values=list(outcome_counts.values()),
            title="Chances Created",
            center_label=str(sum(outcome_counts.values()))
        )
        create_zone_bar_chart(ax_bar, zone_perc)

        # --- Bottom row: 6 sequences ---
        seq_ids_to_plot = unique_sequences[page_idx * max_sequences_per_page:(page_idx + 1) * max_sequences_per_page]

        for i, seq_id in enumerate(seq_ids_to_plot):
            ax_pitch = fig.add_subplot(spec[1 + i // 3, i % 3])
            seq_df = df_sequences[df_sequences['loss_sequence_id'] == seq_id]
            plot_transition_sequence(ax=ax_pitch, df_sequence = seq_df, title=f"Sequence {seq_id}", team_name=team_name)

        fig.suptitle(f"{team_name} Offensive Transitions (vs {opponent_name}) - Page {page_idx + 1}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_plots:
            filename = f"{team_name.replace(' ', '_')}_offensive_transitions_page_{page_idx + 1}.png"
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Saved: {filepath}")
        else:
            plt.show()

        plt.close(fig)
