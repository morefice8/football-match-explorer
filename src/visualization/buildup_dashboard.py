# src/visualization/buildup_dashboard.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np

from .pitch_plots import plot_transition_sequence
from src.utils.plot_helpers import get_transition_outcome_priority, create_donut_chart, create_zone_bar_chart


def plot_buildup_dashboard_with_sequences_multipage(
    df_sequences,
    summary_table_df,
    team_name,
    opponent_name,
    save_path="output/plots",
    save_plots=True,
    max_sequences_per_page=6,
    keywords_priority=None
):
    """
    Plots a multipage dashboard for buildup phases starting from defensive third.
    Each page includes: Donut (top left), Summary Table (top center), Bar chart (top right), and up to 6 pitch sequences below.
    """
    os.makedirs(save_path, exist_ok=True)

    if keywords_priority is None:
        keywords_priority = ["goal", "shot", "chance"]

    # Donut data (count outcomes)
    total_outcomes = df_sequences[df_sequences['sequence_outcome_type'].str.lower().str.contains('|'.join(keywords_priority), na=False)]
    outcome_counts = total_outcomes['sequence_outcome_type'].value_counts().to_dict()

    # Right bar chart (percent positive outcomes by zone)
    zone_total = df_sequences['start_zone'].value_counts()
    zone_positive = total_outcomes['start_zone'].value_counts()
    zone_perc = (zone_positive / zone_total * 100).fillna(0).round(1).to_dict()

    # Sort table by importance
    summary_table_df = summary_table_df.copy()
    summary_table_df["sort_rank"] = summary_table_df["sequence_outcome_type"].apply(
        lambda x: get_transition_outcome_priority(x, keywords_priority)
    )
    summary_table_df = summary_table_df.sort_values("sort_rank").drop(columns="sort_rank")

    unique_sequences = df_sequences['buildup_sequence_id'].unique()
    total_pages = int(np.ceil(len(unique_sequences) / max_sequences_per_page))

    for page_idx in range(total_pages):
        fig = plt.figure(figsize=(16, 10))
        spec = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 3])

        ax_donut = fig.add_subplot(spec[0, 0])
        ax_table = fig.add_subplot(spec[0, 1:3])
        ax_bar = fig.add_subplot(spec[0, 3])

        create_donut_chart(
            ax_donut,
            list(outcome_counts.keys()),
            list(outcome_counts.values()),
            title="Effective Buildups",
            center_label=str(sum(outcome_counts.values()))
        )

        ax_table.axis('off')
        ax_table.table(
            cellText=summary_table_df.values,
            colLabels=summary_table_df.columns,
            loc='center',
            cellLoc='center'
        )

        create_zone_bar_chart(ax_bar, zone_perc, title="% Buildups with Outcome")

        seq_ids_to_plot = unique_sequences[page_idx * max_sequences_per_page:(page_idx + 1) * max_sequences_per_page]

        for i, seq_id in enumerate(seq_ids_to_plot):
            ax_pitch = fig.add_subplot(spec[1 + i // 3, i % 3])
            seq_df = df_sequences[df_sequences['buildup_sequence_id'] == seq_id]
            plot_transition_sequence(df_sequence=seq_df, ax=ax_pitch, team_name=team_name)

        fig.suptitle(f"{team_name} Buildup Phases (vs {opponent_name}) - Page {page_idx + 1}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        if save_plots:
            filename = f"{team_name.replace(' ', '_')}_buildups_page_{page_idx + 1}.png"
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, dpi=300)
            print(f"Saved: {filepath}")
        else:
            plt.show()

        plt.close(fig)
