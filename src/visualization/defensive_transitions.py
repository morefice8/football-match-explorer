# src/data_preparation_for_plots/defensive_transitions.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from src import config
from ..config import KEYWORDS_PRIORITY_FOR_OFFENSIVE_TRANSITIONS
from src.visualization import pitch_plots
from src.utils import plot_helpers

# This function plots opponent buildup sequences with summary statistics.
def plot_defensive_transitions_with_summary(df_sequences, df_summary, lost_team_name, buildup_team_name,
                                        buildup_team_color, is_buildup_team_away, fig_prefix,
                                        loss_side_short, output_dir, save_plots, team_short_home, team_short_away):
    """
    Plot defensive transitions with summary statistics for opponent buildup sequences after possession loss.
    This function visualizes sequences where a team loses possession and the opponent initiates a buildup, grouped by the zone of loss. For each zone, it creates figures containing multiple sequence plots and a summary statistics table. The summary table highlights outcomes (e.g., goals, shots, regains) with color coding, and the sequence plots can be optionally saved to disk.
        df_sequences (pd.DataFrame): DataFrame containing detailed sequence data, including loss zones and sequence IDs.
        df_summary (pd.DataFrame): DataFrame containing summary statistics for each loss zone and outcome type.
        buildup_team_name (str): Name of the team that gained possession and started the buildup.
        buildup_team_color (str): Color used for plotting the buildup team.
        is_buildup_team_away (bool): Indicates if the buildup team is the away team.
        fig_prefix (str): Prefix for figure filenames when saving plots.
        loss_side_short (str): Short identifier for the side where the loss occurred.
        output_dir (str): Directory path to save the generated plots.
        save_plots (bool): Whether to save the plots to disk.
        team_short_home (str): Short name or code for the home team.
        team_short_away (str): Short name or code for the away team.
    Returns:
        None. The function generates and displays (and optionally saves) summary figures for defensive transitions.
    """

    for loss_zone_name, zone_group_df in df_sequences.groupby('loss_zone'):
        if zone_group_df.empty:
            continue

        unique_seq_ids_in_zone = zone_group_df['loss_sequence_id'].unique()
        if len(unique_seq_ids_in_zone) == 0:
            continue

        print(f"  Plotting {len(unique_seq_ids_in_zone)} sequences by {buildup_team_name} after {lost_team_name} loss in {loss_zone_name}...")

        PLOTS_PER_FIGURE = 6
        COLS_PER_FIGURE = 3

        for fig_num, i in enumerate(range(0, len(unique_seq_ids_in_zone), PLOTS_PER_FIGURE)):
            current_chunk_seq_ids = unique_seq_ids_in_zone[i: i + PLOTS_PER_FIGURE]
            rows = int(np.ceil(len(current_chunk_seq_ids) / COLS_PER_FIGURE))

            fig = plt.figure(figsize=(COLS_PER_FIGURE * 6.5, rows * 6.0 + 3.0), facecolor=config.BG_COLOR)
            gs = fig.add_gridspec(nrows=(rows + 1), ncols=COLS_PER_FIGURE, height_ratios=[1.2] + [1] * rows)

            fig.suptitle(f"{lost_team_name} Defensive Transitions\nPossession Lost in {loss_zone_name} - Page {fig_num + 1}",
                         fontsize=16, fontweight='bold', color=config.LINE_COLOR, y=0.98)

            ax_table = fig.add_subplot(gs[0, 1])
            ax_table.axis('off')

            zone_summary = df_summary[df_summary['loss_zone'] == loss_zone_name].copy()
            if zone_summary.empty:
                continue

            col_rename = {
                'sequence_outcome_type': 'Outcome',
                'count': 'Count',
                'avg_opp_passes_before_regain': 'Avg. Passes Before Regain'
            }
            zone_summary = zone_summary.rename(columns=col_rename)
            zone_summary = zone_summary[list(col_rename.values())]
            zone_summary["sort_rank"] = zone_summary["Outcome"].apply(
                lambda outcome: plot_helpers.get_transition_outcome_priority(outcome, KEYWORDS_PRIORITY_FOR_OFFENSIVE_TRANSITIONS)
            )
            zone_summary = zone_summary.sort_values(by="sort_rank").drop(columns="sort_rank")

            table_vals = zone_summary.values.tolist()
            table_cols = list(zone_summary.columns)

            zone_summary['Avg. Passes Before Regain'] = pd.to_numeric(zone_summary['Avg. Passes Before Regain'], errors='coerce')
            valid_vals = zone_summary['Avg. Passes Before Regain'].dropna()
            # norm = colors.Normalize(vmin=valid_vals.min(), vmax=min(valid_vals.max(), 12)) if not valid_vals.empty else None
            # cmap = cm.get_cmap('cool')

            cell_colors = []
            for _, row in zone_summary.iterrows():
                row_color = []
                outcome_text = str(row['Outcome']).lower()
                for col in table_cols:
                    if "goal" in outcome_text:
                        row_color.append('salmon')
                    elif "shot" in outcome_text:
                        row_color.append('orange')
                    elif "chance" in outcome_text:
                        row_color.append('yellow')
                    elif "regain" in outcome_text:
                        row_color.append('lightgreen')
                    else:
                        row_color.append('lightgrey')
                cell_colors.append(row_color)

            tab = ax_table.table(cellText=table_vals, colLabels=table_cols, cellColours=cell_colors,
                                 colWidths=[0.4, 0.2, 0.3], loc='center', cellLoc='center')
            tab.auto_set_font_size(False)
            tab.set_fontsize(11)
            tab.scale(1.5, 1.6)
            for i in range(len(table_cols)):
                tab[0, i].set_text_props(ha='center', weight='bold')
                for j in range(len(table_vals)):
                    tab[j + 1, i].set_text_props(ha='center')

            # --- Donut Chart (Top Left - gs[0, 0]) ---
            ax_donut = fig.add_subplot(gs[0, 0])
            if not zone_summary.empty and 'Count' in zone_summary.columns and 'Outcome' in zone_summary.columns:
                counts_for_donut = zone_summary['Count'].tolist()
                labels_for_donut = zone_summary['Outcome'].tolist()

                # Map outcome labels to consistent colors for the donut
                donut_segment_colors = [plot_helpers.DEFAULT_DEFENSIVE_COLOR_MAP.get(lbl, 'grey') for lbl in labels_for_donut]
                # Example: Explode all segments slightly
                explode_vals = [0.05] * len(counts_for_donut)

                # Filter out zero counts for pie chart
                valid_indices = [k for k, count_val in enumerate(counts_for_donut) if count_val > 0]
                final_counts = [counts_for_donut[k] for k in valid_indices]
                final_labels = [labels_for_donut[k] for k in valid_indices]
                final_colors = [donut_segment_colors[k] for k in valid_indices]
                final_explode = [explode_vals[k] for k in valid_indices] if explode_vals else None

                plot_helpers.plot_donut_chart(
                    ax_donut,
                    values=final_counts,
                    labels=final_labels,
                    title=f"Defensive Transitions Breakdown", # Title for donut
                    colors=final_colors,
                    explode_values=final_explode,
                    wedge_width=0.45, # Make it thick
                    edge_color='white', edge_width=1,
                    number_fontsize=15, 
                    number_color='black',
                    center_total_fontsize=45,
                    title_fontsize=12,
                    legend_kwargs=None
                )
            else:
                ax_donut.text(0.5, 0.5, "No Donut Data", ha='center', va='center'); ax_donut.axis('off')
            # --- End Donut ---

            # --- Placeholder for Right KPI (gs[0, 2]) ---
            ax_kpi_right = fig.add_subplot(gs[0, 2])
            ax_kpi_right.text(0.5, 0.5, "Right KPI\n(e.g., Success % by Zone)", ha='center', va='center', fontsize=10)
            ax_kpi_right.axis('off')

            axs_opp_bu = [fig.add_subplot(gs[r + 1, c]) for r in range(rows) for c in range(COLS_PER_FIGURE)]

            for plot_idx, seq_id in enumerate(current_chunk_seq_ids):
                if plot_idx < len(axs_opp_bu):
                    ax = axs_opp_bu[plot_idx]
                    sequence_data = zone_group_df[zone_group_df['loss_sequence_id'] == seq_id]
                    outcomes = sequence_data['sequence_outcome_type'].dropna().astype(str).str.lower().unique()
                    highlight = any(kw in outcome for outcome in outcomes for kw in ['goal', 'shot', 'chance'])

                    pitch_plots.plot_opponent_buildup_after_loss(
                                ax, sequence_data,
                                lost_team_name,
                                buildup_team_name,
                                buildup_team_color,
                                seq_id, loss_zone_name,
                                is_buildup_team_away=is_buildup_team_away
                            )

                    if highlight:
                        ax.set_facecolor('#fff2cc')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#ff9900')
                            spine.set_linewidth(2)

            for j in range(len(current_chunk_seq_ids), len(axs_opp_bu)):
                axs_opp_bu[j].axis('off')

            plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.1, hspace=0.35)
            plt.show()

            if save_plots:
                save_path = os.path.join(output_dir, f"{team_short_home}_{team_short_away}_{fig_prefix}_in_{loss_zone_name.replace(' ', '')}_p{fig_num + 1}.png")
                fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                print(f"Saved: {save_path}")
            plt.close(fig)
