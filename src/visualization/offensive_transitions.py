# src/visualization/offensive_transitions.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors

from src import config
from ..config import KEYWORDS_PRIORITY_FOR_OFFENSIVE_TRANSITIONS
from src.visualization import pitch_plots
from src.utils import plot_helpers

def plot_offensive_transitions_with_summary(df_sequences, df_summary, gaining_team_name,
                                             losing_team_name, team_color, is_away_team,
                                             fig_prefix, team_short_home, team_short_away,
                                             output_dir, save_plots):
    """
    Plot offensive transition sequences after a team gains possession.
    Similar to defensive transitions, grouped by recovery zone.
    """
    for zone, zone_group_df in df_sequences.groupby('recovery_zone'):
        if zone_group_df.empty:
            continue

        unique_seq_ids_in_zone = zone_group_df['loss_sequence_id'].unique()
        if len(unique_seq_ids_in_zone) == 0:
            continue

        print(f"  Plotting {len(unique_seq_ids_in_zone)} offensive transitions by {gaining_team_name} in {zone}...")

        PLOTS_PER_FIGURE = 6
        COLS_PER_FIGURE = 3

        for fig_num, i in enumerate(range(0, len(unique_seq_ids_in_zone), PLOTS_PER_FIGURE)):
            current_chunk_seq_ids = unique_seq_ids_in_zone[i: i + PLOTS_PER_FIGURE]
            rows = int(np.ceil(len(current_chunk_seq_ids) / COLS_PER_FIGURE))

            fig = plt.figure(figsize=(COLS_PER_FIGURE * 6.5, rows * 6.0 + 3.0), facecolor=config.BG_COLOR)
            gs = fig.add_gridspec(nrows=(rows + 1), ncols=COLS_PER_FIGURE, height_ratios=[1.2] + [1] * rows)

            fig.suptitle(f"{gaining_team_name} Offensive Transitions\nRecovered in {zone} - Page {fig_num + 1}",
                         fontsize=16, fontweight='bold', color=config.LINE_COLOR, y=0.98)

            ax_table = fig.add_subplot(gs[0, 1])
            ax_table.axis('off')

            zone_summary = df_summary[df_summary['recovery_zone'] == zone].copy()
            if zone_summary.empty:
                continue

            col_rename = {
                'sequence_outcome_type': 'Outcome',
                'count': 'Count',
                'avg_opp_passes_before_regain': 'Avg. Passes'
            }
            zone_summary = zone_summary.rename(columns=col_rename)
            zone_summary = zone_summary[list(col_rename.values())]

            zone_summary["sort_rank"] = zone_summary["Outcome"].apply(
                lambda outcome: plot_helpers.get_transition_outcome_priority(outcome, KEYWORDS_PRIORITY_FOR_OFFENSIVE_TRANSITIONS)
            )
            zone_summary = zone_summary.sort_values(by="sort_rank").drop(columns="sort_rank")

            table_vals = zone_summary.values.tolist()
            table_cols = list(zone_summary.columns)

            zone_summary['Avg. Passes'] = pd.to_numeric(zone_summary['Avg. Passes'], errors='coerce')
            valid_vals = zone_summary['Avg. Passes'].dropna()
            # norm = colors.Normalize(vmin=valid_vals.min(), vmax=min(valid_vals.max(), 12)) if not valid_vals.empty else None
            # cmap = cm.get_cmap('Greens')

            cell_colors = []
            for _, row in zone_summary.iterrows():
                row_color = []
                outcome_text = str(row['Outcome']).lower()
                for col in table_cols:
                    if "goal" in outcome_text:
                        row_color.append('green')
                    elif "shot" in outcome_text:
                        row_color.append('lightgreen')
                    elif "chance" in outcome_text:
                        row_color.append('skyblue')
                    elif "lost" in outcome_text:
                        row_color.append('lightcoral')
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
                donut_segment_colors = [plot_helpers.DEFAULT_OFFENSIVE_COLOR_MAP.get(lbl, 'grey') for lbl in labels_for_donut]
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
                    title=f"Offensive Transitions Breakdown", # Title for donut
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

            # # --- Transition Success Rate Bar Chart (gs[0, 2]) ---
            # ax_trans_success = fig.add_subplot(gs[0, 2])
            # # df_zone_success_summary should ALREADY be specific to the current team.
            # # If it contains all zones, plot_transition_success_rate_bars will handle it.
            # # If you want to show only the *current zone's* success rate here, filter it:
            # # current_zone_success_data = df_zone_success_summary[df_zone_success_summary['recovery_zone'] == zone]
            # # However, the bar chart is designed to show ALL zones, so pass the full team's summary
            # plot_helpers.plot_transition_success_rate_bars(
            #     ax_trans_success,
            #     zone_summary, # Pass the DataFrame for the current team
            #     f"Transition Success by Zone"
            # )
            # # --- End Transition Bar Chart ---

            # --- Plots Buildups Sequences ---
            axs = [fig.add_subplot(gs[r + 1, c]) for r in range(rows) for c in range(COLS_PER_FIGURE)]

            for plot_idx, seq_id in enumerate(current_chunk_seq_ids):
                if plot_idx < len(axs):
                    ax = axs[plot_idx]
                    sequence_data = zone_group_df[zone_group_df['loss_sequence_id'] == seq_id]
                    outcomes = sequence_data['sequence_outcome_type'].dropna().astype(str).str.lower().unique()
                    highlight_goals = any(kw in outcome for outcome in outcomes for kw in ['goal'])
                    highlight_chances = any(kw in outcome for outcome in outcomes for kw in ['shot', 'chance'])

                    pitch_plots.plot_opponent_buildup_after_loss(
                        ax, sequence_data,
                        losing_team_name,
                        gaining_team_name,
                        team_color,
                        seq_id, zone,
                        is_buildup_team_away=is_away_team,
                        metric_to_analyze='offensive_transitions'
                    )

                    if highlight_goals:
                        ax.set_facecolor('lightgreen')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#ff9900')
                            spine.set_linewidth(2)
                    elif highlight_chances:
                        ax.set_facecolor('lightyellow')
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#ff9900')
                            spine.set_linewidth(2)

            for j in range(len(current_chunk_seq_ids), len(axs)):
                axs[j].axis('off')

            plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.92, wspace=0.1, hspace=0.35)
            plt.show()

            if save_plots:
                save_path = os.path.join(output_dir, f"{team_short_home}_{team_short_away}_{fig_prefix}_in_{zone.replace(' ', '')}_p{fig_num + 1}.png")
                fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                print(f"Saved: {save_path}")
            plt.close(fig)
