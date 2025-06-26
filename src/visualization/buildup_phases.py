# src/visualization/buildup_phases.py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import os
from mplsoccer import Pitch

from ..config import LINE_COLOR, KEYWORDS_PRIORITY_FOR_OFFENSIVE_TRANSITIONS, BG_COLOR

from src.visualization import pitch_plots
from src.utils import plot_helpers

def plot_buildup_phases_with_summary(df_sequences, df_summary, attacking_team_name,
                                             defending_team_name, team_color, is_away_team,
                                             fig_prefix, team_short_home, team_short_away,
                                             output_dir, save_plots):
    """
    Plot buildup sequences of each team.
    Similar to defensive transitions, grouped by trigger zone.
    """
    for zone, zone_group_df in df_sequences.groupby('trigger_zone'):
        if zone_group_df.empty:
            continue

        unique_seq_ids_in_zone = zone_group_df['trigger_sequence_id'].unique()
        if len(unique_seq_ids_in_zone) == 0:
            continue

        print(f"  Plotting {len(unique_seq_ids_in_zone)} buildup sequences by {attacking_team_name} in {zone}...")

        PLOTS_PER_FIGURE = 6
        COLS_PER_FIGURE = 3

        for fig_num, i in enumerate(range(0, len(unique_seq_ids_in_zone), PLOTS_PER_FIGURE)):
            current_chunk_seq_ids = unique_seq_ids_in_zone[i: i + PLOTS_PER_FIGURE]
            rows = int(np.ceil(len(current_chunk_seq_ids) / COLS_PER_FIGURE))

            fig = plt.figure(figsize=(COLS_PER_FIGURE * 6.5, rows * 6.0 + 3.0), facecolor=BG_COLOR)
            gs = fig.add_gridspec(nrows=(rows + 1), ncols=COLS_PER_FIGURE, height_ratios=[1.2] + [1] * rows)

            fig.suptitle(f"{attacking_team_name} Offensive Buildups\nStarting from {zone} - Page {fig_num + 1}",
                         fontsize=16, fontweight='bold', color=LINE_COLOR, y=0.98)

            ax_table = fig.add_subplot(gs[0, 1])
            ax_table.axis('off')

            zone_summary = df_summary[df_summary['trigger_zone'] == zone].copy()
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
                    title=f"Offensive Buildups Breakdown", # Title for donut
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
            # # current_zone_success_data = df_zone_success_summary[df_zone_success_summary['trigger_zone'] == zone]
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
                    sequence_data = zone_group_df[zone_group_df['trigger_sequence_id'] == seq_id]
                    outcomes = sequence_data['sequence_outcome_type'].dropna().astype(str).str.lower().unique()
                    highlight_goals = any(kw in outcome for outcome in outcomes for kw in ['goal'])
                    highlight_chances = any(kw in outcome for outcome in outcomes for kw in ['shot', 'chance'])

                    pitch_plots.plot_opponent_buildup_after_loss(
                        ax, sequence_data,
                        defending_team_name,
                        attacking_team_name,
                        team_color,
                        seq_id, zone,
                        is_buildup_team_away=is_away_team,
                        metric_to_analyze='buildup_phases'
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


# def plot_buildup_dashboard_with_sequences(
#     df_sequences: pd.DataFrame,
#     team_name: str,
#     opponent_name: str,
#     save_path: str = "output/plots",
#     save_plots: bool = True,
#     max_sequences_per_page: int = 6
# ):
#     """
#     Plots a dashboard per zone with: donut (chances), table (summary), bar chart (% successful progression),
#     and up to 6 pitch sequences per page.
#     """
#     os.makedirs(save_path, exist_ok=True)

#     # --- 1. Prepare summary table ---
#     summary = df_sequences.groupby('buildup_id').agg({
#         'reached_middle': 'max',
#         'reached_final_third': 'max',
#         'reached_danger_zone': 'max',
#         'ended_in_shot': 'max',
#         'ended_in_goal': 'max',
#         'lost_in_own_half': 'max'
#     }).reset_index()

#     summary_table = pd.DataFrame({
#         "Phase Outcome": [
#             "Lost in Own Half", "Reached Midfield", "Reached Final Third",
#             "Reached Danger Zone", "Shot Taken", "Goal Scored"
#         ],
#         "Count": [
#             summary['lost_in_own_half'].sum(),
#             summary['reached_middle'].sum(),
#             summary['reached_final_third'].sum(),
#             summary['reached_danger_zone'].sum(),
#             summary['ended_in_shot'].sum(),
#             summary['ended_in_goal'].sum()
#         ]
#     })

#     total = len(summary)
#     summary_table["%"] = (summary_table["Count"] / total * 100).round(1)

#     # --- 2. Define pages ---
#     unique_buildups = df_sequences['buildup_id'].unique()
#     total_pages = int(np.ceil(len(unique_buildups) / max_sequences_per_page))

#     for page in range(total_pages):
#         fig = plt.figure(figsize=(16, 10))
#         spec = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 3])

#         ax_donut = fig.add_subplot(spec[0, 0])
#         ax_table = fig.add_subplot(spec[0, 1:3])
#         ax_bar = fig.add_subplot(spec[0, 3])

#         # --- Donut ---
#         donut_labels = summary_table["Phase Outcome"]
#         donut_values = summary_table["Count"]
#         create_donut_chart(ax_donut, labels=donut_labels, values=donut_values, title="Buildup Outcomes", center_label=str(total))

#         # --- Table ---
#         ax_table.axis('off')
#         ax_table.table(
#             cellText=summary_table.values,
#             colLabels=summary_table.columns,
#             loc='center',
#             cellLoc='center'
#         )

#         # --- Bar chart ---
#         perc_dict = dict(zip(summary_table["Phase Outcome"], summary_table["%"]))
#         create_zone_bar_chart(ax_bar, perc_dict, title="% of Sequences")

#         # --- Sequences ---
#         seq_ids_to_plot = unique_buildups[page * max_sequences_per_page:(page + 1) * max_sequences_per_page]
#         for i, seq_id in enumerate(seq_ids_to_plot):
#             ax_pitch = fig.add_subplot(spec[1 + i // 3, i % 3])
#             seq_df = df_sequences[df_sequences['buildup_id'] == seq_id]
#             plot_transition_sequence(df_sequence=seq_df, ax=ax_pitch, team_name=team_name)

#         fig.suptitle(f"{team_name} Buildup Phases (vs {opponent_name}) - Page {page + 1}", fontsize=14)
#         plt.tight_layout(rect=[0, 0, 1, 0.97])

#         if save_plots:
#             filename = f"{team_name.replace(' ', '_')}_buildup_page_{page + 1}.png"
#             plt.savefig(os.path.join(save_path, filename), dpi=300)
#             print(f"Saved: {filename}")
#         else:
#             plt.show()

#         plt.close(fig)

# def plot_buildup_dashboard_with_sequences_multipage(
#     df_sequences: pd.DataFrame,
#     summary_table_df: pd.DataFrame,
#     team_name: str,
#     opponent_name: str,
#     save_path: str = "output/plots",
#     save_plots: bool = True,
#     max_sequences_per_page: int = 6,
#     keywords_priority: list[str] = None
# ):
#     """
#     Plots a multipage dashboard for buildup phases.
#     Each page includes: Donut (top left), Summary Table (top center), Bar chart (top right), and up to 6 pitch sequences below.
#     """
#     os.makedirs(save_path, exist_ok=True)

#     if keywords_priority is None:
#         keywords_priority = ["goal", "shot", "chance", "final third"]

#     # 1. Prepare donut values
#     total_outcomes = df_sequences[df_sequences['sequence_outcome_type'].str.lower().str.contains('|'.join(keywords_priority), na=False)]
#     outcome_counts = total_outcomes['sequence_outcome_type'].value_counts().to_dict()

#     # 2. Prepare zone % bar chart
#     zone_total = df_sequences['start_zone'].value_counts()
#     zone_positive = total_outcomes['start_zone'].value_counts()
#     zone_perc = (zone_positive / zone_total * 100).fillna(0).round(1).to_dict()

#     # 3. Prepare summary table
#     summary_table_df = summary_table_df.copy()
#     summary_table_df["sort_rank"] = summary_table_df["sequence_outcome_type"].apply(
#         lambda x: get_transition_outcome_priority(x, keywords_priority)
#     )
#     summary_table_df = summary_table_df.sort_values("sort_rank").drop(columns="sort_rank")

#     unique_sequences = df_sequences['buildup_sequence_id'].unique()
#     total_pages = int(np.ceil(len(unique_sequences) / max_sequences_per_page))

#     for page_idx in range(total_pages):
#         fig = plt.figure(figsize=(16, 10))
#         spec = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 3])

#         # Top row
#         ax_donut = fig.add_subplot(spec[0, 0])
#         ax_table = fig.add_subplot(spec[0, 1:3])
#         ax_bar = fig.add_subplot(spec[0, 3])

#         create_donut_chart(ax_donut, list(outcome_counts.keys()), list(outcome_counts.values()), title="Build-up Outcomes", center_label=str(sum(outcome_counts.values())))
#         ax_table.axis('off')
#         ax_table.table(
#             cellText=summary_table_df.values,
#             colLabels=summary_table_df.columns,
#             loc='center',
#             cellLoc='center'
#         )
#         create_zone_bar_chart(ax_bar, zone_perc, title="% Positive Outcomes by Zone")

#         # Sequences
#         seq_ids_to_plot = unique_sequences[page_idx * max_sequences_per_page:(page_idx + 1) * max_sequences_per_page]
#         for i, seq_id in enumerate(seq_ids_to_plot):
#             ax_pitch = fig.add_subplot(spec[1 + i // 3, i % 3])
#             seq_df = df_sequences[df_sequences['buildup_sequence_id'] == seq_id]
#             plot_transition_sequence(df_sequence=seq_df, ax=ax_pitch, team_name=team_name)

#         fig.suptitle(f"{team_name} Build-up Phases vs {opponent_name} - Page {page_idx + 1}", fontsize=14)
#         plt.tight_layout(rect=[0, 0, 1, 0.97])

#         if save_plots:
#             filename = f"{team_name.replace(' ', '_')}_buildup_phases_page_{page_idx + 1}.png"
#             filepath = os.path.join(save_path, filename)
#             plt.savefig(filepath, dpi=300)
#             print(f"Saved: {filepath}")
#         else:
#             plt.show()

#         plt.close(fig)

def plot_single_buildup_sequence(ax, sequence_df, team_color):
    """
    Plots a single, complete buildup sequence on a pitch.
    """
    pitch = Pitch(pitch_type='opta', pitch_color=BG_COLOR, line_color=LINE_COLOR, line_zorder=2)
    pitch.draw(ax=ax)

    if sequence_df.empty:
        ax.text(50, 50, "No Sequence Data", ha='center', va='center', color='red', fontsize=12)
        return

    # Separate passes from the final shot/event
    passes_df = sequence_df[sequence_df['type_name'] == 'Pass']
    final_event_df = sequence_df.iloc[-1:] # The last event in the sequence

    # Plot passes with a consistent color
    pitch.lines(
        xstart=passes_df['x'], ystart=passes_df['y'],
        xend=passes_df['end_x'], yend=passes_df['end_y'],
        ax=ax, color='#a3a3a3', lw=2, zorder=3, transparent=True, comet=True
    )

    # Highlight the final shot/event with the team color
    pitch.arrows(
        xstart=final_event_df['x'], ystart=final_event_df['y'],
        xend=final_event_df['end_x'], yend=final_event_df['end_y'],
        ax=ax, color=team_color, zorder=5, headwidth=8, headlength=10, width=3
    )

    # Plot player nodes
    pitch.scatter(
        x=sequence_df['x'], y=sequence_df['y'],
        ax=ax, s=200, color=BG_COLOR, edgecolors=LINE_COLOR, zorder=4, lw=2
    )
    
    # Add a title with key info
    outcome = final_event_df['sequence_outcome_type'].iloc[0]
    pass_count = sequence_df['buildup_pass_count'].iloc[0]
    ax.set_title(f"Outcome: {outcome} ({pass_count} Passes)", color='white', fontsize=14, pad=10)