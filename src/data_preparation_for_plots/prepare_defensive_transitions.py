# src/data_preparation_for_plots/prepare_opponent_buildup.py
import pandas as pd
import numpy as np
from typing import Dict, Optional

from ..data_processing.pass_processing import get_passes_df
from ..metrics.transition_metrics import find_buildup_after_possession_loss
from ..config import LOSS_TYPES  # Move LOSS_TYPES to config.py for central management

# This module prepares opponent buildup data and summary tables for both teams after possession losses.
def create_buildup_summary(df_sequences_for_summary: pd.DataFrame, team_that_lost_possession_name: str) -> Optional[pd.DataFrame]:
    """
    Create a summary table for opponent buildup sequences after a team loses possession.
    Args:
        df_sequences_for_summary (pd.DataFrame): DataFrame containing sequences data.
        team_that_lost_possession_name (str): Name of the team that lost possession.
    """
    if df_sequences_for_summary.empty or 'loss_sequence_id' not in df_sequences_for_summary.columns:
        print(f"Warning: No sequence data to create summary table for {team_that_lost_possession_name} losses.")
        return None

    summary_df = df_sequences_for_summary.drop_duplicates(subset=['loss_sequence_id'], keep='last').copy()
    required_summary_cols = ['loss_zone', 'sequence_outcome_type', 'opponent_pass_count']
    for col in required_summary_cols:
        if col not in summary_df.columns:
            print(f"Warning: Column '{col}' missing. Adding default.")
            summary_df[col] = "Unknown" if col != 'opponent_pass_count' else np.nan

    summary_df['opponent_pass_count'] = pd.to_numeric(summary_df['opponent_pass_count'], errors='coerce')

    table_outcome_counts = summary_df.groupby(['loss_zone', 'sequence_outcome_type'], as_index=False) \
                                     .size().rename(columns={'size': 'count'})

    avg_passes = summary_df[summary_df['opponent_pass_count'].notna()] \
        .groupby(['loss_zone', 'sequence_outcome_type']) \
        .agg(avg_opp_passes_before_regain=('opponent_pass_count', 'mean')) \
        .round(1).reset_index()

    final_table = table_outcome_counts.merge(avg_passes, on=['loss_zone', 'sequence_outcome_type'], how='left')
    final_table = final_table.sort_values(by=['loss_zone', 'count'], ascending=[True, False])

    print(f"\nDEBUG: Summary table for {team_that_lost_possession_name}:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(final_table)

    return final_table


# This function enriches the processed DataFrame with receiver and jersey number information from the passes DataFrame.
def enrich_with_receiver_info(df_processed: pd.DataFrame, passes_df: pd.DataFrame) -> pd.DataFrame:
    """Attach receiver and jersey number info to df_processed.
    Args:
        df_processed (pd.DataFrame): Processed DataFrame with event data.
        passes_df (pd.DataFrame): DataFrame containing pass data.
    """
    df = df_processed.copy()
    if not passes_df.empty and 'receiver' in passes_df.columns and 'receiver_jersey_number' in passes_df.columns:
        df = pd.merge(
            df,
            passes_df[['eventId', 'receiver', 'receiver_jersey_number']],
            on='eventId',
            how='left'
        )
    else:
        print("Warning: passes_df or receiver info missing.")
        df['receiver'] = pd.NA
        df['receiver_jersey_number'] = pd.NA
    return df


# This function prepares the opponent buildup data and summary tables for both teams after possession losses.
def prepare_opponent_buildup_data(df_processed: pd.DataFrame, HTEAM_NAME: str, ATEAM_NAME: str) -> Dict[str, pd.DataFrame]:
    """
    Prepares buildup and summary tables for both teams after possession losses.
    Args:
        df_processed (pd.DataFrame): Processed DataFrame with event data.
        HTEAM_NAME (str): Name of the home team.
        ATEAM_NAME (str): Name of the away team.
    """
    passes_df = get_passes_df(df_processed)
    df_with_receiver_info = enrich_with_receiver_info(df_processed, passes_df)

    df_away = find_buildup_after_possession_loss(
        df_with_receiver_info, HTEAM_NAME, possession_loss_types=LOSS_TYPES)

    df_home = find_buildup_after_possession_loss(
        df_with_receiver_info, ATEAM_NAME, possession_loss_types=LOSS_TYPES)

    summary_away = create_buildup_summary(df_away, HTEAM_NAME)
    summary_home = create_buildup_summary(df_home, ATEAM_NAME)

    return {
        'df_away_buildup_after_home_loss': df_away,
        'df_home_buildup_after_away_loss': df_home,
        'home_loss_summary_table_data': summary_away,
        'away_loss_summary_table_data': summary_home
    }
