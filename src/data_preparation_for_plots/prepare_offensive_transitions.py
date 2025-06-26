# src/data_preparation_for_plots/prepare_offensive_transitions.py
import pandas as pd
import numpy as np
from typing import Dict, Optional

from ..data_processing.pass_processing import get_passes_df
from ..metrics.transition_metrics import find_buildup_after_possession_loss
from ..config import LOSS_TYPES


def create_offensive_transition_summary(df_sequences: pd.DataFrame, team_that_recovered_possession: str) -> Optional[pd.DataFrame]:
    """
    Create a summary table for offensive transition sequences after a team regains possession.
    """
    if df_sequences.empty or 'loss_sequence_id' not in df_sequences.columns:
        print(f"Warning: No sequences found for {team_that_recovered_possession}.")
        return None

    summary_df = df_sequences.drop_duplicates(subset=['loss_sequence_id'], keep='last').copy()
    required_cols = ['loss_zone', 'sequence_outcome_type', 'opponent_pass_count']
    for col in required_cols:
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

    print(f"\nDEBUG: Offensive transition summary for {team_that_recovered_possession}:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(final_table)
    
    final_table = final_table.rename(columns={'loss_zone': 'recovery_zone'})
    print(f"Michele : {final_table}")

    return final_table


def enrich_with_receiver_info(df_processed: pd.DataFrame, passes_df: pd.DataFrame) -> pd.DataFrame:
    df = df_processed.copy()
    if not passes_df.empty and 'receiver' in passes_df.columns and 'receiver_jersey_number' in passes_df.columns:
        df = pd.merge(
            df,
            passes_df[['eventId', 'receiver', 'receiver_jersey_number']],
            on='eventId', how='left'
        )
    else:
        print("Warning: passes_df or receiver info missing.")
        df['receiver'] = pd.NA
        df['receiver_jersey_number'] = pd.NA
    return df


def prepare_offensive_transitions_data(df_processed: pd.DataFrame, HTEAM_NAME: str, ATEAM_NAME: str) -> Dict[str, pd.DataFrame]:
    """
    Prepares offensive transition sequences and summary tables for both teams after regaining possession.
    """
    passes_df = get_passes_df(df_processed)
    df_with_receiver_info = enrich_with_receiver_info(df_processed, passes_df)

    df_home = find_buildup_after_possession_loss(
        df_with_receiver_info, ATEAM_NAME, possession_loss_types=LOSS_TYPES,
        metric_to_analyze='offensive_transitions')

    df_away = find_buildup_after_possession_loss(
        df_with_receiver_info, HTEAM_NAME, possession_loss_types=LOSS_TYPES,
        metric_to_analyze='offensive_transitions')

    summary_home = create_offensive_transition_summary(df_home, ATEAM_NAME)
    summary_away = create_offensive_transition_summary(df_away, HTEAM_NAME)

    df_home = df_home.rename(columns={"loss_zone": "recovery_zone"})
    df_away = df_away.rename(columns={"loss_zone": "recovery_zone"})

    return {
        'df_home_transitions': df_home,
        'df_away_transitions': df_away,
        'home_transition_summary': summary_home,
        'away_transition_summary': summary_away
    }
