# src/data_preparation_for_plots/prepare_buildup_phase.py
import pandas as pd
import numpy as np
from typing import Dict, Optional

from ..data_processing.pass_processing import get_passes_df
from ..metrics.buildup_metrics import find_buildup_sequences
from ..config import TRIGGER_TYPES_FOR_BUILDUPS


def create_buildup_phase_summary(df_sequences: pd.DataFrame, team_building_up: str) -> Optional[pd.DataFrame]:
    """
    Generates a summary DataFrame for buildup phases for a given team.
    Args:
        df_sequences (pd.DataFrame): DataFrame containing buildup sequences.
        team_building_up (str): Name of the team building up a.

    Returns:
        Optional[pd.DataFrame]: Summary DataFrame with counts and average opponent passes before regain, or None if input is invalid.
    """
    if df_sequences.empty or 'trigger_sequence_id' not in df_sequences.columns:
        print(f"Warning: No sequences found for {team_building_up}.")
        return None

    summary_df = df_sequences.drop_duplicates(subset=['trigger_sequence_id'], keep='last').copy()
    required_cols = ['trigger_zone', 'sequence_outcome_type', 'buildup_pass_count']
    for col in required_cols:
        if col not in summary_df.columns:
            print(f"Warning: Column '{col}' missing. Adding default.")
            summary_df[col] = "Unknown" if col != 'buildup_pass_count' else np.nan

    summary_df['buildup_pass_count'] = pd.to_numeric(summary_df['buildup_pass_count'], errors='coerce')

    table_outcome_counts = summary_df.groupby(['trigger_zone', 'sequence_outcome_type'], as_index=False) \
                                     .size().rename(columns={'size': 'count'})

    avg_passes = summary_df[summary_df['buildup_pass_count'].notna()] \
        .groupby(['trigger_zone', 'sequence_outcome_type']) \
        .agg(avg_opp_passes_before_regain=('buildup_pass_count', 'mean')) \
        .round(1).reset_index()

    final_table = table_outcome_counts.merge(avg_passes, on=['trigger_zone', 'sequence_outcome_type'], how='left')
    final_table = final_table.sort_values(by=['trigger_zone', 'count'], ascending=[True, False])

    print(f"\nDEBUG: Offensive transition summary for {team_building_up}:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(final_table)
    
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


def prepare_offensive_buildups_data(df_processed: pd.DataFrame, HTEAM_NAME: str, ATEAM_NAME: str, metric_to_analyze: str) -> Dict[str, pd.DataFrame]:
    """
    Prepares offensive transition sequences and summary tables for both teams after regaining possession.
    """
    passes_df = get_passes_df(df_processed)
    df_with_receiver_info = enrich_with_receiver_info(df_processed, passes_df)
    print(f"Michele : Looking for {TRIGGER_TYPES_FOR_BUILDUPS}")

    df_home = find_buildup_sequences(
        df_with_receiver_info, HTEAM_NAME, ATEAM_NAME, metric_to_analyze, triggers_buildups=TRIGGER_TYPES_FOR_BUILDUPS
        )

    df_away = find_buildup_sequences(
        df_with_receiver_info, ATEAM_NAME, HTEAM_NAME, metric_to_analyze, triggers_buildups=TRIGGER_TYPES_FOR_BUILDUPS
        )

    summary_home = create_buildup_phase_summary(df_home, ATEAM_NAME)
    summary_away = create_buildup_phase_summary(df_away, HTEAM_NAME)

    print(f"Michele df_home_buildups: {df_home}")

    return {
        'df_home_buildups': df_home,
        'df_away_buildups': df_away,
        'home_buildups_summary': summary_home,
        'away_buildups_summary': summary_away
    }
