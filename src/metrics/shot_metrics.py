import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from src.metrics.shot_classification import classify_shots, shot_stats_for_team
from src.metrics.shot_geometry import GOAL_X, GOAL_Y, shot_distance_m


def calculate_shot_stats(df_processed, hteamName, ateamName, hxG, axG, hxGOT, axGOT,
                         pitch_length_meters=105.0, pitch_width_meters=68.0):
    """
    Calculate shared canonical shot statistics for both teams.

    ``Attempt Saved`` is disambiguated through Opta qualifiers by
    ``shot_classification.classify_shots``. In particular, qualifier 82
    (``Blocked``) is a blocked shot and therefore not a shot on target.
    """
    logger.debug("Calculating canonical shot statistics (all shots target X=100)...")
    shots_df = classify_shots(df_processed)

    if shots_df.empty:
        logger.debug("No shot events found.")
        return pd.DataFrame(), {}, {}

    if 'x' not in shots_df.columns or 'y' not in shots_df.columns:
        logger.warning("Shot coordinate columns ('x', 'y') not found. Cannot calculate distance.")
        shots_df['shot_distance_m'] = np.nan
    else:
        x_opta = shots_df['x'].fillna(GOAL_X)
        y_opta = shots_df['y'].fillna(GOAL_Y)
        shots_df['shot_distance_m'] = shot_distance_m(
            x_opta,
            y_opta,
            pitch_length_m=pitch_length_meters,
            pitch_width_m=pitch_width_meters,
        )
        logger.debug("Calculated shot distances in meters (to X=100 goal).")

    def _stats(team_name, xg, xgot):
        team_shots = shots_df[shots_df['team_name'] == team_name]
        stats = shot_stats_for_team(shots_df, team_name)
        counted = team_shots.loc[
            team_shots['shot_counts_as_shot'].fillna(False)
        ]
        stats.update({
            'xg': xg,
            'xgot': xgot,
            'avg_shot_distance': (
                counted['shot_distance_m'].mean()
                if 'shot_distance_m' in counted.columns and not counted.empty
                else np.nan
            ),
            'xg_per_shot': (
                xg / stats['total_shots']
                if stats['total_shots'] > 0 and xg is not None
                else 0.0
            ),
        })
        return stats

    home_stats = _stats(hteamName, hxG, hxGOT)
    away_stats = _stats(ateamName, axG, axGOT)

    logger.info(
        "Shot classification complete: %s SoT=%s blocked=%s; %s SoT=%s blocked=%s",
        hteamName,
        home_stats['shots_on_target'],
        home_stats['blocked_shots'],
        ateamName,
        away_stats['shots_on_target'],
        away_stats['blocked_shots'],
    )
    return shots_df, home_stats, away_stats
