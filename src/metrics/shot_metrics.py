# src/metrics/shot_metrics.py
import pandas as pd
import numpy as np

def calculate_shot_stats(df_processed, hteamName, ateamName, hxG, axG, hxGOT, axGOT,
                         pitch_length_meters=105.0, pitch_width_meters=68.0):
    """
    Calculates various shot statistics for both teams, including avg distance in meters.
    ASSUMES Opta coordinates in df_processed are normalized so ALL shots target X=100.
    Calculates distance to the goal at X=100, Y=50 for all shots.
    """
    print("Calculating shot statistics (assuming all shots target X=100)...")
    shot_types = ['Miss', 'Attempt Saved', 'Post', 'Goal']
    # Ensure we work with a copy
    shots_df = df_processed[df_processed['type_name'].isin(shot_types)].copy()

    if shots_df.empty:
        print("No shot events found.")
        return pd.DataFrame(), {}, {}

    # --- Define SINGLE Target Goal Center (Opta Coords) ---
    # Based on user clarification, all shots are relative to attacking this goal
    target_goal_center_x, target_goal_center_y = 100, 50

    # --- Calculate Distance to the TARGET Goal (X=100) in METERS ---
    distances_m = []
    if 'x' not in shots_df.columns or 'y' not in shots_df.columns:
        print("Warning: Shot coordinate columns ('x', 'y') not found. Cannot calculate distance.")
        shots_df['shot_distance_m'] = np.nan
    else:
        # Vectorized calculation is much faster than iteration
        x_opta = shots_df['x'].fillna(target_goal_center_x) # Fill NaN with goal X for 0 distance
        y_opta = shots_df['y'].fillna(target_goal_center_y) # Fill NaN with goal Y for 0 distance

        # Calculate distance in Opta units to the target goal (X=100)
        dx_opta = target_goal_center_x - x_opta
        dy_opta = target_goal_center_y - y_opta
        dist_opta = np.sqrt(dx_opta**2 + dy_opta**2)

        # Convert Opta distance to Meters
        dist_meters = dist_opta * (pitch_length_meters / 100.0)

        # Assign the calculated meter distance as a new column
        shots_df['shot_distance_m'] = dist_meters
        print(f"  Calculated shot distances in meters (to X=100 goal).")

    # --- Aggregate Stats Per Team ---
    # (Aggregation logic remains the same, using the new 'shot_distance_m')
    home_shots = shots_df[shots_df['team_name'] == hteamName]
    away_shots = shots_df[shots_df['team_name'] == ateamName]

    home_stats = {
        'goals': (home_shots['type_name'] == 'Goal').sum(),
        'total_shots': len(home_shots),
        'shots_on_target': ((home_shots['type_name'] == 'Goal') | (home_shots['type_name'] == 'Attempt Saved')).sum(),
        'xg': hxG, 'xgot': hxGOT,
        'avg_shot_distance': home_shots['shot_distance_m'].mean() if 'shot_distance_m' in home_shots.columns else np.nan,
        'xg_per_shot': hxG / len(home_shots) if len(home_shots) > 0 and hxG is not None else 0.0
    }

    away_stats = {
        'goals': (away_shots['type_name'] == 'Goal').sum(),
        'total_shots': len(away_shots),
        'shots_on_target': ((away_shots['type_name'] == 'Goal') | (away_shots['type_name'] == 'Attempt Saved')).sum(),
        'xg': axG, 'xgot': axGOT,
        'avg_shot_distance': away_shots['shot_distance_m'].mean() if 'shot_distance_m' in away_shots.columns else np.nan,
        'xg_per_shot': axG / len(away_shots) if len(away_shots) > 0 and axG is not None else 0.0
    }

    print("Finished calculating shot statistics.")
    return shots_df, home_stats, away_stats