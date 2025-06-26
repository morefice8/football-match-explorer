# src/metrics/turnover_metrics.py
import pandas as pd
import numpy as np

# This function identifies turnovers within the radius using Opta units for filtering,
# but returns the DFs with ORIGINAL Opta coordinates for plotting flexibility.
def calculate_high_turnovers(df_processed, hteamName, ateamName,
                             radius_meters=40.0,
                             pitch_length_meters=105.0):
    """
    Identifies high turnovers (recoveries within a defined radius of opponent's goal).
    Filters using Opta coordinates derived from the meter radius, but returns
    DataFrames containing the original Opta coordinates of the turnovers found.

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame.
        hteamName (str): Home team name.
        ateamName (str): Away team name.
        radius_meters (float, optional): Radius in meters from goal center. Defaults to 40.0.
        pitch_length_meters (float, optional): Pitch length in meters for scaling. Defaults to 105.0.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame of home team high turnovers (with Opta coords).
            - pd.DataFrame: DataFrame of away team high turnovers (with Opta coords).
            - int: Count of home team high turnovers.
            - int: Count of away team high turnovers.
            # Removed radius_opta return, as it's recalculated in plotting if needed
            # or meter radius is used directly
    """
    print(f"Calculating high turnovers (radius: {radius_meters}m)...")

    # --- Calculate Radius in Opta Units FOR FILTERING ---
    if pitch_length_meters <= 0:
        print("Error: Pitch length must be positive.")
        return pd.DataFrame(), pd.DataFrame(), 0, 0
    radius_opta = radius_meters * (100.0 / pitch_length_meters)
    print(f"  Filtering radius in Opta units: {radius_opta:.2f}")

    # --- Filter for Recovery Events ---
    recovery_types = ['Ball recovery', 'Interception']
    required_cols = ['team_name', 'type_name', 'x', 'y']
    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        print(f"Error: Missing required columns for turnover analysis: {missing}")
        return pd.DataFrame(), pd.DataFrame(), 0, 0

    recoveries_df = df_processed[df_processed['type_name'].isin(recovery_types)].copy()

    if recoveries_df.empty:
        print("No recovery events found.")
        return pd.DataFrame(), pd.DataFrame(), 0, 0

    # Define opponent goal centers in OPTA coordinates for filtering
    home_goal_center_x, home_goal_center_y = 100, 50
    away_goal_center_x, away_goal_center_y = 0, 50

    # --- Calculate Home Team High Turnovers ---
    home_recoveries = recoveries_df[recoveries_df['team_name'] == hteamName].copy()
    if not home_recoveries.empty:
        dist_to_away_goal = np.sqrt(
            (home_recoveries['x'].fillna(away_goal_center_x) - away_goal_center_x)**2 +
            (home_recoveries['y'].fillna(away_goal_center_y) - away_goal_center_y)**2
        )
        # Filter using radius_opta
        home_high_to_df = home_recoveries[dist_to_away_goal <= radius_opta].copy() # Use .copy()
        hto_count = len(home_high_to_df)
    else:
        home_high_to_df = pd.DataFrame()
        hto_count = 0

    # --- Calculate Away Team High Turnovers ---
    away_recoveries = recoveries_df[recoveries_df['team_name'] == ateamName].copy()
    if not away_recoveries.empty:
        dist_to_home_goal = np.sqrt(
            (away_recoveries['x'].fillna(home_goal_center_x) - home_goal_center_x)**2 +
            (away_recoveries['y'].fillna(home_goal_center_y) - home_goal_center_y)**2
        )
        # Filter using radius_opta
        away_high_to_df = away_recoveries[dist_to_home_goal <= radius_opta].copy() # Use .copy()
        ato_count = len(away_high_to_df)
    else:
        away_high_to_df = pd.DataFrame()
        ato_count = 0

    print(f"  Found {hto_count} high turnovers for {hteamName}")
    print(f"  Found {ato_count} high turnovers for {ateamName}")

    # Return the filtered DFs (still with Opta coords) and counts
    return home_high_to_df, away_high_to_df, hto_count, ato_count