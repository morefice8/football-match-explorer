# src/metrics/pass_metrics.py
import pandas as pd

# --- Pass Network Data Calculation ---
# This function calculates the average player locations and pass counts between players for a specific team.
def calculate_pass_network_data(passes_df, team_name):
    """
    Calculates average player locations and pass counts between players for a specific team.

    Args:
        passes_df (pd.DataFrame): DataFrame containing pass events (output of get_passes_df).
        team_name (str): The name of the team to analyze.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Passes between players with counts and start/end locations.
            - pd.DataFrame: Average location and pass count for each player.
            Returns (empty DataFrame, empty DataFrame) if no passes for the team.
    """
    print(f"Calculating pass network data for {team_name}...")
    # Filter the passes DataFrame for the specified team
    team_passes_df = passes_df[passes_df['team_name'] == team_name].copy()

    if team_passes_df.empty:
        print(f"Warning: No passes found for team {team_name}.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Calculate average locations and counts per player ---
    # Group by player name and aggregate median coordinates and count passes
    # Also get the first jersey number associated with the player
    player_agg = team_passes_df.groupby('playerName').agg(
        pass_avg_x=('x', 'median'),
        pass_avg_y=('y', 'median'),
        pass_count=('id', 'count'), # Count passes made by the player
        jersey_number=('Mapped Jersey Number', 'first') # Get jersey number
    )
    # Reset index to make 'playerName' a column again
    average_locs_and_count_df = player_agg.reset_index()
    print(f"Calculated average locations for {len(average_locs_and_count_df)} players.")


    # --- Calculate passes between player pairs ---
    # Select relevant columns for pairing
    # Ensure 'receiver' column exists and handle potential NaN values before processing
    if 'receiver' not in team_passes_df.columns:
         print("Error: 'receiver' column not found in passes data. Cannot calculate pairs.")
         return pd.DataFrame(), average_locs_and_count_df # Return what we have so far

    # Drop rows where receiver is NaN as they cannot form a pair
    team_passes_pairs_df = team_passes_df.dropna(subset=['playerName', 'receiver']).copy()

    # Convert player names to string just in case
    team_passes_pairs_df['playerName'] = team_passes_pairs_df['playerName'].astype(str)
    team_passes_pairs_df['receiver'] = team_passes_pairs_df['receiver'].astype(str)

    # Create a unique, sorted tuple for each passer-receiver pair to group A->B and B->A together
    team_passes_pairs_df['player_pair'] = team_passes_pairs_df.apply(
        lambda row: tuple(sorted((row['playerName'], row['receiver']))), axis=1
    )

    # Count the number of passes for each unique pair
    passes_between_df = team_passes_pairs_df.groupby('player_pair').size().reset_index(name='pass_count')

    # Split the pair tuple back into two columns
    passes_between_df[['player1', 'player2']] = pd.DataFrame(passes_between_df['player_pair'].tolist(), index=passes_between_df.index)
    # Drop the temporary pair column
    passes_between_df = passes_between_df.drop(columns=['player_pair'])

    # --- Merge average locations onto the pairs data ---
    # Set 'playerName' as index in average locations df for easier merging
    average_locs_indexed = average_locs_and_count_df.set_index('playerName')

    # Merge based on player1 to get their location
    passes_between_df = passes_between_df.merge(
        average_locs_indexed[['pass_avg_x', 'pass_avg_y']],
        left_on='player1',
        right_index=True,
        how='left'
    )
    # Merge based on player2 to get their location (end location of the line)
    passes_between_df = passes_between_df.merge(
        average_locs_indexed[['pass_avg_x', 'pass_avg_y']],
        left_on='player2',
        right_index=True,
        how='left',
        suffixes=('', '_end') # Add suffix to distinguish player2's coords
    )

    # Rename columns for clarity if needed (e.g., pass_avg_x_end)
    # The suffixes already handle this: 'pass_avg_x', 'pass_avg_y', 'pass_avg_x_end', 'pass_avg_y_end'

    print(f"Calculated {len(passes_between_df)} links between players.")

    # Return both the pair data and the individual player average locations
    return passes_between_df, average_locs_and_count_df

# --- Progressive Passes ---
# This function identifies progressive passes based on Opta definitions and calculates counts and percentages per vertical third of the pitch.
def analyze_progressive_passes(df_input,
                               pitch_length_meters=105.0,
                               exclude_qualifiers=None,
                               return_ids_only=False):
    """
    Identifies progressive passes based on distance gained towards the opponent's goal,
    converting meter-based thresholds to Opta coordinate units. Allows dynamic exclusion
    of passes based ONLY on the specified qualifier columns provided.

    Definition inspired by WyScout:
    Progressive if distance between start and next touch is:
    - >= 30m closer to goal if start/end in own half.
    - >= 15m closer to goal if start in own half, end in opp half.
    - >= 10m closer to goal if start/end in opp half.

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame.
        pitch_length_meters (float, optional): The standard length of the pitch in meters
                                               used for coordinate conversion. Defaults to 105.0.
        exclude_qualifiers (list, optional): A list of column names (strings) representing
                                             qualifiers to exclude from progressive pass
                                             consideration (e.g., ['cross', 'Launch']).
                                             If None or empty, NO qualifiers are excluded.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame containing only the progressive passes.
            - dict: Dictionary with overall progressive pass counts per starting zone
                    {'total': count, 'left': count, 'mid': count, 'right': count}.
            Returns (empty DataFrame, empty dict) if analysis fails or no passes found.
    """
    print("Analyzing progressive passes...")

    if df_input.empty:
        return (pd.DataFrame(), {}) if not return_ids_only else []

    # La funzione ora si aspetta un df che può essere o l'intero df_processed o un df di passaggi.
    # Assicuriamoci di lavorare solo sui passaggi riusciti.
    pass_filter = (df_input['type_name'] == 'Pass') & (df_input['outcome'] == 'Successful')
    df_passes = df_input[pass_filter].copy()

    if df_passes.empty:
        return (pd.DataFrame(), {}) if not return_ids_only else []
        
    # --- Gestione Esclusioni ---
    if exclude_qualifiers:
        for qualifier_name in exclude_qualifiers:
            if qualifier_name in df_passes.columns:
                is_present = df_passes[qualifier_name].fillna(0).isin([1, '1', True])
                df_passes = df_passes[~is_present]
    
    if df_passes.empty:
        return (pd.DataFrame(), {}) if not return_ids_only else []

    # --- Calcolo Criteri di Progressione (logica invariata) ---
    opta_units_per_meter_x = 100.0 / pitch_length_meters
    prog_thresh_30m_opta = 30.0 * opta_units_per_meter_x
    prog_thresh_15m_opta = 15.0 * opta_units_per_meter_x
    prog_thresh_10m_opta = 10.0 * opta_units_per_meter_x

    x = df_passes['x'].fillna(50)
    end_x = df_passes['end_x'].fillna(x)
    distance_gained = end_x - x
    
    progression_criteria = (
        (distance_gained > 0) & 
        (((x <= 50) & (end_x <= 50) & (distance_gained >= prog_thresh_30m_opta)) |
         ((x <= 50) & (end_x > 50)  & (distance_gained >= prog_thresh_15m_opta)) |
         ((x > 50)  & (end_x > 50)  & (distance_gained >= prog_thresh_10m_opta)))
    )
    
    df_prog_passes = df_passes.loc[progression_criteria]

    # --- Output Condizionale ---
    if return_ids_only:
        return df_prog_passes['id'].tolist()
    else:
        # Calcola le statistiche di zona solo se richiesto l'output completo
        total_count = len(df_prog_passes)
        y_start = df_prog_passes['y'].fillna(50)
        right_prog = (y_start < 33.33).sum()
        mid_prog = ((y_start >= 33.33) & (y_start < 66.67)).sum()
        left_prog = (y_start >= 66.67).sum()
        zone_counts = {'total': total_count, 'left': left_prog, 'mid': mid_prog, 'right': right_prog}
        
        return df_prog_passes, zone_counts

    # # --- Handle Exclusions: Ensure it's a list, even if empty ---
    # if exclude_qualifiers is None:
    #     exclude_qualifiers = [] # Treat None as an empty list - no exclusions
    #     print("  No specific exclusions provided.")
    # elif not exclude_qualifiers: # Checks if the list is empty
    #      print("  Empty exclusion list provided - no qualifiers will be excluded.")
    # else:
    #      print(f"  Attempting to exclude passes with qualifiers: {exclude_qualifiers}")


    # # --- Define Pitch Conversion Factors ---
    # if pitch_length_meters <= 0:
    #     print("Error: Pitch length must be positive.")
    #     return pd.DataFrame(), {}
    # opta_units_per_meter_x = 100.0 / pitch_length_meters
    # prog_thresh_30m_opta = 30.0 * opta_units_per_meter_x
    # prog_thresh_15m_opta = 15.0 * opta_units_per_meter_x
    # prog_thresh_10m_opta = 10.0 * opta_units_per_meter_x
    # print(f"  Using thresholds (Opta Units): OwnHalf={prog_thresh_30m_opta:.2f}, DiffHalf={prog_thresh_15m_opta:.2f}, OppHalf={prog_thresh_10m_opta:.2f}")


    # # --- Check Required Columns ---
    # required_cols = ['team_name', 'type_name', 'outcome', 'x', 'y', 'end_x', 'end_y']
    # if not all(col in df_processed.columns for col in required_cols):
    #     missing = set(required_cols) - set(df_processed.columns)
    #     print(f"Error: Missing required columns for progressive pass analysis: {missing}")
    #     return pd.DataFrame(), {}

    # # --- Base Filtering (Successful Passes) ---
    # base_filter = (
    #     (df_processed['type_name'] == 'Pass') &
    #     (df_processed['outcome'] == 'Successful')
    # )

    # # --- Apply Dynamic Exclusions (Only if exclude_qualifiers is not empty) ---
    # valid_exclusions_applied = []
    # if exclude_qualifiers: # Only loop if the list is not empty
    #     for qualifier_name in exclude_qualifiers:
    #         if qualifier_name in df_processed.columns:
    #             # Check if the qualifier is present (value is 1, '1', or True)
    #             # Fill NaN with 0 (or False) so they are NOT excluded
    #             is_present = df_processed[qualifier_name].fillna(0).isin([1, '1', True])
    #             # Add filter to exclude rows where the qualifier IS present
    #             base_filter &= (~is_present)
    #             valid_exclusions_applied.append(qualifier_name)
    #         else:
    #             print(f"    - Warning: Qualifier column '{qualifier_name}' provided for exclusion not found in DataFrame.")

    #     if valid_exclusions_applied:
    #         print(f"  Successfully applied exclusion filters for: {valid_exclusions_applied}")


    # # Apply base filters (including any dynamic exclusions applied)
    # df_passes_filtered = df_processed[base_filter].copy()

    # if df_passes_filtered.empty:
    #     print("No successful passes matching the criteria (after any specified exclusions) found.")
    #     return pd.DataFrame(), {'total': 0, 'left': 0, 'mid': 0, 'right': 0}

    # # --- Calculate Progression Criteria using Converted Thresholds ---
    # # (This part remains the same as the previous version)
    # x = df_passes_filtered['x'].fillna(50)
    # y = df_passes_filtered['y'].fillna(50)
    # end_x = df_passes_filtered['end_x'].fillna(x)
    # distance_gained_opta = end_x - x

    # progression_criteria = (
    #    (distance_gained_opta > 0) &
    #    (
    #        ((x <= 50) & (end_x <= 50) & (distance_gained_opta >= prog_thresh_30m_opta)) |
    #        ((x <= 50) & (end_x > 50) & (distance_gained_opta >= prog_thresh_15m_opta)) |
    #        ((x > 50) & (end_x > 50) & (distance_gained_opta >= prog_thresh_10m_opta))
    #    )
    # )

    # # Apply the progression criteria filter
    # df_prog_passes = df_passes_filtered.loc[progression_criteria].copy()
    # total_prog_count = len(df_prog_passes)

    # print(f"Found {total_prog_count} progressive passes overall (after exclusions and progression checks).")

    # if total_prog_count == 0:
    #     return pd.DataFrame(), {'total': 0, 'left': 0, 'mid': 0, 'right': 0}

    # # --- Calculate Starting Zone Counts ---
    # # (This part remains the same as the previous version)
    # if 'y' not in df_prog_passes.columns:
    #     print("Error: 'y' column missing, cannot calculate zones for progressive passes.")
    #     return df_prog_passes, {'total': total_prog_count, 'left': 0, 'mid': 0, 'right': 0}

    # y_start = df_prog_passes['y'].fillna(50)
    # right_prog = (y_start < 33.33).sum()
    # mid_prog = ((y_start >= 33.33) & (y_start < 66.67)).sum()
    # left_prog = (y_start >= 66.67).sum()

    # zone_counts = { 'total': total_prog_count, 'left': left_prog, 'mid': mid_prog, 'right': right_prog }
    # print(f"  Progressive Pass Start Zones: Left={left_prog}, Mid={mid_prog}, Right={right_prog}")

    # return df_prog_passes, zone_counts

# --- Final Third Passes (Zone 14 / Half-Spaces) ---
# This function identifies successful passes ending in Zone 14 or Left/Right Half-Spaces for a specific team.
def analyze_final_third_passes(passes_df_team_successful):
    """
    Identifies successful passes ending in Zone 14 or Left/Right Half-Spaces
    for a specific team. Excludes passes starting very close to the corner flag.

    Args:
        passes_df_team_successful (pd.DataFrame): DataFrame containing ONLY successful passes
                                                 for the team being analyzed.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame with Zone 14 passes.
            - pd.DataFrame: DataFrame with Left Half-Space passes.
            - pd.DataFrame: DataFrame with Right Half-Space passes.
            - dict: Counts {'zone14': count, 'hs_left': count, 'hs_right': count,
                           'hs_total': count, 'total_final_third': count}.
            Returns (empty df, empty df, empty df, default counts dict) if no relevant passes.
    """
    print(f"Analyzing Zone 14 / Half-Space passes...")

    default_counts = {'zone14': 0, 'hs_left': 0, 'hs_right': 0, 'hs_total': 0, 'total_final_third': 0}
    empty_df = pd.DataFrame()

    # Ensure required coordinate columns exist
    required_cols = ['x', 'y', 'end_x', 'end_y']
    if not all(col in passes_df_team_successful.columns for col in required_cols):
        missing = set(required_cols) - set(passes_df_team_successful.columns)
        print(f"Error: Missing required columns for final third analysis: {missing}")
        return empty_df, empty_df, empty_df, default_counts

    # Filter out passes starting too close to corner flag (x > ~99 is likely corner)
    df_filtered = passes_df_team_successful[passes_df_team_successful['x'] < 99.5].copy()

    if df_filtered.empty:
        print("Info: No relevant successful passes found after filtering.")
        return empty_df, empty_df, empty_df, default_counts

    # Define Zone Boundaries (Opta Coordinates 0-100)
    zone14_x_min, zone14_x_max = 66.67, 82.0
    zone14_y_min, zone14_y_max = 100/3, 200/3
    halfspace_x_min = 66.67
    rhs_y_min, rhs_y_max = 100/6, 100/3
    lhs_y_min, lhs_y_max = 200/3, 500/6 # Using original 200/3 to 500/6 (~66.7 to 83.3)

    # --- Classify Passes based on END coordinates ---
    # Zone 14 Passes
    zone14_mask = (
        (df_filtered['end_x'] >= zone14_x_min) & (df_filtered['end_x'] <= zone14_x_max) &
        (df_filtered['end_y'] >= zone14_y_min) & (df_filtered['end_y'] <= zone14_y_max)
    )
    df_zone14 = df_filtered[zone14_mask].copy()
    z14_count = len(df_zone14)

    # Right Half-Space Passes (Low Y values)
    rhs_mask = (
        (df_filtered['end_x'] >= halfspace_x_min) &
        (df_filtered['end_y'] >= rhs_y_min) & (df_filtered['end_y'] < rhs_y_max)
    )
    df_rhs = df_filtered[rhs_mask & (~zone14_mask)].copy() # Exclude Zone 14 overlap
    rhs_count = len(df_rhs)

    # Left Half-Space Passes (High Y values)
    lhs_mask = (
        (df_filtered['end_x'] >= halfspace_x_min) &
        (df_filtered['end_y'] >= lhs_y_min) & (df_filtered['end_y'] <= lhs_y_max)
    )
    df_lhs = df_filtered[lhs_mask & (~zone14_mask)].copy() # Exclude Zone 14 overlap
    lhs_count = len(df_lhs)

    # Calculate totals
    hs_total_count = lhs_count + rhs_count
    total_final_third_count = z14_count + hs_total_count

    # Store counts
    zone_stats_dict = {
        'zone14': z14_count,
        'hs_left': lhs_count,
        'hs_right': rhs_count,
        'hs_total': hs_total_count,
        'total_final_third': total_final_third_count
    }
    print(f"Found: Zone 14={z14_count}, L HS={lhs_count}, R HS={rhs_count} (Total FT={total_final_third_count})")

    return df_zone14, df_lhs, df_rhs, zone_stats_dict

# --- Analyze Chance Creation Passes ---
# This function identifies chance-creating passes (Key Passes + Assists) based on specific qualifier values.
def analyze_chance_creation(df_processed, hteamName, ateamName,
                            assist_qualifier_col='Assist', # Column name for assist qualifiers
                            key_pass_values=[13, 14, 15], # Values indicating key pass type
                            assist_values=[16]             # Value(s) indicating assist type
                           ):
    """
    Identifies chance-creating passes (Key Passes + Assists) based on specific
    qualifier values in the processed DataFrame.

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame.
        hteamName (str): Home team name.
        ateamName (str): Away team name.
        assist_qualifier_col (str): The name of the column in df_processed that
                                    contains the qualifier values distinguishing
                                    key passes and assists.
        key_pass_values (list): List of values in assist_qualifier_col that signify a Key Pass.
        assist_values (list): List of values in assist_qualifier_col that signify an Assist.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: DataFrame of home team chance-creating passes with flags.
            - pd.DataFrame: DataFrame of away team chance-creating passes with flags.
            Returns (empty DF, empty DF) if analysis fails or no passes found.
    """
    print("Analyzing chance creation passes...")

    # --- Check required columns ---
    required_cols = ['team_name', 'type_name', 'outcome', 'x', 'y', 'end_x', 'end_y']
    if assist_qualifier_col not in df_processed.columns:
        print(f"Error: Assist qualifier column '{assist_qualifier_col}' not found. Cannot analyze chances.")
        return pd.DataFrame(), pd.DataFrame()
    required_cols.append(assist_qualifier_col) # Add it for the main check

    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        print(f"Error: Missing required columns for chance creation analysis: {missing}")
        return pd.DataFrame(), pd.DataFrame()

    # --- Filter for successful passes first ---
    df_succ_passes = df_processed[
        (df_processed['type_name'] == 'Pass') &
        (df_processed['outcome'] == 'Successful') &
        # Ensure end coordinates are valid for plotting arrows
        (df_processed['end_x'].notna()) &
        (df_processed['end_y'].notna()) &
        (df_processed['end_x'] >= 0) & # Basic check for within-pitch end points
        (df_processed['end_y'] >= 0) &
        (df_processed['end_x'] <= 100) &
        (df_processed['end_y'] <= 100)
    ].copy()

    if df_succ_passes.empty:
         print("No successful passes with valid end coordinates found.")
         return pd.DataFrame(), pd.DataFrame()

    # --- Identify Key Passes and Assists ---
    # Convert qualifier column to numeric, coercing errors (handles strings/numbers)
    assist_qual_numeric = pd.to_numeric(df_succ_passes[assist_qualifier_col], errors='coerce')

    # Create boolean masks based on the provided values
    key_pass_filter = assist_qual_numeric.isin(key_pass_values)
    assist_filter = assist_qual_numeric.isin(assist_values) # Use isin for lists

    # Combine filters: must be either a key pass or an assist
    chance_creation_filter = key_pass_filter | assist_filter

    # Apply the filter to get only chance-creating passes
    df_chances = df_succ_passes[chance_creation_filter].copy()

    if df_chances.empty:
         print("No passes matched the Key Pass or Assist criteria.")
         return pd.DataFrame(), pd.DataFrame()

    # --- Add boolean flags for easier plotting distinction ---
    # Apply masks again to the filtered df_chances to ensure correct index alignment
    df_chances['is_key_pass'] = key_pass_filter.reindex(df_chances.index).fillna(False)
    df_chances['is_assist'] = assist_filter.reindex(df_chances.index).fillna(False)

    print(f"Found {len(df_chances)} chance-creating passes.")

    # --- Split by team ---
    df_chances_home = df_chances[df_chances['team_name'] == hteamName].copy()
    df_chances_away = df_chances[df_chances['team_name'] == ateamName].copy()

    print(f"  Home Chances: {len(df_chances_home)}, Away Chances: {len(df_chances_away)}")

    return df_chances_home, df_chances_away