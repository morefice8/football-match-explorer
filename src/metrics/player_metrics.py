# src/metrics/player_metrics.py
import pandas as pd
import numpy as np
from .pass_metrics import analyze_progressive_passes # Assuming correct relative import

def calculate_player_stats(df_processed, assist_qualifier_col='Assist',
                           key_pass_values=[13, 14, 15], assist_values=[16],
                           prog_pass_exclusions=None):
    """
    Calculates a variety of statistics aggregated per player.
    """
    print("Calculating aggregated player statistics...")
    if df_processed.empty:
        print("Warning: Input DataFrame is empty.")
        return pd.DataFrame()

    # --- Pre-calculate Flags ---
    # (Keep the logic to ensure is_key_pass and is_assist flags exist)
    if 'is_key_pass' not in df_processed.columns or 'is_assist' not in df_processed.columns:
        print("Info: 'is_key_pass'/'is_assist' flags not found, calculating them now...")
        if assist_qualifier_col not in df_processed.columns:
            print(f"Error: Required qualifier column '{assist_qualifier_col}' not found.")
            return pd.DataFrame()
        assist_qual_numeric = pd.to_numeric(df_processed[assist_qualifier_col], errors='coerce')
        # Ensure flags are only True if it's also a Pass event
        df_processed['is_key_pass'] = assist_qual_numeric.isin(key_pass_values) & (df_processed['type_name'] == 'Pass')
        df_processed['is_assist'] = assist_qual_numeric.isin(assist_values) & (df_processed['type_name'] == 'Pass')
    else:
        df_processed['is_key_pass'] = df_processed['is_key_pass'].fillna(False).astype(bool)
        df_processed['is_assist'] = df_processed['is_assist'].fillna(False).astype(bool)

    # Define shot types
    shot_types = ['Miss', 'Attempt Saved', 'Post', 'Goal']

    # --- Group by Player ---
    grouped_player = df_processed.groupby('playerName')

    # --- Calculate Stats using apply or separate aggregations ---
    print("  Aggregating player stats...")
    player_stats_list = []

    required_cols_check = ['type_name', 'outcome', 'x', 'y', 'end_x', 'end_y', 'is_key_pass', 'is_assist']
    if not all(col in df_processed.columns for col in required_cols_check):
         print(f"Error: Missing one or more required columns for calculation: {set(required_cols_check) - set(df_processed.columns)}")
         return pd.DataFrame()

    for name, group in grouped_player:
        stats = {}
        stats['playerName'] = name # Keep player name

        # Shooting Sequence
        stats['Shots'] = (group['type_name'].isin(shot_types)).sum()
        stats['Shot Assists'] = (group['is_key_pass']).sum() + (group['is_assist']).sum()

        # Defensive Actions
        stats['Tackles Won'] = ((group['type_name'] == 'Tackle') & (group['outcome'] == 'Successful')).sum()
        stats['Interceptions'] = (group['type_name'] == 'Interception').sum()
        stats['Clearances'] = (group['type_name'] == 'Clearance').sum()
        stats['Ball recovery'] = (group['type_name'] == 'Ball recovery').sum()
        stats['Aerials Won'] = ((group['type_name'] == 'Aerial') & (group['outcome'] == 'Successful')).sum()

        # Passing Types
        stats['Passes into Box'] = ((group['type_name'] == 'Pass') &
                                    (group['end_x'].fillna(-1) >= 83) &
                                    (group['end_y'].fillna(-1) >= 21.1) &
                                    (group['end_y'].fillna(-1) <= 78.9)).sum()
        stats['Key Passes'] = stats['Shot Assists'] # Assuming they are the same by definition used

        # Basic Pass Stats
        stats['Total Passes'] = (group['type_name'] == 'Pass').sum()
        stats['Successful Passes'] = ((group['type_name'] == 'Pass') & (group['outcome'] == 'Successful')).sum()

        # Append player's stats dictionary to list
        player_stats_list.append(stats)

    # Convert list of dicts to DataFrame
    player_stats = pd.DataFrame(player_stats_list)
    if player_stats.empty:
        print("No player data after initial aggregation.")
        return pd.DataFrame()
    player_stats.set_index('playerName', inplace=True) # Set index after creation

    # --- Calculate Progressive Passes per Player ---
    # (Keep this logic as it merges counts from a separate calculation)
    print("  Calculating progressive passes per player...")
    df_prog_passes_all, _ = analyze_progressive_passes(
        df_processed, exclude_qualifiers=prog_pass_exclusions
    )
    if not df_prog_passes_all.empty:
        prog_passes_counts = df_prog_passes_all.groupby('playerName')['id'].count().rename('Progressive Passes')
        player_stats = player_stats.merge(prog_passes_counts, on='playerName', how='left')
        player_stats['Progressive Passes'] = player_stats['Progressive Passes'].fillna(0).astype(int)
    else:
        player_stats['Progressive Passes'] = 0

    # --- Calculate "Buildup to Shot" (Shift Logic) ---
    # (Keep this logic as it uses shift on the original df and merges)
    print("  Calculating 'Buildup to Shot' (shift logic)...")
    df_processed['next_event_is_kp'] = df_processed['is_key_pass'].shift(-1).fillna(False)
    buildup_df = df_processed[(df_processed['type_name'] == 'Pass') & df_processed['next_event_is_kp']]
    if not buildup_df.empty:
        buildup_counts = buildup_df.groupby('playerName')['id'].count().rename('Buildup to Shot')
        player_stats = player_stats.merge(buildup_counts, on='playerName', how='left')
        player_stats['Buildup to Shot'] = player_stats['Buildup to Shot'].fillna(0).astype(int)
    else:
        player_stats['Buildup to Shot'] = 0
    # Consider removing the temporary column if df_processed is used later
    # df_processed.drop(columns=['next_event_is_kp'], inplace=True, errors='ignore')


    # --- Calculate Totals ---
    # (Keep total calculation logic, ensure column names match exactly)
    shooting_seq_cols = ['Shots', 'Shot Assists', 'Buildup to Shot']
    if all(col in player_stats.columns for col in shooting_seq_cols): player_stats['Shooting Seq Total'] = player_stats[shooting_seq_cols].sum(axis=1)
    else: print("Warning: Could not calculate 'Shooting Seq Total'."); player_stats['Shooting Seq Total'] = 0

    offensive_pass_cols = ['Progressive Passes', 'Passes into Box', 'Shot Assists']
    if all(col in player_stats.columns for col in offensive_pass_cols): player_stats['Offensive Pass Total'] = player_stats[offensive_pass_cols].sum(axis=1)
    else: print("Warning: Could not calculate 'Offensive Pass Total'."); player_stats['Offensive Pass Total'] = 0

    # defensive_cols = ['Tackles Won', 'Interceptions', 'Clearances']
    # if all(col in player_stats.columns for col in defensive_cols): player_stats['Defensive Actions Total'] = player_stats[defensive_cols].sum(axis=1)
    # else: print("Warning: Could not calculate 'Defensive Actions Total'."); player_stats['Defensive Actions Total'] = 0

    defensive_cols = ['Tackles Won', 'Interceptions', 'Clearances', 'Ball recovery', 'Aerials Won']
    if all(col in player_stats.columns for col in defensive_cols):
        player_stats['Defensive Actions Total'] = player_stats[defensive_cols].sum(axis=1)
    else:
        print("Warning: Could not calculate 'Defensive Actions Total'.")
        player_stats['Defensive Actions Total'] = 0


    # --- Final Touches ---
    player_stats.fillna(0, inplace=True)
    count_cols = player_stats.select_dtypes(include=np.number).columns
    player_stats[count_cols] = player_stats[count_cols].astype(int)

    print(f"Finished calculating stats for {len(player_stats)} players.")
    print("Final columns in player_stats_df:", player_stats.columns.tolist())
    return player_stats

# --- Calculate Median Touch Location ---
def calculate_median_touch_location(df_processed, exclude_event_types=None):
    """
    Calculates the median x, y location for each player based on all
    events (excluding specified types like maybe Formation Change?).

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame.
        exclude_event_types (list, optional): List of 'type_name' values to exclude
                                             from the location calculation.
                                             Defaults to ['Formation Change', 'Deleted event'].

    Returns:
        pd.DataFrame: DataFrame indexed by 'playerName' containing median_x, median_y,
                      touch_count, jersey_number, and positional_role.
                      Returns empty DataFrame if errors or no relevant events.
    """
    print("Calculating median player touch locations...")
    if df_processed.empty:
        print("Warning: Input DataFrame is empty.")
        return pd.DataFrame()

    if exclude_event_types is None:
        exclude_event_types = ['Formation Change', 'Deleted event', 'End', 'Start'] # Add event types with no real location

    # Ensure necessary columns exist
    required_cols = ['playerName', 'team_name', 'x', 'y', 'type_name', 'Mapped Jersey Number', 'positional_role', 'id']
    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        print(f"Error: Missing required columns for median touch location: {missing}")
        return pd.DataFrame()

    # Filter out excluded event types and events without valid coordinates
    df_touches = df_processed[
        (~df_processed['type_name'].isin(exclude_event_types)) &
        (df_processed['x'].notna()) &
        (df_processed['y'].notna())
    ].copy()

    if df_touches.empty:
        print("Warning: No valid touch events found after filtering.")
        return pd.DataFrame()

    # Group by player and calculate median location, count, and get first jersey/role
    player_loc_agg = df_touches.groupby('playerName').agg(
        median_x=('x', 'median'),
        median_y=('y', 'median'),
        touch_count=('id', 'count'), # Count events considered as touches
        jersey_number=('Mapped Jersey Number', 'first'), # Get representative jersey
        positional_role=('positional_role', 'first') # Get representative role
    ).reset_index() # Make playerName a column

    print(f"Calculated median locations for {len(player_loc_agg)} players.")
    return player_loc_agg

def calculate_defensive_action_rates(df_player_actions):
    """
    Calculates the success rate for various defensive actions.
    Assumes an 'outcome' column exists where applicable (e.g., for Tackles, Aerials).
    """
    if df_player_actions.empty:
        return pd.DataFrame(columns=['Action', 'Successful', 'Total', 'Success Rate'])

    # Define which actions have a success/fail outcome vs. are always successful
    outcome_based_actions = ['Tackle', 'Aerial']
    always_successful_actions = ['Interception', 'Clearance', 'Ball recovery', 'Blocked pass']
    other_actions = ['Foul'] # Actions that are counted but not successful

    all_action_types = outcome_based_actions + always_successful_actions + other_actions
    stats_list = []

    for action in all_action_types:
        df_action = df_player_actions[df_player_actions['type_name'] == action]
        total = len(df_action)
        
        if total == 0:
            continue # Skip actions the player didn't attempt

        successful = 0
        if action in outcome_based_actions:
            # Assumes 'outcome' column has 'Successful' or 'Unsuccessful'
            successful = len(df_action[df_action['outcome'] == 'Successful'])
        elif action in always_successful_actions:
            successful = total # These actions are successful by definition
        # For 'Foul' and others, successful remains 0

        # Avoid division by zero for actions like Fouls
        rate = (successful / total) * 100 if total > 0 and action not in other_actions else 0

        stats_list.append({
            'Action': action,
            'Successful': successful,
            'Total': total,
            'Success Rate': f"{rate:.1f}%"
        })

    return pd.DataFrame(stats_list)

def get_mean_positions_data(df_processed, team_name):
    """
    Prepares data for plotting mean player positions. It calculates the median 
    position for each player based on all their touch-based events.

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame.
        team_name (str): The name of the team to analyze.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: All touch events for the team.
            - pd.DataFrame: Aggregated data per player (median_x, median_y, etc.).
    """
    # Definisci quali eventi contano come un "tocco"
    TOUCH_EVENT_TYPES = [
        'Pass', 'Take On', 'Ball touch', 'Shot', 'Dispossessed', 'Ball recovery', 
        'Clearance', 'Interception', 'Tackle', 'Goal'
    ]
    
    # Filtra tutti i tocchi per la squadra specificata
    df_all_touches = df_processed[
        (df_processed['team_name'] == team_name) &
        (df_processed['type_name'].isin(TOUCH_EVENT_TYPES))
    ].copy()

    if df_all_touches.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Calcola la posizione mediana e il conteggio delle azioni per ogni giocatore
    df_player_agg = df_all_touches.groupby('playerName').agg(
        median_x=('x', 'median'),
        median_y=('y', 'median'),
        action_count=('eventId', 'count') # Manteniamo il conteggio per l'hover
    ).reset_index()

    # Aggiungi informazioni sui giocatori (numero di maglia, se titolare)
    player_info = df_processed[
        df_processed['playerName'].isin(df_player_agg['playerName'])
    ][['playerName', 'Mapped Jersey Number', 'Is Starter']].drop_duplicates(subset='playerName')
    
    df_player_agg = pd.merge(df_player_agg, player_info, on='playerName', how='left')
    
    # Assicura che 'Is Starter' sia un booleano
    df_player_agg['Is Starter'] = df_player_agg['Is Starter'].fillna(False).astype(bool)

    return df_all_touches, df_player_agg
