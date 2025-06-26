# src/metrics/sequence_metrics.py
import pandas as pd
import numpy as np
from collections import Counter # For counting sequences

# --- Find_shot_sequences ---
# This function identifies sequences of passes leading to a shot
def find_shot_sequences(df_processed,
                        shot_types=['Goal', 'Miss', 'Attempt Saved', 'Post'],
                        goal_mouth_y_col='GoalMouthY'): # Adjust if your col name is 'Goal mouth y co-ordinate'
    """
    Identifies sequences ending in a shot using backward looping logic.
    Selects more columns to allow for detailed title information in plots.
    """
    print("Identifying pass-to-shot sequences (backward loop, more detail)...")

    # --- Define Base and Optional Columns to Select ---
    # Base columns always needed
    base_required_cols = ['id', 'eventId', 'team_name', 'type_name', 'outcome',
                          'x', 'y', 'end_x', 'end_y', 'playerName',
                          'timeMin', 'timeSec'] 
    optional_info_cols = [
        'shorter_name', 'Mapped Jersey Number', 'positional_role',
        goal_mouth_y_col, # Or 'Goal mouth y co-ordinate' if that's the final name
        'is_key_pass', 'is_assist',

        # Qualifiers for Pattern of Play that generated the shot
        'Set piece', # Q24 - Shot occurred from a crossed free kick
        'Regular play', # Q22 - Shot occurred from regular play
        'Fast break', # Q23 - Shot occurred from a fast break
        'From corner', # Q25 - Shot occurred from a corner kick
        'Free kick', # Q26 - Shot occurred from a direct free kick
        'Throw-in set piece', # Q160 - Shot occurred from a throw-in set piece
        'Corner situation', # Q96 - Shot occurred from a 2nd phase attack following a corner situation

        # Qualifiers for the description of shot taken
        'Penalty', # Q9 - Shot occurred from a penalty kick

        # Qualifiers for the type of pass leading to the shot
        'Corner taken', # From Q6 - "Corner taken" is a common mapped name
        'Free kick taken', # From Q5 - "Free kick taken"
        'cross', # From Q2 - "Cross"
        'lb', # From Q1 - "Long ball"

        # Qualifiers for the body part used to take the shot
        'Head', # From Q15 - "Head" (for shots or passes)
        'Right footed', # From Q72 - "Right foot" (or a generic "BodyPart" column from Q21)
        'Left footed', # From Q73 - "Left foot"
        'Other body part' # From Q21 if a general "BodyPart" qualifier column
    ]

    # Check base requirements
    if not all(col in df_processed.columns for col in base_required_cols):
        missing = set(base_required_cols) - set(df_processed.columns)
        print(f"Error: Missing base required columns for sequence analysis: {missing}")
        return pd.DataFrame()

    # Build list of columns to actually select
    cols_to_select = base_required_cols
    found_optional = []
    for col in optional_info_cols:
        if col in df_processed.columns:
            cols_to_select.append(col)
            found_optional.append(col)
    # Use set to ensure unique columns if any overlap, then convert back to list
    cols_to_select = list(set(cols_to_select))
    # print(f"  Including optional columns found: {found_optional}")

    df = df_processed[cols_to_select].copy()
    df = df.reset_index(drop=True)
    # print(df.head())

    shots_filter = df['type_name'].isin(shot_types)
    shots_df = df[shots_filter].copy()

    if shots_df.empty: print("No shot events found."); return pd.DataFrame()
    print(f"Found {len(shots_df)} total shot events. Tracing sequences backward...")

    # --- Backward Loop Logic (Storing Indices) ---
    all_sequence_indices_with_id = []; sequence_id_counter = 0
    for shot_original_idx in shots_df.index:
        shot_event_series = df.iloc[shot_original_idx]; shot_team = shot_event_series['team_name']
        current_sequence_indices = [shot_original_idx]; current_idx = shot_original_idx - 1
        #print(f"  Shot event {shot_original_idx} ({shot_event_series['type_name']}) by {shot_team} at ({shot_event_series['x']}, {shot_event_series['y']})")
        while current_idx >= 0:
            prev_action_series = df.iloc[current_idx]
            is_pass = prev_action_series['type_name'] == 'Pass'; is_successful = prev_action_series['outcome'] == 'Successful'; is_same_team = prev_action_series['team_name'] == shot_team
            # print(f"  Checking event {current_idx} ({prev_action_series['type_name']}) by {prev_action_series['team_name']} at ({prev_action_series['x']}, {prev_action_series['y']})")
            if is_pass and is_same_team:
                current_sequence_indices.append(current_idx)
                current_idx -= 1
            elif prev_action_series['type_name'] in ('Aerial', 'Ball touch', 'Clearance'): 
                print(f"  Skipping {prev_action_series['type_name']} event at index {current_idx} in sequence {sequence_id_counter}.")
                current_idx -= 1
                continue; # Skip aerials and ball touches
            else: break # Break if not a pass or not same team
        for event_idx in reversed(current_sequence_indices): all_sequence_indices_with_id.append((event_idx, sequence_id_counter))
        sequence_id_counter += 1

    # --- Create Final DataFrame ---
    if not all_sequence_indices_with_id: print("No sequences constructed."); return pd.DataFrame()
    indices_to_keep = [idx for idx, seq_id in all_sequence_indices_with_id]; sequence_ids = [seq_id for idx, seq_id in all_sequence_indices_with_id]
    df_all_sequences = df.iloc[indices_to_keep].copy()
    df_all_sequences['sequence_id'] = sequence_ids # Assign sequence_id
    print(f"Constructed {df_all_sequences['sequence_id'].nunique()} shot sequences.")

    # --- Adjust Shot Coordinates ---
    shot_event_filter = df_all_sequences['type_name'].isin(shot_types)
    df_all_sequences.loc[shot_event_filter, 'end_x'] = 100.0
    # Use .get for safety, then fillna
    df_all_sequences.loc[shot_event_filter, 'end_y'] = df_all_sequences.loc[shot_event_filter].apply(
        lambda row: row.get(goal_mouth_y_col, 50.0), axis=1
    ).fillna(50.0)

    # Rename goal mouth column if it exists and was selected
    if goal_mouth_y_col in df_all_sequences.columns and goal_mouth_y_col != 'shot_end_y':
         df_all_sequences.rename(columns={goal_mouth_y_col: 'shot_end_y'}, inplace=True)
    elif goal_mouth_y_col not in df_all_sequences.columns and 'shot_end_y' not in df_all_sequences.columns:
        # If neither original nor renamed exists, add it based on end_y for shots
        df_all_sequences.loc[shot_event_filter, 'shot_end_y'] = df_all_sequences.loc[shot_event_filter, 'end_y']


    # Ensure boolean flags are boolean type if they exist
    for flag_col in ['is_key_pass', 'is_assist']:
        if flag_col in df_all_sequences.columns:
            df_all_sequences[flag_col] = df_all_sequences[flag_col].fillna(False).astype(bool)

    # Select final column order (include all potentially selected columns)
    # Ensure all columns in final_cols_order actually exist in df_all_sequences
    final_cols_order = ['id', 'eventId', 'sequence_id', 'team_name', 'type_name', 'outcome',
                        'playerName', 'shorter_name', 'Mapped Jersey Number', 'positional_role',
                        'timeMin', 'timeSec',
                        'x', 'y', 'end_x', 'end_y', 'shot_end_y',
                        'is_key_pass', 'is_assist',
                        'Regular play', 'Set piece', 'Fast break', 'From corner', 'Throw-in set piece', 'Corner situation', 
                        'Penalty', 'Corner taken', 'Free kick taken', 
                        'cross', 'lb', 'Head', 'Right footed', 'Left footed', 'Other body part'] # Add more as needed

    existing_final_cols = [col for col in final_cols_order if col in df_all_sequences.columns]
    # print(f"Final sequence columns being returned: {existing_final_cols}")
    return df_all_sequences[existing_final_cols]

# --- Helper function to define zones (Example: 3x3 grid) ---
# This function assigns a zone based on Opta coordinates (x, y)
def get_zone_3x3(x, y):
    """Assigns a 3x3 zone based on Opta coordinates."""
    if pd.isna(x) or pd.isna(y): return 'Unknown'
    row = int(y // (100/3)) # 0, 1, 2 from bottom to top
    col = int(x // (100/3)) # 0, 1, 2 from left to right

    row_map = {0: 'Bot', 1: 'Mid', 2: 'Top'}
    col_map = {0: 'Def', 1: 'Mid', 2: 'Att'}

    # Handle edge case where y=100 or x=100
    row = min(row, 2)
    col = min(col, 2)

    return f"{col_map[col]}{row_map[row]}" # e.g., 'DefBot', 'MidMid', 'AttTop'

# --- Function to find sequence patterns ---
# This function identifies sequences of passes leading to a shot
def find_sequence_patterns(df_all_sequences, pattern_type='zone', n_last_events=3):
    """
    Finds common patterns (player or zone sequences) in the last N events
    leading up to each shot.
    """
    print(f"Finding common {pattern_type} patterns (last {n_last_events} events)...")
    if df_all_sequences is None or df_all_sequences.empty: print("Warning: Input sequence DataFrame is empty."); return pd.Series(dtype=int)
    if 'sequence_id' not in df_all_sequences.columns: print("Error: 'sequence_id' column missing."); return pd.Series(dtype=int)

    patterns = []
    grouped_sequences = df_all_sequences.groupby('sequence_id')

    for seq_id, group in grouped_sequences:
        if 'eventId' in group.columns: group = group.sort_values('eventId')
        sequence_tail = group.tail(n_last_events)

        if len(sequence_tail) < n_last_events and n_last_events > 1: continue

        if pattern_type == 'zone':
            required_cols = ['x', 'y']
            if not all(col in sequence_tail.columns for col in required_cols): continue
            zone_sequence = tuple(sequence_tail.apply(lambda row: get_zone_3x3(row['x'], row['y']), axis=1))
            patterns.append(zone_sequence)
        elif pattern_type == 'player':
            required_cols = ['playerName']
            if not all(col in sequence_tail.columns for col in required_cols): continue
            player_sequence = tuple(sequence_tail['playerName'].fillna('Unknown'))
            patterns.append(player_sequence)
        elif pattern_type == 'role':
            required_cols = ['positional_role'] # Check for the new role column
            if not all(col in sequence_tail.columns for col in required_cols):
                print(f"Warning: Missing 'positional_role' column for sequence {seq_id}. Skipping pattern.")
                continue
            # Create tuple of roles, handling potential None/NaN from the mapping
            pattern = tuple(sequence_tail['positional_role'].fillna('Unknown'))
            patterns.append(pattern)
        else: print(f"Error: Unknown pattern_type '{pattern_type}'. Use 'zone', 'player', or 'role'."); return pd.Series(dtype=int)

    if not patterns: print("No patterns generated."); return pd.Series(dtype=int)
    pattern_counts = Counter(patterns)
    patterns_series = pd.Series(pattern_counts).sort_values(ascending=False)
    print(f"Found {len(patterns_series)} unique {pattern_type} patterns.")
    return patterns_series

# --- Helper function to define zones (e.g., 7x6 grid like chance creation) ---
def get_bin_location(x, y, bins=(7, 6)):
    """Assigns a bin tuple (col_idx, row_idx) based on coordinates and grid size."""
    if pd.isna(x) or pd.isna(y): return (np.nan, np.nan) # Return NaN tuple if coords invalid
    # Ensure coordinates are within bounds [0, 100]
    x = np.clip(x, 0, 100)
    y = np.clip(y, 0, 100)
    # Avoid index out of bounds for exactly 100
    col_bin_size = 100 / bins[0]
    row_bin_size = 100 / bins[1]
    col_idx = min(int(x // col_bin_size), bins[0] - 1)
    row_idx = min(int(y // row_bin_size), bins[1] - 1)
    return (col_idx, row_idx)

# --- Calculate Binned Sequence Transitions ---
def calculate_binned_sequence_stats(df_all_sequences, bins=(7, 6),
                                   shot_types=['Goal', 'Miss', 'Attempt Saved', 'Post']):
    """
    Calculates the frequency of transitions between pitch bins within shot sequences,
    identifies the dominant positional role making each frequent transition,
    and counts shot origins from each bin.

    Args:
        df_all_sequences (pd.DataFrame): DataFrame from find_shot_sequences
                                         (MUST contain 'sequence_id' and 'positional_role').
        bins (tuple): Grid size (width_bins, height_bins). Defaults to (7, 6).
        shot_types (list): List of 'type_name' values identifying shots.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Counts of transitions including dominant role
                            (start_bin, end_bin, total_transition_count, dominant_role, dominant_role_count).
            - pd.DataFrame: Counts of shots originating from each bin (start_bin, shot_origin_count).
            Returns (empty DF, empty DF) if input is empty or error occurs.
    """
    print(f"Calculating binned stats & dominant roles for shot sequences (Grid: {bins[0]}x{bins[1]})...")
    if df_all_sequences is None or df_all_sequences.empty:
        print("Warning: Input sequence DataFrame is empty."); return pd.DataFrame(columns=['start_bin', 'end_bin', 'total_transition_count', 'dominant_role', 'dominant_role_count']), pd.DataFrame(columns=['start_bin', 'shot_origin_count']) # Return empty DFs with expected columns

    # Ensure required columns exist, including positional_role
    required_cols = ['x', 'y', 'end_x', 'end_y', 'type_name', 'sequence_id', 'positional_role']
    if not all(col in df_all_sequences.columns for col in required_cols):
        missing = set(required_cols) - set(df_all_sequences.columns)
        print(f"Error: Missing required columns for binned role analysis: {missing}")
        return pd.DataFrame(columns=['start_bin', 'end_bin', 'total_transition_count', 'dominant_role', 'dominant_role_count']), pd.DataFrame(columns=['start_bin', 'shot_origin_count'])

    df_seq = df_all_sequences.copy()

    # --- Calculate Start and End Bins ---
    df_seq['start_bin'] = df_seq.apply(lambda row: get_bin_location(row['x'], row['y'], bins), axis=1)
    df_seq['end_bin'] = df_seq.apply(lambda row: get_bin_location(row['end_x'], row['end_y'], bins) \
                                     if row['type_name'] == 'Pass' else (np.nan, np.nan), axis=1)
    
    # --- Find Receiver Role for each Pass ---
    # Sort by eventId to use shift reliably
    df_seq = df_seq.sort_values('eventId')
    # Shift the passer's role up to align with the pass event
    df_seq['receiver_role'] = df_seq['positional_role'].shift(-1)
    # Clear receiver_role for non-pass events or last event of sequence
    df_seq.loc[df_seq['type_name'] != 'Pass', 'receiver_role'] = np.nan
    # Clear receiver role if the next event is a new sequence
    df_seq.loc[df_seq['sequence_id'] != df_seq['sequence_id'].shift(-1), 'receiver_role'] = np.nan

    # --- Aggregate Pass Transitions BY ROLE ---
    # Filter for valid passes with valid bins and known roles
    passes_in_seq = df_seq[
        (df_seq['type_name'] == 'Pass') &
        df_seq['start_bin'].apply(lambda b: isinstance(b, tuple) and not pd.isna(b[0])) &
        df_seq['end_bin'].apply(lambda b: isinstance(b, tuple) and not pd.isna(b[0])) &
        df_seq['positional_role'].notna() & # Need passer role
        df_seq['receiver_role'].notna() &   # Need receiver role
        ~df_seq['positional_role'].isin(['Sub/Unknown', 'UnknownFormation', 'UnknownPosNum']) &
        ~df_seq['receiver_role'].isin(['Sub/Unknown', 'UnknownFormation', 'UnknownPosNum'])
    ].copy()

    # Initialize empty DataFrame with correct columns in case no valid passes are found
    df_bin_transitions_final = pd.DataFrame(columns=['start_bin', 'end_bin', 'total_transition_count', 'dominant_passer_role', 'dominant_receiver_role', 'dominant_pair_count'])

    if passes_in_seq.empty:
        print("Warning: No valid pass transitions with known passer and receiver roles found.")
    else:
        # Count transitions for each specific role pair
        role_pair_transition_counts = passes_in_seq.groupby(
            ['start_bin', 'end_bin', 'positional_role', 'receiver_role'] # Group by all 4
        ).size().reset_index(name='role_pair_count')

        if not role_pair_transition_counts.empty:
            # Find the index of the max count role PAIR within each (start_bin, end_bin) group
            idx_max = role_pair_transition_counts.groupby(
                ['start_bin', 'end_bin']
            )['role_pair_count'].idxmax()

            # Get the dominant role pair information
            dominant_pairs = role_pair_transition_counts.loc[idx_max].copy() # Work on copy
            dominant_pairs.rename(columns={'positional_role': 'dominant_passer_role',
                                           'receiver_role': 'dominant_receiver_role',
                                           'role_pair_count': 'dominant_pair_count'}, inplace=True)

            # Calculate TOTAL transitions between bins (using the same filtered passes_in_seq)
            total_transitions = passes_in_seq.groupby(
                ['start_bin', 'end_bin']
            ).size().reset_index(name='total_transition_count')

            # Merge total counts and dominant pair info
            df_bin_transitions_final = pd.merge(
                total_transitions, dominant_pairs,
                on=['start_bin', 'end_bin'],
                how='left'
            )
            print(f"Found {len(df_bin_transitions_final)} unique bin transitions with dominant role pairs identified.")
        else:
             print("Warning: Grouping by role pair yielded no results.")


    # --- Aggregate Shot Origins (Same as before) ---
    shots_in_seq = df_seq[
        df_seq['type_name'].isin(shot_types) &
        df_seq['start_bin'].apply(lambda b: isinstance(b, tuple) and not pd.isna(b[0])) # Check start_bin is valid tuple
    ].copy()

    # Initialize empty DataFrame with correct columns
    df_shot_origins = pd.DataFrame(columns=['start_bin', 'shot_origin_count'])

    if shots_in_seq.empty:
        print("Warning: No valid shot origins found within sequences.")
    else:
        df_shot_origins = shots_in_seq.groupby('start_bin').size().reset_index(name='shot_origin_count')
        print(f"Found {len(df_shot_origins)} unique shot origin bins.")

    return df_bin_transitions_final, df_shot_origins


# --- Find Build-up Sequences from Defensive Third ---
# This function identifies sequences of passes originating from deep in a team's own half
# (e.g., from the goalkeeper or defensive third) and staying predominantly
# def find_buildup_sequences(df_processed,
#                            start_x_thresh_deep=15.0, # Max x-coord for the initial pass
#                            buildup_max_x_thresh=66.67): # Max x-coord for subsequent passes in own half
#     """
#     Identifies sequences of passes originating from deep in a team's own half
#     (e.g., from the goalkeeper or defensive third) and staying predominantly
#     within their own half or just beyond.

#     Args:
#         df_processed (pd.DataFrame): The main processed DataFrame.
#                            Must include 'id', 'eventId', 'team_name', 'type_name',
#                            'outcome', 'x', 'y', 'end_x', 'end_y', 'playerName',
#                            'Mapped Jersey Number', 'receiver', 'receiver_jersey_number'.
#         start_x_thresh_deep (float): Maximum x-coordinate for the *first* pass
#                                      of a buildup sequence to be considered.
#         buildup_max_x_thresh (float): Maximum x-coordinate for *subsequent* successful
#                                       passes in the sequence to remain part of the buildup
#                                       (e.g., staying within own half + a bit).

#     Returns:
#         pd.DataFrame: Flattened DataFrame containing all events part of a buildup sequence,
#                       with a 'buildup_sequence_id' column.
#                       Returns empty DataFrame if errors or no sequences found.
#     """
#     print(f"Identifying buildup sequences (starting x <= {start_x_thresh_deep}, continuing x <= {buildup_max_x_thresh})...")

#     # --- Input Validation and Setup ---
#     required_cols = ['id', 'eventId', 'team_name', 'type_name', 'outcome', 'x', 'y',
#                      'end_x', 'end_y', 'playerName', 'Mapped Jersey Number',
#                      'receiver', 'receiver_jersey_number'] # receiver_jersey_number is from get_passes_df
#     # Add optional ones if they exist
#     if 'shorter_name' in df_processed.columns: required_cols.append('shorter_name')
#     if 'is_key_pass' in df_processed.columns: required_cols.append('is_key_pass')
#     if 'is_assist' in df_processed.columns: required_cols.append('is_assist')


#     if not all(col in df_processed.columns for col in required_cols):
#         missing = set(required_cols) - set(df_processed.columns)
#         print(f"Error: Missing required columns for buildup sequence analysis: {missing}")
#         return pd.DataFrame()

#     # Work with a copy, sort by eventId, reset index for iloc
#     df = df_processed.copy()
#     df = df.sort_values('eventId').reset_index(drop=True)
#     # df = df.reset_index(drop=True)

#     # --- Identify Potential Starting Passes ---
#     # Passes by any team starting from deep
#     start_passes_x_values = pd.to_numeric(df['x'], errors='coerce')
#     deep_start_passes_filter = (
#         (df['type_name'] == 'Pass') &
#         (start_passes_x_values.fillna(start_x_thresh_deep + 1) <= start_x_thresh_deep)
#     )
#     potential_start_passes_df = df[deep_start_passes_filter]

#     if potential_start_passes_df.empty:
#         print("No deep starting passes found to initiate buildup sequences.")
#         return pd.DataFrame()

#     print(f"Found {len(potential_start_passes_df)} potential deep starting passes. Tracing sequences forward...")

#     # --- Trace Sequences Forward (Similar to Original Logic) ---
#     all_buildup_sequences_data = [] # Stores event data dictionaries
#     sequence_id_counter = 0
#     events_in_a_sequence = set() # To track event indices already part of a sequence

#     # Iterate through each potential starting pass
#     for start_pass_idx, start_pass_row in potential_start_passes_df.iterrows():
#         # Skip if this event is already part of another sequence
#         if start_pass_idx in events_in_a_sequence:
#             continue

#         current_sequence_events = [] # For this specific sequence
#         current_team = start_pass_row['team_name']

#         # Add the starting pass
#         start_pass_data = start_pass_row.to_dict() 
#         start_pass_data['buildup_sequence_id'] = sequence_id_counter 
#         current_sequence_events.append(start_pass_data) 
#         events_in_a_sequence.add(start_pass_idx) 

#         # Trace forward
#         current_event_idx = start_pass_idx 
#         last_pass_outcome_successful = (start_pass_row['outcome'] == 'Successful') # Track if the last pass was successful

#         # Only continue tracing if the first pass was successful
#         if not last_pass_outcome_successful:
#             all_buildup_sequences_data.extend(current_sequence_events) # Add the single pass sequence
#             sequence_id_counter += 1
#             continue # Move to the next potential starting pass

#         # Now trace forward for successful passes
#         while current_event_idx < len(df) - 1:
#             next_event_idx = current_event_idx + 1
#             next_action = df.iloc[next_event_idx]

#             # Conditions to continue the sequence:
#             # 1. It's a Pass
#             # 2. Same team
#             # 3. Start of pass is within the buildup_max_x_thresh (e.g., own half + midfield)
#             # 4. Event not already processed
#             is_pass = next_action['type_name'] == 'Pass'
#             is_same_team = next_action['team_name'] == current_team
#             # is_in_buildup_zone = next_action['x'].fillna(buildup_max_x_thresh + 1) <= buildup_max_x_thresh
#             is_new_event = next_event_idx not in events_in_a_sequence
#             next_action_x = pd.to_numeric(next_action['x'], errors='coerce') # Ensure numeric
#             if pd.isna(next_action_x):
#                 # If x is NaN, it's not in the buildup zone (unless thresh is also NaN, which it isn't)
#                 is_in_buildup_zone = False
#             else:
#                 is_in_buildup_zone = next_action_x <= buildup_max_x_thresh

#             if is_pass and is_same_team and is_in_buildup_zone and is_new_event:
#                 next_action_data = next_action.to_dict()
#                 next_action_data['buildup_sequence_id'] = sequence_id_counter
#                 current_sequence_events.append(next_action_data)
#                 events_in_a_sequence.add(next_event_idx)
#                 current_event_idx = next_event_idx # Move pointer forward
#                 # If this pass was unsuccessful, the sequence ends here
#                 if next_action['outcome'] != 'Successful':
#                     break
#             else:
#                 # Break if conditions not met
#                 break

#         all_buildup_sequences_data.extend(current_sequence_events)
#         sequence_id_counter += 1

#     # --- Create Final DataFrame ---
#     if not all_buildup_sequences_data:
#         print("No buildup sequences constructed.")
#         return pd.DataFrame()

#     df_all_buildup_sequences = pd.DataFrame(all_buildup_sequences_data)
#     print(f"Constructed {df_all_buildup_sequences['buildup_sequence_id'].nunique()} buildup sequences.")

#     if 'id' in df_all_buildup_sequences.columns and 'buildup_sequence_id' in df_all_buildup_sequences.columns:
#         num_rows_before_drop = len(df_all_buildup_sequences)
#         df_all_buildup_sequences.drop_duplicates(subset=['id', 'buildup_sequence_id'], keep='first', inplace=True)
#         num_rows_after_drop = len(df_all_buildup_sequences)
#         if num_rows_before_drop != num_rows_after_drop:
#             print(f"  Dropped {num_rows_before_drop - num_rows_after_drop} duplicate event entries within sequences.")
#     elif 'eventId' in df_all_buildup_sequences.columns and 'buildup_sequence_id' in df_all_buildup_sequences.columns:
#         # Fallback to eventId if 'id' is missing
#         num_rows_before_drop = len(df_all_buildup_sequences)
#         df_all_buildup_sequences.drop_duplicates(subset=['eventId', 'buildup_sequence_id'], keep='first', inplace=True)
#         num_rows_after_drop = len(df_all_buildup_sequences)
#         if num_rows_before_drop != num_rows_after_drop:
#             print(f"  Dropped {num_rows_before_drop - num_rows_after_drop} duplicate event entries within sequences using eventId.")
#     else:
#         print("Warning: Cannot drop duplicates effectively without 'id' or 'eventId' and 'buildup_sequence_id'.")

#     print(df_all_buildup_sequences)
#     return df_all_buildup_sequences

def find_buildup_sequences(df_processed,
                           start_x_thresh_deep=15.0, # Max x-coord for the initial pass
                           buildup_max_x_thresh=66.67): # Max x-coord for subsequent passes in own half
    """
    Identifies sequences of passes originating from deep in a team's own half
    (e.g., from the goalkeeper or defensive third) and staying predominantly
    within their own half or just beyond.

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame.
                           Must include 'id', 'eventId', 'team_name', 'type_name',
                           'outcome', 'x', 'y', 'end_x', 'end_y', 'playerName',
                           'Mapped Jersey Number', 'receiver', 'receiver_jersey_number'.
        start_x_thresh_deep (float): Maximum x-coordinate for the *first* pass
                                     of a buildup sequence to be considered.
        buildup_max_x_thresh (float): Maximum x-coordinate for *subsequent* successful
                                      passes in the sequence to remain part of the buildup
                                      (e.g., staying within own half + a bit).

    Returns:
        pd.DataFrame: Flattened DataFrame containing all events part of a buildup sequence,
                      with a 'buildup_sequence_id' column.
                      Returns empty DataFrame if errors or no sequences found.
    """
    print(f"Identifying buildup sequences (starting x <= {start_x_thresh_deep}, continuing x <= {buildup_max_x_thresh})...")

    # --- Input Validation and Setup ---
    required_cols = ['id', 'eventId', 'team_name', 'type_name', 'outcome', 'x', 'y',
                     'end_x', 'end_y', 'playerName', 'Mapped Jersey Number',
                     'receiver', 'receiver_jersey_number'] # receiver_jersey_number is from get_passes_df
    # Add optional ones if they exist
    if 'shorter_name' in df_processed.columns: required_cols.append('shorter_name')
    if 'is_key_pass' in df_processed.columns: required_cols.append('is_key_pass')
    if 'is_assist' in df_processed.columns: required_cols.append('is_assist')


    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        print(f"Error: Missing required columns for buildup sequence analysis: {missing}")
        return pd.DataFrame()

    # Work with a copy, sort by eventId, reset index for iloc
    df = df_processed.copy()
    df = df.reset_index(drop=True)
    # df = df.reset_index(drop=True)

    # --- Identify Potential Starting Passes ---
    # Passes by any team starting from deep
    potential_start_passes_df = df[(df['x'] <= start_x_thresh_deep) & (df['type_name'] == 'Pass')].copy()

    if potential_start_passes_df.empty:
        print("No deep starting passes found to initiate buildup sequences.")
        return pd.DataFrame()

    print(f"Found {len(potential_start_passes_df)} potential deep starting passes. Tracing sequences forward...")

    # --- Trace Sequences Forward (Similar to Original Logic) ---
    all_buildup_sequences_data = [] # Stores event data dictionaries
    sequence_id_counter = 0
    events_in_a_sequence = set() # To track event indices already part of a sequence

    # Iterate through each potential starting pass
    for start_pass_idx, start_pass_row in potential_start_passes_df.iterrows():
        # Skip if this event is already part of another sequence
        if start_pass_idx in events_in_a_sequence:
            continue

        current_sequence_events = [] # For this specific sequence
        current_team = start_pass_row['team_name']

        # Add the starting pass
        start_pass_data = start_pass_row.to_dict() 
        start_pass_data['buildup_sequence_id'] = sequence_id_counter 
        current_sequence_events.append(start_pass_data) 
        events_in_a_sequence.add(start_pass_idx) 

        # Trace forward
        current_event_idx = start_pass_idx 


        # Now trace forward for successful passes
        while current_event_idx < len(df) - 1:
            next_event_idx = current_event_idx + 1
            next_action = df.iloc[next_event_idx]

            # Conditions to continue the sequence:
            # 1. It's a Pass
            # 2. Same team
            # 3. Start of pass is within the buildup_max_x_thresh (e.g., own half + midfield)
            # 4. Event not already processed
            is_pass = next_action['type_name'] == 'Pass'
            is_same_team = next_action['team_name'] == current_team
            # is_in_buildup_zone = next_action['x'].fillna(buildup_max_x_thresh + 1) <= buildup_max_x_thresh
            is_new_event = next_event_idx not in events_in_a_sequence
            next_action_x = pd.to_numeric(next_action['x'], errors='coerce') # Ensure numeric
            if pd.isna(next_action_x):
                # If x is NaN, it's not in the buildup zone (unless thresh is also NaN, which it isn't)
                is_in_buildup_zone = False
            else:
                is_in_buildup_zone = next_action_x <= buildup_max_x_thresh

            if is_pass and is_same_team and next_action['x'] <= buildup_max_x_thresh:
                if not is_new_event:
                    continue # Skip if already in sequence
                next_action_data = next_action.to_dict()
                next_action_data['buildup_sequence_id'] = sequence_id_counter
                current_sequence_events.append(next_action_data)
                events_in_a_sequence.add(next_event_idx)
                current_event_idx = next_event_idx # Move pointer forward
                # If this pass was unsuccessful, the sequence ends here
                if next_action['outcome'] != 'Successful':
                    break
            else:
                # Break if conditions not met
                break

        all_buildup_sequences_data.extend(current_sequence_events)
        sequence_id_counter += 1

    # --- Create Final DataFrame ---
    if not all_buildup_sequences_data:
        print("No buildup sequences constructed.")
        return pd.DataFrame()
    
    # # --- Flatten the list and create DataFrame ---
    # flattened_pass_data = [event_data for sequence in all_buildup_sequences_data for event_data in sequence]

    # if not flattened_pass_data:
    #     print("No sequences constructed.")
    #     return pd.DataFrame()

    df_all_buildup_sequences = pd.DataFrame(all_buildup_sequences_data)
    print(f"Constructed {df_all_buildup_sequences['buildup_sequence_id'].nunique()} buildup sequences.")

    # if 'id' in df_all_buildup_sequences.columns and 'buildup_sequence_id' in df_all_buildup_sequences.columns:
    #     num_rows_before_drop = len(df_all_buildup_sequences)
    #     df_all_buildup_sequences.drop_duplicates(subset=['id', 'buildup_sequence_id'], keep='first', inplace=True)
    #     num_rows_after_drop = len(df_all_buildup_sequences)
    #     if num_rows_before_drop != num_rows_after_drop:
    #         print(f"  Dropped {num_rows_before_drop - num_rows_after_drop} duplicate event entries within sequences.")
    # elif 'eventId' in df_all_buildup_sequences.columns and 'buildup_sequence_id' in df_all_buildup_sequences.columns:
    #     # Fallback to eventId if 'id' is missing
    #     num_rows_before_drop = len(df_all_buildup_sequences)
    #     df_all_buildup_sequences.drop_duplicates(subset=['eventId', 'buildup_sequence_id'], keep='first', inplace=True)
    #     num_rows_after_drop = len(df_all_buildup_sequences)
    #     if num_rows_before_drop != num_rows_after_drop:
    #         print(f"  Dropped {num_rows_before_drop - num_rows_after_drop} duplicate event entries within sequences using eventId.")
    # else:
    #     print("Warning: Cannot drop duplicates effectively without 'id' or 'eventId' and 'buildup_sequence_id'.")

    print(df_all_buildup_sequences)
    return df_all_buildup_sequences

# --- Get Detailed Play and Last Pass Description ---
def get_shot_context_description(shot_event_series, last_pass_event_series=None):
    """
    Generates a detailed description string for a shot's context,
    including play type and last pass type.

    Args:
        shot_event_series (pd.Series): The event data for the shot.
        last_pass_event_series (pd.Series, optional): The event data for the
                                                     pass immediately preceding the shot.

    Returns:
        str: A descriptive string like "Open Play - Cross", "Set Piece - Direct", etc.
    """
    play_type_desc = "Open Play" # Default
    last_pass_action = ""

    # Determine Play Type from SHOT event
    # Ensure qualifier column names match your df_processed
    if str(shot_event_series.get('Set piece')) == '1': play_type_desc = "Set Piece" # Q24 - Shot occurred from a crossed free kick
    elif str(shot_event_series.get('Regular play')) == '1': play_type_desc = "Open play" # Q22 - Shot occurred from regular play
    elif str(shot_event_series.get('Fast break')) == '1': play_type_desc = "Fast break" # Q23 - Shot occurred from a fast break
    elif str(shot_event_series.get('From corner')) == '1': play_type_desc = "Corner" # Q25 - Shot occurred from a corner kick
    elif str(shot_event_series.get('Free kick')) == '1': play_type_desc = "Direct Free Kick" # Q26 - Shot occurred from a direct free kick
    elif str(shot_event_series.get('Throw-in set piece')) == '1': play_type_desc = "Throw-in set piece" # Q160 - Shot occurred from a throw-in set piece
    elif str(shot_event_series.get('Corner situation')) == '1': play_type_desc = "2nd Phase Corner" # Q96 - Shot occurred from a 2nd phase attack following a corner situation
    elif str(shot_event_series.get('Penalty')) == '1': play_type_desc = "Penalty" # Q9 - Shot occurred from a penalty kick

    # --- Extract Information from the LAST PASS in the sequence (if any) ---
    # Determine Last Pass Type (if a last pass exists in the sequence)
    if last_pass_event_series is not None and last_pass_event_series.get('type_name') == 'Pass':
        if str(last_pass_event_series.get('cross')) == '1': last_pass_action = "Cross"
        elif str(last_pass_event_series.get('lb')) == '1': last_pass_action = "Long Ball"
        elif str(last_pass_event_series.get('Corner taken')) == '1': last_pass_action = "Corner Pass"
        elif str(last_pass_event_series.get('Free kick taken')) == '1': last_pass_action = "FK Pass"
        else: last_pass_action = "Pass" # Generic pass
    elif len(shot_event_series.get('sequence_data', [])) == 1 : # If shot is the only event in its own minimal sequence data
        last_pass_action = "Direct Shot" # No preceding pass in *this specific* context

    if last_pass_action:
        return f"{play_type_desc} - {last_pass_action}"
    else: # Only play type if no distinct last pass action
        return play_type_desc