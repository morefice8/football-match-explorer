# src/metrics/transition_metrics.py
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from collections import defaultdict
from dash import Dash, html as dash_html, dcc, Input, Output, dash_table, no_update
# import dash.html as html
from src import config
# --- Set display options to show all columns and more rows ---
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.max_rows', None)    # Show all rows (be careful with very large DFs)
pd.set_option('display.width', None)       # Auto-detect width to avoid line wrapping if possible
pd.set_option('display.max_colwidth', None) # Show full content of each column

# Define Opta typeIds (ensure these match your df_processed['typeId'] values)
TACKLE_ID = 7
BALL_RECOVERY_ID = 49  # Often used for loose ball recoveries
PASS_ID = 1

# Define outcome categories (can be moved to config)
SUCCESSFUL_TRANSITION_CATEGORIES = ['Goals', 'Shot', 'Big Chance', 'Chance Created'] # What counts as "successful"
FAILED_TRANSITION_CATEGORIES = ['Possession Lost', 'Turnover'] # What counts as "failed"

# --- Helper function to define pitch thirds (based on X-coordinate) ---
def get_pitch_third(x_coord):
    """Categorizes an x-coordinate into pitch thirds."""
    if pd.isna(x_coord):
        return "Unknown Third"
    if x_coord <= 33.33:
        return "Defensive Third"
    elif x_coord <= 66.67:
        return "Middle Third"
    else:
        return "Attacking Third"

# --- Function to find recoveries and subsequent first pass ---
def find_recovery_to_first_pass(df_processed,
                                tackle_type_id=TACKLE_ID,
                                ball_recovery_type_id=BALL_RECOVERY_ID,
                                pass_type_id=PASS_ID):
    """
    Identifies ball recoveries (tackles won in play, ball recovery events)
    and the immediate successful pass made by the same team afterwards.
    Categorizes these by the zone of recovery.

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame. Must include
                                     'eventId', 'typeId', 'team_name', 'playerName',
                                     'x', 'y', 'end_x', 'end_y', 'outcome',
                                     'Mapped Jersey Number'.
                                     Also needs a column for 'Out of play' if filtering tackles.
        tackle_type_id (int): Opta typeId for Tackle events.
        ball_recovery_type_id (int): Opta typeId for Ball Recovery events.
        pass_type_id (int): Opta typeId for Pass events.

    Returns:
        pd.DataFrame: DataFrame containing rows for each recovery and its subsequent
                      first pass, with added columns: 'recovery_x', 'recovery_y',
                      'recovery_zone', 'recovery_player', 'recovery_jersey',
                      'first_pass_player', 'first_pass_jersey',
                      'first_pass_x', 'first_pass_y', 'first_pass_end_x', 'first_pass_end_y'.
                      Returns empty DataFrame if errors or no such sequences.
    """
    print("Identifying recoveries and subsequent first passes...")

    # Ensure necessary columns exist
    required_cols = ['eventId', 'typeId', 'team_name', 'playerName', 'x', 'y',
                     'end_x', 'end_y', 'outcome', 'Mapped Jersey Number']
    # Check for 'Out of play' if filtering tackles (assuming it's a renamed qualifier)
    OUT_OF_PLAY_COL = 'Out of play' 
    if OUT_OF_PLAY_COL not in df_processed.columns:
        print(f"Warning: Column '{OUT_OF_PLAY_COL}' not found. Tackles won't be filtered for staying in play.")
        # Decide if this is critical. For now, proceed without it.

    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        # If OUT_OF_PLAY_COL was optional and missing, remove it from the error message
        if OUT_OF_PLAY_COL not in df_processed.columns and OUT_OF_PLAY_COL in missing:
            missing.remove(OUT_OF_PLAY_COL)
        if missing:
            print(f"Error: Missing required columns for recovery analysis: {missing}")
            return pd.DataFrame()

    # Sort by eventId to reliably use shift(-1) for next event
    df = df_processed.reset_index(drop=True)

    # --- Identify Recovery Events ---
    # Condition 1: Successful Tackle that stays in play
    tackle_filter = (
        (df['typeId'] == tackle_type_id) &
        (df['outcome'] == 'Successful')
    )
    if OUT_OF_PLAY_COL in df.columns: # Only apply if column exists
        # Assume 'Out of play' being NaN or '0' or False means ball stayed in play
        tackle_filter &= (~df[OUT_OF_PLAY_COL].isin(['1', 1, True]))
    else: # If column doesn't exist, we can't filter this, so include all successful tackles
        pass

    # Condition 2: Ball Recovery event type
    ball_recovery_filter = (df['typeId'] == ball_recovery_type_id)
    # Note: 'Ball recovery' events usually imply possession gained. Outcome might not always be 'Successful'.
    # Check your data for typical outcome of typeId=49. Assuming any typeId=49 is a recovery.

    recovery_event_filter = tackle_filter | ball_recovery_filter
    # recovery_event_filter = ball_recovery_filter

    df_recoveries = df[recovery_event_filter].copy()

    if df_recoveries.empty:
        print("No recovery events (tackles in play / ball recoveries) found.")
        return pd.DataFrame()

    print(f"Found {len(df_recoveries)} potential recovery events.")

    # --- Find the Immediate Next Successful Pass by the Same Team ---
    recovery_first_pass_data = []

    # Get indices of recovery events in the sorted DataFrame 'df'
    recovery_indices = df_recoveries.index.tolist()

    for rec_idx in recovery_indices:
        if rec_idx + 1 >= len(df): # Recovery is the last event in the game
            continue

        recovery_event = df.iloc[rec_idx]
        next_event = df.iloc[rec_idx + 1]

        # Check if next event is a successful pass by the same team
        is_pass = (next_event['typeId'] == pass_type_id)
        is_successful = (next_event['outcome'] == 'Successful')
        is_same_team = (next_event['team_name'] == recovery_event['team_name'])

        if is_pass:
            recovery_first_pass_data.append({
                'recovery_event_id': recovery_event['eventId'],
                'recovery_player': recovery_event['playerName'],
                'recovery_jersey': recovery_event.get('Mapped Jersey Number'),
                'recovery_x': recovery_event['x'],
                'recovery_y': recovery_event['y'],
                'recovery_zone': get_pitch_third(recovery_event['x']),
                'team_name': recovery_event['team_name'], # Team making recovery & first pass
                'first_pass_event_id': next_event['eventId'],
                'first_pass_player': next_event['playerName'],
                'first_pass_jersey': next_event.get('Mapped Jersey Number'),
                'first_pass_x': next_event['x'], # Start of the first pass
                'first_pass_y': next_event['y'],
                'first_pass_end_x': next_event['end_x'],
                'first_pass_end_y': next_event['end_y'],
                'first_pass_outcome': next_event['outcome'],
                'timeMin': recovery_event['timeMin'], 
                'timeSec': recovery_event['timeSec']
            })

    if not recovery_first_pass_data:
        print("No recoveries were immediately followed by a successful pass by the same team.")
        return pd.DataFrame()

    df_final = pd.DataFrame(recovery_first_pass_data)
    print(f"Found {len(df_final)} recovery-to-first-pass sequences.")
    return df_final

# --- Function: Find Opponent Buildup After Specific Team's Loss ---
def find_buildup_after_possession_loss(df_processed,
                                       team_that_lost_possession, # Team that lost possession
                                       possession_loss_types=['Goal', 'Pass', 'Take On', 'Error', 'Dispossessed', 'Aerial', 'Challenge', 'Clearance', 'Save'], # Types of loss
                                       max_passes_in_buildup_sequence=35,
                                       shot_types=['Goal', 'Miss', 'Attempt Saved', 'Post'],
                                       metric_to_analyze='defensive_transitions',
                                       time_threshold_seconds=2):
    """
    Identifies sequences of successful passes by the TEAM THAT GAINED POSSESSION
    immediately following a possession loss by the specified 'team_that_lost_possession'.
    Categorizes by the zone of the initial loss.

    Args:
        df_processed (pd.DataFrame): Main processed DataFrame.
        team_that_lost_possession (str): Name of the team whose loss triggers analysis.
        possession_loss_types (list): 'type_name' values for 'team_that_lost_possession'
                                      that signify losing possession.
        max_passes_in_buildup_sequence (int): Max passes to trace for the team that gained possession.

    Returns:
        pd.DataFrame: DataFrame of buildup sequences by the team that gained possession,
                      with 'loss_sequence_id' and 'loss_zone' (where possession was lost).
    """
    print(f"Identifying buildup sequences after {team_that_lost_possession} lost possession...")

    # --- Define Base and Optional Columns to Select ---
    # Required columns always needed
    required_cols = ['id', 'eventId', 'team_name', 'type_name', 'outcome', 'x', 'y',
                     'end_x', 'end_y', 'playerName', 'Mapped Jersey Number',
                     'timeMin', 'timeSec']
    # Optional columns that may be present
    optional_cols = ['receiver', 'receiver_jersey_number', 'Own goal', 'From corner', 'Goal mouth y co-ordinate']

    all_teams = df_processed['team_name'].unique()
    team_that_gained_possession = [t for t in all_teams if t != team_that_lost_possession][0]

    # Check base requirements
    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        print(f"Error: Missing required columns: {missing}"); 
        return pd.DataFrame()
    
    # Build list of columns to actually select
    cols_to_select = required_cols
    found_optional = []
    for col in optional_cols:
        if col in df_processed.columns:
            cols_to_select.append(col)
            found_optional.append(col)

    # Use set to ensure unique columns if any overlap, then convert back to list
    cols_to_select = list(set(cols_to_select))
    print(f"  Including optional columns found: {found_optional}")

    df = df_processed[cols_to_select].copy()
    df = df.reset_index(drop=True)
    df['total_seconds'] = df['timeMin'] * 60 + df['timeSec']

    # --- Identify Possession Loss Events by 'team_that_lost_possession' ---
    loss_filter = pd.Series(False, index=df.index)
    # ... (build loss_filter based on possession_loss_types) ...
    if 'Goal' in possession_loss_types: loss_filter |= ((df['type_name'] == 'Goal'))
    if 'Pass' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Pass') & (df['outcome'] == 'Unsuccessful'))
    if 'Take On' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Take On') & (df['outcome'] == 'Unsuccessful'))
    if 'Error' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Error'))
    if 'Dispossessed' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Dispossessed'))
    #if 'Clearance' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Clearance') & (df['outcome'] == 'Unsuccessful'))
    if 'Clearance' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Clearance'))
    if 'Aerial' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Aerial') & (df['outcome'] == 'Unsuccessful'))
    if 'Challenge' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Challenge') & (df['outcome'] == 'Unsuccessful'))
    if 'Save' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_gained_possession) & (df['type_name'] == 'Save') & (df['outcome'] == 'Successful'))

    df_losses_raw = df[loss_filter].copy()
    if df_losses_raw.empty: print(f"No loss events for {team_that_lost_possession}."); return pd.DataFrame()
    
    df_losses = df_losses_raw.drop_duplicates(subset=['id'], keep='first').copy()
    if df_losses.empty: print(f"No unique loss events after deduplication by 'id' for {team_that_lost_possession}."); return pd.DataFrame()
    print(f"Found {len(df_losses)} unique possession loss events by {team_that_lost_possession}. Tracing...")

    # indices_to_keep = []
    # last_trigger_time = -9999
    
    # for index, row in df_losses_raw.iterrows():
    #     current_time = row['total_seconds']
    #     # Se l'evento attuale è troppo vicino al precedente, lo saltiamo.
    #     if current_time - last_trigger_time > time_threshold_seconds:
    #         indices_to_keep.append(index)
    #         last_trigger_time = current_time
    
    # df_losses = df_losses_raw.loc[indices_to_keep].drop_duplicates(subset=['id'], keep='first').copy()
    # if df_losses.empty: print(f"No unique loss events after deduplication by 'id' for {team_that_lost_possession}."); return pd.DataFrame()
    # print(f"Found {len(df_losses)} unique possession loss events by {team_that_lost_possession}. Tracing...")


    # --- Trace Subsequent Sequences ---
    all_buildup_events_with_loss_info = [] # Stores dictionaries
    team_building_up = [t for t in df['team_name'].unique() if t != team_that_lost_possession][0]
    sequence_id_counter = 0

    processed_loss_event_ids = set()

    for loss_original_df_idx in df_losses.index:
        loss_event = df.iloc[loss_original_df_idx]

        # Zone where possession was gained by the opponent
        if metric_to_analyze == 'defensive_transitions':
            if loss_event['type_name'] in ('Pass'):
                loss_zone = get_pitch_third(loss_event['end_x'])
                print(f"DEBUG: Loss event {loss_event['type_name']} in {loss_event['end_x']} for zone '{loss_zone}'")
            else: 
                loss_zone = get_pitch_third(loss_event['x']) 
                print(f"DEBUG: Loss event {loss_event['type_name']} in {loss_event['x']} for zone '{loss_zone}'")
            
        else: # Offensive transitions
            if loss_event['type_name'] in ('Aerial', 'Dispossessed', 'Challenge', 'Take On', 'Error'):
                recovery_coord = 100 - loss_event['x']
                loss_zone = get_pitch_third(recovery_coord)
            else:
                recovery_coord = 100 - loss_event['end_x']
                loss_zone = get_pitch_third(recovery_coord)

        time_min_at_loss = loss_event.get('timeMin'); time_sec_at_loss = loss_event.get('timeSec')
        type_of_loss = loss_event.get('type_name', 'Unknown Loss')
        if loss_event.get('outcome') == 'Unsuccessful' and type_of_loss not in ['Error', 'Dispossessed']:
            if metric_to_analyze == 'defensive_transitions':
                type_of_loss = f"Unsuccessful {type_of_loss}"
            elif type_of_loss == 'Pass':
                type_of_loss = f"{type_of_loss} Interception"
            elif type_of_loss == 'Take On':
                type_of_loss = f"Ground Duel won (failed {type_of_loss})"
            elif type_of_loss == 'Aerial':
                type_of_loss = f"Aerial Duel won"

        current_opponent_sequence_events = []
        num_passes_in_seq = 0
        sequence_outcome_type = 'Unknown' # Default value
        current_event_original_df_idx = loss_original_df_idx # Start from the loss event index

        # Trace forward to find the opponent's sequence
        if loss_event['id'] in processed_loss_event_ids:
            continue
        if loss_event['id'] not in processed_loss_event_ids:
            while current_event_original_df_idx < len(df) - 1 and num_passes_in_seq < max_passes_in_buildup_sequence:
                current_event_original_df_idx += 1 # Move to the event *after* the loss or last pass
                
                action_by_gaining_team = df.iloc[current_event_original_df_idx]
                action_data = action_by_gaining_team.to_dict()
                action_data['loss_sequence_id'] = sequence_id_counter
                action_data['loss_zone'] = loss_zone
                action_data['triggering_loss_Opta_id'] = loss_event['id']
                action_data['timeMin_at_loss'] = time_min_at_loss
                action_data['timeSec_at_loss'] = time_sec_at_loss
                action_data['type_of_initial_loss'] = type_of_loss

                is_own_goal = action_by_gaining_team.get('Own goal') in [1, '1', True]

                if is_own_goal and action_by_gaining_team['team_name'] == team_that_lost_possession:
                    current_opponent_sequence_events.append(action_data)
                    if metric_to_analyze == 'defensive_transitions':
                        sequence_outcome_type = "Own Goal Conceded"
                    else: # offensive_transitions
                        sequence_outcome_type = "Forced Own Goal"
                    break

                is_correct_team = (action_by_gaining_team['team_name'] == team_building_up)
                is_pass = (action_by_gaining_team['type_name'] == 'Pass')
                is_unknown = (action_by_gaining_team['type_name'] == 'Unknown')
                is_successful_event = (action_by_gaining_team['outcome'] == 'Successful')
                is_team_that_lost_possession = (action_by_gaining_team['team_name'] == team_that_lost_possession)
                is_not_successful_event = (action_by_gaining_team['outcome'] == 'Unsuccessful')
                is_shot = (action_by_gaining_team['type_name'] in shot_types)
                is_end_sequence = (action_by_gaining_team['type_name'] in ('Foul', 'Out', 'Keeper pick-up', 'Claim', 'Dispossessed', 'Offside Pass', 'Corner Awarded'))
                is_take_on = (action_by_gaining_team['type_name'] == 'Take On')
                is_ball_touch = (action_by_gaining_team['type_name'] == 'Ball touch')
                is_ball_recovery = (action_by_gaining_team['type_name'] == 'Ball recovery')
                is_from_corner = (action_by_gaining_team.get('From corner') in [1, '1', True])

                if is_end_sequence and len(current_opponent_sequence_events) > 0:
                    current_opponent_sequence_events.append(action_data)
                    if action_by_gaining_team['type_name'] == 'Foul':
                        sequence_outcome_type = 'Foul'
                    elif action_by_gaining_team['type_name'] == 'Offside Pass':
                        sequence_outcome_type = 'Offside'
                    elif action_by_gaining_team['type_name'] == 'Out':
                        sequence_outcome_type = 'Out'
                    elif action_by_gaining_team['type_name'] == 'Corner Awarded':
                        sequence_outcome_type = 'Corner'
                    elif action_by_gaining_team['type_name'] == 'Dispossessed':
                        if metric_to_analyze == 'defensive_transitions':
                            sequence_outcome_type = f"Regained Possessions"
                        else:
                            sequence_outcome_type = f"Lost Possessions"
                    break # End the sequence here

                elif is_end_sequence:
                    break # End the sequence here
                
                elif is_correct_team and is_pass:
                    if is_successful_event: #successful pass
                        current_opponent_sequence_events.append(action_data)
                        num_passes_in_seq += 1
                    elif is_not_successful_event: # Unsuccessful pass
                        current_opponent_sequence_events.append(action_data)
                        if action_by_gaining_team['end_x'] >= 83 and (21.1 <= action_by_gaining_team['end_y'] <= 78.9): # If in the goal area
                            if metric_to_analyze == 'defensive_transitions':
                                sequence_outcome_type = f"Big Chances conceded"
                            else:
                                sequence_outcome_type = f"Big Chances"
                        else:
                            if metric_to_analyze == 'defensive_transitions':
                                sequence_outcome_type = f"Regained Possessions"
                            else:
                                sequence_outcome_type = f"Lost Possessions"
                        break # End the sequence here
                
                elif is_correct_team and is_shot and not is_own_goal: # It's a regular shot/goal
                    action_data['shot_end_y'] = action_by_gaining_team.get('Goal mouth y co-ordinate')
                    current_opponent_sequence_events.append(action_data)
                    
                    if action_by_gaining_team['type_name'] == 'Goal':
                        if metric_to_analyze == 'defensive_transitions':
                            sequence_outcome_type = "Goals conceded"
                        else: # offensive_transitions
                            sequence_outcome_type = "Goals"
                    else:
                        if metric_to_analyze == 'defensive_transitions':
                            sequence_outcome_type = "Shots conceded"
                        else: # offensive_transitions
                            sequence_outcome_type = "Shots"
                            
                    break # End the sequence here


                elif is_unknown: # Unknown event type
                    continue # Skip this event   
                elif is_ball_touch and is_successful_event: # Any unintentional ball touch
                    continue # Skip this event
                elif is_correct_team and is_successful_event: # Gaining team still has ball
                    continue # Skip this event
                elif is_correct_team and is_take_on and is_not_successful_event: # Gaining team lost possession due to unsuccessful take on
                    current_opponent_sequence_events.append(action_data)
                    if metric_to_analyze == 'defensive_transitions':
                        sequence_outcome_type = f"Regained Possessions"
                    else: 
                        sequence_outcome_type = f"Lost Possessions"
                    break
                elif is_correct_team and is_ball_touch and is_not_successful_event: # Gaining team lost possession due to unsuccessful control
                    if len(current_opponent_sequence_events) < 1: # If no events yet, don't count this as a sequence
                        break
                    current_opponent_sequence_events.append(action_data)
                    if metric_to_analyze == 'defensive_transitions':
                        sequence_outcome_type = f"Regained Possessions"
                    else:
                        sequence_outcome_type = f"Lost Possessions"
                    break
                elif is_team_that_lost_possession and is_not_successful_event: # Losing team fail to regain possession
                    # print(f"DEBUG: {action_by_gaining_team['type_name']} event by {action_by_gaining_team['team_name']} (id: {action_by_gaining_team['id']}) after loss event (id: {loss_event['id']})")
                    processed_loss_event_ids.add(action_by_gaining_team['id'])
                    # print(f"DEBUG: Seen lost possession events: {processed_loss_event_ids}")
                    continue # Skip this event

                elif is_correct_team and is_ball_recovery and is_successful_event:
                    processed_loss_event_ids.add(action_by_gaining_team['id'])
                    # print(f"DEBUG: Seen lost possession events: {processed_loss_event_ids}")
                    continue # Skip this event      
                                      
                elif is_team_that_lost_possession and is_successful_event: # Initial team regained possession
                    break
                else: # Some other event or end of data
                    break

        # pass_count = sum(
        #     1 for e in current_opponent_sequence_events
        #     if e['type_name'] == 'Pass' and e['outcome'] == 'Successful'
        # )

        # Deduplicate and filter to keep only events by the correct team
        df_seq = pd.DataFrame(current_opponent_sequence_events)

        if not df_seq.empty and 'team_name' in df_seq.columns:
            df_seq_deduped = df_seq.drop_duplicates(subset=[
                'eventId', 'team_name', 'type_name', 'x', 'y', 'end_x', 'end_y', 'timeMin', 'timeSec'
            ])
            pass_count = df_seq_deduped[
                (df_seq_deduped['type_name'] == 'Pass') & (df_seq_deduped['outcome'] == 'Successful')
            ].shape[0]

            for event_data_dict in df_seq_deduped.to_dict('records'):
                event_data_dict['opponent_pass_count'] = pass_count
                event_data_dict['sequence_outcome_type'] = sequence_outcome_type
                all_buildup_events_with_loss_info.append(event_data_dict)

            sequence_id_counter += 1
            # print(f"DEBUG Sequence {sequence_id_counter}: {pass_count} real passes, {len(current_opponent_sequence_events)} events, num_passes = {num_passes_in_seq}, outcome = {sequence_outcome_type}")

    # --- End Loop ---

    if not all_buildup_events_with_loss_info: return pd.DataFrame()
    df_all_sequences = pd.DataFrame(all_buildup_events_with_loss_info)
    print(f"Constructed {df_all_sequences['loss_sequence_id'].nunique()} opponent buildup sequences (incl. terminating event).")
    return df_all_sequences

def calculate_transition_success_by_zone(df_processed, team_name,
                                         recovery_event_types=None,
                                         possession_loss_types_opponent=None):
    """
    Calculates counts of successful, failed, and neutral offensive transitions
    originating from different pitch thirds.

    Returns:
        pd.DataFrame: Columns: ['recovery_zone', 'successful_transitions',
                                'failed_transitions', 'neutral_transitions', 'total_transitions']
    """
    print(f"Calculating transition success rates by zone for {team_name}...")
    if recovery_event_types is None: recovery_event_types = ['Ball recovery', 'Interception']
    if possession_loss_types_opponent is None: possession_loss_types_opponent = ['Pass', 'Take On', 'Error']


    df = df_processed.sort_values('eventId').reset_index(drop=True)
    df['next_team'] = df['team_name'].shift(-1)
    # ... (gains_filter logic - same as before) ...
    gains_filter = (((df['team_name'] == team_name) & df['type_name'].isin(recovery_event_types)) | ((df['team_name'] != team_name) & df['outcome'] == 'Unsuccessful' & df['type_name'].isin(possession_loss_types_opponent) & (df['next_team'] == team_name)))
    df_possession_gains = df[gains_filter].copy()

    if df_possession_gains.empty: return pd.DataFrame(columns=['recovery_zone', 'successful_transitions', 'failed_transitions', 'neutral_transitions', 'total_transitions'])

    transition_outcomes_detailed = []
    MAX_EVENTS_IN_TRANSITION = 10

    for gain_idx, gain_event in df_possession_gains.iterrows():
        recovery_zone = get_pitch_third(gain_event['x'])
        current_event_idx = gain_idx
        outcome_category = "Neutral End" # Default

        for _ in range(MAX_EVENTS_IN_TRANSITION):
            if current_event_idx + 1 >= len(df): break
            current_event_idx += 1
            next_event = df.iloc[current_event_idx]

            if next_event['team_name'] != team_name: outcome_category = "Possession Lost"; break
            # Define more specific outcomes based on event types
            if next_event['type_name'] == 'Goal': outcome_category = "Goal"; break
            if next_event['type_name'] in config.DEFAULT_SHOT_TYPES: outcome_category = "Shot"; break # Includes 'Attempt Saved', 'Post', 'Miss'
            if next_event.get('is_key_pass', False) or next_event.get('is_assist', False):
                if outcome_category not in ["Goal", "Shot"]: outcome_category = "Chance Created"
                # Don't break on chance created, allow sequence to continue to see if it leads to shot/goal
            if next_event['type_name'] == 'End': break # End of period

        transition_outcomes_detailed.append({'recovery_zone': recovery_zone, 'final_outcome': outcome_category})

    if not transition_outcomes_detailed: return pd.DataFrame(columns=['recovery_zone', 'successful_transitions', 'failed_transitions', 'neutral_transitions', 'total_transitions'])

    df_outcomes = pd.DataFrame(transition_outcomes_detailed)

    # Aggregate counts
    summary_list = []
    for zone, group in df_outcomes.groupby('recovery_zone'):
        total = len(group)
        successful = group['final_outcome'].isin(SUCCESSFUL_TRANSITION_CATEGORIES).sum()
        failed = group['final_outcome'].isin(FAILED_TRANSITION_CATEGORIES).sum()
        neutral = total - successful - failed
        summary_list.append({
            'recovery_zone': zone,
            'successful_transitions': successful,
            'failed_transitions': failed,
            'neutral_transitions': neutral,
            'total_transitions': total
        })

    df_zone_summary = pd.DataFrame(summary_list)
    print(f"Transition success summary for {team_name}:\n{df_zone_summary}")
    return df_zone_summary


# def calculate_def_transition_stats(sequence_list, is_away):
#     """
#     Calcola le statistiche riassuntive per le transizioni difensive basandosi sulle sequenze.
#     """
#     if not sequence_list:
#         return {}

#     total_sequences = len(sequence_list)

#     # 1. Outcomes (ultimo evento della sequenza)
#     outcomes = [seq.iloc[-1]['sequence_outcome_type'] for seq in sequence_list if not seq.empty and 'sequence_outcome_type' in seq.columns]
#     outcome_counts = pd.Series(outcomes).value_counts().to_dict()

#     # 2. Flank dominante (dove si sviluppa la sequenza)
#     flanks = []
#     for seq in sequence_list:
#         if seq.empty:
#             continue
#         right = seq['x'] > 66
#         left = seq['x'] < 33
#         center = (seq['x'] >= 33) & (seq['x'] <= 66)

#         n_right = right.sum()
#         n_left = left.sum()
#         n_center = center.sum()

#         if is_away:
#             # Inverti destra e sinistra se la squadra è in trasferta
#             n_right, n_left = n_left, n_right

#         max_side = max(n_right, n_left, n_center)
#         if max_side == n_right:
#             flanks.append("Right")
#         elif max_side == n_left:
#             flanks.append("Left")
#         else:
#             flanks.append("Center")

#     flank_counts = pd.Series(flanks).value_counts().to_dict()

#     # 3. Tipo di perdita iniziale
#     loss_types = [seq.iloc[0].get("type_of_initial_loss", "Unknown") for seq in sequence_list if not seq.empty]
#     loss_type_counts = pd.Series(loss_types).value_counts().to_dict()

#     return {
#         "total": total_sequences,
#         "outcomes": outcome_counts,
#         "flanks": flank_counts,
#         "types": loss_type_counts
#     }

def calculate_flank(y_vals):
    """
    Determines the dominant flank based on y-coordinates.
    Convention: y > 66 is left, y < 33 is right, 33 <= y <= 66 is center.
    """
    n_left = (y_vals > 66).sum()
    n_right = (y_vals < 33).sum()
    n_center = ((y_vals >= 33) & (y_vals <= 66)).sum()

    if max(n_left, n_right, n_center) == n_left:
        return "Left"
    elif max(n_left, n_right, n_center) == n_right:
        return "Right"
    else:
        return "Center"
    
def assign_bin(x, y, grid_size=6):
    """
    Assigns a bin number based on x and y coordinates using the grid size.
    Bins are numbered row-wise starting from the top-left corner.
    """
    bin_edges = np.linspace(0, 100, grid_size + 1)
    x_bin = np.digitize(x, bin_edges) - 1  # Bin index for x
    y_bin = np.digitize(y, bin_edges) - 1  # Bin index for y

    if x_bin < 0 or x_bin >= grid_size or y_bin < 0 or y_bin >= grid_size:
        return None  # Out of bounds

    return y_bin * grid_size + x_bin + 1  # Bin number (row-wise numbering)

def calculate_def_transition_stats(sequence_list, is_away=False):
    """
    Calcola le statistiche riassuntive per le transizioni difensive basandosi sulle sequenze.
    Include:
    - Outcomes finali
    - Flank dominante
    - Tipo di perdita iniziale
    - Tempo medio di transizione, passaggi medi, passaggi prima della riconquista per zona/flank
    """
    if not sequence_list:
        return {}

    total_sequences = len(sequence_list)

    # --- 1. Outcomes ---
    outcomes = [seq.iloc[-1]['sequence_outcome_type'] for seq in sequence_list if not seq.empty and 'sequence_outcome_type' in seq.columns]
    outcome_counts = pd.Series(outcomes).value_counts().to_dict()

    # --- 2. Flanks (dominant) ---
    flanks = []
    for seq in sequence_list:
        if seq.empty:
            continue
        flank = calculate_flank(seq["y"])  # Use raw y-coordinates
        flanks.append(flank)
    flank_counts = pd.Series(flanks).value_counts().to_dict()

    # --- 3. Initial Loss Type ---
    loss_types = [seq.iloc[0].get("type_of_initial_loss", "Unknown") for seq in sequence_list if not seq.empty]
    loss_type_counts = pd.Series(loss_types).value_counts().to_dict()

    # --- 4. Transition Profile per zona/flank ---
    profile = defaultdict(lambda: defaultdict(list))

    for seq in sequence_list:
        if seq.empty:
            continue

        zone = seq.iloc[0].get("loss_zone", "Unknown")
        # # Assign bin based on the first event's coordinates
        # x = seq.iloc[0]["x"]
        # y = seq.iloc[0]["y"]
        # bin_number = assign_bin(x, y, grid_size)

        # if bin_number is None:
        #     continue  # Skip if coordinates are out of bounds

        flank = calculate_flank(seq["y"])

        # Durata in secondi
        start_sec = seq.iloc[0]["timeMin"] * 60 + seq.iloc[0]["timeSec"]
        end_sec = seq.iloc[-1]["timeMin"] * 60 + seq.iloc[-1]["timeSec"]
        duration = end_sec - start_sec

        # Numero passaggi
        num_passes = (seq["type_name"] == "Pass").sum()

        # Numero passaggi prima di eventuale riconquista avversaria
        passes_before_recovery = seq[seq["team_name"] != seq.iloc[0]["team_name"]]["type_name"].eq("Pass").sum()

        key = (zone, flank)
        profile[key]["duration"].append(duration)
        profile[key]["passes"].append(num_passes)
        profile[key]["recoveries"].append(passes_before_recovery)

    # --- 5. Tabella riassuntiva ---
    profile_table = []
    for (zone, flank), vals in profile.items():
        profile_table.append({
            "Loss Zone": zone,
            "Counterattack Side": flank,
            "Avg Duration (s)": round(np.mean(vals["duration"]), 2),
            "Avg Passes": round(np.mean(vals["passes"]), 2),
            "Num_Sequences": len(vals["duration"])
        })

    return {
        "total": total_sequences,
        "outcomes": outcome_counts,
        "flanks": flank_counts,
        "types": loss_type_counts,
        "transition_profile_table": pd.DataFrame(profile_table)
    }

def generate_transition_profile_table(df):
    if df.empty:
        return dbc.Alert("No transition profile data available.", color="secondary")

    df_display = df.rename(columns={
        "Loss_Zone": "Loss Zone",
        "Flank": "Flank",
        "Avg_Duration": "⏱ Avg Duration (s)",
        "Avg_Passes": "🔁 Avg Passes",
        "Num_Sequences": "🔢 Sequences"
    })

    return dash_table.DataTable(
        columns=[{"name": col, "id": col} for col in df.columns],
        data=df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_cell={
            'padding': '5px',
            'textAlign': 'center',
            'fontFamily': 'Arial',
            'backgroundColor': '#f8f9fa'
        },
        style_header={
            'fontWeight': 'bold',
            'backgroundColor': '#dee2e6'
        }
    )



def create_def_transition_summary_cards(stats, active_filter=None):
    if not stats or stats.get("total", 0) == 0:
        return dbc.Alert("No summary data to display.", color="secondary")

    def is_active(filter_type, value):
        if not active_filter:
            return False
        return str(active_filter.get(filter_type)) == str(value)

    # Card 1: Outcomes
    outcome_order = ['Goals conceded', 'Own Goal Conceded', 'Forced Own Goal', 'Shots conceded', 'Big Chances conceded', 'Regained Possessions', 'Out', 'Offside', 'Foul']
    outcome_rank = {v: i for i, v in enumerate(outcome_order)}
    outcome_items = sorted(stats['outcomes'].items(), key=lambda x: outcome_rank.get(x[0], 99))

    outcome_list = []
    for outcome, count in outcome_items:
        if count == 0 and not is_active('outcomes', outcome):
            continue
        if active_filter and active_filter.get('outcomes') and active_filter['outcomes'] != outcome:
            continue
        active = is_active('outcomes', outcome)
        outcome_list.append(
            dbc.ListGroupItem(
                [dash_html.Div(outcome),
                 dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
                id={'type': 'def-filter', 'filter_type': 'outcomes', 'value': outcome},
                action=True, n_clicks=0,
                active=active,
                className="d-flex justify-content-between align-items-center"
            )
        )
    outcome_card = dbc.Card([
        dbc.CardHeader("Conceded Outcomes"), 
        dbc.ListGroup(outcome_list, flush=True)
    ], className="mb-3")

    # Card 2: Dominant Flank
    flank_list = []
    for flank, count in stats['flanks'].items():
        if count == 0 and not is_active('flanks', flank):
            continue
        if active_filter and active_filter.get('flanks') and active_filter['flanks'] != flank:
            continue
        active = is_active('flanks', flank)
        flank_list.append(
            dbc.ListGroupItem(
                [dash_html.Div(flank),
                 dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
                id={'type': 'def-filter', 'filter_type': 'flanks', 'value': flank},
                action=True, n_clicks=0,
                active=active,
                className="d-flex justify-content-between align-items-center"
            )
        )
    flank_card = dbc.Card([dbc.CardHeader("Counterattack Side"), dbc.ListGroup(flank_list, flush=True)], className="mb-3")

    # Card 3: Type of Loss
    type_list = []
    for loss_type, count in stats['types'].items():
        if count == 0 and not is_active('types', loss_type):
            continue
        if active_filter and active_filter.get('types') and active_filter['types'] != loss_type:
            continue
        active = is_active('types', loss_type)
        type_list.append(
            dbc.ListGroupItem(
                [dash_html.Div(loss_type),
                 dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
                id={'type': 'def-filter', 'filter_type': 'types', 'value': loss_type},
                action=True, n_clicks=0,
                active=active,
                className="d-flex justify-content-between align-items-center"
            )
        )
    type_card = dbc.Card([dbc.CardHeader("Type of Loss"), dbc.ListGroup(type_list, flush=True)], className="mb-3")

    return dbc.Row([
        dbc.Col(outcome_card, md=4),
        dbc.Col(flank_card, md=4),
        dbc.Col(type_card, md=4)
    ])


def calculate_off_transition_stats(sequence_list, is_away=False):
    """
    Calcola le statistiche riassuntive per le transizioni offensive.
    È quasi identica a quella difensiva, ma le etichette potrebbero cambiare.
    """
    if not sequence_list:
        return {}

    total_sequences = len(sequence_list)

    # 1. Outcomes (esattamente come per quelle difensive, ma il significato è invertito)
    outcomes = [seq.iloc[-1]['sequence_outcome_type'] for seq in sequence_list if not seq.empty]
    outcome_counts = pd.Series(outcomes).value_counts().to_dict()

    # 2. Flanks (dove si sviluppa la transizione offensiva)
    flanks = [calculate_flank(seq["y"]) for seq in sequence_list if not seq.empty]
    flank_counts = pd.Series(flanks).value_counts().to_dict()

    # 3. Tipo di recupero palla iniziale (era "tipo di perdita")
    recovery_types = [seq.iloc[0].get("type_of_initial_loss", "Unknown") for seq in sequence_list if not seq.empty]
    recovery_type_counts = pd.Series(recovery_types).value_counts().to_dict()
    
    # 4. Profilo di transizione (riutilizziamo la stessa logica)
    profile = defaultdict(lambda: defaultdict(list))
    for seq in sequence_list:
        if seq.empty: continue
        
        # Qui la zona è dove la palla è stata RECUPERATA
        zone = seq.iloc[0].get("loss_zone", "Unknown") # La funzione find_buildup... calcola già la zona corretta
        flank = calculate_flank(seq["y"])

        start_sec = seq.iloc[0]["timeMin"] * 60 + seq.iloc[0]["timeSec"]
        end_sec = seq.iloc[-1]["timeMin"] * 60 + seq.iloc[-1]["timeSec"]
        duration = end_sec - start_sec
        num_passes = (seq["type_name"] == "Pass").sum()

        key = (zone, flank)
        profile[key]["duration"].append(duration)
        profile[key]["passes"].append(num_passes)

    profile_table_data = []
    for (zone, flank), vals in profile.items():
        profile_table_data.append({
            "Recovery Zone": zone, # Etichetta cambiata
            "Attack Side": flank,  # Etichetta cambiata
            "Avg Duration (s)": round(np.mean(vals["duration"]), 2),
            "Avg Passes": round(np.mean(vals["passes"]), 2),
            "Num_Sequences": len(vals["duration"])
        })

    return {
        "total": total_sequences,
        "outcomes": outcome_counts,
        "flanks": flank_counts,
        "types": recovery_type_counts, # Ora rappresenta i tipi di recupero
        "transition_profile_table": pd.DataFrame(profile_table_data)
    }

# NUOVA FUNZIONE per creare le card riassuntive offensive
def create_off_transition_summary_cards(stats, active_filter=None):
    """
    Crea le card interattive per il riepilogo delle transizioni offensive.
    """
    if not stats or stats.get("total", 0) == 0:
        return dbc.Alert("No summary data to display.", color="secondary")

    def is_active(filter_type, value):
        return active_filter is not None and str(active_filter.get(filter_type)) == str(value)

    # Card 1: Outcomes
    outcome_order = ['Goals', 'Forced Own Goal', 'Shots', 'Big Chances', 'Lost Possessions', 'Out', 'Offside', 'Foul']
    outcome_rank = {v: i for i, v in enumerate(outcome_order)}
    outcome_items = sorted(stats.get('outcomes', {}).items(), key=lambda x: outcome_rank.get(x[0], 99))
    
    outcome_list = [
        dbc.ListGroupItem(
            [dash_html.Div(outcome), dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
            id={'type': 'off-filter', 'filter_type': 'outcomes', 'value': outcome}, # ID cambiato
            action=True, n_clicks=0, active=is_active('outcomes', outcome),
            className="d-flex justify-content-between align-items-center"
        ) for outcome, count in outcome_items
    ]
    outcome_card = dbc.Card([dbc.CardHeader("Attack Outcomes"), dbc.ListGroup(outcome_list, flush=True)], className="mb-3")

    # Card 2: Flank
    flank_list = [
        dbc.ListGroupItem(
            [dash_html.Div(flank), dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
            id={'type': 'off-filter', 'filter_type': 'flanks', 'value': flank}, # ID cambiato
            action=True, n_clicks=0, active=is_active('flanks', flank),
            className="d-flex justify-content-between align-items-center"
        ) for flank, count in stats.get('flanks', {}).items()
    ]
    flank_card = dbc.Card([dbc.CardHeader("Attack Side"), dbc.ListGroup(flank_list, flush=True)], className="mb-3")

    # Card 3: Recovery Type
    type_list = [
        dbc.ListGroupItem(
            [dash_html.Div(rec_type), dbc.Badge(f"{count} ({count / stats['total']:.0%})", className="ms-auto")],
            id={'type': 'off-filter', 'filter_type': 'types', 'value': rec_type}, # ID cambiato
            action=True, n_clicks=0, active=is_active('types', rec_type),
            className="d-flex justify-content-between align-items-center"
        ) for rec_type, count in stats.get('types', {}).items()
    ]
    type_card = dbc.Card([dbc.CardHeader("Type of Recovery"), dbc.ListGroup(type_list, flush=True)], className="mb-3")

    return dbc.Row([
        dbc.Col(outcome_card, md=4),
        dbc.Col(flank_card, md=4),
        dbc.Col(type_card, md=4)
    ])

