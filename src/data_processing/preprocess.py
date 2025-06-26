# src/data_processing/preprocess.py
import os
import pandas as pd
from src.utils import position_mapper

def get_shorter_name(full_name):
    if not isinstance(full_name, str) or not full_name.strip(): return None
    parts = full_name.split()
    if len(parts) == 1: return full_name
    return parts[0][0] + ". " + parts[-1] if len(parts) >= 2 else full_name

def process_opta_events(opta_data, event_mapping, qualifier_mapping, match_info, debug_excel_path=None):
    print("Starting Opta event processing...")
    contestants_map = {c['id']: c['name'] for c in opta_data.get('matchInfo', {}).get('contestant', [])}
    event_details = opta_data.get('liveData', {}).get('event', [])
    if not event_details:
        print("Warning: No event data found. Returning empty DataFrame and empty dicts.")
        return pd.DataFrame(), {}, {}, {}

    initial_lineup_player_metadata = {}
    initial_team_formations = {}
    all_formation_events_by_team = {name: [] for name in contestants_map.values()}
    print("Processing lineup/formation events (typeId 34)...")
    QID_PLAYER_IDS = 30; QID_JERSEY_NUM = 59; QID_POSITION_NUM = 131; QID_ROLE_CODE = 44; QID_FORMATION_ID = 130

    for event in event_details:
        if isinstance(event, dict) and event.get('typeId') == 34:
            team_id = event.get('contestantId'); team_name = contestants_map.get(team_id)
            qualifiers = event.get('qualifier', [])
            if not team_id or not team_name or not isinstance(qualifiers, list): continue
            q_vals = {q['qualifierId']: q.get('value') for q in qualifiers if isinstance(q, dict) and 'qualifierId' in q}
            formation_id_str = q_vals.get(QID_FORMATION_ID)
            current_formation_id = None
            if formation_id_str is not None:
                try: current_formation_id = int(formation_id_str)
                except ValueError: pass
            if current_formation_id is not None:
                time_marker = (event.get('periodId', 0), event.get('timeMin', 0), event.get('timeSec', 0), event.get('eventId', 0))
                all_formation_events_by_team.setdefault(team_name, []).append({'time_marker': time_marker, 'formation_id': current_formation_id, 'q_vals': q_vals})

    for team_name_key, formation_events_list in all_formation_events_by_team.items():
        if not formation_events_list: continue
        formation_events_list.sort(key=lambda x: x['time_marker'])
        initial_formation_event = formation_events_list[0]
        initial_formation_id = initial_formation_event['formation_id']
        team_id_for_meta = next((c_id for c_id, name_val in contestants_map.items() if name_val == team_name_key), None)
        if team_id_for_meta is None: continue
        initial_team_formations[team_id_for_meta] = initial_formation_id
        q_vals = initial_formation_event['q_vals']
        player_ids_str = q_vals.get(QID_PLAYER_IDS); jerseys_str = q_vals.get(QID_JERSEY_NUM)
        positions_num_str = q_vals.get(QID_POSITION_NUM); roles_code_str = q_vals.get(QID_ROLE_CODE)
        if not all([player_ids_str, jerseys_str, positions_num_str, roles_code_str]): continue
        try:
            player_ids = [pid.strip() for pid in str(player_ids_str).split(',') if pid.strip()]
            jerseys = [j.strip() for j in str(jerseys_str).split(',') if j.strip()]
            position_nums = [int(p.strip()) if p.strip().isdigit() else 0 for p in str(positions_num_str).split(',')]
            role_codes = [int(r.strip()) if r.strip().isdigit() else 0 for r in str(roles_code_str).split(',')]
            is_starter_list = [(p_num > 0 and p_num <= 11) for p_num in position_nums]
            if not (len(player_ids) == len(jerseys) == len(position_nums) == len(role_codes) == len(is_starter_list)): continue
            metadata_map = {pid: {"jersey": jerseys[idx], "role_code": role_codes[idx], "position_num": position_nums[idx], "is_starter": is_starter_list[idx]} for idx, pid in enumerate(player_ids)}
            initial_lineup_player_metadata[team_id_for_meta] = metadata_map
        except Exception as e: print(f"Error building INITIAL lineup metadata for team {team_id_for_meta}: {e}")
    print(f"Initial Team Formations: {initial_team_formations}")
    print(f"Initial Player Metadata sets: {len(initial_lineup_player_metadata)}")

    processed_events = []
    unique_qualifiers_found = set()
    print("Processing all game events...")    
    current_team_formations_state = initial_team_formations.copy()
    
    event_details_sorted = sorted(
        event_details, 
        key=lambda x: (
            x.get('periodId', 0), 
            x.get('timeMin', 0), 
            x.get('timeSec', 0), 
            x.get('timeStamp', "0000-00-00T00:00:00.000Z"),
            x.get('id', 0)
        )
    )

    for event in event_details_sorted:
        event_dict = {
             "id": event.get("id"), "eventId": event.get("eventId"), "typeId": event.get("typeId"),
             "periodId": event.get("periodId"), "timeMin": event.get("timeMin"), "timeSec": event.get("timeSec"),
             "contestantId": event.get("contestantId"), "playerId": event.get("playerId"),
             "playerName": event.get("playerName"), "outcome": event.get("outcome"),
             "x": event.get("x"), "y": event.get("y"), "timeStamp": event.get("timeStamp"),
        }
        current_team_id = event.get("contestantId")
        event_dict["team_name"] = contestants_map.get(current_team_id)

        qualifiers_list = event.get('qualifier', [])

        if event.get('typeId') == 19: # Substitution event
            for q_dict in qualifiers_list:
                if isinstance(q_dict, dict):
                    q_id = q_dict.get('qualifierId')
                    q_val = q_dict.get('value')
                    
                    if q_id == 55: # Related eventId
                        event_dict['related_eventId'] = q_val
                    elif q_id == 292: # Formation slot
                        event_dict['Formation slot'] = q_val
                        
        if isinstance(qualifiers_list, list):
            for q_dict in qualifiers_list:
                if isinstance(q_dict, dict) and 'qualifierId' in q_dict:
                    q_id_val = q_dict['qualifierId']
                    q_raw_value = q_dict.get('value') # Get the value if it exists
                    
                    # Try to process q_id_val as int, fallback to string
                    try: q_id_processed = int(q_id_val)
                    except ValueError: q_id_processed = str(q_id_val)
                    
                    unique_qualifiers_found.add(q_id_processed)
                    
                    # If 'value' key is NOT present, it's a flag, assign 1 (integer).
                    # Otherwise, store the value AS A STRING to handle mixed types.
                    event_dict[f"qualifier_{q_id_processed}"] = 1 if q_raw_value is None else str(q_raw_value)
        
        if event.get('typeId') == 34:
            q_vals_event = {q['qualifierId']: q.get('value') for q in qualifiers_list if isinstance(q, dict) and 'qualifierId' in q}
            formation_id_str_event = q_vals_event.get(QID_FORMATION_ID)
            if formation_id_str_event is not None:
                try: current_team_formations_state[current_team_id] = int(formation_id_str_event)
                except ValueError: pass

        


        current_player_id = str(event.get("playerId", ""))
        player_initial_meta = initial_lineup_player_metadata.get(current_team_id, {}).get(current_player_id)
        active_formation_id = current_team_formations_state.get(current_team_id)
        event_dict['Mapped Jersey Number'] = pd.NA; event_dict['Mapped Position Number'] = pd.NA
        event_dict['positional_role'] = 'Sub/Unknown'; event_dict['Is Starter'] = False
        if player_initial_meta:
            event_dict['Mapped Jersey Number'] = player_initial_meta.get('jersey')
            initial_pos_num = player_initial_meta.get('position_num')
            event_dict['Mapped Position Number'] = initial_pos_num
            event_dict['Is Starter'] = player_initial_meta.get('is_starter', False)
            event_dict['positional_role'] = position_mapper.get_role_from_formation(active_formation_id, initial_pos_num if event_dict['Is Starter'] else None)
        if event.get('typeId') == 18 and event_dict['positional_role'] in ['Sub/Unknown', 'UnknownFormation', 'UnknownPosNum']: pass
        processed_events.append(event_dict)

    df = pd.DataFrame(processed_events)
    if df.empty:
        print("Warning: DataFrame is empty after processing events.")
        return pd.DataFrame(), initial_lineup_player_metadata, initial_team_formations, all_formation_events_by_team

    print(f"DataFrame created with shape: {df.shape}. Columns initially (sample): {df.columns.tolist()[:25]}...")
    print(f"DEBUG: Unique qualifier IDs found in data: {sorted(list(str(qid) for qid in unique_qualifiers_found))}")

    print("Adding type names...")
    if 'typeId' in df.columns and event_mapping: df['type_name'] = df['typeId'].map(event_mapping).fillna('Unknown Type')
    else: df['type_name'] = 'Unknown Type'
    print("Renaming outcome...")
    if 'outcome' in df.columns: df['outcome'] = df['outcome'].map({0: 'Unsuccessful', 1: 'Successful'}).fillna('Unknown')
    else: print("Warning: 'outcome' column not found.")

    print("Renaming qualifier columns...")
    rename_dict = {}; qualifier_name_clashes = {}
    qualifiers_renamed_count = 0
    
    if qualifier_mapping and unique_qualifiers_found:
        for q_id_from_set in unique_qualifiers_found:
            temp_col_name = f"qualifier_{q_id_from_set}"
            if temp_col_name in df.columns:
                qualifier_info = None; q_id_lookup_key = None
                if isinstance(q_id_from_set, int): q_id_lookup_key = q_id_from_set
                elif isinstance(q_id_from_set, str) and q_id_from_set.isdigit(): q_id_lookup_key = int(q_id_from_set)
                else: q_id_lookup_key = str(q_id_from_set)
                
                qualifier_info = qualifier_mapping.get(q_id_lookup_key)
                if not qualifier_info: 
                    if isinstance(q_id_lookup_key, int): qualifier_info = qualifier_mapping.get(str(q_id_lookup_key))
                    elif isinstance(q_id_lookup_key, str):
                        try: qualifier_info = qualifier_mapping.get(int(q_id_lookup_key))
                        except ValueError: pass

                if qualifier_info:
                    qualifier_name = qualifier_info.get('name', f"UnknownQ_{q_id_from_set}")
                    if qualifier_name in rename_dict.values():
                         qualifier_name_clashes[qualifier_name] = qualifier_name_clashes.get(qualifier_name, 0) + 1
                         qualifier_name = f"{qualifier_name}_{qualifier_name_clashes[qualifier_name]}"
                    rename_dict[temp_col_name] = qualifier_name
                    qualifiers_renamed_count +=1
                else: print(f"Warning: QID '{q_id_from_set}' (type: {type(q_id_from_set)}) not found in qualifier_mapping file.")
        if rename_dict: df.rename(columns=rename_dict, inplace=True)
        print(f"Renamed {qualifiers_renamed_count} qualifier columns out of {len(unique_qualifiers_found)} unique QIDs.")
        if qualifiers_renamed_count == 0 and unique_qualifiers_found: print(f"WARNING: No qualifier columns were renamed. DF columns: {df.columns.tolist()[:25]}")
    else:
        if not unique_qualifiers_found: print("Info: No unique qualifiers were found.")
        if not qualifier_mapping: print("Warning: Qualifier mapping empty.")

    # --- INIZIO BLOCCO CORRETTO E FINALE PER I FLAG ---
    print("Creating special pass flags (is_key_pass, is_assist)...")
    
    # La tua colonna 'Assist' (da Q210) è quella che contiene i valori numerici.
    # La usiamo come fonte di verità.
    TARGET_COL = 'Assist' 
    
    if TARGET_COL in df.columns:
        # La colonna esiste, procediamo
        pass_type_numeric = pd.to_numeric(df[TARGET_COL], errors='coerce')
        
        key_pass_values = [13, 14, 15]
        assist_values = [16]

        # Un evento è un key pass/assist SE è un passaggio E ha il valore corretto nella colonna target
        is_pass_filter = (df['type_name'] == 'Pass')
        
        df['is_key_pass'] = (pass_type_numeric.isin(key_pass_values)) & is_pass_filter
        df['is_assist'] = (pass_type_numeric.isin(assist_values)) & is_pass_filter
        
        # Pulisci i NaN che potrebbero derivare dal booleano se la conversione numerica fallisce
        df['is_key_pass'] = df['is_key_pass'].fillna(False)
        df['is_assist'] = df['is_assist'].fillna(False)
        
        print(f"  Flags created. Assists found: {df['is_assist'].sum()}. Key Passes found: {df['is_key_pass'].sum()}.")
    else:
        # Questo non dovrebbe succedere, ma è una sicurezza
        print(f"  CRITICAL WARNING: Column '{TARGET_COL}' not found. Cannot create key pass/assist flags.")
        df['is_key_pass'] = False
        df['is_assist'] = False
    
    # --- FINE BLOCCO CORRETTO ---

    common_renames = {'Pass End X': 'end_x', 'Pass End Y': 'end_y', 'Throw-in': 'ThrowIn', 'Cross': 'cross', 'Long ball': 'lb'}
    actual_renames_to_apply = {k: v for k, v in common_renames.items() if k in df.columns}
    if actual_renames_to_apply: df.rename(columns=actual_renames_to_apply, inplace=True)
    print(f"Applied {len(actual_renames_to_apply)} common renames.")

    print("Converting data types...")
    coord_cols = ['x', 'y', 'end_x', 'end_y']
    for col in coord_cols:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
    
    known_numeric_renamed_qualifiers = ['Angle', 'Length'] # Add more as identified
    # Example: 'Related event ID' if you expect it to be always numeric AFTER specific cleaning.
    # For now, let's assume it can be mixed and keep it as object.

    for col_name in known_numeric_renamed_qualifiers:
        if col_name in df.columns:
            # Ensure the column is first treated as string before numeric conversion if it might contain mixed types
            df[col_name] = pd.to_numeric(df[col_name].astype(str), errors='coerce')
            print(f"Attempted conversion of column '{col_name}' to numeric.")

    flag_like_renamed_columns = ['KeyPass', 'Assist', 'Chipped', 'Corner taken'] # Add your actual renamed flag column names
    # The "Foul" column is problematic because it comes from "Related event ID" which can have "243"
    # So, "Foul" (if it's qualifier_13 renamed) should NOT be in flag_like_renamed_columns unless you specifically
    # handle the "243" to mean something else (e.g., map it to NaN before numeric conversion, or handle it as a category).
    # For now, we will NOT attempt to convert "Foul" to 0/1 int if it's the target of "Related event ID"
    
    # Example: if 'Foul' is a distinct flag qualifier AND NOT from qualifier_13
    # if 'Foul' in df.columns and 'Foul' not in known_numeric_renamed_qualifiers:
    #    flag_like_renamed_columns.append('Foul')


    for col_name in flag_like_renamed_columns:
        if col_name in df.columns:
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).astype(int)
            print(f"Processed flag-like column '{col_name}' to 0/1 int.")

    columns_to_ensure_string = []
    if 'Foul' in df.columns: # Assuming 'Foul' is the renamed column for qualifier_13
        columns_to_ensure_string.append('Foul')
    # Add any other such columns here, e.g., if 'Card type' (qualifier_243) was renamed to 'Card'
    # and its values are descriptive strings or numeric-like strings not meant for calculation.
    # if 'Card' in df.columns:
    #     columns_to_ensure_string.append('Card')

    for col_name in columns_to_ensure_string:
        if col_name in df.columns:
            print(f"Ensuring column '{col_name}' is string type before saving.")
            df[col_name] = df[col_name].astype(str).replace('nan', pd.NA) # Convert to string, then replace 'nan' string with actual pd.NA for Parquet
            # Alternatively, to keep 1 as int and '243' as str, and NaNs as NaNs:
            # df[col_name] = df[col_name].apply(lambda x: str(x) if pd.notna(x) and not isinstance(x, (int, float)) else x)
            # However, for Parquet, having a consistent string type (with pd.NA for missing) is often safer for object columns
            # if true mixed types (int and str) are causing issues with the writer.
            # The most robust is to make the whole column string if it has non-convertible strings like '243'.
            
    if 'Mapped Jersey Number' in df.columns: df['Mapped Jersey Number'] = pd.to_numeric(df['Mapped Jersey Number'], errors='coerce').astype('Int64')
    if 'Mapped Position Number' in df.columns: df['Mapped Position Number'] = pd.to_numeric(df['Mapped Position Number'], errors='coerce').astype('Int64')
    if 'Is Starter' in df.columns: df['Is Starter'] = df['Is Starter'].astype('boolean')

    print("Removing unwanted events...")
    values_to_remove = ['Deleted event']; original_rows = len(df)
    if 'type_name' in df.columns: df = df[~df['type_name'].isin(values_to_remove)].copy(); print(f"Removed {original_rows - len(df)} rows.")
    else: print("Warning: 'type_name' column not found for filtering.")

    print("Creating shorter names...")
    if 'playerName' in df.columns: df['shorter_name'] = df['playerName'].apply(get_shorter_name)
    else: print("Warning: 'playerName' column not found.")

    print("Dropping duplicate events based on 'id'...")
    if 'id' in df.columns:
        original_rows = len(df); df.drop_duplicates(subset='id', inplace=True, keep='first'); print(f"Removed {original_rows - len(df)} duplicate rows.")
    else: print("Warning: 'id' column not found for deduplication.")

    if 'end_x' not in df.columns or 'end_y' not in df.columns: print("Warning: 'end_x' or 'end_y' columns missing.")
    if 'positional_role' not in df.columns: print("Warning: 'positional_role' column failed to be added.")
    elif 'positional_role' in df.columns and df['positional_role'].isnull().all(): print("Warning: 'positional_role' column added but all values are Null/Unknown.")

    print(f"Preprocessing complete. Final DataFrame shape: {df.shape}")
    print(f"Final Columns (sample): {df.columns.tolist()[:25]}...")
    # For debugging the problematic 'Foul' column if the error persists:
    if 'Foul' in df.columns:
        print(f"DEBUG: 'Foul' column dtype: {df['Foul'].dtype}")
        print(f"DEBUG: Unique values in 'Foul' column: {df['Foul'].unique()[:20]}") # Print some unique values

    # print("Calculating running score...")
    
    # # Identifica le squadre di casa e ospiti
    # home_team_id = match_info.get('home_team_id')
    # away_team_id = match_info.get('away_team_id')
    
    # # Filtra solo gli eventi 'Goal' (typeId 16)
    # goals = df[df['typeId'] == 16].copy()
    
    # # Calcola i punteggi parziali
    # goals['home_score'] = (goals['contestantId'] == home_team_id).cumsum()
    # goals['away_score'] = (goals['contestantId'] == away_team_id).cumsum()
    
    # # Unisci i punteggi parziali al DataFrame principale
    # df = pd.merge(df, goals[['id', 'home_score', 'away_score']], on='id', how='left')
    
    # # Propaga l'ultimo punteggio valido in avanti (forward fill)
    # df[['home_score', 'away_score']] = df[['home_score', 'away_score']].ffill().fillna(0).astype(int)
    
    # print("Running score calculated.")

    if debug_excel_path:
        try:
            print(f"Attempting to save debug Excel file to: {debug_excel_path}")
            debug_dir = os.path.dirname(debug_excel_path)
            if debug_dir and not os.path.exists(debug_dir): os.makedirs(debug_dir, exist_ok=True)
            df.to_excel(debug_excel_path, index=False, engine='openpyxl')
            print(f"Successfully saved debug Excel file: {debug_excel_path}")
        except ImportError: print("Error saving Excel: `openpyxl` library not found.")
        except Exception as e: print(f"Error saving debug Excel file: {e}")

    return df, initial_lineup_player_metadata, initial_team_formations, all_formation_events_by_team