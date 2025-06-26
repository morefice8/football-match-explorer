# src/data_processing/pass_processing.py
import pandas as pd
from ..metrics import pass_metrics  # Assuming this is a module you have for pass metrics

def get_passes_df(df_processed):
    """
    Versione 3: Corregge il calcolo dei passaggi progressivi e assicura che
    tutte le colonne booleane e temporali necessarie siano presenti.
    """
    print("Extracting pass data in get_passes_df...")
    if df_processed.empty:
        print("  get_passes_df: Input df_processed is empty.")
        return pd.DataFrame()

    pass_events_filter = df_processed['type_name'] == 'Pass'
    if not pass_events_filter.any():
        print("  get_passes_df: No 'Pass' events found in df_processed.")
        return pd.DataFrame()

    # Lavora su una copia degli eventi di passaggio
    passes = df_processed[pass_events_filter].copy()

    # --- Calcolo dei flag booleani ---

    # 1. Passaggi Progressivi (calcolati SOLO sui passaggi riusciti)
    successful_passes = passes[passes['outcome'] == 'Successful'].copy()
    if not successful_passes.empty:
        exclusions_for_map = ['cross', 'Launch', 'ThrowIn'] 
        progressive_ids = pass_metrics.analyze_progressive_passes(
            passes, # Passiamo il df di passaggi, la funzione filtrerà per quelli riusciti
            exclude_qualifiers=exclusions_for_map,
            return_ids_only=True # <-- Diciamo alla funzione di restituire solo gli ID
        )
        passes['is_progressive'] = passes['id'].isin(progressive_ids)
        # progressive_pass_ids = pass_metrics.get_progressive_pass_ids(successful_passes)
        # passes['is_progressive'] = passes['id'].isin(progressive_pass_ids)
    else:
        passes['is_progressive'] = False # Nessun passaggio progressivo se non ci sono passaggi riusciti

    # 2. Passaggi in Area (calcolati su tutti i passaggi, anche falliti)
    passes['is_into_box'] = (passes['end_x'] >= 83.5) & (passes['end_y'].between(21.1, 78.9))
    
    # 3. Informazioni sul ricevitore (logica a scorrimento, come prima)
    passes["receiver"] = df_processed["playerName"].shift(-1).loc[passes.index]
    passes["receiver_jersey_number"] = df_processed["Mapped Jersey Number"].shift(-1).loc[passes.index]

    # 4. Assicura che i flag di key pass/assist esistano
    for flag_col in ['is_key_pass', 'is_assist']:
        if flag_col in passes.columns:
            passes[flag_col] = passes[flag_col].fillna(False).astype(bool)
        else:
            # Questo è il messaggio che stai vedendo. Significa che il problema è a monte.
            print(f"  get_passes_df: Flag column '{flag_col}' NOT found. Creating as all False.")
            passes[flag_col] = False

    # --- Define final columns to select ---
    columns_to_select = [
        "id", "eventId", 
        "timeMin", "timeSec", # Assicurati che siano incluse
        "x", "y", "end_x", "end_y", "team_name",
        "playerName", "shorter_name", "Mapped Jersey Number",
        "receiver", "receiver_jersey_number", "type_name", "outcome",
        "is_key_pass", "is_assist", "is_progressive", "is_into_box"
    ]

    # Seleziona solo le colonne che esistono effettivamente nel DataFrame
    final_present_columns = [col for col in columns_to_select if col in passes.columns]
    
    df_final_passes = passes[final_present_columns]

    print(f"  get_passes_df: Extracted {len(df_final_passes)} pass events. Columns: {df_final_passes.columns.tolist()}")
    return df_final_passes


# def get_passes_df(df_processed):
#     """
#     Filters the processed DataFrame for passes and adds receiver information.
#     Also includes 'is_key_pass' and 'is_assist' flags if they exist.
#     """
#     print("Extracting pass data...")
#     pass_events_filter = df_processed['type_name'] == 'Pass'
#     if not pass_events_filter.any():
#         print("Warning: No 'Pass' events found.")
#         # Define expected cols including potential flags
#         expected_cols = ["id", "x", "y", "end_x", "end_y", "team_name",
#                          "playerName", "shorter_name", "Mapped Jersey Number",
#                          "receiver", "receiver_jersey_number", "type_name", "outcome",
#                          "is_key_pass", "is_assist"] # Add flags here
#         return pd.DataFrame(columns=expected_cols)

#     passes_indices = df_processed.index[pass_events_filter]
#     df_temp_passes = df_processed.loc[passes_indices].copy()

#     # Add receiver info
#     df_temp_passes["receiver"] = df_processed["playerName"].shift(-1)
#     df_temp_passes["receiver_jersey_number"] = df_processed["Mapped Jersey Number"].shift(-1)

#     # --- Select final columns, INCLUDING the flags ---
#     columns_to_keep = ["id", "eventId", "x", "y", "end_x", "end_y", "team_name",
#                        "playerName", "shorter_name", "Mapped Jersey Number",
#                        "receiver", "receiver_jersey_number", "type_name", "outcome",
#                        "is_key_pass", "is_assist"] # Flags included

#     # Only keep columns that actually exist in the dataframe
#     final_columns = [col for col in columns_to_keep if col in df_temp_passes.columns]
#     # Ensure 'eventId' is definitely present if it was in df_temp_passes initially
#     if 'eventId' in df_temp_passes.columns and 'eventId' not in final_columns:
#         final_columns.append('eventId') # Should already be included by columns_to_keep

#     missing_cols = set(columns_to_keep) - set(final_columns)
#     # Don't warn about missing flags, but maybe about eventId if it's crucial and missing
#     if 'eventId' not in final_columns and 'eventId' in df_temp_passes.columns:
#         print("CRITICAL WARNING: eventId was in df_temp_passes but lost during column selection.")

#     # Don't warn loudly about missing flags, they might genuinely not exist
#     # if no key passes/assists occurred or the qualifier wasn't found.
#     # if missing_cols:
#     #     print(f"Warning: Missing expected columns in pass data: {missing_cols}")

#     df_passes = df_temp_passes[final_columns].copy()

#     print(f"Extracted {len(df_passes)} pass events. Columns: {df_passes.columns.tolist()}") # Print columns to verify
#     return df_passes


def get_sub_list(df_processed):
    """
    Identifies players who came on as substitutes.

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame.

    Returns:
        list: A list of unique player names who have a 'Player on' event.
    """
    if 'type_name' in df_processed.columns and 'playerName' in df_processed.columns:
        df_sub = df_processed[df_processed['type_name'] == 'Player on']
        sub_list = df_sub['playerName'].unique().tolist()
        print(f"Identified substitutes: {sub_list}")
        return sub_list
    else:
        print("Warning: Cannot determine substitutes. 'type_name' or 'playerName' column missing.")
        return []