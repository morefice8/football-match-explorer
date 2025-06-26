import pandas as pd
import os
import time
import json

# Import functions from our structured modules
from src.data_processing import load_opta, preprocess # preprocess is now critical
from src.utils import mapping_loader
from src import config

# --- Global Configuration (Paths that are generally fixed) ---
RAW_DATA_DIR = config.RAW_DATA_DIR
PROCESSED_DATA_DIR = config.PROCESSED_DATA_DIR
MAPPINGS_DIR = config.MAPPINGS_DIR

# Path to mapping files from config
EVENTS_MAPPING_FILE = config.OPTA_EVENTS_XLSX
QUALIFIERS_MAPPING_FILE = config.OPTA_QUALIFIERS_JSON

def run_processing(input_txt_file_path): # Ensure this takes the path as an argument
    """Main function to orchestrate data loading and preprocessing for a single match."""
    start_time = time.time()
    print(f"--- Starting processing for {input_txt_file_path} ---")

    # --- Preparations: check input file, create output dir ---
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    if not os.path.exists(input_txt_file_path):
        print(f"Error: Input file not found at {input_txt_file_path}")
        return

    # --- Step 1: Load and Clean Raw JSON Data ---
    print("Step 1: Loading and cleaning raw JSON...")
    raw_data = load_opta.clean_json(input_txt_file_path) # Use the passed argument
    if not raw_data:
        print(f"Failed to load or clean raw data from {input_txt_file_path}. Exiting.")
        return

    # --- Step 2: Extract Match Info ---
    print("Step 2: Extracting match info...")
    match_info = config.extract_match_info(raw_data)
    
    # --- Check if we got essential info for filenames ---
    hteam_short = match_info.get('hteamShortName')
    ateam_short = match_info.get('ateamShortName')

    if not hteam_short or not ateam_short:
        print("Warning: Could not extract one or both team short names for generating output filenames.")
        input_filename_stem = os.path.splitext(os.path.basename(input_txt_file_path))[0]
        base_filename = f"{input_filename_stem}_processed"
        if hteam_short and not ateam_short: base_filename = f"{hteam_short}_UNKNOWN-AWAY_processed"
        elif not hteam_short and ateam_short: base_filename = f"UNKNOWN-HOME_{ateam_short}_processed"
        print(f"Using fallback base filename: {base_filename}")
    else:
        base_filename = f"{hteam_short}_{ateam_short}_processed"
    
    output_parquet_file = os.path.join(PROCESSED_DATA_DIR, f"{base_filename}.parquet")
    output_metadata_file = os.path.join(PROCESSED_DATA_DIR, f"{base_filename}_metadata.json")

    print("Match Info Extracted:")
    print(f"  Home Team: {match_info.get('hteamName', 'N/A')} ({match_info.get('hteamShortName', 'N/A')})")
    print(f"  Away Team: {match_info.get('ateamName', 'N/A')} ({match_info.get('ateamShortName', 'N/A')})")
    # ... (print other info) ...

    # --- Step 3: Load Mappings ---
    print("Step 3: Loading event and qualifier mappings...")
    event_map = mapping_loader.load_opta_event_mapping(EVENTS_MAPPING_FILE)
    qualifier_map = mapping_loader.load_opta_qualifier_mapping(QUALIFIERS_MAPPING_FILE)
    if not event_map or qualifier_map is None: # Check if qualifier_map could be None on failure
         print("Failed to load one or more mapping files. Exiting.")
         return
    
    debug_hteam_short = hteam_short if hteam_short else 'HOME'
    debug_ateam_short = ateam_short if ateam_short else 'AWAY'
    debug_filename = f"debug_{debug_hteam_short}_{debug_ateam_short}_processed.xlsx"
    debug_filepath = os.path.join("debug_files", debug_filename)
    # Ensure the directory for debug files exists before process_opta_events is called
    os.makedirs(os.path.dirname(debug_filepath), exist_ok=True)


    # --- Step 4: Process Events into DataFrame ---
    print("Step 4: Processing events into DataFrame...")
    # Your preprocess.process_opta_events now returns a tuple of 4 items:
    # (df, initial_lineup_player_metadata, initial_team_formations, all_formation_events_by_team)
    processing_result_tuple = preprocess.process_opta_events(
        raw_data, event_map, qualifier_map, match_info, debug_excel_path=debug_filepath
    )

    df_processed = None
    # Unpack the tuple; the DataFrame is the first element
    if isinstance(processing_result_tuple, tuple) and len(processing_result_tuple) > 0:
        if isinstance(processing_result_tuple[0], pd.DataFrame):
            df_processed = processing_result_tuple[0]
            # You can optionally unpack and use the other elements if needed:
            # initial_lineup_player_metadata = processing_result_tuple[1]
            # initial_team_formations = processing_result_tuple[2]
            # all_formation_events_by_team = processing_result_tuple[3]
            print("Successfully unpacked DataFrame from processing result.")
        else:
            print(f"Error: First element of the tuple returned by process_opta_events is not a DataFrame. Got: {type(processing_result_tuple[0])}")
            return
    elif isinstance(processing_result_tuple, pd.DataFrame): # Fallback if it somehow returns just DF
        df_processed = processing_result_tuple
        print("Warning: process_opta_events returned a DataFrame directly, not a tuple as expected by the new preprocess.py.")
    else:
        print(f"Error: process_opta_events returned an unexpected type: {type(processing_result_tuple)}. Expected a tuple containing a DataFrame as its first element.")
        return

    # This is where the error occurred (around line 78 in your traceback)
    if df_processed is None or df_processed.empty:
        print("Processing resulted in an empty or None DataFrame. Exiting.")
        return
    
    print("--- Processed DataFrame Info ---")
    print(f"Shape: {df_processed.shape}")

    # --- Step 5a: Save the Processed DataFrame ---
    print(f"Step 5a: Saving processed DataFrame to {output_parquet_file}...")
    try:
        df_processed.to_parquet(output_parquet_file, index=False)
        print(f"Successfully saved processed DataFrame.")
    except Exception as e:
        print(f"An unexpected error occurred while saving processed DataFrame: {e}")
        return

    # --- Step 5b: Save the Match Metadata ---
    print(f"Step 5b: Saving match metadata to {output_metadata_file}...")
    try:
        with open(output_metadata_file, 'w', encoding='utf-8') as f:
            json.dump(match_info, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved match metadata.")
    except TypeError as e:
         print(f"Error: Could not serialize match_info to JSON: {e}. Check for non-serializable types.")
    except Exception as e:
        print(f"An unexpected error occurred while saving metadata: {e}")

    end_time = time.time()
    print(f"--- Processing Finished for {input_txt_file_path} in {end_time - start_time:.2f} seconds ---")


if __name__ == "__main__":
    os.makedirs("debug_files", exist_ok=True) # For debug Excel files

    # --- Input Selection (Example for your structure) ---
    # Your successful output showed: data/raw/PremierLeague/Season24-25/Week-30\eve-whu.txt
    # This implies RAW_DATA_DIR is something like "data/raw"
    # And the subsequent path components are "PremierLeague", "Season24-25/Week-30", "eve-whu.txt"

    selected_league = "PremierLeague"
    # Combine season and week like this if your folders are structured that way
    selected_game_week_and_season = os.path.join("Season24-25", "Week-30") 
    selected_match_filename = "1_Arsenal_Wolves_9xlphha9gt647b84dy64ts8b8.json"

    # Construct the full path to the input file
    # Ensure RAW_DATA_DIR is correctly set in your config (e.g., "data/raw")
    input_file_to_process = os.path.join(
        RAW_DATA_DIR, 
        selected_league, 
        selected_game_week_and_season, 
        selected_match_filename
    )
    
    # This is line 114 (or around it) from your traceback.
    # Ensure you call run_processing WITH the argument
    run_processing(input_file_to_process)