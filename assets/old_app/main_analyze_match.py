# main_analyze_match.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.table import Table 
from matplotlib.gridspec import GridSpec
from src.visualization.defensive_transitions import plot_defensive_transitions_with_summary
from src.data_preparation_for_plots.prepare_defensive_transitions import prepare_opponent_buildup_data, create_buildup_summary
from src.visualization.defensive_transitions import plot_defensive_transitions_with_summary
from src.data_preparation_for_plots.prepare_offensive_transitions import prepare_offensive_transitions_data
from src.visualization.offensive_transitions import plot_offensive_transitions_with_summary
from src.visualization.offensive_dashboard import plot_offensive_dashboard_with_sequences_multipage
from src.data_preparation_for_plots.prepare_buildup_phase import prepare_offensive_buildups_data
from src.visualization.buildup_phases import plot_buildup_phases_with_summary
from src.metrics.buildup_metrics import prepare_cross_analysis_data

import os
import json
import numpy as np
import argparse

# Import custom modules
from src.data_processing import pass_processing
from src.metrics import (
    pass_metrics,
    shot_metrics,
    turnover_metrics,
    defensive_metrics,
    player_metrics,
    sequence_metrics,
    transition_metrics
)
from src.visualization import pitch_plots, player_plots, pattern_plots
from src import config

# Configuration
MATCH_ID = "EVE_WHU"  # Default match if no args provided
SAVE_PLOTS = False
OUTPUT_VIZ_DIR = "visualizations"
os.makedirs(OUTPUT_VIZ_DIR, exist_ok=True)
HOME_HEATMAP_CMAP = 'Reds'
AWAY_HEATMAP_CMAP = 'Blues'
DEFAULT_HXG = 1.00; DEFAULT_AXG = 0.83; DEFAULT_HXGOT = 1.46; DEFAULT_AXGOT = 1.82
# Define colors needed for dashboard plots (get from config if defined there)
VIOLET_COLOR = config.VIOLET if hasattr(config, 'VIOLET') else '#a369ff'
GREEN_COLOR = config.GREEN if hasattr(config, 'GREEN') else '#69f900'


def run_analysis(match_id, plots_to_generate,
                 def_action_ids_arg=None,
                 buildup_start_zone_x_max=33.3, # Default: defensive third
                 buildup_end_zone_x_min=66.7,   # Default: attacking third
                 buildup_end_zone_y_bounds=(20, 80), # Default: central/half-space Y range
                 h_passer_arg="TOP", a_passer_arg="TOP",
                 h_receiver_arg="TOP", a_receiver_arg="TOP",
                 h_defender_arg="TOP", a_defender_arg="TOP"
                 ):
    """Loads data/metadata, generates SPECIFIED visualizations sequentially."""
    print(f"--- Starting Match Analysis for {match_id} ---")
    print(f"--- Plots requested: {plots_to_generate} ---")

    # --- Determine Defensive Action IDs to Use ---
    if def_action_ids_arg: # If user provided IDs via command line
        def_action_ids_to_use = def_action_ids_arg
        print(f"Using user-specified defensive action IDs: {def_action_ids_to_use}")
    else: # Use default from config
        def_action_ids_to_use = config.DEFAULT_PPDA_DEF_ACTION_IDS
        print(f"Using default defensive action IDs: {def_action_ids_to_use}")
    # ---

    # --- Derive File Paths ---
    base_filename = f"{match_id}_processed"
    input_parquet_file = os.path.join(config.PROCESSED_DATA_DIR, f"{base_filename}.parquet")
    input_metadata_file = os.path.join(config.PROCESSED_DATA_DIR, f"{base_filename}_metadata.json")

    # --- Load Metadata ---
    if not os.path.exists(input_metadata_file): print(f"Error: Metadata file not found: {input_metadata_file}"); return
    try:
        with open(input_metadata_file, 'r', encoding='utf-8') as f: match_info = json.load(f)
        print("Loaded match metadata.")
    except Exception as e: print(f"Error loading metadata: {e}"); return

    # Extract info from metadata
    HTEAM_NAME = match_info.get('hteamName', 'Home'); ATEAM_NAME = match_info.get('ateamName', 'Away')
    HTEAM_SHORT_NAME = match_info.get('hteamShortName', 'HOM'); ATEAM_SHORT_NAME = match_info.get('ateamShortName', 'AWY')
    HTEAM_COLOR = config.DEFAULT_HCOL; ATEAM_COLOR = config.DEFAULT_ACOL
    GAMEWEEK = match_info.get('gw', '?'); LEAGUE = match_info.get('league', '?'); MATCH_DATE = match_info.get('date_formatted', '?')
    MATCH_HXG = match_info.get('hxg', DEFAULT_HXG); MATCH_AXG = match_info.get('axg', DEFAULT_AXG)
    MATCH_HXGOT = match_info.get('hxgot', DEFAULT_HXGOT); MATCH_AXGOT = match_info.get('axgot', DEFAULT_AXGOT)

    # --- Load Processed Data ---
    if not os.path.exists(input_parquet_file): print(f"Error: Data file not found: {input_parquet_file}"); return
    try:
        df_processed = pd.read_parquet(input_parquet_file)
        print(f"Loaded processed data. Shape: {df_processed.shape}")
        print(f"Columns: {df_processed.columns.tolist()}")
    except Exception as e: print(f"Error loading data: {e}"); return

    # --- *** START: Pre-calculate Flags on df_processed *** ---
    # This ensures flags are available before other metric/processing steps
    print("Pre-calculating key pass/assist flags...")
    # *** IMPORTANT: Verify 'Assist' is the correct column name ***
    assist_qualifier_col='Assist' # ADJUST IF NEEDED
    key_pass_values=[13, 14, 15]; assist_values=[16] # Values from original code

    if assist_qualifier_col not in df_processed.columns:
         print(f"Warning: Assist qualifier column '{assist_qualifier_col}' not found in df_processed. Key Pass/Assist flags cannot be determined.")
         # Create empty/False columns so downstream code doesn't break, but results will be inaccurate
         df_processed['is_key_pass'] = False
         df_processed['is_assist'] = False
    else:
        assist_qual_numeric = pd.to_numeric(df_processed[assist_qualifier_col], errors='coerce')
        # Calculate and ensure flags are boolean
        if 'is_key_pass' not in df_processed.columns:
             print("Info: Adding 'is_key_pass' flag to df_processed.")
             df_processed['is_key_pass'] = assist_qual_numeric.isin(key_pass_values) & (df_processed['type_name'] == 'Pass')
        df_processed['is_key_pass'] = df_processed['is_key_pass'].fillna(False).astype(bool)

        if 'is_assist' not in df_processed.columns:
            print("Info: Adding 'is_assist' flag to df_processed.")
            df_processed['is_assist'] = assist_qual_numeric.isin(assist_values) & (df_processed['type_name'] == 'Pass')
        df_processed['is_assist'] = df_processed['is_assist'].fillna(False).astype(bool)
    print("Flags pre-calculation complete.")
    # --- *** END: Pre-calculate Flags *** ---


    # --- Initialize Data Structures ---
    # Initialize all potentially needed structures
    passes_df = pd.DataFrame(); successful_passes = pd.DataFrame()
    shots_df = pd.DataFrame(); home_stats = {}; away_stats = {}
    df_prog_passes_all = pd.DataFrame(); home_prog_passes = pd.DataFrame(); away_prog_passes = pd.DataFrame()
    home_prog_zone_stats = {}; away_prog_zone_stats = {}
    home_ft_stats_dict = None; away_ft_stats_dict = None
    home_z14_df, home_lhs_df, home_rhs_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    away_z14_df, away_lhs_df, away_rhs_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    sub_list = []
    home_high_to_df = pd.DataFrame(); away_high_to_df = pd.DataFrame(); hto_count = 0; ato_count = 0
    home_chances_df = pd.DataFrame(); away_chances_df = pd.DataFrame()
    df_def_actions = pd.DataFrame(); home_def_agg_df = pd.DataFrame(); away_def_agg_df = pd.DataFrame()
    player_stats_df = pd.DataFrame() 
    df_bin_trans_home_role, df_shot_origin_home = pd.DataFrame(), pd.DataFrame()
    df_bin_trans_away_role, df_shot_origin_away = pd.DataFrame(), pd.DataFrame()
    home_loc_agg_df = pd.DataFrame(); away_loc_agg_df = pd.DataFrame() # For mean positions
    df_touches_home = pd.DataFrame(); df_touches_away = pd.DataFrame() # For heatmap background
    home_ppda = np.nan; away_ppda = np.nan # Initialize PPDA vars
    event_map_for_plots = {} # To store event mapping for titles
    initial_lineup_meta_home_df = pd.DataFrame()
    initial_lineup_meta_away_df = pd.DataFrame()
    home_starting_formation_id = None
    away_starting_formation_id = None
    all_formation_changes_log = {}
    df_recovery_first_passes = pd.DataFrame()
    df_away_buildup_after_home_loss = pd.DataFrame()
    df_home_buildup_after_away_loss = pd.DataFrame()
    home_loss_summary_table_data = None
    away_loss_summary_table_data = None
    df_away_buildup = pd.DataFrame()
    df_home_buildup = pd.DataFrame()
    home_buildup_summary_table_data = None
    away_buildup_summary_table_data = None
    home_cross_df, away_cross_df = pd.DataFrame(), pd.DataFrame()
    home_cross_summary, away_cross_summary = [], []
    home_total_crosses, away_total_crosses = 0, 0
    cross_summary_cols = []

    if 'formation' in plots_to_generate:
        FORMATION_ID_COL_NAME = 'Team formation' # *** VERIFY THIS RENAMED COLUMN FOR Q130 ***
        lineup_type34_events = df_processed[df_processed['typeId'] == 34]

        # Get initial formation IDs (first typeId=34 event for each team)
        home_first_lineup_event = lineup_type34_events[lineup_type34_events['team_name'] == HTEAM_NAME].sort_values('eventId').head(1)
        if not home_first_lineup_event.empty and FORMATION_ID_COL_NAME in home_first_lineup_event.columns:
            try: home_starting_formation_id = int(home_first_lineup_event[FORMATION_ID_COL_NAME].iloc[0])
            except: home_starting_formation_id = None
        away_first_lineup_event = lineup_type34_events[lineup_type34_events['team_name'] == ATEAM_NAME].sort_values('eventId').head(1)
        if not away_first_lineup_event.empty and FORMATION_ID_COL_NAME in away_first_lineup_event.columns:
            try: away_starting_formation_id = int(away_first_lineup_event[FORMATION_ID_COL_NAME].iloc[0])
            except: away_starting_formation_id = None

        print(f"  Initial Home Formation ID: {home_starting_formation_id}")
        print(f"  Initial Away Formation ID: {away_starting_formation_id}")

        # Reconstruct initial player metadata (starters only for the plot)
        # This uses the already processed 'positional_role' and 'Is Starter' on df_processed
        # We need to make sure these are correctly derived from the *initial* lineup event
        cols_for_meta = ['playerName', 'Mapped Jersey Number', 'Mapped Position Number', 'positional_role', 'Is Starter', 'team_name']
        if all(c in df_processed.columns for c in cols_for_meta):
            # Get players who were starters in the first lineup event
            # This assumes 'Is Starter' and 'Mapped Position Number' are correctly set by preprocess
            # based on the *initial* typeId=34 event.
            potential_starters = df_processed[
                (df_processed['Is Starter'] == True) &
                (df_processed['Mapped Position Number'].between(1, 11))
            ].drop_duplicates(subset=['playerName', 'team_name'], keep='first') # Get their first state as starter

            initial_lineup_meta_home_df = potential_starters[potential_starters['team_name'] == HTEAM_NAME].copy()
            initial_lineup_meta_away_df = potential_starters[potential_starters['team_name'] == ATEAM_NAME].copy()
            print(f"  Extracted {len(initial_lineup_meta_home_df)} home starters for formation plot.")
            print(f"  Extracted {len(initial_lineup_meta_away_df)} away starters for formation plot.")
        else:
            print(f"Warning: Missing metadata columns for formation plot reconstruction.")

        # --- Extract Formation Change Log from df_processed (simplified) ---
        # This requires 'typeId' and your renamed formation ID column to be in df_processed
        if 'typeId' in df_processed.columns and FORMATION_ID_COL_NAME in df_processed.columns:
            formation_change_events = df_processed[df_processed['typeId'] == 34].sort_values('eventId')
            for team_n in [HTEAM_NAME, ATEAM_NAME]:
                all_formation_changes_log[team_n] = []
                team_form_changes = formation_change_events[formation_change_events['team_name'] == team_n]
                # Skip the first one as it's the starting formation
                for idx, row in team_form_changes.iloc[1:].iterrows():
                    try:
                        form_id = int(row[FORMATION_ID_COL_NAME])
                        time_m = int(row['timeMin'])
                        all_formation_changes_log[team_n].append(f"{time_m}' -> Form. {form_id}")
                    except: continue # Skip if conversion fails
        # --- End Formation Change Log ---


    # --- Calculate ALL Player Stats (Needed for Dashboard rankings) ---
    if any(p in plots_to_generate for p in ['dashboard', 'player_passmap', 'player_def_actions', 'player_received']):
        print("Calculating aggregated player stats...")
        player_stats_df = player_metrics.calculate_player_stats(
            df_processed.copy(), # Pass df with flags already added
            assist_qualifier_col=assist_qualifier_col # Pass original name (used internally if needed)
            # Add prog_pass_exclusions here if needed by calculate_player_stats
        )
        if not player_stats_df.empty:
            player_team_map = df_processed.drop_duplicates(subset=['playerName'])[['playerName', 'team_name']].set_index('playerName')['team_name']
            player_stats_df = player_stats_df.merge(player_team_map, left_index=True, right_index=True, how='left')
            print("Aggregated player stats calculated.")
        else:
            print("Warning: Failed to calculate player stats. Dashboard/player plots might be empty.")


    # --- Calculate Event-Specific DataFrames (Conditionally) ---
    # Passes & Subs (Now gets flags from df_processed)
    if any(p in plots_to_generate for p in ['heatmap', 'network', 'progressive', 'final_third', 'chances', 'defensive_block', 'hull', 'voronoi', 'dashboard', 'player_passmap', 'player_received', 'player_def_actions']):
        print("Processing pass data & sub list...")
        passes_df = pass_processing.get_passes_df(df_processed) # Should now contain flags
        sub_list = pass_processing.get_sub_list(df_processed)
        if not passes_df.empty:
            successful_passes = passes_df[passes_df['outcome'] == 'Successful'].copy()
            df_pass_home_succ = successful_passes[successful_passes['team_name'] == HTEAM_NAME]
            df_pass_away_succ = successful_passes[successful_passes['team_name'] == ATEAM_NAME]
        print("Pass data & sub list processed.")

    # Shots
    if any(p in plots_to_generate for p in ['shotmap', 'dashboard']):
        print("Calculating shot data...")
        shots_df, home_stats, away_stats = shot_metrics.calculate_shot_stats(df_processed, HTEAM_NAME, ATEAM_NAME, MATCH_HXG, MATCH_AXG, MATCH_HXGOT, MATCH_AXGOT)
        print("Shot data calculated.")

    # Progressive Passes
    if 'progressive' in plots_to_generate:
         print("Calculating progressive pass data...")
         prog_pass_exclusions = ['cross', 'Launch', 'ThrowIn']
         df_prog_passes_all, _ = pass_metrics.analyze_progressive_passes(df_processed, exclude_qualifiers=prog_pass_exclusions)
         if not df_prog_passes_all.empty:
             home_prog_passes = df_prog_passes_all[df_prog_passes_all['team_name'] == HTEAM_NAME].copy()
             away_prog_passes = df_prog_passes_all[df_prog_passes_all['team_name'] == ATEAM_NAME].copy()
             def calculate_team_prog_zone_stats(df_team_prog):
                 if df_team_prog.empty or 'y' not in df_team_prog.columns: return {'total': 0, 'left': 0, 'mid': 0, 'right': 0}
                 y_start = df_team_prog['y'].fillna(50); total = len(df_team_prog); right = (y_start < 33.33).sum(); mid = ((y_start >= 33.33) & (y_start < 66.67)).sum(); left = (y_start >= 66.67).sum()
                 return {'total': total, 'left': left, 'mid': mid, 'right': right}
             home_prog_zone_stats = calculate_team_prog_zone_stats(home_prog_passes)
             away_prog_zone_stats = calculate_team_prog_zone_stats(away_prog_passes)

    # Final Third Passes
    if 'final_third' in plots_to_generate:
        print("Calculating final third pass data...")
        default_ft_counts = {'zone14': 0, 'hs_left': 0, 'hs_right': 0, 'hs_total': 0, 'total_final_third': 0}
        # Ensure successful passes calculated before using them
        if 'df_pass_home_succ' in locals() and not df_pass_home_succ.empty: home_z14_df, home_lhs_df, home_rhs_df, home_ft_stats_dict = pass_metrics.analyze_final_third_passes(df_pass_home_succ)
        else: home_z14_df, home_lhs_df, home_rhs_df, home_ft_stats_dict = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), default_ft_counts.copy()
        if 'df_pass_away_succ' in locals() and not df_pass_away_succ.empty: away_z14_df, away_lhs_df, away_rhs_df, away_ft_stats_dict = pass_metrics.analyze_final_third_passes(df_pass_away_succ)
        else: away_z14_df, away_lhs_df, away_rhs_df, away_ft_stats_dict = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), default_ft_counts.copy()

    # High Turnovers
    if 'turnover' in plots_to_generate:
        print("Calculating high turnover data...")
        turnover_radius_meters = 40.0; pitch_len_meters = 105.0
        home_high_to_df, away_high_to_df, hto_count, ato_count = turnover_metrics.calculate_high_turnovers(df_processed, HTEAM_NAME, ATEAM_NAME, turnover_radius_meters, pitch_len_meters)

    # Chance Creation Passes (relies on flags being on df_processed)
    if 'chances' in plots_to_generate:
        print("Calculating chance creation pass data...")
        home_chances_df, away_chances_df = pass_metrics.analyze_chance_creation(df_processed, HTEAM_NAME, ATEAM_NAME, assist_qualifier_col=assist_qualifier_col)

    # Defensive Actions
    if any(p in plots_to_generate for p in ['defensive_block', 'hull', 'voronoi', 'dashboard', 'player_def_actions']):
        print("Calculating defensive action data...")
        def_action_types_to_use = None
        df_def_actions = defensive_metrics.get_defensive_actions(df_processed, defensive_action_types=def_action_types_to_use)
        if not df_def_actions.empty:
            home_def_agg_df = defensive_metrics.calculate_defensive_agg(df_def_actions, HTEAM_NAME)
            away_def_agg_df = defensive_metrics.calculate_defensive_agg(df_def_actions, ATEAM_NAME)

    # Shot Sequences
    if any(p in plots_to_generate for p in ['shot_sequence', 'patterns', 'binned_sequence']):
        print("Calculating shot sequence data...")
        df_shot_sequences = sequence_metrics.find_shot_sequences(
            df_processed,
            goal_mouth_y_col='Goal mouth y co-ordinate'
        )

    # --- Calculate Binned Sequence Stats if needed ---
    if 'binned_sequence' in plots_to_generate:
        if 'df_shot_sequences' in locals() and not df_shot_sequences.empty:
            print("Calculating binned sequence stats with dominant role pair...") # Updated print
            bin_grid = (7, 6)
            home_seq_df = df_shot_sequences[df_shot_sequences['team_name'] == HTEAM_NAME]
            df_bin_trans_home_pair, df_shot_origin_home = \
                sequence_metrics.calculate_binned_sequence_stats(home_seq_df, bins=bin_grid)
            away_seq_df = df_shot_sequences[df_shot_sequences['team_name'] == ATEAM_NAME]
            df_bin_trans_away_pair, df_shot_origin_away = \
                sequence_metrics.calculate_binned_sequence_stats(away_seq_df, bins=bin_grid)
        else: print("Skipping binned stats: No sequence data.")

    # --- Calculate Mean Position Data if needed ---
    if 'mean_pos' in plots_to_generate:
        print("Calculating median touch location data...")
        # Define event types to exclude from touch calculation
        exclude_types = ['Formation Change', 'Deleted event', 'End', 'Start', 'Save attempt'] # Add more?
        # Calculate aggregated locations
        player_loc_agg_all = player_metrics.calculate_median_touch_location(
            df_processed, exclude_event_types=exclude_types
        )
        if not player_loc_agg_all.empty:
            # Add team name for filtering (reuse map if player_stats_df was calculated)
            if not player_stats_df.empty and 'team_name' in player_stats_df.columns:
                 player_loc_agg_all = player_loc_agg_all.merge(
                     player_stats_df['team_name'], # Select only team_name column
                     left_on='playerName',
                     right_index=True,
                     how='left'
                 )
            else: # Fallback if player_stats_df wasn't calculated
                player_team_map = df_processed.drop_duplicates(subset=['playerName'])[['playerName', 'team_name']].set_index('playerName')['team_name']
                player_loc_agg_all = player_loc_agg_all.merge(player_team_map, on='playerName', how='left')

            home_loc_agg_df = player_loc_agg_all[player_loc_agg_all['team_name'] == HTEAM_NAME].copy()
            away_loc_agg_df = player_loc_agg_all[player_loc_agg_all['team_name'] == ATEAM_NAME].copy()

            # Get all touches per team for KDE plot background
            df_touches = df_processed[
                (~df_processed['type_name'].isin(exclude_types)) &
                (df_processed['x'].notna()) & (df_processed['y'].notna())
            ].copy()
            df_touches_home = df_touches[df_touches['team_name'] == HTEAM_NAME]
            df_touches_away = df_touches[df_touches['team_name'] == ATEAM_NAME]
        else:
             print("Warning: Failed to calculate median touch locations.")

    # --- Calculate PPDA if requested (or always if needed for reports) ---
    if 'ppda_actions' in plots_to_generate:
        print("\nCalculating PPDA...")
        # Use the determined def_action_ids_to_use
        ppda_pass_zone = config.DEFAULT_PPDA_PASS_ZONE_THRESH
        ppda_def_zone = config.DEFAULT_PPDA_DEF_ZONE_THRESH

        try: home_ppda = defensive_metrics.calculate_ppda_opta(df_processed, HTEAM_NAME, ATEAM_NAME, def_action_ids=def_action_ids_to_use, def_action_zone_thresh=ppda_def_zone, pass_zone_thresh=ppda_pass_zone)
        except Exception as e: print(f"Error calc home PPDA: {e}")
        try: away_ppda = defensive_metrics.calculate_ppda_opta(df_processed, ATEAM_NAME, HTEAM_NAME, def_action_ids=def_action_ids_to_use, def_action_zone_thresh=ppda_def_zone, pass_zone_thresh=ppda_pass_zone)
        except Exception as e: print(f"Error calc away PPDA: {e}")

        # Print results
        ppda_h_str = f"{home_ppda:.2f}" if pd.notna(home_ppda) and home_ppda != np.inf else "inf" if home_ppda == np.inf else "N/A"
        ppda_a_str = f"{away_ppda:.2f}" if pd.notna(away_ppda) and away_ppda != np.inf else "inf" if away_ppda == np.inf else "N/A"
        print(f"\n--- PPDA Results ---")
        print(f"{HTEAM_NAME}: {ppda_h_str}")
        print(f"{ATEAM_NAME}: {ppda_a_str}")
        print(f"--------------------")

    # --- Calculate Buildup Sequences if needed ---
    if 'buildup_sequence' in plots_to_generate:
        print("Calculating buildup sequence data...")
        # Ensure passes_df is calculated as it might have 'receiver' info
        if 'passes_df' not in locals() or passes_df.empty:
             print("  Dependency: Calculating pass data for receiver info...")
             passes_df = pass_processing.get_passes_df(df_processed) # Ensure flags are on df_processed

        df_with_receiver_info = df_processed.copy()
        if not passes_df.empty and 'receiver' in passes_df.columns and 'receiver_jersey_number' in passes_df.columns:
            # Merge receiver info from passes_df to df_processed based on eventId or index
            # This assumes passes_df was created from df_processed and shares index/eventId
            # A more robust way is to calculate receiver info directly in find_buildup_sequences if needed.
            df_with_receiver_info = pd.merge(
                df_with_receiver_info,
                passes_df[['eventId', 'receiver', 'receiver_jersey_number']], # Select relevant columns
                on='eventId',
                how='left'
            )
        else:
             print("Warning: passes_df or receiver info not available for buildup sequences.")
             # Create dummy columns if missing to prevent errors in find_buildup_sequences
             if 'receiver' not in df_with_receiver_info.columns: df_with_receiver_info['receiver'] = pd.NA
             if 'receiver_jersey_number' not in df_with_receiver_info.columns: df_with_receiver_info['receiver_jersey_number'] = pd.NA


        df_buildup_sequences = sequence_metrics.find_buildup_sequences(
            df_with_receiver_info, # Pass df with attempts to add receiver info
            start_x_thresh_deep=buildup_start_zone_x_max, # Start zone (defensive third)
            buildup_max_x_thresh=66.67
        )
    
    # --- Calculate Recovery to First Pass Data ---
    if 'recovery_pass' in plots_to_generate:
        print("Calculating recovery to first pass data...")
        # *** VERIFY Out of play column name if used ***
        df_recovery_first_passes = transition_metrics.find_recovery_to_first_pass(df_processed)

    # --- Calculate Defensive Transitions After Loss ---
    if 'defensive_transitions' in plots_to_generate or 'opponent_buildup_summary' in plots_to_generate:
        print("Calculating opponent buildup sequences after possession loss...")
        buildup_data = prepare_opponent_buildup_data(df_processed, HTEAM_NAME, ATEAM_NAME)
        df_away_buildup_after_home_loss = buildup_data['df_away_buildup_after_home_loss']
        df_home_buildup_after_away_loss = buildup_data['df_home_buildup_after_away_loss']
        home_loss_summary_table_data = buildup_data['home_loss_summary_table_data']
        away_loss_summary_table_data = buildup_data['away_loss_summary_table_data']
    
    # --- Calculate Offensive Transitions After Possession Loss ---
    if 'offensive_transitions' in plots_to_generate or 'try_dashboard' in plots_to_generate:
        print("Calculating offensive transitions after possession loss...")
        offensive_data = prepare_offensive_transitions_data(df_processed, HTEAM_NAME, ATEAM_NAME)
        df_home_offensive = offensive_data['df_home_transitions']
        df_away_offensive = offensive_data['df_away_transitions']
        home_offensive_summary = offensive_data['home_transition_summary']
        away_offensive_summary = offensive_data['away_transition_summary']

    if 'buildup_phase' in plots_to_generate:
        print("Calculating buildup sequences of each team ...")
        buildup_phase_data = prepare_offensive_buildups_data(df_processed, HTEAM_NAME, ATEAM_NAME, 'buildup_phase')
        df_home_buildup = buildup_phase_data['df_home_buildups']
        df_away_buildup = buildup_phase_data['df_away_buildups']
        home_buildup_summary_table_data = buildup_phase_data['home_buildups_summary']
        away_buildup_summary_table_data = buildup_phase_data['away_buildups_summary']
        print(f"Michele df_home_buildup: {df_home_buildup}")

    if 'set_piece' in plots_to_generate:
        print("Calculating buildup sequences of each team ...")
        buildup_phase_data = prepare_offensive_buildups_data(df_processed, HTEAM_NAME, ATEAM_NAME, 'set_piece')
        df_home_buildup = buildup_phase_data['df_home_buildups']
        df_away_buildup = buildup_phase_data['df_away_buildups']
        home_buildup_summary_table_data = buildup_phase_data['home_buildups_summary']
        away_buildup_summary_table_data = buildup_phase_data['away_buildups_summary']
        print(f"Michele df_home_buildup: {df_home_buildup}")

    # --- Calculate Cross Analysis Data if needed ---
    if 'crosses' in plots_to_generate:
        print("Calculating cross analysis data...")
        try:
            cross_data_dict = prepare_cross_analysis_data(df_processed, HTEAM_NAME, ATEAM_NAME)
            home_cross_df = cross_data_dict['home_crosses_df']
            home_cross_summary = cross_data_dict['home_cross_summary_data']
            home_total_crosses = cross_data_dict['home_total_crosses']
            away_cross_df = cross_data_dict['away_crosses_df']
            away_cross_summary = cross_data_dict['away_cross_summary_data']
            away_total_crosses = cross_data_dict['away_total_crosses']
            cross_summary_cols = cross_data_dict['summary_cols']
            print("  Cross analysis data calculated.")
        except Exception as e:
            print(f"  Error calculating cross data: {e}")
            # Ensure defaults remain so script doesn't break
            home_cross_df, away_cross_df = pd.DataFrame(), pd.DataFrame()
            home_cross_summary, away_cross_summary = [], []
            home_total_crosses, away_total_crosses = 0, 0
            cross_summary_cols = ["Metric", "Count"]


    # --- Generate Requested Plots Sequentially ---
    print("\n--- Generating Plots ---")

    # --- Heatmap Plot ---
    if 'heatmap' in plots_to_generate:
        if not passes_df.empty:
            print(f"Generating Pass Density & Heatmap Plots...")
            df_pass_home = passes_df[passes_df['team_name'] == HTEAM_NAME]
            df_pass_away = passes_df[passes_df['team_name'] == ATEAM_NAME]
            fig_heatmap_home, axes_heatmap_home = plt.subplots(1, 2, figsize=(10, 7)); fig_heatmap_home.suptitle(f'{HTEAM_NAME} - Pass Locations', fontsize=16)
            pitch_plots.plot_pass_density(axes_heatmap_home[0], df_pass_home, HTEAM_NAME, cmap=HOME_HEATMAP_CMAP, is_away_team=False)
            pitch_plots.plot_pass_heatmap(axes_heatmap_home[1], df_pass_home, HTEAM_NAME, cmap=HOME_HEATMAP_CMAP, is_away_team=False)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show();
            if SAVE_PLOTS: save_path_home = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_pass_location_plots_{HTEAM_SHORT_NAME}.png"); fig_heatmap_home.savefig(save_path_home, dpi=150, bbox_inches='tight'); print(f"Saved: {save_path_home}")
            plt.close(fig_heatmap_home)
            fig_heatmap_away, axes_heatmap_away = plt.subplots(1, 2, figsize=(10, 7)); fig_heatmap_away.suptitle(f'{ATEAM_NAME} - Pass Locations', fontsize=16)
            pitch_plots.plot_pass_density(axes_heatmap_away[0], df_pass_away, ATEAM_NAME, cmap=AWAY_HEATMAP_CMAP, is_away_team=True)
            pitch_plots.plot_pass_heatmap(axes_heatmap_away[1], df_pass_away, ATEAM_NAME, cmap=AWAY_HEATMAP_CMAP, is_away_team=True)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show();
            if SAVE_PLOTS: save_path_away = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_pass_location_plots_{ATEAM_SHORT_NAME}.png"); fig_heatmap_away.savefig(save_path_away, dpi=150, bbox_inches='tight'); print(f"Saved: {save_path_away}")
            plt.close(fig_heatmap_away)
        else: print("Skipping heatmap: Pass data unavailable.")

    # --- Network Plot ---
    if 'network' in plots_to_generate:
        if 'successful_passes' in locals() and not successful_passes.empty:
             print("Generating Pass Network Plot...")
             home_passes_between, home_avg_locs = pass_metrics.calculate_pass_network_data(successful_passes, HTEAM_NAME)
             away_passes_between, away_avg_locs = pass_metrics.calculate_pass_network_data(successful_passes, ATEAM_NAME)
             fig_network, axs_network = plt.subplots(1, 2, figsize=(20, 10), facecolor='white'); fig_network.suptitle(f'{HTEAM_NAME} vs {ATEAM_NAME} - Passing Networks', fontsize=20, fontweight='bold')
             pitch_plots.plot_pass_network(axs_network[0], home_passes_between, home_avg_locs, HTEAM_COLOR, HTEAM_NAME, sub_list, False)
             pitch_plots.plot_pass_network(axs_network[1], away_passes_between, away_avg_locs, ATEAM_COLOR, ATEAM_NAME, sub_list, True)
             plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show();
             if SAVE_PLOTS: save_path_network = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_pass_network_plots.png"); fig_network.savefig(save_path_network, dpi=150, bbox_inches='tight'); print(f"Saved: {save_path_network}")
             plt.close(fig_network)
        else: print("Skipping network: Successful pass data unavailable.")

    # --- Progressive Pass Plot ---
    if 'progressive' in plots_to_generate:
        if not home_prog_passes.empty or not away_prog_passes.empty:
             print("Generating Progressive Pass Plot...")
             fig_prog, axs_prog = plt.subplots(1, 2, figsize=(20, 8), facecolor='white'); fig_prog.suptitle(f'{HTEAM_NAME} vs {ATEAM_NAME} - Progressive Passes', fontsize=20, fontweight='bold')
             pitch_plots.plot_progressive_passes(axs_prog[0], home_prog_passes, home_prog_zone_stats, HTEAM_NAME, HTEAM_COLOR, False, prog_pass_exclusions)
             pitch_plots.plot_progressive_passes(axs_prog[1], away_prog_passes, away_prog_zone_stats, ATEAM_NAME, ATEAM_COLOR, True, prog_pass_exclusions)
             plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show();
             if SAVE_PLOTS: save_path_prog = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_progressive_pass_plots.png"); fig_prog.savefig(save_path_prog, dpi=150, bbox_inches='tight', facecolor=fig_prog.get_facecolor()); print(f"Saved: {save_path_prog}")
             plt.close(fig_prog)
        else: print("Skipping progressive pass plot: No data.")

    # --- Shot Map Plot ---
    if 'shotmap' in plots_to_generate:
        if not shots_df.empty:
             print("Generating Shot Map & Stats Bar Plot...")
             fig_shot, ax_shot = plt.subplots(figsize=(12, 8)); fig_shot.set_facecolor(config.BG_COLOR)
             pitch_plots.plot_shot_map_and_stats(ax_shot, shots_df, home_stats, away_stats, HTEAM_NAME, ATEAM_NAME, HTEAM_COLOR, ATEAM_COLOR, GAMEWEEK, LEAGUE, MATCH_DATE, config.BG_COLOR, config.LINE_COLOR)
             plt.show();
             if SAVE_PLOTS: save_path_shot = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_shot_map_stats.png"); fig_shot.savefig(save_path_shot, dpi=150, bbox_inches='tight', facecolor=fig_shot.get_facecolor()); print(f"Saved: {save_path_shot}")
             plt.close(fig_shot)
        else: print("Skipping shotmap: Shot data unavailable.")

    # --- Final Third Plot ---
    if 'final_third' in plots_to_generate:
        if home_ft_stats_dict is not None and away_ft_stats_dict is not None:
            if home_ft_stats_dict.get('total_final_third', 0) > 0 or away_ft_stats_dict.get('total_final_third', 0) > 0:
                 print("Generating Zone 14 & Half-Space Pass Plot...")
                 fig_ft, axs_ft = plt.subplots(1, 2, figsize=(20, 8), facecolor='white'); fig_ft.suptitle(f'{HTEAM_NAME} vs {ATEAM_NAME} - Final Third Entries (Successful Passes)', fontsize=20, fontweight='bold')
                 pitch_plots.plot_zone14_halfspace_map(axs_ft[0], home_z14_df, home_lhs_df, home_rhs_df, home_ft_stats_dict, HTEAM_NAME, HTEAM_COLOR, False, halfspace_color=HTEAM_COLOR)
                 pitch_plots.plot_zone14_halfspace_map(axs_ft[1], away_z14_df, away_lhs_df, away_rhs_df, away_ft_stats_dict, ATEAM_NAME, ATEAM_COLOR, True, halfspace_color=ATEAM_COLOR)
                 plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show();
                 if SAVE_PLOTS: save_path_ft = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_zone14_halfspace_plots.png"); fig_ft.savefig(save_path_ft, dpi=150, bbox_inches='tight', facecolor=fig_ft.get_facecolor()); print(f"Saved: {save_path_ft}")
                 plt.close(fig_ft)
            else: print("Skipping final third plot: No relevant passes found.")
        else: print("Skipping final third plot: Data unavailable.")

    # --- High Turnover Plot ---
    if 'turnover' in plots_to_generate:
        # Re-fetch params if not calculated earlier or set defaults
        turnover_radius_meters = 40.0; pitch_len_meters = 105.0; pitch_wid_meters=68.0
        if hto_count > 0 or ato_count > 0:
             print("Generating High Turnover Plot...")
             fig_to, ax_to = plt.subplots(figsize=(12, 8)); fig_to.set_facecolor(config.BG_COLOR)
             pitch_plots.plot_high_turnovers(ax_to, home_high_to_df, away_high_to_df, hto_count, ato_count, HTEAM_NAME, ATEAM_NAME, HTEAM_COLOR, ATEAM_COLOR, turnover_radius_meters, pitch_len_meters, pitch_wid_meters, config.BG_COLOR, config.LINE_COLOR)
             ax_to.set_title(f"High Turnovers ({turnover_radius_meters}m Radius)", fontsize=20, fontweight='bold'); plt.tight_layout(); plt.show();
             if SAVE_PLOTS: save_path_to = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_high_turnovers_meters.png"); fig_to.savefig(save_path_to, dpi=150, bbox_inches='tight', facecolor=fig_to.get_facecolor()); print(f"Saved: {save_path_to}")
             plt.close(fig_to)
        else: print("Skipping high turnover plot: No data.")

    # --- Chance Creation Plot ---
    if 'chances' in plots_to_generate:
        if not home_chances_df.empty or not away_chances_df.empty:
             print("Generating Chance Creation Plot...")
             fig_ch, axs_ch = plt.subplots(1, 2, figsize=(20, 10), facecolor='white'); fig_ch.suptitle(f'{HTEAM_NAME} vs {ATEAM_NAME} - Chance Creation Zones & Passes', fontsize=20, fontweight='bold')
             pitch_plots.plot_chance_creation(axs_ch[0], home_chances_df, HTEAM_NAME, HTEAM_COLOR, False)
             pitch_plots.plot_chance_creation(axs_ch[1], away_chances_df, ATEAM_NAME, ATEAM_COLOR, True)
             plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show();
             if SAVE_PLOTS: save_path_ch = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_chance_creation.png"); fig_ch.savefig(save_path_ch, dpi=150, bbox_inches='tight', facecolor=fig_ch.get_facecolor()); print(f"Saved: {save_path_ch}")
             plt.close(fig_ch)
        else: print("Skipping chance creation plot: No data.")

    # --- Defensive Block Plot ---
    if 'defensive_block' in plots_to_generate:
        if not home_def_agg_df.empty or not away_def_agg_df.empty:
             print("Generating Defensive Block Plot...")
             fig_db, axs_db = plt.subplots(1, 2, figsize=(20, 10), facecolor='white'); fig_db.suptitle(f'{HTEAM_NAME} vs {ATEAM_NAME} - Defensive Blocks', fontsize=20, fontweight='bold')
             df_def_home = df_def_actions[df_def_actions['team_name'] == HTEAM_NAME]
             df_def_away = df_def_actions[df_def_actions['team_name'] == ATEAM_NAME]
             show_scatter = True
             pitch_plots.plot_defensive_block(axs_db[0], df_def_home, home_def_agg_df, HTEAM_NAME, HTEAM_COLOR, sub_list, False, scatter_actions=show_scatter)
             pitch_plots.plot_defensive_block(axs_db[1], df_def_away, away_def_agg_df, ATEAM_NAME, ATEAM_COLOR, sub_list, True, scatter_actions=show_scatter)
             plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show();
             if SAVE_PLOTS: save_path_db = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_defensive_block.png"); fig_db.savefig(save_path_db, dpi=150, bbox_inches='tight', facecolor=fig_db.get_facecolor()); print(f"Saved: {save_path_db}")
             plt.close(fig_db)
        else: print("Skipping defensive block plot: No data.")

    # --- Defensive Hull Plot ---
    if 'hull' in plots_to_generate:
        if not home_def_agg_df.empty or not away_def_agg_df.empty:
             print("Generating Defensive Hull Plot...")
             fig_hull, axs_hull = plt.subplots(1, 2, figsize=(20, 10), facecolor='white'); fig_hull.suptitle(f'{HTEAM_NAME} vs {ATEAM_NAME} - Defensive Shape (Hull)', fontsize=20, fontweight='bold')
             pitch_plots.plot_defensive_hull(axs_hull[0], home_def_agg_df, HTEAM_NAME, HTEAM_COLOR, sub_list, False)
             pitch_plots.plot_defensive_hull(axs_hull[1], away_def_agg_df, ATEAM_NAME, ATEAM_COLOR, sub_list, True)
             plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show();
             if SAVE_PLOTS: save_path_hull = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_defensive_hull.png"); fig_hull.savefig(save_path_hull, dpi=150, bbox_inches='tight', facecolor=fig_hull.get_facecolor()); print(f"Saved: {save_path_hull}")
             plt.close(fig_hull)
        else: print("Skipping defensive hull plot: No data.")

    # --- Defensive Voronoi Plot ---
    if 'voronoi' in plots_to_generate:
        if not home_def_agg_df.empty or not away_def_agg_df.empty:
             print("Generating Defensive Voronoi Plot...")
             fig_vor, axs_vor = plt.subplots(1, 2, figsize=(20, 10), facecolor='white'); fig_vor.suptitle(f'{HTEAM_NAME} vs {ATEAM_NAME} - Defensive Coverage (Voronoi)', fontsize=20, fontweight='bold')
             pitch_plots.plot_defensive_voronoi(axs_vor[0], home_def_agg_df, HTEAM_NAME, HTEAM_COLOR, sub_list, False)
             pitch_plots.plot_defensive_voronoi(axs_vor[1], away_def_agg_df, ATEAM_NAME, ATEAM_COLOR, sub_list, True)
             plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show();
             if SAVE_PLOTS: save_path_vor = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_defensive_voronoi.png"); fig_vor.savefig(save_path_vor, dpi=150, bbox_inches='tight', facecolor=fig_vor.get_facecolor()); print(f"Saved: {save_path_vor}")
             plt.close(fig_vor)
        else: print("Skipping defensive Voronoi plot: No data.")

    # --- Player Dashboard Plot ---
    if 'dashboard' in plots_to_generate:
        # Check if necessary dataframes were calculated and are not empty
        if not player_stats_df.empty and 'passes_df' in locals() and not passes_df.empty and 'df_def_actions' in locals() and not df_def_actions.empty:
            print("\n--- Generating Player Dashboard ---")

            # --- Identify Players for the Dashboard ---
            top_passers_all = player_stats_df.sort_values('Offensive Pass Total', ascending=False).head(10)
            top_shooters_all = player_stats_df.sort_values('Shooting Seq Total', ascending=False).head(10)
            top_defenders_all = player_stats_df.sort_values('Defensive Actions Total', ascending=False).head(10)

            home_players = player_stats_df[player_stats_df['team_name'] == HTEAM_NAME]
            away_players = player_stats_df[player_stats_df['team_name'] == ATEAM_NAME]

            # --- Select Players for Maps based on args ---
            # Home Passer Map Player
            if h_passer_arg == "TOP" or h_passer_arg not in home_players.index:
                if not home_players.empty: top_home_passer_name = home_players.sort_values('Offensive Pass Total', ascending=False).index[0]
                else: top_home_passer_name = "N/A"
                if h_passer_arg != "TOP" and h_passer_arg not in home_players.index: print(f"Warning: Home passer '{h_passer_arg}' not found, using top passer.")
            else: top_home_passer_name = h_passer_arg

            # Away Passer Map Player
            if a_passer_arg == "TOP" or a_passer_arg not in away_players.index:
                if not away_players.empty: top_away_passer_name = away_players.sort_values('Offensive Pass Total', ascending=False).index[0]
                else: top_away_passer_name = "N/A"
                if a_passer_arg != "TOP" and a_passer_arg not in away_players.index: print(f"Warning: Away passer '{a_passer_arg}' not found, using top passer.")
            else: top_away_passer_name = a_passer_arg

            # Home Defender Map Player
            if h_defender_arg == "TOP" or h_defender_arg not in home_players.index:
                if not home_players.empty: top_home_defender_name = home_players.sort_values('Defensive Actions Total', ascending=False).index[0]
                else: top_home_defender_name = "N/A"
                if h_defender_arg != "TOP" and h_defender_arg not in home_players.index: print(f"Warning: Home defender '{h_defender_arg}' not found, using top defender.")
            else: top_home_defender_name = h_defender_arg

            # Away Defender Map Player
            if a_defender_arg == "TOP" or a_defender_arg not in away_players.index:
                if not away_players.empty: top_away_defender_name = away_players.sort_values('Defensive Actions Total', ascending=False).index[0]
                else: top_away_defender_name = "N/A"
                if a_defender_arg != "TOP" and a_defender_arg not in away_players.index: print(f"Warning: Away defender '{a_defender_arg}' not found, using top defender.")
            else: top_away_defender_name = a_defender_arg

            # Home Received Map Player (Target Forward)
            if h_receiver_arg == "TOP" or h_receiver_arg not in home_players.index:
                if not home_players.empty: target_home_forward = home_players.sort_values('Shooting Seq Total', ascending=False).index[0] # Default: Top shooter
                else: target_home_forward = "N/A"
                if h_receiver_arg != "TOP" and h_receiver_arg not in home_players.index: print(f"Warning: Home receiver '{h_receiver_arg}' not found, using default target.")
            else: target_home_forward = h_receiver_arg

            # Away Received Map Player (Target Forward)
            if a_receiver_arg == "TOP" or a_receiver_arg not in away_players.index:
                if not away_players.empty: target_away_forward = away_players.sort_values('Shooting Seq Total', ascending=False).index[0] # Default: Top shooter
                else: target_away_forward = "N/A"
                if a_receiver_arg != "TOP" and a_receiver_arg not in away_players.index: print(f"Warning: Away receiver '{a_receiver_arg}' not found, using default target.")
            else: target_away_forward = a_receiver_arg

            print(f"Dashboard Players Selected:")
            print(f"  H Passer Map: {top_home_passer_name}")
            print(f"  A Passer Map: {top_away_passer_name}")
            print(f"  H Receiver Map: {target_home_forward}")
            print(f"  A Receiver Map: {target_away_forward}")
            print(f"  H Defender Map: {top_home_defender_name}")
            print(f"  A Defender Map: {top_away_defender_name}")
            # --- End Player Selection ---

            # --- Filter Data for Specific Player Maps ---
            df_top_home_passer = passes_df[passes_df['playerName'] == top_home_passer_name].copy() if top_home_passer_name != "N/A" else pd.DataFrame()
            df_top_away_passer = passes_df[passes_df['playerName'] == top_away_passer_name].copy() if top_away_passer_name != "N/A" else pd.DataFrame()
            df_top_home_defender = df_def_actions[df_def_actions['playerName'] == top_home_defender_name].copy() if top_home_defender_name != "N/A" else pd.DataFrame()
            df_top_away_defender = df_def_actions[df_def_actions['playerName'] == top_away_defender_name].copy() if top_away_defender_name != "N/A" else pd.DataFrame()

            # --- Create 3x3 Grid ---
            fig_dash, axs_dash = plt.subplots(3, 3, figsize=(27, 22), facecolor=config.BG_COLOR)
            fig_dash.suptitle("Top Players Dashboard", fontsize=34, fontweight='bold', color=config.LINE_COLOR, y=0.98)
            if home_stats and away_stats:
                score_text = f"{HTEAM_NAME} {int(home_stats.get('goals',0))} - {int(away_stats.get('goals',0))} {ATEAM_NAME}"
                fig_dash.text(0.5, 0.94, score_text, fontsize=20, ha='center', va='top', color=config.LINE_COLOR) # Adjust y=0.94 and va='top'
            else:
                print("Warning: Shot stats unavailable, cannot display score.")

            # --- Populate Grid ---
            # Row 0: Passing Focus
            player_plots.plot_player_pass_map(axs_dash[0, 0], df_top_home_passer, top_home_passer_name, HTEAM_COLOR, False)
            player_plots.plot_passer_stats_bar(axs_dash[0, 1], top_passers_all, num_players=10)
            player_plots.plot_player_pass_map(axs_dash[0, 2], df_top_away_passer, top_away_passer_name, ATEAM_COLOR, True)

            # Row 1: Shooting Focus
            player_plots.plot_player_received_passes(axs_dash[1, 0], passes_df, target_home_forward, HTEAM_COLOR, False)
            player_plots.plot_shot_sequence_bar(axs_dash[1, 1], top_shooters_all, num_players=10)
            player_plots.plot_player_received_passes(axs_dash[1, 2], passes_df, target_away_forward, ATEAM_COLOR, True)

            # Row 2: Defending Focus
            player_plots.plot_player_defensive_actions(axs_dash[2, 0], df_top_home_defender, top_home_defender_name, HTEAM_COLOR, False)
            player_plots.plot_defender_stats_bar(axs_dash[2, 1], top_defenders_all, num_players=10)
            player_plots.plot_player_defensive_actions(axs_dash[2, 2], df_top_away_defender, top_away_defender_name, ATEAM_COLOR, True)

            # --- Final Adjustments ---
            plt.subplots_adjust(left=0.04, right=0.96, bottom=0.03, top=0.90, wspace=0.25, hspace=0.35) # Adjust spacing values as needed
            # plt.tight_layout(rect=[0, 0.01, 1, 0.93])
            plt.show() # Blocking show
            if SAVE_PLOTS:
                save_path_dash = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_player_dashboard.png")
                fig_dash.savefig(save_path_dash, dpi=150, bbox_inches='tight', facecolor=fig_dash.get_facecolor())
                print(f"Saved player dashboard plot: {save_path_dash}")
            plt.close(fig_dash)
        else:
             # Check which specific dataframe was missing
             missing_dfs = []
             if player_stats_df.empty: missing_dfs.append("player_stats_df")
             if 'passes_df' not in locals() or passes_df.empty: missing_dfs.append("passes_df")
             if 'df_def_actions' not in locals() or df_def_actions.empty: missing_dfs.append("df_def_actions")
             print(f"Skipping dashboard: Required dataframes unavailable: {', '.join(missing_dfs)}")

    # --- Shot Sequence Plot ---
    if 'shot_sequence' in plots_to_generate:
        if 'df_shot_sequences' in locals() and not df_shot_sequences.empty:
            print("\n--- Generating Shot Sequence Plots (Paginated with Summary) ---")

            # --- PARAMETERS FOR GRIDDING PER FIGURE ---
            PLOTS_PER_PAGE_MAX = 6 # Max pitches per page (e.g., 3x3)
            COLS_PITCHES = 3      # Columns for pitch plots
            # --- END PARAMETERS ---

            for team_name, team_short_name, team_color, is_away in [
                    (HTEAM_NAME, HTEAM_SHORT_NAME, HTEAM_COLOR, False),
                    (ATEAM_NAME, ATEAM_SHORT_NAME, ATEAM_COLOR, True)
                ]:
                team_sequences_df = df_shot_sequences[df_shot_sequences['team_name'] == team_name]
                unique_seq_ids = team_sequences_df['sequence_id'].unique()
                num_total_team_sequences = len(unique_seq_ids)

                if num_total_team_sequences == 0:
                    print(f"No shot sequences found for {team_name}.")
                    continue

                # --- *** 1. Calculate OVERALL Combined Play Type Counts for the Team *** ---
                overall_combined_play_type_counts = {}
                for seq_id in unique_seq_ids:
                    sequence_data = team_sequences_df[team_sequences_df['sequence_id'] == seq_id]
                    if sequence_data.empty: continue
                    shot_event_series = sequence_data[~sequence_data['type_name'].isin(['Pass', None])].iloc[-1]
                    last_pass_event_series = None
                    if len(sequence_data) > 1:
                        # Get the event directly preceding the shot
                        event_before_shot = sequence_data.iloc[-2]
                        if event_before_shot.get('type_name') == 'Pass':
                            last_pass_event_series = event_before_shot

                    # Use the helper function from sequence_metrics
                    combined_desc = sequence_metrics.get_shot_context_description(shot_event_series, last_pass_event_series)
                    overall_combined_play_type_counts[combined_desc] = overall_combined_play_type_counts.get(combined_desc, 0) + 1
                # --- *** End Overall Count Calculation *** ---

                # --- Loop to create multiple figures (pages) if needed ---
                print(f"Generating paginated shot sequence plots for {team_name}...")
                for fig_num, i in enumerate(range(0, num_total_team_sequences, PLOTS_PER_PAGE_MAX)):
                    current_chunk_seq_ids = unique_seq_ids[i : i + PLOTS_PER_PAGE_MAX]
                    if not current_chunk_seq_ids.any(): continue
                    num_sequences_in_chunk = len(current_chunk_seq_ids)

                    # --- Grid Layout with GridSpec ---
                    # +1 row for the table spanning the top
                    rows_for_pitches = int(np.ceil(num_sequences_in_chunk / COLS_PITCHES))
                    total_rows_in_fig = rows_for_pitches + 1 # Add a row for the table

                    fig_height_per_pitch_row = 5.0 # Height for each row of pitch plots
                    fig_width_per_pitch_col = 6.5  # Width for each column of pitch plots
                    table_height_ratio = 0.15 # Fraction of total height for the table row
                                             # Or fixed height: table_height_inches = 2

                    # Calculate figure size
                    fig_width = COLS_PITCHES * fig_width_per_pitch_col
                    fig_height = (rows_for_pitches * fig_height_per_pitch_row) / (1 - table_height_ratio) # Adjust total height

                    fig_seq = plt.figure(figsize=(fig_width, fig_height), facecolor=config.BG_COLOR)
                    # Define GridSpec: first row for table, rest for pitches
                    gs = GridSpec(total_rows_in_fig, COLS_PITCHES, figure=fig_seq,
                                  height_ratios=[table_height_ratio] + [ (1-table_height_ratio)/rows_for_pitches ] * rows_for_pitches, # Distribute remaining height
                                  hspace=0.4, wspace=0.15) # Adjust spacing

                    fig_seq.suptitle(f"{team_name} ({team_short_name}) - Shot Sequences (Page {fig_num + 1})", fontsize=18, fontweight='bold', color=config.LINE_COLOR, y=0.99)

                    # --- Add Summary Table to the First Row (Spanning Columns) ---
                    if overall_combined_play_type_counts:
                        ax_table = fig_seq.add_subplot(gs[0, 1]) # Span all columns in the first row (gs[0, :])
                        ax_table.axis('off')
                        # ax_table.set_title(f"Summary: Shots by Play Context (Page {fig_num+1})", fontsize=12, loc='center') # Or overall?

                        sorted_play_types = sorted(overall_combined_play_type_counts.items(), key=lambda item: item[1], reverse=True)
                        max_count_table = sorted_play_types[0][1] if sorted_play_types else 0
                        table_data_display = [[ptype, count] for ptype, count in sorted_play_types]
                        col_labels_table = ["Play Context", "Count"]
                        cell_colors_table = []
                        for ptype, count in table_data_display:
                            if count == max_count_table and max_count_table > 0: cell_colors_table.append([team_color, team_color])
                            else: cell_colors_table.append(['lightgrey', 'lightgrey'])

                        tab = ax_table.table(cellText=table_data_display, colLabels=col_labels_table, loc='center',
                                             cellLoc='left', cellColours=cell_colors_table)
                        tab.auto_set_font_size(False); tab.set_fontsize(9); tab.scale(1, 1.2)
                    # --- End Summary Table ---

                    # --- Plot each sequence in the grid below the table ---
                    ax_idx_offset = 0 # Start plotting pitches from the first cell after the table row
                    for plot_idx, seq_id in enumerate(current_chunk_seq_ids):
                        # Calculate current row and col for pitches grid
                        current_pitch_row = plot_idx // COLS_PITCHES
                        current_pitch_col = plot_idx % COLS_PITCHES
                        if current_pitch_row < rows_for_pitches: # Ensure we don't exceed pitch rows
                             ax = fig_seq.add_subplot(gs[current_pitch_row + 1, current_pitch_col]) # +1 to row index because table is in row 0
                             sequence_data = team_sequences_df[team_sequences_df['sequence_id'] == seq_id]
                             pitch_plots.plot_individual_shot_sequence(ax, sequence_data, team_name, team_color, seq_id)
                             if is_away: ax.invert_xaxis(); ax.invert_yaxis()
                             ax_idx_offset +=1

                    # Hide any unused axes in the pitch grid part
                    for j in range(ax_idx_offset, rows_for_pitches * COLS_PITCHES):
                        # Calculate row and col for remaining axes
                        remaining_row = j // COLS_PITCHES
                        remaining_col = j % COLS_PITCHES
                        if remaining_row < rows_for_pitches : # only if valid row index for pitches
                            fig_seq.add_subplot(gs[remaining_row + 1, remaining_col]).axis('off')


                    # plt.tight_layout(rect=[0, 0.01, 1, 0.94]) # tight_layout can conflict with GridSpec
                    plt.show() # Show each figure
                    if SAVE_PLOTS:
                        save_path_seq = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_shot_sequences_{team_short_name}_p{fig_num + 1}.png")
                        fig_seq.savefig(save_path_seq, dpi=150, bbox_inches='tight', facecolor=fig_seq.get_facecolor())
                        print(f"Saved shot sequence plot: {save_path_seq}")
                    plt.close(fig_seq)
                # --- End Loop for Figures ---
        else:
            print("Skipping shot sequence plot: No sequence data calculated.")
    # --- End Shot Sequence Plot ---

    # --- *** Analyze Sequence Patterns *** ---
    if 'patterns' in plots_to_generate:
        if not df_shot_sequences.empty:
            print("\n--- Analyzing & Plotting Shot Sequence Patterns ---")
            n_last_3 = 3 # Look at last 3 events
            n_last_2 = 2 # Look at last 2 events
            n_last_1 = 1 # Look at last 1 event (shot origin/shooter)
            top_n_patterns_display = 5 # How many top patterns to show in each bar chart

            # --- Create Figure for Patterns ---
            # One figure per team, 2 rows (zone, player), 3 cols (N=3, N=2, N=1)
            for team_name, team_short_name, team_color, is_away in [
                    (HTEAM_NAME, HTEAM_SHORT_NAME, HTEAM_COLOR, False),
                    (ATEAM_NAME, ATEAM_SHORT_NAME, ATEAM_COLOR, True) # Use specific Away Color
                ]:

                print(f"\n--- {team_name} Patterns ---")
                team_sequences = df_shot_sequences[df_shot_sequences['team_name'] == team_name]

                if team_sequences.empty:
                    print("No sequences for this team to analyze patterns.")
                    continue

                # Calculate all pattern types first
                zone_patt_3 = sequence_metrics.find_sequence_patterns(team_sequences, 'zone', n_last_3)
                zone_patt_2 = sequence_metrics.find_sequence_patterns(team_sequences, 'zone', n_last_2)
                zone_patt_1 = sequence_metrics.find_sequence_patterns(team_sequences, 'zone', n_last_1)
                player_patt_3 = sequence_metrics.find_sequence_patterns(team_sequences, 'player', n_last_3)
                player_patt_2 = sequence_metrics.find_sequence_patterns(team_sequences, 'player', n_last_2)
                player_patt_1 = sequence_metrics.find_sequence_patterns(team_sequences, 'player', n_last_1)
                role_patt_3 = sequence_metrics.find_sequence_patterns(team_sequences, 'role', n_last_3)
                role_patt_2 = sequence_metrics.find_sequence_patterns(team_sequences, 'role', n_last_2)
                role_patt_1 = sequence_metrics.find_sequence_patterns(team_sequences, 'role', n_last_1)

                # Create Figure
                fig_patt, axs_patt = plt.subplots(2, 3, figsize=(20, 15), facecolor=config.BG_COLOR) # 3 rows, 3 cols
                fig_patt.suptitle(f"{team_name} ({team_short_name}) - Common Shot Sequence Patterns", fontsize=20, fontweight='bold', color=config.LINE_COLOR)

                # Plot Zone Patterns (Row 0)
                pattern_plots.plot_pattern_bar_chart(axs_patt[0, 0], zone_patt_3, f"Zone (Last {n_last_3})", team_color, top_n_patterns_display)
                pattern_plots.plot_pattern_bar_chart(axs_patt[0, 1], zone_patt_2, f"Zone (Last {n_last_2})", team_color, top_n_patterns_display)
                pattern_plots.plot_pattern_bar_chart(axs_patt[0, 2], zone_patt_1, f"Shot Zone (Last {n_last_1})", team_color, top_n_patterns_display)

                # Plot Player Patterns (Row 1)
                # pattern_plots.plot_pattern_bar_chart(axs_patt[1, 0], player_patt_3, f"Player (Last {n_last_3})", team_color, top_n_patterns_display)
                # pattern_plots.plot_pattern_bar_chart(axs_patt[1, 1], player_patt_2, f"Player (Last {n_last_2})", team_color, top_n_patterns_display)
                # pattern_plots.plot_pattern_bar_chart(axs_patt[1, 2], player_patt_1, f"Shooter (Last {n_last_1})", team_color, top_n_patterns_display)

                # *** Plot Role Patterns (Row 2) ***
                pattern_plots.plot_pattern_bar_chart(axs_patt[1, 0], role_patt_3, f"Role (Last {n_last_3})", team_color, top_n_patterns_display)
                pattern_plots.plot_pattern_bar_chart(axs_patt[1, 1], role_patt_2, f"Role (Last {n_last_2})", team_color, top_n_patterns_display)
                pattern_plots.plot_pattern_bar_chart(axs_patt[1, 2], role_patt_1, f"Shooter Role (Last {n_last_1})", team_color, top_n_patterns_display)

                # Adjust layout
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect for suptitle
                plt.show() # Blocking show

                if SAVE_PLOTS:
                    save_path_patt = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_sequence_patterns_{team_short_name}.png")
                    fig_patt.savefig(save_path_patt, dpi=150, bbox_inches='tight', facecolor=fig_patt.get_facecolor())
                    print(f"Saved sequence pattern plot: {save_path_patt}")
                plt.close(fig_patt)

        else:
            print("Skipping pattern analysis: No sequence data calculated.")
    # --- *** End Pattern Analysis *** ---

    # --- *** Binned Sequence Flow Plot *** ---
    if 'binned_sequence' in plots_to_generate:
        # Check using the new df names
        if ('df_bin_trans_home_pair' in locals() and 'df_shot_origin_home' in locals() and \
           'df_bin_trans_away_pair' in locals() and 'df_shot_origin_away' in locals()) and \
           (not df_bin_trans_home_pair.empty or not df_shot_origin_home.empty or \
            not df_bin_trans_away_pair.empty or not df_shot_origin_away.empty):
            print("\nGenerating Binned Sequence Flow Plot (with Role Pairs)...") # Updated print
            fig_bins, axs_bins = plt.subplots(1, 2, figsize=(20, 8), facecolor='white')
            fig_bins.suptitle("Aggregated Shot Buildup Flow & Dominant Role Pairs", fontsize=20, fontweight='bold')
            bin_grid_plot = (7, 6); min_trans_plot = 2

            # *** Pass the correct DataFrame to the plotting function ***
            pitch_plots.plot_binned_sequence_flow(
                axs_bins[0], df_bin_trans_home_pair, df_shot_origin_home, # Pass df with pairs
                HTEAM_NAME, HTEAM_COLOR, bins=bin_grid_plot,
                min_transition_count=min_trans_plot, shot_origin_cmap='Reds', annotate_roles=True
            )
            pitch_plots.plot_binned_sequence_flow(
                axs_bins[1], df_bin_trans_away_pair, df_shot_origin_away, # Pass df with pairs
                ATEAM_NAME, ATEAM_COLOR, bins=bin_grid_plot,
                min_transition_count=min_trans_plot, shot_origin_cmap='Blues', annotate_roles=True
            )
            axs_bins[1].invert_xaxis(); axs_bins[1].invert_yaxis()

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            if SAVE_PLOTS:
                save_path_bins = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_binned_sequence_flow.png")
                fig_bins.savefig(save_path_bins, dpi=150, bbox_inches='tight', facecolor=fig_bins.get_facecolor())
                print(f"Saved binned sequence flow plot: {save_path_bins}")
            plt.close(fig_bins)
        else:
            print("Skipping binned sequence plot: Data unavailable or calculation failed.")
    # --- *** End Binned Sequence Flow Plot *** ---

    # --- *** Mean Positions Plot *** ---
    if 'mean_pos' in plots_to_generate:
        if not home_loc_agg_df.empty or not away_loc_agg_df.empty:
            print("\n--- Generating Mean Positions Plot ---")
            fig_mp, axs_mp = plt.subplots(1, 2, figsize=(20, 10), facecolor='white') # Adjust size
            fig_mp.suptitle(f'{HTEAM_NAME} vs {ATEAM_NAME} - Mean Player Positions', fontsize=20, fontweight='bold')

            # Plot Home Team Positions
            pitch_plots.plot_mean_positions(
                axs_mp[0], home_loc_agg_df, df_touches_home,
                HTEAM_NAME, HTEAM_COLOR, sub_list, is_away_team=False, annotate_role=True
            )
            # Plot Away Team Positions
            pitch_plots.plot_mean_positions(
                axs_mp[1], away_loc_agg_df, df_touches_away,
                ATEAM_NAME, ATEAM_COLOR, sub_list, is_away_team=True, annotate_role=True
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            if SAVE_PLOTS:
                save_path_mp = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_mean_positions.png")
                fig_mp.savefig(save_path_mp, dpi=150, bbox_inches='tight', facecolor=fig_mp.get_facecolor())
                print(f"Saved mean positions plot: {save_path_mp}")
            plt.close(fig_mp)
        else:
            print("Skipping mean positions plot: Data unavailable.")
    # --- *** End Mean Positions Plot *** ---

    # --- *** Pressure Map Plot *** ---
    event_map_for_plots = {} # Initialize empty
    if 'ppda_actions' in plots_to_generate: # Only load if needed
        try:
            from src.utils import mapping_loader
            event_map_for_plots = mapping_loader.load_opta_event_mapping(config.OPTA_EVENTS_XLSX)
            if not event_map_for_plots: print("Warning: Event mapping empty.")
        except ImportError: print("Warning: mapping_loader not found.")
        except Exception as e: print(f"Warning: Error loading event mappings: {e}")

    if 'ppda_actions' in plots_to_generate:
        if not df_processed.empty:
            print("\n--- Generating PPDA Defensive Actions Map ---")
            fig_ppda, axs_ppda = plt.subplots(1, 2, figsize=(20, 8), facecolor='white') # Adjust size
            fig_ppda.suptitle("PPDA Defensive Actions Location", fontsize=20, fontweight='bold')

            # Use the same threshold used in PPDA calculation
            ppda_def_action_threshold = config.DEFAULT_PPDA_DEF_ZONE_THRESH # Or the variable used above

            # Plot Home Team Actions in Zone
            pitch_plots.plot_ppda_actions( # Call renamed function
                axs_ppda[0], df_processed, HTEAM_NAME, HTEAM_COLOR,
                action_ids_to_plot=def_action_ids_to_use, # Use the consistent list
                def_action_zone_thresh=ppda_def_action_threshold, # Pass threshold
                plot_type='both', kde_cmap='viridis', is_away_team=False, # kde_cmap options: https://matplotlib.org/stable/tutorials/colors/colormaps.html
                event_mapping=event_map_for_plots
            )
            # Plot Away Team Actions in Zone
            pitch_plots.plot_ppda_actions( # Call renamed function
                axs_ppda[1], df_processed, ATEAM_NAME, ATEAM_COLOR,
                action_ids_to_plot=def_action_ids_to_use, # Use the consistent list
                def_action_zone_thresh=ppda_def_action_threshold, # Pass threshold
                plot_type='both', kde_cmap='Blues', is_away_team=True, # kde_cmap options: https://matplotlib.org/stable/tutorials/colors/colormaps.html
                event_mapping=event_map_for_plots
            )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
            if SAVE_PLOTS:
                save_path_ppda = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_ppda_actions_map.png")
                fig_ppda.savefig(save_path_ppda, dpi=150, bbox_inches='tight', facecolor=fig_ppda.get_facecolor())
                print(f"Saved PPDA actions map plot: {save_path_ppda}")
            plt.close(fig_ppda)
        else:
            print("Skipping PPDA actions map plot: Processed data unavailable.")
    # --- *** End PPDA Actions Plot *** ---

    # --- *** Buildup Sequence Plot *** ---
    if 'buildup_sequence' in plots_to_generate:
        if 'df_buildup_sequences' in locals() and not df_buildup_sequences.empty:
            print(f"\n--- Generating Buildup Sequence Plots (Filtered by Zones) ---")
            print(f"  Filtering: Start X <= {buildup_start_zone_x_max}, "
                  f"End X >= {buildup_end_zone_x_min}, "
                  f"End Y between {buildup_end_zone_y_bounds[0]}-{buildup_end_zone_y_bounds[1]}")

            # --- *** FILTER SEQUENCES BASED ON END LOCATION OF LAST EVENT *** ---
            filtered_sequence_ids = []
            if 'buildup_sequence_id' in df_buildup_sequences.columns:
                for seq_id, group in df_buildup_sequences.groupby('buildup_sequence_id'):
                    if group.empty: continue
                    last_event = group.iloc[-1] # Last event of the sequence
                    # Check if the end coordinates of the last event meet criteria
                    # Ensure end_x and end_y are numeric before comparison
                    end_x = pd.to_numeric(last_event.get('end_x'), errors='coerce')
                    end_y = pd.to_numeric(last_event.get('end_y'), errors='coerce')

                    if pd.notna(end_x) and pd.notna(end_y):
                        if (end_x >= buildup_end_zone_x_min and
                            end_y >= buildup_end_zone_y_bounds[0] and
                            end_y <= buildup_end_zone_y_bounds[1]):
                            # Also ensure the first pass of the sequence started deep (already handled by find_buildup_sequences)
                            filtered_sequence_ids.append(seq_id)
            else:
                 print("Warning: 'buildup_sequence_id' not found in df_buildup_sequences.")

            if not filtered_sequence_ids:
                print("No buildup sequences found matching the specified start/end zone criteria.")
            else:
                selected_sequences_df = df_buildup_sequences[df_buildup_sequences['buildup_sequence_id'].isin(filtered_sequence_ids)]
                print(f"Found {len(filtered_sequence_ids)} sequences matching zone criteria.")

                # --- *** PARAMETERS FOR GRIDDING PER FIGURE *** ---
                PLOTS_PER_FIGURE = 9
                COLS_PER_FIGURE = 3
                # --- *** END PARAMETERS *** ---

                for team_name, team_short_name, team_color, opponent_color, is_away in [
                        (HTEAM_NAME, HTEAM_SHORT_NAME, HTEAM_COLOR, ATEAM_COLOR, False),
                        (ATEAM_NAME, ATEAM_SHORT_NAME, ATEAM_COLOR, HTEAM_COLOR, True)
                    ]:
                    team_filtered_sequences_df = selected_sequences_df[selected_sequences_df['team_name'] == team_name]
                    unique_seq_ids_to_plot = team_filtered_sequences_df['buildup_sequence_id'].unique()
                    num_sequences_to_plot = len(unique_seq_ids_to_plot)

                    if num_sequences_to_plot == 0:
                        print(f"No matching buildup sequences to plot for {team_name}.")
                        continue

                    # --- Loop to create multiple figures if needed ---
                    for fig_num, i in enumerate(range(0, num_sequences_to_plot, PLOTS_PER_FIGURE)):
                        current_chunk_seq_ids = unique_seq_ids_to_plot[i : i + PLOTS_PER_FIGURE]
                        if not current_chunk_seq_ids.any(): continue

                        num_sequences_in_chunk = len(current_chunk_seq_ids)
                        print(f"  Plotting figure {fig_num + 1} for {team_name} with {num_sequences_in_chunk} sequences...")

                        rows = int(np.ceil(num_sequences_in_chunk / COLS_PER_FIGURE))
                        fig_height_per_row = 6; fig_width_per_col = 8
                        fig_buildup, axes_buildup = plt.subplots(rows, COLS_PER_FIGURE, figsize=(COLS_PER_FIGURE * fig_width_per_col, rows * fig_height_per_row), facecolor=config.BG_COLOR, squeeze=False)
                        axes_buildup = axes_buildup.flatten()

                        fig_buildup.suptitle(f"{team_name} ({team_short_name}) - Buildups to Dangerous Zones (Page {fig_num + 1})", fontsize=18, fontweight='bold', color=config.LINE_COLOR, y=0.99)

                        for plot_idx, seq_id in enumerate(current_chunk_seq_ids):
                            if plot_idx < len(axes_buildup):
                                ax = axes_buildup[plot_idx]
                                sequence_data = team_filtered_sequences_df[team_filtered_sequences_df['buildup_sequence_id'] == seq_id]
                                # --- Pass Zone Boundaries to Plotting Function ---
                                pitch_plots.plot_buildup_sequence(
                                    ax, sequence_data, team_name, team_color, opponent_color, seq_id,
                                    target_zone_x_min=buildup_end_zone_x_min,
                                    target_zone_y_bounds=buildup_end_zone_y_bounds,
                                    target_zone_color='lightgreen' # Example color
                                )
                                # --- End Pass Zone Boundaries ---
                                if is_away: ax.invert_xaxis(); ax.invert_yaxis()

                        for j in range(num_sequences_in_chunk, len(axes_buildup)): axes_buildup[j].axis('off')
                        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.03, top=0.93, wspace=0.1, hspace=0.2)
                        plt.show()
                        if SAVE_PLOTS:
                            save_path_buildup = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_filtered_buildup_seq_{team_short_name}_p{fig_num + 1}.png")
                            fig_buildup.savefig(save_path_buildup, dpi=150, bbox_inches='tight', facecolor=fig_buildup.get_facecolor())
                            print(f"Saved filtered buildup sequence plot: {save_path_buildup}")
                        plt.close(fig_buildup)
                    # --- End Loop for Figures ---
        else:
            print("Skipping buildup sequence plot: No sequence data calculated.")
    # --- End Buildup Sequence Plot ---

    # --- Formation and Substitutions Plot *** ---
    if 'formation' in plots_to_generate:
        # Check if we have data for at least one team
        can_plot_home = not initial_lineup_meta_home_df.empty and home_starting_formation_id is not None
        can_plot_away = not initial_lineup_meta_away_df.empty and away_starting_formation_id is not None

        if can_plot_home or can_plot_away:
            print(f"\n--- Generating Combined Formation Plot ---")
            # Create a figure with 1 row, 2 columns for side-by-side
            fig_form_combined, axs_form = plt.subplots(1, 2, figsize=(16, 9), facecolor=config.BG_COLOR, squeeze=False) # squeeze=False
            axs_form = axs_form.flatten() # axs_form will be [ax_home, ax_away]

            fig_form_combined.suptitle("Starting Formations & Substitutions", fontsize=18, fontweight='bold', y=0.98)

            # Plot Home Team
            if can_plot_home:
                pitch_plots.plot_formations_and_subs(
                    axs_form[0], # Pass the first axes
                    df_processed, initial_lineup_meta_home_df,
                    HTEAM_NAME, HTEAM_COLOR, home_starting_formation_id,
                    formation_changes=all_formation_changes_log.get(HTEAM_NAME, []),
                    is_away_team=False
                )
            else:
                axs_form[0].text(0.5, 0.5, f"{HTEAM_NAME}\nData N/A", ha='center', va='center')
                axs_form[0].axis('off')
                print(f"Skipping formation plot for {HTEAM_NAME}: Missing data.")

            # Plot Away Team
            if can_plot_away:
                pitch_plots.plot_formations_and_subs(
                    axs_form[1], # Pass the second axes
                    df_processed, initial_lineup_meta_away_df,
                    ATEAM_NAME, ATEAM_COLOR, away_starting_formation_id,
                    formation_changes=all_formation_changes_log.get(ATEAM_NAME, []),
                    is_away_team=True
                )
            else:
                axs_form[1].text(0.5, 0.5, f"{ATEAM_NAME}\nData N/A", ha='center', va='center')
                axs_form[1].axis('off')
                print(f"Skipping formation plot for {ATEAM_NAME}: Missing data.")

            plt.tight_layout(rect=[0, 0, 1, 1]) # Adjust rect for suptitle
            plt.show()
            if SAVE_PLOTS:
                save_path_form_comb = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_formations_combined.png")
                fig_form_combined.savefig(save_path_form_comb, dpi=150, bbox_inches='tight')
                print(f"Saved combined formation plot: {save_path_form_comb}")
            plt.close(fig_form_combined)
        else:
            print("Skipping combined formation plot: No data for either team.")
    # --- End Formation Plot ---

    # --- *** Recovery to First Pass Plot (by Zone) *** ---
    if 'recovery_pass' in plots_to_generate:
        if 'df_recovery_first_passes' in locals() and not df_recovery_first_passes.empty:
            print("\n--- Generating Combined Recovery to First Pass Plot (by Zone) ---")

            # Create a figure with 2 rows, 3 columns
            fig_rec_combined, axs_rec_combined = plt.subplots(2, 3, figsize=(22, 12), facecolor=config.BG_COLOR) # Adjust figsize
            fig_rec_combined.suptitle("First Pass Direction After Recovery by Zone",
                                      fontsize=18, fontweight='bold', color=config.LINE_COLOR, y=0.97)

            teams_data = [
                (HTEAM_NAME, HTEAM_SHORT_NAME, HTEAM_COLOR, False), # Home Team Data
                (ATEAM_NAME, ATEAM_SHORT_NAME, ATEAM_COLOR, True)  # Away Team Data
            ]
            zones = ["Defensive Third", "Middle Third", "Attacking Third"]

            for row_idx, (team_name, team_short_name, team_color, is_away) in enumerate(teams_data):
                team_recovery_df = df_recovery_first_passes[df_recovery_first_passes['team_name'] == team_name]

                if team_recovery_df.empty:
                    print(f"No recovery-first-pass sequences for {team_name} to plot in combined grid.")
                    # Optionally fill empty subplots with a message
                    for col_idx in range(3):
                        ax = axs_rec_combined[row_idx, col_idx]
                        ax.text(0.5, 0.5, f"{team_name}\nNo Data", ha='center', va='center', fontsize=10, color='grey')
                        ax.axis('off')
                    continue # Skip to the next team if no data

                for col_idx, zone_name in enumerate(zones):
                    ax = axs_rec_combined[row_idx, col_idx] # Get the specific subplot axes
                    zone_specific_df = team_recovery_df[team_recovery_df['recovery_zone'] == zone_name]

                    # Call the existing plotting function for each zone
                    pitch_plots.plot_recovery_first_pass(
                        ax, zone_specific_df, team_name, team_color, zone_name # Pass team_name for title
                    )
                    # Invert axes for the away team's entire row AFTER plotting
                    if is_away:
                        ax.invert_xaxis()
                        ax.invert_yaxis() # If you want them attacking "up" when on the left

            # Adjust layout to prevent overlap and fit suptitle
            plt.tight_layout(rect=[0, 0.01, 1, 0.94])
            plt.show()

            if SAVE_PLOTS:
                save_path_rec_comb = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_recovery_first_pass_combined.png")
                fig_rec_combined.savefig(save_path_rec_comb, dpi=150, bbox_inches='tight', facecolor=fig_rec_combined.get_facecolor())
                print(f"Saved combined recovery first pass plot: {save_path_rec_comb}")
            plt.close(fig_rec_combined)
        else:
            print("Skipping combined recovery to first pass plot: No data.")
    # --- *** End Recovery to First Pass Plot *** ---

    # --- *** Defensive Transitions After Loss Plot (Paginated by Loss Zone) *** ---
    if 'defensive_transitions' in plots_to_generate:
        # ATEAM buildup after HTEAM loss
        if 'df_away_buildup_after_home_loss' in locals() and not df_away_buildup_after_home_loss.empty:
            print(f"\n--- Generating {ATEAM_NAME} Buildup Plots (after {HTEAM_NAME} loss) ---")
            plot_defensive_transitions_with_summary(
                df_sequences=df_away_buildup_after_home_loss,
                df_summary=home_loss_summary_table_data,
                lost_team_name=HTEAM_NAME,
                buildup_team_name=ATEAM_NAME,
                buildup_team_color=ATEAM_COLOR,
                is_buildup_team_away=True,
                fig_prefix="home_defensive_transitions_after_loss",
                loss_side_short=f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}",
                output_dir=OUTPUT_VIZ_DIR,
                save_plots=SAVE_PLOTS,
                team_short_home=HTEAM_SHORT_NAME,
                team_short_away=ATEAM_SHORT_NAME
            )
        else:
            print(f"No buildup sequences by {ATEAM_NAME} found after {HTEAM_NAME} lost possession.")

        # HTEAM buildup after ATEAM loss
        if 'df_home_buildup_after_away_loss' in locals() and not df_home_buildup_after_away_loss.empty:
            print(f"\n--- Generating {HTEAM_NAME} Buildup Plots (after {ATEAM_NAME} loss) ---")
            plot_defensive_transitions_with_summary(
                df_sequences=df_home_buildup_after_away_loss,
                df_summary=away_loss_summary_table_data,
                lost_team_name=ATEAM_NAME,
                buildup_team_name=HTEAM_NAME,
                buildup_team_color=HTEAM_COLOR,
                is_buildup_team_away=False,
                fig_prefix="away_defensive_transitions_after_loss",
                loss_side_short=f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}",
                output_dir=OUTPUT_VIZ_DIR,
                save_plots=SAVE_PLOTS,
                team_short_home=HTEAM_SHORT_NAME,
                team_short_away=ATEAM_SHORT_NAME
            )
        else:
            print(f"No buildup sequences by {HTEAM_NAME} found after {ATEAM_NAME} lost possession.")
    
    # --- End Opponent Buildup After Loss Plot ---

    # --- *** Offensive Transitions After Possession Gain (Paginated by Loss Zone) *** ---
    if 'offensive_transitions' in plots_to_generate:
    # Home team (after regaining possession from away)
        if 'df_home_buildup_after_away_loss' in locals() and not df_home_offensive.empty:
            print(f"\n--- Generating {HTEAM_NAME} Offensive Transitions (after {ATEAM_NAME} loss) ---")
            plot_offensive_transitions_with_summary(
                df_home_offensive, home_offensive_summary,
                gaining_team_name = HTEAM_NAME, 
                losing_team_name= ATEAM_NAME,
                team_color= HTEAM_COLOR,
                is_away_team=False,
                fig_prefix="offensive_transitions_home",
                team_short_home=HTEAM_SHORT_NAME,
                team_short_away=ATEAM_SHORT_NAME,
                output_dir=OUTPUT_VIZ_DIR,
                save_plots=SAVE_PLOTS
            )
        else:
            print(f"No offensive transitions by {HTEAM_NAME} found after gain possession.")

        # Away team (after regaining possession from home)
        if 'df_away_buildup_after_home_loss' in locals() and not df_away_offensive .empty:
            print(f"\n--- Generating {HTEAM_NAME} Buildup Plots (after {ATEAM_NAME} loss) ---")
            plot_offensive_transitions_with_summary(
                df_away_offensive, away_offensive_summary,
                gaining_team_name = ATEAM_NAME, 
                losing_team_name = HTEAM_NAME,
                team_color= ATEAM_COLOR,
                is_away_team=True,
                fig_prefix="offensive_transitions_away",
                team_short_home=HTEAM_SHORT_NAME,
                team_short_away=ATEAM_SHORT_NAME,
                output_dir=OUTPUT_VIZ_DIR,
                save_plots=SAVE_PLOTS
            )
        else:
            print(f"No offensive transitions by {ATEAM_NAME} found after gain possession.")
    
    # --- End Offensive Transitions After Possession Gain Plot ---

    if 'try_dashboard' in plots_to_generate:
        print(f"--- Generating Offensive Dashboard for {HTEAM_NAME} ---")
        plot_offensive_dashboard_with_sequences_multipage(
        df_sequences=df_away_offensive,
        summary_table_df=away_offensive_summary,
        team_name=ATEAM_NAME,
        opponent_name=HTEAM_NAME,
        save_path="output/plots",
        save_plots=True
    )

    # --- *** Plot Summary Tables *** ---
    if 'opponent_buildup_summary' in plots_to_generate:
        print("\n--- Generating Opponent Buildup Summary Tables ---")

        def generate_buildup_table(df, fig_title, save_path, team_color):
            if df is None or df.empty:
                print(f"No summary data to plot: {fig_title}")
                return

            fig, ax = plt.subplots(figsize=(9, max(3, len(df) * 0.35 + 1.5)), facecolor=config.BG_COLOR)
            ax.axis('off')
            ax.set_title(fig_title, fontsize=12, fontweight='bold', pad=15)

            # Copy and format data
            df_plot = df.copy()

            # Rename headers
            col_rename = {
                'loss_zone': 'Loss Zone',
                'sequence_outcome_type': 'Outcome',
                'count': 'Count',
                'avg_opp_passes_before_regain': 'Avg. Passes Before Regain'
            }
            df_plot = df_plot.rename(columns=col_rename)

            # Ensure numeric values for heatmap
            if 'Avg. Passes Before Regain' in df_plot.columns:
                df_plot['Avg. Passes Before Regain'] = pd.to_numeric(df_plot['Avg. Passes Before Regain'], errors='coerce')
                valid_vals = df_plot['Avg. Passes Before Regain'].dropna()
                if not valid_vals.empty:
                    vmin = valid_vals.min()
                    vmax = min(valid_vals.max(), 12)
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                else:
                    norm = None
            else:
                norm = None

            cmap = cm.get_cmap('cool') 
            table_cols = list(col_rename.values())
            table_values = df_plot[table_cols].values.tolist()

            # Build color and bold matrix
            cell_colors = []
            bold_cells = set()

            for i, row in df_plot.iterrows():
                row_colors = []
                outcome_text = str(row['Outcome']).lower().strip()
                is_goal = "goal" in outcome_text
                for j, col in enumerate(table_cols):
                    val = row[col]
                    if col == 'Avg. Passes Before Regain' and pd.notna(val) and norm:
                        rgba = cmap(norm(val))
                        hex_color = colors.to_hex(rgba)
                        row_colors.append(hex_color)
                    elif is_goal:
                        row_colors.append('red')
                        # bold_cells.add((i, j))
                    elif "shot conceded" in outcome_text or "chance conceded" in outcome_text:
                        row_colors.append('orange')
                        # bold_cells.add((i + 1, j))
                    elif "possession regained" in outcome_text:
                        row_colors.append('lightgreen')
                    else:
                        row_colors.append('lightgrey')
                cell_colors.append(row_colors)

            # Column widths
            default_widths = {
                'Loss Zone': 0.25,
                'Outcome': 0.40,
                'Count': 0.10,
                'Avg. Passes Before Regain': 0.25
            }
            col_widths = [default_widths.get(col, 0.20) for col in table_cols]

            # Draw table
            tab = ax.table(cellText=table_values,
                        colLabels=table_cols,
                        colWidths=col_widths,
                        cellColours=cell_colors,
                        loc='center', cellLoc='center')

            # Bold specific cells
            for (r, c) in bold_cells:
                tab[r, c].set_text_props(weight='bold')

            tab.auto_set_font_size(False)
            tab.set_fontsize(10)
            tab.scale(1.2, 1.5)

            # Center headers and values
            for i, col in enumerate(table_cols):
                tab[0, i].set_text_props(ha='center', weight='bold')
                for row in range(len(table_values)):
                    tab[row + 1, i].set_text_props(ha='center')

            plt.tight_layout(pad=0.5)
            plt.show()

            if SAVE_PLOTS:
                fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
                print(f"Saved Summary Table: {save_path}")
            plt.close(fig)


        # Home table (ATEAM buildup after HTEAM loses possession)
        generate_buildup_table(
            df=home_loss_summary_table_data,
            fig_title=f"{ATEAM_NAME} Buildup Outcomes After {HTEAM_NAME} Loss",
            save_path=os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_home_loss_summary_table.png"),
            team_color=ATEAM_COLOR
        )

        # Away table (HTEAM buildup after ATEAM loses possession)
        generate_buildup_table(
            df=away_loss_summary_table_data,
            fig_title=f"{HTEAM_NAME} Buildup Outcomes After {ATEAM_NAME} Loss",
            save_path=os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_away_loss_summary_table.png"),
            team_color=HTEAM_COLOR
        )
        
    # --- Buildup Sequences ---
    if 'buildup_phase' in plots_to_generate:
        # Home team (after regaining possession from away)
        if 'df_home_buildup' in locals() and not df_home_buildup.empty:
            print(f"\n--- Generating {HTEAM_NAME} Buildup sequences ---")
            plot_buildup_phases_with_summary(
                df_home_buildup, home_buildup_summary_table_data,
                attacking_team_name = HTEAM_NAME, 
                defending_team_name= ATEAM_NAME,
                team_color= HTEAM_COLOR,
                is_away_team=False,
                fig_prefix="buildup_sequences_home",
                team_short_home=HTEAM_SHORT_NAME,
                team_short_away=ATEAM_SHORT_NAME,
                output_dir=OUTPUT_VIZ_DIR,
                save_plots=SAVE_PLOTS
            )
        else:
            print(f"No Buildups by {HTEAM_NAME} found.")

        # Away team (after regaining possession from home)
        if 'df_away_buildup' in locals() and not df_away_buildup .empty:
            print(f"\n--- Generating {ATEAM_NAME} Buildup sequences ---")
            plot_buildup_phases_with_summary(
                df_away_buildup, away_buildup_summary_table_data,
                attacking_team_name = ATEAM_NAME, 
                defending_team_name = HTEAM_NAME,
                team_color= ATEAM_COLOR,
                is_away_team=True,
                fig_prefix="buildup_sequences_away",
                team_short_home=HTEAM_SHORT_NAME,
                team_short_away=ATEAM_SHORT_NAME,
                output_dir=OUTPUT_VIZ_DIR,
                save_plots=SAVE_PLOTS
            )
        else:
            print(f"No Buildups by {ATEAM_NAME} found.")
    
    # --- End Buildup Sequences ---

    # --- Set Pieces Sequences ---
    if 'set_piece' in plots_to_generate:
        # Home team (after regaining possession from away)
        if 'df_home_buildup' in locals() and not df_home_buildup.empty:
            print(f"\n--- Generating {HTEAM_NAME} Set Pieces sequences ---")
            plot_buildup_phases_with_summary(
                df_home_buildup, home_buildup_summary_table_data,
                attacking_team_name = HTEAM_NAME, 
                defending_team_name= ATEAM_NAME,
                team_color= HTEAM_COLOR,
                is_away_team=False,
                fig_prefix="setpieces_home",
                team_short_home=HTEAM_SHORT_NAME,
                team_short_away=ATEAM_SHORT_NAME,
                output_dir=OUTPUT_VIZ_DIR,
                save_plots=SAVE_PLOTS
            )
        else:
            print(f"No Set Pieces by {HTEAM_NAME} found.")

        # Away team (after regaining possession from home)
        if 'df_away_buildup' in locals() and not df_away_buildup .empty:
            print(f"\n--- Generating {ATEAM_NAME} Buildup sequences ---")
            plot_buildup_phases_with_summary(
                df_away_buildup, away_buildup_summary_table_data,
                attacking_team_name = ATEAM_NAME, 
                defending_team_name = HTEAM_NAME,
                team_color= ATEAM_COLOR,
                is_away_team=True,
                fig_prefix="setpieces_away",
                team_short_home=HTEAM_SHORT_NAME,
                team_short_away=ATEAM_SHORT_NAME,
                output_dir=OUTPUT_VIZ_DIR,
                save_plots=SAVE_PLOTS
            )
        else:
            print(f"No Set Pieces by {ATEAM_NAME} found.")
    
    # --- End Set Pieces ---

    # --- Cross Analysis Plot ---
    if 'crosses' in plots_to_generate:
        if (not home_cross_df.empty and home_total_crosses > 0) or \
           (not away_cross_df.empty and away_total_crosses > 0) :
            print("Generating Cross Analysis Plots...")

            # Home Team Cross Plot
            if not home_cross_df.empty and home_total_crosses > 0:
                fig_cross_h = plt.figure(figsize=(10, 12), facecolor=config.BG_COLOR) # Adjusted figsize for table
                gs_h = fig_cross_h.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.1)
                ax_pitch_h = fig_cross_h.add_subplot(gs_h[0, 0])
                ax_table_h = fig_cross_h.add_subplot(gs_h[1, 0])
                ax_table_h.axis('off') # Important to turn off table axes decorations

                pitch_plots.plot_cross_heatmap_and_summary(
                    fig=fig_cross_h,
                    ax_pitch=ax_pitch_h,
                    ax_table=ax_table_h,
                    team_crosses_df=home_cross_df,
                    total_crosses=home_total_crosses,
                    attacking_team_name=HTEAM_NAME,
                    summary_data=home_cross_summary,
                    summary_cols=cross_summary_cols,
                    pitch_type='opta', # Explicitly Opta
                    heatmap_cmap=HOME_HEATMAP_CMAP,
                    pitch_background_color=config.BG_COLOR, # Pass from main
                    pitch_line_color=config.LINE_COLOR,     # Pass from main
                    text_color=config.LINE_COLOR           # Pass from main
                )
                # fig_cross_h.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for suptitle
                plt.show()
                if SAVE_PLOTS:
                    save_path_ch = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_cross_analysis_{HTEAM_SHORT_NAME}.png")
                    fig_cross_h.savefig(save_path_ch, dpi=150, bbox_inches='tight', facecolor=fig_cross_h.get_facecolor())
                    print(f"Saved: {save_path_ch}")
                plt.close(fig_cross_h)
            else:
                print(f"  No plottable crosses for {HTEAM_NAME}.")

            # Away Team Cross Plot
            if not away_cross_df.empty and away_total_crosses > 0:
                fig_cross_a = plt.figure(figsize=(10, 12), facecolor=config.BG_COLOR)
                gs_a = fig_cross_a.add_gridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.1)
                ax_pitch_a = fig_cross_a.add_subplot(gs_a[0, 0])
                ax_table_a = fig_cross_a.add_subplot(gs_a[1, 0])
                ax_table_a.axis('off')

                # For Opta, away team attacking direction is usually L-R like home.
                # If you wanted to visually flip their plot to attack left, you'd invert axes AFTER plotting.
                # However, heatmap data is based on absolute coordinates.
                # Transformation of away_cross_df's x,y (100-x, 100-y) would be needed
                # in prepare_cross_analysis_data if you want heatmap bins to reflect attacking left.
                # Current setup: Away heatmap also shows absolute Opta locations.
                pitch_plots.plot_cross_heatmap_and_summary(
                    fig=fig_cross_a,
                    ax_pitch=ax_pitch_a,
                    ax_table=ax_table_a,
                    team_crosses_df=away_cross_df,
                    total_crosses=away_total_crosses,
                    attacking_team_name=ATEAM_NAME,
                    summary_data=away_cross_summary,
                    summary_cols=cross_summary_cols,
                    pitch_type='opta',
                    heatmap_cmap=AWAY_HEATMAP_CMAP,
                    pitch_background_color=config.BG_COLOR,
                    pitch_line_color=config.LINE_COLOR,
                    text_color=config.LINE_COLOR
                )
                # If you want the away team to appear attacking Left on the pitch display
                # ax_pitch_a.invert_xaxis()
                # ax_pitch_a.invert_yaxis()
                # This would visually flip the pitch lines, but the heatmap data is absolute.

                # fig_cross_a.tight_layout(rect=[0, 0, 1, 0.96])
                plt.show()
                if SAVE_PLOTS:
                    save_path_ca = os.path.join(OUTPUT_VIZ_DIR, f"{HTEAM_SHORT_NAME}_{ATEAM_SHORT_NAME}_cross_analysis_{ATEAM_SHORT_NAME}.png")
                    fig_cross_a.savefig(save_path_ca, dpi=150, bbox_inches='tight', facecolor=fig_cross_a.get_facecolor())
                    print(f"Saved: {save_path_ca}")
                plt.close(fig_cross_a)
            else:
                print(f"  No plottable crosses for {ATEAM_NAME}.")
        else:
            print("Skipping cross analysis plot: No cross data available for plotting.")

    # End of Crosses


    print("\n--- Match Analysis Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate visualizations for a football match.")
    parser.add_argument("--match", default=MATCH_ID, help=f"Match ID base filename (e.g., EVE_WHU). Default: {MATCH_ID}")
    parser.add_argument("--plot", nargs='+',
                        choices=['formation', 'heatmap', 'network', 'progressive', 'shotmap',
                                 'final_third', 'turnover', 'chances',
                                 'defensive_block', 'hull', 'voronoi',
                                 'dashboard', 'shot_sequence',
                                 'binned_sequence', 'mean_pos', 'ppda_actions', 
                                 'buildup_sequence', 'recovery_pass', 'defensive_transitions', 'opponent_buildup_summary', 
                                 'offensive_transitions', 'try_dashboard', 'buildup_phase', 'set_piece', 'crosses', 'all'],
                        default=['all'],
                        help="Specify which plot(s) to generate. Use 'all' for default set.")
    
    # --- New Arguments for Player Selection ---
    parser.add_argument("--h-passer", default="TOP", help="Name of home player for pass map (default: top passer)")
    parser.add_argument("--a-passer", default="TOP", help="Name of away player for pass map (default: top passer)")
    parser.add_argument("--h-receiver", default="TOP", help="Name of home player for received pass map (default: top shooter)")
    parser.add_argument("--a-receiver", default="TOP", help="Name of away player for received pass map (default: top shooter)")
    parser.add_argument("--h-defender", default="TOP", help="Name of home player for defensive actions map (default: top defender)")
    parser.add_argument("--a-defender", default="TOP", help="Name of away player for defensive actions map (default: top defender)")
    parser.add_argument("--def-actions", nargs='+', type=int, default=None, # Expect space-separated integers
                        help="List of Opta type IDs to consider as defensive/pressure actions "
                             f"(default: {config.DEFAULT_PPDA_DEF_ACTION_IDS})")
    parser.add_argument("--buildup-start-x", type=float, default=33.3,
                        help="Max X-coordinate for the start of a buildup sequence (default: 33.3, defensive third).")
    parser.add_argument("--buildup-end-x", type=float, default=66.7,
                        help="Min X-coordinate for the end of a buildup sequence (default: 66.7, attacking third).")
    parser.add_argument("--buildup-end-y", type=float, nargs=2, default=[20, 80],
                        metavar=('Y_MIN', 'Y_MAX'),
                        help="Y-coordinate bounds (min max) for the end of a buildup sequence (default: 20 80, central channel).")
    # --- End New Arguments ---

    args = parser.parse_args()
    plots_requested = args.plot

    if 'all' in plots_requested:
        # Define the complete set for 'all'
        plots_to_generate = ['formation', 'heatmap', 'network', 'progressive', 'shotmap',
                             'final_third', 'turnover', 'chances',
                             'defensive_block', 'hull', 'voronoi',
                             'dashboard', 'shot_sequence', 'binned_sequence','mean_pos',
                             'ppda_actions','buildup_sequence', 'recovery_pass', 'defensive_transitions',
                             'opponent_buildup_summary', 'offensive_transitions', 'try_dashboard', 'buildup_phase', 'set_piece', 'crosses']
        print(f"'all' specified, generating: {', '.join(plots_to_generate)}")
    else:
        plots_to_generate = plots_requested

    run_analysis(
        match_id=args.match,
        plots_to_generate=plots_to_generate,
        def_action_ids_arg=args.def_actions,
        buildup_start_zone_x_max=args.buildup_start_x,
        buildup_end_zone_x_min=args.buildup_end_x,
        buildup_end_zone_y_bounds=args.buildup_end_y,
        h_passer_arg=args.h_passer,
        a_passer_arg=args.a_passer,
        h_receiver_arg=args.h_receiver,
        a_receiver_arg=args.a_receiver,
        h_defender_arg=args.h_defender,
        a_defender_arg=args.a_defender
    )