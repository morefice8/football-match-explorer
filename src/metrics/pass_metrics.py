# src/metrics/pass_metrics.py
import logging
logger = logging.getLogger(__name__)

import numpy as np
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
    logger.debug(f"Calculating pass network data for {team_name}...")
    # Filter the passes DataFrame for the specified team
    team_passes_df = passes_df[passes_df['team_name'] == team_name].copy()

    if team_passes_df.empty:
        logger.warning(f"Warning: No passes found for team {team_name}.")
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
    logger.info(f"Calculated average locations for {len(average_locs_and_count_df)} players.")


    # --- Calculate passes between player pairs ---
    # Select relevant columns for pairing
    # Ensure 'receiver' column exists and handle potential NaN values before processing
    if 'receiver' not in team_passes_df.columns:
         logger.warning("Error: 'receiver' column not found in passes data. Cannot calculate pairs.")
         return pd.DataFrame(), average_locs_and_count_df # Return what we have so far

    # Only receiver attributions that passed the temporal, team and spatial
    # validation may form a network edge. This remains backwards-compatible
    # with older datasets that do not expose receiver_is_reliable.
    if 'receiver_is_reliable' in team_passes_df.columns:
        reliable_receiver_mask = team_passes_df['receiver_is_reliable'].fillna(False).astype(bool)
        team_passes_df = team_passes_df[reliable_receiver_mask].copy()

    # Drop rows where receiver is NaN as they cannot form a pair.
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

    logger.info(f"Calculated {len(passes_between_df)} links between players.")

    # Return both the pair data and the individual player average locations
    return passes_between_df, average_locs_and_count_df

# --- Progressive Passes ---
# This function identifies progressive passes based on Opta definitions and calculates counts and percentages per vertical third of the pitch.
def _legacy_analyze_progressive_passes(df_input,
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
    logger.debug("Analyzing progressive passes...")

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

PROGRESSIVE_RESULT_COLUMNS = [
    'is_progressive_attempt',
    'is_progressive',
    'progressive_distance_m',
    'progressive_threshold_m',
    'progressive_phase',
    'progressive_channel',
    'progressive_is_open_play',
    'progressive_exclusion_reason',
]

# Opta qualifier names vary slightly between mapping-file versions. Each tuple
# lists accepted aliases for the same non-open-play action.
PROGRESSIVE_EXCLUSION_ALIASES = (
    ('cross', 'Cross'),
    ('ThrowIn', 'Throw-in', 'Throw in'),
    ('Corner taken',),
    ('Free kick taken', 'Freekick taken'),
    ('Goal kick', 'Goal kick taken'),
)


def _flag_mask(df, aliases):
    """Return a mask for a flag that may use any of the supplied aliases."""
    mask = pd.Series(False, index=df.index)
    for column in aliases:
        if column not in df.columns:
            continue
        values = df[column]
        numeric = pd.to_numeric(values, errors='coerce')
        text = values.fillna('').astype(str).str.strip().str.lower()
        mask |= numeric.eq(1) | text.isin({'true', 'yes', 'y'})
    return mask


def classify_progressive_passes(
    df_input,
    pitch_length_meters=105.0,
    pitch_width_meters=68.0,
    exclude_qualifiers=None,
):
    """Classify progressive attempts and completions transparently.

    Progression is the reduction in straight-line distance to the centre of
    the opponent's goal, not simply ``end_x - x``. The required gain is 30 m
    within the own half, 15 m when crossing halfway and 10 m within the
    opposition half. Crosses and restarts are excluded; open-play long passes
    remain eligible. Both successful and unsuccessful attempts are retained.
    """
    if df_input is None or df_input.empty:
        base_columns = list(df_input.columns) if df_input is not None else []
        return pd.DataFrame(columns=base_columns + PROGRESSIVE_RESULT_COLUMNS)
    if pitch_length_meters <= 0 or pitch_width_meters <= 0:
        raise ValueError('Pitch dimensions must be positive.')

    classified = df_input.copy()
    for column in PROGRESSIVE_RESULT_COLUMNS:
        classified[column] = (
            False
            if column.startswith('is_') or column == 'progressive_is_open_play'
            else pd.NA
        )

    if 'type_name' in classified.columns:
        pass_mask = classified['type_name'].fillna('').astype(str).str.lower().eq('pass')
    elif 'typeId' in classified.columns:
        pass_mask = pd.to_numeric(classified['typeId'], errors='coerce').eq(1)
    else:
        return classified

    coordinates = {}
    for column in ('x', 'y', 'end_x', 'end_y'):
        if column not in classified.columns:
            classified.loc[pass_mask, 'progressive_exclusion_reason'] = 'missing_coordinates'
            return classified
        coordinates[column] = pd.to_numeric(classified[column], errors='coerce')

    valid_coordinates = pass_mask.copy()
    for values in coordinates.values():
        valid_coordinates &= values.between(0, 100, inclusive='both')

    excluded = pd.Series(False, index=classified.index)
    exclusion_reason = pd.Series(pd.NA, index=classified.index, dtype='object')
    reason_aliases = (
        ('cross', PROGRESSIVE_EXCLUSION_ALIASES[0]),
        ('throw_in', PROGRESSIVE_EXCLUSION_ALIASES[1]),
        ('corner', PROGRESSIVE_EXCLUSION_ALIASES[2]),
        ('free_kick', PROGRESSIVE_EXCLUSION_ALIASES[3]),
        ('goal_kick', PROGRESSIVE_EXCLUSION_ALIASES[4]),
    )
    for reason, aliases in reason_aliases:
        current = _flag_mask(classified, aliases)
        exclusion_reason = exclusion_reason.mask(current & exclusion_reason.isna(), reason)
        excluded |= current

    # Keep the public argument for callers that need additional exclusions.
    for qualifier in exclude_qualifiers or []:
        current = _flag_mask(classified, (qualifier,))
        exclusion_reason = exclusion_reason.mask(
            current & exclusion_reason.isna(), str(qualifier)
        )
        excluded |= current

    x = coordinates['x']
    y = coordinates['y']
    end_x = coordinates['end_x']
    end_y = coordinates['end_y']
    x_scale = pitch_length_meters / 100.0
    y_scale = pitch_width_meters / 100.0
    start_goal_distance = np.hypot((100.0 - x) * x_scale, (50.0 - y) * y_scale)
    end_goal_distance = np.hypot(
        (100.0 - end_x) * x_scale, (50.0 - end_y) * y_scale
    )
    distance_gained = start_goal_distance - end_goal_distance

    own_half = (x <= 50) & (end_x <= 50)
    crosses_halfway = (x <= 50) & (end_x > 50)
    opposition_half = (x > 50) & (end_x > 50)
    threshold = pd.Series(np.nan, index=classified.index, dtype='float64')
    threshold.loc[own_half] = 30.0
    threshold.loc[crosses_halfway] = 15.0
    threshold.loc[opposition_half] = 10.0

    phase = pd.Series(pd.NA, index=classified.index, dtype='object')
    phase.loc[own_half] = 'Own half'
    phase.loc[crosses_halfway] = 'Across halfway'
    phase.loc[opposition_half] = 'Opposition half'

    channel = pd.Series('Central', index=classified.index, dtype='object')
    channel.loc[y < (100 / 3)] = 'Right'
    channel.loc[y >= (200 / 3)] = 'Left'

    open_play = pass_mask & valid_coordinates & ~excluded
    attempt = open_play & threshold.notna() & distance_gained.ge(threshold)
    outcome = classified.get('outcome', pd.Series('', index=classified.index))
    outcome_text = outcome.fillna('').astype(str).str.strip().str.lower()
    outcome_numeric = pd.to_numeric(outcome, errors='coerce')
    successful = outcome_text.eq('successful') | outcome_numeric.eq(1)

    classified.loc[pass_mask, 'progressive_distance_m'] = distance_gained.round(1)
    classified.loc[pass_mask, 'progressive_threshold_m'] = threshold
    classified.loc[pass_mask, 'progressive_phase'] = phase
    classified.loc[pass_mask, 'progressive_channel'] = channel
    classified.loc[pass_mask, 'progressive_is_open_play'] = open_play
    classified.loc[pass_mask, 'progressive_exclusion_reason'] = exclusion_reason
    classified.loc[
        pass_mask & ~valid_coordinates, 'progressive_exclusion_reason'
    ] = 'missing_coordinates'
    classified['is_progressive_attempt'] = attempt.fillna(False).astype(bool)
    classified['is_progressive'] = (attempt & successful).fillna(False).astype(bool)
    classified['progressive_is_open_play'] = (
        classified['progressive_is_open_play'].fillna(False).astype(bool)
    )
    return classified


def progressive_pass_summary(passes_df):
    """Return volume, completion, distance gained and channel usage."""
    default = {
        'attempted': 0,
        'successful': 0,
        'unsuccessful': 0,
        'completion_pct': 0.0,
        'total_progression_m': 0.0,
        'average_progression_m': 0.0,
        'main_channel': 'N/A',
        'channel_counts': {'Left': 0, 'Central': 0, 'Right': 0},
    }
    if (
        passes_df is None
        or passes_df.empty
        or 'is_progressive_attempt' not in passes_df.columns
    ):
        return default

    attempts = passes_df[
        passes_df['is_progressive_attempt'].fillna(False).astype(bool)
    ].copy()
    if attempts.empty:
        return default
    completed = attempts[attempts['is_progressive'].fillna(False).astype(bool)]
    channel_counts = attempts.get(
        'progressive_channel', pd.Series(dtype='object')
    ).value_counts()
    channels = {
        channel: int(channel_counts.get(channel, 0))
        for channel in ('Left', 'Central', 'Right')
    }
    main_channel = max(channels, key=channels.get) if any(channels.values()) else 'N/A'
    progression = pd.to_numeric(
        completed.get('progressive_distance_m'), errors='coerce'
    )
    attempted = len(attempts)
    successful = len(completed)
    return {
        'attempted': int(attempted),
        'successful': int(successful),
        'unsuccessful': int(attempted - successful),
        'completion_pct': successful / attempted * 100.0 if attempted else 0.0,
        'total_progression_m': float(progression.sum()) if not progression.empty else 0.0,
        'average_progression_m': float(progression.mean()) if not progression.empty else 0.0,
        'main_channel': main_channel,
        'channel_counts': channels,
    }


def progressive_pass_player_summary(passes_df, limit=5):
    """Rank players while keeping failed progressive attempts visible."""
    columns = ['Player', 'Successful', 'Attempted', 'Completion %', 'Progression m']
    if (
        passes_df is None
        or passes_df.empty
        or 'is_progressive_attempt' not in passes_df.columns
    ):
        return pd.DataFrame(columns=columns)

    attempts = passes_df[
        passes_df['is_progressive_attempt'].fillna(False).astype(bool)
    ].copy()
    if attempts.empty or 'playerName' not in attempts.columns:
        return pd.DataFrame(columns=columns)

    attempts['_successful_progression_m'] = pd.to_numeric(
        attempts.get('progressive_distance_m'), errors='coerce'
    ).where(attempts['is_progressive'].fillna(False).astype(bool), 0.0)
    grouped = attempts.groupby('playerName', dropna=True).agg(
        Successful=('is_progressive', 'sum'),
        Attempted=('is_progressive_attempt', 'sum'),
        **{'Progression m': ('_successful_progression_m', 'sum')},
    ).reset_index().rename(columns={'playerName': 'Player'})
    grouped['Successful'] = grouped['Successful'].astype(int)
    grouped['Attempted'] = grouped['Attempted'].astype(int)
    grouped['Completion %'] = (
        grouped['Successful'] / grouped['Attempted'] * 100
    ).round(0).astype(int)
    grouped['Progression m'] = grouped['Progression m'].round(0).astype(int)
    return grouped.sort_values(
        ['Successful', 'Progression m', 'Attempted'],
        ascending=[False, False, False],
    ).head(limit)[columns].reset_index(drop=True)


def analyze_progressive_passes(
    df_input,
    pitch_length_meters=105.0,
    exclude_qualifiers=None,
    return_ids_only=False,
):
    """Backward-compatible access to completed progressive passes."""
    logger.debug('Analyzing progressive passes...')
    classified = classify_progressive_passes(
        df_input,
        pitch_length_meters=pitch_length_meters,
        exclude_qualifiers=exclude_qualifiers,
    )
    if classified.empty:
        return [] if return_ids_only else (pd.DataFrame(), {})

    progressive = classified[classified['is_progressive']].copy()
    if return_ids_only:
        if 'id' in progressive.columns:
            return progressive['id'].tolist()
        return progressive.index.tolist()

    channel_counts = progressive.get(
        'progressive_channel', pd.Series(dtype='object')
    ).value_counts()
    zone_counts = {
        'total': int(len(progressive)),
        'left': int(channel_counts.get('Left', 0)),
        'mid': int(channel_counts.get('Central', 0)),
        'right': int(channel_counts.get('Right', 0)),
    }
    return progressive, zone_counts


# --- Final Third Passes (Zone 14 / Half-Spaces) ---
# This function identifies successful passes ending in Zone 14 or Left/Right Half-Spaces for a specific team.
# def analyze_final_third_passes(passes_df_team_successful):
#     """
#     Identifies successful passes ending in Zone 14 or Left/Right Half-Spaces
#     for a specific team. Excludes passes starting very close to the corner flag.

#     Args:
#         passes_df_team_successful (pd.DataFrame): DataFrame containing ONLY successful passes
#                                                  for the team being analyzed.

#     Returns:
#         tuple: A tuple containing:
#             - pd.DataFrame: DataFrame with Zone 14 passes.
#             - pd.DataFrame: DataFrame with Left Half-Space passes.
#             - pd.DataFrame: DataFrame with Right Half-Space passes.
#             - dict: Counts {'zone14': count, 'hs_left': count, 'hs_right': count,
#                            'hs_total': count, 'total_final_third': count}.
#             Returns (empty df, empty df, empty df, default counts dict) if no relevant passes.
#     """
#     print(f"Analyzing Zone 14 / Half-Space passes...")

#     default_counts = {'zone14': 0, 'hs_left': 0, 'hs_right': 0, 'hs_total': 0, 'total_final_third': 0}
#     empty_df = pd.DataFrame()

#     # Ensure required coordinate columns exist
#     required_cols = ['x', 'y', 'end_x', 'end_y']
#     if not all(col in passes_df_team_successful.columns for col in required_cols):
#         missing = set(required_cols) - set(passes_df_team_successful.columns)
#         print(f"Error: Missing required columns for final third analysis: {missing}")
#         return empty_df, empty_df, empty_df, default_counts

#     # Filter out passes starting too close to corner flag (x > ~99 is likely corner)
#     df_filtered = passes_df_team_successful[passes_df_team_successful['x'] < 99.5].copy()

#     if df_filtered.empty:
#         print("Info: No relevant successful passes found after filtering.")
#         return empty_df, empty_df, empty_df, default_counts

#     # Define Zone Boundaries (Opta Coordinates 0-100)
#     zone14_x_min, zone14_x_max = 66.67, 82.0
#     zone14_y_min, zone14_y_max = 100/3, 200/3
#     halfspace_x_min = 66.67
#     rhs_y_min, rhs_y_max = 100/6, 100/3
#     lhs_y_min, lhs_y_max = 200/3, 500/6 # Using original 200/3 to 500/6 (~66.7 to 83.3)

#     # --- Classify Passes based on END coordinates ---
#     # Zone 14 Passes
#     zone14_mask = (
#         (df_filtered['end_x'] >= zone14_x_min) & (df_filtered['end_x'] <= zone14_x_max) &
#         (df_filtered['end_y'] >= zone14_y_min) & (df_filtered['end_y'] <= zone14_y_max)
#     )
#     df_zone14 = df_filtered[zone14_mask].copy()
#     z14_count = len(df_zone14)

#     # Right Half-Space Passes (Low Y values)
#     rhs_mask = (
#         (df_filtered['end_x'] >= halfspace_x_min) &
#         (df_filtered['end_y'] >= rhs_y_min) & (df_filtered['end_y'] < rhs_y_max)
#     )
#     df_rhs = df_filtered[rhs_mask & (~zone14_mask)].copy() # Exclude Zone 14 overlap
#     rhs_count = len(df_rhs)

#     # Left Half-Space Passes (High Y values)
#     lhs_mask = (
#         (df_filtered['end_x'] >= halfspace_x_min) &
#         (df_filtered['end_y'] >= lhs_y_min) & (df_filtered['end_y'] <= lhs_y_max)
#     )
#     df_lhs = df_filtered[lhs_mask & (~zone14_mask)].copy() # Exclude Zone 14 overlap
#     lhs_count = len(df_lhs)

#     # Calculate totals
#     hs_total_count = lhs_count + rhs_count
#     total_final_third_count = z14_count + hs_total_count

#     # Store counts
#     zone_stats_dict = {
#         'zone14': z14_count,
#         'hs_left': lhs_count,
#         'hs_right': rhs_count,
#         'hs_total': hs_total_count,
#         'total_final_third': total_final_third_count
#     }
#     print(f"Found: Zone 14={z14_count}, L HS={lhs_count}, R HS={rhs_count} (Total FT={total_final_third_count})")

#     return df_zone14, df_lhs, df_rhs, zone_stats_dict

# --- Final Third Entries via Pass ---
def analyze_final_third_passes(passes_df_team_successful):
    """
    Identifies completed passes that ENTER the attacking final third.

    A pass counts as a Final Third Entry only when:
        - it starts outside the attacking final third
        - it ends inside the attacking final third

    Opta coordinates are assumed to be 0-100, attacking from left to right.

    Each entry is also classified by:
        - entry channel: Left / Central / Right
        - destination zone:
            Zone 14
            Left Half-Space
            Right Half-Space
            Wide / Other

    Args:
        passes_df_team_successful (pd.DataFrame):
            Successful passes for one team.

    Returns:
        tuple:
            df_zone14,
            df_lhs,
            df_rhs,
            stats

        stats contains:
            total_final_third
            zone14
            hs_left
            hs_right
            hs_total
            wide_other
            channel_left
            channel_central
            channel_right
    """

    final_third_x = 100 * 2 / 3

    default_counts = {
        'zone14': 0,
        'hs_left': 0,
        'hs_right': 0,
        'hs_total': 0,
        'wide_other': 0,
        'channel_left': 0,
        'channel_central': 0,
        'channel_right': 0,
        'total_final_third': 0,
    }

    empty_df = pd.DataFrame()

    required_cols = ['x', 'y', 'end_x', 'end_y']

    if not all(col in passes_df_team_successful.columns for col in required_cols):
        missing = set(required_cols) - set(passes_df_team_successful.columns)
        logger.warning(f"Error: Missing required columns for final third analysis: {missing}")
        return empty_df, empty_df, empty_df, default_counts

    df = passes_df_team_successful.copy()

    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=required_cols)

    if df.empty:
        return empty_df, empty_df, empty_df, default_counts

    # ---------------------------------------------------------
    # TRUE FINAL THIRD ENTRY
    #
    # The ball must cross the final-third boundary.
    # Passes starting inside the final third are NOT new entries.
    # ---------------------------------------------------------
    entry_mask = (
        (df['x'] < final_third_x) &
        (df['end_x'] >= final_third_x)
    )

    entries = df[entry_mask].copy()

    if entries.empty:
        return empty_df, empty_df, empty_df, default_counts

    # ---------------------------------------------------------
    # ENTRY CHANNEL
    # Based on where the ball crosses / arrives into final third.
    #
    # Pitch divided into three equal vertical channels.
    # ---------------------------------------------------------
    def classify_channel(end_y):
        if end_y < 100 / 3:
            return 'Right'
        elif end_y <= 200 / 3:
            return 'Central'
        else:
            return 'Left'

    entries['final_third_channel'] = entries['end_y'].apply(classify_channel)

    # ---------------------------------------------------------
    # DESTINATION ZONES
    # ---------------------------------------------------------

    # Zone 14:
    # central area immediately outside the penalty area.
    zone14_mask = (
        (entries['end_x'] >= final_third_x) &
        (entries['end_x'] <= 82.0) &
        (entries['end_y'] >= 100 / 3) &
        (entries['end_y'] <= 200 / 3)
    )

    # Right half-space
    rhs_mask = (
        (entries['end_y'] >= 100 / 6) &
        (entries['end_y'] < 100 / 3)
    )

    # Left half-space
    lhs_mask = (
        (entries['end_y'] > 200 / 3) &
        (entries['end_y'] <= 500 / 6)
    )

    df_zone14 = entries[zone14_mask].copy()

    # Keep categories mutually exclusive.
    df_rhs = entries[rhs_mask & ~zone14_mask].copy()
    df_lhs = entries[lhs_mask & ~zone14_mask].copy()

    classified_mask = (
        zone14_mask |
        rhs_mask |
        lhs_mask
    )

    df_wide_other = entries[~classified_mask].copy()

    channel_counts = entries['final_third_channel'].value_counts()

    z14_count = len(df_zone14)
    lhs_count = len(df_lhs)
    rhs_count = len(df_rhs)

    stats = {
        'zone14': int(z14_count),
        'hs_left': int(lhs_count),
        'hs_right': int(rhs_count),
        'hs_total': int(lhs_count + rhs_count),

        'wide_other': int(len(df_wide_other)),

        'channel_left': int(channel_counts.get('Left', 0)),
        'channel_central': int(channel_counts.get('Central', 0)),
        'channel_right': int(channel_counts.get('Right', 0)),

        # THIS is now the true total.
        'total_final_third': int(len(entries)),
    }

    logger.debug("Final Third Entries via pass: "
        f"Total={stats['total_final_third']}, "
        f"Zone14={stats['zone14']}, "
        f"LHS={stats['hs_left']}, "
        f"RHS={stats['hs_right']}, "
        f"Other={stats['wide_other']}")

    return df_zone14, df_lhs, df_rhs, stats


def analyze_final_third_entries(
    successful_passes_df,
    carries_df=None,
    *,
    high_confidence_carries_only=True,
    carry_boundary_buffer=1.0,
    min_carry_x_progress=3.0,
):
    """
    Identify entries into the attacking final third.

    An entry occurs when the ball crosses the final-third boundary:
        start_x < 66.67
        end_x >= 66.67

    Supported entry types:
        - Pass: completed passes
        - Carry: high-confidence inferred carries with a robust boundary cross

    Pass coordinates are provider observations and use the exact 66.67 line.
    Carry coordinates are reconstructed between consecutive events, so the
    primary KPI applies two additional safeguards by default:

        - the carry must start at least ``carry_boundary_buffer`` Opta points
          before the line and end the same distance beyond it;
        - longitudinal progress must be at least ``min_carry_x_progress``.

    Rejected carry candidates remain visible in the returned diagnostics.

    Each entry is classified by:
        - entry_type: Pass / Carry
        - final_third_channel: Left / Central / Right
        - destination_zone:
            Zone 14
            Left Half-Space
            Right Half-Space
            Other

    Returns:
        entries_df, stats
    """

    final_third_x = 100 * 2 / 3

    stats_default = {
        'total_final_third': 0,
        'pass_entries': 0,
        'carry_entries': 0,
        'carry_entry_candidates': 0,
        'carry_entries_excluded_confidence': 0,
        'carry_entries_excluded_boundary': 0,
        'carry_entries_excluded_total': 0,

        'channel_left': 0,
        'channel_central': 0,
        'channel_right': 0,

        'zone14': 0,
        'hs_left': 0,
        'hs_right': 0,
        'hs_total': 0,
        'other': 0,
    }

    entry_frames = []

    # ---------------------------------------------------------
    # PASS ENTRIES
    # ---------------------------------------------------------
    if successful_passes_df is not None and not successful_passes_df.empty:
        required_cols = ['x', 'y', 'end_x', 'end_y']

        if all(col in successful_passes_df.columns for col in required_cols):
            passes = successful_passes_df.copy()

            for col in required_cols:
                passes[col] = pd.to_numeric(passes[col], errors='coerce')

            passes = passes.dropna(subset=required_cols)

            pass_entry_mask = (
                (passes['x'] < final_third_x) &
                (passes['end_x'] >= final_third_x)
            )

            pass_entries = passes[pass_entry_mask].copy()

            if not pass_entries.empty:
                pass_entries['entry_type'] = 'Pass'
                entry_frames.append(pass_entries)

    # ---------------------------------------------------------
    # CARRY ENTRIES
    # ---------------------------------------------------------
    if carries_df is not None and not carries_df.empty:
        required_cols = ['x', 'y', 'end_x', 'end_y']

        if all(col in carries_df.columns for col in required_cols):
            carries = carries_df.copy()

            for col in required_cols:
                carries[col] = pd.to_numeric(carries[col], errors='coerce')

            carries = carries.dropna(subset=required_cols)

            # First keep only candidates whose event continuity passed the
            # inference checks. This does not yet make them KPI-eligible.
            if 'carry_is_reliable' in carries.columns:
                carries = carries[
                    carries['carry_is_reliable'].fillna(False).astype(bool)
                ]

            raw_crossing_mask = (
                (carries['x'] < final_third_x) &
                (carries['end_x'] >= final_third_x)
            )
            carry_candidates = carries[raw_crossing_mask].copy()
            stats_default['carry_entry_candidates'] = int(
                len(carry_candidates)
            )

            if not carry_candidates.empty:
                if high_confidence_carries_only:
                    if 'carry_confidence' in carry_candidates.columns:
                        confidence_mask = (
                            carry_candidates['carry_confidence']
                            .fillna('unknown')
                            .astype(str)
                            .str.lower()
                            .eq('high')
                        )
                    else:
                        confidence_mask = pd.Series(
                            False,
                            index=carry_candidates.index,
                        )
                else:
                    confidence_mask = pd.Series(
                        True,
                        index=carry_candidates.index,
                    )

                stats_default[
                    'carry_entries_excluded_confidence'
                ] = int((~confidence_mask).sum())

                confidence_eligible = carry_candidates[
                    confidence_mask
                ].copy()

                robust_crossing_mask = (
                    (
                        confidence_eligible['x']
                        <= final_third_x - carry_boundary_buffer
                    )
                    & (
                        confidence_eligible['end_x']
                        >= final_third_x + carry_boundary_buffer
                    )
                    & (
                        confidence_eligible['end_x']
                        - confidence_eligible['x']
                        >= min_carry_x_progress
                    )
                )

                stats_default[
                    'carry_entries_excluded_boundary'
                ] = int((~robust_crossing_mask).sum())

                carry_entries = confidence_eligible[
                    robust_crossing_mask
                ].copy()
            else:
                carry_entries = pd.DataFrame()

            stats_default['carry_entries_excluded_total'] = int(
                stats_default['carry_entries_excluded_confidence']
                + stats_default['carry_entries_excluded_boundary']
            )

            if not carry_entries.empty:
                carry_entries['entry_type'] = 'Carry'
                carry_entries['entry_method'] = 'Inferred carry'
                entry_frames.append(carry_entries)

    if not entry_frames:
        return pd.DataFrame(), stats_default

    entries = pd.concat(entry_frames, ignore_index=True, sort=False)

    # ---------------------------------------------------------
    # ENTRY CHANNEL
    # ---------------------------------------------------------
    def classify_channel(end_y):
        if end_y < 100 / 3:
            return 'Right'
        elif end_y <= 200 / 3:
            return 'Central'
        else:
            return 'Left'

    entries['final_third_channel'] = (
        entries['end_y'].apply(classify_channel)
    )

    # ---------------------------------------------------------
    # DESTINATION ZONES
    # ---------------------------------------------------------
    def classify_destination(row):
        end_x = row['end_x']
        end_y = row['end_y']

        # Zone 14
        if (
            final_third_x <= end_x <= 82.0
            and 100 / 3 <= end_y <= 200 / 3
        ):
            return 'Zone 14'

        # Right half-space: low Opta Y
        if 100 / 6 <= end_y < 100 / 3:
            return 'Right Half-Space'

        # Left half-space: high Opta Y
        if 200 / 3 < end_y <= 500 / 6:
            return 'Left Half-Space'

        return 'Other'

    entries['destination_zone'] = entries.apply(
        classify_destination,
        axis=1,
    )

    type_counts = entries['entry_type'].value_counts()
    channel_counts = entries['final_third_channel'].value_counts()
    zone_counts = entries['destination_zone'].value_counts()

    stats = {
        'total_final_third': int(len(entries)),

        'pass_entries': int(type_counts.get('Pass', 0)),
        'carry_entries': int(type_counts.get('Carry', 0)),
        'carry_entry_candidates': stats_default[
            'carry_entry_candidates'
        ],
        'carry_entries_excluded_confidence': stats_default[
            'carry_entries_excluded_confidence'
        ],
        'carry_entries_excluded_boundary': stats_default[
            'carry_entries_excluded_boundary'
        ],
        'carry_entries_excluded_total': stats_default[
            'carry_entries_excluded_total'
        ],

        'channel_left': int(channel_counts.get('Left', 0)),
        'channel_central': int(channel_counts.get('Central', 0)),
        'channel_right': int(channel_counts.get('Right', 0)),

        'zone14': int(zone_counts.get('Zone 14', 0)),
        'hs_left': int(zone_counts.get('Left Half-Space', 0)),
        'hs_right': int(zone_counts.get('Right Half-Space', 0)),
        'hs_total': int(
            zone_counts.get('Left Half-Space', 0)
            + zone_counts.get('Right Half-Space', 0)
        ),
        'other': int(zone_counts.get('Other', 0)),
    }

    return entries, stats

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
    logger.debug("Analyzing chance creation passes...")

    # --- Check required columns ---
    required_cols = ['team_name', 'type_name', 'outcome', 'x', 'y', 'end_x', 'end_y']
    if assist_qualifier_col not in df_processed.columns:
        logger.warning(f"Error: Assist qualifier column '{assist_qualifier_col}' not found. Cannot analyze chances.")
        return pd.DataFrame(), pd.DataFrame()
    required_cols.append(assist_qualifier_col) # Add it for the main check

    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        logger.warning(f"Error: Missing required columns for chance creation analysis: {missing}")
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
         logger.debug("No successful passes with valid end coordinates found.")
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
         logger.debug("No passes matched the Key Pass or Assist criteria.")
         return pd.DataFrame(), pd.DataFrame()

    # --- Add boolean flags for easier plotting distinction ---
    # Apply masks again to the filtered df_chances to ensure correct index alignment
    df_chances['is_key_pass'] = key_pass_filter.reindex(df_chances.index).fillna(False)
    df_chances['is_assist'] = assist_filter.reindex(df_chances.index).fillna(False)

    logger.info(f"Found {len(df_chances)} chance-creating passes.")

    # --- Split by team ---
    df_chances_home = df_chances[df_chances['team_name'] == hteamName].copy()
    df_chances_away = df_chances[df_chances['team_name'] == ateamName].copy()

    logger.debug(f"  Home Chances: {len(df_chances_home)}, Away Chances: {len(df_chances_away)}")

    return df_chances_home, df_chances_away
