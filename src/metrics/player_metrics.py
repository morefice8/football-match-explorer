# src/metrics/player_metrics.py
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from .pass_metrics import classify_progressive_passes
from .shot_sequence_metrics import calculate_shot_sequence_player_stats


PENALTY_AREA_X_MIN = 83.5
PENALTY_AREA_Y_MIN = 21.1
PENALTY_AREA_Y_MAX = 78.9


def _successful_pass_mask(df):
    """Return successful Pass events, accepting text or numeric outcomes."""
    pass_mask = df['type_name'].fillna('').astype(str).str.lower().eq('pass')
    outcome = df.get('outcome', pd.Series('', index=df.index))
    outcome_text = outcome.fillna('').astype(str).str.strip().str.lower()
    outcome_numeric = pd.to_numeric(outcome, errors='coerce')
    return pass_mask & (
        outcome_text.eq('successful')
        | outcome_numeric.eq(1)
    )


def _event_key_series(df):
    """Return stable event identities, preferring Opta's global ``id``."""
    keys = pd.Series(index=df.index, dtype='object')

    if 'id' in df.columns:
        has_id = df['id'].notna()
        keys.loc[has_id] = [
            ('id', value)
            for value in df.loc[has_id, 'id']
        ]

    if 'eventId' in df.columns:
        missing = keys.isna() & df['eventId'].notna()
        keys.loc[missing] = [
            ('eventId', value)
            for value in df.loc[missing, 'eventId']
        ]

    missing = keys.isna()
    keys.loc[missing] = [
        ('index', index)
        for index in df.index[missing]
    ]
    return keys


def build_player_passing_event_sets(
    df_processed,
    prog_pass_exclusions=None,
):
    """
    Build explicit event sets for the individual passing metrics.

    Contract:
    - Key Pass: completed pass flagged as a key pass; restarts are allowed.
    - Assist: completed pass flagged as an assist; restarts are allowed.
    - Completed Pass into Box: completed open-play pass ending in the box.
    - Progressive Pass: completed open-play pass satisfying the canonical
      progressive-pass definition.
    - Unique Offensive Contribution: union of the four event sets above.

    Set unions prevent overlapping categories from double-counting the same
    event.
    """
    empty = {
        'key_pass': set(),
        'assist': set(),
        'completed_pass_into_box': set(),
        'progressive_pass': set(),
        'shot_assist': set(),
        'unique_offensive_contribution': set(),
    }
    if df_processed is None or df_processed.empty:
        return empty

    df = df_processed.copy()
    if 'type_name' not in df.columns:
        return empty

    for flag_col in ('is_key_pass', 'is_assist'):
        if flag_col in df.columns:
            df[flag_col] = df[flag_col].fillna(False).astype(bool)
        else:
            df[flag_col] = False

    classified = classify_progressive_passes(
        df,
        exclude_qualifiers=prog_pass_exclusions,
    )
    successful_pass = _successful_pass_mask(df)
    open_play = classified.get(
        'progressive_is_open_play',
        pd.Series(False, index=df.index),
    ).fillna(False).astype(bool)

    end_x = pd.to_numeric(
        df.get(
            'end_x',
            pd.Series(float('nan'), index=df.index),
        ),
        errors='coerce',
    )
    end_y = pd.to_numeric(
        df.get(
            'end_y',
            pd.Series(float('nan'), index=df.index),
        ),
        errors='coerce',
    )

    key_pass_mask = successful_pass & df['is_key_pass']
    assist_mask = successful_pass & df['is_assist']
    into_box_mask = (
        successful_pass
        & open_play
        & end_x.ge(PENALTY_AREA_X_MIN)
        & end_y.between(
            PENALTY_AREA_Y_MIN,
            PENALTY_AREA_Y_MAX,
            inclusive='both',
        )
    )
    progressive_mask = classified.get(
        'is_progressive',
        pd.Series(False, index=df.index),
    ).fillna(False).astype(bool)

    event_keys = _event_key_series(df)
    key_passes = set(event_keys.loc[key_pass_mask])
    assists = set(event_keys.loc[assist_mask])
    into_box = set(event_keys.loc[into_box_mask])
    progressive = set(event_keys.loc[progressive_mask])
    shot_assists = key_passes | assists

    return {
        'key_pass': key_passes,
        'assist': assists,
        'completed_pass_into_box': into_box,
        'progressive_pass': progressive,
        'shot_assist': shot_assists,
        'unique_offensive_contribution': (
            progressive
            | into_box
            | key_passes
            | assists
        ),
    }


def calculate_offensive_pass_contributions(
    df_processed,
    df_progressive_passes=None,
    prog_pass_exclusions=None,
):
    """Count unique canonical offensive pass events per player."""
    del df_progressive_passes  # Backward-compatible parameter.

    if df_processed is None or df_processed.empty:
        return pd.Series(
            dtype='int64',
            name='Offensive Pass Contributions',
        )

    df = df_processed.copy()
    event_sets = build_player_passing_event_sets(
        df,
        prog_pass_exclusions=prog_pass_exclusions,
    )
    contribution_events = event_sets[
        'unique_offensive_contribution'
    ]
    if not contribution_events:
        return pd.Series(
            dtype='int64',
            name='Offensive Pass Contributions',
        )

    event_keys = _event_key_series(df)
    selected = df.loc[
        event_keys.isin(contribution_events),
        ['playerName'],
    ].copy()
    selected['_event_key'] = event_keys.loc[selected.index]

    return (
        selected
        .dropna(subset=['playerName'])
        .drop_duplicates(['playerName', '_event_key'])
        .groupby('playerName')
        .size()
        .rename('Offensive Pass Contributions')
        .astype(int)
    )


def calculate_player_stats(df_processed, assist_qualifier_col='Assist',
                           key_pass_values=[13, 14, 15], assist_values=[16],
                           prog_pass_exclusions=None):
    """
    Calculates a variety of statistics aggregated per player.
    """
    logger.debug("Calculating aggregated player statistics...")
    if df_processed.empty:
        logger.warning("Warning: Input DataFrame is empty.")
        return pd.DataFrame()

    df_processed = df_processed.copy()

    # --- Pre-calculate Flags ---
    # (Keep the logic to ensure is_key_pass and is_assist flags exist)
    if 'is_key_pass' not in df_processed.columns or 'is_assist' not in df_processed.columns:
        logger.warning("Info: 'is_key_pass'/'is_assist' flags not found, calculating them now...")
        if assist_qualifier_col not in df_processed.columns:
            logger.warning(f"Error: Required qualifier column '{assist_qualifier_col}' not found.")
            return pd.DataFrame()
        assist_qual_numeric = pd.to_numeric(df_processed[assist_qualifier_col], errors='coerce')
        # Ensure flags are only True if it's also a Pass event
        df_processed['is_key_pass'] = assist_qual_numeric.isin(key_pass_values) & (df_processed['type_name'] == 'Pass')
        df_processed['is_assist'] = assist_qual_numeric.isin(assist_values) & (df_processed['type_name'] == 'Pass')
    else:
        df_processed['is_key_pass'] = df_processed['is_key_pass'].fillna(False).astype(bool)
        df_processed['is_assist'] = df_processed['is_assist'].fillna(False).astype(bool)

    successful_pass = _successful_pass_mask(df_processed)
    df_processed['is_key_pass'] &= successful_pass
    df_processed['is_assist'] &= successful_pass

    # Define shot types
    shot_types = ['Miss', 'Attempt Saved', 'Post', 'Goal']

    # --- Group by Player ---
    grouped_player = df_processed.groupby('playerName')

    # --- Calculate Stats using apply or separate aggregations ---
    logger.debug("  Aggregating player stats...")
    player_stats_list = []

    required_cols_check = ['type_name', 'outcome', 'x', 'y', 'end_x', 'end_y', 'is_key_pass', 'is_assist']
    if not all(col in df_processed.columns for col in required_cols_check):
         logger.warning(f"Error: Missing one or more required columns for calculation: {set(required_cols_check) - set(df_processed.columns)}")
         return pd.DataFrame()

    passing_event_sets = build_player_passing_event_sets(
        df_processed,
        prog_pass_exclusions=prog_pass_exclusions,
    )

    for name, group in grouped_player:
        stats = {}
        stats['playerName'] = name # Keep player name
        group_event_keys = set(_event_key_series(group))

        # Shooting Sequence
        stats['Shots'] = (group['type_name'].isin(shot_types)).sum()
        stats['Shot Assists'] = len(
            group_event_keys
            & passing_event_sets['shot_assist']
        )

        # Defensive Actions
        stats['Tackles Won'] = ((group['type_name'] == 'Tackle') & (group['outcome'] == 'Successful')).sum()
        stats['Interceptions'] = (group['type_name'] == 'Interception').sum()
        stats['Clearances'] = (group['type_name'] == 'Clearance').sum()
        stats['Ball recovery'] = (group['type_name'] == 'Ball recovery').sum()
        stats['Aerials Won'] = ((group['type_name'] == 'Aerial') & (group['outcome'] == 'Successful')).sum()

        # Passing Types
        stats['Progressive Passes'] = len(
            group_event_keys
            & passing_event_sets['progressive_pass']
        )
        stats['Passes into Box'] = len(
            group_event_keys
            & passing_event_sets['completed_pass_into_box']
        )
        stats['Key Passes'] = len(
            group_event_keys
            & passing_event_sets['key_pass']
        )
        stats['Assists'] = len(
            group_event_keys
            & passing_event_sets['assist']
        )

        # Basic Pass Stats
        stats['Total Passes'] = (group['type_name'] == 'Pass').sum()
        stats['Successful Passes'] = ((group['type_name'] == 'Pass') & (group['outcome'] == 'Successful')).sum()

        # Append player's stats dictionary to list
        player_stats_list.append(stats)

    # Convert list of dicts to DataFrame
    player_stats = pd.DataFrame(player_stats_list)
    if player_stats.empty:
        logger.debug("No player data after initial aggregation.")
        return pd.DataFrame()
    player_stats.set_index('playerName', inplace=True) # Set index after creation

    # --- Canonical Shot Sequence Involvement (REL-08) ---
    logger.debug("  Calculating canonical shot-sequence involvement...")
    shot_sequence_stats = calculate_shot_sequence_player_stats(
        df_processed,
        shot_types=shot_types,
    )

    shot_sequence_columns = [
        'Shot Sequence Shots',
        'Shot Sequence Assists',
        'Shot Sequence Pre-Assists',
        'Shot Sequence Involvements',
    ]

    if shot_sequence_stats.empty:
        for column in shot_sequence_columns:
            player_stats[column] = 0
    else:
        player_stats = player_stats.merge(
            shot_sequence_stats,
            left_index=True,
            right_index=True,
            how='left',
        )
        for column in shot_sequence_columns:
            player_stats[column] = (
                player_stats[column]
                .fillna(0)
                .astype(int)
            )

    # Temporary compatibility aliases. The legacy names remain available to
    # callers until the graph is redesigned, but are now sourced from the
    # possession-chain contract rather than shift(-1).
    player_stats['Buildup to Shot'] = (
        player_stats['Shot Sequence Pre-Assists']
    )
    player_stats['Shooting Seq Total'] = (
        player_stats['Shot Sequence Involvements']
    )

        # --- Unique Offensive Pass Contributions ---
    offensive_contribution_counts = (
        calculate_offensive_pass_contributions(
            df_processed,
            prog_pass_exclusions=prog_pass_exclusions,
        )
    )

    player_stats = player_stats.merge(
        offensive_contribution_counts,
        left_index=True,
        right_index=True,
        how='left',
    )

    player_stats['Offensive Pass Contributions'] = (
        player_stats['Offensive Pass Contributions']
        .fillna(0)
        .astype(int)
    )

    # defensive_cols = ['Tackles Won', 'Interceptions', 'Clearances']
    # if all(col in player_stats.columns for col in defensive_cols): player_stats['Defensive Actions Total'] = player_stats[defensive_cols].sum(axis=1)
    # else: print("Warning: Could not calculate 'Defensive Actions Total'."); player_stats['Defensive Actions Total'] = 0

    defensive_cols = ['Tackles Won', 'Interceptions', 'Clearances', 'Ball recovery', 'Aerials Won']
    if all(col in player_stats.columns for col in defensive_cols):
        player_stats['Defensive Actions Total'] = player_stats[defensive_cols].sum(axis=1)
    else:
        logger.warning("Warning: Could not calculate 'Defensive Actions Total'.")
        player_stats['Defensive Actions Total'] = 0


    # --- Final Touches ---
    player_stats.fillna(0, inplace=True)
    count_cols = player_stats.select_dtypes(include=np.number).columns
    player_stats[count_cols] = player_stats[count_cols].astype(int)

    logger.info(f"Finished calculating stats for {len(player_stats)} players.")
    logger.debug('%s %s', "Final columns in player_stats_df:", player_stats.columns.tolist())
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
    logger.debug("Calculating median player touch locations...")
    if df_processed.empty:
        logger.warning("Warning: Input DataFrame is empty.")
        return pd.DataFrame()

    if exclude_event_types is None:
        exclude_event_types = ['Formation Change', 'Deleted event', 'End', 'Start'] # Add event types with no real location

    # Ensure necessary columns exist
    required_cols = ['playerName', 'team_name', 'x', 'y', 'type_name', 'Mapped Jersey Number', 'positional_role', 'id']
    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        logger.warning(f"Error: Missing required columns for median touch location: {missing}")
        return pd.DataFrame()

    # Filter out excluded event types and events without valid coordinates
    df_touches = df_processed[
        (~df_processed['type_name'].isin(exclude_event_types)) &
        (df_processed['x'].notna()) &
        (df_processed['y'].notna())
    ].copy()

    if df_touches.empty:
        logger.warning("Warning: No valid touch events found after filtering.")
        return pd.DataFrame()

    # Group by player and calculate median location, count, and get first jersey/role
    player_loc_agg = df_touches.groupby('playerName').agg(
        median_x=('x', 'median'),
        median_y=('y', 'median'),
        touch_count=('id', 'count'), # Count events considered as touches
        jersey_number=('Mapped Jersey Number', 'first'), # Get representative jersey
        positional_role=('positional_role', 'first') # Get representative role
    ).reset_index() # Make playerName a column

    logger.info(f"Calculated median locations for {len(player_loc_agg)} players.")
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

# PLOT-03 — median-position territorial profile
# ---------------------------------------------------------------------------

MEAN_POSITIONS_MIN_MINUTES = 15.0

MEAN_POSITION_TOUCH_TYPES = {
    "Pass",
    "Take On",
    "Ball touch",
    "Shot",
    "Miss",
    "Attempt Saved",
    "Post",
    "Goal",
    "Dispossessed",
    "Ball recovery",
    "Clearance",
    "Interception",
    "Tackle",
}


def _mean_position_event_seconds(row):
    minute = pd.to_numeric(
        row.get("timeMin"),
        errors="coerce",
    )
    second = pd.to_numeric(
        row.get("timeSec"),
        errors="coerce",
    )

    if pd.isna(minute):
        return None

    if pd.isna(second):
        second = 0

    return float(minute) * 60.0 + float(second)


def _mean_position_player_key(row):
    player_id = row.get("playerId")

    if player_id is not None and not pd.isna(player_id):
        text = str(player_id).strip()
        if text:
            return f"id:{text}"

    player_name = row.get("playerName")

    if player_name is not None and not pd.isna(player_name):
        text = str(player_name).strip()
        if text:
            return f"name:{text}"

    return None


def _mean_position_truthy(value):
    if value is None:
        return False

    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return value != 0

    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "red",
        "rc",
    }


def _mean_position_is_dismissal(row):
    for column in (
        "Red card",
        "Red Card",
        "Second yellow",
        "Second Yellow",
        "Second yellow card",
    ):
        if (
            column in row.index
            and _mean_position_truthy(
                row.get(column)
            )
        ):
            return True

    return False


def _mean_position_match_windows(df):
    event_seconds = df.apply(
        _mean_position_event_seconds,
        axis=1,
    )

    valid_seconds = event_seconds.dropna()

    match_end = (
        float(valid_seconds.max())
        if not valid_seconds.empty
        else 90.0 * 60.0
    )

    period_numeric = pd.to_numeric(
        df.get(
            "periodId",
            pd.Series(
                index=df.index,
                dtype=float,
            ),
        ),
        errors="coerce",
    )

    first_half_seconds = event_seconds[
        period_numeric.eq(1)
    ].dropna()

    second_half_seconds = event_seconds[
        period_numeric.eq(2)
    ].dropna()

    first_half_end = (
        float(first_half_seconds.max())
        if not first_half_seconds.empty
        else min(
            45.0 * 60.0,
            match_end,
        )
    )

    second_half_start = (
        float(second_half_seconds.min())
        if not second_half_seconds.empty
        else first_half_end
    )

    second_half_end = (
        float(second_half_seconds.max())
        if not second_half_seconds.empty
        else match_end
    )

    return {
        "full": (
            0.0,
            max(
                match_end,
                1.0,
            ),
        ),
        "1h": (
            0.0,
            max(
                first_half_end,
                1.0,
            ),
        ),
        "2h": (
            second_half_start,
            max(
                second_half_end,
                second_half_start + 1.0,
            ),
        ),
    }


def _mean_position_on_pitch_intervals(
    df_team,
    window_end,
):
    """
    Infer one on-pitch interval per player from starter/substitution/card data.

    Football does not allow re-entry, so a single [on, off] interval is the
    correct contract for normal match data.
    """
    if df_team.empty:
        return {}

    player_rows = df_team.copy()
    player_rows["_player_key"] = player_rows.apply(
        _mean_position_player_key,
        axis=1,
    )
    player_rows["_event_seconds"] = player_rows.apply(
        _mean_position_event_seconds,
        axis=1,
    )

    intervals = {}

    for player_key, group in player_rows.dropna(
        subset=["_player_key"]
    ).groupby("_player_key"):
        is_starter = (
            group.get(
                "Is Starter",
                pd.Series(
                    False,
                    index=group.index,
                ),
            )
            .fillna(False)
            .astype(bool)
            .any()
        )

        type_ids = pd.to_numeric(
            group["typeId"],
            errors="coerce",
        )

        on_events = group.loc[
            type_ids.eq(19)
        ]["_event_seconds"].dropna()

        off_events = group.loc[
            type_ids.eq(18)
        ]["_event_seconds"].dropna()

        dismissal_events = group.loc[
            group.apply(
                _mean_position_is_dismissal,
                axis=1,
            )
        ]["_event_seconds"].dropna()

        if is_starter:
            on_second = 0.0
        elif not on_events.empty:
            on_second = float(
                on_events.min()
            )
        else:
            # A player with normal on-ball events but no explicit starter/sub
            # metadata is not assigned invented minutes.
            continue

        off_candidates = [
            float(value)
            for value in off_events.tolist()
            if float(value) >= on_second
        ]
        off_candidates.extend(
            float(value)
            for value in dismissal_events.tolist()
            if float(value) >= on_second
        )

        off_second = (
            min(off_candidates)
            if off_candidates
            else float(window_end)
        )

        intervals[player_key] = (
            float(on_second),
            max(
                float(on_second),
                float(off_second),
            ),
        )

    return intervals


def _mean_position_minutes_in_window(
    interval,
    window,
):
    if interval is None:
        return 0.0

    on_second, off_second = interval
    window_start, window_end = window

    overlap_start = max(
        float(on_second),
        float(window_start),
    )
    overlap_end = min(
        float(off_second),
        float(window_end),
    )

    return max(
        0.0,
        overlap_end - overlap_start,
    ) / 60.0



def _mean_position_substitution_chains(
    df_team,
    player_keys,
):
    """
    Build starter-slot substitution chains.

    Each starting player defines one football slot. Incoming substitutes inherit
    the slot of the player they replace. Players that cannot be linked to a
    starter slot are not allowed to create an artificial 12th/13th marker.
    """
    player_keys = {
        str(key)
        for key in (player_keys or [])
        if key
    }

    if not player_keys:
        return []

    working = df_team.copy()

    if "event_sequence_index" not in working.columns:
        working = working.reset_index().rename(
            columns={
                "index": "event_sequence_index",
            }
        )

    working["_player_key"] = working.apply(
        _mean_position_player_key,
        axis=1,
    )
    working["_event_seconds"] = working.apply(
        _mean_position_event_seconds,
        axis=1,
    )

    type_ids = pd.to_numeric(
        working["typeId"],
        errors="coerce",
    )

    off_rows = working.loc[
        type_ids.eq(18)
        & working["_player_key"].notna()
    ].sort_values(
        "event_sequence_index",
        kind="stable",
    )

    on_rows = working.loc[
        type_ids.eq(19)
        & working["_player_key"].notna()
    ].copy()

    successor = {}
    predecessor = {}
    used_on_indices = set()

    for _, off_row in off_rows.iterrows():
        off_key = str(
            off_row["_player_key"]
        )

        if off_key not in player_keys:
            continue

        candidate = None
        off_event_id = off_row.get(
            "eventId"
        )

        if (
            off_event_id is not None
            and not pd.isna(off_event_id)
            and "related_eventId"
            in on_rows.columns
        ):
            linked = on_rows.loc[
                on_rows[
                    "related_eventId"
                ].astype(str).eq(
                    str(off_event_id)
                )
            ]

            linked = linked.loc[
                ~linked.index.isin(
                    used_on_indices
                )
            ]

            if not linked.empty:
                candidate = (
                    linked.sort_values(
                        "event_sequence_index",
                        kind="stable",
                    )
                    .iloc[0]
                )

        if candidate is None:
            available = on_rows.loc[
                ~on_rows.index.isin(
                    used_on_indices
                )
            ].copy()

            contestant_id = off_row.get(
                "contestantId"
            )

            if (
                contestant_id is not None
                and not pd.isna(
                    contestant_id
                )
                and "contestantId"
                in available.columns
            ):
                available = available.loc[
                    available[
                        "contestantId"
                    ].astype(str).eq(
                        str(contestant_id)
                    )
                ]

            off_second = off_row.get(
                "_event_seconds"
            )

            if (
                off_second is not None
                and not pd.isna(off_second)
                and not available.empty
            ):
                available = available.assign(
                    _delta=(
                        available[
                            "_event_seconds"
                        ]
                        .sub(
                            float(
                                off_second
                            )
                        )
                        .abs()
                    )
                )

                available = available.loc[
                    available[
                        "_delta"
                    ].le(3.0)
                ]

            if not available.empty:
                candidate = (
                    available.sort_values(
                        [
                            "_delta"
                            if "_delta"
                            in available.columns
                            else "event_sequence_index",
                            "event_sequence_index",
                        ],
                        kind="stable",
                    )
                    .iloc[0]
                )

        if candidate is None:
            continue

        on_key = str(
            candidate[
                "_player_key"
            ]
        )

        if on_key not in player_keys:
            continue

        successor[
            off_key
        ] = on_key
        predecessor[
            on_key
        ] = off_key
        used_on_indices.add(
            candidate.name
        )

    starter_series = working.get(
        "Is Starter",
        pd.Series(
            False,
            index=working.index,
        ),
    ).fillna(False).astype(bool)

    starter_keys = [
        str(value)
        for value in working.loc[
            starter_series
            & working[
                "_player_key"
            ].notna(),
            "_player_key",
        ].drop_duplicates().tolist()
        if str(value) in player_keys
    ]

    roots = (
        starter_keys
        if starter_keys
        else sorted(
            key
            for key in player_keys
            if key not in predecessor
        )
    )

    chains = []
    assigned = set()

    for root in roots:
        chain = []
        current = root
        visited = set()

        while (
            current
            and current not in visited
            and current in player_keys
        ):
            chain.append(
                current
            )
            visited.add(
                current
            )
            assigned.add(
                current
            )
            current = successor.get(
                current
            )

        if chain:
            chains.append(
                chain
            )

    # If starter metadata is missing, retain disconnected players as their own
    # slots. With normal Opta starter metadata they are intentionally ignored.
    if not starter_keys:
        for player_key in sorted(
            player_keys.difference(
                assigned
            )
        ):
            chains.append(
                [
                    player_key
                ]
            )

    return chains


def _mean_position_representatives(
    df_team,
    minutes_by_player,
    min_minutes,
):
    """
    Select one representative per substitution chain.

    The player with the most minutes in the selected window represents that
    football slot. Ties prefer the later occupant, so an incoming substitute
    replaces the outgoing player rather than appearing alongside them.
    """
    chains = _mean_position_substitution_chains(
        df_team,
        minutes_by_player.keys(),
    )

    representatives = set()

    for chain in chains:
        candidates = []

        for index, player_key in enumerate(
            chain
        ):
            minutes_played = float(
                minutes_by_player.get(
                    player_key,
                    0.0,
                )
            )

            if minutes_played < float(
                min_minutes
            ):
                continue

            candidates.append(
                (
                    minutes_played,
                    index,
                    player_key,
                )
            )

        if not candidates:
            continue

        _, _, representative = max(
            candidates,
            key=lambda item: (
                item[0],
                item[1],
            ),
        )

        representatives.add(
            representative
        )

    return representatives


def _mean_position_period_mask(df, period):
    period = str(
        period or "full"
    ).lower()

    if period == "full":
        return pd.Series(
            True,
            index=df.index,
        )

    period_id = (
        1
        if period == "1h"
        else 2
        if period == "2h"
        else None
    )

    if period_id is None:
        raise ValueError(
            f"Unsupported mean-position period: {period}"
        )

    numeric_period = pd.to_numeric(
        df.get(
            "periodId",
            pd.Series(
                index=df.index,
                dtype=float,
            ),
        ),
        errors="coerce",
    )

    return numeric_period.eq(
        period_id
    )


def get_mean_positions_profile(
    df_processed,
    team_name,
    *,
    period="full",
    min_minutes=MEAN_POSITIONS_MIN_MINUTES,
):
    """
    Build a robust territorial-location profile.

    Player encoding:
      * centre = median touch location
      * footprint = x/y interquartile range
      * marker size = share of team touch events
      * eligibility = minutes played in selected period

    Team metrics are calculated over eligible outfield players when role
    metadata identifies a goalkeeper; otherwise all eligible players are used.
    """
    empty_summary = {
        "period": str(
            period or "full"
        ).lower(),
        "min_minutes": float(
            min_minutes
        ),
        "eligible_players": 0,
        "touches": 0,
        "team_length_m": None,
        "team_width_m": None,
        "centroid_x": None,
        "centroid_y": None,
        "average_height_m": None,
    }

    if (
        df_processed is None
        or df_processed.empty
        or not team_name
    ):
        return (
            pd.DataFrame(),
            empty_summary,
        )

    df = df_processed.copy()

    required = {
        "team_name",
        "typeId",
        "type_name",
        "playerName",
        "x",
        "y",
        "timeMin",
        "timeSec",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Mean Positions requires columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df_team = df[
        df["team_name"].eq(
            team_name
        )
    ].copy()

    if df_team.empty:
        return (
            pd.DataFrame(),
            empty_summary,
        )

    df_team["_player_key"] = df_team.apply(
        _mean_position_player_key,
        axis=1,
    )

    windows = _mean_position_match_windows(
        df
    )
    selected_period = str(
        period or "full"
    ).lower()

    if selected_period not in windows:
        raise ValueError(
            "period must be one of: full, 1h, 2h"
        )

    selected_window = windows[
        selected_period
    ]

    intervals = _mean_position_on_pitch_intervals(
        df_team,
        windows["full"][1],
    )

    minutes_by_player = {
        player_key: _mean_position_minutes_in_window(
            interval,
            selected_window,
        )
        for player_key, interval in intervals.items()
    }

    representative_keys = _mean_position_representatives(
        df_team,
        minutes_by_player,
        min_minutes,
    )

    period_mask = _mean_position_period_mask(
        df_team,
        selected_period,
    )

    type_names = df_team["type_name"].astype(
        str
    )

    touch_mask = type_names.isin(
        MEAN_POSITION_TOUCH_TYPES
    )

    x_numeric = pd.to_numeric(
        df_team["x"],
        errors="coerce",
    )
    y_numeric = pd.to_numeric(
        df_team["y"],
        errors="coerce",
    )

    touch_rows = df_team.loc[
        period_mask
        & touch_mask
        & x_numeric.notna()
        & y_numeric.notna()
        & df_team["_player_key"].notna()
    ].copy()

    if touch_rows.empty:
        return (
            pd.DataFrame(),
            empty_summary,
        )

    touch_rows["x"] = pd.to_numeric(
        touch_rows["x"],
        errors="coerce",
    )
    touch_rows["y"] = pd.to_numeric(
        touch_rows["y"],
        errors="coerce",
    )

    total_team_touches = int(
        len(touch_rows)
    )

    player_records = []

    for player_key, group in touch_rows.groupby(
        "_player_key",
        sort=False,
    ):
        minutes_played = float(
            minutes_by_player.get(
                player_key,
                0.0,
            )
        )

        if minutes_played < float(
            min_minutes
        ):
            continue

        if (
            representative_keys
            and player_key
            not in representative_keys
        ):
            continue

        median_x = float(
            group["x"].median()
        )
        median_y = float(
            group["y"].median()
        )

        q25_x = float(
            group["x"].quantile(0.25)
        )
        q75_x = float(
            group["x"].quantile(0.75)
        )
        q25_y = float(
            group["y"].quantile(0.25)
        )
        q75_y = float(
            group["y"].quantile(0.75)
        )

        dx_m = (
            group["x"] - median_x
        ) * 1.05
        dy_m = (
            group["y"] - median_y
        ) * 0.68

        radial_distance_m = np.sqrt(
            dx_m.pow(2)
            + dy_m.pow(2)
        )

        dispersion_m = float(
            radial_distance_m.median()
        )

        sample_row = group.iloc[0]

        jersey = pd.to_numeric(
            group.get(
                "Mapped Jersey Number",
                pd.Series(
                    index=group.index,
                    dtype=float,
                ),
            ),
            errors="coerce",
        ).dropna()

        role_values = (
            group.get(
                "positional_role",
                pd.Series(
                    index=group.index,
                    dtype=object,
                ),
            )
            .dropna()
            .astype(str)
        )

        touch_count = int(
            len(group)
        )

        touch_share = (
            100.0
            * touch_count
            / total_team_touches
            if total_team_touches
            else 0.0
        )

        player_records.append(
            {
                "player_key": player_key,
                "playerName": str(
                    sample_row.get(
                        "playerName",
                        "Unknown",
                    )
                ),
                "playerId": (
                    sample_row.get(
                        "playerId"
                    )
                    if "playerId" in group.columns
                    else None
                ),
                "Mapped Jersey Number": (
                    int(
                        jersey.iloc[0]
                    )
                    if not jersey.empty
                    else None
                ),
                "positional_role": (
                    role_values.iloc[0]
                    if not role_values.empty
                    else "Unknown"
                ),
                "median_x": median_x,
                "median_y": median_y,
                "q25_x": q25_x,
                "q75_x": q75_x,
                "q25_y": q25_y,
                "q75_y": q75_y,
                "iqr_x": max(
                    0.0,
                    q75_x - q25_x,
                ),
                "iqr_y": max(
                    0.0,
                    q75_y - q25_y,
                ),
                "dispersion_m": dispersion_m,
                "touch_count": touch_count,
                "touch_share": touch_share,
                "minutes_played": minutes_played,
            }
        )

    player_profile = pd.DataFrame(
        player_records
    )

    if player_profile.empty:
        summary = dict(
            empty_summary
        )
        summary["touches"] = (
            total_team_touches
        )
        return (
            player_profile,
            summary,
        )

    role_upper = player_profile[
        "positional_role"
    ].astype(str).str.upper()

    outfield = player_profile[
        ~role_upper.eq("GK")
        & ~role_upper.str.contains(
            "GOALKEEP",
            na=False,
        )
    ].copy()

    structural = (
        outfield
        if not outfield.empty
        else player_profile
    )

    centroid_x = float(
        structural["median_x"].mean()
    )
    centroid_y = float(
        structural["median_y"].mean()
    )

    summary = {
        "period": selected_period,
        "min_minutes": float(
            min_minutes
        ),
        "eligible_players": int(
            len(player_profile)
        ),
        "representative_players": int(
            len(player_profile)
        ),
        "touches": total_team_touches,
        "team_length_m": float(
            (
                structural["median_x"].max()
                - structural["median_x"].min()
            )
            * 1.05
        ),
        "team_width_m": float(
            (
                structural["median_y"].max()
                - structural["median_y"].min()
            )
            * 0.68
        ),
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "average_height_m": float(
            centroid_x * 1.05
        ),
        "structural_min_x": float(
            structural["median_x"].min()
        ),
        "structural_max_x": float(
            structural["median_x"].max()
        ),
        "structural_min_y": float(
            structural["median_y"].min()
        ),
        "structural_max_y": float(
            structural["median_y"].max()
        ),
    }

    return (
        player_profile.sort_values(
            [
                "median_x",
                "median_y",
            ],
            kind="stable",
        ).reset_index(
            drop=True
        ),
        summary,
    )
