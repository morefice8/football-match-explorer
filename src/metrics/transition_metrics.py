# src/metrics/transition_metrics.py
import logging
logger = logging.getLogger(__name__)

import pandas as pd
from src.utils.derived_cache import cache_derived_result
from src.metrics.data_quality import attach_sequence_coverage
import numpy as np
from collections import defaultdict
from src import config
from src.utils.sequence_outcomes import apply_sequence_outcome_contract
# --- Set display options to show all columns and more rows ---
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.max_rows', None)    # Show all rows (be careful with very large DFs)
pd.set_option('display.width', None)       # Auto-detect width to avoid line wrapping if possible
pd.set_option('display.max_colwidth', None) # Show full content of each column

# Define Opta typeIds (ensure these match your df_processed['typeId'] values)
TACKLE_ID = 7
BALL_RECOVERY_ID = 49  # Often used for loose ball recoveries
PASS_ID = 1

# Transition-domain contract. Keep the default in one place while still
# allowing callers/tests to override it explicitly when needed.
TRANSITION_WINDOW_SECONDS = 12.0

# The 12-second value defines the active transition-development window.
# A terminal shot/goal can still complete the same transition shortly after
# the deadline, but only if the attack had already reached the final third
# inside the base window and possession remained continuous.
TRANSITION_TERMINAL_GRACE_SECONDS = 4.0
TRANSITION_ADVANCED_X_THRESHOLD = 66.67

# Explicit possession-gain markers that confirm the gaining team has
# controlled the ball. Recording these markers also keeps zero-pass
# transitions observable.
TRANSITION_RECOVERY_EVENT_TYPES = frozenset({
    'Ball recovery',
    'Tackle',
    'Interception',
})

# Opta may describe one physical duel with two adjacent rows:
#   attacking player -> Dispossessed
#   opponent         -> Tackle/Challenge Unsuccessful
# The failed opponent action is evidence that control was NOT won, so the
# transition must not be terminated solely by the Dispossessed row.
FAILED_OPPONENT_CHALLENGE_TYPES = frozenset({
    'Tackle',
    'Challenge',
})
FAILED_CHALLENGE_PAIR_WINDOW_SECONDS = 1.0
FAILED_CHALLENGE_PAIR_MAX_DISTANCE = 5.0

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

# Controlled actions that can establish possession before an Opta Error.
# Keep this conservative: an Error is a transition trigger only when the team
# committing it had demonstrable control immediately beforehand.
ERROR_PRIOR_CONTROL_EVENT_TYPES = frozenset({
    'Pass',
    'Take On',
    'Ball recovery',
    'Interception',
    'Tackle',
    'Keeper pick-up',
    'Claim',
    'Ball touch',
})

ERROR_PRIOR_CONTROL_LOOKBACK_SECONDS = 6.0


def error_has_prior_team_control(
    df,
    error_idx,
    team_name,
    max_lookback_seconds=ERROR_PRIOR_CONTROL_LOOKBACK_SECONDS,
):
    """
    Return True only when an Opta Error is preceded by evidence that the
    error team actually controlled possession.

    This prevents defensive errors occurring during an opponent possession
    from being misclassified as fresh transition regains.

    The scan:
    - never crosses a period boundary;
    - ignores administrative / failed-opponent events that do not establish
      control;
    - accepts a recent successful controlled action by ``team_name``;
    - rejects once the opponent is seen to have successful control.
    """
    if error_idx < 0 or error_idx >= len(df):
        return False

    error_event = df.iloc[error_idx]

    if error_event.get('type_name') != 'Error':
        return True

    error_period = error_event.get('periodId')
    error_time = pd.to_numeric(
        error_event.get('total_seconds'),
        errors='coerce',
    )

    administrative_types = {
        'Card',
        'Foul',
        'Unknown',
        'Unknown Type',
    }

    clear_team_losses = {
        'Dispossessed',
        'Offside Pass',
        'Out',
    }

    for candidate_idx in range(error_idx - 1, -1, -1):
        candidate = df.iloc[candidate_idx]

        candidate_period = candidate.get('periodId')
        if (
            pd.notna(error_period)
            and pd.notna(candidate_period)
            and candidate_period != error_period
        ):
            break

        candidate_time = pd.to_numeric(
            candidate.get('total_seconds'),
            errors='coerce',
        )

        if pd.notna(error_time) and pd.notna(candidate_time):
            elapsed = float(error_time) - float(candidate_time)

            if elapsed < 0:
                continue

            if elapsed > max_lookback_seconds:
                break

        event_type = candidate.get('type_name')
        outcome = candidate.get('outcome')
        candidate_team = candidate.get('team_name')

        if event_type in administrative_types:
            continue

        if candidate_team == team_name:
            if event_type in clear_team_losses:
                return False

            if (
                event_type in ERROR_PRIOR_CONTROL_EVENT_TYPES
                and outcome == 'Successful'
            ):
                return True

            # A failed same-team possession action is stronger evidence that
            # control was already lost before the Error.
            if (
                event_type in {'Pass', 'Take On', 'Aerial', 'Challenge'}
                and outcome == 'Unsuccessful'
            ):
                return False

            continue

        # Opponent unsuccessful actions do not prove possession changed.
        if outcome == 'Unsuccessful':
            continue

        # A successful controlled opponent action means the Error happened
        # during the opponent's possession, so it is not a regain trigger.
        if (
            event_type in ERROR_PRIOR_CONTROL_EVENT_TYPES
            and outcome == 'Successful'
        ):
            return False

    return False


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
    logger.debug("Identifying recoveries and subsequent first passes...")

    # Ensure necessary columns exist
    required_cols = ['eventId', 'typeId', 'team_name', 'playerName', 'x', 'y',
                     'end_x', 'end_y', 'outcome', 'Mapped Jersey Number']
    # Check for 'Out of play' if filtering tackles (assuming it's a renamed qualifier)
    OUT_OF_PLAY_COL = 'Out of play'
    if OUT_OF_PLAY_COL not in df_processed.columns:
        logger.warning(f"Warning: Column '{OUT_OF_PLAY_COL}' not found. Tackles won't be filtered for staying in play.")
        # Decide if this is critical. For now, proceed without it.

    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        # If OUT_OF_PLAY_COL was optional and missing, remove it from the error message
        if OUT_OF_PLAY_COL not in df_processed.columns and OUT_OF_PLAY_COL in missing:
            missing.remove(OUT_OF_PLAY_COL)
        if missing:
            logger.warning(f"Error: Missing required columns for recovery analysis: {missing}")
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
        logger.debug("No recovery events (tackles in play / ball recoveries) found.")
        return pd.DataFrame()

    logger.info(f"Found {len(df_recoveries)} potential recovery events.")

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
        logger.debug("No recoveries were immediately followed by a successful pass by the same team.")
        return pd.DataFrame()

    df_final = pd.DataFrame(recovery_first_pass_data)
    logger.info(f"Found {len(df_final)} recovery-to-first-pass sequences.")
    return df_final

# --- Function: Find Opponent Buildup After Specific Team's Loss ---
@cache_derived_result("transition_sequences")
def find_buildup_after_possession_loss(df_processed,
                                       team_that_lost_possession, # Team that lost possession
                                       possession_loss_types=['Pass', 'Take On', 'Error', 'Dispossessed', 'Ball touch', 'Aerial', 'Challenge', 'Clearance', 'Save'], # Types of loss
                                       max_passes_in_buildup_sequence=35,
                                       shot_types=['Goal', 'Miss', 'Attempt Saved', 'Post'],
                                       metric_to_analyze='defensive_transitions',
                                       max_transition_seconds=TRANSITION_WINDOW_SECONDS):
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

    # --- Define Base and Optional Columns to Select ---
    # Required columns always needed
    required_cols = ['id', 'eventId', 'team_name', 'type_name', 'outcome', 'x', 'y',
                     'end_x', 'end_y', 'playerName', 'Mapped Jersey Number',
                     'timeMin', 'timeSec']
    # Optional columns that may be present
    optional_cols = [
        'receiver',
        'receiver_jersey_number',
        'Own goal',
        'Penalty',
        'From corner',
        'Goal mouth y co-ordinate',
        'periodId',
        'Red card',
        'Second yellow',
        'Out of play',
    ]

    all_teams = df_processed['team_name'].unique()
    team_that_gained_possession = [t for t in all_teams if t != team_that_lost_possession][0]

    # Check base requirements
    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        logger.warning(f"Error: Missing required columns: {missing}");
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

    df = df_processed[cols_to_select].copy()
    df = df.reset_index(drop=True)
    df['total_seconds'] = df['timeMin'] * 60 + df['timeSec']

    # --- Identify Possession Loss Events by 'team_that_lost_possession' ---
    loss_filter = pd.Series(False, index=df.index)
    # ... (build loss_filter based on possession_loss_types) ...
    #if 'Goal' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Goal'))
    if 'Pass' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Pass') & (df['outcome'] == 'Unsuccessful'))
    if 'Take On' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Take On') & (df['outcome'] == 'Unsuccessful'))
    if 'Error' in possession_loss_types:
        error_candidates = (
            (df['team_name'] == team_that_lost_possession)
            & (df['type_name'] == 'Error')
        )

        confirmed_error_losses = pd.Series(
            False,
            index=df.index,
        )

        for error_idx in df.index[error_candidates]:
            confirmed_error_losses.loc[error_idx] = (
                error_has_prior_team_control(
                    df,
                    error_idx,
                    team_that_lost_possession,
                )
            )

        loss_filter |= confirmed_error_losses
    if 'Dispossessed' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Dispossessed'))
    if 'Ball touch' in possession_loss_types:
        loss_filter |= (
            (df['team_name'] == team_that_lost_possession)
            & (df['type_name'] == 'Ball touch')
            & (df['outcome'] == 'Unsuccessful')
        )
    #if 'Clearance' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Clearance') & (df['outcome'] == 'Unsuccessful'))
    if 'Clearance' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Clearance'))
    if 'Aerial' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Aerial') & (df['outcome'] == 'Unsuccessful'))
    if 'Challenge' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_lost_possession) & (df['type_name'] == 'Challenge') & (df['outcome'] == 'Unsuccessful'))
    if 'Save' in possession_loss_types: loss_filter |= ((df['team_name'] == team_that_gained_possession) & (df['type_name'] == 'Save') & (df['outcome'] == 'Successful'))

    # Offensive transitions can start from an explicit Opta possession-gain
    # marker even when the opponent's preceding loss is ambiguous or is not
    # represented by one of the legacy loss-event types above.
    #
    # This is intentionally a fallback candidate source. If the same recovery
    # marker is already observed inside a transition opened by an opponent
    # loss, ``processed_loss_event_ids`` suppresses the duplicate candidate
    # later in the loop.
    if metric_to_analyze == 'offensive_transitions':
        direct_gain_filter = (
            (df['team_name'] == team_that_gained_possession)
            & (df['type_name'].isin(TRANSITION_RECOVERY_EVENT_TYPES))
            & (df['outcome'] == 'Successful')
        )

        if 'Out of play' in df.columns:
            direct_gain_filter &= ~(
                (df['type_name'] == 'Tackle')
                & (df['Out of play'].isin([1, '1', True]))
            )

        loss_filter |= direct_gain_filter

    df_losses_raw = df[loss_filter].copy()
    if df_losses_raw.empty: return pd.DataFrame()

    df_losses = df_losses_raw.drop_duplicates(subset=['id'], keep='first').copy()
    if df_losses.empty: return pd.DataFrame()

    # REL-10B: detector candidate count is measured after
    # canonical loss eligibility/deduplication and before UI filters.
    coverage_candidate_count = int(len(df_losses))

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

    def _action_in_transition_frame(event):
        """
        Return an event expressed in the coordinate frame of the
        team currently in transition.

        Processed Match Analysis coordinates are team-relative.
        Events belonging to the opposite team therefore require
        a 180-degree rotation before they can coexist in the same
        transition sequence.
        """
        action_data = event.to_dict()

        if event.get('team_name') == team_building_up:
            return action_data

        for coord in (
            'x',
            'y',
            'end_x',
            'end_y',
        ):
            value = pd.to_numeric(
                action_data.get(coord),
                errors='coerce',
            )

            if pd.notna(value):
                action_data[coord] = (
                    100.0 - float(value)
                )

        return action_data

    def _find_paired_failed_opponent_challenge(
        dispossessed_idx,
        max_lookahead_seconds=FAILED_CHALLENGE_PAIR_WINDOW_SECONDS,
        max_distance=FAILED_CHALLENGE_PAIR_MAX_DISTANCE,
    ):
        """
        Return the opponent Tackle/Challenge row when it is the paired failed
        side of the same duel as a gaining-team Dispossessed event.

        The pair must:
        - belong to the same period;
        - occur almost immediately afterwards;
        - be by the team that had lost possession;
        - be an unsuccessful Tackle/Challenge;
        - occur at the same location after converting both events into the
          transition team's coordinate frame.

        If coordinates are missing, the strict temporal/event contract is used
        rather than inventing a possession change from unavailable geometry.
        """
        dispossessed = df.iloc[dispossessed_idx]

        if (
            dispossessed.get('team_name') != team_building_up
            or dispossessed.get('type_name') != 'Dispossessed'
        ):
            return None

        start_period = dispossessed.get('periodId')
        start_time = dispossessed.get('total_seconds')
        start_data = _action_in_transition_frame(dispossessed)

        start_x = pd.to_numeric(
            start_data.get('x'),
            errors='coerce',
        )
        start_y = pd.to_numeric(
            start_data.get('y'),
            errors='coerce',
        )

        for candidate_idx in range(dispossessed_idx + 1, len(df)):
            candidate = df.iloc[candidate_idx]

            candidate_period = candidate.get('periodId')
            if (
                pd.notna(start_period)
                and pd.notna(candidate_period)
                and candidate_period != start_period
            ):
                break

            candidate_time = candidate.get('total_seconds')
            if pd.notna(start_time) and pd.notna(candidate_time):
                elapsed = float(candidate_time) - float(start_time)

                if elapsed < 0:
                    continue

                if elapsed > max_lookahead_seconds:
                    break

            if candidate.get('team_name') != team_that_lost_possession:
                continue

            if candidate.get('type_name') not in FAILED_OPPONENT_CHALLENGE_TYPES:
                continue

            if candidate.get('outcome') != 'Unsuccessful':
                continue

            if candidate.get('Out of play') in [1, '1', True]:
                continue

            candidate_data = _action_in_transition_frame(candidate)
            candidate_x = pd.to_numeric(
                candidate_data.get('x'),
                errors='coerce',
            )
            candidate_y = pd.to_numeric(
                candidate_data.get('y'),
                errors='coerce',
            )

            if (
                pd.notna(start_x)
                and pd.notna(start_y)
                and pd.notna(candidate_x)
                and pd.notna(candidate_y)
            ):
                distance = float(
                    np.hypot(
                        float(candidate_x) - float(start_x),
                        float(candidate_y) - float(start_y),
                    )
                )

                if distance > max_distance:
                    continue

            return candidate

        return None

    def _find_immediate_penalty_award(
        start_idx,
        max_lookahead_seconds=2.0,
    ):
        """
        Find the penalty-award foul immediately following an action.

        In Opta eventing a dribble/take-on can be recorded as unsuccessful
        just before the successful Foul event carrying qualifier 9. Without
        this look-ahead the transition is incorrectly terminated as a lost
        possession before the penalty award is reached.
        """
        start_event = df.iloc[start_idx]
        start_period = start_event.get('periodId')
        start_time = start_event.get('total_seconds')
        fallback_event = None

        for candidate_idx in range(start_idx + 1, len(df)):
            candidate = df.iloc[candidate_idx]
            candidate_period = candidate.get('periodId')

            if (
                pd.notna(start_period)
                and pd.notna(candidate_period)
                and candidate_period != start_period
            ):
                break

            candidate_time = candidate.get('total_seconds')
            if (
                pd.notna(start_time)
                and pd.notna(candidate_time)
                and float(candidate_time) - float(start_time)
                    > max_lookahead_seconds
            ):
                break

            if candidate.get('type_name') != 'Foul':
                continue

            if candidate.get('Penalty') not in [1, '1', True]:
                continue

            if (
                candidate.get('team_name') == team_building_up
                and candidate.get('outcome') == 'Successful'
            ):
                return candidate

            if (
                fallback_event is None
                and candidate.get('team_name')
                    == team_that_lost_possession
                and candidate.get('outcome') == 'Unsuccessful'
            ):
                fallback_event = candidate

        return fallback_event

    def _first_gaining_team_x_after_trigger(
        trigger_idx,
    ):
        """
        Find the first known location of the team that gains
        possession after the trigger.

        Coordinates of that event are already expressed in
        the gaining team's attacking direction.
        """

        trigger_event = df.iloc[
            trigger_idx
        ]

        trigger_time = trigger_event.get(
            'total_seconds',
            np.nan,
        )

        trigger_period = trigger_event.get(
            'periodId',
            np.nan,
        )

        for candidate_idx in range(
            trigger_idx + 1,
            len(df),
        ):
            candidate = df.iloc[
                candidate_idx
            ]

            # Never borrow a recovery location
            # from another period.
            candidate_period = candidate.get(
                'periodId',
                np.nan,
            )

            if (
                pd.notna(trigger_period)
                and pd.notna(candidate_period)
                and candidate_period
                    != trigger_period
            ):
                break

            candidate_time = candidate.get(
                'total_seconds',
                np.nan,
            )

            if (
                pd.notna(trigger_time)
                and pd.notna(candidate_time)
            ):
                elapsed = (
                    float(candidate_time)
                    - float(trigger_time)
                )

                if elapsed > max_transition_seconds:
                    break

            if (
                candidate.get('team_name')
                == team_that_gained_possession
                and pd.notna(
                    candidate.get('x')
                )
            ):
                return float(
                    candidate.get('x')
                )

        return np.nan

    for loss_original_df_idx in df_losses.index:
        loss_event = df.iloc[loss_original_df_idx]
        loss_period = loss_event.get('periodId')
        loss_total_seconds = loss_event.get('total_seconds')

        # Zone where possession was gained by the opponent
        loss_x = np.nan
        loss_y = np.nan

        # Zone where possession was lost / gained.
        if metric_to_analyze == 'defensive_transitions':

            if (
                loss_event.get('type_name') == 'Pass'
                and pd.notna(
                    loss_event.get('end_x')
                )
            ):
                # For an unsuccessful pass, the turnover
                # happens at the pass destination.
                loss_x = pd.to_numeric(
                    loss_event.get('end_x'),
                    errors='coerce',
                )

                loss_y = pd.to_numeric(
                    loss_event.get('end_y'),
                    errors='coerce',
                )

            else:
                # Point-like turnovers:
                # dispossession, duel, clearance, etc.
                loss_x = pd.to_numeric(
                    loss_event.get('x'),
                    errors='coerce',
                )

                loss_y = pd.to_numeric(
                    loss_event.get('y'),
                    errors='coerce',
                )

            loss_zone = get_pitch_third(
                loss_x
            )

        else:  # Offensive transitions

            event_type = loss_event.get(
                'type_name'
            )

            # -----------------------------------------------------
            # 1. Recovery event recorded directly for the team
            #    that gains possession.
            # -----------------------------------------------------

            is_direct_recovery_trigger = (
                loss_event.get('team_name')
                == team_building_up
                and event_type
                in TRANSITION_RECOVERY_EVENT_TYPES
                and loss_event.get('outcome')
                == 'Successful'
                and not (
                    event_type == 'Tackle'
                    and loss_event.get('Out of play')
                    in [1, '1', True]
                )
            )

            if is_direct_recovery_trigger:

                loss_x = pd.to_numeric(
                    loss_event.get('x'),
                    errors='coerce',
                )
                loss_y = pd.to_numeric(
                    loss_event.get('y'),
                    errors='coerce',
                )
                recovery_coord = loss_x

            elif event_type == 'Save':

                recovery_coord = loss_event.get(
                    'x',
                    np.nan,
                )

            # -----------------------------------------------------
            # 2. Point-like turnover events recorded from the
            #    losing team's coordinate system.
            # -----------------------------------------------------

            elif event_type in (
                'Aerial',
                'Dispossessed',
                'Challenge',
                'Take On',
                'Error',
            ):

                loss_x = loss_event.get(
                    'x',
                    np.nan,
                )

                recovery_coord = (
                    100.0 - float(loss_x)
                    if pd.notna(loss_x)
                    else np.nan
                )

            # -----------------------------------------------------
            # 3. Events with a known destination.
            # -----------------------------------------------------

            elif pd.notna(
                loss_event.get('end_x')
            ):

                recovery_coord = (
                    100.0
                    - float(
                        loss_event.get('end_x')
                    )
                )

            # -----------------------------------------------------
            # 4. Missing destination, e.g. Clearance.
            #
            #    Use the first actual location recorded for the
            #    team that gains possession.
            # -----------------------------------------------------

            else:

                recovery_coord = (
                    _first_gaining_team_x_after_trigger(
                        loss_original_df_idx
                    )
                )

            loss_zone = get_pitch_third(
                recovery_coord
            )

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
        termination_reason_override = None
        reached_final_third_in_base_window = False
        is_direct_recovery_trigger = (
            metric_to_analyze == 'offensive_transitions'
            and loss_event.get('team_name')
                == team_building_up
            and loss_event.get('type_name')
                in TRANSITION_RECOVERY_EVENT_TYPES
            and loss_event.get('outcome')
                == 'Successful'
            and not (
                loss_event.get('type_name') == 'Tackle'
                and loss_event.get('Out of play')
                    in [1, '1', True]
            )
        )

        # Loss-triggered sequences begin with the event after the loss.
        # Explicit recovery-triggered sequences begin with the recovery marker
        # itself so the transition clock, recovery zone and first plotted action
        # all share the same canonical start.
        current_event_original_df_idx = (
            loss_original_df_idx - 1
            if is_direct_recovery_trigger
            else loss_original_df_idx
        )

        # Trace forward to find the opponent's sequence
        if loss_event['id'] in processed_loss_event_ids:
            continue
        if loss_event['id'] not in processed_loss_event_ids:
            while current_event_original_df_idx < len(df) - 1 and num_passes_in_seq < max_passes_in_buildup_sequence:
                current_event_original_df_idx += 1 # Move to the event *after* the loss or last pass

                action_by_gaining_team = df.iloc[
                current_event_original_df_idx
                ]

                # 1. Never cross period boundary
                action_period = action_by_gaining_team.get(
                    'periodId'
                )

                if (
                    pd.notna(loss_period)
                    and pd.notna(action_period)
                    and action_period != loss_period
                ):
                    if current_opponent_sequence_events:
                        termination_reason_override = (
                            'period_boundary'
                        )
                    break

                # 2. Transition timing contract
                #
                # The base window describes ACTIVE transition development.
                # It is deliberately not a blind hard cutoff for a shot that
                # completes an already-advanced attack a few seconds later.
                action_total_seconds = (
                    action_by_gaining_team.get(
                        'total_seconds'
                    )
                )

                if (
                    pd.notna(loss_total_seconds)
                    and pd.notna(action_total_seconds)
                ):
                    elapsed_seconds = (
                        action_total_seconds
                        - loss_total_seconds
                    )

                    # Track whether the team in transition reached the final
                    # third while the base window was still active. This is the
                    # eligibility condition for terminal-action grace.
                    if (
                        elapsed_seconds <= max_transition_seconds
                        and action_by_gaining_team.get('team_name')
                            == team_building_up
                    ):
                        action_x = pd.to_numeric(
                            action_by_gaining_team.get('x'),
                            errors='coerce',
                        )
                        action_end_x = pd.to_numeric(
                            action_by_gaining_team.get('end_x'),
                            errors='coerce',
                        )

                        if (
                            (
                                pd.notna(action_x)
                                and float(action_x)
                                    >= TRANSITION_ADVANCED_X_THRESHOLD
                            )
                            or (
                                pd.notna(action_end_x)
                                and float(action_end_x)
                                    >= TRANSITION_ADVANCED_X_THRESHOLD
                            )
                        ):
                            reached_final_third_in_base_window = True

                    if elapsed_seconds > max_transition_seconds:
                        action_is_terminal_shot = (
                            action_by_gaining_team.get('team_name')
                                == team_building_up
                            and action_by_gaining_team.get('type_name')
                                in shot_types
                        )

                        inside_terminal_grace = (
                            elapsed_seconds
                            <= (
                                max_transition_seconds
                                + TRANSITION_TERMINAL_GRACE_SECONDS
                            )
                        )

                        can_complete_advanced_transition = (
                            action_is_terminal_shot
                            and inside_terminal_grace
                            and reached_final_third_in_base_window
                        )

                        if can_complete_advanced_transition:
                            # Keep tracing this single terminal action through
                            # the normal shot/goal outcome logic below.
                            termination_reason_override = (
                                'terminal_action_grace'
                            )
                        else:
                            if current_opponent_sequence_events:
                                if (
                                    metric_to_analyze
                                    == 'defensive_transitions'
                                ):
                                    sequence_outcome_type = (
                                        'Opponent Possession Consolidated'
                                    )
                                else:
                                    sequence_outcome_type = (
                                        'Possession Consolidated'
                                    )

                                termination_reason_override = (
                                    'time_window_elapsed'
                                )

                            break

                action_data = _action_in_transition_frame(action_by_gaining_team)
                action_data['loss_sequence_id'] = sequence_id_counter
                action_data['loss_zone'] = loss_zone
                action_data['loss_x'] = loss_x
                action_data['loss_y'] = loss_y
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
                is_unknown = (action_by_gaining_team['type_name'] in ('Unknown','Unknown Type',))
                is_successful_event = (action_by_gaining_team['outcome'] == 'Successful')
                is_team_that_lost_possession = (action_by_gaining_team['team_name'] == team_that_lost_possession)
                is_not_successful_event = (action_by_gaining_team['outcome'] == 'Unsuccessful')
                is_shot = (action_by_gaining_team['type_name'] in shot_types)
                is_end_sequence = (action_by_gaining_team['type_name'] in ('Foul', 'Out', 'Keeper pick-up', 'Claim', 'Dispossessed', 'Offside Pass', 'Corner Awarded'))
                is_take_on = (action_by_gaining_team['type_name'] == 'Take On')
                is_ball_touch = (action_by_gaining_team['type_name'] == 'Ball touch')
                is_out_of_play = (
                    action_by_gaining_team.get('Out of play')
                    in [1, '1', True]
                )

                is_recovery_marker = (
                    action_by_gaining_team['type_name']
                    in TRANSITION_RECOVERY_EVENT_TYPES
                    and not (
                        action_by_gaining_team['type_name']
                        == 'Tackle'
                        and is_out_of_play
                    )
                )
                is_card = (
                    action_by_gaining_team['type_name']
                    == 'Card'
                )
                is_from_corner = (action_by_gaining_team.get('From corner') in [1, '1', True])
                is_penalty_awarded = (action_by_gaining_team.get('Penalty') in [1, '1', True])
                is_penalty_for_building_team = (
                    is_penalty_awarded
                    and (
                        (
                            action_by_gaining_team['team_name']
                            == team_building_up
                            and is_successful_event
                        )
                        or (
                            action_by_gaining_team['team_name']
                            == team_that_lost_possession
                            and is_not_successful_event
                        )
                    )
                )

                # Cards (including dismissals) are administrative events,
                # not evidence that possession changed. Keep tracing the
                # transition around them.
                if is_card:
                    continue

                # A penalty award is a terminal event even when it is the
                # first event after the turnover. Do not require an earlier
                # pass/action in the transition before recording it.
                if (
                    action_by_gaining_team['type_name'] == 'Foul'
                    and is_penalty_for_building_team
                ):
                    current_opponent_sequence_events.append(action_data)
                    if metric_to_analyze == 'defensive_transitions':
                        sequence_outcome_type = 'Penalty conceded'
                    else:
                        sequence_outcome_type = 'Penalty won'
                    break

                # Opta can emit an unsuccessful take-on immediately before
                # the penalty-award foul. Look ahead briefly before treating
                # the current gaining-team action as a turnover.
                if is_correct_team:
                    penalty_award_event = (
                        _find_immediate_penalty_award(
                            current_event_original_df_idx
                        )
                    )

                    if penalty_award_event is not None:
                        current_opponent_sequence_events.append(action_data)

                        penalty_data = _action_in_transition_frame(penalty_award_event)
                        penalty_data['loss_sequence_id'] = sequence_id_counter
                        penalty_data['loss_zone'] = loss_zone
                        penalty_data['loss_x'] = loss_x
                        penalty_data['loss_y'] = loss_y
                        penalty_data['triggering_loss_Opta_id'] = loss_event['id']
                        penalty_data['timeMin_at_loss'] = time_min_at_loss
                        penalty_data['timeSec_at_loss'] = time_sec_at_loss
                        penalty_data['type_of_initial_loss'] = type_of_loss
                        current_opponent_sequence_events.append(penalty_data)

                        if metric_to_analyze == 'defensive_transitions':
                            sequence_outcome_type = 'Penalty conceded'
                        else:
                            sequence_outcome_type = 'Penalty won'
                        break

                # A Dispossessed row is not always a confirmed turnover.
                # Opta can immediately pair it with an unsuccessful opponent
                # Tackle/Challenge at the same time/location. In that case the
                # attacking team retained the loose ball and the transition
                # remains alive. The opponent row itself will be skipped by the
                # existing failed-regain branch on the next loop iteration.
                if (
                    is_correct_team
                    and action_by_gaining_team['type_name'] == 'Dispossessed'
                    and _find_paired_failed_opponent_challenge(
                        current_event_original_df_idx
                    ) is not None
                ):
                    current_opponent_sequence_events.append(
                        action_data
                    )
                    continue

                if (
                    is_end_sequence
                    and current_opponent_sequence_events
                ):
                    current_opponent_sequence_events.append(
                        action_data
                    )

                    if (
                        action_by_gaining_team['type_name']
                        == 'Foul'
                    ):
                        if is_penalty_awarded:
                            if (
                                metric_to_analyze
                                == 'defensive_transitions'
                            ):
                                sequence_outcome_type = (
                                    'Penalty conceded'
                                )
                            else:
                                sequence_outcome_type = (
                                    'Penalty won'
                                )
                        else:
                            sequence_outcome_type = 'Foul'

                    elif (
                        action_by_gaining_team['type_name']
                        == 'Offside Pass'
                    ):
                        sequence_outcome_type = 'Offside'

                    elif (
                        action_by_gaining_team['type_name']
                        == 'Out'
                    ):
                        sequence_outcome_type = 'Out'

                    elif (
                        action_by_gaining_team['type_name']
                        == 'Corner Awarded'
                    ):
                        sequence_outcome_type = 'Corner'

                    elif (
                        action_by_gaining_team['type_name']
                        == 'Dispossessed'
                    ):
                        if (
                            metric_to_analyze
                            == 'defensive_transitions'
                        ):
                            sequence_outcome_type = (
                                'Regained Possessions'
                            )
                        else:
                            sequence_outcome_type = (
                                'Lost Possessions'
                            )

                    elif (
                        action_by_gaining_team['type_name']
                        in ('Keeper pick-up', 'Claim')
                    ):
                        if (
                            action_by_gaining_team['team_name']
                            == team_that_lost_possession
                            and is_successful_event
                        ):
                            # The defending team regained control
                            # through its goalkeeper.
                            if (
                                metric_to_analyze
                                == 'defensive_transitions'
                            ):
                                sequence_outcome_type = (
                                    'Regained Possessions'
                                )
                            else:
                                sequence_outcome_type = (
                                    'Lost Possessions'
                                )

                        elif (
                            action_by_gaining_team['team_name']
                            == team_building_up
                            and is_successful_event
                        ):
                            # The team in transition retained the ball
                            # but the fast phase ended with keeper control.
                            if (
                                metric_to_analyze
                                == 'defensive_transitions'
                            ):
                                sequence_outcome_type = (
                                    'Opponent Possession Consolidated'
                                )
                            else:
                                sequence_outcome_type = (
                                    'Possession Consolidated'
                                )

                            termination_reason_override = (
                                'keeper_control'
                            )

                    break

                elif is_end_sequence:
                    # A terminal event alone does not prove that
                    # the opponent actually established possession.
                    break

                elif is_correct_team and is_pass:
                    if is_successful_event: #successful pass
                        current_opponent_sequence_events.append(action_data)
                        num_passes_in_seq += 1
                    elif is_not_successful_event:
                        current_opponent_sequence_events.append(
                            action_data
                        )

                        if metric_to_analyze == 'defensive_transitions':
                            sequence_outcome_type = (
                                'Regained Possessions'
                            )
                        else:
                            sequence_outcome_type = (
                                'Lost Possessions'
                            )

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
                elif (
                    is_correct_team
                    and is_recovery_marker
                    and is_successful_event
                ):
                    # An explicit recovery is a controlled action even when
                    # no pass follows it. Store it so immediate regains and
                    # zero-pass transitions remain observable.
                    current_opponent_sequence_events.append(
                        action_data
                    )
                    processed_loss_event_ids.add(
                        action_by_gaining_team['id']
                    )
                    continue
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

                elif (is_team_that_lost_possession and is_successful_event):
                    if metric_to_analyze == 'defensive_transitions':
                        sequence_outcome_type = (
                            'Regained Possessions'
                        )
                    else:
                        sequence_outcome_type = (
                            'Lost Possessions'
                        )

                    break

                else:  # Some other event or end of data
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

            df_seq_deduped['opponent_pass_count'] = pass_count
            df_seq_deduped['sequence_outcome_type'] = sequence_outcome_type
            viewpoint = (
                'defending'
                if metric_to_analyze == 'defensive_transitions'
                else 'attacking'
            )
            df_seq_deduped = apply_sequence_outcome_contract(
                df_seq_deduped,
                viewpoint=viewpoint,
                legacy_outcome=sequence_outcome_type,
                termination_reason=(
                    termination_reason_override
                ),
            )

            all_buildup_events_with_loss_info.extend(
                df_seq_deduped.to_dict('records')
            )

            sequence_id_counter += 1
            # print(f"DEBUG Sequence {sequence_id_counter}: {pass_count} real passes, {len(current_opponent_sequence_events)} events, num_passes = {num_passes_in_seq}, outcome = {sequence_outcome_type}")

    # --- End Loop ---

    if not all_buildup_events_with_loss_info:
        return attach_sequence_coverage(
            pd.DataFrame(),
            candidates=coverage_candidate_count,
            sequence_id_column="loss_sequence_id",
        )
    df_all_sequences = pd.DataFrame(all_buildup_events_with_loss_info)
    return attach_sequence_coverage(
        df_all_sequences,
        candidates=coverage_candidate_count,
        sequence_id_column="loss_sequence_id",
    )

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
    logger.debug(f"Calculating transition success rates by zone for {team_name}...")
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
    logger.debug(f"Transition success summary for {team_name}:\n{df_zone_summary}")
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
    terminal_outcomes = [
        seq.iloc[-1].get('terminal_outcome', 'unknown')
        for seq in sequence_list
        if not seq.empty
    ]
    terminal_outcome_counts = (
        pd.Series(terminal_outcomes).value_counts().to_dict()
    )

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
        "terminal_outcomes": terminal_outcome_counts,
        "flanks": flank_counts,
        "types": loss_type_counts,
        "transition_profile_table": pd.DataFrame(profile_table)
    }


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
    terminal_outcomes = [
        seq.iloc[-1].get('terminal_outcome', 'unknown')
        for seq in sequence_list
        if not seq.empty
    ]
    terminal_outcome_counts = (
        pd.Series(terminal_outcomes).value_counts().to_dict()
    )

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
        "terminal_outcomes": terminal_outcome_counts,
        "flanks": flank_counts,
        "types": recovery_type_counts, # Ora rappresenta i tipi di recupero
        "transition_profile_table": pd.DataFrame(profile_table_data)
    }

# NUOVA FUNZIONE per creare le card riassuntive offensive

