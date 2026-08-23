# src/data_processing/pass_processing.py
import math

import numpy as np
import pandas as pd
from src.utils.derived_cache import cache_derived_result
from ..metrics import pass_metrics  # Assuming this is a module you have for pass metrics


# Events that describe match administration rather than a new on-ball action.
# They may sit between a pass and the receiver's next technical event and must
# therefore be ignored when inferring the receiving player.
RECEIVER_ADMIN_EVENT_TYPE_IDS = frozenset({
    17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 30, 32, 34, 35, 36,
    37, 38, 39, 40, 43, 47,
})
RECEIVER_ADMIN_EVENT_NAMES = frozenset({
    'card', 'player off', 'player on', 'player retired', 'player returns',
    'start', 'end', 'start delay', 'end delay', 'team set up',
    'formation change', 'player changed position', 'deleted event',
    'rescinded card', 'collection end', 'temporary stop', 'temporary start',
})

RECEIVER_RESULT_COLUMNS = [
    'receiver',
    'receiver_player_id',
    'receiver_jersey_number',
    'receiver_event_id',
    'receiver_confidence',
    'receiver_is_reliable',
    'receiver_reason',
    'receiver_time_gap_seconds',
    'receiver_spatial_gap_m',
]


def _numeric(value):
    try:
        numeric = float(value)
        return numeric if math.isfinite(numeric) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _event_seconds(row):
    minute = _numeric(row.get('timeMin'))
    second = _numeric(row.get('timeSec'))
    if pd.isna(minute):
        return np.nan
    return minute * 60 + (0.0 if pd.isna(second) else second)


def _is_successful_pass(row):
    outcome = row.get('outcome')
    if isinstance(outcome, str):
        return outcome.strip().lower() == 'successful'
    return _numeric(outcome) == 1


def _same_team(pass_event, candidate):
    for column in ('contestantId', 'team_name'):
        pass_team = pass_event.get(column)
        candidate_team = candidate.get(column)
        if pd.notna(pass_team) and pd.notna(candidate_team):
            return str(pass_team) == str(candidate_team)
    return False


def _is_administrative_event(row):
    type_id = _numeric(row.get('typeId'))
    if pd.notna(type_id) and int(type_id) in RECEIVER_ADMIN_EVENT_TYPE_IDS:
        return True
    type_name = str(row.get('type_name') or '').strip().lower()
    return type_name in RECEIVER_ADMIN_EVENT_NAMES


@cache_derived_result("pass_receivers")
def infer_pass_receivers(
    df_processed,
    max_time_gap_seconds=12.0,
    high_confidence_distance_m=15.0,
    max_distance_m=25.0,
):
    """
    Infer pass receivers from the next credible on-ball event.

    Opta's event stream used by the app does not expose a receiver directly.
    A receiver is therefore assigned only when all of these conditions hold:

    - the pass is successful;
    - the first subsequent technical event is in the same period and belongs
      to the passing team;
    - the event occurs within ``max_time_gap_seconds``;
    - its start location is no more than ``max_distance_m`` from the pass end.

    Coordinates are converted from Opta's 0-100 scale to approximate metres
    before distance is evaluated (105 x 68 metre pitch). Ambiguous passes keep
    a null receiver and an explicit ``receiver_reason`` instead of being
    attributed to the next arbitrary row.
    """
    if df_processed is None or df_processed.empty:
        return pd.DataFrame(columns=RECEIVER_RESULT_COLUMNS)

    pass_mask = pd.Series(False, index=df_processed.index)
    if 'typeId' in df_processed.columns:
        pass_mask |= pd.to_numeric(df_processed['typeId'], errors='coerce').eq(1)
    if 'type_name' in df_processed.columns:
        pass_mask |= df_processed['type_name'].fillna('').astype(str).str.lower().eq('pass')

    pass_indices = df_processed.index[pass_mask]
    results = pd.DataFrame(index=pass_indices, columns=RECEIVER_RESULT_COLUMNS)
    if results.empty:
        return results

    results['receiver_is_reliable'] = False
    results['receiver_reason'] = 'no_next_technical_event'
    results['receiver_time_gap_seconds'] = np.nan
    results['receiver_spatial_gap_m'] = np.nan

    events = df_processed.copy()
    events['_receiver_source_index'] = events.index
    sort_columns = [
        column for column in
        ('periodId', 'timeMin', 'timeSec', 'timeStamp', 'id', 'eventId')
        if column in events.columns
    ]
    if sort_columns:
        events = events.sort_values(sort_columns, kind='mergesort')
    events = events.reset_index(drop=True)

    for position, pass_event in events.iterrows():
        source_index = pass_event['_receiver_source_index']
        is_pass = (
            _numeric(pass_event.get('typeId')) == 1
            or str(pass_event.get('type_name') or '').strip().lower() == 'pass'
        )
        if not is_pass or source_index not in results.index:
            continue

        if not _is_successful_pass(pass_event):
            results.at[source_index, 'receiver_reason'] = 'unsuccessful_pass'
            continue

        end_x = _numeric(pass_event.get('end_x'))
        end_y = _numeric(pass_event.get('end_y'))
        if pd.isna(end_x) or pd.isna(end_y):
            results.at[source_index, 'receiver_reason'] = 'missing_pass_endpoint'
            continue

        pass_seconds = _event_seconds(pass_event)
        pass_period = pass_event.get('periodId')
        candidate = None
        candidate_gap_seconds = np.nan

        for candidate_position in range(position + 1, len(events)):
            next_event = events.iloc[candidate_position]
            if pd.notna(pass_period) and pd.notna(next_event.get('periodId')):
                if str(next_event.get('periodId')) != str(pass_period):
                    break

            next_seconds = _event_seconds(next_event)
            gap_seconds = next_seconds - pass_seconds if pd.notna(pass_seconds) and pd.notna(next_seconds) else np.nan
            if pd.notna(gap_seconds) and gap_seconds < 0:
                continue
            if pd.notna(gap_seconds) and gap_seconds > max_time_gap_seconds:
                break

            if _is_administrative_event(next_event):
                continue
            if pd.isna(next_event.get('playerName')) and pd.isna(next_event.get('playerId')):
                continue
            if pd.isna(_numeric(next_event.get('x'))) or pd.isna(_numeric(next_event.get('y'))):
                continue

            candidate = next_event
            candidate_gap_seconds = gap_seconds
            break

        if candidate is None:
            continue

        results.at[source_index, 'receiver_time_gap_seconds'] = candidate_gap_seconds
        if not _same_team(pass_event, candidate):
            results.at[source_index, 'receiver_reason'] = 'possession_changed_before_next_touch'
            continue

        passer_id = pass_event.get('playerId')
        candidate_id = candidate.get('playerId')
        passer_name = pass_event.get('playerName')
        candidate_name = candidate.get('playerName')
        same_player = (
            pd.notna(passer_id) and pd.notna(candidate_id) and str(passer_id) == str(candidate_id)
        ) or (
            pd.isna(passer_id) and pd.isna(candidate_id)
            and pd.notna(passer_name) and pd.notna(candidate_name)
            and str(passer_name) == str(candidate_name)
        )
        if same_player:
            results.at[source_index, 'receiver_reason'] = 'same_player_next_event'
            continue

        delta_x_m = (_numeric(candidate.get('x')) - end_x) * 1.05
        delta_y_m = (_numeric(candidate.get('y')) - end_y) * 0.68
        spatial_gap_m = math.hypot(delta_x_m, delta_y_m)
        results.at[source_index, 'receiver_spatial_gap_m'] = spatial_gap_m
        if spatial_gap_m > max_distance_m:
            results.at[source_index, 'receiver_reason'] = 'spatial_mismatch'
            continue

        confidence = 'high' if spatial_gap_m <= high_confidence_distance_m else 'medium'
        results.at[source_index, 'receiver'] = candidate_name
        results.at[source_index, 'receiver_player_id'] = candidate_id
        results.at[source_index, 'receiver_jersey_number'] = candidate.get('Mapped Jersey Number')
        results.at[source_index, 'receiver_event_id'] = candidate.get('id') or candidate.get('eventId')
        results.at[source_index, 'receiver_confidence'] = confidence
        results.at[source_index, 'receiver_is_reliable'] = True
        results.at[source_index, 'receiver_reason'] = 'inferred_from_next_technical_event'

    results['receiver_is_reliable'] = results['receiver_is_reliable'].fillna(False).astype(bool)
    return results

CARRY_RESULT_COLUMNS = [
    'playerName',
    'playerId',
    'team_name',
    'contestantId',
    'x',
    'y',
    'end_x',
    'end_y',
    'carry_distance_m',
    'carry_time_gap_seconds',
    'carry_source_event_id',
    'carry_end_event_id',
    'carry_source_type',
    'carry_confidence',
    'carry_is_inferred',
    'carry_is_reliable',
]


def _same_player(event_a, event_b):
    player_a_id = event_a.get('playerId')
    player_b_id = event_b.get('playerId')

    if pd.notna(player_a_id) and pd.notna(player_b_id):
        return str(player_a_id) == str(player_b_id)

    player_a_name = event_a.get('playerName')
    player_b_name = event_b.get('playerName')

    return (
        pd.notna(player_a_name)
        and pd.notna(player_b_name)
        and str(player_a_name) == str(player_b_name)
    )


@cache_derived_result("carries")
def infer_carries(
    df_processed,
    min_distance_m=2.0,
    max_distance_m=30.0,
    max_time_gap_seconds=12.0,
):
    """
    Infer ball carries from the gap between consecutive credible technical events.

    A carry is accepted when possession continuity can be established and the
    movement from the previous event endpoint to the next event start is
    spatially and temporally plausible.

    Two situations are supported:

    1. Previous event is a successful pass:
       the next player must be the reliably inferred receiver of that pass.

    2. Previous event is another on-ball event:
       the next technical event must belong to the same player.

    The inferred carry belongs to the player performing the NEXT event, since
    that player moved the ball from the previous endpoint to their new action
    location. These rows are reconstructed from event continuity rather than
    observed carry events, so ``carry_is_inferred`` and ``carry_source_type``
    are retained as explicit provenance fields.
    """

    if df_processed is None or df_processed.empty:
        return pd.DataFrame(columns=CARRY_RESULT_COLUMNS)

    events = df_processed.copy()
    events['_carry_source_index'] = events.index

    sort_columns = [
        column for column in
        ('periodId', 'timeMin', 'timeSec', 'timeStamp', 'id', 'eventId')
        if column in events.columns
    ]

    if sort_columns:
        events = events.sort_values(sort_columns, kind='mergesort')

    events = events.reset_index(drop=True)

    # Reuse the reliable receiver inference already implemented.
    receiver_info = infer_pass_receivers(
        df_processed,
        max_time_gap_seconds=max_time_gap_seconds,
    )

    carry_rows = []

    for position in range(len(events) - 1):
        source = events.iloc[position]

        start_x = _numeric(source.get('end_x'))
        start_y = _numeric(source.get('end_y'))

        if pd.isna(start_x) or pd.isna(start_y):
            continue

        source_seconds = _event_seconds(source)
        source_period = source.get('periodId')

        candidate = None
        candidate_gap_seconds = np.nan

        # Find the next credible technical event.
        for candidate_position in range(position + 1, len(events)):
            next_event = events.iloc[candidate_position]

            if pd.notna(source_period) and pd.notna(next_event.get('periodId')):
                if str(next_event.get('periodId')) != str(source_period):
                    break

            next_seconds = _event_seconds(next_event)

            gap_seconds = (
                next_seconds - source_seconds
                if pd.notna(source_seconds) and pd.notna(next_seconds)
                else np.nan
            )

            if pd.notna(gap_seconds) and gap_seconds < 0:
                continue

            if (
                pd.notna(gap_seconds)
                and gap_seconds > max_time_gap_seconds
            ):
                break

            if _is_administrative_event(next_event):
                continue

            if (
                pd.isna(next_event.get('playerName'))
                and pd.isna(next_event.get('playerId'))
            ):
                continue

            end_x = _numeric(next_event.get('x'))
            end_y = _numeric(next_event.get('y'))

            if pd.isna(end_x) or pd.isna(end_y):
                continue

            candidate = next_event
            candidate_gap_seconds = gap_seconds
            break

        if candidate is None:
            continue

        # Possession must remain with the same team.
        if not _same_team(source, candidate):
            continue

        source_is_pass = (
            _numeric(source.get('typeId')) == 1
            or str(source.get('type_name') or '').strip().lower() == 'pass'
        )

        if source_is_pass:
            # An unsuccessful pass cannot start a reliable carry.
            if not _is_successful_pass(source):
                continue

            source_index = source['_carry_source_index']

            if source_index not in receiver_info.index:
                continue

            receiver = receiver_info.loc[source_index]

            if not bool(receiver.get('receiver_is_reliable', False)):
                continue

            receiver_id = receiver.get('receiver_player_id')
            candidate_id = candidate.get('playerId')

            receiver_name = receiver.get('receiver')
            candidate_name = candidate.get('playerName')

            receiver_matches_candidate = (
                pd.notna(receiver_id)
                and pd.notna(candidate_id)
                and str(receiver_id) == str(candidate_id)
            ) or (
                pd.isna(receiver_id)
                and pd.isna(candidate_id)
                and pd.notna(receiver_name)
                and pd.notna(candidate_name)
                and str(receiver_name) == str(candidate_name)
            )

            if not receiver_matches_candidate:
                continue

            confidence = receiver.get('receiver_confidence', 'medium')
            carry_source_type = 'pass_reception_gap'

        else:
            # For other event types, require continuity of the same player.
            if not _same_player(source, candidate):
                continue

            confidence = 'high'
            carry_source_type = 'same_player_event_gap'

        end_x = _numeric(candidate.get('x'))
        end_y = _numeric(candidate.get('y'))

        delta_x_m = (end_x - start_x) * 1.05
        delta_y_m = (end_y - start_y) * 0.68

        carry_distance_m = math.hypot(delta_x_m, delta_y_m)

        if carry_distance_m < min_distance_m:
            continue

        if carry_distance_m > max_distance_m:
            continue

        carry_rows.append({
            'playerName': candidate.get('playerName'),
            'playerId': candidate.get('playerId'),
            'team_name': candidate.get('team_name'),
            'contestantId': candidate.get('contestantId'),

            'x': start_x,
            'y': start_y,
            'end_x': end_x,
            'end_y': end_y,

            'carry_distance_m': carry_distance_m,
            'carry_time_gap_seconds': candidate_gap_seconds,

            'carry_source_event_id': (
                source.get('id') or source.get('eventId')
            ),
            'carry_end_event_id': (
                candidate.get('id') or candidate.get('eventId')
            ),

            'carry_source_type': carry_source_type,
            'carry_confidence': confidence,
            'carry_is_inferred': True,
            'carry_is_reliable': True,
        })

    if not carry_rows:
        return pd.DataFrame(columns=CARRY_RESULT_COLUMNS)

    return pd.DataFrame(carry_rows, columns=CARRY_RESULT_COLUMNS)


def receiver_coverage_summary(passes_df):
    """Return transparent receiver-attribution coverage for successful passes."""
    if passes_df is None or passes_df.empty:
        return {'eligible': 0, 'resolved': 0, 'high': 0, 'medium': 0, 'coverage_pct': 0.0}

    if 'outcome' in passes_df.columns:
        outcome_text = passes_df['outcome'].fillna('').astype(str).str.lower()
        outcome_numeric = pd.to_numeric(passes_df['outcome'], errors='coerce')
        successful = outcome_text.eq('successful') | outcome_numeric.eq(1)
    else:
        successful = pd.Series(True, index=passes_df.index)
    eligible = passes_df.loc[successful]
    resolved_mask = eligible.get(
        'receiver_is_reliable',
        eligible.get('receiver', pd.Series(index=eligible.index, dtype='object')),
    ).fillna(False).astype(bool)
    confidence = eligible.get('receiver_confidence', pd.Series(index=eligible.index, dtype='object'))
    resolved = int(resolved_mask.sum())
    total = int(len(eligible))
    return {
        'eligible': total,
        'resolved': resolved,
        'high': int(confidence.eq('high').sum()),
        'medium': int(confidence.eq('medium').sum()),
        'coverage_pct': (resolved / total * 100) if total else 0.0,
    }

@cache_derived_result("passes")
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

    # 1. Progressive passes. Keep both attempts and completions so that the UI
    # can distinguish volume from efficiency. The classifier also exposes the
    # canonical open-play flag used by other individual passing metrics.
    classified_passes = pass_metrics.classify_progressive_passes(passes)
    for column in pass_metrics.PROGRESSIVE_RESULT_COLUMNS:
        passes[column] = classified_passes[column]

    outcome = passes.get(
        'outcome',
        pd.Series('', index=passes.index),
    )
    outcome_text = (
        outcome.fillna('').astype(str).str.strip().str.lower()
    )
    outcome_numeric = pd.to_numeric(
        outcome,
        errors='coerce',
    )
    successful_pass = (
        outcome_text.eq('successful')
        | outcome_numeric.eq(1)
    )

    # 2. Completed open-play passes ending inside the penalty area.
    # Reuse the same open-play contract as Progressive Passes so corners,
    # free kicks, throw-ins, goal kicks and crosses do not leak into the
    # individual "Passes into Box" metric.
    passes['is_into_box'] = (
        successful_pass
        & classified_passes[
            'progressive_is_open_play'
        ].fillna(False).astype(bool)
        & pd.to_numeric(
            passes['end_x'],
            errors='coerce',
        ).ge(83.5)
        & pd.to_numeric(
            passes['end_y'],
            errors='coerce',
        ).between(
            21.1,
            78.9,
            inclusive='both',
        )
    )

    # 3. Receiver information. Never use a blind shift: the next raw row can
    # belong to the opponent or be an administrative event.
    receiver_info = infer_pass_receivers(df_processed)
    overlapping_receiver_columns = [
        column for column in RECEIVER_RESULT_COLUMNS if column in passes.columns
    ]
    if overlapping_receiver_columns:
        passes = passes.drop(columns=overlapping_receiver_columns)
    passes = passes.join(receiver_info, how='left')
    passes['receiver_is_reliable'] = passes['receiver_is_reliable'].fillna(False).astype(bool)

    coverage = receiver_coverage_summary(passes)
    print(
        "  get_passes_df: Reliable receiver inferred for "
        f"{coverage['resolved']}/{coverage['eligible']} successful passes "
        f"({coverage['coverage_pct']:.1f}%; "
        f"high={coverage['high']}, medium={coverage['medium']})."
    )

    # 4. Ensure Key Pass / Assist flags exist and represent completed passes.
    for flag_col in ['is_key_pass', 'is_assist']:
        if flag_col in passes.columns:
            passes[flag_col] = (
                passes[flag_col]
                .fillna(False)
                .astype(bool)
                & successful_pass
            )
        else:
            print(
                f"  get_passes_df: Flag column "
                f"'{flag_col}' NOT found. Creating as all False."
            )
            passes[flag_col] = False

    # --- Define final columns to select ---
    columns_to_select = [
        "id", "eventId", 
        "timeMin", "timeSec", # Assicurati che siano incluse
        "x", "y", "end_x", "end_y", "team_name",
        "playerName", "shorter_name", "Mapped Jersey Number",
        "receiver", "receiver_jersey_number", "type_name", "outcome",
        "receiver_player_id", "receiver_event_id", "receiver_confidence",
        "receiver_is_reliable", "receiver_reason", "receiver_time_gap_seconds",
        "receiver_spatial_gap_m", "is_key_pass", "is_assist",
        "is_progressive_attempt", "is_progressive",
        "progressive_distance_m", "progressive_threshold_m",
        "progressive_phase", "progressive_channel",
        "progressive_is_open_play", "progressive_exclusion_reason",
        "is_into_box"
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
