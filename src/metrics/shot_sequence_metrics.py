# src/metrics/shot_sequence_metrics.py
"""Canonical player shot-sequence metrics.

REL-08 replaces adjacency-based ``shift(-1)`` logic with an explicit
possession-chain reconstruction.
"""

import pandas as pd

from ..data_processing.pass_processing import infer_pass_receivers


DEFAULT_SHOT_TYPES = ('Goal', 'Miss', 'Attempt Saved', 'Post')

# Own-team actions that can explicitly mark the start of a new possession.
POSSESSION_START_TYPES = {
    'Ball recovery',
    'Interception',
    'Keeper pick-up',
    'Claim',
}

# Opponent actions that are strong evidence of controlled possession.
# Deflections / blocks / clearances are intentionally not included here.
OPPONENT_CONTROL_TYPES = {
    'Pass',
    'Take On',
    'Ball recovery',
    'Interception',
    'Tackle',
    'Keeper pick-up',
    'Claim',
    'Save',
    'Aerial',
    'Ball touch',
}

OWN_POSSESSION_END_TYPES = {
    'Dispossessed',
    'Offside Pass',
    'Out',
}


def _flag(value):
    """Interpret common Opta flag encodings."""
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes'}
    return bool(value)


def _is_successful(row):
    """Accept both text and numeric Opta outcome encodings."""
    outcome = row.get('outcome')
    if isinstance(outcome, str):
        return outcome.strip().lower() == 'successful'

    numeric = pd.to_numeric(
        pd.Series([outcome]),
        errors='coerce',
    ).iloc[0]
    return pd.notna(numeric) and float(numeric) == 1.0


def _event_key(row, fallback):
    """Stable event identity, preferring Opta global ``id``."""
    value = row.get('id')
    if pd.notna(value):
        return ('id', value)

    value = row.get('eventId')
    if pd.notna(value):
        return ('eventId', value)

    return ('source_order', fallback)


def _same_period(left, right):
    """Missing period data is tolerated; present period data is strict."""
    left_period = left.get('periodId')
    right_period = right.get('periodId')

    if pd.isna(left_period) or pd.isna(right_period):
        return True

    return left_period == right_period


def _is_restart_pass(row):
    """Return True when a pass explicitly starts from a restart."""
    if row.get('type_name') != 'Pass':
        return False

    aliases = (
        'Corner taken',
        'Free kick taken',
        'Freekick taken',
        'ThrowIn',
        'Throw-in set piece',
        'Goal kick',
        'Goal Kick',
    )

    return any(
        _flag(row.get(column))
        for column in aliases
        if column in row.index
    )


def _is_direct_restart_shot(row):
    """Penalty and direct-free-kick shots do not inherit stale open-play events."""
    if row.get('type_name') not in DEFAULT_SHOT_TYPES:
        return False

    for column in ('Penalty', 'Free kick'):
        if column in row.index and _flag(row.get(column)):
            return True

    return False


def _is_shot(row, shot_types):
    """Own goals are not credited as shooter sequences for the scoring side."""
    return (
        row.get('type_name') in shot_types
        and not _flag(row.get('Own goal'))
    )


def _is_completed_pass(row):
    return (
        row.get('type_name') == 'Pass'
        and _is_successful(row)
    )


def _is_shot_assist_pass(row):
    """
    Canonical shot assist: one completed pass explicitly linked to the shot.

    Key-pass and assist flags may overlap on the same Opta event; the event
    receives one ``shot_assist`` role only.
    """
    if not _is_completed_pass(row):
        return False

    return (
        _flag(row.get('is_key_pass'))
        or _flag(row.get('is_assist'))
    )


def _own_possession_starts_here(row):
    """Explicit recovery/restart boundary for the attacking team."""
    if _is_restart_pass(row):
        return True

    event_type = row.get('type_name')

    if (
        event_type in POSSESSION_START_TYPES
        and _is_successful(row)
    ):
        return True

    return (
        event_type == 'Tackle'
        and _is_successful(row)
    )


def _own_possession_ended_here(row):
    """Events before this point cannot belong to the shot possession."""
    event_type = row.get('type_name')

    if (
        event_type == 'Pass'
        and not _is_successful(row)
    ):
        return True

    if (
        event_type in {
            'Take On',
            'Ball touch',
            'Aerial',
            'Challenge',
        }
        and not _is_successful(row)
    ):
        return True

    return event_type in OWN_POSSESSION_END_TYPES


def _opponent_controls(row):
    """Opponent contact only breaks the chain when it signals control."""
    event_type = row.get('type_name')

    if event_type in DEFAULT_SHOT_TYPES:
        return True

    return (
        event_type in OPPONENT_CONTROL_TYPES
        and _is_successful(row)
    )


def _prepare_events(df_processed):
    """
    Establish deterministic chronology.

    ``eventId`` is not assumed to be globally chronological. Match time and
    period are authoritative, with source order as the stable tie-breaker.
    """
    df = df_processed.copy()
    # Keep the original DataFrame index so receiver inference can be joined
    # back to the chronologically sorted shot-sequence rows.
    df['_source_index'] = df.index
    df['_source_order'] = range(len(df))

    for column in ('periodId', 'timeMin', 'timeSec'):
        if column not in df.columns:
            df[column] = pd.NA

        df[f'_sort_{column}'] = pd.to_numeric(
            df[column],
            errors='coerce',
        )

    df = df.sort_values(
        [
            '_sort_periodId',
            '_sort_timeMin',
            '_sort_timeSec',
            '_source_order',
        ],
        kind='stable',
        na_position='last',
    ).reset_index(drop=True)

    df['_event_key'] = [
        _event_key(
            row,
            int(row['_source_order']),
        )
        for _, row in df.iterrows()
    ]

    return df



def _pass_receiver_matches_player(pass_row, player_row, receiver_info):
    """
    Return True only when a completed pass has a reliably inferred receiver
    matching ``player_row``.

    REL-08 pre-assists are therefore not "the previous completed pass". They
    are the pass RECEIVED by the player who subsequently makes the
    shot-creating pass.
    """
    source_index = pass_row.get('_source_index')

    if source_index not in receiver_info.index:
        return False

    receiver = receiver_info.loc[source_index]
    if isinstance(receiver, pd.DataFrame):
        # Defensive guard for non-unique input indexes.
        receiver = receiver.iloc[0]

    if not bool(receiver.get('receiver_is_reliable', False)):
        return False

    receiver_id = receiver.get('receiver_player_id')
    player_id = player_row.get('playerId')

    if pd.notna(receiver_id) and pd.notna(player_id):
        return str(receiver_id) == str(player_id)

    receiver_name = receiver.get('receiver')
    player_name = player_row.get('playerName')

    return (
        pd.notna(receiver_name)
        and pd.notna(player_name)
        and str(receiver_name) == str(player_name)
    )


def build_shot_sequences(
    df_processed,
    shot_types=None,
):
    """
    Build canonical possession chains ending in a shot.

    Contract:
    - one attacking team per sequence;
    - one period per sequence;
    - opponent events never enter the returned sequence;
    - the backward trace stops at opponent control, an own possession-ending
      event, a previous shot, an explicit possession start/restart, or the
      period boundary;
    - every sequence terminates with exactly one shot;
    - shooter, shot assist and pre-assist are event-exclusive roles.

    An opponent unsuccessful challenge may appear in the raw stream without
    ending possession. It is skipped rather than inserted into the sequence.
    """
    if df_processed is None or df_processed.empty:
        return pd.DataFrame()

    required = {
        'team_name',
        'type_name',
        'playerName',
    }
    if not required.issubset(df_processed.columns):
        return pd.DataFrame()

    shot_types = tuple(
        shot_types or DEFAULT_SHOT_TYPES
    )

    # Pre-assists need a real passer -> receiver relationship. Reuse the
    # conservative receiver inference already used by the passing module
    # instead of assuming the previous pass belongs to the shot creator.
    receiver_info = infer_pass_receivers(df_processed)

    df = _prepare_events(df_processed)

    sequence_frames = []
    sequence_number = 0

    for shot_idx in range(len(df)):
        shot = df.iloc[shot_idx]

        if not _is_shot(shot, shot_types):
            continue

        shot_team = shot.get('team_name')
        if pd.isna(shot_team):
            continue

        sequence_number += 1
        selected_indices = [shot_idx]
        boundary_reason = 'data_start'

        # Penalties/direct free kicks are self-contained shot possessions.
        if _is_direct_restart_shot(shot):
            boundary_reason = 'direct_restart_shot'

        else:
            current_idx = shot_idx - 1

            while current_idx >= 0:
                event = df.iloc[current_idx]

                if not _same_period(event, shot):
                    boundary_reason = 'period_boundary'
                    break

                # A previous shot closes the previous shot possession. This
                # prevents one assist/pre-assist event from being reused for a
                # rebound or second shot.
                if event.get('type_name') in shot_types:
                    boundary_reason = 'previous_shot'
                    break

                if event.get('team_name') != shot_team:
                    if _opponent_controls(event):
                        boundary_reason = 'opponent_control'
                        break

                    # Failed opponent duel / deflection: possession can remain
                    # with the shooting team, but the opponent row itself is
                    # never inserted into the attacking sequence.
                    current_idx -= 1
                    continue

                if _own_possession_ended_here(event):
                    boundary_reason = 'own_possession_loss'
                    break

                selected_indices.append(current_idx)

                if _own_possession_starts_here(event):
                    boundary_reason = 'possession_start'
                    break

                current_idx -= 1

        selected_indices = list(
            reversed(selected_indices)
        )
        sequence = df.iloc[
            selected_indices
        ].copy()

        # Hard guards for the public contract.
        sequence = sequence[
            sequence['team_name'].eq(shot_team)
        ].copy()

        if (
            'periodId' in sequence.columns
            and pd.notna(shot.get('periodId'))
        ):
            sequence = sequence[
                sequence['periodId'].eq(
                    shot.get('periodId')
                )
            ].copy()

        sequence['shot_sequence_id'] = (
            sequence_number
        )
        sequence['shot_event_key'] = [
            shot['_event_key']
        ] * len(sequence)
        sequence[
            'sequence_boundary_reason'
        ] = boundary_reason
        sequence['sequence_role'] = pd.NA

        shot_key = shot['_event_key']
        shot_key_mask = sequence[
            '_event_key'
        ].map(
            lambda value: value == shot_key
        )
        sequence.loc[
            shot_key_mask,
            'sequence_role',
        ] = 'shooter'

        # Shot assist = latest completed pass explicitly flagged as a key
        # pass or assist inside this possession.
        before_shot = sequence.iloc[:-1]
        assist_candidates = before_shot[
            before_shot.apply(
                _is_shot_assist_pass,
                axis=1,
            )
        ]

        if not assist_candidates.empty:
            assist_idx = (
                assist_candidates.index[-1]
            )
            sequence.loc[
                assist_idx,
                'sequence_role',
            ] = 'shot_assist'

            # Pre-assist = latest completed pass RECEIVED by the player who
            # makes the shot-creating pass. This is deliberately stricter than
            # selecting the previous completed pass in the raw sequence.
            assist_position = (
                sequence.index.get_loc(
                    assist_idx
                )
            )
            assist_player = sequence.loc[assist_idx]
            before_assist = sequence.iloc[
                :assist_position
            ]

            pre_candidates = before_assist[
                before_assist.apply(
                    _is_completed_pass,
                    axis=1,
                )
            ]

            if not pre_candidates.empty:
                receiver_match_mask = pre_candidates.apply(
                    lambda pass_row: _pass_receiver_matches_player(
                        pass_row,
                        assist_player,
                        receiver_info,
                    ),
                    axis=1,
                )
                pre_candidates = pre_candidates[
                    receiver_match_mask
                ]

            if not pre_candidates.empty:
                pre_idx = (
                    pre_candidates.index[-1]
                )

                # Defensive guard: one event can never occupy two roles.
                if pre_idx != assist_idx:
                    sequence.loc[
                        pre_idx,
                        'sequence_role',
                    ] = 'pre_assist'

        sequence_frames.append(sequence)

    if not sequence_frames:
        return pd.DataFrame()

    result = pd.concat(
        sequence_frames,
        ignore_index=True,
    )

    keep_internal = {'_event_key'}
    drop_columns = [
        column
        for column in result.columns
        if (
            column.startswith('_sort_')
            or (
                column.startswith('_')
                and column not in keep_internal
            )
        )
    ]

    return result.drop(
        columns=drop_columns,
        errors='ignore',
    )


def calculate_shot_sequence_player_stats(
    df_processed,
    shot_types=None,
):
    """
    Count canonical shot-sequence roles by player.

    A player may legitimately occupy two different roles through two different
    events (for example, pre-assist and shooter in a give-and-go), but the same
    event is never counted twice.
    """
    sequences = build_shot_sequences(
        df_processed,
        shot_types=shot_types,
    )

    output_columns = [
        'Shot Sequence Shots',
        'Shot Sequence Assists',
        'Shot Sequence Pre-Assists',
        'Shot Sequence Involvements',
    ]

    if sequences.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    role_to_column = {
        'shooter': 'Shot Sequence Shots',
        'shot_assist': (
            'Shot Sequence Assists'
        ),
        'pre_assist': (
            'Shot Sequence Pre-Assists'
        ),
    }

    role_rows = sequences[
        sequences['sequence_role'].isin(
            role_to_column
        )
        & sequences['playerName'].notna()
    ].copy()

    if role_rows.empty:
        return pd.DataFrame(
            columns=output_columns
        )

    role_rows['metric'] = (
        role_rows['sequence_role']
        .map(role_to_column)
    )

    role_rows = role_rows.drop_duplicates(
        subset=[
            'playerName',
            '_event_key',
            'metric',
        ]
    )

    stats = (
        role_rows
        .groupby(
            ['playerName', 'metric']
        )
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=output_columns[:-1],
            fill_value=0,
        )
        .astype(int)
    )

    stats[
        'Shot Sequence Involvements'
    ] = stats[
        output_columns[:-1]
    ].sum(axis=1)

    return stats[output_columns]
