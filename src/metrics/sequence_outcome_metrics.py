import numpy as np
import pandas as pd

from src.utils.sequence_outcomes import apply_sequence_outcome_contract


MIDDLE_THIRD_X = 33.33
OPPOSITION_HALF_X = 50.0
FINAL_THIRD_X = 66.67

PENALTY_AREA_X = 83.0
PENALTY_AREA_Y_MIN = 21.1
PENALTY_AREA_Y_MAX = 78.9

SHOT_TYPES = {
    'Goal',
    'Miss',
    'Attempt Saved',
    'Post',
}


def _point_is_in_box(x, y):
    if pd.isna(x) or pd.isna(y):
        return False

    return (
        x >= PENALTY_AREA_X
        and PENALTY_AREA_Y_MIN
        <= y
        <= PENALTY_AREA_Y_MAX
    )


def calculate_sequence_milestones(
    sequence_df,
):
    """
    Calculate independent territorial and attacking milestones
    for a single possession sequence.

    Territorial milestones are based on controlled ball locations:
    - every event start location counts as controlled;
    - an event end location counts only when that event is successful.

    This prevents an unsuccessful forward pass from being treated as
    a successfully reached zone.
    """

    default_result = {
        'max_controlled_x': np.nan,
        'reached_middle_third': False,
        'reached_opposition_half': False,
        'reached_final_third': False,
        'entered_penalty_area': False,
        'produced_shot': False,
        'produced_goal': False,
        'milestone_final_third': False,
        'milestone_box': False,
        'milestone_shot': False,
        'milestone_goal': False,
    }

    if (
        sequence_df is None
        or sequence_df.empty
    ):
        return default_result

    df = sequence_df.copy()

    # ---------------------------------------------------------
    # NUMERIC COORDINATES
    # ---------------------------------------------------------
    for col in (
        'x',
        'y',
        'end_x',
        'end_y',
    ):
        if col not in df.columns:
            df[col] = np.nan

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce',
        )

    if 'outcome' not in df.columns:
        df['outcome'] = None

    if 'type_name' not in df.columns:
        df['type_name'] = None

    # ---------------------------------------------------------
    # CONTROLLED LOCATIONS
    # ---------------------------------------------------------
    controlled_x = []

    # Event starting point:
    # if an action starts here, the team has reached this location.
    controlled_x.extend(
        df['x']
        .dropna()
        .tolist()
    )

    # Successful end locations also represent retained control.
    successful_events = df[
        df['outcome'].eq('Successful')
    ]

    controlled_x.extend(
        successful_events['end_x']
        .dropna()
        .tolist()
    )

    if controlled_x:
        max_controlled_x = float(
            max(controlled_x)
        )
    else:
        max_controlled_x = np.nan

    # ---------------------------------------------------------
    # PENALTY AREA
    # ---------------------------------------------------------
    start_in_box = df.apply(
        lambda row: _point_is_in_box(
            row['x'],
            row['y'],
        ),
        axis=1,
    )

    successful_end_in_box = (
        successful_events.apply(
            lambda row: _point_is_in_box(
                row['end_x'],
                row['end_y'],
            ),
            axis=1,
        )
        if not successful_events.empty
        else pd.Series(dtype=bool)
    )

    entered_penalty_area = bool(
        start_in_box.any()
        or successful_end_in_box.any()
    )

    # ---------------------------------------------------------
    # SHOTS / GOALS
    # ---------------------------------------------------------
    shot_mask = df[
        'type_name'
    ].isin(SHOT_TYPES)

    produced_shot = bool(
        shot_mask.any()
    )

    produced_goal = bool(
        df['type_name']
        .eq('Goal')
        .any()
    )

    # Canonical terminal outcome is authoritative for special cases
    # such as forced own goals that may not be represented by a shot row
    # belonging to the attacking team.
    if 'terminal_outcome' in df.columns:
        terminal_values = (
            df['terminal_outcome']
            .dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if terminal_values.eq('goal').any():
            produced_goal = True
            produced_shot = True
    elif 'sequence_outcome_type' in df.columns:
        # Backward compatibility for historical flattened sequence frames.
        outcome_text = ' '.join(
            df['sequence_outcome_type']
            .dropna()
            .astype(str)
            .unique()
        ).lower()

        if 'goal' in outcome_text:
            produced_goal = True
            produced_shot = True

    # ---------------------------------------------------------
    # RESULT
    # ---------------------------------------------------------
    return {
        'max_controlled_x': max_controlled_x,

        'reached_middle_third': (
            pd.notna(max_controlled_x)
            and max_controlled_x
            >= MIDDLE_THIRD_X
        ),

        'reached_opposition_half': (
            pd.notna(max_controlled_x)
            and max_controlled_x
            >= OPPOSITION_HALF_X
        ),

        'reached_final_third': (
            pd.notna(max_controlled_x)
            and max_controlled_x
            >= FINAL_THIRD_X
        ),

        'entered_penalty_area':
            entered_penalty_area,

        'produced_shot':
            produced_shot,

        'produced_goal':
            produced_goal,

        # Canonical milestone aliases used by the sequence contract.
        'milestone_final_third': (
            pd.notna(max_controlled_x)
            and max_controlled_x >= FINAL_THIRD_X
        ),
        'milestone_box': entered_penalty_area,
        'milestone_shot': produced_shot,
        'milestone_goal': produced_goal,
    }

def _first_non_null(series, default=None):
    if series is None:
        return default

    values = series.dropna()

    if values.empty:
        return default

    return values.iloc[0]


def _calculate_duration_seconds(
    sequence_df,
    start_minute,
    start_second,
):
    if (
        pd.isna(start_minute)
        or pd.isna(start_second)
        or 'timeMin' not in sequence_df.columns
        or 'timeSec' not in sequence_df.columns
    ):
        return np.nan

    valid_times = sequence_df[
        ['timeMin', 'timeSec']
    ].copy()

    valid_times['timeMin'] = pd.to_numeric(
        valid_times['timeMin'],
        errors='coerce',
    )

    valid_times['timeSec'] = pd.to_numeric(
        valid_times['timeSec'],
        errors='coerce',
    )

    valid_times = valid_times.dropna()

    if valid_times.empty:
        return np.nan

    last_event = valid_times.iloc[-1]

    start_total = (
        float(start_minute) * 60
        + float(start_second)
    )

    end_total = (
        float(last_event['timeMin']) * 60
        + float(last_event['timeSec'])
    )

    return max(
        0.0,
        end_total - start_total,
    )


def summarize_sequences(
    df_sequences,
    sequence_kind,
):
    """
    Convert a flattened sequence-event DataFrame into one row
    per buildup / transition sequence.

    Parameters
    ----------
    df_sequences : pd.DataFrame
        Flattened output returned by the existing sequence detectors.

    sequence_kind : str
        One of:
        - 'buildup'
        - 'offensive_transition'
        - 'defensive_transition'

    Returns
    -------
    pd.DataFrame
        One row per sequence, including final outcome, duration,
        pass count and independent territorial milestones.
    """

    output_columns = [
        'sequence_id',
        'sequence_kind',
        'team_name',
        'start_zone',
        'initial_action_type',
        'terminal_outcome',
        'termination_reason',
        'viewpoint',
        'final_outcome',
        'pass_count',
        'event_count',
        'duration_seconds',
        'restart_delay_seconds',
        'trigger_minute',
        'trigger_second',
        'start_minute',
        'start_second',
        'end_minute',
        'end_second',
        'max_controlled_x',
        'reached_middle_third',
        'reached_opposition_half',
        'reached_final_third',
        'entered_penalty_area',
        'produced_shot',
        'produced_goal',
        'milestone_final_third',
        'milestone_box',
        'milestone_shot',
        'milestone_goal',
    ]

    if (
        df_sequences is None
        or df_sequences.empty
    ):
        return pd.DataFrame(
            columns=output_columns
        )

    if sequence_kind == 'buildup':
        sequence_id_col = 'trigger_sequence_id'
        sequence_viewpoint = 'attacking'
        zone_col = 'trigger_zone'
        pass_count_col = 'buildup_pass_count'
        initial_action_col = (
            'type_of_initial_trigger'
        )
        start_minute_col = (
            'timeMin_at_trigger'
        )
        start_second_col = (
            'timeSec_at_trigger'
        )

    elif sequence_kind in (
        'offensive_transition',
        'defensive_transition',
    ):
        sequence_id_col = 'loss_sequence_id'
        sequence_viewpoint = (
            'defending'
            if sequence_kind == 'defensive_transition'
            else 'attacking'
        )
        zone_col = 'loss_zone'
        pass_count_col = 'opponent_pass_count'
        initial_action_col = (
            'type_of_initial_loss'
        )
        start_minute_col = (
            'timeMin_at_loss'
        )
        start_second_col = (
            'timeSec_at_loss'
        )

    else:
        raise ValueError(
            'sequence_kind must be one of: '
            "'buildup', "
            "'offensive_transition', "
            "'defensive_transition'"
        )

    if sequence_id_col not in df_sequences.columns:
        raise ValueError(
            f"Missing required sequence id column: "
            f"{sequence_id_col}"
        )

    summaries = []

    grouped = df_sequences.groupby(
        sequence_id_col,
        sort=False,
    )

    for sequence_id, sequence_df in grouped:
        sequence_df = apply_sequence_outcome_contract(
            sequence_df,
            viewpoint=sequence_viewpoint,
        )

        first_event = sequence_df.iloc[0]
        last_event = sequence_df.iloc[-1]

        milestones = (
            calculate_sequence_milestones(
                sequence_df,
            )
        )

        # -----------------------------------------------------
        # START CONTEXT
        # -----------------------------------------------------

        start_zone = first_event.get(
            zone_col,
            'Unknown',
        )

        initial_action_type = first_event.get(
            initial_action_col,
            'Unknown',
        )

        trigger_minute = first_event.get(
            start_minute_col,
            np.nan,
        )

        trigger_second = first_event.get(
            start_second_col,
            np.nan,
        )

        first_action_minute = first_event.get(
            'timeMin',
            np.nan,
        )

        first_action_second = first_event.get(
            'timeSec',
            np.nan,
        )

        # -----------------------------------------------------
        # PHASE START
        # -----------------------------------------------------
        #
        # Buildup:
        # trigger can be a foul, out, offside, keeper action, etc.
        # Dead/restart time must not count as active buildup duration.
        #
        # Transition:
        # the transition starts at the possession-loss event itself.

        if sequence_kind == 'buildup':
            start_minute = first_action_minute
            start_second = first_action_second

            if (
                pd.notna(trigger_minute)
                and pd.notna(trigger_second)
                and pd.notna(first_action_minute)
                and pd.notna(first_action_second)
            ):
                trigger_total_seconds = (
                    float(trigger_minute) * 60
                    + float(trigger_second)
                )

                first_action_total_seconds = (
                    float(first_action_minute) * 60
                    + float(first_action_second)
                )

                restart_delay_seconds = max(
                    0.0,
                    first_action_total_seconds
                    - trigger_total_seconds,
                )
            else:
                restart_delay_seconds = np.nan

        else:
            start_minute = trigger_minute
            start_second = trigger_second
            restart_delay_seconds = np.nan

            # Fallback if loss metadata is unavailable.
            if pd.isna(start_minute):
                start_minute = first_action_minute

            if pd.isna(start_second):
                start_second = first_action_second

        # -----------------------------------------------------
        # END CONTEXT
        # -----------------------------------------------------

        end_minute = last_event.get(
            'timeMin',
            np.nan,
        )

        end_second = last_event.get(
            'timeSec',
            np.nan,
        )

        duration_seconds = (
            _calculate_duration_seconds(
                sequence_df,
                start_minute,
                start_second,
            )
        )

        # -----------------------------------------------------
        # OUTCOME
        # -----------------------------------------------------

        terminal_outcome = last_event.get(
            'terminal_outcome',
            'unknown',
        )
        termination_reason = last_event.get(
            'termination_reason',
            'unknown',
        )
        viewpoint = last_event.get(
            'viewpoint',
            sequence_viewpoint,
        )

        # Compatibility/UI label. Canonical logic must use the three
        # fields above rather than branching on this string.
        final_outcome = last_event.get(
            'sequence_outcome_type',
            'Unknown',
        )

        if pd.isna(final_outcome):
            final_outcome = 'Unknown'

        # -----------------------------------------------------
        # PASS COUNT
        # -----------------------------------------------------

        if pass_count_col in sequence_df.columns:
            pass_count = _first_non_null(
                sequence_df[
                    pass_count_col
                ],
                default=np.nan,
            )
        else:
            pass_count = np.nan

        if pd.isna(pass_count):
            if (
                'type_name'
                in sequence_df.columns
                and 'outcome'
                in sequence_df.columns
            ):
                pass_count = int(
                    (
                        sequence_df[
                            'type_name'
                        ].eq('Pass')
                        &
                        sequence_df[
                            'outcome'
                        ].eq('Successful')
                    ).sum()
                )
            else:
                pass_count = 0

        # -----------------------------------------------------
        # TEAM
        # -----------------------------------------------------

        if 'team_name' in sequence_df.columns:
            team_name = _first_non_null(
                sequence_df['team_name'],
                default='Unknown',
            )
        else:
            team_name = 'Unknown'

        summaries.append({
            'sequence_id':
                sequence_id,

            'sequence_kind':
                sequence_kind,

            'team_name':
                team_name,

            'start_zone':
                start_zone,

            'initial_action_type':
                initial_action_type,

            'terminal_outcome':
                terminal_outcome,

            'termination_reason':
                termination_reason,

            'viewpoint':
                viewpoint,

            'final_outcome':
                final_outcome,

            'pass_count':
                int(pass_count),

            'event_count':
                int(len(sequence_df)),

            'duration_seconds':
                duration_seconds,

            'restart_delay_seconds':
                restart_delay_seconds,

            'trigger_minute':
                trigger_minute,

            'trigger_second':
                trigger_second,

            'start_minute':
                start_minute,

            'start_second':
                start_second,

            'end_minute':
                end_minute,

            'end_second':
                end_second,

            **milestones,
        })

    return pd.DataFrame(
        summaries,
        columns=output_columns,
    )

def aggregate_sequence_outcomes(
    sequence_summary,
):
    """
    Aggregate a one-row-per-sequence summary into metrics
    suitable for comparison and visualization.

    Returns:
    - total sequence count;
    - milestone counts and percentages;
    - terminal outcome counts and percentages;
    - average/median duration;
    - average/median completed passes.
    """

    milestone_columns = [
        'reached_middle_third',
        'reached_opposition_half',
        'reached_final_third',
        'entered_penalty_area',
        'produced_shot',
        'produced_goal',
    ]

    if (
        sequence_summary is None
        or sequence_summary.empty
    ):
        return {
            'total_sequences': 0,
            'milestones': {
                milestone: {
                    'count': 0,
                    'percentage': 0.0,
                }
                for milestone
                in milestone_columns
            },
            'outcomes': {},
            'terminal_outcomes': {},
            'termination_reasons': {},
            'avg_duration_seconds': np.nan,
            'median_duration_seconds': np.nan,
            'avg_completed_passes': np.nan,
            'median_completed_passes': np.nan,
        }

    df = sequence_summary.copy()

    total_sequences = len(df)

    # ---------------------------------------------------------
    # MILESTONES
    # ---------------------------------------------------------

    milestones = {}

    for milestone in milestone_columns:
        if milestone not in df.columns:
            count = 0
        else:
            count = int(
                df[milestone]
                .fillna(False)
                .astype(bool)
                .sum()
            )

        percentage = (
            count
            / total_sequences
            * 100
        )

        milestones[milestone] = {
            'count': count,
            'percentage': percentage,
        }

    # ---------------------------------------------------------
    # TERMINAL OUTCOMES
    # ---------------------------------------------------------

    def _count_categories(column, fallback):
        if column not in df.columns:
            return {}

        counts = (
            df[column]
            .fillna(fallback)
            .astype(str)
            .value_counts()
        )

        return {
            str(value): {
                'count': int(count),
                'percentage': (
                    int(count)
                    / total_sequences
                    * 100
                ),
            }
            for value, count in counts.items()
        }

    # Compatibility/UI labels retained for existing cards and plots.
    outcomes = _count_categories(
        'final_outcome',
        'Unknown',
    )

    # Canonical domain aggregation. New logic should consume these keys.
    terminal_outcomes = _count_categories(
        'terminal_outcome',
        'unknown',
    )

    termination_reasons = _count_categories(
        'termination_reason',
        'unknown',
    )

    # ---------------------------------------------------------
    # DURATION
    # ---------------------------------------------------------

    if 'duration_seconds' in df.columns:
        durations = pd.to_numeric(
            df['duration_seconds'],
            errors='coerce',
        )
    else:
        durations = pd.Series(dtype=float)

    # ---------------------------------------------------------
    # COMPLETED PASSES
    # ---------------------------------------------------------

    if 'pass_count' in df.columns:
        passes = pd.to_numeric(
            df['pass_count'],
            errors='coerce',
        )
    else:
        passes = pd.Series(dtype=float)

    return {
        'total_sequences':
            total_sequences,

        'milestones':
            milestones,

        'outcomes':
            outcomes,

        'terminal_outcomes':
            terminal_outcomes,

        'termination_reasons':
            termination_reasons,

        'avg_duration_seconds': (
            float(durations.mean())
            if not durations.dropna().empty
            else np.nan
        ),

        'median_duration_seconds': (
            float(durations.median())
            if not durations.dropna().empty
            else np.nan
        ),

        'avg_completed_passes': (
            float(passes.mean())
            if not passes.dropna().empty
            else np.nan
        ),

        'median_completed_passes': (
            float(passes.median())
            if not passes.dropna().empty
            else np.nan
        ),
    }

MILESTONE_LABELS = {
    'reached_middle_third': 'Middle third',
    'reached_opposition_half': 'Opposition half',
    'reached_final_third': 'Final third',
    'entered_penalty_area': 'Penalty area',
    'produced_shot': 'Shot',
    'produced_goal': 'Goal',
}


def build_sequence_comparison(
    home_summary,
    away_summary,
    home_team,
    away_team,
    milestone_keys=None,
):
    """
    Build a Home vs Away comparison structure ready for UI use.

    The function does not calculate sequence logic itself.
    It only combines the already validated aggregates produced by
    aggregate_sequence_outcomes().
    """

    if milestone_keys is None:
        milestone_keys = [
            'reached_middle_third',
            'reached_opposition_half',
            'reached_final_third',
            'entered_penalty_area',
            'produced_shot',
            'produced_goal',
        ]

    home_agg = aggregate_sequence_outcomes(
        home_summary
    )

    away_agg = aggregate_sequence_outcomes(
        away_summary
    )

    # ---------------------------------------------------------
    # FUNNEL / MILESTONES
    # ---------------------------------------------------------

    funnel = [
        {
            'key': 'total_sequences',
            'label': 'Sequences',

            'home_count':
                home_agg['total_sequences'],

            'home_percentage': (
                100.0
                if home_agg['total_sequences'] > 0
                else 0.0
            ),

            'away_count':
                away_agg['total_sequences'],

            'away_percentage': (
                100.0
                if away_agg['total_sequences'] > 0
                else 0.0
            ),
        }
    ]

    for milestone in milestone_keys:
        home_data = (
            home_agg[
                'milestones'
            ].get(
                milestone,
                {
                    'count': 0,
                    'percentage': 0.0,
                },
            )
        )

        away_data = (
            away_agg[
                'milestones'
            ].get(
                milestone,
                {
                    'count': 0,
                    'percentage': 0.0,
                },
            )
        )

        funnel.append({
            'key':
                milestone,

            'label':
                MILESTONE_LABELS.get(
                    milestone,
                    milestone,
                ),

            'home_count':
                home_data['count'],

            'home_percentage':
                home_data['percentage'],

            'away_count':
                away_data['count'],

            'away_percentage':
                away_data['percentage'],
        })

    # ---------------------------------------------------------
    # TERMINAL OUTCOMES
    # ---------------------------------------------------------

    all_outcomes = set(
        home_agg['outcomes'].keys()
    ) | set(
        away_agg['outcomes'].keys()
    )

    # Most common combined outcomes first.
    sorted_outcomes = sorted(
        all_outcomes,
        key=lambda outcome: (
            -(
                home_agg[
                    'outcomes'
                ].get(
                    outcome,
                    {'count': 0},
                )['count']
                +
                away_agg[
                    'outcomes'
                ].get(
                    outcome,
                    {'count': 0},
                )['count']
            ),
            outcome,
        ),
    )

    outcome_rows = []

    for outcome in sorted_outcomes:
        home_data = (
            home_agg[
                'outcomes'
            ].get(
                outcome,
                {
                    'count': 0,
                    'percentage': 0.0,
                },
            )
        )

        away_data = (
            away_agg[
                'outcomes'
            ].get(
                outcome,
                {
                    'count': 0,
                    'percentage': 0.0,
                },
            )
        )

        outcome_rows.append({
            'outcome':
                outcome,

            'home_count':
                home_data['count'],

            'home_percentage':
                home_data['percentage'],

            'away_count':
                away_data['count'],

            'away_percentage':
                away_data['percentage'],
        })

    canonical_outcome_rows = []
    canonical_outcomes = set(
        home_agg['terminal_outcomes'].keys()
    ) | set(
        away_agg['terminal_outcomes'].keys()
    )

    for outcome in sorted(canonical_outcomes):
        home_data = home_agg['terminal_outcomes'].get(
            outcome,
            {'count': 0, 'percentage': 0.0},
        )
        away_data = away_agg['terminal_outcomes'].get(
            outcome,
            {'count': 0, 'percentage': 0.0},
        )

        canonical_outcome_rows.append({
            'terminal_outcome': outcome,
            'home_count': home_data['count'],
            'home_percentage': home_data['percentage'],
            'away_count': away_data['count'],
            'away_percentage': away_data['percentage'],
        })

    # ---------------------------------------------------------
    # SEQUENCE PROFILE
    # ---------------------------------------------------------

    profile = {
        'home': {
            'avg_duration_seconds':
                home_agg[
                    'avg_duration_seconds'
                ],

            'median_duration_seconds':
                home_agg[
                    'median_duration_seconds'
                ],

            'avg_completed_passes':
                home_agg[
                    'avg_completed_passes'
                ],

            'median_completed_passes':
                home_agg[
                    'median_completed_passes'
                ],
        },

        'away': {
            'avg_duration_seconds':
                away_agg[
                    'avg_duration_seconds'
                ],

            'median_duration_seconds':
                away_agg[
                    'median_duration_seconds'
                ],

            'avg_completed_passes':
                away_agg[
                    'avg_completed_passes'
                ],

            'median_completed_passes':
                away_agg[
                    'median_completed_passes'
                ],
        },
    }

    return {
        'home_team':
            home_team,

        'away_team':
            away_team,

        'home_total':
            home_agg['total_sequences'],

        'away_total':
            away_agg['total_sequences'],

        'funnel':
            funnel,

        'outcomes':
            outcome_rows,

        'terminal_outcomes':
            canonical_outcome_rows,

        'profile':
            profile,
    }