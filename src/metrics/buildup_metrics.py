# src/metrics/transition_metrics.py
import pandas as pd
from src.utils.derived_cache import cache_derived_result
from src.metrics.data_quality import attach_sequence_coverage
import numpy as np
from src import config
from src.metrics.transition_metrics import get_pitch_third
from src.utils.sequence_outcomes import apply_sequence_outcome_contract
from src.metrics.restart_metrics import classify_restart_event
# --- Set display options to show all columns and more rows ---
pd.set_option('display.max_columns', None) # Show all columns
pd.set_option('display.max_rows', None)    # Show all rows (be careful with very large DFs)
pd.set_option('display.width', None)       # Auto-detect width to avoid line wrapping if possible
pd.set_option('display.max_colwidth', None) # Show full content of each column

# --- Define Semicircle Parameters (should be globally consistent or passed) ---
# These constants define the "big chance" area and MUST MATCH those used in
# your `find_buildup_sequences` function and its `is_in_big_chance_area` helper.
# Assuming attacking L->R (goal at x=100) by default for these base definitions.
# Penalty box implied by original check: x >= 83, y between 21.1 and 78.9
BC_STD_BOX_EDGE_X_FURTHEST_FROM_GOAL = 83.0  # X-coord of penalty box edge (e.g., 18-yard line)
BC_STD_BOX_EDGE_X_GOAL_LINE = 100.0
BC_STD_BOX_EDGE_Y_MIN = 21.1
BC_STD_BOX_EDGE_Y_MAX = 78.9
BC_STD_BOX_CENTER_Y = (BC_STD_BOX_EDGE_Y_MIN + BC_STD_BOX_EDGE_Y_MAX) / 2.0 # 50.0

# Semicircle's flat diameter is on the line x = BC_STD_BOX_EDGE_X_FURTHEST_FROM_GOAL
# and it's centered vertically at BC_STD_BOX_CENTER_Y.
BC_SEMICIRCLE_DIAMETER_X_STD = BC_STD_BOX_EDGE_X_FURTHEST_FROM_GOAL # Center of the full circle for Arc/Wedge
BC_SEMICIRCLE_CENTER_Y_STD = BC_STD_BOX_CENTER_Y

# Calculate the radius needed for the semicircle to encompass the box.
_dx = BC_STD_BOX_EDGE_X_GOAL_LINE - BC_SEMICIRCLE_DIAMETER_X_STD
_dy = BC_STD_BOX_EDGE_Y_MAX - BC_SEMICIRCLE_CENTER_Y_STD # or use BC_STD_BOX_EDGE_Y_MIN
BC_SEMICIRCLE_RADIUS_STD = np.sqrt(_dx**2 + _dy**2)
BC_SEMICIRCLE_RADIUS_SQUARED_STD = BC_SEMICIRCLE_RADIUS_STD**2

# --- Helper function for plot text consistency (optional, if sequence_outcome_type isn't enough) ---
def _is_point_in_plot_big_chance_area(point_x, point_y, is_attacking_right_to_left):
    """
    Checks if a point is in the big chance area, considering attack direction.
    Uses the global-like BC_... constants.
    """
    if is_attacking_right_to_left: # Attacking R->L (goal at x=0 in data)
        # Semicircle flat diameter at data x_val = 100 - BC_SEMICIRCLE_DIAMETER_X_STD (e.g., 17)
        # Bulges towards x=0. Point must be to the left of or on the diameter.
        ref_x = 100.0 - BC_SEMICIRCLE_DIAMETER_X_STD
        if point_x > ref_x: # point_x is to the right of diameter, not in bulge
            return False
        # Check distance from center of diameter (ref_x, BC_SEMICIRCLE_CENTER_Y_STD)
        dist_sq = (point_x - ref_x)**2 + (point_y - BC_SEMICIRCLE_CENTER_Y_STD)**2
        return dist_sq <= BC_SEMICIRCLE_RADIUS_SQUARED_STD
    else: # Attacking L->R (goal at x=100 in data)
        # Semicircle flat diameter at data x_val = BC_SEMICIRCLE_DIAMETER_X_STD (e.g., 83)
        # Bulges towards x=100. Point must be to the right of or on the diameter.
        ref_x = BC_SEMICIRCLE_DIAMETER_X_STD
        if point_x < ref_x: # point_x is to the left of diameter, not in bulge
            return False
        # Check distance from center of diameter (ref_x, BC_SEMICIRCLE_CENTER_Y_STD)
        dist_sq = (point_x - ref_x)**2 + (point_y - BC_SEMICIRCLE_CENTER_Y_STD)**2
        return dist_sq <= BC_SEMICIRCLE_RADIUS_SQUARED_STD

# --- Function: Find Opponent Buildup After Specific Team's trigger ---

FIRST_PHASE_EXIT_X = 50.0
MAX_ACTIVE_BUILDUP_SECONDS = 20.0

DEFAULT_FIRST_PHASE_BUILDUP_TRIGGERS = (
    'Throw-in',
    'Free Kick',
    'Goal Kick',
    'Offside provoked',
    'Keeper pick-up',
    'Claim',
    'Ball recovery',
)
GOAL_KICK_FLAG_ALIASES = ('Goal kick', 'Goal kick taken')
FIRST_PHASE_ACTIVE_EVENT_TYPES = frozenset({
    'Pass', 'Take On', 'Ball touch', 'Dispossessed', 'Offside Pass',
    'Foul', 'Out', 'Goal', 'Miss', 'Attempt Saved', 'Post',
})


def _flag_is_true(value):
    if value is None or pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'y'}
    return bool(value)


def _row_has_any_flag(row, aliases):
    return any(
        alias in row.index and _flag_is_true(row.get(alias))
        for alias in aliases
    )


def _event_total_seconds(row):
    minute = pd.to_numeric(pd.Series([row.get('timeMin')]), errors='coerce').iloc[0]
    second = pd.to_numeric(pd.Series([row.get('timeSec')]), errors='coerce').iloc[0]
    if pd.isna(minute) or pd.isna(second):
        return np.nan
    return float(minute) * 60.0 + float(second)


def _attacking_perspective_x(event, attacking_team):
    x = pd.to_numeric(pd.Series([event.get('x')]), errors='coerce').iloc[0]
    if pd.isna(x):
        return np.nan
    return float(x) if event.get('team_name') == attacking_team else 100.0 - float(x)


def _is_goal_kick_pass(event, attacking_team):
    return (
        event.get('team_name') == attacking_team
        and event.get('type_name') == 'Pass'
        and _row_has_any_flag(event, GOAL_KICK_FLAG_ALIASES)
    )


def _is_first_phase_active_event(event, attacking_team):
    return (
        event.get('team_name') == attacking_team
        and event.get('type_name') in FIRST_PHASE_ACTIVE_EVENT_TYPES
    )


def classify_buildup_type(sequence_df):
    """
    Canonical first-phase distribution:
    - Long Ball: first pass attempt is long;
    - Short-Long: first pass is short, a later first-phase pass is long;
    - Short-Short: all first-phase pass attempts are short.
    """
    if sequence_df is None or sequence_df.empty or 'type_name' not in sequence_df.columns:
        return None

    passes = sequence_df[sequence_df['type_name'].eq('Pass')]
    if passes.empty:
        return None
    if _flag_is_true(passes.iloc[0].get('lb')):
        return 'Long Ball'
    if len(passes) > 1 and passes.iloc[1:]['lb'].apply(_flag_is_true).any():
        return 'Short-Long'
    return 'Short-Short'


def _build_first_phase_trigger_mask(df, attacking_team, defending_team, triggers_buildups):
    trigger_types = set(triggers_buildups or DEFAULT_FIRST_PHASE_BUILDUP_TRIGGERS)

    # Goal kicks and goalkeeper recoveries remain canonical REL-05 origins.
    trigger_types.update({'Goal Kick', 'Ball recovery'})

    mask = pd.Series(False, index=df.index, dtype=bool)

    canonical_restart_types = (
        {'Throw-in', 'Free Kick', 'Goal Kick'}
        & trigger_types
    )
    if canonical_restart_types:
        mask |= df.apply(
            lambda row: (
                row.get('team_name') == attacking_team
                and classify_restart_event(row) in canonical_restart_types
            ),
            axis=1,
        )

    if 'Out' in trigger_types:
        mask |= (
            df['team_name'].eq(defending_team)
            & df['type_name'].eq('Out')
            & df['outcome'].eq('Unsuccessful')
        )
    if 'Foul' in trigger_types:
        mask |= (
            df['team_name'].eq(defending_team)
            & df['type_name'].eq('Foul')
            & df['outcome'].eq('Unsuccessful')
        )
    if 'Offside provoked' in trigger_types:
        mask |= (
            df['team_name'].eq(attacking_team)
            & df['type_name'].eq('Offside provoked')
            & df['outcome'].eq('Successful')
        )
    if 'Keeper pick-up' in trigger_types:
        mask |= (
            df['team_name'].eq(attacking_team)
            & df['type_name'].eq('Keeper pick-up')
            & df['outcome'].eq('Successful')
        )
    if 'Claim' in trigger_types:
        mask |= (
            df['team_name'].eq(attacking_team)
            & df['type_name'].eq('Claim')
            & df['outcome'].eq('Successful')
        )
    if 'Ball recovery' in trigger_types:
        positional_roles = df.get(
            'positional_role',
            pd.Series(
                index=df.index,
                dtype='object',
            ),
        )

        mask |= (
            df['team_name'].eq(attacking_team)
            & df['type_name'].eq('Ball recovery')
            & df['outcome'].eq('Successful')
            & positional_roles.eq('GK')
        )
    return mask


def _controlled_event_reaches_phase_exit(event, phase_exit_x):
    x = pd.to_numeric(pd.Series([event.get('x')]), errors='coerce').iloc[0]
    end_x = pd.to_numeric(pd.Series([event.get('end_x')]), errors='coerce').iloc[0]
    if pd.notna(x) and float(x) >= phase_exit_x:
        return True
    return (
        event.get('outcome') == 'Successful'
        and pd.notna(end_x)
        and float(end_x) >= phase_exit_x
    )


def _find_first_phase_buildup_sequences(
    df_processed,
    attacking_team,
    defending_team,
    *,
    triggers_buildups=None,
    max_passes_in_buildup_sequence=50,
    shot_types=None,
    start_x=50,
    phase_exit_x=FIRST_PHASE_EXIT_X,
    max_active_buildup_seconds=MAX_ACTIVE_BUILDUP_SECONDS,
):
    """
    Canonical first-phase build-up contract.

    Trigger: own-half restart/control event.
    Active start: first on-ball action after the trigger; an explicit goal kick
    is both trigger and active start.
    End: controlled entry into the opposition half, 20s active-possession
    boundary, terminal event, turnover, or period boundary.
    Dead/restart time is stored separately and excluded from active duration.
    """
    shot_types = shot_types or ['Goal', 'Miss', 'Attempt Saved', 'Post']

    required_cols = [
        'id', 'eventId', 'team_name', 'type_name', 'outcome', 'x', 'y',
        'end_x', 'end_y', 'playerName', 'Mapped Jersey Number', 'timeMin',
        'timeSec', 'lb', 'Length', 'cross', 'Corner taken',
    ]
    missing = [col for col in required_cols if col not in df_processed.columns]
    if missing:
        print(f"Error: Missing required columns: {set(missing)}")
        return pd.DataFrame()

    optional_cols = [
        'positional_role', 'receiver', 'receiver_jersey_number', 'In-swinger',
        'Out-swinger', 'Straight', 'Right footed', 'Left footed', 'Own goal',
        'Penalty', 'Blocked', 'Goal mouth y co-ordinate', 'periodId',
        'ThrowIn', 'Throw in', 'Throw-in', 'Throw-in taken',
        'Free kick taken', 'Freekick taken', 'Free Kick taken',
        'Free kick', 'Freekick',
        *GOAL_KICK_FLAG_ALIASES,
    ]
    cols = required_cols + [
        col for col in optional_cols
        if col in df_processed.columns and col not in required_cols
    ]
    df = df_processed[cols].copy().reset_index(drop=True)

    trigger_mask = _build_first_phase_trigger_mask(
        df, attacking_team, defending_team, triggers_buildups
    )
    triggers = df[trigger_mask].copy()
    if triggers.empty:
        return pd.DataFrame()

    triggers['_attacking_trigger_x'] = triggers.apply(
        lambda row: _attacking_perspective_x(row, attacking_team),
        axis=1,
    )
    triggers = triggers[triggers['_attacking_trigger_x'].le(float(start_x))]
    triggers = triggers.drop_duplicates(subset=['id'], keep='first')
    if triggers.empty:
        return pd.DataFrame()

    # REL-10B: detector candidate count is measured after
    # canonical trigger eligibility and before UI filters.
    coverage_candidate_count = int(len(triggers))

    all_rows = []
    consumed_event_ids = set()
    sequence_id = 0

    for trigger_idx in triggers.index:
        trigger_event = df.iloc[trigger_idx]
        if trigger_event.get('id') in consumed_event_ids:
            continue

        trigger_period = trigger_event.get('periodId')
        trigger_x = _attacking_perspective_x(trigger_event, attacking_team)
        trigger_zone = get_pitch_third(trigger_x)
        canonical_restart_type = classify_restart_event(trigger_event)
        trigger_is_restart_delivery = canonical_restart_type in {
            'Throw-in',
            'Free Kick',
            'Goal Kick',
        }
        trigger_type = (
            canonical_restart_type
            if trigger_is_restart_delivery
            else str(trigger_event.get('type_name', 'Unknown trigger'))
        )

        first_active_idx = None
        # The actual restart delivery is already the first active on-ball action.
        scan_start = (
            trigger_idx
            if trigger_is_restart_delivery
            else trigger_idx + 1
        )

        for candidate_idx in range(scan_start, len(df)):
            candidate = df.iloc[candidate_idx]
            candidate_period = candidate.get('periodId')

            if (
                pd.notna(trigger_period)
                and pd.notna(candidate_period)
                and candidate_period != trigger_period
            ):
                break

            if _is_first_phase_active_event(candidate, attacking_team):
                first_active_idx = candidate_idx
                break

            if (
                candidate.get('team_name') == defending_team
                and candidate.get('outcome') == 'Successful'
            ):
                break

        if first_active_idx is None:
            continue

        first_active = df.iloc[first_active_idx]
        active_start_total = _event_total_seconds(first_active)
        active_start_min = first_active.get('timeMin')
        active_start_sec = first_active.get('timeSec')
        first_active_type = (
            'Goal kick'
            if _is_goal_kick_pass(first_active, attacking_team)
            else first_active.get('type_name', 'Unknown')
        )

        sequence_events = []
        successful_passes = 0
        pass_attempts = 0
        legacy_outcome = 'Unknown'
        termination_reason = None

        for event_idx in range(first_active_idx, len(df)):
            event = df.iloc[event_idx]
            event_period = event.get('periodId')

            if (
                pd.notna(trigger_period)
                and pd.notna(event_period)
                and event_period != trigger_period
            ):
                if sequence_events:
                    legacy_outcome = 'Possession Retained'
                    termination_reason = 'period_boundary'
                break

            event_total = _event_total_seconds(event)
            if (
                sequence_events
                and pd.notna(active_start_total)
                and pd.notna(event_total)
                and max_active_buildup_seconds is not None
                and event_total - active_start_total > float(max_active_buildup_seconds)
            ):
                legacy_outcome = 'Possession Consolidated'
                termination_reason = 'time_window_elapsed'
                break

            event_team = event.get('team_name')
            event_type = event.get('type_name')
            outcome = event.get('outcome')
            successful = outcome == 'Successful'
            unsuccessful = outcome == 'Unsuccessful'

            # A foul can be recorded on the defending team that committed it.
            # It must terminate the attacking buildup before generic
            # opponent-event handling skips unsuccessful opponent actions.
            if event_type == 'Foul':
                if sequence_events:
                    is_penalty_foul = (
                        event.get('Penalty')
                        in [1, '1', True]
                    )

                    legacy_outcome = (
                        'Penalty won'
                        if is_penalty_foul
                        else 'Foul'
                    )

                    termination_reason = (
                        'penalty_awarded'
                        if is_penalty_foul
                        else 'foul'
                    )

                break

            if event_team == defending_team:
                if unsuccessful:
                    continue
                if successful:
                    legacy_outcome = 'Lost Possessions'
                    termination_reason = 'opponent_regain'
                    break
                continue

            if event_team != attacking_team or event_type == 'Unknown':
                continue

            action = event.to_dict()
            action.update({
                'trigger_sequence_id': sequence_id,
                'trigger_zone': trigger_zone,
                'triggering_trigger_Opta_id': trigger_event.get('id'),
                'timeMin_at_trigger': trigger_event.get('timeMin'),
                'timeSec_at_trigger': trigger_event.get('timeSec'),
                'type_of_initial_trigger': trigger_type,
                'timeMin_at_active_start': active_start_min,
                'timeSec_at_active_start': active_start_sec,
                'first_active_action_type': first_active_type,
            })

            if event_type == 'Pass':
                sequence_events.append(action)
                pass_attempts += 1

                if unsuccessful:
                    legacy_outcome = 'Lost Possessions'
                    termination_reason = 'unsuccessful_pass'
                    break

                if successful:
                    successful_passes += 1

                    if _controlled_event_reaches_phase_exit(event, float(phase_exit_x)):
                        legacy_outcome = 'Possession Consolidated'
                        termination_reason = 'opposition_half_reached'
                        break

                    if (
                        max_passes_in_buildup_sequence is not None
                        and pass_attempts >= int(max_passes_in_buildup_sequence)
                    ):
                        legacy_outcome = 'Possession Consolidated'
                        termination_reason = 'safety_pass_limit'
                        break

                continue

            if event_type in shot_types:
                action['shot_end_y'] = event.get('Goal mouth y co-ordinate')
                sequence_events.append(action)
                if event_type == 'Goal':
                    legacy_outcome = (
                        'Own Goal'
                        if event.get('Own goal') in [1, '1', True]
                        else 'Goals'
                    )
                else:
                    legacy_outcome = 'Shots'
                break

            if event_type == 'Offside Pass':
                if sequence_events:
                    sequence_events.append(action)
                    legacy_outcome = 'Offside'
                break

            if event_type == 'Out':
                if sequence_events:
                    sequence_events.append(action)
                    legacy_outcome = 'Out'
                break

            if event_type == 'Dispossessed':
                if sequence_events:
                    sequence_events.append(action)
                    legacy_outcome = 'Lost Possessions'
                    termination_reason = 'dispossessed'
                break

            if event_type == 'Take On' and unsuccessful:
                sequence_events.append(action)
                legacy_outcome = 'Lost Possessions'
                termination_reason = 'failed_take_on'
                break

            if event_type == 'Ball touch' and unsuccessful:
                if sequence_events:
                    sequence_events.append(action)
                    legacy_outcome = 'Lost Possessions'
                    termination_reason = 'failed_control'
                break

            if event_type in {'Take On', 'Ball touch'} and successful:
                sequence_events.append(action)
                if _controlled_event_reaches_phase_exit(event, float(phase_exit_x)):
                    legacy_outcome = 'Possession Consolidated'
                    termination_reason = 'opposition_half_reached'
                    break

        if sequence_events and legacy_outcome == 'Unknown':
            legacy_outcome = 'Possession Retained'
            termination_reason = 'data_end'

        if not sequence_events or pass_attempts == 0:
            continue

        df_seq = pd.DataFrame(sequence_events)
        df_seq = df_seq[df_seq['team_name'].eq(attacking_team)].drop_duplicates(
            subset=[
                'eventId', 'team_name', 'type_name', 'x', 'y',
                'end_x', 'end_y', 'timeMin', 'timeSec',
            ]
        ).copy()
        if df_seq.empty:
            continue

        buildup_type = classify_buildup_type(df_seq)

        if (
            termination_reason == 'time_window_elapsed'
            and max_active_buildup_seconds is not None
        ):
            active_duration = float(max_active_buildup_seconds)
        else:
            last_total = _event_total_seconds(df_seq.iloc[-1])
            active_duration = (
                max(0.0, float(last_total) - float(active_start_total))
                if pd.notna(last_total) and pd.notna(active_start_total)
                else np.nan
            )

        df_seq['buildup_pass_count'] = int(successful_passes)
        df_seq['buildup_active_duration_seconds'] = active_duration
        df_seq['buildup_type'] = buildup_type
        df_seq['sequence_outcome_type'] = legacy_outcome
        df_seq = apply_sequence_outcome_contract(
            df_seq,
            viewpoint='attacking',
            legacy_outcome=legacy_outcome,
            termination_reason=termination_reason,
        )

        all_rows.extend(df_seq.to_dict('records'))
        consumed_event_ids.update(df_seq['id'].dropna().tolist())
        consumed_event_ids.add(trigger_event.get('id'))
        sequence_id += 1

    if not all_rows:
        return attach_sequence_coverage(
            pd.DataFrame(),
            candidates=coverage_candidate_count,
            sequence_id_column="trigger_sequence_id",
        )

    result = pd.DataFrame(all_rows)
    print(
        f"Constructed {result['trigger_sequence_id'].nunique()} "
        "canonical first-phase buildup sequences."
    )
    return attach_sequence_coverage(
        result,
        candidates=coverage_candidate_count,
        sequence_id_column="trigger_sequence_id",
    )


def _find_legacy_buildup_sequences(df_processed, attacking_team,
                        defending_team, # Team that lost possession
                        metric_to_analyze = 'buildup_phase',
                        triggers_buildups=['Out', 'Foul', 'Card', 'Miss', 'Offside provoked', 'Save', 'Claim', 'Keeper pick-up', 'Ball recovery', 'Corner Awarded', 'Attempt Saved'], # Triggers that start a buildup
                        max_passes_in_buildup_sequence=50,
                        shot_types=['Goal', 'Miss', 'Attempt Saved', 'Post'],
                        start_x = 50):
    """
    Identifies sequences of successful passes by the TEAM THAT GAINED POSSESSION
    immediately following a possession trigger by the specified 'defending_team'.
    Categorizes by the zone of the initial trigger.

    Args:
        df_processed (pd.DataFrame): Main processed DataFrame.
        defending_team (str): Name of the team whose trigger triggers analysis.
        triggers_buildups (list): 'type_name' values for 'defending_team'
                                      that signify losing possession.
        max_passes_in_buildup_sequence (int): Max passes to trace for the team that gained possession.

    Returns:
        pd.DataFrame: DataFrame of buildup sequences by the team that gained possession,
                      with 'trigger_sequence_id' and 'trigger_zone' (where possession was lost).
    """
    print(f"Identifying buildup sequences for {attacking_team} starting in its Defending Third")

    # --- Define Base and Optional Columns to Select ---
    # Required columns always needed
    required_cols = ['id', 'eventId', 'team_name', 'type_name', 'outcome', 'x', 'y',
                     'end_x', 'end_y', 'playerName', 'Mapped Jersey Number',
                     'timeMin', 'timeSec', 'lb', 'Length',  'cross', 'Corner taken']
    # Optional columns that may be present
    optional_cols = ['positional_role', 'receiver', 'receiver_jersey_number', 'In-swinger', 'Out-swinger', 'Straight', 'Right footed', 'Left footed', 'Own goal', 'Penalty', 'Blocked', 'Goal mouth y co-ordinate','periodId']

    # Check base requirements
    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        print(f"Error: Missing required columns: {missing}");
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
    print(f"  Including optional columns found: {found_optional}")

    df = df_processed[cols_to_select].copy()
    df = df.reset_index(drop=True)

    # --- Identify Buildup Triggers ---
    triggers_filter = pd.Series(False, index=df.index)
    if 'Out' in triggers_buildups:
        triggers_filter |= ((df['team_name'] == defending_team) & (df['type_name'] == 'Out') & (df['outcome'] == 'Unsuccessful'))
    if 'Foul' in triggers_buildups:
        triggers_filter |= ((df['team_name'] == defending_team) & (df['type_name'] == 'Foul') & (df['outcome'] == 'Unsuccessful'))
    if 'Card' in triggers_buildups: triggers_filter |= ((df['team_name'] == defending_team) & (df['type_name'] == 'Card') & (df['outcome'] == 'Successful'))
    if 'Keeper pick-up' in triggers_buildups: triggers_filter |= ((df['team_name'] == attacking_team) & (df['type_name'] == 'Keeper pick-up') & (df['outcome'] == 'Successful'))
    if 'Claim' in triggers_buildups: triggers_filter |= ((df['team_name'] == attacking_team) & (df['type_name'] == 'Claim') & (df['outcome'] == 'Successful'))
    # if 'Attempt Saved' in triggers_buildups: triggers_filter |= ((df['team_name'] == defending_team) & (df['type_name'] == 'Attempt Saved') & (df['outcome'] == 'Successful') & (df['Blocked'].isin([1, '1', True])))
    if 'Offside provoked' in triggers_buildups: triggers_filter |= ((df['team_name'] == attacking_team) & (df['type_name'] == 'Offside provoked') & (df['outcome'] == 'Successful'))
    if 'Ball recovery' in triggers_buildups: triggers_filter |= ((df['team_name'] == attacking_team) & (df['type_name'] == 'Ball recovery') & (df['outcome'] == 'Successful') & (df['positional_role'] == 'GK'))
    if 'Corner Awarded' in triggers_buildups: triggers_filter |= ((df['type_name'] == 'Corner Awarded') & (df['team_name'] == defending_team) & (df['outcome'] == 'Unsuccessful')) # | ((df['team_name'] == attacking_team) & (df['outcome'] == 'Successful'))))

    df_triggers_raw = df[triggers_filter].copy()
    if df_triggers_raw.empty: print(f"No triggers events for {attacking_team}."); return pd.DataFrame()

    if metric_to_analyze == 'buildup_phase':
        own_half_filter = (
            (
                (df_triggers_raw['x'] <= start_x)
                & (
                    df_triggers_raw['team_name']
                    == attacking_team
                )
            )
            |
            (
                (df_triggers_raw['x'] >= start_x)
                & (
                    df_triggers_raw['team_name']
                    == defending_team
                )
            )
        )
        df_triggers_own_half = df_triggers_raw.loc[own_half_filter].copy()
        if df_triggers_own_half.empty: print(f"No triggers events for {attacking_team} in its own half"); return pd.DataFrame()

        # df_triggers = df_triggers_raw.drop_duplicates(subset=['id'], keep='first').copy()
        # if df_triggers_raw.empty: print(f"No unique trigger events after deduplication by 'id' for {defending_team}."); return pd.DataFrame()
        # print(f"Found {len(df_triggers)} unique possession trigger events by {defending_team}. Tracing...")

        df_triggers = df_triggers_own_half.drop_duplicates(subset=['id'], keep='first').copy()
        if df_triggers.empty: print(f"No unique trigger events after deduplication by 'id' for {defending_team}."); return pd.DataFrame()
        print(f"Found {len(df_triggers)} unique possession trigger events by {defending_team}. Tracing...")

    elif metric_to_analyze == 'set_piece':
        opponent_half_filter = (
            (
                (df_triggers_raw['x'] >= start_x)
                & (
                    df_triggers_raw['team_name']
                    == attacking_team
                )
            )
            |
            (
                (df_triggers_raw['x'] <= start_x)
                & (
                    df_triggers_raw['team_name']
                    == defending_team
                )
            )
        )
        df_triggers_own_half = df_triggers_raw.loc[opponent_half_filter].copy()
        if df_triggers_own_half.empty: print(f"No triggers events for {attacking_team} in its own half"); return pd.DataFrame()

        df_triggers = df_triggers_own_half.drop_duplicates(subset=['id'], keep='first').copy()
        if df_triggers.empty: print(f"No unique trigger events after deduplication by 'id' for {defending_team}."); return pd.DataFrame()
        print(f"Found {len(df_triggers)} unique possession trigger events by {defending_team}. Tracing...")


    # --- Trace Subsequent Sequences ---
    all_buildup_events_with_trigger_info = [] # Stores dictionaries
    team_building_up = [t for t in df['team_name'].unique() if t != defending_team][0]
    sequence_id_counter = 0

    processed_trigger_event_ids = set()

    for trigger_original_df_idx in df_triggers.index:
        trigger_event = df.iloc[trigger_original_df_idx]
        trigger_period = trigger_event.get('periodId')

        # Zone where possession was gained by attacking team
        if trigger_event['type_name'] in ('Out', 'Foul'):
            trigger_coord = 100 - trigger_event['x']
            trigger_zone = get_pitch_third(trigger_coord)
        elif trigger_event['type_name'] in ('Card', 'Offside provoked'):
            next_event_original_df_idx = trigger_original_df_idx + 1
            next_event = df.iloc[next_event_original_df_idx]
            trigger_zone = get_pitch_third(next_event['x'])
        elif trigger_event['type_name'] in ('Keeper pick-up', 'Claim', 'Ball recovery'):
            trigger_zone = get_pitch_third(trigger_event['x'])
        elif trigger_event['type_name'] in ('Corner Awarded'):
            if trigger_event['team_name'] == attacking_team:
                trigger_zone = get_pitch_third(trigger_event['x'])
            else:
                trigger_coord = 100 - trigger_event['x']
                trigger_zone = get_pitch_third(trigger_coord)



        time_min_at_trigger = trigger_event.get('timeMin'); time_sec_at_trigger = trigger_event.get('timeSec')
        is_penalty_trigger = (
            trigger_event.get('type_name') == 'Foul'
            and trigger_event.get('Penalty') in [1, '1', True]
        )
        type_of_trigger = (
            'Penalty'
            if is_penalty_trigger
            else trigger_event.get('type_name', 'Unknown trigger')
        )

        current_opponent_sequence_events = []
        num_passes_in_seq = 0
        sequence_outcome_type = 'Unknown' # Default value
        current_event_original_df_idx = trigger_original_df_idx # Start from the trigger event index

        # Trace forward to find the opponent's sequence
        if trigger_event['id'] not in processed_trigger_event_ids:
            while current_event_original_df_idx < len(df) - 1 and num_passes_in_seq < max_passes_in_buildup_sequence:
                current_event_original_df_idx += 1 # Move to the event *after* the trigger or last pass
                action_by_gaining_team = df.iloc[current_event_original_df_idx]
                action_period = action_by_gaining_team.get(
                        'periodId'
                )
                if (
                    pd.notna(trigger_period)
                    and pd.notna(action_period)
                    and action_period != trigger_period
                ):
                    break

                is_correct_team = (action_by_gaining_team['team_name'] == team_building_up)
                is_pass = (action_by_gaining_team['type_name'] == 'Pass')
                is_unknown = (action_by_gaining_team['type_name'] == 'Unknown')
                is_successful_event = (action_by_gaining_team['outcome'] == 'Successful')
                is_defending_team = (action_by_gaining_team['team_name'] == defending_team)
                is_not_successful_event = (action_by_gaining_team['outcome'] == 'Unsuccessful')
                is_shot = (action_by_gaining_team['type_name'] in shot_types)
                is_end_sequence = (action_by_gaining_team['type_name'] in ('Foul', 'Out', 'Keeper pick-up', 'Claim', 'Dispossessed', 'Offside Pass'))
                is_take_on = (action_by_gaining_team['type_name'] == 'Take On')
                is_ball_touch = (action_by_gaining_team['type_name'] == 'Ball touch')
                is_penalty = (action_by_gaining_team.get('Penalty') in [1, '1', True])

                action_data = action_by_gaining_team.to_dict()
                action_data['trigger_sequence_id'] = sequence_id_counter
                action_data['trigger_zone'] = trigger_zone
                action_data['triggering_trigger_Opta_id'] = trigger_event['id']
                action_data['timeMin_at_trigger'] = time_min_at_trigger
                action_data['timeSec_at_trigger'] = time_sec_at_trigger
                action_data['type_of_initial_trigger'] = type_of_trigger

                if is_end_sequence and len(current_opponent_sequence_events) > 0:
                    current_opponent_sequence_events.append(action_data)
                    if action_by_gaining_team['type_name'] == 'Foul':
                        if is_penalty:
                            sequence_outcome_type = 'Penalty won'
                        else:
                            sequence_outcome_type = 'Foul'
                    elif action_by_gaining_team['type_name'] == 'Offside Pass':
                        sequence_outcome_type = 'Offside'
                    elif action_by_gaining_team['type_name'] == 'Out':
                        sequence_outcome_type = 'Out'
                    elif action_by_gaining_team['type_name'] == 'Dispossessed':
                            sequence_outcome_type = f"Lost Possessions"
                    break # End the sequence here
                elif is_end_sequence:
                    break # End the sequence here
                elif is_correct_team and is_pass:
                    if is_successful_event: #successful pass
                        current_opponent_sequence_events.append(action_data)
                        num_passes_in_seq += 1
                    elif is_not_successful_event:
                        current_opponent_sequence_events.append(action_data)
                        sequence_outcome_type = 'Lost Possessions'
                        break

                elif is_correct_team and is_shot: # Gaining team attempted a shot
                    action_data['shot_end_y'] = action_by_gaining_team['Goal mouth y co-ordinate']
                    current_opponent_sequence_events.append(action_data)

                    is_own_goal = action_by_gaining_team.get('Own goal') == 1

                    if (
                        metric_to_analyze == 'set_piece'
                        and is_penalty
                    ):
                        if action_by_gaining_team['type_name'] == 'Goal':
                            sequence_outcome_type = 'Penalty Goal'
                        elif action_by_gaining_team['type_name'] == 'Attempt Saved':
                            sequence_outcome_type = 'Penalty Saved'
                        else:
                            sequence_outcome_type = 'Penalty Missed'
                    elif action_by_gaining_team['type_name'] == 'Goal':
                        if is_own_goal:
                            sequence_outcome_type = "Own Goal"
                        else:
                            sequence_outcome_type = f"Goals"
                    else:
                        sequence_outcome_type = f"Shots"
                    break # End the sequence here
                elif is_unknown: # Unknown event type
                    continue # Skip this event
                elif is_ball_touch and is_successful_event: # Any unintentional ball touch
                    current_opponent_sequence_events.append(action_data)
                    continue # Skip this event
                elif is_correct_team and is_successful_event: # Gaining team still has ball
                    continue # Skip this event
                elif is_correct_team and is_take_on and is_not_successful_event: # Gaining team lost possession due to unsuccessful take on
                    current_opponent_sequence_events.append(action_data)
                    sequence_outcome_type = f"Lost Possessions"
                    break
                elif is_correct_team and is_ball_touch and is_not_successful_event: # Gaining team lost possession due to unsuccessful control
                    if len(current_opponent_sequence_events) < 1: # If no events yet, don't count this as a sequence
                        break
                    current_opponent_sequence_events.append(action_data)
                    sequence_outcome_type = f"Lost Possessions"
                    break
                elif is_defending_team and is_not_successful_event: # Losing team fail to regain possession
                    processed_trigger_event_ids.add(action_by_gaining_team['id'])
                    continue # Skip this event
                elif (is_defending_team and is_successful_event):
                    sequence_outcome_type = 'Lost Possessions'
                    break
                else: # Some other event or end of data
                    break

        # Deduplicate and filter to keep only events by the correct team
        df_seq = pd.DataFrame(current_opponent_sequence_events)

        if not df_seq.empty and 'team_name' in df_seq.columns:
            df_seq_deduped = df_seq[df_seq['team_name'] == team_building_up].drop_duplicates(subset=[
                'eventId', 'team_name', 'type_name', 'x', 'y', 'end_x', 'end_y', 'timeMin', 'timeSec'
            ])
            pass_count = df_seq_deduped[
                (df_seq_deduped['type_name'] == 'Pass') & (df_seq_deduped['outcome'] == 'Successful')
            ].shape[0]

            df_seq_deduped['buildup_pass_count'] = pass_count
            df_seq_deduped['sequence_outcome_type'] = sequence_outcome_type
            df_seq_deduped = apply_sequence_outcome_contract(
                df_seq_deduped,
                viewpoint='attacking',
                legacy_outcome=sequence_outcome_type,
            )

            all_buildup_events_with_trigger_info.extend(
                df_seq_deduped.to_dict('records')
            )

            sequence_id_counter += 1

    # --- End Loop ---

    if not all_buildup_events_with_trigger_info: return pd.DataFrame()
    df_all_sequences = pd.DataFrame(all_buildup_events_with_trigger_info)
    print(f"Constructed {df_all_sequences['trigger_sequence_id'].nunique()} opponent buildup sequences (incl. terminating event).")
    return df_all_sequences


@cache_derived_result("buildup_sequences")
def find_buildup_sequences(
    df_processed,
    attacking_team,
    defending_team,
    metric_to_analyze='buildup_phase',
    triggers_buildups=None,
    max_passes_in_buildup_sequence=50,
    shot_types=None,
    start_x=50,
    phase_exit_x=FIRST_PHASE_EXIT_X,
    max_active_buildup_seconds=MAX_ACTIVE_BUILDUP_SECONDS,
):
    """
    Public detector. Build-up uses the REL-05 first-phase contract; set pieces
    retain their established sequence definition.
    """
    shot_types = shot_types or ['Goal', 'Miss', 'Attempt Saved', 'Post']

    if metric_to_analyze == 'buildup_phase':
        return _find_first_phase_buildup_sequences(
            df_processed,
            attacking_team,
            defending_team,
            triggers_buildups=triggers_buildups,
            max_passes_in_buildup_sequence=max_passes_in_buildup_sequence,
            shot_types=shot_types,
            start_x=start_x,
            phase_exit_x=phase_exit_x,
            max_active_buildup_seconds=max_active_buildup_seconds,
        )

    legacy_triggers = (
        list(triggers_buildups)
        if triggers_buildups is not None
        else [
            'Out', 'Foul', 'Card', 'Miss', 'Offside provoked', 'Save',
            'Claim', 'Keeper pick-up', 'Ball recovery', 'Corner Awarded',
            'Attempt Saved',
        ]
    )
    return _find_legacy_buildup_sequences(
        df_processed,
        attacking_team,
        defending_team,
        metric_to_analyze=metric_to_analyze,
        triggers_buildups=legacy_triggers,
        max_passes_in_buildup_sequence=max_passes_in_buildup_sequence,
        shot_types=shot_types,
        start_x=start_x,
    )


def calculate_team_cross_stats(df_team_crosses, team_name):
    """Helper to calculate summary stats for one team's crosses."""
    if df_team_crosses.empty:
        return [], 0 # Empty summary, 0 total crosses

    total_crosses = len(df_team_crosses)
    in_swingers = df_team_crosses[df_team_crosses['swinger_type'] == 'In-swinger'].shape[0]
    out_swingers = df_team_crosses[df_team_crosses['swinger_type'] == 'Out-swinger'].shape[0]
    unknown_straight_swingers = df_team_crosses[df_team_crosses['swinger_type'] == 'Unknown/Straight'].shape[0]

    corner_crosses_df = df_team_crosses[df_team_crosses['is_corner']]
    total_corners = corner_crosses_df.shape[0]
    corner_in_swingers = corner_crosses_df[corner_crosses_df['swinger_type'] == 'In-swinger'].shape[0]
    corner_out_swingers = corner_crosses_df[corner_crosses_df['swinger_type'] == 'Out-swinger'].shape[0]
    corner_unknown_straight_swingers = corner_crosses_df[corner_crosses_df['swinger_type'] == 'Unknown/Straight'].shape[0]

    summary_data = [
        ["Total Crosses", total_crosses],
        ["  In-swingers", in_swingers],
        ["  Out-swingers", out_swingers],
        ["  Unknown/Straight", unknown_straight_swingers],
        ["Total Corner Crosses", total_corners],
        ["  Corner In-swingers", corner_in_swingers],
        ["  Corner Out-swingers", corner_out_swingers],
        ["  Corner Unknown/Straight", corner_unknown_straight_swingers],
    ]
    return summary_data, total_crosses

#
def prepare_cross_analysis_data(df_events, home_team_name, away_team_name):
    """
    Prepares cross data and summary statistics for both home and away teams.

    Args:
        df_events (pd.DataFrame): The main event DataFrame.
        home_team_name (str): Name of the home team.
        away_team_name (str): Name of the away team.

    Returns:
        dict: Containing processed DataFrames and summary data for crosses.
    """
    print("  Preparing cross analysis data...")
    results = {
        'home_crosses_df': pd.DataFrame(), 'home_cross_summary_data': [], 'home_total_crosses': 0,
        'away_crosses_df': pd.DataFrame(), 'away_cross_summary_data': [], 'away_total_crosses': 0,
        'summary_cols': ["Metric", "Count"]
    }

    # Ensure relevant flag columns are numeric, coercing errors.
    flag_cols_to_convert = ['cross', 'Out-swinger', 'In-swinger', 'Corner taken']
    temp_df = df_events.copy()
    for col in flag_cols_to_convert:
        if col in temp_df.columns:
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
        else:
            print(f"    Warning: Flag column '{col}' not found. Cross analysis might be incomplete.")
            temp_df[col] = 0 # Add as 0 if missing to avoid errors

    # Filter for all crosses first
    all_crosses_df = temp_df[
        (temp_df['type_name'] == 'Pass') &
        (temp_df['cross'] == 1)
    ].copy()

    if all_crosses_df.empty:
        print("    No cross events (type_name='Pass' & cross=1) found in the dataset.")
        return results

    # Determine Swinger Type from existing columns
    def determine_swinger(row):
        if row['Out-swinger'] == 1: return 'Out-swinger'
        elif row['In-swinger'] == 1: return 'In-swinger'
        else: return 'Unknown/Straight'
    all_crosses_df['swinger_type'] = all_crosses_df.apply(determine_swinger, axis=1)

    # Identify Corner Crosses from existing column
    all_crosses_df['is_corner'] = all_crosses_df['Corner taken'] == 1

    # Ensure 'x' and 'y' are numeric and handle potential NaNs for plottable crosses
    all_crosses_df['x'] = pd.to_numeric(all_crosses_df['x'], errors='coerce')
    all_crosses_df['y'] = pd.to_numeric(all_crosses_df['y'], errors='coerce')

    # Home Team
    home_df = all_crosses_df[all_crosses_df['team_name'] == home_team_name].copy()
    home_df_plottable = home_df.dropna(subset=['x', 'y']) # For count in summary
    results['home_crosses_df'] = home_df # Pass full df for plotting function to handle NaNs
    summary, total = calculate_team_cross_stats(home_df_plottable, home_team_name)
    results['home_cross_summary_data'] = summary
    results['home_total_crosses'] = total
    print(f"    Home team ({home_team_name}) - {total} plottable crosses found.")

    # Away Team
    away_df = all_crosses_df[all_crosses_df['team_name'] == away_team_name].copy()
    away_df_plottable = away_df.dropna(subset=['x','y'])
    results['away_crosses_df'] = away_df
    summary, total = calculate_team_cross_stats(away_df_plottable, away_team_name)
    results['away_cross_summary_data'] = summary
    results['away_total_crosses'] = total
    print(f"    Away team ({away_team_name}) - {total} plottable crosses found.")

    return results

# def calculate_transition_success_by_zone(df_processed, team_name,
#                                          recovery_event_types=None,
#                                          triggers_buildups_opponent=None):
#     """
#     Calculates counts of successful, failed, and neutral offensive transitions
#     originating from different pitch thirds.

#     Returns:
#         pd.DataFrame: Columns: ['recovery_zone', 'successful_transitions',
#                                 'failed_transitions', 'neutral_transitions', 'total_transitions']
#     """
#     print(f"Calculating transition success rates by zone for {team_name}...")
#     if recovery_event_types is None: recovery_event_types = ['Ball recovery', 'Interception']
#     if triggers_buildups_opponent is None: triggers_buildups_opponent = ['Pass', 'Take On', 'Error']


#     df = df_processed.sort_values('eventId').reset_index(drop=True)
#     df['next_team'] = df['team_name'].shift(-1)
#     # ... (gains_filter logic - same as before) ...
#     gains_filter = (((df['team_name'] == team_name) & df['type_name'].isin(recovery_event_types)) | ((df['team_name'] != team_name) & df['outcome'] == 'Unsuccessful' & df['type_name'].isin(triggers_buildups_opponent) & (df['next_team'] == team_name)))
#     df_possession_gains = df[gains_filter].copy()

#     if df_possession_gains.empty: return pd.DataFrame(columns=['recovery_zone', 'successful_transitions', 'failed_transitions', 'neutral_transitions', 'total_transitions'])

#     transition_outcomes_detailed = []
#     MAX_EVENTS_IN_TRANSITION = 10

#     for gain_idx, gain_event in df_possession_gains.iterrows():
#         recovery_zone = get_pitch_third(gain_event['x'])
#         current_event_idx = gain_idx
#         outcome_category = "Neutral End" # Default

#         for _ in range(MAX_EVENTS_IN_TRANSITION):
#             if current_event_idx + 1 >= len(df): break
#             current_event_idx += 1
#             next_event = df.iloc[current_event_idx]

#             if next_event['team_name'] != team_name: outcome_category = "Possession Lost"; break
#             # Define more specific outcomes based on event types
#             if next_event['type_name'] == 'Goal': outcome_category = "Goal"; break
#             if next_event['type_name'] in config.DEFAULT_SHOT_TYPES: outcome_category = "Shot"; break # Includes 'Attempt Saved', 'Post', 'Miss'
#             if next_event.get('is_key_pass', False) or next_event.get('is_assist', False):
#                 if outcome_category not in ["Goal", "Shot"]: outcome_category = "Chance Created"
#                 # Don't break on chance created, allow sequence to continue to see if it leads to shot/goal
#             if next_event['type_name'] == 'End': break # End of period

#         transition_outcomes_detailed.append({'recovery_zone': recovery_zone, 'final_outcome': outcome_category})

#     if not transition_outcomes_detailed: return pd.DataFrame(columns=['recovery_zone', 'successful_transitions', 'failed_transitions', 'neutral_transitions', 'total_transitions'])

#     df_outcomes = pd.DataFrame(transition_outcomes_detailed)

#     # Aggregate counts
#     summary_list = []
#     for zone, group in df_outcomes.groupby('recovery_zone'):
#         total = len(group)
#         successful = group['final_outcome'].isin(SUCCESSFUL_TRANSITION_CATEGORIES).sum()
#         failed = group['final_outcome'].isin(FAILED_TRANSITION_CATEGORIES).sum()
#         neutral = total - successful - failed
#         summary_list.append({
#             'recovery_zone': zone,
#             'successful_transitions': successful,
#             'failed_transitions': failed,
#             'neutral_transitions': neutral,
#             'total_transitions': total
#         })

#     df_zone_summary = pd.DataFrame(summary_list)
#     print(f"Transition success summary for {team_name}:\n{df_zone_summary}")
#     return df_zone_summary


import pandas as pd
import dash_bootstrap_components as dbc
from dash import html as dash_html

# --- New Function 1 ---
def calculate_buildup_stats(sequence_list, attacking_team_is_home):
    """
    Takes a list of sequence DataFrames and calculates aggregate stats based on
    detailed column data like 'lb' and trigger zone.
    """
    if not sequence_list:
        return {}

    total_sequences = len(sequence_list)

    # 1. Outcome Analysis (this logic is correct)
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

    # --- FIX 2: Correctly calculate flank based on the 'trigger_zone' ---
    # Your `find_buildup_sequences` function already calculates this for us.
    flanks = [seq.iloc[0].get('dominant_flank', 'Unknown') for seq in sequence_list if not seq.empty]
    flank_counts = pd.Series(flanks).value_counts().to_dict()

    # 3. Canonical first-phase distribution type.
    buildup_type_map = {'Short-Short': 0, 'Short-Long': 0, 'Long Ball': 0}
    for seq in sequence_list:
        if seq.empty:
            continue
        buildup_type = seq.iloc[-1].get('buildup_type')
        if not buildup_type or buildup_type not in buildup_type_map:
            buildup_type = classify_buildup_type(seq)
        if buildup_type in buildup_type_map:
            buildup_type_map[buildup_type] += 1

    return {
        "total": total_sequences,
        "outcomes": outcome_counts,
        "terminal_outcomes": terminal_outcome_counts,
        "flanks": flank_counts,
        "types": buildup_type_map
    }

def create_buildup_summary_cards(stats, active_filter=None):
    """
    Creates a layout of interactive cards displaying the calculated buildup stats.
    The active filter will be highlighted with a custom style.
    """
    if not stats or stats.get("total", 0) == 0:
        return dbc.Alert("No summary data to display.", color="secondary")

    def is_active(filter_type, value):
        if not active_filter:
            return False
        return str(active_filter.get(filter_type)) == str(value)

    # Card 1: Outcomes
    outcome_hierarchy = ['Goals', 'Penalty won', 'Shots', 'Big Chances', 'Lost Possessions', 'Foul', 'Offside', 'Out']
    outcome_rank_map = {outcome: i for i, outcome in enumerate(outcome_hierarchy)}
    sorted_outcome_items = sorted(stats.get('outcomes', {}).items(), key=lambda item: outcome_rank_map.get(item[0], 99))

    outcome_list_items = []
    for outcome, count in sorted_outcome_items:
        if count == 0:
            continue
        if active_filter and 'outcomes' in active_filter and active_filter['outcomes'] != outcome:
            continue
        active = is_active('outcomes', outcome)
        outcome_list_items.append(
            dbc.ListGroupItem(
                [
                    dash_html.Div(f"{outcome}"),
                    dbc.Badge(f"{count} ({count / stats['total']:.0%})", color="light", text_color="dark", className="ms-auto")
                ],
                id={'type': 'buildup-filter', 'filter_type': 'outcomes', 'value': outcome},
                action=True,
                n_clicks=0,
                active=active,
                className=f"match-summary-row {'is-active' if active else ''}"
            )
        )
    outcome_card = dbc.Card([
        dbc.CardHeader([dash_html.I(className="fa-solid fa-bullseye me-2"), "Buildup outcomes"]),
        dbc.ListGroup(outcome_list_items, flush=True)
    ], className="match-summary-card")

    # Card 2: Flanks
    flank_list_items = []
    for flank, count in stats['flanks'].items():
        if count == 0:
            continue
        if active_filter and 'flanks' in active_filter and active_filter['flanks'] != flank:
            continue
        active = is_active('flanks', flank)
        flank_list_items.append(
            dbc.ListGroupItem(
                [
                    dash_html.Div(flank),
                    dbc.Badge(f"{count} ({count / stats['total']:.0%})", color="light", text_color="dark", className="ms-auto")
                ],
                id={'type': 'buildup-filter', 'filter_type': 'flanks', 'value': flank},
                action=True,
                n_clicks=0,
                active=active,
                className=f"match-summary-row {'is-active' if active else ''}"
            )
        )
    flank_card = dbc.Card([
        dbc.CardHeader([dash_html.I(className="fa-solid fa-arrows-left-right me-2"), "Dominant flank"]),
        dbc.ListGroup(flank_list_items, flush=True)
    ], className="match-summary-card")

    # Card 3: Buildup Type
    type_list_items = []
    for b_type, count in stats['types'].items():
        if count == 0 and not (active_filter and 'types' in active_filter and active_filter['types'] == b_type):
            continue
        if active_filter and 'types' in active_filter and active_filter['types'] != b_type:
            continue
        active = is_active('types', b_type)
        type_list_items.append(
            dbc.ListGroupItem(
                [
                    dash_html.Div(b_type),
                    dbc.Badge(f"{count} ({count / stats['total']:.0%})", color="light", text_color="dark", className="ms-auto")
                ],
                id={'type': 'buildup-filter', 'filter_type': 'types', 'value': b_type},
                action=True,
                n_clicks=0,
                active=active,
                className=f"match-summary-row {'is-active' if active else ''}"
            )
        )
    type_card = dbc.Card([
        dbc.CardHeader([dash_html.I(className="fa-solid fa-route me-2"), "Initial buildup type"]),
        dbc.ListGroup(type_list_items, flush=True)
    ], className="match-summary-card")

    return dbc.Row([
        dbc.Col(outcome_card, md=4),
        dbc.Col(flank_card, md=4),
        dbc.Col(type_card, md=4)
    ], className="g-3 match-summary-grid")




# # --- New Function 2 ---
# def create_buildup_summary_cards(stats):
#     """
#     Creates a layout of cards displaying the calculated buildup stats,
#     with the Outcomes card sorted by a qualitative hierarchy.
#     """
#     if not stats or stats.get("total", 0) == 0:
#         return dbc.Alert("No summary data to display.", color="secondary")

#     # --- START OF FIX: Implement custom sorting for the Outcomes card ---
#     # 1. Define the desired order
#     outcome_hierarchy = [
#         'Goals',
#         'Shots',
#         'Big Chances', # Or 'Big Chances'
#         'Lost Possessions',
#         'Foul',
#         'Offside',
#         'Out'
#         # Any other outcomes will be appended at the end
#     ]

#     # 2. Sort the outcome items based on the hierarchy list
#     outcome_counts = stats.get('outcomes', {})

#     # Create a mapping from outcome name to its rank in the hierarchy
#     # Unlisted outcomes get a high rank to place them at the end
#     outcome_rank_map = {outcome: i for i, outcome in enumerate(outcome_hierarchy)}

#     # Sort the dictionary items by the rank of their key
#     sorted_outcome_items = sorted(
#         outcome_counts.items(),
#         key=lambda item: outcome_rank_map.get(item[0], 99) # Use .get() for safety
#     )

#     # 3. Create the list items from the now-sorted data
#     outcome_items = [
#         dash_html.Li(f"{outcome}: {count} ({count/stats['total']:.0%})", className="list-group-item bg-dark text-white border-secondary")
#         for outcome, count in sorted_outcome_items
#     ]
#     # --- END OF FIX ---

#     outcome_card = dbc.Card([
#         dbc.CardHeader("Buildup Outcomes"),
#         dbc.ListGroup(outcome_items, flush=True)
#     ], className="mb-3")

#     # Card 2: Starting Flank (no changes needed)
#     flank_items = [
#         dash_html.Li(f"{flank}: {count} ({count/stats['total']:.0%})", className="list-group-item bg-dark text-white border-secondary")
#         for flank, count in stats['flanks'].items()
#     ]
#     flank_card = dbc.Card([
#         dbc.CardHeader("Starting Flank"),
#         dbc.ListGroup(flank_items, flush=True)
#     ], className="mb-3")

#     # Card 3: Buildup Type (no changes needed)
#     type_items = [
#         dash_html.Li(f"{b_type}: {count} ({count/stats['total']:.0%})", className="list-group-item bg-dark text-white border-secondary")
#         for b_type, count in stats['types'].items()
#     ]
#     type_card = dbc.Card([
#         dbc.CardHeader("Initial Buildup Type"),
#         dbc.ListGroup(type_items, flush=True)
#     ], className="mb-3")

#     return dbc.Row([
#         dbc.Col(outcome_card, md=4),
#         dbc.Col(flank_card, md=4),
#         dbc.Col(type_card, md=4)
#     ])

def assign_flank_to_sequences(sequence_list, is_away=False):
    updated = []
    for seq_df in sequence_list:
        if seq_df.empty:
            updated.append(seq_df)
            continue

        flank_zone = "Central"
        y_coords = seq_df[seq_df['type_name'].isin(['Pass', 'Carry', 'Shot'])]['y']

        if not y_coords.empty:
            avg_y = y_coords.mean()
            if avg_y < 33:
                flank_zone = "Left"
            elif avg_y > 66:
                flank_zone = "Right"
            else:
                flank_zone = "Central"

        # Inverti se squadra è in trasferta
        if is_away:
            if flank_zone == "Left":
                flank_zone = "Right"
            elif flank_zone == "Right":
                flank_zone = "Left"

        seq_df = seq_df.copy()
        seq_df['dominant_flank'] = flank_zone
        updated.append(seq_df)

    return updated
