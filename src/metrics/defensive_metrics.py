# src/metrics/defensive_metrics.py
import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

# --- Define Default Defensive Action Types ---
DEFAULT_DEFENSIVE_TYPES = [
    'Ball recovery', 'Blocked pass', 'Challenge',
    'Clearance', 'Error', 'Foul', 'Interception', 'Tackle'
]

PPDA_ACTION_TYPES = {
    'tackle', 'challenge', 'interception', 'blocked pass', 'foul'
}
PPDA_PASS_ZONE_THRESHOLD = 60.0
PPDA_DEFENSIVE_ZONE_THRESHOLD = 40.0

def get_defensive_actions(df_processed, defensive_action_types=None):
    """
    Filters the DataFrame for relevant defensive actions, allowing dynamic selection.
    Includes Aerial duels in the defensive third by default.

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame.
        defensive_action_types (list, optional): A list of strings specifying the
                                                 event type names to consider as
                                                 defensive actions. If None, uses
                                                 DEFAULT_DEFENSIVE_TYPES.

    Returns:
        pd.DataFrame: DataFrame containing only the selected defensive action events.
    """
    logger.debug("Filtering for defensive actions...")

    # --- Determine action types to use ---
    if defensive_action_types is None:
        types_to_use = DEFAULT_DEFENSIVE_TYPES
        logger.debug(f"  Using default defensive types: {types_to_use}")
    else:
        types_to_use = defensive_action_types
        logger.debug(f"  Using specified defensive types: {types_to_use}")

    # --- Check required columns ---
    required_cols = ['team_name', 'type_name', 'x', 'y'] # Base requirements
    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        logger.warning(f"Error: Missing base required columns for defensive action analysis: {missing}")
        return pd.DataFrame()
    # Check if 'type_name' column actually exists before filtering
    if 'type_name' not in df_processed.columns:
         logger.warning(f"Error: 'type_name' column missing, cannot filter actions.")
         return pd.DataFrame()


    # --- Apply Filters ---
    # Base filter for standard defensive types provided in the list
    # Handle potential NaNs in type_name just in case, though unlikely
    defensive_filter = df_processed['type_name'].fillna('').isin(types_to_use)

    # Specific filter for Aerial duels in defensive third (always included for now)
    # Ensure 'x' exists before applying this filter
    if 'x' in df_processed.columns:
        aerial_filter = (df_processed['type_name'] == 'Aerial') & (df_processed['x'].fillna(101) <= 33.33)
        logger.debug("  Including 'Aerial' actions occurring in defensive third (x <= 33.33).")
        # Combine filters: must be one of the specified types OR a defensive third aerial
        final_filter = defensive_filter | aerial_filter
    else:
        logger.warning("  Warning: 'x' column missing, cannot apply Aerial location filter.")
        final_filter = defensive_filter # Only use standard types if 'x' is missing

    # --- Select relevant columns ---
    # Ensure we only select columns that actually exist after preprocessing
    relevant_cols = ["id", "x", "y", "team_name", "playerName",
                     "Mapped Jersey Number", "type_name", "outcome"]
    existing_cols = [col for col in relevant_cols if col in df_processed.columns]

    df_defensive_actions = df_processed.loc[final_filter, existing_cols].copy()
    logger.info(f"Found {len(df_defensive_actions)} defensive actions matching criteria.")
    return df_defensive_actions


def calculate_defensive_agg(df_defensive_actions, team_name):
    """
    Calculates the median location and count of defensive actions per player for a team.

    Args:
        df_defensive_actions (pd.DataFrame): DataFrame filtered for defensive actions.
        team_name (str): The name of the team to analyze.

    Returns:
        pd.DataFrame: DataFrame with player name, median x/y, action count, and jersey number.
                      Returns empty DataFrame if no actions for the team.
    """
    logger.debug(f"Calculating aggregated defensive metrics for {team_name}...")
    team_actions_df = df_defensive_actions[df_defensive_actions["team_name"] == team_name].copy()

    if team_actions_df.empty:
        logger.warning(f"Warning: No defensive actions found for team {team_name}.")
        return pd.DataFrame()

    # Check for required columns for aggregation
    required_agg_cols = ['playerName', 'x', 'y', 'id', 'Mapped Jersey Number']
    if not all(col in team_actions_df.columns for col in required_agg_cols):
        missing = set(required_agg_cols) - set(team_actions_df.columns)
        logger.warning(f"Error: Missing columns needed for aggregation: {missing}")
        return pd.DataFrame()

    # Group by player and calculate median location and count
    player_agg = team_actions_df.groupby('playerName').agg(
        median_x=('x', 'median'),
        median_y=('y', 'median'),
        action_count=('id', 'count'), # Count defensive actions
        jersey_number=('Mapped Jersey Number', 'first') # Get jersey number
    ).reset_index() # Make playerName a column again

    logger.info(f"Calculated metrics for {len(player_agg)} players.")
    return player_agg

# --- PPDA Calculation Function ---
# Note: PPDA is a measure of pressing intensity, calculated as:
#       PPDA = Opponent Passes / Your Team's Defensive Actions in the opponent's half
def calculate_ppda_opta(df, team_of_interest, opponent_team, def_action_ids, # Opta IDs for def actions
                        event_id_col='typeId', # Use typeId as likely column name
                        pass_zone_thresh=60.0, # X-coordinate threshold (float)
                        def_action_zone_thresh=60.0, # X-coordinate threshold (float)
                        pass_type_id=1 # Opta ID for Pass
                        ): 
    """
    Calculates PPDA (Passes Per Defensive Action) using Opta event type IDs.
    Measures pressing intensity in the opponent's half/midfield.

    Args:
        df (pd.DataFrame): DataFrame with Opta event data (needs team_name, event_id_col, x).
        team_of_interest (str): The name of the team to calculate PPDA FOR.
        opponent_team (str): The name of the opponent team.
        event_id_col (str): Column name containing Opta event type IDs.
        pass_zone_thresh (float): Max x-coordinate for opponent passes to be included
                                 (usually lower 60% of the pitch).
        def_action_zone_thresh (float): Min x-coordinate for the team's defensive actions
                                       to be included (usually upper 40% of the pitch).
        pass_type_id (int): The typeId for Pass events.
        def_action_ids (list): REQUIRED List of typeIds counting as defensive actions for PPDA.

    Returns:
        float: Calculated PPDA value (Opponent Passes / Team Def Actions in zone).
               Returns np.inf if no defensive actions occurred in the zone.
               Returns None if input validation fails.
    """
    logger.debug(f"Calculating PPDA for {team_of_interest} (vs {opponent_team})...")

    # --- Input Validation ---
    required_cols = ['team_name', event_id_col, 'x']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.warning(f"Error [PPDA]: DataFrame missing required columns: {missing}")
        return None
    if team_of_interest not in df['team_name'].unique():
        logger.warning(f"Error [PPDA]: Team '{team_of_interest}' not found.")
        return None
    if opponent_team not in df['team_name'].unique():
        logger.warning(f"Error [PPDA]: Opponent '{opponent_team}' not found.")
        return None
    # Ensure coordinate column is numeric
    if not pd.api.types.is_numeric_dtype(df['x']):
         logger.warning(f"Error [PPDA]: Column 'x' must be numeric.")
         return None
     # Ensure event id column is numeric (or convertible)
    try:
        df[event_id_col] = pd.to_numeric(df[event_id_col], errors='coerce')
        if df[event_id_col].isnull().any():
             logger.warning(f"Warning [PPDA]: Column '{event_id_col}' contains non-numeric values.")
             # Decide whether to proceed by dropping NaNs or return None
             # df = df.dropna(subset=[event_id_col]) # Option to proceed
             # return None # Option to stop
    except Exception as e:
         logger.warning(f"Error [PPDA]: Could not process event ID column '{event_id_col}': {e}")
         return None


    # --- Numerator: Opponent Passes in their Defensive Zone ---
    # Convert threshold to float just in case
    pass_zone_thresh = float(pass_zone_thresh)
    opponent_passes_in_zone = df[
        (df['team_name'] == opponent_team) &
        (df[event_id_col] == pass_type_id) &
        (df['x'].fillna(pass_zone_thresh + 1) < pass_zone_thresh) 
    ]
    num_opponent_passes = len(opponent_passes_in_zone)

    # --- Denominator: Your Team's Defensive Actions in Opponent's Half ---
    def_action_zone_thresh = float(def_action_zone_thresh)
    team_defensive_actions_in_zone = df[
        (df['team_name'] == team_of_interest) &
        (df[event_id_col].isin(def_action_ids)) & 
        (df['x'].fillna(def_action_zone_thresh - 1) >= def_action_zone_thresh)
    ]
    num_team_def_actions = len(team_defensive_actions_in_zone)

    # --- Calculate PPDA ---
    if num_team_def_actions == 0:
        ppda = np.inf
        logger.debug(f"  {team_of_interest}: 0 defensive actions in zone (x >= {def_action_zone_thresh}). PPDA = inf")
    else:
        ppda = num_opponent_passes / num_team_def_actions
        logger.debug(f"  {opponent_team} Passes (x < {pass_zone_thresh}): {num_opponent_passes}")
        logger.debug(f"  {team_of_interest} Def Actions (x >= {def_action_zone_thresh}): {num_team_def_actions}")
        logger.info(f"  Calculated PPDA: {ppda:.2f}")

    return ppda

def get_defensive_block_data(df_processed, team_name):
    """
    Prepares data for plotting a defensive block. It calculates the median position
    and action count for each player based on their defensive actions.

    Args:
        df_processed (pd.DataFrame): The main processed DataFrame.
        team_name (str): The name of the team to analyze.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: All defensive actions for the team.
            - pd.DataFrame: Aggregated data per player (median_x, median_y, action_count, etc.).
    """
    DEFENSIVE_ACTION_TYPES = [
        'Tackle', 'Interception', 'Clearance', 'Blocked pass', 
        'Ball recovery', 'Foul', 'Aerial'
    ]
    
    # Filtra le azioni difensive per la squadra specificata
    df_def_actions = df_processed[
        (df_processed['team_name'] == team_name) &
        (df_processed['type_name'].isin(DEFENSIVE_ACTION_TYPES))
    ].copy()

    if df_def_actions.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Calcola la posizione mediana e il conteggio delle azioni per ogni giocatore
    df_player_agg = df_def_actions.groupby('playerName').agg(
        median_x=('x', 'median'),
        median_y=('y', 'median'),
        action_count=('eventId', 'count')
    ).reset_index()

    # Aggiungi informazioni sui giocatori (numero di maglia, se titolare)
    player_info = df_processed[
        df_processed['playerName'].isin(df_player_agg['playerName'])
    ][['playerName', 'Mapped Jersey Number', 'Is Starter']].drop_duplicates(subset='playerName')
    
    df_player_agg = pd.merge(df_player_agg, player_info, on='playerName', how='left')
    
    # Assicura che 'Is Starter' sia un booleano
    df_player_agg['Is Starter'] = df_player_agg['Is Starter'].fillna(False).astype(bool)

    return df_def_actions, df_player_agg


def _truthy_qualifier(value):
    """Return True for Opta flag qualifiers without treating zero/NaN as flags."""
    if pd.isna(value):
        return False
    return str(value).strip().lower() not in {'', '0', '0.0', 'false', 'nan', 'none'}


def _ppda_snapshot(
    df_processed,
    team_name,
    opponent_name,
    row_mask=None,
    pass_zone_threshold=PPDA_PASS_ZONE_THRESHOLD,
    defensive_zone_threshold=PPDA_DEFENSIVE_ZONE_THRESHOLD,
):
    """Calculate one PPDA sample and retain its numerator/denominator events."""
    if df_processed is None or df_processed.empty:
        return {
            'ppda': float('inf'),
            'opponent_passes': 0,
            'defensive_actions': 0,
            'df_opponent_passes': pd.DataFrame(),
            'df_defensive_actions': pd.DataFrame(),
        }

    df = df_processed.copy()
    if row_mask is not None:
        aligned_mask = pd.Series(row_mask, index=df_processed.index).fillna(False).astype(bool)
        df = df.loc[aligned_mask].copy()

    x_numeric = pd.to_numeric(df.get('x'), errors='coerce')
    type_normalized = df.get('type_name', pd.Series('', index=df.index)).fillna('').astype(str).str.strip().str.lower()
    outcome_normalized = df.get('outcome', pd.Series('', index=df.index)).fillna('').astype(str).str.strip().str.lower()

    # Numerator: passes attempted by the opponent while building in its first 60%.
    opponent_passes_filter = (
        (df.get('team_name') == opponent_name)
        & (type_normalized == 'pass')
        & (x_numeric < float(pass_zone_threshold))
    )

    # Denominator: pressing actions made by the team in the corresponding zone.
    defensive_actions_filter = (
        (df.get('team_name') == team_name)
        & type_normalized.isin(PPDA_ACTION_TYPES)
        & (x_numeric >= float(defensive_zone_threshold))
    )
    # Opta logs committed fouls as unsuccessful and fouls suffered as successful.
    defensive_actions_filter &= ~((type_normalized == 'foul') & (outcome_normalized == 'successful'))

    df_opponent_passes = df.loc[opponent_passes_filter].copy()
    df_defensive_actions = df.loc[defensive_actions_filter].copy()
    num_passes = len(df_opponent_passes)
    num_actions = len(df_defensive_actions)

    return {
        'ppda': num_passes / num_actions if num_actions else float('inf'),
        'opponent_passes': num_passes,
        'defensive_actions': num_actions,
        'df_opponent_passes': df_opponent_passes,
        'df_defensive_actions': df_defensive_actions,
    }


def _build_ppda_player_stats(df_defensive_actions):
    """Create an interpretable action-count table without a synthetic success rate."""
    columns = ['Player', 'Actions', 'Tackles', 'Interceptions', 'Challenges', 'Blocks', 'Fouls']
    if df_defensive_actions is None or df_defensive_actions.empty:
        return pd.DataFrame(columns=columns)

    actions = df_defensive_actions.copy()
    actions['_action'] = actions['type_name'].fillna('').astype(str).str.strip().str.lower()
    actions['_count'] = 1

    index_cols = ['playerName']
    if 'Mapped Jersey Number' in actions.columns:
        index_cols.append('Mapped Jersey Number')

    pivot = actions.pivot_table(
        index=index_cols,
        columns='_action',
        values='_count',
        aggfunc='sum',
        fill_value=0,
    ).reset_index()

    for action in PPDA_ACTION_TYPES:
        if action not in pivot.columns:
            pivot[action] = 0

    def player_label(row):
        jersey = row.get('Mapped Jersey Number')
        if pd.notna(jersey):
            try:
                return f"#{int(float(jersey))} - {row.get('playerName', 'Unknown')}"
            except (TypeError, ValueError):
                pass
        return str(row.get('playerName', 'Unknown'))

    pivot['Player'] = pivot.apply(player_label, axis=1)
    pivot['Actions'] = pivot[list(PPDA_ACTION_TYPES)].sum(axis=1).astype(int)
    pivot = pivot.rename(columns={
        'tackle': 'Tackles',
        'interception': 'Interceptions',
        'challenge': 'Challenges',
        'blocked pass': 'Blocks',
        'foul': 'Fouls',
    })
    return pivot[columns].sort_values(['Actions', 'Player'], ascending=[False, True]).reset_index(drop=True)


def calculate_ppda_data(
    df_processed,
    team_name,
    opponent_name,
    pass_zone_threshold=PPDA_PASS_ZONE_THRESHOLD,
    defensive_zone_threshold=PPDA_DEFENSIVE_ZONE_THRESHOLD,
):
    """
    Calculate full-match PPDA.

    PPDA = opponent passes starting in its first 60% / pressing actions by the
    defending team from x=40 onward. All coordinates are expected to be
    normalised so that each team attacks from left to right.

    The four-item return value is kept for compatibility with existing callers.
    """
    snapshot = _ppda_snapshot(
        df_processed,
        team_name,
        opponent_name,
        pass_zone_threshold=pass_zone_threshold,
        defensive_zone_threshold=defensive_zone_threshold,
    )
    player_stats = _build_ppda_player_stats(snapshot['df_defensive_actions'])
    return (
        snapshot['ppda'],
        snapshot['df_defensive_actions'],
        snapshot['df_opponent_passes'],
        player_stats,
    )


def calculate_ppda_profile(
    df_processed,
    team_name,
    opponent_name,
    pass_zone_threshold=PPDA_PASS_ZONE_THRESHOLD,
    defensive_zone_threshold=PPDA_DEFENSIVE_ZONE_THRESHOLD,
    interval_minutes=15,
):
    """Return full-match, half-by-half and fixed-interval PPDA information."""
    df = df_processed.copy()
    time_min = pd.to_numeric(df.get('timeMin'), errors='coerce')
    period_id = pd.to_numeric(df.get('periodId'), errors='coerce') if 'periodId' in df.columns else None

    if period_id is not None and period_id.notna().any():
        first_half_mask = period_id == 1
        second_half_mask = period_id == 2
    else:
        first_half_mask = time_min < 45
        second_half_mask = time_min >= 45

    overall = _ppda_snapshot(
        df, team_name, opponent_name,
        pass_zone_threshold=pass_zone_threshold,
        defensive_zone_threshold=defensive_zone_threshold,
    )
    first_half = _ppda_snapshot(
        df, team_name, opponent_name, first_half_mask,
        pass_zone_threshold, defensive_zone_threshold,
    )
    second_half = _ppda_snapshot(
        df, team_name, opponent_name, second_half_mask,
        pass_zone_threshold, defensive_zone_threshold,
    )

    timeline_rows = []
    period_specs = [
        ('1H', first_half_mask, 0.0, max(45.0, float(time_min[first_half_mask].max()) if first_half_mask.any() else 45.0)),
        ('2H', second_half_mask, 45.0, max(90.0, float(time_min[second_half_mask].max()) if second_half_mask.any() else 90.0)),
    ]

    for period_label, base_mask, period_start, period_end in period_specs:
        interval_starts = list(np.arange(period_start, period_start + 3 * interval_minutes, interval_minutes))
        for interval_index, interval_start in enumerate(interval_starts):
            nominal_end = interval_start + interval_minutes
            interval_end = period_end if interval_index == len(interval_starts) - 1 else nominal_end
            if interval_index == len(interval_starts) - 1:
                interval_mask = base_mask & (time_min >= interval_start) & (time_min <= interval_end)
            else:
                interval_mask = base_mask & (time_min >= interval_start) & (time_min < interval_end)
            snapshot = _ppda_snapshot(
                df, team_name, opponent_name, interval_mask,
                pass_zone_threshold, defensive_zone_threshold,
            )
            raw_ppda = snapshot['ppda']
            pressure_rate = (
                snapshot['defensive_actions'] / snapshot['opponent_passes'] * 100
                if snapshot['opponent_passes'] > 0
                else 0.0
            )
            timeline_rows.append({
                'team_name': team_name,
                'period': period_label,
                'interval_label': f"{interval_start:.0f}'–{nominal_end:.0f}'",
                'window_start': round(interval_start, 1),
                'window_end': round(interval_end, 1),
                'minute': round(interval_start + interval_minutes / 2, 1),
                'ppda': raw_ppda,
                'pressure_rate': pressure_rate,
                'opponent_passes': snapshot['opponent_passes'],
                'defensive_actions': snapshot['defensive_actions'],
                'low_sample': snapshot['defensive_actions'] < 2,
            })

    return {
        'team_name': team_name,
        'opponent_name': opponent_name,
        'pass_zone_threshold': float(pass_zone_threshold),
        'defensive_zone_threshold': float(defensive_zone_threshold),
        'interval_minutes': int(interval_minutes),
        'overall': overall,
        'first_half': first_half,
        'second_half': second_half,
        'timeline': pd.DataFrame(timeline_rows),
        'player_stats': _build_ppda_player_stats(overall['df_defensive_actions']),
    }


def extract_ppda_key_events(df_processed):
    """Extract goals and dismissals to contextualise the PPDA timeline."""
    columns = ['minute', 'event_type', 'team_name', 'playerName', 'label']
    if df_processed is None or df_processed.empty:
        return pd.DataFrame(columns=columns)

    df = df_processed.copy()
    type_id = pd.to_numeric(df.get('typeId'), errors='coerce')
    type_name = df.get('type_name', pd.Series('', index=df.index)).fillna('').astype(str).str.lower()
    goal_mask = (type_id == 16) | (type_name == 'goal')

    dismissal_mask = pd.Series(False, index=df.index)
    for column in ('Red card', 'Second yellow'):
        if column in df.columns:
            dismissal_mask |= df[column].map(_truthy_qualifier)
    dismissal_mask &= (type_id == 17) | (type_name == 'card')

    events = []
    for _, row in df.loc[goal_mask | dismissal_mask].sort_values(['timeMin', 'timeSec']).iterrows():
        minute = pd.to_numeric(pd.Series([row.get('timeMin')]), errors='coerce').iloc[0]
        if pd.isna(minute):
            continue
        is_goal = bool(goal_mask.loc[row.name])
        event_type = 'goal' if is_goal else 'red_card'
        player = row.get('playerName') or 'Unknown player'
        team = row.get('team_name') or 'Unknown team'
        events.append({
            'minute': float(minute),
            'event_type': event_type,
            'team_name': team,
            'playerName': player,
            'label': f"Goal · {player} ({team})" if is_goal else f"Red card · {player} ({team})",
        })
    return pd.DataFrame(events, columns=columns)

# ============================================================================
# PLOT-10 — Defensive Shape
# ============================================================================

DEFENSIVE_SHAPE_WINDOW_MINUTES = 15
DEFENSIVE_SHAPE_MIN_OUTFIELD_PLAYERS = 7
DEFENSIVE_SHAPE_PITCH_LENGTH_M = 105.0
DEFENSIVE_SHAPE_PITCH_WIDTH_M = 68.0


def _defensive_shape_time_minutes(df):
    """Return a numeric match-minute Series using the best available clock."""
    import pandas as pd

    candidates = [
        "expandedMinute",
        "timeMin",
        "minute",
        "matchMinute",
    ]

    for column in candidates:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.notna().any():
                if column == "timeMin" and "timeSec" in df.columns:
                    seconds = pd.to_numeric(
                        df["timeSec"],
                        errors="coerce",
                    ).fillna(0)
                    values = values + seconds / 60.0
                return values

    return pd.Series(
        float("nan"),
        index=df.index,
        dtype=float,
    )


def _defensive_shape_period_mask(df, period):
    import pandas as pd

    period = (period or "full").lower()
    minutes = _defensive_shape_time_minutes(df)

    if period == "1h":
        if "periodId" in df.columns:
            period_id = pd.to_numeric(
                df["periodId"],
                errors="coerce",
            )
            return period_id.eq(1)
        return minutes.lt(45.0)

    if period == "2h":
        if "periodId" in df.columns:
            period_id = pd.to_numeric(
                df["periodId"],
                errors="coerce",
            )
            return period_id.eq(2)
        return minutes.ge(45.0)

    return pd.Series(
        True,
        index=df.index,
        dtype=bool,
    )


def _defensive_shape_is_goalkeeper(df):
    import pandas as pd

    mask = pd.Series(
        False,
        index=df.index,
        dtype=bool,
    )

    for column in (
        "position",
        "positionName",
        "Mapped Position",
        "mapped_position",
        "role",
    ):
        if column not in df.columns:
            continue

        values = (
            df[column]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        mask = mask | values.isin(
            {
                "gk",
                "goalkeeper",
                "keeper",
                "goal keeper",
            }
        )

    return mask


def _defensive_shape_convex_hull(points):
    """Monotonic-chain convex hull; returns ordered (x, y) tuples."""
    unique = sorted(
        {
            (float(x), float(y))
            for x, y in points
            if x is not None and y is not None
        }
    )

    if len(unique) <= 2:
        return unique

    def cross(origin, a, b):
        return (
            (a[0] - origin[0])
            * (b[1] - origin[1])
            - (a[1] - origin[1])
            * (b[0] - origin[0])
        )

    lower = []
    for point in unique:
        while (
            len(lower) >= 2
            and cross(
                lower[-2],
                lower[-1],
                point,
            )
            <= 0
        ):
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(unique):
        while (
            len(upper) >= 2
            and cross(
                upper[-2],
                upper[-1],
                point,
            )
            <= 0
        ):
            upper.pop()
        upper.append(point)

    return lower[:-1] + upper[:-1]


def _defensive_shape_polygon_area_m2(hull):
    if len(hull) < 3:
        return 0.0

    area_coordinate_units = 0.0

    for index, (x1, y1) in enumerate(hull):
        x2, y2 = hull[
            (index + 1) % len(hull)
        ]
        area_coordinate_units += (
            x1 * y2 - x2 * y1
        )

    area_coordinate_units = (
        abs(area_coordinate_units)
        / 2.0
    )

    return (
        area_coordinate_units
        * (
            DEFENSIVE_SHAPE_PITCH_LENGTH_M
            / 100.0
        )
        * (
            DEFENSIVE_SHAPE_PITCH_WIDTH_M
            / 100.0
        )
    )


def _defensive_shape_snapshot(
    player_locations,
    window_start,
    window_end,
):
    points = list(
        zip(
            player_locations["median_x"],
            player_locations["median_y"],
        )
    )
    hull = _defensive_shape_convex_hull(
        points
    )

    block_height_m = (
        float(
            player_locations[
                "median_x"
            ].median()
        )
        * DEFENSIVE_SHAPE_PITCH_LENGTH_M
        / 100.0
    )

    width_m = (
        float(
            player_locations[
                "median_y"
            ].max()
            - player_locations[
                "median_y"
            ].min()
        )
        * DEFENSIVE_SHAPE_PITCH_WIDTH_M
        / 100.0
    )

    footprint_m2 = (
        _defensive_shape_polygon_area_m2(
            hull
        )
    )

    pitch_area_m2 = (
        DEFENSIVE_SHAPE_PITCH_LENGTH_M
        * DEFENSIVE_SHAPE_PITCH_WIDTH_M
    )

    centroid_x = float(
        player_locations[
            "median_x"
        ].median()
    )
    centroid_y = float(
        player_locations[
            "median_y"
        ].median()
    )

    dx_m = (
        player_locations[
            "median_x"
        ].astype(float)
        - centroid_x
    ) * (
        DEFENSIVE_SHAPE_PITCH_LENGTH_M
        / 100.0
    )

    dy_m = (
        player_locations[
            "median_y"
        ].astype(float)
        - centroid_y
    ) * (
        DEFENSIVE_SHAPE_PITCH_WIDTH_M
        / 100.0
    )

    compactness_m = float(
        (
            (
                dx_m.pow(2)
                + dy_m.pow(2)
            ).mean()
        )
        ** 0.5
    )

    return {
        "window_start":
            float(window_start),
        "window_end":
            float(window_end),
        "player_count":
            int(len(player_locations)),
        "block_height_m":
            block_height_m,
        "width_m":
            width_m,
        "footprint_m2":
            footprint_m2,
        "compactness_m":
            compactness_m,
        "centroid_x":
            centroid_x,
        "centroid_y":
            centroid_y,
        "player_locations":
            player_locations.copy(),
        "hull":
            hull,
    }


def _defensive_shape_lineup_change_minutes(
    df,
    team_name,
):
    """Return team substitution minutes used to reject mixed-lineup windows."""
    import pandas as pd

    if (
        df is None
        or df.empty
        or "team_name"
        not in df.columns
    ):
        return []

    team_df = df.loc[
        df["team_name"].eq(
            team_name
        )
    ].copy()

    if team_df.empty:
        return []

    mask = pd.Series(
        False,
        index=team_df.index,
        dtype=bool,
    )

    if "typeId" in team_df.columns:
        type_ids = pd.to_numeric(
            team_df["typeId"],
            errors="coerce",
        )
        mask = mask | type_ids.isin(
            [18, 19]
        )

    if "type_name" in team_df.columns:
        names = (
            team_df["type_name"]
            .fillna("")
            .astype(str)
            .str.lower()
        )

        mask = mask | names.str.contains(
            "substitution",
            regex=False,
        )
        mask = mask | names.isin(
            {
                "player off",
                "player on",
            }
        )

    if not mask.any():
        return []

    minutes = (
        _defensive_shape_time_minutes(
            team_df.loc[mask]
        )
        .dropna()
        .astype(float)
        .tolist()
    )

    return sorted(
        set(minutes)
    )


def build_defensive_shape_profile(
    df_processed,
    team_name,
    period="full",
    window_minutes=
        DEFENSIVE_SHAPE_WINDOW_MINUTES,
    min_outfield_players=
        DEFENSIVE_SHAPE_MIN_OUTFIELD_PLAYERS,
):
    """
    Build an event-derived defensive shape profile.

    Density uses all defensive actions in the selected period.

    Shape never creates one whole-match hull. It creates player-location
    snapshots inside stable-lineup match-time windows, rejects windows that
    contain substitutions, requires at least seven outfield contributors and
    displays one real representative window closest to the period median
    structure.
    """
    import numpy as np
    import pandas as pd

    empty = {
        "team_name": team_name,
        "period": period or "full",
        "actions": pd.DataFrame(),
        "action_count": 0,
        "snapshot_count": 0,
        "block_height_m": None,
        "width_m": None,
        "compactness_m": None,
        "footprint_m2": None,
        "representative": None,
        "window_minutes":
            int(window_minutes),
    }

    if (
        df_processed is None
        or df_processed.empty
    ):
        return empty

    required = {
        "team_name",
        "type_name",
        "x",
        "y",
    }
    if not required.issubset(
        df_processed.columns
    ):
        return empty

    # Keep timing / period / positional metadata on the source frame.
    #
    # get_defensive_actions intentionally returns a compact projection with
    # event, player and location columns only. Defensive Shape also needs the
    # match clock to build coherent time windows, so filter the source first
    # and then reattach the derived clock by preserved original index.
    source_df = df_processed.copy()

    lineup_change_minutes = (
        _defensive_shape_lineup_change_minutes(
            source_df,
            team_name,
        )
    )

    source_df[
        "__match_minute"
    ] = (
        _defensive_shape_time_minutes(
            source_df
        )
    )

    period_mask = (
        _defensive_shape_period_mask(
            source_df,
            period,
        )
    )
    source_df = source_df.loc[
        period_mask
    ].copy()

    source_df = source_df.loc[
        ~_defensive_shape_is_goalkeeper(
            source_df
        )
    ].copy()

    actions = get_defensive_actions(
        source_df
    )

    if (
        actions is None
        or actions.empty
    ):
        return empty

    actions = actions.loc[
        actions["team_name"].eq(
            team_name
        )
    ].copy()

    if actions.empty:
        return empty

    actions[
        "__match_minute"
    ] = source_df.loc[
        actions.index,
        "__match_minute",
    ].to_numpy()

    actions = actions.loc[
        actions["x"].notna()
        & actions["y"].notna()
        & actions[
            "__match_minute"
        ].notna()
    ].copy()

    if actions.empty:
        return empty

    player_column = next(
        (
            candidate
            for candidate in (
                "playerName",
                "player_name",
                "playerId",
                "player_id",
            )
            if candidate
            in actions.columns
        ),
        None,
    )

    result = {
        **empty,
        "actions": actions,
        "action_count":
            int(len(actions)),
    }

    if player_column is None:
        return result

    actions["__window_start"] = (
        np.floor(
            actions[
                "__match_minute"
            ].astype(float)
            / float(window_minutes)
        )
        * float(window_minutes)
    )

    snapshots = []

    for (
        window_start,
        window_df,
    ) in actions.groupby(
        "__window_start"
    ):
        window_start = float(
            window_start
        )
        window_end = (
            window_start
            + float(
                window_minutes
            )
        )

        contains_lineup_change = any(
            (
                minute
                >= window_start
                and minute
                < window_end
            )
            for minute
            in lineup_change_minutes
        )

        if contains_lineup_change:
            continue

        aggregations = {
            "median_x":
                ("x", "median"),
            "median_y":
                ("y", "median"),
            "action_count":
                ("x", "size"),
        }

        jersey_column = next(
            (
                candidate
                for candidate in (
                    "Mapped Jersey Number",
                    "jersey_number",
                    "shirtNumber",
                )
                if candidate
                in window_df.columns
            ),
            None,
        )

        if jersey_column is not None:
            aggregations[
                "jersey_number"
            ] = (
                jersey_column,
                "first",
            )

        player_locations = (
            window_df
            .groupby(
                player_column,
                dropna=True,
            )
            .agg(**aggregations)
            .reset_index()
            .rename(
                columns={
                    player_column:
                        "player_name"
                }
            )
        )

        player_locations = (
            player_locations.loc[
                player_locations[
                    "median_x"
                ].notna()
                & player_locations[
                    "median_y"
                ].notna()
            ].copy()
        )

        if (
            len(player_locations)
            < int(
                min_outfield_players
            )
        ):
            continue

        snapshots.append(
            _defensive_shape_snapshot(
                player_locations,
                window_start=
                    window_start,
                window_end=
                    window_end,
            )
        )

    result[
        "snapshot_count"
    ] = int(len(snapshots))

    if not snapshots:
        return result

    block_values = np.array(
        [
            snapshot[
                "block_height_m"
            ]
            for snapshot
            in snapshots
        ],
        dtype=float,
    )
    width_values = np.array(
        [
            snapshot["width_m"]
            for snapshot
            in snapshots
        ],
        dtype=float,
    )
    compact_values = np.array(
        [
            snapshot[
                "compactness_m"
            ]
            for snapshot
            in snapshots
        ],
        dtype=float,
    )
    footprint_values = np.array(
        [
            snapshot[
                "footprint_m2"
            ]
            for snapshot
            in snapshots
        ],
        dtype=float,
    )

    median_block = float(
        np.median(block_values)
    )
    median_width = float(
        np.median(width_values)
    )
    median_compact = float(
        np.median(compact_values)
    )
    median_footprint = float(
        np.median(footprint_values)
    )

    def distance(snapshot):
        block_scale = max(
            float(
                np.ptp(
                    block_values
                )
            ),
            1.0,
        )
        width_scale = max(
            float(
                np.ptp(
                    width_values
                )
            ),
            1.0,
        )
        compact_scale = max(
            float(
                np.ptp(
                    compact_values
                )
            ),
            1.0,
        )

        return (
            abs(
                snapshot[
                    "block_height_m"
                ]
                - median_block
            )
            / block_scale
            + abs(
                snapshot["width_m"]
                - median_width
            )
            / width_scale
            + abs(
                snapshot[
                    "compactness_m"
                ]
                - median_compact
            )
            / compact_scale
        )

    representative = min(
        snapshots,
        key=distance,
    )

    result.update(
        {
            "block_height_m":
                median_block,
            "width_m":
                median_width,
            "compactness_m":
                median_compact,
            "footprint_m2":
                median_footprint,
            "representative":
                representative,
            "snapshots":
                snapshots,
        }
    )

    return result

