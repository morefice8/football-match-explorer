# src/metrics/defensive_metrics.py
import pandas as pd
import numpy as np

# --- Define Default Defensive Action Types ---
DEFAULT_DEFENSIVE_TYPES = [
    'Ball recovery', 'Blocked pass', 'Challenge',
    'Clearance', 'Error', 'Foul', 'Interception', 'Tackle'
]

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
    print("Filtering for defensive actions...")

    # --- Determine action types to use ---
    if defensive_action_types is None:
        types_to_use = DEFAULT_DEFENSIVE_TYPES
        print(f"  Using default defensive types: {types_to_use}")
    else:
        types_to_use = defensive_action_types
        print(f"  Using specified defensive types: {types_to_use}")

    # --- Check required columns ---
    required_cols = ['team_name', 'type_name', 'x', 'y'] # Base requirements
    if not all(col in df_processed.columns for col in required_cols):
        missing = set(required_cols) - set(df_processed.columns)
        print(f"Error: Missing base required columns for defensive action analysis: {missing}")
        return pd.DataFrame()
    # Check if 'type_name' column actually exists before filtering
    if 'type_name' not in df_processed.columns:
         print(f"Error: 'type_name' column missing, cannot filter actions.")
         return pd.DataFrame()


    # --- Apply Filters ---
    # Base filter for standard defensive types provided in the list
    # Handle potential NaNs in type_name just in case, though unlikely
    defensive_filter = df_processed['type_name'].fillna('').isin(types_to_use)

    # Specific filter for Aerial duels in defensive third (always included for now)
    # Ensure 'x' exists before applying this filter
    if 'x' in df_processed.columns:
        aerial_filter = (df_processed['type_name'] == 'Aerial') & (df_processed['x'].fillna(101) <= 33.33)
        print("  Including 'Aerial' actions occurring in defensive third (x <= 33.33).")
        # Combine filters: must be one of the specified types OR a defensive third aerial
        final_filter = defensive_filter | aerial_filter
    else:
        print("  Warning: 'x' column missing, cannot apply Aerial location filter.")
        final_filter = defensive_filter # Only use standard types if 'x' is missing

    # --- Select relevant columns ---
    # Ensure we only select columns that actually exist after preprocessing
    relevant_cols = ["id", "x", "y", "team_name", "playerName",
                     "Mapped Jersey Number", "type_name", "outcome"]
    existing_cols = [col for col in relevant_cols if col in df_processed.columns]

    df_defensive_actions = df_processed.loc[final_filter, existing_cols].copy()
    print(f"Found {len(df_defensive_actions)} defensive actions matching criteria.")
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
    print(f"Calculating aggregated defensive metrics for {team_name}...")
    team_actions_df = df_defensive_actions[df_defensive_actions["team_name"] == team_name].copy()

    if team_actions_df.empty:
        print(f"Warning: No defensive actions found for team {team_name}.")
        return pd.DataFrame()

    # Check for required columns for aggregation
    required_agg_cols = ['playerName', 'x', 'y', 'id', 'Mapped Jersey Number']
    if not all(col in team_actions_df.columns for col in required_agg_cols):
        missing = set(required_agg_cols) - set(team_actions_df.columns)
        print(f"Error: Missing columns needed for aggregation: {missing}")
        return pd.DataFrame()

    # Group by player and calculate median location and count
    player_agg = team_actions_df.groupby('playerName').agg(
        median_x=('x', 'median'),
        median_y=('y', 'median'),
        action_count=('id', 'count'), # Count defensive actions
        jersey_number=('Mapped Jersey Number', 'first') # Get jersey number
    ).reset_index() # Make playerName a column again

    print(f"Calculated metrics for {len(player_agg)} players.")
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
    print(f"Calculating PPDA for {team_of_interest} (vs {opponent_team})...")

    # --- Input Validation ---
    required_cols = ['team_name', event_id_col, 'x']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        print(f"Error [PPDA]: DataFrame missing required columns: {missing}")
        return None
    if team_of_interest not in df['team_name'].unique():
        print(f"Error [PPDA]: Team '{team_of_interest}' not found.")
        return None
    if opponent_team not in df['team_name'].unique():
        print(f"Error [PPDA]: Opponent '{opponent_team}' not found.")
        return None
    # Ensure coordinate column is numeric
    if not pd.api.types.is_numeric_dtype(df['x']):
         print(f"Error [PPDA]: Column 'x' must be numeric.")
         return None
     # Ensure event id column is numeric (or convertible)
    try:
        df[event_id_col] = pd.to_numeric(df[event_id_col], errors='coerce')
        if df[event_id_col].isnull().any():
             print(f"Warning [PPDA]: Column '{event_id_col}' contains non-numeric values.")
             # Decide whether to proceed by dropping NaNs or return None
             # df = df.dropna(subset=[event_id_col]) # Option to proceed
             # return None # Option to stop
    except Exception as e:
         print(f"Error [PPDA]: Could not process event ID column '{event_id_col}': {e}")
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
        print(f"  {team_of_interest}: 0 defensive actions in zone (x >= {def_action_zone_thresh}). PPDA = inf")
    else:
        ppda = num_opponent_passes / num_team_def_actions
        print(f"  {opponent_team} Passes (x < {pass_zone_thresh}): {num_opponent_passes}")
        print(f"  {team_of_interest} Def Actions (x >= {def_action_zone_thresh}): {num_team_def_actions}")
        print(f"  Calculated PPDA: {ppda:.2f}")

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


def calculate_ppda_data(df_processed, team_name, opponent_name, zone_threshold=40.0):
    """
    Calcola il PPDA e restituisce i dati per la visualizzazione.
    - Tackle e Challenge sono in formato "successi/totale".
    - Interception, Blocked Pass e Foul sono mostrati come conteggio totale.
    """
    # Azioni che hanno un esito variabile (successo/fallimento)
    ACTIONS_WITH_OUTCOME = ['Tackle', 'Challenge']
    # Azioni che per natura sono "riuscite" o di cui contiamo solo il totale
    ACTIONS_AS_COUNT = ['Interception', 'Blocked Pass', 'Foul']
    
    # Filtro per le azioni con esito
    base_actions_filter = (
        (df_processed['team_name'] == team_name) &
        (df_processed['type_name'].isin(ACTIONS_WITH_OUTCOME)) &
        (df_processed['x'] >= zone_threshold)
    )
    
    # Filtro per le azioni da contare
    count_actions_filter = (
        (df_processed['team_name'] == team_name) &
        (df_processed['type_name'].isin(ACTIONS_AS_COUNT)) &
        # Per i falli, contiamo solo quelli commessi (Unsuccessful)
        (~((df_processed['type_name'] == 'Foul') & (df_processed['outcome'] == 'Successful'))) &
        (df_processed['x'] >= zone_threshold)
    )
    
    defensive_actions_filter = base_actions_filter | count_actions_filter
    df_defensive_actions = df_processed[defensive_actions_filter].copy()
    num_defensive_actions = len(df_defensive_actions)

    # ... (logica per opponent_passes e ppda_value invariata) ...
    opponent_passes_filter = (
        (df_processed['team_name'] == opponent_name) &
        (df_processed['type_name'] == 'Pass') &
        (df_processed['x'] >= zone_threshold)
    )
    df_opponent_passes = df_processed[opponent_passes_filter]
    num_opponent_passes = len(df_opponent_passes)
    ppda_value = (num_opponent_passes / num_defensive_actions) if num_defensive_actions > 0 else float('inf')

    df_player_stats = pd.DataFrame()
    if not df_defensive_actions.empty:
        df_defensive_actions['successful_action'] = (df_defensive_actions['outcome'] == 'Successful').astype(int)
        
        pivot = df_defensive_actions.pivot_table(
            index=['playerName', 'Mapped Jersey Number'],
            columns='type_name',
            values='successful_action',
            aggfunc=['sum', 'count'],
            fill_value=0
        )
        pivot.columns = [f'{level1}_{level0}' for level0, level1 in pivot.columns]
        
        # --- START: LOGICA DI FORMATTAZIONE SEPARATA ---
        # Azioni con formato "success/total"
        for action_type in ACTIONS_WITH_OUTCOME:
            sum_col, count_col = f'{action_type}_sum', f'{action_type}_count'
            if sum_col in pivot.columns and count_col in pivot.columns:
                pivot[action_type] = pivot.apply(lambda row: f"{int(row[sum_col])}/{int(row[count_col])}", axis=1)
            else:
                pivot[action_type] = "0/0"
        
        # Azioni con formato "conteggio totale"
        for action_type in ACTIONS_AS_COUNT:
            count_col = f'{action_type}_count'
            if count_col in pivot.columns:
                pivot[action_type] = pivot[count_col]
            else:
                pivot[action_type] = 0
        # --- END: LOGICA DI FORMATTAZIONE SEPARATA ---

        pivot['Total_Actions'] = pivot.filter(regex='_count$').sum(axis=1)
        pivot['Successful_Actions'] = pivot.filter(regex='_sum$').sum(axis=1)
        pivot['Success Rate'] = (pivot['Successful_Actions'] / pivot['Total_Actions'] * 100).round(1).fillna(0)
        
        df_player_stats = pivot.reset_index()
        df_player_stats['Player'] = "#" + df_player_stats['Mapped Jersey Number'].astype(int).astype(str) + " - " + df_player_stats['playerName']
        
        df_player_stats = df_player_stats.rename(columns={
            "Total_Actions": "Tot", "Successful_Actions": "Succ", "Success Rate": "Succ %",
            "Tackle": "Tkl", "Interception": "Int", "Blocked Pass": "Blk", "Challenge": "Chl", "Foul": "Fls"
        })
        
        final_cols = ['Player', 'Tot', 'Succ', 'Succ %', 'Tkl', 'Int', 'Blk', 'Chl', 'Fls']
        final_cols_exist = [col for col in final_cols if col in df_player_stats.columns]
        
        df_player_stats = df_player_stats[final_cols_exist].sort_values(by='Tot', ascending=False)

    return ppda_value, df_defensive_actions, df_opponent_passes, df_player_stats