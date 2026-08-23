# src/metrics/set_piece_metrics.py

import pandas as pd
from src.utils.derived_cache import cache_derived_result
import numpy as np
import dash_bootstrap_components as dbc
from dash import html
from src.utils.sequence_outcomes import apply_sequence_outcome_contract

# Adjust Opta typeIds if necessary
# Note: Freekick might be complex (Pass, Shot, etc.). We'll focus on the 'Pass' part for deliveries.
# We assume Throw-ins are also captured under a specific type or qualifier. For now, let's use a placeholder.
CORNER_AWARDED_ID = 6  # The event that awards the corner
PASS_ID = 1
THROW_IN_ID = 1008  # Placeholder, adjust if you have a real ID
FREE_KICK_PASS_ID = 3 # This is often used for freekick passes, check your data

PENALTY_SHOT_TYPES = {
    'Goal',
    'Miss',
    'Attempt Saved',
    'Post',
}


@cache_derived_result("penalty_restart_sequences")
def extract_penalty_set_piece_sequences(
    df_processed,
    team_name,
):
    """
    Build one self-contained set-piece sequence per penalty kick.

    Opta can separate the penalty-award event from the eventual kick by
    several minutes (VAR, cards, treatment, goalkeeper preparation, etc.).
    Penalties therefore must be identified from the shot event itself rather
    than by trying to trace continuously from the foul that awarded them.
    """
    if df_processed is None or df_processed.empty:
        return []

    required_columns = {
        'team_name',
        'type_name',
        'Penalty',
    }
    if not required_columns.issubset(df_processed.columns):
        return []

    penalty_mask = (
        (df_processed['team_name'] == team_name)
        & df_processed['type_name'].isin(PENALTY_SHOT_TYPES)
        & df_processed['Penalty'].isin([1, '1', True])
    )

    penalty_shots = df_processed.loc[penalty_mask].copy()
    if penalty_shots.empty:
        return []

    sort_columns = [
        column
        for column in (
            'periodId',
            'timeMin',
            'timeSec',
            'eventId',
            'id',
        )
        if column in penalty_shots.columns
    ]
    if sort_columns:
        penalty_shots = penalty_shots.sort_values(
            sort_columns,
            kind='stable',
        )

    outcome_by_type = {
        'Goal': 'Penalty Goal',
        'Attempt Saved': 'Penalty Saved',
        'Miss': 'Penalty Missed',
        'Post': 'Penalty Missed',
    }

    sequences = []

    for row_number, (_, penalty_shot) in enumerate(
        penalty_shots.iterrows(),
        start=1,
    ):
        event_id = penalty_shot.get('id')
        if pd.isna(event_id):
            event_id = penalty_shot.get('eventId')
        if pd.isna(event_id):
            event_id = row_number

        sequence_id = f"penalty-{event_id}"
        event_data = penalty_shot.to_dict()
        event_data.update({
            'trigger_sequence_id': sequence_id,
            'trigger_zone': 'Attacking Third',
            'triggering_trigger_Opta_id': event_id,
            'timeMin_at_trigger': penalty_shot.get('timeMin'),
            'timeSec_at_trigger': penalty_shot.get('timeSec'),
            'type_of_initial_trigger': 'Penalty',
            'buildup_pass_count': 0,
            'sequence_outcome_type': outcome_by_type.get(
                penalty_shot.get('type_name'),
                'Penalty Missed',
            ),
        })

        sequence_df = pd.DataFrame([event_data])
        sequence_df = apply_sequence_outcome_contract(
            sequence_df,
            viewpoint='attacking',
            legacy_outcome=event_data['sequence_outcome_type'],
        )
        sequences.append(sequence_df)

    return sequences

def analyze_offensive_set_pieces(df_processed, team_name):
    """
    Identifies and analyzes all offensive set pieces for a given team.
    VERSIONE CORRETTA: Gestisce le colonne potenzialmente mancanti in modo robusto.
    """
    
    # --- Step 1: Identify all relevant set piece events ---
    
    # Filtro base
    team_passes = df_processed[(df_processed['team_name'] == team_name) & (df_processed['typeId'] == 1)].copy()
    offensive_half_passes = team_passes[team_passes['x'] > 50]

    # Lista per contenere i DataFrame dei vari tipi di calci piazzati
    set_piece_dfs = []

    # A. Corners
    if 'Corner taken' in offensive_half_passes.columns:
        corners = offensive_half_passes[offensive_half_passes['Corner taken'] == 1].copy()
        corners['Set Piece Type'] = 'Corner'
        set_piece_dfs.append(corners)

    # B. Free Kicks
    if 'Freekick taken' in offensive_half_passes.columns:
        free_kicks = offensive_half_passes[offensive_half_passes['Freekick taken'] == 1].copy()
        free_kicks['Set Piece Type'] = 'Free Kick'
        set_piece_dfs.append(free_kicks)

    # C. Offensive Throw-ins
    # Controlliamo sia typeId che type_name per sicurezza
    throw_in_filter = (df_processed['team_name'] == team_name) & (df_processed['x'] > 50)
    if 'type_name' in df_processed.columns and 'Throw-in' in df_processed['type_name'].unique():
        throw_in_filter &= (df_processed['type_name'] == 'Throw-in')
    # Aggiungi qui un eventuale controllo su typeId se hai un ID specifico per le rimesse
    # elif 'typeId' in df_processed.columns:
    #     throw_in_filter &= (df_processed['typeId'] == THROW_IN_ID)
    
    throw_ins = df_processed[throw_in_filter].copy()
    if not throw_ins.empty:
        throw_ins['Set Piece Type'] = 'Throw-in'
        set_piece_dfs.append(throw_ins)

    # Controlla se abbiamo trovato dei calci piazzati prima di continuare
    if not set_piece_dfs:
        return pd.DataFrame()

    df_set_pieces = pd.concat(set_piece_dfs, ignore_index=True)
    if df_set_pieces.empty:
        return pd.DataFrame()

    # --- Step 2: Enrich the data ---
    
    analyzed_data = []
    df_sorted = df_processed.sort_values('eventId').reset_index()

    for _, sp_event in df_set_pieces.iterrows():
        
        # A. Delivery Type
        delivery_type = "Cross" if sp_event.get('cross') == 1 else "Short Pass"
        if sp_event['Set Piece Type'] == 'Throw-in':
            delivery_type = 'Throw-in'

        # B. Player Foot and Swing Type
        player_foot = 'Unknown'
        if 'Right footed' in sp_event and sp_event['Right footed'] == 1:
            player_foot = 'Right'
        elif 'Left footed' in sp_event and sp_event['Left footed'] == 1:
            player_foot = 'Left'
        
        swing = "N/A"
        if delivery_type == "Cross":
            if 'In-swinger' in sp_event and sp_event['In-swinger'] == 1:
                swing = 'In-swinger'
            elif 'Out-swinger' in sp_event and sp_event['Out-swinger'] == 1:
                swing = 'Out-swinger'
            # Fallback (logica identica a prima)
            elif sp_event['Set Piece Type'] == 'Corner':
                if sp_event['y'] > 50 and player_foot == 'Right': swing = "In-swinger"
                elif sp_event['y'] > 50 and player_foot == 'Left': swing = "Out-swinger"
                elif sp_event['y'] < 50 and player_foot == 'Right': swing = "Out-swinger"
                elif sp_event['y'] < 50 and player_foot == 'Left': swing = "In-swinger"

        # C. Sequence Outcome
        outcome = "Possession Retained"
        start_index_list = df_sorted.index[df_sorted['eventId'] == sp_event['eventId']].tolist()
        if not start_index_list:
            continue # Se non troviamo l'evento, saltiamo
        
        start_index = start_index_list[0]
        sequence = df_sorted.iloc[start_index : start_index + 8]

        for i, event in sequence.iloc[1:].iterrows():
            if event['team_name'] != sp_event['team_name']:
                outcome = "Possession Lost"
                break
            if event['type_name'] in ['Shot', 'Goal', 'Attempt Saved', 'Post']:
                outcome = "Shot"
                if event['type_name'] == 'Goal':
                    outcome = "Goal"
                break
        
        analyzed_data.append({
            'Minute': sp_event.get('timeMin'),
            'Player': sp_event.get('playerName'),
            'Action Type': sp_event.get('Set Piece Type'),
            'Delivery': delivery_type,
            'Foot': player_foot,
            'Swing': swing,
            'Outcome': outcome,
            'x_start': sp_event.get('x'), 'y_start': sp_event.get('y'),
            'x_end': sp_event.get('end_x'), 'y_end': sp_event.get('end_y'),
        })

    return pd.DataFrame(analyzed_data)

def calculate_set_piece_stats(sequence_list):
    """
    Calcola le statistiche aggregate per una lista di sequenze di calci piazzati.
    """
    if not sequence_list:
        return pd.DataFrame()

    data = []
    for seq in sequence_list:
        if seq.empty:
            continue
        
        first_event = seq.iloc[0]
        
        # Determina il tipo di calcio piazzato dal trigger
        trigger_type = first_event.get('type_of_initial_trigger', 'Unknown')
        if trigger_type == 'Foul':
            action_type = 'Free Kick'
        elif trigger_type == 'Corner Awarded':
            action_type = 'Corner'
        elif trigger_type in ('Goal kick', 'Goal kick taken'):
            action_type = 'Goal Kick'
        elif trigger_type in (
            'Corner',
            'Free Kick',
            'Throw-in',
            'Goal Kick',
            'Penalty',
        ):
            action_type = trigger_type
        else:
            action_type = trigger_type

        # Delivery Type
        is_cross = first_event.get('cross') == 1
        delivery = (
            'Penalty kick'
            if action_type == 'Penalty'
            else ('Cross' if is_cross else 'Short Pass')
        )

        # Foot and Swing
        player_foot = 'Right' if first_event.get('Right footed') == 1 else ('Left' if first_event.get('Left footed') == 1 else 'Unknown')
        swing = 'N/A'
        if is_cross:
            if first_event.get('In-swinger') == 1: swing = 'In-swinger'
            elif first_event.get('Out-swinger') == 1: swing = 'Out-swinger'
            elif first_event.get('Straight') == 1: swing = 'Straight'

        data.append({
            'Action Type': action_type,
            'Delivery': delivery,
            'Foot': player_foot,
            'Swing': swing,
            'Outcome': seq.iloc[-1]['sequence_outcome_type']
        })

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    
    # Crea una tabella pivot per riassumere i dati
    summary_table = pd.pivot_table(
        df,
        index=['Action Type', 'Delivery', 'Swing'],
        columns='Outcome',
        aggfunc='size', # Conta le occorrenze
        fill_value=0
    )
    
    summary_table['Total'] = summary_table.sum(axis=1)
    return summary_table.reset_index()


def get_cross_destination_zone(start_y, end_x, end_y):
    """
    Categorizes the destination of a cross, considering the starting side.
    """
    if not all(pd.notna([start_y, end_x, end_y])):
        return "Unknown"

    # --- Area di Porta (6-yard box) ---
    # Definiamo i limiti dell'area di porta
    six_yard_box_x_start = 94.5
    six_yard_box_y_min = 36.8
    six_yard_box_y_max = 63.2

    if end_x >= six_yard_box_x_start and (six_yard_box_y_min <= end_y <= six_yard_box_y_max):
        # Il cross arriva nell'area piccola
        
        # Cross da DESTRA (y bassa)
        if start_y < 45: # Usiamo 45 come soglia per essere sicuri
            if end_y < 50:
                return "Near Post"
            else:
                return "Far Post"
        
        # Cross da SINISTRA (y alta)
        elif start_y > 55: # Usiamo 55 come soglia
            if end_y > 50:
                return "Near Post"
            else:
                return "Far Post"
        
        # Cross dal CENTRO
        else:
            return "Center of 6-Yard Box"

    # --- Area di Rigore (Penalty Area), ma fuori dall'area di porta ---
    penalty_area_x_start = 83.5
    if end_x >= penalty_area_x_start:
        return "Center Box"  # Il cross arriva nell'area di rigore ma non nell'area piccola

    # --- Fuori dall'area ---
    else:
        return "Edge of Box / Other"

def analyze_and_summarize_set_pieces(sequence_list):
    """
    Analyzes set piece sequences.
    VERSIONE FINALE: Gestisce corner corti e cerca l'evento di cross corretto.
    """
    if not sequence_list:
        return pd.DataFrame(), {}

    detailed_data = []
    for seq in sequence_list:
        if seq.empty: continue
        
        trigger_event = seq.iloc[0]
        trigger_type_raw = trigger_event.get('type_of_initial_trigger', 'Unknown')
        
        action_map = {
            'Foul': 'Free Kick',
            'Corner Awarded': 'Corner',
            'Goal kick': 'Goal Kick',
            'Goal kick taken': 'Goal Kick',
            'Corner': 'Corner',
            'Free Kick': 'Free Kick',
            'Throw-in': 'Throw-in',
            'Goal Kick': 'Goal Kick',
            'Penalty': 'Penalty',
        }
        action_type = action_map.get(trigger_type_raw, trigger_type_raw)

        # --- LOGICA DI ESTRAZIONE DATI POTENZIATA ---

        if action_type == 'Penalty':
            # A penalty set piece can be a direct shot with no pass
            # delivery, so it must not be discarded by the pass-only
            # logic used for corners/free kicks/throw-ins.
            penalty_shots = seq[
                seq['type_name'].isin(
                    ['Goal', 'Miss', 'Attempt Saved', 'Post']
                )
            ]

            if 'Penalty' in seq.columns:
                flagged_penalties = penalty_shots[
                    penalty_shots['Penalty'].isin(
                        [1, '1', True]
                    )
                ]
            else:
                flagged_penalties = pd.DataFrame()

            if not flagged_penalties.empty:
                main_delivery_event = flagged_penalties.iloc[0]
            elif not penalty_shots.empty:
                main_delivery_event = penalty_shots.iloc[0]
            else:
                continue

            cross_event = None
            delivery = 'Penalty kick'

        elif (
            action_type == 'Free Kick'
            and trigger_event.get('type_name')
                in ['Goal', 'Miss', 'Attempt Saved', 'Post']
        ):
            main_delivery_event = trigger_event
            cross_event = None
            delivery = 'Direct shot'

        else:
            # L'evento di battuta iniziale (può essere un passaggio corto)
            initial_delivery = seq[seq['type_name'] == 'Pass'].iloc[0] if not seq[seq['type_name'] == 'Pass'].empty else None
            if initial_delivery is None: continue

            # L'evento di cross, se esiste nella sequenza
            cross_event = seq[seq['cross'] == 1].iloc[0] if not seq[seq['cross'] == 1].empty else None

            # Se c'è un cross, usiamo quello per le metriche di delivery. Altrimenti, usiamo la battuta iniziale.
            main_delivery_event = cross_event if cross_event is not None else initial_delivery

            # Determina il tipo di delivery
            is_short_corner = action_type == 'Corner' and cross_event is not None and initial_delivery['eventId'] != cross_event['eventId']
            if is_short_corner:
                delivery = "Short Corner + Cross"
            elif main_delivery_event.get('cross') == 1:
                delivery = "Direct Cross"
            else:
                delivery = "Short Pass"

            if action_type == 'Throw-in':
                delivery = 'Throw-in'
            elif action_type == 'Goal Kick':
                delivery = 'Goal kick'

        # Estrai le altre metriche dall'evento di cross/delivery principale
        player_name = main_delivery_event.get('playerName')
        side = "Center"
        if main_delivery_event['x'] > 50:
            if main_delivery_event['y'] > 67: side = "Left"
            elif main_delivery_event['y'] < 33: side = "Right"

        canonical_delivery = trigger_event.get(
            'restart_delivery_type'
        )
        if (
            canonical_delivery is not None
            and not pd.isna(canonical_delivery)
            and str(canonical_delivery).strip()
        ):
            delivery = str(canonical_delivery)

        player_foot = 'Unknown'
        if main_delivery_event.get('Right footed') == 1: player_foot = 'Right'
        elif main_delivery_event.get('Left footed') == 1: player_foot = 'Left'
        
        swing = 'N/A'
        if 'Cross' in delivery:
            if main_delivery_event.get('In-swinger') == 1: swing = 'In-swinger'
            elif main_delivery_event.get('Out-swinger') == 1: swing = 'Out-swinger'
            elif main_delivery_event.get('Straight') == 1: swing = 'Straight'

        destination = get_cross_destination_zone(main_delivery_event.get('y'), main_delivery_event.get('end_x'), main_delivery_event.get('end_y')) if 'Cross' in delivery else 'N/A'
        
        detailed_data.append({
            'sequence_id': trigger_event.get('trigger_sequence_id'),
            'playerName': player_name,
            'Action Type': action_type,
            'Side': side,
            'Delivery': delivery,
            'Swing': swing,
            'Foot': player_foot,
            'Destination': destination,
            'terminal_outcome': seq.iloc[-1].get('terminal_outcome', 'unknown'),
            'termination_reason': seq.iloc[-1].get('termination_reason', 'unknown'),
            'viewpoint': seq.iloc[-1].get('viewpoint', 'attacking'),
            'Outcome': trigger_event.get(
                'restart_execution_outcome',
                seq.iloc[-1].get('sequence_outcome_type', 'Unknown'),
            )
        })
        
    if not detailed_data:
        return pd.DataFrame(), {}
        
    df = pd.DataFrame(detailed_data)
    
    # Le statistiche aggregate ora saranno molto più accurate
    stats = {
        'total': len(df),
        'action_types': df['Action Type'].value_counts().to_dict(),
        'sides': df['Side'].value_counts().to_dict(),
        'deliveries': df['Delivery'].value_counts().to_dict(),
        'swings': df[df['Swing'] != 'N/A']['Swing'].value_counts().to_dict(),
        'feet': df[df['Foot'] != 'Unknown']['Foot'].value_counts().to_dict(),
        'destinations': df[df['Destination'] != 'N/A']['Destination'].value_counts().to_dict(),
        'outcomes': df['Outcome'].value_counts().to_dict(),
        'terminal_outcomes': df['terminal_outcome'].value_counts().to_dict(),
    }
    
    return df, stats


def create_set_piece_summary_cards(stats, active_filter=None):
    """Creates a full set of detailed, interactive summary cards for set pieces."""
    if not stats or stats.get("total", 0) == 0:
        return dbc.Alert("No restart data to display.", color="secondary")

    def create_card(title, data_dict, filter_type):
        items = []
        if not data_dict: return None
        data_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
        if filter_type == 'outcome':
            outcome_order = ['Penalty Goal', 'Goals', 'Penalty Saved', 'Penalty Missed', 'Shots', 'Big Chances', 'Lost Possessions', 'Foul']
            outcome_rank = {v: i for i, v in enumerate(outcome_order)}
            # Ri-ordina la lista 'data_items' basandosi sulla gerarchia definita
            data_items = sorted(data_items, key=lambda item: outcome_rank.get(item[0], 99))
        for value, count in data_items:
            active = active_filter is not None and str(active_filter.get(filter_type)) == str(value)
            items.append(dbc.ListGroupItem(
                [html.Div(value), dbc.Badge(f"{count}", color="light", className="ms-auto")],
                id={'type': 'sp-filter', 'filter_type': filter_type, 'value': value},
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            ))
        return dbc.Col(dbc.Card([dbc.CardHeader(title), dbc.ListGroup(items, flush=True)]), md=4)

    active_filter = active_filter or {}
    
    # Creiamo una lista di tutte le card che vogliamo visualizzare
    cards = [
        create_card("Action Type", stats.get('action_types'), 'action'),
        create_card("Action Side", stats.get('sides'), 'side'),
        create_card("Delivery Type", stats.get('deliveries'), 'delivery'),
        create_card("Cross Swing", stats.get('swings'), 'swing'),
        create_card("Taker Foot", stats.get('feet'), 'foot'),
        create_card("Corner Cross Destination", stats.get('destinations'), 'destination'),
        create_card("Execution Outcome", stats.get('outcomes'), 'outcome')
    ]
    
    # Rimuoviamo le card che non hanno dati (es. se non ci sono cross, non mostrare la card "Swing")
    valid_cards = [card for card in cards if card is not None]
    
    # Dividiamo le card in righe di 3 per una migliore visualizzazione
    rows = [dbc.Row(valid_cards[i:i+3], className="mb-3") for i in range(0, len(valid_cards), 3)]
    
    return html.Div(rows)


def create_takers_card(df_analyzed, player_jersey_map, active_filter=None):
    """
    Crea una card che mostra i giocatori che hanno battuto i calci piazzati.
    VERSIONE CORRETTA E ROBUSTA.
    """
    if df_analyzed.empty:
        return None

    # REL-09B: taker patterns are meaningful for corners/free kicks,
    # not for a mixed list of throw-ins and goal kicks.
    if 'Action Type' in df_analyzed.columns:
        df_analyzed = df_analyzed[
            df_analyzed['Action Type'].isin(['Corner', 'Free Kick'])
        ].copy()

    if df_analyzed.empty:
        return None

    # Conta quanti calci piazzati ha battuto ogni giocatore e prende il piede
    # (assumendo che un giocatore usi prevalentemente lo stesso piede per i calci piazzati)
    takers_summary = df_analyzed.groupby('playerName').agg(
        Count=('playerName', 'size'),
        Foot=('Foot', lambda x: x.mode()[0] if not x.mode().empty else 'Unknown') # Usa il piede più frequente
    ).sort_values('Count', ascending=False).reset_index()

    active_filter = active_filter or {}
    card_items = []
    # Mostra al massimo i primi 5-6 tiratori per non affollare la card
    for _, row in takers_summary.head(6).iterrows():
        player_name = row['playerName']
        jersey_number = player_jersey_map.get(player_name)
        jersey_display = f"#{int(jersey_number)}" if pd.notna(jersey_number) else "#?"

        active = active_filter.get('taker') == player_name

        card_items.append(
            dbc.ListGroupItem(
                [
                    html.Div([
                        html.Span(f"{jersey_display}", className="fw-bold me-2", style={'minWidth': '35px', 'display': 'inline-block'}),
                        html.Span(player_name)
                    ]),
                    dbc.Badge(f"{row['Foot']} | ✖ {row['Count']}", color="light", className="ms-auto", pill=True)
                ],
                id={'type': 'sp-filter', 'filter_type': 'taker', 'value': player_name}, 
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            )
        )
    
    if not card_items:
        return None
    
    return dbc.Col(dbc.Card([dbc.CardHeader("Set-piece Takers"), dbc.ListGroup(card_items, flush=True)]), md=4)
