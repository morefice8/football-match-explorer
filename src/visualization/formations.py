# src/analysis/formation_analysis.py

import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch, FontManager
from src.utils import formation_layouts
import io
import base64
from dash import html
import dash_bootstrap_components as dbc

# --- 1. FUNZIONE PER PREPARARE I DATI ---

def _extract_player_positions(event):
    """Estrae {playerId: position} da un evento."""
    players_str = event.get('Involved')
    positions_str = event.get('Team player formation')
    if pd.isna(players_str) or pd.isna(positions_str): return {}
    player_ids = [pid.strip() for pid in str(players_str).split(',')]
    positions = [pos.strip() for pos in str(positions_str).split(',')]
    if len(player_ids) != len(positions): return {}
    return {pid: int(pos) for pid, pos in zip(player_ids, positions) if pos.isdigit() and int(pos) > 0}

# --- 1. NUOVA FUNZIONE PER LA TIMELINE UNIFICATA ---
# def create_unified_timeline(df_processed, home_id, away_id, player_data_map):
#     """
#     Versione 2: Arricchisce la timeline con score post-evento e descrizioni dettagliate.
#     """
#     event_types = {16: 'Goal', 17: 'Card', 18: 'Substitution', 40: 'Formation Change'}
#     key_events_df = df_processed[df_processed['typeId'].isin(event_types.keys())].sort_values(by=['timeMin', 'timeSec', 'eventId'])

#     timeline = []
#     processed_subs = set()

#     for _, event in key_events_df.iterrows():
#         time_min, event_type = event['timeMin'], event_types.get(event['typeId'])
        
#         # Calcola lo score DOPO questo evento
#         events_up_to_now = df_processed[df_processed['event_sequence_index'] <= event['event_sequence_index']]
#         goals = events_up_to_now[events_up_to_now['typeId'] == 16]
#         score = f"{(goals['contestantId'] == home_id).sum()} - {(goals['contestantId'] == away_id).sum()}"
        
#         # Costruisci descrizione e dati evento
#         if event_type == 'Substitution':
#             if event['eventId'] in processed_subs: continue
#             sub_on = df_processed[(df_processed['typeId'] == 19) & (df_processed['related_eventId'] == event['eventId'])].iloc[0]
#             processed_subs.add(sub_on['eventId'])
            
#             p_in_jersey = player_data_map.get(sub_on['playerId'], {}).get('jersey', '?')
#             p_out_jersey = player_data_map.get(event['playerId'], {}).get('jersey', '?')
            
#             description = html.Div([
#                 html.Strong(f"({score}) "),
#                 html.I(className="fas fa-arrows-alt-v me-2"),
#                 f"Sub {event['team_name']}: ",
#                 html.I(className="fas fa-arrow-up me-1 text-success"), f"{sub_on['playerName']} ({p_in_jersey})",
#                 html.I(className="fas fa-arrow-down ms-2 me-1 text-danger"), f"{event['playerName']} ({p_out_jersey})"
#             ])
#             # Aggiungiamo i dati di entrambi gli eventi per l'aggiornamento dello stato
#             event_data = {'sub_off': event.to_dict(), 'sub_on': sub_on.to_dict()}

#         elif event_type == 'Goal':
#             description = html.Div([
#                 html.Strong(f"({score}) "),
#                 html.I(className="fas fa-futbol me-2"),
#                 f"GOAL! {event['playerName']} ({event['team_name']})"
#             ])
#             event_data = event.to_dict()
            
#         elif event_type == 'Card':
#             card_type, card_color = ("Red", "red") if ('Red card' in event and pd.notna(event['Red card'])) or ('Second yellow' in event and pd.notna(event['Second yellow'])) else ("Yellow", "yellow")
#             description = html.Div([
#                 html.Strong(f"({score}) "),
#                 html.I(className="fas fa-square me-2", style={'color': card_color}),
#                 f"{card_type} Card for {event['playerName']} ({event['team_name']})"
#             ])
#             event_data = event.to_dict()

#         elif event_type == 'Formation Change':
#             description = html.Div([
#                  html.Strong(f"({score}) "),
#                  html.I(className="fas fa-sitemap me-2"),
#                  f"{event['team_name']} changes formation"
#             ])
#             event_data = event.to_dict()
#         else: continue
            
#         timeline.append({
#             'time_min': time_min,
#             'time_str': f"{time_min}'",
#             'team_id': event['contestantId'],
#             'typeId': event['typeId'],
#             'event_data': event_data, # Ora può contenere un dizionario con sub_on/sub_off
#             'description_component': description
#         })
#     return timeline
def create_unified_timeline(df_processed, home_id, away_id, player_data_map):
    """
    Versione 3: Gestisce correttamente gli autogol nel calcolo dello score e nella descrizione.
    """

    if 'Own goal' not in df_processed.columns:
        df_processed['Own goal'] = 0

    event_types = {16: 'Goal', 17: 'Card', 18: 'Substitution', 40: 'Formation Change'}
    key_events_df = df_processed[df_processed['typeId'].isin(event_types.keys())].sort_values(by=['timeMin', 'timeSec', 'eventId'])

    timeline = []
    processed_subs = set()

    for _, event in key_events_df.iterrows():
        time_min, event_type = event['timeMin'], event_types.get(event['typeId'])
        
        # --- SEZIONE 1: CALCOLO DELLO SCORE CORRETTO ---
        events_up_to_now = df_processed[df_processed['event_sequence_index'] <= event['event_sequence_index']]
        goals_df = events_up_to_now[events_up_to_now['typeId'] == 16]

        home_score = 0
        away_score = 0
        if not goals_df.empty:
            # Gol segnati dalla squadra di casa (non autogol)
            home_score += goals_df[(goals_df['contestantId'] == home_id) & (goals_df['Own goal'] != 1)].shape[0]
            # Autogol della squadra ospite (a favore della squadra di casa)
            home_score += goals_df[(goals_df['contestantId'] == away_id) & (goals_df['Own goal'] == 1)].shape[0]

            # Gol segnati dalla squadra ospite (non autogol)
            away_score += goals_df[(goals_df['contestantId'] == away_id) & (goals_df['Own goal'] != 1)].shape[0]
            # Autogol della squadra di casa (a favore della squadra ospite)
            away_score += goals_df[(goals_df['contestantId'] == home_id) & (goals_df['Own goal'] == 1)].shape[0]
        
        score = f"{home_score} - {away_score}"
        # --- FINE SEZIONE 1 ---
        
        # Costruisci descrizione e dati evento
        if event_type == 'Substitution':
            if event['eventId'] in processed_subs: continue
            sub_on_query = df_processed[(df_processed['typeId'] == 19) & (df_processed['related_eventId'] == event['eventId'])]
            if sub_on_query.empty: continue
            sub_on = sub_on_query.iloc[0]
            processed_subs.add(sub_on['eventId'])
            
            p_in_jersey = player_data_map.get(sub_on['playerId'], {}).get('jersey', '?')
            p_out_jersey = player_data_map.get(event['playerId'], {}).get('jersey', '?')
            
            description = html.Div([
                html.Strong(f"({score}) "),
                html.I(className="fas fa-arrows-alt-v me-2"),
                f"Sub {event['team_name']}: ",
                html.I(className="fas fa-arrow-up me-1 text-success"), f"{sub_on['playerName']} ({p_in_jersey})",
                html.I(className="fas fa-arrow-down ms-2 me-1 text-danger"), f"{event['playerName']} ({p_out_jersey})"
            ])
            event_data = {'sub_off': event.to_dict(), 'sub_on': sub_on.to_dict()}

        # --- SEZIONE 2: DESCRIZIONE DELL'EVENTO GOL CORRETTA ---
        elif event_type == 'Goal':
            # Controlla se è un autogol
            if event['Own goal'] == 1:
                description = html.Div([
                    html.Strong(f"({score}) "),
                    html.I(className="fas fa-futbol me-2", style={'color': 'red'}),
                    f"OWN GOAL! {event['playerName']} ({event['team_name']})"
                ])
            else:
                description = html.Div([
                    html.Strong(f"({score}) "),
                    html.I(className="fas fa-futbol me-2"),
                    f"GOAL! {event['playerName']} ({event['team_name']})"
                ])
            event_data = event.to_dict()
        # --- FINE SEZIONE 2 ---
            
        elif event_type == 'Card':
            card_type, card_color = ("Red", "red") if ('Red card' in event and pd.notna(event['Red card'])) or ('Second yellow' in event and pd.notna(event['Second yellow'])) else ("Yellow", "yellow")
            description = html.Div([
                html.Strong(f"({score}) "),
                html.I(className="fas fa-square me-2", style={'color': card_color}),
                f"{card_type} Card for {event['playerName']} ({event['team_name']})"
            ])
            event_data = event.to_dict()

        elif event_type == 'Formation Change':
            description = html.Div([
                 html.Strong(f"({score}) "),
                 html.I(className="fas fa-sitemap me-2"),
                 f"{event['team_name']} changes formation"
            ])
            event_data = event.to_dict()
        else: continue
            
        timeline.append({
            'time_min': time_min,
            'time_str': f"{time_min}'",
            'team_id': event['contestantId'],
            'typeId': event['typeId'],
            'event_data': event_data,
            'description_component': description
        })
    return timeline


def get_key_formation_states(df_processed, team_name, home_team_id, away_team_id):
    """
    Estrae gli stati chiave della formazione e calcola il punteggio corretto per ciascuno.
    """
    df_team = df_processed[df_processed['team_name'] == team_name].copy()
    
    # Evento iniziale
    start_event_series = df_team[df_team['typeId'] == 34].sort_values('eventId')
    if start_event_series.empty:
        return [] # Non possiamo procedere senza una formazione iniziale
    start_event = start_event_series.iloc[0]
    
    # Eventi di cambio formazione
    formation_change_events = df_team[df_team['typeId'] == 40].sort_values('eventId')

    key_states = []
    
    # Aggiungi stato iniziale
    key_states.append({
        'time_str': "0'",
        'description': f"Starting XI: {formation_layouts.get_formation_name(start_event['Team formation'])}",
        'score': "0 - 0", # La partita inizia sempre 0-0
        'state': {
            'formation_id': int(start_event['Team formation']),
            'players': _extract_player_positions(start_event)
        }
    })

    # Aggiungi i cambi di formazione
    for _, event in formation_change_events.iterrows():
        # Calcola il punteggio contando tutti i gol avvenuti PRIMA del minuto di questo evento
        minute_of_change = event['timeMin']
        goals_until_then = df_processed[
            (df_processed['typeId'] == 16) & 
            (df_processed['timeMin'] < minute_of_change)
        ]
        home_score = (goals_until_then['contestantId'] == home_team_id).sum()
        away_score = (goals_until_then['contestantId'] == away_team_id).sum()
        
        key_states.append({
            'time_str': f"{event['timeMin']}'",
            'description': f"Formation Change to {formation_layouts.get_formation_name(event['Team formation'])}",
            'score': f"{home_score} - {away_score}", # Aggiungi lo score calcolato
            'state': {
                'formation_id': int(event['Team formation']),
                'players': _extract_player_positions(event)
            }
        })
        
    return key_states

# --- 2. FUNZIONE PER LA LISTA DELLE SOSTITUZIONI ---

def create_substitutions_list(df_processed, team_name, player_data_map):
    """
    Versione 2: Usa 'related_eventId' per un matching corretto delle sostituzioni multiple.
    """
    df_team = df_processed[df_processed['team_name'] == team_name].copy()
    
    # Prendi tutti gli eventi di sostituzione per questa squadra
    subs_off = df_team[df_team['typeId'] == 18].sort_values('timeMin')
    subs_on = df_team[df_team['typeId'] == 19]
    
    if subs_off.empty:
        return dbc.ListGroupItem("No substitutions made.")

    # Crea un dizionario per un accesso veloce agli eventi 'Player On'
    # La chiave è il related_eventId (che punta all'eventId del Player Off)
    subs_on_map = {}
    if 'related_eventId' in subs_on.columns:
        # Assicurati che related_eventId sia dello stesso tipo di eventId
        subs_on['related_eventId'] = pd.to_numeric(subs_on['related_eventId'], errors='coerce')
        subs_off['eventId'] = pd.to_numeric(subs_off['eventId'], errors='coerce')
        subs_on_map = subs_on.set_index('related_eventId').to_dict('index')
    else:
        # Se 'related_eventId' non è stato mappato, non possiamo fare il matching.
        # Stampiamo un avviso e torniamo.
        print("WARNING: 'related_eventId' column not found. Cannot perform accurate substitution matching.")
        return dbc.ListGroupItem("Substitution data is incomplete (missing event links).")


    list_items = [html.H5("Substitutions", className="text-white mb-3")]
    
    for _, sub_off_event in subs_off.iterrows():
        time_min = sub_off_event['timeMin']
        player_off_name = sub_off_event['playerName']
        off_event_id = sub_off_event['eventId']
        
        # Cerca il giocatore che entra usando la mappa
        sub_on_event_dict = subs_on_map.get(off_event_id)
        
        player_on_name = "Matching Error"
        if sub_on_event_dict:
            player_on_name = sub_on_event_dict.get('playerName', 'N/A')
            
        list_items.append(
            dbc.ListGroupItem(
                [
                    html.Span(f"{time_min}'", className="fw-bold me-3"),
                    # Usiamo le icone di Font Awesome che hai già importato
                    html.I(className="fas fa-arrow-up me-1", style={'color': '#28a745'}),
                    html.Span(f"{player_on_name}", className="me-3"),
                    html.I(className="fas fa-arrow-down me-1", style={'color': '#dc3545'}),
                    html.Span(f"{player_off_name}"),
                ], 
                className="bg-transparent text-white border-secondary"
            )
        )
        
    return dbc.ListGroup(list_items, flush=True)

# --- 3. FUNZIONE DI PLOTTING CON EVIDENZIAZIONE ---

def plot_formation_snapshot_with_changes(current_snapshot, previous_state, player_data_map, team_color):
    """
    Versione Semplificata: si concentra su plot, punteggio e highlight dei cambi di posizione.
    """
    pitch = Pitch(pitch_type='opta', pitch_color='#2E3439', line_color='white', line_zorder=2)
    fig, ax = pitch.draw(figsize=(11, 8.5)) # Dimensioni grandi
    fig.set_facecolor('#2E3439')
    
    current_state = current_snapshot['state']
    formation_id = current_state['formation_id']
    score = current_snapshot.get('score', '')
    
    # Titolo del grafico con lo score
    title = f"{current_snapshot['time_str']}  |  Score: {score}  |  {current_snapshot['description']}"
    ax.set_title(title, color='white', fontsize=16, pad=15)

    for player_id, current_pos in current_state['players'].items():
        x, y = formation_layouts.get_formation_layout_coords(formation_id, current_pos)
        if x is None: continue

        player_info = player_data_map.get(player_id, {})
        jersey = str(int(player_info.get('jersey', 0))) if str(player_info.get('jersey', '?')).isdigit() else '?'
        name = player_info.get('name', 'N/A')
        
        # Colore ciano per evidenziare chi ha cambiato posizione
        circle_color = team_color
        if previous_state and previous_state['players'].get(player_id) != current_pos:
            circle_color = '#00FFFF' # Ciano
        
        pitch.scatter(x, y, s=800, c=circle_color, edgecolors='white', ax=ax, zorder=3)
        pitch.annotate(jersey, (x, y), va='center', ha='center', color='black', fontsize=12, fontweight='bold', ax=ax, zorder=4)
        pitch.annotate(name.split()[-1], (x, y - 8), va='center', ha='center', color='white', fontsize=10, ax=ax, zorder=4)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=120)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    
    return f"data:image/png;base64,{img_b64}"

# --- FUNZIONE DI PLOTTING (MODIFICATA PER ACCETTARE STATO E PUNTEGGIO) ---
def plot_formation_snapshot(current_state, player_colors, player_data_map, base_color, title_text, is_away=False):
    """
    Versione 5: Il titolo ora è più pulito e non include più lo score.
    """
    pitch = Pitch(pitch_type='opta', pitch_color='#2E3439', line_color='white', line_zorder=2)
    fig, ax = pitch.draw(figsize=(8, 6))
    fig.set_facecolor('#2E3439')
    
    formation_id = current_state['formation_id']
    ax.set_title(title_text, color='white', fontsize=12, pad=8)

    for player_id, pos_num in current_state['players'].items():
        x, y = formation_layouts.get_formation_layout_coords(formation_id, pos_num)
        if x is None: continue
        
        if is_away: x, y = 100 - x, 100 - y

        player_info = player_data_map.get(player_id, {})
        jersey = player_info.get('jersey', '?')
        name = player_info.get('name', 'N/A')
        
        circle_color = player_colors.get(player_id, base_color)
        
        pitch.scatter(x, y, s=650, c=circle_color, edgecolors='white', ax=ax, zorder=3)
        pitch.annotate(jersey, (x, y), va='center', ha='center', color='black', fontsize=10, fontweight='bold', ax=ax, zorder=4)
        pitch.annotate(name.split()[-1], (x, y - 7.5), va='center', ha='center', color='white', fontsize=8, ax=ax, zorder=4)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), dpi=100)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('ascii')
    plt.close(fig)
    
    return f"data:image/png;base64,{img_b64}"