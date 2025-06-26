# src/metrics/cross_metrics.py
import pandas as pd
import dash_bootstrap_components as dbc
from dash import html, dash_table

def analyze_crosses(df_processed, team_name):
    """
    Identifies all crosses for a team and enriches them with detailed metrics.
    """
    # Filtra tutti gli eventi di tipo "Pass" con il qualifier "cross"
    crosses_df = df_processed[
        (df_processed['team_name'] == team_name) & 
        (df_processed['type_name'] == 'Pass') &
        (df_processed['cross'] == 1)
    ].copy()

    if crosses_df.empty:
        return pd.DataFrame()

    analyzed_data = []
    
    # Determina la zona di origine e destinazione per ogni cross
    def get_pitch_zone(x, y):
        if y > 67:
            side = "Left"
        elif y < 33:
            side = "Right"
        else:
            side = "Center"
        
        if x > 80:
            area = "Deep"
        elif x > 60:
            area = "Advanced"
        else:
            area = "Midfield"
            
        return f"{side} {area}"

    for _, cross in crosses_df.iterrows():
        play_type = "Open Play"
        if cross.get('Corner taken') == 1:
            play_type = "From Corner"
        elif cross.get('Freekick taken') == 1:
            play_type = "From Free Kick"
        foot = 'Right' if cross.get('Right footed') == 1 else ('Left' if cross.get('Left footed') == 1 else 'Unknown')
        swing = 'N/A'
        if cross.get('In-swinger') == 1: swing = 'In-swinger'
        elif cross.get('Out-swinger') == 1: swing = 'Out-swinger'
        elif cross.get('Straight') == 1: swing = 'Straight'
        
        origin_zone = get_pitch_zone(cross['x'], cross['y'])
        destination_zone = get_pitch_zone(cross['end_x'], cross['end_y'])
        
        # Semplificazione dell'outcome
        outcome = "Retained"
        if cross['outcome'] == 'Unsuccessful':
            outcome = "Lost"
        # Potremmo aggiungere una logica per 'Shot' o 'Goal' se analizziamo la sequenza successiva,
        # per ora ci limitiamo al successo del cross stesso.

        analyzed_data.append({
            'cross_id': cross['eventId'],
            'playerName': cross['playerName'],
            'Play Type': play_type,
            'Foot': foot,
            'Swing': swing,
            'Origin Zone': origin_zone,
            'Destination Zone': destination_zone,
            'Outcome': outcome,
            'x': cross['x'],
            'y': cross['y'],
            'end_x': cross['end_x'],
            'end_y': cross['end_y']
        })
        
    return pd.DataFrame(analyzed_data)

def create_cross_summary_cards(df_analyzed, active_filter=None):
    """Creates a full set of detailed, interactive summary cards for crosses."""
    if df_analyzed.empty:
        return html.Div()

    # La funzione helper interna ora è corretta
    def create_card(title, data_dict, filter_type):
        items = []
        if not data_dict: return None
        sorted_items = sorted(data_dict.items(), key=lambda item: item[1], reverse=True)
        # Usiamo l'active_filter definito nello scope esterno
        for value, count in sorted_items:
            active = active_filter is not None and str(active_filter.get(filter_type)) == str(value)
            
            items.append(dbc.ListGroupItem(
                [html.Div(value), dbc.Badge(f"{count}", color="light", className="ms-auto")],
                id={'type': 'cross-filter', 'filter_type': filter_type, 'value': value},
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            ))
        return dbc.Col(dbc.Card([dbc.CardHeader(title), dbc.ListGroup(items, flush=True)]), md=4)

    stats = {
        'origin': df_analyzed['Origin Zone'].value_counts().to_dict(),
        'destination': df_analyzed['Destination Zone'].value_counts().to_dict(),
        'swing': df_analyzed[df_analyzed['Swing'] != 'N/A']['Swing'].value_counts().to_dict(),
        'outcome': df_analyzed['Outcome'].value_counts().to_dict(),
        'feet': df_analyzed[df_analyzed['Foot'] != 'Unknown']['Foot'].value_counts().to_dict(),
        'takers': df_analyzed['playerName'].value_counts().to_dict(),
        'play_type': df_analyzed['Play Type'].value_counts().to_dict()
    }
    
    active_filter = active_filter or {}
    
    # --- CHIAMATE CORRETTE (con 3 argomenti) ---
    cards = [
        create_card("Origin Zone", stats.get('origin'), 'origin'),
        create_card("Destination Zone", stats.get('destination'), 'destination'),
        create_card("Play Type", stats.get('play_type'), 'play_type'),
        create_card("Swing Type", stats.get('swing'), 'swing'),
        create_card("Taker Foot", stats.get('feet'), 'foot'),
        create_card("Outcome", stats.get('outcome'), 'outcome'),
    ]
    
    # Creiamo la card dei crossatori a parte per gestire il filter_type 'taker'
    takers_stats = stats.get('takers')
    if takers_stats:
        takers_items = []
        # Mostra solo i primi 5 per non affollare
        for player, count in sorted(takers_stats.items(), key=lambda item: item[1], reverse=True)[:5]:
            active = active_filter.get('taker') == player
            takers_items.append(dbc.ListGroupItem(
                [html.Div(player), dbc.Badge(f"{count}", color="light", className="ms-auto")],
                id={'type': 'cross-filter', 'filter_type': 'taker', 'value': player},
                action=True, n_clicks=0, active=active,
                className="d-flex justify-content-between align-items-center"
            ))
        cards.append(dbc.Col(dbc.Card([dbc.CardHeader("Top Crossers"), dbc.ListGroup(takers_items, flush=True)]), md=4))

    valid_cards = [card for card in cards if card is not None]
    
    # Layout a 3 colonne
    rows = [dbc.Row(valid_cards[i:i+3], className="mb-3") for i in range(0, len(valid_cards), 3)]
    
    return html.Div(rows)

def generate_cross_flow_table(df_analyzed):
    """
    Crea una tabella a matrice di flusso (heatmap tabellare) che mostra
    le origini e le destinazioni dei cross.
    VERSIONE CORRETTA E SEMPLIFICATA
    """
    if df_analyzed.empty:
        return dbc.Alert("No cross flow data to display.", color="info")

    flow_matrix = pd.crosstab(
        df_analyzed['Origin Zone'],
        df_analyzed['Destination Zone']
    )

    if flow_matrix.empty:
        return dbc.Alert("No cross flow data to display.", color="info")

    flow_matrix['Total Out'] = flow_matrix.sum(axis=1)
    flow_matrix.loc['Total In'] = flow_matrix.sum(axis=0)
    flow_matrix = flow_matrix.reset_index()

    columns = [{"name": i, "id": i} for i in flow_matrix.columns]
    data = flow_matrix.to_dict('records')

    # --- Stile "Fancy" Semplificato (Heatmap di Colori) ---
    style_data_conditional = []
    
    # Itera sulle colonne numeriche per applicare la scala di colori
    numeric_cols = flow_matrix.columns.drop(['Origin Zone'])
    
    for col in numeric_cols:
        # Trova il massimo valore nella colonna per normalizzare
        max_val = flow_matrix[col].max()
        if max_val > 0:
            # Applica una scala di colori (es. da trasparente a blu)
            style_data_conditional.extend([
                {
                    'if': {
                        'filter_query': f'{{{col}}} = {val}',
                        'column_id': col
                    },
                    # Calcola l'opacità in base al valore della cella
                    'backgroundColor': f'rgba(91, 192, 222, {val / max_val})', # Colore Info di Bootstrap
                    'color': 'white'
                } for val in flow_matrix[col].unique() if val > 0
            ])

    return dash_table.DataTable(
        columns=columns,
        data=data,
        style_as_list_view=True,
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white',
            'fontWeight': 'bold',
            'border': '1px solid rgb(80, 80, 80)',
        },
        style_cell={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white',
            'textAlign': 'center',
            'minWidth': '90px', 'width': '90px', 'maxWidth': '90px',
            'border': '1px solid rgb(80, 80, 80)'
        },
        style_data_conditional=[
            {
                'if': {'row_index': 'odd'},
                'backgroundColor': 'rgb(60, 60, 60)'
            },
            {
                'if': {'column_id': 'Total Out'},
                'backgroundColor': 'rgb(40, 40, 40)',
                'fontWeight': 'bold'
            },
            {
                'if': {'row_index': len(flow_matrix) - 2}, # La riga dei totali
                'backgroundColor': 'rgb(40, 40, 40)',
                'fontWeight': 'bold'
            }
        ] + style_data_conditional # Aggiungi la nostra heatmap
    )